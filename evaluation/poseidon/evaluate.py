"""SAFE-SVD-v0 Poseidon evaluation, using this repository's exported models.

Dataset construction, ScOT prediction/loss, previous-time inference and physics
aggregation follow SAFE-SVD-v0/evaluation/poseidon. No quantization is run here.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from evaluation.reporting import build_dataset_results
from quantization.calibration import dataset_metadata, _fingerprint
from .physics import evaluate_and_result


class FidelityDataset(Dataset):
    """Use original predictions as labels, matching SAFE-SVD's oncomp mode."""

    def __init__(self, dataset, predictions):
        self.ds = dataset
        self.origin_preds = torch.as_tensor(predictions)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = dict(self.ds[index])
        item["labels"] = self.origin_preds[index]
        return item

    def __getattr__(self, name):
        if name == "ds":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "ds"), name)


def build_test_dataset(get_dataset, args):
    """Pass the same arguments as SAFE-SVD, without truncating held-out data."""
    time_kwargs = {}
    if ".time" not in args.dataset:
        time_kwargs = dict(fix_input_to_time_step=None, time_step_size=1,
                           max_num_time_steps=args.final_time)
    return get_dataset(
        dataset=args.dataset, which="test",
        num_trajectories=64 if args.num_trajectories is None else args.num_trajectories,
        data_path=str(args.data_path), **time_kwargs,
    )


def build_trainer(model, args):
    """Mirror ScOT get_trainer without reloading an FP model for each artifact.

    SAFE-SVD's physical report only consumes predict()'s _loss, so the channel
    statistics callback in scOT.inference.get_trainer is unnecessary here.
    Keep labels and pixel_mask: they define ScOT's loss and fixed channels.
    """
    from scOT.trainer import Trainer, TrainingArguments

    device = torch.device(args.DEV)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    training_args = TrainingArguments(
        output_dir=".", per_device_eval_batch_size=args.batch_size,
        eval_accumulation_steps=16, dataloader_num_workers=0,
        remove_unused_columns=False, label_names=["labels"],
        use_cpu=device.type == "cpu", report_to=[],
    )
    return Trainer(model=model.to(device).eval(), args=training_args)


def rollout(trainer, dataset):
    """Same single direct prediction per time pair as scOT.inference.rollout."""
    trainer.set_ar_steps(ar_steps=1, output_all_steps=False)
    prediction = trainer.predict(dataset, metric_key_prefix="")
    return prediction.predictions, prediction.label_ids, prediction.metrics


def validate_args(args):
    if not args.model_path and not args.checkpoint:
        raise ValueError("Provide --model_path and/or --checkpoint.")
    if args.eval_obj == "oncomp" and not args.checkpoint:
        raise ValueError("--eval_obj oncomp requires --checkpoint for original predictions.")
    if args.split != "test" or args.initial_time != 0 or args.ar_steps != 1:
        raise ValueError("SAFE-SVD uses the test split and all time pairs with ar_steps=1; "
                         "use --protocol endpoint for fixed-endpoint/AR evaluation.")
    if args.num_workers != 0:
        raise ValueError("SAFE-SVD evaluation uses --num_workers 0.")
    count = 64 if args.num_trajectories is None else args.num_trajectories
    if min(count, args.final_time, args.batch_size, args.physics_batch_size) <= 0:
        raise ValueError("Trajectory, time, and batch-size arguments must be positive.")
    if args.DEV.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --DEV cpu.")


def run(args):
    validate_args(args)
    from scOT.problems.base import get_dataset

    test_ds = build_test_dataset(get_dataset, args)
    if not len(test_ds):
        raise ValueError("Evaluation dataset is empty.")
    # Record the real held-out count: ScOT num_trajectories is training capacity.
    samples_per_trajectory = int(getattr(test_ds, "multiplier", 1))
    selection = {
        "schema": "safe-svd-poseidon-evaluation-v1", "split": "test",
        "num_samples": len(test_ds),
        "num_test_trajectories": len(test_ds) // samples_per_trajectory,
        "samples_per_trajectory": samples_per_trajectory,
        "time_pairs": getattr(test_ds, "time_indices", None),
        "source": dataset_metadata(test_ds),
    }
    selection["fingerprint"] = _fingerprint(selection)
    origin_metrics = None
    if args.checkpoint:
        from scOT.model import ScOT

        model = ScOT.from_pretrained(str(args.checkpoint), local_files_only=True)
        trainer = build_trainer(model, args)
        if args.eval_obj == "oncomp":
            predictions, _, _ = rollout(trainer, test_ds)
            test_ds = FidelityDataset(test_ds, predictions)
        origin_metrics = evaluate_and_result(
            trainer, test_ds, args.dataset, rollout, args.physics_batch_size)
        # Avoid keeping the full original model on GPU during compressed inference.
        trainer.model.cpu()
        del trainer, model

    compressed_metrics = None
    artifact_metadata = {}
    if args.model_path:
        from quantization.artifacts import load_quantized_artifact

        model, artifact_metadata = load_quantized_artifact(args.model_path, "cpu")
        trainer = build_trainer(model, args)
        compressed_metrics = evaluate_and_result(
            trainer, test_ds, args.dataset, rollout, args.physics_batch_size)

    metadata = {
        "protocol": "safe-svd", "eval_obj": args.eval_obj,
        "num_trajectories": 64 if args.num_trajectories is None else args.num_trajectories,
        "final_time": args.final_time, "batch_size": args.batch_size,
        "physics_batch_size": args.physics_batch_size,
        "evaluation": selection,
        "metric_definitions": {
            "base_loss": "ScOT Trainer _loss, using the saved model config; not RelL1 percent.",
            "physics": "SAFE-SVD-v0 PDE_calc: denormalized, dx=dy=1/128, Re=2500; "
                       "compressible dt=1/20, wave dt=1/15.",
            "aggregation": "Unweighted mean over physical batches, including the final partial batch.",
            "relative": "rel_* are ratios; compressed.increase is percent change from origin.",
            "prediction": "Direct inference on every test time pair (including t1=t2); "
                          "ScOT labels/pixel_mask conventions; previous-time pass for wave/compressible.",
        },
    }
    if args.model_path:
        metadata["artifact"] = artifact_metadata
    if origin_metrics is not None:
        return build_dataset_results(
            Path(args.checkpoint), {args.dataset: origin_metrics},
            compressed_model=Path(args.model_path) if args.model_path else None,
            compressed_metrics={args.dataset: compressed_metrics} if args.model_path else None,
            metadata=metadata,
        )
    # Standalone compressed models do not require the original checkpoint.
    return {args.dataset: {
        **metadata, "dataset": args.dataset,
        "compressed": {"model": str(args.model_path), "metrics": compressed_metrics},
    }}


if __name__ == "__main__":
    from evaluation.evaluate import main
    main()

"""Evaluate a saved quantized artifact without recalibration or its FP checkpoint.

Run with ``python -m quantization.evaluate --model_path ... --dataset ...``.
Reported L1/RelL1 follow scOT.metrics (normalized fields, sample mean, percent).
Physical structure metrics use denormalized fields and unit-domain grid spacing.
No target is passed to the model, including during autoregressive rollout.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from quantization.calibration import dataset_metadata, _jsonable, _fingerprint, _underlying_dataset


def get_dataset(*args, **kwargs):
    from scOT.problems.base import get_dataset as factory
    return factory(*args, **kwargs)


class _PrefixDataset(Dataset):
    def __init__(self, source, length):
        self.source_dataset = source
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.source_dataset[index]

    def __getattr__(self, name):
        if name == "source_dataset":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "source_dataset"), name)


def build_evaluation_loader(args):
    if args.split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test.")
    static = "elliptic" in args.dataset or "steady" in args.dataset
    if args.ar_steps < 1:
        raise ValueError("ar_steps must be positive.")
    if static and args.ar_steps != 1:
        raise ValueError("Autoregressive rollout is undefined for a static problem.")
    kwargs = {"data_path": str(args.data_path)}
    if not static:
        if args.initial_time < 0 or args.final_time <= args.initial_time:
            raise ValueError("Require 0 <= initial_time < final_time.")
        horizon = args.final_time - args.initial_time
        if horizon % args.ar_steps:
            raise ValueError("final_time - initial_time must be divisible by ar_steps.")
        kwargs.update(fix_input_to_time_step=args.initial_time, time_step_size=horizon,
                      max_num_time_steps=1)
    source = get_dataset(dataset=args.dataset, which=args.split, num_trajectories=-1, **kwargs)
    # ScOT num_trajectories describes training capacity even for val/test splits.
    # Each static sample or fixed-endpoint sample is exactly one held-out trajectory.
    available = len(source)
    requested = getattr(args, "num_trajectories", None)
    if requested is not None and (requested < 1 or requested > available):
        raise ValueError(f"num_trajectories must be in [1, {available}], or omitted for the full split.")
    count = available if requested is None else requested
    dataset = source if count == available else _PrefixDataset(source, count)
    metadata = {
        "schema": "quantized-artifact-evaluation-v1", "dataset": args.dataset,
        "split": args.split, "available_trajectories": available,
        "num_trajectories": count, "num_samples": len(dataset),
        "trajectory_indices": list(range(count)),
        "absolute_trajectory_indices": [int(getattr(_underlying_dataset(source), "start", 0)) + i
                                        for i in range(count)],
        "time_pairs": None if static else [[args.initial_time, args.final_time]],
        "ar_steps": args.ar_steps, "source": dataset_metadata(source),
    }
    metadata["fingerprint"] = _fingerprint(metadata)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    return dataset, loader, metadata


def _prediction(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "output"):
        return output.output
    if isinstance(output, dict) and "output" in output:
        return output["output"]
    raise TypeError("Model must return a Tensor or a ScOT output with .output.")


def _mask_for_output(mask, prediction):
    mask = mask.to(device=prediction.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask[:, :prediction.shape[1], None, None]
    elif mask.ndim == 4:
        mask = mask[:, :prediction.shape[1]]
    else:
        raise ValueError("Expected a channel or spatial ScOT pixel_mask.")
    return torch.broadcast_to(mask, prediction.shape)


@torch.no_grad()
def autoregressive_predict(model, batch, *, device, ar_steps=1):
    """Roll out predictions; preserve masked/input-only channels from inputs.

    ScOT normally copies masked channels from labels in forward(). Here these
    nonpredicted channels are copied from the current input instead, so future
    labels cannot leak into rollout. Extra input-only channels follow ScOT's
    Trainer convention and are carried forward unchanged.
    """
    if ar_steps < 1:
        raise ValueError("ar_steps must be positive.")
    x = batch["pixel_values"].to(device)
    time = batch.get("time")
    if time is not None:
        time = time.to(device=device, dtype=x.dtype) / ar_steps
    elif ar_steps != 1:
        raise ValueError("A time-conditioned batch is required for autoregression.")
    mask = batch.get("pixel_mask")
    for _ in range(ar_steps):
        kwargs = {"pixel_values": x, "return_dict": True}
        if time is not None:
            kwargs["time"] = time
        pred = _prediction(model(**kwargs))
        if pred.ndim != 4 or pred.shape[0] != x.shape[0] or pred.shape[-2:] != x.shape[-2:]:
            raise ValueError("Model output must match the input batch and spatial shape.")
        if pred.shape[1] > x.shape[1]:
            raise ValueError("ScOT output has more channels than its input.")
        if mask is not None:
            pred = torch.where(_mask_for_output(mask, pred), x[:, :pred.shape[1]], pred)
        x = torch.cat((pred, x[:, pred.shape[1]:]), dim=1)
    return pred


def denormalize_output(value, dataset, dataset_name):
    dataset = _underlying_dataset(dataset)
    constants = getattr(dataset, "constants", {})
    channels = value.shape[1]
    if "wave" in dataset_name:
        mean = [constants["mean"], constants["mean_c"]][:channels]
        std = [constants["std"], constants["std_c"]][:channels]
    elif "mean_solution" in constants:
        mean, std = [constants["mean_solution"]], [constants["std_solution"]]
    elif "mean" in constants and "std" in constants:
        mean = torch.as_tensor(constants["mean"]).flatten().tolist()
        std = torch.as_tensor(constants["std"]).flatten().tolist()
        names = [name.strip() for group in re.findall(r"\[([^\]]+)\]", getattr(dataset, "label_description", ""))
                 for name in group.split(",")]
        while channels > len(mean) and len(mean) < len(names):
            name = names[len(mean)]
            if name == "tracer" and "incompressible" in dataset_name:
                mean.append(constants["tracer_mean"])
                std.append(constants["tracer_std"])
            elif name == "tracer":
                # CompressibleBase appends its tracer without normalization.
                mean.append(0.0)
                std.append(1.0)
            elif name == "g" and "mean_forcing" in constants:
                mean.append(constants["mean_forcing"])
                std.append(constants["std_forcing"])
            else:
                break
    elif hasattr(dataset, "mean") and hasattr(dataset, "std"):
        mean, std = [dataset.mean], [dataset.std]
    else:
        raise ValueError("Dataset does not expose output normalization for physical metrics.")
    mean = torch.as_tensor(mean, dtype=value.dtype, device=value.device).flatten()
    std = torch.as_tensor(std, dtype=value.dtype, device=value.device).flatten()
    if mean.numel() not in (1, channels) and mean.numel() < channels:
        raise ValueError("Output normalization does not cover all channels.")
    result = value * std[:channels].view(1, -1, 1, 1) + mean[:channels].view(1, -1, 1, 1)
    if hasattr(dataset, "mean_pressure") and channels > 3:
        # CompressibleBase subtracts a dataset-specific pressure offset *before*
        # applying the shared normalization constants.
        result[:, 3] += float(dataset.mean_pressure)
    return result


def _sample_mean(value):
    return value.flatten(1).mean(1)


def per_sample_metrics(pred, target, dataset, dataset_name):
    """Return vectors, so partial final batches receive correct sample weights."""
    dataset = _underlying_dataset(dataset)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}.")
    pred, target = pred.double(), target.double()
    error = (pred - target).abs().flatten(1).sum(1)
    norm = target.abs().flatten(1).sum(1)
    denominator = torch.where(norm == 0, torch.full_like(norm, 1e-10), norm)
    metrics = {"L1": error, "RelL1": 100 * error / denominator}
    p = denormalize_output(pred, dataset, dataset_name)
    y = denormalize_output(target, dataset, dataset_name)
    dx, dy = 1.0 / p.shape[-1], 1.0 / p.shape[-2]
    if min(p.shape[-2:]) < 3:
        raise ValueError("Physical structure metrics require at least a 3 x 3 grid.")
    dx_p, dx_y = torch.diff(p, dim=-1) / dx, torch.diff(y, dim=-1) / dx
    dy_p, dy_y = torch.diff(p, dim=-2) / dy, torch.diff(y, dim=-2) / dy
    sob = _sample_mean((p-y).abs()) + _sample_mean((dx_p-dx_y).abs()) + _sample_mean((dy_p-dy_y).abs())
    scale = _sample_mean(y.abs()) + _sample_mean(dx_y.abs()) + _sample_mean(dy_y.abs())
    metrics.update(sobolev_s01=sob, rel_sobolev_s01=sob / scale.clamp_min(1e-12))
    if "incompressible" in dataset_name:
        desc = getattr(dataset, "label_description", "")
        names = [name.strip() for group in re.findall(r"\[([^\]]+)\]", desc) for name in group.split(",")]
        if "u" not in names or "v" not in names:
            raise ValueError("NS physical metrics require named u,v output channels.")
        # Incompressible HDF5 uses x,y; transpose=True already switches to y,x.
        if not getattr(dataset, "transpose", False):
            p, y = p.transpose(-2, -1), y.transpose(-2, -1)
        dx, dy = 1.0 / p.shape[-1], 1.0 / p.shape[-2]
        def grads(field):
            gx = (field[..., 1:-1, 2:] - field[..., 1:-1, :-2]) / (2*dx)
            gy = (field[..., 2:, 1:-1] - field[..., :-2, 1:-1]) / (2*dy)
            return gx, gy
        ux, uy = grads(p[:, names.index("u")])
        vx, vy = grads(p[:, names.index("v")])
        ux_y, uy_y = grads(y[:, names.index("u")])
        vx_y, vy_y = grads(y[:, names.index("v")])
        div = _sample_mean((ux+vy).abs())
        vort = _sample_mean(((vx-uy)-(vx_y-uy_y)).abs())
        metrics.update(continuity=div,
                       rel_continuity=div/(_sample_mean(ux_y.abs())+_sample_mean(vy_y.abs())).clamp_min(1e-12),
                       vorticity=vort, rel_vorticity=vort/_sample_mean((vx_y-uy_y).abs()).clamp_min(1e-12))
    return metrics


@torch.no_grad()
def evaluate_model(model, loader, *, dataset, dataset_name, device, ar_steps=1):
    model.to(device).eval()
    totals, count = {}, 0
    for batch in loader:
        pred = autoregressive_predict(model, batch, device=device, ar_steps=ar_steps)
        target = batch["labels"].to(device=device)
        # The model can predict fewer channels than the dataset input/labels.
        target = target[:, :pred.shape[1]]
        values = per_sample_metrics(pred, target, dataset, dataset_name)
        for key, value in values.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {key} encountered during evaluation.")
            totals[key] = totals.get(key, 0.0) + value.sum().item()
        count += pred.shape[0]
    if not count:
        raise ValueError("Evaluation dataset is empty.")
    return {"num_samples": count, "metrics": {key: total/count for key, total in totals.items()}}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--DEV", default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--initial_time", type=int, default=0)
    parser.add_argument("--final_time", type=int, default=14)
    parser.add_argument("--ar_steps", type=int, default=1)
    parser.add_argument("--num_trajectories", type=int, default=None,
                        help="Omit to evaluate the complete split; otherwise use its deterministic prefix.")
    parser.add_argument("--output_path", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from quantization.artifacts import load_quantized_artifact
    model, artifact_meta = load_quantized_artifact(args.model_path, args.DEV)
    dataset, loader, selection = build_evaluation_loader(args)
    result = evaluate_model(model, loader, dataset=dataset, dataset_name=args.dataset,
                            device=args.DEV, ar_steps=args.ar_steps)
    result.update(artifact=_jsonable(artifact_meta), evaluation=selection, metric_definitions={
        "L1": "Mean over samples of summed absolute error in normalized output channels.",
        "RelL1": "Mean per-sample relative L1, percent; all predicted output channels.",
        "sobolev_s01": "Denormalized mean absolute error plus first forward-difference errors; dx=1/W, dy=1/H.",
        "continuity": "Mean absolute central-difference velocity divergence, NS only.",
        "vorticity": "Mean absolute curl error relative to ground truth, NS only.",
        "relative_structure_metrics": "Dimensionless ratios, not percentages.",
        "rollout": "Endpoint only; predicted states feed subsequent steps; masked channels copied from input.",
    })
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output_path:
        output = Path(args.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized)
    return result


if __name__ == "__main__":
    main()

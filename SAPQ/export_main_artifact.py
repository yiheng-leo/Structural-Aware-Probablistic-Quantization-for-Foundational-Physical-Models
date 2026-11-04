"""Materialize main's final SAPQ steps as a ritchie evaluation artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


def materialize_weights(model, steps):
    """Match PPQ.metrics' weight-only quantization hooks exactly."""
    modules = dict(model.named_modules())
    validated = []
    if not steps:
        raise ValueError("No quantization steps were saved.")
    for name, entry in steps.items():
        layer = modules.get(name)
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"Checkpoint has no Linear layer named {name}.")
        step = entry[0] if isinstance(entry, (tuple, list)) else entry
        if not torch.is_tensor(step) or step.shape != (layer.out_features,):
            raise ValueError(f"Invalid output-channel step shape for {name}.")
        step = step.detach().to(device=layer.weight.device, dtype=layer.weight.dtype)
        if not torch.isfinite(step).all() or not (step > 0).all():
            raise ValueError(f"Nonpositive or nonfinite quantization steps for {name}.")
        validated.append((layer, step))
    # Validate every layer before modifying any weights.
    with torch.no_grad():
        for layer, step in validated:
            scale = step.reshape(-1, 1)
            layer.weight.copy_(torch.round(layer.weight / scale) * scale)
    return model.eval().requires_grad_(False)


def final_history(history_path):
    if not history_path.is_file():
        return None, None
    entries = json.loads(history_path.read_text())
    epochs = [entry for entry in entries if "epoch" in entry and "avg_bits" in entry]
    if not epochs:
        raise ValueError(f"No epoch bit statistics in {history_path}.")
    final = max(epochs, key=lambda entry: entry["epoch"])
    bits = float(final["avg_bits"])
    if not math.isfinite(bits):
        raise ValueError("Final average bits must be finite.")
    return bits, int(final["epoch"])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(step_path, save_path, checkpoint=None, history_path=None):
    from scOT.model import ScOT

    step_path, save_path = Path(step_path), Path(save_path)
    if save_path.exists() and any(save_path.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {save_path}")
    saved = torch.load(step_path, map_location="cpu", weights_only=False)
    steps, source_meta = saved["step_sizes_dict"], saved.get("meta", {})
    checkpoint = checkpoint or source_meta.get("model_path")
    if not checkpoint:
        raise ValueError("Specify the original --checkpoint used during SAPQ training.")
    history_path = Path(history_path) if history_path else step_path.with_name("sapq_global_history.json")
    bits, epoch = final_history(history_path)
    print(f"Loading original checkpoint: {checkpoint}", flush=True)
    model = ScOT.from_pretrained(str(checkpoint), local_files_only=True).cpu().eval()
    materialize_weights(model, steps)
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / "compressed_model.pt"
    print(f"Saving {len(steps)} quantized Linear layers: {model_file}", flush=True)
    torch.save(model, model_file)
    torch.save(saved, save_path / "quantization_state.pt")
    metadata = {
        "schema_version": 1, "artifact_type": "poseidon_quantization",
        "model_file": model_file.name, "model_sha256": sha256(model_file),
        "source_checkpoint": str(checkpoint), "source_steps": str(step_path.resolve()),
        "source_steps_sha256": sha256(step_path), "source_meta": source_meta,
        "source_history": str(history_path.resolve()) if history_path.is_file() else None,
        "quantization": {
            "method": "sapq", "source_branch": "main", "avg_weight_bits": bits,
            "epochs": epoch, "num_layers": len(steps), "layer_names": list(steps),
            "representation": "materialized_float_fake_quantized_weights", "weight_only": True,
        },
    }
    (save_path / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Export complete: epoch={epoch}, avg_weight_bits={bits}", flush=True)
    if bits is None:
        print("History unavailable: evaluation will report avg_weight_bits=null; target_bits is not substituted.")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step_path", required=True)
    parser.add_argument("--checkpoint", help="Same original fine-tuned checkpoint used for SAPQ.")
    parser.add_argument("--history_path", help="Defaults to sapq_global_history.json beside the steps file.")
    parser.add_argument("--save_path", required=True, help="New artifact directory, preferably outside Git checkout.")
    args = parser.parse_args(argv)
    export(args.step_path, args.save_path, args.checkpoint, args.history_path)


if __name__ == "__main__":
    main()

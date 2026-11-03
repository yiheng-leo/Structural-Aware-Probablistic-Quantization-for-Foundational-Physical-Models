#!/usr/bin/env python3
"""Prepare missing input caches for main's original global SAPQ entry point.

Run from a main checkout, after setting PPQ_MODEL_PATH/PPQ_DATA_PATH. New cache
directories live outside the checkout; existing directories are preserved.
This does not run SAPQ step optimization or
change its loss/config. Missing historical caches yield a new seeded draw,
not recovery of the author's original calibration observations.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import importlib
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys
import types


def link_directory(path, target):
    """Replace broken symlinks only; preserve existing cache directories."""
    path, target = Path(path), Path(target).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        return path
    target.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        print(f"Relink {path}: {os.readlink(path)} -> {target}", flush=True)
        path.unlink()  # Only the symlink; never delete files at its target.
    if not path.exists():
        path.symlink_to(target, target_is_directory=True)
    elif not path.is_dir():
        raise FileExistsError(f"Expected a directory: {path}")
    return path


def link_file(path, target):
    path, target = Path(path), Path(target).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() and not path.exists():
        path.unlink()
    if not path.exists():
        path.symlink_to(target)
    return path


def load_poseidon_sensitivity_helpers(root):
    """Load original Poseidon code without unrelated BRECQ.models imports.

    BRECQ.quant.__init__ eagerly imports ImageNet models absent from main.
    In this preparation process only, expose the real quant_layer module and
    main's exact BaseQuantBlock class. Poseidon wrappers, quantizers, Sobolev
    formula, sensitivity reduction and mapping still use main's source files.
    No source files are rewritten and no placeholder network classes are used.
    """
    import torch
    from torch import nn

    package = types.ModuleType("BRECQ.quant")
    package.__path__ = [str(root / "BRECQ/quant")]
    sys.modules["BRECQ.quant"] = package
    quant = importlib.import_module("BRECQ.quant.quant_layer")
    file = root / "BRECQ/quant/quant_block.py"
    tree = ast.parse(file.read_text())
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseQuantBlock")
    block = types.ModuleType("BRECQ.quant.quant_block")
    block.__file__ = str(file)
    block.__dict__.update(torch=torch, nn=nn, QuantModule=quant.QuantModule,
                          UniformAffineQuantizer=quant.UniformAffineQuantizer,
                          StraightThrough=quant.StraightThrough)
    exec(compile(ast.Module(body=[base], type_ignores=[]), str(file), "exec"), block.__dict__)
    sys.modules[block.__name__] = block
    return importlib.import_module("SAPQ.sapq_sobo_sensitivity")


def compute_sensitivity(model, batches, dataset, cfg, root, device):
    import torch
    from tqdm.auto import tqdm

    helpers = load_poseidon_sensitivity_helpers(root)
    qmodel = helpers.PoseidonQuantModel(model).to(device).eval()
    qmodel.requires_grad_(False)
    blocks = [(name, module) for name, module in qmodel.named_modules()
              if isinstance(module, helpers.TARGET_BLOCK_TYPES)]
    if not blocks:
        raise ValueError("No Poseidon structural blocks were found for sensitivity.")
    captured, handles = {}, []
    totals, counts = {}, defaultdict(int)
    try:
        for name, module in blocks:
            def capture(module, inputs, output, name=name):
                captured[name] = helpers.get_model_output_tensor(output)
            handles.append(module.register_forward_hook(capture))
        for batch in tqdm(batches, desc="Prepare Sobolev sensitivity", unit="batch"):
            inputs = {key: value.to(device) if torch.is_tensor(value) else value
                      for key, value in batch.items()}
            qmodel.set_quant_state(False, False)
            with torch.no_grad():
                clean = helpers.get_model_output_tensor(qmodel(**inputs)).detach()
            captured.clear()
            qmodel.set_quant_state(True, False)
            inputs["pixel_values"] = inputs["pixel_values"].detach().requires_grad_(True)
            pred = helpers.get_model_output_tensor(qmodel(**inputs))
            loss = helpers.compute_sobolev_h1_loss_denorm(
                pred, clean, dataset, cfg.dataset_name,
                sobolev_weight=float(getattr(cfg, "sobolev_weight", 1.0)),
                transpose=bool(getattr(cfg, "sobolev_transpose", False)))
            entries = list(captured.items())
            grads = torch.autograd.grad(loss, [value for _, value in entries], allow_unused=True)
            for (name, _), grad in zip(entries, grads):
                if grad is None:
                    raise ValueError(f"No sensitivity gradient reached {name}.")
                value = helpers.reduce_channel_sensitivity(grad).detach().cpu()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite sensitivity for {name}.")
                totals[name] = totals.get(name, torch.zeros_like(value)) + value
                counts[name] += 1
            captured.clear()
            del pred, clean, loss, grads, entries, inputs
        raw = {name: value / counts[name] for name, value in totals.items()}
        normalized = helpers.normalize_blockwise_minmax(raw)
        return {
            "block_sensitivity_raw": raw, "block_sensitivity_norm": normalized,
            "layer_sensitivity_raw": helpers.build_block_sensitivity_to_inner_layer_dict(qmodel, raw),
            "layer_sensitivity_norm": helpers.build_block_sensitivity_to_inner_layer_dict(qmodel, normalized),
        }
    finally:
        for handle in handles:
            handle.remove()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frozen-only", action="store_true", help="Only create layer list and calibration batches.")
    args = parser.parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    if not (root / "SAPQ/run_sapq_network_global.py").is_file():
        parser.error("Run from the main checkout (or pass --repo-root); no branch switch is performed.")
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from PPQ.config import PPQConfig
    from scOT.problems.base import get_dataset

    cfg = PPQConfig()
    cache = (args.cache_root or root.parent / "sapq-input-cache").expanduser().resolve() / cfg.dataset_name
    if cache == root or root in cache.parents:
        parser.error("--cache-root must be outside the Git checkout to survive branch switches.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    layer_pt = root / "inspect_layers/L_quantize_layers.pt"
    if not layer_pt.exists():
        names = json.loads(layer_pt.with_suffix(".json").read_text())
        torch.save({"quantize_layers": names}, layer_pt)
        print(f"Created layer list: {layer_pt}", flush=True)

    artifacts = link_directory(root / "ppq_artifacts", cache / "ppq_artifacts")
    frozen = artifacts / "frozen_calibration_batches.pt"
    nested_frozen = artifacts / f"{Path(cfg.data_path).name}-calib" / frozen.name
    if not frozen.exists() and nested_frozen.is_file():
        # The original cache producer saved here, whereas global reads above.
        link_file(frozen, nested_frozen)
    dataset = get_dataset(dataset=cfg.dataset_name, which="train", num_trajectories=2048,
                          data_path=cfg.data_path)
    if frozen.is_file():
        print(f"Reuse calibration: {frozen.resolve()}", flush=True)
        batches = torch.load(frozen, map_location="cpu", weights_only=False)
    else:
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=False)
        from tqdm.auto import tqdm
        batches = list(tqdm(islice(loader, 512), total=512, desc="Freeze calibration", unit="batch"))
        if len(batches) != 512 or sum(len(batch["pixel_values"]) for batch in batches) != 1024:
            raise ValueError("Expected 512 batches of two samples from the original training capacity.")
        torch.save(batches, frozen)
        meta = {"dataset": cfg.dataset_name, "seed": args.seed, "num_samples": 1024,
                "batch_size": 2, "num_batches": 512, "num_trajectories": 2048,
                "time_pairs": getattr(dataset, "time_indices", None),
                "data_path": str(Path(cfg.data_path).resolve()), "historical_cache": False}
        (artifacts / "calibration_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Created calibration: {frozen.resolve()}", flush=True)
    if not isinstance(batches, list) or not batches:
        raise ValueError("Frozen calibration file must contain a nonempty list of batches.")
    # Original global reads the top-level file; its sensitivity producer reads
    # <data-directory-name>-calib/frozen_calibration_batches.pt. Supply both.
    link_file(nested_frozen, frozen)
    if not nested_frozen.samefile(frozen):
        raise FileExistsError(f"Different calibration files exist at {frozen} and {nested_frozen}; "
                              "choose one cache before preparing sensitivity.")
    print(f"Calibration ready: {len(batches)} batches, "
          f"{sum(len(batch['pixel_values']) for batch in batches)} samples", flush=True)
    if args.frozen_only:
        return

    tag = Path(cfg.model_path).name
    sensitivity_dir = link_directory(root / "SAPQ/prior_sensitivity_sobo" / tag,
                                     cache / "sensitivity" / tag)
    sensitivity_file = sensitivity_dir / "prior_sensitivity.pt"
    if sensitivity_file.exists():
        saved = torch.load(sensitivity_file, map_location="cpu", weights_only=False)
        if "layer_sensitivity_norm" not in saved:
            raise ValueError(f"No layer_sensitivity_norm in {sensitivity_file}")
        print(f"Reuse sensitivity: {sensitivity_file.resolve()}", flush=True)
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu for a CPU preparation run.")
        from scOT.model import ScOT
        torch.set_float32_matmul_precision("high")
        model = ScOT.from_pretrained(cfg.model_path, local_files_only=True).to(device).eval()
        saved = compute_sensitivity(model, batches, dataset, cfg, root, device)
        saved["meta"] = {"dataset_name": cfg.dataset_name, "model_path": cfg.model_path,
                         "seed": args.seed, "num_calibration_batches": len(batches),
                         "loss_for_sensitivity": "sobolev_h1_denorm(final_quant_output, final_fp_output)",
                         "normalization": "per-block minmax", "sobolev_order": 1}
        torch.save(saved, sensitivity_file)
        print(f"Created sensitivity: {sensitivity_file.resolve()}", flush=True)
    print("Ready: python -u -m SAPQ.run_sapq_network_global", flush=True)
    print("This prepares missing inputs using a new seeded sample selection; it does not recover historical caches.",
          flush=True)


if __name__ == "__main__":
    main()

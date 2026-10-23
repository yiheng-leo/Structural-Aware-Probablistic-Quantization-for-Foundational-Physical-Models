# Historical diagnostic helpers; new quantization runs do not use these caches.
# SAPQ/run_sapq_network_global.py
from __future__ import annotations

import json
from pathlib import Path
import copy
import os

import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import (
    load_poseidon_model,
    build_poseidon_loaders,
)
from PPQ.ranges import (
    load_precalculated_ranges_if_exists,
    compute_data_ranges_poseidon,
)
from PPQ.metrics import (
    evaluate_with_stepsizes,
    compute_dynamic_stepsizes,
)
from SAPQ.sapq_trainer_global import SAPQTrainerGlobal


def load_candidate_layers(model, quant_layer_path: Path):
    """
    Load candidate quantization layer names and keep only raw-model nn.Linear layers.
    Supports:
      - .pt with {"quantize_layers": [...]}
      - .json as list
      - .json with {"quantize_layers": [...]}
    """
    if not quant_layer_path.exists():
        raise FileNotFoundError(f"Quant layer file not found: {quant_layer_path}")

    if quant_layer_path.suffix == ".pt":
        obj = torch.load(quant_layer_path, map_location="cpu")
        if isinstance(obj, dict) and "quantize_layers" in obj:
            layer_names = obj["quantize_layers"]
        else:
            raise ValueError(f"Unsupported PT quant layer format: {quant_layer_path}")
    elif quant_layer_path.suffix == ".json":
        with open(quant_layer_path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            layer_names = obj
        elif isinstance(obj, dict) and "quantize_layers" in obj:
            layer_names = obj["quantize_layers"]
        else:
            raise ValueError(f"Unsupported JSON quant layer format: {quant_layer_path}")
    else:
        raise ValueError(f"Unsupported quant layer file type: {quant_layer_path}")

    name2mod = dict(model.named_modules())
    candidate_layers = [
        name for name in layer_names
        if isinstance(name2mod.get(name, None), nn.Linear)
    ]

    print(f"[INFO] Loaded {len(candidate_layers)} candidate Linear layers.")
    return candidate_layers


def load_sapq_sensitivity(cfg: PPQConfig, device: torch.device):
    """
    Load precomputed SAPQ sensitivity from:
      SAPQ/prior_sensitivity/<model_name>/prior_sensitivity.pt

    Uses:
      save_obj["layer_sensitivity_norm"]
    which is channel-wise sensitivity in ORIGINAL PPQ namespace.
    """
    model_name = Path(cfg.model_path).name
    sens_path = (
        Path(cfg.repo_root)
        / "SAPQ"
        / "prior_sensitivity_sobo"       # or for divergence version ,it shoudl be "prior_sensitivity_div"
        / model_name
        / "prior_sensitivity.pt"
    )

    if not sens_path.exists():
        raise FileNotFoundError(
            f"SAPQ sensitivity file not found: {sens_path}\n"
            f"Please run SAPQ/sapq_sensitivity.py first."
        )

    print(f"[INFO] Loading SAPQ sensitivity from: {sens_path}")
    obj = torch.load(sens_path, map_location="cpu")

    if "layer_sensitivity_norm" not in obj:
        raise ValueError(f"'layer_sensitivity_norm' not found in: {sens_path}")

    sens_dict = {
        name: tensor.to(device)
        for name, tensor in obj["layer_sensitivity_norm"].items()
    }

    print(f"[INFO] Loaded sensitivity for {len(sens_dict)} layers.")
    return sens_dict


def maybe_load_or_compute_ranges(cfg, model, frozen_iter, candidate_layers, device):
    ranges_dict = load_precalculated_ranges_if_exists(
        model_path=cfg.model_path,
        percentile_prob=cfg.percentile_prob,
        repo_root=cfg.repo_root,
        device=device,
    )

    if ranges_dict is None:
        print(f"[INFO] Computing ranges with percentile_prob={cfg.percentile_prob} ...")
        ranges_dict = compute_data_ranges_poseidon(
            model=model,
            dataloader=frozen_iter,
            device=device,
            layer_names=candidate_layers,
            percentile_prob=cfg.percentile_prob,
        )
        for name, value in ranges_dict.items():
            value["weight_ranges"] = value["weight_ranges"].to(device)
            value["activation_ranges"] = value["activation_ranges"].to(device)
    else:
        print("[INFO] Using cached precalculated ranges.")

    return ranges_dict


def load_frozen_calibration_batches(cfg, device: torch.device):
    """
    Load pre-frozen calibration batches from:
        <repo_root>/ppq_artifacts/frozen_calibration_batches.pt

    Expected format:
        list[dict] with keys:
            pixel_values, labels, time, pixel_mask
    """
    frozen_path = Path(cfg.repo_root) / "ppq_artifacts" / "frozen_calibration_batches.pt"

    if not frozen_path.exists():
        raise FileNotFoundError(
            f"Frozen calibration file not found: {frozen_path}"
        )

    print(f"[INFO] Loading frozen calibration batches from: {frozen_path}")
    frozen_batches = torch.load(frozen_path, map_location="cpu")

    if not isinstance(frozen_batches, list):
        raise ValueError(
            f"Expected frozen_batches to be a list, got {type(frozen_batches)}"
        )
    if len(frozen_batches) == 0:
        raise ValueError("Frozen calibration batch list is empty.")

    required_keys = {"pixel_values", "labels", "time", "pixel_mask"}
    first_batch = frozen_batches[0]

    if not isinstance(first_batch, dict):
        raise ValueError(
            f"Expected each frozen batch to be a dict, got {type(first_batch)}"
        )

    missing = required_keys - set(first_batch.keys())
    if missing:
        raise ValueError(
            f"Frozen calibration batch missing keys: {missing}"
        )

    print(f"[INFO] Loaded {len(frozen_batches)} frozen calibration batches.")
    return frozen_batches



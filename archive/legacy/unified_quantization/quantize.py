"""Load calibration data, quantize, and save; evaluation is a separate command."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time


METHODS = (
    "fp", "uniform", "dynamic", "static", "ppq", "ppq-layerwise",
    "sapq", "sapq-block", "sapq-layerwise", "sapq-network", "brecq", "repq", "ptq4vit",
)
PROBABILISTIC_METHODS = {"ppq", "ppq-layerwise", "sapq", "sapq-block", "sapq-layerwise", "sapq-network"}


def build_parser(default_method=None, defaults=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default=default_method,
                        required=default_method is None)
    parser.add_argument("--model_name", default="Poseidon-L")
    parser.add_argument("--dataset", default=os.environ.get(
        "PPQ_DATASET_NAME", "fluids.incompressible.PiecewiseConstants"))
    parser.add_argument("--traj_num", type=int, default=64)
    parser.add_argument("--random_calibration", choices=("y", "n"), default="n")
    parser.add_argument("--calibration_seed", type=int, default=42)
    parser.add_argument("--final_time", type=int, default=20)
    parser.add_argument("--time_back", type=int, default=0)
    parser.add_argument("--use_phyloss", default="no_phyloss")
    parser.add_argument("--DEV", default="cuda")
    parser.add_argument("--model_path", default=os.environ.get("PPQ_MODEL_PATH"))
    parser.add_argument("--data_path", default=os.environ.get(
        "PPQ_DATA_PATH", "poseidon_main/camlab-ethz/down_streams"))
    parser.add_argument("--save_path")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Memory batch size; every selected trajectory/time pair is consumed.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="Quantization RNG seed (independent of trajectory selection).")
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--activation_bits", type=int, default=32,
                        help="32 keeps activations floating point; method support is checked explicitly.")
    parser.add_argument("--init_bits", type=int, default=9)
    parser.add_argument("--target_bits", type=float, default=8)
    parser.add_argument("--epochs", type=int, default=21)
    parser.add_argument("--mc_samples", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=9.1e-4)
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--prior_mode", choices=("ppq", "block_no_sens", "block_sens"),
                        default=os.environ.get("SAPQ_PRIOR_MODE", "block_sens"))
    parser.add_argument("--prior_scale", type=float, default=1e-4)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.005)
    parser.add_argument("--sensitivity", choices=("sobolev", "mse", "divergence"), default="sobolev")
    parser.add_argument("--sensitivity_normalization", choices=("raw", "normalized"), default="raw")
    parser.add_argument("--percentile", type=float, default=1e-4)
    parser.add_argument("--bmax_bits", type=int, default=20)
    parser.add_argument("--avg_cap_bits", type=float, default=8.0)
    parser.add_argument("--avg_cap_lam", type=float, default=400.0)
    parser.add_argument("--avg_cap_alpha", type=float, default=10.0)
    parser.add_argument("--reconstruction_iterations", type=int, default=20000)
    parser.add_argument("--rec_batchsize", type=int, default=32)
    parser.add_argument("--ptq_candidates", type=int, default=100)
    parser.add_argument("--ptq_search_rounds", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_work_dir", action="store_true",
                        help="Keep this run's temporary calibration/reconstruction caches.")
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def validate_args(args):
    for name in ("traj_num", "batch_size", "epochs", "mc_samples", "rec_batchsize", "ptq_candidates", "ptq_search_rounds"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    if args.num_workers < 0 or args.reconstruction_iterations < 1:
        raise ValueError("num_workers must be nonnegative and reconstruction_iterations positive")
    if not 2 <= args.weight_bits <= 16 or args.activation_bits not in (*range(2, 17), 32):
        raise ValueError("weight_bits must be 2..16; activation_bits must be 2..16 or 32")
    if not 0 < args.percentile < 0.5 or args.eta <= 0 or args.learning_rate <= 0:
        raise ValueError("Require 0 < percentile < 0.5 and positive eta/learning_rate")
    if not 0 < args.target_bits <= args.bmax_bits or not 1 <= args.init_bits <= args.bmax_bits:
        raise ValueError("target_bits/init_bits must be positive and no greater than bmax_bits")
    if args.bmax_bits > 30:
        raise ValueError("bmax_bits must not exceed 30")
    if args.sigma0 <= 0 or args.prior_scale < 0 or args.alpha < 0 or args.gamma < 0:
        raise ValueError("sigma0 must be positive; prior_scale, alpha and gamma must be nonnegative")
    if args.method in PROBABILISTIC_METHODS and args.activation_bits != 32:
        raise ValueError("PPQ/SAPQ currently implement weight-only quantization; use --activation_bits 32")
    if args.model_path is None:
        args.model_path = str(Path("checkpoints") / args.model_name / args.use_phyloss
                              / args.dataset / "origin" / "none")
    if args.save_path is None:
        tag = f"traj{args.traj_num}"
        if args.random_calibration == "y":
            tag += f"_random_seed{args.calibration_seed}"
        tag += f"_ftime{args.final_time}_timeback{args.time_back}"
        budget = (f"init{args.init_bits}_target{args.target_bits:g}"
                  if args.method in PROBABILISTIC_METHODS
                  else f"w{args.weight_bits}a{args.activation_bits}")
        if args.method.startswith("sapq"):
            budget += f"_{args.prior_mode}_{args.sensitivity}_{args.sensitivity_normalization}"
        args.save_path = str(Path("quantization_artifacts") / args.model_name / args.dataset
                             / args.method / f"{tag}_{budget}_seed{args.seed}")
    return args


def _load_model(path, device):
    import torch
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint is unavailable: {path}. Set --model_path explicitly.")
    if path.is_file():
        model = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("A model file must contain an nn.Module; use a pretrained directory for state dicts.")
    else:
        from scOT.model import ScOT
        model = ScOT.from_pretrained(str(path), local_files_only=True)
    return model.to(device).eval()


def _revision():
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _implementation_hashes():
    """Identify the actual source even when running uncommitted branch changes."""
    root = Path(__file__).resolve().parents[1]
    files = set((root / "quantization").glob("*.py"))
    for directory in (root / "BRECQ" / "quant", root / "PPQ", root / "scOT"):
        files.update(directory.rglob("*.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def run_quantization(args, *, model=None, calibration_loader=None, calibration_metadata=None):
    """Single production path, also accepting injected models/data for integration tests."""
    import numpy as np
    import torch
    from .artifacts import save_quantized_artifact
    from .calibration import build_calibration_loader
    from .common import resolve_device

    args = validate_args(args)
    device = resolve_device(args.DEV)
    destination = Path(args.save_path)
    if (destination / "compressed_model.pt").exists() and not args.overwrite:
        raise FileExistsError(f"Artifact already exists: {destination}; choose a new --save_path or --overwrite")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    started = time.perf_counter()
    if calibration_loader is None:
        _, calibration_loader, calibration_metadata = build_calibration_loader(args)
    elif calibration_metadata is None:
        raise ValueError("Injected calibration_loader requires calibration_metadata")
    if model is None:
        model = _load_model(args.model_path, device)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    destination.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="work-", dir=destination))
    print("Calibration: " + json.dumps({key: calibration_metadata.get(key) for key in
          ("dataset", "num_trajectories", "num_samples", "selection", "fingerprint")}), flush=True)
    print(f"Quantizing with {args.method} on {device}; output: {destination}", flush=True)
    succeeded = False
    try:
        if args.method in PROBABILISTIC_METHODS:
            from .probabilistic import quantize_probabilistic
            result = quantize_probabilistic(model, calibration_loader, args, device, work_dir)
        else:
            from .baselines import quantize_baseline
            result = quantize_baseline(model, calibration_loader, args, device, work_dir)
        metadata = {
            "method": args.method,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": _revision(),
            "implementation_sha256": _implementation_hashes(),
            "source_model_path": str(Path(args.model_path).resolve()),
            "settings": vars(args).copy(),
            "calibration": calibration_metadata,
            "torch_version": torch.__version__,
            "elapsed_quantization_seconds": time.perf_counter() - started,
            "evaluation_performed": False,
        }
        metadata = save_quantized_artifact(result, destination, metadata, overwrite=args.overwrite)
        succeeded = True
    finally:
        if succeeded and not args.keep_work_dir:
            shutil.rmtree(work_dir)
        elif work_dir.exists():
            print(f"Work files retained at {work_dir}", flush=True)
    print(f"Saved {destination / 'compressed_model.pt'}", flush=True)
    print("Evaluate separately with: python -m quantization.evaluate --model_path "
          f"'{destination}' --dataset '{args.dataset}' --data_path '{args.data_path}' --DEV {args.DEV}", flush=True)
    return destination, metadata


def main(argv=None, *, default_method=None, defaults=None):
    parser = build_parser(default_method, defaults)
    args = parser.parse_args(argv)
    try:
        run_quantization(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

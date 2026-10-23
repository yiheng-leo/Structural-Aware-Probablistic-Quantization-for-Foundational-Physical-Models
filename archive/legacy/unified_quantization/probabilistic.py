"""Disk-streamed PPQ/SAPQ calibration, returning ordinary materialized models.

The objectives follow PPQ/loss.py and SAPQ/sapq_loss.py: PPQ uses mean MC
reconstruction, SAPQ uses a log-mean-exp likelihood and a selectable bit prior.
SAPQ's actual training convention is sigma = sigma0 * (1 + alpha * sensitivity),
not the different variance convention in analysis.py. We reuse the independent
PPQ kernels; the small SAPQ formulas are kept here to avoid importing the legacy
BRECQ wrappers and their model/package dependencies.

Only the supplied calibration loader is consumed. No old ranges, step-size,
sensitivity or layer-IO caches are read, and no evaluation loader is constructed.
Temporary batches and one reconstruction unit's IO live on disk, not the GPU.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile

import torch
from torch import nn
import torch.nn.functional as F

from PPQ.loss import compute_mdl_prior, prior_weighted_avg_bits_cap_poseidon
from PPQ.noise import add_quantization_noise
from PPQ.optimize import get_lr_for_epoch
from quantization.common import (
    QuantizationResult, forward_model, move_to_device, output_tensor,
    select_quantizable_layers,
)


_METHODS = {"ppq", "ppq-layerwise", "sapq", "sapq-block", "sapq-network", "sapq-layerwise"}
_BLOCK_TYPES = {"ScOTLayer", "ConvNeXtBlock", "ResNetBlock"}
_EPS = 1e-8


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return value


def _inputs(batch):
    """Calibration targets are never used, even to compute the model's own loss."""
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        return {key: batch[key] for key in ("pixel_values", "time", "pixel_mask")
                if key in batch and batch[key] is not None}
    raise TypeError("Calibration batches must be tensors, ScOT dicts, or (input, target).")


def _num_samples(batch):
    tensor = batch if torch.is_tensor(batch) else batch.get("pixel_values")
    if not torch.is_tensor(tensor) or tensor.ndim == 0:
        raise ValueError("Calibration batch has no batched input tensor.")
    return int(tensor.shape[0])


def _requires_input_grad(batch, device):
    batch = move_to_device(batch, device)
    x = batch if torch.is_tensor(batch) else batch["pixel_values"]
    if not x.is_floating_point():
        raise ValueError("Physical model calibration inputs must be floating point.")
    x = x.detach().requires_grad_(True)
    return x if torch.is_tensor(batch) else {**batch, "pixel_values": x}


def _load(path):
    # These files are tensor-only scratch data created by this invocation.
    return torch.load(path, map_location="cpu", weights_only=True)


@contextmanager
def _scratch_directory(parent, prefix, keep):
    if keep:
        yield Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    else:
        with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as directory:
            yield Path(directory)


def _freeze_calibration(model, loader, directory, device):
    """Consume the loader once, retaining every batch including its partial tail."""
    paths, samples = [], 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            batch = _cpu(_inputs(batch))
            count = _num_samples(batch)
            if count == 0:
                raise ValueError("Empty calibration batches are unsupported.")
            clean = _cpu(forward_model(model, batch, device))
            if not torch.isfinite(clean).all():
                raise ValueError(f"Non-finite FP prediction in calibration batch {index}.")
            path = directory / f"batch_{index:06d}.pt"
            torch.save({"batch": batch, "clean": clean}, path)
            paths.append(path)
            samples += count
    if not paths:
        raise ValueError("Calibration loader is empty.")
    return paths, samples


def _weight_ranges(layers, percentile):
    """Original PPQ Gaussian range: 2 * Phi^-1(1-p) * per-channel std."""
    z = math.sqrt(2.0) * torch.special.erfinv(
        torch.tensor(1.0 - 2.0 * percentile, dtype=torch.float64)).item()
    return {name: {"weight_ranges": (2.0 * z * layer.weight.detach().float().std(
        dim=1, unbiased=False)).clamp_min(_EPS)} for name, layer in layers.items()}


@dataclass
class _Unit:
    name: str
    module: nn.Module
    layers: dict


def _block_units(model, layers):
    """Preserve ScOT residual/attention blocks; cover non-block Linear layers too."""
    modules = dict(model.named_modules())
    blocks = {name: mod for name, mod in modules.items()
              if type(mod).__name__ in _BLOCK_TYPES}
    groups = {}
    for name, layer in layers.items():
        ancestors = [block for block in blocks if name.startswith(block + ".") or block == ""]
        owner = max(ancestors, key=len) if ancestors else name
        groups.setdefault(owner, {})[name] = layer
    return [_Unit(name, modules[name], groups[name]) for name in modules if name in groups]


def _first_tensor(output):
    if isinstance(output, (tuple, list)):
        return _first_tensor(output[0])
    return output_tensor(output)


@contextmanager
def _weight_hooks(layers, steps, *, noisy):
    """Functional weights retain gradients to steps; Parameter replacement does not."""
    handles = []
    try:
        for name, module in layers.items():
            step = steps[name]

            def hook(mod, inputs, output, step=step):
                weight = mod.weight
                if noisy:
                    weight = add_quantization_noise(weight, step.to(weight.dtype), 0)
                else:
                    scale = step.to(weight.dtype).view(-1, 1)
                    weight = torch.round(weight / scale) * scale
                return F.linear(inputs[0], weight, mod.bias)

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _reference_quantizers(model, bits):
    """Legacy PoseidonQuantModel reference: affine per-tensor Linear/Conv weights.

    This reference is used only for sensitivity/Fisher/asymmetric inputs. The
    learned and exported quantizer remains selected Linear, per-output-channel.
    """
    modules = dict(model.named_modules())
    excluded = [name for name, mod in modules.items() if isinstance(mod, (nn.LayerNorm, nn.BatchNorm2d))
                or type(mod).__name__ in {"LayerNorm", "ConditionalLayerNorm"}]
    layers = {name: mod for name, mod in modules.items() if isinstance(mod, (nn.Linear, nn.Conv2d))
              and not any(name == skip or name.startswith(skip + ".") for skip in excluded)}
    quantizers = {}
    for name, layer in layers.items():
        weight = layer.weight.detach()
        lower, upper = min(weight.min().item(), 0.0), max(weight.max().item(), 0.0)
        delta = max((upper - lower) / (2 ** bits - 1), _EPS)
        quantizers[name] = (weight.new_tensor(delta), round(-lower / delta), 2 ** bits - 1)
    return layers, quantizers


@contextmanager
def _reference_hooks(layers, quantizers):
    handles = []
    try:
        for name, layer in layers.items():
            delta, zero, maximum = quantizers[name]

            def hook(module, inputs, output, delta=delta, zero=zero, maximum=maximum):
                quantized = ((torch.round(module.weight / delta) + zero).clamp(0, maximum) - zero) * delta
                if isinstance(module, nn.Conv2d):
                    return module._conv_forward(inputs[0], quantized, module.bias)
                return F.linear(inputs[0], quantized, module.bias)

            handles.append(layer.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _reference_prefix(model, unit, layers):
    names = list(dict(model.named_modules()))
    end = max(index for index, name in enumerate(names)
              if name == unit.name or name.startswith(unit.name + ".") or unit.name == "")
    return {name: layers[name] for name in names[:end + 1] if name in layers}


def _dataset_constants(loader):
    dataset = getattr(loader, "dataset", None)
    seen = set()
    while dataset is not None and id(dataset) not in seen:
        seen.add(id(dataset))
        constants = getattr(dataset, "constants", None)
        if constants is not None:
            return constants
        dataset = getattr(dataset, "source_dataset", getattr(dataset, "dataset", None))
    return None


def _structure_loss(pred, clean, mode, constants, dataset_name):
    if mode == "mse":
        return F.mse_loss(pred, clean)
    if pred.ndim != 4:
        raise ValueError(f"{mode} sensitivity requires [B,C,H,W] predictions.")
    if mode == "divergence":
        # Matches SAPQ/sapq_sensitivity.py: divergence in model-output units.
        channels = pred.shape[1]
        u, v = (1, 2) if channels >= 4 else (0, 1)
        if channels < 2 or min(pred.shape[-2:]) < 3:
            raise ValueError("Divergence sensitivity requires two velocity channels and H,W >= 3.")

        def divergence(field):
            dx, dy = 1.0 / (field.shape[-1] - 1), 1.0 / (field.shape[-2] - 1)
            return ((field[:, u, 1:-1, 2:] - field[:, u, 1:-1, :-2]) / (2 * dx)
                    + (field[:, v, 2:, 1:-1] - field[:, v, :-2, 1:-1]) / (2 * dy))

        return F.mse_loss(pred, clean) + F.mse_loss(divergence(pred), divergence(clean))
    if mode != "sobolev":
        raise ValueError(f"Unknown sensitivity mode: {mode}")
    # Preserve the existing discrete H1 convention: forward differences without
    # division by grid spacing. Constant means cancel in prediction differences.
    if constants is not None:
        std = torch.as_tensor(constants["std"], device=pred.device, dtype=pred.dtype)
        if std.numel() not in (1, pred.shape[1]):
            raise ValueError("Legacy Sobolev sensitivity std does not cover the output channels; "
                             "use mse sensitivity for this dataset or supply matching constants.")
        if std.numel() > 1:
            std = std.view(1, -1, 1, 1)
        pred, clean = pred * std, clean * std
    if ("incompressible" in dataset_name.lower() or dataset_name.lower().startswith("ns")):
        if pred.shape[1] == 4:
            pred, clean = pred[:, 1:], clean[:, 1:]
    if min(pred.shape[-2:]) < 2:
        raise ValueError("Sobolev sensitivity requires H,W >= 2.")
    return (F.mse_loss(pred, clean)
            + F.mse_loss(pred[..., 1:] - pred[..., :-1], clean[..., 1:] - clean[..., :-1])
            + F.mse_loss(pred[..., 1:, :] - pred[..., :-1, :],
                         clean[..., 1:, :] - clean[..., :-1, :]))


def _channel_sensitivity(grad):
    dims = (0, 2, 3) if grad.ndim == 4 else tuple(range(grad.ndim - 1))
    return grad.square().mean(dim=dims)


def _sensitivity(model, units, layers, reference_steps, paths, device, args, constants):
    """One backward pass per batch captures every block, keeping only channel sums."""
    totals, counts = {}, {}
    for path in paths:
        record = _load(path)
        captured, handles = {}, []
        try:
            batch = _requires_input_grad(record["batch"], device)
            with _reference_hooks(layers, reference_steps):
                # Capture after the weight hook when a unit is itself Linear.
                for unit in units:
                    def capture(mod, inputs, output, name=unit.name):
                        if name in captured:
                            raise ValueError(f"Repeated reconstruction unit {name!r} is unsupported.")
                        captured[name] = _first_tensor(output)
                    handles.append(unit.module.register_forward_hook(capture))
                pred = forward_model(model, batch, device)
            loss = _structure_loss(pred, record["clean"].to(device), args.sensitivity,
                                   constants, getattr(args, "dataset", ""))
            active = [(name, tensor) for name, tensor in captured.items() if tensor.requires_grad]
            if not active:
                raise ValueError("No differentiable reconstruction outputs were captured.")
            grads = torch.autograd.grad(loss, [value for _, value in active], allow_unused=True)
            for (name, _), grad in zip(active, grads):
                if grad is None:
                    continue
                value = _channel_sensitivity(grad).detach().cpu()
                totals[name] = totals.get(name, torch.zeros_like(value)) + value
                counts[name] = counts.get(name, 0) + 1
        finally:
            for handle in handles:
                handle.remove()
    raw, normalized = {}, {}
    for unit in units:
        if unit.name not in totals:
            raise ValueError(f"No sensitivity gradient reached reconstruction unit {unit.name!r}.")
        sens = totals[unit.name] / counts[unit.name]
        norm = (sens - sens.min()) / (sens.max() - sens.min() + _EPS)
        for name, layer in unit.layers.items():
            raw[name] = sens.clone() if sens.numel() == layer.out_features else sens.mean().repeat(layer.out_features)
            normalized[name] = norm.clone() if norm.numel() == layer.out_features else norm.mean().repeat(layer.out_features)
    return raw, normalized


def _prior(steps, ranges, sensitivity, args, *, is_ppq):
    mode = "ppq" if is_ppq else args.prior_mode
    if mode == "ppq":
        return compute_mdl_prior({name: (value, None) for name, value in steps.items()},
                                 ranges, gamma=args.gamma)
    if mode not in {"block_sens", "block_no_sens"}:
        raise ValueError(f"Unknown SAPQ prior mode: {mode}")
    prior = next(iter(steps.values())).new_zeros(())
    for name, step in steps.items():
        bits = torch.log2(ranges[name]["weight_ranges"].clamp_min(_EPS) / step.clamp_min(_EPS))
        sens = sensitivity[name].to(step.device) if mode == "block_sens" else torch.zeros_like(step)
        sigma = (args.sigma0 * (1.0 + args.alpha * sens)).clamp_min(_EPS)
        prior = prior + ((bits - args.target_bits).square() / (2.0 * sigma.square())).sum()
    return args.prior_scale * prior


def _clamp(steps, ranges, max_bits):
    with torch.no_grad():
        for name, step in steps.items():
            maximum = ranges[name]["weight_ranges"]
            step.clamp_(min=(maximum / (2.0 ** max_bits)).clamp_min(_EPS), max=maximum)


def _average_bits(steps, ranges, layers):
    with torch.no_grad():
        total, parameters = 0.0, 0
        for name, step in steps.items():
            bits = torch.log2(ranges[name]["weight_ranges"].clamp_min(_EPS) / step.clamp_min(_EPS))
            total += float(bits.sum()) * layers[name].in_features
            parameters += layers[name].weight.numel()
    return total / parameters


@contextmanager
def _seed(seed, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        elif device.type == "mps":
            torch.mps.manual_seed(seed)
        try:
            yield
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


def _energy(pred, target, eta, fisher=None, *, output_sum=False):
    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {tuple(pred.shape)} differs from target {tuple(target.shape)}.")
    error = (pred - target).square()
    if fisher is not None:
        if fisher.shape != pred.shape:
            raise ValueError("Fisher gradients must match the reconstruction output.")
        return 0.5 * (error * fisher.square()).reshape(pred.shape[0], -1).sum(dim=1)
    flat = error.reshape(pred.shape[0], -1)
    return (flat.sum(dim=1) if output_sum else flat.mean(dim=1)) / (2.0 * eta)


def _backward_likelihood(forward, target, args, device, *, marginal, fisher=None, loss_scale=1.0):
    """Exact log-mean-exp gradient with only one MC computation graph resident.

    A no-grad pass obtains mixture responsibilities; replaying each seed gives
    identical noise and the weighted energy gradient. This is algebraically the
    gradient of the legacy stacked MC loss, not a changed likelihood estimator.
    """
    seeds = torch.randint(0, 2 ** 31 - 1, (args.mc_samples,)).tolist()
    if not marginal:
        loss_value = 0.0
        for seed in seeds:
            with _seed(seed, device):
                loss = loss_scale * _energy(forward(), target, args.eta).mean() / args.mc_samples
            loss.backward()
            loss_value += float(loss.detach())
        return loss_value
    scores = []
    with torch.no_grad():
        for seed in seeds:
            with _seed(seed, device):
                scores.append(-_energy(forward(), target, args.eta, fisher,
                                       output_sum=args.method == "sapq-network"))
        scores = torch.stack(scores)
        probabilities = scores.softmax(dim=0)
        loss_value = -loss_scale * (scores.logsumexp(dim=0) - math.log(args.mc_samples)).sum().item()
    for index, seed in enumerate(seeds):
        with _seed(seed, device):
            energy = _energy(forward(), target, args.eta, fisher,
                             output_sum=args.method == "sapq-network")
        (loss_scale * probabilities[index] * energy).sum().backward()
    return loss_value


class _StopCapture(Exception):
    pass


def _capture_unit(model, unit, batch, device, *, stop=True):
    captured = {}

    def capture(mod, inputs, kwargs, output):
        captured.update(args=_cpu(inputs), kwargs=_cpu(kwargs), target=_cpu(_first_tensor(output)))
        if stop:
            raise _StopCapture

    handle = unit.module.register_forward_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            try:
                forward_model(model, batch, device)
            except _StopCapture:
                pass
    finally:
        handle.remove()
    if not captured:
        raise ValueError(f"Reconstruction unit {unit.name!r} did not execute.")
    return captured


def _cache_unit(model, unit, layers, prefix_layers, reference_steps, paths, directory, device, *, block):
    """Keep only one unit's IO; block inputs use a quantized prefix (asym=True)."""
    result = []
    for index, path in enumerate(paths):
        record = _load(path)
        captured = _capture_unit(model, unit, record["batch"], device)
        if block:
            # The target remains the clean FP block output; inputs reflect an
            # independently quantized prefix, as in original BRECQ asym caching.
            with _reference_hooks(layers, reference_steps):
                quantized = _capture_unit(model, unit, record["batch"], device)
            captured.update(args=quantized["args"], kwargs=quantized["kwargs"])
            output = []
            handle = None
            try:
                with _reference_hooks(prefix_layers, reference_steps):
                    handle = unit.module.register_forward_hook(lambda mod, inputs, value: output.append(_first_tensor(value)))
                    pred = forward_model(model, _requires_input_grad(record["batch"], device), device)
                if len(output) != 1:
                    raise ValueError(f"Expected one execution of {unit.name!r}, got {len(output)}.")
                loss = F.mse_loss(pred, record["clean"].to(device))
                grad = torch.autograd.grad(loss, output[0], allow_unused=True)[0]
                if grad is None:
                    raise ValueError(f"No Fisher gradient reached {unit.name!r}.")
                # save_grad_data's default is abs(gradient) + damping=1.0;
                # the block energy squares this damped quantity, not raw grad.
                captured["fisher"] = _cpu(grad.abs() + 1.0)
            finally:
                if handle is not None:
                    handle.remove()
        local_path = directory / f"unit_{index:06d}.pt"
        torch.save(captured, local_path)
        result.append(local_path)
    return result


def _optimize(model, layers, steps, ranges, sensitivity, paths, args, device,
              *, unit=None, block=False, network_unit=None):
    local_layers = (unit.layers if unit is not None else
                    network_unit.layers if network_unit is not None else layers)
    local_steps = {name: steps[name] for name in local_layers}
    optimizer = torch.optim.Adam(list(local_steps.values()), lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        lr = get_lr_for_epoch(epoch, args.learning_rate, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals = {"likelihood": 0.0, "prior": 0.0, "total": 0.0}
        for path in paths:
            record = _load(path)
            target = record["clean" if unit is None else "target"].to(device)
            if unit is None:
                batch = move_to_device(record["batch"], device)
                def forward():
                    with _weight_hooks(local_layers, local_steps, noisy=True):
                        return forward_model(model, batch, device)
            else:
                inputs = move_to_device(record["args"], device)
                kwargs = move_to_device(record["kwargs"], device)
                def forward():
                    with _weight_hooks(local_layers, local_steps, noisy=True):
                        return _first_tensor(unit.module(*inputs, **kwargs))
            optimizer.zero_grad(set_to_none=True)
            likelihood = _backward_likelihood(
                forward, target, args, device,
                marginal=args.method in {"sapq", "sapq-block", "sapq-network"},
                fisher=record["fisher"].to(device) if block else None,
                # Original PPQ layerwise averages likelihood across ALL layers,
                # while MDL is a sum. Its objective is separable; streaming one
                # layer at a time must retain this global divisor.
                loss_scale=1.0 / len(layers) if args.method == "ppq-layerwise" else 1.0,
            )
            if args.method == "ppq":
                # PPQ/trainer.py's active objective excludes MDL and uses the
                # parameter-weighted smooth average-bit cap instead.
                prior = prior_weighted_avg_bits_cap_poseidon(
                    {name: (step, None) for name, step in local_steps.items()}, ranges,
                    {name: step.new_full(step.shape, local_layers[name].in_features)
                     for name, step in local_steps.items()},
                    target_bits=getattr(args, "avg_cap_bits", 8.0),
                    lam=getattr(args, "avg_cap_lam", 400.0),
                    alpha=getattr(args, "avg_cap_alpha", 10.0),
                )
            else:
                prior = _prior(local_steps, ranges, sensitivity, args,
                               is_ppq=args.method == "ppq-layerwise")
            # Preserve the original SAPQ block objective's additional factor.
            prior_weight = 1e-4 if block else 1e-10 if network_unit is not None else 1.0
            (prior_weight * prior).backward()
            if not math.isfinite(likelihood) or not torch.isfinite(prior):
                raise FloatingPointError("Non-finite probabilistic quantization loss.")
            if any(step.grad is None or not torch.isfinite(step.grad).all() for step in local_steps.values()):
                raise FloatingPointError("Missing or non-finite step-size gradients.")
            optimizer.step()
            _clamp(local_steps, ranges, args.bmax_bits)
            totals["likelihood"] += likelihood
            totals["prior"] += float(prior.detach())
            totals["total"] += likelihood + prior_weight * float(prior.detach())
        history.append({"unit": unit.name if unit is not None else (
                            network_unit.name if network_unit is not None else "network"),
                        "epoch": epoch, "learning_rate": lr, "num_batches": len(paths),
                        "avg_bits": _average_bits(steps, ranges, layers),
                        **{key: value / len(paths) for key, value in totals.items()}})
        print(f"[{args.method}] {history[-1]['unit']} epoch {epoch}/{args.epochs} "
              f"loss={history[-1]['total']:.6g} bits={history[-1]['avg_bits']:.4f}", flush=True)
    return history


def _validate(args):
    if args.method not in _METHODS:
        raise ValueError(f"Unknown probabilistic method: {args.method}")
    if args.activation_bits != 32:
        raise ValueError("PPQ/SAPQ adapters support weight-only quantization; activation_bits must be 32.")
    if args.epochs < 1 or args.mc_samples < 1:
        raise ValueError("epochs and mc_samples must be positive.")
    if not (0 < args.percentile < 0.5) or args.eta <= 0 or args.learning_rate <= 0:
        raise ValueError("Require 0 < percentile < 0.5, eta > 0 and learning_rate > 0.")
    if not 0 < args.init_bits <= args.bmax_bits or args.bmax_bits > 30:
        raise ValueError("Require 0 < init_bits <= bmax_bits <= 30.")
    if args.sigma0 <= 0 or args.alpha < 0 or args.prior_scale < 0 or args.gamma < 0:
        raise ValueError("Prior sigma0 must be positive; alpha, prior_scale and gamma must be nonnegative.")
    if getattr(args, "sensitivity_normalization", "raw") not in {"raw", "normalized"}:
        raise ValueError("sensitivity_normalization must be raw or normalized.")


def quantize_probabilistic(model, calibration_loader, args, device, work_dir):
    """Learn steps from all supplied calibration samples and materialize weights.

    Model parameters are frozen during calibration; only channel steps are
    optimized. Returned weights are floating point fake-quantized values, not a
    packed integer representation. Inference needs no hooks or calibration files.
    """
    _validate(args)
    device = torch.device(device)
    model = model.to(device).eval()
    layers = select_quantizable_layers(model)
    if not layers:
        raise ValueError("No eligible Linear layers were found for probabilistic quantization.")
    original_flags = {name: param.requires_grad for name, param in model.named_parameters()}
    for param in model.parameters():
        param.requires_grad_(False)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        keep = getattr(args, "keep_work_dir", False)
        with _seed(args.seed, device), _scratch_directory(work_dir, "probabilistic-", keep) as scratch:
            paths, num_samples = _freeze_calibration(model, calibration_loader, scratch, device)
            ranges = _weight_ranges(layers, args.percentile)
            steps = {name: nn.Parameter(rec["weight_ranges"].clone() / (2.0 ** args.init_bits))
                     for name, rec in ranges.items()}
            _clamp(steps, ranges, args.bmax_bits)
            initial_steps = {name: value.detach().cpu().clone() for name, value in steps.items()}
            units = _block_units(model, layers)
            reference_layers, reference_steps = _reference_quantizers(model, args.weight_bits)
            raw, normalized, sensitivity = {}, {}, {}
            constants = _dataset_constants(calibration_loader)
            if args.method.startswith("sapq") and args.prior_mode == "block_sens":
                raw, normalized = _sensitivity(model, units, reference_layers, reference_steps, paths,
                                                device, args, constants)
                sensitivity = normalized if getattr(args, "sensitivity_normalization", "raw") == "normalized" else raw
            history = []
            if args.method in {"ppq", "sapq"}:
                history = _optimize(model, layers, steps, ranges, sensitivity, paths, args, device)
            else:
                if args.method not in {"sapq-block", "sapq-network"}:
                    units = [_Unit(name, module, {name: module}) for name, module in layers.items()]
                for unit in units:
                    if args.method == "sapq-network":
                        history.extend(_optimize(model, layers, steps, ranges, sensitivity,
                                                 paths, args, device, network_unit=unit))
                        continue
                    with _scratch_directory(scratch, "unit-", keep) as unit_dir:
                        prefix = _reference_prefix(model, unit, reference_layers)
                        unit_paths = _cache_unit(model, unit, reference_layers, prefix, reference_steps,
                                                 paths, Path(unit_dir), device, block=args.method == "sapq-block")
                        history.extend(_optimize(model, layers, steps, ranges, sensitivity,
                                                 unit_paths, args, device, unit=unit,
                                                 block=args.method == "sapq-block"))
            avg_bits = _average_bits(steps, ranges, layers)
            with torch.no_grad():
                for name, module in layers.items():
                    scale = steps[name].to(module.weight.dtype).view(-1, 1)
                    module.weight.copy_(torch.round(module.weight / scale) * scale)
            state = {"steps": _cpu(steps), "initial_steps": initial_steps, "ranges": _cpu(ranges),
                     "sensitivity": sensitivity, "sensitivity_raw": raw,
                     "sensitivity_normalized": normalized, "history": history}
            metadata = {
                "method": args.method, "representation": "materialized_float_fake_quantized_weights",
                "weight_only": True, "activation_bits": 32, "num_layers": len(layers),
                "layer_names": list(layers), "calibration_samples": num_samples,
                "calibration_batches": len(paths), "calibration_storage": "temporary_disk",
                "initial_bits": args.init_bits, "target_bits": args.target_bits,
                "avg_weight_bits": avg_bits, "epochs": args.epochs, "mc_samples": args.mc_samples,
                "likelihood": "mc_log_mean_exp_fisher" if args.method == "sapq-block" else (
                    "mc_log_mean_exp_output_mse" if args.method == "sapq" else (
                        "mc_log_mean_exp_output_sum_squared" if args.method == "sapq-network" else "mc_mean_mse")),
                "prior_mode": "weighted_average_bit_cap" if args.method == "ppq" else (
                    "ppq" if args.method == "ppq-layerwise" else args.prior_mode),
                "gamma_used": args.method == "ppq-layerwise" or (
                    args.method.startswith("sapq") and args.prior_mode == "ppq"),
                "avg_cap_bits": getattr(args, "avg_cap_bits", 8.0) if args.method == "ppq" else None,
                "sensitivity": args.sensitivity if sensitivity else None,
                "sensitivity_normalization": getattr(args, "sensitivity_normalization", "raw"),
                "sensitivity_reference_bits": args.weight_bits,
                "sensitivity_reference_quantizer": "per_tensor_affine_linear_and_conv2d",
                "sobolev_denormalized": args.sensitivity == "sobolev" and constants is not None,
                "sobolev_scaling_protocol": "legacy_dataset_std_broadcast_all_channels",
                "sobolev_differences": "forward_difference_without_grid_spacing",
                "variance_convention": "sigma0^2 * (1 + alpha * sensitivity)^2",
                "block_prior_multiplier": 1e-4 if args.method == "sapq-block" else (
                    1e-10 if args.method == "sapq-network" else 1.0),
                "block_fisher_damping": 1.0 if args.method == "sapq-block" else None,
                "layerwise_likelihood_divisor": len(layers) if args.method == "ppq-layerwise" else 1,
                "evaluated_test_data": False,
            }
            return QuantizationResult(model=model, metadata=metadata, state=state)
    finally:
        for name, param in model.named_parameters():
            param.requires_grad_(original_flags[name])

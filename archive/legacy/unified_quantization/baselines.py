"""Calibration-only baselines and portable fake-quantized Linear modules.

RepQ's scale/shift reparameterization follows
``RepQ-ViT/RepQ_NS-PwC_2048.py``. We retain the input affine explicitly:
ScOT's conditional, post-residual normalizations cannot generally be folded
using timm's pre-norm LayerNorm-to-Linear rewrite.

The PTQ4ViT adaptation follows ``PTQ4ViT/quant_layers/linear.py``: alternating
Hessian-guided interval searches and twin uniform quantization after GELU.
Its Hessian proxy uses the physical regression calibration loss. Both adapters
cover the selected Linear operations, not attention matmul/softmax kernels.
All exported weights and activation scales are self-contained floating-point
quantize/dequantize simulations; no integer speedup or packed storage is claimed.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

from .common import QuantizationResult, forward_model, move_to_device, select_quantizable_layers


def _replace(model, name, module):
    if not name:
        return module
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)
    return model


def _bits(value, name):
    value = int(value)
    if value != 32 and not 2 <= value <= 16:
        raise ValueError(f"{name} must be 2..16 or 32 (unquantized), got {value}")
    return value


def _symmetric_weight(weight, bits, *, grid="symmetric"):
    if bits == 32:
        return weight.detach().clone(), torch.ones(weight.shape[0], device=weight.device)
    maximum = weight.detach().abs().amax(dim=1)
    if grid == "ppq_reference":
        # Preserve PPQ.metrics.compute_dynamic_stepsizes/evaluate_with_stepsizes:
        # the grid has 2^b - 1 intervals across [-max, max], with no clipping.
        scale = (2 * maximum / (2 ** bits - 1)).clamp_min(1e-12)
        return ((weight / scale[:, None]).round() * scale[:, None]).detach(), scale
    qmax = 2 ** (bits - 1) - 1
    # DynamicQ adds epsilon before division, rather than clamping the maximum.
    maximum = maximum + 1e-8 if grid == "dynamic_reference" else maximum.clamp_min(1e-12)
    scale = maximum / qmax
    quantized = (weight / scale[:, None]).round().clamp(-qmax, qmax) * scale[:, None]
    return quantized.detach(), scale


class QuantizedLinear(nn.Linear):
    """Materialized QDQ weights with optional runtime input quantization.

    Buffers (including the RepQ input transform) participate in .to() and
    state_dict(). Full-module artifacts resolve this stable repository class.
    """

    def __init__(self, source, weight, activation_bits=32, input_mode="none",
                 scale=None, zero_point=None, negative_scale=None,
                 input_ratio=None, input_shift=None, signed_full_range=False):
        super().__init__(source.in_features, source.out_features, source.bias is not None,
                         device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            self.weight.copy_(weight)
            if self.bias is not None:
                self.bias.copy_(source.bias)
        self.requires_grad_(False)
        self.activation_bits = int(activation_bits)
        self.input_mode = input_mode
        self.signed_full_range = bool(signed_full_range)
        for name, value, default in (
            ("input_scale", scale, 1.0), ("input_zero_point", zero_point, 0.0),
            ("negative_scale", negative_scale, 1.0), ("input_ratio", input_ratio, 1.0),
            ("input_shift", input_shift, 0.0),
        ):
            value = torch.as_tensor(default if value is None else value,
                                    device=weight.device, dtype=weight.dtype).detach().clone()
            self.register_buffer(name, value)

    def quantize_input(self, value):
        value = value / self.input_ratio - self.input_shift
        if self.activation_bits == 32 or self.input_mode == "none":
            return value
        qmax = 2 ** (self.activation_bits - 1) - 1
        qmin = -qmax - int(self.signed_full_range)
        if self.input_mode == "dynamic":
            scale = (value.reshape(-1, self.in_features).abs().amax(0) + 1e-8) / qmax
            return (value / scale).round().clamp(-qmax, qmax) * scale
        if self.input_mode == "symmetric":
            return (value / self.input_scale).round().clamp(qmin, qmax) * self.input_scale
        if self.input_mode == "affine":
            integer = (value / self.input_scale).round() + self.input_zero_point
            return (integer.clamp(0, 2 ** self.activation_bits - 1) - self.input_zero_point) * self.input_scale
        if self.input_mode == "twin":
            positive = (value / self.input_scale).round().clamp(0, qmax) * self.input_scale
            negative = (value / self.negative_scale).round().clamp(-qmax - 1, 0) * self.negative_scale
            return positive + negative
        raise RuntimeError(f"Unknown input quantization mode: {self.input_mode}")

    def forward(self, value):
        return F.linear(self.quantize_input(value), self.weight, self.bias)


class _Calibration:
    """A fresh disk snapshot of every supplied calibration batch, never an old cache."""

    def __init__(self, loader, directory):
        self.paths = []
        self.samples = 0
        if loader is None:
            raise ValueError("This method requires a nonempty calibration loader")
        for index, batch in enumerate(loader):
            batch = move_to_device(batch, "cpu")
            value = batch if torch.is_tensor(batch) else batch.get("pixel_values")
            if not torch.is_tensor(value) or value.ndim == 0:
                raise ValueError("Calibration batches need a batched pixel_values tensor")
            path = Path(directory) / f"calibration-{index:06d}.pt"
            torch.save(batch, path)
            self.paths.append(path)
            self.samples += value.shape[0]
        if not self.paths or not self.samples:
            raise ValueError("Calibration loader is empty")

    def __iter__(self):
        for path in self.paths:
            yield torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def _activation_ranges(model, layers, calibration, device):
    ranges = {}
    handles = []
    for name, layer in layers.items():
        def collect(module, inputs, name=name):
            value = inputs[0].detach().reshape(-1, module.in_features)
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite calibration activations in {name}")
            low, high = value.amin(0), value.amax(0)
            if name in ranges:
                low = torch.minimum(low, ranges[name][0])
                high = torch.maximum(high, ranges[name][1])
            ranges[name] = (low, high)
        handles.append(layer.register_forward_pre_hook(collect))
    try:
        for batch in calibration:
            forward_model(model, batch, device)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(layers) - set(ranges)
    if missing:
        raise ValueError(f"Calibration did not execute selected layers: {sorted(missing)}")
    return ranges


def _basic(model, layers, method, weight_bits, activation_bits, calibration, device):
    ranges = _activation_ranges(model, layers, calibration, device) if method == "static" and activation_bits < 32 else {}
    state = {}
    for name, layer in layers.items():
        grid = {"uniform": "ppq_reference", "dynamic": "dynamic_reference"}.get(method, "symmetric")
        weight, weight_scale = _symmetric_weight(layer.weight, weight_bits, grid=grid)
        mode, scale = "none", None
        if activation_bits < 32:
            if method == "dynamic":
                mode = "dynamic"
            elif method == "static":
                low, high = ranges[name]
                scale = torch.maximum(low.abs(), high.abs()).clamp_min(1e-12) / (2 ** (activation_bits - 1) - 1)
                mode = "symmetric"
        model = _replace(model, name, QuantizedLinear(layer, weight, activation_bits, mode, scale=scale))
        state[name] = {"weight_scale": weight_scale.cpu(), "activation_scale": None if scale is None else scale.cpu()}
    return model, state


def _repq_weight(weight, bits):
    """RepQ percentile/MSE calibration, vectorized over output channels."""
    if bits == 32:
        return weight.detach().clone(), None
    source = weight.detach().float()
    quantiles = torch.tensor([0.001, 0.0001, 0.00001, 0.999, 0.9999, 0.99999], device=source.device)
    values = torch.quantile(source, quantiles, dim=1)
    low, high = values[:3], values[3:]
    scale = (high - low).clamp_min(1e-12) / (2 ** bits - 1)
    zero = (-low / scale).round()
    integer = (source[None] / scale[:, :, None]).round() + zero[:, :, None]
    candidates = (integer.clamp(0, 2 ** bits - 1) - zero[:, :, None]) * scale[:, :, None]
    best = (candidates - source[None]).square().mean(-1).argmin(0)
    index = torch.arange(source.shape[0], device=source.device)
    output = candidates[best, index]
    # Preserve constant channels exactly; their zero range has no quantization error.
    constant = source.amax(1) == source.amin(1)
    output[constant] = source[constant]
    return output.to(weight.dtype), {"scale": scale[best, index].cpu(), "zero_point": zero[best, index].cpu()}


def _repq(model, layers, calibration, weight_bits, activation_bits, device):
    if activation_bits == 32:
        raise ValueError("RepQ scale reparameterization requires activation_bits < 32; use uniform for weight-only quantization")
    ranges = _activation_ranges(model, layers, calibration, device)
    state = {}
    for name, layer in layers.items():
        low, high = ranges[name]
        # Full-calibration min/max replaces first-batch percentile initialization
        # in the old script, so every supplied sample contributes to the scales.
        delta = (high - low).clamp_min(1e-8) / (2 ** activation_bits - 1)
        zp = (-low / delta).round()
        target_delta, target_zp = delta.mean(), zp.mean()
        ratio = delta / target_delta
        shift = (-zp * delta) / ratio + target_zp * target_delta
        transformed = copy.deepcopy(layer)
        with torch.no_grad():
            transformed.weight.mul_(ratio[None])
            compensation = transformed.weight @ shift
            if transformed.bias is None:
                transformed.bias = nn.Parameter(compensation, requires_grad=False)
            else:
                transformed.bias.add_(compensation)
        weight, weight_state = _repq_weight(transformed.weight, weight_bits)
        replacement = QuantizedLinear(transformed, weight, activation_bits, "affine",
                                      scale=target_delta, zero_point=target_zp,
                                      input_ratio=ratio, input_shift=shift)
        model = _replace(model, name, replacement)
        state[name] = {"input_ratio": ratio.cpu(), "input_shift": shift.cpu(),
                       "activation_scale": target_delta.cpu(), "activation_zero_point": target_zp.cpu(),
                       "weight_quantizer": weight_state}
    return model, state


def _cache_ptq_layer(model, name, layer, calibration, directory, device):
    """Cache all layer IO and regression-loss gradients, a batch at a time."""
    paths, captured, gelu_outputs = [], [], set()
    post_gelu = []
    handles = []

    def mark_gelu(module, inputs, output):
        if torch.is_tensor(output):
            gelu_outputs.add(id(output))

    def capture(module, inputs, output):
        output.retain_grad()
        captured.append((inputs[0].detach(), output))
        post_gelu.append(id(inputs[0]) in gelu_outputs)

    for module in model.modules():
        if "gelu" in module.__class__.__name__.lower():
            handles.append(module.register_forward_hook(mark_gelu))
    handles.append(layer.register_forward_hook(capture))
    original_requires_grad = {parameter: parameter.requires_grad for parameter in model.parameters()}
    model.requires_grad_(True)
    try:
        for index, batch in enumerate(calibration):
            if not isinstance(batch, dict) or "labels" not in batch:
                raise ValueError("PTQ4ViT's regression Hessian proxy requires calibration labels")
            model.zero_grad(set_to_none=True)
            captured.clear()
            gelu_outputs.clear()
            prediction = forward_model(model, batch, device)
            labels = batch["labels"].to(prediction)
            if prediction.shape != labels.shape:
                raise ValueError("PTQ4ViT calibration labels and predictions must have identical shapes")
            # Sum over samples gives each observation the same weight even when
            # the final loader batch is smaller than the others.
            loss = (prediction - labels).square().flatten(1).mean(1).sum()
            loss.backward()
            for invocation, (value, output) in enumerate(captured):
                if output.grad is None:
                    raise ValueError(f"Layer {name} is disconnected from the calibration loss")
                record = {"x": value.cpu(), "y": output.detach().cpu(), "g": output.grad.detach().cpu()}
                if not all(torch.isfinite(tensor).all() for tensor in record.values()):
                    raise ValueError(f"Non-finite PTQ4ViT calibration values in {name}")
                path = Path(directory) / f"layer-{index:06d}-{invocation}.pt"
                torch.save(record, path)
                paths.append(path)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for parameter, enabled in original_requires_grad.items():
            parameter.requires_grad_(enabled)
    if not paths:
        raise ValueError(f"Calibration did not execute selected layer {name}")
    return paths, bool(post_gelu) and all(post_gelu)


def _ptq_weight(weight, scale, bits):
    if bits == 32:
        return weight
    qmax = 2 ** (bits - 1)
    return (weight / scale).round().clamp(-qmax, qmax - 1) * scale


def _ptq_input(value, scale, negative_scale, bits, twin):
    if bits == 32:
        return value
    qmax = 2 ** (bits - 1)
    if twin:
        return ((value / scale).round().clamp(0, qmax - 1) * scale
                + (value / negative_scale).round().clamp(-qmax, 0) * negative_scale)
    return (value / scale).round().clamp(-qmax, qmax - 1) * scale


@torch.no_grad()
def _ptq_search(layer, paths, weight_bits, activation_bits, twin, device,
                search_rounds=3, candidate_count=100):
    weight = layer.weight.detach()
    weight_scale = weight.abs().amax().clamp_min(1e-12) / (2 ** (weight_bits - 1) - 0.5) if weight_bits < 32 else weight.new_tensor(1.)
    maximum = weight.new_tensor(0.)
    gradient_energy = 0.0
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        value = record["x"].to(device)
        maximum = torch.maximum(maximum, value.amax() if twin else value.abs().amax())
        gradient_energy += float(record["g"].double().square().sum())
    if gradient_energy == 0:
        raise ValueError("PTQ4ViT calibration gradients are all zero; choose nontrivial labeled calibration samples")
    activation_scale = maximum.clamp_min(1e-12) / (2 ** (activation_bits - 1) - 0.5) if activation_bits < 32 else weight.new_tensor(1.)
    # GELU's negative lobe is bounded by approximately -0.17, as in PTQ4ViT.
    negative_scale = weight.new_tensor(0.1699712425 / (2 ** (activation_bits - 1)))
    initial_weight_scale, initial_activation_scale = weight_scale.clone(), activation_scale.clone()
    factors = torch.linspace(0.01, 1.2, candidate_count, device=device)

    def score_candidates(candidates, search_weight):
        scores = torch.zeros(len(candidates), dtype=torch.float64, device=device)
        # Each record contains a complete loader batch. Never subsample tokens
        # or observations; candidate chunks keep GPU memory bounded.
        for path in paths:
            record = torch.load(path, map_location="cpu", weights_only=False)
            value, target, gradient = (record[key].to(device) for key in ("x", "y", "g"))
            for index, candidate in enumerate(candidates):
                ws = candidate if search_weight else weight_scale
                activation_step = activation_scale if search_weight else candidate
                quantized = F.linear(_ptq_input(value, activation_step, negative_scale, activation_bits, twin),
                                     _ptq_weight(weight, ws, weight_bits), layer.bias)
                scores[index] += ((quantized - target) * gradient).double().square().sum()
        return candidates[int(scores.argmin())]

    for _ in range(search_rounds):
        if weight_bits < 32:
            weight_scale = score_candidates(initial_weight_scale * factors, True)
        if activation_bits < 32:
            activation_scale = score_candidates(initial_activation_scale * factors, False)
    return weight_scale, activation_scale, negative_scale, gradient_energy


def _ptq4vit(model, layers, calibration, weight_bits, activation_bits, device, directory, args):
    rounds = int(getattr(args, "ptq_search_rounds", 3))
    candidates = int(getattr(args, "ptq_candidates", 100))
    if rounds < 1 or candidates < 2:
        raise ValueError("PTQ4ViT requires at least one search round and two candidate intervals")
    state, replacements, twin_layers = {}, {}, []
    # Collect and optimize against the unchanged FP network (parallel PTQ),
    # then apply all replacements together.
    for name, layer in layers.items():
        with tempfile.TemporaryDirectory(prefix="ptq-layer-", dir=directory) as layer_directory:
            paths, twin = _cache_ptq_layer(model, name, layer, calibration, layer_directory, device)
            ws, activation_scale, negative_scale, energy = _ptq_search(
                layer, paths, weight_bits, activation_bits, twin, device, rounds, candidates)
        if twin:
            twin_layers.append(name)
        replacements[name] = QuantizedLinear(layer, _ptq_weight(layer.weight, ws, weight_bits).detach(),
                                            activation_bits, "twin" if twin else "symmetric",
                                            scale=activation_scale, negative_scale=negative_scale,
                                            signed_full_range=True)
        state[name] = {"weight_scale": ws.cpu(), "activation_scale": activation_scale.cpu(),
                       "negative_activation_scale": negative_scale.cpu(), "twin_uniform": twin,
                       "calibration_gradient_energy": energy}
    for name, replacement in replacements.items():
        model = _replace(model, name, replacement)
    return model, state, {"search_rounds": rounds, "candidate_intervals": candidates,
                          "twin_uniform_layers": twin_layers,
                          "sensitivity_metric": "squared regression-loss gradient weighted output reconstruction"}


class _QuantStateModel(nn.Module):
    """Transient BRECQ adapter, removed before exporting the model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def set_quant_state(self, weight_quant=False, act_quant=False):
        for module in self.model.modules():
            if hasattr(module, "set_quant_state"):
                module.set_quant_state(weight_quant, act_quant)

    def train(self, mode=True):
        # The legacy IO-capture helper calls train() after calibration. PTQ
        # reconstructs inference behavior; keep dropout/BN in eval throughout.
        return super().train(False)

    def forward(self, **batch):
        return forward_model(self.model, batch, next(self.parameters()).device)


def _initialize_brecq_weight_quantizer(module):
    """Use vendor MSE scale search, handling constant channels explicitly."""
    quantizer = module.weight_quantizer
    weight = module.weight.detach()
    flattened = weight.reshape(weight.shape[0], -1)
    constant = flattened.amax(1) == flattened.amin(1)
    if not constant.any():
        quantizer(weight)
        return
    scales, zeros = [], []
    for index, row in enumerate(weight):
        if constant[index]:
            value = flattened[index, 0]
            # A one-step representation preserves every constant channel,
            # including zero-initialized conditioning projections.
            scale = value.abs().clamp_min(1e-8)
            zero = value.new_tensor(0 if value >= 0 else quantizer.n_levels - 1)
        else:
            scale, zero = quantizer.init_quantization_scale(row, channel_wise=False)
        scales.append(scale)
        zeros.append(zero)
    shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
    quantizer.delta = torch.stack(scales).reshape(shape)
    quantizer.zero_point = torch.stack(zeros).reshape(shape)
    quantizer.inited = True


def _brecq_calibration_batches(calibration, batch_size):
    """Rebatch every calibration sample to the requested reconstruction size."""
    from torch.utils.data import default_collate
    if batch_size < 1:
        raise ValueError("BRECQ rec_batchsize must be positive")
    pending = []
    for batch in calibration:
        if not isinstance(batch, dict):
            raise ValueError("BRECQ calibration requires dictionary batches with pixel_values")
        size = batch["pixel_values"].shape[0]
        # The model adapter only needs inputs/conditioning. Reconstruction
        # targets are captured FP block outputs, never supervised test labels.
        for index in range(size):
            pending.append({key: value[index] for key, value in batch.items()
                            if key in ("pixel_values", "time") and torch.is_tensor(value)})
            if len(pending) == batch_size:
                yield default_collate(pending)
                pending = []
    if pending:
        yield default_collate(pending)


def _reconstruct_brecq_block(qnn, block, calibration, iterations, batch_size, directory, device):
    """Vendor AdaRound/BRECQ objective with a bounded, disk-backed IO cache."""
    from BRECQ.quant.adaptive_rounding import AdaRoundQuantizer
    from BRECQ.quant.quant_layer import QuantModule
    from BRECQ.quant.poseidon_block_recon import PoseidonLossFunction
    from BRECQ.quant.poseidon_data_utils import (
        PoseidonGetLayerInpOut, move_block_args_to_device, get_reconstruction_output,
    )

    qnn.set_quant_state(False, False)
    block.set_quant_state(True, False)
    parameters = []
    for module in block.modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer = AdaRoundQuantizer(
                uaq=module.weight_quantizer, round_mode="learned_hard_sigmoid",
                weight_tensor=module.org_weight.detach())
            module.weight_quantizer.soft_targets = True
            parameters.append(module.weight_quantizer.alpha)
    if not parameters:
        raise ValueError("BRECQ reconstruction block contains no weight quantizers")
    optimizer = torch.optim.Adam(parameters)
    loss_function = PoseidonLossFunction(block, round_loss="relaxation", weight=0.01,
                                        max_count=iterations, rec_loss="mse",
                                        b_range=(20, 2), warmup=0.2, p=2.)
    get_io = PoseidonGetLayerInpOut(qnn, block, torch.device(device), asym=True, act_quant=False)
    count = 0
    total_samples = 0
    for count, batch in enumerate(_brecq_calibration_batches(calibration, batch_size), start=1):
        total_samples += int(batch["pixel_values"].shape[0])
        inputs, target = get_io(batch)
        if inputs is None or target is None:
            raise ValueError("Calibration did not execute a BRECQ reconstruction block")
        torch.save({"inputs": inputs, "target": target}, Path(directory) / f"block-{count-1:06d}.pt")
        # The vendor hook object retains its most recent tensors. Release
        # these as soon as that batch is written, including device tensors.
        get_io.data_saver.input_store = None
        get_io.data_saver.output_store = None
    if not count:
        raise ValueError("BRECQ calibration loader is empty")
    for _ in range(iterations):
        # Draw a sample uniformly, then reconstruct its stored batch. Batch
        # selection is proportional to batch size; averaging its loss gives
        # equal expected weight to every sample, including the final partial
        # batch. Keep whole batches because attention folds windows into B.
        index = int(torch.randint(total_samples, (1,))) // batch_size
        record = torch.load(Path(directory) / f"block-{index:06d}.pt", map_location="cpu", weights_only=False)
        inputs = move_block_args_to_device(record["inputs"], device)
        target = record["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = get_reconstruction_output(block(*inputs))
        loss_function(output, target).backward()
        optimizer.step()
    for module in block.modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False
    return count


def _brecq(model, layers, calibration, args, weight_bits, activation_bits, device, directory):
    if activation_bits != 32:
        raise ValueError("The Poseidon BRECQ adapter currently supports weight-only reconstruction; set activation_bits=32")
    if not 2 <= weight_bits <= 8:
        raise ValueError("BRECQ's AdaRound quantizer requires weight_bits in 2..8")
    iterations = int(getattr(args, "reconstruction_iterations", 20000))
    if iterations < 1:
        raise ValueError("BRECQ reconstruction_iterations must be positive")
    from BRECQ.quant.quant_layer import QuantModule

    original = copy.deepcopy(model)
    weight_parameters = {"n_bits": weight_bits, "channel_wise": True, "scale_method": "mse"}
    activation_parameters = {"n_bits": 8, "channel_wise": False, "scale_method": "max"}
    poseidon = model.__class__.__module__ == "scOT.model"
    if poseidon:
        from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
        from BRECQ.quant.poseidon_quant_block import QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock
        poseidon_wrapper = PoseidonQuantModel(model, weight_parameters, activation_parameters).to(device).eval()
        qnn = _QuantStateModel(poseidon_wrapper.model).to(device).eval()
        block_types = (QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock)
        blocks = [(name, module) for name, module in qnn.model.named_modules() if isinstance(module, block_types)]
    else:
        for name, layer in layers.items():
            model = _replace(model, name, QuantModule(layer, weight_parameters, activation_parameters))
        qnn = _QuantStateModel(model).to(device).eval()
        blocks = []
    quant_modules = {name: module for name, module in qnn.model.named_modules() if isinstance(module, QuantModule)}
    if not quant_modules:
        raise ValueError("BRECQ found no quantizable operations")
    # Initialize every weight quantizer, including layers outside reconstructed
    # blocks, directly from weights; no old scales/checkpoints are consumed.
    with torch.no_grad():
        for module in quant_modules.values():
            _initialize_brecq_weight_quantizer(module)
    covered = {id(module) for _, block in blocks for module in block.modules() if isinstance(module, QuantModule)}
    blocks.extend((name, module) for name, module in quant_modules.items() if id(module) not in covered)
    reconstruction_batch_size = int(getattr(args, "rec_batchsize", 32))
    qnn.requires_grad_(False)
    cached_batches = 0
    for name, block in blocks:
        qnn.eval()
        with tempfile.TemporaryDirectory(prefix="brecq-block-", dir=directory) as block_directory:
            cached_batches = _reconstruct_brecq_block(
                qnn, block, calibration, iterations, reconstruction_batch_size, block_directory, device)
    qnn.eval()
    state = {}
    original_modules = dict(original.named_modules())
    with torch.no_grad():
        for name, module in quant_modules.items():
            quantizer = module.weight_quantizer
            if hasattr(quantizer, "soft_targets"):
                quantizer.soft_targets = False
            quantized_weight = quantizer(module.weight).detach()
            target = original_modules[name]
            target.weight.copy_(quantized_weight)
            state[name] = {"scale": quantizer.delta.detach().cpu(),
                           "zero_point": quantizer.zero_point.detach().cpu(),
                           "rounding_alpha": getattr(quantizer, "alpha", None)}
            if state[name]["rounding_alpha"] is not None:
                state[name]["rounding_alpha"] = state[name]["rounding_alpha"].detach().cpu()
        # Verify the saved, ordinary model reproduces the reconstructed
        # wrapper on calibration input before discarding transient modules.
        qnn.set_quant_state(True, False)
        example = next(iter(calibration))
        wrapped_prediction = forward_model(qnn, example, device)
        materialized_prediction = forward_model(original.eval(), example, device)
        difference = (wrapped_prediction - materialized_prediction).abs().amax()
        if not torch.allclose(wrapped_prediction, materialized_prediction, atol=1e-6, rtol=1e-5):
            raise RuntimeError(f"BRECQ weight materialization changed predictions (max difference {float(difference):.3g})")
    return original.eval(), state, {"reconstruction_blocks": [name for name, _ in blocks],
                                    "reconstruction_iterations": iterations,
                                    "reconstruction_batch_size": reconstruction_batch_size,
                                    "reconstruction_cached_batches": cached_batches,
                                    "calibration_cache": "disk-backed complete calibration and one block at a time",
                                    "reconstruction_sampling": "uniform sample index mapped to its cached batch (batch-size-proportional sampling)",
                                    "materialization_max_abs_difference": float(difference),
                                    "quantized_layers": list(quant_modules),
                                    "scope": "Poseidon Conv2d and Linear, including conditional norm projections" if poseidon else "selected Linear"}


def quantize_baseline(model, calibration_loader, args, device, work_dir):
    """Quantize a supplied model using only the supplied calibration loader."""
    method = str(args.method).lower()
    methods = {"fp", "uniform", "dynamic", "static", "brecq", "repq", "ptq4vit"}
    if method not in methods:
        raise ValueError(f"Unknown baseline {method!r}; expected one of {sorted(methods)}")
    weight_bits = _bits(getattr(args, "weight_bits", 8), "weight_bits")
    activation_bits = _bits(getattr(args, "activation_bits", 32), "activation_bits")
    if method == "uniform" and activation_bits != 32:
        raise ValueError("uniform is the static weight-only max-abs baseline; use static or dynamic for activation quantization")
    if method == "repq" and activation_bits == 32:
        raise ValueError("RepQ requires activation quantization (activation_bits < 32)")
    if method == "brecq" and (activation_bits != 32 or not 2 <= weight_bits <= 8):
        raise ValueError("BRECQ supports weight_bits=2..8 and activation_bits=32")
    torch.manual_seed(int(getattr(args, "seed", 0)))
    model = model.to(device).eval()
    layers = select_quantizable_layers(model)
    if method != "fp" and not layers:
        raise ValueError("The model has no eligible Linear layers")
    for name, layer in layers.items():
        if not torch.isfinite(layer.weight).all():
            raise ValueError(f"Non-finite weights in {name}")
    metadata = {"method": method, "weight_bits": 32 if method == "fp" else weight_bits,
                "activation_bits": 32 if method == "fp" else activation_bits,
                "execution": "floating_point" if method == "fp" else "fake_quantization",
                "packed_integer_weights": False, "scope": "selected Linear",
                "quantized_layers": [] if method == "fp" else list(layers),
                "calibration_samples": 0, "calibration_batches": 0}
    if method == "fp":
        return QuantizationResult(model, metadata, {})
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    needs_calibration = method in {"repq", "ptq4vit", "brecq"} or (method == "static" and activation_bits < 32)
    with tempfile.TemporaryDirectory(prefix="baseline-calibration-", dir=work_dir) as directory:
        calibration = _Calibration(calibration_loader, directory) if needs_calibration else None
        if calibration is not None:
            metadata.update(calibration_samples=calibration.samples, calibration_batches=len(calibration.paths))
        if method in {"uniform", "dynamic", "static"}:
            model, state = _basic(model, layers, method, weight_bits, activation_bits, calibration, device)
            metadata["weight_scale_rule"] = "per-output-channel symmetric max-abs"
            if method == "uniform":
                metadata["quantization_grid"] = "PPQ.metrics reference: step=2*max_abs/(2^bits-1), round(W/step)*step without clipping"
            elif method == "dynamic":
                metadata["quantization_grid"] = "DynamicQ reference: step=(max_abs+1e-8)/(2^(bits-1)-1), signed symmetric QDQ"
                metadata["activation_scale_rule"] = "per-input-channel max-abs over each current input batch"
            else:
                metadata["quantization_grid"] = "step=max_abs/(2^(bits-1)-1), signed symmetric QDQ"
                metadata["activation_scale_rule"] = "per-input-channel max-abs over the complete calibration set"
        elif method == "repq":
            model, state = _repq(model, layers, calibration, weight_bits, activation_bits, device)
            metadata.update(algorithm="RepQ scale/shift reparameterization with explicit input affine",
                            activation_calibration="full-calibration per-input-channel min/max",
                            norm_folding=False, attention_matmul_quantized=False, softmax_quantized=False)
        elif method == "ptq4vit":
            model, state, details = _ptq4vit(model, layers, calibration, weight_bits, activation_bits, device, directory, args)
            metadata.update(details)
            metadata.update(algorithm="PTQ4ViT Linear adaptation: Hessian-guided search and post-GELU twin uniform",
                            attention_matmul_quantized=False, softmax_quantized=False)
        else:
            model, state, details = _brecq(model, layers, calibration, args, weight_bits, activation_bits, device, directory)
            metadata.update(details)
            metadata["algorithm"] = "BRECQ block reconstruction with AdaRound; materialized QDQ weights"
    return QuantizationResult(model.eval(), metadata, state)

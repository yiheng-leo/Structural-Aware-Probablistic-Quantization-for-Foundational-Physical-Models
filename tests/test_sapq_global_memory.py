"""Compare memory-bounded global SAPQ against the original stacked objective."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from PPQ.metrics import evaluate_with_stepsizes
from PPQ.noise import add_quantization_noise
from SAPQ.sapq_loss import (
    compute_mc_negative_loglikelihood_network_global,
    compute_prior_by_mode,
    compute_sapq_loss_with_prior_global,
)
from SAPQ.sapq_trainer_global import SAPQTrainerGlobal


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ToyPoseidon(nn.Module):
    def __init__(self, width=4, hidden=7):
        super().__init__()
        self.first = nn.Linear(width, hidden)
        self.last = nn.Linear(hidden, width)
        self.forward_sizes = []
        self.fail = False

    def forward(self, pixel_values, time=None, pixel_mask=None, labels=None):
        self.forward_sizes.append(len(pixel_values))
        hidden = self.first(pixel_values)
        if self.fail:
            raise RuntimeError("injected forward failure")
        hidden = torch.tanh(hidden + time[:, None, None, None])
        result = self.last(hidden)
        if pixel_mask is not None:
            result = torch.where(pixel_mask[:, :, None, None], labels, result)
        return SimpleNamespace(output=result)


def batch_for(model, samples=2):
    return {
        "pixel_values": torch.randn(samples, 2, 3, model.first.in_features, dtype=model.first.weight.dtype),
        "time": torch.linspace(0.1, 0.3, samples, dtype=model.first.weight.dtype),
        "pixel_mask": torch.tensor([[True, False]]).expand(samples, -1),
        "labels": torch.randn(samples, 2, 3, model.last.out_features, dtype=model.first.weight.dtype),
    }


def forward_batch(model, batch):
    return model(**{name: value.to(model.first.weight.device) for name, value in batch.items()}).output


def steps_for(model):
    return {name: (nn.Parameter(torch.full((module.out_features,), 0.025,
                                         device=module.weight.device, dtype=module.weight.dtype)), None)
            for name, module in model.named_modules() if isinstance(module, nn.Linear)}


def reference_likelihood(model, steps, batch, target, samples, eta):
    """Independent explicit forward with the original sequential noise draws."""
    device = model.first.weight.device
    inputs = {name: value.to(device) for name, value in batch.items()}
    scores = []
    for _ in range(samples):
        first_weight = add_quantization_noise(model.first.weight, steps["first"][0], 0)
        hidden = F.linear(inputs["pixel_values"], first_weight, model.first.bias)
        hidden = torch.tanh(hidden + inputs["time"][:, None, None, None])
        last_weight = add_quantization_noise(model.last.weight, steps["last"][0], 0)
        output = F.linear(hidden, last_weight, model.last.bias)
        output = torch.where(inputs["pixel_mask"][:, :, None, None], inputs["labels"], output)
        scores.append(-(output - target.to(device)).square().mean((1, 2, 3)) / (2 * eta))
    return -(torch.stack(scores).logsumexp(0) - torch.log(torch.tensor(float(samples), device=device))).sum()


def rng_state(device):
    return [torch.get_rng_state()] + ([torch.cuda.get_rng_state(device)] if device.type == "cuda" else [])


@pytest.mark.parametrize("prior_mode", ["block_sens", "block_no_sens", "ppq"])
@pytest.mark.parametrize("device_name", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_loss_gradients_adam_updates_and_rng_match_original(prior_mode, device_name):
    torch.manual_seed(17)
    device = torch.device(device_name)
    template = ToyPoseidon().double().to(device)
    batch = batch_for(template)
    with torch.no_grad():
        target = forward_batch(template, batch).cpu()
    ranges = {name: {"weight_ranges": torch.linspace(0.7, 1.2, mod.out_features,
                                                     dtype=torch.float64, device=device)}
              for name, mod in template.named_modules() if isinstance(mod, nn.Linear)}
    sensitivity = {name: torch.linspace(0, 1, len(rec["weight_ranges"]),
                                       dtype=torch.float64, device=device) for name, rec in ranges.items()}
    settings = dict(prior_mode=prior_mode, b_target=8.0, sigma0=0.5, alpha=1.0,
                    prior_scale=1e-4, gamma=0.005)

    def run(checkpointed):
        model = deepcopy(template)
        if checkpointed:
            model.requires_grad_(False)
        steps = steps_for(model)
        optimizer = torch.optim.Adam([step[0] for step in steps.values()], lr=9.1e-4)
        torch.manual_seed(23)
        records = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            if checkpointed:
                total, likelihood, prior = compute_sapq_loss_with_prior_global(
                    model, steps, [batch], [target], ranges, sensitivity,
                    num_mc_samples=5, eta=1e-6, device=device,
                    memory_efficient=True, **settings)
            else:
                likelihood = reference_likelihood(model, steps, batch, target, 5, 1e-6)
                prior = compute_prior_by_mode(step_sizes_dict=steps, ranges_dict=ranges,
                                              sens_dict=sensitivity, **settings)
                total = likelihood + prior
            before_backward = rng_state(device)
            total.backward()
            for before, after in zip(before_backward, rng_state(device)):
                assert torch.equal(before, after)
            gradients = [step[0].grad.detach().clone() for step in steps.values()]
            optimizer.step()
            records.append((torch.stack([total, likelihood, prior]).detach(), gradients,
                            [step[0].detach().clone() for step in steps.values()],
                            torch.rand(8, device=device)))
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, dict(template.named_parameters())[name], rtol=0, atol=0)
            if checkpointed:
                assert parameter.grad is None
        assert not any(module._forward_hooks for module in model.modules())
        return records

    expected, actual = run(False), run(True)
    for old, new in zip(expected, actual):
        torch.testing.assert_close(new, old, rtol=1e-10, atol=1e-10)


def test_checkpoint_retains_small_inputs_instead_of_all_mc_activations():
    torch.manual_seed(5)
    model = ToyPoseidon(width=64, hidden=256).requires_grad_(False)
    batch = batch_for(model)
    with torch.no_grad():
        target = forward_batch(model, batch)

    def saved_elements(checkpointed):
        saved = []

        def pack(tensor):
            saved.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            value = compute_mc_negative_loglikelihood_network_global(
                model, steps_for(model), [batch], [target], 0, num_mc_samples=5,
                device="cpu", memory_efficient=checkpointed)
        value.backward()
        return sum(saved)

    original, checkpointed = saved_elements(False), saved_elements(True)
    assert checkpointed < original / 10


@pytest.mark.parametrize("quantized", [False, True])
def test_validation_chunks_preserve_samples_and_original_batch_means(quantized):
    torch.manual_seed(9)
    model = ToyPoseidon().double().requires_grad_(False)
    batches = [batch_for(model, 5), batch_for(model, 3)]
    weight_steps = steps_for(model) if quantized else {}
    settings = dict(model=model, val_loader=lambda: iter(batches), weight_steps=weight_steps,
                    act_steps=None, layer_names=["first", "last"], device="cpu")
    expected = evaluate_with_stepsizes(**settings)
    assert model.forward_sizes == [5, 3]
    model.forward_sizes.clear()
    actual = evaluate_with_stepsizes(**settings, forward_batch_size=2)
    assert model.forward_sizes == [2, 2, 1, 2, 1]
    assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert not any(module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("operation", ["likelihood", "evaluation"])
def test_forward_failure_removes_quantization_hooks(operation):
    model = ToyPoseidon().requires_grad_(False)
    batch = batch_for(model)
    with torch.no_grad():
        target = forward_batch(model, batch)
    model.fail = True
    steps = steps_for(model)
    with pytest.raises(RuntimeError, match="injected forward failure"):
        if operation == "likelihood":
            compute_mc_negative_loglikelihood_network_global(
                model, steps, [batch], [target], 0, num_mc_samples=5,
                device="cpu", memory_efficient=True)
        else:
            evaluate_with_stepsizes(model, [batch], steps, None, list(steps),
                                    device="cpu", forward_batch_size=1)
    assert not any(module._forward_hooks for module in model.modules())


def test_trainer_reuses_fp_model_and_only_updates_channel_steps(tmp_path):
    torch.manual_seed(10)
    model = ToyPoseidon()
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    cfg = SimpleNamespace(model_path=str(tmp_path / "toy"), repo_root=tmp_path,
                          percentile_prob=1e-4, init_bits=9, bmax_bits=20, weight_only=True,
                          base_lr=9.1e-4, num_epochs=2, min_epochs=10, early_stop_bits=4,
                          num_mc_samples=5, eta=1e-6, updates_per_batch=1,
                          target_bits=8, sigma0=0.5, alpha=1.0, prior_scale=1e-4,
                          prior_mode="block_sens", log_every=1, eval_every=1)
    trainer = SAPQTrainerGlobal(model, cfg, ["first", "last"], device="cpu")
    assert trainer.model is model
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
    batches = [batch_for(model), batch_for(model)]
    ranges = {name: {"weight_ranges": torch.ones(mod.out_features),
                     "activation_ranges": torch.ones(mod.in_features)}
              for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
    sensitivity = {name: torch.zeros_like(rec["weight_ranges"]) for name, rec in ranges.items()}
    evaluations = []

    def evaluate(epoch, steps, unused_ranges):
        evaluations.append(evaluate_with_stepsizes(model, batches, steps, None, list(steps),
                                                  device="cpu", forward_batch_size=1))

    steps, _, history = trainer.train(batches, ranges, sensitivity, evaluate)
    assert len(evaluations) == 2
    assert len([entry for entry in history if "epoch" in entry]) == 2
    assert all(np.isfinite(metric) for metrics in evaluations for metric in metrics.values())
    for step, _ in steps.values():
        assert step.grad is not None and torch.isfinite(step.grad).all()
        assert not torch.equal(step, torch.full_like(step, 1 / 2**9))
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
        assert parameter.grad is None


def test_poseidon_checkpoint_matches_original_loss_and_step_gradients():
    from scOT.model import ScOT, ScOTConfig

    torch.manual_seed(13)
    template = ScOT(ScOTConfig(image_size=8, patch_size=2, embed_dim=8, depths=[1, 1],
                               num_heads=[2, 4], window_size=2, skip_connections=[1, 1],
                               num_channels=4, num_out_channels=4, use_conditioning=True,
                               hidden_dropout_prob=0, attention_probs_dropout_prob=0,
                               drop_path_rate=0)).eval()
    batch = {"pixel_values": torch.randn(2, 4, 8, 8), "labels": torch.randn(2, 4, 8, 8),
             "time": torch.tensor([0.1, 0.2]), "pixel_mask": torch.tensor([[True, False, False, False]] * 2)}
    with torch.no_grad():
        target = template(**batch).output
    records = []
    for checkpointed in (False, True):
        model = deepcopy(template)
        if checkpointed:
            model.requires_grad_(False)
        steps = steps_for(model)
        torch.manual_seed(31)
        likelihood = compute_mc_negative_loglikelihood_network_global(
            model, steps, [batch], [target], 0, num_mc_samples=5, eta=1e-6,
            device="cpu", memory_efficient=checkpointed)
        state = torch.get_rng_state()
        likelihood.backward()
        assert torch.equal(state, torch.get_rng_state())
        records.append((likelihood.detach(), {name: value[0].grad for name, value in steps.items()}))
        assert not any(module._forward_hooks for module in model.modules())
    torch.testing.assert_close(records[1], records[0], rtol=2e-5, atol=2e-5)

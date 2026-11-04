from contextlib import ExitStack
from io import BytesIO
import json
import os
import subprocess
import sys
import tarfile

import pytest
import torch
from torch import nn

from SAPQ.export_main_artifact import export, materialize_weights


@pytest.mark.parametrize("invalid", ["shape", "nonpositive", "missing"])
def test_bad_steps_do_not_partially_modify_model(invalid):
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    original = {key: value.clone() for key, value in model.state_dict().items()}
    steps = {"0": (torch.full((3,), 0.1), None), "1": (torch.full((2,), 0.1), None)}
    if invalid == "shape":
        steps["1"] = torch.ones(3)
    elif invalid == "nonpositive":
        steps["1"] = torch.tensor([0.0, 0.1])
    else:
        steps["missing"] = steps.pop("1")
    with pytest.raises(ValueError):
        materialize_weights(model, steps)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_export_matches_main_hooks_and_loads_in_ritchie_evaluation(tmp_path):
    from scOT.model import ScOT, ScOTConfig

    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = ScOT(ScOTConfig(image_size=8, patch_size=2, embed_dim=8, depths=[1, 1],
                            num_heads=[2, 4], window_size=2, skip_connections=[1, 1],
                            num_channels=4, num_out_channels=4, use_conditioning=True,
                            drop_path_rate=0)).eval()
    checkpoint = tmp_path / "original"
    model.save_pretrained(checkpoint)
    layers = {name: layer for name, layer in model.named_modules() if isinstance(layer, nn.Linear)
              and "norm" not in name and "time" not in name}
    steps = {name: (torch.full((layer.out_features,), 0.05), torch.ones(layer.in_features))
             for name, layer in layers.items()}
    step_path = tmp_path / "sapq_global_step_sizes.pt"
    torch.save({"step_sizes_dict": steps, "meta": {"model_path": str(checkpoint), "target_bits": 8}}, step_path)
    (tmp_path / "sapq_global_history.json").write_text(json.dumps([
        {"epoch": 1, "avg_bits": 4.1}, {"epoch": 10, "avg_bits": 3.8695},
        {"type": "batch_update_history_epoch1", "records": []}]))
    batch = {"pixel_values": torch.randn(2, 4, 8, 8), "labels": torch.randn(2, 4, 8, 8),
             "time": torch.tensor([0.1, 0.2]), "pixel_mask": torch.tensor([[False, False, False, True]] * 2)}
    # Same hooks used by main's PPQ.metrics.evaluate_with_stepsizes.
    with ExitStack() as cleanup, torch.no_grad():
        for name, layer in layers.items():
            scale = steps[name][0].reshape(-1, 1)

            def hook(module, inputs, output, scale=scale):
                weight = torch.round(module.weight / scale) * scale
                return nn.functional.linear(inputs[0], weight, module.bias)

            handle = layer.register_forward_hook(hook)
            cleanup.callback(handle.remove)
        expected = model(**batch).output
    artifact = tmp_path / "exported"
    metadata = export(step_path, artifact)
    assert metadata["quantization"]["avg_weight_bits"] == 3.8695
    assert metadata["quantization"]["epochs"] == 10
    restored = torch.load(artifact / "compressed_model.pt", weights_only=False)
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch).output, expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(FileExistsError):
        export(step_path, artifact)

    # Exercise the actual other branch's loader, ScOT classes and compact report
    # in an isolated process, without switching this checkout or touching data.
    refs = [ref for ref in ("ritchie", "origin/ritchie") if subprocess.run(
        ["git", "rev-parse", "--verify", ref], capture_output=True).returncode == 0]
    if not refs:
        pytest.skip("ritchie source unavailable for cross-branch loading check")
    archive = subprocess.check_output(["git", "archive", refs[0], "quantization", "scOT", "physical", "evaluation/reporting.py"])
    root = tmp_path / "ritchie-source"
    root.mkdir()
    with tarfile.open(fileobj=BytesIO(archive)) as files:
        files.extractall(root, filter="data")
    reference = tmp_path / "reference.pt"
    torch.save((batch, expected), reference)
    env = {**os.environ, "PYTHONPATH": str(root), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    code = '''
import sys, torch
from quantization.artifacts import load_quantized_artifact
from evaluation.reporting import final_average_weight_bits, build_results
torch.set_num_threads(1)
model, metadata = load_quantized_artifact(sys.argv[1], "cpu")
batch, expected = torch.load(sys.argv[2], weights_only=False)
with torch.no_grad():
    torch.testing.assert_close(model(**batch).output, expected, rtol=1e-6, atol=1e-6)
bits = final_average_weight_bits(metadata)
assert bits == 3.8695
assert build_results(avg_weight_bits=bits)["avg_weight_bits"] == bits
'''
    completed = subprocess.run([sys.executable, "-c", code, str(artifact), str(reference)],
                               cwd=root, env=env, text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr

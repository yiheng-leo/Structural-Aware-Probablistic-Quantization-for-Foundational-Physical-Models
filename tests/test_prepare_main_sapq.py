"""Standalone preparation against main's original cache consumers/formulas."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def main_sources(tmp_path):
    root = tmp_path / "main-checkout"
    root.mkdir()
    files = ["PPQ/config.py", "PPQ/path_utils.py", "PPQ/poseidon_utils.py",
             "SAPQ/sapq_sobo_sensitivity.py", "BRECQ/quant/quant_layer.py", "BRECQ/quant/quant_block.py",
             "BRECQ/quant/poseidon_quant_block.py", "BRECQ/quant/poseidon_quant_model.py",
             "BRECQ/quant/poseidon_data_utils.py"]
    for file in files:
        result = subprocess.run(["git", "show", f"main:{file}"], capture_output=True)
        if result.returncode:
            pytest.skip("Original main branch source is unavailable for direct comparison.")
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result.stdout)
    for package in ("SAPQ", "PPQ", "BRECQ"):
        (root / package / "__init__.py").touch()
    (root / "SAPQ/run_sapq_network_global.py").write_text("# original main entry marker\n")
    (root / "inspect_layers").mkdir()
    (root / "inspect_layers/L_quantize_layers.json").write_text('["layer"]')
    return root


def run_script(code, root, tmp_path):
    project = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(root), str(project)]),
               HF_HOME=str(tmp_path / "hf-cache"), TRANSFORMERS_CACHE=str(tmp_path / "hf-cache"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run([sys.executable, "-c", code, str(project / "tools/prepare_main_sapq.py"),
                             str(root), str(tmp_path)], cwd=root, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_freeze_cache_uses_external_storage_and_both_main_paths(main_sources, tmp_path):
    run_script(r'''
import hashlib, json, os, runpy, sys
from pathlib import Path
import torch
from torch.utils.data import Dataset
from scOT.problems import base
root, scratch = Path(sys.argv[2]), Path(sys.argv[3])
os.environ["PPQ_MODEL_PATH"] = "models/poseidon/fluids.incompressible.PiecewiseConstants"
os.environ["PPQ_DATA_PATH"] = "poseidon/camlab-ethz/down_streams"
os.environ["PPQ_DATASET_NAME"] = "fluids.incompressible.PiecewiseConstants"
calls=[]
class Data(Dataset):
    def __len__(self): return 2048
    def __getitem__(self,index):
        return {"pixel_values":torch.full((4,4,4),float(index)),
                "labels":torch.zeros(4,4,4), "time":torch.tensor(0.1),
                "pixel_mask":torch.zeros(4,dtype=torch.bool)}
def factory(**kw):
    calls.append(kw)
    return Data()
base.get_dataset=factory
(root/"ppq_artifacts").symlink_to("/nonexistent-author-cluster/ppq_artifacts")
main=runpy.run_path(sys.argv[1])["main"]
arguments=["--repo-root",str(root),"--cache-root",str(scratch/"external-cache"),"--frozen-only"]
main(arguments)
p=root/"ppq_artifacts/frozen_calibration_batches.pt"
assert p.is_file() and root not in p.resolve().parents
batches=torch.load(p,weights_only=True)
assert len(batches)==512 and sum(len(b["pixel_values"]) for b in batches)==1024
assert all(b["pixel_values"].device.type=="cpu" for b in batches)
assert p.samefile(root/"ppq_artifacts/down_streams-calib/frozen_calibration_batches.pt")
assert (root/"inspect_layers/L_quantize_layers.pt").is_file()
meta=json.loads((root/"ppq_artifacts/calibration_meta.json").read_text())
assert meta["seed"]==0 and meta["historical_cache"] is False
assert calls[0]["num_trajectories"]==2048 and "max_num_time_steps" not in calls[0]
digest=hashlib.sha256(p.read_bytes()).hexdigest()
main(arguments)
assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
''', main_sources, tmp_path)


def test_existing_nested_cache_and_valid_directory_symlink_are_preserved(main_sources, tmp_path):
    run_script(r'''
import hashlib, os, runpy, sys
from pathlib import Path
import torch
from scOT.problems import base
root, scratch = Path(sys.argv[2]), Path(sys.argv[3])
os.environ["PPQ_DATA_PATH"] = "poseidon/camlab-ethz/down_streams"
base.get_dataset=lambda **kw: object()
existing=scratch/"historical-cache"
nested=existing/"down_streams-calib/frozen_calibration_batches.pt"
nested.parent.mkdir(parents=True)
batch={"pixel_values":torch.randn(2,4,4,4),"labels":torch.zeros(2,4,4,4),
       "time":torch.tensor([0.1,0.2]),"pixel_mask":torch.zeros(2,4,dtype=torch.bool)}
torch.save([batch],nested)
digest=hashlib.sha256(nested.read_bytes()).hexdigest()
(root/"ppq_artifacts").symlink_to(existing,target_is_directory=True)
runpy.run_path(sys.argv[1])["main"](["--repo-root",str(root),
    "--cache-root",str(scratch/"new-cache"),"--frozen-only"])
assert (root/"ppq_artifacts").resolve()==existing
assert (root/"ppq_artifacts/frozen_calibration_batches.pt").samefile(nested)
assert hashlib.sha256(nested.read_bytes()).hexdigest()==digest
''', main_sources, tmp_path)


def test_sensitivity_matches_original_per_block_backward(main_sources, tmp_path):
    run_script(r'''
from copy import deepcopy
from pathlib import Path
import runpy, sys
from types import SimpleNamespace
import torch
from scOT.model import ScOT, ScOTConfig
torch.set_num_threads(1)
torch.manual_seed(5)
functions=runpy.run_path(sys.argv[1])
root=Path(sys.argv[2])
helpers=functions["load_poseidon_sensitivity_helpers"](root)
model=ScOT(ScOTConfig(image_size=8,patch_size=2,embed_dim=8,depths=[1,1],num_heads=[2,4],
    window_size=2,skip_connections=[1,1],num_channels=4,num_out_channels=4,use_conditioning=True,
    hidden_dropout_prob=0,attention_probs_dropout_prob=0,drop_path_rate=0)).eval()
reference_model=deepcopy(model)
batches=[{"pixel_values":torch.randn(2,4,8,8),"time":torch.tensor([0.1,0.2])} for _ in range(2)]
dataset=SimpleNamespace(constants={"mean":[0.8,0,0,0],"std":[0.31,0.391,0.356,0.185]})
cfg=SimpleNamespace(dataset_name="fluids.incompressible.PiecewiseConstants")
actual=functions["compute_sensitivity"](model,batches,dataset,cfg,root,torch.device("cpu"))
qmodel=helpers.PoseidonQuantModel(reference_model).eval()
blocks=[(name,mod) for name,mod in qmodel.named_modules() if isinstance(mod,helpers.TARGET_BLOCK_TYPES)]
assert any(isinstance(mod,helpers.QuantConvNeXtBlock) for _,mod in blocks)
raw={}
for name,block in blocks:
    hook=helpers.BlockGradHook()
    handle=block.register_full_backward_hook(hook)
    values=[]
    for batch in batches:
        qmodel.zero_grad(set_to_none=True)
        qmodel.set_quant_state(False,False)
        clean=helpers.get_model_output_tensor(qmodel(**batch)).detach()
        qmodel.set_quant_state(True,False)
        pred=helpers.get_model_output_tensor(qmodel(**batch))
        loss=helpers.compute_sobolev_h1_loss_denorm(pred,clean,dataset,cfg.dataset_name)
        loss.backward()
        assert hook.grad is not None
        values.append(helpers.reduce_channel_sensitivity(hook.grad).detach())
    handle.remove()
    raw[name]=torch.stack(values).mean(0)
assert actual["block_sensitivity_raw"].keys()==raw.keys()
for name,expected in raw.items():
    torch.testing.assert_close(actual["block_sensitivity_raw"][name],expected,rtol=2e-5,atol=1e-18)
normalized=helpers.normalize_blockwise_minmax(raw)
expected_layers=helpers.build_block_sensitivity_to_inner_layer_dict(qmodel,normalized)
assert actual["layer_sensitivity_norm"].keys()==expected_layers.keys()
for name,expected in expected_layers.items():
    torch.testing.assert_close(actual["layer_sensitivity_norm"][name],expected,rtol=2e-5,atol=1e-14)
''', main_sources, tmp_path)

# Preserved VICON BRECQ workflow

This is the VICON-specific workflow formerly located at `BRECQ/run_vicon_brecq.py`
and `BRECQ/quant/vicon_*`. It remains separate from the unified Poseidon runner:
its calibration data, model interface and saved wrapper format differ.

From the repository root, the relocated entry is:

```bash
python -m workflows.vicon.BRECQ.run_vicon_brecq --dataset_name NS2D --device cuda
```

The weight-scale preprocessing entry is
`python -m workflows.vicon.BRECQ.quant.precompute_vicon_weight_scales`.
The original example arguments are retained in `quant/viconcommand.txt`.
VICON attention/block/model adapters remain in `quant/`; shared BRECQ quantizers
are imported from `BRECQ.quantizers`. No active imports depend on the preserved
third-party source tree.

## Existing missing dependency

Both entries require the helper functions `load_vicon_ckpt` and, for
reconstruction, `vicon_preprocess_bc`. The old scripts imported these from
`cache_vicon_calib_layer_io.py`, but that file was already absent from the
repository before the layout migration. The relocated import expects
`workflows/vicon/cache_vicon_calib_layer_io.py`. No replacement algorithm or
placeholder has been invented. These entries cannot run until the original
helper is supplied.

The workflow also requires the VICON source/dependencies, an external model
checkpoint and existing frozen calibration at
`ppq_artifacts/<dataset>-calib/frozen_calibration_batches.pt`. Its historical
`calib_steps`, eager loading and scale-cache behavior are preserved; it does not
inherit the unified Poseidon runner's complete, disk-streamed calibration
contract. Optional distributed reconstruction requires an available `linklink`
module. This relocation has been syntax/import checked for the available
modules, but no VICON calibration or reconstruction was run.

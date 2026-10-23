# Preserved upstream source

These directories preserve the third-party trees that were previously mixed into
the active method folders. They are reference material, not dependencies of the
Poseidon quantization runner. Their original READMEs and available licenses are
retained. Examples here may still use their original imports, datasets and paths;
they are not a second supported Poseidon implementation.

| Previous location | Preserved location | Active Poseidon implementation |
| --- | --- | --- |
| `BRECQ/` ImageNet programs, distributed helpers and generic `quant/` kernels | `brecq/` | `BRECQ/quantize.py`, `quantizers.py`, `poseidon.py`, `reconstruction.py` |
| `RepQ-ViT/classification`, `detection`, images and documentation | `repq_vit/` | `RepQ/quantize.py` |
| `PTQ4ViT/configs`, `example`, `quant_layers`, `utils` | `ptq4vit/` | `PTQ4ViT/quantize.py`, `hessian.py`, `search.py` |

The active adapters preserve the preceding unified Poseidon implementation's
numerical behavior. They do not claim full upstream algorithm coverage:

- BRECQ uses weight-only MSE scale initialization and AdaRound reconstruction,
  including Poseidon convolution and conditional-normalization projections.
- RepQ adapts selected Linear layers with explicit affine input
  reparameterization and compensating weights/biases. It does not fold transforms
  into LayerNorm or quantize attention matrix multiplication or softmax.
- PTQ4ViT adapts selected Linear layers with interval search and the post-GELU
  twin grid. Its curvature weights come from physical regression-label MSE
  gradients, not the upstream classification KL objective. Attention matrix
  multiplication and softmax are outside this adapter's scope.

All active adapters export floating-point QDQ models; this layout change does
not add packed integer execution. Old project-specific Poseidon experiments
are under `archive/legacy`, while the separate VICON workflow is under
`workflows/vicon`.

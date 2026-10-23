"""Lazy exports keep Poseidon independent of optional ImageNet architectures."""
from importlib import import_module

_EXPORTS = {
    "block_reconstruction": "block_recon",
    "layer_reconstruction": "layer_recon",
    "BaseQuantBlock": "quant_block",
    "QuantModule": "quant_layer",
    "QuantModel": "quant_model",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value

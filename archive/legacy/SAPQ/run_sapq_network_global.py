"""MySVD-style sapq entry: load calibration, quantize, save; evaluate separately."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantization.quantize import main as quantize_main


def main():
    return quantize_main(default_method='sapq', defaults={'sensitivity_normalization': 'normalized'})


if __name__ == "__main__":
    main()


def __getattr__(name):
    # Imported by old plotting/debug scripts; never used by the new runner.
    if name in {"load_candidate_layers", "maybe_load_or_compute_ranges",
                "load_sapq_sensitivity", "load_frozen_calibration_batches"}:
        from SAPQ import legacy_global_helpers
        return getattr(legacy_global_helpers, name)
    raise AttributeError(name)

"""MySVD-style sapq-layerwise entry: load calibration, quantize, save; evaluate separately."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantization.quantize import main as quantize_main


def main():
    return quantize_main(default_method='sapq-layerwise', defaults={'sensitivity_normalization': 'normalized'})


if __name__ == "__main__":
    main()

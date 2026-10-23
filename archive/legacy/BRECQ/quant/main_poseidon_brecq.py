"""MySVD-style brecq entry: load calibration, quantize, save; evaluate separately."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantization.quantize import main as quantize_main


def main():
    return quantize_main(default_method='brecq')


if __name__ == "__main__":
    main()

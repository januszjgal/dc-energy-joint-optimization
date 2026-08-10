"""Repository entry point for the independent six-market energy model v3."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from energy_model_v3.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

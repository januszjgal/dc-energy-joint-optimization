"""Build and validate the real energy-v3 ramp-v6 factory handoff."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from energy_model_v3.ramp_factory import build_factory, validate_factory  # noqa: E402


def main() -> None:
    build_factory()
    print(json.dumps(validate_factory(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

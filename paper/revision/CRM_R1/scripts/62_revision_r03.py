#!/usr/bin/env python3
"""R03 offline preparation or explicitly requested, bounded local-model development."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from v17_r03 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

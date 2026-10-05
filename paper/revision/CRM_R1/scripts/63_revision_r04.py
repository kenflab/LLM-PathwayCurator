#!/usr/bin/env python3
"""R03 diagnosis and offline preparation; --live runs only the R04 reading probe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from v17_r04 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run R01 using the source package in this checkout, not another editable clone."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from v17_revision import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

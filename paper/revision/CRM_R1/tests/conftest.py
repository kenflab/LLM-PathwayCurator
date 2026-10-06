"""Repository-only revision tests import their uninstalled development modules."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "paper/revision/CRM_R1/experiments"))

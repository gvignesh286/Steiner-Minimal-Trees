"""Helpers shared by the scripts (paths, sys.path setup)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
README = ROOT / "README.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

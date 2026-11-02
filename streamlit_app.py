"""Streamlit Cloud Root Entry Point for StreamShield-MLOps."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Run dashboard module
import runpy

runpy.run_path(str(REPO_ROOT / "src" / "dashboard" / "app.py"), run_name="__main__")

#!/usr/bin/env python3
"""Entry point for: python -m models.decision"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.decision.cli import main

if __name__ == "__main__":
    main()

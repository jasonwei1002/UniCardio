"""Shared pytest fixtures and path setup for UniCardio tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Expose the repo root so ``import src.*`` works without installation.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

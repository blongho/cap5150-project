"""Provenance recorded with every run: git commit and library versions."""

from __future__ import annotations

import importlib.util
import platform
import subprocess

import pandas as pd
import sklearn


def git_commit() -> str:
    """Current commit hash, or "unknown" outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def torch_version() -> str | None:
    """Installed PyTorch version, or None where torch is unavailable."""
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch.__version__


def run_info() -> dict[str, str | None]:
    return {
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "torch_version": torch_version(),
        "pandas_version": pd.__version__,
    }

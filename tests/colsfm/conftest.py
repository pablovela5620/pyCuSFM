"""Shared fixtures for the colsfm tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
"""Repo root, so data paths resolve regardless of the working directory."""


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def galileo_input_meta(repo_root: Path) -> Path:
    """The 226-keyframe input metadata shipped with r2b_galileo."""
    return repo_root / "data" / "r2b_galileo" / "frames_meta.json"


@pytest.fixture(scope="session")
def galileo_run_dir(repo_root: Path) -> Path:
    """The shipped cuSFM Galileo run used as the reference blob output."""
    return repo_root / "data" / "cusfm_runs" / "galileo"

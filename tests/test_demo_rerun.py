"""The demo's side of the shared COLMAP text reader.

The parser's own behaviour is pinned once, in
``tests/colsfm/test_colmap_text_model.py``. What this file checks is the reason
that parser is shared rather than duplicated: the demo runs in a CUDA-13
environment with a TensorRT dependency override and **no pycolmap**, so the
module it reads models through must not pull pycolmap's ceres/CUDA stack in
behind it.

Importing ``demo_rerun`` needs the Rerun SDK and simplecv, so environments
without them (``raco``) skip rather than fail to collect. Run with
``pixi run -e colsfm demo-test``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
"""Repository root, so ``demo_rerun`` imports without being installed."""

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

demo_rerun = pytest.importorskip("demo_rerun", reason="demo_rerun needs rerun + simplecv; run in the colsfm environment")

if TYPE_CHECKING:
    from colsfm.colmap_text_model import ColmapModel

FIRST_IMAGE_UNOBSERVED: str = (
    "# Image list with two lines of data per image:\n"
    "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
    "# Number of images: 2, mean observations per image: 0.5\n"
    "1 1 0 0 0 0 0 0 1 cam_00/000000.jpg\n"
    "\n"  # image 1 observed nothing: COLMAP still writes its (empty) 2D line
    "2 0.7071067811865476 0 0 0.7071067811865476 1 2 3 1 cam_01/000000.jpg\n"
    "100.5 200.5 7\n"
)
"""Two-image model whose first image has an empty observation line."""

IMPORT_WITHOUT_PYCOLMAP: str = """
import sys
import colsfm.colmap_text_model  # noqa: F401
assert "pycolmap" not in sys.modules, sorted(name for name in sys.modules if "colmap" in name)
print("ok")
"""
"""Probe run in a fresh interpreter: importing the reader must not import pycolmap."""


def test_the_shared_reader_does_not_import_pycolmap() -> None:
    """The whole reason the parser is shared instead of duplicated into the demo.

    Run in a subprocess: this test session has already imported pycolmap through
    the rest of the suite, so ``sys.modules`` here would prove nothing.
    """
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", IMPORT_WITHOUT_PYCOLMAP],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_demo_still_exposes_the_reader(tmp_path: Path) -> None:
    """`demo_rerun.read_colmap_model` keeps working, from its new home."""
    (tmp_path / "images.txt").write_text(FIRST_IMAGE_UNOBSERVED)

    model: ColmapModel = demo_rerun.read_colmap_model(tmp_path)

    assert sorted(model.world_T_cam) == ["cam_00/000000.jpg", "cam_01/000000.jpg"]
    np.testing.assert_allclose(model.world_T_cam["cam_00/000000.jpg"], np.eye(4), atol=1e-12)


def test_the_entry_point_holds_only_the_command_line() -> None:
    """`demo_rerun.py` is the entry point; the work lives in the `demo` package."""
    source: str = (REPO_ROOT / "demo_rerun.py").read_text()

    assert source.count("\ndef ") == 0
    assert source.count("\nclass ") == 0
    assert "from demo.app import Config, main" in source

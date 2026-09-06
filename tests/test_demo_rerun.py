"""Behavior tests for the demo's hand-rolled COLMAP text-model reader.

Pure parsing only: no GPU, no cuSFM binaries, and no Rerun Viewer. Importing
``demo_rerun`` still pulls in the Rerun SDK and simplecv, so environments
without them (``raco``) skip instead of failing to collect. Run with
``pixi run -e colsfm demo-test``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
"""Repository root, so ``demo_rerun`` imports without being installed."""

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

demo_rerun = pytest.importorskip("demo_rerun", reason="demo_rerun needs rerun + simplecv; run in the colsfm environment")

if TYPE_CHECKING:
    from demo_rerun import ColmapImage, ColmapModel

IMAGES_TXT_HEADER: str = (
    "# Image list with two lines of data per image:\n"
    "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
    "# Number of images: 2, mean observations per image: 0.5\n"
)
"""The exact comment block COLMAP writes above the image records."""

FIRST_IMAGE_UNOBSERVED: str = (
    f"{IMAGES_TXT_HEADER}"
    "1 1 0 0 0 0 0 0 1 cam_00/000000.jpg\n"
    "\n"  # image 1 observed nothing: COLMAP still writes its (empty) 2D line
    "2 0.7071067811865476 0 0 0.7071067811865476 1 2 3 1 cam_01/000000.jpg\n"
    "100.5 200.5 7\n"
)
"""Two-image model whose first image has an empty observation line."""


def test_empty_observation_line_does_not_shift_pairing() -> None:
    """An image with zero observations must not drop every later image."""
    images: list[ColmapImage] = demo_rerun.parse_colmap_images_text(FIRST_IMAGE_UNOBSERVED)

    assert [image.image_id for image in images] == [1, 2]
    assert [image.name for image in images] == ["cam_00/000000.jpg", "cam_01/000000.jpg"]


def test_pose_is_inverted_into_world_from_camera() -> None:
    """``world_T_cam`` is the inverse of COLMAP's stored ``cam_T_world``."""
    images: list[ColmapImage] = demo_rerun.parse_colmap_images_text(FIRST_IMAGE_UNOBSERVED)

    np.testing.assert_allclose(images[0].world_T_cam, np.eye(4), atol=1e-12)

    # 90 degrees about +Z, then a translation of (1, 2, 3), inverted.
    cam_T_world: Float64[ndarray, "4 4"] = np.eye(4)
    cam_T_world[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_T_world[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(images[1].world_T_cam, np.linalg.inv(cam_T_world), atol=1e-9)


def test_trailing_blank_lines_are_tolerated() -> None:
    """Newlines at the end of the file yield no phantom image."""
    images: list[ColmapImage] = demo_rerun.parse_colmap_images_text(f"{FIRST_IMAGE_UNOBSERVED}\n\n\n")

    assert [image.image_id for image in images] == [1, 2]


def test_header_only_model_parses_to_nothing() -> None:
    """A comment-only ``images.txt`` is a valid, empty model."""
    assert demo_rerun.parse_colmap_images_text(IMAGES_TXT_HEADER) == []


def test_read_colmap_model_keys_every_registered_image(tmp_path: Path) -> None:
    """The on-disk reader keeps both images, keyed by name, alongside the points."""
    (tmp_path / "images.txt").write_text(FIRST_IMAGE_UNOBSERVED)
    (tmp_path / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n1 0.5 1.5 2.5 10 20 30 0.1 2 0\n"
    )

    model: ColmapModel = demo_rerun.read_colmap_model(tmp_path)

    assert sorted(model.world_T_cam) == ["cam_00/000000.jpg", "cam_01/000000.jpg"]
    np.testing.assert_allclose(model.points_xyz, [[0.5, 1.5, 2.5]])
    np.testing.assert_array_equal(model.points_rgb, [[10, 20, 30]])

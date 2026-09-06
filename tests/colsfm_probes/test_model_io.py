"""Probe (g): model I/O, and whether demo_rerun.py can read what pycolmap writes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float
from numpy import ndarray

from colsfm_probes.synthetic import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SyntheticScene,
    add_ground_truth_points,
    make_pinhole_camera,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
"""Repository root, so `demo_rerun` can be imported without installing it."""


def _load_read_colmap_model():
    """Import `demo_rerun.read_colmap_model` from the repository root.

    Returns:
        The `read_colmap_model` function.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import demo_rerun

    return demo_rerun.read_colmap_model


def test_write_text_binary_and_read_round_trip(scene: SyntheticScene, tmp_path: Path) -> None:
    """`write_text`, `write_binary` and `read` round-trip the whole model.

    COLMAP 4.2 emits five files in each format — `rigs` and `frames` are new
    alongside `cameras`, `images` and `points3D`.  `Reconstruction.write` is an
    alias for `write_binary`; `read` sniffs whichever pair is present.
    """
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    add_ground_truth_points(scene)

    text_dir: Path = tmp_path / "text"
    binary_dir: Path = tmp_path / "binary"
    text_dir.mkdir()
    binary_dir.mkdir()
    reconstruction.write_text(text_dir)
    reconstruction.write_binary(binary_dir)

    assert {path.name for path in text_dir.iterdir()} == {
        "cameras.txt",
        "images.txt",
        "points3D.txt",
        "rigs.txt",
        "frames.txt",
    }
    assert {path.name for path in binary_dir.iterdir()} == {
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "rigs.bin",
        "frames.bin",
    }

    for source in (text_dir, binary_dir):
        loaded: pycolmap.Reconstruction = pycolmap.Reconstruction()
        loaded.read(source)
        assert loaded.num_cameras() == reconstruction.num_cameras()
        assert loaded.num_rigs() == reconstruction.num_rigs()
        assert loaded.num_frames() == reconstruction.num_frames()
        assert loaded.num_images() == reconstruction.num_images()
        assert loaded.num_points3D() == reconstruction.num_points3D()
        for frame_id in sorted(reconstruction.frames):
            assert np.allclose(
                loaded.frame(frame_id).rig_from_world.matrix(),
                reconstruction.frame(frame_id).rig_from_world.matrix(),
                atol=1e-9,
            )


def test_images_txt_column_order(scene: SyntheticScene, tmp_path: Path) -> None:
    """`images.txt` is `IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`, then a POINTS2D line.

    demo_rerun's parser reads the quaternion from fields 1:5 as wxyz, the
    translation from 5:8 and the name from field 9 onwards, so this ordering is
    the interop contract.
    """
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    add_ground_truth_points(scene)
    reconstruction.write_text(tmp_path)

    lines: list[str] = [
        line
        for line in (tmp_path / "images.txt").read_text().splitlines()
        if not line.startswith("#")
    ]
    pose_line: list[str] = lines[0].split()
    assert len(pose_line) == 10
    image: pycolmap.Image = reconstruction.image(int(pose_line[0]))
    quaternion_wxyz: Float[ndarray, " 4"] = np.asarray(pose_line[1:5], dtype=np.float64)
    cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
    assert np.allclose(quaternion_wxyz, np.roll(cam_from_world.rotation.quat, 1), atol=1e-9)
    assert np.allclose(
        np.asarray(pose_line[5:8], dtype=np.float64), cam_from_world.translation, atol=1e-9
    )
    assert int(pose_line[8]) == image.camera_id
    assert pose_line[9] == image.name


def test_demo_rerun_reads_a_pycolmap_model(scene: SyntheticScene, tmp_path: Path) -> None:
    """A fully observed pycolmap model parses correctly in `demo_rerun.read_colmap_model`."""
    read_colmap_model = _load_read_colmap_model()
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    add_ground_truth_points(scene)
    reconstruction.write_text(tmp_path)

    model = read_colmap_model(tmp_path)
    assert len(model.world_T_cam) == reconstruction.num_images()
    assert len(model.points_xyz) == reconstruction.num_points3D()
    assert set(model.world_T_cam) == set(scene.image_names.values())
    for image_id, name in scene.image_names.items():
        expected: Float[ndarray, "4 4"] = np.eye(4)
        cam_from_world: pycolmap.Rigid3d = reconstruction.image(image_id).cam_from_world()
        expected[:3, :] = cam_from_world.matrix()
        assert np.allclose(model.world_T_cam[name], np.linalg.inv(expected), atol=1e-6)
    assert np.allclose(np.sort(model.points_xyz, axis=0), np.sort(scene.points_xyz, axis=0), atol=1e-9)


def test_demo_rerun_parser_handles_images_without_observations(tmp_path: Path) -> None:
    """A registered image with no 2D points must not shift the pose/POINTS2D pairing.

    COLMAP writes two lines per image: a pose line and a POINTS2D line.  When an
    image has no observations, the POINTS2D line is *empty*.  demo_rerun's parser
    used to drop empty lines before taking every second line, so from that point
    on POINTS2D lines were read as poses: it returned the same *number* of
    entries with a real pose replaced by nonsense parsed out of a coordinate
    list.  It now consumes strict pairs, so this file survives intact.
    """
    read_colmap_model = _load_read_colmap_model()
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera_with_trivial_rig(make_pinhole_camera(1))
    for image_id in (1, 2, 3):
        reconstruction.add_image_with_trivial_frame(
            pycolmap.Image(name=f"img_{image_id}.png", camera_id=1, image_id=image_id)
        )
        reconstruction.frame(image_id).rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(), np.array([-0.5 * image_id, 0.0, 0.0])
        )
        reconstruction.register_frame(image_id)

    points_xyz: Float[ndarray, "n 3"] = np.array(
        [[0.0, 0.0, 5.0], [0.5, 0.2, 6.0], [-0.4, 0.3, 5.5], [0.3, -0.4, 7.0], [-0.2, 0.1, 6.5]]
    )
    # Images 1 and 3 observe every point; image 2 observes nothing.
    for image_id in (1, 3):
        image: pycolmap.Image = reconstruction.image(image_id)
        image.points2D = pycolmap.Point2DList(
            [pycolmap.Point2D(image.project_point(xyz)) for xyz in points_xyz]
        )
    for point_index, xyz in enumerate(points_xyz):
        point3D_id: int = reconstruction.add_point3D(
            xyz,
            pycolmap.Track(
                [pycolmap.TrackElement(1, point_index), pycolmap.TrackElement(3, point_index)]
            ),
            np.array([10, 20, 30], dtype=np.uint8),
        )
        for image_id in (1, 3):
            reconstruction.image(image_id).set_point3D_for_point2D(point_index, point3D_id)

    reconstruction.write_text(tmp_path)
    images_txt: str = (tmp_path / "images.txt").read_text()
    body: list[str] = [line for line in images_txt.splitlines() if not line.startswith("#")]
    # Two lines per image, and image 2's POINTS2D line is empty.
    assert len(body) == 6
    assert body[3] == ""

    model = read_colmap_model(tmp_path)
    assert set(model.world_T_cam) == {"img_1.png", "img_2.png", "img_3.png"}

    # Every pose matches pycolmap's own reader on the same file.
    reloaded: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reloaded.read_text(tmp_path)
    assert reloaded.num_reg_images() == 3
    for image_id in (1, 2, 3):
        expected: Float[ndarray, "4 4"] = np.eye(4)
        expected[:3, :] = reloaded.image(image_id).cam_from_world().matrix()
        assert np.allclose(model.world_T_cam[f"img_{image_id}.png"], np.linalg.inv(expected), atol=1e-6)


def test_points3d_txt_carries_colour_and_track(scene: SyntheticScene, tmp_path: Path) -> None:
    """`points3D.txt` is `ID X Y Z R G B ERROR TRACK[]`, matching demo_rerun's slices."""
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    add_ground_truth_points(scene)
    reconstruction.write_text(tmp_path)
    line: str = next(
        line
        for line in (tmp_path / "points3D.txt").read_text().splitlines()
        if line and not line.startswith("#")
    )
    fields: list[str] = line.split()
    point: pycolmap.Point3D = reconstruction.point3D(int(fields[0]))
    assert np.allclose(np.asarray(fields[1:4], dtype=np.float64), point.xyz, atol=1e-9)
    assert np.array_equal(np.asarray(fields[4:7], dtype=np.uint8), point.color)
    assert float(fields[7]) == pytest.approx(point.error)
    assert (len(fields) - 8) % 2 == 0
    assert (len(fields) - 8) // 2 == point.track.length()


def test_export_ply(scene: SyntheticScene, tmp_path: Path) -> None:
    """`export_PLY` writes the point cloud for quick external inspection."""
    add_ground_truth_points(scene)
    ply_path: Path = tmp_path / "points.ply"
    scene.reconstruction.export_PLY(ply_path)
    assert ply_path.is_file()
    assert ply_path.read_bytes().startswith(b"ply")


def test_camera_dimensions_survive_text_round_trip(scene: SyntheticScene, tmp_path: Path) -> None:
    """Image width/height and params come back unchanged from `cameras.txt`."""
    scene.reconstruction.write_text(tmp_path)
    loaded: pycolmap.Reconstruction = pycolmap.Reconstruction()
    loaded.read_text(tmp_path)
    camera: pycolmap.Camera = loaded.camera(1)
    assert camera.width == IMAGE_WIDTH
    assert camera.height == IMAGE_HEIGHT
    assert camera.model == pycolmap.CameraModelId.PINHOLE
    assert np.allclose(camera.params, scene.reconstruction.camera(1).params)

"""Export writers, checked against the artifacts the blob left in `data/cusfm_runs/galileo`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from colsfm.export import (
    RuntimeRecord,
    append_runtime_record,
    read_runtime_records,
    write_colmap_model,
    write_optimised_frames_meta,
    write_pose_files,
)
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.geometry import parse_tum_line, quaternion_wxyz


@pytest.fixture(scope="module")
def optimised_meta(galileo_run_dir: Path) -> FramesMeta:
    """The blob's post-BA metadata, the input `extract_pose_from_map_main` was given."""
    return read_frames_meta(galileo_run_dir / "kpmap" / "keyframes" / "frames_meta.json")


def test_colmap_model_round_trips_through_write_text(galileo_run_dir: Path, tmp_path: Path) -> None:
    """A blob-written model survives our writer with every pose, id and point intact.

    Byte equality is not the target — see the module docstring — so this checks
    the model semantically: same cameras, same image ids and names, poses to
    1e-6, same points and track lengths.
    """
    original: pycolmap.Reconstruction = pycolmap.Reconstruction(str(galileo_run_dir / "sparse"))
    sparse_dir: Path = write_colmap_model(tmp_path, original)
    reloaded: pycolmap.Reconstruction = pycolmap.Reconstruction(str(sparse_dir))

    assert sorted(reloaded.cameras) == sorted(original.cameras)
    assert sorted(reloaded.images) == sorted(original.images)
    assert reloaded.num_points3D() == original.num_points3D() == 1275
    assert len(reloaded.images) == 28

    for image_id, expected_image in original.images.items():
        actual_image: pycolmap.Image = reloaded.images[image_id]
        assert actual_image.name == expected_image.name
        assert actual_image.camera_id == expected_image.camera_id
        assert np.allclose(actual_image.cam_from_world().translation, expected_image.cam_from_world().translation, atol=1e-6)
        assert np.allclose(quaternion_wxyz(actual_image.cam_from_world()), quaternion_wxyz(expected_image.cam_from_world()), atol=1e-6)
        assert len(actual_image.points2D) == len(expected_image.points2D)

    for point_id, expected_point in original.points3D.items():
        actual_point: pycolmap.Point3D = reloaded.points3D[point_id]
        assert np.allclose(actual_point.xyz, expected_point.xyz, atol=1e-6)
        assert actual_point.track.length() == expected_point.track.length()
    assert sum(point.track.length() for point in reloaded.points3D.values()) == 5835


def test_colmap_model_writes_the_three_named_files(galileo_run_dir: Path, tmp_path: Path) -> None:
    """`cameras.txt`, `images.txt` and `points3D.txt` land in `<out>/sparse/`.

    COLMAP 4.2 adds `rigs.txt` and `frames.txt`; that is a deliberate difference
    from cuSFM, which flattens the rig away.
    """
    sparse_dir: Path = write_colmap_model(tmp_path, pycolmap.Reconstruction(str(galileo_run_dir / "sparse")))
    assert sparse_dir == tmp_path / "sparse"
    written: set[str] = {path.name for path in sparse_dir.iterdir()}
    assert {"cameras.txt", "images.txt", "points3D.txt"} <= written
    assert written - {"cameras.txt", "images.txt", "points3D.txt"} == {"rigs.txt", "frames.txt"}


def assert_tum_files_agree(ours: Path, theirs: Path, tolerance: float = 1e-14) -> None:
    """Assert two TUM files hold the same poses at the same timestamps.

    The timestamp column must match as text — that pins the `us * 1e-6` multiply
    and the fixed 16-decimal formatting exactly. The seven pose columns are
    compared numerically because composing the rotation through scipy rather than
    Eigen moves the last bit or two: the worst disagreement over all 15 merged
    lines is 4e-16, which is float64 noise, not a convention error.

    Args:
        ours: File written by `colsfm.export`.
        theirs: The blob's own file.
        tolerance: Maximum absolute difference allowed per pose column.
    """
    our_lines: list[str] = ours.read_text().splitlines()
    their_lines: list[str] = theirs.read_text().splitlines()
    assert len(our_lines) == len(their_lines), f"{ours.name}: {len(our_lines)} lines vs {len(their_lines)}"
    for our_line, their_line in zip(our_lines, their_lines, strict=True):
        assert our_line.split(" ", 1)[0] == their_line.split(" ", 1)[0]
        our_values: list[float] = [float(field) for field in our_line.split()[1:]]
        their_values: list[float] = [float(field) for field in their_line.split()[1:]]
        assert np.allclose(our_values, their_values, rtol=0.0, atol=tolerance), f"{ours.name}: {our_line} != {their_line}"


def test_merged_tum_file_reproduces_the_blob(optimised_meta: FramesMeta, galileo_run_dir: Path, tmp_path: Path) -> None:
    """`merged_pose_file.tum` is reproduced to float64 noise, all 15 lines.

    Exercises the whole chain: `world_T_cam * cam_T_vehicle`, the timestamp-keyed
    map that collapses 32 keyframes to 15 unique microseconds, `us * 1e-6`, fixed
    16 decimals and the `qx qy qz qw` order.
    """
    write_pose_files(tmp_path, optimised_meta)
    assert_tum_files_agree(tmp_path / "output_poses" / "merged_pose_file.tum", galileo_run_dir / "output_poses" / "merged_pose_file.tum")


def test_per_camera_tum_files_reproduce_the_blob(optimised_meta: FramesMeta, galileo_run_dir: Path, tmp_path: Path) -> None:
    """All eight `camera_name-<sensor>_pose_file.tum` files match, in session `0`."""
    written: list[Path] = write_pose_files(tmp_path, optimised_meta)
    blob_dir: Path = galileo_run_dir / "output_poses" / "0"
    expected_names: set[str] = {path.name for path in blob_dir.iterdir()}
    ours_dir: Path = tmp_path / "output_poses" / "0"
    assert {path.name for path in ours_dir.iterdir()} == expected_names
    assert len(written) == len(expected_names) + 1
    for name in sorted(expected_names):
        assert_tum_files_agree(ours_dir / name, blob_dir / name)


def test_camera_frame_export_writes_the_stored_poses(optimised_meta: FramesMeta, tmp_path: Path) -> None:
    """With `export_pose_in_vehicle_frame=False` the per-camera file is `camera_to_world` verbatim.

    export.md §4.2 verified that to 5.6e-17 m. The merged file is meaningless in
    that mode and is not written.
    """
    written: list[Path] = write_pose_files(tmp_path, optimised_meta, export_pose_in_vehicle_frame=False)
    assert not (tmp_path / "output_poses" / "merged_pose_file.tum").exists()
    assert len(written) == len(optimised_meta.cameras)

    keyframes_by_id = optimised_meta.keyframe_by_id()
    camera_zero_keyframes = [k for k in optimised_meta.keyframes if k.camera_params_id == 0]
    sensor_name: str = optimised_meta.cameras[0].sensor_name
    lines: list[str] = (tmp_path / "output_poses" / "0" / f"camera_name-{sensor_name}_pose_file.tum").read_text().splitlines()
    assert len(lines) == len(camera_zero_keyframes)
    for line in lines:
        parsed = parse_tum_line(line)
        expected = next(k for k in camera_zero_keyframes if k.timestamp_microseconds == parsed.timestamp_microseconds)
        assert keyframes_by_id[expected.keyframe_id] is expected
        assert np.allclose(parsed.world_T_body.translation, expected.world_T_cam.translation, atol=1e-12)


def test_optimised_frames_meta_matches_the_blob_output(galileo_run_dir: Path, tmp_path: Path) -> None:
    """Feeding the blob's own optimised poses back in reproduces its `frames_meta.json`.

    The BA input is `pose_graph/frames_meta.json` and the BA output is
    `kpmap/keyframes/frames_meta.json`; they differ only in `camera_to_world` and
    `initial_pose_type` (export.md §7).
    """
    ba_input: FramesMeta = read_frames_meta(galileo_run_dir / "pose_graph" / "frames_meta.json")
    ba_output: FramesMeta = read_frames_meta(galileo_run_dir / "kpmap" / "keyframes" / "frames_meta.json")
    optimised_poses = {keyframe.keyframe_id: keyframe.world_T_cam for keyframe in ba_output.keyframes}

    path: Path = write_optimised_frames_meta(tmp_path, ba_input, optimised_poses)
    assert path == tmp_path / "kpmap" / "keyframes" / "frames_meta.json"
    written: FramesMeta = read_frames_meta(path)

    assert written.initial_pose_type == "ALIGNMENT" == ba_output.initial_pose_type
    assert [k.keyframe_id for k in written.keyframes] == [k.keyframe_id for k in ba_input.keyframes]
    written_by_id = written.keyframe_by_id()
    assert sorted(written_by_id) == sorted(k.keyframe_id for k in ba_output.keyframes)
    for theirs in ba_output.keyframes:
        ours = written_by_id[theirs.keyframe_id]
        assert ours.image_name == theirs.image_name
        assert ours.camera_params_id == theirs.camera_params_id
        assert ours.timestamp_microseconds == theirs.timestamp_microseconds
        assert ours.synced_sample_id == theirs.synced_sample_id
        assert np.allclose(ours.world_T_cam.translation, theirs.world_T_cam.translation, atol=1e-12)
        assert np.allclose(quaternion_wxyz(ours.world_T_cam), quaternion_wxyz(theirs.world_T_cam), atol=1e-12)
    for camera_params_id, camera in ba_output.cameras.items():
        assert np.allclose(written.cameras[camera_params_id].vehicle_T_cam.translation, camera.vehicle_T_cam.translation, atol=0.0)


def test_runtime_csv_matches_the_command_runner_format(tmp_path: Path, galileo_run_dir: Path) -> None:
    """The header and dialect are `pycusfm/command_runner.py`'s, so the logs concatenate."""
    blob_header: str = (galileo_run_dir / "runtime.csv").read_text().splitlines()[0]
    records: list[RuntimeRecord] = [
        RuntimeRecord(command="feature_extractor_main --min_inter_frame_distance 0.0", runtime_seconds=207.74518609046936),
        RuntimeRecord(command="kpmap_to_colmap --map_dir a, --output_dir b", runtime_seconds=1.4205365180969238),
    ]
    for record in records:
        path: Path = append_runtime_record(tmp_path, record)
    assert path == tmp_path / "runtime.csv"
    assert path.read_text().splitlines()[0] == blob_header
    assert read_runtime_records(path) == records


def test_the_blob_runtime_log_reads_back(galileo_run_dir: Path) -> None:
    """The shipped Galileo log parses with the same reader, 9 stages."""
    records: list[RuntimeRecord] = read_runtime_records(galileo_run_dir / "runtime.csv")
    assert len(records) == 9
    assert records[0].command.startswith("feature_extractor_main ")
    assert records[-1].command.startswith("extract_pose_from_map_main ")
    assert records[0].runtime_seconds == pytest.approx(207.74518609046936)

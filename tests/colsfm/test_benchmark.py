"""Benchmark metrics, checked against synthetic runs with known answers.

Every synthetic fixture is written to disk in the **real** cuSFM layout, through
the real writers (`pycolmap.Reconstruction.write_text` via
`colsfm.export.write_colmap_model`, `write_optimised_frames_meta`,
`append_runtime_record`), so `read_run` is exercised end to end rather than
handed an in-memory shortcut.

Expected values come from an independent source: a closed-form residual for a
scaled trajectory, the pixel offset the observations were built with, the
`(1 - 1/k) d` a single displaced camera puts into a k-camera mean, and — for
the real-data test — the numbers `demo_rerun.py` printed for the same blob run.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde.json import from_json, to_json

from colsfm.benchmark import (
    AcceptanceBounds,
    Comparison,
    RigidAlignment,
    RigTrack,
    RunArtifacts,
    RunMetrics,
    RuntimeRecord,
    Stage,
    align_rigid,
    check_acceptance,
    classify_stage,
    compare_runs,
    latest_run_records,
    match_timestamps,
    read_ground_truth,
    read_run,
    render_markdown_report,
    rig_rigidity_spread_millimeters,
    rig_track_from_frames_meta,
    stage_runtime_seconds,
)
from colsfm.export import append_runtime_record, write_colmap_model, write_optimised_frames_meta
from colsfm.frames_meta import CameraParams, FramesMeta, KeyframeMeta, read_frames_meta

SYNTHETIC_ERROR_PX: float = 0.75
"""Pixel offset baked into every synthetic observation, so the recomputed mean is known."""

BLOB_ERROR_PLACEHOLDER: float = 2.0
"""`kpmap_to_colmap` hardcodes this into `points3D.txt`; the synthetic runs mimic it."""

POINT_DEPTH_METERS: float = 4.0
"""Distance in front of a camera the synthetic points are placed at."""

POINTS_PER_CAMERA: int = 12
"""Points generated in front of each camera; enough for tracks of length > 1."""

DEMO_REPROJECTION_PX: float = 1.55
"""What `demo_rerun.py` reported for the Galileo blob reference run."""

DEMO_REGISTERED_IMAGES: int = 224
"""Registered images the plan's "Blob reference runs" table records for Galileo."""

RIGID_TOLERANCE_MM: float = 0.02
"""Rigidity noise floor: axis-angle-in-degrees storage costs ~10 um per pose."""


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic cuSFM-layout runs
# ─────────────────────────────────────────────────────────────────────────────


def transform_rig_trajectory(frames_meta: FramesMeta, *, rig_transform: pycolmap.Rigid3d, scale: float = 1.0) -> FramesMeta:
    """Move every keyframe as if the whole rig trajectory had been transformed.

    Rewrites each `camera_to_world` as
    `new_world_T_vehicle * vehicle_T_cam`, where `new_world_T_vehicle` applies
    `rig_transform` and then scales the position about the world origin. Keeping
    the rig extrinsics untouched means the collection stays perfectly rigid, so
    the rigidity spread is unaffected and only the trajectory metrics move.

    Args:
        frames_meta: Collection to transform.
        rig_transform: Rigid transform applied to every rig pose.
        scale: Multiplier applied to the rig position after the rigid transform.

    Returns:
        A new collection with the transformed poses.
    """
    poses: dict[int, pycolmap.Rigid3d] = {}
    for keyframe in frames_meta.keyframes:
        camera: CameraParams = frames_meta.cameras[keyframe.camera_params_id]
        world_T_vehicle: pycolmap.Rigid3d = frames_meta.world_T_vehicle(keyframe)
        moved: pycolmap.Rigid3d = rig_transform * world_T_vehicle
        scaled: pycolmap.Rigid3d = pycolmap.Rigid3d(moved.rotation, np.asarray(moved.translation, dtype=np.float64) * scale)
        poses[keyframe.keyframe_id] = scaled * camera.vehicle_T_cam
    return frames_meta.with_camera_to_world(poses)


def build_reconstruction(frames_meta: FramesMeta, error_px: float) -> pycolmap.Reconstruction:
    """Build a COLMAP model consistent with a collection, off by a known pixel error.

    One PINHOLE camera per `camera_params_id`, one rig whose reference sensor is
    the lowest camera id, one frame per `synced_sample_id`, and one image per
    keyframe. Points are placed in front of each camera and observed wherever they
    project inside an image; every observation is displaced by exactly `error_px`
    along +x, so the recomputed mean reprojection error is `error_px`.

    Args:
        frames_meta: Poses and calibration to mirror.
        error_px: Displacement applied to every 2D observation.

    Returns:
        A reconstruction with points, tracks and `error` set to the blob's placeholder.
    """
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    camera_ids: list[int] = sorted(frames_meta.cameras)
    for camera_params_id in camera_ids:
        camera_params: CameraParams = frames_meta.cameras[camera_params_id]
        k_matrix: Float64[ndarray, "3 3"] = np.asarray(camera_params.projection_matrix, dtype=np.float64)[:3, :3]
        camera: pycolmap.Camera = pycolmap.Camera.create_from_model_id(
            camera_params_id + 1,
            pycolmap.CameraModelId.PINHOLE,
            float(k_matrix[0, 0]),
            camera_params.image_width,
            camera_params.image_height,
        )
        camera.params = [float(k_matrix[0, 0]), float(k_matrix[1, 1]), float(k_matrix[0, 2]), float(k_matrix[1, 2])]
        reconstruction.add_camera(camera)

    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_ids[0] + 1))
    reference_vehicle_T_cam: pycolmap.Rigid3d = frames_meta.cameras[camera_ids[0]].vehicle_T_cam
    for camera_params_id in camera_ids[1:]:
        cam_T_reference: pycolmap.Rigid3d = frames_meta.cameras[camera_params_id].vehicle_T_cam.inverse() * reference_vehicle_T_cam
        rig.add_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_params_id + 1), cam_T_reference)
    reconstruction.add_rig(rig)

    by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    image_ids_by_keyframe: dict[int, int] = {}
    for frame_index, rig_frame in enumerate(frames_meta.rig_frames()):
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = frame_index + 1
        frame.rig_id = 1
        # The rig frame is the reference *camera*, not the vehicle, so the rig
        # pose carries the reference camera's extrinsic.
        frame.rig_from_world = (rig_frame.world_T_vehicle * reference_vehicle_T_cam).inverse()
        pending: list[tuple[int, KeyframeMeta]] = []
        for keyframe_id in rig_frame.keyframe_ids:
            keyframe: KeyframeMeta = by_id[keyframe_id]
            image_id: int = len(image_ids_by_keyframe) + 1
            image_ids_by_keyframe[keyframe_id] = image_id
            frame.add_data_id(pycolmap.data_t(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, keyframe.camera_params_id + 1), image_id))
            pending.append((image_id, keyframe))
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)
        for image_id, keyframe in pending:
            image: pycolmap.Image = pycolmap.Image(name=keyframe.image_name, camera_id=keyframe.camera_params_id + 1, image_id=image_id)
            image.frame_id = frame.frame_id
            reconstruction.add_image(image)

    points_xyz: Float64[ndarray, "n_points 3"] = _points_in_front_of_cameras(frames_meta)
    _observe(reconstruction, points_xyz, error_px)
    for point in reconstruction.points3D.values():
        # kpmap_to_colmap hardcodes ERROR = 2.0; the harness must recompute it.
        point.error = BLOB_ERROR_PLACEHOLDER
    return reconstruction


def _points_in_front_of_cameras(frames_meta: FramesMeta) -> Float64[ndarray, "n_points 3"]:
    """Scatter points a few metres in front of each camera at its first keyframe."""
    generator: np.random.Generator = np.random.default_rng(7)
    seen: set[int] = set()
    points: list[Float64[ndarray, "3"]] = []
    for keyframe in frames_meta.keyframes:
        if keyframe.camera_params_id in seen:
            continue
        seen.add(keyframe.camera_params_id)
        world_T_cam: pycolmap.Rigid3d = keyframe.world_T_cam
        local: Float64[ndarray, "k 3"] = np.column_stack(
            [
                generator.uniform(-1.0, 1.0, POINTS_PER_CAMERA),
                generator.uniform(-0.8, 0.8, POINTS_PER_CAMERA),
                np.full(POINTS_PER_CAMERA, POINT_DEPTH_METERS),
            ]
        )
        points.extend(local @ np.asarray(world_T_cam.rotation.matrix()).T + np.asarray(world_T_cam.translation))
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _observe(reconstruction: pycolmap.Reconstruction, points_xyz: Float64[ndarray, "n_points 3"], error_px: float) -> None:
    """Add every visible projection as an observation, displaced by `error_px` in +x."""
    observations: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(points_xyz))}
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        points_in_cam: Float64[ndarray, "n_points 3"] = (
            points_xyz @ np.asarray(cam_from_world.rotation.matrix()).T + np.asarray(cam_from_world.translation)
        )
        projected: Float64[ndarray, "n_points 2"] = np.asarray(camera.img_from_cam(points_in_cam), dtype=np.float64)
        visible: np.ndarray = (
            np.isfinite(projected).all(axis=1)
            & (points_in_cam[:, 2] > 0.0)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < camera.width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < camera.height)
        )
        indices: Int64[ndarray, "k"] = np.flatnonzero(visible).astype(np.int64)
        image.points2D = pycolmap.Point2DList(
            [pycolmap.Point2D(projected[index] + np.array([error_px, 0.0])) for index in indices]
        )
        for slot, index in enumerate(indices):
            observations[int(index)].append((image_id, slot))

    for index, track_entries in observations.items():
        if len(track_entries) < 2:
            continue
        track: pycolmap.Track = pycolmap.Track()
        for image_id, slot in track_entries:
            track.add_element(image_id, slot)
        point_id: int = reconstruction.add_point3D(points_xyz[index], track, np.array([180, 190, 200], dtype=np.uint8))
        for image_id, slot in track_entries:
            reconstruction.image(image_id).set_point3D_for_point2D(slot, point_id)


def write_synthetic_run(run_dir: Path, frames_meta: FramesMeta, *, error_px: float, runtimes: dict[str, float]) -> Path:
    """Write a complete cuSFM-layout run to disk through the production writers.

    Args:
        run_dir: Destination workspace, the equivalent of `.../cusfm`.
        frames_meta: Poses and calibration for the run.
        error_px: Pixel error to bake into the observations.
        runtimes: `runtime.csv` rows as `command -> seconds`, in insertion order.

    Returns:
        `run_dir`, now holding `sparse/`, `kpmap/keyframes/frames_meta.json` and `runtime.csv`.
    """
    write_colmap_model(run_dir, build_reconstruction(frames_meta, error_px))
    write_optimised_frames_meta(run_dir, frames_meta, {})
    for command, seconds in runtimes.items():
        append_runtime_record(run_dir, RuntimeRecord(command=command, runtime_seconds=seconds))
    return run_dir


BLOB_RUNTIMES: dict[str, float] = {
    "feature_extractor_main --input_image_directory in": 8.0,
    "generate_bow_vocabulary_main --keyframe_directory kf": 6.0,
    "generate_bow_index_main --keyframe_directory kf": 4.0,
    "generate_association_main --keyframe_dir kf": 3.0,
    "pose_graph_main --keyframe_directory kf": 3.5,
    "feature_matcher_task_builder_main --current_keyframe_directory kf": 2.0,
    "feature_matcher_main --current_keyframe_directory kf": 2.5,
    "keypoints_mapper_main --keyframe_dir kf": 6.0,
    "kpmap_to_colmap --map_dir kpmap": 2.0,
    "extract_pose_from_map_main --output_pose_dir poses": 0.5,
}
"""A blob run's `runtime.csv`, one row per binary; totals 37.5 s."""

COLSFM_RUNTIMES: dict[str, float] = {
    "colsfm.features --input in": 16.0,
    "colsfm.retrieval --keyframes kf": 5.0,
    "colsfm.loop_closure --keyframes kf": 6.0,
    "colsfm.pose_graph --keyframes kf": 7.0,
    "colsfm.pairs --keyframes kf": 4.0,
    "colsfm.matching --keyframes kf": 5.0,
    "colsfm.mapping --keyframes kf": 30.0,
    "colsfm.export --output out": 2.0,
}
"""A colsfm run's `runtime.csv`, one row per stage name; totals 75.0 s (2.0x the blob)."""

COLSFM_PIPELINE_RUNTIMES: dict[str, float] = {
    "keyframe_selection --input in": 1.0,
    "feature_extraction --input in": 15.0,
    "pair_selection --keyframes kf": 4.0,
    "matching --keyframes kf": 5.0,
    "loop_closure --keyframes kf": 6.0,
    "pose_graph --keyframes kf": 7.0,
    "reconstruction --keyframes kf": 30.0,
    "export --output out": 2.0,
}
"""The names and the order `colsfm.pipeline` actually writes — matching precedes
loop closure, so pipeline-order run detection would wrongly split this."""


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def galileo_input(galileo_input_meta: Path) -> FramesMeta:
    """The 226-keyframe input metadata shipped with r2b_galileo."""
    return read_frames_meta(galileo_input_meta)


@pytest.fixture(scope="module")
def three_samples(galileo_input: FramesMeta) -> FramesMeta:
    """The first three synchronised samples of the Galileo input, ~24 keyframes."""
    keep: list[int] = [keyframe_id for rig_frame in galileo_input.rig_frames()[:3] for keyframe_id in rig_frame.keyframe_ids]
    return galileo_input.filtered(keep)


@pytest.fixture(scope="module")
def galileo_blobref_dir(repo_root: Path) -> Path:
    """The 2026-09-05 blob reference run the plan's table was measured on."""
    return repo_root / "data" / "cusfm_runs" / "galileo_blobref" / "cusfm"


@pytest.fixture(scope="module")
def galileo_blobref(galileo_blobref_dir: Path) -> RunArtifacts:
    """The blob reference run, read through the production reader."""
    return read_run(galileo_blobref_dir, "blobref")


# ─────────────────────────────────────────────────────────────────────────────
# Alignment
# ─────────────────────────────────────────────────────────────────────────────


def test_a_rigidly_moved_trajectory_aligns_back_exactly() -> None:
    """A rotation plus a translation is undone by the fit, leaving unit scale."""
    generator: np.random.Generator = np.random.default_rng(3)
    source_xyz: Float64[ndarray, "n 3"] = generator.normal(0.0, 1.0, (40, 3))
    quaternion_xyzw: Float64[ndarray, "4"] = np.array([0.1, -0.2, 0.3, 0.9])
    rotation: Float64[ndarray, "3 3"] = pycolmap.Rotation3d(quaternion_xyzw / np.linalg.norm(quaternion_xyzw)).matrix()
    target_xyz: Float64[ndarray, "n 3"] = source_xyz @ np.asarray(rotation).T + np.array([2.0, -3.0, 0.5])

    alignment: RigidAlignment = align_rigid(source_xyz, target_xyz)

    assert alignment.rmse_meters == pytest.approx(0.0, abs=1e-12)
    assert alignment.max_error_meters == pytest.approx(0.0, abs=1e-12)
    assert alignment.would_be_scale == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(alignment.apply(source_xyz), target_xyz, atol=1e-12)


def test_a_scaled_trajectory_reports_the_scale_the_fit_refused_to_absorb() -> None:
    """Holding scale at 1 turns a 5 % shrink into a residual with a closed form.

    For `source = s * target` about the centroid, the scale-1 fit leaves
    `|s - 1| * rms_radius` of residual, and the similarity fit that was declined
    would have chosen `1 / s`.
    """
    generator: np.random.Generator = np.random.default_rng(11)
    target_xyz: Float64[ndarray, "n 3"] = generator.normal(0.0, 2.0, (60, 3))
    scale: float = 0.95
    source_xyz: Float64[ndarray, "n 3"] = scale * target_xyz

    alignment: RigidAlignment = align_rigid(source_xyz, target_xyz)

    centred: Float64[ndarray, "n 3"] = target_xyz - target_xyz.mean(axis=0)
    rms_radius: float = float(np.sqrt((centred**2).sum(axis=1).mean()))
    assert alignment.would_be_scale == pytest.approx(1.0 / scale, rel=1e-9)
    assert alignment.rmse_meters == pytest.approx(abs(scale - 1.0) * rms_radius, rel=1e-9)


def test_alignment_refuses_mismatched_trajectories() -> None:
    """Two trajectories of different lengths cannot be aligned."""
    with pytest.raises(ValueError, match="matched trajectories"):
        align_rigid(np.zeros((3, 3)), np.zeros((4, 3)))


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp joins
# ─────────────────────────────────────────────────────────────────────────────


def test_the_timestamp_join_keeps_only_pairs_inside_the_tolerance() -> None:
    """A 20 us window accepts a 2 us gap and rejects a 50 us one."""
    query: Int64[ndarray, "n"] = np.array([1000, 2000, 3000], dtype=np.int64)
    reference: Int64[ndarray, "k"] = np.array([1002, 2050, 3000], dtype=np.int64)

    query_indices, reference_indices = match_timestamps(query, reference, 20)

    assert query_indices.tolist() == [0, 2]
    assert reference_indices.tolist() == [0, 2]


def test_an_exact_join_demands_an_exact_timestamp() -> None:
    """Tolerance 0 is what the input-trajectory comparison uses."""
    query: Int64[ndarray, "n"] = np.array([10, 20], dtype=np.int64)
    reference: Int64[ndarray, "k"] = np.array([10, 21], dtype=np.int64)

    query_indices, _ = match_timestamps(query, reference, 0)

    assert query_indices.tolist() == [0]


# ─────────────────────────────────────────────────────────────────────────────
# Runtime mapping
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("feature_extractor_main --input_image_directory x", "extraction"),
        ("generate_bow_vocabulary_main --keyframe_directory x", "retrieval"),
        ("generate_bow_index_main --keyframe_directory x", "retrieval"),
        ("generate_association_main --keyframe_dir x", "association"),
        ("pose_graph_main --keyframe_directory x", "pose_graph"),
        ("pose_graph_association_main --keyframe_directory x", "pose_graph"),
        ("feature_matcher_task_builder_main --current_keyframe_directory x", "pair_selection"),
        ("feature_matcher_main --current_keyframe_directory x", "matching"),
        ("keypoints_mapper_main --keyframe_dir x", "mapping"),
        ("kpmap_to_colmap --map_dir x", "export"),
        ("extract_pose_from_map_main --output_pose_dir x", "export"),
        ("update_keyframe_pose_main --input x", "export"),
        ("colsfm.features --input x", "extraction"),
        ("colsfm.retrieval --keyframes x", "retrieval"),
        ("colsfm.loop_closure --keyframes x", "association"),
        ("colsfm.pose_graph --keyframes x", "pose_graph"),
        ("colsfm.pairs --keyframes x", "pair_selection"),
        ("colsfm.matching --keyframes x", "matching"),
        ("colsfm.mapping --keyframes x", "mapping"),
        ("colsfm.export --output x", "export"),
        # The names `colsfm.pipeline` actually writes.
        ("keyframe_selection --input x", "extraction"),
        ("feature_extraction --input x", "extraction"),
        ("pair_selection --keyframes x", "pair_selection"),
        ("matching --keyframes x", "matching"),
        ("loop_closure --keyframes x", "association"),
        ("pose_graph --keyframes x", "pose_graph"),
        ("reconstruction --keyframes x", "mapping"),
        ("export --output x", "export"),
    ],
)
def test_every_stage_command_maps_onto_a_plan_stage(command: str, expected: Stage) -> None:
    """Both vocabularies — blob binaries and colsfm stage names — land on the plan's rows."""
    assert classify_stage(command) == expected


def test_an_unknown_command_is_not_forced_into_a_stage() -> None:
    """A row the harness does not recognise is dropped, not silently mis-attributed."""
    assert classify_stage("cuvslam_main --input x") is None


def test_an_appended_runtime_log_reports_only_its_last_run(tmp_path: Path) -> None:
    """`runtime.csv` is appended to across runs; the newest block is the run."""
    for command, seconds in BLOB_RUNTIMES.items():
        append_runtime_record(tmp_path, RuntimeRecord(command=command, runtime_seconds=seconds))
    for command, seconds in BLOB_RUNTIMES.items():
        append_runtime_record(tmp_path, RuntimeRecord(command=command, runtime_seconds=2.0 * seconds))

    from colsfm.export import read_runtime_records

    records: list[RuntimeRecord] = read_runtime_records(tmp_path / "runtime.csv")
    assert len(records) == 2 * len(BLOB_RUNTIMES)
    assert len(latest_run_records(records)) == len(BLOB_RUNTIMES)

    totals: dict[Stage, float] = stage_runtime_seconds(records)
    assert totals["extraction"] == pytest.approx(16.0)
    assert totals["retrieval"] == pytest.approx(20.0)
    assert totals["export"] == pytest.approx(5.0)
    assert sum(totals.values()) == pytest.approx(2.0 * sum(BLOB_RUNTIMES.values()))


def test_a_run_whose_stages_are_out_of_pipeline_order_is_not_split() -> None:
    """`colsfm` matches before it closes loops; that is one run, not two."""
    rows: list[RuntimeRecord] = [
        RuntimeRecord(command=command, runtime_seconds=seconds) for command, seconds in COLSFM_PIPELINE_RUNTIMES.items()
    ]

    assert len(latest_run_records(rows)) == len(rows)
    assert sum(stage_runtime_seconds(rows).values()) == pytest.approx(sum(COLSFM_PIPELINE_RUNTIMES.values()))


def test_a_row_repeated_verbatim_mid_run_is_not_a_run_boundary() -> None:
    """`kpmap_to_colmap` takes no config path, so its row repeats across runs.

    Keying the block boundary on the whole row would cut the second run in half
    at that line; keying it on the command *name* does not.
    """
    rows: list[RuntimeRecord] = [
        RuntimeRecord(command="feature_extractor_main --config isaac", runtime_seconds=1.0),
        RuntimeRecord(command="kpmap_to_colmap --map_dir kpmap", runtime_seconds=2.0),
        RuntimeRecord(command="feature_extractor_main --config fixed", runtime_seconds=3.0),
        RuntimeRecord(command="kpmap_to_colmap --map_dir kpmap", runtime_seconds=4.0),
    ]

    kept: list[RuntimeRecord] = latest_run_records(rows)

    assert [record.runtime_seconds for record in kept] == [3.0, 4.0]


def test_a_sharded_stage_stays_inside_one_run(tmp_path: Path) -> None:
    """Several `feature_matcher_main` task files are one stage, not two runs."""
    rows: list[RuntimeRecord] = [
        RuntimeRecord(command="feature_extractor_main --input x", runtime_seconds=1.0),
        RuntimeRecord(command="feature_matcher_main --task_file a", runtime_seconds=2.0),
        RuntimeRecord(command="feature_matcher_main --task_file b", runtime_seconds=3.0),
    ]

    totals: dict[Stage, float] = stage_runtime_seconds(rows)

    assert totals["extraction"] == pytest.approx(1.0)
    assert totals["matching"] == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rig geometry
# ─────────────────────────────────────────────────────────────────────────────


def test_the_shipped_galileo_rig_is_rigid(galileo_input: FramesMeta) -> None:
    """Every camera of a sample agrees on the rig pose to within 10 um.

    Not exactly zero: `camera_to_world` is stored as an axis and an angle in
    *degrees*, so each camera's pose carries its own float64 round-trip noise.
    The demo prints this as `0.00 mm`; `RIGID_TOLERANCE_MM` is that, resolved.
    """
    assert rig_rigidity_spread_millimeters(galileo_input) < RIGID_TOLERANCE_MM


def test_displacing_one_camera_moves_the_rigidity_spread_by_one_minus_one_over_k(galileo_input: FramesMeta) -> None:
    """With `k` cameras, an offset `d` on one leaves `(1 - 1/k) * d` after the mean.

    The mean of the `k` per-camera estimates moves by `d/k`, so the displaced
    camera sits `(1 - 1/k) d` from it — the largest deviation in the sample, and
    hence the reported spread.
    """
    sample = galileo_input.rig_frames()[0]
    num_cameras: int = len(sample.keyframe_ids)
    victim: int = sample.keyframe_ids[-1]
    original: pycolmap.Rigid3d = galileo_input.keyframe_by_id()[victim].world_T_cam
    displacement_meters: float = 0.008
    moved: pycolmap.Rigid3d = pycolmap.Rigid3d(
        original.rotation, np.asarray(original.translation, dtype=np.float64) + np.array([displacement_meters, 0.0, 0.0])
    )

    spread: float = rig_rigidity_spread_millimeters(galileo_input.with_camera_to_world({victim: moved}))

    expected_millimeters: float = 1000.0 * (1.0 - 1.0 / num_cameras) * displacement_meters
    assert spread == pytest.approx(expected_millimeters, abs=RIGID_TOLERANCE_MM)


def test_the_rig_track_follows_the_synchronised_samples(galileo_input: FramesMeta) -> None:
    """One rig pose per `synced_sample_id`, in ascending timestamp order."""
    track: RigTrack = rig_track_from_frames_meta(galileo_input)

    assert len(track) == len(galileo_input.rig_frames())
    assert np.all(np.diff(track.timestamps_microseconds) > 0)
    assert track.world_t_rig.shape == (len(track), 3)
    assert track.world_R_rig.shape == (len(track), 3, 3)


def test_the_rig_track_drops_samples_no_image_registered(galileo_input: FramesMeta) -> None:
    """Passing a registered-image filter keeps only the samples it covers."""
    first = galileo_input.rig_frames()[0]
    by_id: dict[int, KeyframeMeta] = galileo_input.keyframe_by_id()
    names: set[str] = {by_id[keyframe_id].image_name for keyframe_id in first.keyframe_ids}

    track: RigTrack = rig_track_from_frames_meta(galileo_input, keep_image_names=names)

    assert len(track) == 1
    assert int(track.timestamps_microseconds[0]) == first.timestamp_microseconds


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth
# ─────────────────────────────────────────────────────────────────────────────


def test_galileo_ground_truth_joins_onto_every_rig_sample(repo_root: Path, galileo_input: FramesMeta) -> None:
    """`ground_truth.txt` matches all 29 rig samples inside the 0.02 ms window."""
    ground_truth: RigTrack | None = read_ground_truth(repo_root / "data" / "r2b_galileo")
    assert ground_truth is not None

    track: RigTrack = rig_track_from_frames_meta(galileo_input)
    query_indices, _ = match_timestamps(track.timestamps_microseconds, ground_truth.timestamps_microseconds, 20)

    assert len(query_indices) == len(track)


def test_a_dataset_without_ground_truth_reports_none(tmp_path: Path) -> None:
    """RoboCap ships none; the metric must be absent, not zero."""
    assert read_ground_truth(tmp_path) is None


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end on synthetic runs
# ─────────────────────────────────────────────────────────────────────────────


def test_a_synthetic_run_reports_the_error_it_was_built_with(three_samples: FramesMeta, tmp_path: Path) -> None:
    """The recomputed reprojection error is the pixel offset, not the stored 2.0."""
    run_dir: Path = write_synthetic_run(tmp_path / "run", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES)

    stored: pycolmap.Reconstruction = pycolmap.Reconstruction(str(run_dir / "sparse"))
    assert stored.compute_mean_reprojection_error() == pytest.approx(BLOB_ERROR_PLACEHOLDER)

    artifacts: RunArtifacts = read_run(run_dir, "synthetic")
    assert artifacts.reconstruction.compute_mean_reprojection_error() == pytest.approx(SYNTHETIC_ERROR_PX, abs=1e-6)


def test_a_rigidly_offset_run_scores_zero_against_the_reference(three_samples: FramesMeta, tmp_path: Path, repo_root: Path) -> None:
    """Moving the whole rig trajectory rigidly is invisible to the ATE, by design."""
    offset: pycolmap.Rigid3d = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([0.0, 0.0, np.sin(0.15), np.cos(0.15)])), np.array([1.5, -0.25, 0.75])
    )
    run_a: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "a", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES), "blob"
    )
    moved: FramesMeta = transform_rig_trajectory(three_samples, rig_transform=offset)
    run_b: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "b", moved, error_px=SYNTHETIC_ERROR_PX, runtimes=COLSFM_RUNTIMES), "colsfm"
    )

    comparison: Comparison = compare_runs(run_a, run_b, repo_root / "data" / "r2b_galileo", "galileo")

    assert comparison.run_b.vs_input.num_matched == 3
    assert comparison.run_b.vs_input.rmse_millimeters == pytest.approx(0.0, abs=1e-6)
    assert comparison.run_b.vs_input.would_be_scale == pytest.approx(1.0, abs=1e-6)
    assert comparison.pose_delta.position_rmse_millimeters == pytest.approx(0.0, abs=1e-6)
    assert comparison.pose_delta.rotation_rmse_degrees == pytest.approx(0.0, abs=1e-6)


def test_a_shrunken_run_reports_the_scale_and_the_residual(three_samples: FramesMeta, tmp_path: Path, repo_root: Path) -> None:
    """A 2 % shrink shows up as a would-be scale and a closed-form residual."""
    scale: float = 0.98
    run_a: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "a", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES), "blob"
    )
    shrunk: FramesMeta = transform_rig_trajectory(three_samples, rig_transform=pycolmap.Rigid3d(), scale=scale)
    run_b: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "b", shrunk, error_px=SYNTHETIC_ERROR_PX, runtimes=COLSFM_RUNTIMES), "colsfm"
    )

    delta = compare_runs(run_a, run_b, repo_root / "data" / "r2b_galileo", "galileo").pose_delta

    reference: Float64[ndarray, "n 3"] = run_a.track.world_t_rig
    centred: Float64[ndarray, "n 3"] = reference - reference.mean(axis=0)
    rms_radius: float = float(np.sqrt((centred**2).sum(axis=1).mean()))
    assert delta.num_matched == 3
    assert delta.position_rmse_millimeters == pytest.approx(1000.0 * abs(scale - 1.0) * rms_radius, rel=1e-6)
    assert delta.rotation_rmse_degrees == pytest.approx(0.0, abs=1e-9)


def test_the_runtime_table_maps_both_vocabularies_onto_one_set_of_rows(
    three_samples: FramesMeta, tmp_path: Path, repo_root: Path
) -> None:
    """Blob binaries and colsfm stage names share the plan's eight rows, with a B/A ratio."""
    run_a: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "a", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES), "blob"
    )
    run_b: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "b", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=COLSFM_RUNTIMES), "colsfm"
    )

    comparison: Comparison = compare_runs(run_a, run_b, repo_root / "data" / "r2b_galileo", "galileo")

    assert [row.stage for row in comparison.stages] == [
        "extraction",
        "retrieval",
        "association",
        "pose_graph",
        "pair_selection",
        "matching",
        "mapping",
        "export",
    ]
    by_stage: dict[Stage, float | None] = {row.stage: row.seconds_a for row in comparison.stages}
    assert by_stage["retrieval"] == pytest.approx(10.0)
    assert by_stage["export"] == pytest.approx(2.5)
    assert comparison.run_a.total_runtime_seconds == pytest.approx(sum(BLOB_RUNTIMES.values()))
    assert comparison.run_b.total_runtime_seconds == pytest.approx(sum(COLSFM_RUNTIMES.values()))
    assert comparison.total_runtime_ratio == pytest.approx(2.0)
    mapping_row = next(row for row in comparison.stages if row.stage == "mapping")
    assert mapping_row.ratio_b_over_a == pytest.approx(5.0)


def _synthetic_metrics(run_dir: Path, frames_meta: FramesMeta, repo_root: Path) -> RunMetrics:
    """Metrics of one synthetic run, for tests that drive `check_acceptance` directly.

    Args:
        run_dir: Where to write the run.
        frames_meta: Poses and calibration for it.
        repo_root: Repository root, for the Galileo input directory.

    Returns:
        The run's metrics, as `compare_runs` would compute them.
    """
    run: RunArtifacts = read_run(
        write_synthetic_run(run_dir, frames_meta, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES), "blob"
    )
    return compare_runs(run, run, repo_root / "data" / "r2b_galileo", "galileo").run_a


def test_a_run_that_matches_the_reference_meets_every_bound(
    three_samples: FramesMeta, tmp_path: Path, repo_root: Path
) -> None:
    """The bounds are relative, so reproducing run A exactly passes all four.

    Under the plan's old absolute figures this same pair failed: 24 registered
    images is far below 220, however faithfully B reproduces A.
    """
    run_a: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "a", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES), "blob"
    )
    run_b: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "b", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=COLSFM_RUNTIMES), "colsfm"
    )

    comparison: Comparison = compare_runs(run_a, run_b, repo_root / "data" / "r2b_galileo", "galileo")

    results: dict[str, bool] = {check.name: check.passed for check in comparison.acceptance}
    assert set(results) == {
        "registered images",
        "mean reprojection error (px)",
        "ATE vs ground truth (mm)",
        "total runtime ratio",
    }
    assert all(results.values())


def test_acceptance_flags_a_candidate_that_falls_behind_the_reference(
    three_samples: FramesMeta, tmp_path: Path, repo_root: Path
) -> None:
    """Twice A's reprojection error and twice A's runtime budget both fail."""
    slow_runtimes: dict[str, float] = {
        command: 3.0 * seconds for command, seconds in COLSFM_RUNTIMES.items()
    }
    run_a: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "a", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES), "blob"
    )
    run_b: RunArtifacts = read_run(
        write_synthetic_run(
            tmp_path / "b", three_samples, error_px=2.0 * SYNTHETIC_ERROR_PX, runtimes=slow_runtimes
        ),
        "colsfm",
    )

    comparison: Comparison = compare_runs(run_a, run_b, repo_root / "data" / "r2b_galileo", "galileo")

    results: dict[str, bool] = {check.name: check.passed for check in comparison.acceptance}
    assert results["mean reprojection error (px)"] is False
    assert results["total runtime ratio"] is False
    assert results["registered images"] is True


def test_the_registered_image_bound_allows_a_small_shortfall(
    three_samples: FramesMeta, tmp_path: Path, repo_root: Path
) -> None:
    """B may register up to four fewer images than A, and no more."""
    metrics_a: RunMetrics = _synthetic_metrics(tmp_path / "a", three_samples, repo_root)
    bounds: AcceptanceBounds = AcceptanceBounds()

    for shortfall, expected in ((bounds.registered_images_allowance, True), (bounds.registered_images_allowance + 1, False)):
        metrics_b: RunMetrics = dataclasses.replace(
            metrics_a, name="colsfm", registered_images=metrics_a.registered_images - shortfall
        )
        checks: dict[str, bool] = {
            check.name: check.passed for check in check_acceptance(metrics_a, metrics_b, 1.0, bounds)
        }
        assert checks["registered images"] is expected


def test_robocap_gets_no_acceptance_bounds(three_samples: FramesMeta, tmp_path: Path, repo_root: Path) -> None:
    """The plan sets bounds for Galileo only; RoboCap ships no ground truth."""
    run_a: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "a", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES), "blob"
    )
    run_b: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "b", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=COLSFM_RUNTIMES), "colsfm"
    )

    comparison: Comparison = compare_runs(run_a, run_b, repo_root / "data" / "r2b_galileo", "robocap")

    assert comparison.acceptance == ()


def test_the_comparison_round_trips_through_pyserde(three_samples: FramesMeta, tmp_path: Path, repo_root: Path) -> None:
    """The JSON dump is lossless, so the report and the machine-readable form agree."""
    run_a: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "a", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES), "blob"
    )
    run_b: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "b", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=COLSFM_RUNTIMES), "colsfm"
    )
    comparison: Comparison = compare_runs(run_a, run_b, repo_root / "data" / "r2b_galileo", "galileo")

    restored: Comparison = from_json(Comparison, to_json(comparison))

    assert restored == comparison
    assert "| Stage | A (s) | B (s) | B/A |" in render_markdown_report(restored)


# ─────────────────────────────────────────────────────────────────────────────
# Real data
# ─────────────────────────────────────────────────────────────────────────────


def test_the_blob_reference_run_reproduces_the_demos_numbers(galileo_blobref: RunArtifacts) -> None:
    """224 registered images and 1.55 px, the figures the demo and the plan record.

    The tolerance on the reprojection error is 0.2 px because the demo printed
    the value to two decimals after a bundle adjustment it did not re-run; the
    point of the check is that recomputing from the model lands on the measured
    figure and nowhere near the hardcoded 2.0.
    """
    reconstruction: pycolmap.Reconstruction = galileo_blobref.reconstruction
    registered: int = sum(1 for image in reconstruction.images.values() if image.num_points3D > 0)

    assert registered == DEMO_REGISTERED_IMAGES
    assert reconstruction.compute_mean_reprojection_error() == pytest.approx(DEMO_REPROJECTION_PX, abs=0.2)
    assert len(galileo_blobref.frames_meta.keyframes) == 226
    assert rig_rigidity_spread_millimeters(galileo_blobref.frames_meta) < RIGID_TOLERANCE_MM


def test_the_blob_reference_run_agrees_with_its_input_trajectory(galileo_blobref: RunArtifacts, repo_root: Path) -> None:
    """The plan records 1.8 mm RMSE against the input trajectory over 0.66 m."""
    input_meta: FramesMeta = read_frames_meta(repo_root / "data" / "r2b_galileo" / "frames_meta.json")
    from colsfm.benchmark import compute_run_metrics

    metrics = compute_run_metrics(galileo_blobref, rig_track_from_frames_meta(input_meta), None)

    assert metrics.vs_input.num_matched == 29
    assert metrics.vs_input.rmse_millimeters == pytest.approx(1.8, abs=0.3)
    assert metrics.vs_input.reference_path_length_meters == pytest.approx(0.66, abs=0.05)
    assert metrics.vs_ground_truth is None


def test_the_blob_reference_runtime_totals_match_the_plans_table(galileo_blobref: RunArtifacts) -> None:
    """40.1 s total, as recorded in the plan's "Blob reference runs" table."""
    totals: dict[Stage, float] = galileo_blobref.runtime_seconds_by_stage

    assert set(totals) == {
        "extraction",
        "retrieval",
        "association",
        "pose_graph",
        "pair_selection",
        "matching",
        "mapping",
        "export",
    }
    assert totals["extraction"] == pytest.approx(8.6, abs=0.1)
    assert totals["retrieval"] == pytest.approx(11.7, abs=0.1)
    assert totals["mapping"] == pytest.approx(6.7, abs=0.1)
    assert sum(totals.values()) == pytest.approx(40.1, abs=0.1)

"""Tests for `colsfm.loop_closure`, the Python replacement for cuSFM's loop-closure stage.

Seams under test (all public):

* `find_loop_edges` — retrieval, verification, banding, rig conversion, gating.
* `select_best_candidates` — the `SelectBestCandidates` banding rule.
* `is_good_match` / `minimum_time_gap_seconds` / `session_duration_seconds` — the gates.
* `estimate_relative_camera_pose` / `calibrated_cameras` — the two-view verification.

The scene is a synthetic rig walking a square and coming back along its first side, so
there is one true revisit with a known ground-truth relative pose. Its images are 3-D
points projected into pinhole cameras with pixel noise and gross outliers, and its matcher
is a plain `match_fn`, so nothing here needs a GPU, a network or the matching worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
import pytest
from google.protobuf.message import Message
from jaxtyping import Bool, Float32, Float64, Int
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm.frames_meta import FramesMeta, parse_message, read_frames_meta
from colsfm.loop_closure import (
    LoopClosureConfig,
    LoopClosureResult,
    MatchFunction,
    RetrievalHit,
    calibrated_cameras,
    find_loop_edges,
    is_good_match,
    minimum_time_gap_seconds,
    select_best_candidates,
    session_duration_seconds,
)
from colsfm.loop_pose import Matches
from colsfm.pose_graph import PoseGraphEdge, RigNode, gate_loop_edges, sequential_edges, solve_pose_graph
from colsfm.retrieval import (
    ALIKED_DESCRIPTOR_DIM,
    RetrievalConfig,
    RetrievalIndex,
    build_retrieval_index,
    default_good_score_threshold,
)
from colsfm.schema import KEYFRAMES_METADATA_COLLECTION, load_schema

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
"""Repo root, so data paths resolve regardless of the working directory."""

GALILEO_KEYFRAMES_META: Final[Path] = REPO_ROOT / "data/cusfm_runs/galileo/cusfm/keyframes/frames_meta.json"
"""The 226-keyframe blob run, used to check the temporal gate against real timestamps."""

RIG_PERIOD_US: Final[int] = 500_000
"""Half a second between rig frames, so a 48-frame sequence spans 23.5 s."""

IMAGE_WIDTH: Final[int] = 640
"""Synthetic image width in pixels."""

IMAGE_HEIGHT: Final[int] = 480
"""Synthetic image height in pixels."""

FOCAL_LENGTH_PX: Final[float] = 400.0
"""Synthetic focal length, giving a 77 deg horizontal field of view."""

KEYPOINTS_PER_IMAGE: Final[int] = 150
"""Observations kept per image, so every image carries the same descriptor count."""

VEHICLE_R_CAMERA: Final[Float64[ndarray, "3 3"]] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
"""Optical frame (x right, y down, z forward) expressed in the vehicle FLU frame."""


# --------------------------------------------------------------------------------------
# synthetic scene
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SquareRigScene:
    """A two-camera rig walking a square and retracing its first side."""

    frames_meta: FramesMeta
    """Metadata whose `camera_to_world` holds slightly noisy priors."""
    database_path: Path
    """COLMAP database holding one keypoint block per keyframe."""
    index: RetrievalIndex
    """Retrieval index over the same keyframes."""
    world_T_rig: dict[int, pycolmap.Rigid3d]
    """Exact ground-truth rig pose per `synced_sample_id`."""
    point_index_by_keypoint: dict[int, Int[ndarray, "n_keypoints"]]
    """Which 3-D point each keypoint of each keyframe observes."""
    revisit_offset: int
    """Rig frames between an outbound frame and the frame that retraces it."""
    n_rigs: int
    """How many rig frames the sequence has."""


def _rigid(rotation_matrix: Float64[ndarray, "3 3"], translation: Float64[ndarray, "3"]) -> pycolmap.Rigid3d:
    """Build a `Rigid3d` from a rotation matrix and a translation.

    Args:
        rotation_matrix: Float64 rotation with shape `[3, 3]`.
        translation: Float64 translation in metres with shape `[3]`.

    Returns:
        The rigid transform.
    """
    quaternion_xyzw: Float64[ndarray, "4"] = Rotation.from_matrix(rotation_matrix).as_quat()
    return pycolmap.Rigid3d(pycolmap.Rotation3d(quaternion_xyzw), np.asarray(translation, dtype=np.float64))


def _square_trajectory(side_frames: int = 10, side_meters: float = 4.0, revisit_frames: int = 8, lateral_offset_m: float = 0.3) -> list[pycolmap.Rigid3d]:
    """Walk a square, then retrace the first side 30 cm to the left.

    The lateral offset is what makes the revisit usable: two exactly coincident views give
    a degenerate essential matrix, so the pair needs a real baseline.

    Args:
        side_frames: Rig frames per side of the square.
        side_meters: Side length in metres.
        revisit_frames: Rig frames spent retracing the first side.
        lateral_offset_m: How far left of the outbound track the retrace runs.

    Returns:
        `world_T_vehicle` per rig frame, in time order.
    """
    step: float = side_meters / side_frames
    corners: list[tuple[float, float]] = [(0.0, 0.0), (side_meters, 0.0), (side_meters, side_meters), (0.0, side_meters)]
    directions: list[tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    yaws_deg: list[float] = [0.0, 90.0, 180.0, 270.0]

    poses: list[pycolmap.Rigid3d] = []
    for side in range(4):
        for frame in range(side_frames):
            position: Float64[ndarray, "3"] = np.array(
                [corners[side][0] + directions[side][0] * step * frame, corners[side][1] + directions[side][1] * step * frame, 0.0]
            )
            poses.append(_rigid(Rotation.from_euler("z", yaws_deg[side], degrees=True).as_matrix(), position))
    for frame in range(revisit_frames):
        position = np.array([step * frame, lateral_offset_m, 0.0])
        poses.append(_rigid(Rotation.from_euler("z", 0.0, degrees=True).as_matrix(), position))
    return poses


def _camera_extrinsics(baseline_m: float = 0.15) -> dict[int, pycolmap.Rigid3d]:
    """Two forward-looking cameras straddling the vehicle centre.

    Args:
        baseline_m: Distance between the two optical centres, in metres.

    Returns:
        `vehicle_T_cam` per `camera_params_id`; 0 is the left camera, 1 the right.
    """
    return {
        0: _rigid(VEHICLE_R_CAMERA, np.array([0.0, baseline_m / 2.0, 0.0])),
        1: _rigid(VEHICLE_R_CAMERA, np.array([0.0, -baseline_m / 2.0, 0.0])),
    }


def _project(
    cam_T_world: pycolmap.Rigid3d, points_xyz: Float64[ndarray, "n_points 3"]
) -> tuple[Float64[ndarray, "n_points 2"], Bool[ndarray, "n_points"], Float64[ndarray, "n_points"]]:
    """Project world points into a pinhole camera and say which land on the sensor.

    Args:
        cam_T_world: Pose of the world in the camera frame.
        points_xyz: Float64 world points with shape `[n_points, 3]`.

    Returns:
        Pixel coordinates `[n_points, 2]`, a visibility mask `[n_points]`, and the depth
        along the optical axis `[n_points]`.
    """
    rotated: Float64[ndarray, "n_points 3"] = points_xyz @ np.asarray(cam_T_world.rotation.matrix(), dtype=np.float64).T
    in_camera: Float64[ndarray, "n_points 3"] = rotated + np.asarray(cam_T_world.translation, dtype=np.float64)
    depth: Float64[ndarray, "n_points"] = in_camera[:, 2]
    safe_depth: Float64[ndarray, "n_points"] = np.where(depth > 1e-6, depth, 1.0)
    pixels: Float64[ndarray, "n_points 2"] = np.stack(
        [
            FOCAL_LENGTH_PX * in_camera[:, 0] / safe_depth + IMAGE_WIDTH / 2.0,
            FOCAL_LENGTH_PX * in_camera[:, 1] / safe_depth + IMAGE_HEIGHT / 2.0,
        ],
        axis=1,
    )
    visible: Bool[ndarray, "n_points"] = (
        (depth > 0.5) & (pixels[:, 0] >= 0.0) & (pixels[:, 0] < IMAGE_WIDTH) & (pixels[:, 1] >= 0.0) & (pixels[:, 1] < IMAGE_HEIGHT)
    )
    return pixels, visible, depth


def _projection_matrix_message(message: Message) -> None:
    """Fill a `MatrixD` with the synthetic 3x4 rectified projection matrix.

    Args:
        message: The `calibration_parameters.projection_matrix` submessage to fill.
    """
    message.row_count = 3
    message.column_count = 4
    message.data.extend(
        [FOCAL_LENGTH_PX, 0.0, IMAGE_WIDTH / 2.0, 0.0, 0.0, FOCAL_LENGTH_PX, IMAGE_HEIGHT / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )


def _write_rigid(message: Message, pose: pycolmap.Rigid3d) -> None:
    """Write a `Rigid3d` into a `RigidTransform3d` message as axis-angle in degrees.

    Args:
        message: The `RigidTransform3d` submessage to fill.
        pose: The transform to store.
    """
    rotation_vector: Float64[ndarray, "3"] = Rotation.from_quat(pose.rotation.quat).as_rotvec()
    angle_rad: float = float(np.linalg.norm(rotation_vector))
    axis: Float64[ndarray, "3"] = np.array([1.0, 0.0, 0.0]) if angle_rad == 0.0 else rotation_vector / angle_rad
    message.axis_angle.x, message.axis_angle.y, message.axis_angle.z = (float(value) for value in axis)
    message.axis_angle.angle_degrees = float(np.rad2deg(angle_rad))
    message.translation.x, message.translation.y, message.translation.z = (float(value) for value in pose.translation)


def _build_frames_meta(
    world_T_cam: dict[int, pycolmap.Rigid3d],
    camera_of_keyframe: dict[int, int],
    rig_of_keyframe: dict[int, int],
    timestamps_us: dict[int, int],
    extrinsics: dict[int, pycolmap.Rigid3d],
) -> FramesMeta:
    """Assemble a `KeyframesMetadataCollection` for the synthetic rig.

    Args:
        world_T_cam: Prior camera pose per keyframe id.
        camera_of_keyframe: `camera_params_id` per keyframe id.
        rig_of_keyframe: `synced_sample_id` per keyframe id.
        timestamps_us: Capture time in microseconds per keyframe id.
        extrinsics: `vehicle_T_cam` per `camera_params_id`.

    Returns:
        The parsed metadata.
    """
    collection_class = load_schema().message_class(KEYFRAMES_METADATA_COLLECTION)
    message: Message = collection_class()
    for keyframe_id in sorted(world_T_cam):
        entry = message.keyframes_metadata.add()
        entry.id = keyframe_id
        entry.camera_params_id = camera_of_keyframe[keyframe_id]
        entry.timestamp_microseconds = timestamps_us[keyframe_id]
        entry.image_name = f"cam{camera_of_keyframe[keyframe_id]}/{keyframe_id}.jpeg"
        entry.synced_sample_id = rig_of_keyframe[keyframe_id]
        _write_rigid(entry.camera_to_world, world_T_cam[keyframe_id])
    for camera_id, vehicle_T_cam in extrinsics.items():
        sensor = message.camera_params_id_to_camera_params[camera_id]
        sensor.sensor_meta_data.sensor_id = camera_id
        sensor.sensor_meta_data.sensor_name = f"cam{camera_id}"
        sensor.sensor_meta_data.frequency = 2.0
        _write_rigid(sensor.sensor_meta_data.sensor_to_vehicle_transform, vehicle_T_cam)
        sensor.calibration_parameters.image_width = IMAGE_WIDTH
        sensor.calibration_parameters.image_height = IMAGE_HEIGHT
        _projection_matrix_message(sensor.calibration_parameters.projection_matrix)
    stereo = message.stereo_pair.add()
    stereo.left_camera_param_id = 0
    stereo.right_camera_param_id = 1
    stereo.baseline_meters = 0.15
    return parse_message(message)


def make_square_rig_scene(tmp_path: Path, n_points: int = 12000, pixel_noise_px: float = 0.5, seed: int = 3) -> SquareRigScene:
    """Build the whole synthetic scene: poses, points, images, database and index.

    Priors written into `frames_meta.json` carry 2 mm / 0.1 deg of noise, so the metric
    scale a loop edge takes from them (`docs/spec/generate_association_main.md` §7.3) is not
    the ground truth and the accuracy assertions are not tautological.

    Args:
        tmp_path: Directory the COLMAP database is written into.
        n_points: How many 3-D landmarks to scatter around the trajectory.
        pixel_noise_px: One-sigma keypoint noise in pixels.
        seed: Seed for the landmarks, the descriptors and the noise.

    Returns:
        The scene.
    """
    generator: np.random.Generator = np.random.default_rng(seed)
    trajectory: list[pycolmap.Rigid3d] = _square_trajectory()
    extrinsics: dict[int, pycolmap.Rigid3d] = _camera_extrinsics()
    points_xyz: Float64[ndarray, "n_points 3"] = np.stack(
        [
            generator.uniform(-6.0, 12.0, n_points),
            generator.uniform(-6.0, 12.0, n_points),
            generator.uniform(-1.5, 2.5, n_points),
        ],
        axis=1,
    )
    point_descriptors: Float32[ndarray, "n_points 128"] = generator.standard_normal((n_points, ALIKED_DESCRIPTOR_DIM)).astype(np.float32)
    point_descriptors /= np.linalg.norm(point_descriptors, axis=1, keepdims=True)

    world_T_rig: dict[int, pycolmap.Rigid3d] = {}
    world_T_cam_true: dict[int, pycolmap.Rigid3d] = {}
    world_T_cam_prior: dict[int, pycolmap.Rigid3d] = {}
    camera_of_keyframe: dict[int, int] = {}
    rig_of_keyframe: dict[int, int] = {}
    timestamps_us: dict[int, int] = {}
    keypoints_by_keyframe: dict[int, Float64[ndarray, "n_keypoints 2"]] = {}
    point_index_by_keypoint: dict[int, Int[ndarray, "n_keypoints"]] = {}
    descriptors_by_keyframe: dict[int, Float32[ndarray, "n_keypoints 128"]] = {}

    for rig_index, pose in enumerate(trajectory):
        rig_id: int = rig_index + 1
        world_T_rig[rig_id] = pose
        for camera_id, vehicle_T_cam in extrinsics.items():
            keyframe_id: int = rig_index * len(extrinsics) + camera_id + 1
            world_T_camera: pycolmap.Rigid3d = pose * vehicle_T_cam
            pixels, visible, depth = _project(world_T_camera.inverse(), points_xyz)
            # Nearest first: a two-view estimate constrains the translation component ALONG
            # the optical axis far more weakly than the two lateral ones, and near points
            # are what fix it. With the far points included the direction error on a 0.3 m
            # revisit baseline reaches 2 deg, i.e. 12 mm of translation error.
            visible_indices: Int[ndarray, "n_visible"] = np.flatnonzero(visible)
            chosen: Int[ndarray, "n_keypoints"] = visible_indices[np.argsort(depth[visible_indices])][:KEYPOINTS_PER_IMAGE]
            if len(chosen) < KEYPOINTS_PER_IMAGE:
                raise ValueError(f"keyframe {keyframe_id} sees only {len(chosen)} points; scatter more landmarks")
            observed: Float64[ndarray, "n_keypoints 2"] = pixels[chosen] + generator.standard_normal((len(chosen), 2)) * pixel_noise_px
            keypoints_by_keyframe[keyframe_id] = observed
            point_index_by_keypoint[keyframe_id] = chosen
            noisy_descriptors: Float32[ndarray, "n_keypoints 128"] = point_descriptors[chosen] + generator.standard_normal(
                (len(chosen), ALIKED_DESCRIPTOR_DIM)
            ).astype(np.float32) * np.float32(0.02)
            descriptors_by_keyframe[keyframe_id] = np.ascontiguousarray(
                noisy_descriptors / np.linalg.norm(noisy_descriptors, axis=1, keepdims=True)
            )
            world_T_cam_true[keyframe_id] = world_T_camera
            world_T_cam_prior[keyframe_id] = pycolmap.Rigid3d(
                pycolmap.Rotation3d(
                    (Rotation.from_quat(world_T_camera.rotation.quat) * Rotation.from_rotvec(generator.standard_normal(3) * np.deg2rad(0.1))).as_quat()
                ),
                np.asarray(world_T_camera.translation, dtype=np.float64) + generator.standard_normal(3) * 0.002,
            )
            camera_of_keyframe[keyframe_id] = camera_id
            rig_of_keyframe[keyframe_id] = rig_id
            timestamps_us[keyframe_id] = rig_index * RIG_PERIOD_US

    database_path: Path = tmp_path / "database.db"
    database: pycolmap.Database = pycolmap.Database.open(str(database_path))
    frames_meta: FramesMeta = _build_frames_meta(world_T_cam_prior, camera_of_keyframe, rig_of_keyframe, timestamps_us, extrinsics)
    for camera_id, camera in calibrated_cameras(frames_meta).items():
        database.write_camera(camera, use_camera_id=True)
        del camera_id
    for keyframe_id, keypoints in keypoints_by_keyframe.items():
        database.write_image(
            pycolmap.Image(image_id=keyframe_id, name=f"cam{camera_of_keyframe[keyframe_id]}/{keyframe_id}.png", camera_id=camera_of_keyframe[keyframe_id]),
            use_image_id=True,
        )
        database.write_keypoints(keyframe_id, np.ascontiguousarray(keypoints, dtype=np.float32))
    database.close()

    index: RetrievalIndex = build_retrieval_index(descriptors_by_keyframe, RetrievalConfig(max_descriptors_per_image=KEYPOINTS_PER_IMAGE))
    return SquareRigScene(
        frames_meta=frames_meta,
        database_path=database_path,
        index=index,
        world_T_rig=world_T_rig,
        point_index_by_keypoint=point_index_by_keypoint,
        revisit_offset=40,
        n_rigs=len(trajectory),
    )


def make_match_function(scene: SquareRigScene, outlier_ratio: float = 0.25, seed: int = 11) -> MatchFunction:
    """Build a `match_fn` from the known point correspondences, with gross outliers added.

    Args:
        scene: The synthetic scene.
        outlier_ratio: Fraction of extra, wrong matches relative to the true ones.
        seed: Seed for the outlier draw.

    Returns:
        A callable `match_fn(image_id_a, image_id_b) -> Int[ndarray, "n_matches 2"]`.
    """
    generator: np.random.Generator = np.random.default_rng(seed)

    def match_fn(image_id_a: int, image_id_b: int) -> Matches:
        """Return true correspondences plus a fixed fraction of random wrong pairs."""
        points_a: Int[ndarray, "n_a"] = scene.point_index_by_keypoint[image_id_a]
        points_b: Int[ndarray, "n_b"] = scene.point_index_by_keypoint[image_id_b]
        position_in_b: dict[int, int] = {int(point): position for position, point in enumerate(points_b)}
        true_matches: list[tuple[int, int]] = [
            (position, position_in_b[int(point)]) for position, point in enumerate(points_a) if int(point) in position_in_b
        ]
        n_outliers: int = round(outlier_ratio * len(true_matches))
        outliers: list[tuple[int, int]] = [
            (int(generator.integers(len(points_a))), int(generator.integers(len(points_b)))) for _ in range(n_outliers)
        ]
        return np.array(true_matches + outliers, dtype=np.int64).reshape(-1, 2)

    return match_fn


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> SquareRigScene:
    """The synthetic square-walking rig, built once for the module."""
    return make_square_rig_scene(tmp_path_factory.mktemp("loop_closure"))


@pytest.fixture(scope="module")
def match_fn(scene: SquareRigScene) -> MatchFunction:
    """A matcher over the synthetic scene."""
    return make_match_function(scene)


@pytest.fixture(scope="module")
def loop_result(scene: SquareRigScene, match_fn: MatchFunction) -> LoopClosureResult:
    """The loop-closure run over the synthetic scene, with the stage switched on."""
    return find_loop_edges(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig(enabled=True), match_fn)


# --------------------------------------------------------------------------------------
# c. loop closure end to end
# --------------------------------------------------------------------------------------


def test_the_stage_is_off_by_default(scene: SquareRigScene, match_fn: MatchFunction) -> None:
    """`LoopClosureConfig.enabled` is False, and nothing runs until a caller flips it.

    The pose-graph spec measured that loops move Galileo by 5-13 mm against a 5 mm ATE
    budget, so the default must be inert (`docs/spec/pose_graph_main.md` §8).
    """
    assert LoopClosureConfig().enabled is False
    result: LoopClosureResult = find_loop_edges(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig(), match_fn)
    assert result.edges == []
    assert result.diagnostics.enabled is False
    assert result.diagnostics.queries == 0


def test_the_score_gate_falls_back_to_the_index_backend_and_can_be_overridden(
    scene: SquareRigScene, match_fn: MatchFunction, loop_result: LoopClosureResult
) -> None:
    """`good_score_threshold` is None by default and resolves against the index's backend.

    The two `colsfm.retrieval` backends score different quantities, so one hard-coded
    constant would be right for at most one of them. The diagnostics report the value that
    was actually applied, exactly as they do for the temporal gap.
    """
    assert LoopClosureConfig().good_score_threshold is None
    assert scene.index.backend == "brute_force"
    assert loop_result.diagnostics.good_score_threshold == default_good_score_threshold("brute_force")

    strict: LoopClosureResult = find_loop_edges(
        scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig(enabled=True, good_score_threshold=0.99), match_fn
    )
    assert strict.diagnostics.good_score_threshold == 0.99
    assert strict.diagnostics.rejected_by_score > 0
    assert strict.edges == []


def test_loop_edges_recover_the_ground_truth_relative_rig_pose(scene: SquareRigScene, loop_result: LoopClosureResult) -> None:
    """Every emitted edge is within 1 cm and 0.5 deg of the true rig-to-rig transform.

    The rotation and the translation *direction* come from the images through
    `estimate_two_view_geometry`; only the magnitude comes from the priors, which carry
    2 mm of noise. So this checks the essential-matrix estimate, the prior-pose scale and
    the camera-to-rig conversion together.
    """
    assert loop_result.edges, f"no loop edges: {loop_result.diagnostics}"
    for edge in loop_result.edges:
        assert edge.kind == "loop"
        expected: pycolmap.Rigid3d = scene.world_T_rig[edge.source].inverse() * scene.world_T_rig[edge.target]
        error: pycolmap.Rigid3d = expected.inverse() * edge.source_T_target
        translation_error_m: float = float(np.linalg.norm(error.translation))
        rotation_error_deg: float = float(np.rad2deg(np.linalg.norm(Rotation.from_quat(error.rotation.quat).as_rotvec())))
        assert translation_error_m < 0.01, f"edge {edge.source}->{edge.target} off by {translation_error_m * 1e3:.1f} mm"
        assert rotation_error_deg < 0.5, f"edge {edge.source}->{edge.target} off by {rotation_error_deg:.3f} deg"


def test_loop_edges_link_the_outbound_track_to_its_retrace(scene: SquareRigScene, loop_result: LoopClosureResult) -> None:
    """The edges connect the first side of the square to the frames that retrace it."""
    linked: set[tuple[int, int]] = {(min(edge.source, edge.target), max(edge.source, edge.target)) for edge in loop_result.edges}
    assert linked
    for low, high in linked:
        assert high - low >= scene.revisit_offset - 4
    print(f"loop closure diagnostics: {loop_result.diagnostics}")


def test_loop_edges_reduce_trajectory_error_when_the_odometry_drifts(scene: SquareRigScene, loop_result: LoopClosureResult) -> None:
    """A pose graph of drifting sequential edges plus these loop edges lands closer to truth.

    The sequential measurements carry a small per-step error that integrates into a
    trajectory drift, exactly as odometry does; the loop edges are the ones just verified
    against ground truth. Solving with both must beat solving with the sequential edges
    alone, which reproduce the drift exactly.
    """
    generator: np.random.Generator = np.random.default_rng(5)
    rig_ids: list[int] = sorted(scene.world_T_rig)
    drifted: dict[int, pycolmap.Rigid3d] = {rig_ids[0]: scene.world_T_rig[rig_ids[0]]}
    for previous, current in zip(rig_ids, rig_ids[1:], strict=False):
        true_step: pycolmap.Rigid3d = scene.world_T_rig[previous].inverse() * scene.world_T_rig[current]
        perturbation: pycolmap.Rigid3d = pycolmap.Rigid3d(
            pycolmap.Rotation3d(Rotation.from_rotvec(generator.standard_normal(3) * np.deg2rad(0.25)).as_quat()),
            generator.standard_normal(3) * 0.004,
        )
        drifted[current] = drifted[previous] * (true_step * perturbation)

    nodes: list[RigNode] = [
        RigNode(rig_id=rig_id, world_T_rig=drifted[rig_id], timestamp_us=(rig_id - 1) * RIG_PERIOD_US) for rig_id in rig_ids
    ]
    odometry_edges: list[PoseGraphEdge] = sequential_edges(nodes, connected_keyframe_num=1)

    def trajectory_error(poses: dict[int, pycolmap.Rigid3d]) -> float:
        """Mean absolute position error against ground truth, in metres."""
        return float(np.mean([np.linalg.norm(poses[rig_id].translation - scene.world_T_rig[rig_id].translation) for rig_id in rig_ids]))

    without_loops: float = trajectory_error(solve_pose_graph(nodes, odometry_edges).world_T_rig)
    with_loops: float = trajectory_error(solve_pose_graph(nodes, [*odometry_edges, *loop_result.edges]).world_T_rig)
    print(f"pose graph mean position error: {without_loops * 1e3:.1f} mm without loops, {with_loops * 1e3:.1f} mm with them")
    assert without_loops > 0.02, "the injected drift is too small for the test to mean anything"
    assert with_loops < 0.6 * without_loops


def test_stored_weights_scale_the_information_matrix_by_the_inlier_count(scene: SquareRigScene, match_fn: MatchFunction) -> None:
    """`use_stored_weights` reproduces the blob's `inliers * loop_residual_weight * I6`.

    It is off by default: the blob's own solver discards the stored matrix, and measured
    against its archived Galileo result the identity lands 0.28 mm away against 2.17 mm
    (`docs/spec/pose_graph_main.md` §8 item 6).

    The check reads the inlier count off the candidate rather than comparing two runs
    edge for edge: both generalized estimators are RANSAC and pycolmap's own
    `set_random_seed` does not reach them, so a pair sitting on the inlier gate falls either
    way between runs. The poses themselves agree to 0.1 mm; only the gate decision moves.
    """
    config: LoopClosureConfig = LoopClosureConfig(enabled=True, use_stored_weights=True)
    identity_run: LoopClosureResult = find_loop_edges(
        scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig(enabled=True), match_fn
    )
    weighted_run: LoopClosureResult = find_loop_edges(scene.frames_meta, scene.database_path, scene.index, config, match_fn)
    assert identity_run.edges
    assert weighted_run.edges
    for edge in identity_run.edges:
        np.testing.assert_array_equal(edge.information, np.eye(6))

    inliers_by_pair: dict[tuple[int, int], int] = {
        (candidate.source_rig_id, candidate.target_rig_id): candidate.num_inliers for candidate in weighted_run.candidates
    }
    for edge in weighted_run.edges:
        expected: float = inliers_by_pair[(edge.source, edge.target)] * config.loop_residual_weight
        np.testing.assert_allclose(edge.information, np.eye(6) * expected)


# --------------------------------------------------------------------------------------
# c. candidate banding
# --------------------------------------------------------------------------------------


def _hit(delta_seconds: float, score: float, candidate_keyframe_id: int) -> RetrievalHit:
    """Build a retrieval hit with a chosen time gap and score.

    Args:
        delta_seconds: `t_query - t_candidate` in seconds.
        score: Retrieval score.
        candidate_keyframe_id: Id to identify the hit by in the assertions.

    Returns:
        The hit.
    """
    return RetrievalHit(
        query_keyframe_id=1,
        candidate_keyframe_id=candidate_keyframe_id,
        source_rig_id=1,
        target_rig_id=candidate_keyframe_id + 100,
        score=score,
        delta_seconds=delta_seconds,
    )


def test_two_hits_in_the_same_band_collapse_to_the_better_scoring_one() -> None:
    """`SelectBestCandidates` keeps one hit per 10 s band of `|dt|` (spec §6.4).

    The band is ranked by retrieval score, not by inlier count: nothing has been matched
    when the band is chosen. See `select_best_candidates` for why.
    """
    kept: list[RetrievalHit] = select_best_candidates(
        [_hit(20.0, 0.2, candidate_keyframe_id=1), _hit(23.0, 0.5, candidate_keyframe_id=2)], band_seconds=10.0
    )
    assert [hit.candidate_keyframe_id for hit in kept] == [2]


def test_a_weaker_hit_does_not_displace_the_incumbent_of_its_band() -> None:
    """The first hit of a band survives a later, lower-scoring one."""
    kept: list[RetrievalHit] = select_best_candidates(
        [_hit(20.0, 0.7, candidate_keyframe_id=1), _hit(22.0, 0.2, candidate_keyframe_id=2)], band_seconds=10.0
    )
    assert [hit.candidate_keyframe_id for hit in kept] == [1]


def test_hits_in_different_bands_both_survive() -> None:
    """A revisit far from the first one opens a new band and is kept."""
    kept: list[RetrievalHit] = select_best_candidates(
        [_hit(20.0, 0.5, candidate_keyframe_id=1), _hit(45.0, 0.4, candidate_keyframe_id=2)], band_seconds=10.0
    )
    assert [hit.candidate_keyframe_id for hit in kept] == [1, 2]


# --------------------------------------------------------------------------------------
# d. the gates
# --------------------------------------------------------------------------------------


def test_is_good_accepts_a_strong_pair_and_rejects_a_weak_one() -> None:
    """`inliers > 30 and inliers / matches > 0.25`, both strict (spec §6.6, §7.1 item 6)."""
    config: LoopClosureConfig = LoopClosureConfig()
    assert is_good_match(num_inliers=60, num_matches=200, config=config) is True
    assert is_good_match(num_inliers=30, num_matches=100, config=config) is False
    assert is_good_match(num_inliers=50, num_matches=200, config=config) is False
    assert is_good_match(num_inliers=0, num_matches=0, config=config) is False


def _loop_edge(translation_m: float, rotation_deg: float) -> PoseGraphEdge:
    """Build a loop edge with a chosen translation magnitude and rotation angle.

    Args:
        translation_m: Translation magnitude in metres, along `x`.
        rotation_deg: Rotation angle in degrees, about `z`.

    Returns:
        The edge.
    """
    return PoseGraphEdge(
        source=1,
        target=2,
        source_T_target=pycolmap.Rigid3d(
            pycolmap.Rotation3d(Rotation.from_euler("z", rotation_deg, degrees=True).as_quat()), np.array([translation_m, 0.0, 0.0])
        ),
        information=np.eye(6),
        kind="loop",
    )


def test_the_translation_gate_accepts_a_short_loop_and_rejects_a_long_one() -> None:
    """`loop_edge_translation_threshold_meters` (spec §6.4 step 7), at the 3.0 m default."""
    config: LoopClosureConfig = LoopClosureConfig()
    inside: PoseGraphEdge = _loop_edge(translation_m=2.0, rotation_deg=0.0)
    outside: PoseGraphEdge = _loop_edge(translation_m=5.0, rotation_deg=0.0)
    kept: list[PoseGraphEdge] = gate_loop_edges([inside, outside], config.max_translation_m, config.max_rotation_deg)
    assert kept == [inside]


def test_the_rotation_gate_accepts_a_small_turn_and_rejects_a_large_one() -> None:
    """`loop_edge_rotation_threshold_degrees` (spec §6.4 step 7), at the 30 deg default."""
    config: LoopClosureConfig = LoopClosureConfig()
    inside: PoseGraphEdge = _loop_edge(translation_m=1.0, rotation_deg=20.0)
    outside: PoseGraphEdge = _loop_edge(translation_m=1.0, rotation_deg=75.0)
    kept: list[PoseGraphEdge] = gate_loop_edges([inside, outside], config.max_translation_m, config.max_rotation_deg)
    assert kept == [inside]


def test_the_shipped_zero_thresholds_are_dead() -> None:
    """The blob's 0.0 defaults reject every candidate, which is why the loop path is dead."""
    edges: list[PoseGraphEdge] = [_loop_edge(translation_m=0.3, rotation_deg=1.0)]
    assert gate_loop_edges(edges, max_translation_m=0.0, max_rotation_deg=0.0) == []


# --------------------------------------------------------------------------------------
# d. the temporal gate
# --------------------------------------------------------------------------------------


def test_the_time_gate_is_the_larger_of_the_two_specs() -> None:
    """A candidate must pass both the fixed threshold and the session-ratio one."""
    long_session: list[int] = [0, 400_000_000]
    config: LoopClosureConfig = LoopClosureConfig()
    assert session_duration_seconds(long_session) == pytest.approx(400.0)
    assert minimum_time_gap_seconds(long_session, config) == pytest.approx(32.0)
    short_session: list[int] = [0, 20_000_000]
    assert minimum_time_gap_seconds(short_session, config) == pytest.approx(10.0)


def test_the_fixed_ten_second_gate_is_unusable_on_galileo() -> None:
    """Galileo lasts 0.933 s, so the blob's 10 s gate rejects every candidate in the dataset.

    Only the ratio gate (0.08 x 0.933 = 75 ms, about two rig frames) is live there, which is
    why `LoopClosureConfig.loop_interval_threshold_in_seconds` must be lowered per dataset
    and why the diagnostics report the gap that was applied.
    """
    if not GALILEO_KEYFRAMES_META.is_file():
        pytest.skip(f"Galileo metadata is not present at {GALILEO_KEYFRAMES_META}")
    frames_meta: FramesMeta = read_frames_meta(GALILEO_KEYFRAMES_META)
    timestamps_us: list[int] = [keyframe.timestamp_microseconds for keyframe in frames_meta.keyframes]
    duration: float = session_duration_seconds(timestamps_us)
    assert duration == pytest.approx(0.933, abs=0.01)
    assert minimum_time_gap_seconds(timestamps_us, LoopClosureConfig()) > duration
    lowered: LoopClosureConfig = LoopClosureConfig(loop_interval_threshold_in_seconds=0.0)
    assert minimum_time_gap_seconds(timestamps_us, lowered) == pytest.approx(0.08 * duration)


# --------------------------------------------------------------------------------------
# e. the cameras
# --------------------------------------------------------------------------------------


def test_every_camera_is_built_with_a_prior_focal_length(scene: SquareRigScene) -> None:
    """`calibrated_cameras` sets `has_prior_focal_length`, which nothing else does.

    `docs/spec/pycolmap-capabilities.md` §5: a freshly built camera has the flag false, and
    with it false every COLMAP estimator that can fall back to an uncalibrated model does —
    a fisheye pair comes back `DEGENERATE` with zero inliers and no warning. The loop stage
    no longer estimates a two-view geometry, but the same cameras go on to the mapper.
    """
    naive: pycolmap.Camera = pycolmap.Camera(
        camera_id=0, model="PINHOLE", width=IMAGE_WIDTH, height=IMAGE_HEIGHT, params=[FOCAL_LENGTH_PX, FOCAL_LENGTH_PX, IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2]
    )
    assert naive.has_prior_focal_length is False
    cameras: dict[int, pycolmap.Camera] = calibrated_cameras(scene.frames_meta)
    assert cameras
    for camera in cameras.values():
        assert camera.has_prior_focal_length is True

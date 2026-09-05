"""Triangulation and bundle adjustment, on synthetic rigs and against the Galileo blob.

The seams are `colsfm.mapping.load_correspondences` and
`colsfm.mapping.run_mapping`, both observed through the `pycolmap.Reconstruction`
they leave behind and the `MappingResult` they return.

Two kinds of input feed them:

- **Synthetic** — a rig whose geometry, points and observation noise are known,
  so a recovered point count or a pose error is measured against ground truth
  rather than against the code's own opinion.
- **Galileo** — the blob's own keyframes and matches, loaded into a COLMAP
  database by `_write_blob_database` below, so the only thing that differs
  between cuSFM's numbers and ours is the mapper.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from conftest import pose_delta, project_and_mask, read_blob_keyframes
from google.protobuf.message import Message
from jaxtyping import Bool, Float64, Int, UInt32
from numpy import ndarray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from colsfm.cameras import colmap_cameras
from colsfm.config import BundleAdjustmentConfig, CusfmConfig, VisionMappingConfig, read_config_directory
from colsfm.export import read_runtime_records
from colsfm.frames_meta import FramesMeta, KeyframeMeta, parse_message, read_frames_meta, write_rigid_transform
from colsfm.mapping import (
    CASPAR_STOCK_CAMERA_MODELS,
    Correspondences,
    MappingOptions,
    MappingResult,
    RoundStats,
    apply_caspar_options,
    backend_name,
    bundle_adjustment_options,
    caspar_supported_camera_models,
    ceres_polish_options,
    filter_degenerate_points,
    filter_projection_failures,
    load_correspondences,
    pixel_error_schedule,
    projection_sanity_bound_px,
    run_mapping,
)
from colsfm.reconstruction import (
    RIG_ID,
    PosedModel,
    build_reconstruction,
    camera_sensor_id,
    gauge_camera_params_id,
    gauge_rig_frame,
)
from colsfm.schema import KEYFRAMES_METADATA_COLLECTION, load_schema

IMAGE_WIDTH: int = 640
"""Width in pixels of every synthetic camera."""

IMAGE_HEIGHT: int = 480
"""Height in pixels of every synthetic camera."""

FOCAL_LENGTH_PX: float = 500.0
"""Focal length in pixels of every synthetic camera."""

PERTURBATION_M: float = 0.01
"""Translation the pose-recovery test knocks every non-gauge rig frame off by."""

PERTURBATION_DEG: float = 0.5
"""Rotation the pose-recovery test knocks every non-gauge rig frame off by."""

SYNTHETIC_NOISE_PX: float = 0.3
"""Standard deviation of the Gaussian noise added to every synthetic observation."""

MULTIVIEW_OVERSAMPLING: int = 8
"""How many candidate points to draw per kept point, before the visibility filter."""

VEHICLE_R_CAM: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
"""Rotation from the camera's RDF axes to the vehicle's FLU axes (docs/camera.md)."""


@pytest.fixture(autouse=True, scope="module")
def _quiet_glog() -> None:
    """Silence COLMAP's own logging so the tests' own numbers stay readable."""
    pycolmap.logging.minloglevel = 2


@dataclass(slots=True)
class SyntheticRig:
    """A rig sequence with known poses, known points and noisy observations."""

    frames_meta: FramesMeta
    """Metadata carrying the true (or perturbed) poses, as the mapper would read it."""
    points_xyz: Float64[ndarray, "num_points 3"]
    """Ground-truth world points in metres."""
    keypoints_px: dict[int, Float64[ndarray, "num_observations 2"]]
    """Noisy pixel observations per keyframe id, in write order."""
    point_indices: dict[int, Int[ndarray, " num_observations"]]
    """Index into `points_xyz` of every observation, per keyframe id."""
    world_T_vehicle: dict[int, pycolmap.Rigid3d]
    """The true vehicle pose per `synced_sample_id`, before any perturbation."""


def _pinhole_projection(camera_matrix: Float64[ndarray, "3 3"]) -> list[float]:
    """A 3x4 projection matrix, row-major, from a 3x3 intrinsic matrix.

    Args:
        camera_matrix: The intrinsics.

    Returns:
        The twelve row-major entries cuSFM stores in `projection_matrix`.
    """
    projection: Float64[ndarray, "3 4"] = np.hstack([camera_matrix, np.zeros((3, 1))])
    return [float(value) for value in projection.reshape(-1)]


def _camera_matrix() -> Float64[ndarray, "3 3"]:
    """The synthetic cameras' shared intrinsic matrix.

    Returns:
        A 3x3 intrinsic matrix in pixels.
    """
    return np.array(
        [[FOCAL_LENGTH_PX, 0.0, IMAGE_WIDTH / 2.0], [0.0, FOCAL_LENGTH_PX, IMAGE_HEIGHT / 2.0], [0.0, 0.0, 1.0]]
    )


def _fill_camera(
    message: Message,
    camera_params_id: int,
    vehicle_T_cam: pycolmap.Rigid3d,
    projection_model: str,
    fisheye: bool,
) -> None:
    """Fill one `CameraSensor` entry of a synthetic collection.

    Args:
        message: The `protos.common.sensor.CameraSensor` to fill.
        camera_params_id: Its key in the map.
        vehicle_T_cam: The rig extrinsic, camera to vehicle.
        projection_model: Name of the `CameraProjectionModelType` value.
        fisheye: Write `camera_matrix` plus four distortion coefficients rather
            than a rectified `projection_matrix`.
    """
    message.sensor_meta_data.sensor_id = camera_params_id
    message.sensor_meta_data.sensor_name = f"camera_{camera_params_id}"
    message.sensor_meta_data.frequency = 30.0
    write_rigid_transform(message.sensor_meta_data.sensor_to_vehicle_transform, vehicle_T_cam)
    message.calibration_parameters.image_width = IMAGE_WIDTH
    message.calibration_parameters.image_height = IMAGE_HEIGHT
    if fisheye:
        message.calibration_parameters.camera_matrix.row_count = 3
        message.calibration_parameters.camera_matrix.column_count = 3
        message.calibration_parameters.camera_matrix.data.extend(
            [float(value) for value in _camera_matrix().reshape(-1)]
        )
        message.calibration_parameters.distortion_coefficients.row_count = 1
        message.calibration_parameters.distortion_coefficients.column_count = 4
        message.calibration_parameters.distortion_coefficients.data.extend([0.02, -0.005, 0.001, 0.0])
    else:
        message.calibration_parameters.projection_matrix.row_count = 3
        message.calibration_parameters.projection_matrix.column_count = 4
        message.calibration_parameters.projection_matrix.data.extend(_pinhole_projection(_camera_matrix()))
    message.camera_projection_model_type = message.DESCRIPTOR.fields_by_name[
        "camera_projection_model_type"
    ].enum_type.values_by_name[projection_model].number


def _world_T_vehicle_at(frame_index: int) -> pycolmap.Rigid3d:
    """The synthetic trajectory: a forward drive with a slow yaw.

    Args:
        frame_index: Zero-based rig frame index.

    Returns:
        The vehicle pose in the world frame.
    """
    yaw_degrees: float = 1.5 * frame_index
    rotation: Rotation = Rotation.from_euler("z", yaw_degrees, degrees=True)
    translation: Float64[ndarray, "3"] = np.array([0.6 * frame_index, 0.05 * frame_index, 0.0])
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rotation.as_quat()), translation)


def _project(
    reconstruction: pycolmap.Reconstruction,
    frames_meta: FramesMeta,
    points_xyz: Float64[ndarray, "num_points 3"],
) -> tuple[dict[int, Float64[ndarray, "num_points 2"]], dict[int, Bool[ndarray, " num_points"]]]:
    """Project a point cloud into every image of a reconstruction.

    Args:
        reconstruction: Poses and cameras.
        frames_meta: The collection the images came from.
        points_xyz: World points in metres.

    Returns:
        Exact pixel projections per keyframe id, and a visibility mask per keyframe
        id: in front of the camera, finite, and inside the image.
    """
    projections: dict[int, Float64[ndarray, "num_points 2"]] = {}
    visibility: dict[int, Bool[ndarray, " num_points"]] = {}
    for keyframe in frames_meta.keyframes:
        image: pycolmap.Image = reconstruction.image(keyframe.keyframe_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        projected, visible = project_and_mask(camera, image.cam_from_world(), points_xyz)
        projections[keyframe.keyframe_id] = projected
        visibility[keyframe.keyframe_id] = visible
    return projections, visibility


def build_synthetic_rig(
    num_frames: int = 20,
    num_points: int = 500,
    noise_px: float = SYNTHETIC_NOISE_PX,
    seed: int = 7,
    fisheye: bool = False,
) -> SyntheticRig:
    """Build a two-camera rig sequence with exact poses and noisy observations.

    The cameras sit half a metre apart across the vehicle and look forward, so
    every point is seen by both cameras of several rig frames — the multi-view
    geometry cuSFM's `VEHICLE_RIG` mode assumes.

    Args:
        num_frames: Rig frames along the trajectory.
        num_points: World points to scatter in front of the vehicle.
        noise_px: Standard deviation of the Gaussian pixel noise.
        seed: PRNG seed.
        fisheye: Give the cameras the OPENCV_FISHEYE model.

    Returns:
        The synthetic rig, its metadata and its observations.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    schema = load_schema()
    collection: Message = schema.message_class(KEYFRAMES_METADATA_COLLECTION)()
    projection_model: str = "OPENCV_FISHEYE" if fisheye else "PINHOLE"

    vehicle_T_cams: dict[int, pycolmap.Rigid3d] = {}
    for camera_params_id, lateral_offset_m in enumerate((0.25, -0.25)):
        vehicle_T_cams[camera_params_id] = pycolmap.Rigid3d(
            pycolmap.Rotation3d(Rotation.from_matrix(VEHICLE_R_CAM).as_quat()),
            np.array([1.2, lateral_offset_m, 1.0]),
        )
        _fill_camera(
            collection.camera_params_id_to_camera_params[camera_params_id],
            camera_params_id,
            vehicle_T_cams[camera_params_id],
            projection_model,
            fisheye,
        )

    world_T_vehicle: dict[int, pycolmap.Rigid3d] = {}
    keyframe_id: int = 0
    for frame_index in range(num_frames):
        world_T_vehicle[frame_index] = _world_T_vehicle_at(frame_index)
        for camera_params_id in sorted(vehicle_T_cams):
            entry = collection.keyframes_metadata.add()
            entry.id = keyframe_id
            entry.camera_params_id = camera_params_id
            entry.timestamp_microseconds = 1_000_000 + frame_index * 33_333
            entry.image_name = f"camera_{camera_params_id}/{frame_index:04d}.jpeg"
            entry.synced_sample_id = frame_index
            write_rigid_transform(
                entry.camera_to_world, world_T_vehicle[frame_index] * vehicle_T_cams[camera_params_id]
            )
            keyframe_id += 1

    frames_meta: FramesMeta = parse_message(collection)
    reconstruction: pycolmap.Reconstruction = build_reconstruction(frames_meta).reconstruction
    candidates: Float64[ndarray, "num_candidates 3"] = np.column_stack(
        [
            rng.uniform(2.0, 8.0 + 0.6 * num_frames, MULTIVIEW_OVERSAMPLING * num_points),
            rng.uniform(-9.0, 9.0, MULTIVIEW_OVERSAMPLING * num_points),
            rng.uniform(-1.0, 4.0, MULTIVIEW_OVERSAMPLING * num_points),
        ]
    )
    projections, visibility = _project(reconstruction, frames_meta, candidates)

    # Keep only points several images actually see: a point observed twice or
    # less cannot be triangulated under `min_correspondences: 3`, so leaving it
    # in the ground truth would measure the sampler rather than the mapper.
    seen: Int[ndarray, " num_candidates"] = np.sum(np.stack(list(visibility.values())), axis=0)
    keep: Int[ndarray, " num_kept"] = np.flatnonzero(seen >= 3)[:num_points]
    if len(keep) < num_points:
        raise ValueError(f"Only {len(keep)} of {num_points} synthetic points are seen three times")

    points_xyz: Float64[ndarray, "num_points 3"] = candidates[keep]
    keypoints: dict[int, Float64[ndarray, "num_observations 2"]] = {}
    indices: dict[int, Int[ndarray, " num_observations"]] = {}
    for keyframe in frames_meta.keyframes:
        visible: Bool[ndarray, " num_kept"] = visibility[keyframe.keyframe_id][keep]
        exact: Float64[ndarray, "num_observations 2"] = projections[keyframe.keyframe_id][keep][visible]
        keypoints[keyframe.keyframe_id] = exact + rng.normal(0.0, noise_px, exact.shape)
        indices[keyframe.keyframe_id] = np.flatnonzero(visible).astype(np.int64)
    return SyntheticRig(
        frames_meta=frames_meta,
        points_xyz=points_xyz,
        keypoints_px=keypoints,
        point_indices=indices,
        world_T_vehicle=world_T_vehicle,
    )


def _write_database(
    database_path: Path,
    frames_meta: FramesMeta,
    keypoints_px: dict[int, Float64[ndarray, "num_observations 2"]],
    matches: dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]],
    image_id_offset: int = 0,
) -> int:
    """Write cameras, images, keypoints and verified two-view geometries to SQLite.

    This stands in for the feature and matching stages: the mapper only ever
    reads keypoints and `two_view_geometries` back out, so a database carrying
    those is indistinguishable from a real one.

    Args:
        database_path: Where to create the COLMAP database.
        frames_meta: The collection the images come from.
        keypoints_px: Pixel keypoints per keyframe id.
        matches: Inlier matches per ordered image pair, as point2D index columns.
        image_id_offset: Added to every keyframe id, to write a database that
            numbers the same images differently.

    Returns:
        The number of image pairs written.
    """
    cameras: dict[int, pycolmap.Camera] = colmap_cameras(frames_meta)
    with pycolmap.Database.open(database_path) as database:
        for camera_params_id in sorted(cameras):
            camera: pycolmap.Camera = cameras[camera_params_id]
            camera.has_prior_focal_length = True
            database.write_camera(camera, use_camera_id=True)
        for keyframe in frames_meta.keyframes:
            image_id: int = keyframe.keyframe_id + image_id_offset
            image: pycolmap.Image = pycolmap.Image(
                name=keyframe.image_name, camera_id=keyframe.camera_params_id, image_id=image_id
            )
            database.write_image(image, use_image_id=True)
            database.write_keypoints(image_id, keypoints_px[keyframe.keyframe_id].astype(np.float32))
        num_pairs: int = 0
        for (image_id1, image_id2), pair_matches in sorted(matches.items()):
            if len(pair_matches) == 0:
                continue
            database.write_matches(image_id1 + image_id_offset, image_id2 + image_id_offset, pair_matches)
            geometry: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
            geometry.config = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
            geometry.inlier_matches = pair_matches
            database.write_two_view_geometry(image_id1 + image_id_offset, image_id2 + image_id_offset, geometry)
            num_pairs += 1
    return num_pairs


def _synthetic_matches(rig: SyntheticRig, min_matches: int = 20) -> dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]]:
    """Ground-truth matches between every pair of synthetic images.

    Args:
        rig: The synthetic rig.
        min_matches: Skip pairs sharing fewer points than this.

    Returns:
        Inlier matches per ordered image pair.
    """
    matches: dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]] = {}
    keyframe_ids: list[int] = sorted(rig.point_indices)
    for image_id1, image_id2 in itertools.combinations(keyframe_ids, 2):
        left: Int[ndarray, " n"] = rig.point_indices[image_id1]
        right: Int[ndarray, " n"] = rig.point_indices[image_id2]
        common: Int[ndarray, " n"] = np.intersect1d(left, right)
        if len(common) < min_matches:
            continue
        matches[(image_id1, image_id2)] = np.stack(
            [np.searchsorted(left, common), np.searchsorted(right, common)], axis=1
        ).astype(np.uint32)
    return matches


@pytest.fixture(scope="module")
def isaac_config(repo_root: Path) -> CusfmConfig:
    """The isaac profile: 5 BA rounds, a 25 -> 5 px gate, CAUCHY at sigma 4."""
    return read_config_directory(repo_root / "pycusfm" / "configs" / "isaac")


@pytest.fixture(scope="module")
def synthetic_rig() -> SyntheticRig:
    """A 20-frame, two-camera rig with 500 points and 0.3 px observation noise."""
    return build_synthetic_rig()


@pytest.fixture(scope="module")
def synthetic_database(synthetic_rig: SyntheticRig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic rig's observations as a COLMAP database."""
    database_path: Path = tmp_path_factory.mktemp("synthetic") / "database.db"
    _write_database(
        database_path, synthetic_rig.frames_meta, synthetic_rig.keypoints_px, _synthetic_matches(synthetic_rig)
    )
    return database_path


@pytest.fixture(scope="module")
def fisheye_rig() -> SyntheticRig:
    """A 6-frame, two-camera OPENCV_FISHEYE rig: the model stock CASPAR cannot project."""
    return build_synthetic_rig(num_frames=6, num_points=200, fisheye=True, seed=3)


@pytest.fixture(scope="module")
def fisheye_database(fisheye_rig: SyntheticRig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fisheye rig's observations as a COLMAP database."""
    database_path: Path = tmp_path_factory.mktemp("fisheye") / "database.db"
    _write_database(database_path, fisheye_rig.frames_meta, fisheye_rig.keypoints_px, _synthetic_matches(fisheye_rig))
    return database_path


def _quiet_options(**overrides: object) -> MappingOptions:
    """Mapping options with the per-round printing turned off.

    Args:
        **overrides: Fields to override.

    Returns:
        The options.
    """
    return dataclasses.replace(MappingOptions(verbose=False), **overrides)


def _every_round(mapping_config: VisionMappingConfig) -> VisionMappingConfig:
    """The same configuration with the early exit switched off.

    `run_mapping` leaves the outer loop when the observation change falls below
    `max_observation_change`, and the statistic is never negative, so a threshold of
    0.0 runs every one of `num_ba_iterations` rounds.

    Args:
        mapping_config: The mapping configuration.

    Returns:
        A copy whose `max_observation_change` is 0.0.
    """
    return dataclasses.replace(mapping_config, max_observation_change=0.0)


def _match_tracks_to_truth(
    reconstruction: pycolmap.Reconstruction, rig: SyntheticRig
) -> tuple[set[int], int, list[float]]:
    """Match every triangulated track back to the point that generated it.

    Identity comes from the observations, not from proximity: each track element
    names an image and a keypoint index, and the synthetic rig knows which world
    point that keypoint came from. A track whose elements disagree has merged two
    points and is a genuine failure, however close its position looks.

    Args:
        reconstruction: The triangulated model.
        rig: The synthetic rig the observations came from.

    Returns:
        The truth indices recovered, the number of tracks mixing two points, and
        the position error of each recovered point as a fraction of its distance
        to the first camera that saw it.
    """
    recovered: set[int] = set()
    mixed_tracks: int = 0
    relative_errors: list[float] = []
    for point3D_id in reconstruction.point3D_ids():
        point: pycolmap.Point3D = reconstruction.point3D(point3D_id)
        truth_indices: set[int] = {
            int(rig.point_indices[element.image_id][element.point2D_idx]) for element in point.track.elements
        }
        if len(truth_indices) != 1:
            mixed_tracks += 1
            continue
        truth_index: int = truth_indices.pop()
        recovered.add(truth_index)
        truth_xyz: Float64[ndarray, "3"] = rig.points_xyz[truth_index]
        first_image: pycolmap.Image = reconstruction.image(point.track.elements[0].image_id)
        range_m: float = float(np.linalg.norm(truth_xyz - first_image.projection_center()))
        relative_errors.append(float(np.linalg.norm(point.xyz - truth_xyz)) / range_m)
    return recovered, mixed_tracks, relative_errors


def test_pixel_error_schedule_is_the_isaac_ramp(isaac_config: CusfmConfig) -> None:
    """The gate decays linearly across the outer loop: 25, 20, 15, 10, 5 (§5.6).

    Read off a `--v=3` re-run of the blob, not derived from the code.
    """
    assert pixel_error_schedule(isaac_config.vision_mapping) == (25.0, 20.0, 15.0, 10.0, 5.0)


def test_load_correspondences_fills_points2d_and_the_graph(
    synthetic_rig: SyntheticRig, synthetic_database: Path
) -> None:
    """Keypoints land on the images and the verified pairs become the graph.

    Nothing is re-estimated: the poses the reconstruction went in with come back
    out bit-identical.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    poses_before: dict[int, Float64[ndarray, "3 4"]] = {
        frame_id: reconstruction.frame(frame_id).rig_from_world.matrix().copy() for frame_id in sorted(reconstruction.frames)
    }

    correspondences = load_correspondences(reconstruction, synthetic_database, min_num_matches=15)

    assert correspondences.num_images == reconstruction.num_images() == 40
    assert correspondences.num_image_pairs > 100
    for keyframe in synthetic_rig.frames_meta.keyframes:
        image: pycolmap.Image = reconstruction.image(keyframe.keyframe_id)
        expected: Float64[ndarray, "n 2"] = synthetic_rig.keypoints_px[keyframe.keyframe_id]
        assert image.num_points2D() == len(expected)
        assert np.allclose(np.array([point.xy for point in image.points2D]), expected, atol=1e-4)
        assert correspondences.graph.num_correspondences_for_image(keyframe.keyframe_id) > 0
    for frame_id, matrix in poses_before.items():
        assert np.array_equal(reconstruction.frame(frame_id).rig_from_world.matrix(), matrix)


def test_load_correspondences_rejects_a_database_with_other_image_ids(
    synthetic_rig: SyntheticRig, tmp_path: Path
) -> None:
    """A database that numbers the same image differently is an error, not a silent miss.

    The correspondence graph is keyed by database image id, so a mismatch would
    triangulate one image's keypoints against another's.
    """
    database_path: Path = tmp_path / "shifted.db"
    _write_database(
        database_path,
        synthetic_rig.frames_meta,
        synthetic_rig.keypoints_px,
        _synthetic_matches(synthetic_rig),
        image_id_offset=1000,
    )

    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    with pytest.raises(ValueError, match="disagree on image ids"):
        load_correspondences(reconstruction, database_path)


def test_run_mapping_recovers_the_synthetic_points(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """With exact poses, at least 95 % of the points come back at the noise floor.

    Ground truth is the point cloud the observations were generated from, so a
    recovered point is only counted when it lands within a centimetre of one.
    """
    model: PosedModel = build_reconstruction(synthetic_rig.frames_meta)
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    result: MappingResult = run_mapping(model, synthetic_database, mapping_config, _quiet_options())

    recovered, mixed_tracks, relative_errors = _match_tracks_to_truth(result.reconstruction, synthetic_rig)
    fraction: float = len(recovered) / len(synthetic_rig.points_xyz)
    print(
        f"[colsfm] synthetic: {result.num_points3D} points, {fraction:.1%} of the truth recovered, "
        f"{mixed_tracks} tracks mixing two points, reprojection {result.mean_reprojection_error_px:.4f} px, "
        f"median depth error {np.median(relative_errors) * 100:.3f} % of range"
    )
    assert fraction >= 0.95
    assert mixed_tracks == 0, "a track must not merge observations of two different points"
    # A 0.3 px noise on a 0.5 m stereo baseline puts a 25 m point a few
    # decimetres out along the ray, so the position check is relative to range.
    assert float(np.median(relative_errors)) < 0.01
    assert result.mean_reprojection_error_px <= 1.5 * SYNTHETIC_NOISE_PX
    assert result.num_registered_images == 40
    assert result.num_images_with_observations == 40
    schedule: tuple[float, ...] = pixel_error_schedule(mapping_config)
    assert [round_stats.max_pixel_error for round_stats in result.rounds] == list(schedule[: len(result.rounds)])
    assert all(round_stats.ba_termination != "FAILURE" for round_stats in result.rounds)


def test_the_outer_loop_can_run_to_the_end_or_exit_early(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """On data where nothing moves, the observation-change test ends the loop at once.

    `observation_change = (merged + completed + filtered) / observations` (§5.4).
    On exact synthetic geometry no track merges, completes or gets filtered, so
    the statistic is 0 and the loop stops after one round. cuSFM's own log says
    `use num_ba_iterations to stop BA`, i.e. its two stopping rules are
    exclusive, and a `max_observation_change` of 0.0 is how the configuration asks
    for every round regardless.
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    early: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta), synthetic_database, mapping_config, _quiet_options()
    )
    assert len(early.rounds) == 1
    assert early.rounds[0].observation_change < mapping_config.max_observation_change

    full: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        _every_round(mapping_config),
        _quiet_options(),
    )
    assert len(full.rounds) == mapping_config.num_ba_iterations == 5
    assert [round_stats.max_pixel_error for round_stats in full.rounds] == [25.0, 20.0, 15.0, 10.0, 5.0]


def _random_offset(
    rng: np.random.Generator, translation_m: float, rotation_deg: float
) -> pycolmap.Rigid3d:
    """One rigid offset of an exact size, in a random direction.

    Both the pose-recovery and the extrinsic-recovery tests need the same thing: an
    error whose magnitude is known to the metre and the degree, so that what is
    measured afterwards is how much of it came back.

    Args:
        rng: The generator; the rotation is drawn before the translation.
        translation_m: Length of the translation, metres.
        rotation_deg: Angle of the rotation, degrees.

    Returns:
        The offset transform.
    """
    rotation_vector: Float64[ndarray, "3"] = rng.normal(0.0, 1.0, 3)
    rotation_vector = rotation_vector / np.linalg.norm(rotation_vector) * np.deg2rad(rotation_deg)
    translation: Float64[ndarray, "3"] = rng.normal(0.0, 1.0, 3)
    translation = translation / np.linalg.norm(translation) * translation_m
    return pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_rotvec(rotation_vector).as_quat()), translation)


def _perturbed_metadata(rig: SyntheticRig, seed: int = 11) -> FramesMeta:
    """Knock every rig frame but the gauge one 1 cm and 0.5 deg off its true pose.

    The offset is applied to the **vehicle** pose and then pushed through the
    extrinsics, so the perturbed metadata stays rig-exact and the only thing
    bundle adjustment has to undo is the rig pose itself.

    Args:
        rig: The synthetic rig holding the true poses.
        seed: PRNG seed for the offset directions.

    Returns:
        A copy of the metadata with perturbed `camera_to_world` poses.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    keyframe_by_id: dict[int, KeyframeMeta] = rig.frames_meta.keyframe_by_id()
    gauge_sample_id: int = gauge_rig_frame(rig.frames_meta).synced_sample_id
    perturbed_poses: dict[int, pycolmap.Rigid3d] = {}
    for rig_frame in rig.frames_meta.rig_frames():
        offset: pycolmap.Rigid3d = (
            pycolmap.Rigid3d()
            if rig_frame.synced_sample_id == gauge_sample_id
            else _random_offset(rng, PERTURBATION_M, PERTURBATION_DEG)
        )
        world_T_vehicle: pycolmap.Rigid3d = rig_frame.world_T_vehicle * offset
        for keyframe_id in rig_frame.keyframe_ids:
            camera_params_id: int = keyframe_by_id[keyframe_id].camera_params_id
            perturbed_poses[keyframe_id] = world_T_vehicle * rig.frames_meta.cameras[camera_params_id].vehicle_T_cam
    return rig.frames_meta.with_camera_to_world(perturbed_poses)


@pytest.mark.parametrize(
    ("noise_px", "max_translation_m", "max_rotation_deg"),
    [(0.0, 1e-5, 1e-4), (0.1, 1e-3, 0.05)],
    ids=["exact", "0.1px-noise"],
)
def test_bundle_adjustment_pulls_perturbed_rig_poses_back(
    tmp_path: Path, isaac_config: CusfmConfig, noise_px: float, max_translation_m: float, max_rotation_deg: float
) -> None:
    """Poses knocked 1 cm and 0.5 deg off come back to the observation noise floor.

    The gauge frame is held constant, so the recovered trajectory is directly
    comparable with the truth — no alignment step hides the error. What is left
    is estimator variance, and it scales with the pixel noise: measured
    0.0 px -> 0 mm, 0.1 px -> 0.85 mm, 0.3 px -> 2.5 mm on this geometry. The
    exact case is the real proof that the perturbation itself is removed.

    Extrinsics must not move: `optimize_extrinsics` is off.
    """
    rig: SyntheticRig = build_synthetic_rig(noise_px=noise_px)
    database_path: Path = tmp_path / "perturbed.db"
    _write_database(database_path, rig.frames_meta, rig.keypoints_px, _synthetic_matches(rig))
    perturbed_meta: FramesMeta = _perturbed_metadata(rig)

    model: PosedModel = build_reconstruction(perturbed_meta)
    reconstruction: pycolmap.Reconstruction = model.reconstruction
    extrinsics_before: dict[int, Float64[ndarray, "3 4"]] = {
        camera_params_id: reconstruction.rig(RIG_ID).sensor_from_rig(camera_sensor_id(camera_params_id)).matrix().copy()
        for camera_params_id in sorted(perturbed_meta.cameras)
    }
    start_translation_m: float = max(
        pose_delta(reconstruction.frame(rig_frame.synced_sample_id).rig_from_world.inverse(), rig_frame.world_T_vehicle)[0]
        for rig_frame in rig.frames_meta.rig_frames()
    )
    assert start_translation_m == pytest.approx(PERTURBATION_M, abs=1e-6)

    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    result: MappingResult = run_mapping(
        model, database_path, _every_round(mapping_config), _quiet_options()
    )

    worst_translation_m: float = 0.0
    worst_rotation_deg: float = 0.0
    for rig_frame in rig.frames_meta.rig_frames():
        frame: pycolmap.Frame = result.reconstruction.frame(rig_frame.synced_sample_id)
        translation_m, rotation_deg = pose_delta(frame.rig_from_world.inverse(), rig_frame.world_T_vehicle)
        worst_translation_m = max(worst_translation_m, translation_m)
        worst_rotation_deg = max(worst_rotation_deg, rotation_deg)
    print(
        f"[colsfm] perturbed rig at {noise_px} px noise: worst pose error "
        f"{worst_translation_m * 1e3:.3f} mm, {worst_rotation_deg:.4f} deg after {len(result.rounds)} rounds"
    )
    assert worst_translation_m < max_translation_m
    assert worst_rotation_deg < max_rotation_deg
    for camera_params_id, matrix in extrinsics_before.items():
        after: Float64[ndarray, "3 4"] = (
            result.reconstruction.rig(RIG_ID).sensor_from_rig(camera_sensor_id(camera_params_id)).matrix()
        )
        assert np.array_equal(after, matrix), "extrinsics must not move when optimize_extrinsics is off"


def test_run_mapping_is_deterministic_on_one_thread(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """Two single-threaded runs agree on every point count and every round.

    cuSFM's own config says the same thing about its triangulator: a re-run at
    eight threads gave 1281 points against a shipped 1275 (§5.3).
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    results: list[MappingResult] = [
        run_mapping(
            build_reconstruction(synthetic_rig.frames_meta),
            synthetic_database,
            mapping_config,
            _quiet_options(num_threads=1),
        )
        for _ in range(2)
    ]
    assert results[0].num_points3D == results[1].num_points3D
    assert results[0].num_observations == results[1].num_observations
    assert [stats.num_points3D for stats in results[0].rounds] == [stats.num_points3D for stats in results[1].rounds]
    assert results[0].mean_reprojection_error_px == pytest.approx(results[1].mean_reprojection_error_px, abs=1e-9)


def test_fisheye_rig_triangulates(tmp_path: Path, isaac_config: CusfmConfig) -> None:
    """An OPENCV_FISHEYE rig produces points, with the prior focal length declared.

    This is the guard on pycolmap-capabilities.md §5: a fisheye camera whose
    `has_prior_focal_length` is left at its False default degrades to
    `DEGENERATE` with zero inliers, and nothing warns.
    """
    rig: SyntheticRig = build_synthetic_rig(num_frames=8, num_points=200, fisheye=True, seed=3)
    database_path: Path = tmp_path / "fisheye.db"
    _write_database(database_path, rig.frames_meta, rig.keypoints_px, _synthetic_matches(rig))

    model: PosedModel = build_reconstruction(rig.frames_meta)
    reconstruction: pycolmap.Reconstruction = model.reconstruction
    assert {camera.model.name for camera in reconstruction.cameras.values()} == {"OPENCV_FISHEYE"}
    assert all(camera.has_prior_focal_length for camera in reconstruction.cameras.values())

    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    result: MappingResult = run_mapping(model, database_path, mapping_config, _quiet_options())
    print(f"[colsfm] fisheye: {result.num_points3D} points, reprojection {result.mean_reprojection_error_px:.4f} px")
    assert result.num_points3D > 100
    assert result.mean_reprojection_error_px < 1.0


# ---------------------------------------------------------------------------
# The CASPAR bundle-adjustment backend and its three fallbacks
# ---------------------------------------------------------------------------


def test_the_caspar_backend_reaches_the_options_the_solver_reads(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig
) -> None:
    """A PINHOLE rig under `ba_backend="caspar"` gets CASPAR and the settings it demands.

    CASPAR throws on a free `sensor_from_rig` and on a focal length refined apart
    from the distortion block, so the options builder owes all three: the backend,
    the frozen extrinsics and the two refinement flags in agreement.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    assert {camera.model.name for camera in reconstruction.cameras.values()} == {"PINHOLE"}

    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment, _quiet_options(ba_backend="caspar"), reconstruction
    )
    assert backend_name(ba_options) == "caspar"
    assert ba_options.refine_sensor_from_rig is False
    assert ba_options.refine_extra_params == ba_options.refine_focal_length
    assert ba_options.caspar.gpu_index == "-1"
    on_gpu: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment,
        _quiet_options(ba_backend="caspar", use_gpu=True),
        reconstruction,
    )
    assert on_gpu.caspar.gpu_index == "0"


def test_caspar_solver_options_reach_the_options_object(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig
) -> None:
    """`MappingOptions.caspar_options` lands on `BundleAdjustmentOptions.caspar`, cast.

    The convergence sweep of `docs/caspar-build.md` moves CASPAR's stopping rules
    from the command line, so the dict has to arrive at the solver intact — and the
    two iteration counters are C++ ints, which pybind11 will not take a float for.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment,
        _quiet_options(
            ba_backend="caspar",
            caspar_options={"solver_iter_max": 1000, "pcg_iter_max": 80.0, "pcg_rel_error_exit": 1e-6},
        ),
        reconstruction,
    )
    assert backend_name(ba_options) == "caspar"
    assert ba_options.caspar.solver_iter_max == 1000
    assert isinstance(ba_options.caspar.solver_iter_max, int)
    assert ba_options.caspar.pcg_iter_max == 80
    assert isinstance(ba_options.caspar.pcg_iter_max, int)
    assert ba_options.caspar.pcg_rel_error_exit == pytest.approx(1e-6)
    # Untouched knobs keep COLMAP's defaults, and `use_gpu` still owns `gpu_index`.
    defaults: dict[str, object] = pycolmap.BundleAdjustmentOptions().caspar.todict()
    assert ba_options.caspar.diag_init == defaults["diag_init"]
    assert ba_options.caspar.gpu_index == "-1"


def test_caspar_solver_options_are_inert_on_ceres_and_reject_unknown_names(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig
) -> None:
    """Ceres never reads the CASPAR block, and a misspelt knob is an error, not a no-op."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    on_ceres: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment,
        _quiet_options(caspar_options={"solver_iter_max": 1000.0}),
        reconstruction,
    )
    assert backend_name(on_ceres) == "ceres"
    assert on_ceres.caspar.solver_iter_max == pycolmap.BundleAdjustmentOptions().caspar.solver_iter_max

    with pytest.raises(KeyError, match="solver_iter_maxx"):
        apply_caspar_options(pycolmap.BundleAdjustmentOptions().caspar, {"solver_iter_maxx": 1.0})
    with pytest.raises(KeyError, match="gpu_index"):
        apply_caspar_options(pycolmap.BundleAdjustmentOptions().caspar, {"gpu_index": 0.0})


def test_the_default_backend_is_ceres_and_leaves_caspar_alone(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig
) -> None:
    """Without the flag nothing about the solve changes: Ceres, as every run before."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment, _quiet_options(), reconstruction
    )
    assert backend_name(ba_options) == "ceres"
    assert ba_options.caspar.gpu_index == "-1"


def test_a_camera_model_this_caspar_build_cannot_project_falls_back_to_ceres(
    isaac_config: CusfmConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    """An OPENCV_FISHEYE rig goes to Ceres on a build whose adapters stop at SIMPLE_RADIAL.

    CASPAR skips the observations of a model it has no adapter for with a log line
    and still reports success, so a silent half-problem is what the fallback exists
    to prevent. The supported set is injected rather than probed, so the test states
    one rule — model not in the set means Ceres — on every environment.
    """
    rig: SyntheticRig = build_synthetic_rig(num_frames=4, num_points=50, fisheye=True, seed=3)
    reconstruction: pycolmap.Reconstruction = build_reconstruction(rig.frames_meta).reconstruction

    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment,
        _quiet_options(ba_backend="caspar", caspar_supported_models=CASPAR_STOCK_CAMERA_MODELS),
        reconstruction,
    )
    assert backend_name(ba_options) == "ceres"
    printed: str = capsys.readouterr().out
    assert "OPENCV_FISHEYE" in printed
    # The warning has to say what this build *can* do, or the reader cannot tell a
    # missing adapter from a missing build.
    assert "SIMPLE_RADIAL" in printed


def test_a_fisheye_rig_stays_on_caspar_when_the_build_has_the_adapter(
    isaac_config: CusfmConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same rig keeps CASPAR once OPENCV_FISHEYE is in the supported set.

    This is the whole point of making the pre-flight capability-aware: the
    `colsfm-caspar-fisheye` build projects OPENCV_FISHEYE, so a fisheye rig must
    not be sent to Ceres there.
    """
    rig: SyntheticRig = build_synthetic_rig(num_frames=4, num_points=50, fisheye=True, seed=3)
    reconstruction: pycolmap.Reconstruction = build_reconstruction(rig.frames_meta).reconstruction

    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment,
        _quiet_options(
            ba_backend="caspar", caspar_supported_models=CASPAR_STOCK_CAMERA_MODELS | {"OPENCV_FISHEYE"}
        ),
        reconstruction,
    )
    assert backend_name(ba_options) == "caspar"
    assert "OPENCV_FISHEYE" not in capsys.readouterr().out


def test_the_probe_reports_the_camera_models_this_caspar_build_projects(
    caspar_enabled: bool, pixi_environment_name: str
) -> None:
    """`caspar_supported_camera_models` reads the build, not a hard-coded list.

    Stock CASPAR (COLMAP 4.2.0) ships adapters for PINHOLE and SIMPLE_RADIAL only;
    `colsfm-caspar-fisheye` adds OPENCV_FISHEYE (`docs/caspar-fisheye-adapter.md`).
    The expectation comes from the environment name, so it is independent of the
    solve the probe runs to find out.
    """
    if not caspar_enabled:
        pytest.skip("this pycolmap is built without CASPAR_ENABLED; run under `pixi run -e colsfm-caspar`")
    supported: frozenset[str] = caspar_supported_camera_models()
    print(f"[colsfm] {pixi_environment_name or 'unknown env'} CASPAR projects {sorted(supported)}")

    assert CASPAR_STOCK_CAMERA_MODELS <= supported
    if pixi_environment_name == "colsfm-caspar-fisheye":
        assert "OPENCV_FISHEYE" in supported
    else:
        assert supported == CASPAR_STOCK_CAMERA_MODELS


def test_the_probe_answers_once_and_quickly(caspar_enabled: bool) -> None:
    """The probe is cached, so the mapper pays for it once per process.

    It runs a real CASPAR solve per candidate model, which is only acceptable
    because it happens once and takes well under a second.
    """
    if not caspar_enabled:
        pytest.skip("this pycolmap is built without CASPAR_ENABLED; run under `pixi run -e colsfm-caspar`")
    caspar_supported_camera_models()
    started: float = time.perf_counter()
    cached: frozenset[str] = caspar_supported_camera_models()
    assert time.perf_counter() - started < 0.01
    assert cached is caspar_supported_camera_models()


def test_the_probe_reports_nothing_without_a_caspar_build(caspar_enabled: bool) -> None:
    """A pycolmap without CASPAR_ENABLED supports no model, so every rig falls back."""
    if caspar_enabled:
        pytest.skip("this pycolmap has CASPAR; the empty answer only exists on the stock build")
    assert caspar_supported_camera_models() == frozenset()


def test_refining_extrinsics_falls_back_to_ceres(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    """`optimize_extrinsics` and CASPAR are exclusive, so the request wins the solver.

    CASPAR hard-throws on a free `sensor_from_rig` in a multi-sensor frame. Forcing
    the flag off instead would turn an asked-for refinement into a silent no-op.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment,
        _quiet_options(ba_backend="caspar", optimize_extrinsics=True),
        reconstruction,
    )
    assert backend_name(ba_options) == "ceres"
    assert ba_options.refine_sensor_from_rig is True
    assert "optimize-extrinsics" in capsys.readouterr().out


def test_the_mapper_runs_on_caspar_or_reports_the_build_that_cannot(
    synthetic_rig: SyntheticRig,
    synthetic_database: Path,
    isaac_config: CusfmConfig,
    caspar_enabled: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--ba-backend caspar` maps the synthetic rig, in either environment.

    On a CASPAR-enabled pycolmap the GPU backend solves and the map matches the
    Ceres one; on the stock one the missing capability is reported once and Ceres
    finishes the run, so the flag never costs a result. The map itself is checked
    against the ground-truth points, not against the solver's own opinion.
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    ceres: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta), synthetic_database, mapping_config, _quiet_options()
    )
    capsys.readouterr()
    caspar: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        mapping_config,
        _quiet_options(ba_backend="caspar"),
    )
    printed: str = capsys.readouterr().out

    if not caspar_enabled:
        assert caspar.ba_backend == "ceres"
        assert "CASPAR_ENABLED" in printed
        assert caspar.num_points3D == ceres.num_points3D
        return

    assert caspar.ba_backend == "caspar"
    assert "CASPAR_ENABLED" not in printed
    recovered, mixed_tracks, relative_errors = _match_tracks_to_truth(caspar.reconstruction, synthetic_rig)
    print(
        f"[colsfm] caspar synthetic: {caspar.num_points3D} points against ceres' {ceres.num_points3D}, "
        f"reprojection {caspar.mean_reprojection_error_px:.4f} px against {ceres.mean_reprojection_error_px:.4f} px"
    )
    assert mixed_tracks == 0
    assert len(recovered) / len(synthetic_rig.points_xyz) >= 0.95
    assert float(np.median(relative_errors)) < 0.01
    assert caspar.num_registered_images == ceres.num_registered_images
    assert caspar.num_points3D == pytest.approx(ceres.num_points3D, rel=0.05)
    assert caspar.mean_reprojection_error_px == pytest.approx(ceres.mean_reprojection_error_px, abs=0.05)



def test_a_fisheye_rig_maps_on_caspar_only_where_the_adapter_exists(
    fisheye_rig: SyntheticRig,
    fisheye_database: Path,
    isaac_config: CusfmConfig,
    caspar_enabled: bool,
    pixi_environment_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End to end on an OPENCV_FISHEYE rig, the run the capability probe exists for.

    `colsfm-caspar-fisheye` carries the adapter, so the rounds run on the GPU and the
    Ceres polish finishes them; `colsfm-caspar` does not, so the pre-flight names the
    model and the whole run is Ceres, which has already converged and needs no polish.
    Either way the map is checked against the rig's own points, not against the solver.
    """
    if not caspar_enabled:
        pytest.skip("this pycolmap is built without CASPAR_ENABLED; run under `pixi run -e colsfm-caspar`")
    result: MappingResult = run_mapping(
        build_reconstruction(fisheye_rig.frames_meta),
        fisheye_database,
        isaac_config.vision_mapping,
        _quiet_options(ba_backend="caspar"),
    )
    printed: str = capsys.readouterr().out
    recovered, mixed_tracks, relative_errors = _match_tracks_to_truth(result.reconstruction, fisheye_rig)
    print(
        f"[colsfm] fisheye rig on {pixi_environment_name or 'unknown env'}: backend {result.ba_backend}, "
        f"{result.num_points3D} points, {result.mean_reprojection_error_px:.4f} px"
    )
    assert mixed_tracks == 0
    assert len(recovered) / len(fisheye_rig.points_xyz) >= 0.95
    assert float(np.median(relative_errors)) < 0.01

    if pixi_environment_name == "colsfm-caspar-fisheye":
        assert result.ba_backend == "caspar"
        assert "OPENCV_FISHEYE" not in printed
        assert result.polish is not None
        assert result.polish.num_observations > 0
        assert result.polish.mean_reprojection_error_after_px <= result.polish.mean_reprojection_error_before_px
    else:
        assert result.ba_backend == "ceres"
        assert "OPENCV_FISHEYE" in printed
        assert result.polish is None



# ---------------------------------------------------------------------------
# Galileo parity: the blob's own keyframes and matches, so only the mapper differs
# ---------------------------------------------------------------------------

BLOB_RUN_DIRS: dict[str, Path] = {
    "galileo-32": Path("data") / "cusfm_runs" / "galileo",
    "galileo-226": Path("data") / "cusfm_runs" / "galileo_blobref" / "cusfm",
}
"""The shipped cuSFM runs parity is measured against, relative to the repo root.

`galileo-32` is the 32-keyframe run the mapper spec documents throughout;
`galileo-226` is the acceptance run of `docs/open-pipeline-plan.md`. Both are
plain `VEHICLE_RIG` invocations with `--optimize_extrinsics` off. The
226-keyframe run under `data/cusfm_runs/galileo/cusfm` is deliberately not used:
its artifacts were last written with `--optimize_extrinsics=True`, which adds
777 `relative_extrinsic` and 8 `absolute_extrinsic` residual blocks that
pycolmap's bundle adjuster cannot express.
"""

DROPPED_PAIRS_PATTERN: str = r"Filter image pairs: by min matches=\d+: (\d+)"
"""The blob's own count of pairs rejected under `min_num_matches_per_pair`."""


def _blob_keypoints(keyframe_dir: Path, frames_meta: FramesMeta) -> dict[int, Float64[ndarray, "num_keypoints 2"]]:
    """Take the extractor's keypoint pixel columns out of the blob's own keyframes.

    Only `keypoint_vector.x` and `.y` are read; the descriptors are what the matcher
    needed and the mapper never sees them.

    Args:
        keyframe_dir: The run's `keyframes` directory.
        frames_meta: The collection naming the images.

    Returns:
        Keypoints per keyframe id, in the file's own order — which is what the
        match files index into.
    """
    return {
        keyframe_id: np.stack(
            [
                np.asarray(message.keypoint_vector.x, dtype=np.float64),
                np.asarray(message.keypoint_vector.y, dtype=np.float64),
            ],
            axis=1,
        )
        for keyframe_id, message in read_blob_keyframes(keyframe_dir, frames_meta).items()
    }


def _blob_matches(matches_dir: Path) -> dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]]:
    """Read every `FrameMatches` result the blob's matcher wrote.

    `FramePairMatch.frames` names the two **keyframe ids** and each
    `FeatureMatch` is a pair of indices into those keyframes' keypoint vectors
    (feature_matcher_main.md §4.2). One `FramePairMatch` is written per input
    pair, empty ones included. COLMAP keys a pair by the lower image id, so the
    pairs are normalised here rather than relying on the database to swap them.

    Args:
        matches_dir: The run's `matches` directory.

    Returns:
        Matches per ordered image pair, empty pairs included.
    """
    frame_matches_class = load_schema().message_class("protos.visual.cusfm.FrameMatches")
    matches: dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]] = {}
    for path in sorted(matches_dir.glob("*_result.pb")):
        message: Message = frame_matches_class()
        message.ParseFromString(path.read_bytes())
        for pair in message.feature_matches:
            source_id: int = int(pair.frames.source_frame_id)
            target_id: int = int(pair.frames.target_frame_id)
            columns: UInt32[ndarray, "num_matches 2"] = np.array(
                [[int(match.source_point_id), int(match.target_point_id)] for match in pair.matches], dtype=np.uint32
            ).reshape(-1, 2)
            if source_id > target_id:
                source_id, target_id = target_id, source_id
                columns = columns[:, ::-1].copy()
            matches[(source_id, target_id)] = columns
    return matches


@dataclass(frozen=True, slots=True)
class BlobRun:
    """One shipped cuSFM run, loaded as a mapper input and a reference output."""

    name: str
    """Key in `BLOB_RUN_DIRS`."""
    input_meta: FramesMeta
    """`pose_graph/frames_meta.json`: the poses the mapper was given."""
    output_meta: FramesMeta
    """`kpmap/keyframes/frames_meta.json`: the poses the mapper produced."""
    database_path: Path
    """The blob's keypoints and matches, rewritten as a COLMAP database."""
    reference: pycolmap.Reconstruction
    """`sparse/`, the blob's exported model, with point errors computed."""
    num_match_pairs: int
    """Image pairs the matcher wrote, empty ones included."""
    num_dropped_pairs: int
    """Pairs the blob's own log says it dropped under `min_num_matches_per_pair`."""
    runtime_seconds: float
    """What `runtime.csv` recorded for `keypoints_mapper_main`."""


@pytest.fixture(scope="module", params=sorted(BLOB_RUN_DIRS))
def galileo_blob(request: pytest.FixtureRequest, repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> BlobRun:
    """A shipped run, with its matches rewritten into a COLMAP database.

    Args:
        request: Carries the run name.
        repo_root: Repository root.
        tmp_path_factory: Where the database goes.

    Returns:
        The run's input, its reference output and a database of its matches.
    """
    name: str = str(request.param)
    run_dir: Path = repo_root / BLOB_RUN_DIRS[name]
    if not (run_dir / "sparse" / "images.txt").is_file():
        pytest.skip(f"No cuSFM reference run at {run_dir}")

    input_meta: FramesMeta = read_frames_meta(run_dir / "pose_graph" / "frames_meta.json")
    matches: dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]] = _blob_matches(run_dir / "matches")
    database_path: Path = tmp_path_factory.mktemp(name) / "blob_matches.db"
    _write_database(database_path, input_meta, _blob_keypoints(run_dir / "keyframes", input_meta), matches)

    reference: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reference.read_text(str(run_dir / "sparse"))
    reference.update_point_3d_errors()
    log_text: str = (run_dir / "kpmap" / "keypoints_mapper_main.txt").read_text()
    dropped_match: re.Match[str] | None = re.search(DROPPED_PAIRS_PATTERN, log_text)
    assert dropped_match is not None, "the blob's log always reports how many pairs it dropped"
    runtime_seconds: float = next(
        record.runtime_seconds
        for record in reversed(read_runtime_records(run_dir / "runtime.csv"))
        if record.command.startswith("keypoints_mapper_main")
    )
    return BlobRun(
        name=name,
        input_meta=input_meta,
        output_meta=read_frames_meta(run_dir / "kpmap" / "keyframes" / "frames_meta.json"),
        database_path=database_path,
        reference=reference,
        num_match_pairs=len(matches),
        num_dropped_pairs=int(dropped_match.group(1)),
        runtime_seconds=runtime_seconds,
    )


@dataclass(frozen=True, slots=True)
class ParityRun:
    """One `run_mapping` call over a blob run's own matches."""

    result: MappingResult
    """What the mapper produced."""
    elapsed_seconds: float
    """Wall-clock seconds the call took, for the runtime comparison."""


@pytest.fixture(scope="module")
def parity(galileo_blob: BlobRun, isaac_config: CusfmConfig) -> ParityRun:
    """Map a blob run's own matches with the isaac configuration.

    Args:
        galileo_blob: The reference run.
        isaac_config: The profile the blob ran with.

    Returns:
        The mapping result and its wall-clock time.
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    started: float = time.perf_counter()
    result: MappingResult = run_mapping(
        build_reconstruction(galileo_blob.input_meta), galileo_blob.database_path, mapping_config, _quiet_options()
    )
    return ParityRun(result=result, elapsed_seconds=time.perf_counter() - started)


def test_the_blob_database_keeps_the_pairs_the_blob_kept(galileo_blob: BlobRun, isaac_config: CusfmConfig) -> None:
    """The loader and `min_num_matches_per_pair` agree with the blob's own log.

    `Filter image pairs: by min matches=15: N` is the blob's count of rejected
    pairs, so the pairs the correspondence graph ends up with must be the pairs
    written minus that N. This is what makes the comparison a mapper comparison:
    both sides start from the same graph.

    An image whose every pair was dropped never enters `DatabaseCache` at all,
    so the graph can hold fewer images than the metadata — one fewer on the
    32-keyframe run. Those images stay registered and simply observe nothing,
    which is also why the blob exports 28 images out of 32.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(galileo_blob.input_meta).reconstruction
    correspondences = load_correspondences(
        reconstruction, galileo_blob.database_path, isaac_config.vision_mapping.min_num_matches_per_pair
    )
    expected_pairs: int = galileo_blob.num_match_pairs - galileo_blob.num_dropped_pairs
    print(
        f"[colsfm] {galileo_blob.name}: {galileo_blob.num_match_pairs} pairs written, "
        f"{galileo_blob.num_dropped_pairs} dropped by the blob, {correspondences.num_image_pairs} in the graph"
    )
    assert correspondences.num_image_pairs == expected_pairs
    assert galileo_blob.reference.num_images() <= correspondences.num_images <= len(galileo_blob.input_meta.keyframes)


def test_galileo_parity_against_the_blob(galileo_blob: BlobRun, parity: ParityRun) -> None:
    """The whole mapper, on the blob's own matches, against the blob's own output.

    Everything upstream is held fixed — the blob's keyframes, its matches and its
    input poses — so any difference is the mapper's. The reference numbers come
    from `sparse/`, not from the log.

    The point count is **not** compared directly. COLMAP's LO-RANSAC
    triangulation does a local optimisation and a final refit that cuSFM's plain
    MSAC does not (§9.3), and cuSFM grows one disjoint track per connected
    component of the match graph while COLMAP grows one per reference
    observation. The result is that colsfm keeps weak three-view tracks cuSFM
    rejects: a superset, at a lower reprojection error. What is asserted instead
    is that none of the blob's structure is lost.
    """
    result: MappingResult = parity.result
    blob_reprojection_px: float = galileo_blob.reference.compute_mean_reprojection_error()
    print(
        f"[colsfm] {galileo_blob.name} parity | images {result.num_images_with_observations} vs "
        f"{galileo_blob.reference.num_images()} | points {result.num_points3D} vs "
        f"{galileo_blob.reference.num_points3D()} | observations {result.num_observations} vs "
        f"{galileo_blob.reference.compute_num_observations()} | reprojection "
        f"{result.mean_reprojection_error_px:.4f} vs {blob_reprojection_px:.4f} px | track "
        f"{result.mean_track_length:.3f} vs {galileo_blob.reference.compute_mean_track_length():.3f} | "
        f"{parity.elapsed_seconds:.2f} s vs {galileo_blob.runtime_seconds:.2f} s "
        f"({parity.elapsed_seconds / galileo_blob.runtime_seconds:.2f}x)"
    )

    assert result.num_registered_images == len(galileo_blob.input_meta.keyframes)
    assert abs(result.num_images_with_observations - galileo_blob.reference.num_images()) <= 2
    assert abs(result.mean_reprojection_error_px - blob_reprojection_px) < 0.3
    # `docs/open-pipeline-plan.md` asks for a per-stage runtime within 2x of the
    # blob's; the benchmark harness is where that bound is enforced. This guard
    # only catches a gross regression (an order of magnitude), because the test
    # measures wall clock on a machine that other runs share: standalone the stage
    # takes ~7 s, under a concurrent RoboCap run it measured 28 s against a 6.7 s
    # blob reference, and a 3x bound failed for reasons unrelated to the code.
    assert parity.elapsed_seconds < 10.0 * galileo_blob.runtime_seconds
    assert result.num_points3D >= 0.9 * galileo_blob.reference.num_points3D()

    blob_points: Float64[ndarray, "num_blob_points 3"] = np.array(
        [galileo_blob.reference.point3D(point_id).xyz for point_id in galileo_blob.reference.point3D_ids()]
    )
    our_points: Float64[ndarray, "num_points 3"] = np.array(
        [result.reconstruction.point3D(point_id).xyz for point_id in result.reconstruction.point3D_ids()]
    )
    nearest_m: Float64[ndarray, " num_blob_points"] = cKDTree(our_points).query(blob_points)[0]
    coverage: float = float(np.mean(nearest_m < 0.05))
    print(f"[colsfm] {galileo_blob.name} parity | {coverage:.1%} of the blob's points have one of ours within 5 cm")
    assert coverage >= 0.95


def test_galileo_rig_poses_match_the_blob(galileo_blob: BlobRun, parity: ParityRun) -> None:
    """Every rig pose lands within a fraction of the correction the blob applied.

    The gauge frame is the one holding the lowest keyframe id, which cuSFM also
    holds constant (§6.3), so the two trajectories are comparable directly with
    no alignment step to hide an error. The yardstick is how far the blob moved
    the poses it was given — about 12 mm and half a degree on the last rig frame.
    """
    blob_correction_m: float = 0.0
    blob_correction_deg: float = 0.0
    worst_translation_m: float = 0.0
    worst_rotation_deg: float = 0.0
    input_by_sample: dict[int, pycolmap.Rigid3d] = {
        rig_frame.synced_sample_id: rig_frame.world_T_vehicle for rig_frame in galileo_blob.input_meta.rig_frames()
    }
    for rig_frame in galileo_blob.output_meta.rig_frames():
        correction_m, correction_deg = pose_delta(
            input_by_sample[rig_frame.synced_sample_id], rig_frame.world_T_vehicle
        )
        blob_correction_m = max(blob_correction_m, correction_m)
        blob_correction_deg = max(blob_correction_deg, correction_deg)
        translation_m, rotation_deg = pose_delta(
            parity.result.reconstruction.frame(rig_frame.synced_sample_id).rig_from_world.inverse(),
            rig_frame.world_T_vehicle,
        )
        worst_translation_m = max(worst_translation_m, translation_m)
        worst_rotation_deg = max(worst_rotation_deg, rotation_deg)

    print(
        f"[colsfm] {galileo_blob.name} parity | pose delta {worst_translation_m * 1e3:.3f} mm / "
        f"{worst_rotation_deg:.4f} deg, against a blob correction of {blob_correction_m * 1e3:.3f} mm / "
        f"{blob_correction_deg:.4f} deg"
    )
    assert worst_translation_m < 5e-3
    assert worst_rotation_deg < 0.15
    assert worst_translation_m < blob_correction_m
    assert worst_rotation_deg < blob_correction_deg

    # Both sides hold the same rig frame constant, so it cannot have moved.
    gauge_sample_id: int = gauge_rig_frame(galileo_blob.input_meta).synced_sample_id
    gauge_translation_m, gauge_rotation_deg = pose_delta(
        parity.result.reconstruction.frame(gauge_sample_id).rig_from_world.inverse(),
        next(
            rig_frame
            for rig_frame in galileo_blob.output_meta.rig_frames()
            if rig_frame.synced_sample_id == gauge_sample_id
        ).world_T_vehicle,
    )
    assert gauge_translation_m < 1e-9
    assert gauge_rotation_deg < 1e-9


def test_the_cauchy_scale_is_the_sigma_whitening_equivalent(galileo_blob: BlobRun, isaac_config: CusfmConfig) -> None:
    """A loss scale equal to sigma is what actually reproduces the blob's poses.

    cuSFM whitens the residual by `reprojection_error_standard_deviation = 4.0`
    and then applies `CauchyLoss(1.0)`; COLMAP does not whiten, so the port
    carries the whitening in the loss scale (§9.2). That is an argument, not a
    measurement — this is the measurement. On the 32-keyframe run the worst pose
    difference against the blob is 5.51 mm at scale 1.0, 2.06 mm at 2.0,
    **1.26 mm at 4.0** and 2.63 mm at 8.0: the minimum sits exactly at sigma.
    """
    if galileo_blob.name != "galileo-32":
        pytest.skip("one run is enough to locate the minimum")
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    deltas_mm: dict[float, float] = {}
    for sigma in (1.0, 4.0, 8.0):
        scaled_config: VisionMappingConfig = dataclasses.replace(
            mapping_config,
            bundle_adjustment=dataclasses.replace(
                mapping_config.bundle_adjustment, reprojection_error_standard_deviation=sigma
            ),
        )
        result: MappingResult = run_mapping(
            build_reconstruction(galileo_blob.input_meta),
            galileo_blob.database_path,
            scaled_config,
            _quiet_options(),
        )
        deltas_mm[sigma] = 1e3 * max(
            pose_delta(
                result.reconstruction.frame(rig_frame.synced_sample_id).rig_from_world.inverse(),
                rig_frame.world_T_vehicle,
            )[0]
            for rig_frame in galileo_blob.output_meta.rig_frames()
        )
    print(f"[colsfm] pose delta by Cauchy scale: {', '.join(f'{k}: {v:.3f} mm' for k, v in deltas_mm.items())}")
    assert deltas_mm[4.0] < deltas_mm[1.0]
    assert deltas_mm[4.0] < deltas_mm[8.0]


# ─────────────────────────────────────────────────────────────────────────────
# The pre-solve degeneracy guard
# ─────────────────────────────────────────────────────────────────────────────


DEPTH_THRESHOLD_M: float = 300.0
"""`vision_mapping_config.depth_threshold`, the isaac profile's world-Z cap in metres."""


def _triangulated(rig: SyntheticRig, database_path: Path, config: CusfmConfig) -> tuple[pycolmap.Reconstruction, Correspondences]:
    """Triangulate a rig and hand back the model with its live observation manager.

    `run_mapping` loads its own correspondences, so a caller that wants the
    manager afterwards has to build the model itself; a manager without the
    correspondence graph silently declines to delete anything.

    Args:
        rig: The synthetic rig.
        database_path: Its COLMAP database.
        config: The isaac profile.

    Returns:
        The triangulated reconstruction and the correspondences that own its manager.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(rig.frames_meta).reconstruction
    correspondences: Correspondences = load_correspondences(
        reconstruction, database_path, config.vision_mapping.min_num_matches_per_pair
    )
    triangulator: pycolmap.IncrementalTriangulator = pycolmap.IncrementalTriangulator(
        correspondences.graph, reconstruction, correspondences.observation_manager
    )
    options: pycolmap.IncrementalTriangulatorOptions = pycolmap.IncrementalTriangulatorOptions()
    for image_id in sorted(reconstruction.images):
        triangulator.triangulate_image(options, image_id)
    reconstruction.update_point_3d_errors()
    return reconstruction, correspondences


def _corrupt_point(reconstruction: pycolmap.Reconstruction, point3D_id: int, xyz: Float64[ndarray, "3"]) -> None:
    """Overwrite one point's world position in place.

    Args:
        reconstruction: The model to edit.
        point3D_id: The point to move.
        xyz: Its new world position.
    """
    reconstruction.point3D(point3D_id).xyz = xyz


def test_degeneracy_guard_drops_points_ceres_cannot_survive(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """Non-finite, astronomically distant and beyond-cap points all go; sane ones stay.

    Ceres reported ~1e152 px mean reprojection on the first RoboCap rounds, which
    is what a point at 1e30 m or a NaN does to a normal equation. This is the
    guard that keeps such a point out of the solver.
    """
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    observation_manager: pycolmap.ObservationManager = correspondences.observation_manager

    point3D_ids: list[int] = sorted(reconstruction.points3D)
    assert len(point3D_ids) >= 4
    doomed: list[int] = point3D_ids[:3]
    survivor: int = point3D_ids[3]
    survivor_xyz: Float64[ndarray, "3"] = np.asarray(reconstruction.point3D(survivor).xyz, dtype=np.float64).copy()
    _corrupt_point(reconstruction, doomed[0], np.array([np.nan, 0.0, 0.0]))
    _corrupt_point(reconstruction, doomed[1], np.array([1e30, 0.0, 0.0]))
    _corrupt_point(reconstruction, doomed[2], np.array([0.0, 0.0, 2.0 * DEPTH_THRESHOLD_M]))

    dropped: int = filter_degenerate_points(observation_manager, reconstruction, DEPTH_THRESHOLD_M)

    assert dropped > 0
    for point3D_id in doomed:
        assert point3D_id not in reconstruction.points3D
    assert survivor in reconstruction.points3D
    assert np.array_equal(np.asarray(reconstruction.point3D(survivor).xyz, dtype=np.float64), survivor_xyz)


def test_degeneracy_guard_is_a_no_op_on_a_healthy_model(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """A model whose points all sit inside the cap loses nothing to the guard."""
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    observation_manager: pycolmap.ObservationManager = correspondences.observation_manager
    num_points_before: int = reconstruction.num_points3D()

    assert filter_degenerate_points(observation_manager, reconstruction, DEPTH_THRESHOLD_M) == 0
    assert reconstruction.num_points3D() == num_points_before


def test_projection_failure_guard_drops_a_point_pushed_into_a_camera(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """A point moved a micron in front of a camera loses its observations.

    This is RoboCap's failure mode exactly: 4 points of 209 905 ended up 5-90 mm
    from a camera centre, where the projection diverges and the reprojection error
    reaches 4.5e153 px, while the world position stays perfectly ordinary and
    every coordinate bound is satisfied.
    """
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    observation_manager: pycolmap.ObservationManager = correspondences.observation_manager
    bound_px: float = projection_sanity_bound_px(reconstruction)
    assert bound_px > 0.0

    point3D_ids: list[int] = sorted(reconstruction.points3D)
    doomed: int = point3D_ids[0]
    survivor: int = point3D_ids[1]
    survivor_track_length: int = reconstruction.point3D(survivor).track.length()
    observer: pycolmap.Image = reconstruction.image(reconstruction.point3D(doomed).track.elements[0].image_id)
    _corrupt_point(reconstruction, doomed, observer.cam_from_world().inverse() * np.array([0.5, 0.5, 1e-6]))
    reconstruction.update_point_3d_errors()
    assert reconstruction.point3D(doomed).error > bound_px

    removed: int = filter_projection_failures(observation_manager, reconstruction)

    assert removed > 0
    assert doomed not in reconstruction.points3D
    assert reconstruction.point3D(survivor).track.length() == survivor_track_length
    reconstruction.update_point_3d_errors()
    assert all(point.error <= bound_px for point in reconstruction.points3D.values())


def test_projection_failure_guard_is_a_no_op_on_a_healthy_model(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """Sub-pixel observations are nowhere near an image width, so nothing is dropped."""
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    observation_manager: pycolmap.ObservationManager = correspondences.observation_manager
    num_observations_before: int = reconstruction.compute_num_observations()

    assert filter_projection_failures(observation_manager, reconstruction) == 0
    assert reconstruction.compute_num_observations() == num_observations_before


# ── extrinsic refinement (`--optimize_extrinsics`) ───────────────────────────────────

WEAKLY_CONSTRAINED_MOVE_M: float = 0.03
"""Above this an unregularised refinement is drifting, not correcting; see
`test_the_unregularised_refinement_drifts_on_weakly_covisible_cameras`."""

EXTRINSIC_PERTURBATION_M: float = 0.02
"""20 mm of translation error the refinement has to remove."""

EXTRINSIC_RECOVERY_M: float = 2e-3
"""How close to the unperturbed refinement the perturbed one must land: 2 mm.

Not 1 mm, which is what this was and which made the test flaky. Neither side of
this comparison is a fixed point: both runs are the same unregularised
block-coordinate descent over the same matches, but they start 20 mm apart and
stop on their own tolerances, so the gap between them is a property of the
descent, not of the optimum. Measured 0.660 mm on the committed Galileo database
and 1.453 mm on a regenerated one — a factor of 2.2 between two artifacts of the
same run, so the bound has to sit above both."""

EXTRINSIC_RECOVERY_DEG: float = 0.05
"""How close to the unperturbed refinement the perturbed one must land: 0.05 deg.

Measured 0.0019 deg, i.e. 26x inside the bound; rotation is far better
conditioned here than translation."""

EXTRINSIC_RESIDUAL_FRACTION: float = 0.5
"""Fraction of the perturbation that may survive against the *calibration*.

This bound is a fraction rather than a millimetre figure on purpose. The
refinement is unregularised — colsfm has no equivalent of cuSFM's absolute and
relative extrinsic priors (NOTES.md deviation 9) — so its optimum does not sit on
the factory calibration and there is no reason for it to: on the committed
Galileo database the *unperturbed* refinement already lands 7.07 mm and 0.30 deg
away from it, while cutting the mean reprojection error from 1.027 px to 0.860 px.
So "came back to the calibration" is not the claim this test can make; "removed
most of a 20 mm perturbation" is, and 7.07 mm of 20 mm is inside this bound with
room to spare."""


def _stereo_partner(frames_meta: FramesMeta, camera_params_id: int) -> int:
    """The other camera of the declared stereo pair holding this one.

    The refinement tests perturb the reference camera's stereo partner, because
    that is the one relative extrinsic Galileo's imagery genuinely determines:
    the pair shares a whole field of view, whereas two cameras facing different
    ways are linked only through the trajectory.

    Args:
        frames_meta: The collection, for its `stereo_pair` list.
        camera_params_id: One camera of a declared pair.

    Returns:
        The partner's `camera_params_id`.

    Raises:
        ValueError: When the camera is in no declared stereo pair.
    """
    for pair in frames_meta.stereo_pairs:
        if pair.left_camera_params_id == camera_params_id:
            return pair.right_camera_params_id
        if pair.right_camera_params_id == camera_params_id:
            return pair.left_camera_params_id
    raise ValueError(f"camera {camera_params_id} is in no declared stereo pair")


@pytest.fixture(scope="module")
def galileo_pose_graph_meta(repo_root: Path) -> FramesMeta:
    """The rig-exact 226-keyframe Galileo poses a colsfm run's pose-graph stage wrote."""
    path: Path = repo_root / "data" / "cusfm_runs" / "galileo_colsfm" / "cusfm" / "pose_graph" / "frames_meta.json"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    return read_frames_meta(path)


@pytest.fixture(scope="module")
def galileo_colsfm_database(repo_root: Path) -> Path:
    """The COLMAP database that same run's matching stage wrote."""
    path: Path = repo_root / "data" / "cusfm_runs" / "galileo_colsfm" / "cusfm" / "database.db"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    return path


def _perturbed_extrinsic(pose: pycolmap.Rigid3d, seed: int = 5) -> pycolmap.Rigid3d:
    """Knock one extrinsic 20 mm and 0.5 deg off.

    Args:
        pose: The extrinsic to disturb.
        seed: PRNG seed for the offset directions.

    Returns:
        The perturbed transform.
    """
    return pose * _random_offset(np.random.default_rng(seed), EXTRINSIC_PERTURBATION_M, PERTURBATION_DEG)


@dataclass(frozen=True, slots=True)
class ExtrinsicRuns:
    """Three Galileo mappings that differ only in the start point and the flag."""

    fixed: MappingResult
    """Perturbed metadata, `optimize_extrinsics` off: the control that keeps the perturbation."""
    refined: MappingResult
    """Perturbed metadata, `optimize_extrinsics` on."""
    refined_from_clean: MappingResult
    """Unperturbed metadata, `optimize_extrinsics` on: where the refinement's own optimum is."""
    perturbed_meta: FramesMeta
    """The metadata the first two runs started from."""
    reference_camera_params_id: int
    """The rig origin, which is also the fixed camera."""
    perturbed_camera_params_id: int
    """The camera the first two runs started 20 mm and 0.5 deg off."""


@pytest.fixture(scope="module")
def extrinsic_runs(
    galileo_pose_graph_meta: FramesMeta, galileo_colsfm_database: Path, isaac_config: CusfmConfig
) -> ExtrinsicRuns:
    """Map Galileo three times over the same matches; see `ExtrinsicRuns`.

    Args:
        galileo_pose_graph_meta: The rig-exact input poses.
        galileo_colsfm_database: The matches every run reads.
        isaac_config: The mapping configuration.

    Returns:
        The three results and what they started from.
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    reference_camera_params_id: int = gauge_camera_params_id(galileo_pose_graph_meta)
    perturbed_camera_params_id: int = _stereo_partner(galileo_pose_graph_meta, reference_camera_params_id)
    perturbed_meta: FramesMeta = galileo_pose_graph_meta.with_extrinsics(
        {perturbed_camera_params_id: _perturbed_extrinsic(
            galileo_pose_graph_meta.cameras[perturbed_camera_params_id].vehicle_T_cam
        )}
    )

    def _map(frames_meta: FramesMeta, optimize_extrinsics: bool) -> MappingResult:
        return run_mapping(
            build_reconstruction(frames_meta, reference_camera_params_id=reference_camera_params_id),
            galileo_colsfm_database,
            mapping_config,
            _quiet_options(optimize_extrinsics=optimize_extrinsics, num_threads=1),
        )

    return ExtrinsicRuns(
        fixed=_map(perturbed_meta, False),
        refined=_map(perturbed_meta, True),
        refined_from_clean=_map(galileo_pose_graph_meta, True),
        perturbed_meta=perturbed_meta,
        reference_camera_params_id=reference_camera_params_id,
        perturbed_camera_params_id=perturbed_camera_params_id,
    )


def test_refinement_removes_a_20_mm_extrinsic_perturbation(
    extrinsic_runs: ExtrinsicRuns, galileo_pose_graph_meta: FramesMeta
) -> None:
    """Camera 1, knocked 20 mm and 0.5 deg off, lands where the clean refinement lands.

    The substantive claim is that the perturbation is *removed*, not merely
    reduced, and the evidence for that is the unperturbed refinement over the
    same matches: two descents starting 20 mm apart stop within
    `EXTRINSIC_RECOVERY_M` of each other. The calibration cannot serve as that
    reference, because this refinement is **unregularised** — colsfm has no
    equivalent of cuSFM's absolute and relative extrinsic priors (§7) — so its
    optimum is several millimetres off the factory numbers with nothing perturbed
    at all (7.07 mm on the committed Galileo database, at a *lower* reprojection
    error). Against the calibration the test therefore asserts only the weak,
    artifact-stable claim: at most `EXTRINSIC_RESIDUAL_FRACTION` of the
    perturbation survives.

    This test was flaky at a 1 mm refinement-against-refinement bound: it
    measured 0.660 mm on the committed database and 1.453 mm on a regenerated
    one. Neither number is an error — the descent stops on a tolerance, not on a
    fixed point — so the bound is 2 mm rather than a seed or a tightened
    tolerance, which would only hide the same variation.

    The fixed-extrinsics run over the same matches is the control: it keeps all
    20 mm, which is the regression this whole change exists for.
    """
    truth: pycolmap.Rigid3d = galileo_pose_graph_meta.cameras[extrinsic_runs.perturbed_camera_params_id].vehicle_T_cam
    started: pycolmap.Rigid3d = extrinsic_runs.perturbed_meta.cameras[extrinsic_runs.perturbed_camera_params_id].vehicle_T_cam
    assert pose_delta(started, truth)[0] == pytest.approx(EXTRINSIC_PERTURBATION_M, abs=1e-9)

    assert extrinsic_runs.fixed.refined_extrinsics is None, "nothing is refined when the flag is off"
    kept: dict[int, pycolmap.Rigid3d] = extrinsic_runs.fixed.reference.vehicle_T_cam_by_camera_params_id(
        extrinsic_runs.fixed.reconstruction
    )
    assert pose_delta(kept[extrinsic_runs.perturbed_camera_params_id], truth)[0] == pytest.approx(EXTRINSIC_PERTURBATION_M, abs=1e-9)

    assert extrinsic_runs.refined.refined_extrinsics is not None
    assert extrinsic_runs.refined_from_clean.refined_extrinsics is not None
    recovered: pycolmap.Rigid3d = extrinsic_runs.refined.refined_extrinsics[extrinsic_runs.perturbed_camera_params_id]
    clean: pycolmap.Rigid3d = extrinsic_runs.refined_from_clean.refined_extrinsics[extrinsic_runs.perturbed_camera_params_id]
    translation_m, rotation_deg = pose_delta(recovered, clean)
    residual_m, residual_deg = pose_delta(recovered, truth)
    print(
        f"[colsfm] extrinsic refinement: camera {extrinsic_runs.perturbed_camera_params_id} started "
        f"{EXTRINSIC_PERTURBATION_M * 1e3:.1f} mm / {PERTURBATION_DEG} deg off, landed "
        f"{translation_m * 1e3:.3f} mm / {rotation_deg:.4f} deg from the clean refinement and "
        f"{residual_m * 1e3:.3f} mm / {residual_deg:.4f} deg from the calibration; reprojection "
        f"{extrinsic_runs.fixed.mean_reprojection_error_px:.4f} -> "
        f"{extrinsic_runs.refined.mean_reprojection_error_px:.4f} px"
    )
    assert translation_m < EXTRINSIC_RECOVERY_M
    assert rotation_deg < EXTRINSIC_RECOVERY_DEG
    assert residual_m < EXTRINSIC_RESIDUAL_FRACTION * EXTRINSIC_PERTURBATION_M
    assert extrinsic_runs.refined.mean_reprojection_error_px <= extrinsic_runs.fixed.mean_reprojection_error_px


def test_the_fixed_camera_extrinsic_is_untouched(
    extrinsic_runs: ExtrinsicRuns, galileo_pose_graph_meta: FramesMeta
) -> None:
    """The fixed camera is the rig origin, so its `vehicle_T_cam` is the input one exactly.

    Bundle adjustment holds no parameter block for the reference sensor at all —
    COLMAP fixes it to identity — which is what makes it the exact bridge back to
    the vehicle frame.
    """
    assert extrinsic_runs.refined.refined_extrinsics is not None
    camera_params_id: int = extrinsic_runs.reference_camera_params_id
    translation_m, rotation_deg = pose_delta(
        extrinsic_runs.refined.refined_extrinsics[camera_params_id],
        galileo_pose_graph_meta.cameras[camera_params_id].vehicle_T_cam,
    )
    assert translation_m == pytest.approx(0.0, abs=1e-12)
    assert rotation_deg == pytest.approx(0.0, abs=1e-9)


def test_the_unregularised_refinement_drifts_on_weakly_covisible_cameras(
    extrinsic_runs: ExtrinsicRuns, galileo_pose_graph_meta: FramesMeta
) -> None:
    """`refined_extrinsics` covers the whole rig, and records how far each camera walked.

    This pins the measured cost of having no extrinsic priors. cuSFM regularises
    its refinement with absolute and relative extrinsic residuals (§7), which
    pycolmap's bundle adjuster cannot express, so colsfm's optimum is the
    reprojection optimum alone. On Galileo's 0.66 m sweep that is well determined
    only inside a stereo pair — the reference camera's partner moves millimetres,
    the four cameras facing other directions tens of millimetres, because nothing
    but the trajectory links them to the reference.
    """
    refined: dict[int, pycolmap.Rigid3d] | None = extrinsic_runs.refined_from_clean.refined_extrinsics
    assert refined is not None
    assert sorted(refined) == sorted(extrinsic_runs.refined_from_clean.reconstruction.cameras)
    moved_m: dict[int, float] = {
        camera_params_id: pose_delta(pose, galileo_pose_graph_meta.cameras[camera_params_id].vehicle_T_cam)[0]
        for camera_params_id, pose in refined.items()
    }
    print(
        "[colsfm] unregularised extrinsic moves (mm): "
        + ", ".join(f"{key}={value * 1e3:.2f}" for key, value in sorted(moved_m.items()))
    )
    reference_camera_params_id: int = extrinsic_runs.reference_camera_params_id
    assert moved_m[reference_camera_params_id] == pytest.approx(0.0, abs=1e-12)
    partner: int = _stereo_partner(galileo_pose_graph_meta, reference_camera_params_id)
    assert 0.0 < moved_m[partner] < WEAKLY_CONSTRAINED_MOVE_M, "the reference's own stereo pair stays put"
    elsewhere: list[float] = [
        value for key, value in moved_m.items() if key not in {reference_camera_params_id, partner}
    ]
    assert max(elsewhere) > WEAKLY_CONSTRAINED_MOVE_M, "the other cameras drift; there is no prior holding them"


def test_refining_extrinsics_on_a_vehicle_referenced_rig_is_refused(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """A vehicle-referenced rig cannot refine extrinsics, and says so instead of no-opping.

    COLMAP freezes every `sensor_from_rig` of a rig whose reference sensor is not
    itself parameterised, and the vehicle body owns no images, so the request
    would otherwise be silently ignored.
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    with pytest.raises(ValueError, match="reference sensor"):
        run_mapping(
            build_reconstruction(synthetic_rig.frames_meta),
            synthetic_database,
            mapping_config,
            _quiet_options(optimize_extrinsics=True),
        )


def test_the_round_callback_sees_every_round_as_it_finishes(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """`MappingOptions.round_callback` fires once per round, with that round's model.

    The hook a live viewer needs. It runs after the round's filter and solve and
    before the early-exit check, so the number of calls equals the number of
    `RoundStats` the mapper returns, and the reconstruction it is handed already
    holds `RoundStats.num_points3D` points.
    """
    seen: list[tuple[int, int, int]] = []

    def record(stats: RoundStats, reconstruction: pycolmap.Reconstruction) -> None:
        """Record the round index, its reported point count and the live one."""
        seen.append((stats.round_index, stats.num_points3D, reconstruction.num_points3D()))

    model: PosedModel = build_reconstruction(synthetic_rig.frames_meta)
    result: MappingResult = run_mapping(
        model, synthetic_database, isaac_config.vision_mapping, _quiet_options(round_callback=record)
    )
    assert len(seen) == len(result.rounds)
    assert [entry[0] for entry in seen] == [stats.round_index for stats in result.rounds]
    assert all(reported == live for _index, reported, live in seen)
    assert seen[-1][1] == result.reconstruction.num_points3D()


# ---------------------------------------------------------------------------
# The Ceres polish that finishes a CASPAR mapping run
# ---------------------------------------------------------------------------


def test_the_polish_option_set_is_the_ceres_one_built_from_the_same_config(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig
) -> None:
    """`ceres_polish_options` differs from the CASPAR round options in the backend only.

    `docs/caspar-build.md` § Ceres polish experiment measured the recovery with
    colsfm's own solve — the config's loss and scale, its iteration cap, SPARSE_SCHUR
    and the same frozen extrinsics — so the polish has to be that solve and not a
    fresh set of defaults.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    ba_config: BundleAdjustmentConfig = isaac_config.vision_mapping.bundle_adjustment
    options: MappingOptions = _quiet_options(ba_backend="caspar")
    rounds: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(ba_config, options, reconstruction)
    polish: pycolmap.BundleAdjustmentOptions = ceres_polish_options(ba_config, options, reconstruction)

    assert backend_name(rounds) == "caspar"
    assert backend_name(polish) == "ceres"
    assert polish.ceres.loss_function_type == rounds.ceres.loss_function_type
    assert polish.ceres.loss_function_scale == pytest.approx(
        ba_config.reprojection_error_standard_deviation * ba_config.loss_function_scale
    )
    assert polish.ceres.solver_options.max_num_iterations == ba_config.max_num_iterations
    assert polish.ceres.solver_options.linear_solver_type == rounds.ceres.solver_options.linear_solver_type
    assert polish.refine_sensor_from_rig is False
    assert polish.refine_focal_length is False
    assert polish.refine_extra_params is False


def test_the_polish_is_on_by_default_and_switches_off() -> None:
    """The ablation switch is a field, and the default is the shipped behaviour."""
    assert MappingOptions().caspar_ceres_polish is True
    assert _quiet_options(caspar_ceres_polish=False).caspar_ceres_polish is False


def test_the_ceres_backend_never_polishes(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """A Ceres mapping run has already converged, so nothing is appended to it.

    This also covers every CASPAR fallback: they all end with `ba_backend == "ceres"`,
    which is the condition the polish is keyed on.
    """
    result: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        isaac_config.vision_mapping,
        _quiet_options(),
    )
    assert result.ba_backend == "ceres"
    assert result.polish is None


def test_the_caspar_run_finishes_with_one_ceres_bundle_adjustment(
    synthetic_rig: SyntheticRig,
    synthetic_database: Path,
    isaac_config: CusfmConfig,
    caspar_enabled: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--ba-backend caspar` maps on the GPU and ends on one Ceres solve.

    The polish moves the model without touching the observation set, so the point
    and observation counts are the ones the rounds left behind and only the
    reprojection error may change.
    """
    if not caspar_enabled:
        pytest.skip("this pycolmap is built without CASPAR_ENABLED; run under `pixi run -e colsfm-caspar`")
    result: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        isaac_config.vision_mapping,
        _quiet_options(ba_backend="caspar", verbose=True),
    )
    printed: str = capsys.readouterr().out

    assert result.ba_backend == "caspar"
    assert result.polish is not None
    assert result.polish.seconds > 0.0
    assert result.polish.mean_reprojection_error_after_px == pytest.approx(result.mean_reprojection_error_px)
    assert result.polish.num_observations == result.num_observations
    assert result.polish.ba_termination in {"CONVERGENCE", "NO_CONVERGENCE"}
    assert result.bundle_adjustment_seconds >= result.polish.seconds
    assert "polish" in printed
    print(
        f"[colsfm] caspar polish: {result.polish.ba_num_iterations} iterations, "
        f"{result.polish.mean_reprojection_error_before_px:.4f} px -> "
        f"{result.polish.mean_reprojection_error_after_px:.4f} px in {result.polish.seconds:.2f} s"
    )


def test_the_polish_can_be_switched_off_for_an_ablation(
    synthetic_rig: SyntheticRig,
    synthetic_database: Path,
    isaac_config: CusfmConfig,
    caspar_enabled: bool,
) -> None:
    """`caspar_ceres_polish=False` leaves a CASPAR run exactly where CASPAR stopped."""
    if not caspar_enabled:
        pytest.skip("this pycolmap is built without CASPAR_ENABLED; run under `pixi run -e colsfm-caspar`")
    result: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        isaac_config.vision_mapping,
        _quiet_options(ba_backend="caspar", caspar_ceres_polish=False),
    )
    assert result.ba_backend == "caspar"
    assert result.polish is None

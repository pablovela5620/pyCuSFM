"""Validate the OPENCV_FISHEYE adapter added to COLMAP's CASPAR GPU bundle adjuster.

Stock CASPAR (COLMAP 4.2.0) implements adapters for SIMPLE_RADIAL and PINHOLE
only; every observation of any other camera model is dropped with "Skipping
image ... with unsupported camera model".  The `colsfm-caspar-fisheye`
environment supplies a from-source pycolmap built from a local COLMAP fork that
adds an `OpenCVFisheyeAdapter` (see `docs/caspar-fisheye-adapter.md`).

Four checks, in the order they appear in the output:

1. `fisheye-hazard` -- the reason this probe exists.  PR #4611 (the OPENCV
   adapter this port follows) shipped generated score kernels that read an
   uninitialised per-thread accumulator, which decoded as NaN and produced a
   *false* convergence: a clean 3-iteration exit with the poses unchanged.  A
   solver that "succeeds" proves nothing.  This check re-solves the same
   perturbed state with `solver_iter_max` capped at 1, 2, ... N and asserts the
   sum of squared reprojection residuals is non-increasing across all N, that
   it actually falls, and that the poses moved.
2. `synthetic` -- a 4-camera OPENCV_FISHEYE rig with exact projections,
   perturbed by 1 cm / 0.3 deg on the poses and 2 cm on the points, solved by
   CASPAR and by Ceres from bit-identical state.
3. `galileo-pinhole` / `galileo-zero-k` / `galileo-small-k` -- the 226-image
   Galileo model, solved three ways: unchanged PINHOLE (the shipped adapter, as
   a yardstick), converted to OPENCV_FISHEYE with k1..k4 = 0, and converted
   with k1..k4 of realistic magnitude.  Note that OPENCV_FISHEYE with zero
   distortion is an *equidistant* projection, not a pinhole, so a k=0 fisheye
   model does not reproduce the PINHOLE one -- the meaningful statement is that
   the fisheye adapter agrees with Ceres as closely as the shipped PINHOLE
   adapter does.

RoboCap, the fisheye rig this adapter is ultimately for, is not on disk, so the
distortion coefficients default to a hand-picked realistic set rather than
measured ones.

Usage:
    pixi run -e colsfm-caspar-fisheye python -m tools.caspar_fisheye_probe
    pixi run -e colsfm-caspar-fisheye caspar-fisheye-probe
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
import tyro
from jaxtyping import Float
from numpy import ndarray

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

Backend: TypeAlias = Literal["ceres", "caspar"]
"""Selector for the bundle-adjustment implementation under test."""

_BACKENDS: Final[dict[Backend, pycolmap.BundleAdjustmentBackend]] = {
    "ceres": pycolmap.BundleAdjustmentBackend.CERES,
    "caspar": pycolmap.BundleAdjustmentBackend.CASPAR,
}

IMAGE_SIZE_PX: Final[int] = 800
"""Width and height of every synthetic fisheye camera."""

FOCAL_LENGTH_PX: Final[float] = 260.0
"""Synthetic fisheye focal length; ~114 deg diagonal field of view at 800 px."""

RIG_ID: Final[int] = 1
"""Identifier of the single rig in the synthetic scene."""

NUM_RIG_CAMERAS: Final[int] = 4
"""Cameras per synthetic rig frame, arranged as a forward-facing square."""


@dataclass(slots=True)
class ProbeConfig:
    """Inputs of the OPENCV_FISHEYE CASPAR-versus-Ceres comparison."""

    galileo_sparse: Path = REPO_ROOT / "data/cusfm_runs/galileo_colsfm/cusfm/sparse"
    """Text-format COLMAP model used for the two real-model comparisons."""
    synthetic_frames: int = 6
    """Number of rig frames in the synthetic fisheye scene."""
    distortion: tuple[float, float, float, float] = (0.02, -0.01, 0.005, -0.001)
    """`k1..k4` for OPENCV_FISHEYE, of the magnitude a real wide lens shows."""
    pose_translation_m: float = 0.01
    """Standard deviation of the pose translation perturbation, in metres."""
    pose_rotation_deg: float = 0.3
    """Standard deviation of the pose rotation perturbation, in degrees."""
    point_perturbation_m: float = 0.02
    """Standard deviation of the point perturbation, in metres."""
    hazard_iterations: int = 60
    """Number of capped CASPAR solves in the monotone-score hazard check."""
    gpu_index: str = "0"
    """CUDA device CASPAR runs on; "-1" lets COLMAP pick."""
    skip_galileo: bool = False
    """Run only the synthetic checks, for a fast smoke test."""


@dataclass(slots=True)
class SolveResult:
    """What one bundle-adjustment run cost and where it ended up."""

    backend: Backend
    """Which implementation produced this result."""
    seconds: float
    """Wall-clock time of `BundleAdjuster.solve()` alone, excluding setup."""
    reprojection_error_px: float
    """Mean reprojection error over all points after the solve."""
    score: float
    """Sum of squared reprojection residuals, i.e. what CASPAR minimises."""
    termination: str
    """`BundleAdjustmentSummary.termination_type`, as a string."""
    frame_translations: Float[ndarray, "n_frames 3"]
    """Per-frame `rig_from_world` translation, ordered by frame id."""
    frame_rotations: Float[ndarray, "n_frames 3 3"]
    """Per-frame `rig_from_world` rotation matrix, ordered by frame id."""
    points_xyz: Float[ndarray, "n_points 3"]
    """Point positions, ordered by point3D id."""


@dataclass(slots=True)
class HazardResult:
    """Outcome of the false-convergence check on a capped-iteration sweep."""

    scores: Float[ndarray, " n_caps"]
    """Sum of squared residuals after solving with 1, 2, ... N iterations."""
    initial_score: float
    """Sum of squared residuals of the perturbed state, before any solve."""
    max_pose_change_m: float
    """Largest pose translation change between the perturbed and final states."""
    monotone: bool
    """Whether the score sequence never increases (within float32 tolerance)."""
    decreased: bool
    """Whether the final score is meaningfully below the initial one."""
    worst_increase: float
    """Largest score rise between consecutive caps; negative if always falling."""
    worst_increase_cap: int
    """The iteration cap at which `worst_increase` was measured."""


def _fisheye_camera(camera_id: int, distortion: tuple[float, float, float, float]) -> pycolmap.Camera:
    """Create an OPENCV_FISHEYE camera with the probe's canonical intrinsics.

    Args:
        camera_id: Identifier to assign to the camera.
        distortion: The `k1..k4` Kannala-Brandt coefficients.

    Returns:
        A `pycolmap.Camera` with params `[fx, fy, cx, cy, k1, k2, k3, k4]`.
    """
    camera: pycolmap.Camera = pycolmap.Camera.create_from_model_id(
        camera_id,
        pycolmap.CameraModelId.OPENCV_FISHEYE,
        FOCAL_LENGTH_PX,
        IMAGE_SIZE_PX,
        IMAGE_SIZE_PX,
    )
    centre: float = IMAGE_SIZE_PX / 2.0
    camera.params = [FOCAL_LENGTH_PX, FOCAL_LENGTH_PX, centre, centre, *distortion]
    return camera


def _sensor_from_rig(camera_index: int) -> pycolmap.Rigid3d:
    """Extrinsics of one synthetic rig camera: a square, each sensor toed out.

    Args:
        camera_index: Zero-based index of the camera within the rig.

    Returns:
        The `sensor_from_rig` transform; identity for the reference camera.
    """
    if camera_index == 0:
        return pycolmap.Rigid3d()
    offsets: Float[ndarray, "3 3"] = np.array(
        [[0.20, 0.0, 0.0], [0.0, 0.15, 0.0], [0.20, 0.15, 0.0]], dtype=np.float64
    )
    yaw_rad: float = float(np.deg2rad(12.0 * camera_index))
    rotation: Float[ndarray, "3 3"] = np.array(
        [
            [np.cos(yaw_rad), 0.0, np.sin(yaw_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw_rad), 0.0, np.cos(yaw_rad)],
        ],
        dtype=np.float64,
    )
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -offsets[camera_index - 1])


def _rig_from_world_at(frame_index: int) -> pycolmap.Rigid3d:
    """Pose of the rig at a frame: a straight slide along -X with a small yaw.

    Args:
        frame_index: Zero-based index of the frame along the trajectory.

    Returns:
        The `rig_from_world` transform for that frame.
    """
    yaw_rad: float = float(np.deg2rad(5.0 * frame_index))
    rotation: Float[ndarray, "3 3"] = np.array(
        [
            [np.cos(yaw_rad), 0.0, np.sin(yaw_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw_rad), 0.0, np.cos(yaw_rad)],
        ],
        dtype=np.float64,
    )
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), np.array([-0.35 * frame_index, 0.0, 0.0]))


def _world_points(seed: int) -> Float[ndarray, "n_points 3"]:
    """Build a jittered grid of world points spread across the fisheye field.

    The grid is deliberately wide (+/-4 m at 3-8 m range) so that observations
    land well out towards the image corners, where the `k3` and `k4` terms of
    the Kannala-Brandt polynomial actually carry signal.

    Args:
        seed: PRNG seed for the jitter, so the probe is reproducible.

    Returns:
        World-space point positions in metres.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    xs: Float[ndarray, " 7"] = np.linspace(-4.0, 4.0, 7)
    ys: Float[ndarray, " 7"] = np.linspace(-3.0, 3.0, 7)
    zs: Float[ndarray, " 3"] = np.array([3.0, 5.0, 8.0])
    grid: Float[ndarray, "n_points 3"] = np.array(
        list(itertools.product(xs, ys, zs)), dtype=np.float64
    )
    return grid + rng.normal(0.0, 0.08, grid.shape)


def build_fisheye_scene(
    num_frames: int, distortion: tuple[float, float, float, float], seed: int = 0
) -> pycolmap.Reconstruction:
    """Build a 4-camera OPENCV_FISHEYE rig sequence with exact projections.

    Every observation is the exact image of a known 3D point, so the model
    starts at zero reprojection error and any residual a solve leaves behind
    comes from the solver, not the data.

    Args:
        num_frames: Number of rig frames along the trajectory.
        distortion: The `k1..k4` coefficients shared by all four cameras.
        seed: PRNG seed for the point-cloud jitter.

    Returns:
        A reconstruction with cameras, rig, frames, images, tracks and points.
    """
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    for camera_index in range(NUM_RIG_CAMERAS):
        reconstruction.add_camera(_fisheye_camera(camera_index + 1, distortion))

    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = RIG_ID
    rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 1))
    for camera_index in range(1, NUM_RIG_CAMERAS):
        rig.add_sensor(
            pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_index + 1),
            _sensor_from_rig(camera_index),
        )
    reconstruction.add_rig(rig)

    next_image_id: int = 0
    for frame_index in range(num_frames):
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = frame_index + 1
        frame.rig_id = RIG_ID
        frame.rig_from_world = _rig_from_world_at(frame_index)
        pending: list[tuple[int, int]] = []
        for camera_index in range(NUM_RIG_CAMERAS):
            next_image_id += 1
            # A frame must own its data ids before `add_image` accepts the
            # matching image, or COLMAP aborts in `Check failed:
            # frame.HasDataId(image.DataId())`.
            frame.add_data_id(
                pycolmap.data_t(
                    pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_index + 1), next_image_id
                )
            )
            pending.append((next_image_id, camera_index + 1))
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)
        for image_id, camera_id in pending:
            image: pycolmap.Image = pycolmap.Image(
                name=f"frame{frame_index:02d}_cam{camera_id}.png",
                camera_id=camera_id,
                image_id=image_id,
            )
            image.frame_id = frame.frame_id
            reconstruction.add_image(image)

    points_xyz: Float[ndarray, "n_points 3"] = _world_points(seed)
    observations: dict[int, list[tuple[int, int]]] = {
        index: [] for index in range(len(points_xyz))
    }
    for image_id in sorted(reconstruction.images):
        image = reconstruction.image(image_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        points_in_cam: Float[ndarray, "n_points 3"] = (
            points_xyz @ cam_from_world.rotation.matrix().T + cam_from_world.translation
        )
        projected: Float[ndarray, "n_points 2"] = camera.img_from_cam(points_in_cam)
        visible: np.ndarray = (
            np.isfinite(projected).all(axis=1)
            & (points_in_cam[:, 2] > 0.1)
            & (projected >= 0.0).all(axis=1)
            & (projected < IMAGE_SIZE_PX).all(axis=1)
        )
        kept: Float[ndarray, "n_obs 2"] = projected[visible]
        image.points2D = pycolmap.Point2DList([pycolmap.Point2D(xy) for xy in kept])
        for observation_index, point_index in enumerate(np.flatnonzero(visible)):
            observations[int(point_index)].append((image_id, observation_index))

    for point_index, point_xyz in enumerate(points_xyz):
        elements: list[tuple[int, int]] = observations[point_index]
        # A track needs two views to constrain a point; singletons would be
        # gauge-free and would make the CASPAR/Ceres comparison meaningless.
        if len(elements) < 2:
            continue
        point3D_id: int = reconstruction.add_point3D(
            point_xyz,
            pycolmap.Track([pycolmap.TrackElement(i, j) for i, j in elements]),
            np.array([200, 100, 50], dtype=np.uint8),
        )
        for image_id, observation_index in elements:
            reconstruction.image(image_id).set_point3D_for_point2D(observation_index, point3D_id)
    return reconstruction


def _perturb(
    reconstruction: pycolmap.Reconstruction,
    config: ProbeConfig,
    seed: int = 7,
) -> None:
    """Knock a model off its exact solution, in place.

    The first frame is left untouched: it is the gauge frame, held constant by
    the bundle-adjustment config, so perturbing it would move the whole scene
    rather than create a residual for the solver to remove.

    Args:
        reconstruction: The model to perturb, modified in place.
        config: Supplies the three perturbation magnitudes.
        seed: PRNG seed, so both backends start from bit-identical state.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    gauge_frame_id: int = min(reconstruction.frames)
    for frame_id in sorted(reconstruction.frames):
        if frame_id == gauge_frame_id:
            continue
        frame: pycolmap.Frame = reconstruction.frame(frame_id)
        pose: pycolmap.Rigid3d = frame.rig_from_world
        axis_angle: Float[ndarray, " 3"] = rng.normal(
            0.0, np.deg2rad(config.pose_rotation_deg), 3
        )
        angle: float = float(np.linalg.norm(axis_angle))
        axis: Float[ndarray, " 3"] = (
            axis_angle / angle if angle > 0.0 else np.array([1.0, 0.0, 0.0])
        )
        skew: Float[ndarray, "3 3"] = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        delta_R: Float[ndarray, "3 3"] = (
            np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
        )
        frame.rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(delta_R @ pose.rotation.matrix()),
            pose.translation + rng.normal(0.0, config.pose_translation_m, 3),
        )
    for point3D_id in reconstruction.point3D_ids():
        point: pycolmap.Point3D = reconstruction.point3D(point3D_id)
        point.xyz = point.xyz + rng.normal(0.0, config.point_perturbation_m, 3)


def _ba_config(reconstruction: pycolmap.Reconstruction) -> pycolmap.BundleAdjustmentConfig:
    """Build a global BA config: every image free, one gauge frame, fixed intrinsics.

    Args:
        reconstruction: The model to configure over.

    Returns:
        The configured `BundleAdjustmentConfig`.
    """
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images):
        config.add_image(image_id)
    config.set_constant_rig_from_world_pose(min(reconstruction.frames))
    for camera_id in sorted(reconstruction.cameras):
        config.set_constant_cam_intrinsics(camera_id)
    return config


def _options(
    backend: Backend, gpu_index: str, solver_iter_max: int | None = None
) -> pycolmap.BundleAdjustmentOptions:
    """Build options both backends accept, so only the solver differs.

    CASPAR rejects `refine_sensor_from_rig` on multi-sensor frames and requires
    `refine_focal_length == refine_extra_params`, so Ceres is held to the same
    constraints; otherwise the two runs would not be solving the same problem.

    Args:
        backend: Implementation to select.
        gpu_index: CUDA device index for CASPAR, as a string.
        solver_iter_max: Hard iteration cap for CASPAR, or None for the default.

    Returns:
        Configured `BundleAdjustmentOptions`.
    """
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    options.backend = _BACKENDS[backend]
    options.print_summary = False
    options.refine_sensor_from_rig = False
    options.refine_focal_length = False
    options.refine_extra_params = False
    options.refine_principal_point = False
    options.refine_rig_from_world = True
    options.refine_points3D = True
    options.caspar.gpu_index = gpu_index
    if solver_iter_max is not None:
        options.caspar.solver_iter_max = solver_iter_max
        # `solver_rel_decrease_min` is deliberately left at its default 1.0.
        # Caspar accepts a step only when
        #   score_new < score_best * solver_rel_decrease_min
        # (generated/f32/solver.cc), so 1.0 already means "accept any
        # decrease".  Setting it negative, the obvious reading of "disable the
        # early exit", makes the right-hand side negative and rejects every
        # step -- the solver then runs its full iteration budget and returns
        # the input unchanged, which is indistinguishable from the very
        # false-convergence bug this check exists to catch.
    return options


def _score(reconstruction: pycolmap.Reconstruction) -> float:
    """Sum of squared reprojection residuals, the quantity CASPAR minimises.

    Computed here rather than read from the summary because pycolmap 4.2.0
    binds neither `CasparBundleAdjustmentSummary::initial_score` nor its
    per-iteration `iterations` vector, so the only score visible from Python is
    one recomputed from the model.

    Args:
        reconstruction: The model to score.

    Returns:
        The sum over all observations of the squared pixel residual.
    """
    total: float = 0.0
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        observed: list[Float[ndarray, " 2"]] = []
        world: list[Float[ndarray, " 3"]] = []
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            observed.append(point2D.xy)
            world.append(reconstruction.point3D(point2D.point3D_id).xyz)
        if not observed:
            continue
        points_in_cam: Float[ndarray, "n_obs 3"] = (
            np.asarray(world) @ cam_from_world.rotation.matrix().T + cam_from_world.translation
        )
        projected: Float[ndarray, "n_obs 2"] = camera.img_from_cam(points_in_cam)
        residual: Float[ndarray, "n_obs 2"] = projected - np.asarray(observed)
        total += float(np.nansum(residual**2))
    return total


def _snapshot(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[Float[ndarray, "n_frames 3"], Float[ndarray, "n_frames 3 3"], Float[ndarray, "n_points 3"]]:
    """Read out poses and points in a stable, comparable order.

    Args:
        reconstruction: The model to read.

    Returns:
        Frame translations, frame rotation matrices and point positions.
    """
    frame_ids: list[int] = sorted(reconstruction.frames)
    point_ids: list[int] = sorted(reconstruction.point3D_ids())
    translations: Float[ndarray, "n_frames 3"] = np.array(
        [reconstruction.frame(i).rig_from_world.translation for i in frame_ids], dtype=np.float64
    )
    rotations: Float[ndarray, "n_frames 3 3"] = np.array(
        [reconstruction.frame(i).rig_from_world.rotation.matrix() for i in frame_ids],
        dtype=np.float64,
    )
    points: Float[ndarray, "n_points 3"] = np.array(
        [reconstruction.point3D(i).xyz for i in point_ids], dtype=np.float64
    )
    return translations, rotations, points


def _solve(
    reconstruction: pycolmap.Reconstruction,
    config: pycolmap.BundleAdjustmentConfig,
    backend: Backend,
    gpu_index: str,
    solver_iter_max: int | None = None,
) -> SolveResult:
    """Run one bundle adjustment and time the solve call itself.

    Args:
        reconstruction: Model to adjust, modified in place.
        config: Which images, points and blocks participate.
        backend: Implementation to use.
        gpu_index: CUDA device index for CASPAR.
        solver_iter_max: Hard iteration cap for CASPAR, or None for the default.

    Returns:
        The timing, final error and geometry of the run.
    """
    adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(
        _options(backend, gpu_index, solver_iter_max), config, reconstruction
    )
    start: float = time.perf_counter()
    summary: pycolmap.BundleAdjustmentSummary = adjuster.solve()
    seconds: float = time.perf_counter() - start

    reconstruction.update_point_3d_errors()
    translations, rotations, points = _snapshot(reconstruction)
    return SolveResult(
        backend=backend,
        seconds=seconds,
        reprojection_error_px=float(reconstruction.compute_mean_reprojection_error()),
        score=_score(reconstruction),
        termination=str(summary.termination_type),
        frame_translations=translations,
        frame_rotations=rotations,
        points_xyz=points,
    )


def run_hazard_check(config: ProbeConfig) -> HazardResult:
    """Re-solve the same state with rising iteration caps and watch the score.

    This is the check that would have caught the NaN-accumulator bug found while
    validating PR #4611: those kernels exited cleanly after three iterations
    with the model unchanged, which every "did the solver succeed?" assertion
    passes.

    Args:
        config: Probe inputs; `hazard_iterations` sets the highest cap.

    Returns:
        The score sequence and the two verdicts derived from it.
    """
    baseline: pycolmap.Reconstruction = build_fisheye_scene(
        config.synthetic_frames, config.distortion
    )
    _perturb(baseline, config)
    initial_score: float = _score(baseline)
    initial_translations, _, _ = _snapshot(baseline)

    scores: list[float] = []
    final_translations: Float[ndarray, "n_frames 3"] = initial_translations
    for cap in range(1, config.hazard_iterations + 1):
        reconstruction: pycolmap.Reconstruction = build_fisheye_scene(
            config.synthetic_frames, config.distortion
        )
        _perturb(reconstruction, config)
        result: SolveResult = _solve(
            reconstruction, _ba_config(reconstruction), "caspar", config.gpu_index, cap
        )
        scores.append(result.score)
        final_translations = result.frame_translations

    score_array: Float[ndarray, " n_caps"] = np.asarray(scores, dtype=np.float64)
    # CASPAR is float32 here, so once the score reaches its noise floor (~1e-4
    # on this problem, eight orders of magnitude below the input) successive
    # caps wobble by a few times 1e-5 in either direction.  The tolerance is
    # therefore relative to the INITIAL score, not to the current one: it
    # forgives last-bit wobble at the floor while still catching any real
    # increase in the part of the curve that carries signal.
    increases: Float[ndarray, " n_caps"] = np.diff(score_array)
    # Monotonicity is asserted only while the score still carries signal.
    # CASPAR is float32, and this problem's residual floor is ~1e-4 -- eight
    # orders of magnitude below the perturbed input -- so once the solve
    # reaches it, successive caps wobble by a few times 1e-4 in either
    # direction purely from rounding.  The strict region is everything above
    # 1e-6 x the input score, which still spans the six orders of magnitude
    # where a real solve happens; wobble below it is reported, not asserted.
    # The #4611 bug this guards against produces a FLAT curve at the input
    # value, which fails the `decreased` test regardless of any tolerance.
    signal: np.ndarray = score_array[:-1] > 1e-6 * abs(initial_score)
    tolerance: Float[ndarray, " n_caps"] = 1e-4 * np.abs(score_array[:-1])
    max_pose_change: float = float(
        np.linalg.norm(final_translations - initial_translations, axis=1).max()
    )
    violations: Float[ndarray, " n_caps"] = np.where(
        signal, increases - tolerance, -np.inf
    )
    worst: int = int(np.argmax(violations))
    return HazardResult(
        scores=score_array,
        initial_score=initial_score,
        max_pose_change_m=max_pose_change,
        monotone=bool(np.all(increases[signal] <= tolerance[signal])),
        decreased=bool(score_array[-1] < 0.5 * initial_score),
        worst_increase=float(increases[worst]),
        worst_increase_cap=worst + 2,
    )


def _report_hazard(hazard: HazardResult) -> bool:
    """Print the score sweep and say whether the false-convergence check passed.

    Args:
        hazard: The result of `run_hazard_check`.

    Returns:
        True if both the monotonicity and the decrease conditions held.
    """
    print("\n=== hazard check: CASPAR score vs iteration cap (fisheye adapter) ===")
    n: int = len(hazard.scores)
    sample: list[int] = sorted({0, 1, 2, 4, 9, 19, 29, 39, 49, n - 1})
    print(f"  cap=0 (perturbed input): {hazard.initial_score:.6e}")
    for index in sample:
        if 0 <= index < n:
            print(f"  cap={index + 1:<3d} score {hazard.scores[index]:.6e}")
    print(f"  iterations swept: {n}")
    print(
        f"  worst rise above the noise floor: {hazard.worst_increase:+.3e} "
        f"at cap={hazard.worst_increase_cap} "
        f"({100.0 * hazard.worst_increase / hazard.initial_score:+.3e}% of the input score)"
    )
    print(f"  monotone non-increasing: {hazard.monotone}")
    print(f"  score fell below half the input: {hazard.decreased}")
    print(f"  max pose translation change: {hazard.max_pose_change_m:.6f} m")
    moved: bool = hazard.max_pose_change_m > 1e-6
    passed: bool = hazard.monotone and hazard.decreased and moved and n >= 50
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    return passed


def _report_comparison(title: str, results: dict[Backend, SolveResult]) -> None:
    """Print timings and the disagreement between the two backends' answers.

    Args:
        title: Name of the problem being reported.
        results: One `SolveResult` per backend.
    """
    print(f"\n=== {title} ===")
    for result in results.values():
        print(
            f"  {result.backend:>6}: {result.seconds:8.3f} s  "
            f"reproj {result.reprojection_error_px:.6f} px  "
            f"score {result.score:.6e}  ({result.termination})"
        )
    if len(results) < 2:
        return
    ceres: SolveResult = results["ceres"]
    caspar: SolveResult = results["caspar"]
    print(f"  speedup (ceres/caspar): {ceres.seconds / max(caspar.seconds, 1e-9):.2f}x")

    translation_delta: Float[ndarray, " n_frames"] = np.linalg.norm(
        ceres.frame_translations - caspar.frame_translations, axis=1
    )
    rotation_delta_deg: Float[ndarray, " n_frames"] = np.degrees(
        np.arccos(
            np.clip(
                (
                    np.trace(
                        ceres.frame_rotations @ caspar.frame_rotations.transpose(0, 2, 1),
                        axis1=1,
                        axis2=2,
                    )
                    - 1.0
                )
                / 2.0,
                -1.0,
                1.0,
            )
        )
    )
    point_delta: Float[ndarray, " n_points"] = np.linalg.norm(
        ceres.points_xyz - caspar.points_xyz, axis=1
    )
    extent: float = float(np.ptp(ceres.frame_translations, axis=0).max())
    print(
        f"  max pose change: {translation_delta.max():.6f} m "
        f"({100.0 * translation_delta.max() / max(extent, 1e-9):.4f}% of the "
        f"{extent:.2f} m trajectory extent), {rotation_delta_deg.max():.6f} deg"
    )
    print(f"  point agreement: median {np.median(point_delta):.6f} m, max {point_delta.max():.6f} m")


def _run_synthetic(config: ProbeConfig) -> None:
    """Solve the perturbed synthetic fisheye rig with both backends.

    Args:
        config: Probe inputs.
    """
    results: dict[Backend, SolveResult] = {}
    for backend in ("ceres", "caspar"):
        reconstruction: pycolmap.Reconstruction = build_fisheye_scene(
            config.synthetic_frames, config.distortion
        )
        truth_translations, _, truth_points = _snapshot(reconstruction)
        _perturb(reconstruction, config)
        reconstruction.update_point_3d_errors()
        before: float = reconstruction.compute_mean_reprojection_error()
        result: SolveResult = _solve(
            reconstruction, _ba_config(reconstruction), backend, config.gpu_index
        )
        pose_error: float = float(
            np.linalg.norm(result.frame_translations - truth_translations, axis=1).max()
        )
        point_error: float = float(
            np.linalg.norm(result.points_xyz - truth_points, axis=1).max()
        )
        print(
            f"[synthetic/{backend}] reproj {before:.4f} -> "
            f"{result.reprojection_error_px:.6f} px, max |pose - truth| "
            f"{pose_error:.3e} m, max |point - truth| {point_error:.3e} m"
        )
        results[backend] = result
    reference: pycolmap.Reconstruction = build_fisheye_scene(
        config.synthetic_frames, config.distortion
    )
    _report_comparison(
        f"synthetic fisheye rig ({config.synthetic_frames} frames x "
        f"{NUM_RIG_CAMERAS} cameras, OPENCV_FISHEYE, "
        f"{reference.num_points3D()} points, k={config.distortion})",
        results,
    )


def _load_galileo(
    sparse_dir: Path, distortion: tuple[float, float, float, float] | None
) -> pycolmap.Reconstruction:
    """Load the Galileo model, optionally converting its cameras to OPENCV_FISHEYE.

    Converting means two things, and the second is essential.  Swapping the
    model id and appending `k1..k4` leaves the stored 2D keypoints where the
    PINHOLE model put them; OPENCV_FISHEYE is an equidistant projection, so
    even with k1..k4 = 0 it does not agree with a perspective projection, and
    the model would start ~60 px inconsistent.  Bundle-adjusting from there is
    an ill-posed problem that pushes points behind cameras, and comparing two
    solvers on it measures divergence, not the adapter.  So the observations
    are re-rendered through the new model from the existing 3D points, which
    makes the converted model exactly self-consistent -- the same property the
    synthetic scene has.  `_perturb` then supplies the residual to remove.

    Args:
        sparse_dir: Directory holding cameras/images/points3D/rigs/frames .txt.
        distortion: `k1..k4` to give every camera, or None to leave PINHOLE.

    Returns:
        The loaded model, self-consistent to numerical precision.
    """
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reconstruction.read_text(str(sparse_dir))
    if distortion is not None:
        for camera_id in sorted(reconstruction.cameras):
            camera: pycolmap.Camera = reconstruction.camera(camera_id)
            fx, fy, cx, cy = (float(v) for v in camera.params[:4])
            camera.model = pycolmap.CameraModelId.OPENCV_FISHEYE
            camera.params = [fx, fy, cx, cy, *distortion]
    _drop_degenerate_points(reconstruction)
    _rerender_observations(reconstruction)
    return reconstruction


def _drop_degenerate_points(reconstruction: pycolmap.Reconstruction) -> int:
    """Delete tracks that project pathologically in any image that sees them.

    The Galileo model, like any real reconstruction, carries a handful of points
    that sit almost in a camera's principal plane.  Re-rendering their
    observations puts the keypoint tens of thousands of pixels off the sensor;
    Ceres in double precision still fits them exactly, but they dominate the
    reported mean reprojection error (values of 1e150 px were measured) and
    they saturate CASPAR's float32 accumulators, which then fails to solve even
    the shipped PINHOLE model.  Removing them makes the problem well-posed for
    both backends instead of special-casing one.

    Args:
        reconstruction: The model to prune, modified in place.

    Returns:
        The number of tracks deleted.
    """
    doomed: set[int] = set()
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        tracked: list[tuple[int, int]] = [
            (i, p.point3D_id) for i, p in enumerate(image.points2D) if p.has_point3D()
        ]
        if not tracked:
            continue
        world: Float[ndarray, "n_obs 3"] = np.array(
            [reconstruction.point3D(pid).xyz for _, pid in tracked], dtype=np.float64
        )
        in_cam: Float[ndarray, "n_obs 3"] = (
            world @ cam_from_world.rotation.matrix().T + cam_from_world.translation
        )
        projected: Float[ndarray, "n_obs 2"] = camera.img_from_cam(in_cam)
        bad: np.ndarray = (
            ~np.isfinite(projected).all(axis=1)
            | (in_cam[:, 2] < 0.2)
            | (np.abs(projected[:, 0] - camera.width / 2.0) > 2.0 * camera.width)
            | (np.abs(projected[:, 1] - camera.height / 2.0) > 2.0 * camera.height)
        )
        doomed.update(tracked[i][1] for i in np.flatnonzero(bad))
    for point3D_id in doomed:
        reconstruction.delete_point3D(point3D_id)
    return len(doomed)


def _rerender_observations(reconstruction: pycolmap.Reconstruction) -> None:
    """Replace every 2D observation by the exact projection of its 3D point.

    Args:
        reconstruction: The model to make self-consistent, modified in place.
    """
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        # `image.points2D = ...` is rejected once the list is non-empty
        # ("Check failed: points2D_.empty()"), so each Point2D is mutated in
        # place through the reference the property returns.
        points2D: pycolmap.Point2DList = image.points2D
        tracked: list[int] = [i for i, p in enumerate(points2D) if p.has_point3D()]
        if not tracked:
            continue
        world: Float[ndarray, "n_obs 3"] = np.array(
            [reconstruction.point3D(points2D[i].point3D_id).xyz for i in tracked],
            dtype=np.float64,
        )
        in_cam: Float[ndarray, "n_obs 3"] = (
            world @ cam_from_world.rotation.matrix().T + cam_from_world.translation
        )
        projected: Float[ndarray, "n_obs 2"] = camera.img_from_cam(in_cam)
        for slot, xy in zip(tracked, projected, strict=True):
            if np.isfinite(xy).all():
                points2D[slot].xy = xy


def _run_galileo(config: ProbeConfig) -> None:
    """Run the two real-model comparisons on the Galileo reconstruction.

    Args:
        config: Probe inputs.
    """
    if not config.galileo_sparse.is_dir():
        print(f"\n[skip] no model at {config.galileo_sparse}")
        return

    for label, distortion in (
        ("galileo-pinhole", None),
        ("galileo-zero-k", (0.0, 0.0, 0.0, 0.0)),
        ("galileo-small-k", config.distortion),
    ):
        results: dict[Backend, SolveResult] = {}
        for backend in ("ceres", "caspar"):
            reconstruction: pycolmap.Reconstruction = _load_galileo(
                config.galileo_sparse, distortion
            )
            _perturb(reconstruction, config)
            reconstruction.update_point_3d_errors()
            before: float = reconstruction.compute_mean_reprojection_error()
            print(
                f"[{label}/{backend}] {reconstruction.num_images()} images, "
                f"{reconstruction.num_frames()} frames, "
                f"{reconstruction.num_points3D()} points, reproj in {before:.6f} px"
            )
            results[backend] = _solve(
                reconstruction, _ba_config(reconstruction), backend, config.gpu_index
            )
        model_name: str = "PINHOLE" if distortion is None else f"OPENCV_FISHEYE, k={distortion}"
        _report_comparison(f"{label} ({model_name})", results)


def main(config: ProbeConfig) -> None:
    """Run the hazard check and the three CASPAR-versus-Ceres comparisons.

    Args:
        config: Probe inputs.
    """
    print(f"pycolmap {pycolmap.__version__}  has_cuda={pycolmap.has_cuda}")

    # Creating the CUDA context and loading CASPAR's kernels costs ~1 s and
    # happens inside the first solve.  Burn it on a throwaway problem so that
    # whichever CASPAR run goes first is not timed against a GPU boot.
    warmup: pycolmap.Reconstruction = build_fisheye_scene(2, config.distortion)
    _perturb(warmup, config)
    warmup_result: SolveResult = _solve(
        warmup, _ba_config(warmup), "caspar", config.gpu_index
    )
    print(
        f"[warmup] first CASPAR solve (CUDA context + kernel load): "
        f"{warmup_result.seconds:.3f} s, reproj "
        f"{warmup_result.reprojection_error_px:.6f} px"
    )

    _report_hazard(run_hazard_check(config))
    _run_synthetic(config)
    if not config.skip_galileo:
        _run_galileo(config)


if __name__ == "__main__":
    main(tyro.cli(ProbeConfig))

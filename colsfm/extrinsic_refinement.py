"""Regularised rig-extrinsic refinement — the prior terms pycolmap's bundle adjuster has not.

cuSFM's second `keypoints_mapper_main` pass (`--optimize_extrinsics=true`) does not
optimise reprojection alone. Its Ceres problem carries two extra cost groups, both
visible in the shipped Galileo log
(`data/cusfm_runs/galileo/cusfm/kpmap/keypoints_mapper_main.txt`):

```
bundle_adjustment_solver.cc:1058] Added 8 absolute pose constraints of extrinsics to problem
bundle_adjustment_solver.cc:1138] Added 777 camera rig extrinsic constraints
[BA COST] initial group 'relative_extrinsic' cost = 0 (777 blocks)
[BA COST] initial group 'absolute_extrinsic' cost = 0 (8 blocks)
```

which is the paper's §4.4: reprojection (Eq. 5) plus an absolute extrinsic prior
(Eq. 14) plus the inter-camera relative extrinsic constraint (Eq. 6). pycolmap's
`BundleAdjuster` can express neither and its Ceres problem cannot be extended from
standalone pyceres (`docs/spec/pycolmap-capabilities.md` §8, the pybind11 registry
split), so this module rebuilds the extrinsic half of that problem in pyceres and
alternates it with pycolmap's own bundle adjustment.

## What the two block counts mean, measured

**8 absolute blocks** is one per camera — all eight, including the one whose extrinsic
is held constant and which therefore vanishes from Ceres' `Reduced` column. Only the free
cameras get one here; see deviation 4 below.

**777 relative blocks** is *not* `4 declared stereo pairs x N frames`. Galileo's 226
keyframes fall into 29 rig frames holding 8, 8, ..., 6 and 4 cameras, and

```
sum over rig frames of C(cameras in that frame, 2) = 27*28 + 15 + 6 = 777
```

exactly. So the constraint is added for **every unordered pair of cameras that fired in
the same rig frame**, not only for the declared stereo pairs, and it is added once per
rig frame. Because a rig's extrinsics are shared across frames, all the blocks of one
pair carry an identical residual, so the 777 blocks reduce to 28 distinct pairs with
multiplicities — which is how they are built here (`RepeatedCauchyLoss`).

## Which sigmas, measured

`vision_mapping_config.pb.txt` offers two candidate pairs. Reproducing the blob's own
final group costs from its exported extrinsics settles it. With
`extrinsic_error_meters = 0.01`, `extrinsic_error_degrees = 2` and `CauchyLoss(1.0)`:

| group | recomputed here | blob's last logged round |
|---|---|---|
| `absolute_extrinsic` | 0.7230 | 0.722949 |
| `relative_extrinsic` | 77.80 | 87.50 |

The absolute figure matches to four significant figures. `relative_pose_*_error_*`
(0.1 m / 5 deg) gives 0.09 and 10.5, eight times too small in both groups. The residual
is `[log(R_err); t_err]` with rotation in slots 0-2, whitened by
`chol(diag(I/sigma_rot_rad^2, I/sigma_trans_m^2))`, exactly the blob's
`DefaultPoseInformationMatrix` (`docs/spec/keypoints_mapper_main.md` §6.4).

## Whitening convention

`colsfm.mapping` hands pycolmap an unwhitened pixel residual with
`loss_function_scale = 4.0`, which is the same minimum as the blob's whitened residual
under `CauchyLoss(1.0)` because `CauchyLoss(a)` at `s` equals `a^2` times `CauchyLoss(1)`
at `s/a^2`. That equivalence only survives when reprojection is the *whole* objective.
Here it is mixed with priors, so the common scale has to be the blob's: every residual
in this module is whitened (reprojection by `1/4 px`, priors by their sigmas) and every
loss is `CauchyLoss(1.0)`.

## Why one residual block can hold a whole camera

A Python `pyceres` cost function costs roughly a millisecond per call, and Galileo's
refinement carries ~30 000 observations. One block per observation would be minutes per
Ceres iteration. `RigReprojectionCost` therefore holds **all** of one camera's
observations in a single block of `2 * n_obs` residuals, and applies the robust loss
*inside* `Evaluate`.

That substitution is exact, not an approximation. Ceres wraps a lossy block through
`ceres::Corrector`, which for any loss with `rho'' <= 0` — Cauchy, Huber and SoftL1 all
qualify — drops its rank-one term and reduces to scaling. Rather than reproduce the
scaling (which would report the wrong cost to the trust region), the cost function
returns the **square-root form** `r' = sqrt(rho(s)/s) * r` per observation, so that

* `|r'|^2 = rho(s)`, i.e. the block's cost is exactly `0.5 * sum_i rho(s_i)`, and
* `J'^T r' = rho'(s) J^T r`, i.e. the gradient is exactly Ceres' corrected gradient
  (`d(s * rho(s)/s)/ds = rho'(s)` does the algebra).

Only the Gauss-Newton Hessian differs from Ceres' by the Triggs-style rank-one term, and
both approximate the same objective, so the stationary point is the same.

## The block-coordinate loop

Extrinsics, rig poses and points cannot go into one pyceres problem — 30 000 Python
residual blocks again — so the refinement alternates:

* **(A)** extrinsics only, in pyceres, with rig poses and points frozen at their current
  values, carrying reprojection plus both priors;
* **(B)** pycolmap's own bundle adjustment with `refine_sensor_from_rig = False`, poses
  and points free.

It stops when a round moves no extrinsic by more than 0.1 mm / 0.005 deg, or after
`num_rounds`. `ExtrinsicRefinementOptions.regularised = False` restores the plain
pycolmap path (`run_mapping(..., optimize_extrinsics=True)`) for comparison.

`refine_extrinsics` takes an already-mapped `MappingResult` and keeps adjusting its
model: the caller maps its camera-referenced `PosedModel` once and the alternation
picks up from there, rather than the same database being mapped twice.

## Deliberate deviations

1. **The reference camera's extrinsic is pinned**, because COLMAP requires the rig's
   reference sensor to be identity (`colsfm.reconstruction`). The blob leaves all eight
   free and lets the absolute priors fix the gauge. Every extrinsic here therefore moves
   relative to the gauge camera, which is the quantity NOTES.md gotcha 13 tabulates.
2. **The gauge keyframe's observations are not un-whitened.** The blob leaves its 13
   constant-frame residuals at weight 1.0 (`docs/spec/keypoints_mapper_main.md` §6.4, an
   upstream bug); with poses frozen in solve (A) those residuals are constant anyway.
3. **The camera projection Jacobian is numeric**, six batched `Camera.img_from_cam`
   calls per evaluation. pycolmap exposes no analytic camera Jacobian and the numeric one
   is model-agnostic; the pose half of the chain rule is analytic.
4. **The pinned camera gets no absolute prior block**, so seven are built where the blob
   logs eight. A block whose every parameter is constant is evaluated by Ceres'
   `Program::RemoveFixedBlocks` with a null residual pointer, and pyceres 2.6's Python
   trampoline segfaults on it. The block is inert in the blob too, so nothing is lost.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pyceres
import pycolmap
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

from colsfm.ceres_pose import (
    CeresSolveOptions,
    LinearSolver,
    PoseBlocks,
    SolverStats,
    attach_quaternion_manifolds,
    batched_skew,
    pose_parameter_blocks,
    rigid3d_from_parameter_blocks,
    solver_options,
    solver_stats,
)
from colsfm.config import BundleAdjustmentConfig, VisionMappingConfig
from colsfm.geometry import MILLIMETRES_PER_METRE, Matrix3, Vector3
from colsfm.mapping import (
    MappingOptions,
    MappingResult,
    backend_name,
    bundle_adjustment_options,
    registered_image_ids,
    run_mapping,
    solve_bundle_adjustment,
)
from colsfm.pose_graph import (
    Information6,
    RelativePoseCost,
    default_information,
    quaternion_plus_jacobian_transpose,
    rotation_matrix_from_quat_xyzw,
)
from colsfm.reconstruction import RIG_ID, PosedModel, RigReference, camera_sensor_id

PointsInVehicle: TypeAlias = Float64[ndarray, "n_obs 3"]
"""Observed 3D points expressed in the vehicle (FLU) frame of their own rig instant."""

PixelObservations: TypeAlias = Float64[ndarray, "n_obs 2"]
"""Measured keypoint positions in pixels."""

CameraPairKey: TypeAlias = tuple[int, int]
"""Two `camera_params_id`s, ascending; one inter-camera extrinsic constraint."""

EXTRINSIC_LINEAR_SOLVER: Final[LinearSolver] = "SPARSE_NORMAL_CHOLESKY"
"""The extrinsics-only problem has a handful of pose blocks and no Schur structure."""

_SMALL_SQUARED_NORM: Final[float] = 1e-8
"""Below this squared residual the Cauchy scale is taken from its series expansion."""

_PROJECTION_STEP_SCALE: Final[float] = 1e-6
"""Relative step of the central-difference camera Jacobian, in metres per metre."""

_LOCAL_PARAMETERS: Final[int] = 6
"""Rotation (3) then translation (3): the tangent space of one `vehicle_T_cam`."""


# ======================================================================================
# options and results
# ======================================================================================


@dataclass(frozen=True, slots=True)
class ExtrinsicRefinementOptions:
    """Knobs for the alternating refinement; the defaults are the blob's own numbers."""

    regularised: bool = True
    """Run the prior-carrying pyceres refinement. False falls back to the plain pycolmap
    path, `run_mapping(..., optimize_extrinsics=True)`, which is unregularised and on
    Galileo overfits (NOTES.md deviation 9)."""
    num_rounds: int = 20
    """Ceiling on the (A)-then-(B) rounds; the tolerances below normally stop it first.
    Block-coordinate descent converges linearly, and on Galileo the per-round motion decays
    by only ~0.88 per round: 0.47, 0.25, 0.18, 0.15, ... mm, reaching the tolerances after
    **19 rounds** and 3.2 s. Three rounds would stop at a third of the final extrinsic move,
    a worse ATE (4.38 mm against 4.28) and a worse reprojection error (0.956 px against
    0.898). A dataset whose bundle adjustment is expensive — RoboCap's is ~20 s a round —
    wants a smaller ceiling."""
    translation_tolerance_mm: float = 0.1
    """Stop once no extrinsic moved further than this in a round."""
    rotation_tolerance_deg: float = 0.005
    """Stop once no extrinsic rotated further than this in a round."""
    extrinsic_translation_sigma_m: float = 0.01
    """`BundleAdjustmentConfig.extrinsic_error_meters`; one sigma of the extrinsic priors.
    `solve_extrinsics` uses whatever this says, but `refine_extrinsics` OVERRIDES it with
    the mapping configuration's own value: the two must agree, and the config is the
    authority. The default is the shipped isaac number, kept so a standalone
    `solve_extrinsics` call is still the blob's problem."""
    extrinsic_rotation_sigma_deg: float = 2.0
    """`BundleAdjustmentConfig.extrinsic_error_degrees`; one sigma of the extrinsic priors.
    Overridden from the configuration by `refine_extrinsics`, like the field above."""
    reprojection_sigma_px: float = 4.0
    """`BundleAdjustmentConfig.reprojection_error_standard_deviation`; whitens the pixel
    residual. Overridden from the configuration by `refine_extrinsics`."""
    use_absolute_prior: bool = True
    """Add the per-camera `absolute_extrinsic` blocks. Off only for ablations."""
    use_relative_prior: bool = True
    """Add the per-camera-pair `relative_extrinsic` blocks. Off only for ablations."""
    max_num_iterations: int = 200
    """`BundleAdjustmentConfig.max_num_iterations`, applied to the extrinsics-only solve.
    Overridden from the configuration by `refine_extrinsics`."""
    num_threads: int = 1
    """Ceres threads for the extrinsics-only solve. The residuals are Python, so worker
    threads serialise on the GIL; `colsfm.pose_graph` measured 8 threads slower than 1."""
    verbose: bool = True
    """Print one line per round."""


@dataclass(frozen=True, slots=True)
class ExtrinsicSolveStats:
    """What one extrinsics-only pyceres solve contained and how it ended."""

    num_cameras: int
    """Cameras with a parameter block, the pinned reference camera included."""
    num_reprojection_blocks: int
    """One vectorised block per camera that both owns observations and is free."""
    num_reprojection_residuals: int
    """`2 * observations`, i.e. every scalar reprojection residual in the problem."""
    num_absolute_prior_blocks: int
    """`absolute_extrinsic` blocks: one per FREE camera, so seven where the blob logs eight
    (its eighth is the pinned camera's, which is constant and contributes nothing)."""
    num_relative_prior_blocks: int
    """Distinct co-observed camera pairs, each carrying its own multiplicity."""
    num_relative_prior_frames: int
    """Sum of those multiplicities: the blob's own block count, 777 on Galileo."""
    iterations: int
    """Ceres minimizer iterations, successful and unsuccessful."""
    initial_cost: float
    """Ceres cost before the solve."""
    final_cost: float
    """Ceres cost after the solve."""
    termination: str
    """`CONVERGENCE`, `NO_CONVERGENCE` or `FAILURE`."""
    seconds: float
    """Wall-clock seconds, problem construction included."""


@dataclass(frozen=True, slots=True)
class ExtrinsicRefinementRound:
    """One (A) extrinsics solve followed by one (B) bundle adjustment."""

    round_index: int
    """Zero-based round number."""
    solve: ExtrinsicSolveStats
    """The extrinsics-only solve of step (A)."""
    max_translation_change_mm: float
    """Largest extrinsic origin move this round made, in millimetres."""
    max_rotation_change_deg: float
    """Largest extrinsic rotation this round made, in degrees."""
    bundle_adjustment_seconds: float
    """Seconds inside step (B)."""
    mean_reprojection_error_px: float
    """Mean reprojection error of the whole model after step (B)."""
    seconds: float
    """Wall-clock seconds for the round."""


@dataclass(frozen=True, slots=True)
class ExtrinsicRefinementResult:
    """What `refine_extrinsics` produced.

    The extrinsics and the elapsed time are read straight off `mapping`, which is a
    live view of the adjusted model (`colsfm.mapping.MappingResult`); duplicating them
    here is what let the two disagree.
    """

    mapping: MappingResult
    """The mapping result: the round-by-round statistics of the pass that produced the
    model, with the timings extended to cover the alternation that followed."""
    rounds: tuple[ExtrinsicRefinementRound, ...]
    """Per-round statistics of the alternation; empty when `regularised` was False."""
    converged: bool
    """Whether a round fell inside both tolerances rather than exhausting `num_rounds`."""
    regularised: bool
    """Whether the prior-carrying path ran; the pipeline reports it and nothing derives
    from it."""

    @property
    def refined_extrinsics(self) -> dict[int, pycolmap.Rigid3d]:
        """`vehicle_T_cam` per `camera_params_id` after the refinement.

        Returns:
            The extrinsics read back out of the adjusted rig.

        Raises:
            ValueError: When the mapping result reports none, which cannot happen for a
                model this function accepted — it needs a camera-referenced rig — and
                would mean the model was swapped underneath it.
        """
        refined: dict[int, pycolmap.Rigid3d] | None = self.mapping.refined_extrinsics
        if refined is None:
            raise ValueError(
                "the refined model reports no extrinsics: its rig reference is not a "
                "camera, so `refine_extrinsics` cannot have produced it"
            )
        return refined

    @property
    def seconds(self) -> float:
        """Wall-clock seconds for the whole stage, the mapping pass included.

        Returns:
            `mapping.total_seconds`, which `refine_extrinsics` extends to cover the
            alternation.
        """
        return self.mapping.total_seconds


# ======================================================================================
# gathering the frozen half of the problem
# ======================================================================================


@dataclass(frozen=True, slots=True)
class CameraObservations:
    """Every observation of one camera, with the frozen geometry already folded in.

    Rig poses and 3D points are constant during solve (A), so each observation reduces
    to the point expressed in its own rig instant's vehicle frame. Projecting it needs
    only `vehicle_T_cam`, which is the parameter.
    """

    camera_params_id: int
    """The camera these observations belong to."""
    camera: pycolmap.Camera
    """Its intrinsics, used batched through `Camera.img_from_cam`."""
    points_in_vehicle: PointsInVehicle
    """`vehicle_T_world(frame) * X` per observation, metres."""
    observed_px: PixelObservations
    """The measured keypoint of each observation, pixels."""


@dataclass(frozen=True, slots=True)
class ImageObservationIndex:
    """Which points one image observes, and where in that image it saw them."""

    frame_id: int
    """The rig instant this image belongs to; picks the pose to fold in."""
    point3D_ids: Int64[ndarray, " n_obs"]
    """The observed point ids, in the image's own `points2D` order."""
    observed_px: PixelObservations
    """The measured keypoint of each of those observations, pixels."""


@dataclass(frozen=True, slots=True)
class CameraObservationIndex:
    """One camera's share of the observation graph, without any geometry folded in.

    This is the half of `CameraObservations` that a round of the alternation cannot
    change: `solve_bundle_adjustment` moves poses and points but adds and removes no
    observation, so the images, the point ids and the pixels are the same every round
    and are built once. Only the geometry has to be gathered again.
    """

    camera_params_id: int
    """The camera these observations belong to."""
    camera: pycolmap.Camera
    """Its intrinsics, used batched through `Camera.img_from_cam`."""
    images: tuple[ImageObservationIndex, ...]
    """Its images that observe at least one point, in ascending image id order."""


def build_observation_index(reconstruction: pycolmap.Reconstruction) -> dict[int, CameraObservationIndex]:
    """Index every observation by camera and image, once, for the whole alternation.

    Args:
        reconstruction: A posed, triangulated model.

    Returns:
        One `CameraObservationIndex` per `camera_params_id` that owns at least one
        observation, keyed by that id.
    """
    images_by_camera: dict[int, list[ImageObservationIndex]] = {}
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        point3D_ids: list[int] = []
        observed: list[Vector3] = []
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            point3D_ids.append(int(point2D.point3D_id))
            observed.append(point2D.xy)
        if not point3D_ids:
            continue
        images_by_camera.setdefault(image.camera_id, []).append(
            ImageObservationIndex(
                frame_id=image.frame_id,
                point3D_ids=np.asarray(point3D_ids, dtype=np.int64),
                observed_px=np.asarray(observed, dtype=np.float64),
            )
        )
    return {
        camera_params_id: CameraObservationIndex(
            camera_params_id=camera_params_id,
            camera=reconstruction.camera(camera_params_id),
            images=tuple(images),
        )
        for camera_params_id, images in sorted(images_by_camera.items())
    }


def _sorted_points(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[Int64[ndarray, " n_points"], Float64[ndarray, "n_points 3"]]:
    """Read every current point position out in one pass, sorted by point id.

    Args:
        reconstruction: The model to read.

    Returns:
        Ascending point3D ids and their world positions in the same order, so that a
        `searchsorted` turns an observation's point id into a row.
    """
    point3D_ids: list[int] = []
    positions: list[Vector3] = []
    for point3D_id, point in reconstruction.points3D.items():
        point3D_ids.append(int(point3D_id))
        positions.append(point.xyz)
    if not point3D_ids:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    ids: Int64[ndarray, " n_points"] = np.asarray(point3D_ids, dtype=np.int64)
    points_xyz: Float64[ndarray, "n_points 3"] = np.asarray(positions, dtype=np.float64)
    order: Int64[ndarray, " n_points"] = np.argsort(ids)
    return ids[order], points_xyz[order]


def camera_observations(
    reconstruction: pycolmap.Reconstruction,
    rig_reference: RigReference,
    index: Mapping[int, CameraObservationIndex] | None = None,
) -> dict[int, CameraObservations]:
    """Fold the frozen rig poses and points into per-camera observation arrays.

    Only this half is per-round work: it is one pass over `points3D` plus fancy
    indexing, where the index it reads (`build_observation_index`) is built once. An
    observation whose point has disappeared since the index was built is dropped rather
    than raising — `solve_bundle_adjustment` inside the alternation filters nothing, so
    that is not expected to happen, but a caller may hand in any model.

    Args:
        reconstruction: A posed, triangulated model built on a camera-referenced rig.
        rig_reference: How that rig was built, from `colsfm.reconstruction.rig_reference`.
        index: A prebuilt observation index for this model, or None to build one.

    Returns:
        One `CameraObservations` per `camera_params_id` that owns at least one
        observation, keyed by that id.
    """
    resolved_index: Mapping[int, CameraObservationIndex] = (
        build_observation_index(reconstruction) if index is None else index
    )
    vehicle_T_world_by_frame_id: dict[int, pycolmap.Rigid3d] = {
        frame_id: world_T_vehicle.inverse()
        for frame_id, world_T_vehicle in rig_reference.world_T_vehicle_by_frame_id(reconstruction).items()
    }
    point3D_ids, points_xyz = _sorted_points(reconstruction)

    observations: dict[int, CameraObservations] = {}
    for camera_params_id, camera_index in sorted(resolved_index.items()):
        points_in_vehicle: list[PointsInVehicle] = []
        pixels: list[PixelObservations] = []
        for image_index in camera_index.images:
            vehicle_T_world: pycolmap.Rigid3d | None = vehicle_T_world_by_frame_id.get(image_index.frame_id)
            if vehicle_T_world is None:
                continue
            rows: Int64[ndarray, " n_obs"] = np.clip(
                np.searchsorted(point3D_ids, image_index.point3D_ids), 0, max(len(point3D_ids) - 1, 0)
            )
            present: Bool[ndarray, " n_obs"] = (
                np.zeros(len(image_index.point3D_ids), dtype=bool)
                if len(point3D_ids) == 0
                else point3D_ids[rows] == image_index.point3D_ids
            )
            if not present.any():
                continue
            world_points: PointsInVehicle = points_xyz[rows[present]]
            points_in_vehicle.append(
                world_points @ np.asarray(vehicle_T_world.rotation.matrix(), dtype=np.float64).T
                + np.asarray(vehicle_T_world.translation, dtype=np.float64)
            )
            pixels.append(image_index.observed_px[present])
        if not points_in_vehicle:
            continue
        observations[camera_params_id] = CameraObservations(
            camera_params_id=camera_params_id,
            camera=camera_index.camera,
            points_in_vehicle=np.ascontiguousarray(np.concatenate(points_in_vehicle, axis=0)),
            observed_px=np.ascontiguousarray(np.concatenate(pixels, axis=0)),
        )
    return observations


def co_observed_camera_pairs(reconstruction: pycolmap.Reconstruction) -> dict[CameraPairKey, int]:
    """Count the rig frames in which each unordered pair of cameras fired together.

    This is the blob's `camera rig extrinsic constraint` set: one block per pair per rig
    frame. On Galileo the counts sum to 777, matching
    `bundle_adjustment_solver.cc:1138`.

    Args:
        reconstruction: A model whose frames group the images of one rig instant.

    Returns:
        `{(lower camera_params_id, higher camera_params_id): number of rig frames}`.
    """
    cameras_by_frame: dict[int, set[int]] = {}
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        cameras_by_frame.setdefault(image.frame_id, set()).add(image.camera_id)
    pair_counts: dict[CameraPairKey, int] = {}
    for cameras in cameras_by_frame.values():
        for lower, higher in itertools.combinations(sorted(cameras), 2):
            pair_counts[(lower, higher)] = pair_counts.get((lower, higher), 0) + 1
    return dict(sorted(pair_counts.items()))


# ======================================================================================
# the Ceres problem
# ======================================================================================


class RepeatedCauchyLoss(pyceres.LossFunction):
    """`CauchyLoss(1.0)` summed over `multiplicity` identical residual blocks.

    A rig's extrinsics are shared across rig frames, so all the blocks the blob adds for
    one camera pair evaluate the same residual. Scaling `rho` and its derivatives by the
    count is exactly equivalent to adding the blocks, and turns Galileo's 777 Python
    evaluations per iteration into 28.
    """

    def __init__(self, multiplicity: int) -> None:
        """Set up the loss.

        Args:
            multiplicity: How many identical blocks this one stands for; at least 1.
        """
        super().__init__()
        if multiplicity < 1:
            raise ValueError(f"multiplicity must be at least 1, got {multiplicity}")
        self._multiplicity: float = float(multiplicity)

    def Evaluate(self, squared_norm: float, out: Float64[ndarray, "3"]) -> None:
        """Write `multiplicity * [rho, rho', rho'']` of `CauchyLoss(1.0)` into `out`.

        The trampoline pyceres 2.6 installs is the C++ signature `Evaluate(double, double*)`,
        not the `evaluate(double) -> array` that the concrete losses expose to Python; a
        subclass that overrides the latter never gets called and Ceres aborts with
        `<Evaluate> not implemented`.

        Args:
            squared_norm: Squared norm of the block's residual.
            out: Three-element output buffer for `rho`, `rho'` and `rho''`.
        """
        denominator: float = 1.0 + squared_norm
        out[0] = self._multiplicity * np.log1p(squared_norm)
        out[1] = self._multiplicity / denominator
        out[2] = -self._multiplicity / denominator**2


def cauchy_square_root_scale(squared_norm: Float64[ndarray, " n_obs"]) -> tuple[Float64[ndarray, " n_obs"], Float64[ndarray, " n_obs"]]:
    """Per-observation `sqrt(rho(s)/s)` of `CauchyLoss(1.0)` and the derivative of `rho(s)/s`.

    Multiplying a residual by the first return value makes its squared norm equal
    `rho(s)`, so a block that carries many observations reports exactly the cost Ceres
    would report for the same observations as separate lossy blocks.

    Args:
        squared_norm: `s = |r|^2` per observation, whitened.

    Returns:
        `(sqrt(rho(s)/s), d(rho(s)/s)/ds)`, both shaped like the input. Near `s = 0` the
        series `rho(s)/s = 1 - s/2 + s^2/3` supplies both, since the closed forms are
        `0/0` there.
    """
    squared: Float64[ndarray, " n_obs"] = np.asarray(squared_norm, dtype=np.float64)
    is_small: Bool[ndarray, " n_obs"] = squared < _SMALL_SQUARED_NORM
    safe: Float64[ndarray, " n_obs"] = np.where(is_small, 1.0, squared)
    ratio: Float64[ndarray, " n_obs"] = np.where(is_small, 1.0 - 0.5 * squared, np.log1p(safe) / safe)
    derivative: Float64[ndarray, " n_obs"] = np.where(
        is_small, -0.5 + (2.0 / 3.0) * squared, (safe / (1.0 + safe) - np.log1p(safe)) / safe**2
    )
    return np.sqrt(ratio), derivative


class RigReprojectionCost(pyceres.CostFunction):
    """Every reprojection residual of one camera, in one block, against `vehicle_T_cam`.

    Parameter blocks are `[quaternion (4, x-y-z-w), translation (3)]` holding
    `vehicle_T_cam`; the residual is `2 * n_obs` long and already carries the whitening
    and the robust loss (see the module docstring for why that is exact).

    The projected point is `p_cam = vehicle_R_cam^T (p_vehicle - vehicle_t_cam)`, so with
    the manifold's right perturbation `R <- R Exp(d)` the analytic chain rule is
    `d p_cam / d[d, dt] = [skew(p_cam), -vehicle_R_cam^T]`; only `d img / d p_cam` is
    numeric.
    """

    def __init__(self, observations: CameraObservations, reprojection_sigma_px: float) -> None:
        """Set up one camera's block.

        Args:
            observations: That camera's frozen points and measured pixels.
            reprojection_sigma_px: `reprojection_error_standard_deviation`, in pixels.

        Raises:
            ValueError: When the camera has no observations or the sigma is not positive.
        """
        super().__init__()
        num_observations: int = int(observations.points_in_vehicle.shape[0])
        if num_observations == 0:
            raise ValueError(f"camera {observations.camera_params_id} has no observations to refine against")
        if reprojection_sigma_px <= 0.0:
            raise ValueError(f"reprojection sigma must be positive, got {reprojection_sigma_px}")
        self.set_num_residuals(2 * num_observations)
        self.set_parameter_block_sizes([4, 3])
        self._camera: pycolmap.Camera = observations.camera
        self._points_in_vehicle: PointsInVehicle = observations.points_in_vehicle
        self._observed_px: PixelObservations = observations.observed_px
        self._inverse_sigma: float = 1.0 / reprojection_sigma_px
        self._num_observations: int = num_observations
        self._vehicle_R_cam: Matrix3 = np.empty((3, 3), dtype=np.float64)
        self._plus_jacobian_t: Float64[ndarray, "3 4"] = np.empty((3, 4), dtype=np.float64)

    def _projection_jacobian(self, points_in_cam: Float64[ndarray, "n_obs 3"]) -> Float64[ndarray, "n_obs 2 3"]:
        """Central-difference `d img_from_cam / d p_cam`, six batched projections.

        Args:
            points_in_cam: The observed points in camera coordinates, metres.

        Returns:
            A 2x3 Jacobian per observation, pixels per metre.
        """
        step: Float64[ndarray, " n_obs"] = _PROJECTION_STEP_SCALE * np.maximum(
            1.0, np.abs(points_in_cam).max(axis=1)
        )
        jacobian: Float64[ndarray, "n_obs 2 3"] = np.empty((self._num_observations, 2, 3), dtype=np.float64)
        for axis in range(3):
            offset: Float64[ndarray, "n_obs 3"] = np.zeros_like(points_in_cam)
            offset[:, axis] = step
            forward: PixelObservations = self._camera.img_from_cam(points_in_cam + offset)
            backward: PixelObservations = self._camera.img_from_cam(points_in_cam - offset)
            jacobian[:, :, axis] = (forward - backward) / (2.0 * step)[:, None]
        return jacobian

    def Evaluate(self, parameters: list, residuals: ndarray, jacobians: list | None) -> bool:
        """Evaluate the whitened, robustified residual and, when asked, ambient Jacobians."""
        quat_xyzw, vehicle_t_cam = parameters
        vehicle_R_cam: Matrix3 = rotation_matrix_from_quat_xyzw(quat_xyzw, self._vehicle_R_cam)
        points_in_cam: Float64[ndarray, "n_obs 3"] = (self._points_in_vehicle - vehicle_t_cam) @ vehicle_R_cam
        error: PixelObservations = (self._camera.img_from_cam(points_in_cam) - self._observed_px) * self._inverse_sigma
        squared_norm: Float64[ndarray, " n_obs"] = np.einsum("ij,ij->i", error, error)
        scale, ratio_derivative = cauchy_square_root_scale(squared_norm)
        residuals[:] = (scale[:, None] * error).ravel()

        if jacobians is None or (jacobians[0] is None and jacobians[1] is None):
            return True

        projection_jacobian: Float64[ndarray, "n_obs 2 3"] = self._projection_jacobian(points_in_cam)
        point_jacobian: Float64[ndarray, "n_obs 3 6"] = np.empty(
            (self._num_observations, 3, _LOCAL_PARAMETERS), dtype=np.float64
        )
        point_jacobian[:, :, :3] = batched_skew(points_in_cam)
        point_jacobian[:, :, 3:] = -vehicle_R_cam.T
        local_jacobian: Float64[ndarray, "n_obs 2 6"] = self._inverse_sigma * (projection_jacobian @ point_jacobian)

        # d(g r)/dp with g = sqrt(rho(s)/s): the scaling plus the rank-one term that makes
        # the returned residual an exact function of the parameters, so that the analytic
        # Jacobian survives a finite-difference check.
        error_times_jacobian: Float64[ndarray, "n_obs 6"] = np.einsum("ni,nij->nj", error, local_jacobian)
        robust: Float64[ndarray, "n_obs 2 6"] = (
            scale[:, None, None] * local_jacobian
            + (ratio_derivative / scale)[:, None, None] * error[:, :, None] * error_times_jacobian[:, None, :]
        )
        if jacobians[0] is not None:
            plus_jacobian_t: Float64[ndarray, "3 4"] = quaternion_plus_jacobian_transpose(
                quat_xyzw, self._plus_jacobian_t
            )
            jacobians[0][:] = (robust[:, :, :3] @ plus_jacobian_t).ravel()
        if jacobians[1] is not None:
            jacobians[1][:] = robust[:, :, 3:].ravel()
        return True


# ======================================================================================
# solve (A): extrinsics only
# ======================================================================================


def _add_reprojection_blocks(
    problem: pyceres.Problem,
    poses: Mapping[int, PoseBlocks],
    observations: Mapping[int, CameraObservations],
    reference_camera_params_id: int,
    reprojection_sigma_px: float,
    costs: list[pyceres.CostFunction],
) -> tuple[int, int]:
    """Add one vectorised reprojection block per free camera.

    Args:
        problem: The problem being built.
        poses: The parameter arrays per `camera_params_id`.
        observations: That round's frozen points and pixels per camera.
        reference_camera_params_id: The pinned rig origin, which gets no block: its
            extrinsic never moves, so its residual would be constant and Ceres would
            drop it anyway.
        reprojection_sigma_px: Whitening sigma, in pixels.
        costs: Kept-alive list every cost is appended to; pyceres holds raw pointers
            into it and a dropped reference is a segfault, not an error.

    Returns:
        The number of blocks added and the number of scalar residuals in them.
    """
    num_blocks: int = 0
    num_residuals: int = 0
    for camera_params_id, camera_obs in observations.items():
        if camera_params_id == reference_camera_params_id:
            continue
        blocks: PoseBlocks = poses[camera_params_id]
        cost: RigReprojectionCost = RigReprojectionCost(camera_obs, reprojection_sigma_px)
        costs.append(cost)
        problem.add_residual_block(cost, None, [blocks.quat_xyzw, blocks.translation])
        num_blocks += 1
        num_residuals += 2 * int(camera_obs.points_in_vehicle.shape[0])
    return num_blocks, num_residuals


def _add_absolute_priors(
    problem: pyceres.Problem,
    poses: Mapping[int, PoseBlocks],
    calibration_vehicle_T_cam: Mapping[int, pycolmap.Rigid3d],
    reference_camera_params_id: int,
    information: Information6,
    identity: PoseBlocks,
    costs: list[pyceres.CostFunction],
    losses: list[pyceres.LossFunction],
) -> int:
    """Add the blob's `absolute_extrinsic` block for every free camera (Eq. 14).

    `RelativePoseCost` against a constant identity source IS the absolute prior: its
    error is `measurement^-1 * (I^-1 * vehicle_T_cam)`.

    Args:
        problem: The problem being built.
        poses: The parameter arrays per `camera_params_id`.
        calibration_vehicle_T_cam: The calibration each prior pulls towards.
        reference_camera_params_id: The pinned rig origin. It would give a block whose
            every parameter is constant; the blob counts such a block (8 on Galileo,
            one inert) but Ceres' `Program::RemoveFixedBlocks` evaluates it with a null
            residual pointer, which segfaults the pyceres 2.6 trampoline. It
            contributes nothing, so it is left out.
        information: The 6x6 weight both prior groups share.
        identity: The constant identity source, owned by the caller so that it outlives
            the problem; set constant here once a block references it.
        costs: Kept-alive list for the cost functions.
        losses: Kept-alive list for the loss functions.

    Returns:
        The number of blocks added.
    """
    num_blocks: int = 0
    for camera_params_id in sorted(poses):
        if camera_params_id == reference_camera_params_id:
            continue
        blocks: PoseBlocks = poses[camera_params_id]
        cost: RelativePoseCost = RelativePoseCost(calibration_vehicle_T_cam[camera_params_id], information)
        loss: pyceres.CauchyLoss = pyceres.CauchyLoss(1.0)
        costs.append(cost)
        losses.append(loss)
        problem.add_residual_block(
            cost, loss, [identity.quat_xyzw, identity.translation, blocks.quat_xyzw, blocks.translation]
        )
        num_blocks += 1
    if num_blocks:
        problem.set_parameter_block_constant(identity.quat_xyzw)
        problem.set_parameter_block_constant(identity.translation)
    return num_blocks


def _add_relative_priors(
    problem: pyceres.Problem,
    poses: Mapping[int, PoseBlocks],
    calibration_vehicle_T_cam: Mapping[int, pycolmap.Rigid3d],
    pair_counts: Mapping[CameraPairKey, int],
    information: Information6,
    costs: list[pyceres.CostFunction],
    losses: list[pyceres.LossFunction],
) -> tuple[int, int]:
    """Add one `relative_extrinsic` block per co-observed camera pair (Eq. 6).

    Each block carries its pair's rig-frame multiplicity through
    `RepeatedCauchyLoss`, which is exactly equivalent to the blob's repeated blocks
    (see the module docstring).

    Args:
        problem: The problem being built.
        poses: The parameter arrays per `camera_params_id`.
        calibration_vehicle_T_cam: The calibration the relative measurement comes from.
        pair_counts: Rig frames per co-observed camera pair.
        information: The 6x6 weight both prior groups share.
        costs: Kept-alive list for the cost functions.
        losses: Kept-alive list for the loss functions.

    Returns:
        The number of blocks added, and the number of blob blocks they stand for.
    """
    num_blocks: int = 0
    num_frames: int = 0
    for (lower, higher), count in pair_counts.items():
        if lower not in poses or higher not in poses:
            continue
        # `higher_T_lower`, the calibrated relative pose the blob preserves (Eq. 6).
        measured: pycolmap.Rigid3d = calibration_vehicle_T_cam[higher].inverse() * calibration_vehicle_T_cam[lower]
        cost: RelativePoseCost = RelativePoseCost(measured, information)
        loss: RepeatedCauchyLoss = RepeatedCauchyLoss(count)
        costs.append(cost)
        losses.append(loss)
        problem.add_residual_block(
            cost,
            loss,
            [
                poses[higher].quat_xyzw,
                poses[higher].translation,
                poses[lower].quat_xyzw,
                poses[lower].translation,
            ],
        )
        num_blocks += 1
        num_frames += count
    return num_blocks, num_frames


def solve_extrinsics(
    reconstruction: pycolmap.Reconstruction,
    rig_reference: RigReference,
    calibration_vehicle_T_cam: Mapping[int, pycolmap.Rigid3d],
    options: ExtrinsicRefinementOptions | None = None,
    observations: Mapping[int, CameraObservations] | None = None,
    pair_counts: Mapping[CameraPairKey, int] | None = None,
) -> tuple[dict[int, pycolmap.Rigid3d], ExtrinsicSolveStats]:
    """Refine every free `vehicle_T_cam` with rig poses and points held fixed.

    The problem is the extrinsic half of the blob's second mapping pass: one vectorised
    reprojection block per free camera, one `absolute_extrinsic` block per camera and one
    `relative_extrinsic` block per co-observed camera pair carrying that pair's rig-frame
    multiplicity. The reference camera's two blocks are constant, because it is the rig
    origin.

    Args:
        reconstruction: A posed, triangulated model on a camera-referenced rig; read only.
        rig_reference: How that rig was built.
        calibration_vehicle_T_cam: `vehicle_T_cam` per `camera_params_id` as the input
            calibration declared it — the mean of both priors.
        options: Sigmas and solver settings; the defaults when None. `refine_extrinsics`
            derives the sigmas from the mapping configuration before calling this.
        observations: This round's folded-in observations, from `camera_observations`;
            gathered here when None. The alternation passes them so the observation
            index behind them is built once rather than once per round.
        pair_counts: Co-observed camera pairs, from `co_observed_camera_pairs`; computed
            here when None. They never change during the alternation.

    Returns:
        The refined `vehicle_T_cam` per `camera_params_id`, and what the solve contained.

    Raises:
        ValueError: When the rig is not camera-referenced, or when a camera in the
            reconstruction has no calibration to be primed from.
    """
    resolved: ExtrinsicRefinementOptions = ExtrinsicRefinementOptions() if options is None else options
    reference_camera_params_id: int | None = rig_reference.camera_params_id
    if reference_camera_params_id is None:
        raise ValueError(
            "extrinsic refinement needs a camera-referenced rig: the vehicle-body reference "
            "owns no images, so COLMAP freezes every extrinsic (NOTES.md gotcha 13)"
        )
    started: float = time.perf_counter()

    current: dict[int, pycolmap.Rigid3d] = rig_reference.vehicle_T_cam_by_camera_params_id(reconstruction)
    missing: set[int] = set(current) - set(calibration_vehicle_T_cam)
    if missing:
        raise ValueError(f"no calibration extrinsic for camera_params_id {sorted(missing)}")

    poses: dict[int, PoseBlocks] = {
        camera_params_id: pose_parameter_blocks(vehicle_T_cam)
        for camera_params_id, vehicle_T_cam in current.items()
    }
    information: Information6 = default_information(
        relative_rotation_sigma_deg=resolved.extrinsic_rotation_sigma_deg,
        relative_translation_sigma_m=resolved.extrinsic_translation_sigma_m,
    )
    problem: pyceres.Problem = pyceres.Problem()
    manifold: pyceres.EigenQuaternionManifold = pyceres.EigenQuaternionManifold()
    # Kept alive for the lifetime of the problem: pyceres holds raw pointers into these.
    costs: list[pyceres.CostFunction] = []
    losses: list[pyceres.LossFunction] = []
    identity: PoseBlocks = pose_parameter_blocks(pycolmap.Rigid3d())

    resolved_observations: Mapping[int, CameraObservations] = (
        camera_observations(reconstruction, rig_reference) if observations is None else observations
    )
    num_reprojection_blocks, num_reprojection_residuals = _add_reprojection_blocks(
        problem,
        poses,
        resolved_observations,
        reference_camera_params_id,
        resolved.reprojection_sigma_px,
        costs,
    )
    num_absolute_blocks: int = 0
    if resolved.use_absolute_prior:
        num_absolute_blocks = _add_absolute_priors(
            problem, poses, calibration_vehicle_T_cam, reference_camera_params_id, information, identity, costs, losses
        )
    resolved_pairs: Mapping[CameraPairKey, int] = {}
    if resolved.use_relative_prior:
        resolved_pairs = co_observed_camera_pairs(reconstruction) if pair_counts is None else pair_counts
    num_relative_blocks, num_relative_frames = _add_relative_priors(
        problem, poses, calibration_vehicle_T_cam, resolved_pairs, information, costs, losses
    )

    attach_quaternion_manifolds(
        problem, {camera_params_id: blocks.quat_xyzw for camera_params_id, blocks in poses.items()}, manifold
    )
    reference_blocks: PoseBlocks = poses[reference_camera_params_id]
    if problem.has_parameter_block(reference_blocks.quat_xyzw):
        problem.set_parameter_block_constant(reference_blocks.quat_xyzw)
        problem.set_parameter_block_constant(reference_blocks.translation)

    summary: pyceres.SolverSummary = pyceres.SolverSummary()
    pyceres.solve(
        solver_options(
            CeresSolveOptions(
                linear_solver=EXTRINSIC_LINEAR_SOLVER,
                max_num_iterations=resolved.max_num_iterations,
                num_threads=resolved.num_threads,
            )
        ),
        problem,
        summary,
    )

    refined: dict[int, pycolmap.Rigid3d] = {
        camera_params_id: rigid3d_from_parameter_blocks(blocks.quat_xyzw, blocks.translation)
        for camera_params_id, blocks in poses.items()
    }
    solved: SolverStats = solver_stats(summary)
    stats: ExtrinsicSolveStats = ExtrinsicSolveStats(
        num_cameras=len(poses),
        num_reprojection_blocks=num_reprojection_blocks,
        num_reprojection_residuals=num_reprojection_residuals,
        num_absolute_prior_blocks=num_absolute_blocks,
        num_relative_prior_blocks=num_relative_blocks,
        num_relative_prior_frames=num_relative_frames,
        iterations=solved.iterations,
        initial_cost=solved.initial_cost,
        final_cost=solved.final_cost,
        termination=solved.termination,
        seconds=time.perf_counter() - started,
    )
    return refined, stats


def apply_extrinsics(
    reconstruction: pycolmap.Reconstruction,
    rig_reference: RigReference,
    vehicle_T_cam_by_camera_params_id: Mapping[int, pycolmap.Rigid3d],
) -> None:
    """Write refined extrinsics back into the reconstruction's rig, in place.

    `sensor_from_rig = cam_T_cam_ref = vehicle_T_cam^-1 * vehicle_T_cam_ref`, the inverse
    of what `colsfm.reconstruction.build_rig` writes. The reference camera is skipped:
    COLMAP holds its `sensor_from_rig` at identity by construction, so the rig frames'
    `rig_from_world` keeps its meaning and no pose has to move.

    Args:
        reconstruction: The model to update.
        rig_reference: How its rig was built; must be camera-referenced.
        vehicle_T_cam_by_camera_params_id: The refined extrinsics.

    Raises:
        ValueError: When the rig is not camera-referenced.
    """
    if rig_reference.camera_params_id is None:
        raise ValueError("cannot write extrinsics back into a vehicle-referenced rig")
    vehicle_T_reference: pycolmap.Rigid3d = vehicle_T_cam_by_camera_params_id[rig_reference.camera_params_id]
    rig: pycolmap.Rig = reconstruction.rig(RIG_ID)
    for camera_params_id, vehicle_T_cam in vehicle_T_cam_by_camera_params_id.items():
        if camera_params_id == rig_reference.camera_params_id:
            continue
        rig.set_sensor_from_rig(camera_sensor_id(camera_params_id), vehicle_T_cam.inverse() * vehicle_T_reference)


def extrinsic_deltas(
    before: Mapping[int, pycolmap.Rigid3d], after: Mapping[int, pycolmap.Rigid3d]
) -> tuple[float, float]:
    """Largest translation and rotation change between two sets of extrinsics.

    Args:
        before: `vehicle_T_cam` per `camera_params_id` before a step.
        after: The same cameras after it.

    Returns:
        `(largest translation change in millimetres, largest rotation change in degrees)`;
        `(0.0, 0.0)` when the two share no camera.
    """
    translation_mm: list[float] = []
    rotation_deg: list[float] = []
    for camera_params_id, previous in before.items():
        if camera_params_id not in after:
            continue
        updated: pycolmap.Rigid3d = after[camera_params_id]
        translation_mm.append(
            float(np.linalg.norm(np.asarray(updated.translation) - np.asarray(previous.translation)))
            * MILLIMETRES_PER_METRE
        )
        rotation_deg.append(float(np.rad2deg((previous.rotation.inverse() * updated.rotation).angle())))
    if not translation_mm:
        return 0.0, 0.0
    return max(translation_mm), max(rotation_deg)


# ======================================================================================
# the alternation
# ======================================================================================


def refine_extrinsics(
    mapping: MappingResult,
    database_path: Path,
    mapping_config: VisionMappingConfig,
    mapping_options: MappingOptions,
    calibration_vehicle_T_cam: Mapping[int, pycolmap.Rigid3d],
    options: ExtrinsicRefinementOptions | None = None,
) -> ExtrinsicRefinementResult:
    """Alternate extrinsics-only solves with bundle adjustment, on an already-mapped model.

    ```
    (the caller has already run run_mapping on a camera-referenced model)
    for k in 0 .. num_rounds - 1:
        (A) solve_extrinsics(...)                        # poses and points frozen
        (B) bundle adjustment with sensor_from_rig fixed # extrinsics frozen
        stop when this round moved no extrinsic past the tolerances
    ```

    The mapping pass is the caller's, not this function's: under
    `--optimize-extrinsics` the pipeline builds **one** camera-referenced `PosedModel`,
    maps it once and hands the adjusted result here, instead of mapping the same
    database twice.

    With `options.regularised` False the whole thing collapses to
    `run_mapping(..., optimize_extrinsics=True)` — the unregularised pycolmap path,
    kept for comparison — run on that same already-adjusted model.

    The three sigmas and the iteration ceiling in `options` are **overridden** from
    `mapping_config.bundle_adjustment`: the pyceres problem and the pycolmap bundle
    adjustment it alternates with have to whiten by the same numbers, and the
    configuration is the authority on what those are.

    Args:
        mapping: What `run_mapping` returned for a camera-referenced `PosedModel`; its
            reconstruction is adjusted further, in place.
        database_path: COLMAP database with keypoints and verified two-view geometries;
            read only by the unregularised path.
        mapping_config: `vision_mapping_config.pb.txt`, including the
            `BundleAdjustmentConfig` every sigma comes from.
        mapping_options: Command-line style mapper knobs; `optimize_extrinsics` is
            overridden by this function.
        calibration_vehicle_T_cam: The input `vehicle_T_cam` per `camera_params_id`, which
            is what both priors pull towards.
        options: Refinement settings; the defaults when None.

    Returns:
        The updated mapping result and the per-round statistics; the refined extrinsics
        are read off the result's own model.
    """
    ba_config: BundleAdjustmentConfig = mapping_config.bundle_adjustment
    resolved: ExtrinsicRefinementOptions = dataclasses.replace(
        ExtrinsicRefinementOptions() if options is None else options,
        extrinsic_translation_sigma_m=ba_config.extrinsic_error_meters,
        extrinsic_rotation_sigma_deg=ba_config.extrinsic_error_degrees,
        reprojection_sigma_px=ba_config.reprojection_error_standard_deviation,
        max_num_iterations=ba_config.max_num_iterations,
    )
    reconstruction: pycolmap.Reconstruction = mapping.reconstruction
    rig_reference: RigReference = mapping.reference
    started: float = time.perf_counter()

    if not resolved.regularised:
        unregularised: MappingResult = run_mapping(
            PosedModel(reconstruction=reconstruction, reference=rig_reference),
            database_path,
            mapping_config,
            dataclasses.replace(mapping_options, optimize_extrinsics=True),
        )
        return ExtrinsicRefinementResult(
            mapping=dataclasses.replace(
                unregularised,
                bundle_adjustment_seconds=mapping.bundle_adjustment_seconds
                + unregularised.bundle_adjustment_seconds,
                total_seconds=mapping.total_seconds + (time.perf_counter() - started),
            ),
            rounds=(),
            converged=True,
            regularised=False,
        )

    # `mapping.ba_backend` rather than `mapping_options.ba_backend`: whatever fallback
    # the mapping pass already resolved holds here too, and re-requesting a backend that
    # just failed would only repeat its message once per round.
    base_options: MappingOptions = dataclasses.replace(
        mapping_options, optimize_extrinsics=False, ba_backend=mapping.ba_backend
    )
    # The same gauge `run_mapping` picked for the mapping pass: the frame owning the
    # lowest registered image id, which is cuSFM's own choice (spec §6.3).
    gauge_frame_id: int = reconstruction.image(min(registered_image_ids(reconstruction))).frame_id
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(ba_config, base_options, reconstruction)
    # Neither of these changes across the rounds: bundle adjustment moves poses and
    # points but adds and removes no observation, so which image saw which point — and
    # therefore which cameras co-observed — is fixed. Only the geometry is re-gathered.
    observation_index: dict[int, CameraObservationIndex] = build_observation_index(reconstruction)
    pair_counts: dict[CameraPairKey, int] = co_observed_camera_pairs(reconstruction)

    rounds: list[ExtrinsicRefinementRound] = []
    converged: bool = False
    for round_index in range(resolved.num_rounds):
        round_started: float = time.perf_counter()
        before: dict[int, pycolmap.Rigid3d] = rig_reference.vehicle_T_cam_by_camera_params_id(reconstruction)
        refined, stats = solve_extrinsics(
            reconstruction,
            rig_reference,
            calibration_vehicle_T_cam,
            resolved,
            observations=camera_observations(reconstruction, rig_reference, observation_index),
            pair_counts=pair_counts,
        )
        apply_extrinsics(reconstruction, rig_reference, refined)
        _, ba_seconds = solve_bundle_adjustment(reconstruction, ba_options, gauge_frame_id, None)
        reconstruction.update_point_3d_errors()
        after: dict[int, pycolmap.Rigid3d] = rig_reference.vehicle_T_cam_by_camera_params_id(reconstruction)
        translation_mm, rotation_deg = extrinsic_deltas(before, after)
        rounds.append(
            ExtrinsicRefinementRound(
                round_index=round_index,
                solve=stats,
                max_translation_change_mm=translation_mm,
                max_rotation_change_deg=rotation_deg,
                bundle_adjustment_seconds=ba_seconds,
                mean_reprojection_error_px=reconstruction.compute_mean_reprojection_error(),
                seconds=time.perf_counter() - round_started,
            )
        )
        if resolved.verbose:
            latest: ExtrinsicRefinementRound = rounds[-1]
            print(
                f"[colsfm] extrinsic round {round_index}: {stats.num_reprojection_residuals // 2} observations "
                f"in {stats.num_reprojection_blocks} blocks, {stats.num_absolute_prior_blocks} absolute and "
                f"{stats.num_relative_prior_frames} relative priors, cost "
                f"{stats.initial_cost:.6g} -> {stats.final_cost:.6g} ({stats.termination}), "
                f"moved {translation_mm:.3f} mm / {rotation_deg:.4f} deg, "
                f"reprojection {latest.mean_reprojection_error_px:.4f} px, "
                f"{latest.seconds:.2f}s"
            )
        if translation_mm < resolved.translation_tolerance_mm and rotation_deg < resolved.rotation_tolerance_deg:
            converged = True
            break

    # Only the timings and the "extrinsics moved" flag need replacing: every count on a
    # `MappingResult` is a property over this very reconstruction, so they are already up
    # to date with the alternation that just finished.
    updated: MappingResult = dataclasses.replace(
        mapping,
        bundle_adjustment_seconds=mapping.bundle_adjustment_seconds
        + sum(round_stats.bundle_adjustment_seconds for round_stats in rounds),
        total_seconds=mapping.total_seconds + (time.perf_counter() - started),
        extrinsics_refined=True,
        ba_backend=backend_name(ba_options),
    )
    return ExtrinsicRefinementResult(
        mapping=updated, rounds=tuple(rounds), converged=converged, regularised=True
    )

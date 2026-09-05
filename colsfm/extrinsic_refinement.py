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
from jaxtyping import Bool, Float
from numpy import ndarray

from colsfm.config import BundleAdjustmentConfig, VisionMappingConfig
from colsfm.mapping import (
    MappingOptions,
    MappingResult,
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
from colsfm.reconstruction import RIG_ID, RigReference, camera_sensor_id

QuaternionXYZW: TypeAlias = Float[ndarray, "4"]
"""Unit quaternion in Eigen order, the layout `pyceres.EigenQuaternionManifold` wants."""

Vector3: TypeAlias = Float[ndarray, "3"]
"""A translation or a rotation vector, metres or radians."""

Matrix3: TypeAlias = Float[ndarray, "3 3"]
"""A rotation matrix."""

PointsInVehicle: TypeAlias = Float[ndarray, "n_obs 3"]
"""Observed 3D points expressed in the vehicle (FLU) frame of their own rig instant."""

PixelObservations: TypeAlias = Float[ndarray, "n_obs 2"]
"""Measured keypoint positions in pixels."""

CameraPairKey: TypeAlias = tuple[int, int]
"""Two `camera_params_id`s, ascending; one inter-camera extrinsic constraint."""

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
    **19 rounds** and 9.9 s. Three rounds would stop at a third of the final extrinsic move,
    a worse ATE (4.38 mm against 4.28) and a worse reprojection error (0.956 px against
    0.898). A dataset whose bundle adjustment is expensive — RoboCap's is ~20 s a round —
    wants a smaller ceiling."""
    translation_tolerance_mm: float = 0.1
    """Stop once no extrinsic moved further than this in a round."""
    rotation_tolerance_deg: float = 0.005
    """Stop once no extrinsic rotated further than this in a round."""
    extrinsic_translation_sigma_m: float = 0.01
    """`BundleAdjustmentConfig.extrinsic_error_meters`; one sigma of the extrinsic priors."""
    extrinsic_rotation_sigma_deg: float = 2.0
    """`BundleAdjustmentConfig.extrinsic_error_degrees`; one sigma of the extrinsic priors."""
    reprojection_sigma_px: float = 4.0
    """`BundleAdjustmentConfig.reprojection_error_standard_deviation`; whitens the pixel residual."""
    use_absolute_prior: bool = True
    """Add the per-camera `absolute_extrinsic` blocks. Off only for ablations."""
    use_relative_prior: bool = True
    """Add the per-camera-pair `relative_extrinsic` blocks. Off only for ablations."""
    max_num_iterations: int = 200
    """`BundleAdjustmentConfig.max_num_iterations`, applied to the extrinsics-only solve."""
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
    """What `refine_extrinsics` produced."""

    mapping: MappingResult
    """The mapping result, with the round-by-round statistics of the base pass and its
    final counters brought up to date with the alternation that followed."""
    refined_extrinsics: dict[int, pycolmap.Rigid3d]
    """`vehicle_T_cam` per `camera_params_id` after the refinement."""
    rounds: tuple[ExtrinsicRefinementRound, ...]
    """Per-round statistics of the alternation; empty when `regularised` was False."""
    converged: bool
    """Whether a round fell inside both tolerances rather than exhausting `num_rounds`."""
    regularised: bool
    """Whether the prior-carrying path ran."""
    seconds: float
    """Wall-clock seconds for the whole stage, the base mapping pass included."""


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


def camera_observations(
    reconstruction: pycolmap.Reconstruction, rig_reference: RigReference
) -> dict[int, CameraObservations]:
    """Fold the frozen rig poses and points into per-camera observation arrays.

    Args:
        reconstruction: A posed, triangulated model built on a camera-referenced rig.
        rig_reference: How that rig was built, from `colsfm.reconstruction.rig_reference`.

    Returns:
        One `CameraObservations` per `camera_params_id` that owns at least one
        observation, keyed by that id.
    """
    vehicle_T_world_by_frame_id: dict[int, pycolmap.Rigid3d] = {
        frame_id: world_T_vehicle.inverse()
        for frame_id, world_T_vehicle in rig_reference.world_T_vehicle_by_frame_id(reconstruction).items()
    }
    points_by_camera: dict[int, list[PointsInVehicle]] = {}
    pixels_by_camera: dict[int, list[PixelObservations]] = {}
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        vehicle_T_world: pycolmap.Rigid3d | None = vehicle_T_world_by_frame_id.get(image.frame_id)
        if vehicle_T_world is None:
            continue
        observed: list[Float[ndarray, "2"]] = []
        world_points: list[Vector3] = []
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            observed.append(point2D.xy)
            world_points.append(reconstruction.point3D(point2D.point3D_id).xyz)
        if not observed:
            continue
        points_xyz: Float[ndarray, "n_image_obs 3"] = np.asarray(world_points, dtype=np.float64)
        points_by_camera.setdefault(image.camera_id, []).append(
            points_xyz @ np.asarray(vehicle_T_world.rotation.matrix(), dtype=np.float64).T
            + np.asarray(vehicle_T_world.translation, dtype=np.float64)
        )
        pixels_by_camera.setdefault(image.camera_id, []).append(np.asarray(observed, dtype=np.float64))

    return {
        camera_params_id: CameraObservations(
            camera_params_id=camera_params_id,
            camera=reconstruction.camera(camera_params_id),
            points_in_vehicle=np.ascontiguousarray(np.concatenate(chunks, axis=0)),
            observed_px=np.ascontiguousarray(np.concatenate(pixels_by_camera[camera_params_id], axis=0)),
        )
        for camera_params_id, chunks in sorted(points_by_camera.items())
    }


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

    def Evaluate(self, squared_norm: float, out: Float[ndarray, "3"]) -> None:
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


def cauchy_square_root_scale(squared_norm: Float[ndarray, " n_obs"]) -> tuple[Float[ndarray, " n_obs"], Float[ndarray, " n_obs"]]:
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
    squared: Float[ndarray, " n_obs"] = np.asarray(squared_norm, dtype=np.float64)
    is_small: Bool[ndarray, " n_obs"] = squared < _SMALL_SQUARED_NORM
    safe: Float[ndarray, " n_obs"] = np.where(is_small, 1.0, squared)
    ratio: Float[ndarray, " n_obs"] = np.where(is_small, 1.0 - 0.5 * squared, np.log1p(safe) / safe)
    derivative: Float[ndarray, " n_obs"] = np.where(
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
        self._plus_jacobian_t: Float[ndarray, "3 4"] = np.empty((3, 4), dtype=np.float64)

    def _projection_jacobian(self, points_in_cam: Float[ndarray, "n_obs 3"]) -> Float[ndarray, "n_obs 2 3"]:
        """Central-difference `d img_from_cam / d p_cam`, six batched projections.

        Args:
            points_in_cam: The observed points in camera coordinates, metres.

        Returns:
            A 2x3 Jacobian per observation, pixels per metre.
        """
        step: Float[ndarray, " n_obs"] = _PROJECTION_STEP_SCALE * np.maximum(
            1.0, np.abs(points_in_cam).max(axis=1)
        )
        jacobian: Float[ndarray, "n_obs 2 3"] = np.empty((self._num_observations, 2, 3), dtype=np.float64)
        for axis in range(3):
            offset: Float[ndarray, "n_obs 3"] = np.zeros_like(points_in_cam)
            offset[:, axis] = step
            forward: PixelObservations = self._camera.img_from_cam(points_in_cam + offset)
            backward: PixelObservations = self._camera.img_from_cam(points_in_cam - offset)
            jacobian[:, :, axis] = (forward - backward) / (2.0 * step)[:, None]
        return jacobian

    def Evaluate(self, parameters: list, residuals: ndarray, jacobians: list | None) -> bool:
        """Evaluate the whitened, robustified residual and, when asked, ambient Jacobians."""
        quat_xyzw, vehicle_t_cam = parameters
        vehicle_R_cam: Matrix3 = rotation_matrix_from_quat_xyzw(quat_xyzw, self._vehicle_R_cam)
        points_in_cam: Float[ndarray, "n_obs 3"] = (self._points_in_vehicle - vehicle_t_cam) @ vehicle_R_cam
        error: PixelObservations = (self._camera.img_from_cam(points_in_cam) - self._observed_px) * self._inverse_sigma
        squared_norm: Float[ndarray, " n_obs"] = np.einsum("ij,ij->i", error, error)
        scale, ratio_derivative = cauchy_square_root_scale(squared_norm)
        residuals[:] = (scale[:, None] * error).ravel()

        if jacobians is None or (jacobians[0] is None and jacobians[1] is None):
            return True

        projection_jacobian: Float[ndarray, "n_obs 2 3"] = self._projection_jacobian(points_in_cam)
        point_jacobian: Float[ndarray, "n_obs 3 6"] = np.empty(
            (self._num_observations, 3, _LOCAL_PARAMETERS), dtype=np.float64
        )
        point_jacobian[:, :, :3] = _batched_skew(points_in_cam)
        point_jacobian[:, :, 3:] = -vehicle_R_cam.T
        local_jacobian: Float[ndarray, "n_obs 2 6"] = self._inverse_sigma * (projection_jacobian @ point_jacobian)

        # d(g r)/dp with g = sqrt(rho(s)/s): the scaling plus the rank-one term that makes
        # the returned residual an exact function of the parameters, so that the analytic
        # Jacobian survives a finite-difference check.
        error_times_jacobian: Float[ndarray, "n_obs 6"] = np.einsum("ni,nij->nj", error, local_jacobian)
        robust: Float[ndarray, "n_obs 2 6"] = (
            scale[:, None, None] * local_jacobian
            + (ratio_derivative / scale)[:, None, None] * error[:, :, None] * error_times_jacobian[:, None, :]
        )
        if jacobians[0] is not None:
            plus_jacobian_t: Float[ndarray, "3 4"] = quaternion_plus_jacobian_transpose(
                quat_xyzw, self._plus_jacobian_t
            )
            jacobians[0][:] = (robust[:, :, :3] @ plus_jacobian_t).ravel()
        if jacobians[1] is not None:
            jacobians[1][:] = robust[:, :, 3:].ravel()
        return True


def _batched_skew(vectors: Float[ndarray, "n 3"]) -> Float[ndarray, "n 3 3"]:
    """Skew-symmetric matrix of every row of `vectors`.

    Args:
        vectors: One 3-vector per row.

    Returns:
        `skew(v)` per row, so that `skew(v) @ w == np.cross(v, w)`.
    """
    skew: Float[ndarray, "n 3 3"] = np.zeros((vectors.shape[0], 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew


# ======================================================================================
# solve (A): extrinsics only
# ======================================================================================


def solve_extrinsics(
    reconstruction: pycolmap.Reconstruction,
    rig_reference: RigReference,
    calibration_vehicle_T_cam: Mapping[int, pycolmap.Rigid3d],
    options: ExtrinsicRefinementOptions | None = None,
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
        options: Sigmas and solver settings; the defaults when None.

    Returns:
        The refined `vehicle_T_cam` per `camera_params_id`, and what the solve contained.

    Raises:
        ValueError: When the rig is not camera-referenced, or when a camera in the
            reconstruction has no calibration to be primed from.
    """
    resolved: ExtrinsicRefinementOptions = ExtrinsicRefinementOptions() if options is None else options
    if rig_reference.camera_params_id is None:
        raise ValueError(
            "extrinsic refinement needs a camera-referenced rig: the vehicle-body reference "
            "owns no images, so COLMAP freezes every extrinsic (NOTES.md gotcha 13)"
        )
    started: float = time.perf_counter()

    current: dict[int, pycolmap.Rigid3d] = rig_reference.vehicle_T_cam_by_camera_params_id(reconstruction)
    missing: set[int] = set(current) - set(calibration_vehicle_T_cam)
    if missing:
        raise ValueError(f"no calibration extrinsic for camera_params_id {sorted(missing)}")

    quaternions: dict[int, QuaternionXYZW] = {}
    translations: dict[int, Vector3] = {}
    for camera_params_id, vehicle_T_cam in current.items():
        quat: QuaternionXYZW = np.asarray(vehicle_T_cam.rotation.quat, dtype=np.float64).copy()
        quaternions[camera_params_id] = quat / np.linalg.norm(quat)
        translations[camera_params_id] = np.asarray(vehicle_T_cam.translation, dtype=np.float64).copy()

    information: Information6 = default_information(
        relative_rotation_sigma_deg=resolved.extrinsic_rotation_sigma_deg,
        relative_translation_sigma_m=resolved.extrinsic_translation_sigma_m,
    )
    problem: pyceres.Problem = pyceres.Problem()
    manifold: pyceres.EigenQuaternionManifold = pyceres.EigenQuaternionManifold()
    # Kept alive for the lifetime of the problem: pyceres holds raw pointers into these.
    costs: list[pyceres.CostFunction] = []
    losses: list[pyceres.LossFunction] = []

    observations: dict[int, CameraObservations] = camera_observations(reconstruction, rig_reference)
    num_reprojection_residuals: int = 0
    num_reprojection_blocks: int = 0
    for camera_params_id, camera_obs in observations.items():
        # The reference camera's extrinsic is the rig origin and never moves, so its
        # reprojection block would be constant; Ceres would drop it anyway.
        if camera_params_id == rig_reference.camera_params_id:
            continue
        cost: RigReprojectionCost = RigReprojectionCost(camera_obs, resolved.reprojection_sigma_px)
        costs.append(cost)
        problem.add_residual_block(cost, None, [quaternions[camera_params_id], translations[camera_params_id]])
        num_reprojection_blocks += 1
        num_reprojection_residuals += 2 * int(camera_obs.points_in_vehicle.shape[0])

    identity_quat: QuaternionXYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    identity_translation: Vector3 = np.zeros(3, dtype=np.float64)
    num_absolute_blocks: int = 0
    if resolved.use_absolute_prior:
        for camera_params_id in sorted(current):
            # The pinned reference camera would give a block whose every parameter is
            # constant. The blob counts such a block (8 on Galileo, one inert) but Ceres'
            # `Program::RemoveFixedBlocks` evaluates it with a null residual pointer, which
            # segfaults the pyceres 2.6 trampoline. It contributes nothing, so it is left out.
            if camera_params_id == rig_reference.camera_params_id:
                continue
            # `RelativePoseCost` against a constant identity source IS the absolute prior:
            # its error is `measurement^-1 * (I^-1 * vehicle_T_cam)`.
            absolute_cost: RelativePoseCost = RelativePoseCost(
                calibration_vehicle_T_cam[camera_params_id], information
            )
            absolute_loss: pyceres.CauchyLoss = pyceres.CauchyLoss(1.0)
            costs.append(absolute_cost)
            losses.append(absolute_loss)
            problem.add_residual_block(
                absolute_cost,
                absolute_loss,
                [identity_quat, identity_translation, quaternions[camera_params_id], translations[camera_params_id]],
            )
            num_absolute_blocks += 1
    if num_absolute_blocks:
        problem.set_parameter_block_constant(identity_quat)
        problem.set_parameter_block_constant(identity_translation)

    pair_counts: dict[CameraPairKey, int] = co_observed_camera_pairs(reconstruction) if resolved.use_relative_prior else {}
    num_relative_blocks: int = 0
    num_relative_frames: int = 0
    for (lower, higher), count in pair_counts.items():
        if lower not in quaternions or higher not in quaternions:
            continue
        # `higher_T_lower`, the calibrated relative pose the blob preserves (Eq. 6).
        measured: pycolmap.Rigid3d = (
            calibration_vehicle_T_cam[higher].inverse() * calibration_vehicle_T_cam[lower]
        )
        relative_cost: RelativePoseCost = RelativePoseCost(measured, information)
        relative_loss: RepeatedCauchyLoss = RepeatedCauchyLoss(count)
        costs.append(relative_cost)
        losses.append(relative_loss)
        problem.add_residual_block(
            relative_cost,
            relative_loss,
            [quaternions[higher], translations[higher], quaternions[lower], translations[lower]],
        )
        num_relative_blocks += 1
        num_relative_frames += count

    for camera_params_id, quat in quaternions.items():
        if problem.has_parameter_block(quat):
            problem.set_manifold(quat, manifold)
        if camera_params_id == rig_reference.camera_params_id and problem.has_parameter_block(quat):
            problem.set_parameter_block_constant(quat)
            problem.set_parameter_block_constant(translations[camera_params_id])

    ceres_options: pyceres.SolverOptions = pyceres.SolverOptions()
    ceres_options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    ceres_options.max_num_iterations = resolved.max_num_iterations
    ceres_options.num_threads = resolved.num_threads
    ceres_options.minimizer_progress_to_stdout = False
    summary: pyceres.SolverSummary = pyceres.SolverSummary()
    pyceres.solve(ceres_options, problem, summary)

    refined: dict[int, pycolmap.Rigid3d] = {}
    for camera_params_id, quat in quaternions.items():
        unit: QuaternionXYZW = quat / np.linalg.norm(quat)
        refined[camera_params_id] = pycolmap.Rigid3d(
            pycolmap.Rotation3d(unit), translations[camera_params_id].copy()
        )
    stats: ExtrinsicSolveStats = ExtrinsicSolveStats(
        num_cameras=len(quaternions),
        num_reprojection_blocks=num_reprojection_blocks,
        num_reprojection_residuals=num_reprojection_residuals,
        num_absolute_prior_blocks=num_absolute_blocks,
        num_relative_prior_blocks=num_relative_blocks,
        num_relative_prior_frames=num_relative_frames,
        iterations=int(summary.num_successful_steps) + int(summary.num_unsuccessful_steps),
        initial_cost=float(summary.initial_cost),
        final_cost=float(summary.final_cost),
        termination=str(summary.termination_type).rsplit(".", maxsplit=1)[-1],
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
            float(np.linalg.norm(np.asarray(updated.translation) - np.asarray(previous.translation))) * 1e3
        )
        rotation_deg.append(float(np.rad2deg((previous.rotation.inverse() * updated.rotation).angle())))
    if not translation_mm:
        return 0.0, 0.0
    return max(translation_mm), max(rotation_deg)


# ======================================================================================
# the alternation
# ======================================================================================


def refine_extrinsics(
    reconstruction: pycolmap.Reconstruction,
    database_path: Path,
    mapping_config: VisionMappingConfig,
    ba_config: BundleAdjustmentConfig,
    mapping_options: MappingOptions,
    rig_reference: RigReference,
    calibration_vehicle_T_cam: Mapping[int, pycolmap.Rigid3d],
    options: ExtrinsicRefinementOptions | None = None,
) -> ExtrinsicRefinementResult:
    """Map the scene, then alternate extrinsics-only solves with bundle adjustment.

    ```
    run_mapping(..., optimize_extrinsics=False)          # the base pass
    for k in 0 .. num_rounds - 1:
        (A) solve_extrinsics(...)                        # poses and points frozen
        (B) bundle adjustment with sensor_from_rig fixed # extrinsics frozen
        stop when this round moved no extrinsic past the tolerances
    ```

    With `options.regularised` False the whole thing collapses to
    `run_mapping(..., optimize_extrinsics=True)`, the unregularised pycolmap path.

    Args:
        reconstruction: A camera-referenced, posed reconstruction to map and adjust.
        database_path: COLMAP database with keypoints and verified two-view geometries.
        mapping_config: `vision_mapping_config.pb.txt`.
        ba_config: The `BundleAdjustmentConfig` inside it.
        mapping_options: Command-line style mapper knobs; `optimize_extrinsics` is
            overridden by this function.
        rig_reference: How the reconstruction's rig was built.
        calibration_vehicle_T_cam: The input `vehicle_T_cam` per `camera_params_id`, which
            is what both priors pull towards.
        options: Refinement settings; the defaults when None.

    Returns:
        The refined extrinsics, the updated mapping result and the per-round statistics.
    """
    resolved: ExtrinsicRefinementOptions = ExtrinsicRefinementOptions() if options is None else options
    started: float = time.perf_counter()

    if not resolved.regularised:
        unregularised: MappingResult = run_mapping(
            reconstruction,
            database_path,
            mapping_config,
            ba_config,
            dataclasses.replace(mapping_options, optimize_extrinsics=True),
            rig_reference=rig_reference,
        )
        assert unregularised.refined_extrinsics is not None, "run_mapping returns them whenever it refines"
        return ExtrinsicRefinementResult(
            mapping=unregularised,
            refined_extrinsics=unregularised.refined_extrinsics,
            rounds=(),
            converged=True,
            regularised=False,
            seconds=time.perf_counter() - started,
        )

    base_options: MappingOptions = dataclasses.replace(mapping_options, optimize_extrinsics=False)
    mapping: MappingResult = run_mapping(reconstruction, database_path, mapping_config, ba_config, base_options)
    # The same gauge `run_mapping` picked for the base pass: the frame owning the lowest
    # registered image id, which is cuSFM's own choice (spec §6.3).
    gauge_frame_id: int = reconstruction.image(min(registered_image_ids(reconstruction))).frame_id
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(ba_config, base_options)

    rounds: list[ExtrinsicRefinementRound] = []
    converged: bool = False
    for round_index in range(resolved.num_rounds):
        round_started: float = time.perf_counter()
        before: dict[int, pycolmap.Rigid3d] = rig_reference.vehicle_T_cam_by_camera_params_id(reconstruction)
        refined, stats = solve_extrinsics(reconstruction, rig_reference, calibration_vehicle_T_cam, resolved)
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

    final_extrinsics: dict[int, pycolmap.Rigid3d] = rig_reference.vehicle_T_cam_by_camera_params_id(reconstruction)
    updated: MappingResult = dataclasses.replace(
        mapping,
        num_points3D=reconstruction.num_points3D(),
        num_observations=reconstruction.compute_num_observations(),
        mean_reprojection_error_px=reconstruction.compute_mean_reprojection_error(),
        mean_track_length=reconstruction.compute_mean_track_length(),
        bundle_adjustment_seconds=mapping.bundle_adjustment_seconds
        + sum(round_stats.bundle_adjustment_seconds for round_stats in rounds),
        total_seconds=time.perf_counter() - started,
        refined_extrinsics=final_extrinsics,
    )
    return ExtrinsicRefinementResult(
        mapping=updated,
        refined_extrinsics=final_extrinsics,
        rounds=tuple(rounds),
        converged=converged,
        regularised=True,
        seconds=time.perf_counter() - started,
    )

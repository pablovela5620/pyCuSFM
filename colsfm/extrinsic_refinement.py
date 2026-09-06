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

**777 relative blocks** is one per unordered pair of cameras that fired in the same rig
frame, once per rig frame, and not `4 declared stereo pairs x N frames`; the arithmetic
that establishes it is in `colsfm.extrinsic_observations`, which counts the pairs, and
the multiplicities it returns are what `RepeatedCauchyLoss` turns back into 777 blocks'
worth of cost.

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
in this stage is whitened (reprojection by `1/4 px`, priors by their sigmas) and every
loss is `CauchyLoss(1.0)`.

## Why one residual block can hold a whole camera

A Python `pyceres` cost function costs roughly a millisecond per call, and Galileo's
refinement carries ~30 000 observations, so `colsfm.extrinsic_costs` holds **all** of one
camera's observations in a single block and applies the robust loss inside `Evaluate`.
That module states why the substitution is exact rather than an approximation.

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
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyceres
import pycolmap

from colsfm.ceres_pose import (
    CeresSolveOptions,
    LinearSolver,
    PoseBlocks,
    SolverStats,
    attach_quaternion_manifolds,
    pose_parameter_blocks,
    rigid3d_from_parameter_blocks,
    solver_options,
    solver_stats,
)
from colsfm.config import BundleAdjustmentConfig, VisionMappingConfig
from colsfm.extrinsic_costs import RepeatedCauchyLoss, RigReprojectionCost
from colsfm.extrinsic_observations import (
    CameraObservationIndex,
    CameraObservations,
    CameraPairKey,
    build_observation_index,
    camera_observations,
    co_observed_camera_pairs,
)
from colsfm.geometry import MILLIMETRES_PER_METRE
from colsfm.mapping import (
    MappingOptions,
    MappingResult,
    backend_name,
    bundle_adjustment_options,
    registered_image_ids,
    run_mapping,
    solve_bundle_adjustment,
)
from colsfm.pose_graph import Information6, RelativePoseCost, default_information
from colsfm.reconstruction import RIG_ID, PosedModel, RigReference, camera_sensor_id

EXTRINSIC_LINEAR_SOLVER: Final[LinearSolver] = "SPARSE_NORMAL_CHOLESKY"
"""The extrinsics-only problem has a handful of pose blocks and no Schur structure."""


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

"""Triangulation and global bundle adjustment: the `keypoints_mapper_main` replacement.

Implements `docs/spec/keypoints_mapper_main.md` §5.3-§5.6 (the triangulate /
merge / complete / filter / BA outer loop) and §6 (the bundle adjustment) on
pycolmap's `IncrementalTriangulator`, `ObservationManager` and
`create_default_bundle_adjuster`, against a reconstruction built by
`colsfm.reconstruction.build_reconstruction` and a COLMAP database written by
the feature and matching stages.

```
frames_meta.json ──> build_reconstruction ──┐
                                            ├──> run_mapping ──> MappingResult
database.db (keypoints + two-view geoms) ───┘
```

Poses are a **prior, not an initialisation**: every rig frame is registered
before the first triangulation, and only bundle adjustment moves them.

## What is deliberately not reproduced

Two cuSFM behaviours are known defects and are left out (§9.5):

1. **Unweighted constant-frame residuals.** cuSFM whitens every reprojection
   residual by `1/sigma` except the ones observing the gauge keyframe, whose
   functor has no weight member — 13 of 6276 blocks in Galileo. COLMAP applies
   one weighting uniformly and there is no way to ask for the inconsistency.
2. **The 3-D depth residual** (`VehicleCameraReprojectionCost3D`) for keypoints
   carrying depth. pycolmap's bundle adjuster is 2-D only, and Galileo sets
   `keypoint_feature_has_depth: false`.

## Extrinsic refinement is real but unregularised

`optimize_extrinsics` needs a **camera-referenced** rig; on the default
vehicle-referenced one COLMAP freezes every `sensor_from_rig` and the refinement
is a silent no-op, so `run_mapping` raises instead (see `colsfm.reconstruction`).
Given one, it does refine: a 20 mm perturbation of a Galileo camera comes back to
0.70 mm.

What it cannot reproduce is cuSFM's **regularisation**. The blob's refinement
pass adds absolute-extrinsic priors (the paper's Eq. 14, one block per camera)
and relative-extrinsic constraints between cameras (Eq. 6, 777 blocks on
Galileo), both with sigmas from `vision_mapping_config.pb.txt` (§7).
`pycolmap.BundleAdjustmentOptions` exposes neither, and the Ceres problem
`create_default_bundle_adjuster` builds cannot be extended from standalone
pyceres (the pybind11 registry split, pycolmap-capabilities.md §8). colsfm
therefore optimises reprojection alone, and on Galileo that trades trajectory
accuracy for pixels: 0.861 px against 1.334 px, but 5.70 mm ATE against 4.33 mm,
with extrinsics moving up to 234 mm where the blob moves 11 mm. Only a stereo
pair's own relative extrinsic is well determined by a 0.66 m sweep. The flag is
off by default; NOTES.md deviation 9 carries the table.

## Overrides forced on pycolmap, and why

| Setting | pycolmap default | Here | Reason |
|---|---|---|---|
| `IncrementalTriangulatorOptions.min_angle` | 1.5 | `min_triangulation_deg` (2.0) | config §3.1 |
| `merge_max_reproj_error`, `complete_max_reproj_error` | 4.0 | the round's gate `g_k` | cuSFM merges and completes under the same decaying gate (§5.4, §5.6) |
| `ignore_two_view_tracks` | True | `min_correspondences > 2` | `min_correspondences: 3` drops two-view tracks |
| `random_seed` | -1 | `random_seed` (1) | determinism |
| `refine_focal_length` | True | False | `--optimize_intrinsics=false` |
| `refine_extra_params` | True | False | same |
| `refine_principal_point` | False | False | same; already off |
| `refine_sensor_from_rig` | True | `MappingOptions.optimize_extrinsics` (False) | `--optimize_extrinsics=false`; needs a camera-referenced rig, see below |
| `ceres.loss_function_type` | TRIVIAL | `loss_type` (CAUCHY) | config §3.2 |
| `ceres.loss_function_scale` | 1.0 | `sigma * loss_function_scale` (4.0) | see below |
| `ceres.auto_select_solver_type` | True | False | otherwise the explicit solver choice is ignored |
| `solver_options.linear_solver_type` | SPARSE_NORMAL_CHOLESKY | `MAPPING_LINEAR_SOLVER` (SPARSE_SCHUR) | cuSFM: SPARSE_SCHUR on CPU, SPARSE_NORMAL_CHOLESKY on the cuDSS path (§6.5) |
| `solver_options.max_num_iterations` | 100 | 200 | config §3.2 |
| `solver_options.max_linear_solver_iterations` | 200 | 100 | config §3.2 |
| `solver_options.num_threads` | -1 | `num_threads` (8) | config §3.2 |
| `solver_options.function_tolerance` | **0.0** | 1e-6 | cuSFM leaves Ceres' defaults; pycolmap overrides them |
| `solver_options.gradient_tolerance` | **1e-4** | 1e-10 | same |
| `solver_options.parameter_tolerance` | **0.0** | 1e-8 | same |

**The loss scale carries the whitening.** cuSFM divides the 2-vector
reprojection residual by `reprojection_error_standard_deviation = 4.0 px` and
then applies `CauchyLoss(1.0)`. COLMAP does not whiten. `CauchyLoss(a)` is
`a^2 log(1 + s/a^2)` in the squared residual `s`, so evaluating it at `s/sigma^2`
with `a = 1` equals evaluating it at `s` with `a = sigma`, up to the constant
factor `sigma^2` that cannot move the minimum. The port therefore sets
`loss_function_scale = sigma * loss_function_scale = 4.0` (§9.2).

## What §9.3 promises that pycolmap cannot deliver

`IncrementalTriangulatorOptions` has **no** RANSAC block and no residual-type
switch: `ransac.max_error`, `confidence`, `max_num_trials`, `min_inlier_ratio`
and `TriangulationResidualType.REPROJECTION_ERROR` live on
`EstimateTriangulationOptions`, which only the standalone
`pycolmap.estimate_triangulation` takes. Driving the triangulator therefore
means accepting COLMAP's own angular create/continue thresholds (2.0 deg) and
enforcing the pixel gate `g_k` where it is reachable: on merging, on completion
and on the per-round filter. Deriving the create threshold from the gate through
the focal length (`atan(g_k / f)`) was tried and moves nothing: 2035 points
against 2035, 1.6706 px against 1.6706.

## Measured against the blob, on the blob's own matches

| | Galileo 32 keyframes | Galileo 226 keyframes |
|---|---|---|
| images with observations | 28 vs 28 | 224 vs 224 |
| points | 2035 vs 1275 | 14 496 vs 5065 |
| mean reprojection | 1.671 vs 1.699 px | 1.415 vs 1.550 px |
| worst rig-pose difference | 1.26 mm / 0.118 deg | 2.55 mm / 0.092 deg |
| the blob's own pose correction | 11.27 mm / 0.569 deg | 12.66 mm / 0.364 deg |
| runtime | 0.9 vs 1.1 s | 8.0 vs 6.7 s |

The point count is the one figure that does not match. cuSFM grows **one**
disjoint track per connected component of the match graph and triangulates it
with plain MSAC — no local optimisation, no final refit — so it rejects roughly
half its candidate tracks. COLMAP's LO-RANSAC keeps them. The result is a
superset at a lower reprojection error: 98-99 % of the blob's points have one of
ours within 5 cm, and merging is already at a fixed point (a second
`merge_all_tracks` pass returns 0), so the difference is structural rather than
a missing merge.

## `ba_backend = "caspar"`: COLMAP's GPU bundle adjustment

`MappingOptions.ba_backend` picks the solver behind the same
`create_default_bundle_adjuster` call. `caspar` needs a `CASPAR_ENABLED` build,
which only the `colsfm-caspar` environment has (`docs/caspar-build.md`); the
stock environment raises when the adjuster is built and this module falls back to
Ceres and says so, so the flag is safe to pass anywhere.

**CASPAR drops the robust loss.** It solves plain least squares whatever
`ceres.loss_function_type` says, so the CAUCHY term at sigma 4 that whitens
cuSFM's residual (§9.2, the table above) is simply not applied: every outlier
enters the normal equations at full weight. What still rejects them is the outer
loop — the per-round gate `g_k` filters at 25, 20, 15, 10, 5 px *before* each
solve, and `filter_degenerate_points` / `filter_projection_failures` run after it.
That is enough: an ablation on KITTI 06 — Ceres with `loss_type: TRIVIAL` — scores
0.899 m Sim(3) ATE against the robust run's 0.895 m, so the loss is not what
CASPAR's 1.299 m costs (NOTES.md "CASPAR backend"). Its float32 arithmetic is.

Three more constraints, all from COLMAP 4.2.0's source and all handled here:
observations from a camera that is not PINHOLE or SIMPLE_RADIAL are *silently
skipped*, so `resolve_ba_backend` falls back to Ceres rather than solve a
quietly smaller problem; `refine_sensor_from_rig = True` is a hard throw on a
multi-sensor frame, so `optimize_extrinsics` falls back too (the regularised
`colsfm.extrinsic_refinement` path keeps CASPAR: it holds the extrinsics fixed
inside the pycolmap solve by construction); and `refine_focal_length` must equal
`refine_extra_params`, which it does — both are False.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64
from numpy import ndarray

from colsfm.ceres_pose import COLMAP_LINEAR_SOLVER, LinearSolver
from colsfm.config import BundleAdjustmentConfig, LossFunctionType, VisionMappingConfig
from colsfm.reconstruction import (
    RIG_ID,
    PosedModel,
    RigReference,
    camera_sensor_id,
)
from colsfm.reconstruction import num_registered_images as num_observing_images

MAPPING_LINEAR_SOLVER: Final[LinearSolver] = "SPARSE_SCHUR"
"""cuSFM's CPU linear solver, and the only one this stage ever wants: a bundle adjustment
is exactly the problem the Schur complement exists for. The cuDSS path uses
SPARSE_NORMAL_CHOLESKY instead (§6.5), which colsfm does not build."""

CERES_FUNCTION_TOLERANCE: Final[float] = 1e-6
"""Ceres' own default, which cuSFM keeps and pycolmap overrides to 0.0."""

CERES_GRADIENT_TOLERANCE: Final[float] = 1e-10
"""Ceres' own default, which cuSFM keeps and pycolmap overrides to 1e-4."""

CERES_PARAMETER_TOLERANCE: Final[float] = 1e-8
"""Ceres' own default, which cuSFM keeps and pycolmap overrides to 0.0."""

LOSS_FUNCTION_BY_NAME: Final[dict[LossFunctionType, pycolmap.LossFunctionType]] = {
    "TRIVIAL": pycolmap.LossFunctionType.TRIVIAL,
    "SOFT_L1": pycolmap.LossFunctionType.SOFT_L1,
    "CAUCHY": pycolmap.LossFunctionType.CAUCHY,
    "HUBER": pycolmap.LossFunctionType.HUBER,
}
"""cuSFM's `loss_type` enum to pycolmap's; the two sets coincide exactly."""

BaBackend: TypeAlias = Literal["ceres", "caspar"]
"""Which implementation solves the global bundle adjustment; `MappingOptions.ba_backend`."""

BUNDLE_ADJUSTMENT_BACKEND: Final[dict[BaBackend, pycolmap.BundleAdjustmentBackend]] = {
    "ceres": pycolmap.BundleAdjustmentBackend.CERES,
    "caspar": pycolmap.BundleAdjustmentBackend.CASPAR,
}
"""The name to pycolmap's enum. The enum is bound unconditionally, so its presence proves
nothing about the build — only a solve does (`docs/caspar-build.md`)."""

CASPAR_CAMERA_MODELS: Final[frozenset[pycolmap.CameraModelId]] = frozenset(
    {pycolmap.CameraModelId.PINHOLE, pycolmap.CameraModelId.SIMPLE_RADIAL}
)
"""The only two models CASPAR projects. Observations of any other model are dropped with
a `LOG(WARNING)` and the solve still reports success, so this is a pre-flight, not a guard."""

CASPAR_DISABLED_MARKER: Final[str] = "CASPAR_ENABLED"
"""What COLMAP's "built without CASPAR_ENABLED" `ValueError` says; the capability test."""

CasparOptions: TypeAlias = dict[str, float | int]
"""Numeric overrides for `pycolmap.BundleAdjustmentOptions.caspar`, by attribute name.

The stopping and damping knobs of CASPAR's Levenberg-Marquardt / PCG loop —
`solver_iter_max`, `pcg_iter_max`, `pcg_rel_error_exit`, `pcg_rel_score_exit`,
`pcg_rel_decrease_min`, `solver_rel_decrease_min`, `score_exit_value`, `diag_*`.
Values are cast to the attribute's own type on the way in, so the integer counters
take integers whichever way they were written. `gpu_index` is a string and is not
settable here; `MappingOptions.use_gpu` owns it."""

BRIEF_REPORT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"Iterations:\s*(\d+),\s*Initial cost:\s*([0-9.eE+-]+),\s*Final cost:\s*([0-9.eE+-]+)"
)
"""`BundleAdjustmentSummary` exposes iterations and cost only inside `brief_report()`."""


@dataclass(frozen=True, slots=True)
class RoundStats:
    """One outer triangulate / filter / bundle-adjust round (§5.4)."""

    round_index: int
    """Zero-based round number, `k` in the gate schedule."""
    max_pixel_error: float
    """The round's reprojection gate `g_k`, in pixels."""
    num_merged: int
    """Observations merged by `merge_all_tracks`."""
    num_completed: int
    """Observations added by `complete_all_tracks`."""
    num_filtered: int
    """Observations dropped by the reprojection, angle, depth and track-length filters."""
    num_diverged: int
    """Observations dropped by the degeneracy guard *after* the solve, i.e. points bundle
    adjustment itself pushed out of bounds. Normally 0; see `filter_degenerate_points`."""
    num_observations: int
    """Observations left in the reconstruction after filtering; cuSFM's `num observed`."""
    observation_change: float
    """`(merged + completed + filtered) / observations`, the early-exit statistic."""
    num_points3D: int
    """Points left after this round's bundle adjustment."""
    mean_reprojection_error_px: float
    """Mean reprojection error after this round, in pixels."""
    ba_num_iterations: int
    """Ceres iterations the solver reported."""
    ba_initial_cost: float
    """Ceres cost before the solve. Not cuSFM's logged `Initial cost`, which is a normalised RMS."""
    ba_final_cost: float
    """Ceres cost after the solve."""
    ba_termination: str
    """`CONVERGENCE`, `NO_CONVERGENCE` or `FAILURE`."""
    seconds: float
    """Wall-clock seconds the round took, bundle adjustment included."""


@dataclass(frozen=True, slots=True)
class PolishStats:
    """The one Ceres bundle adjustment that finishes a CASPAR mapping run.

    Nothing is merged, completed or filtered around it, so the observation set that
    goes in is the one that comes out and every number here is the solve's alone
    (`docs/caspar-build.md` § Shipped: CASPAR + Ceres polish).
    """

    num_observations: int
    """Observations the solve parameterised; unchanged by it, since nothing filters."""
    ba_num_iterations: int
    """Ceres iterations, from `brief_report()`. 22 on KITTI 06 from CASPAR's answer."""
    ba_initial_cost: float
    """Ceres cost of the model the rounds left behind."""
    ba_final_cost: float
    """Ceres cost after the polish; 3.4 % below the initial one on KITTI 06."""
    ba_termination: str
    """`CONVERGENCE`, `NO_CONVERGENCE` or `FAILURE`."""
    mean_reprojection_error_before_px: float
    """Mean reprojection error as CASPAR left it, in pixels."""
    mean_reprojection_error_after_px: float
    """Mean reprojection error after the polish, over the same observations."""
    seconds: float
    """Wall-clock seconds inside `BundleAdjuster.solve()`; 9.8 s on KITTI 06's 2156 images."""


RoundCallback: TypeAlias = Callable[[RoundStats, pycolmap.Reconstruction], None]
"""What `MappingOptions.round_callback` takes: one round's statistics and the live model."""


@dataclass(frozen=True, slots=True)
class MappingOptions:
    """Knobs the mapper takes from the command line rather than from a config file."""

    optimize_extrinsics: bool = False
    """Refine `sensor_from_rig`; cuSFM's `--optimize_extrinsics`, off by default."""
    fixed_camera_params_id: int | None = None
    """Camera whose extrinsic stays fixed while the others are refined; cuSFM's `--fixed_camera_name`."""
    num_threads: int | None = None
    """Ceres threads; the config's `num_threads` when None. Use 1 for a reproducible solve.
    Triangulation is single-threaded in COLMAP either way, so it does not carry cuSFM's
    threaded-triangulation non-determinism (§5.3)."""
    use_gpu: bool = False
    """Hand the Ceres solve to the GPU; COLMAP still needs 50+ images before it switches
    (`min_num_images_gpu_solver`, pycolmap-capabilities.md §11). Off by default because the
    measured gain does not exist: 1.68 s against 1.60 s on Galileo's 226 images, and 19.2 s
    against 20.9 s on 800 RoboCap images — 5 % the wrong way, then 8 % the right way. The
    blob's own mapper is Ceres on the CPU, so that is where the default stays. Which GPU
    is left to Ceres, i.e. pycolmap's `gpu_index` default of `-1`."""
    ba_backend: BaBackend = "ceres"
    """Which implementation solves the global bundle adjustment.

    `caspar` is COLMAP's GPU backend and needs a `CASPAR_ENABLED` build — the
    `colsfm-caspar` environment. It falls back to `ceres`, loudly, on three
    conditions: a camera model outside PINHOLE / SIMPLE_RADIAL (whose observations
    CASPAR would silently drop), `optimize_extrinsics` (which CASPAR cannot honour),
    and a pycolmap built without it. See the module docstring for the robust loss
    CASPAR does not apply."""
    caspar_ceres_polish: bool = True
    """Finish a CASPAR run with one Ceres global bundle adjustment over the final model.

    Read only when `ba_backend` actually resolves to `caspar`: every fallback ends on
    Ceres, which has already converged, so there is nothing to polish. CASPAR's
    `CONVERGED_DIAG_EXIT` fires while a few percent of the cost is still reachable,
    and on KITTI 06 that costs 0.4 m of trajectory; one Ceres solve on CASPAR's own
    observation set gets all of it back — 1.299 m to 0.904 m Sim(3) ATE in 9.8 s, for a
    41.3 s mapping stage against the Ceres mapper's 0.895 m in 148.5 s
    (`docs/caspar-build.md` § Shipped: CASPAR + Ceres polish). Set False for the ablation."""
    caspar_options: CasparOptions | None = None
    """Overrides applied to `BundleAdjustmentOptions.caspar` when `ba_backend` is `caspar`.

    None keeps COLMAP's defaults. Read only on the CASPAR backend: on Ceres the
    options object is never consulted, so a dict here is silently inert, which is
    what a sweep driver that flips only `--ba-backend` wants. Unknown names raise
    (`apply_caspar_options`) rather than being ignored."""
    verbose: bool = True
    """Print one line per triangulation pass and per outer round, as cuSFM does."""
    round_callback: RoundCallback | None = None
    """Called once at the end of every outer round, after the round's filter and solve.

    The hook a live viewer needs: it receives the round's `RoundStats` and the
    `pycolmap.Reconstruction` in exactly the state that round left it, so a caller
    can log the cloud and the poses per round without re-running the mapper or
    scraping `verbose` output. It runs *before* the early-exit check, so the last
    round is always reported. Nothing in the mapper reads what it returns, and a
    callback that raises aborts the mapping — it is a plain call, not a try/except."""


@dataclass(slots=True)
class Correspondences:
    """The correspondence graph a triangulator needs, kept alive alongside its owner.

    `IncrementalTriangulator` holds raw references to the graph and the
    reconstruction, so the `DatabaseCache` that owns the graph has to outlive
    both. Keeping all three in one object is what guarantees that.
    """

    database_cache: pycolmap.DatabaseCache
    """Owns the correspondence graph; must outlive every triangulator built on it."""
    graph: pycolmap.CorrespondenceGraph
    """Keypoint correspondences from the database's verified two-view geometries."""
    observation_manager: pycolmap.ObservationManager
    """Bookkeeping for observations and the filters, bound to the reconstruction."""
    num_images: int
    """Images the database contributed correspondences for."""
    num_image_pairs: int
    """Verified image pairs that survived `min_num_matches`."""


@dataclass(frozen=True, slots=True)
class MappingResult:
    """What `run_mapping` produced, and how long each phase took.

    Every count is a **property** computed from `reconstruction` on demand, not a
    number copied out of it at return time. `colsfm.extrinsic_refinement` keeps
    adjusting the very same model after `run_mapping` has returned, and a stored
    count would then be quietly describing a model that no longer exists.
    """

    reconstruction: pycolmap.Reconstruction
    """The adjusted reconstruction; the same object that was passed in."""
    reference: RigReference
    """Where the rig origin sits, carried through from the `PosedModel` that was mapped."""
    rounds: tuple[RoundStats, ...]
    """Per-round statistics, in order."""
    correspondence_seconds: float
    """Time spent loading keypoints and two-view geometries from the database."""
    triangulation_seconds: float
    """Time spent in the initial triangulation passes."""
    bundle_adjustment_seconds: float
    """Time spent inside `BundleAdjuster.solve`, summed over the rounds and the polish."""
    total_seconds: float
    """Wall-clock seconds for the whole call."""
    extrinsics_refined: bool = False
    """Whether this pass refined `sensor_from_rig`; what makes `refined_extrinsics`
    something other than the input calibration read back out of the rig."""
    ba_backend: BaBackend = "ceres"
    """Which backend the solves actually ran on — `ceres` whenever one of the three
    fallbacks of `MappingOptions.ba_backend` fired, so this is evidence rather than a
    request."""
    polish: PolishStats | None = None
    """The closing Ceres solve of a CASPAR run, or None when none ran.

    None means either `ba_backend == "ceres"` — a fallback included, since a Ceres
    run is already at its own fixed point — or `caspar_ceres_polish` switched off.
    Its seconds are part of `bundle_adjustment_seconds`."""

    @property
    def num_registered_images(self) -> int:
        """Images belonging to a registered frame, whether or not they see a point.

        Returns:
            The count `registered_image_ids` reports, which is every image of a
            `build_reconstruction` model.
        """
        return len(registered_image_ids(self.reconstruction))

    @property
    def num_images_with_observations(self) -> int:
        """Images that observe at least one point; what a COLMAP export writes out.

        Returns:
            `colsfm.reconstruction.num_registered_images` of this model — the other,
            narrower sense of "registered"; see that function.
        """
        return num_observing_images(self.reconstruction)

    @property
    def num_points3D(self) -> int:
        """Triangulated points that survived every filter.

        Returns:
            The point count.
        """
        return self.reconstruction.num_points3D()

    @property
    def num_observations(self) -> int:
        """Point-to-image observations in the final map.

        Returns:
            The observation count.
        """
        return self.reconstruction.compute_num_observations()

    @property
    def mean_reprojection_error_px(self) -> float:
        """Mean reprojection error over all observations, in pixels.

        Returns:
            The mean error; valid only after `update_point_3d_errors`, which
            `run_mapping` calls at the end of every round.
        """
        return self.reconstruction.compute_mean_reprojection_error()

    @property
    def mean_track_length(self) -> float:
        """Mean observations per point; cuSFM's `Mean length`.

        Returns:
            The mean track length.
        """
        return self.reconstruction.compute_mean_track_length()

    @property
    def refined_extrinsics(self) -> dict[int, pycolmap.Rigid3d] | None:
        """`vehicle_T_cam` per `camera_params_id` after the solve.

        Returns:
            The extrinsics read back out of the adjusted rig, or None when nothing
            refined them — either because `optimize_extrinsics` was off or because
            the rig is vehicle-referenced, which COLMAP would freeze anyway. The
            reference camera's entry is the input one exactly: it is the rig origin,
            so bundle adjustment holds no block for it.
        """
        if not self.extrinsics_refined or self.reference.camera_params_id is None:
            return None
        return self.reference.vehicle_T_cam_by_camera_params_id(self.reconstruction)


@dataclass(frozen=True, slots=True)
class _FilterCounts:
    """Observations removed by one round's filters, split by cause."""

    reprojection_and_angle: int = 0
    """Dropped by `filter_all_points3D` (reprojection, negative depth, triangulation angle)."""
    negative_depth: int = 0
    """Dropped by the explicit cheirality pass."""
    short_track: int = 0
    """Dropped for having fewer than `min_correspondences` observations."""
    degenerate: int = 0
    """Dropped by the pre-solve guard: the world-Z cap, a non-finite coordinate or a
    coordinate whose magnitude exceeds `depth_threshold`."""

    def total(self) -> int:
        """Total observations removed.

        Returns:
            The sum of every cause.
        """
        return self.reprojection_and_angle + self.negative_depth + self.short_track + self.degenerate


def pixel_error_schedule(mapping_config: VisionMappingConfig) -> tuple[float, ...]:
    """The linear per-round reprojection gate, `initial` to `final` (§5.6).

    `g_k = initial + (final - initial) * k / (N - 1)`, which is 25, 20, 15, 10, 5
    for the isaac profile and 40, 32, 24, 16, 8 for AV. A single-round
    configuration stays at `initial`.

    Args:
        mapping_config: The mapping configuration.

    Returns:
        One gate per outer round, in pixels.
    """
    num_rounds: int = max(mapping_config.num_ba_iterations, 1)
    if num_rounds == 1:
        return (mapping_config.initial_max_pixel_error,)
    span: float = mapping_config.final_max_pixel_error - mapping_config.initial_max_pixel_error
    return tuple(mapping_config.initial_max_pixel_error + span * index / (num_rounds - 1) for index in range(num_rounds))


def triangulator_options(mapping_config: VisionMappingConfig, max_pixel_error: float) -> pycolmap.IncrementalTriangulatorOptions:
    """Triangulator options for one round's gate.

    Args:
        mapping_config: The mapping configuration.
        max_pixel_error: The round's reprojection gate, in pixels.

    Returns:
        Options with the module docstring's overrides applied.
    """
    options: pycolmap.IncrementalTriangulatorOptions = pycolmap.IncrementalTriangulatorOptions()
    options.min_angle = mapping_config.min_triangulation_deg
    options.merge_max_reproj_error = max_pixel_error
    options.complete_max_reproj_error = max_pixel_error
    options.ignore_two_view_tracks = mapping_config.min_correspondences > 2
    options.random_seed = mapping_config.random_seed
    return options


def resolve_ba_backend(options: MappingOptions, reconstruction: pycolmap.Reconstruction | None = None) -> BaBackend:
    """The backend that will actually run, after the two pre-flight fallbacks.

    Neither condition is an error: CASPAR would *silently* drop the observations of
    an unsupported camera model, and would hold `sensor_from_rig` fixed where the
    caller asked for it to move. Both are reported and Ceres takes the solve.

    Args:
        options: Command-line style knobs; `ba_backend` is what is being resolved.
        reconstruction: The model about to be adjusted, for its camera models. When
            None the camera check is skipped — the caller has none to offer.

    Returns:
        `ceres` or `caspar`.
    """
    if options.ba_backend != "caspar":
        return options.ba_backend
    if options.optimize_extrinsics:
        print(
            "[colsfm] CASPAR holds `sensor_from_rig` fixed and throws when asked to refine it; "
            "`--optimize-extrinsics` therefore falls back to Ceres for this solve"
        )
        return "ceres"
    if reconstruction is not None:
        unsupported: set[str] = {
            camera.model.name for camera in reconstruction.cameras.values() if camera.model not in CASPAR_CAMERA_MODELS
        }
        if unsupported:
            print(
                f"[colsfm] CASPAR projects only PINHOLE and SIMPLE_RADIAL, and this model has "
                f"{sorted(unsupported)}, whose observations it would silently drop; using Ceres"
            )
            return "ceres"
    return "caspar"


def backend_name(ba_options: pycolmap.BundleAdjustmentOptions) -> BaBackend:
    """Which backend a set of options selects, as this module's name for it.

    Args:
        ba_options: The options a solve ran with.

    Returns:
        `caspar` when the options select CASPAR, `ceres` otherwise.
    """
    return "caspar" if ba_options.backend == BUNDLE_ADJUSTMENT_BACKEND["caspar"] else "ceres"


def apply_caspar_options(caspar: pycolmap.CasparBundleAdjustmentOptions, overrides: CasparOptions) -> None:
    """Set CASPAR solver knobs by name, casting each value to the attribute's own type.

    `CasparBundleAdjustmentOptions` binds `solver_iter_max` and `pcg_iter_max` as
    C++ ints, which pybind11 refuses to take a Python float for, so every value
    goes through the type the default carries. `gpu_index` is a string knob owned
    by `MappingOptions.use_gpu` and is rejected here.

    Args:
        caspar: The `BundleAdjustmentOptions.caspar` block, modified in place.
        overrides: Attribute name to value.

    Raises:
        KeyError: When a name is not a numeric CASPAR option.
    """
    for name, value in overrides.items():
        current: object = getattr(caspar, name, None)
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            raise KeyError(
                f"[colsfm] `{name}` is not a numeric CASPAR solver option; "
                f"the settable ones are {sorted(caspar_option_names())}"
            )
        setattr(caspar, name, type(current)(value))


def caspar_option_names() -> frozenset[str]:
    """The numeric option names `apply_caspar_options` accepts.

    Returns:
        Every key of a default `caspar.todict()` whose value is a number, i.e.
        everything but the string `gpu_index`.
    """
    defaults: dict[str, object] = pycolmap.BundleAdjustmentOptions().caspar.todict()
    return frozenset(name for name, value in defaults.items() if isinstance(value, (int, float)) and not isinstance(value, bool))


def bundle_adjustment_options(
    ba_config: BundleAdjustmentConfig,
    options: MappingOptions,
    reconstruction: pycolmap.Reconstruction | None = None,
) -> pycolmap.BundleAdjustmentOptions:
    """Bundle adjustment options mirroring cuSFM's `BundleAdjustmentConfig` (§9.2).

    Args:
        ba_config: The Ceres settings from `vision_mapping_config.pb.txt`.
        options: Command-line style knobs.
        reconstruction: The model about to be adjusted, read only for its camera
            models when `ba_backend` is `caspar`; see `resolve_ba_backend`.

    Returns:
        Options with every override of the module docstring applied, on the backend
        `resolve_ba_backend` allows.

    Raises:
        KeyError: When the configuration names a loss pycolmap does not have.
    """
    ba_options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = False
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = False
    ba_options.refine_rig_from_world = True
    ba_options.refine_sensor_from_rig = options.optimize_extrinsics
    ba_options.refine_points3D = True
    # pycolmap defaults this to True, which prints a full Ceres report per round.
    ba_options.print_summary = False

    num_threads: int = ba_config.num_threads if options.num_threads is None else options.num_threads
    ba_options.ceres.loss_function_type = LOSS_FUNCTION_BY_NAME[ba_config.loss_type]
    ba_options.ceres.loss_function_scale = ba_config.reprojection_error_standard_deviation * ba_config.loss_function_scale
    ba_options.ceres.use_gpu = options.use_gpu
    ba_options.ceres.auto_select_solver_type = False
    solver_options = ba_options.ceres.solver_options
    solver_options.linear_solver_type = COLMAP_LINEAR_SOLVER[MAPPING_LINEAR_SOLVER]
    solver_options.num_threads = num_threads
    solver_options.max_num_iterations = ba_config.max_num_iterations
    solver_options.max_num_consecutive_invalid_steps = ba_config.max_invalid_steps
    solver_options.max_linear_solver_iterations = ba_config.max_linear_solver_iterations
    solver_options.function_tolerance = CERES_FUNCTION_TOLERANCE
    solver_options.gradient_tolerance = CERES_GRADIENT_TOLERANCE
    solver_options.parameter_tolerance = CERES_PARAMETER_TOLERANCE
    solver_options.minimizer_progress_to_stdout = False

    backend: BaBackend = resolve_ba_backend(options, reconstruction)
    ba_options.backend = BUNDLE_ADJUSTMENT_BACKEND[backend]
    if backend == "caspar":
        # CASPAR throws on a multi-sensor frame whose extrinsics are free, and throws
        # again unless focal length and the distortion block agree; both are False here.
        ba_options.refine_sensor_from_rig = False
        ba_options.refine_extra_params = ba_options.refine_focal_length
        # `-1` is COLMAP's own pick, which is what `use_gpu` off means everywhere else
        # in this dataclass. Both were measured to solve on the 5090.
        ba_options.caspar.gpu_index = "0" if options.use_gpu else "-1"
        if options.caspar_options is not None:
            apply_caspar_options(ba_options.caspar, options.caspar_options)
    return ba_options


def ceres_polish_options(
    ba_config: BundleAdjustmentConfig,
    options: MappingOptions,
    reconstruction: pycolmap.Reconstruction | None = None,
) -> pycolmap.BundleAdjustmentOptions:
    """The same solve as the rounds, on Ceres: what polishes a finished CASPAR model.

    One config, one builder, one difference — the backend. `docs/caspar-build.md`
    § Ceres polish experiment measured the recovery with colsfm's own settings (the
    config's Cauchy loss at scale 4, 200 iterations, SPARSE_SCHUR, extrinsics and
    intrinsics fixed), so the polish has to be that solve rather than a fresh set of
    pycolmap defaults.

    Args:
        ba_config: The Ceres settings from `vision_mapping_config.pb.txt`.
        options: The run's knobs; only `ba_backend` is overridden.
        reconstruction: The model about to be adjusted; unread on the Ceres backend.

    Returns:
        Options selecting Ceres, otherwise identical to `bundle_adjustment_options`.
    """
    return bundle_adjustment_options(ba_config, replace(options, ba_backend="ceres"), reconstruction)


def load_correspondences(
    reconstruction: pycolmap.Reconstruction,
    database_path: Path,
    min_num_matches: int = 0,
) -> Correspondences:
    """Load keypoints and verified two-view geometries into the reconstruction.

    The database's keypoints become each image's `points2D` and its verified
    two-view geometries become the correspondence graph the triangulator walks.
    Nothing is re-estimated: the images already carry poses, and COLMAP's
    `DatabaseCache` is used only for the keypoints and the graph.

    Database image ids must equal the reconstruction's, because the graph is
    keyed by them. Both come from cuSFM's keyframe ids, so they agree by
    construction; a mismatch is a bug in whoever wrote the database and is
    reported rather than silently mis-triangulated.

    Args:
        reconstruction: A reconstruction with registered, pose-carrying frames.
        database_path: COLMAP database written by the feature and matching stages.
        min_num_matches: Drop image pairs with fewer inlier matches; cuSFM's
            `min_num_matches_per_pair`.

    Returns:
        The graph, the observation manager and the cache that owns them.

    Raises:
        FileNotFoundError: When the database does not exist.
        ValueError: When a database image name maps to a different id than the
            reconstruction's, or when the database holds none of its images.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    cache_options: pycolmap.DatabaseCacheOptions = pycolmap.DatabaseCacheOptions()
    cache_options.min_num_matches = min_num_matches
    cache_options.ignore_watermarks = False
    cache_options.image_names = {image.name for image in reconstruction.images.values()}
    with pycolmap.Database.open(database_path) as database:
        database_cache: pycolmap.DatabaseCache = pycolmap.DatabaseCache.create(database, cache_options)

    image_id_by_name: dict[str, int] = {image.name: image_id for image_id, image in reconstruction.images.items()}
    mismatched: list[str] = []
    num_loaded: int = 0
    for image_id, cached_image in database_cache.images.items():
        expected_image_id: int | None = image_id_by_name.get(cached_image.name)
        if expected_image_id is None:
            continue
        if expected_image_id != image_id:
            mismatched.append(f"{cached_image.name}: database id {image_id}, reconstruction id {expected_image_id}")
            continue
        reconstruction.image(image_id).points2D = cached_image.points2D
        num_loaded += 1
    if mismatched:
        raise ValueError(f"Database and reconstruction disagree on image ids: {mismatched[:5]}")
    if num_loaded == 0:
        raise ValueError(f"{database_path} holds none of the reconstruction's {reconstruction.num_images()} images")

    graph: pycolmap.CorrespondenceGraph = database_cache.correspondence_graph
    observation_manager: pycolmap.ObservationManager = pycolmap.ObservationManager(reconstruction, graph)
    return Correspondences(
        database_cache=database_cache,
        graph=graph,
        observation_manager=observation_manager,
        num_images=num_loaded,
        num_image_pairs=graph.num_image_pairs(),
    )


def registered_image_ids(reconstruction: pycolmap.Reconstruction) -> list[int]:
    """Image ids belonging to a registered frame, ascending.

    Args:
        reconstruction: The model.

    Returns:
        Sorted image ids.
    """
    image_ids: list[int] = []
    for frame_id in reconstruction.reg_frame_ids():
        for data_id in reconstruction.frame(frame_id).data_ids:
            if data_id.sensor_id.type == pycolmap.SensorType.CAMERA:
                image_ids.append(data_id.id)
    return sorted(image_ids)


def _triangulate(
    triangulator: pycolmap.IncrementalTriangulator,
    image_ids: Sequence[int],
    mapping_config: VisionMappingConfig,
    max_pixel_error: float,
    verbose: bool,
) -> int:
    """Run the inner triangulation loop until a pass adds nothing (§5.3).

    cuSFM repeats up to `iterative_triangulation_times` passes and exits early
    when a pass adds no point; observed 1 to 8 passes on Galileo.

    Args:
        triangulator: The triangulator, already bound to the reconstruction.
        image_ids: Registered images to triangulate, in ascending id order.
        mapping_config: The mapping configuration.
        max_pixel_error: The first round's gate, in pixels.
        verbose: Print one line per pass.

    Returns:
        Total observations triangulated.
    """
    options: pycolmap.IncrementalTriangulatorOptions = triangulator_options(mapping_config, max_pixel_error)
    total: int = 0
    for pass_index in range(max(mapping_config.iterative_triangulation_times, 1)):
        added: int = sum(triangulator.triangulate_image(options, image_id) for image_id in image_ids)
        total += added
        if verbose:
            print(f"[colsfm] triangulation pass {pass_index}: {added} observations")
        if added == 0:
            break
    return total


def filter_degenerate_points(
    observation_manager: pycolmap.ObservationManager,
    reconstruction: pycolmap.Reconstruction,
    depth_threshold: float,
) -> int:
    """Delete every point whose world position is non-finite or out of bounds.

    A point goes when any coordinate is NaN or infinite, or when the largest
    coordinate magnitude exceeds `depth_threshold`. That is deliberately stricter
    than cuSFM, whose only spatial filter is a one-sided cap on the world Z axis
    (keypoints_mapper_main.md §5.5); a bound on all three axes and both signs
    subsumes it, and cuSFM's asymmetry looks like an oversight rather than a
    design. On Galileo the difference is nothing — the guard removes 0 points in
    all five rounds.

    Two things make the guard necessary, and neither is caught by COLMAP's own
    filters:

    1. **NaN is invisible to every threshold test.** They all ask
       `error > threshold`, and any comparison against NaN is false, so a NaN
       point survives filtering and then poisons the whole normal equation.
    2. **A converged solve can still throw a point to infinity.** Measured on 200
       RoboCap rig frames: round 0's bundle adjustment reports CONVERGENCE and
       halves its cost while pushing 5 of 31 017 points out to 4941 m. Under the
       fisheye model those points project next to the projection singularity, so
       their reprojection error comes out at 4.5e153 px — which is where the
       round's reported mean of 1.4e149 px came from. The pre-solve filter alone
       cannot help: the model going *into* that solve was clean (max error
       22.4 px). The guard therefore runs after the solve as well as before it.

    Args:
        observation_manager: Bookkeeping bound to the reconstruction.
        reconstruction: The model to filter, edited in place.
        depth_threshold: `vision_mapping_config.depth_threshold`, in metres.

    Returns:
        Observations removed, i.e. the summed track length of the deleted points.
    """
    point3D_ids: list[int] = list(reconstruction.points3D)
    if not point3D_ids:
        return 0
    points_xyz: Float64[ndarray, "num_points 3"] = np.array(
        [reconstruction.point3D(point3D_id).xyz for point3D_id in point3D_ids], dtype=np.float64
    )
    # `np.abs(nan) > threshold` is False, so the finiteness term has to carry the NaNs.
    out_of_bounds: Bool[ndarray, "num_points"] = ~np.isfinite(points_xyz).all(axis=1) | (
        np.abs(points_xyz).max(axis=1) > depth_threshold
    )
    removed: int = 0
    for point3D_id in np.asarray(point3D_ids)[out_of_bounds].tolist():
        removed += reconstruction.point3D(point3D_id).track.length()
        observation_manager.delete_point3D(point3D_id)
    return removed


def projection_sanity_bound_px(reconstruction: pycolmap.Reconstruction) -> float:
    """The largest reprojection error that can still be a measurement.

    Args:
        reconstruction: The model, for its cameras' pixel dimensions.

    Returns:
        The longest image side over every camera, or 0.0 for a model without
        cameras. An observation further than a whole image from its point is not
        a bad measurement, it is a failed projection.
    """
    sides: list[float] = [float(max(camera.width, camera.height)) for camera in reconstruction.cameras.values()]
    return max(sides) if sides else 0.0


def filter_projection_failures(
    observation_manager: pycolmap.ObservationManager, reconstruction: pycolmap.Reconstruction
) -> int:
    """Drop observations a solve has made unprojectable, at a numeric-sanity bound.

    Not a quality filter — the round's own gate (25 down to 5 px) is far tighter
    and owns that job. This only catches projection *failures*, at
    `projection_sanity_bound_px`, and it exists because bundle adjustment can
    produce them out of a clean model:

    On RoboCap's 4528 fisheye images, 4 points of 209 905 end up within
    5-90 mm of a camera centre — one of them at **negative** depth. At that
    distance `OPENCV_FISHEYE` projects them thousands of image widths away and
    their reprojection error comes out at 4.5e153 px, which is where the
    `1e152 px` mean and the CHOLMOD "not positive definite" warnings in the first
    three rounds came from. The model going *into* each of those solves was clean
    (max 23.5 px), so no pre-solve filter can prevent it.

    Deleting them now rather than in the next round changes no result — the next
    gate is at most 25 px, so anything above a whole image width was already
    doomed — but it keeps the round's reported metric meaningful and keeps the
    next solve from linearising the same singularity.

    Args:
        observation_manager: Bookkeeping bound to the reconstruction.
        reconstruction: The model to filter, edited in place.

    Returns:
        Observations removed.
    """
    bound_px: float = projection_sanity_bound_px(reconstruction)
    if bound_px <= 0.0:
        return 0
    # `filter_points3D` covers reprojection error, cheirality and triangulation angle;
    # a zero angle threshold switches the last of the three off, leaving the two that
    # describe a failed projection.
    return observation_manager.filter_points3D(bound_px, 0.0, set(reconstruction.points3D))


def _filter_points(
    observation_manager: pycolmap.ObservationManager,
    reconstruction: pycolmap.Reconstruction,
    mapping_config: VisionMappingConfig,
    max_pixel_error: float,
) -> _FilterCounts:
    """Apply cuSFM's point filters for one round, before its bundle adjustment (§5.4-§5.5).

    The reprojection gate decays per round; the triangulation angle is the max
    over observing pairs, matching cuSFM's `HasLargeEnoughTriangulateAngle`; the
    depth test is a **world Z** cap, one axis and one-sided, not a camera depth.
    `filter_degenerate_points` adds the non-finite and absurd-coordinate guard
    cuSFM has no equivalent of; see its docstring.

    Args:
        observation_manager: Bookkeeping bound to the reconstruction.
        reconstruction: The model to filter.
        mapping_config: The mapping configuration.
        max_pixel_error: The round's gate, in pixels.

    Returns:
        Observations removed, split by cause.
    """
    reprojection_and_angle: int = observation_manager.filter_all_points3D(
        max_pixel_error, mapping_config.min_triangulation_deg
    )
    negative_depth: int = observation_manager.filter_observations_with_negative_depth()
    short_track: int = observation_manager.filter_points3D_with_short_tracks(mapping_config.min_correspondences)
    degenerate: int = filter_degenerate_points(observation_manager, reconstruction, mapping_config.depth_threshold)
    return _FilterCounts(
        reprojection_and_angle=reprojection_and_angle,
        negative_depth=negative_depth,
        short_track=short_track,
        degenerate=degenerate,
    )


def solve_bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    ba_options: pycolmap.BundleAdjustmentOptions,
    gauge_frame_id: int,
    fixed_camera_params_id: int | None,
) -> tuple[pycolmap.BundleAdjustmentSummary, float]:
    """Run one global bundle adjustment over every registered image.

    The gauge is one constant rig frame — cuSFM fixes a single keyframe, which
    through the shared rig fixes its whole rig pose (§6.3). COLMAP's own
    `fix_gauge` is deliberately not used: it would pick its own two frames and
    the resulting trajectory would no longer be anchored where cuSFM anchors it.

    A pycolmap built without `CASPAR_ENABLED` says so only when the adjuster is
    built, which is where the capability check therefore lives: the first
    construction that raises it switches `ba_options` back to Ceres — in place, so
    every later round follows without repeating the message — and builds again.
    Nothing else is retried; any other `ValueError` propagates.

    Args:
        reconstruction: The model to adjust, in place.
        ba_options: Solver and refinement options.
        gauge_frame_id: Frame whose `rig_from_world` stays constant.
        fixed_camera_params_id: Camera whose `sensor_from_rig` stays constant while
            the others are refined, or None to leave every extrinsic as the options
            say. Ignored when it names the rig's reference sensor, which COLMAP
            already fixes to identity.

    Returns:
        The solver summary and the seconds `solve()` took.
    """
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in registered_image_ids(reconstruction):
        config.add_image(image_id)
    config.set_constant_rig_from_world_pose(gauge_frame_id)
    if fixed_camera_params_id is not None:
        fixed_sensor: pycolmap.sensor_t = camera_sensor_id(fixed_camera_params_id)
        if not reconstruction.rig(RIG_ID).is_ref_sensor(fixed_sensor):
            config.set_constant_sensor_from_rig_pose(fixed_sensor)
    try:
        adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(ba_options, config, reconstruction)
    except ValueError as error:
        if CASPAR_DISABLED_MARKER not in str(error):
            raise
        print(
            "[colsfm] this pycolmap is built without CASPAR_ENABLED, so the GPU backend "
            "cannot solve; falling back to Ceres for the rest of the run "
            "(the `colsfm-caspar` environment has the build that can)"
        )
        ba_options.backend = BUNDLE_ADJUSTMENT_BACKEND["ceres"]
        adjuster = pycolmap.create_default_bundle_adjuster(ba_options, config, reconstruction)
    started: float = time.perf_counter()
    summary: pycolmap.BundleAdjustmentSummary = adjuster.solve()
    return summary, time.perf_counter() - started


def _parse_brief_report(summary: pycolmap.BundleAdjustmentSummary) -> tuple[int, float, float]:
    """Read iterations and costs out of Ceres' one-line report.

    `BundleAdjustmentSummary` exposes `termination_type`, `num_residuals` and
    `is_solution_usable`, but the iteration count and the costs only appear in
    `brief_report()`.

    Args:
        summary: The solver summary.

    Returns:
        Iterations, initial cost and final cost; zeros when the report does not parse.
    """
    match: re.Match[str] | None = BRIEF_REPORT_PATTERN.search(summary.brief_report())
    if match is None:
        return 0, 0.0, 0.0
    return int(match.group(1)), float(match.group(2)), float(match.group(3))


def _check_extrinsics_are_refinable(model: PosedModel) -> None:
    """Refuse an extrinsic refinement COLMAP would silently turn into a no-op.

    COLMAP's `ParameterizeRigsAndFrames` fixes every non-reference
    `sensor_from_rig` of a rig whose reference sensor is not itself among the
    parameterised sensors — "otherwise, the relative pose between the sensors is
    not well constrained". A vehicle-body reference owns no images, so it never
    is, and `refine_sensor_from_rig = True` then moves nothing at all.

    A `PosedModel` carries the reference it was built with, so this is the only
    condition left to test: the second half of the old guard — a caller who had a
    reconstruction but no `rig_reference` to invert the bookkeeping with — cannot
    be expressed any more.

    Args:
        model: The model about to be adjusted.

    Raises:
        ValueError: When the rig's reference sensor is not a camera.
    """
    if model.reference.camera_params_id is None:
        raise ValueError(
            "Cannot refine rig extrinsics: this rig's reference sensor is "
            f"{model.reconstruction.rig(RIG_ID).ref_sensor_id}, which owns no images. "
            "COLMAP holds every `sensor_from_rig` constant when the reference sensor is "
            "not part of the problem, so the refinement would be a silent no-op. Build "
            "the reconstruction with "
            "`build_reconstruction(..., reference_camera_params_id=...)` to put the rig "
            "origin on a camera."
        )


def _polish_with_ceres(
    reconstruction: pycolmap.Reconstruction,
    ba_config: BundleAdjustmentConfig,
    options: MappingOptions,
    gauge_frame_id: int,
    fixed_camera_params_id: int | None,
) -> PolishStats:
    """One Ceres global bundle adjustment over a finished CASPAR model.

    The whole step: the same gauge frame, the same observation set, no
    re-triangulation, no merge, no complete, no filter — so the reprojection error
    the round loop reported and the one this returns describe the same points, and
    every difference between them is the solve's. `docs/caspar-build.md`
    § Shipped: CASPAR + Ceres polish measured 22 iterations and 9.8 s on KITTI 06's
    2156 images, which recovers the 0.4 m of trajectory CASPAR left on the table.

    Args:
        reconstruction: The model the rounds left behind, adjusted in place.
        ba_config: The Ceres settings from `vision_mapping_config.pb.txt`.
        options: The run's knobs; the backend is forced to Ceres.
        gauge_frame_id: The same frame the rounds held constant.
        fixed_camera_params_id: The same camera the rounds held constant, or None.

    Returns:
        What the solve cost and how far it moved the reprojection error.
    """
    polish_options: pycolmap.BundleAdjustmentOptions = ceres_polish_options(ba_config, options, reconstruction)
    before_px: float = reconstruction.compute_mean_reprojection_error()
    summary, solve_seconds = solve_bundle_adjustment(reconstruction, polish_options, gauge_frame_id, fixed_camera_params_id)
    reconstruction.update_point_3d_errors()
    num_iterations, initial_cost, final_cost = _parse_brief_report(summary)
    stats: PolishStats = PolishStats(
        num_observations=reconstruction.compute_num_observations(),
        ba_num_iterations=num_iterations,
        ba_initial_cost=initial_cost,
        ba_final_cost=final_cost,
        ba_termination=summary.termination_type.name,
        mean_reprojection_error_before_px=before_px,
        mean_reprojection_error_after_px=reconstruction.compute_mean_reprojection_error(),
        seconds=solve_seconds,
    )
    if options.verbose:
        cost_cut: float = 1.0 - stats.ba_final_cost / stats.ba_initial_cost if stats.ba_initial_cost else 0.0
        print(
            f"[colsfm] ceres polish: {stats.ba_num_iterations} iterations, cost {stats.ba_initial_cost:.6g} -> "
            f"{stats.ba_final_cost:.6g} ({cost_cut:.2%}), reprojection {stats.mean_reprojection_error_before_px:.4f} -> "
            f"{stats.mean_reprojection_error_after_px:.4f} px in {stats.seconds:.1f} s ({stats.ba_termination})"
        )
    return stats


def run_mapping(
    model: PosedModel,
    database_path: Path,
    mapping_config: VisionMappingConfig,
    options: MappingOptions | None = None,
    gauge_frame_id: int | None = None,
) -> MappingResult:
    """Triangulate and bundle-adjust, cuSFM's `keypoints_mapper_main` §5.3-§5.6.

    ```
    load correspondences
    triangulate every registered image, repeating until a pass adds nothing
    for k in 0 .. num_ba_iterations - 1:
        merge -> complete -> filter at gate g_k -> global bundle adjustment -> guards
        stop early when (merged + completed + filtered) / observations < max_observation_change
    one Ceres global bundle adjustment, when the rounds ran on CASPAR
    ```

    The trailing guards are the one step cuSFM has no equivalent of:
    `filter_degenerate_points` and `filter_projection_failures` between them
    remove what a converged solve can still leave behind — a point outside the
    world, or one inside a camera. Both docstrings carry the measurements.

    The early exit is the configuration's own: set
    `mapping_config.max_observation_change` to 0.0 to run every round, since the
    statistic is never negative. cuSFM logs `use num_ba_iterations to stop BA`, so
    its two stopping rules are exclusive.

    Args:
        model: A reconstruction with registered, pose-carrying frames and the rig
            reference it was built with. The reconstruction is adjusted in place.
        database_path: COLMAP database with keypoints and verified two-view geometries.
        mapping_config: `vision_mapping_config.pb.txt`; its `bundle_adjustment` block
            is what the Ceres solve is configured from.
        options: Command-line style knobs; the defaults when None.
        gauge_frame_id: Frame held constant for gauge; the frame owning the
            lowest image id when None, which is cuSFM's own choice (§6.3).

    The closing solve is the polish of `docs/caspar-build.md`
    § Shipped: CASPAR + Ceres polish: CASPAR stops while a few percent of the cost is
    still reachable, so a CASPAR run ends on one Ceres bundle adjustment over the
    final model, with no filtering around it. It is skipped whenever the solves ran
    on Ceres — every fallback included — because Ceres' own answer is a fixed point
    of it (one iteration, no cost change, 25 femtometres of rig motion, measured).

    Returns:
        The adjusted reconstruction with per-round statistics and timings, the
        closing `polish` when one ran, plus `refined_extrinsics` when extrinsics
        were refined.

    Raises:
        FileNotFoundError: When the database does not exist.
        ValueError: When the database and the reconstruction disagree on image
            ids, when no frame is registered, or when extrinsic refinement is
            asked for on a rig COLMAP would silently freeze.
    """
    resolved_options: MappingOptions = MappingOptions() if options is None else options
    reconstruction: pycolmap.Reconstruction = model.reconstruction
    ba_config: BundleAdjustmentConfig = mapping_config.bundle_adjustment
    started: float = time.perf_counter()
    image_ids: list[int] = registered_image_ids(reconstruction)
    if not image_ids:
        raise ValueError("The reconstruction has no registered frames to map")
    resolved_gauge_frame_id: int = (
        reconstruction.image(min(image_ids)).frame_id if gauge_frame_id is None else gauge_frame_id
    )
    fixed_camera_params_id: int | None = None
    if resolved_options.optimize_extrinsics:
        _check_extrinsics_are_refinable(model)
        # cuSFM's constant keyframe is the lowest id in the map (§6.3), and
        # `--fixed_camera_name` pins that keyframe's camera when extrinsics move.
        fixed_camera_params_id = (
            reconstruction.image(min(image_ids)).camera_id
            if resolved_options.fixed_camera_params_id is None
            else resolved_options.fixed_camera_params_id
        )

    correspondence_started: float = time.perf_counter()
    correspondences: Correspondences = load_correspondences(
        reconstruction, database_path, mapping_config.min_num_matches_per_pair
    )
    correspondence_seconds: float = time.perf_counter() - correspondence_started

    triangulator: pycolmap.IncrementalTriangulator = pycolmap.IncrementalTriangulator(
        correspondences.graph, reconstruction, correspondences.observation_manager
    )
    schedule: tuple[float, ...] = pixel_error_schedule(mapping_config)
    triangulation_started: float = time.perf_counter()
    _triangulate(triangulator, image_ids, mapping_config, schedule[0], resolved_options.verbose)
    triangulation_seconds: float = time.perf_counter() - triangulation_started
    # Round 0's gate would drop these anyway; doing it here only makes the line below
    # mean something. Without it RoboCap's triangulation reports 8.9e152 px, because a
    # handful of tracks resolve to a point sitting inside one of their own cameras.
    filter_projection_failures(correspondences.observation_manager, reconstruction)
    reconstruction.update_point_3d_errors()
    if resolved_options.verbose:
        print(
            f"[colsfm] triangulation finished: {reconstruction.num_points3D()} points, "
            f"reprojection {reconstruction.compute_mean_reprojection_error():.4f} px, "
            f"mean length {reconstruction.compute_mean_track_length():.4f}"
        )

    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(ba_config, resolved_options, reconstruction)
    rounds: list[RoundStats] = []
    bundle_adjustment_seconds: float = 0.0
    for round_index, max_pixel_error in enumerate(schedule):
        round_started: float = time.perf_counter()
        round_options: pycolmap.IncrementalTriangulatorOptions = triangulator_options(mapping_config, max_pixel_error)
        num_merged: int = triangulator.merge_all_tracks(round_options)
        num_completed: int = triangulator.complete_all_tracks(round_options)
        filtered: _FilterCounts = _filter_points(
            correspondences.observation_manager, reconstruction, mapping_config, max_pixel_error
        )
        num_observations: int = reconstruction.compute_num_observations()
        observation_change: float = (
            (num_merged + num_completed + filtered.total()) / num_observations if num_observations else 0.0
        )

        summary, solve_seconds = solve_bundle_adjustment(reconstruction, ba_options, resolved_gauge_frame_id, fixed_camera_params_id)
        bundle_adjustment_seconds += solve_seconds
        # The guards run on both sides of the solve. Before it, so Ceres never linearises a
        # NaN; after it, because a *converged* solve can still walk a weakly-constrained
        # point out of the world (`filter_degenerate_points`) or into a camera centre
        # (`filter_projection_failures`), and the next round would otherwise inherit both.
        num_diverged: int = filter_degenerate_points(
            correspondences.observation_manager, reconstruction, mapping_config.depth_threshold
        ) + filter_projection_failures(correspondences.observation_manager, reconstruction)
        reconstruction.update_point_3d_errors()
        num_iterations, initial_cost, final_cost = _parse_brief_report(summary)
        rounds.append(
            RoundStats(
                round_index=round_index,
                max_pixel_error=max_pixel_error,
                num_merged=num_merged,
                num_completed=num_completed,
                num_filtered=filtered.total(),
                num_diverged=num_diverged,
                num_observations=num_observations,
                observation_change=observation_change,
                num_points3D=reconstruction.num_points3D(),
                mean_reprojection_error_px=reconstruction.compute_mean_reprojection_error(),
                ba_num_iterations=num_iterations,
                ba_initial_cost=initial_cost,
                ba_final_cost=final_cost,
                ba_termination=summary.termination_type.name,
                seconds=time.perf_counter() - round_started,
            )
        )
        latest: RoundStats = rounds[-1]
        if resolved_options.round_callback is not None:
            resolved_options.round_callback(latest, reconstruction)
        if resolved_options.verbose:
            print(
                f"[colsfm] round {round_index} gate {max_pixel_error:.1f} px: merged {num_merged}, "
                f"completed {num_completed}, filtered {filtered.total()}, diverged {num_diverged}, "
                f"change {observation_change:.6f}, points {latest.num_points3D}, "
                f"reprojection {latest.mean_reprojection_error_px:.4f} px"
            )
        if observation_change < mapping_config.max_observation_change:
            if resolved_options.verbose:
                print(f"[colsfm] observation change {observation_change:.6f} below {mapping_config.max_observation_change}")
            break

    # The rounds are done; CASPAR's answer is not yet a stationary point of the
    # objective, so one Ceres solve finishes it. Nothing filters around it.
    polish: PolishStats | None = None
    if resolved_options.caspar_ceres_polish and backend_name(ba_options) == "caspar":
        polish = _polish_with_ceres(
            reconstruction, ba_config, resolved_options, resolved_gauge_frame_id, fixed_camera_params_id
        )
        bundle_adjustment_seconds += polish.seconds

    return MappingResult(
        reconstruction=reconstruction,
        reference=model.reference,
        rounds=tuple(rounds),
        correspondence_seconds=correspondence_seconds,
        triangulation_seconds=triangulation_seconds,
        bundle_adjustment_seconds=bundle_adjustment_seconds,
        polish=polish,
        total_seconds=time.perf_counter() - started,
        extrinsics_refined=resolved_options.optimize_extrinsics,
        # Read off the options the solves ran with, not off the request: the
        # capability fallback rewrites them in place at the first solve.
        ba_backend=backend_name(ba_options),
    )

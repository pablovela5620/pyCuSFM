"""Property test: CASPAR's OPENCV_FISHEYE adapter agrees with Ceres on random rigs.

`docs/caspar-fisheye-adapter.md` measures the local `OpenCVFisheyeAdapter` on
*one* synthetic rig (§5.2: four cameras, six frames, 147 points, one hand-picked
`k1..k4`) and on *one* real model (§5.3, Galileo).  Both pass, but both are
single points in a large space, and the interesting failure modes of a
Symforce-generated GPU kernel -- an uninitialised accumulator that only bites at
some node count, a `theta/r` limit that only bites at some distortion -- are
exactly the ones a single scene can miss.  What the adapter is supposed to
satisfy is a statement quantified over rigs:

    for every well-posed OPENCV_FISHEYE rig, CASPAR from a given initialisation
    lands within the documented float32 shortfall of where Ceres lands, on the
    same residuals, and one Ceres polish closes the remaining gap.

The module holds three tests. `test_caspar_matches_ceres_on_random_fisheye_rigs`
is that property. `test_caspar_score_is_stable_across_the_thread_block_boundary`
is a deterministic repeat check at 1023 / 1024 / 1025 observations, aimed at the
uninitialised padding-thread read the generated score kernels still carry; it is
currently a negative result, and says so.
`test_points_behind_the_cameras_never_reach_the_solvers` fixes the domain: the
two backends disagree about non-positive depth, so no scene here may contain it.

## What "the same problem" means here

The comparison is void unless both backends minimise the same function over the
same variables from the same start, so all of the following are pinned rather
than left to defaults:

- **Loss.**  Both are given `LossFunctionType.TRIVIAL`.  CASPAR solves plain
  least squares whatever loss is requested, so a robust loss on the Ceres side
  would silently turn this into a comparison of loss functions.  (The pipeline
  itself asks for Cauchy; that difference belongs to a pipeline test, not here.)
- **Gauge.**  One rig frame is held constant, intrinsics are constant, and
  `refine_sensor_from_rig` is false.  Rotation and translation are fixed by the
  frame; *scale* is fixed by the rig, which is why every drawn rig has at least
  two cameras with known extrinsics.  A single fixed monocular pose would leave
  scale free and the two answers could differ by a similarity without either
  being wrong.
- **Coverage.**  `measure_coverage` recounts the eligible observations from the
  reconstruction and both summaries must report exactly twice that many
  residuals.  Equality between the two backends alone would not catch a build in
  which both dropped the same images.
- **Domain.**  Every observation must project to a finite pixel at positive
  depth, checked before any error is averaged.  There is no `nansum` anywhere in
  this path: a projection that fails must break the test, not vanish from it.

## Why Hypothesis, and where it stops paying

Every example is three GPU bundle adjustments, so the budget is `max_examples`
in the low tens rather than the hundreds Hypothesis normally wants.  At that
size Hypothesis is *not* buying search: eight draws will not find a rare corner
that eight hand-written `@pytest.mark.parametrize` cases would miss.  Three
other things do pay, and they are why this is a Hypothesis test:

1. **Shrinking.**  A failure on a 4-camera / 30-frame / 500-point rig is close
   to undebuggable.  Hypothesis reports the *minimal* rig that still fails --
   typically 2 cameras, 5 frames, the smallest distortion -- which is a bug
   report rather than a data dump.  Shrinking costs many more GPU solves, but
   only on the run that already failed.
2. **The example database.**  A rig that fails once is replayed first on every
   later run, so a flaky-looking float32 boundary cannot be lost by re-running.
3. **The strategy is the specification.**  `fisheye_problems` states, in
   executable form, the input domain the adapter claims to cover: 2-4 cameras,
   5-30 cm baselines, unequal `fx`/`fy`, an off-centre principal point,
   independently drawn `k1..k4` of realistic magnitude, observations out to
   45 deg and beyond off-axis, sub-pixel noise.  A parametrised list states only
   the points somebody happened to think of.

The dividend was immediate and worth recording. On the second run Hypothesis
shrank a failure to a rig sliding in a *straight line*, which is the one
trajectory for which fitting a rigid alignment to the rig centres alone is
under-determined -- a bug in this module's own comparison, reporting 149 deg of
rotation disagreement on two answers that agreed to 0.6 um.  On the third it
walked to the 2-camera / 5 cm / 5-frame / 0.5 px corner and exposed a real
property of the family, recorded under `POSITION_RMSE_FRACTION_OF_EXTENT`.
Neither is a case a hand-written list would have contained.

Two concessions the small budget forces, recorded so the next reader does not
mistake them for oversights:

- **Only the structural knobs are drawn.**  Counts, magnitudes and ranges come
  from Hypothesis; the 100-500 point positions and the per-camera extrinsics are
  filled in by `numpy.random.default_rng(seed)` from a drawn seed.  Handing
  Hypothesis 1500 floats to shrink would spend the whole budget on noise that
  cannot be a cause.
- **Well-posedness is constructed, not filtered.**  The strategy places points
  in a box the rig actually looks at and scales the trajectory to its length, so
  no draw has to be rejected.  The one `assume` left is a tripwire, not a filter:
  if it ever fires, the generator is wrong.

## Tolerances

`docs/caspar-build.md` and `docs/caspar-fisheye-adapter.md` §9.4/§9.5 both
measure the same thing on real rigs: CASPAR's float32 solve stops ~4 % short of
Ceres in mean reprojection error, and one Ceres round closes it.  The bounds
below were then measured on this generator (see the module's git history and the
report accompanying it) and set with headroom above the worst example seen.

The reprojection bound is relative *plus* a small absolute floor because
`pixel_noise_px` may be drawn as zero: on a noiseless scene Ceres reaches ~1e-8
px, CASPAR's float32 residual floor sits near 1e-4 px, and no relative bound can
straddle both.  The floor is that fp32 floor, not slack.

The geometric bounds come in two strengths, and the gap between them is the
module's main finding.  The *polished* answer meets tight, scale-relative targets
(1e-4 of the trajectory extent in position, 0.002 deg in rotation) at every
conditioning drawn, subject to a floor at Ceres' own stopping precision.  The
*raw* fp32 answer does not: on the badly conditioned corner it sits 2e-2 of
extent out, two hundred times that target, while reaching the same cost as Ceres
to five decimal places.  Since `colsfm` runs the polish by
default (`MappingOptions.caspar_ceres_polish`), the tight bound is the one that
describes what the pipeline delivers; the loose one describes what the GPU
solver alone delivers, and it is stated as a measurement rather than hidden.

Skips cleanly when this pycolmap has no CASPAR at all (the plain `colsfm`
environment) or has CASPAR without the fisheye adapter (`colsfm-caspar`,
`colsfm-caspar64`), using the same `num_residuals` capability probe the pipeline
itself uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pycolmap
import pytest
from hypothesis import HealthCheck, assume, given, note, settings
from hypothesis import strategies as st
from jaxtyping import Bool, Float
from numpy import ndarray

from colsfm.mapping import caspar_supported_camera_models
from tools.caspar_fisheye_probe import (
    FisheyeScenePlan,
    ProbeConfig,
    SolveResult,
    build_scene_from_plan,
    bundle_adjustment_config,
    drop_degenerate_points,
    fisheye_camera,
    perturb_model,
    solve_with_backend,
)

FISHEYE_MODEL: Final[str] = "OPENCV_FISHEYE"
"""The camera model this suite is about; a `pycolmap.CameraModelId` member name."""

CASPAR_MODELS: Final[frozenset[str]] = caspar_supported_camera_models()
"""What this build's CASPAR actually projects, measured once at import.

Empty means pycolmap was built without `CASPAR_ENABLED`; `{PINHOLE,
SIMPLE_RADIAL}` means stock CASPAR. Costs five sub-millisecond GPU solves."""

GPU_INDEX: Final[str] = "-1"
"""CUDA device CASPAR runs on; `-1` lets COLMAP pick, as `colsfm.mapping` does."""

MAX_EXAMPLES: Final[int] = 12
"""Rigs drawn per run. Three GPU solves each, so this is minutes, not seconds."""

REPROJECTION_RELATIVE_TOLERANCE: Final[float] = 0.05
"""How far above Ceres CASPAR's mean reprojection error may land, as a fraction.

5 % against the 4 % that `docs/caspar-fisheye-adapter.md` §9.4 and §9.5 measured
on the 600-image and 4528-image RoboCap runs, and against the 0.08 % worst case
this generator produced (a noiseless draw where Ceres reached 0.0013 px)."""

REPROJECTION_FLOOR_PX: Final[float] = 0.001
"""Absolute slack added to the relative bound, in pixels: CASPAR's float32 floor.

Needed because a draw may set `pixel_noise_px = 0`, where Ceres converges to ~1e-8
px and no relative bound could hold. §5.2 of the adapter document measures the
floor at 1.4e-4 px on its own rig; the worst this generator produced is 4.3e-5 px
absolute on a noiseless draw and 1.7e-5 px of |CASPAR - Ceres| on a noisy one, so
this is that floor with a factor of twenty of room. It is deliberately not the
0.02 px the document's *large* real-model cases need: this family is small and
normalised, and a floor sized for RoboCap would assert nothing here."""

POSITION_RMSE_FLOOR_M: Final[float] = 0.001
"""Constant part of the raw-CASPAR rig-centre bound: one millimetre."""

POSITION_RMSE_FRACTION_OF_EXTENT: Final[float] = 0.05
"""Scaled part of the raw-CASPAR rig-centre bound, as a fraction of the trajectory.

This is a sanity bound, and saying so is more useful than pretending otherwise.
Over the drawn family it is loose by a factor of several hundred on a
well-conditioned rig (0.08 mm measured against a 44 mm bound) and by a factor of
two and a half on the corner Hypothesis walks to. That spread is the family's,
not the bound's, and it cannot be closed by one scalar.

The corner -- two cameras, the 5 cm minimum baseline, five frames, 100 points,
the full 0.5 px of noise, and motion pointed at the point cloud -- is well posed
but very badly conditioned, and its number is *predicted* rather than merely
tolerated. A point at 3.4 m seen through a 5 cm baseline at half a pixel has a
depth uncertainty of order Z^2 sigma / (f b) = 0.36 m; averaged over 100 points
that is ~36 mm of rig position along the view axis, and the eight measured
repeats of that corner land at 18.5-41.9 mm with a CASPAR-to-Ceres reprojection
ratio of 1.0000. Both solvers are sitting inside the data's own noise floor at
the same cost. 5 % of a 2 m arc is 101 mm, which covers it with room for the tail.

The tight statement is `POLISHED_POSITION_FRACTION_OF_EXTENT` below, which holds
at every conditioning measured, and the two tight statements about the raw answer
are the reprojection bound and `ROTATION_TOLERANCE_DEG` -- neither of which the
depth valley touches."""

ROTATION_TOLERANCE_DEG: Final[float] = 0.05
"""Largest raw-CASPAR rig-rotation disagreement allowed, in degrees.

Rotation is not affected by the depth valley above: the worst measured over the
sweep, that corner included, is 0.0030 deg, so this is a factor of 17 of room."""

POLISH_RELATIVE_TOLERANCE: Final[float] = 0.002
"""How far the polished CASPAR answer may sit from Ceres in reprojection error.

Both are Ceres solves of the same problem from different starts, so they should
reach the same local minimum; this is 0.2 %, twenty-five times tighter than the
raw CASPAR bound, and is the assertion that the shipped polish does its job. The
measured ratio is 1.000000 on every rig drawn so far."""

POLISH_FLOOR_PX: Final[float] = 1e-4
"""Absolute slack on the polish bound, in pixels, for the noiseless draws."""

POLISHED_POSITION_FRACTION_OF_EXTENT: Final[float] = 1e-4
"""Rig-centre RMSE allowed between the *polished* CASPAR answer and Ceres.

As a fraction of the trajectory extent: 0.2 mm on a 2 m arc. This is the
review's stated target for a well-conditioned scene, and it holds here at every
conditioning -- including the corner where raw CASPAR sits 42 mm out, i.e. 2e-2
of extent, two hundred times the target. Measured worst polished value over the
sweep: 0.0016 mm, which is 1e-6 of extent. That contrast is the finding: the
target is met by what the pipeline ships (`MappingOptions.caspar_ceres_polish`
is on by default), not by the raw fp32 answer."""

POLISHED_POSITION_FLOOR_M: Final[float] = 3e-4
"""Constant part of the polished rig-centre bound: Ceres' own stopping precision.

Not padding, and not the value this started at. The first version used 1e-5 m,
and Hypothesis failed it on a *noiseless*, weakly conditioned draw -- two
cameras, 5 cm baseline, five frames, zero distortion, zero pixel noise -- where
the polish landed 0.069 mm from Ceres on a 0.5 m arc, i.e. 1.4e-4 of extent
against a 1.2e-4 bound. Reading the rest of that example explains it: Ceres
stopped at 3.4e-6 px and the polish at 2.6e-6 px, a reprojection *ratio of
0.869*. The polish was the better of the two answers; the disagreement is where
the reference stopped, not where the polish did. Below about 1e-5 px of cost
there is nothing left to steer either solver, and 0.07 mm is how wide that basin
is on this family.

So the floor is 0.3 mm, four times the observed worst, and it is what makes the
1e-4-of-extent target meaningful only where the scene is large enough for it to
exceed this: on a 2 m arc the bound is 0.5 mm against a measured 0.0016 mm.
`test_caspar_matches_ceres_on_random_fisheye_rigs` also asserts that the polish
never moves the answer *further* from Ceres than raw CASPAR was, which is the
same claim in a form the basin cannot flatten."""

POLISHED_ROTATION_DEG: Final[float] = 0.002
"""Rig-rotation disagreement allowed after the polish, in degrees.

The review's target. Measured worst over the sweep: 2e-6 deg."""

MIN_TRACKS: Final[int] = 50
"""Tracks a drawn rig must produce. A tripwire on the generator, not a filter."""

MIN_INITIAL_DEPTH_M: Final[float] = 0.5
"""Depth margin every observation of the *perturbed* start must keep, in metres.

Half a metre against a point box that starts at 2.5 m and a perturbation of at
most 5 cm and 0.5 deg, so the prune touches only the grazing observations the
wide box exists to create. See `_initial_state` for why it has to exist at
all."""

MIN_PERIPHERAL_THETA_DEG: Final[float] = 45.0
"""Incidence angle the widest observation must reach, or the rig proves nothing.

`k3` and `k4` multiply theta^6 and theta^8. A scene whose observations all sit
near the optical axis exercises the adapter's polynomial at its least
interesting point and would pass whatever those two coefficients were. This is a
coverage tripwire on the generator, asserted per example."""

THREAD_BLOCK_OBSERVATIONS: Final[tuple[int, int, int]] = (1023, 1024, 1025)
"""Observation counts straddling CASPAR's 1024-thread block, for the padding check."""

THREAD_BLOCK_REPEATS: Final[int] = 5
"""Times each of those counts is re-solved from identical state."""

THREAD_BLOCK_SCORE_TOLERANCE: Final[float] = 1e-5
"""Relative spread allowed across repeats of one identical problem.

Measured on the 5090: 1e-8 at 1023 observations, 2.5e-7 at 1024 and 2.7e-7 at
1025. That the full block -- where a padding thread cannot exist -- carries the
same spread as the ragged sizes is what says this is reduction-order
nondeterminism in a float32 sum, not the uninitialised read. The bound is 1e-5,
about forty times the worst seen."""

THREAD_BLOCK_RAGGED_SPREAD_FACTOR: Final[float] = 50.0
"""How much noisier a ragged block size may be than the full one before it is a finding.

The padding read can only affect a thread block that is not full, so its
signature is not "noisy" but "noisy *at 1023 and 1025 and not at 1024*". Measured
ratio: 1.09 at 1025 and 0.04 at 1023, against a factor of 50."""

THREAD_BLOCK_SPREAD_FLOOR: Final[float] = 1e-7
"""Floor under the full-block spread in that ratio, so a fluke-small 1024 run does
not turn the comparison into a division by nothing."""


def _skip_reason() -> str:
    """Say why this build cannot run the comparison, or the empty string when it can.

    Returns:
        A `pytest.skip` reason, or `""` when CASPAR projects OPENCV_FISHEYE.
    """
    if FISHEYE_MODEL in CASPAR_MODELS:
        return ""
    if not CASPAR_MODELS:
        return (
            "this pycolmap has no CASPAR at all (built without CASPAR_ENABLED, or no GPU "
            "answered); the comparison needs the `colsfm-caspar-fisheye` environment"
        )
    return (
        f"this CASPAR build projects {sorted(CASPAR_MODELS)} and not {FISHEYE_MODEL}, so it "
        f"would silently skip every fisheye observation; use `colsfm-caspar-fisheye`"
    )


pytestmark = pytest.mark.skipif(bool(_skip_reason()), reason=_skip_reason() or "runnable")


@dataclass(frozen=True, slots=True)
class FisheyeProblem:
    """One drawn OPENCV_FISHEYE bundle-adjustment problem, before it is built.

    Everything here is a structural knob Hypothesis shrinks. The bulk geometry --
    where each point sits, which way each camera is toed out -- is derived from
    `seed` in `scene_plan`, because 1500 individually shrinkable floats would
    spend the whole example budget without ever being the cause of a failure.
    """

    num_cameras: int
    """Cameras on the rig, 2-4. Camera 1 is the rig's reference sensor."""
    num_frames: int
    """Rig frames along the trajectory, 5-30."""
    num_points: int
    """Candidate world points, 100-500. Those seen by fewer than two images drop out."""
    image_size_px: int
    """Square sensor width and height, in pixels."""
    focal_ratio: float
    """`fx / image_size_px`. 0.30-0.50 spans the plausible wide-fisheye range."""
    aspect_ratio: float
    """`fy / fx`. Drawn away from 1 so `fx` and `fy` are never interchangeable.

    The adapter packs the merged calibration as `[fx, fy, k1..k4, cx, cy]` where
    COLMAP stores `[fx, fy, cx, cy, k1..k4]`; equal focal lengths would let a
    transposed pack pass."""
    principal_point_offset_ratio: tuple[float, float]
    """`(cx, cy)` displacement from the image centre, as a fraction of the width.

    Same reason as `aspect_ratio`: a centred principal point hides a swap of `cx`
    and `cy`, and a zero offset hides dropping them."""
    distortion: tuple[float, float, float, float]
    """`k1..k4`, drawn within the magnitudes a real Kannala-Brandt lens shows.

    Drawn independently, so no coefficient is ever a multiple of another and each
    term of `theta^2 (k1 + theta^2 (k2 + theta^2 (k3 + theta^2 k4)))` is separately
    identifiable."""
    max_baseline_m: float
    """Largest rig baseline, 0.05-0.30 m; every non-reference camera sits within it."""
    path_length_m: float
    """Total length of the rig trajectory, in metres."""
    travel_heading_deg: float
    """Angle between the rig's forward axis and its direction of travel.

    Zero is a drive straight at the point cloud, which is the classic weak
    configuration for depth from motion; +/-90 is a sideways sweep, which is what
    the probe's own §5.2 scene does. Drawing it covers both."""
    curvature_deg_per_frame: float
    """Yaw change per frame, which makes the trajectory an arc rather than a line."""
    pixel_noise_px: float
    """Standard deviation of the Gaussian noise on every observation, 0-0.5 px."""
    pose_translation_m: float
    """Standard deviation of the initial pose translation error handed to both solvers."""
    pose_rotation_deg: float
    """Standard deviation of the initial pose rotation error, in degrees."""
    point_perturbation_m: float
    """Standard deviation of the initial point error, in metres."""
    seed: int
    """PRNG seed for the bulk geometry, the observation noise and the perturbation."""


@dataclass(frozen=True, slots=True)
class PoseAgreement:
    """How far apart two solvers left the rig trajectory, after rigid alignment."""

    position_rmse_m: float
    """Root-mean-square distance between corresponding rig centres."""
    position_max_m: float
    """Largest single rig-centre distance."""
    rotation_rmse_deg: float
    """Root-mean-square rig-rotation angle difference."""
    rotation_max_deg: float
    """Largest single rig-rotation angle difference."""
    point_median_m: float
    """Median distance between corresponding 3D points, under the same alignment."""
    trajectory_extent_m: float
    """Largest side of the reference trajectory's bounding box, for scale."""


@st.composite
def fisheye_problems(draw: st.DrawFn) -> FisheyeProblem:
    """Draw a well-posed synthetic OPENCV_FISHEYE rig problem.

    The ranges are the specification of what the adapter claims to handle: a
    2-4 camera rig with 5-30 cm baselines, a wide fisheye at 0.30-0.50 of the
    sensor width, `k1..k4` shrinking by roughly an order of magnitude per term as
    a real Kannala-Brandt fit does, a short smooth trajectory, sub-pixel
    observation noise, and an initialisation knocked centimetres off.

    Args:
        draw: Hypothesis' draw callback.

    Returns:
        The drawn problem; `scene_plan` turns it into geometry.
    """
    return FisheyeProblem(
        num_cameras=draw(st.integers(min_value=2, max_value=4)),
        num_frames=draw(st.integers(min_value=5, max_value=30)),
        num_points=draw(st.integers(min_value=100, max_value=500)),
        image_size_px=draw(st.sampled_from((640, 800, 960))),
        focal_ratio=draw(st.floats(min_value=0.30, max_value=0.50)),
        aspect_ratio=draw(st.floats(min_value=0.92, max_value=1.08)),
        principal_point_offset_ratio=(
            draw(st.floats(min_value=-0.03, max_value=0.03)),
            draw(st.floats(min_value=-0.03, max_value=0.03)),
        ),
        distortion=(
            draw(st.floats(min_value=-0.05, max_value=0.05)),
            draw(st.floats(min_value=-0.02, max_value=0.02)),
            draw(st.floats(min_value=-0.008, max_value=0.008)),
            draw(st.floats(min_value=-0.002, max_value=0.002)),
        ),
        max_baseline_m=draw(st.floats(min_value=0.05, max_value=0.30)),
        path_length_m=draw(st.floats(min_value=0.5, max_value=2.5)),
        travel_heading_deg=draw(st.floats(min_value=-90.0, max_value=90.0)),
        curvature_deg_per_frame=draw(st.floats(min_value=-3.0, max_value=3.0)),
        pixel_noise_px=draw(st.floats(min_value=0.0, max_value=0.5)),
        pose_translation_m=draw(st.floats(min_value=0.005, max_value=0.03)),
        pose_rotation_deg=draw(st.floats(min_value=0.05, max_value=0.5)),
        point_perturbation_m=draw(st.floats(min_value=0.005, max_value=0.05)),
        seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
    )


def _yaw_rotation(yaw_rad: float) -> Float[ndarray, "3 3"]:
    """Rotation about the world +Y axis, the rig's up axis in this scene.

    Args:
        yaw_rad: Angle in radians.

    Returns:
        The rotation matrix.
    """
    return np.array(
        [
            [np.cos(yaw_rad), 0.0, np.sin(yaw_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw_rad), 0.0, np.cos(yaw_rad)],
        ],
        dtype=np.float64,
    )


def scene_plan(problem: FisheyeProblem) -> FisheyeScenePlan:
    """Turn a drawn problem into the geometry `build_scene_from_plan` consumes.

    The trajectory is an arc in the world XZ plane travelled by a rig that looks
    along its own +Z; the points fill a box in front of the arc's midpoint, wide
    enough that observations reach the image corners -- where `k3` and `k4` carry
    what signal they have -- and deep enough that the rig baselines triangulate
    them. Both are sized from the trajectory, so no draw needs rejecting.

    Args:
        problem: The drawn knobs.

    Returns:
        Cameras, extrinsics, poses, points and noise for one scene.
    """
    rng: np.random.Generator = np.random.default_rng(problem.seed)
    focal_length_px: float = problem.focal_ratio * problem.image_size_px

    cameras: list[pycolmap.Camera] = []
    sensor_from_rig: list[pycolmap.Rigid3d] = []
    offset_x_ratio, offset_y_ratio = problem.principal_point_offset_ratio
    for camera_index in range(problem.num_cameras):
        # Every camera on a rig has its own calibration, so the per-sensor jitter
        # is on top of the drawn aspect and offset rather than instead of them:
        # four identical cameras would let a per-sensor indexing error pass.
        cameras.append(
            fisheye_camera(
                camera_id=camera_index + 1,
                distortion=problem.distortion,
                focal_length_px=focal_length_px * float(rng.uniform(0.98, 1.02)),
                image_size_px=problem.image_size_px,
                focal_length_y_px=focal_length_px * problem.aspect_ratio,
                principal_point_px=(
                    problem.image_size_px * (0.5 + offset_x_ratio + float(rng.uniform(-0.01, 0.01))),
                    problem.image_size_px * (0.5 + offset_y_ratio + float(rng.uniform(-0.01, 0.01))),
                ),
            )
        )
        if camera_index == 0:
            sensor_from_rig.append(pycolmap.Rigid3d())
            continue
        # Baselines are drawn in the rig's XY plane and never below 5 cm, so no
        # two sensors coincide and the rig always carries stereo information.
        direction_rad: float = float(rng.uniform(0.0, 2.0 * np.pi))
        radius_m: float = float(rng.uniform(0.05, problem.max_baseline_m))
        centre_in_rig_m: Float[ndarray, " 3"] = radius_m * np.array(
            [np.cos(direction_rad), np.sin(direction_rad), 0.0], dtype=np.float64
        )
        sensor_R_rig: Float[ndarray, "3 3"] = _yaw_rotation(
            float(np.deg2rad(rng.uniform(-15.0, 15.0)))
        )
        sensor_from_rig.append(
            pycolmap.Rigid3d(
                pycolmap.Rotation3d(sensor_R_rig), -sensor_R_rig @ centre_in_rig_m
            )
        )

    step_m: float = problem.path_length_m / max(problem.num_frames - 1, 1)
    curvature_rad: float = float(np.deg2rad(problem.curvature_deg_per_frame))
    heading_rad: float = float(np.deg2rad(problem.travel_heading_deg))
    travel_in_rig: Float[ndarray, " 3"] = np.array(
        [np.sin(heading_rad), 0.0, np.cos(heading_rad)], dtype=np.float64
    )
    centres_m: list[Float[ndarray, " 3"]] = []
    rig_from_world: list[pycolmap.Rigid3d] = []
    position_m: Float[ndarray, " 3"] = np.zeros(3, dtype=np.float64)
    for frame_index in range(problem.num_frames):
        yaw_rad: float = curvature_rad * frame_index
        rig_R_world: Float[ndarray, "3 3"] = _yaw_rotation(yaw_rad)
        centres_m.append(position_m.copy())
        rig_from_world.append(
            pycolmap.Rigid3d(pycolmap.Rotation3d(rig_R_world), -rig_R_world @ position_m)
        )
        # Advance along a fixed direction in the rig's own frame, so the yaw makes
        # the path an arc and the rig keeps facing the point cloud whatever the
        # heading is.
        position_m = position_m + step_m * rig_R_world.T @ travel_in_rig

    # The box is wide and shallow on purpose: a point 6 m off-axis at 2.5 m depth
    # subtends theta ~= 67 deg, which is where k3 and k4 have anything to say. A
    # narrow frustum of points would exercise the polynomial only near theta = 0
    # and would pass whatever k3 and k4 were. Anything the cameras cannot see is
    # dropped by `build_scene_from_plan`'s cheirality and sensor-bounds test.
    midpoint_m: Float[ndarray, " 3"] = np.mean(np.asarray(centres_m), axis=0)
    depth_m: float = 2.5 + problem.path_length_m
    points_xyz: Float[ndarray, "n_points 3"] = midpoint_m + np.column_stack(
        (
            rng.uniform(-6.0, 6.0, problem.num_points),
            rng.uniform(-4.0, 4.0, problem.num_points),
            rng.uniform(depth_m, depth_m + 8.0, problem.num_points),
        )
    )
    return FisheyeScenePlan(
        cameras=tuple(cameras),
        sensor_from_rig=tuple(sensor_from_rig),
        rig_from_world=tuple(rig_from_world),
        points_xyz=points_xyz,
        pixel_noise_px=problem.pixel_noise_px,
        noise_seed=problem.seed,
    )


@dataclass(frozen=True, slots=True)
class Coverage:
    """What a built scene actually hands the solvers, counted independently.

    Independently is the operative word: every number here is recomputed from the
    reconstruction rather than read out of a solver summary, so it can be compared
    against `BundleAdjustmentSummary.num_residuals` instead of agreeing with it by
    construction.
    """

    num_observations: int
    """2D points of a registered image that carry a 3D point; the eligible residuals / 2."""
    num_tracks: int
    """Distinct 3D points with at least one such observation."""
    min_track_length: int
    """Shortest track. Below two the point would be gauge-free."""
    max_theta_deg: float
    """Largest incidence angle any observation was formed at, in degrees."""
    min_depth_m: float
    """Smallest observation depth, in metres. Must be positive: CASPAR accepts
    negative depth through its `z + eps sign(z)` guard where COLMAP's own
    projection would refuse, so the scene must not contain any."""
    max_depth_m: float
    """Largest observation depth, in metres."""
    all_projections_finite: bool
    """Whether every observation reprojects to a finite pixel."""


def measure_coverage(reconstruction: pycolmap.Reconstruction) -> Coverage:
    """Count the eligible observations and check the projection domain.

    Args:
        reconstruction: The model about to be, or just, adjusted.

    Returns:
        The counts and the domain limits.
    """
    num_observations: int = 0
    tracks: set[int] = set()
    track_lengths: list[int] = []
    max_theta_rad: float = 0.0
    min_depth_m: float = np.inf
    max_depth_m: float = 0.0
    all_finite: bool = True
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        point3D_ids: list[int] = [p.point3D_id for p in image.points2D if p.has_point3D()]
        if not point3D_ids:
            continue
        num_observations += len(point3D_ids)
        tracks.update(point3D_ids)
        world_xyz: Float[ndarray, "n_obs 3"] = np.array(
            [reconstruction.point3D(pid).xyz for pid in point3D_ids], dtype=np.float64
        )
        in_camera_xyz: Float[ndarray, "n_obs 3"] = (
            world_xyz @ np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64).T
            + np.asarray(cam_from_world.translation, dtype=np.float64)
        )
        depth_m: Float[ndarray, " n_obs"] = in_camera_xyz[:, 2]
        min_depth_m = min(min_depth_m, float(depth_m.min()))
        max_depth_m = max(max_depth_m, float(depth_m.max()))
        radius: Float[ndarray, " n_obs"] = np.linalg.norm(
            in_camera_xyz[:, :2] / depth_m[:, None], axis=1
        )
        max_theta_rad = max(max_theta_rad, float(np.arctan(radius).max()))
        projected: Float[ndarray, "n_obs 2"] = np.asarray(
            camera.img_from_cam(in_camera_xyz), dtype=np.float64
        )
        all_finite = all_finite and bool(np.isfinite(projected).all())
    for point3D_id in tracks:
        track_lengths.append(reconstruction.point3D(point3D_id).track.length())
    return Coverage(
        num_observations=num_observations,
        num_tracks=len(tracks),
        min_track_length=min(track_lengths) if track_lengths else 0,
        max_theta_deg=float(np.degrees(max_theta_rad)),
        min_depth_m=float(min_depth_m),
        max_depth_m=max_depth_m,
        all_projections_finite=all_finite,
    )


def _probe_config(problem: FisheyeProblem) -> ProbeConfig:
    """The perturbation magnitudes, in the shape `perturb_model` reads them.

    Args:
        problem: The drawn knobs.

    Returns:
        A `ProbeConfig` carrying only the three perturbation standard deviations.
    """
    return ProbeConfig(
        distortion=problem.distortion,
        pose_translation_m=problem.pose_translation_m,
        pose_rotation_deg=problem.pose_rotation_deg,
        point_perturbation_m=problem.point_perturbation_m,
        gpu_index=GPU_INDEX,
    )


def _rig_centres_m(result: SolveResult) -> Float[ndarray, "n_frames 3"]:
    """World positions of the rig, from the `rig_from_world` poses a solve left.

    Args:
        result: One solver's answer.

    Returns:
        `-R^T t` per frame, in metres.
    """
    return -np.einsum("nji,nj->ni", result.frame_rotations, result.frame_translations)


def _rigid_alignment(
    source_xyz: Float[ndarray, "n 3"], target_xyz: Float[ndarray, "n 3"]
) -> tuple[Float[ndarray, "3 3"], Float[ndarray, " 3"]]:
    """Kabsch: the rotation and translation that best map `source` onto `target`.

    Scale is deliberately not estimated. A multi-camera rig with fixed extrinsics
    fixes the scale of its own reconstruction, so a scale difference between the
    two solvers would be a real disagreement and must not be aligned away.

    Args:
        source_xyz: Points to move, in metres.
        target_xyz: Points to move them onto, in metres.

    Returns:
        The rotation and the translation, applied as `rotation @ x + translation`.
    """
    source_centroid: Float[ndarray, " 3"] = source_xyz.mean(axis=0)
    target_centroid: Float[ndarray, " 3"] = target_xyz.mean(axis=0)
    covariance: Float[ndarray, "3 3"] = (target_xyz - target_centroid).T @ (
        source_xyz - source_centroid
    )
    left, _, right = np.linalg.svd(covariance)
    reflection: Float[ndarray, "3 3"] = np.diag(
        np.array([1.0, 1.0, float(np.sign(np.linalg.det(left @ right)))])
    )
    rotation: Float[ndarray, "3 3"] = left @ reflection @ right
    return rotation, target_centroid - rotation @ source_centroid


def pose_agreement(reference: SolveResult, candidate: SolveResult) -> PoseAgreement:
    """How far `candidate` sits from `reference` once rigidly aligned onto it.

    The alignment is fitted to the rig centres *and* the 3D points, not to the
    centres alone. Hypothesis found the reason on its second example: a rig that
    slides in a straight line leaves the roll about that line unconstrained, so
    Kabsch on the centres returns an arbitrary rotation about the trajectory
    axis, which maps the line onto itself and then reports ~150 deg of rotation
    disagreement on two answers that in fact agree to 0.6 um. The point cloud
    spans all three axes on every draw and fixes the alignment outright.

    Args:
        reference: The answer to measure against; Ceres here.
        candidate: The answer under test; CASPAR here.

    Returns:
        Position, rotation and point disagreement, plus the trajectory's extent.
    """
    reference_centres_m: Float[ndarray, "n_frames 3"] = _rig_centres_m(reference)
    candidate_centres_m: Float[ndarray, "n_frames 3"] = _rig_centres_m(candidate)
    rotation, translation = _rigid_alignment(
        np.vstack((candidate_centres_m, candidate.points_xyz)),
        np.vstack((reference_centres_m, reference.points_xyz)),
    )

    aligned_centres_m: Float[ndarray, "n_frames 3"] = (
        candidate_centres_m @ rotation.T + translation
    )
    position_error_m: Float[ndarray, " n_frames"] = np.linalg.norm(
        aligned_centres_m - reference_centres_m, axis=1
    )
    # A world rotation of `rotation` turns candidate-world vectors into
    # reference-world ones, so the comparable rig orientation is R_candidate @ R^T.
    aligned_rotations: Float[ndarray, "n_frames 3 3"] = candidate.frame_rotations @ rotation.T
    relative: Float[ndarray, "n_frames 3 3"] = (
        reference.frame_rotations @ aligned_rotations.transpose(0, 2, 1)
    )
    rotation_error_deg: Float[ndarray, " n_frames"] = np.degrees(
        np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
    )
    point_error_m: Float[ndarray, " n_points"] = np.linalg.norm(
        candidate.points_xyz @ rotation.T + translation - reference.points_xyz, axis=1
    )
    return PoseAgreement(
        position_rmse_m=float(np.sqrt(np.mean(position_error_m**2))),
        position_max_m=float(position_error_m.max()),
        rotation_rmse_deg=float(np.sqrt(np.mean(rotation_error_deg**2))),
        rotation_max_deg=float(rotation_error_deg.max()),
        point_median_m=float(np.median(point_error_m)),
        trajectory_extent_m=float(np.ptp(reference_centres_m, axis=0).max()),
    )


def _initial_state(
    problem: FisheyeProblem, plan: FisheyeScenePlan
) -> tuple[pycolmap.Reconstruction, float]:
    """Build the scene and knock it off its solution, the same way every time.

    Both backends get their own reconstruction from this function, which makes
    their starting states bit-identical without either of them having to be
    solved first.

    The prune afterwards is not cosmetic and was not there first. The point box
    is deliberately wide, so some observations sit at grazing incidence; a
    centimetre of point noise then pushes a few of them through a camera's
    principal plane, and *there* the two backends stop agreeing about the
    problem: Ceres' reprojection cost returns zero for a projection COLMAP's
    model refuses, while CASPAR projects it anyway through `z + eps sign(z)`.
    Hypothesis found this within twelve examples -- a rig on which Ceres reached
    0.000000 px and CASPAR reported 2.5e153 px. Dropping those tracks from the
    *perturbed* state, deterministically and identically for both backends,
    restores a common problem; `MIN_INITIAL_DEPTH_M` is the margin.

    Args:
        problem: The drawn knobs; supplies the perturbation and the seed.
        plan: The geometry to build.

    Returns:
        The perturbed model and its mean reprojection error before any solve.
    """
    reconstruction: pycolmap.Reconstruction = build_scene_from_plan(plan)
    perturb_model(reconstruction, _probe_config(problem), seed=problem.seed)
    drop_degenerate_points(reconstruction, min_depth_m=MIN_INITIAL_DEPTH_M)
    reconstruction.update_point_3d_errors()
    return reconstruction, float(reconstruction.compute_mean_reprojection_error())


def _report(
    problem: FisheyeProblem,
    coverage: Coverage,
    initial_reprojection_px: float,
    ceres: SolveResult,
    caspar: SolveResult,
    polish: SolveResult,
    agreement: PoseAgreement,
    polished_agreement: PoseAgreement,
) -> str:
    """Everything one example measured, on one line per fact.

    Attached to every assertion in the property so that a failure carries the
    numbers that explain it rather than just the one bound that broke.

    Args:
        problem: The drawn knobs.
        coverage: What the built scene handed the solvers.
        initial_reprojection_px: Mean reprojection error of the perturbed state.
        ceres: The Ceres answer.
        caspar: The CASPAR answer.
        polish: The CASPAR answer after one Ceres round.
        agreement: How far raw CASPAR sits from Ceres.
        polished_agreement: How far the polished CASPAR answer sits from Ceres.

    Returns:
        A multi-line diagnostic block.
    """
    ceres_px: float = max(ceres.reprojection_error_px, 1e-12)
    lines: list[str] = [
        "",
        f"rig: {problem.num_cameras} cameras x {problem.num_frames} frames, "
        + f"{coverage.num_tracks}/{problem.num_points} tracks, {coverage.num_observations} observations, "
        + f"{problem.image_size_px} px, f={problem.focal_ratio * problem.image_size_px:.1f} px",
        f"  intrinsics          fy/fx {problem.aspect_ratio:.4f}, principal point offset "
        + f"{problem.principal_point_offset_ratio[0]:+.4f}, {problem.principal_point_offset_ratio[1]:+.4f} of width",
        f"  k1..k4              {problem.distortion}",
        f"  observation domain  theta <= {coverage.max_theta_deg:.2f} deg, depth "
        + f"{coverage.min_depth_m:.3f}-{coverage.max_depth_m:.3f} m, shortest track "
        + f"{coverage.min_track_length}, all projections finite: {coverage.all_projections_finite}",
        f"  baseline <= {problem.max_baseline_m:.3f} m, path {problem.path_length_m:.3f} m, "
        + f"heading {problem.travel_heading_deg:+.1f} deg, "
        + f"curvature {problem.curvature_deg_per_frame:+.2f} deg/frame, seed {problem.seed}",
        f"  noise {problem.pixel_noise_px:.3f} px; start off by {problem.pose_translation_m:.3f} m / "
        + f"{problem.pose_rotation_deg:.3f} deg / {problem.point_perturbation_m:.3f} m",
        f"  reprojection (px)   start  {initial_reprojection_px:.6f}",
        f"                      ceres  {ceres.reprojection_error_px:.6f}  ({ceres.termination}, "
        + f"{ceres.seconds:.3f} s, {ceres.num_residuals} residuals)",
        f"                      caspar {caspar.reprojection_error_px:.6f}  ({caspar.termination}, "
        + f"{caspar.seconds:.3f} s, {caspar.num_residuals} residuals)",
        f"                      caspar+polish {polish.reprojection_error_px:.6f}  ({polish.termination}, "
        + f"{polish.seconds:.3f} s)",
        f"  caspar/ceres reprojection ratio {caspar.reprojection_error_px / ceres_px:.4f}",
        f"  polish/ceres reprojection ratio {polish.reprojection_error_px / ceres_px:.6f}",
        f"  caspar vs ceres     position RMSE {agreement.position_rmse_m * 1e3:.4f} mm, max "
        + f"{agreement.position_max_m * 1e3:.4f} mm over a {agreement.trajectory_extent_m:.2f} m trajectory",
        f"                      rotation RMSE {agreement.rotation_rmse_deg:.6f} deg, max "
        + f"{agreement.rotation_max_deg:.6f} deg, point median {agreement.point_median_m * 1e3:.4f} mm",
        f"  polished vs ceres   position RMSE {polished_agreement.position_rmse_m * 1e3:.4f} mm, max "
        + f"{polished_agreement.position_max_m * 1e3:.4f} mm",
        f"                      rotation RMSE {polished_agreement.rotation_rmse_deg:.6f} deg, max "
        + f"{polished_agreement.rotation_max_deg:.6f} deg, "
        + f"point median {polished_agreement.point_median_m * 1e3:.4f} mm",
    ]
    return "\n".join(lines)


@settings(
    deadline=None,
    max_examples=MAX_EXAMPLES,
    # Every example builds a scene and runs three bundle adjustments, two of them
    # on a shared GPU; Hypothesis' data-generation and per-example timers both
    # flag that, and neither is measuring anything about the code under test.
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(problem=fisheye_problems())
def test_caspar_matches_ceres_on_random_fisheye_rigs(problem: FisheyeProblem) -> None:
    """CASPAR lands where Ceres does on any well-posed OPENCV_FISHEYE rig.

    Four claims, all on one solve pair so the GPU is paid for once:

    1. CASPAR parameterised the *same* residuals as Ceres. A build without the
       adapter drops every fisheye image with "unsupported camera model" and
       reports `num_residuals == 0`, and its solve still "succeeds" -- so
       equality with Ceres' count is what proves the adapter ran.
    2. Its mean reprojection error is no worse than Ceres' by more than the
       documented float32 shortfall.
    3. Its trajectory agrees with Ceres' after rigid alignment, to a bound that
       scales with the trajectory because the weakest rigs in the drawn family
       have a genuinely flat depth valley (see
       `POSITION_RMSE_FRACTION_OF_EXTENT`).
    4. One Ceres round on CASPAR's own answer -- the polish colsfm ships -- lands
       on Ceres' reprojection error *and* on Ceres' geometry, at every
       conditioning. This is what makes the fp32 shortfall a cost and not a
       defect, and it is the tightest bound in the module.

    Args:
        problem: A rig drawn by `fisheye_problems`.
    """
    plan: FisheyeScenePlan = scene_plan(problem)
    ceres_model, initial_reprojection_px = _initial_state(problem, plan)
    caspar_model, caspar_initial_px = _initial_state(problem, plan)

    coverage: Coverage = measure_coverage(ceres_model)
    # A tripwire, not a filter: `scene_plan` sizes the point box from the
    # trajectory precisely so that this always holds.
    assume(coverage.num_tracks >= MIN_TRACKS)
    assert caspar_initial_px == initial_reprojection_px, (
        "the two backends must start from bit-identical state, or the comparison "
        f"measures the initialisation: {caspar_initial_px} != {initial_reprojection_px}"
    )
    assert measure_coverage(caspar_model) == coverage, (
        "the two backends were handed different problems; the whole comparison is void"
    )

    ceres: SolveResult = solve_with_backend(
        ceres_model, bundle_adjustment_config(ceres_model), "ceres", GPU_INDEX
    )
    caspar: SolveResult = solve_with_backend(
        caspar_model, bundle_adjustment_config(caspar_model), "caspar", GPU_INDEX
    )
    # The polish is one more Ceres round over the model CASPAR left behind, with
    # nothing filtered in between -- `colsfm.mapping._polish_with_ceres` in
    # miniature (`docs/caspar-build.md` § Shipped: CASPAR + Ceres polish).
    polish: SolveResult = solve_with_backend(
        caspar_model, bundle_adjustment_config(caspar_model), "ceres", GPU_INDEX
    )
    agreement: PoseAgreement = pose_agreement(ceres, caspar)
    polished_agreement: PoseAgreement = pose_agreement(ceres, polish)
    report: str = _report(
        problem,
        coverage,
        initial_reprojection_px,
        ceres,
        caspar,
        polish,
        agreement,
        polished_agreement,
    )
    note(report)

    # The problem's domain, checked before any error is summed: a NaN projection
    # would otherwise vanish into a mean and make every bound below meaningless.
    assert coverage.all_projections_finite, (
        f"an observation of the *input* model does not project to a finite pixel.{report}"
    )
    assert coverage.min_depth_m > 0.0, (
        f"the scene contains an observation at non-positive depth ({coverage.min_depth_m:.4f} m). "
        "CASPAR projects it anyway through its `z + eps sign(z)` guard where COLMAP's own model "
        f"refuses, so the two backends would not be minimising the same function.{report}"
    )
    assert coverage.min_track_length >= 2, (
        f"a track of length {coverage.min_track_length} is gauge-free.{report}"
    )
    assert coverage.max_theta_deg >= MIN_PERIPHERAL_THETA_DEG, (
        f"the widest observation sits at only {coverage.max_theta_deg:.2f} deg, so k3 and k4 "
        f"barely enter the residual and this rig would pass with either of them wrong.{report}"
    )

    # Exact coverage, both ways: two residuals per eligible observation, counted
    # from the reconstruction rather than taken from either summary. Zero on the
    # CASPAR side is what "unsupported camera model" looks like, and that is
    # precisely what a `==` against Ceres alone would not distinguish from a
    # build in which *both* dropped the same images.
    expected_residuals: int = 2 * coverage.num_observations
    assert ceres.num_residuals == expected_residuals, (
        f"Ceres parameterised {ceres.num_residuals} residuals for {coverage.num_observations} "
        f"eligible observations, not the {expected_residuals} expected.{report}"
    )
    assert caspar.num_residuals == expected_residuals, (
        f"CASPAR parameterised {caspar.num_residuals} residuals for {coverage.num_observations} "
        f"eligible observations, not the {expected_residuals} expected. Zero means this build has "
        "no OPENCV_FISHEYE adapter and skipped every image; anything else in between means it "
        f"silently dropped part of the problem.{report}"
    )
    assert np.isfinite(caspar.score) and np.isfinite(caspar.reprojection_error_px), (
        f"CASPAR left the model with a non-finite score.{report}"
    )
    assert caspar.reprojection_error_px < initial_reprojection_px, (
        "CASPAR returned a model no better than the one it was given, which is what the "
        f"PR #4611 false-convergence bug looked like.{report}"
    )

    reprojection_bound_px: float = (
        ceres.reprojection_error_px * (1.0 + REPROJECTION_RELATIVE_TOLERANCE)
        + REPROJECTION_FLOOR_PX
    )
    assert caspar.reprojection_error_px <= reprojection_bound_px, (
        f"CASPAR's float32 solve stopped further short of the optimum than the "
        f"{REPROJECTION_RELATIVE_TOLERANCE:.0%} + {REPROJECTION_FLOOR_PX} px this build is "
        f"allowed: {caspar.reprojection_error_px:.6f} px > {reprojection_bound_px:.6f} px.{report}"
    )

    position_bound_m: float = (
        POSITION_RMSE_FLOOR_M
        + POSITION_RMSE_FRACTION_OF_EXTENT * agreement.trajectory_extent_m
    )
    assert agreement.position_rmse_m <= position_bound_m, (
        f"the two trajectories disagree by {agreement.position_rmse_m * 1e3:.3f} mm RMSE, above the "
        f"{position_bound_m * 1e3:.3f} mm this trajectory's length allows.{report}"
    )
    assert agreement.rotation_max_deg <= ROTATION_TOLERANCE_DEG, (
        f"the two trajectories disagree by {agreement.rotation_max_deg:.6f} deg, above the "
        f"{ROTATION_TOLERANCE_DEG} deg bound.{report}"
    )

    polish_bound_px: float = (
        ceres.reprojection_error_px * (1.0 + POLISH_RELATIVE_TOLERANCE) + POLISH_FLOOR_PX
    )
    assert polish.reprojection_error_px <= polish_bound_px, (
        f"the Ceres polish did not recover Ceres' own answer: {polish.reprojection_error_px:.8f} px "
        f"> {polish_bound_px:.8f} px. Both are Ceres solves of the same problem, so this is a "
        f"claim about the model CASPAR handed over, not about Ceres.{report}"
    )
    polished_position_bound_m: float = (
        POLISHED_POSITION_FLOOR_M
        + POLISHED_POSITION_FRACTION_OF_EXTENT * agreement.trajectory_extent_m
    )
    assert polished_agreement.position_rmse_m <= polished_position_bound_m, (
        f"the polish left the trajectory {polished_agreement.position_rmse_m * 1e3:.4f} mm from "
        f"Ceres' own, above the {polished_position_bound_m * 1e3:.4f} mm bound. Raw CASPAR was "
        f"{agreement.position_rmse_m * 1e3:.4f} mm out, so if that number is large and this one is "
        f"too, the two solvers found different minima rather than the same one.{report}"
    )
    assert polished_agreement.rotation_max_deg <= POLISHED_ROTATION_DEG, (
        f"the polish left the rig rotations {polished_agreement.rotation_max_deg:.8f} deg from "
        f"Ceres' own, above the {POLISHED_ROTATION_DEG} deg bound.{report}"
    )
    # "The polish closes the gap", stated so that the flat basin of a noiseless
    # scene cannot flatten it into a coin toss: whatever the absolute numbers, the
    # polished answer must not be further from Ceres than the raw one was.
    assert (
        polished_agreement.position_rmse_m
        <= agreement.position_rmse_m + POLISHED_POSITION_FLOOR_M
    ), (
        f"the polish moved the trajectory *away* from Ceres: raw CASPAR was "
        f"{agreement.position_rmse_m * 1e3:.4f} mm out and the polished answer is "
        f"{polished_agreement.position_rmse_m * 1e3:.4f} mm out.{report}"
    )


BOUNDARY_PROBLEM: Final[FisheyeProblem] = FisheyeProblem(
    num_cameras=2,
    num_frames=6,
    num_points=200,
    image_size_px=800,
    focal_ratio=0.35,
    aspect_ratio=1.03,
    principal_point_offset_ratio=(0.012, -0.009),
    distortion=(0.02, -0.01, 0.005, -0.001),
    max_baseline_m=0.22,
    path_length_m=1.4,
    travel_heading_deg=55.0,
    curvature_deg_per_frame=1.5,
    pixel_noise_px=0.25,
    pose_translation_m=0.01,
    pose_rotation_deg=0.3,
    point_perturbation_m=0.02,
    seed=20260905,
)
"""One fixed, well-conditioned rig, trimmed to an exact observation count.

Not drawn: the thread-block check needs the *count* to be the variable and
everything else held, so a Hypothesis draw would only add noise to it."""


def _trim_to_observations(reconstruction: pycolmap.Reconstruction, target: int) -> int:
    """Delete observations until the model carries exactly `target` of them.

    Observations are taken one at a time off the longest tracks, so no track ever
    falls below two elements and the problem stays gauge-determined; a whole point
    is dropped only when the remaining excess is at least two and no long track is
    left. This is how an *exact* count -- 1023, not "about a thousand" -- is
    reached without hand-building a reconstruction.

    Args:
        reconstruction: The model to trim, modified in place.
        target: Observations wanted.

    Returns:
        The observation count actually reached.
    """
    while True:
        excess: int = int(reconstruction.compute_num_observations()) - target
        if excess <= 0:
            return int(reconstruction.compute_num_observations())
        longest_id: int | None = None
        longest_length: int = 2
        for point3D_id in reconstruction.point3D_ids():
            length: int = reconstruction.point3D(point3D_id).track.length()
            if length > longest_length:
                longest_id, longest_length = point3D_id, length
        if longest_id is not None:
            element = reconstruction.point3D(longest_id).track.element(0)
            reconstruction.delete_observation(element.image_id, element.point2D_idx)
            continue
        if excess < 2:
            raise AssertionError(
                f"cannot reach {target} observations: every track has length two and "
                f"{excess} observation(s) of excess remain"
            )
        reconstruction.delete_point3D(next(iter(reconstruction.point3D_ids())))


def _solve_at_size(num_observations: int) -> tuple[SolveResult, list[float], list[float], str]:
    """Solve the fixed boundary rig, trimmed to an exact size, on Ceres then CASPAR N times.

    Args:
        num_observations: Eligible observations to trim to.

    Returns:
        The Ceres answer, the CASPAR scores, the CASPAR reprojection errors, and a
        one-line summary for the failure message.
    """
    plan: FisheyeScenePlan = scene_plan(BOUNDARY_PROBLEM)
    ceres_model, initial_px = _initial_state(BOUNDARY_PROBLEM, plan)
    _trim_to_observations(ceres_model, num_observations)
    ceres_model.update_point_3d_errors()
    coverage: Coverage = measure_coverage(ceres_model)
    assert coverage.num_observations == num_observations, (
        f"trimming reached {coverage.num_observations} observations, not {num_observations}"
    )
    assert coverage.all_projections_finite and coverage.min_depth_m > 0.0, (
        f"the trimmed model at {num_observations} observations left the projection domain"
    )
    ceres: SolveResult = solve_with_backend(
        ceres_model, bundle_adjustment_config(ceres_model), "ceres", GPU_INDEX
    )

    scores: list[float] = []
    reprojections_px: list[float] = []
    for _ in range(THREAD_BLOCK_REPEATS):
        caspar_model, _ = _initial_state(BOUNDARY_PROBLEM, plan)
        _trim_to_observations(caspar_model, num_observations)
        assert measure_coverage(caspar_model) == coverage, (
            f"the trimmed CASPAR model at {num_observations} observations differs from the "
            "trimmed Ceres model"
        )
        caspar: SolveResult = solve_with_backend(
            caspar_model, bundle_adjustment_config(caspar_model), "caspar", GPU_INDEX
        )
        assert caspar.num_residuals == 2 * num_observations, (
            f"CASPAR parameterised {caspar.num_residuals} residuals at {num_observations} "
            f"observations, not {2 * num_observations}; Ceres reported {ceres.num_residuals}"
        )
        assert np.isfinite(caspar.score), (
            f"CASPAR's score is not finite at {num_observations} observations, which is what an "
            "uninitialised padding-thread accumulator would produce"
        )
        scores.append(caspar.score)
        reprojections_px.append(caspar.reprojection_error_px)

    summary: str = (
        f"\n  {num_observations:5d} observations: start {initial_px:.6f} px, "
        f"ceres {ceres.reprojection_error_px:.8f} px, caspar "
        f"{min(reprojections_px):.8f}-{max(reprojections_px):.8f} px, scores "
        f"{[format(value, '.6e') for value in scores]}"
    )
    return ceres, scores, reprojections_px, summary


def test_caspar_score_is_stable_across_the_thread_block_boundary() -> None:
    """A problem size that does not fill CASPAR's last thread block still solves right.

    The generated score kernels declare their per-thread accumulator without an
    initialiser, assign it inside a `global_thread_idx < problem_size` guard, and
    then pass it *by value* to `SumStore`, which masks it only after the argument
    has been evaluated. Threads in the padding of the final block therefore read
    an uninitialised register. That is the source-level defect behind PR #4611's
    false convergence, and it can only bite when the factor count is not a
    multiple of the block width.

    So the shape of the check follows the shape of the defect. 1024 observations
    fill the block exactly and are the control; 1023 and 1025 leave it ragged.
    Each size is solved five times from identical state, because an uninitialised
    read returns whatever the previous kernel left behind and is therefore a
    *flakiness* bug, not a wrong-answer bug. The finding is not "is CASPAR noisy"
    -- a float32 reduction is always a little noisy -- but "is CASPAR noisier
    when the block is ragged than when it is full".

    Measured on an RTX 5090 with this build: it is not. The relative spread over
    five repeats is 1e-8 at 1023, 2.5e-7 at 1024 and 2.7e-7 at 1025, so the full
    block is as noisy as the ragged ones and the source-level defect does not
    reach the answer here. That is a negative result about one GPU and one
    compiler, not a proof that the defect is harmless.
    """
    plan: FisheyeScenePlan = scene_plan(BOUNDARY_PROBLEM)
    available: int = int(build_scene_from_plan(plan).compute_num_observations())
    assert available > max(THREAD_BLOCK_OBSERVATIONS), (
        f"the fixed boundary rig produces {available} observations, too few to trim to "
        f"{max(THREAD_BLOCK_OBSERVATIONS)}"
    )

    spreads: dict[int, float] = {}
    summaries: list[str] = []
    for num_observations in THREAD_BLOCK_OBSERVATIONS:
        ceres, scores, reprojections_px, summary = _solve_at_size(num_observations)
        summaries.append(summary)
        spreads[num_observations] = float(np.ptp(scores)) / max(
            abs(float(np.mean(scores))), 1e-12
        )
        bound_px: float = (
            ceres.reprojection_error_px * (1.0 + REPROJECTION_RELATIVE_TOLERANCE)
            + REPROJECTION_FLOOR_PX
        )
        assert max(reprojections_px) <= bound_px, (
            f"CASPAR landed at {max(reprojections_px):.8f} px against a bound of {bound_px:.8f} px "
            f"at {num_observations} observations.{''.join(summaries)}"
        )

    report: str = "".join(summaries) + "\n  spreads: " + ", ".join(
        f"{size}: {spread:.3e}" for size, spread in spreads.items()
    )
    for num_observations, spread in spreads.items():
        assert spread <= THREAD_BLOCK_SCORE_TOLERANCE, (
            f"{THREAD_BLOCK_REPEATS} runs of one identical {num_observations}-observation problem "
            f"gave scores spread over {spread:.3e} relative, above "
            f"{THREAD_BLOCK_SCORE_TOLERANCE:.0e}.{report}"
        )

    full_block_spread: float = max(spreads[1024], THREAD_BLOCK_SPREAD_FLOOR)
    for num_observations in (1023, 1025):
        assert spreads[num_observations] <= THREAD_BLOCK_RAGGED_SPREAD_FACTOR * full_block_spread, (
            f"the ragged block size {num_observations} is {spreads[num_observations] / full_block_spread:.1f}x "
            f"noisier than the full block of 1024. A padding thread cannot exist at 1024, so a "
            f"gap this size is the signature of the uninitialised accumulator read.{report}"
        )


def test_points_behind_the_cameras_never_reach_the_solvers() -> None:
    """Non-positive depth is an excluded domain, not an input the solvers must handle.

    CASPAR's cores divide by `z + eps sign(z)`, which happily projects a point
    *behind* the camera to a finite pixel; COLMAP's own `OpenCVFisheyeCameraModel`
    refuses, and its Ceres cost function then contributes zero. So a scene holding
    such an observation makes the two backends minimise different functions, and
    every comparison in this module would be void. The scene builder's cheirality
    test is what keeps them out, and this is the test of that test: adding a wall
    of points behind every camera must change nothing at all about the problem
    handed to either solver.
    """
    plan: FisheyeScenePlan = scene_plan(BOUNDARY_PROBLEM)
    clean: pycolmap.Reconstruction = build_scene_from_plan(plan)
    clean_coverage: Coverage = measure_coverage(clean)

    rng: np.random.Generator = np.random.default_rng(BOUNDARY_PROBLEM.seed)
    centres_m: Float[ndarray, "n_frames 3"] = np.array(
        [-pose.rotation.matrix().T @ pose.translation for pose in plan.rig_from_world],
        dtype=np.float64,
    )
    candidates_xyz: Float[ndarray, "n_candidates 3"] = centres_m.mean(axis=0) + np.column_stack(
        (
            rng.uniform(-8.0, 8.0, 4000),
            rng.uniform(-6.0, 6.0, 4000),
            rng.uniform(-12.0, 0.0, 4000),
        )
    )
    # "Behind the cameras" has to mean behind *every* camera, and the rig arcs, so
    # a slab behind the trajectory midpoint is not enough -- a first version of
    # this test used one and five of its points landed 12 cm in FRONT of a late
    # frame. The depth is therefore computed per image and the candidates kept
    # only where all of them are non-positive.
    depths_m: list[Float[ndarray, " n_candidates"]] = []
    for image_id in sorted(clean.images):
        cam_from_world: pycolmap.Rigid3d = clean.image(image_id).cam_from_world()
        viewing_axis: Float[ndarray, " 3"] = np.asarray(
            cam_from_world.rotation.matrix(), dtype=np.float64
        )[2]
        offset_m: float = float(np.asarray(cam_from_world.translation, dtype=np.float64)[2])
        depths_m.append(candidates_xyz @ viewing_axis + offset_m)
    behind: Bool[ndarray, " n_candidates"] = (np.asarray(depths_m) <= 0.0).all(axis=0)
    behind_xyz: Float[ndarray, "n_behind 3"] = candidates_xyz[behind]
    assert len(behind_xyz) >= 100, (
        f"only {len(behind_xyz)} candidates sit behind every camera; the test has nothing to add"
    )
    polluted: pycolmap.Reconstruction = build_scene_from_plan(
        FisheyeScenePlan(
            cameras=plan.cameras,
            sensor_from_rig=plan.sensor_from_rig,
            rig_from_world=plan.rig_from_world,
            points_xyz=np.vstack((plan.points_xyz, behind_xyz)),
            pixel_noise_px=plan.pixel_noise_px,
            noise_seed=plan.noise_seed,
        )
    )
    polluted_coverage: Coverage = measure_coverage(polluted)

    assert polluted_coverage.min_depth_m > 0.1, (
        f"a point at {polluted_coverage.min_depth_m:.4f} m depth reached the problem; the "
        "cheirality test in `build_scene_from_plan` no longer holds"
    )
    assert polluted_coverage.all_projections_finite, (
        "an observation of the polluted scene does not project to a finite pixel"
    )
    assert polluted_coverage.num_observations == clean_coverage.num_observations, (
        f"{len(behind_xyz)} points behind every camera changed the problem: "
        f"{clean_coverage.num_observations} "
        f"observations became {polluted_coverage.num_observations}"
    )
    assert polluted_coverage.num_tracks == clean_coverage.num_tracks, (
        f"{clean_coverage.num_tracks} tracks became {polluted_coverage.num_tracks}"
    )

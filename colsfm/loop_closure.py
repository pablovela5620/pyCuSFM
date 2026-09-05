"""Loop-closure candidate selection and verification — cuSFM's `RetrievalLoopAssociations`.

This is stage 3 of the plan of record: turn the retrieval hits of `colsfm.retrieval` into
verified `colsfm.pose_graph.PoseGraphEdge`s of kind `"loop"`. It reimplements
`AssociationWorker::RetrievalLoopAssociationsForFrame`
(`docs/spec/generate_association_main.md` §6.4) and
`PoseGraph::FindBestLoopCandidateForFrame` (`docs/spec/pose_graph_main.md` §6.4), which
are two views of the same pipeline, and deliberately departs from the blob in four
places — every one of them measured, every one documented below.

Pipeline per rig frame
----------------------

1. Query keyframes: the **left** camera of each declared stereo pair, matching both blob
   paths ("Skip, due to frame N is not a left camera frame"). Metadata that declares no
   stereo pair falls back to every camera.
2. Retrieval: `RetrievalIndex.query(top_k=20)` with a temporal gate (below), then the
   `good_score_threshold` gate.
3. Verification: `match_fn(image_a, image_b)` for the raw matches — injected so this
   module never depends on the matching worker — then
   `pycolmap.estimate_two_view_geometry` with `compute_relative_pose=True` on **calibrated**
   cameras.
4. `is_good`: `inliers > min_inliers and inliers / matches > min_inlier_ratio`
   (30 and 0.25, `docs/spec/generate_association_main.md` §6.6, §7.1 item 6).
5. Metric scale: rotation and translation *direction* from the two-view geometry, magnitude
   from the prior poses in `frames_meta` (§7.3 "Scale").
6. `select_best_candidates`: one candidate per 10 s band of `|dt|`, ranked by inlier count
   then retrieval score (§6.4, §7.1 item 5).
7. Rig conversion, one edge per unordered rig pair, then
   `colsfm.pose_graph.gate_loop_edges` with real thresholds.

The temporal gate
-----------------

The two blob binaries gate differently and neither number is usable as shipped:

* `generate_association_main` drops candidates with
  `|dt| < loop_interval_threshold_in_seconds` (10 s).
* `pose_graph_main` drops candidates with
  `|dt| < loop_closure_interval_ratio * session_duration_seconds` (0.08 x the session).

`minimum_time_gap_seconds` takes the **maximum** of the two, i.e. a candidate must pass
both. That is the conservative reading, and it is what a union of the two specs means for
a pipeline that runs the association stage and the pose graph back to back.

**Measured trap:** the shipped Galileo sequence lasts **0.933 s** end to end (226
keyframes, 29 rig frames, a 0.66 m sweep). The fixed 10 s gate therefore rejects every
candidate in the dataset, and only the ratio gate (0.08 x 0.933 = 0.075 s, about two rig
frames) is live. A caller that wants loops on Galileo must lower
`loop_interval_threshold_in_seconds`; `LoopClosureDiagnostics.min_time_gap_seconds`
reports what was actually applied.

What is deliberately not copied
-------------------------------

* `max_mean_point_to_epipolarline_error: 10e-6`. Measured on RoboCap it rejects 168 of 180
  candidates and the blob produces **zero** loop associations
  (`docs/spec/generate_association_main.md` §6.4). The gate here is
  `ransac_max_error_px` (4.0 px, COLMAP's own default), expressed in pixels.
* The four-view stereo estimator with baseline-locked scale, replaced by a plain two-view
  estimate plus prior-pose scale (§7.2).
* The `scale = 0.47` fallback for an out-of-range scale — the candidate is rejected instead.
* `loop_edge_translation_threshold_meters` / `loop_edge_rotation_threshold_degrees` default
  to **0.0** in every shipped config, which rejects every candidate and is why
  `pose_graph_main` never emits a loop edge out of the box. The defaults here are 3.0 m and
  30 deg (`docs/spec/pose_graph_main.md` §8 "Should fix").

Information matrices
--------------------

Identity by default. The blob writes `inliers * loop_residual_weight * I6` on the
association path and `(1/||t||) * loop_residual_weight * I6` on the pose-graph path, then
**overwrites both with the identity at solve time** because
`relative_constraint_use_pose_covariance` is false in every shipped config
(`docs/spec/pose_graph_main.md` §8 item 6). Measured against the blob's archived 8-loop
Galileo result, the identity lands 0.28 mm away and the stored weights 2.17 mm away — so
identity is the default and `LoopClosureConfig.use_stored_weights` is the documented
option, weighting by inlier count (the more standard of the blob's two schemes).

Measured on RoboCap: the retrieval is solved, the estimator is not
--------------------------------------------------------------------

Once `colsfm.retrieval`'s vocab backend gave this stage real candidates on RoboCap (4528
keyframes, 1132 rig frames, a 124 m walk), the whole funnel was run end to end against the
blob's own 90 LOOP edges. Retrieval and selection are **right**; the two-view estimator of
step 3-5 above is **not good enough**, and these are the numbers that show it.

Funnel, `good_score_threshold` 0.1, temporal gap 12.07 s (0.08 x the 150.8 s session):

| stage | count |
|---|---|
| query keyframes (left camera of the one stereo pair) | 1132 |
| retrieval hits after the temporal gate | 22 640 |
| dropped by `good_score_threshold` | 4 926 |
| dropped by `estimate_two_view_geometry` | 782 |
| dropped by `is_good` | 0 |
| dropped by the scale range | 5 |
| verified | 16 927 |
| after `select_best_candidates` | 1 729 |
| after one edge per rig pair | 1 585 |
| after `gate_loop_edges` (3.0 m / 30 deg) | 488 |

All **90** of the blob's LOOP pairs are covered by those 488 edges to within 2 rig frames
(and by the verified candidates too), so nothing true is being missed.

Pose graph, 1131 sequential edges from the input priors plus the loop edges, against the
blob's `vehicle_pose.tum` after rigid alignment:

| edges fed to `solve_pose_graph` | RMSE vs the blob's PGO |
|---|---|
| sequential only (the input trajectory) | 460.8 mm |
| sequential + the blob's own 90 LOOP edges | **135.0 mm** |
| sequential + **oracle** poses on our 488 rig pairs | **39.9 mm** |
| sequential + our 488 two-view + prior-scale edges | 609.0 mm |

The middle two rows are the controls that localise the fault. Our `sequential_edges`
reproduce the blob's CONSECUTIVE edges to **0.00 mm**, and feeding the blob's own loop
edges through `colsfm.pose_graph.solve_pose_graph` lands 135 mm from its result — so the
solver and the edge conventions are right. Replacing only the *measurement* on our own 488
rig pairs with the blob's optimised relative poses lands at 39.9 mm — so the **pair
selection is right too**. What is left is the measurement itself.

Why the measurement is weak, and what would fix it
--------------------------------------------------

`docs/spec/generate_association_main.md` §7.2 recommends replacing the blob's four-view
stereo estimator with "a plain 2-view essential-matrix estimate between query and
candidate; recover metric scale afterwards from the given poses". On RoboCap that
recommendation is **wrong**, and not for the reason it looks like:

* Taking the magnitude from the prior means the edge can never contradict the prior about
  distance — which is exactly the drift a loop closure exists to remove. Our edges have a
  median `||t||` of 0.417 m where the blob's have 0.157 m.
* The translation *direction* is the harder problem. A revisit baseline here is 0.2-0.4 m
  against a scene several metres deep, so the essential matrix barely constrains it. Our
  direction sits 30 deg from the prior's — but the prior and the blob's own PGO sit
  **44 deg** apart on the same pairs, so a few tens of degrees is simply the noise floor of
  a 0.3 m vector, not a bug.

Replacing the whole estimate with metric stereo triangulation plus PnP — triangulate the
query rig's own stereo matches against its **86 mm** known baseline, then
`pycolmap.estimate_and_refine_absolute_pose` of the candidate camera against those metric
points — was measured on the same 488 pairs and is the only variant that moves the
trajectory the way the blob's did:

| loop edge measurement | RMSE vs the blob's PGO |
|---|---|
| two-view + prior scale (what this module ships) | 609.0 mm (worse than the input) |
| stereo triangulation + PnP | **434.5 mm** (better than the input's 460.8 mm) |

It is not implemented here because 434.5 mm is still far from the 39.9 mm the same pairs
allow: an 86 mm baseline triangulating a scene metres deep leaves depth errors of the same
order as the loop translation itself. Closing the rest of the gap needs the blob's
refinement and its selectivity as well — `StereoPoseRefineSolver`, the occupied-area and
inlier gates of `docs/spec/generate_association_main.md` §6.4, and the ~8 % acceptance rate
that leaves it with 90 edges where this module keeps 488. That is the next piece of work,
and it is an estimator problem, not a retrieval one.

**Consequence for callers today:** on a long sequence this stage now produces hundreds of
loop edges where it used to produce none, and on RoboCap they make the pose graph
measurably worse. `LoopClosureConfig.enabled` stays **False**, and a caller that turns it
on must measure the trajectory before and after rather than assume loops help.

Everything is behind `LoopClosureConfig.enabled`, which is **False**. On Galileo, turning
loops on moves camera positions by 5-13 mm against a 5 mm ATE budget
(`docs/spec/pose_graph_main.md` §8): on a short, low-drift sequence they can easily hurt.
The pipeline decides per dataset.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Float32, Float64, Int, UInt32
from numpy import ndarray

from colsfm.cameras import colmap_cameras
from colsfm.frames_meta import CameraParams, FramesMeta, KeyframeMeta, RigFrame
from colsfm.pose_graph import Information6, PoseGraphEdge, gate_loop_edges, loop_edge_information
from colsfm.retrieval import Candidate, RetrievalIndex

Matches: TypeAlias = Int[ndarray, "n_matches 2"]
"""Raw feature matches: keypoint index in the first image, keypoint index in the second."""

MatchFunction: TypeAlias = Callable[[int, int], Matches]
"""`match_fn(image_id_a, image_id_b) -> matches`, injected by the caller.

The production implementation is LightGlue (TensorRT engine or torch); the tests use a
synthetic one. Keeping it a callable is what stops this module depending on the matching
stage, which another worker owns.
"""

Keypoints: TypeAlias = Float64[ndarray, "n_keypoints 2"]
"""Keypoint pixel coordinates, `x` then `y`, in the original image."""

MICROSECONDS_PER_SECOND: Final[float] = 1e6
"""Timestamps are integer microseconds everywhere in `frames_meta.json`."""


@dataclass(frozen=True, slots=True)
class LoopClosureConfig:
    """Gates and thresholds for the loop-closure stage."""

    enabled: bool = False
    """Master switch. Off because loops measurably move Galileo by 5-13 mm against a 5 mm
    ATE budget (`docs/spec/pose_graph_main.md` §8); they should matter much more on
    RoboCap. Validate per dataset before turning this on."""
    top_k: int = 20
    """`query_result_number`: retrieval hits considered per query keyframe."""
    good_score_threshold: float | None = None
    """Minimum retrieval score, or None to take the index's own backend default.

    The two `colsfm.retrieval` backends produce different quantities — a
    mutual-nearest-neighbour vote ratio and the blob's DBoW2 L1 score — so a single
    constant would be right for at most one of them. Both were calibrated separately and
    both landed on 0.1; `colsfm.retrieval.default_good_score_threshold` carries the
    measurements. Set this to override, for instance to 0.15 on the vocab backend, which
    halves the pairs handed to the matcher at 95 % rather than 88 % precision."""
    loop_interval_threshold_in_seconds: float = 10.0
    """`association_config.loop_interval_threshold_in_seconds`. Note this exceeds the whole
    0.933 s Galileo session, so it must be lowered to get any candidate there."""
    loop_closure_interval_ratio: float = 0.08
    """`pose_graph_config.loop_closure_interval_ratio`, multiplied by the session duration."""
    candidate_band_seconds: float = 10.0
    """Width of the `|dt|` band `select_best_candidates` collapses to one candidate. The
    blob reuses `loop_interval_threshold_in_seconds` here; it is separate so the temporal
    gate can be lowered without also merging every candidate into one band."""
    min_inliers: int = 30
    """`min_matches_num`: a good loop edge needs strictly more inliers than this."""
    min_inlier_ratio: float = 0.25
    """`min_matches_ratio`: inliers / raw matches must be strictly greater than this."""
    ransac_max_error_px: float = 4.0
    """RANSAC inlier threshold in pixels. Replaces the blob's unusable 1e-5 normalised-plane
    epipolar gate; 4.0 px is COLMAP's own `TwoViewGeometryOptions` default."""
    ransac_confidence: float = 0.999
    """RANSAC confidence, COLMAP's default and the blob's `min_ransac_confidence` rounded up."""
    min_scale_meters: float = 0.02
    """Smallest prior baseline that yields a usable translation direction. Below this the
    two views are effectively co-located and the essential matrix is degenerate."""
    max_scale_meters: float = 50.0
    """`scale_upper_limit`. A prior baseline above this means the retrieval matched across
    a whole session and the candidate is rejected rather than rescaled to the blob's 0.47."""
    max_translation_m: float = 3.0
    """`loop_edge_translation_threshold_meters`, applied by `gate_loop_edges`. The shipped
    isaac config leaves it at 0.0, which rejects everything; `data/cusfm_configs/
    loop-closure-fixed` repairs it to 1.0.

    Measured on the blob's own 90 RoboCap LOOP edges, the relative rig translation runs
    0.007-0.962 m with a median of 0.157 m — right up against its own 1.0 m gate. 3.0 m is
    kept as the default so a slightly worse prior pose does not silently drop a true
    revisit; the geometric verification, not this gate, is what rejects a bad pair."""
    max_rotation_deg: float = 30.0
    """`loop_edge_rotation_threshold_degrees`, applied by `gate_loop_edges`. Also 0.0 in
    the shipped isaac config, and 10.0 in `loop-closure-fixed`.

    The blob's 90 RoboCap LOOP edges rotate by 2.5-12.0 deg, median 7.2 — so its own 10 deg
    gate is already binding on its own output. 30 deg leaves headroom for the same reason
    as the translation gate."""
    use_stored_weights: bool = False
    """When True the edge information is `loop_edge_information(inliers, loop_residual_weight)`
    instead of the identity. The blob's own solver discards the stored value; measured
    against its archived Galileo result the identity is 8x closer."""
    loop_residual_weight: float = 0.25
    """`pose_graph_config.loop_residual_weight`; only read when `use_stored_weights`."""
    query_left_cameras_only: bool = True
    """Query with the left camera of each declared stereo pair, as both blob paths do.
    Metadata declaring no stereo pair queries every camera regardless."""


@dataclass(frozen=True, slots=True)
class LoopCandidate:
    """One retrieval candidate that survived two-view verification."""

    query_keyframe_id: int
    """Keyframe the query was issued from."""
    candidate_keyframe_id: int
    """Retrieved keyframe."""
    source_rig_id: int
    """`synced_sample_id` of the query keyframe's rig frame."""
    target_rig_id: int
    """`synced_sample_id` of the candidate keyframe's rig frame."""
    score: float
    """Retrieval score in `[0, 1]`."""
    delta_seconds: float
    """`t_query - t_candidate` in seconds; `SelectBestCandidates` bands on its magnitude."""
    num_matches: int
    """Raw matches `match_fn` returned."""
    num_inliers: int
    """Matches the two-view geometry kept."""
    source_T_target: pycolmap.Rigid3d
    """Rig-frame relative pose, `world_T_source^-1 * world_T_target`, metrically scaled."""


@dataclass(frozen=True, slots=True)
class LoopClosureDiagnostics:
    """Where the candidates went, so a run can be explained without re-running it."""

    enabled: bool
    """Whether `LoopClosureConfig.enabled` let the stage run at all."""
    session_duration_seconds: float
    """`max(timestamp) - min(timestamp)` over the keyframes, in seconds."""
    min_time_gap_seconds: float
    """The temporal gap actually applied: `max(fixed threshold, ratio * session duration)`."""
    good_score_threshold: float
    """The retrieval score gate actually applied, after the index's backend default was
    resolved. Reported because the two backends' scores are not the same quantity."""
    queries: int
    """Query keyframes the stage issued a retrieval for."""
    candidates_retrieved: int
    """Retrieval hits returned across all queries, before any gate."""
    rejected_by_score: int
    """Hits dropped by `good_score_threshold`."""
    rejected_by_time: int
    """Hits dropped by the temporal gate."""
    rejected_by_same_rig: int
    """Hits whose rig frame is the query's own; an intra-rig pair is an extrinsic edge."""
    rejected_no_matches: int
    """Hits where `match_fn` returned too few matches to estimate a geometry."""
    rejected_by_geometry: int
    """Hits whose two-view geometry was not `CALIBRATED` or recovered no relative pose."""
    rejected_by_is_good: int
    """Hits that failed `inliers > min_inliers and inliers / matches > min_inlier_ratio`."""
    rejected_by_scale: int
    """Hits whose prior baseline fell outside `[min_scale_meters, max_scale_meters]`."""
    verified: int
    """Hits that produced a metric relative pose."""
    after_banding: int
    """Candidates left after `select_best_candidates` over every query."""
    after_deduplication: int
    """Edges left after keeping the best candidate per unordered rig pair."""
    edges: int
    """Edges left after `gate_loop_edges`, i.e. what the caller receives."""


@dataclass(frozen=True, slots=True)
class LoopClosureResult:
    """What `find_loop_edges` produced.

    The edges are the payload; the diagnostics exist because a loop-closure stage that
    silently returns nothing is indistinguishable from one that is switched off.
    """

    edges: list[PoseGraphEdge]
    """Verified `"loop"` edges, ready for `colsfm.pose_graph.solve_pose_graph`."""
    diagnostics: LoopClosureDiagnostics
    """Per-stage counts for the run."""
    candidates: list[LoopCandidate] = field(default_factory=list)
    """Every verified candidate, before banding and gating. Kept for inspection."""


# ======================================================================================
# gates
# ======================================================================================


def is_good_match(num_inliers: int, num_matches: int, config: LoopClosureConfig) -> bool:
    """`DetermineValidityViaInlierMatches` for a LOOP edge (spec §6.6).

    `is_good = matched_num > min_matches_num && matched_num / matches.size() > min_matches_ratio`,
    i.e. 30 inliers and a 25 % inlier ratio, both strict.

    Args:
        num_inliers: Matches the geometric verification kept.
        num_matches: Raw matches the matcher produced.

    Returns:
        True when the pair may become a loop edge.
    """
    if num_matches <= 0:
        return False
    return num_inliers > config.min_inliers and num_inliers / num_matches > config.min_inlier_ratio


def session_duration_seconds(timestamps_us: Sequence[int]) -> float:
    """`GetSessionDurationInSecond`: the span of the whole capture.

    Args:
        timestamps_us: Keyframe timestamps in microseconds.

    Returns:
        `(max - min) / 1e6`, or 0.0 for fewer than two timestamps.
    """
    if len(timestamps_us) < 2:
        return 0.0
    return (max(timestamps_us) - min(timestamps_us)) / MICROSECONDS_PER_SECOND


def minimum_time_gap_seconds(timestamps_us: Sequence[int], config: LoopClosureConfig) -> float:
    """Combine the association and pose-graph temporal gates.

    `generate_association_main` uses a fixed `loop_interval_threshold_in_seconds`;
    `pose_graph_main` uses `loop_closure_interval_ratio * session_duration`. A candidate
    must pass both, so the effective gap is the larger of the two. See the module docstring
    for why the fixed 10 s is unusable on Galileo.

    Args:
        timestamps_us: Keyframe timestamps in microseconds.
        config: The loop-closure settings.

    Returns:
        The minimum `|dt|` in seconds a candidate must have from its query.
    """
    return max(config.loop_interval_threshold_in_seconds, config.loop_closure_interval_ratio * session_duration_seconds(timestamps_us))


def select_best_candidates(candidates: Sequence[LoopCandidate], band_seconds: float) -> list[LoopCandidate]:
    """`SelectBestCandidates`: keep one candidate per band of `|dt|` (spec §6.4).

    Sorts ascending by `|dt|` and walks greedily: a candidate more than `band_seconds`
    beyond the last emitted one opens a new band, otherwise it replaces the last emitted
    one when it has more inliers, or the same inliers and a higher retrieval score. This is
    what stops a single revisit producing 20 near-duplicate loop edges (spec §7.1 item 5).

    Args:
        candidates: Verified candidates for one query keyframe.
        band_seconds: Band width in seconds; the blob uses
            `loop_interval_threshold_in_seconds`.

    Returns:
        The kept candidates, in ascending `|dt|` order.
    """
    ordered: list[LoopCandidate] = sorted(candidates, key=lambda candidate: (abs(candidate.delta_seconds), candidate.candidate_keyframe_id))
    kept: list[LoopCandidate] = []
    last_gap_seconds: float = -np.inf
    for candidate in ordered:
        gap_seconds: float = abs(candidate.delta_seconds)
        if gap_seconds - last_gap_seconds > band_seconds:
            kept.append(candidate)
            last_gap_seconds = gap_seconds
            continue
        incumbent: LoopCandidate = kept[-1]
        better: bool = candidate.num_inliers > incumbent.num_inliers or (
            candidate.num_inliers == incumbent.num_inliers and candidate.score > incumbent.score
        )
        if better:
            kept[-1] = candidate
    return kept


# ======================================================================================
# verification
# ======================================================================================


def _left_camera_ids(frames_meta: FramesMeta, config: LoopClosureConfig) -> set[int]:
    """Camera ids allowed to issue a retrieval query.

    Args:
        frames_meta: The parsed metadata.
        config: The loop-closure settings.

    Returns:
        The left camera of every declared stereo pair, or every camera when the metadata
        declares no pair or `query_left_cameras_only` is off.
    """
    if not config.query_left_cameras_only or not frames_meta.stereo_pairs:
        return set(frames_meta.cameras)
    return {pair.left_camera_params_id for pair in frames_meta.stereo_pairs}


def calibrated_cameras(frames_meta: FramesMeta) -> dict[int, pycolmap.Camera]:
    """Build the `pycolmap.Camera`s the two-view estimator needs.

    `Camera.has_prior_focal_length` defaults to **False**, and with it false COLMAP takes
    the uncalibrated path: a pinhole pair degrades to a fundamental matrix with no relative
    pose, and a fisheye pair comes back `DEGENERATE` with zero inliers, silently
    (`docs/spec/pycolmap-capabilities.md` §5). Every camera built here sets it.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        Calibrated cameras keyed by `camera_params_id`.
    """
    cameras: dict[int, pycolmap.Camera] = colmap_cameras(frames_meta)
    for camera in cameras.values():
        camera.has_prior_focal_length = True
    return cameras


def _two_view_options(config: LoopClosureConfig) -> pycolmap.TwoViewGeometryOptions:
    """Options for `estimate_two_view_geometry`.

    `compute_relative_pose` defaults to False, so without setting it `cam2_from_cam1` is
    None even on a perfectly good calibrated pair.

    Args:
        config: The loop-closure settings.

    Returns:
        The options, with the RANSAC threshold in pixels.
    """
    options: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    options.compute_relative_pose = True
    options.min_num_inliers = config.min_inliers
    options.ransac.max_error = config.ransac_max_error_px
    options.ransac.confidence = config.ransac_confidence
    return options


def estimate_relative_camera_pose(
    camera_query: pycolmap.Camera,
    keypoints_query: Keypoints,
    camera_candidate: pycolmap.Camera,
    keypoints_candidate: Keypoints,
    matches: Matches,
    config: LoopClosureConfig,
) -> tuple[pycolmap.Rigid3d, int] | None:
    """Verify one pair and return `cam_candidate_T_cam_query` with a unit translation.

    Args:
        camera_query: Calibrated camera of the query keyframe.
        keypoints_query: Float64 keypoint pixels of the query, shape `[n_keypoints, 2]`.
        camera_candidate: Calibrated camera of the candidate keyframe.
        keypoints_candidate: Float64 keypoint pixels of the candidate, `[n_keypoints, 2]`.
        matches: Int match indices with shape `[n_matches, 2]`.
        config: The loop-closure settings.

    Returns:
        The direction-only relative pose and the inlier count, or None when the geometry is
        not calibrated or recovers no pose.
    """
    match_pairs: UInt32[ndarray, "n_matches 2"] = np.ascontiguousarray(matches, dtype=np.uint32)
    geometry: pycolmap.TwoViewGeometry = pycolmap.estimate_two_view_geometry(
        camera_query,
        np.ascontiguousarray(keypoints_query, dtype=np.float64),
        camera_candidate,
        np.ascontiguousarray(keypoints_candidate, dtype=np.float64),
        match_pairs,
        _two_view_options(config),
    )
    if geometry.config != pycolmap.TwoViewGeometryConfiguration.CALIBRATED:
        return None
    if geometry.cam2_from_cam1 is None:
        return None
    return geometry.cam2_from_cam1, len(geometry.inlier_matches)


def _metric_relative_rig_pose(
    direction_only: pycolmap.Rigid3d,
    query: KeyframeMeta,
    candidate: KeyframeMeta,
    cameras: Mapping[int, CameraParams],
    config: LoopClosureConfig,
) -> pycolmap.Rigid3d | None:
    """Scale a two-view estimate with the prior poses and move it into the rig frame.

    The essential matrix fixes rotation and translation *direction* only. The blob recovers
    the magnitude from the known stereo baseline through a four-view estimator because it
    refuses to trust the prior pose; with `init_pose_mode: GIVEN` the prior is right there,
    so the magnitude is `||prior_cam_candidate_T_cam_query.translation||`
    (`docs/spec/generate_association_main.md` §7.3).

    The rig conversion is `docs/spec/pose_graph_main.md` §6.7:
    `rig_source_T_rig_target = vehicle_T_cam(query) * cam_query_T_cam_candidate *
    vehicle_T_cam(candidate)^-1`, which is exactly `world_T_rig_query^-1 *
    world_T_rig_candidate` when the estimate agrees with the priors.

    Args:
        direction_only: `cam_candidate_T_cam_query` with a unit-norm translation.
        query: The query keyframe's metadata.
        candidate: The candidate keyframe's metadata.
        cameras: Calibration and rig extrinsic per `camera_params_id`.
        config: The loop-closure settings.

    Returns:
        The metric rig-frame relative pose, or None when the prior baseline is degenerate
        or implausibly long.
    """
    prior_candidate_T_query: pycolmap.Rigid3d = candidate.world_T_cam.inverse() * query.world_T_cam
    scale_meters: float = float(np.linalg.norm(prior_candidate_T_query.translation))
    if not config.min_scale_meters <= scale_meters <= config.max_scale_meters:
        return None

    unit_translation: Float64[ndarray, "3"] = np.asarray(direction_only.translation, dtype=np.float64)
    scaled: pycolmap.Rigid3d = pycolmap.Rigid3d(direction_only.rotation, unit_translation * scale_meters)
    cam_query_T_cam_candidate: pycolmap.Rigid3d = scaled.inverse()
    vehicle_T_cam_query: pycolmap.Rigid3d = cameras[query.camera_params_id].vehicle_T_cam
    vehicle_T_cam_candidate: pycolmap.Rigid3d = cameras[candidate.camera_params_id].vehicle_T_cam
    return vehicle_T_cam_query * cam_query_T_cam_candidate * vehicle_T_cam_candidate.inverse()


def _read_keypoints(database_path: Path, image_ids: Sequence[int]) -> dict[int, Keypoints]:
    """Read the `x, y` columns of every image's keypoints out of a COLMAP database.

    `write_keypoints` stores whatever column count it is handed — Nx2, Nx4 and Nx6 are all
    legal — so only the first two columns are read (`docs/spec/pycolmap-capabilities.md` §4).

    Args:
        database_path: Path to `database.db`.
        image_ids: Images to read.

    Returns:
        Float64 keypoint pixels per image id; images without a keypoint row are skipped.

    Raises:
        FileNotFoundError: When the database file does not exist.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    database: pycolmap.Database = pycolmap.Database.open(str(database_path))
    try:
        keypoints: dict[int, Keypoints] = {}
        for image_id in image_ids:
            if not database.exists_keypoints(image_id):
                continue
            stored: Float32[ndarray, "n_keypoints n_columns"] = database.read_keypoints(image_id)
            keypoints[image_id] = np.ascontiguousarray(stored[:, :2], dtype=np.float64)
        return keypoints
    finally:
        database.close()


# ======================================================================================
# the stage
# ======================================================================================


def find_loop_edges(
    frames_meta: FramesMeta,
    database_path: Path,
    index: RetrievalIndex,
    config: LoopClosureConfig,
    match_fn: MatchFunction,
) -> LoopClosureResult:
    """Retrieve, verify and convert loop candidates into rig-level pose-graph edges.

    Returns a `LoopClosureResult` rather than a bare list so the caller can report why a
    run produced no edge, which is the normal outcome on a short sequence.

    Args:
        frames_meta: Parsed `frames_meta.json`; supplies the rig grouping, the prior poses
            used for scale, the rig extrinsics and the calibration.
        database_path: COLMAP database holding the keypoints `match_fn`'s indices refer to.
        index: Retrieval index over the same keyframe ids.
        config: Gates and thresholds. Nothing runs unless `config.enabled`.
        match_fn: `match_fn(query_image_id, candidate_image_id) -> Int[ndarray, "m 2"]`.

    Returns:
        The gated loop edges and the diagnostics of the run.

    Raises:
        FileNotFoundError: When `database_path` does not exist and the stage is enabled.
    """
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    timestamps_us: dict[int, int] = {keyframe_id: keyframe.timestamp_microseconds for keyframe_id, keyframe in keyframe_by_id.items()}
    duration_seconds: float = session_duration_seconds(list(timestamps_us.values()))
    gap_seconds: float = minimum_time_gap_seconds(list(timestamps_us.values()), config)
    score_threshold: float = index.good_score_threshold if config.good_score_threshold is None else config.good_score_threshold

    counters: dict[str, int] = dict.fromkeys(
        (
            "queries",
            "candidates_retrieved",
            "rejected_by_score",
            "rejected_by_time",
            "rejected_by_same_rig",
            "rejected_no_matches",
            "rejected_by_geometry",
            "rejected_by_is_good",
            "rejected_by_scale",
            "verified",
            "after_banding",
            "after_deduplication",
        ),
        0,
    )

    def diagnostics(edge_count: int) -> LoopClosureDiagnostics:
        """Freeze the counters into the returned dataclass."""
        return LoopClosureDiagnostics(
            enabled=config.enabled,
            session_duration_seconds=duration_seconds,
            min_time_gap_seconds=gap_seconds,
            good_score_threshold=score_threshold,
            edges=edge_count,
            **counters,
        )

    if not config.enabled:
        print("Loop closure is disabled; set LoopClosureConfig.enabled to run it.")
        return LoopClosureResult(edges=[], diagnostics=diagnostics(0))

    query_camera_ids: set[int] = _left_camera_ids(frames_meta, config)
    cameras: dict[int, pycolmap.Camera] = calibrated_cameras(frames_meta)
    keypoints: dict[int, Keypoints] = _read_keypoints(database_path, index.image_ids)
    gap_us: int = round(gap_seconds * MICROSECONDS_PER_SECOND)

    indexed: set[int] = set(index.image_ids)
    verified: list[LoopCandidate] = []
    banded: list[LoopCandidate] = []
    for rig in frames_meta.rig_frames():
        for query_id in rig.keyframe_ids:
            query: KeyframeMeta = keyframe_by_id[query_id]
            if query.camera_params_id not in query_camera_ids or query_id not in keypoints or query_id not in indexed:
                continue
            counters["queries"] += 1
            # The blob applies the time gate AFTER retrieval, so temporal neighbours eat
            # slots out of its top-20 and it needs an early abort when more than half the
            # hits are too close (spec §6.4). Pushing the gate into the index instead keeps
            # all `top_k` slots useful; the ungated query is issued only to report how many
            # hits the gate would have consumed.
            for hit in index.query(query_id, top_k=config.top_k):
                if abs(timestamps_us[hit.image_id] - query.timestamp_microseconds) < gap_us:
                    counters["rejected_by_time"] += 1
            for_query: list[LoopCandidate] = []
            for candidate in index.query(query_id, top_k=config.top_k, min_time_gap_us=gap_us, timestamps=timestamps_us):
                counters["candidates_retrieved"] += 1
                found: LoopCandidate | None = _verify_candidate(
                    rig=rig,
                    query=query,
                    candidate=candidate,
                    keyframe_by_id=keyframe_by_id,
                    keypoints=keypoints,
                    cameras=cameras,
                    frames_meta=frames_meta,
                    config=config,
                    score_threshold=score_threshold,
                    match_fn=match_fn,
                    counters=counters,
                )
                if found is not None:
                    for_query.append(found)
            verified.extend(for_query)
            banded.extend(select_best_candidates(for_query, config.candidate_band_seconds))

    counters["after_banding"] = len(banded)
    best_per_rig_pair: dict[tuple[int, int], LoopCandidate] = {}
    for candidate in banded:
        key: tuple[int, int] = (min(candidate.source_rig_id, candidate.target_rig_id), max(candidate.source_rig_id, candidate.target_rig_id))
        incumbent: LoopCandidate | None = best_per_rig_pair.get(key)
        if incumbent is None or candidate.num_inliers > incumbent.num_inliers:
            best_per_rig_pair[key] = candidate
    counters["after_deduplication"] = len(best_per_rig_pair)

    edges: list[PoseGraphEdge] = []
    for key in sorted(best_per_rig_pair):
        candidate = best_per_rig_pair[key]
        information: Information6 = (
            loop_edge_information(float(candidate.num_inliers), config.loop_residual_weight)
            if config.use_stored_weights
            else np.eye(6, dtype=np.float64)
        )
        edges.append(
            PoseGraphEdge(
                source=candidate.source_rig_id,
                target=candidate.target_rig_id,
                source_T_target=candidate.source_T_target,
                information=information,
                kind="loop",
            )
        )
    gated: list[PoseGraphEdge] = gate_loop_edges(edges, max_translation_m=config.max_translation_m, max_rotation_deg=config.max_rotation_deg)
    return LoopClosureResult(edges=gated, diagnostics=diagnostics(len(gated)), candidates=verified)


def _verify_candidate(
    rig: RigFrame,
    query: KeyframeMeta,
    candidate: Candidate,
    keyframe_by_id: Mapping[int, KeyframeMeta],
    keypoints: Mapping[int, Keypoints],
    cameras: Mapping[int, pycolmap.Camera],
    frames_meta: FramesMeta,
    config: LoopClosureConfig,
    score_threshold: float,
    match_fn: MatchFunction,
    counters: dict[str, int],
) -> LoopCandidate | None:
    """Run one candidate through the score, rig, match, geometry, `is_good` and scale gates.

    Args:
        rig: Rig frame the query keyframe belongs to.
        query: The query keyframe's metadata.
        candidate: One retrieval hit.
        keyframe_by_id: Every keyframe, indexed by id.
        keypoints: Keypoint pixels per keyframe id.
        cameras: Calibrated `pycolmap.Camera` per `camera_params_id`.
        frames_meta: The parsed metadata, for the rig extrinsics.
        config: Gates and thresholds.
        score_threshold: The resolved `good_score_threshold` for the index's backend.
        match_fn: The injected matcher.
        counters: Diagnostic counters, mutated in place.

    Returns:
        The verified candidate, or None when any gate rejected it.
    """
    if candidate.score < score_threshold:
        counters["rejected_by_score"] += 1
        return None
    target: KeyframeMeta | None = keyframe_by_id.get(candidate.image_id)
    if target is None or candidate.image_id not in keypoints:
        counters["rejected_no_matches"] += 1
        return None
    if target.synced_sample_id == rig.synced_sample_id:
        counters["rejected_by_same_rig"] += 1
        return None

    matches: Matches = np.asarray(match_fn(query.keyframe_id, target.keyframe_id))
    if matches.ndim != 2 or matches.shape[0] <= config.min_inliers:
        counters["rejected_no_matches"] += 1
        return None

    estimate: tuple[pycolmap.Rigid3d, int] | None = estimate_relative_camera_pose(
        cameras[query.camera_params_id],
        keypoints[query.keyframe_id],
        cameras[target.camera_params_id],
        keypoints[target.keyframe_id],
        matches,
        config,
    )
    if estimate is None:
        counters["rejected_by_geometry"] += 1
        return None
    direction_only, num_inliers = estimate
    if not is_good_match(num_inliers, int(matches.shape[0]), config):
        counters["rejected_by_is_good"] += 1
        return None

    source_T_target: pycolmap.Rigid3d | None = _metric_relative_rig_pose(direction_only, query, target, frames_meta.cameras, config)
    if source_T_target is None:
        counters["rejected_by_scale"] += 1
        return None

    counters["verified"] += 1
    return LoopCandidate(
        query_keyframe_id=query.keyframe_id,
        candidate_keyframe_id=target.keyframe_id,
        source_rig_id=rig.synced_sample_id,
        target_rig_id=target.synced_sample_id,
        score=candidate.score,
        delta_seconds=(query.timestamp_microseconds - target.timestamp_microseconds) / MICROSECONDS_PER_SECOND,
        num_matches=int(matches.shape[0]),
        num_inliers=num_inliers,
        source_T_target=source_T_target,
    )

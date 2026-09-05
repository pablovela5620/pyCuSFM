"""Loop-closure candidate selection and verification — cuSFM's `RetrievalLoopAssociations`.

This is stage 3 of the plan of record: turn the retrieval hits of `colsfm.retrieval` into
verified `colsfm.pose_graph.PoseGraphEdge`s of kind `"loop"`. It reimplements
`AssociationWorker::RetrievalLoopAssociationsForFrame`
(`docs/spec/generate_association_main.md` §6.4) and
`PoseGraph::FindBestLoopCandidateForFrame` (`docs/spec/pose_graph_main.md` §6.4), which
are two views of the same pipeline, and deliberately departs from the blob in four
places — every one of them measured, every one documented below. The measurement itself
lives in `colsfm.loop_pose`; this module decides *which* rig frames to measure.

Pipeline per rig frame
----------------------

1. Query keyframes: the **left** camera of each declared stereo pair, matching both blob
   paths ("Skip, due to frame N is not a left camera frame"). Metadata that declares no
   stereo pair falls back to every camera.
2. Retrieval: `RetrievalIndex.query(top_k=20)` with a temporal gate (below), then the
   `good_score_threshold` gate and a same-rig-frame gate.
3. `select_best_candidates`: one hit per 10 s band of `|dt|` per query (§6.4, §7.1 item 5),
   then one rig pair overall, keeping the best retrieval score.
4. Measurement: `colsfm.loop_pose.estimate_rig_relative_pose` on each surviving rig pair —
   a metric point cloud triangulated in the source rig frame from its own cameras and its
   temporal neighbours, then `estimate_and_refine_generalized_absolute_pose` for the whole
   target rig. Its gates are `is_good`'s: `inliers > min_inliers` and
   `inliers / observations > min_inlier_ratio` (30 and 0.25,
   `docs/spec/generate_association_main.md` §6.6, §7.1 item 6), plus a cross-check of the
   translation direction against the generalized essential matrix.
5. One edge per rig pair, then `colsfm.pose_graph.gate_loop_edges` with real thresholds.

Steps 1-3 read nothing but the retrieval index and the timestamps. That is deliberate: the
pipeline learns which image pairs to match by running this whole stage once with a matcher
that returns nothing, so any pair whose request depends on a match result would never be
matched in production (`colsfm.pipeline._find_loop_edges`, `colsfm.loop_pose`). It is also
why the band in step 3 is ranked by retrieval score where the blob ranks by inlier count —
see `select_best_candidates`.

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
  `RigPoseConfig.ransac_max_error_px`, expressed in pixels.
* The four-view stereo estimator with baseline-locked scale, replaced by the local metric
  map and generalized resection of `colsfm.loop_pose` — a strictly larger version of the
  same idea, using every camera of the rig and its temporal neighbours rather than one
  stereo pair.
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

Measured on RoboCap: retrieval is solved, the measurement is better, the gap is elsewhere
------------------------------------------------------------------------------------------

RoboCap is 4528 fisheye keyframes, 1132 rig frames, four cameras pointing in three
directions, a 124 m walk over 150.8 s. The blob's own run emits **90** LOOP constraints and
a PGO output; everything below is scored against that output after rigid alignment.

Retrieval and pair selection were settled first, and they are right: the vocab backend's
gated top-20 covers all 90 of the blob's LOOP pairs, and replacing only the *measurement* on
our own 488 gated rig pairs with the blob's optimised relative poses lands **39.9 mm** from
its PGO, against 460.8 mm for the input trajectory alone. So the pair set can carry the
answer; what it is fed decides everything.

Estimators, on that one fixed 488-pair set:

| measurement of `source_T_target` | edges | translation error vs the oracle, median / p90 | rotation error, median | RMSE vs the blob's PGO |
|---|---|---|---|---|
| none (the input trajectory) | 0 | — | — | 460.8 mm |
| the oracle (the blob's own PGO) | 488 | 0 / 0 mm | 0.00 deg | **39.9 mm** |
| the input odometry's own relative pose | 488 | 383 / 799 mm | 6.79 deg | 460.8 mm |
| two-view essential + prior-pose scale (what this stage used to ship) | 488 | 346 / 809 mm | 7.30 deg | 609.0 mm |
| `loop_pose`, stereo pair only (`neighbour_span=0`) | 189 | 197 / 1140 mm | 6.63 deg | 457.0 mm |
| **`loop_pose`, the default (`neighbour_span=1`)** | 425 | 219 / 776 mm | 6.90 deg | **408.1 mm** |
| `loop_pose`, `neighbour_span=2` | 448 | 231 / 773 mm | 6.92 deg | 418.5 mm |
| `loop_pose`, no direction cross-check | 430 | 228 / 776 mm | 6.91 deg | 418.4 mm |
| `loop_pose`, direction cross-check at 25 deg | 414 | 227 / 779 mm | 6.85 deg | 407.6 mm |

The new estimator is a clear improvement on the old one — 609.0 mm to 408.1 mm, from worse
than no loops to better than no loops — and `neighbour_span=1` is the knee: the temporal
neighbours more than double the usable triangulation baseline over the 86 mm stereo pair,
and a second step adds nothing. But 408 mm is not 39.9 mm, and no gate closes the rest.
Every knob was swept on the same edges and every one is flat:

| selectivity or weighting | edges | RMSE vs the blob's PGO |
|---|---|---|
| all edges | 433 | 408.0 mm |
| only the 213 within 2 rig frames of a blob LOOP pair | 213 | 389.1 mm |
| `||t|| <= 0.5 m` / `<= 1 m` | 361 / 425 | 417.4 / 407.0 mm |
| inlier ratio `>= 0.30` / `>= 0.35` / `>= 0.40` | 339 / 274 / 206 | 413.0 / 412.3 / 419.1 mm |
| within 0.3 m / 0.5 m of the prior | 209 / 408 | 439.5 / 405.9 mm |
| loop information `0.3` / `0.1` / `0.01 * I6` | 433 | 408.5 / 409.9 / 413.4 mm |

Nothing here is worth 250 mm, which says the residual is not a few bad edges.

The whole stage, end to end
---------------------------

Everything above varies the measurement on one frozen pair set. Running the stage as
`colsfm.pipeline` runs it — retrieval, banding, batch matching, measurement, gating, with
the `data/cusfm_configs/loop-closure-fixed` profile's own 1.0 m / 10 deg edge gates:

| stage | count |
|---|---|
| query keyframes (left camera of the one stereo pair) | 1132 |
| retrieval hits after the 12.07 s temporal gate | 22 640 |
| dropped by `good_score_threshold` | 4 926 |
| after `select_best_candidates` | 1 747 |
| after one rig pair each | 1 573 |
| dropped by `is_good_match` | 586 |
| dropped by the direction cross-check | 66 |
| measured | 921 |
| **after `gate_loop_edges` (1.0 m / 10 deg)** | **74** |

14 443 image pairs are asked of the matcher, of which 6 057 were new — fewer than the
15 325 the old two-view funnel needed, because banding on the retrieval score shortlists rig
pairs before anything is matched. The measurement pass takes 437 s over 1573 rig pairs. Two
runs of the same code give 74 and 75 edges and 404.4 and 402.4 mm: both generalized
estimators are RANSAC and pycolmap's `set_random_seed` does not reach them, so a pair on the
inlier gate falls either way. The poses themselves are stable to 0.1 mm.

| edges fed to `solve_pose_graph` | RMSE vs the blob's PGO |
|---|---|
| sequential only | 460.8 mm |
| + our 74 loop edges | **404.4 mm** |
| + oracle poses on those same 74 rig pairs | 261.8 mm |
| + the blob's own 90 loop edges | 135.0 mm |

Two things to read off that table. The 74 edges are the *accurate* ones — their translation
error against the oracle is a median of 77 mm and their rotation error 3.80 deg, against
219 mm and 6.90 deg over the unfiltered 488 — so the funnel's selectivity works. But the
oracle on the same 74 pairs only reaches 261.8 mm, so this configuration is capped by its
**pair set**, not by its measurement: `loop_edge_rotation_threshold_degrees: 10` throws away
847 of the 921 measured pairs, and covers 56 of the blob's 90 loop pairs instead of all 90,
because a real RoboCap revisit rotates by a median of 11.8 deg and that gate was tuned
against an estimator whose rotations were smaller. `LoopClosureConfig` defaults to 30 deg
for exactly this reason; the config profile overrides it.

The gap is the rotation, and it is not ours
-------------------------------------------

Running the estimator on the blob's **own** 90 loop rig pairs compares two independent
measurements of the same quantity with no pair selection and no pose graph in between. We
measured 87 of them:

| quantity, median over the 87 pairs | ours | the input odometry | the blob's own LOOP edge |
|---|---|---|---|
| `||t||` of the edge | 132 mm | 355 mm | 160 mm |
| relative rotation angle | 11.8 deg | 11.8 deg | 7.3 deg |
| translation vs the blob's edge | 180 mm | 329 mm | — |
| rotation vs the blob's edge | 6.5 deg | 6.4 deg | — |

The **translation** result is the estimator working: the odometry says these revisits are
355 mm apart, the blob measured 160 mm, we measured 132 mm, and we cut the disagreement with
the blob's own number from 329 mm to 180 mm. A measurement that took its magnitude from the
prior could not do that.

The **rotation** is the whole remaining gap, and the blob is the outlier, not us:

* Two independent image-based estimators — the generalized resection of `colsfm.loop_pose`
  and `pycolmap.estimate_generalized_relative_pose`, which share their correspondences and
  nothing else — agree with each other to **0.35 deg** (p90 0.63 deg) and both sit
  **6.5 deg** from the blob's stored rotation and **1.5 deg** from the odometry's.
* The blob's PGO followed its own edges: its output's relative rotation sits 1.1 deg from
  its LOOP edges and 6.0 deg from the odometry's.
* Rebuilding our edges component by component: our rotation with the blob's translation
  scores 392.9 mm, the blob's rotation with our translation scores **265.6 mm**. All of the
  correction the blob's PGO applies is rotational.
* The convention is not the explanation. Parsed exactly as stored, the blob's 1131
  CONSECUTIVE constraints reproduce `sequential_edges` to **0.0000 mm and 0.0000 deg**;
  inverting them gives 228 mm and 7.9 deg. So `pose_source_to_target` is read correctly,
  rotation included, and the 6.5 deg is a real disagreement about the world.

A 6 deg rotation error is not something a resection with ~1900 inliers at an 8 px threshold
on a 630 px focal length can hide — the residuals would be tens of pixels. The odometry
agreeing with us is also the expected pattern for a visual-inertial front end, whose
orientation drifts far more slowly than its position. RoboCap ships **no ground truth**
(NOTES.md §"RoboCap reconstruction quality"), so this cannot be settled outright; what can
be said is that the reference this stage is scored against carries a rotation two
independent estimators here disagree with, and that scoring against it caps what any
measurement in this module can reach.

Where that leaves the stage
---------------------------

`LoopClosureConfig.enabled` stays **False**. The estimator is a real improvement — it beats
the old one by 200 mm and beats running no loops at all, on the frozen pair set and end to
end — but 402 mm against a 135 mm target is not a result worth switching on by default, and
the one number that would justify switching it on (agreement with something that is actually
ground truth) does not exist for this dataset. A caller who turns it on must measure the
trajectory before and after, and should raise `loop_edge_rotation_threshold_degrees` above
10 first, because that gate now costs more than it saves.

On Galileo the same switch stays off for a different reason: turning loops on moves camera
positions by 5-13 mm against a 5 mm ATE budget (`docs/spec/pose_graph_main.md` §8), because
a 0.93 s, 0.66 m sweep has no drift for a loop to remove. The pipeline decides per dataset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
from jaxtyping import Float32
from numpy import ndarray

from colsfm.cameras import colmap_cameras
from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame
from colsfm.loop_pose import (
    DEFAULT_RIG_POSE_CONFIG,
    Keypoints,
    MatchFunction,
    RigFrameIndex,
    RigGeometry,
    RigImageCache,
    RigLandmarks,
    RigPoseConfig,
    RigPoseEstimate,
    RigPoseOutcome,
    RigPoseRejection,
    build_image_cache,
    build_rig_geometry,
    estimate_rig_relative_pose,
    triangulate_rig_landmarks,
)
from colsfm.pose_graph import Information6, PoseGraphEdge, gate_loop_edges, loop_edge_information
from colsfm.retrieval import RetrievalIndex

MICROSECONDS_PER_SECOND: Final[float] = 1e6
"""Timestamps are integer microseconds everywhere in `frames_meta.json`."""

REJECTION_COUNTER: Final[Mapping[RigPoseRejection | None, str]] = {
    "no_observations": "rejected_no_matches",
    "no_pose": "rejected_by_geometry",
    "too_few_inliers": "rejected_by_is_good",  # the same rule `is_good_match` states
    "direction_disagreement": "rejected_by_direction",
    None: "rejected_by_geometry",
}
"""Which diagnostic counter each `colsfm.loop_pose` rejection lands in."""


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
    """`min_matches_num`: a good loop edge needs strictly more inliers than this. Here the
    inliers are the 2-D-3-D observations `colsfm.loop_pose`'s generalized resection kept."""
    min_inlier_ratio: float = 0.25
    """`min_matches_ratio`: inliers over the observations offered, strictly greater."""
    rig_pose: RigPoseConfig = DEFAULT_RIG_POSE_CONFIG
    """Settings of the metric rig-to-rig estimator in `colsfm.loop_pose`, which measures
    every surviving rig pair. `min_inliers` above overrides the copy in here and
    `min_inlier_ratio` has no copy at all, so the blob's own gate lives once, in
    `is_good_match`."""
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
class RetrievalHit:
    """One retrieval hit that passed the score, time and rig gates, before any matching.

    The funnel bands and deduplicates these, and only the survivors are handed to the
    matcher. Nothing here needs an image pair to have been matched, which is what lets the
    stage name every pair it will need in a single dry run (`find_loop_edges`).
    """

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
    """`t_query - t_candidate` in seconds; `select_best_candidates` bands on its magnitude."""


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
    """2-D-3-D observations the generalized PnP was offered."""
    num_inliers: int
    """2-D-3-D observations it kept."""
    num_landmarks: int
    """Landmarks in the source rig frame's local metric map."""
    direction_disagreement_deg: float
    """Angle between this translation direction and the generalized essential matrix's;
    `nan` when that cross-check was unavailable. See `colsfm.loop_pose`."""
    source_T_target: pycolmap.Rigid3d
    """Rig-frame relative pose, `world_T_source^-1 * world_T_target`, in metres."""


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
    after_banding: int
    """Hits left after `select_best_candidates` over every query."""
    after_deduplication: int
    """Rig pairs left after keeping the best-scoring hit per unordered rig pair; this is
    what the metric estimator is actually run on."""
    rejected_no_matches: int
    """Rig pairs whose local map and loop matches yielded too few 2-D-3-D observations."""
    rejected_by_geometry: int
    """Rig pairs the generalized PnP failed on outright."""
    rejected_by_is_good: int
    """Rig pairs that failed `inliers > min_inliers and inliers / observations > min_inlier_ratio`."""
    rejected_by_direction: int
    """Rig pairs whose refined translation direction disagreed with the generalized
    essential matrix's by more than `RigPoseConfig.max_direction_disagreement_deg`."""
    verified: int
    """Rig pairs that produced a metric relative pose."""
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

    The blob counts two-view feature matches; this stage counts the 2-D-3-D observations of
    `colsfm.loop_pose`'s generalized resection, which is the same quantity one stage further
    on — how much of what the matcher offered the geometry actually explained.

    Args:
        num_inliers: Observations the geometry kept.
        num_matches: Observations it was offered.

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


def select_best_candidates(hits: Sequence[RetrievalHit], band_seconds: float) -> list[RetrievalHit]:
    """`SelectBestCandidates`: keep one hit per band of `|dt|` (spec §6.4).

    Sorts ascending by `|dt|` and walks greedily: a hit more than `band_seconds` beyond the
    last emitted one opens a new band, otherwise it replaces the last emitted one when it
    scores higher. This is what stops a single revisit producing 20 near-duplicate loop
    edges (spec §7.1 item 5).

    **Departure from the blob, measured.** `SelectBestCandidates` ranks a band first by the
    geometric inlier count and only then by the retrieval score. Ranking by inliers needs
    every candidate matched *before* the band is chosen, which is 15 325 image pairs on
    RoboCap where the banded shortlist needs 6 000 — and the pipeline has to name every pair
    it will match in one dry run, before any match result exists (`find_loop_edges`). The
    retrieval score is the only ranking available at that point. Measured on RoboCap the two
    orders pick nearly the same pairs, because a band is a 10 s window of one revisit and its
    candidates are near-duplicates of each other.

    Args:
        hits: Retrieval hits for one query keyframe that passed the score and time gates.
        band_seconds: Band width in seconds; the blob uses
            `loop_interval_threshold_in_seconds`.

    Returns:
        The kept hits, in ascending `|dt|` order.
    """
    ordered: list[RetrievalHit] = sorted(hits, key=lambda hit: (abs(hit.delta_seconds), hit.candidate_keyframe_id))
    kept: list[RetrievalHit] = []
    last_gap_seconds: float = -np.inf
    for hit in ordered:
        gap_seconds: float = abs(hit.delta_seconds)
        if gap_seconds - last_gap_seconds > band_seconds:
            kept.append(hit)
            last_gap_seconds = gap_seconds
            continue
        if hit.score > kept[-1].score:
            kept[-1] = hit
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


def rig_geometry(frames_meta: FramesMeta) -> RigGeometry:
    """The rig the metric estimator resections against.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        Calibrated cameras, their `cam_T_vehicle` extrinsics and the declared stereo
        partnerships, in `colsfm.loop_pose`'s parallel-tuple form.
    """
    partners: dict[int, int] = {}
    for pair in frames_meta.stereo_pairs:
        partners[pair.left_camera_params_id] = pair.right_camera_params_id
        partners[pair.right_camera_params_id] = pair.left_camera_params_id
    return build_rig_geometry(
        calibrated_cameras(frames_meta),
        {camera_id: camera.vehicle_T_cam for camera_id, camera in frames_meta.cameras.items()},
        partners,
    )


def rig_frame_index(frames_meta: FramesMeta) -> RigFrameIndex:
    """The rig frames, their prior poses and their images, in time order.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        The index `colsfm.loop_pose` walks to find a rig frame's neighbours and images.
    """
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    rig_frames: tuple[RigFrame, ...] = frames_meta.rig_frames()
    keyframe_by_rig_camera: dict[tuple[int, int], int] = {}
    for rig in rig_frames:
        for keyframe_id in rig.keyframe_ids:
            keyframe_by_rig_camera.setdefault((rig.synced_sample_id, keyframe_by_id[keyframe_id].camera_params_id), keyframe_id)
    return RigFrameIndex(
        sequence=tuple(rig.synced_sample_id for rig in rig_frames),
        world_T_rig={rig.synced_sample_id: rig.world_T_vehicle for rig in rig_frames},
        keyframe_by_rig_camera=keyframe_by_rig_camera,
    )


def rig_pose_config(config: LoopClosureConfig) -> RigPoseConfig:
    """The estimator settings, with this stage's own inlier floor written over them.

    `min_inliers` is the blob's `min_matches_num` (spec §6.6) and belongs to the
    loop-closure stage, so it is kept in one place rather than duplicated in two configs
    that could drift apart. The ratio half of the rule stays here too, in `is_good_match`,
    which this stage applies to the counts the estimator reports.

    Args:
        config: The loop-closure settings.

    Returns:
        The `RigPoseConfig` the metric estimator actually runs with.
    """
    return replace(config.rig_pose, min_inliers=config.min_inliers)


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
    """Retrieve, shortlist, measure and gate loop candidates into rig-level pose-graph edges.

    The funnel is: retrieval hits, the score, time and same-rig gates, one hit per `|dt|`
    band per query, one rig pair overall, then `colsfm.loop_pose.estimate_rig_relative_pose`
    on each surviving rig pair and `gate_loop_edges` on the result. Everything before the
    estimator needs only the retrieval index and the timestamps, which is what lets a caller
    run the whole stage once with a matcher that returns nothing to learn the pair list, then
    match it, then run it again for real (`colsfm.pipeline._find_loop_edges`).

    Returns a `LoopClosureResult` rather than a bare list so the caller can report why a run
    produced no edge, which is the normal outcome on a short sequence.

    Args:
        frames_meta: Parsed `frames_meta.json`; supplies the rig grouping, the prior poses
            the local map is triangulated with, the rig extrinsics and the calibration.
        database_path: COLMAP database holding the keypoints `match_fn`'s indices refer to.
        index: Retrieval index over the same keyframe ids.
        config: Gates and thresholds. Nothing runs unless `config.enabled`.
        match_fn: `match_fn(image_id_a, image_id_b) -> Int[ndarray, "m 2"]`.

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
            "after_banding",
            "after_deduplication",
            "rejected_no_matches",
            "rejected_by_geometry",
            "rejected_by_is_good",
            "rejected_by_direction",
            "verified",
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

    shortlist: dict[tuple[int, int], RetrievalHit] = _shortlist_rig_pairs(
        frames_meta, index, config, keyframe_by_id, timestamps_us, gap_seconds, score_threshold, counters
    )
    candidates: list[LoopCandidate] = _measure_rig_pairs(frames_meta, database_path, index, config, shortlist, match_fn, counters)

    edges: list[PoseGraphEdge] = []
    for candidate in candidates:
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
    return LoopClosureResult(edges=gated, diagnostics=diagnostics(len(gated)), candidates=candidates)


def _shortlist_rig_pairs(
    frames_meta: FramesMeta,
    index: RetrievalIndex,
    config: LoopClosureConfig,
    keyframe_by_id: Mapping[int, KeyframeMeta],
    timestamps_us: Mapping[int, int],
    gap_seconds: float,
    score_threshold: float,
    counters: dict[str, int],
) -> dict[tuple[int, int], RetrievalHit]:
    """Retrieve, gate, band and deduplicate down to one hit per unordered rig pair.

    Args:
        frames_meta: The parsed metadata.
        index: The retrieval index.
        config: Gates and thresholds.
        keyframe_by_id: Every keyframe, indexed by id.
        timestamps_us: Capture time per keyframe id.
        gap_seconds: The temporal gate actually applied.
        score_threshold: The resolved retrieval score gate.
        counters: Diagnostic counters, mutated in place.

    Returns:
        The best-scoring hit per unordered rig pair, keyed by that pair.
    """
    query_camera_ids: set[int] = _left_camera_ids(frames_meta, config)
    gap_us: int = round(gap_seconds * MICROSECONDS_PER_SECOND)
    indexed: set[int] = set(index.image_ids)

    banded: list[RetrievalHit] = []
    for rig in frames_meta.rig_frames():
        for query_id in rig.keyframe_ids:
            query: KeyframeMeta = keyframe_by_id[query_id]
            if query.camera_params_id not in query_camera_ids or query_id not in indexed:
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
            for_query: list[RetrievalHit] = []
            for candidate in index.query(query_id, top_k=config.top_k, min_time_gap_us=gap_us, timestamps=timestamps_us):
                counters["candidates_retrieved"] += 1
                if candidate.score < score_threshold:
                    counters["rejected_by_score"] += 1
                    continue
                target: KeyframeMeta | None = keyframe_by_id.get(candidate.image_id)
                if target is None:
                    continue
                if target.synced_sample_id == rig.synced_sample_id:
                    counters["rejected_by_same_rig"] += 1
                    continue
                for_query.append(
                    RetrievalHit(
                        query_keyframe_id=query_id,
                        candidate_keyframe_id=target.keyframe_id,
                        source_rig_id=rig.synced_sample_id,
                        target_rig_id=target.synced_sample_id,
                        score=candidate.score,
                        delta_seconds=(query.timestamp_microseconds - target.timestamp_microseconds) / MICROSECONDS_PER_SECOND,
                    )
                )
            banded.extend(select_best_candidates(for_query, config.candidate_band_seconds))

    counters["after_banding"] = len(banded)
    shortlist: dict[tuple[int, int], RetrievalHit] = {}
    for hit in banded:
        key: tuple[int, int] = (min(hit.source_rig_id, hit.target_rig_id), max(hit.source_rig_id, hit.target_rig_id))
        incumbent: RetrievalHit | None = shortlist.get(key)
        if incumbent is None or hit.score > incumbent.score:
            shortlist[key] = hit
    counters["after_deduplication"] = len(shortlist)
    return shortlist


def _measure_rig_pairs(
    frames_meta: FramesMeta,
    database_path: Path,
    index: RetrievalIndex,
    config: LoopClosureConfig,
    shortlist: Mapping[tuple[int, int], RetrievalHit],
    match_fn: MatchFunction,
    counters: dict[str, int],
) -> list[LoopCandidate]:
    """Run the metric rig-to-rig estimator over the shortlisted rig pairs.

    Pairs are visited in source-rig order so that one source rig frame triangulates its
    local map once however many candidates it has.

    Args:
        frames_meta: The parsed metadata.
        database_path: COLMAP database holding the keypoints.
        index: The retrieval index, for the set of images that carry descriptors.
        config: Gates and thresholds.
        shortlist: The best hit per unordered rig pair.
        match_fn: The injected matcher.
        counters: Diagnostic counters, mutated in place.

    Returns:
        One verified candidate per rig pair that passed every gate, in rig-pair order.

    Raises:
        FileNotFoundError: When `database_path` does not exist.
    """
    keypoints: dict[int, Keypoints] = _read_keypoints(database_path, index.image_ids)
    geometry: RigGeometry = rig_geometry(frames_meta)
    rig_index: RigFrameIndex = rig_frame_index(frames_meta)
    pose_config: RigPoseConfig = rig_pose_config(config)
    images: RigImageCache = build_image_cache(rig_index, geometry, keypoints)

    candidates: list[LoopCandidate] = []
    local_map: RigLandmarks | None = None
    current_source: int | None = None
    for key in sorted(shortlist, key=lambda pair: (shortlist[pair].source_rig_id, pair)):
        hit: RetrievalHit = shortlist[key]
        if hit.source_rig_id != current_source:
            current_source = hit.source_rig_id
            local_map = triangulate_rig_landmarks(hit.source_rig_id, rig_index, geometry, keypoints, match_fn, pose_config, images)
        outcome: RigPoseOutcome = estimate_rig_relative_pose(
            hit.source_rig_id,
            hit.target_rig_id,
            index=rig_index,
            geometry=geometry,
            keypoints=keypoints,
            match_fn=match_fn,
            config=pose_config,
            landmarks=local_map,
            cache=images,
        )
        if outcome.estimate is None:
            counters[REJECTION_COUNTER[outcome.rejection]] += 1
            continue
        estimate: RigPoseEstimate = outcome.estimate
        if not is_good_match(estimate.num_inliers, estimate.num_observations, config):
            counters["rejected_by_is_good"] += 1
            continue
        counters["verified"] += 1
        candidates.append(
            LoopCandidate(
                query_keyframe_id=hit.query_keyframe_id,
                candidate_keyframe_id=hit.candidate_keyframe_id,
                source_rig_id=hit.source_rig_id,
                target_rig_id=hit.target_rig_id,
                score=hit.score,
                delta_seconds=hit.delta_seconds,
                num_matches=estimate.num_observations,
                num_inliers=estimate.num_inliers,
                num_landmarks=estimate.num_landmarks,
                direction_disagreement_deg=estimate.direction_disagreement_deg,
                source_T_target=estimate.source_T_target,
            )
        )
    return candidates



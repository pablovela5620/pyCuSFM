"""Metric rig-to-rig relative pose for a loop candidate — the measurement, not the retrieval.

`colsfm.loop_closure` decides *which* rig frames revisit each other; this module decides
*where* one is relative to the other, in metres. It is the piece the loop-closure docstring
measured as the whole remaining problem on RoboCap: retrieval and pair selection there are
right to 39.9 mm with oracle poses, and the shipped two-view-plus-prior-scale measurement
turns that into 609.0 mm — worse than no loops at all.

The design: a local metric map, then generalized absolute pose
---------------------------------------------------------------

For a candidate rig pair `(A, B)`:

1. **Local map.** Triangulate 3-D points expressed in rig frame `A`, from the images of `A`
   itself plus its temporal neighbours `A±1..A±span`. Every ray is placed by the rig
   extrinsics and by the *local* odometry between neighbouring rig frames, which is
   accurate — the drift a loop closure removes accumulates over a session, not over 100 ms.
2. **Generalized PnP.** Feed the 2-D observations those points get in every camera of `B`
   to `pycolmap.estimate_and_refine_generalized_absolute_pose`, which solves for one rig
   pose over all cameras at once and then refines it on every inlier observation. Its
   `rig_from_world` is `B_T_A`, so the edge is its inverse.

Nothing in step 2 reads the prior relation between `A` and `B`, so the edge can contradict
the odometry about distance — which is exactly what a loop closure is for, and exactly what
scaling a two-view direction by the prior baseline cannot do.

Why not the generalized essential matrix
----------------------------------------

`pycolmap.estimate_generalized_relative_pose` solves the same rig pair in one call, and
with non-zero rig baselines its scale is observable in principle. Measured on a synthetic
copy of RoboCap's rig (4 cameras, a 0.24 m span, a 0.36 m motion, a scene 10 m deep):

| keypoint noise | rotation error | recovered `||t||` (truth 0.359 m) |
|---|---|---|
| 0.0 px | 0.000 deg | 0.359 m |
| 0.2 px | 0.171 deg | **0.168 m** |
| 0.7 px | 0.084 deg | **0.217 m** |

The rotation survives; the scale does not. Metric scale in a generalized essential matrix
comes only from the rig baseline, so it is read off a 0.24 m lever arm against a scene tens
of metres deep, and half a pixel of noise halves it. Triangulating first spends the same
baseline on depth, where a redundant, refined least-squares problem averages the noise away
instead of amplifying it. The generalized essential matrix is still used, but only for its
rotation: `RigPoseConfig.max_direction_disagreement_deg` cross-checks the PnP translation
direction against it and drops a pair the two disagree about.

Depth, and why the neighbours matter
------------------------------------

RoboCap's declared stereo pair is 86 mm. A rig hop is 114 mm, so `A±1` alone more than
triples the available baseline and points it along a different axis, and the two side
cameras contribute a third view direction that the front pair never sees. Measured on the
same 488 RoboCap loop rig pairs, scored against the blob's PGO after rigid alignment
(`colsfm.loop_closure`'s docstring has the whole table):

| local map | edges | translation error vs the oracle, median | RMSE |
|---|---|---|---|
| the stereo pair alone (`neighbour_span=0`) | 189 | 197 mm | 457.0 mm |
| **plus `A±1` (the default)** | 425 | **219 mm** | **408.1 mm** |
| plus `A±2` | 448 | 231 mm | 418.5 mm |

`neighbour_span=0` measures fewer pairs because an 86 mm baseline against a scene metres
deep leaves too few landmarks to resection from: it loses 294 of the 488 pairs to the
inlier gate where the default loses 58. A second step of neighbours buys nothing and costs
one more image pair per camera in the matching stage.

What the estimator is and is not good at, measured on the blob's own 90 loop rig pairs:
its translations move decisively off the odometry and towards the blob's own measurement
(median `||t||` 132 mm against the odometry's 355 mm and the blob's 160 mm), while its
rotations agree with the odometry to 1.5 deg and with
`pycolmap.estimate_generalized_relative_pose` — an estimator that shares only the
correspondences — to **0.35 deg**. See `colsfm.loop_closure` for what that costs against a
reference whose own rotations sit 6.5 deg away.

Reading a `match_fn` the caller has not matched yet
---------------------------------------------------

`colsfm.pipeline` discovers which image pairs to match by running the whole stage once with
a matcher that returns nothing, batch-matching what was asked for, then running it again
for real. So this module must ask for **every** pair it could use *before* it gates on
anything: a pair requested only after a successful gate would never be matched in
production. `test_every_needed_image_pair_is_requested_even_when_nothing_matches` pins that.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64, Int
from numpy import ndarray

Matches: TypeAlias = Int[ndarray, "n_matches 2"]
"""Raw feature matches: keypoint index in the first image, keypoint index in the second."""

MatchFunction: TypeAlias = Callable[[int, int], Matches]
"""`match_fn(image_id_a, image_id_b) -> matches`, injected by the caller.

Column 0 indexes the keypoints of `image_id_a`, column 1 those of `image_id_b`, which is
what both `pycolmap.Database.read_matches` and the synthetic matchers in the tests give.
"""

Keypoints: TypeAlias = Float64[ndarray, "n_keypoints 2"]
"""Keypoint pixel coordinates, `x` then `y`, in the original image."""

Bearings: TypeAlias = Float64[ndarray, "n_keypoints 3"]
"""Unit-norm bearing vectors, one per keypoint, in the camera's optical frame."""

RigLandmarks: TypeAlias = dict[int, dict[int, Float64[ndarray, "3"]]]
"""Local map: 3-D points in the rig frame, keyed by anchor image id then keypoint index."""

RigPoseRejection: TypeAlias = Literal["no_observations", "no_pose", "too_few_inliers", "direction_disagreement"]
"""Which gate turned a rig pair down; `RigPoseOutcome.rejection` carries one of these.

The loop-closure stage reports these verbatim in its diagnostics, because a stage that
returns no edge is only explainable if it says which gate emptied it."""


@dataclass(frozen=True, slots=True)
class RigPoseConfig:
    """Gates and thresholds of the metric rig-to-rig estimator."""

    neighbour_span: int = 1
    """Temporal neighbours of the source rig frame that contribute rays to the local map.

    0 triangulates from the rig's own cameras only, which on RoboCap means an 86 mm stereo
    baseline against a scene metres deep. 1 adds `A±1`, i.e. two more views 114 mm away
    along the drive direction, and is the default. Each extra step costs one more image
    pair per camera in the matching stage."""
    min_depth_meters: float = 0.4
    """Landmarks nearer than this to the anchor camera are triangulation failures, not
    scene structure."""
    max_depth_meters: float = 60.0
    """Landmarks beyond this carry no usable depth at a 0.2 m baseline; keeping them only
    lets a near-parallel ray pair dominate the point count."""
    min_triangulation_angle_deg: float = 0.5
    """Smallest angle between the rays of a landmark. COLMAP's own mapper uses 1.5 deg for
    initialisation; 0.5 deg is looser because the rig's baseline is short by construction
    and the generalized PnP re-weights a distant point by its reprojection error anyway."""
    max_landmark_reprojection_px: float = 4.0
    """A landmark must reproject into every contributing view within this, which is the
    matching stage's own RANSAC threshold."""
    ransac_max_error_px: float = 8.0
    """RANSAC inlier threshold of the generalized PnP, in pixels. Twice the matcher's 4.0 px
    because a 2-D-3-D residual carries the landmark's depth error as well as the keypoint's."""
    ransac_confidence: float = 0.9999
    """RANSAC confidence; pycolmap's own `RANSACOptions` default."""
    ransac_max_trials: int = 10_000
    """Trial cap for both RANSAC loops.

    A bare `pycolmap.RANSACOptions()` allows **100 000**, and the generalized essential
    matrix's minimal solver is expensive enough that a degenerate pair then spends tens of
    seconds proving it cannot fit — measured, it is what a first run of this stage on
    RoboCap spends most of its time on. 10 000 is what COLMAP's own
    `TwoViewGeometryOptions` uses."""
    ransac_min_trials: int = 100
    """Trials RANSAC runs before its confidence test may stop it; COLMAP's own
    `TwoViewGeometryOptions` value."""
    min_inliers: int = 30
    """Fewest inliers a solved pose may rest on.

    This is a solvability floor, not the loop-closure stage's acceptance rule: whether an
    edge is *good* is `colsfm.loop_closure.is_good_match`, which owns the blob's
    `min_matches_num` and `min_matches_ratio` (`docs/spec/generate_association_main.md`
    §6.6) and applies them to the counts this estimator reports."""
    min_cameras: int = 1
    """Cameras of the target rig frame that must contribute at least one inlier-eligible
    observation. 1 accepts a revisit seen by one camera only, which is the common case when
    a rig passes an old place facing a different way."""
    max_direction_disagreement_deg: float = 45.0
    """How far the refined translation direction may sit from the generalized essential
    matrix's before the pair is dropped.

    The two estimates share their correspondences but nothing else — one triangulates and
    resections, the other solves an epipolar constraint — so agreement is real evidence and
    disagreement means one of them latched onto a wrong RANSAC support set. 45 deg is loose
    on purpose: on a 0.2 m revisit baseline the direction is the weakest quantity either
    estimator produces (`colsfm.loop_closure` measured the prior and the blob's own result
    44 deg apart on the same pairs), so a tight gate would reject true loops. Set it above
    180.0 to switch the cross-check off."""
    cross_stereo_observations: bool = True
    """Also match the left camera of the source against the right camera of the target, and
    vice versa. The two cameras of a declared stereo pair see the same scene 86 mm apart, so
    the cross pairs are real extra observations rather than duplicates of the same view."""


DEFAULT_RIG_POSE_CONFIG: RigPoseConfig = RigPoseConfig()
"""Shared immutable default, so the signatures below hold no constructor call."""


@dataclass(frozen=True, slots=True)
class RigGeometry:
    """The rig itself: what the two generalized estimators take as their fixed arguments.

    `cams_from_rig` and `cameras` are parallel to `camera_ids`, and a camera's position in
    those tuples is the `camera_idx` both pycolmap estimators expect.
    """

    camera_ids: tuple[int, ...]
    """cuSFM `camera_params_id` per rig sensor, ascending."""
    cams_from_rig: tuple[pycolmap.Rigid3d, ...]
    """`cam_T_vehicle` per camera, i.e. the inverse of the metadata's `vehicle_T_cam`; the
    same convention `colsfm.reconstruction.build_rig` gives COLMAP."""
    cameras: tuple[pycolmap.Camera, ...]
    """Calibrated `pycolmap.Camera` per camera, `has_prior_focal_length` set."""
    stereo_partner: Mapping[int, int] = field(default_factory=dict)
    """The other camera of a declared stereo pair, per `camera_params_id`. Empty when the
    metadata declares no pair, in which case the local map triangulates from the temporal
    neighbours alone."""

    def position_of(self, camera_id: int) -> int:
        """Index of one camera in the parallel tuples.

        Args:
            camera_id: cuSFM `camera_params_id`.

        Returns:
            Its `camera_idx` for the generalized estimators.

        Raises:
            KeyError: When the rig holds no such camera.
        """
        try:
            return self.camera_ids.index(camera_id)
        except ValueError as error:
            raise KeyError(f"camera {camera_id} is not part of this rig") from error


@dataclass(frozen=True, slots=True)
class RigFrameIndex:
    """Rig frames in time order, their prior poses and their member images."""

    sequence: tuple[int, ...]
    """Rig ids in capture order; `neighbours` walks this, not the id arithmetic, because
    `synced_sample_id` need not be contiguous."""
    world_T_rig: Mapping[int, pycolmap.Rigid3d]
    """Prior rig pose per rig id. Only *relative* poses inside a neighbourhood are read, so
    session-scale drift in these is exactly what the loop edge is allowed to contradict."""
    keyframe_by_rig_camera: Mapping[tuple[int, int], int]
    """Image id per `(rig id, camera_params_id)`; a camera may be absent from a rig frame."""

    def keyframe(self, rig_id: int, camera_id: int) -> int | None:
        """The image one camera contributed to one rig frame.

        Args:
            rig_id: The rig frame's `synced_sample_id`.
            camera_id: cuSFM `camera_params_id`.

        Returns:
            The image id, or None when that camera did not fire in that rig frame.
        """
        return self.keyframe_by_rig_camera.get((rig_id, camera_id))

    def neighbours(self, rig_id: int, span: int) -> tuple[int, ...]:
        """Rig frames within `span` steps of one rig frame, in time order.

        Args:
            rig_id: The rig frame to look around.
            span: How many steps to reach in each direction.

        Returns:
            The neighbouring rig ids, excluding `rig_id` itself; empty when `span` is 0 or
            the rig id is unknown.

        Raises:
            ValueError: When `span` is negative.
        """
        if span < 0:
            raise ValueError(f"neighbour span must not be negative, got {span}")
        try:
            position: int = self.sequence.index(rig_id)
        except ValueError:
            return ()
        lower: int = max(0, position - span)
        upper: int = min(len(self.sequence), position + span + 1)
        return tuple(self.sequence[step] for step in range(lower, upper) if step != position)


@dataclass(frozen=True, slots=True)
class RigPoseEstimate:
    """One measured rig-to-rig relative pose and the evidence behind it."""

    source_T_target: pycolmap.Rigid3d
    """Metric relative pose: where the target rig frame sits in the source's own frame."""
    num_inliers: int
    """2-D-3-D observations the generalized PnP kept."""
    num_observations: int
    """2-D-3-D observations it was offered."""
    num_landmarks: int
    """Landmarks in the source rig's local map, over every anchor image."""
    num_cameras: int
    """Cameras of the target rig frame that contributed an observation."""
    direction_disagreement_deg: float
    """Angle between this translation direction and the generalized essential matrix's, or
    `nan` when that estimate was unavailable (too few correspondences, or a degenerate
    configuration pycolmap answered with a panoramic model instead)."""


@dataclass(frozen=True, slots=True)
class RigPoseOutcome:
    """What one rig pair produced: an estimate, or the name of the gate that refused it.

    Exactly one of the two fields is set. The rejection is returned rather than folded into
    a bare None because the loop-closure stage's diagnostics have to say *why* a pair was
    dropped — "no edges" and "no edges because every pair was too far from its epipolar
    direction" are different bugs.
    """

    estimate: RigPoseEstimate | None = None
    """The measurement, or None when a gate rejected the pair."""
    rejection: RigPoseRejection | None = None
    """The gate that rejected the pair, or None when there is an estimate."""


def build_rig_geometry(cameras: Mapping[int, pycolmap.Camera], vehicle_T_cam: Mapping[int, pycolmap.Rigid3d], stereo_partner: Mapping[int, int]) -> RigGeometry:
    """Assemble the fixed arguments of the generalized estimators.

    Args:
        cameras: Calibrated `pycolmap.Camera` per `camera_params_id`.
        vehicle_T_cam: Rig extrinsic per `camera_params_id`, camera to vehicle.
        stereo_partner: The other camera of a declared stereo pair, per camera id.

    Returns:
        The rig geometry, cameras ordered by ascending id.

    Raises:
        KeyError: When a camera has no extrinsic.
    """
    camera_ids: tuple[int, ...] = tuple(sorted(cameras))
    return RigGeometry(
        camera_ids=camera_ids,
        cams_from_rig=tuple(vehicle_T_cam[camera_id].inverse() for camera_id in camera_ids),
        cameras=tuple(cameras[camera_id] for camera_id in camera_ids),
        stereo_partner=dict(stereo_partner),
    )


def _ransac_options(config: RigPoseConfig) -> pycolmap.RANSACOptions:
    """RANSAC settings shared by both generalized estimators.

    Args:
        config: The estimator settings.

    Returns:
        The options, with the trial count capped; see `RigPoseConfig.ransac_max_trials`.
    """
    options: pycolmap.RANSACOptions = pycolmap.RANSACOptions()
    options.max_error = config.ransac_max_error_px
    options.confidence = config.ransac_confidence
    options.min_num_trials = config.ransac_min_trials
    options.max_num_trials = config.ransac_max_trials
    return options


def bearings_of(camera: pycolmap.Camera, keypoints: Keypoints) -> Bearings:
    """Turn keypoint pixels into unit bearing vectors in the camera's optical frame.

    Args:
        camera: The calibrated camera, whose model undoes the distortion.
        keypoints: Float64 keypoint pixels with shape `[n_keypoints, 2]`.

    Returns:
        Float64 unit bearings with shape `[n_keypoints, 3]`.
    """
    normalised: Float64[ndarray, "n_keypoints 2"] = np.asarray(camera.cam_from_img(np.ascontiguousarray(keypoints, dtype=np.float64)), dtype=np.float64)
    homogeneous: Bearings = np.concatenate([normalised, np.ones((len(normalised), 1), dtype=np.float64)], axis=1)
    return homogeneous / np.linalg.norm(homogeneous, axis=1, keepdims=True)


@dataclass(slots=True)
class RigImageCache:
    """Per-image lookups the estimator would otherwise redo for every candidate pair.

    Bearings are the expensive part — one camera-model unprojection per keypoint per image —
    and a rig frame with twenty loop candidates would otherwise unproject its neighbourhood
    twenty times. Build one of these per run and hand it to every call.
    """

    geometry: RigGeometry
    """The rig, for the camera model of each image."""
    keypoints: Mapping[int, Keypoints]
    """Keypoint pixels per image id."""
    camera_of_image: Mapping[int, int]
    """`camera_params_id` per image id."""
    rig_of_image: Mapping[int, int]
    """`synced_sample_id` per image id."""
    bearings: dict[int, Bearings] = field(default_factory=dict, init=False)
    """Memoised unit bearings per image id."""

    def of(self, image_id: int) -> Bearings:
        """Unit bearings of every keypoint of one image.

        Args:
            image_id: The image to unproject.

        Returns:
            Float64 unit bearings with shape `[n_keypoints, 3]`.
        """
        cached: Bearings | None = self.bearings.get(image_id)
        if cached is None:
            camera: pycolmap.Camera = self.geometry.cameras[self.geometry.position_of(self.camera_of_image[image_id])]
            cached = bearings_of(camera, self.keypoints[image_id])
            self.bearings[image_id] = cached
        return cached


def build_image_cache(index: RigFrameIndex, geometry: RigGeometry, keypoints: Mapping[int, Keypoints]) -> RigImageCache:
    """Invert the rig-frame index into the per-image lookups the estimator needs.

    Args:
        index: The rig frames.
        geometry: The rig.
        keypoints: Keypoint pixels per image id.

    Returns:
        An empty cache that knows the camera and the rig frame of every indexed image.
    """
    return RigImageCache(
        geometry=geometry,
        keypoints=keypoints,
        camera_of_image={image_id: camera_id for (_, camera_id), image_id in index.keyframe_by_rig_camera.items()},
        rig_of_image={image_id: rig_id for (rig_id, _), image_id in index.keyframe_by_rig_camera.items()},
    )


def _local_map_partners(rig_id: int, camera_id: int, index: RigFrameIndex, geometry: RigGeometry, config: RigPoseConfig) -> list[int]:
    """Images whose rays join one anchor image's in the local map.

    Args:
        rig_id: The source rig frame.
        camera_id: The anchor image's camera.
        index: The rig frames.
        geometry: The rig.
        config: The estimator settings.

    Returns:
        Image ids of the stereo partner inside the rig frame and of the same camera in every
        temporal neighbour, in a stable order.
    """
    partners: list[int] = []
    partner_camera: int | None = geometry.stereo_partner.get(camera_id)
    if partner_camera is not None:
        stereo_image: int | None = index.keyframe(rig_id, partner_camera)
        if stereo_image is not None:
            partners.append(stereo_image)
    for neighbour in index.neighbours(rig_id, config.neighbour_span):
        neighbour_image: int | None = index.keyframe(neighbour, camera_id)
        if neighbour_image is not None:
            partners.append(neighbour_image)
    return partners


def triangulate_rig_landmarks(
    rig_id: int,
    index: RigFrameIndex,
    geometry: RigGeometry,
    keypoints: Mapping[int, Keypoints],
    match_fn: MatchFunction,
    config: RigPoseConfig = DEFAULT_RIG_POSE_CONFIG,
    cache: RigImageCache | None = None,
) -> RigLandmarks:
    """Triangulate a metric point cloud in one rig frame's own vehicle frame.

    Every camera of the rig frame anchors its own landmarks: a keypoint of the anchor image
    becomes a 3-D point once at least one partner image — the declared stereo partner, or
    the same camera in a temporal neighbour — matches it. Poses come from the rig extrinsics
    and from the *local* odometry between neighbouring rig frames, never from any relation
    to a far-away rig frame.

    Args:
        rig_id: The rig frame to build the map around.
        index: Rig frames, their prior poses and their images.
        geometry: The rig's cameras and extrinsics.
        keypoints: Keypoint pixels per image id.
        match_fn: The injected matcher.
        config: Gates and thresholds.
        cache: Shared per-image cache; a private one is built when None.

    Returns:
        3-D points in the rig frame, keyed by anchor image id then anchor keypoint index.
        Every camera of the rig frame gets an entry, possibly empty.
    """
    images: RigImageCache = cache if cache is not None else build_image_cache(index, geometry, keypoints)
    world_T_rig: pycolmap.Rigid3d = index.world_T_rig[rig_id]
    landmarks: RigLandmarks = {}
    for camera_id in geometry.camera_ids:
        anchor: int | None = index.keyframe(rig_id, camera_id)
        if anchor is None or anchor not in keypoints:
            continue
        anchor_from_rig: pycolmap.Rigid3d = geometry.cams_from_rig[geometry.position_of(camera_id)]
        anchor_projection: Float64[ndarray, "3 4"] = np.asarray(anchor_from_rig.matrix(), dtype=np.float64)
        anchor_bearings: Bearings = images.of(anchor)
        anchor_focal: float = focal_length_of(images, anchor)

        tracks: dict[int, list[tuple[int, int]]] = {}
        projections: dict[int, Float64[ndarray, "3 4"]] = {}
        for partner in _local_map_partners(rig_id, camera_id, index, geometry, config):
            partner_matches: Matches = np.asarray(match_fn(anchor, partner), dtype=np.int64).reshape(-1, 2)
            if partner not in keypoints or len(partner_matches) == 0:
                continue
            partner_camera: int = images.camera_of_image[partner]
            partner_rig: int = images.rig_of_image[partner]
            rig_T_partner_rig: pycolmap.Rigid3d = world_T_rig.inverse() * index.world_T_rig[partner_rig]
            projections[partner] = np.asarray(
                (geometry.cams_from_rig[geometry.position_of(partner_camera)] * rig_T_partner_rig.inverse()).matrix(), dtype=np.float64
            )
            for anchor_index, partner_index in partner_matches:
                tracks.setdefault(int(anchor_index), []).append((partner, int(partner_index)))

        # Group the tracks by which images they were seen in: every track in a group shares
        # one stack of projection matrices, so the whole group triangulates in one batched
        # least-squares call instead of one pybind call per landmark. A four-camera rig with
        # two neighbours has at most a handful of distinct view sets, so the grouping is
        # cheap and the batches are large.
        grouped: dict[tuple[int, ...], list[int]] = {}
        for anchor_index, observations in tracks.items():
            grouped.setdefault(tuple(image_id for image_id, _ in observations), []).append(anchor_index)

        points: dict[int, Float64[ndarray, "3"]] = {}
        for view_set, anchor_indices in grouped.items():
            stacked_projections: Float64[ndarray, "n_views 3 4"] = np.stack([anchor_projection, *(projections[image_id] for image_id in view_set)])
            view_focals: Float64[ndarray, "n_views"] = np.array(
                [anchor_focal, *(focal_length_of(images, image_id) for image_id in view_set)], dtype=np.float64
            )
            observed_indices: dict[int, dict[int, int]] = {}
            for anchor_index in anchor_indices:
                for image_id, index_in_partner in tracks[anchor_index]:
                    observed_indices.setdefault(image_id, {})[anchor_index] = index_in_partner
            batch_bearings: Float64[ndarray, "n_points n_views 3"] = np.stack(
                [
                    anchor_bearings[anchor_indices],
                    *(images.of(image_id)[[observed_indices[image_id][anchor_index] for anchor_index in anchor_indices]] for image_id in view_set),
                ],
                axis=1,
            )
            triangulated, keep = _triangulate_batch(stacked_projections, batch_bearings, view_focals, config)
            for position, anchor_index in enumerate(anchor_indices):
                if keep[position]:
                    points[anchor_index] = triangulated[position]
        landmarks[anchor] = points
    return landmarks


def focal_length_of(images: RigImageCache, image_id: int) -> float:
    """Mean focal length in pixels of the camera that took one image.

    Args:
        images: The per-image cache, which knows each image's camera.
        image_id: The image to look up.

    Returns:
        The camera's mean focal length in pixels; RoboCap's fisheyes sit near 630, Galileo's
        pinholes near 845.
    """
    camera: pycolmap.Camera = images.geometry.cameras[images.geometry.position_of(images.camera_of_image[image_id])]
    return float(camera.mean_focal_length())


def _triangulate_batch(
    projections: Float64[ndarray, "n_views 3 4"],
    bearings: Float64[ndarray, "n_points n_views 3"],
    focal_lengths: Float64[ndarray, "n_views"],
    config: RigPoseConfig,
) -> tuple[Float64[ndarray, "n_points 3"], Bool[ndarray, "n_points"]]:
    """Triangulate a batch of landmarks seen in the same views, then gate them.

    The linear system is COLMAP's own: for a bearing `b` seen through the 3x4 projection `P`,
    `b x (P X) = 0` gives two independent rows, and the point is the right singular vector
    of the stacked rows with the smallest singular value. Doing it batched is what makes the
    stage affordable — a rig frame anchors a few thousand landmarks and a per-landmark
    `pycolmap.triangulate_multi_view_point` call costs about a second per rig frame.

    Args:
        projections: `cam_T_rig` as a 3x4 matrix per view; view 0 is the anchor camera.
        bearings: Unit bearing per landmark per view, `[n_points, n_views, 3]`.
        focal_lengths: Mean focal length in pixels per view, which turns the angular
            residual of a bearing back into the pixels `max_landmark_reprojection_px` is
            expressed in.
        config: Gates and thresholds.

    Returns:
        The points in the rig frame, `[n_points, 3]`, and a mask of which passed the depth,
        triangulation-angle and reprojection gates.
    """
    rotations: Float64[ndarray, "n_views 3 3"] = projections[:, :, :3]
    translations: Float64[ndarray, "n_views 3"] = projections[:, :, 3]
    rows: Float64[ndarray, "n_points n_rows 4"] = np.concatenate(
        [
            bearings[:, :, 0:1] * projections[None, :, 2, :] - bearings[:, :, 2:3] * projections[None, :, 0, :],
            bearings[:, :, 1:2] * projections[None, :, 2, :] - bearings[:, :, 2:3] * projections[None, :, 1, :],
        ],
        axis=1,
    )
    # A fisheye keypoint outside its model's valid domain unprojects to `nan`, and LAPACK's
    # SVD raises rather than returning one — measured on RoboCap, where a handful of
    # keypoints per image sit past the OPENCV_FISHEYE undistortion's convergence radius. The
    # offending rows are dropped before the solve rather than after it.
    solvable: Bool[ndarray, "n_points"] = np.isfinite(rows).all(axis=(1, 2))
    points: Float64[ndarray, "n_points 3"] = np.zeros((len(bearings), 3), dtype=np.float64)
    keep: Bool[ndarray, "n_points"] = np.zeros(len(bearings), dtype=bool)
    if not solvable.any():
        return points, keep
    null_space: Float64[ndarray, "n_solvable 4"] = np.linalg.svd(rows[solvable])[2][:, -1, :]
    homogeneous: Float64[ndarray, "n_solvable"] = null_space[:, 3]
    usable: Bool[ndarray, "n_solvable"] = np.abs(homogeneous) > 1e-12
    points[solvable] = null_space[:, :3] / np.where(usable, homogeneous, 1.0)[:, None]
    keep[solvable] = usable
    keep &= np.isfinite(points).all(axis=1)

    in_views: Float64[ndarray, "n_points n_views 3"] = np.einsum("vij,nj->nvi", rotations, points) + translations[None, :, :]
    depths: Float64[ndarray, "n_points n_views"] = in_views[:, :, 2]
    keep &= (depths[:, 0] > config.min_depth_meters) & (depths[:, 0] < config.max_depth_meters) & (depths.min(axis=1) > 0.0)

    directions: Float64[ndarray, "n_points n_views 3"] = in_views / np.maximum(np.linalg.norm(in_views, axis=2, keepdims=True), 1e-12)
    residuals_px: Float64[ndarray, "n_points n_views"] = np.linalg.norm(directions - bearings, axis=2) * focal_lengths[None, :]
    keep &= residuals_px.max(axis=1) <= config.max_landmark_reprojection_px

    # The widest angle any two views see the point from, computed in the rig frame: a ray
    # direction expressed in its own camera's frame is not comparable with another camera's.
    centres: Float64[ndarray, "n_views 3"] = -np.einsum("vji,vj->vi", rotations, translations)
    from_centres: Float64[ndarray, "n_points n_views 3"] = points[:, None, :] - centres[None, :, :]
    from_centres = from_centres / np.maximum(np.linalg.norm(from_centres, axis=2, keepdims=True), 1e-12)
    cosines: Float64[ndarray, "n_points n_views n_views"] = np.einsum("nvi,nwi->nvw", from_centres, from_centres)
    widest_angle_deg: Float64[ndarray, "n_points"] = np.rad2deg(np.arccos(np.clip(cosines.min(axis=(1, 2)), -1.0, 1.0)))
    keep &= widest_angle_deg >= config.min_triangulation_angle_deg
    return points, keep


def _landmark_arrays(points: Mapping[int, Float64[ndarray, "3"]]) -> tuple[Int[ndarray, "n_landmarks"], Float64[ndarray, "n_landmarks 3"]]:
    """Turn one anchor image's landmarks into a sorted index array and a point array.

    Args:
        points: 3-D points in the rig frame, keyed by anchor keypoint index.

    Returns:
        The ascending keypoint indices and the matching points, so a match list can be
        joined against them with one `searchsorted` instead of a dict lookup per match.
    """
    indices: Int[ndarray, "n_landmarks"] = np.fromiter(sorted(points), dtype=np.int64, count=len(points))
    return indices, np.ascontiguousarray([points[int(index)] for index in indices], dtype=np.float64).reshape(-1, 3)


@dataclass(frozen=True, slots=True)
class _Observations:
    """The 2-D-3-D correspondences one rig pair offers the generalized resection."""

    points2d: Float64[ndarray, "n_observations 2"]
    """Pixels in the target rig's images."""
    points3d: Float64[ndarray, "n_observations 3"]
    """The matching landmarks, in the source rig's frame."""
    camera_indices: list[int]
    """`camera_idx` of the target camera each observation was seen in."""
    cameras: set[int]
    """Which cameras of the target rig frame contributed."""


def _observations(
    source_rig_id: int,
    target_rig_id: int,
    landmarks: RigLandmarks,
    index: RigFrameIndex,
    geometry: RigGeometry,
    keypoints: Mapping[int, Keypoints],
    match_fn: MatchFunction,
    config: RigPoseConfig,
) -> _Observations:
    """Match the source rig's anchor images against the target's and pair 2-D with 3-D.

    Every pair is asked of `match_fn` unconditionally, so a caller probing for the pair list
    with a matcher that returns nothing still sees all of them (module docstring). The join
    between a pair's matches and the anchor's landmarks is a `searchsorted`, not a lookup per
    match: a RoboCap rig pair offers a few thousand matches over six image pairs, and doing
    that in Python is most of the stage's runtime.

    Args:
        source_rig_id: Rig frame the landmarks belong to.
        target_rig_id: Rig frame being localised.
        landmarks: The source rig's local map.
        index: The rig frames.
        geometry: The rig.
        keypoints: Keypoint pixels per image id.
        match_fn: The injected matcher.
        config: Gates and thresholds.

    Returns:
        The correspondences, empty when nothing matched.
    """
    pixel_blocks: list[Keypoints] = []
    point_blocks: list[Float64[ndarray, "n_hits 3"]] = []
    camera_indices: list[int] = []
    contributing: set[int] = set()
    for camera_id in geometry.camera_ids:
        anchor: int | None = index.keyframe(source_rig_id, camera_id)
        if anchor is None:
            continue
        target_cameras: list[int] = [camera_id]
        partner_camera: int | None = geometry.stereo_partner.get(camera_id) if config.cross_stereo_observations else None
        if partner_camera is not None:
            target_cameras.append(partner_camera)
        anchor_points: dict[int, Float64[ndarray, "3"]] = landmarks.get(anchor, {})
        landmark_indices: Int[ndarray, "n_landmarks"] | None = None
        landmark_points: Float64[ndarray, "n_landmarks 3"] | None = None
        for target_camera in target_cameras:
            observed: int | None = index.keyframe(target_rig_id, target_camera)
            if observed is None or observed not in keypoints:
                continue
            pair_matches: Matches = np.asarray(match_fn(anchor, observed), dtype=np.int64).reshape(-1, 2)
            if len(pair_matches) == 0 or not anchor_points:
                continue
            if landmark_indices is None or landmark_points is None:
                landmark_indices, landmark_points = _landmark_arrays(anchor_points)
            insertion: Int[ndarray, "n_matches"] = np.searchsorted(landmark_indices, pair_matches[:, 0])
            clipped: Int[ndarray, "n_matches"] = np.clip(insertion, 0, len(landmark_indices) - 1)
            matched: Bool[ndarray, "n_matches"] = landmark_indices[clipped] == pair_matches[:, 0]
            hits: int = int(matched.sum())
            if hits == 0:
                continue
            pixel_blocks.append(keypoints[observed][pair_matches[matched, 1]])
            point_blocks.append(landmark_points[clipped[matched]])
            camera_indices.extend([geometry.position_of(target_camera)] * hits)
            contributing.add(target_camera)
    if not point_blocks:
        return _Observations(np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float64), [], set())
    return _Observations(
        points2d=np.ascontiguousarray(np.concatenate(pixel_blocks), dtype=np.float64),
        points3d=np.ascontiguousarray(np.concatenate(point_blocks), dtype=np.float64),
        camera_indices=camera_indices,
        cameras=contributing,
    )


def estimate_generalized_rig_pose(
    source_rig_id: int,
    target_rig_id: int,
    index: RigFrameIndex,
    geometry: RigGeometry,
    keypoints: Mapping[int, Keypoints],
    match_fn: MatchFunction,
    config: RigPoseConfig = DEFAULT_RIG_POSE_CONFIG,
) -> pycolmap.Rigid3d | None:
    """Solve the generalized essential matrix over every same-camera match of two rig frames.

    Its rotation and translation *direction* are trustworthy; its magnitude is not, which is
    why the caller only cross-checks a direction against it (module docstring).

    Args:
        source_rig_id: The first rig frame.
        target_rig_id: The second rig frame.
        index: The rig frames.
        geometry: The rig.
        keypoints: Keypoint pixels per image id.
        match_fn: The injected matcher.
        config: Gates and thresholds.

    Returns:
        `source_T_target` with an unreliable magnitude, or None when there are too few
        correspondences or pycolmap answers with a panoramic (pure-rotation) model.
    """
    source_pixels: list[Keypoints] = []
    target_pixels: list[Keypoints] = []
    source_indices: list[int] = []
    target_indices: list[int] = []
    for camera_id in geometry.camera_ids:
        source_image: int | None = index.keyframe(source_rig_id, camera_id)
        target_image: int | None = index.keyframe(target_rig_id, camera_id)
        if source_image is None or target_image is None or source_image not in keypoints or target_image not in keypoints:
            continue
        pair_matches: Matches = np.asarray(match_fn(source_image, target_image), dtype=np.int64).reshape(-1, 2)
        if len(pair_matches) == 0:
            continue
        position: int = geometry.position_of(camera_id)
        source_pixels.append(keypoints[source_image][pair_matches[:, 0]])
        target_pixels.append(keypoints[target_image][pair_matches[:, 1]])
        source_indices.extend([position] * len(pair_matches))
        target_indices.extend([position] * len(pair_matches))
    if len(source_indices) < config.min_inliers:
        return None

    answer: dict | None = pycolmap.estimate_generalized_relative_pose(
        np.ascontiguousarray(np.vstack(source_pixels), dtype=np.float64),
        np.ascontiguousarray(np.vstack(target_pixels), dtype=np.float64),
        source_indices,
        target_indices,
        list(geometry.cams_from_rig),
        list(geometry.cameras),
        _ransac_options(config),
    )
    # pycolmap answers a configuration whose rig baseline cannot fix the scale with a
    # `pano2_from_pano1` key instead, i.e. a pure-rotation model; there is no direction in it.
    if answer is None or "rig2_from_rig1" not in answer:
        return None
    return answer["rig2_from_rig1"].inverse()


def estimate_rig_relative_pose(
    source_rig_id: int,
    target_rig_id: int,
    index: RigFrameIndex,
    geometry: RigGeometry,
    keypoints: Mapping[int, Keypoints],
    match_fn: MatchFunction,
    config: RigPoseConfig = DEFAULT_RIG_POSE_CONFIG,
    landmarks: RigLandmarks | None = None,
    cache: RigImageCache | None = None,
) -> RigPoseOutcome:
    """Measure where one rig frame sits relative to another, in metres.

    Builds the source rig frame's local metric map (or reuses the one given), matches every
    anchor image against the target rig frame's cameras, and resections the target rig as a
    whole with `pycolmap.estimate_and_refine_generalized_absolute_pose`. No relation between
    the two rig frames is read from the priors, so the result can contradict them.

    Args:
        source_rig_id: Rig frame the edge starts at.
        target_rig_id: Rig frame the edge ends at.
        index: Rig frames, their prior poses and their images.
        geometry: The rig's cameras and extrinsics.
        keypoints: Keypoint pixels per image id.
        match_fn: The injected matcher.
        config: Gates and thresholds.
        landmarks: A local map for `source_rig_id` built earlier, or None to build it here.
            Passing one is what stops a source rig frame with twenty candidates
            triangulating its neighbourhood twenty times.
        cache: Shared per-image cache; a private one is built when None.

    Returns:
        The metric relative pose and its evidence, or the name of the gate that rejected the
        pair.
    """
    images: RigImageCache = cache if cache is not None else build_image_cache(index, geometry, keypoints)
    local_map: RigLandmarks = (
        triangulate_rig_landmarks(source_rig_id, index, geometry, keypoints, match_fn, config, images) if landmarks is None else landmarks
    )
    observations: _Observations = _observations(source_rig_id, target_rig_id, local_map, index, geometry, keypoints, match_fn, config)
    num_landmarks: int = sum(len(points) for points in local_map.values())
    num_observations: int = len(observations.points3d)
    if num_observations < config.min_inliers or len(observations.cameras) < config.min_cameras:
        return RigPoseOutcome(rejection="no_observations")

    answer: dict | None = pycolmap.estimate_and_refine_generalized_absolute_pose(
        observations.points2d,
        observations.points3d,
        observations.camera_indices,
        list(geometry.cams_from_rig),
        list(geometry.cameras),
        _ransac_options(config),
        pycolmap.AbsolutePoseRefinementOptions(),
    )
    if answer is None:
        return RigPoseOutcome(rejection="no_pose")
    num_inliers: int = int(answer["num_inliers"])
    if num_inliers < config.min_inliers:
        return RigPoseOutcome(rejection="too_few_inliers")
    source_T_target: pycolmap.Rigid3d = answer["rig_from_world"].inverse()

    epipolar: pycolmap.Rigid3d | None = (
        estimate_generalized_rig_pose(source_rig_id, target_rig_id, index, geometry, keypoints, match_fn, config)
        if config.max_direction_disagreement_deg <= 180.0
        else None
    )
    disagreement_deg: float = float("nan") if epipolar is None else direction_angle_deg(source_T_target, epipolar)
    if not np.isnan(disagreement_deg) and disagreement_deg > config.max_direction_disagreement_deg:
        return RigPoseOutcome(rejection="direction_disagreement")

    return RigPoseOutcome(
        estimate=RigPoseEstimate(
            source_T_target=source_T_target,
            num_inliers=num_inliers,
            num_observations=num_observations,
            num_landmarks=num_landmarks,
            num_cameras=len(observations.cameras),
            direction_disagreement_deg=disagreement_deg,
        )
    )


def direction_angle_deg(first: pycolmap.Rigid3d, second: pycolmap.Rigid3d) -> float:
    """Angle between the translation directions of two poses.

    Args:
        first: The first pose.
        second: The second pose.

    Returns:
        The angle in degrees, or `nan` when either translation is degenerate.
    """
    a: Float64[ndarray, "3"] = np.asarray(first.translation, dtype=np.float64)
    b: Float64[ndarray, "3"] = np.asarray(second.translation, dtype=np.float64)
    norms: float = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norms <= 1e-12:
        return float("nan")
    return float(np.rad2deg(np.arccos(np.clip(float(a @ b) / norms, -1.0, 1.0))))

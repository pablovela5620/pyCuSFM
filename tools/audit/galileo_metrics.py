"""Trajectory metrics shared by the paper-claims audit scripts.

The `colsfm.benchmark` harness compares two runs that both carry a
`kpmap/keyframes/frames_meta.json`. A plain COLMAP reconstruction carries no
such file and no rig, and its scale is free, so the audit needs two extra
things this module supplies:

1. **A per-image ground-truth reference.** For every keyframe the dataset's
   ground truth gives `world_T_vehicle` at that timestamp; composing with the
   calibrated `vehicle_T_cam` gives a ground-truth camera centre per image.
   That is the only reference a monocular reconstruction can be scored against
   without first inventing a rig.
2. **A similarity fit.** COLMAP's monocular output has no metric scale, so it
   must also be scored after a scale-free SIM(3) fit; both numbers are reported.
   That is `align_rigid(..., estimate_scale=True)`, the same function and the
   same Umeyama solve as the rigid fit — this module used to carry its own copy.

Everything measured here comes from `colsfm.benchmark`, so the audit's numbers
and the benchmark report's numbers are produced by the same code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
from jaxtyping import Float64, Int64
from numpy import ndarray

from colsfm.benchmark import (
    GROUND_TRUTH_TOLERANCE_MICROSECONDS,
    MILLIMETRES_PER_METRE,
    Positions,
    RigidAlignment,
    RigTrack,
    align_rigid,
    match_timestamps,
    read_ground_truth,
)
from colsfm.frames_meta import FRAMES_META_NAME, CameraParams, FramesMeta, KeyframeMeta, read_frames_meta
from colsfm.geometry import Vector3, rigid3d_from_matrix

MIN_ALIGNMENT_SAMPLES: Final[int] = 3
"""Below three matched samples a three-dimensional fit is not determined."""


@dataclass(frozen=True, slots=True)
class GalileoReference:
    """Everything the audit needs about the Galileo dataset, read once."""

    frames_meta: FramesMeta
    """The input collection: 226 keyframes, 8 cameras, the prior trajectory."""
    ground_truth: RigTrack
    """The shipped `ground_truth.txt` rig trajectory."""
    gt_center_by_image_name: dict[str, Float64[ndarray, "3"]]
    """Ground-truth camera centre per keyframe image name, in metres."""


def read_galileo_reference(input_dir: Path) -> GalileoReference:
    """Read the Galileo input metadata and expand ground truth to per-image poses.

    The ground truth is a vehicle-frame TUM trajectory sampled far faster than the
    keyframes, so each keyframe timestamp is joined to the nearest ground-truth
    sample within `GROUND_TRUTH_TOLERANCE_MICROSECONDS`, and the calibrated
    `vehicle_T_cam` turns that into a camera pose.

    Args:
        input_dir: Directory holding `frames_meta.json` and `ground_truth.txt`.

    Returns:
        The reference bundle.

    Raises:
        FileNotFoundError: When the dataset ships no ground truth.
    """
    frames_meta: FramesMeta = read_frames_meta(input_dir / FRAMES_META_NAME)
    ground_truth: RigTrack | None = read_ground_truth(input_dir)
    if ground_truth is None:
        raise FileNotFoundError(f"no ground_truth.txt under {input_dir}")

    keyframes: tuple[KeyframeMeta, ...] = tuple(sorted(frames_meta.keyframes, key=lambda item: item.timestamp_microseconds))
    query_microseconds: Int64[ndarray, "n"] = np.asarray([item.timestamp_microseconds for item in keyframes], dtype=np.int64)
    keyframe_indices, ground_truth_indices = match_timestamps(
        query_microseconds, ground_truth.timestamps_microseconds, GROUND_TRUTH_TOLERANCE_MICROSECONDS
    )

    centers: dict[str, Float64[ndarray, "3"]] = {}
    for keyframe_index, ground_truth_index in zip(keyframe_indices, ground_truth_indices, strict=True):
        keyframe: KeyframeMeta = keyframes[int(keyframe_index)]
        camera: CameraParams = frames_meta.cameras[keyframe.camera_params_id]
        world_T_vehicle: pycolmap.Rigid3d = rigid3d_from_matrix(
            ground_truth.world_R_rig[int(ground_truth_index)], ground_truth.world_t_rig[int(ground_truth_index)]
        )
        world_T_cam: pycolmap.Rigid3d = world_T_vehicle * camera.vehicle_T_cam
        centers[keyframe.image_name] = np.asarray(world_T_cam.translation, dtype=np.float64)

    return GalileoReference(frames_meta=frames_meta, ground_truth=ground_truth, gt_center_by_image_name=centers)


@dataclass(frozen=True, slots=True)
class TrajectoryScore:
    """ATE of one trajectory against a reference, both alignments reported."""

    num_matched: int
    """Samples that joined the reference."""
    rigid_rmse_millimeters: float
    """RMSE after a rigid fit with the scale held at 1.0 — the harness's metric."""
    rigid_max_millimeters: float
    """Largest residual of that rigid fit."""
    would_be_scale: float
    """Scale a similarity fit would have chosen; 1.0 means metric scale survived."""
    similarity_rmse_millimeters: float
    """RMSE after a scale-free SIM(3) fit — the only fair metric for monocular COLMAP."""

    @staticmethod
    def empty(num_matched: int) -> TrajectoryScore:
        """The score of a sample set too small to fit.

        `align_rigid` raises below one sample and is undetermined below three, so
        the guard lives here once rather than in each of this module's callers.

        Args:
            num_matched: How many samples did join, for the report.

        Returns:
            A score whose every residual is NaN.
        """
        return TrajectoryScore(
            num_matched=num_matched,
            rigid_rmse_millimeters=float("nan"),
            rigid_max_millimeters=float("nan"),
            would_be_scale=float("nan"),
            similarity_rmse_millimeters=float("nan"),
        )


def score_positions(source_xyz: Positions, target_xyz: Positions) -> TrajectoryScore:
    """Score one set of positions against a matched reference set.

    Args:
        source_xyz: Float64 positions to score, shape `[n, 3]`.
        target_xyz: Float64 reference positions, shape `[n, 3]`.

    Returns:
        Both alignments' residual statistics, in millimetres.
    """
    if len(source_xyz) < MIN_ALIGNMENT_SAMPLES:
        return TrajectoryScore.empty(len(source_xyz))
    rigid: RigidAlignment = align_rigid(source_xyz, target_xyz)
    similarity: RigidAlignment = align_rigid(source_xyz, target_xyz, estimate_scale=True)
    return TrajectoryScore(
        num_matched=len(source_xyz),
        rigid_rmse_millimeters=MILLIMETRES_PER_METRE * rigid.rmse_meters,
        rigid_max_millimeters=MILLIMETRES_PER_METRE * rigid.max_error_meters,
        would_be_scale=rigid.would_be_scale,
        similarity_rmse_millimeters=MILLIMETRES_PER_METRE * similarity.rmse_meters,
    )


def score_track_against_ground_truth(track: RigTrack, ground_truth: RigTrack) -> TrajectoryScore:
    """Join a rig trajectory to ground truth on timestamp and score it.

    Uses exactly the join `colsfm.benchmark.compute_run_metrics` uses for its
    `ATE vs ground truth` row: nearest timestamp within 20 microseconds.

    Args:
        track: The trajectory to score.
        ground_truth: The reference trajectory.

    Returns:
        The score.
    """
    track_indices, reference_indices = match_timestamps(
        track.timestamps_microseconds, ground_truth.timestamps_microseconds, GROUND_TRUTH_TOLERANCE_MICROSECONDS
    )
    return score_positions(track.world_t_rig[track_indices], ground_truth.world_t_rig[reference_indices])


def rig_track_from_reconstruction(
    reconstruction: pycolmap.Reconstruction, frames_meta: FramesMeta, alignment: RigidAlignment | None = None
) -> RigTrack:
    """Derive a rig trajectory from a plain (rig-free) COLMAP reconstruction.

    Follows `FramesMeta.rig_frames`' rule — the rig pose of a sample comes from its
    lowest keyframe id — restricted to the keyframes the reconstruction actually
    registered, since a monocular COLMAP model registers an arbitrary subset.

    Args:
        reconstruction: A model whose image names are the dataset's `image_name`s.
        frames_meta: The input collection supplying `synced_sample_id`, timestamps
            and the calibrated `vehicle_T_cam`.
        alignment: When given, applied to every camera pose first, so the rig track
            comes out in the ground-truth frame at metric scale.

    Returns:
        The rig trajectory over the samples with at least one registered keyframe.
    """
    world_T_cam_by_name: dict[str, pycolmap.Rigid3d] = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        pose: pycolmap.Rigid3d = image.cam_from_world().inverse()
        world_T_cam_by_name[image.name] = alignment.apply_pose(pose) if alignment is not None else pose

    members: dict[int, list[KeyframeMeta]] = {}
    for keyframe in frames_meta.keyframes:
        if keyframe.image_name in world_T_cam_by_name:
            members.setdefault(keyframe.synced_sample_id, []).append(keyframe)

    timestamps: list[int] = []
    positions: list[Float64[ndarray, "3"]] = []
    rotations: list[Float64[ndarray, "3 3"]] = []
    for synced_sample_id in sorted(members):
        reference: KeyframeMeta = min(members[synced_sample_id], key=lambda item: item.keyframe_id)
        camera: CameraParams = frames_meta.cameras[reference.camera_params_id]
        world_T_vehicle: pycolmap.Rigid3d = world_T_cam_by_name[reference.image_name] * camera.vehicle_T_cam.inverse()
        timestamps.append(reference.timestamp_microseconds)
        positions.append(np.asarray(world_T_vehicle.translation, dtype=np.float64))
        rotations.append(world_T_vehicle.rotation.matrix().astype(np.float64))

    return RigTrack(
        timestamps_microseconds=np.asarray(timestamps, dtype=np.int64),
        world_t_rig=np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        world_R_rig=np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3),
    )


def image_center_score(
    reconstruction: pycolmap.Reconstruction, reference: GalileoReference
) -> tuple[TrajectoryScore, RigidAlignment]:
    """Score a reconstruction's camera centres against per-image ground truth.

    The similarity fit is computed once and both returned and used for the
    reported RMSE, so the caller's alignment is the very fit the number came from.

    Args:
        reconstruction: A model whose image names are the dataset's `image_name`s.
        reference: The Galileo reference bundle.

    Returns:
        The score and the scale-estimating fit it used, so callers can reuse the fit.

    Raises:
        ValueError: When fewer than three registered images have ground truth.
    """
    # One pass over the images: `find_image_with_name` is a linear scan, so calling it
    # per name inside a comprehension is quadratic in the model size.
    center_by_name: dict[str, Vector3] = {
        image.name: np.asarray(image.cam_from_world().inverse().translation, dtype=np.float64)
        for image in reconstruction.images.values()
        if image.has_pose and image.name in reference.gt_center_by_image_name
    }
    names: list[str] = sorted(center_by_name)
    if len(names) < MIN_ALIGNMENT_SAMPLES:
        raise ValueError(f"only {len(names)} registered images carry ground truth")
    source_xyz: Positions = np.asarray([center_by_name[name] for name in names], dtype=np.float64).reshape(-1, 3)
    target_xyz: Positions = np.asarray([reference.gt_center_by_image_name[name] for name in names], dtype=np.float64).reshape(-1, 3)
    rigid: RigidAlignment = align_rigid(source_xyz, target_xyz)
    similarity: RigidAlignment = align_rigid(source_xyz, target_xyz, estimate_scale=True)
    score: TrajectoryScore = TrajectoryScore(
        num_matched=len(names),
        rigid_rmse_millimeters=MILLIMETRES_PER_METRE * rigid.rmse_meters,
        rigid_max_millimeters=MILLIMETRES_PER_METRE * rigid.max_error_meters,
        would_be_scale=rigid.would_be_scale,
        similarity_rmse_millimeters=MILLIMETRES_PER_METRE * similarity.rmse_meters,
    )
    return score, similarity

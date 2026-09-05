"""Rigid trajectory alignment and the rig trajectories the benchmark scores.

Everything here answers one of two questions: *where did the rig go* (`RigTrack`,
built either from a `frames_meta.json` or from a TUM `ground_truth.txt`) and *how
far apart are two answers* (`match_timestamps` to join them, `align_rigid` to fit
one onto the other).

Alignment defaults to rigid, with the scale **held at 1.0** and the scale a
similarity fit would have chosen reported separately, which is `umeyama_rigid`'s
contract in `demo_rerun.py`; `align_rigid(..., estimate_scale=True)` is the same
fit with the scale let in, for callers asking whether the *shape* is right rather
than the size. Reimplemented here rather than imported: `demo_rerun` is a
2000-line script in the *default* pixi environment that imports `pycusfm` at call
time, and `colsfm` may not depend on it.

The rig trajectory comes from `FramesMeta.rig_frames()` rather than being
re-derived from COLMAP camera poses and the *input* `rig_T_cam` the way
`cusfm_rig_trajectory` does. Both answer the same question; the metadata route is
the one the export spec defines (`world_T_vehicle = world_T_cam * vehicle_T_cam^-1`,
reference = lowest keyframe id of the sample) and it does not silently depend on
bundle adjustment having left the extrinsics alone. The rigidity spread — the
disagreement between the per-camera estimates of one sample's rig pose — is still
computed exactly as `cusfm_rig_trajectory` computes it, just from the metadata's
poses.

This module deliberately imports nothing from `colsfm.benchmark`: the benchmark
is one consumer of these primitives, and `tools/audit` is another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame
from colsfm.geometry import MILLIMETRES_PER_METRE, Matrix3, TumPose, Vector3, parse_tum_line, rigid3d_from_matrix

Positions: TypeAlias = Float64[ndarray, "n 3"]
"""A trajectory as one world-frame position per sample, in metres."""

Rotations: TypeAlias = Float64[ndarray, "n 3 3"]
"""World-frame rotation matrices, one per sample."""

Timestamps: TypeAlias = Int64[ndarray, "n"]
"""Capture times in integer microseconds since the Unix epoch."""

GROUND_TRUTH_FILE_NAME: Final[str] = "ground_truth.txt"
"""TUM ground truth shipped beside `frames_meta.json`; only r2b_galileo has one."""

GROUND_TRUTH_TOLERANCE_MICROSECONDS: Final[int] = 20
"""0.02 ms, the join window NOTES.md uses to match keyframes to `ground_truth.txt`."""

DEGENERATE_VARIANCE: Final[float] = 1e-15
"""Below this the source cloud is a point and the would-be scale is undefined."""


# ─────────────────────────────────────────────────────────────────────────────
# Rigid alignment
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RigidAlignment:
    """A least-squares fit of one trajectory onto another: SE(3), or SIM(3) on request."""

    target_R_source: Matrix3
    """Rotation taking source positions into the target frame."""
    target_t_source: Vector3
    """Translation taking source positions into the target frame, in metres."""
    rmse_meters: float
    """Root-mean-square residual after the fit."""
    max_error_meters: float
    """Largest single residual after the fit."""
    would_be_scale: float
    """Scale a *similarity* fit would have chosen; 1.0 means metric scale survived.

    Reported whether or not the fit took it: for a rigid fit it is the diagnostic
    the benchmark prints, and for a similarity fit it equals `scale`."""
    scale: float = 1.0
    """Scale the fit actually applied — 1.0 for a rigid fit, `would_be_scale` for a
    similarity one. `apply` and `apply_pose` both honour it, so a caller never has
    to remember which kind of fit it holds."""

    def apply(self, positions: Positions) -> Positions:
        """Map source-frame positions into the target frame.

        Args:
            positions: Float64 source positions with shape `[n, 3]`.

        Returns:
            Float64 positions with shape `[n, 3]` in the target frame.
        """
        return self.scale * (positions @ self.target_R_source.T) + self.target_t_source

    def apply_rotations(self, rotations: Rotations) -> Rotations:
        """Map source-frame rotation matrices into the target frame.

        The scale does not touch a rotation, so this is the same for both fits.

        Args:
            rotations: Float64 rotation matrices with shape `[n, 3, 3]`.

        Returns:
            Float64 rotation matrices with shape `[n, 3, 3]` in the target frame.
        """
        return np.einsum("ij,njk->nik", self.target_R_source, rotations)

    def apply_pose(self, world_T_body: pycolmap.Rigid3d) -> pycolmap.Rigid3d:
        """Map a whole source-frame pose into the target frame.

        Args:
            world_T_body: A pose in the source world frame.

        Returns:
            The same pose expressed in the target world frame: the rotation
            composed with the fit's, the translation through `apply`.
        """
        world_R_body: Matrix3 = np.asarray(world_T_body.rotation.matrix(), dtype=np.float64)
        world_t_body: Positions = np.asarray(world_T_body.translation, dtype=np.float64).reshape(1, 3)
        return rigid3d_from_matrix(self.target_R_source @ world_R_body, self.apply(world_t_body).reshape(3))


def align_rigid(source_xyz: Positions, target_xyz: Positions, *, estimate_scale: bool = False) -> RigidAlignment:
    """Fit `target ~ s * R @ source + t`, with `s` held at 1.0 unless asked otherwise.

    The benchmark holds the scale at 1.0 so that a reconstruction which shrank the
    trajectory shows up as alignment error rather than disappearing into the fit;
    the scale the same fit *would* have chosen comes back as `would_be_scale`
    either way. This is `demo_rerun.umeyama_rigid`'s contract, in float64 for
    millimetre work. `estimate_scale=True` is the SIM(3) fit the audits want when
    the question is "is the *shape* right", with the residuals measured after the
    scaling; it changes nothing about the default path.

    Args:
        source_xyz: Float64 positions to move, shape `[n, 3]`.
        target_xyz: Float64 positions to move onto, shape `[n, 3]`.
        estimate_scale: Let the fit absorb a scale change instead of holding it at 1.0.

    Returns:
        The fitted transform plus its residual statistics.

    Raises:
        ValueError: When the two trajectories differ in length or are empty.
    """
    if source_xyz.shape != target_xyz.shape:
        raise ValueError(f"alignment needs matched trajectories, got {source_xyz.shape} and {target_xyz.shape}")
    if len(source_xyz) == 0:
        raise ValueError("alignment needs at least one sample")

    source_mean: Vector3 = source_xyz.mean(axis=0)
    target_mean: Vector3 = target_xyz.mean(axis=0)
    source_centred: Positions = source_xyz - source_mean
    target_centred: Positions = target_xyz - target_mean

    covariance: Matrix3 = target_centred.T @ source_centred / len(source_xyz)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    sign_fix: Matrix3 = np.eye(3)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0:
        sign_fix[2, 2] = -1.0
    target_R_source: Matrix3 = u_matrix @ sign_fix @ vt_matrix

    source_variance: float = float((source_centred**2).sum() / len(source_xyz))
    would_be_scale: float = (
        float((singular_values * np.diag(sign_fix)).sum() / source_variance) if source_variance > DEGENERATE_VARIANCE else 1.0
    )

    scale: float = would_be_scale if estimate_scale else 1.0
    target_t_source: Vector3 = target_mean - scale * (target_R_source @ source_mean)
    residual: Positions = (scale * (source_xyz @ target_R_source.T) + target_t_source) - target_xyz
    distances: Float64[ndarray, "n"] = np.linalg.norm(residual, axis=1)
    return RigidAlignment(
        target_R_source=target_R_source,
        target_t_source=target_t_source,
        rmse_meters=float(np.sqrt((distances**2).mean())),
        max_error_meters=float(distances.max()),
        would_be_scale=would_be_scale,
        scale=scale,
    )


def path_length_meters(positions: Positions) -> float:
    """Total distance travelled along a trajectory.

    Args:
        positions: Float64 positions with shape `[n, 3]`.

    Returns:
        The summed segment length in metres; 0.0 for fewer than two samples.
    """
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Trajectories
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RigTrack:
    """A rig trajectory: one `world_T_rig` per synchronised sample, in time order."""

    timestamps_microseconds: Timestamps
    """Reference keyframe timestamp of each sample, ascending."""
    world_t_rig: Positions
    """Rig position in the world frame, metres."""
    world_R_rig: Rotations
    """Rig orientation in the world frame."""

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.timestamps_microseconds)

    def take(self, indices: Int64[ndarray, "m"]) -> RigTrack:
        """Select a subset of samples.

        Args:
            indices: Int64 sample indices with shape `[m]`.

        Returns:
            A track holding only those samples, in the given order.
        """
        return RigTrack(
            timestamps_microseconds=self.timestamps_microseconds[indices],
            world_t_rig=self.world_t_rig[indices],
            world_R_rig=self.world_R_rig[indices],
        )


def rig_track_from_frames_meta(frames_meta: FramesMeta, keep_image_names: set[str] | None = None) -> RigTrack:
    """Build the rig trajectory a `frames_meta.json` implies.

    Uses `FramesMeta.rig_frames()`, so the rig pose of each sample is
    `world_T_cam * vehicle_T_cam^-1` evaluated at the sample's **lowest keyframe
    id** — the rule that reproduces cuSFM's own vehicle metadata (pose_graph_main.md §4).

    Args:
        frames_meta: A parsed collection.
        keep_image_names: When given, keep only samples with at least one member
            keyframe whose `image_name` is in this set — i.e. the samples the
            reconstruction actually registered.

    Returns:
        The trajectory, ordered by `synced_sample_id`.
    """
    by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    frames: list[RigFrame] = []
    for rig_frame in frames_meta.rig_frames():
        if keep_image_names is not None:
            names: set[str] = {by_id[keyframe_id].image_name for keyframe_id in rig_frame.keyframe_ids}
            if names.isdisjoint(keep_image_names):
                continue
        frames.append(rig_frame)
    return RigTrack(
        timestamps_microseconds=np.asarray([frame.timestamp_microseconds for frame in frames], dtype=np.int64),
        world_t_rig=np.asarray([frame.world_T_vehicle.translation for frame in frames], dtype=np.float64).reshape(-1, 3),
        world_R_rig=np.asarray(
            [Rotation.from_quat(frame.world_T_vehicle.rotation.quat).as_matrix() for frame in frames], dtype=np.float64
        ).reshape(-1, 3, 3),
    )


def rig_rigidity_spread_millimeters(frames_meta: FramesMeta) -> float:
    """Worst disagreement between a sample's per-camera estimates of the rig pose.

    Every camera of one `synced_sample_id` yields `world_T_cam * vehicle_T_cam^-1`;
    with `ba_frame_type=vehicle_rig` those must coincide, and the spread between
    them is what `demo_rerun.cusfm_rig_trajectory` returns as its rigidity check.
    A non-zero value means the exoego schema's single `world_T_rig` per sample is
    a lie.

    Args:
        frames_meta: A parsed collection.

    Returns:
        The largest per-sample spread across all samples, in millimetres.
    """
    by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    worst_meters: float = 0.0
    for rig_frame in frames_meta.rig_frames():
        if len(rig_frame.keyframe_ids) < 2:
            continue
        estimates: Positions = np.asarray(
            [frames_meta.world_T_vehicle(by_id[keyframe_id]).translation for keyframe_id in rig_frame.keyframe_ids],
            dtype=np.float64,
        ).reshape(-1, 3)
        worst_meters = max(worst_meters, float(np.linalg.norm(estimates - estimates.mean(axis=0), axis=1).max()))
    return MILLIMETRES_PER_METRE * worst_meters


def read_ground_truth(input_dir: Path) -> RigTrack | None:
    """Read `ground_truth.txt` beside the input metadata, if the dataset ships one.

    The file is TUM. Its body frame is the **vehicle (FLU) frame in the same world
    frame the input `camera_to_world` uses**, so no conversion is needed on this
    side: NOTES.md gotcha 10 verified that `camera_to_world @ inv(sensor_to_vehicle_transform)`
    — exactly `FramesMeta.world_T_vehicle` — reproduces these positions to ~1.6 mm.
    Timestamps are seconds and come back as rounded microseconds.

    Args:
        input_dir: Directory holding the dataset's `frames_meta.json`.

    Returns:
        The ground-truth rig trajectory, or None when the dataset ships none.
    """
    path: Path = input_dir / GROUND_TRUTH_FILE_NAME
    if not path.is_file():
        return None
    poses: list[TumPose] = [parse_tum_line(line) for line in path.read_text().splitlines() if line.strip()]
    if not poses:
        return None
    order: Int64[ndarray, "n"] = np.argsort([pose.timestamp_microseconds for pose in poses]).astype(np.int64)
    ordered: list[TumPose] = [poses[index] for index in order]
    return RigTrack(
        timestamps_microseconds=np.asarray([pose.timestamp_microseconds for pose in ordered], dtype=np.int64),
        world_t_rig=np.asarray([pose.world_T_body.translation for pose in ordered], dtype=np.float64).reshape(-1, 3),
        world_R_rig=np.asarray(
            [Rotation.from_quat(pose.world_T_body.rotation.quat).as_matrix() for pose in ordered], dtype=np.float64
        ).reshape(-1, 3, 3),
    )


def match_timestamps(
    query_microseconds: Timestamps, reference_microseconds: Timestamps, tolerance_microseconds: int
) -> tuple[Int64[ndarray, "m"], Int64[ndarray, "m"]]:
    """Nearest-neighbour join of two timestamp streams.

    Args:
        query_microseconds: Int64 timestamps to match, shape `[n]`, ascending.
        reference_microseconds: Int64 timestamps to match against, shape `[k]`, ascending.
        tolerance_microseconds: Largest accepted gap; 0 demands an exact hit.

    Returns:
        Index arrays `(query_indices, reference_indices)` of the accepted pairs.
    """
    if len(query_microseconds) == 0 or len(reference_microseconds) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    insertion: Int64[ndarray, "n"] = np.searchsorted(reference_microseconds, query_microseconds)
    left: Int64[ndarray, "n"] = np.clip(insertion - 1, 0, len(reference_microseconds) - 1)
    right: Int64[ndarray, "n"] = np.clip(insertion, 0, len(reference_microseconds) - 1)
    left_gap: Int64[ndarray, "n"] = np.abs(reference_microseconds[left] - query_microseconds)
    right_gap: Int64[ndarray, "n"] = np.abs(reference_microseconds[right] - query_microseconds)
    nearest: Int64[ndarray, "n"] = np.where(left_gap <= right_gap, left, right)
    gap: Int64[ndarray, "n"] = np.minimum(left_gap, right_gap)
    accepted: Bool[ndarray, "n"] = gap <= tolerance_microseconds
    query_indices: Int64[ndarray, "m"] = np.flatnonzero(accepted).astype(np.int64)
    return query_indices, nearest[accepted].astype(np.int64)

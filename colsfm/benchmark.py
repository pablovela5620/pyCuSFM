"""Metrics that compare two cuSFM-layout runs: the NVIDIA blob against `colsfm`.

Both runs are read from the *same* directory layout (`docs/spec/export.md`), so
nothing here knows which producer wrote which run:

| Artifact | Read for |
|---|---|
| `sparse/` COLMAP text model | registered images, points, observations, reprojection error, track length |
| `kpmap/keyframes/frames_meta.json` | rig trajectory, rig rigidity spread |
| `runtime.csv` | per-stage wall clock |

## Deliberate differences from `demo_rerun.py`'s metric definitions

1. **Reprojection error is recomputed.** `kpmap_to_colmap` hardcodes
   `ERROR = 2.0` on every point (export.md §3.5), so the column in
   `points3D.txt` is not a measurement. `pycolmap.Reconstruction.update_point_3d_errors`
   followed by `compute_mean_reprojection_error` re-derives it from the tracks.
2. **The model is read with `pycolmap.Reconstruction.read_text`,** not
   `demo_rerun.read_colmap_model`. The hand-rolled parser mis-parses
   `images.txt` when an image has zero observations (pycolmap-capabilities.md §10).
3. **The rig trajectory comes from `kpmap/keyframes/frames_meta.json`,** through
   `FramesMeta.rig_frames()`, rather than being re-derived from COLMAP camera
   poses and the *input* `rig_T_cam` the way `cusfm_rig_trajectory` does. Both
   answer the same question; the metadata route is the one the export spec
   defines (`world_T_vehicle = world_T_cam * vehicle_T_cam^-1`, reference =
   lowest keyframe id of the sample) and it does not silently depend on bundle
   adjustment having left the extrinsics alone. The rigidity spread — the
   disagreement between the per-camera estimates of one sample's rig pose — is
   still computed exactly as `cusfm_rig_trajectory` computes it, just from the
   metadata's poses.
4. **`runtime.csv` is cut down to its last run.** The blob's runner *appends*,
   so a directory reused across runs holds one block of rows per run
   (`data/cusfm_runs/galileo/cusfm/runtime.csv` holds three). Summing the file
   would report the history rather than the run, so `latest_run_records` keeps
   only the rows after the last repeat of a command name.

Alignment is always rigid with the scale **held at 1.0** and the scale a
similarity fit would have chosen reported separately, which is `umeyama_rigid`'s
contract in the demo. Reimplemented here rather than imported: `demo_rerun` is a
2000-line script in the *default* pixi environment that imports `pycusfm` at
call time, and `colsfm` may not depend on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde
from serde.json import to_json

from colsfm.export import KEYFRAME_METADATA_SUBPATH, RUNTIME_CSV_NAME, SPARSE_DIR_NAME, RuntimeRecord, read_runtime_records
from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame, read_frames_meta
from colsfm.geometry import TumPose, parse_tum_line

Stage: TypeAlias = Literal[
    "extraction",
    "retrieval",
    "association",
    "pose_graph",
    "pair_selection",
    "matching",
    "mapping",
    "export",
]
"""The eight pipeline stages of `docs/open-pipeline-plan.md` §"Stages and their spec owners"."""

DatasetName: TypeAlias = Literal["galileo", "robocap"]
"""Datasets the harness knows acceptance bounds and ground truth for."""

Positions: TypeAlias = Float64[ndarray, "n 3"]
"""A trajectory as one world-frame position per sample, in metres."""

Rotations: TypeAlias = Float64[ndarray, "n 3 3"]
"""World-frame rotation matrices, one per sample."""

Timestamps: TypeAlias = Int64[ndarray, "n"]
"""Capture times in integer microseconds since the Unix epoch."""

STAGES: Final[tuple[Stage, ...]] = (
    "extraction",
    "retrieval",
    "association",
    "pose_graph",
    "pair_selection",
    "matching",
    "mapping",
    "export",
)
"""Stages in pipeline order; also the row order of the runtime table."""

STAGE_LABELS: Final[dict[Stage, str]] = {
    "extraction": "1 feature extraction + keyframe selection",
    "retrieval": "2 BoW vocabulary + index",
    "association": "3 loop-closure association",
    "pose_graph": "4 pose graph optimisation",
    "pair_selection": "5 match pair selection",
    "matching": "6 feature matching",
    "mapping": "7 triangulation + bundle adjustment",
    "export": "8 COLMAP + TUM export",
}
"""Human labels for the report, numbered as the plan numbers the stages."""

STAGE_PATTERNS: Final[tuple[tuple[str, Stage], ...]] = (
    # Ordered, first match wins. Order is load-bearing: `extract_pose_from_map_main`
    # contains "extract", `feature_matcher_task_builder_main` contains "matcher",
    # and `pose_graph_association_main` contains "association".
    ("kpmap_to_colmap", "export"),
    ("extract_pose", "export"),
    ("update_keyframe_pose", "export"),
    ("export", "export"),
    ("task_builder", "pair_selection"),
    ("pair", "pair_selection"),
    ("pose_graph", "pose_graph"),
    ("association", "association"),
    ("loop_closure", "association"),
    ("bow", "retrieval"),
    ("vocabulary", "retrieval"),
    ("retrieval", "retrieval"),
    ("matcher", "matching"),
    ("matching", "matching"),
    ("mapper", "mapping"),
    ("mapping", "mapping"),
    ("reconstruct", "mapping"),
    ("triangulat", "mapping"),
    ("bundle", "mapping"),
    ("extractor", "extraction"),
    ("feature", "extraction"),
    ("keyframe", "extraction"),
    ("extract", "extraction"),
)
"""Substring rules mapping a `runtime.csv` command onto a canonical stage.

The blob's rows are binary names (`feature_extractor_main`); `colsfm.pipeline`'s
are module-ish stage names. One ordered table covers both because the vocabulary
barely overlaps and the specific rules come first."""

GROUND_TRUTH_FILE_NAME: Final[str] = "ground_truth.txt"
"""TUM ground truth shipped beside `frames_meta.json`; only r2b_galileo has one."""

GROUND_TRUTH_TOLERANCE_MICROSECONDS: Final[int] = 20
"""0.02 ms, the join window NOTES.md uses to match keyframes to `ground_truth.txt`."""

MILLIMETRES_PER_METRE: Final[float] = 1000.0
"""Trajectory errors are reported in millimetres; everything is stored in metres."""

DEGENERATE_VARIANCE: Final[float] = 1e-15
"""Below this the source cloud is a point and the would-be scale is undefined."""


# ─────────────────────────────────────────────────────────────────────────────
# Rigid alignment
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RigidAlignment:
    """A least-squares SE(3) fit of one trajectory onto another, scale held at 1."""

    target_R_source: Float64[ndarray, "3 3"]
    """Rotation taking source positions into the target frame."""
    target_t_source: Float64[ndarray, "3"]
    """Translation taking source positions into the target frame, in metres."""
    rmse_meters: float
    """Root-mean-square residual after the fit."""
    max_error_meters: float
    """Largest single residual after the fit."""
    would_be_scale: float
    """Scale a *similarity* fit would have chosen; 1.0 means metric scale survived."""

    def apply(self, positions: Positions) -> Positions:
        """Map source-frame positions into the target frame.

        Args:
            positions: Float64 source positions with shape `[n, 3]`.

        Returns:
            Float64 positions with shape `[n, 3]` in the target frame.
        """
        return positions @ self.target_R_source.T + self.target_t_source

    def apply_rotations(self, rotations: Rotations) -> Rotations:
        """Map source-frame rotation matrices into the target frame.

        Args:
            rotations: Float64 rotation matrices with shape `[n, 3, 3]`.

        Returns:
            Float64 rotation matrices with shape `[n, 3, 3]` in the target frame.
        """
        return np.einsum("ij,njk->nik", self.target_R_source, rotations)


def align_rigid(source_xyz: Positions, target_xyz: Positions) -> RigidAlignment:
    """Fit `target ~ R @ source + t`, refusing to absorb a scale change.

    The scale is held at 1.0 so that a reconstruction which shrank the trajectory
    shows up as alignment error rather than disappearing into the fit; the scale
    the same fit *would* have chosen is returned as a diagnostic. This is
    `demo_rerun.umeyama_rigid`'s contract, in float64 for millimetre work.

    Args:
        source_xyz: Float64 positions to move, shape `[n, 3]`.
        target_xyz: Float64 positions to move onto, shape `[n, 3]`.

    Returns:
        The fitted transform plus its residual statistics.

    Raises:
        ValueError: When the two trajectories differ in length or are empty.
    """
    if source_xyz.shape != target_xyz.shape:
        raise ValueError(f"alignment needs matched trajectories, got {source_xyz.shape} and {target_xyz.shape}")
    if len(source_xyz) == 0:
        raise ValueError("alignment needs at least one sample")

    source_mean: Float64[ndarray, "3"] = source_xyz.mean(axis=0)
    target_mean: Float64[ndarray, "3"] = target_xyz.mean(axis=0)
    source_centred: Positions = source_xyz - source_mean
    target_centred: Positions = target_xyz - target_mean

    covariance: Float64[ndarray, "3 3"] = target_centred.T @ source_centred / len(source_xyz)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    sign_fix: Float64[ndarray, "3 3"] = np.eye(3)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0:
        sign_fix[2, 2] = -1.0
    target_R_source: Float64[ndarray, "3 3"] = u_matrix @ sign_fix @ vt_matrix

    source_variance: float = float((source_centred**2).sum() / len(source_xyz))
    would_be_scale: float = (
        float((singular_values * np.diag(sign_fix)).sum() / source_variance) if source_variance > DEGENERATE_VARIANCE else 1.0
    )

    target_t_source: Float64[ndarray, "3"] = target_mean - target_R_source @ source_mean
    residual: Positions = (source_xyz @ target_R_source.T + target_t_source) - target_xyz
    distances: Float64[ndarray, "n"] = np.linalg.norm(residual, axis=1)
    return RigidAlignment(
        target_R_source=target_R_source,
        target_t_source=target_t_source,
        rmse_meters=float(np.sqrt((distances**2).mean())),
        max_error_meters=float(distances.max()),
        would_be_scale=would_be_scale,
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


# ─────────────────────────────────────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────────────────────────────────────


def classify_stage(command: str) -> Stage | None:
    """Map one `runtime.csv` command onto a canonical pipeline stage.

    Args:
        command: The `command` column, `"<name> <args...>"`.

    Returns:
        The stage, or None when nothing in `STAGE_PATTERNS` matches.
    """
    head: str = command.split()[0].lower() if command.split() else ""
    for needle, stage in STAGE_PATTERNS:
        if needle in head:
            return stage
    return None


def latest_run_records(records: Sequence[RuntimeRecord]) -> list[RuntimeRecord]:
    """Keep only the rows belonging to the most recent run in an appended log.

    `pycusfm.command_runner` *appends* to `runtime.csv`, so a workspace reused
    across runs holds one block of rows per run
    (`data/cusfm_runs/galileo/cusfm/runtime.csv` holds three). A run invokes each
    command **once**, so a repeated command name starts a new block.

    Two refinements the shipped logs force:

    - Compare the command *name*, not the whole row. `kpmap_to_colmap` takes no
      config path, so its full row is byte-identical between two runs that
      differ everywhere else; keying on the row would cut the block in the
      middle of a run.
    - A repeat of the *immediately preceding* name is a sharded stage, not a new
      run — several `feature_matcher_main` task files land as consecutive rows.

    Stage order is deliberately not used: `colsfm` runs matching before loop
    closure, so "the stage index went backwards" is not a run boundary there.

    Args:
        records: Rows in file order.

    Returns:
        The rows of the last block, in file order.
    """
    boundary: int = 0
    seen_names: set[str] = set()
    previous_name: str = ""
    for index, record in enumerate(records):
        name: str = record.command.split()[0] if record.command.split() else ""
        if name in seen_names and name != previous_name:
            boundary = index
            seen_names = set()
        seen_names.add(name)
        previous_name = name
    return list(records[boundary:])


def stage_runtime_seconds(records: Sequence[RuntimeRecord]) -> dict[Stage, float]:
    """Total each stage's wall clock over the most recent run in the log.

    Args:
        records: Rows in file order, possibly spanning several appended runs.

    Returns:
        Seconds per stage; stages with no matching row are absent.
    """
    totals: dict[Stage, float] = {}
    for record in latest_run_records(records):
        stage: Stage | None = classify_stage(record.command)
        if stage is None:
            continue
        totals[stage] = totals.get(stage, 0.0) + record.runtime_seconds
    return totals


# ─────────────────────────────────────────────────────────────────────────────
# What one run holds
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Everything read off one cuSFM-layout run directory, arrays included.

    Kept apart from `RunMetrics` because it carries numpy arrays and a
    `pycolmap.Reconstruction`, neither of which belongs in a serialised report.
    """

    name: str
    """Short label used in the report and in Rerun metadata."""
    run_dir: Path
    """The `.../cusfm` directory the artifacts were read from."""
    reconstruction: pycolmap.Reconstruction
    """The `sparse/` model, with `update_point_3d_errors` already applied."""
    frames_meta: FramesMeta
    """`kpmap/keyframes/frames_meta.json` — the optimised poses and the calibration."""
    track: RigTrack
    """Rig trajectory over the samples the reconstruction registered."""
    points_xyz: Positions
    """Sparse point positions in the run's world frame."""
    points_rgb: Int64[ndarray, "n_points 3"]
    """Per-point colour; all zero when the run omitted `--output_rgb` (gotcha 11)."""
    runtime_seconds_by_stage: dict[Stage, float]
    """Wall clock per stage, from `runtime.csv`."""


def read_run(run_dir: Path, name: str) -> RunArtifacts:
    """Read one cuSFM-layout run directory.

    Args:
        run_dir: A `.../cusfm` directory holding `sparse/`, `kpmap/` and `runtime.csv`.
        name: Short label for the run.

    Returns:
        The artifacts, with point errors already recomputed from the tracks.

    Raises:
        FileNotFoundError: When the sparse model or the optimised metadata is missing.
    """
    sparse_dir: Path = run_dir / SPARSE_DIR_NAME
    meta_path: Path = run_dir / KEYFRAME_METADATA_SUBPATH
    if not (sparse_dir / "images.txt").is_file() and not (sparse_dir / "images.bin").is_file():
        raise FileNotFoundError(f"no COLMAP model under {sparse_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"no optimised metadata at {meta_path}")

    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    # kpmap_to_colmap hardcodes ERROR = 2.0 (export.md §3.5); recompute from the tracks.
    reconstruction.update_point_3d_errors()
    frames_meta: FramesMeta = read_frames_meta(meta_path)

    registered_names: set[str] = {image.name for image in reconstruction.images.values() if image.num_points3D > 0}
    points_xyz: Positions = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=np.float64).reshape(-1, 3)
    points_rgb: Int64[ndarray, "n_points 3"] = np.asarray(
        [point.color for point in reconstruction.points3D.values()], dtype=np.int64
    ).reshape(-1, 3)

    runtime_path: Path = run_dir / RUNTIME_CSV_NAME
    records: list[RuntimeRecord] = read_runtime_records(runtime_path) if runtime_path.is_file() else []

    return RunArtifacts(
        name=name,
        run_dir=run_dir,
        reconstruction=reconstruction,
        frames_meta=frames_meta,
        track=rig_track_from_frames_meta(frames_meta, keep_image_names=registered_names),
        points_xyz=points_xyz,
        points_rgb=points_rgb,
        runtime_seconds_by_stage=stage_runtime_seconds(records),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Serialisable metrics
# ─────────────────────────────────────────────────────────────────────────────


@serde
@dataclass(frozen=True)
class TrajectoryMetrics:
    """Alignment of one run's rig trajectory onto a reference trajectory."""

    reference: str
    """What it was compared against: `input`, `ground_truth`, or the other run."""
    num_matched: int
    """Samples that joined on timestamp."""
    rmse_millimeters: float
    """Root-mean-square position error after a rigid fit with scale held at 1.0."""
    max_millimeters: float
    """Largest single position error after the same fit."""
    would_be_scale: float
    """Scale a similarity fit would have chosen; 1.0 means metric scale survived."""
    reference_path_length_meters: float
    """Distance travelled along the reference over the matched samples."""


@serde
@dataclass(frozen=True)
class StageRuntime:
    """One stage's wall clock inside a single run."""

    stage: Stage
    """Canonical stage name."""
    seconds: float
    """Summed wall clock of every `runtime.csv` row that mapped to this stage."""


@serde
@dataclass(frozen=True)
class RunMetrics:
    """Everything the report says about one run."""

    name: str
    """Short label for the run."""
    run_dir: str
    """The directory the metrics were read from."""
    registered_images: int
    """Images with at least one 3D observation."""
    total_images: int
    """Keyframes the run was given, from `kpmap/keyframes/frames_meta.json`."""
    num_points3D: int
    """Triangulated points in the sparse model."""
    num_observations: int
    """Total 2D-3D observations across every track."""
    mean_reprojection_error_px: float
    """Recomputed from the model, never read from the `ERROR` column."""
    track_length_mean: float
    """Mean observations per point."""
    track_length_median: float
    """Median observations per point."""
    rig_rigidity_spread_millimeters: float
    """Worst per-sample disagreement between the cameras' rig-pose estimates."""
    num_rig_frames: int
    """Synchronised samples with at least one registered image."""
    stage_runtimes: tuple[StageRuntime, ...]
    """Per-stage wall clock, in pipeline order; missing stages are omitted."""
    total_runtime_seconds: float
    """Sum of every stage that reported a runtime."""
    points_are_black: bool
    """True when every point colour is `(0, 0, 0)` — the run omitted `--output_rgb`."""
    vs_input: TrajectoryMetrics
    """Disagreement against the input trajectory in the dataset's `frames_meta.json`."""
    vs_ground_truth: TrajectoryMetrics | None
    """ATE against `ground_truth.txt`, or None when the dataset ships none."""


@serde
@dataclass(frozen=True)
class PoseDelta:
    """Direct disagreement between the two runs' rig poses."""

    num_matched: int
    """Samples present in both runs, joined on timestamp."""
    position_rmse_millimeters: float
    """Root-mean-square position difference after a rigid fit of B onto A."""
    position_max_millimeters: float
    """Largest single position difference after the same fit."""
    rotation_rmse_degrees: float
    """Root-mean-square orientation difference after the same fit."""
    rotation_max_degrees: float
    """Largest single orientation difference after the same fit."""


@serde
@dataclass(frozen=True)
class StageComparison:
    """One row of the per-stage runtime table."""

    stage: Stage
    """Canonical stage name."""
    label: str
    """Human label used in the report."""
    seconds_a: float | None
    """Run A's wall clock, or None when it reported no row for this stage."""
    seconds_b: float | None
    """Run B's wall clock, or None when it reported no row for this stage."""
    ratio_b_over_a: float | None
    """`seconds_b / seconds_a`, or None when either side is missing or A is zero."""


@serde
@dataclass(frozen=True)
class AcceptanceBounds:
    """The Galileo bounds of `docs/open-pipeline-plan.md` §Acceptance.

    Every bound is **relative to run A as this harness measures it**, not absolute.
    The plan's original absolute figures (ATE <= 5 mm, reprojection <= 1.7 px) came
    from an older NOTES table; measured here the blob reference itself scores
    5.00 mm, so an absolute 5 mm bound asks the port to beat the thing it is
    reproducing. Relative bounds also survive a new machine or a re-run of A.
    """

    registered_images_allowance: int = 4
    """Run B may register up to this many fewer images than run A."""
    max_ate_ratio: float = 1.10
    """Run B's ATE against `ground_truth.txt`, over run A's."""
    max_reprojection_ratio: float = 1.10
    """Run B's recomputed mean reprojection error, over run A's."""
    max_runtime_ratio: float = 2.0
    """Run B's total runtime must stay within this multiple of run A's."""


@serde
@dataclass(frozen=True)
class AcceptanceCheck:
    """One bound, evaluated against run B."""

    name: str
    """What was checked."""
    bound: str
    """The bound as written in the plan, e.g. `">= 220"`."""
    value: float
    """The measured value."""
    passed: bool
    """Whether the measurement satisfies the bound."""


@serde
@dataclass(frozen=True)
class Comparison:
    """The complete benchmark result for one dataset and one pair of runs."""

    dataset: str
    """Dataset label, e.g. `galileo`."""
    run_a: RunMetrics
    """Reference run — the NVIDIA blob."""
    run_b: RunMetrics
    """Candidate run — the `colsfm` pipeline."""
    pose_delta: PoseDelta
    """Rig-pose disagreement between the two runs."""
    stages: tuple[StageComparison, ...]
    """Per-stage runtime table, in pipeline order."""
    total_runtime_ratio: float | None
    """Run B's total runtime over run A's, or None when A reported nothing."""
    acceptance: tuple[AcceptanceCheck, ...]
    """Plan bounds; empty for datasets the plan sets no bounds for."""


# ─────────────────────────────────────────────────────────────────────────────
# Computing the metrics
# ─────────────────────────────────────────────────────────────────────────────


def _trajectory_metrics(reference_name: str, track: RigTrack, reference: RigTrack, tolerance_microseconds: int) -> TrajectoryMetrics:
    """Join two trajectories on timestamp and align the run onto the reference."""
    track_indices, reference_indices = match_timestamps(
        track.timestamps_microseconds, reference.timestamps_microseconds, tolerance_microseconds
    )
    if len(track_indices) < 3:
        return TrajectoryMetrics(
            reference=reference_name,
            num_matched=len(track_indices),
            rmse_millimeters=float("nan"),
            max_millimeters=float("nan"),
            would_be_scale=float("nan"),
            reference_path_length_meters=0.0,
        )
    source_xyz: Positions = track.world_t_rig[track_indices]
    target_xyz: Positions = reference.world_t_rig[reference_indices]
    alignment: RigidAlignment = align_rigid(source_xyz, target_xyz)
    return TrajectoryMetrics(
        reference=reference_name,
        num_matched=len(track_indices),
        rmse_millimeters=MILLIMETRES_PER_METRE * alignment.rmse_meters,
        max_millimeters=MILLIMETRES_PER_METRE * alignment.max_error_meters,
        would_be_scale=alignment.would_be_scale,
        reference_path_length_meters=path_length_meters(target_xyz),
    )


def compute_run_metrics(artifacts: RunArtifacts, input_track: RigTrack, ground_truth: RigTrack | None) -> RunMetrics:
    """Reduce one run's artifacts to the numbers the report prints.

    Args:
        artifacts: A run read by `read_run`.
        input_track: Rig trajectory of the dataset's input `frames_meta.json`.
        ground_truth: Ground-truth rig trajectory, or None.

    Returns:
        The run's metrics.
    """
    reconstruction: pycolmap.Reconstruction = artifacts.reconstruction
    track_lengths: Int64[ndarray, "n_points"] = np.asarray(
        [point.track.length() for point in reconstruction.points3D.values()], dtype=np.int64
    )
    stage_runtimes: tuple[StageRuntime, ...] = tuple(
        StageRuntime(stage=stage, seconds=artifacts.runtime_seconds_by_stage[stage])
        for stage in STAGES
        if stage in artifacts.runtime_seconds_by_stage
    )
    return RunMetrics(
        name=artifacts.name,
        run_dir=str(artifacts.run_dir),
        registered_images=sum(1 for image in reconstruction.images.values() if image.num_points3D > 0),
        total_images=len(artifacts.frames_meta.keyframes),
        num_points3D=reconstruction.num_points3D(),
        num_observations=reconstruction.compute_num_observations(),
        mean_reprojection_error_px=float(reconstruction.compute_mean_reprojection_error()),
        track_length_mean=float(track_lengths.mean()) if len(track_lengths) else 0.0,
        track_length_median=float(np.median(track_lengths)) if len(track_lengths) else 0.0,
        rig_rigidity_spread_millimeters=rig_rigidity_spread_millimeters(artifacts.frames_meta),
        num_rig_frames=len(artifacts.track),
        stage_runtimes=stage_runtimes,
        total_runtime_seconds=float(sum(artifacts.runtime_seconds_by_stage.values())),
        points_are_black=bool(len(artifacts.points_rgb) > 0 and not artifacts.points_rgb.any()),
        vs_input=_trajectory_metrics("input", artifacts.track, input_track, 0),
        vs_ground_truth=(
            None
            if ground_truth is None
            else _trajectory_metrics("ground_truth", artifacts.track, ground_truth, GROUND_TRUTH_TOLERANCE_MICROSECONDS)
        ),
    )


def compute_pose_delta(track_a: RigTrack, track_b: RigTrack) -> PoseDelta:
    """Compare the two runs' rig poses sample by sample.

    A rigid fit of B onto A is applied first. The two runs inherit the same input
    world frame, so the fit is usually near identity, but a gauge difference
    between two independent bundle adjustments would otherwise dominate a
    "direct" difference and hide the shape disagreement, which is what this is
    meant to measure.

    Args:
        track_a: Reference run's rig trajectory.
        track_b: Candidate run's rig trajectory.

    Returns:
        Position and orientation disagreement over the samples both runs hold.
    """
    indices_b, indices_a = match_timestamps(track_b.timestamps_microseconds, track_a.timestamps_microseconds, 0)
    if len(indices_b) < 3:
        return PoseDelta(
            num_matched=len(indices_b),
            position_rmse_millimeters=float("nan"),
            position_max_millimeters=float("nan"),
            rotation_rmse_degrees=float("nan"),
            rotation_max_degrees=float("nan"),
        )
    source_xyz: Positions = track_b.world_t_rig[indices_b]
    target_xyz: Positions = track_a.world_t_rig[indices_a]
    alignment: RigidAlignment = align_rigid(source_xyz, target_xyz)
    aligned_R: Rotations = alignment.apply_rotations(track_b.world_R_rig[indices_b])
    relative_R: Rotations = np.einsum("nij,nkj->nik", aligned_R, track_a.world_R_rig[indices_a])
    angles_degrees: Float64[ndarray, "m"] = np.rad2deg(np.linalg.norm(Rotation.from_matrix(relative_R).as_rotvec(), axis=-1))
    return PoseDelta(
        num_matched=len(indices_b),
        position_rmse_millimeters=MILLIMETRES_PER_METRE * alignment.rmse_meters,
        position_max_millimeters=MILLIMETRES_PER_METRE * alignment.max_error_meters,
        rotation_rmse_degrees=float(np.sqrt((angles_degrees**2).mean())),
        rotation_max_degrees=float(angles_degrees.max()),
    )


def compare_stage_runtimes(run_a: RunArtifacts, run_b: RunArtifacts) -> tuple[StageComparison, ...]:
    """Build the per-stage runtime table.

    Args:
        run_a: Reference run.
        run_b: Candidate run.

    Returns:
        One row per stage of `STAGES`, in pipeline order, including stages
        neither run reported (both columns None) so the table shape is stable.
    """
    rows: list[StageComparison] = []
    for stage in STAGES:
        seconds_a: float | None = run_a.runtime_seconds_by_stage.get(stage)
        seconds_b: float | None = run_b.runtime_seconds_by_stage.get(stage)
        ratio: float | None = seconds_b / seconds_a if seconds_a and seconds_b is not None else None
        rows.append(
            StageComparison(stage=stage, label=STAGE_LABELS[stage], seconds_a=seconds_a, seconds_b=seconds_b, ratio_b_over_a=ratio)
        )
    return tuple(rows)


def check_acceptance(
    metrics_a: RunMetrics,
    metrics_b: RunMetrics,
    total_runtime_ratio: float | None,
    bounds: AcceptanceBounds,
) -> tuple[AcceptanceCheck, ...]:
    """Evaluate the plan's Galileo bounds, every one of them relative to run A.

    Args:
        metrics_a: Reference run's metrics — the blob, measured by this harness.
        metrics_b: Candidate run's metrics.
        total_runtime_ratio: Candidate total over reference total, or None.
        bounds: The bounds to apply.

    Returns:
        One check per bound; the ATE check is omitted when no ground truth exists,
        and the runtime check when the reference reported no runtime.
    """
    min_registered: int = metrics_a.registered_images - bounds.registered_images_allowance
    max_reprojection_px: float = bounds.max_reprojection_ratio * metrics_a.mean_reprojection_error_px
    checks: list[AcceptanceCheck] = [
        AcceptanceCheck(
            name="registered images",
            bound=f">= A - {bounds.registered_images_allowance} = {min_registered}",
            value=float(metrics_b.registered_images),
            passed=metrics_b.registered_images >= min_registered,
        ),
        AcceptanceCheck(
            name="mean reprojection error (px)",
            bound=f"<= {bounds.max_reprojection_ratio:.2f} x A = {max_reprojection_px:.3f}",
            value=metrics_b.mean_reprojection_error_px,
            passed=metrics_b.mean_reprojection_error_px <= max_reprojection_px,
        ),
    ]
    if metrics_a.vs_ground_truth is not None and metrics_b.vs_ground_truth is not None:
        max_ate_millimeters: float = bounds.max_ate_ratio * metrics_a.vs_ground_truth.rmse_millimeters
        checks.append(
            AcceptanceCheck(
                name="ATE vs ground truth (mm)",
                bound=f"<= {bounds.max_ate_ratio:.2f} x A = {max_ate_millimeters:.3f}",
                value=metrics_b.vs_ground_truth.rmse_millimeters,
                passed=bool(metrics_b.vs_ground_truth.rmse_millimeters <= max_ate_millimeters),
            )
        )
    if total_runtime_ratio is not None:
        checks.append(
            AcceptanceCheck(
                name="total runtime ratio",
                bound=f"<= {bounds.max_runtime_ratio} x A",
                value=total_runtime_ratio,
                passed=total_runtime_ratio <= bounds.max_runtime_ratio,
            )
        )
    return tuple(checks)


def compare_runs(
    run_a: RunArtifacts,
    run_b: RunArtifacts,
    input_dir: Path,
    dataset: DatasetName,
    bounds: AcceptanceBounds | None = None,
) -> Comparison:
    """Compute every metric for one pair of runs.

    Args:
        run_a: Reference run, read by `read_run` (the NVIDIA blob).
        run_b: Candidate run, read by `read_run` (the `colsfm` pipeline).
        input_dir: Directory holding the dataset's input `frames_meta.json` and,
            when the dataset ships one, `ground_truth.txt`.
        dataset: Dataset label; acceptance bounds apply to `galileo` only.
        bounds: Acceptance bounds, or None for the plan's defaults.

    Returns:
        The full comparison, ready for `write_markdown_report` and `to_json`.
    """
    input_meta: FramesMeta = read_frames_meta(input_dir / "frames_meta.json")
    input_track: RigTrack = rig_track_from_frames_meta(input_meta)
    ground_truth: RigTrack | None = read_ground_truth(input_dir)

    metrics_a: RunMetrics = compute_run_metrics(run_a, input_track, ground_truth)
    metrics_b: RunMetrics = compute_run_metrics(run_b, input_track, ground_truth)
    ratio: float | None = (
        metrics_b.total_runtime_seconds / metrics_a.total_runtime_seconds if metrics_a.total_runtime_seconds > 0.0 else None
    )
    return Comparison(
        dataset=dataset,
        run_a=metrics_a,
        run_b=metrics_b,
        pose_delta=compute_pose_delta(run_a.track, run_b.track),
        stages=compare_stage_runtimes(run_a, run_b),
        total_runtime_ratio=ratio,
        acceptance=(
            check_acceptance(metrics_a, metrics_b, ratio, bounds or AcceptanceBounds()) if dataset == "galileo" else ()
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────


def _number(value: float | None, decimals: int = 2) -> str:
    """Format an optional number for a markdown cell."""
    if value is None:
        return "—"
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def _trajectory_row(label: str, metrics: TrajectoryMetrics | None) -> str:
    """One markdown row of the trajectory table."""
    if metrics is None:
        return f"| {label} | — | — | — | — |"
    return (
        f"| {label} | {metrics.num_matched} | {_number(metrics.rmse_millimeters)} | "
        f"{_number(metrics.max_millimeters)} | {_number(metrics.would_be_scale, 5)} |"
    )


def render_markdown_report(comparison: Comparison) -> str:
    """Render the comparison as a markdown document.

    Args:
        comparison: The result of `compare_runs`.

    Returns:
        The markdown text, ending in a newline.
    """
    run_a: RunMetrics = comparison.run_a
    run_b: RunMetrics = comparison.run_b
    lines: list[str] = [
        f"# cuSFM benchmark — {comparison.dataset}",
        "",
        f"- **A** `{run_a.name}` — `{run_a.run_dir}`",
        f"- **B** `{run_b.name}` — `{run_b.run_dir}`",
        "",
        "## Reconstruction",
        "",
        "| Metric | A | B |",
        "|---|---|---|",
        f"| registered / total images | {run_a.registered_images} / {run_a.total_images} | {run_b.registered_images} / {run_b.total_images} |",
        f"| 3D points | {run_a.num_points3D} | {run_b.num_points3D} |",
        f"| observations | {run_a.num_observations} | {run_b.num_observations} |",
        f"| mean reprojection error (px, recomputed) | {_number(run_a.mean_reprojection_error_px, 3)} | {_number(run_b.mean_reprojection_error_px, 3)} |",
        (
            f"| track length mean / median | {_number(run_a.track_length_mean)} / {_number(run_a.track_length_median)} "
            f"| {_number(run_b.track_length_mean)} / {_number(run_b.track_length_median)} |"
        ),
        f"| rig rigidity spread (mm) | {_number(run_a.rig_rigidity_spread_millimeters, 3)} | {_number(run_b.rig_rigidity_spread_millimeters, 3)} |",
        f"| rig frames registered | {run_a.num_rig_frames} | {run_b.num_rig_frames} |",
        f"| point colours present | {'no' if run_a.points_are_black else 'yes'} | {'no' if run_b.points_are_black else 'yes'} |",
        "",
        "## Trajectory (rigid alignment, scale held at 1.0)",
        "",
        "| Comparison | matched | RMSE (mm) | max (mm) | would-be scale |",
        "|---|---|---|---|---|",
        _trajectory_row(f"A `{run_a.name}` vs input", run_a.vs_input),
        _trajectory_row(f"B `{run_b.name}` vs input", run_b.vs_input),
        _trajectory_row(f"A `{run_a.name}` vs ground truth", run_a.vs_ground_truth),
        _trajectory_row(f"B `{run_b.name}` vs ground truth", run_b.vs_ground_truth),
        "",
        # Laid out tall rather than wide: six columns of a single row overflow
        # the report pane in the Rerun blueprint and clip the last heading.
        "| A vs B rig poses (B aligned onto A) | Value |",
        "|---|---|",
        f"| samples matched | {comparison.pose_delta.num_matched} |",
        f"| position RMSE (mm) | {_number(comparison.pose_delta.position_rmse_millimeters)} |",
        f"| position max (mm) | {_number(comparison.pose_delta.position_max_millimeters)} |",
        f"| rotation RMSE (deg) | {_number(comparison.pose_delta.rotation_rmse_degrees, 3)} |",
        f"| rotation max (deg) | {_number(comparison.pose_delta.rotation_max_degrees, 3)} |",
        "",
        "## Runtime",
        "",
        "| Stage | A (s) | B (s) | B/A |",
        "|---|---|---|---|",
    ]
    for row in comparison.stages:
        lines.append(f"| {row.label} | {_number(row.seconds_a)} | {_number(row.seconds_b)} | {_number(row.ratio_b_over_a)} |")
    lines.append(
        f"| **total** | **{_number(run_a.total_runtime_seconds)}** | **{_number(run_b.total_runtime_seconds)}** "
        f"| **{_number(comparison.total_runtime_ratio)}** |"
    )

    if comparison.acceptance:
        lines += [
            "",
            "## Acceptance (plan bounds, B relative to A)",
            "",
            "| Check | Bound | Value | Result |",
            "|---|---|---|---|",
        ]
        for check in comparison.acceptance:
            lines.append(f"| {check.name} | {check.bound} | {_number(check.value, 3)} | {'PASS' if check.passed else 'FAIL'} |")
        passed: int = sum(1 for check in comparison.acceptance if check.passed)
        lines += ["", f"**{passed}/{len(comparison.acceptance)} bounds met.**"]
    return "\n".join(lines) + "\n"


def write_markdown_report(path: Path, comparison: Comparison) -> Path:
    """Write the markdown report.

    Args:
        path: Destination file; parent directories are created.
        comparison: The result of `compare_runs`.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(comparison))
    return path


def write_json_report(path: Path, comparison: Comparison) -> Path:
    """Dump the comparison as JSON through pyserde.

    Args:
        path: Destination file; parent directories are created.
        comparison: The result of `compare_runs`.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(comparison, indent=2) + "\n")
    return path


def print_comparison(comparison: Comparison) -> None:
    """Print the markdown report to stdout.

    Args:
        comparison: The result of `compare_runs`.
    """
    print(render_markdown_report(comparison))


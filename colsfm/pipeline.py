"""The cuSFM stages wired into one run — `colsfm`'s `cusfm_runner.run_all`.

```
frames_meta.json ─1─> keyframes/frames_meta.json
                 ─2─> database.db (ALIKED keypoints + descriptors)
                 ─3─> pair list (consecutive + stereo)
                 ─4─> LightGlue matches + two-view geometries in the database
                 ─5─> loop edges (optional; their pairs are matched into the same database)
                 ─6─> pose_graph/{frames_meta.json,vehicle_pose.tum}
                 ─7─> triangulation + bundle adjustment
                 ─7b> the same again with the rig extrinsics free (--optimize-extrinsics)
                 ─8─> sparse/, kpmap/keyframes/frames_meta.json, output_poses/, summary.json
```

The contract mirrors `pycusfm.cusfm_runner.CusfmRunner`: the same `--input-dir`
holding `frames_meta.json` and the images it names, the same output directory
layout (`pycusfm/constants.py`), and the same `runtime.csv` — two columns,
`command,runtime_seconds` — so a colsfm run and a blob run compare row by row.
The `command` column holds the stage name rather than a blob command line;
nothing reads it as a command, and the stage name is what a benchmark table wants.

Ordering notes worth knowing before reading the code:

* **Pair selection cannot see loop pairs.** Stage 3 emits the consecutive and
  stereo pairs from the metadata alone (`colsfm.pairs`); loop pairs only exist
  after retrieval and verification in stage 5, which matches them into the same
  database. `colsfm.mapping.load_correspondences` reads *every* verified
  two-view geometry, so a loop pair matched in stage 5 reaches the triangulator
  without being named in stage 3's list.
* **Loop closure needs matches to verify, and matching is a batch operation.**
  `colsfm.loop_closure.find_loop_edges` pulls matches through an injected
  `match_fn` one pair at a time, and running COLMAP's matcher per pair would
  reload the LightGlue graph per call. The stage therefore runs the search
  twice: once with a recording `match_fn` that returns nothing (which collects
  the candidate pairs — the candidate set depends only on retrieval, the score
  gate and the time gate, never on the matches), then one batch
  `colsfm.matching.match_pairs` over those pairs, then the real search reading
  matches back from the database.
* **`--use-gpu` governs the ONNX stages only.** ALIKED and LightGlue run on the
  GPU; the Ceres solves stay on the CPU by default, as the blob's own
  `keypoints_mapper_main` and `pose_graph_main` do (docs/open-pipeline-plan.md,
  "Facts that shape the design"). `--ba-use-gpu` moves the bundle-adjustment
  linear solve onto the GPU; measured it buys nothing on Galileo (1.68 s against
  1.60 s) and about 8 % on 800 RoboCap images (19.2 s against 20.9 s), so the
  default stays where the blob is.
* **Threads are not the blob's `--num_thread 1`.** That flag counts worker
  *processes*, and the runner starts one per camera; COLMAP is one process, so
  `--num-threads` defaults to -1 (every core) for extraction and matching, and
  Ceres takes the config's 8 unless `--ba-num-threads` says otherwise.
"""

from __future__ import annotations

import contextlib
import dataclasses
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Int
from numpy import ndarray
from serde import serde
from serde.json import to_json

from colsfm.cameras import colmap_cameras
from colsfm.config import CusfmConfig, KeyframeSelectionConfig, PoseGraphConfig, read_config_directory
from colsfm.database import ImagePair, create_database, raw_match_counts
from colsfm.export import (
    KEYFRAME_METADATA_SUBPATH,
    RUNTIME_CSV_NAME,
    RuntimeRecord,
    append_runtime_record,
    colour_points_from_images,
    write_colmap_model,
    write_pose_files,
    write_tum_file,
)
from colsfm.features import ExtractionReport, FeatureOptions, extract_features
from colsfm.frames_meta import CameraParams, FramesMeta, KeyframeMeta, RigFrame, read_frames_meta, write_frames_meta
from colsfm.geometry import TumPose, relative_rotation_degrees
from colsfm.keyframe_selection import KeyframeSelection, apply_selection, select_keyframes
from colsfm.mapping import MappingOptions, MappingResult, run_mapping
from colsfm.matching import BLOB_MATCH_TOP_K, MatchingOptions, MatchReport, match_pairs
from colsfm.pairs import select_pairs
from colsfm.pose_graph import PoseGraphEdge, PoseGraphResult, RigNode, sequential_edges, solve_pose_graph
from colsfm.reconstruction import RigReference, build_reconstruction, gauge_camera_params_id, rig_reference

StageName: TypeAlias = Literal[
    "keyframe_selection",
    "feature_extraction",
    "pair_selection",
    "matching",
    "loop_closure",
    "pose_graph",
    "reconstruction",
    "extrinsic_refinement",
    "export",
]
"""The stages, in the order `CusfmRunner.run_all` runs their blob equivalents."""

STAGE_NAMES: Final[tuple[StageName, ...]] = (
    "keyframe_selection",
    "feature_extraction",
    "pair_selection",
    "matching",
    "loop_closure",
    "pose_graph",
    "reconstruction",
    "export",
)
"""The stages every run records; `stage_names` adds the ninth for `--optimize-extrinsics`."""

EXTRINSIC_REFINEMENT_STAGE: Final[StageName] = "extrinsic_refinement"
"""The second mapping pass `--optimize-extrinsics` adds, mirroring the blob.

`CusfmRunner` runs `keypoints_mapper_main` twice: once with the extrinsics fixed
and once with `--optimize_extrinsics=True`, both over the same matches and the
same `pose_graph/frames_meta.json` poses (`pycusfm/cusfm_runner.py`,
`run_mapping` then `refine_extrinsics`). The second pass overwrites `kpmap/`."""


def stage_names(optimize_extrinsics: bool) -> tuple[StageName, ...]:
    """The stages a run will record, in order.

    Args:
        optimize_extrinsics: Whether the run makes the second mapping pass.

    Returns:
        `STAGE_NAMES`, with `extrinsic_refinement` inserted before `export` when
        the flag is set.
    """
    if not optimize_extrinsics:
        return STAGE_NAMES
    index: int = STAGE_NAMES.index("export")
    return STAGE_NAMES[:index] + (EXTRINSIC_REFINEMENT_STAGE,) + STAGE_NAMES[index:]


CheapStageName: TypeAlias = Literal["keyframe_selection", "pair_selection"]
"""The stages the `stage` subcommand can run on their own: metadata only, no GPU."""

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
"""Repo root, so the default config directory resolves from any working directory."""

DEFAULT_CONFIG_DIR: Final[Path] = REPO_ROOT / "data" / "cusfm_configs" / "loop-closure-fixed"
"""`isaac` with loop closure repaired; what the blob reference runs used."""

FRAMES_META_NAME: Final[str] = "frames_meta.json"
"""The metadata file name, in the input directory and in every output directory."""

KEYFRAME_DIR_NAME: Final[str] = "keyframes"
"""`pycusfm.constants.kKEYFRAME_DIR`: where the selected metadata lands."""

POSE_GRAPH_DIR_NAME: Final[str] = "pose_graph"
"""`pycusfm.constants.kPOSE_GRAPH_DIR`: where the optimised rig trajectory lands."""

DATABASE_NAME: Final[str] = "database.db"
"""The COLMAP database, which has no cuSFM equivalent (the blob writes keyframe protos)."""

VEHICLE_POSE_TUM_NAME: Final[str] = "vehicle_pose.tum"
"""`pose_graph_main`'s own rig trajectory file name."""

SUMMARY_NAME: Final[str] = "summary.json"
"""Machine-readable run report, written by this pipeline and by nothing in cuSFM."""

LOOP_PROBE_MATCHES: Final[Int[ndarray, "0 2"]] = np.zeros((0, 2), dtype=np.int64)
"""What the recording `match_fn` returns: no matches, so no candidate is verified."""


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    """Everything one run needs, mirroring `CusfmRunner`'s constructor arguments."""

    input_dir: Path
    """Directory holding `frames_meta.json` and the images its `image_name`s name."""
    output_dir: Path
    """Workspace root, cuSFM's `cusfm_base_dir`; created if missing."""
    config_dir: Path = DEFAULT_CONFIG_DIR
    """Directory of `.pb.txt` configs; the loop-closure-fixed isaac profile by default."""
    min_inter_frame_distance: float = 0.0
    """Keyframe displacement gate in metres; the demo passes 0.0 to keep every sample."""
    min_inter_frame_rotation_degrees: float = 5.0
    """Keyframe rotation gate in degrees, OR-combined with the distance gate."""
    sample_sync_threshold_microseconds: int = 100
    """Rig grouping window in microseconds; the isaac demo's value, not the gflag default."""
    optimize_extrinsics: bool = False
    """Refine `sensor_from_rig` during bundle adjustment; cuSFM's `--optimize_extrinsics`."""
    loop_closure: bool = False
    """Run retrieval-based loop closure. Off by default: the plan decides per dataset,
    and on the 0.93 s Galileo sweep loops move poses further than the ATE budget allows."""
    use_gpu: bool = True
    """Run ALIKED and LightGlue on the GPU when one is usable."""
    ba_use_gpu: bool = False
    """Hand the bundle-adjustment linear solve to the GPU. Off by default: the measured
    difference is under 10 % and changes sign with the problem size, so the default follows
    the blob, which solves on the CPU. See `colsfm.mapping.MappingOptions.use_gpu`."""
    num_threads: int = -1
    """Threads for feature extraction and matching; -1 lets COLMAP use every core.

    The blob runs `feature_extractor_main --num_thread 1`, but that flag counts *its own*
    worker processes, not the threads inside one: the runner launches one process per
    camera. COLMAP has a single process, so 1 serialises JPEG decode against the GPU and
    costs 4.4x on Galileo (33.9 s against 7.6 s, measured)."""
    ba_num_threads: int | None = None
    """Ceres threads; None takes `bundle_adjustment_config.num_threads` (8) from the config,
    which is what the blob solves with. Set 1 for a bit-reproducible solve."""
    max_matches_per_pair: int | None = BLOB_MATCH_TOP_K
    """Verified matches kept per pair after the spatial subsample; None keeps every inlier.
    The blob's `match_top_k`; see `colsfm.matching.subsample_matches_by_coverage`."""
    stop_on_observation_change: bool = True
    """Leave the mapper's outer loop once the observation change falls below the config's
    `max_observation_change`, instead of always running every round."""

    @property
    def database_path(self) -> Path:
        """Where the COLMAP database lives.

        Returns:
            `<output_dir>/database.db`.
        """
        return self.output_dir / DATABASE_NAME

    @property
    def frames_meta_path(self) -> Path:
        """Where the input metadata lives.

        Returns:
            `<input_dir>/frames_meta.json`.
        """
        return self.input_dir / FRAMES_META_NAME


@serde
@dataclass(frozen=True, slots=True)
class ExtrinsicChange:
    """How far the refinement moved one camera's `sensor_to_vehicle_transform`."""

    camera_params_id: int
    """The camera, as `camera_params_id_to_camera_params` keys it."""
    sensor_name: str
    """The camera folder name, e.g. `front_stereo_camera_left`."""
    translation_change_mm: float
    """Distance between the input and refined extrinsic origins, in millimetres."""
    rotation_change_deg: float
    """Angle between the input and refined extrinsic rotations, in degrees."""
    is_reference: bool
    """Whether this is the rig origin and fixed camera, whose change is exactly zero."""


@serde
@dataclass(frozen=True, slots=True)
class PipelineSummary:
    """`summary.json`: what one run produced and how long each stage took."""

    input_dir: Path
    """Input directory the run read."""
    output_dir: Path
    """Workspace the run wrote."""
    config_dir: Path
    """Config profile the run used."""
    git_sha: str
    """`git rev-parse HEAD` of the working tree, or `"unknown"` outside a checkout."""
    loop_closure_enabled: bool
    """Whether stage 5 ran the retrieval search or reported itself as a no-op."""
    num_input_keyframes: int
    """Keyframes in the input `frames_meta.json`."""
    num_selected_keyframes: int
    """Keyframes that survived keyframe selection."""
    num_rig_frames: int
    """Rig frames (`synced_sample_id` groups) among the selected keyframes."""
    num_pairs: int
    """Image pairs stage 3 selected, before any loop pair."""
    num_loop_pairs: int
    """Extra image pairs matched for loop closure; 0 when the stage is off."""
    num_loop_edges: int
    """Loop constraints that survived verification and gating."""
    num_pose_graph_edges: int
    """Constraints in the solved pose graph, sequential and loop together."""
    pose_graph_translation_change_m: float
    """Largest rig position change the pose-graph solve produced, in metres."""
    num_registered_images: int
    """Images belonging to a registered frame after mapping."""
    num_images_with_observations: int
    """Images observing at least one point; the count a COLMAP export writes out."""
    num_points3D: int
    """Triangulated points in the final model."""
    num_observations: int
    """Point-to-image observations in the final model."""
    mean_reprojection_error_px: float
    """Mean reprojection error over every observation, in pixels."""
    mean_track_length: float
    """Mean observations per point."""
    stage_seconds: dict[str, float]
    """Wall-clock seconds per stage, in stage order; the same rows as `runtime.csv`."""
    total_seconds: float
    """Wall-clock seconds for the whole run."""
    optimize_extrinsics: bool = False
    """Whether the run made the second, extrinsic-refining mapping pass."""
    extrinsic_changes: tuple[ExtrinsicChange, ...] = ()
    """Per-camera extrinsic movement the refinement produced; empty when the flag is off."""


@dataclass(slots=True)
class StageClock:
    """Times stages, appends each to `runtime.csv` and keeps them for the summary."""

    output_dir: Path
    """Workspace root holding `runtime.csv`."""
    stages: tuple[StageName, ...] = STAGE_NAMES
    """The stages this run will record, so the progress line counts the right total."""
    seconds_by_stage: dict[str, float] = field(default_factory=dict)
    """Wall-clock seconds per stage, in completion order."""

    def record(self, stage: StageName, seconds: float) -> None:
        """Store one stage's runtime and append it to the log.

        Args:
            stage: The stage that just finished.
            seconds: Its wall-clock duration.
        """
        self.seconds_by_stage[stage] = seconds
        append_runtime_record(self.output_dir, RuntimeRecord(command=stage, runtime_seconds=seconds))


@contextlib.contextmanager
def timed_stage(clock: StageClock, stage: StageName) -> Iterator[None]:
    """Time the enclosed block and record it as one stage.

    Args:
        clock: The clock to record into.
        stage: The stage the block implements.

    Yields:
        Nothing; the block runs inside the timing window.
    """
    started: float = time.perf_counter()
    print(f"[colsfm] stage {clock.stages.index(stage) + 1}/{len(clock.stages)}: {stage}")
    try:
        yield
    finally:
        elapsed: float = time.perf_counter() - started
        clock.record(stage, elapsed)
        print(f"[colsfm] {stage} finished in {elapsed:.2f}s")


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    """Read the working tree's commit, for the summary's provenance field.

    Args:
        repo_root: Directory inside the repository.

    Returns:
        The 40-character SHA, or `"unknown"` when git is unavailable or the
        directory is not a checkout.
    """
    try:
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def rig_nodes(frames_meta: FramesMeta) -> list[RigNode]:
    """Turn a collection's rig frames into pose-graph nodes.

    Args:
        frames_meta: The selected collection.

    Returns:
        One node per `synced_sample_id`, in rig-frame order, carrying `world_T_rig`
        as `FramesMeta.rig_frames` derives it (from the lowest keyframe id).
    """
    return [
        RigNode(
            rig_id=rig_frame.synced_sample_id,
            world_T_rig=rig_frame.world_T_vehicle,
            timestamp_us=rig_frame.timestamp_microseconds,
        )
        for rig_frame in frames_meta.rig_frames()
    ]


def camera_poses_from_rig_poses(
    frames_meta: FramesMeta,
    world_T_rig_by_rig_id: dict[int, pycolmap.Rigid3d],
) -> dict[int, pycolmap.Rigid3d]:
    """Derive every camera pose from the optimised rig poses.

    `world_T_cam = world_T_rig * rig_T_cam`, where `rig_T_cam` is the metadata's
    `sensor_to_vehicle_transform` — the composition `pose_graph_main` writes back
    into its own `frames_meta.json` (docs/spec/pose_graph_main.md §4).

    Args:
        frames_meta: The collection whose keyframes to re-pose.
        world_T_rig_by_rig_id: Optimised rig pose per `synced_sample_id`.

    Returns:
        `world_T_cam` per keyframe id; keyframes whose rig frame is missing from
        the mapping are left out, so the caller keeps their original pose.
    """
    world_T_cam_by_keyframe_id: dict[int, pycolmap.Rigid3d] = {}
    for keyframe in frames_meta.keyframes:
        world_T_rig: pycolmap.Rigid3d | None = world_T_rig_by_rig_id.get(keyframe.synced_sample_id)
        if world_T_rig is None:
            continue
        rig_T_cam: pycolmap.Rigid3d = frames_meta.cameras[keyframe.camera_params_id].vehicle_T_cam
        world_T_cam_by_keyframe_id[keyframe.keyframe_id] = world_T_rig * rig_T_cam
    return world_T_cam_by_keyframe_id


def optimised_camera_poses(reconstruction: pycolmap.Reconstruction) -> dict[int, pycolmap.Rigid3d]:
    """Read `world_T_cam` per image out of an adjusted reconstruction.

    Args:
        reconstruction: The model bundle adjustment left behind.

    Returns:
        `world_T_cam` per image id, for every image of a registered frame.
    """
    poses: dict[int, pycolmap.Rigid3d] = {}
    for image_id, image in reconstruction.images.items():
        if not image.has_pose:
            continue
        poses[image_id] = image.cam_from_world().inverse()
    return poses


def _write_vehicle_pose_file(path: Path, nodes: Sequence[RigNode], world_T_rig_by_rig_id: dict[int, pycolmap.Rigid3d]) -> None:
    """Write the rig trajectory `pose_graph_main` leaves in its output directory.

    Args:
        path: Destination `.tum` file; parent directories are created.
        nodes: The pose-graph nodes, for their timestamps.
        world_T_rig_by_rig_id: Optimised rig pose per `synced_sample_id`.
    """
    poses: list[TumPose] = [
        TumPose(timestamp_microseconds=node.timestamp_us, world_T_body=world_T_rig_by_rig_id[node.rig_id])
        for node in sorted(nodes, key=lambda entry: (entry.timestamp_us, entry.rig_id))
        if node.rig_id in world_T_rig_by_rig_id
    ]
    write_tum_file(path, poses)


def _largest_translation_change_m(nodes: Sequence[RigNode], world_T_rig_by_rig_id: dict[int, pycolmap.Rigid3d]) -> float:
    """Largest rig position change between the input poses and the solved ones.

    Args:
        nodes: The pose-graph nodes as they entered the solve.
        world_T_rig_by_rig_id: Optimised rig pose per `synced_sample_id`.

    Returns:
        The maximum displacement in metres, or 0.0 for an empty graph.
    """
    changes: list[float] = [
        float(np.linalg.norm(world_T_rig_by_rig_id[node.rig_id].translation - node.world_T_rig.translation))
        for node in nodes
        if node.rig_id in world_T_rig_by_rig_id
    ]
    return max(changes) if changes else 0.0


def extrinsic_changes(
    frames_meta: FramesMeta,
    refined_by_camera_params_id: Mapping[int, pycolmap.Rigid3d],
    reference_camera_params_id: int,
) -> tuple[ExtrinsicChange, ...]:
    """Measure what the refinement did to each camera's extrinsic.

    Args:
        frames_meta: The collection the refinement started from.
        refined_by_camera_params_id: `vehicle_T_cam` per camera after the solve.
        reference_camera_params_id: The rig origin, which is also the fixed camera.

    Returns:
        One record per camera, ordered by `camera_params_id`.
    """
    changes: list[ExtrinsicChange] = []
    for camera_params_id in sorted(refined_by_camera_params_id):
        camera: CameraParams = frames_meta.cameras[camera_params_id]
        refined: pycolmap.Rigid3d = refined_by_camera_params_id[camera_params_id]
        changes.append(
            ExtrinsicChange(
                camera_params_id=camera_params_id,
                sensor_name=camera.sensor_name,
                translation_change_mm=float(
                    np.linalg.norm(np.asarray(refined.translation) - np.asarray(camera.vehicle_T_cam.translation))
                )
                * 1e3,
                rotation_change_deg=relative_rotation_degrees(refined, camera.vehicle_T_cam),
                is_reference=camera_params_id == reference_camera_params_id,
            )
        )
    return tuple(changes)


def _print_mapping(mapping: MappingResult) -> None:
    """Print one mapping pass's headline counts.

    Args:
        mapping: What `run_mapping` returned.
    """
    print(
        f"[colsfm] mapping: {mapping.num_images_with_observations}/{mapping.num_registered_images} images with points, "
        f"{mapping.num_points3D} points, {mapping.mean_reprojection_error_px:.4f} px, "
        f"track {mapping.mean_track_length:.3f}"
    )


@dataclass(frozen=True, slots=True)
class LoopClosureStageResult:
    """What the loop-closure stage produced, whether it ran or not."""

    edges: list[PoseGraphEdge]
    """Verified, gated loop constraints; empty when the stage is off or found nothing."""
    num_pairs_matched: int
    """Candidate image pairs matched into the database for verification."""


def run_loop_closure_stage(
    frames_meta: FramesMeta,
    database_path: Path,
    pose_graph_config: PoseGraphConfig,
    matching_options: MatchingOptions,
    *,
    enabled: bool,
) -> LoopClosureStageResult:
    """Retrieve, match and verify loop candidates, guarded against a missing module.

    `colsfm.retrieval` and `colsfm.loop_closure` are owned by another worker, so
    they are imported here rather than at module scope and every call into them
    is guarded: a run with `--no-loop-closure` never touches them, and a run that
    asks for loops while the modules are absent or their API has moved says so
    and continues as a no-op rather than losing the other seven stages.

    The guard catches `TypeError` because that is what a changed signature (or a
    beartype hint violation, which subclasses it) raises; the message carries the
    exception so a mismatch is reported rather than silently swallowed.

    Args:
        frames_meta: The selected collection, in the pose graph's frame conventions.
        database_path: The database holding the keypoints and descriptors.
        pose_graph_config: `pose_graph_config.pb.txt`, whose loop gates this honours.
        matching_options: Settings for the batch match over the candidate pairs.
        enabled: Whether to run at all.

    Returns:
        The loop edges and how many pairs were matched to find them.
    """
    if not enabled:
        print("[colsfm] loop closure disabled (--loop-closure to enable); no loop edges")
        return LoopClosureStageResult(edges=[], num_pairs_matched=0)
    try:
        return _find_loop_edges(frames_meta, database_path, pose_graph_config, matching_options)
    except (ImportError, AttributeError, TypeError) as error:
        print(f"[colsfm] loop closure unavailable, running the stage as a no-op: {error!r}")
        return LoopClosureStageResult(edges=[], num_pairs_matched=0)


def _find_loop_edges(
    frames_meta: FramesMeta,
    database_path: Path,
    pose_graph_config: PoseGraphConfig,
    matching_options: MatchingOptions,
) -> LoopClosureStageResult:
    """Run retrieval, batch matching and verification over the whole collection.

    Args:
        frames_meta: The selected collection.
        database_path: The database holding the keypoints and descriptors.
        pose_graph_config: Supplies the loop gates the config profile sets.
        matching_options: Settings for the batch match over the candidate pairs.

    Returns:
        The loop edges and how many pairs were matched to find them.

    Raises:
        ImportError: When `colsfm.retrieval` or `colsfm.loop_closure` is absent.
        AttributeError: When either module lacks a name this stage needs.
        TypeError: When either module's signatures have moved.
    """
    from colsfm.loop_closure import LoopClosureConfig, LoopClosureResult, find_loop_edges
    from colsfm.retrieval import Descriptors, RetrievalIndex, build_retrieval_index, read_descriptors_from_database

    descriptors: dict[int, Descriptors] = read_descriptors_from_database(database_path)
    if not descriptors:
        print("[colsfm] loop closure: the database holds no descriptors; no loop edges")
        return LoopClosureStageResult(edges=[], num_pairs_matched=0)
    index: RetrievalIndex = build_retrieval_index(descriptors)
    print(f"[colsfm] loop closure: retrieval index over {len(index.image_ids)} images in {index.build_seconds:.2f}s")

    config: LoopClosureConfig = LoopClosureConfig(
        enabled=True,
        loop_closure_interval_ratio=pose_graph_config.loop_closure_interval_ratio,
        max_translation_m=pose_graph_config.loop_edge_translation_threshold_meters,
        max_rotation_deg=pose_graph_config.loop_edge_rotation_threshold_degrees,
        loop_residual_weight=pose_graph_config.loop_residual_weight,
    )
    requested: set[ImagePair] = set()

    def record_pair(image_id_a: int, image_id_b: int) -> Int[ndarray, "num_matches 2"]:
        """Record a candidate pair and return no matches, so nothing verifies."""
        requested.add((min(image_id_a, image_id_b), max(image_id_a, image_id_b)))
        return LOOP_PROBE_MATCHES

    find_loop_edges(frames_meta, database_path, index, config, record_pair)
    requested_pairs: list[ImagePair] = sorted(requested)
    # Most requested pairs are the consecutive and stereo pairs stage 4 already matched
    # (8 400 of 14 443 on RoboCap); matching them again would only cost time.
    already_matched: dict[ImagePair, int] = raw_match_counts(database_path, requested_pairs)
    candidate_pairs: list[ImagePair] = [pair for pair in requested_pairs if already_matched[pair] == 0]
    print(
        f"[colsfm] loop closure: {len(requested_pairs)} candidate pairs, "
        f"{len(requested_pairs) - len(candidate_pairs)} already matched, {len(candidate_pairs)} to match"
    )
    if candidate_pairs:
        match_pairs(database_path, candidate_pairs, matching_options)

    # One handle for the whole verification pass: a candidate set of tens of thousands of
    # pairs would otherwise pay a SQLite open and close per `match_fn` call.
    with pycolmap.Database.open(database_path) as database:

        def read_matches(image_id_a: int, image_id_b: int) -> Int[ndarray, "num_matches 2"]:
            """Read one pair's raw matches back out of the database."""
            return np.asarray(database.read_matches(image_id_a, image_id_b), dtype=np.int64)

        result: LoopClosureResult = find_loop_edges(frames_meta, database_path, index, config, read_matches)
    edges: list[PoseGraphEdge] = list(result.edges)
    # Print the counters, not just the edge count: zero edges is the normal outcome on a
    # short or a low-recall sequence, and only the rejection breakdown says which it was.
    print(
        f"[colsfm] loop closure: {len(edges)} loop edges | {result.diagnostics.queries} queries, "
        f"{result.diagnostics.candidates_retrieved} retrieved, {result.diagnostics.rejected_by_score} below score, "
        f"{result.diagnostics.rejected_by_time} inside the {result.diagnostics.min_time_gap_seconds:.2f}s gap, "
        f"{result.diagnostics.rejected_by_geometry} failed geometry, {result.diagnostics.rejected_by_is_good} not good, "
        f"{result.diagnostics.verified} verified over {len(requested_pairs)} matched pairs"
    )
    return LoopClosureStageResult(edges=edges, num_pairs_matched=len(requested_pairs))


def run_pipeline(options: PipelineOptions) -> PipelineSummary:
    """Run all eight stages and write cuSFM's outputs into `options.output_dir`.

    Args:
        options: Input, output, config and the per-stage knobs.

    Returns:
        The run's summary, which is also written to `<output_dir>/summary.json`.

    Raises:
        FileNotFoundError: When the input metadata or a config file is missing.
        ValueError: When a stage cannot proceed, e.g. no keyframe survives selection.
    """
    started: float = time.perf_counter()
    if not options.frames_meta_path.is_file():
        raise FileNotFoundError(f"No {FRAMES_META_NAME} at {options.frames_meta_path}")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_csv: Path = options.output_dir / RUNTIME_CSV_NAME
    runtime_csv.unlink(missing_ok=True)
    clock: StageClock = StageClock(output_dir=options.output_dir, stages=stage_names(options.optimize_extrinsics))

    config: CusfmConfig = read_config_directory(
        options.config_dir,
        KeyframeSelectionConfig(
            min_inter_frame_distance_m=options.min_inter_frame_distance,
            min_inter_frame_rotation_degrees=options.min_inter_frame_rotation_degrees,
            sample_sync_threshold_microseconds=options.sample_sync_threshold_microseconds,
        ),
    )
    print(f"[colsfm] config {options.config_dir} | input {options.input_dir} | output {options.output_dir}")

    # ── 1. keyframe selection ────────────────────────────────────────────────────────
    with timed_stage(clock, "keyframe_selection"):
        frames_meta: FramesMeta = read_frames_meta(options.frames_meta_path)
        selection: KeyframeSelection = select_keyframes(
            frames_meta,
            min_inter_frame_distance_m=config.keyframe_selection.min_inter_frame_distance_m,
            min_inter_frame_rotation_degrees=config.keyframe_selection.min_inter_frame_rotation_degrees,
            sample_sync_threshold_microseconds=config.keyframe_selection.sample_sync_threshold_microseconds,
        )
        selected: FramesMeta = apply_selection(frames_meta, selection)
        if not selected.keyframes:
            raise ValueError("keyframe selection kept no frames; loosen the distance and rotation gates")
        write_frames_meta(options.output_dir / KEYFRAME_DIR_NAME / FRAMES_META_NAME, selected)
        rig_frames: tuple[RigFrame, ...] = selected.rig_frames()
        print(
            f"[colsfm] selected {len(selected.keyframes)}/{len(frames_meta.keyframes)} keyframes "
            f"in {len(rig_frames)} rig frames over {len(selected.cameras)} cameras"
        )

    # ── 2. feature extraction ────────────────────────────────────────────────────────
    with timed_stage(clock, "feature_extraction"):
        create_database(options.database_path, selected, colmap_cameras(selected), overwrite=True)
        feature_options: FeatureOptions = FeatureOptions(
            num_threads=options.num_threads,
            device="auto" if options.use_gpu else "cpu",
        )
        extraction: ExtractionReport = extract_features(
            options.database_path,
            options.input_dir,
            [keyframe.image_name for keyframe in selected.keyframes],
            feature_options,
        )

    # ── 3. pair selection ────────────────────────────────────────────────────────────
    with timed_stage(clock, "pair_selection"):
        pairs: list[ImagePair] = select_pairs(selected, config.pose_graph.connected_keyframe_num)
        print(
            f"[colsfm] {len(pairs)} pairs from connected_keyframe_num="
            f"{config.pose_graph.connected_keyframe_num} and {len(selected.stereo_pairs)} stereo declarations"
        )

    # ── 4. matching and verification ─────────────────────────────────────────────────
    matching_options: MatchingOptions = MatchingOptions(
        max_error_px=config.matching_task_worker.verification.max_pixel_error,
        confidence=config.matching_task_worker.verification.min_ransac_confidence,
        num_threads=options.num_threads,
        device="auto" if options.use_gpu else "cpu",
        max_matches_per_pair=options.max_matches_per_pair,
    )
    with timed_stage(clock, "matching"):
        match_report: MatchReport = match_pairs(options.database_path, pairs, matching_options)

    # ── 5. loop closure ──────────────────────────────────────────────────────────────
    with timed_stage(clock, "loop_closure"):
        loops: LoopClosureStageResult = run_loop_closure_stage(
            selected, options.database_path, config.pose_graph, matching_options, enabled=options.loop_closure
        )

    # ── 6. pose graph ────────────────────────────────────────────────────────────────
    with timed_stage(clock, "pose_graph"):
        nodes: list[RigNode] = rig_nodes(selected)
        edges: list[PoseGraphEdge] = sequential_edges(nodes, config.pose_graph.connected_keyframe_num)
        edges.extend(loops.edges)
        world_T_rig_by_rig_id: dict[int, pycolmap.Rigid3d] = {node.rig_id: node.world_T_rig for node in nodes}
        if edges:
            solved: PoseGraphResult = solve_pose_graph(nodes, edges)
            world_T_rig_by_rig_id = solved.world_T_rig
            print(
                f"[colsfm] pose graph: {len(nodes)} nodes, {len(edges)} edges "
                f"({len(loops.edges)} loop), {solved.termination} after {solved.iterations} iterations, "
                f"cost {solved.initial_cost:.6g} -> {solved.final_cost:.6g}"
            )
        else:
            print("[colsfm] pose graph: a single rig frame has no edges; the input poses stand")
        # Written unconditionally, even with the solve a no-op: the camera poses are
        # re-derived through the rig, which is what makes the collection rigid and what
        # `pose_graph_main` itself writes out.
        pose_graph_meta: FramesMeta = selected.with_camera_to_world(
            camera_poses_from_rig_poses(selected, world_T_rig_by_rig_id)
        )
        pose_graph_dir: Path = options.output_dir / POSE_GRAPH_DIR_NAME
        write_frames_meta(pose_graph_dir / FRAMES_META_NAME, pose_graph_meta)
        _write_vehicle_pose_file(pose_graph_dir / VEHICLE_POSE_TUM_NAME, nodes, world_T_rig_by_rig_id)
        pose_graph_change_m: float = _largest_translation_change_m(nodes, world_T_rig_by_rig_id)

    mapping_options: MappingOptions = MappingOptions(
        num_threads=options.ba_num_threads,
        use_gpu=options.ba_use_gpu,
        stop_on_observation_change=options.stop_on_observation_change,
    )

    # ── 7. triangulation and bundle adjustment ───────────────────────────────────────
    with timed_stage(clock, "reconstruction"):
        mapping: MappingResult = run_mapping(
            build_reconstruction(pose_graph_meta),
            options.database_path,
            config.vision_mapping,
            config.vision_mapping.bundle_adjustment,
            mapping_options,
        )
        _print_mapping(mapping)

    # ── 7b. extrinsic refinement ─────────────────────────────────────────────────────
    # The blob's second `keypoints_mapper_main` pass: the same matches and the same
    # pose-graph poses again, this time with `--optimize_extrinsics=True`, overwriting
    # `kpmap/` (`pycusfm.cusfm_runner.refine_extrinsics`). The rig origin moves onto the
    # gauge camera because COLMAP freezes every extrinsic of a rig whose reference
    # sensor owns no images — see `colsfm.reconstruction`.
    mapped_meta: FramesMeta = pose_graph_meta
    changes: tuple[ExtrinsicChange, ...] = ()
    if options.optimize_extrinsics:
        with timed_stage(clock, EXTRINSIC_REFINEMENT_STAGE):
            reference_camera_params_id: int = gauge_camera_params_id(pose_graph_meta)
            reference: RigReference = rig_reference(pose_graph_meta, reference_camera_params_id)
            mapping = run_mapping(
                build_reconstruction(pose_graph_meta, reference_camera_params_id=reference_camera_params_id),
                options.database_path,
                config.vision_mapping,
                config.vision_mapping.bundle_adjustment,
                dataclasses.replace(mapping_options, optimize_extrinsics=True),
                rig_reference=reference,
            )
            assert mapping.refined_extrinsics is not None, "run_mapping returns them whenever it refines"
            changes = extrinsic_changes(pose_graph_meta, mapping.refined_extrinsics, reference_camera_params_id)
            mapped_meta = pose_graph_meta.with_extrinsics(mapping.refined_extrinsics)
            _print_mapping(mapping)
            print(
                f"[colsfm] extrinsic refinement: rig origin on camera {reference_camera_params_id} "
                f"({pose_graph_meta.cameras[reference_camera_params_id].sensor_name}), moves "
                + ", ".join(f"{change.camera_params_id}={change.translation_change_mm:.2f} mm" for change in changes)
            )

    # ── 8. export ────────────────────────────────────────────────────────────────────
    with timed_stage(clock, "export"):
        num_coloured: int = colour_points_from_images(mapping.reconstruction, options.input_dir, options.num_threads)
        print(f"[colsfm] export: coloured {num_coloured}/{mapping.num_points3D} points from the source imagery")
        write_colmap_model(options.output_dir, mapping.reconstruction)
        # `camera_to_world` comes straight off the adjusted reconstruction, so it stays
        # consistent with whatever extrinsics were just written next to it.
        optimised: FramesMeta = mapped_meta.with_camera_to_world(
            optimised_camera_poses(mapping.reconstruction), "ALIGNMENT"
        )
        write_frames_meta(options.output_dir / KEYFRAME_METADATA_SUBPATH, optimised)
        written: list[Path] = write_pose_files(options.output_dir, optimised)
        print(f"[colsfm] export: sparse/ model, {KEYFRAME_METADATA_SUBPATH}, {len(written)} TUM files")

    summary: PipelineSummary = PipelineSummary(
        input_dir=options.input_dir,
        output_dir=options.output_dir,
        config_dir=options.config_dir,
        git_sha=git_sha(),
        loop_closure_enabled=options.loop_closure,
        num_input_keyframes=len(frames_meta.keyframes),
        num_selected_keyframes=len(selected.keyframes),
        num_rig_frames=len(rig_frames),
        num_pairs=len(pairs),
        num_loop_pairs=loops.num_pairs_matched,
        num_loop_edges=len(loops.edges),
        num_pose_graph_edges=len(edges),
        pose_graph_translation_change_m=pose_graph_change_m,
        num_registered_images=mapping.num_registered_images,
        num_images_with_observations=mapping.num_images_with_observations,
        num_points3D=mapping.num_points3D,
        num_observations=mapping.num_observations,
        mean_reprojection_error_px=mapping.mean_reprojection_error_px,
        mean_track_length=mapping.mean_track_length,
        stage_seconds=dict(clock.seconds_by_stage),
        total_seconds=time.perf_counter() - started,
        optimize_extrinsics=options.optimize_extrinsics,
        extrinsic_changes=changes,
    )
    (options.output_dir / SUMMARY_NAME).write_text(to_json(summary) + "\n")
    print(
        f"[colsfm] done in {summary.total_seconds:.2f}s | "
        f"{summary.num_images_with_observations} images, {summary.num_points3D} points, "
        f"{summary.mean_reprojection_error_px:.4f} px | extraction on {extraction.device}, "
        f"{match_report.empty_pairs} empty pairs"
    )
    return summary


@dataclass(frozen=True, slots=True)
class StageResult:
    """What one standalone `stage` invocation produced."""

    stage: CheapStageName
    """The stage that ran."""
    num_selected_keyframes: int
    """Keyframes that survived selection."""
    num_rig_frames: int
    """Rig frames among them."""
    num_pairs: int
    """Pairs the pair-selection stage would match; 0 when only selection ran."""


def run_stage(options: PipelineOptions, stage: CheapStageName) -> StageResult:
    """Run one metadata-only stage, for inspecting a dataset without touching a GPU.

    Args:
        options: The same options a full run takes; only the metadata knobs matter.
        stage: `keyframe_selection` or `pair_selection`.

    Returns:
        The counts the stage produced.

    Raises:
        FileNotFoundError: When the input metadata is missing.
    """
    if not options.frames_meta_path.is_file():
        raise FileNotFoundError(f"No {FRAMES_META_NAME} at {options.frames_meta_path}")
    config: CusfmConfig = read_config_directory(
        options.config_dir,
        KeyframeSelectionConfig(
            min_inter_frame_distance_m=options.min_inter_frame_distance,
            min_inter_frame_rotation_degrees=options.min_inter_frame_rotation_degrees,
            sample_sync_threshold_microseconds=options.sample_sync_threshold_microseconds,
        ),
    )
    frames_meta: FramesMeta = read_frames_meta(options.frames_meta_path)
    selection: KeyframeSelection = select_keyframes(
        frames_meta,
        min_inter_frame_distance_m=config.keyframe_selection.min_inter_frame_distance_m,
        min_inter_frame_rotation_degrees=config.keyframe_selection.min_inter_frame_rotation_degrees,
        sample_sync_threshold_microseconds=config.keyframe_selection.sample_sync_threshold_microseconds,
    )
    selected: FramesMeta = apply_selection(frames_meta, selection)
    keyframes: tuple[KeyframeMeta, ...] = selected.keyframes
    pairs: list[ImagePair] = (
        select_pairs(selected, config.pose_graph.connected_keyframe_num) if stage == "pair_selection" else []
    )
    print(
        f"[colsfm] {stage}: {len(keyframes)}/{len(frames_meta.keyframes)} keyframes, "
        f"{len(selected.rig_frames())} rig frames, {len(pairs)} pairs"
    )
    return StageResult(
        stage=stage,
        num_selected_keyframes=len(keyframes),
        num_rig_frames=len(selected.rig_frames()),
        num_pairs=len(pairs),
    )

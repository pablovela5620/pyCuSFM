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

## One function per stage

Each stage is a `run_<stage>_stage` function taking what it needs and returning a
small result dataclass; `run_pipeline` only composes them and times each with
`timed_stage`. The stage functions carry no timing and write no `runtime.csv`
row, so a caller that wants to step through a dataset one stage at a time — the
Rerun walkthrough, `run_stage`, a notebook — reuses exactly the code the full run
executes rather than a parallel copy of it.

Ordering notes worth knowing before reading the code:

* **Pair selection cannot see loop pairs.** Stage 3 emits the consecutive and
  stereo pairs from the metadata alone (`colsfm.pairs`); loop pairs only exist
  after retrieval and verification in stage 5, which matches them into the same
  database. `colsfm.mapping.load_correspondences` reads *every* verified
  two-view geometry, so a loop pair matched in stage 5 reaches the triangulator
  without being named in stage 3's list.
* **Loop closure needs matches to verify, and matching is a batch operation.**
  The estimator pulls matches through an injected `match_fn` one pair at a time,
  and running COLMAP's matcher per pair would reload the LightGlue graph per
  call. So the stage plans first: `colsfm.loop_closure.plan_loop_search` does the
  retrieval half of the funnel — which reads no match at all — and names the
  image pairs the measurement will ask for; one batch
  `colsfm.matching.match_pairs` fills in whichever of them the database does not
  already hold; `verify_loop_plan` then measures and gates the plan, reading the
  matches back. Presence decides what to match, not count: a pair stage 4
  matched and found nothing in is finished work.
* **`--optimize-extrinsics` maps once, not twice.** The camera-referenced
  reconstruction is built up front when the flag is set, stage 7 maps it, and
  stage 7b refines the extrinsics of the model stage 7 left behind. Both
  references give the same `cam_T_world`, so nothing about stage 7's result
  depends on which one was used (`colsfm.reconstruction`).
* **The two ONNX stages have two backends each.** `--features-backend` and
  `--matching-backend` pick between COLMAP's own ALIKED and LightGlue (the
  default) and the blob's graphs on the blob's TensorRT engines
  (`colsfm.features_trt`, `colsfm.matching_trt`). They are independent, all four
  combinations run, and the choice is invisible past the database: both write the
  same keypoint and descriptor columns and leave the same matches and two-view
  geometries. The TensorRT matcher is the only path that sees a per-match score,
  so it is the only one that runs the blob's real SSC spatial NMS.
* **`--use-gpu` governs the ONNX stages only.** ALIKED and LightGlue run on the
  GPU; the Ceres solves stay on the CPU by default, as the blob's own
  `keypoints_mapper_main` and `pose_graph_main` do (docs/open-pipeline-plan.md,
  "Facts that shape the design"). It is read by the `pycolmap` backends only —
  the TensorRT ones run on CUDA device 0 or not at all. `--ba-use-gpu` moves the
  bundle-adjustment linear solve onto the GPU; measured it buys nothing on Galileo (1.68 s against
  1.60 s) and about 8 % on 800 RoboCap images (19.2 s against 20.9 s), so the
  default stays where the blob is.
* **Threads are not the blob's `--num_thread 1`.** That flag counts worker
  *processes*, and the runner starts one per camera; COLMAP is one process, so
  `--num-threads` defaults to -1 (every core) for extraction and matching, and
  Ceres takes the config's 8 unless `--ba-num-threads` says otherwise.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
from jaxtyping import Float, Int
from numpy import ndarray
from serde import serde
from serde.json import to_json

from colsfm import REPO_ROOT
from colsfm.ba_backend import BaBackend, BaExecutionPlan, resolve_ba_plan
from colsfm.config import CusfmConfig, KeyframeSelectionConfig, PoseGraphConfig, read_config_directory
from colsfm.database import (
    Descriptors,
    ImagePair,
    Keypoints,
    create_database,
    pairs_with_matches,
    read_descriptors_from_database,
    read_keypoints_batch,
)
from colsfm.export import (
    colour_points_from_images,
    write_colmap_model,
    write_optimised_frames_meta,
    write_pose_files,
    write_tum_file,
)
from colsfm.extrinsic_refinement import ExtrinsicRefinementOptions, ExtrinsicRefinementResult, refine_extrinsics
from colsfm.features import DeviceChoice, ExtractionReport, FeatureBackend, FeatureOptions, extract_features
from colsfm.frames_meta import FRAMES_META_NAME, CameraParams, FramesMeta, RigFrame, read_frames_meta, write_frames_meta
from colsfm.geometry import MILLIMETRES_PER_METRE, TumPose, relative_rotation_degrees
from colsfm.keyframe_selection import KeyframeSelection, apply_selection, select_keyframes
from colsfm.loop_closure import (
    LoopClosureConfig,
    LoopClosureDiagnostics,
    LoopClosureResult,
    LoopSearchPlan,
    plan_loop_search,
    verify_loop_plan,
)
from colsfm.mapping import MappingOptions, MappingResult, PolishStats, run_mapping
from colsfm.matching import (
    BLOB_MATCH_TOP_K,
    MatchCapMode,
    MatchingBackend,
    MatchingOptions,
    MatchLimitPolicy,
    MatchReport,
    match_pairs,
)
from colsfm.pairs import select_pairs
from colsfm.pose_graph import PoseGraphEdge, PoseGraphResult, RigNode, sequential_edges, solve_pose_graph
from colsfm.reconstruction import PosedModel, build_reconstruction, gauge_camera_params_id
from colsfm.retrieval import RetrievalIndex, build_retrieval_index
from colsfm.run_lifecycle import (
    DATABASE_NAME,
    EXTRINSIC_REFINEMENT_STAGE,
    KEYFRAME_DIR_NAME,
    LOOP_EDGES_NAME,
    POSE_GRAPH_DIR_NAME,
    SUMMARY_NAME,
    VEHICLE_POSE_TUM_NAME,
    CheapStageName,
    RunWorkspace,
    StageLedger,
    StageName,
    git_sha,
    open_workspace,
    stage_names,
    timed_stage,
)

DEFAULT_CONFIG_DIR: Final[Path] = REPO_ROOT / "data" / "cusfm_configs" / "loop-closure-fixed"
"""`isaac` with loop closure repaired; what the blob reference runs used."""


@serde
@dataclass(frozen=True, slots=True)
class SelectionOptions:
    """What keyframe and pair selection need, and nothing else.

    The metadata-only half of `PipelineOptions`, resolved out of it once at the
    start of a run. `python -m colsfm stage` takes exactly this: those two stages
    read `frames_meta.json` and a config profile, touch no image, open no GPU and
    write nothing, so requiring an output directory of them was asking for a
    workspace nobody would fill.
    """

    input_dir: Path
    """Directory holding `frames_meta.json` and the images its `image_name`s name."""
    config_dir: Path = DEFAULT_CONFIG_DIR
    """Directory of `.pb.txt` configs; the loop-closure-fixed isaac profile by default."""
    min_inter_frame_distance: float = 0.0
    """Keyframe displacement gate in metres; the demo passes 0.0 to keep every sample."""
    min_inter_frame_rotation_degrees: float = 5.0
    """Keyframe rotation gate in degrees, OR-combined with the distance gate."""
    sample_sync_threshold_microseconds: int = 100
    """Rig grouping window in microseconds; the isaac demo's value, not the gflag default."""

    @property
    def frames_meta_path(self) -> Path:
        """Where the input metadata lives.

        Returns:
            `<input_dir>/frames_meta.json`.
        """
        return self.input_dir / FRAMES_META_NAME


@serde
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
    regularised_extrinsics: bool = True
    """Carry the blob's extrinsic priors through `colsfm.extrinsic_refinement`. False falls
    back to pycolmap's own rig bundle adjustment, which has no prior terms and on Galileo
    overfits (NOTES.md deviation 9). Only read when `optimize_extrinsics` is set."""
    extrinsic_refinement_rounds: int = 20
    """Ceiling on the extrinsics-then-poses rounds the regularised refinement may take; it
    stops early on its own tolerances, after 19 rounds and 3.2 s on Galileo."""
    loop_closure: bool = True
    """Run retrieval-based loop closure. On by default, as in the blob, so every run and
    every blob comparison pays for the same stage (NOTES.md decision 10, revised). On the
    0.93 s Galileo sweep the 10 s loop gate rejects every candidate, so the stage costs time
    and changes nothing; `--no-loop-closure` is the explicit ablation."""
    use_gpu: bool = True
    """Run ALIKED and LightGlue on the GPU when one is usable. A bool rather than a
    `DeviceChoice` because `--use-gpu` is the flag the pixi tasks and the plan use; the
    `device` property below is the one place it turns into a device request."""
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
    ba_backend: BaBackend = "ceres"
    """Which implementation solves the bundle adjustments — Ceres on the CPU, or COLMAP's
    CASPAR on the GPU. `caspar` needs a CASPAR-enabled pycolmap and falls back to `ceres`,
    loudly, when it is missing, when `--optimize-extrinsics` is set, or when a camera model
    is one this build's CASPAR has no adapter for. Which models those are is measured at
    run time (`colsfm.mapping.caspar_supported_camera_models`): PINHOLE and SIMPLE_RADIAL
    in `colsfm-caspar` and `colsfm-caspar64`, plus OPENCV_FISHEYE in
    `colsfm-caspar-fisheye` (`docs/caspar-fisheye-adapter.md`). It also drops the robust
    loss; see `colsfm.mapping`'s module docstring."""
    ba_num_threads: int | None = None
    """Ceres threads; None takes `bundle_adjustment_config.num_threads` (8) from the config,
    which is what the blob solves with. Set 1 for a bit-reproducible solve."""
    caspar_ceres_polish: bool = True
    """Finish a `--ba-backend caspar` run with one Ceres global bundle adjustment.

    On by default: it is what makes the GPU backend shippable — KITTI 06 goes from
    1.299 m to 0.904 m Sim(3) ATE for 9.8 s on top of 31.5 s of CASPAR rounds, against
    the Ceres mapper's 0.895 m in 148.5 s (`docs/caspar-build.md` § Shipped: CASPAR +
    Ceres polish). `--no-caspar-ceres-polish` is the ablation. Inert on `ceres`,
    fallbacks included, which have already converged."""
    caspar_option: tuple[str, ...] = ()
    """CASPAR solver overrides as `name=value`, e.g. `--caspar-option solver_iter_max=1000
    pcg_iter_max=80`. Read only when `--ba-backend caspar` actually runs; see
    `colsfm.mapping.CasparOptions` for the names and `docs/caspar-build.md`
    § Convergence sweep for what moving them buys."""
    max_matches_per_pair: int | None = BLOB_MATCH_TOP_K
    """Verified matches kept per pair after the spatial subsample; None keeps every inlier.
    The blob's `match_top_k`; see `colsfm.matching.subsample_matches_by_coverage`."""
    match_cap_mode: MatchCapMode = "fixed"
    """How `max_matches_per_pair` becomes a per-pair number; see
    `colsfm.matching.MatchLimitPolicy`. `fixed` is the 500 every run before the KITTI
    follow-up used, `image_area` scales it with the frame's resolution, `off` keeps every
    verified inlier."""
    features_backend: FeatureBackend = "pycolmap"
    """Which ALIKED runs: COLMAP's own ONNX one, or the blob's TensorRT engine
    (`colsfm.features_trt`). See `colsfm.features.FeatureBackend`."""
    matching_backend: MatchingBackend = "pycolmap"
    """Which LightGlue runs: COLMAP's own ONNX one, or the blob's TensorRT engine
    (`colsfm.matching_trt`), which is the only path with the per-match score the blob's
    SSC spatial NMS needs. See `colsfm.matching.MatchingBackend`."""

    @property
    def selection(self) -> SelectionOptions:
        """The metadata-only options, resolved out of the command line's spellings.

        Returns:
            The five gates and paths keyframe and pair selection read, so those
            stages never see the twenty-one fields they have no use for.
        """
        return SelectionOptions(
            input_dir=self.input_dir,
            config_dir=self.config_dir,
            min_inter_frame_distance=self.min_inter_frame_distance,
            min_inter_frame_rotation_degrees=self.min_inter_frame_rotation_degrees,
            sample_sync_threshold_microseconds=self.sample_sync_threshold_microseconds,
        )

    @property
    def device(self) -> DeviceChoice:
        """Device request for the two ONNX stages, derived from `use_gpu` exactly once.

        Returns:
            `auto` — which falls back to the CPU provider when CUDA or cuDNN is
            missing — or `cpu` when `use_gpu` is off.
        """
        return "auto" if self.use_gpu else "cpu"

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
class MappingStats:
    """The reconstruction counters `summary.json` reports, as one block.

    A snapshot of `colsfm.mapping.MappingResult`'s properties at the moment the
    summary was written: the result itself computes them from the live
    reconstruction, which is exactly why they cannot be stored there.
    """

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
    ba_backend: BaBackend = "ceres"
    """Which backend the bundle adjustments actually ran on, fallbacks applied; the one
    field here that is not a counter, and the only record a finished run keeps of it."""
    polish_seconds: float | None = None
    """Seconds the closing Ceres polish of a CASPAR run took, or None when none ran.

    They are already inside the `reconstruction` stage of `runtime.csv` — the polish
    solves inside `run_mapping` — so this names them rather than adding a row."""
    polish_iterations: int | None = None
    """Ceres iterations that polish took; the measure of how far CASPAR stopped short."""
    ba_fallback_reason: str | None = None
    """Why `ba_backend` is not `options.ba_backend`, or None when it is.

    A `--ba-backend caspar` run that solved on Ceres used to say so on stdout and
    nowhere else, so a finished run could not be asked which backend produced it.
    The sentences are `colsfm.ba_backend`'s: an unsupported camera model,
    `--optimize-extrinsics`, or a pycolmap built without CASPAR_ENABLED."""

    @staticmethod
    def of(mapping: MappingResult) -> MappingStats:
        """Snapshot a mapping result's counters.

        Args:
            mapping: What the last mapping pass produced.

        Returns:
            The six counters, the backend that produced them and its closing polish,
            frozen at this point in the run.
        """
        polish: PolishStats | None = mapping.polish
        return MappingStats(
            num_registered_images=mapping.num_registered_images,
            num_images_with_observations=mapping.num_images_with_observations,
            num_points3D=mapping.num_points3D,
            num_observations=mapping.num_observations,
            mean_reprojection_error_px=mapping.mean_reprojection_error_px,
            mean_track_length=mapping.mean_track_length,
            ba_backend=mapping.ba_backend,
            polish_seconds=None if polish is None else polish.seconds,
            polish_iterations=None if polish is None else polish.ba_num_iterations,
            ba_fallback_reason=mapping.ba_fallback_reason,
        )


@serde
@dataclass(frozen=True, slots=True)
class PipelineSummary:
    """`summary.json`: what one run produced and how long each stage took."""

    options: PipelineOptions
    """The options the run was given, verbatim; the run's own provenance."""
    git_sha: str
    """`git rev-parse HEAD` of the working tree, or `"unknown"` outside a checkout."""
    num_input_keyframes: int
    """Keyframes in the input `frames_meta.json`."""
    num_selected_keyframes: int
    """Keyframes that survived keyframe selection."""
    num_rig_frames: int
    """Rig frames (`synced_sample_id` groups) among the selected keyframes."""
    num_pairs: int
    """Image pairs stage 3 selected, before any loop pair."""
    num_loop_pairs: int
    """Extra image pairs matched for loop closure; 0 when the stage is off. Only the
    pairs stage 5 had to match itself are counted — most candidates are consecutive or
    stereo pairs stage 4 already matched, and matching them again would only cost time."""
    num_loop_edges: int
    """Loop constraints that survived verification and gating."""
    num_pose_graph_edges: int
    """Constraints in the solved pose graph, sequential and loop together."""
    pose_graph_translation_change_m: float
    """Largest rig position change the pose-graph solve produced, in metres."""
    mapping: MappingStats
    """The final model's counters, after the extrinsic refinement when one ran."""
    stage_seconds: dict[str, float]
    """Wall-clock seconds per stage, in stage order; the same rows as `runtime.csv`."""
    total_seconds: float
    """Wall-clock seconds for the whole run."""
    extrinsic_changes: tuple[ExtrinsicChange, ...] = ()
    """Per-camera extrinsic movement the refinement produced; empty when the flag is off."""
    extrinsic_refinement_rounds_run: int = 0
    """Rounds the regularised refinement actually took before it converged."""


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


@serde
@dataclass(frozen=True, slots=True)
class LoopEdgeRecord:
    """One gated loop constraint, in a form a viewer can read without pycolmap.

    `PoseGraphEdge` carries a `pycolmap.Rigid3d` and a 6x6 NumPy array, neither of
    which pyserde can write; this is the same measurement flattened to plain
    numbers. The information matrix is reduced to its diagonal because that is all
    `colsfm.pose_graph.default_information` ever puts in it — the off-diagonal
    blocks are zero by construction — and the diagonal is what a viewer would
    render as an edge weight.
    """

    source: int
    """`rig_id` (cuSFM's `synced_sample_id`) of the source rig frame."""
    target: int
    """`rig_id` of the target rig frame."""
    source_T_target_quaternion_xyzw: tuple[float, float, float, float]
    """Rotation of the measurement, as `pycolmap.Rotation3d.quat` orders it: x, y, z, w."""
    source_T_target_translation: tuple[float, float, float]
    """Translation of the measurement, in metres."""
    information_diagonal: tuple[float, float, float, float, float, float]
    """Diagonal of the 6x6 weight, rotation block first then translation."""
    kind: str
    """`PoseGraphEdge.kind`; always `"loop"` in this file, kept so the record is self-describing."""


def loop_edge_record(edge: PoseGraphEdge) -> LoopEdgeRecord:
    """Flatten one pose-graph edge into its serialisable form.

    Args:
        edge: The constraint, as `colsfm.loop_closure` gated it.

    Returns:
        The same measurement as plain numbers.
    """
    quaternion: Float[ndarray, " 4"] = np.asarray(edge.source_T_target.rotation.quat, dtype=np.float64)
    translation: Float[ndarray, " 3"] = np.asarray(edge.source_T_target.translation, dtype=np.float64)
    diagonal: Float[ndarray, " 6"] = np.asarray(edge.information, dtype=np.float64).diagonal()
    return LoopEdgeRecord(
        source=int(edge.source),
        target=int(edge.target),
        source_T_target_quaternion_xyzw=(
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        ),
        source_T_target_translation=(float(translation[0]), float(translation[1]), float(translation[2])),
        information_diagonal=(
            float(diagonal[0]),
            float(diagonal[1]),
            float(diagonal[2]),
            float(diagonal[3]),
            float(diagonal[4]),
            float(diagonal[5]),
        ),
        kind=edge.kind,
    )


def write_loop_edges(path: Path, loop_edges: Sequence[PoseGraphEdge]) -> None:
    """Write the gated loop constraints beside the pose-graph outputs.

    Written on every run, empty list included: a viewer that draws loop edges needs
    to tell "the stage ran and found none" from "the file is not there yet", and an
    absent file cannot say the first.

    Args:
        path: Destination `.json` file; parent directories are created.
        loop_edges: The constraints stage 5 produced; may be empty.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json([loop_edge_record(edge) for edge in loop_edges]) + "\n")


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

    The per-camera translation and rotation are the same two quantities
    `colsfm.extrinsic_refinement.extrinsic_deltas` maximises over, computed the
    same way — `MILLIMETRES_PER_METRE` times the origin distance, and
    `colsfm.geometry.relative_rotation_degrees` for the angle.

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
        translation_m: float = float(
            np.linalg.norm(np.asarray(refined.translation) - np.asarray(camera.vehicle_T_cam.translation))
        )
        changes.append(
            ExtrinsicChange(
                camera_params_id=camera_params_id,
                sensor_name=camera.sensor_name,
                translation_change_mm=MILLIMETRES_PER_METRE * translation_m,
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


# ══════════════════════════════════════════════════════════════════════════════════════
# the stages
# ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class KeyframeSelectionStageResult:
    """Stage 1: the configuration read and the keyframes that survived selection."""

    config: CusfmConfig
    """The whole config profile, with the command line's selection gates folded in."""
    frames_meta: FramesMeta
    """The input collection, unfiltered."""
    selected: FramesMeta
    """The kept keyframes, renumbered into the dense 1-based rig numbering."""
    rig_frames: tuple[RigFrame, ...]
    """Rig frames among the selected keyframes, one per `synced_sample_id`, in time
    order. Bound once here: `FramesMeta.rig_frames` regroups the whole collection on
    every call, and the stage's own print line and the summary both want it."""


def run_keyframe_selection_stage(options: SelectionOptions) -> KeyframeSelectionStageResult:
    """Read the config and the input metadata and apply cuSFM's keyframe selection.

    Writes nothing; `run_pipeline` is what saves `keyframes/frames_meta.json`, so
    `run_stage` can inspect a dataset without creating a workspace.

    Args:
        options: The metadata-only options: the config directory and the three gates.

    Returns:
        The configuration, the input collection and the selected one.

    Raises:
        FileNotFoundError: When the input metadata or a config file is missing.
        ValueError: When no keyframe survives selection.
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
    if not selected.keyframes:
        raise ValueError("keyframe selection kept no frames; loosen the distance and rotation gates")
    return KeyframeSelectionStageResult(
        config=config, frames_meta=frames_meta, selected=selected, rig_frames=selected.rig_frames()
    )


def run_feature_extraction_stage(
    options: PipelineOptions, selected: FramesMeta, feature_options: FeatureOptions | None = None
) -> ExtractionReport:
    """Create the COLMAP database and run ALIKED over the selected keyframes.

    Args:
        options: The run's options, for the database path and the image root.
        selected: The collection keyframe selection kept.
        feature_options: The extractor's resolved settings; derived from `options`
            when None, which is what a caller stepping through one stage wants.

    Returns:
        The extraction report: per-image keypoint counts, the device used and the
        wall time.

    Raises:
        FileNotFoundError: When the image root is missing.
    """
    resolved: FeatureOptions = (
        FeatureOptions(backend=options.features_backend, num_threads=options.num_threads, device=options.device)
        if feature_options is None
        else feature_options
    )
    create_database(options.database_path, selected, overwrite=True)
    return extract_features(
        options.database_path,
        options.input_dir,
        [keyframe.image_name for keyframe in selected.keyframes],
        resolved,
    )


def run_pair_selection_stage(selected: FramesMeta, config: CusfmConfig) -> list[ImagePair]:
    """Enumerate the pairs `feature_matcher_task_builder_main` would have written.

    Args:
        selected: The collection keyframe selection kept.
        config: The config profile, for `connected_keyframe_num`.

    Returns:
        The normalised, deduplicated, sorted image pairs; no loop pair is here yet.

    Raises:
        ValueError: When `connected_keyframe_num` is below 1.
    """
    pairs: list[ImagePair] = select_pairs(selected, config.pose_graph.connected_keyframe_num)
    print(
        f"[colsfm] {len(pairs)} pairs from connected_keyframe_num="
        f"{config.pose_graph.connected_keyframe_num} and {len(selected.stereo_pairs)} stereo declarations"
    )
    return pairs


def run_matching_stage(
    options: PipelineOptions, pairs: Sequence[ImagePair], matching_options: MatchingOptions
) -> MatchReport:
    """Match and verify every selected pair into the database.

    Args:
        options: The run's options, for the database path.
        pairs: The pairs stage 3 selected.
        matching_options: Matching and verification settings.

    Returns:
        The match report: per-pair raw and inlier counts, the device used and the
        wall time.
    """
    return match_pairs(options.database_path, list(pairs), matching_options)


@dataclass(frozen=True, slots=True)
class LoopClosureStageResult:
    """Stage 5: what the loop-closure search produced, whether it ran or not."""

    edges: list[PoseGraphEdge]
    """Verified, gated loop constraints; empty when the stage is off or found nothing."""
    num_pairs_matched: int
    """Image pairs this stage had to match itself, i.e. the candidates stage 4 had not
    already matched. The candidates it reused are not counted: `summary.json` reports
    the *extra* matching work loop closure cost."""
    diagnostics: LoopClosureDiagnostics | None = None
    """`find_loop_edges`' own rejection breakdown, or None when the stage did not run.

    Zero loop edges is the normal outcome on a short or low-recall sequence, and only
    these counters say which it was. The stage already prints them; carrying the object
    means a caller — a viewer, a notebook — can read them instead of parsing stdout.
    None on the two paths that return before the search: loop closure switched off, and
    a database with no descriptors."""


def _warn_about_closed_loop_gates(pose_graph_config: PoseGraphConfig) -> None:
    """Say so, loudly, when the config's loop gates will reject every edge.

    `gate_loop_edges` accepts a candidate whose relative pose is within
    `max_translation_m` and `max_rotation_deg` of the retrieval estimate, so a
    threshold of exactly 0.0 accepts nothing. Both are proto3 scalars, so the
    stock `pycusfm/configs/isaac` and `pycusfm/configs/av` profiles — which set
    neither — read back as 0.0 and silently discard 100 % of the edges after
    paying for the whole stage. Only `data/cusfm_configs/loop-closure-fixed`
    sets them (1.0 m / 10 deg). The gates keep their meaning; this only refuses
    to be quiet about the outcome.

    Args:
        pose_graph_config: `pose_graph_config.pb.txt` as it was parsed.
    """
    if pose_graph_config.loop_edge_translation_threshold_meters > 0.0:
        return
    if pose_graph_config.loop_edge_rotation_threshold_degrees > 0.0:
        return
    print(
        "[colsfm] WARNING: loop closure is on but both loop gates are 0.0 "
        "(loop_edge_translation_threshold_meters and loop_edge_rotation_threshold_degrees), "
        "so every candidate edge will be rejected and the stage will produce nothing. "
        "The stock isaac and av profiles leave both unset; use "
        "--config-dir data/cusfm_configs/loop-closure-fixed, or set the two thresholds."
    )


def run_loop_closure_stage(
    frames_meta: FramesMeta,
    database_path: Path,
    pose_graph_config: PoseGraphConfig,
    matching_options: MatchingOptions,
    *,
    enabled: bool,
) -> LoopClosureStageResult:
    """Retrieve, match and verify loop candidates.

    Three steps, none of which matches anything it does not have to: `plan_loop_search`
    decides which rig pairs to measure and which image pairs that needs, one batch
    `match_pairs` fills in whatever the database does not already hold, and
    `verify_loop_plan` measures and gates the plan.

    Args:
        frames_meta: The selected collection, in the pose graph's frame conventions.
        database_path: The database holding the keypoints and descriptors.
        pose_graph_config: `pose_graph_config.pb.txt`, whose loop gates this honours.
        matching_options: Settings for the batch match over the missing pairs.
        enabled: Whether to run at all.

    Returns:
        The loop edges and how many extra pairs were matched to find them.
    """
    if not enabled:
        print("[colsfm] loop closure disabled (--loop-closure to enable); no loop edges")
        return LoopClosureStageResult(edges=[], num_pairs_matched=0)
    _warn_about_closed_loop_gates(pose_graph_config)

    descriptors: dict[int, Descriptors] = read_descriptors_from_database(database_path)
    if not descriptors:
        print("[colsfm] loop closure: the database holds no descriptors; no loop edges")
        return LoopClosureStageResult(edges=[], num_pairs_matched=0)
    index: RetrievalIndex = build_retrieval_index(descriptors)
    print(f"[colsfm] loop closure: retrieval index over {len(index.image_ids)} images in {index.build_seconds:.2f}s")

    config: LoopClosureConfig = LoopClosureConfig.from_pose_graph(pose_graph_config)
    # Read once and handed to the plan: the keypoints are the same file either side of
    # the batch match, and on RoboCap they are 148 MB.
    keypoints: dict[int, Keypoints] = read_keypoints_batch(database_path, index.image_ids, dtype=np.float64)

    plan: LoopSearchPlan = plan_loop_search(frames_meta, database_path, index, config, keypoints)
    # Presence, not count: a pair stage 4 matched and found nothing in is finished work,
    # and matching it again would cost a matcher call to produce the same nothing. Most
    # of the plan is stage 4's own consecutive and stereo pairs (8 400 of 14 443 on
    # RoboCap), which is what makes this check worth making at all.
    present: set[ImagePair] = pairs_with_matches(database_path, plan.image_pairs)
    missing_pairs: list[ImagePair] = [pair for pair in plan.image_pairs if pair not in present]
    print(
        f"[colsfm] loop closure: {len(plan.rig_pairs)} shortlisted rig pairs need "
        f"{len(plan.image_pairs)} image pairs, {len(present)} already matched, {len(missing_pairs)} to match"
    )
    if missing_pairs:
        match_pairs(database_path, missing_pairs, matching_options)

    # One handle for the whole verification pass: a plan of tens of thousands of pairs
    # would otherwise pay a SQLite open and close per `match_fn` call.
    with pycolmap.Database.open(database_path) as database:

        def read_matches(image_id_a: int, image_id_b: int) -> Int[ndarray, "num_matches 2"]:
            """Read one pair's raw matches back out of the database."""
            return np.asarray(database.read_matches(image_id_a, image_id_b), dtype=np.int64)

        result: LoopClosureResult = verify_loop_plan(plan, read_matches)
    edges: list[PoseGraphEdge] = list(result.edges)
    # Print the counters, not just the edge count: zero edges is the normal outcome on a
    # short or a low-recall sequence, and only the rejection breakdown says which it was.
    print(
        f"[colsfm] loop closure: {len(edges)} loop edges | {result.diagnostics.queries} queries, "
        f"{result.diagnostics.candidates_retrieved} retrieved, {result.diagnostics.rejected_by_score} below score, "
        f"{result.diagnostics.rejected_by_time} inside the {result.diagnostics.min_time_gap_seconds:.2f}s gap, "
        f"{result.diagnostics.rejected_by_geometry} failed geometry, {result.diagnostics.rejected_by_is_good} not good, "
        f"{result.diagnostics.verified} verified over {len(plan.image_pairs)} image pairs"
    )
    return LoopClosureStageResult(
        edges=edges, num_pairs_matched=len(missing_pairs), diagnostics=result.diagnostics
    )


@dataclass(frozen=True, slots=True)
class PoseGraphStageResult:
    """Stage 6: the solved rig trajectory and the collection re-posed through it."""

    nodes: list[RigNode]
    """The rig nodes as they entered the solve, carrying the input poses."""
    edges: list[PoseGraphEdge]
    """Sequential and loop constraints together, in the order they were solved."""
    frames_meta: FramesMeta
    """The selected collection with every camera pose re-derived through the rig."""
    translation_change_m: float
    """Largest rig position change the solve produced, in metres."""
    solve: PoseGraphResult | None = None
    """The Ceres summary and the solved poses, or None when there was nothing to solve.

    None exactly when `edges` is empty, which happens only on a single-rig-frame
    collection; the input poses stand and no solver ran."""


def run_pose_graph_stage(
    options: PipelineOptions, selected: FramesMeta, config: CusfmConfig, loop_edges: Sequence[PoseGraphEdge]
) -> PoseGraphStageResult:
    """Solve the rig pose graph and write `pose_graph/`.

    Args:
        options: The run's options, for the output directory.
        selected: The collection keyframe selection kept.
        config: The config profile, for `connected_keyframe_num`.
        loop_edges: Extra constraints from stage 5; may be empty.

    Returns:
        The nodes, the edges, the re-posed collection and the largest move.
    """
    nodes: list[RigNode] = rig_nodes(selected)
    edges: list[PoseGraphEdge] = sequential_edges(nodes, config.pose_graph.connected_keyframe_num)
    edges.extend(loop_edges)
    world_T_rig_by_rig_id: dict[int, pycolmap.Rigid3d] = {node.rig_id: node.world_T_rig for node in nodes}
    solved: PoseGraphResult | None = None
    if edges:
        solved = solve_pose_graph(nodes, edges)
        world_T_rig_by_rig_id = solved.world_T_rig
        print(
            f"[colsfm] pose graph: {len(nodes)} nodes, {len(edges)} edges "
            f"({len(loop_edges)} loop), {solved.termination} after {solved.iterations} iterations, "
            f"cost {solved.initial_cost:.6g} -> {solved.final_cost:.6g}"
        )
    else:
        print("[colsfm] pose graph: a single rig frame has no edges; the input poses stand")
    # Written unconditionally, even with the solve a no-op: the camera poses are
    # re-derived through the rig, which is what makes the collection rigid and what
    # `pose_graph_main` itself writes out.
    pose_graph_meta: FramesMeta = selected.with_camera_to_world(camera_poses_from_rig_poses(selected, world_T_rig_by_rig_id))
    pose_graph_dir: Path = options.output_dir / POSE_GRAPH_DIR_NAME
    write_frames_meta(pose_graph_dir / FRAMES_META_NAME, pose_graph_meta)
    _write_vehicle_pose_file(pose_graph_dir / VEHICLE_POSE_TUM_NAME, nodes, world_T_rig_by_rig_id)
    write_loop_edges(pose_graph_dir / LOOP_EDGES_NAME, loop_edges)
    return PoseGraphStageResult(
        nodes=nodes,
        edges=edges,
        frames_meta=pose_graph_meta,
        translation_change_m=_largest_translation_change_m(nodes, world_T_rig_by_rig_id),
        solve=solved,
    )


def run_reconstruction_stage(
    options: PipelineOptions, pose_graph_meta: FramesMeta, config: CusfmConfig, mapping_options: MappingOptions
) -> MappingResult:
    """Triangulate and bundle-adjust the pose-graph poses.

    With `--optimize-extrinsics` the reconstruction is built on the **gauge
    camera** rather than the vehicle body, because COLMAP freezes every
    `sensor_from_rig` of a rig whose reference sensor owns no images
    (`colsfm.reconstruction`). Every `cam_T_world` is identical either way, so
    this pass produces the same model and stage 7b can refine the extrinsics of
    the very model it left behind instead of mapping the scene a second time.

    Args:
        options: The run's options; `optimize_extrinsics` chooses the rig origin.
        pose_graph_meta: The collection stage 6 re-posed.
        config: The config profile, for `vision_mapping_config`.
        mapping_options: Command-line style mapper knobs.

    Returns:
        The adjusted reconstruction, its rig reference and the per-round statistics.

    Raises:
        FileNotFoundError: When the database does not exist.
        ValueError: When no frame is registered.
    """
    reference_camera_params_id: int | None = (
        gauge_camera_params_id(pose_graph_meta) if options.optimize_extrinsics else None
    )
    model: PosedModel = build_reconstruction(pose_graph_meta, reference_camera_params_id=reference_camera_params_id)
    mapping: MappingResult = run_mapping(model, options.database_path, config.vision_mapping, mapping_options)
    _print_mapping(mapping)
    return mapping


@dataclass(frozen=True, slots=True)
class ExtrinsicRefinementStageResult:
    """Stage 7b: the alternating extrinsic refinement's model and its movement."""

    mapping: MappingResult
    """The refined mapping result; the same reconstruction, further adjusted."""
    frames_meta: FramesMeta
    """The pose-graph collection carrying the refined `sensor_to_vehicle_transform`."""
    changes: tuple[ExtrinsicChange, ...]
    """Per-camera movement, ordered by `camera_params_id`."""
    rounds_run: int
    """Alternation rounds the refinement took before it converged."""


def run_extrinsic_refinement_stage(
    options: PipelineOptions,
    pose_graph_meta: FramesMeta,
    config: CusfmConfig,
    mapping_options: MappingOptions,
    mapping: MappingResult,
) -> ExtrinsicRefinementStageResult:
    """Refine the rig extrinsics of the model stage 7 produced.

    The blob's second `keypoints_mapper_main` pass: the same matches and the same
    pose-graph poses, this time with `--optimize_extrinsics=True`, overwriting
    `kpmap/` (`pycusfm.cusfm_runner.refine_extrinsics`).

    Args:
        options: The run's options; the refinement's ceiling and the prior switch.
        pose_graph_meta: The collection stage 6 re-posed — what both priors pull towards.
        config: The config profile, for the sigmas and the solver settings.
        mapping_options: Command-line style mapper knobs, for the (B) half of the alternation.
        mapping: What stage 7 produced, on a camera-referenced rig.

    Returns:
        The refined mapping result, the re-calibrated collection and the movement.

    Raises:
        ValueError: When stage 7 built a vehicle-referenced rig, whose extrinsics
            COLMAP would silently hold constant.
    """
    refinement: ExtrinsicRefinementResult = refine_extrinsics(
        mapping,
        options.database_path,
        config.vision_mapping,
        mapping_options,
        {camera_params_id: camera.vehicle_T_cam for camera_params_id, camera in pose_graph_meta.cameras.items()},
        ExtrinsicRefinementOptions(
            regularised=options.regularised_extrinsics,
            num_rounds=options.extrinsic_refinement_rounds,
            use_absolute_prior=config.vision_mapping.use_camera_extrinsic_constraint,
            use_relative_prior=config.vision_mapping.use_camera_extrinsic_constraint,
        ),
    )
    reference_camera_params_id: int = gauge_camera_params_id(pose_graph_meta)
    changes: tuple[ExtrinsicChange, ...] = extrinsic_changes(
        pose_graph_meta, refinement.refined_extrinsics, reference_camera_params_id
    )
    _print_mapping(refinement.mapping)
    print(
        f"[colsfm] extrinsic refinement "
        f"({'regularised' if refinement.regularised else 'unregularised'}): rig origin on camera "
        f"{reference_camera_params_id} "
        f"({pose_graph_meta.cameras[reference_camera_params_id].sensor_name}), moves "
        + ", ".join(f"{change.camera_params_id}={change.translation_change_mm:.2f} mm" for change in changes)
    )
    return ExtrinsicRefinementStageResult(
        mapping=refinement.mapping,
        frames_meta=pose_graph_meta.with_extrinsics(refinement.refined_extrinsics),
        changes=changes,
        rounds_run=len(refinement.rounds),
    )


@dataclass(frozen=True, slots=True)
class ExportStageResult:
    """Stage 8: the four artifacts a cuSFM run leaves behind."""

    optimised: FramesMeta
    """The collection written to `kpmap/keyframes/frames_meta.json`."""
    num_coloured: int
    """Points that got a colour from the source imagery."""
    pose_files: list[Path]
    """Every TUM trajectory written, per-camera files first."""


def run_export_stage(options: PipelineOptions, mapped_meta: FramesMeta, mapping: MappingResult) -> ExportStageResult:
    """Colour the points and write `sparse/`, `kpmap/` and `output_poses/`.

    Args:
        options: The run's options, for the output directory and the image root.
        mapped_meta: The collection carrying whatever extrinsics were just solved.
        mapping: The final mapping result.

    Returns:
        The exported collection and what was written.

    Raises:
        FileNotFoundError: When the image root is missing.
    """
    num_coloured: int = colour_points_from_images(mapping.reconstruction, options.input_dir, options.num_threads)
    print(f"[colsfm] export: coloured {num_coloured}/{mapping.num_points3D} points from the source imagery")
    write_colmap_model(options.output_dir, mapping.reconstruction)
    # `camera_to_world` comes straight off the adjusted reconstruction, so it stays
    # consistent with whatever extrinsics were just written next to it.
    optimised: FramesMeta = write_optimised_frames_meta(
        options.output_dir, mapped_meta, optimised_camera_poses(mapping.reconstruction), "ALIGNMENT"
    )
    written: list[Path] = write_pose_files(options.output_dir, optimised)
    print(f"[colsfm] export: sparse/ model, kpmap/keyframes/{FRAMES_META_NAME}, {len(written)} TUM files")
    return ExportStageResult(optimised=optimised, num_coloured=num_coloured, pose_files=written)


# ══════════════════════════════════════════════════════════════════════════════════════
# resolved configuration
# ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    """`PipelineOptions` turned into what each stage actually takes, once per run.

    `PipelineOptions` is an *input adapter*: twenty-one fields spanning unrelated
    stages, kept exactly as the command line spells them and as `summary.json`
    records them. It used to be carried unresolved through the whole run, so a
    stage read whichever fields it happened to need, some combinations were inert
    (`--match-cap-mode off` and a `None` budget both meant "no cap"), and the
    CASPAR strings were not parsed until feature extraction, matching, loop
    closure and the pose graph had already been paid for.

    Resolution happens once, before the workspace is created, and it validates:
    `--caspar-option solver_iter_max=2.9` stops the run here rather than being
    truncated to 2 at the first solve. What cannot be resolved this early is the
    matcher's verification block, whose thresholds live in the config profile
    stage 1 reads — `matching_options` folds those in and is the only place that
    does.
    """

    selection: SelectionOptions
    """Stages 1 and 3: the metadata gates and the config directory."""
    features: FeatureOptions
    """Stage 2: which ALIKED runs, on what device, with how many threads."""
    matching_backend: MatchingBackend
    """Stage 4: which LightGlue runs."""
    match_limit: MatchLimitPolicy
    """Stages 4 and 5: how many verified matches one pair may keep, as one policy."""
    mapping: MappingOptions
    """Stages 7 and 7b: the mapper's own knobs, with the validated CASPAR overrides."""
    ba_plan: BaExecutionPlan
    """Which backend the bundle adjustments were planned on, and why, before any solve."""
    device: DeviceChoice
    """Device request the two ONNX stages share; `--use-gpu` resolved exactly once."""
    num_threads: int
    """Threads for extraction and matching; -1 lets COLMAP use every core."""

    def matching_options(self, config: CusfmConfig) -> MatchingOptions:
        """The matcher's settings, once the config profile's verification block is known.

        Args:
            config: The profile stage 1 read.

        Returns:
            The resolved matching settings, identical for stage 4 and stage 5's
            batch match — one object, so the two cannot drift.
        """
        return MatchingOptions(
            backend=self.matching_backend,
            max_error_px=config.matching_task_worker.verification.max_pixel_error,
            confidence=config.matching_task_worker.verification.min_ransac_confidence,
            num_threads=self.num_threads,
            device=self.device,
            match_limit=self.match_limit,
        )


def resolve_run(options: PipelineOptions) -> ResolvedRun:
    """Turn one command line into the per-stage settings the run will execute.

    Called first, before the output directory exists, so everything cheap is
    validated before anything is created or replaced.

    Args:
        options: The command line, verbatim.

    Returns:
        The resolved settings for every stage.

    Raises:
        ValueError: When a `--caspar-option` item is malformed, names an option
            that is not a settable numeric CASPAR knob, or gives an integer knob a
            fractional value.
    """
    ba_plan: BaExecutionPlan = resolve_ba_plan(
        options.ba_backend,
        options.caspar_option,
        ceres_polish=options.caspar_ceres_polish,
        optimize_extrinsics=options.optimize_extrinsics,
    )
    return ResolvedRun(
        selection=options.selection,
        features=FeatureOptions(
            backend=options.features_backend, num_threads=options.num_threads, device=options.device
        ),
        matching_backend=options.matching_backend,
        match_limit=MatchLimitPolicy.of(options.match_cap_mode, options.max_matches_per_pair),
        # `ba_backend` stays the *request*: the camera-model fallback needs the
        # reconstruction, which does not exist until stage 7, so `colsfm.mapping`
        # makes that call and reports it back as `MappingResult.ba_fallback_reason`.
        mapping=MappingOptions(
            num_threads=options.ba_num_threads,
            use_gpu=options.ba_use_gpu,
            ba_backend=ba_plan.requested_backend,
            caspar_ceres_polish=options.caspar_ceres_polish,
            caspar_options=ba_plan.caspar_options or None,
        ),
        ba_plan=ba_plan,
        device=options.device,
        num_threads=options.num_threads,
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# the observer
# ══════════════════════════════════════════════════════════════════════════════════════


class PipelineObserver:
    """What a bystander is told while `run_pipeline` runs; every method a no-op.

    The one seam a viewer, a notebook or a report needs to watch a run without
    re-sequencing it. Each `on_*` method is handed the stage's own typed result,
    *after* its timing window has closed, so nothing an observer does lands in
    `runtime.csv`; `stage_scope` is the only hook inside the window, and exists
    because the stage functions print their counters rather than returning them.

    Two ordering guarantees the visualisation depends on:

    * `on_reconstruction` runs before stage 7b starts, so an observer that wants
      the un-refined model sees it before the refinement mutates it in place.
    * `on_export` runs after `run_export_stage` has written every artifact, so a
      reader may open them.

    Subclass and override what you need. The base class is what `run_pipeline`
    uses when no observer is passed.
    """

    def run_started(self, options: PipelineOptions, stages: tuple[StageName, ...]) -> None:
        """Announce the run before stage 1.

        Args:
            options: The options the stages will execute, with `output_dir` already
                rebased onto the run's staging directory — so an observer that opens
                the database or reads an artifact mid-run looks where the run is
                actually writing. `summary.options` keeps the caller's own paths.
            stages: The stages it will record, in order.
        """

    def stage_scope(self, stage: StageName) -> contextlib.AbstractContextManager[None]:
        """Wrap one stage's body, inside its timing window.

        Args:
            stage: The stage about to run.

        Returns:
            A context manager entered around the stage call; the default does
            nothing. An observer that wants a stage's console output redirects
            stdout here.
        """
        return contextlib.nullcontext()

    def on_keyframe_selection(self, result: KeyframeSelectionStageResult) -> None:
        """Stage 1 finished.

        Args:
            result: The configuration read, the input collection and the selection.
        """

    def on_feature_extraction(self, report: ExtractionReport) -> None:
        """Stage 2 finished.

        Args:
            report: Per-image keypoint counts, the device used and the wall time.
        """

    def on_pair_selection(self, pairs: Sequence[ImagePair]) -> None:
        """Stage 3 finished.

        Args:
            pairs: The consecutive and stereo pairs the matcher will be given.
        """

    def on_matching(self, report: MatchReport) -> None:
        """Stage 4 finished.

        Args:
            report: Per-pair raw and inlier counts, the device used and the wall time.
        """

    def on_loop_closure(self, result: LoopClosureStageResult) -> None:
        """Stage 5 finished.

        Args:
            result: The gated loop edges, the extra pairs matched and the diagnostics.
        """

    def on_pose_graph(self, result: PoseGraphStageResult) -> None:
        """Stage 6 finished.

        Args:
            result: The nodes, the edges, the re-posed collection and the largest move.
        """

    def on_reconstruction(self, mapping: MappingResult) -> None:
        """Stage 7 finished, before stage 7b may refine the same model in place.

        Args:
            mapping: The triangulated, bundle-adjusted model and its round statistics.
        """

    def on_extrinsic_refinement(self, result: ExtrinsicRefinementStageResult) -> None:
        """Stage 7b finished; only called when `--optimize-extrinsics` is set.

        Args:
            result: The refined model, the re-calibrated collection and the movement.
        """

    def on_export(self, result: ExportStageResult, mapping: MappingResult) -> None:
        """Stage 8 finished and every artifact but `summary.json` is on disk.

        Args:
            result: The exported collection and what was written.
            mapping: The final mapping result the export was made from.
        """

    def run_finished(self, summary: PipelineSummary) -> None:
        """The run is over and `summary.json` is written.

        Args:
            summary: What the run produced and how long each stage took.
        """


# ══════════════════════════════════════════════════════════════════════════════════════
# the run
# ══════════════════════════════════════════════════════════════════════════════════════


def run_pipeline(options: PipelineOptions, observer: PipelineObserver | None = None) -> PipelineSummary:
    """Run every stage and write cuSFM's outputs into `options.output_dir`.

    The only sequencer in the package: a caller that wants to watch a run — the
    Rerun walkthrough, a notebook — passes an observer rather than composing the
    stage functions itself.

    Args:
        options: Input, output, config and the per-stage knobs.
        observer: Told about each stage as it completes; None watches nothing.

    Returns:
        The run's summary, which is also written to `<output_dir>/summary.json`.

    Raises:
        FileNotFoundError: When the input metadata or a config file is missing.
        ValueError: When a stage cannot proceed, e.g. no keyframe survives selection.
    """
    watcher: PipelineObserver = PipelineObserver() if observer is None else observer
    started: float = time.perf_counter()
    if not options.frames_meta_path.is_file():
        raise FileNotFoundError(f"No {FRAMES_META_NAME} at {options.frames_meta_path}")
    # Resolved and validated before the workspace exists: a `--caspar-option` typo
    # used to surface at the first bundle adjustment, after feature extraction,
    # matching and loop closure had already been paid for and the previous run's
    # database had already been deleted.
    resolved: ResolvedRun = resolve_run(options)
    stages: tuple[StageName, ...] = stage_names(options.optimize_extrinsics)
    workspace: RunWorkspace = open_workspace(options.output_dir, stages)
    # Every stage writes into the staging directory, the database included, because
    # the stages take an options object whose `output_dir` is that directory. The
    # summary below reports `options`, i.e. the paths the caller asked for.
    staged: PipelineOptions = replace(options, output_dir=workspace.staging_dir)
    ledger: StageLedger = StageLedger(output_dir=workspace.staging_dir, stages=stages)
    print(f"[colsfm] config {options.config_dir} | input {options.input_dir} | output {options.output_dir}")
    watcher.run_started(staged, stages)
    try:
        summary: PipelineSummary = _run_stages(options, staged, resolved, ledger, watcher, started)
    except BaseException as error:
        workspace.fail(ledger, f"{type(error).__name__}: {error}")
        raise
    workspace.publish(ledger)
    watcher.run_finished(summary)
    return summary


def _run_stages(
    options: PipelineOptions,
    staged: PipelineOptions,
    resolved: ResolvedRun,
    ledger: StageLedger,
    watcher: PipelineObserver,
    started: float,
) -> PipelineSummary:
    """Run every stage into the staging workspace and write its summary there.

    Split out of `run_pipeline` so that the publication boundary is one `try` around
    one call rather than a `finally` wrapped around nine stages.

    Args:
        options: The caller's own options, which the summary reports.
        staged: The same options with `output_dir` on the staging directory, which
            every stage writes into.
        resolved: The per-stage settings, resolved once.
        ledger: The run's stage ledger.
        watcher: The observer, told about each stage as it completes.
        started: `time.perf_counter()` at the start of the run.

    Returns:
        The summary, already written to the staging directory.

    Raises:
        FileNotFoundError: When a config file is missing.
        ValueError: When a stage cannot proceed.
    """

    # ── 1. keyframe selection ────────────────────────────────────────────────────────
    with timed_stage(ledger, "keyframe_selection"), watcher.stage_scope("keyframe_selection"):
        selection: KeyframeSelectionStageResult = run_keyframe_selection_stage(resolved.selection)
        write_frames_meta(staged.output_dir / KEYFRAME_DIR_NAME / FRAMES_META_NAME, selection.selected)
        print(
            f"[colsfm] selected {len(selection.selected.keyframes)}/{len(selection.frames_meta.keyframes)} keyframes "
            f"in {len(selection.rig_frames)} rig frames over {len(selection.selected.cameras)} cameras"
        )
    watcher.on_keyframe_selection(selection)
    config: CusfmConfig = selection.config

    # ── 2. feature extraction ────────────────────────────────────────────────────────
    with timed_stage(ledger, "feature_extraction"), watcher.stage_scope("feature_extraction"):
        extraction: ExtractionReport = run_feature_extraction_stage(staged, selection.selected, resolved.features)
    watcher.on_feature_extraction(extraction)

    # ── 3. pair selection ────────────────────────────────────────────────────────────
    with timed_stage(ledger, "pair_selection"), watcher.stage_scope("pair_selection"):
        pairs: list[ImagePair] = run_pair_selection_stage(selection.selected, config)
    watcher.on_pair_selection(pairs)

    # ── 4. matching and verification ─────────────────────────────────────────────────
    # One object for stage 4 and stage 5's batch match: the loop stage matching its
    # candidates with settings of its own is exactly the drift this prevents.
    matching_options: MatchingOptions = resolved.matching_options(config)
    with timed_stage(ledger, "matching"), watcher.stage_scope("matching"):
        matching: MatchReport = run_matching_stage(staged, pairs, matching_options)
    watcher.on_matching(matching)

    # ── 5. loop closure ──────────────────────────────────────────────────────────────
    with timed_stage(ledger, "loop_closure"), watcher.stage_scope("loop_closure"):
        loops: LoopClosureStageResult = run_loop_closure_stage(
            selection.selected, staged.database_path, config.pose_graph, matching_options, enabled=options.loop_closure
        )
    watcher.on_loop_closure(loops)

    # ── 6. pose graph ────────────────────────────────────────────────────────────────
    with timed_stage(ledger, "pose_graph"), watcher.stage_scope("pose_graph"):
        pose_graph: PoseGraphStageResult = run_pose_graph_stage(staged, selection.selected, config, loops.edges)
    watcher.on_pose_graph(pose_graph)

    mapping_options: MappingOptions = resolved.mapping

    # ── 7. triangulation and bundle adjustment ───────────────────────────────────────
    with timed_stage(ledger, "reconstruction"), watcher.stage_scope("reconstruction"):
        reconstruction: MappingResult = run_reconstruction_stage(
            staged, pose_graph.frames_meta, config, mapping_options
        )
    # Before 7b: the refinement adjusts this very reconstruction in place.
    watcher.on_reconstruction(reconstruction)

    # ── 7b. extrinsic refinement ─────────────────────────────────────────────────────
    mapping: MappingResult = reconstruction
    mapped_meta: FramesMeta = pose_graph.frames_meta
    changes: tuple[ExtrinsicChange, ...] = ()
    rounds_run: int = 0
    if options.optimize_extrinsics:
        with timed_stage(ledger, EXTRINSIC_REFINEMENT_STAGE), watcher.stage_scope(EXTRINSIC_REFINEMENT_STAGE):
            refinement: ExtrinsicRefinementStageResult = run_extrinsic_refinement_stage(
                staged, pose_graph.frames_meta, config, mapping_options, mapping
            )
        watcher.on_extrinsic_refinement(refinement)
        mapping = refinement.mapping
        mapped_meta = refinement.frames_meta
        changes = refinement.changes
        rounds_run = refinement.rounds_run

    # ── 8. export ────────────────────────────────────────────────────────────────────
    with timed_stage(ledger, "export"), watcher.stage_scope("export"):
        export: ExportStageResult = run_export_stage(staged, mapped_meta, mapping)
    watcher.on_export(export, mapping)

    summary: PipelineSummary = PipelineSummary(
        options=options,
        git_sha=git_sha(),
        num_input_keyframes=len(selection.frames_meta.keyframes),
        num_selected_keyframes=len(selection.selected.keyframes),
        num_rig_frames=len(selection.rig_frames),
        num_pairs=len(pairs),
        num_loop_pairs=loops.num_pairs_matched,
        num_loop_edges=len(loops.edges),
        num_pose_graph_edges=len(pose_graph.edges),
        pose_graph_translation_change_m=pose_graph.translation_change_m,
        mapping=MappingStats.of(mapping),
        stage_seconds=ledger.seconds_by_stage,
        total_seconds=time.perf_counter() - started,
        extrinsic_changes=changes,
        extrinsic_refinement_rounds_run=rounds_run,
    )
    (staged.output_dir / SUMMARY_NAME).write_text(to_json(summary) + "\n")
    print(
        f"[colsfm] done in {summary.total_seconds:.2f}s | "
        f"{summary.mapping.num_images_with_observations} images, {summary.mapping.num_points3D} points, "
        f"{summary.mapping.mean_reprojection_error_px:.4f} px | extraction on {extraction.device}, "
        f"{matching.empty_pairs} empty pairs"
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


def run_stage(options: SelectionOptions, stage: CheapStageName) -> StageResult:
    """Run one metadata-only stage, for inspecting a dataset without touching a GPU.

    Args:
        options: The metadata-only options; neither stage needs anything else, and
            neither writes, so no output directory is asked for.
        stage: `keyframe_selection` or `pair_selection`.

    Returns:
        The counts the stage produced.

    Raises:
        FileNotFoundError: When the input metadata is missing.
        ValueError: When no keyframe survives selection.
    """
    selection: KeyframeSelectionStageResult = run_keyframe_selection_stage(options)
    pairs: list[ImagePair] = (
        run_pair_selection_stage(selection.selected, selection.config) if stage == "pair_selection" else []
    )
    rig_frames: tuple[RigFrame, ...] = selection.rig_frames
    print(
        f"[colsfm] {stage}: {len(selection.selected.keyframes)}/{len(selection.frames_meta.keyframes)} keyframes, "
        f"{len(rig_frames)} rig frames, {len(pairs)} pairs"
    )
    return StageResult(
        stage=stage,
        num_selected_keyframes=len(selection.selected.keyframes),
        num_rig_frames=len(rig_frames),
        num_pairs=len(pairs),
    )

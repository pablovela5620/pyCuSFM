"""A stage-by-stage Rerun walkthrough of the `colsfm` architecture.

```
pixi run -e colsfm python -m colsfm.walkthrough \
    --input-dir data/r2b_galileo \
    --config-dir data/cusfm_configs/loop-closure-fixed \
    --output data/bench/galileo_walkthrough.rrd
```

The module is an **observer** of `colsfm.pipeline.run_pipeline`, never a second
runner: `run_walkthrough` calls `run_pipeline` with a `WalkthroughObserver`, and
the observer is handed each stage's typed result once that stage's timing window
has closed. One sequencer therefore resolves the options, times the stages into
`runtime.csv` and writes `summary.json`; the walkthrough only draws. A run under
the walkthrough leaves exactly the artifacts `python -m colsfm run` leaves, and a
new stage or backend is added in one place.

Each stage's real intermediate data lands in one recording on a `stage` sequence
timeline. Scrubbing that timeline from 1 to N walks the architecture of Figure 2
of *CuSfM: CUDA-Accelerated Structure-from-Motion* (arXiv:2510.15271):
dictionary construction → loop closure detection → pose graph optimisation on
the first row, feature extraction → feature matching on the second, camera
projection, triangulation and mapping, and extrinsic refinement on the third.

Nothing here re-implements a stage. Everything logged is read back out of the
stage results, the COLMAP database the run wrote, or the run directory the
export stage left behind, so a claim in a panel is a claim about this run.

## Entity layout

```
/walkthrough
  /stage_01                    keyframe selection
    /notes                     TextDocument: what the stage does + this run's numbers
    /input/{frusta,centres}    every input camera pose, grey
    /selected/{frusta,centres} the poses selection kept, blue
    /selected/rig_track        the rig trajectory over the kept rig frames
  /stage_02                    feature extraction
    /samples/<sensor>_<id>/{image,keypoints}   sample keyframes + their ALIKED keypoints
    /keypoint_counts           BarChart, one bar per selected keyframe
  /stage_03                    pair selection
    /centres                   camera centres
    /pairs/{consecutive,stereo}  LineStrips3D between the paired camera centres
  /stage_04                    matching
    /pair/{left,right}/{image,keypoints}   one sample pair, side by side
    /pair/stacked/{image,keypoints,matches}  the same pair concatenated, with match lines
    /inliers_per_pair          BarChart: verified inliers, one bar per pair
    /inlier_histogram          BarChart: the distribution of those counts
  /stage_05                    loop closure
    /robocap/{nodes,trajectory,loop_edges}   RoboCap's revisit graph (see the notes)
  /stage_06                    pose graph
    /before/{nodes,edges}      rig nodes and constraints as they entered the solve
    /after/{nodes,edges}       the same after it
  /stage_07                    triangulation + bundle adjustment
    /points                    the model the mapper left behind
    /rounds/<metric>           Scalars on a `round` timeline, one point per BA round
    /round_chart/<metric>      the same numbers as a BarChart, readable on `stage`
  /stage_08                    extrinsic refinement (only with --optimize-extrinsics)
    /extrinsics/{before,after,shift}   per-camera centres and the shift arrows
  /stage_08 or /stage_09       export
    /points                    the final coloured cloud
    /rigs/rig_NN               input / colsfm / blob reference rigs (exoego layout)
    /tracks/<name>/trajectory  each run's rig trajectory
    /report                    the benchmark table
```

Two deliberate departures are worth knowing before reading the code.

1. **Everything is logged at its stage index, not statically.** A latest-at
   query on `stage` therefore shows stages 1..k at index k and nothing beyond,
   which is what makes the timeline a walkthrough rather than a switchboard.
   The one exception is the rig calibration `simplecv.log_rig_static` writes,
   which is static by construction.
2. **The rigs of the export stage carry two poses.** `log_rig_pose_stream`
   writes `world_T_rig(t)` through `rr.send_columns`, which stamps only the
   `frame_time` timeline — a chunk without a `stage` column is invisible to a
   `stage` query. The export stage therefore also logs each rig's first pose on
   the `stage` timeline, so the rig has a place to stand while the timeline
   walks the architecture, and still animates when `frame_time` is picked.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pycolmap
import rerun as rr
import rerun.blueprint as rrb
import tyro
from jaxtyping import Float, Float64, Int64, UInt8
from numpy import ndarray
from PIL import Image as PillowImage
from simplecv.rerun_rig_logger import log_rig_pose_stream, log_rig_static
from simplecv.rig import Rig, entity_id

from colsfm import REPO_ROOT
from colsfm.alignment import RigTrack, rig_track_from_frames_meta
from colsfm.bench_report import render_markdown_report
from colsfm.benchmark import AcceptanceBounds, Comparison, RunArtifacts, compare_runs, read_run
from colsfm.cameras import colmap_camera
from colsfm.config import CusfmConfig
from colsfm.database import ImagePair, KeypointsXY, read_keypoints, read_two_view_geometry
from colsfm.features import ExtractionReport
from colsfm.frames_meta import FRAMES_META_NAME, FramesMeta, KeyframeMeta, RigFrame, read_frames_meta
from colsfm.mapping import MappingResult, RoundStats
from colsfm.matching import MatchReport, PairMatchStats
from colsfm.pairs import consecutive_pairs, stereo_pairs
from colsfm.pipeline import (
    DEFAULT_CONFIG_DIR,
    ExportStageResult,
    ExtrinsicChange,
    ExtrinsicRefinementStageResult,
    KeyframeSelectionStageResult,
    LoopClosureStageResult,
    PipelineObserver,
    PipelineOptions,
    PipelineSummary,
    PoseGraphStageResult,
    StageName,
    run_pipeline,
    stage_names,
)
from colsfm.pose_graph import PoseGraphEdge, RigNode
from colsfm.rerun_log import IMAGE_PLANE_DISTANCE, POINT_RADIUS, TIMELINE, Rgb, build_rig

# ══════════════════════════════════════════════════════════════════════════════════════
# constants
# ══════════════════════════════════════════════════════════════════════════════════════

WALKTHROUGH_PATH: Final[str] = "walkthrough"
"""Root of the walkthrough's entity tree; every stage hangs one level below it."""

STAGE_TIMELINE: Final[str] = "stage"
"""Sequence timeline carrying one integer per stage, 1-based, in pipeline order."""

ROUND_TIMELINE: Final[str] = "round"
"""Sequence timeline carrying the mapper's bundle-adjustment rounds, 0-based."""

INPUT_COLOR: Final[Rgb] = (135, 135, 145)
"""Grey — the poses that went in, before any stage touched them."""

SELECTED_COLOR: Final[Rgb] = (90, 170, 255)
"""Blue — the colsfm pipeline's own hue, inherited from `colsfm.rerun_log`."""

STEREO_COLOR: Final[Rgb] = (255, 150, 40)
"""Orange — the declared stereo pairs, so they read apart from the consecutive ones."""

LOOP_COLOR: Final[Rgb] = (235, 60, 70)
"""Red — loop constraints, the one edge kind that is not sequential."""

SOLVED_COLOR: Final[Rgb] = (80, 230, 120)
"""Green — the solved side of a before/after pair."""

REFERENCE_COLOR: Final[Rgb] = (255, 150, 40)
"""Orange — the NVIDIA blob reference run, as `colsfm.rerun_log` colours it."""

FRUSTUM_RADIUS_FRACTION: Final[float] = 0.0025
"""Camera-frustum line radius as a fraction of the scene's diagonal."""

EDGE_RADIUS_FRACTION: Final[float] = 0.0035
"""Pose-graph, pair and trajectory line radius as a fraction of the scene's diagonal."""

NODE_RADIUS_FRACTION: Final[float] = 0.008
"""Camera-centre and rig-node marker radius as a fraction of the scene's diagonal.

Every radius here is relative because the walkthrough draws two scenes of very
different size in the same recording: Galileo sweeps 0.66 m, RoboCap walks 124 m,
and a metre-fixed radius makes one an unreadable blob and the other invisible."""

SOLVED_NODE_SHRINK: Final[float] = 0.55
"""How much smaller the solved rig nodes are drawn, so the input ones stay visible
behind them when a solve moved nothing."""

INPUT_FRUSTUM_FRACTION: Final[float] = 0.07
"""Frustum depth of the greyed-out input poses, as a fraction of the scene's diagonal."""

SELECTED_FRUSTUM_FRACTION: Final[float] = 0.11
"""Frustum depth of the poses a stage kept; deliberately longer than the input's, so
the two are told apart even when selection dropped nothing."""

KEYPOINT_RADIUS: Final[float] = 2.5
"""Keypoint marker radius, in pixels."""

MATCH_LINE_RADIUS: Final[float] = 0.6
"""Match-line radius on the concatenated pair image, in pixels."""

STACKED_MAX_WIDTH: Final[int] = 1920
"""Width the concatenated match image is scaled down to, in pixels."""

MAX_MATCH_LINES: Final[int] = 200
"""Match lines drawn on the concatenated image; more is a solid block of colour."""

INLIER_HISTOGRAM_BINS: Final[int] = 24
"""Bars in the per-pair inlier histogram."""

ARC_POINTS: Final[int] = 16
"""Vertices per loop arc; enough that the bulge reads as a curve."""

ARC_LIFT_FRACTION: Final[float] = 0.25
"""Height of a loop arc's bulge as a fraction of the chord length."""

MAX_LOOP_ARCS: Final[int] = 400
"""Strongest loop arcs drawn; the rest would be an unreadable mat of lines."""

MIN_LOOP_RIG_GAP: Final[int] = 2
"""Rig frames a verified pair must span before it counts as a revisit, not a sequence link."""

DEFAULT_EXTRINSIC_ARROW_SCALE: Final[float] = 200.0
"""Extrinsic shifts are sub-millimetre; the arrows are scaled up to be visible."""

DEFAULT_REFERENCE_RUN: Final[Path] = REPO_ROOT / "data" / "cusfm_runs" / "galileo_blobref" / "cusfm"
"""The NVIDIA blob's own Galileo run, used as the closing benchmark's reference."""

DEFAULT_ROBOCAP_LOOP_RUN: Final[Path] = REPO_ROOT / "data" / "cusfm_runs" / "robocap_colsfm_loops" / "cusfm"
"""A RoboCap run whose loop-closure stage actually found revisits (Galileo has none)."""

DEFAULT_WORK_DIR: Final[Path] = REPO_ROOT / "data" / "bench" / "galileo_walkthrough_workspace"
"""Where the walkthrough's own cuSFM-layout workspace goes when none is given."""

DEFAULT_OUTPUT: Final[Path] = REPO_ROOT / "data" / "bench" / "galileo_walkthrough.rrd"
"""Where the recording goes when none is given."""

APPLICATION_ID: Final[str] = "colsfm_walkthrough"
"""Rerun application id stamped into the recording."""

MarkdownRows: TypeAlias = list[tuple[str, str]]
"""Name/value rows rendered into a stage's "this run" table."""


@dataclass(frozen=True, slots=True)
class SamplePane:
    """One sample keyframe's 2D pane: which entity it lives at and what to call it.

    The blueprint cannot be written before the run, because which keyframes get
    sampled depends on the dataset. `log_walkthrough` therefore collects these
    while stage 2 logs and sends the blueprint once, at the end.
    """

    entity_path: str
    """Entity holding the image and its keypoints."""
    name: str
    """Pane title: the sensor and the keyframe id."""


# ══════════════════════════════════════════════════════════════════════════════════════
# the stage vocabulary
# ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class StageCard:
    """What a stage is, independent of any run: the text half of its notes document."""

    title: str
    """Human-readable stage name, also the blueprint tab's label."""
    computes: str
    """One sentence on what the stage turns into what."""
    replaces: str
    """The cuSFM binary (or half of one) this stage stands in for."""
    module: str
    """The `colsfm` modules that implement it."""
    figure: str
    """The box of the paper's Figure 2 this stage occupies."""


STAGE_CARDS: Final[dict[StageName, StageCard]] = {
    "keyframe_selection": StageCard(
        title="1 · Keyframe selection",
        computes=(
            "Reads `frames_meta.json` and the config profile, then drops every keyframe whose rig has not moved far "
            "enough or turned far enough since the last kept one, and regroups what is left into rig frames by "
            "`synced_sample_id`."
        ),
        replaces="`feature_extractor_main` (selection half)",
        module="`colsfm.keyframe_selection`, `colsfm.frames_meta`, `colsfm.config`",
        figure="System inputs — initial poses, image sequences and camera parameters (§2).",
    ),
    "feature_extraction": StageCard(
        title="2 · Feature extraction",
        computes=(
            "Creates the COLMAP database carrying cuSFM's own camera, rig, frame and image ids, then runs ALIKED over "
            "every kept keyframe and stores keypoints and descriptors in it."
        ),
        replaces="`feature_extractor_main` (per-image half)",
        module="`colsfm.database`, `colsfm.cameras`, `colsfm.features`",
        figure="Feature Extraction, second row (§3.1) — ALIKED, the paper's own default.",
    ),
    "pair_selection": StageCard(
        title="3 · Pair selection",
        computes=(
            "Enumerates the image pairs the matcher will be given from the metadata alone: each camera to itself "
            "across the next `connected_keyframe_num` rig frames, plus the two cameras of every declared stereo pair "
            "inside one rig frame. Loop pairs cannot exist yet."
        ),
        replaces="`feature_matcher_task_builder_main`",
        module="`colsfm.pairs`",
        figure="View-Graph Construction via Pose Graph Priors (§3.4) — the non-redundant data association idea.",
    ),
    "matching": StageCard(
        title="4 · Matching",
        computes=(
            "Runs LightGlue over every selected pair, verifies each match set with COLMAP's two-view geometry "
            "estimator, subsamples the inliers over an image-space grid, and writes both into the database."
        ),
        replaces="`feature_matcher_main`",
        module="`colsfm.matching`",
        figure="Feature Matching, second row (§3.1) — LightGlue, the paper's own default.",
    ),
    "loop_closure": StageCard(
        title="5 · Loop closure",
        computes=(
            "Builds a retrieval index over the stored descriptors, shortlists revisit candidates by score and by time "
            "gap, matches the ones the earlier stage had not, measures a metric rig-to-rig pose for each, and gates "
            "the result on the config's translation and rotation thresholds."
        ),
        replaces="`generate_bow_vocabulary_main`, `generate_bow_index_main`, `generate_association_main`",
        module="`colsfm.retrieval`, `colsfm.loop_closure`, `colsfm.loop_pose`",
        figure="Dictionary Construction (§3.2) → Loop Closure Detection (§3.3), first row.",
    ),
    "pose_graph": StageCard(
        title="6 · Pose graph",
        computes=(
            "Turns each rig frame into one node carrying `world_T_rig`, adds a sequential constraint per "
            "`connected_keyframe_num` link and the loop constraints stage 5 produced, and solves the graph on Ceres."
        ),
        replaces="`pose_graph_main`",
        module="`colsfm.pose_graph`, `colsfm.ceres_pose`",
        figure="Pose Graph Optimization (§4.2), first row.",
    ),
    "reconstruction": StageCard(
        title="7 · Triangulation + BA",
        computes=(
            "Builds a `pycolmap.Reconstruction` on the pose-graph poses, triangulates every registered image until a "
            "pass adds nothing, then runs merge → complete → filter → global bundle adjustment for each round of a "
            "decaying reprojection-error schedule."
        ),
        replaces="`keypoints_mapper_main`",
        module="`colsfm.reconstruction`, `colsfm.mapping`",
        figure="Camera Projection (§3.5) feeding Iterative Triangulation and Mapping (§4.3), third row.",
    ),
    "extrinsic_refinement": StageCard(
        title="7b · Extrinsic refinement",
        computes=(
            "Alternates a regularised solve over the rig's `sensor_from_rig` extrinsics — carrying cuSFM's absolute "
            "(Eq. 14) and inter-camera relative (Eq. 6) priors — with pycolmap's own bundle adjustment, holding the "
            "gauge camera fixed."
        ),
        replaces="`keypoints_mapper_main --optimize_extrinsics=True`",
        module="`colsfm.extrinsic_refinement`, `colsfm.mapping`",
        figure="Extrinsic Refinement (§4.4), third row.",
    ),
    "export": StageCard(
        title="8 · Export",
        computes=(
            "Colours the points from the source imagery and writes the four artifacts a cuSFM run leaves behind: the "
            "`sparse/` COLMAP model, `kpmap/keyframes/frames_meta.json`, the TUM trajectories and `runtime.csv`."
        ),
        replaces="`kpmap_to_colmap`, `extract_pose_from_map_main`, `update_keyframe_pose_main`",
        module="`colsfm.export`, `colsfm.benchmark`, `colsfm.bench_report`",
        figure="System outputs — refined trajectory, sparse structure and optimised extrinsics (§2).",
    ),
}
"""One card per `colsfm.pipeline.StageName`, so a stage cannot be logged without its prose."""


@dataclass(frozen=True, slots=True)
class WalkthroughStage:
    """One stage of the walkthrough: its position on the timeline and its prose."""

    index: int
    """1-based position on the `stage` timeline, in pipeline order."""
    name: StageName
    """The `colsfm.pipeline` stage this shows."""
    card: StageCard
    """What the stage is, run-independently."""

    @property
    def entity_root(self) -> str:
        """Where this stage's entities live.

        Returns:
            `/walkthrough/stage_NN`, zero-padded so the paths sort in stage order.
        """
        return f"/{WALKTHROUGH_PATH}/stage_{self.index:02d}"


def walkthrough_stages(optimize_extrinsics: bool) -> tuple[WalkthroughStage, ...]:
    """The stages this walkthrough will record, numbered in order.

    Args:
        optimize_extrinsics: Whether the run makes the second mapping pass, which
            adds stage 7b and pushes export one index later.

    Returns:
        One stage per `colsfm.pipeline.stage_names` entry, 1-based.
    """
    return tuple(
        WalkthroughStage(index=index, name=name, card=STAGE_CARDS[name])
        for index, name in enumerate(stage_names(optimize_extrinsics), start=1)
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# configuration
# ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class WalkthroughConfig:
    """Run one dataset through every stage and log the walkthrough to a `.rrd`."""

    input_dir: Path
    """Directory holding `frames_meta.json` and the images its `image_name`s name."""
    output: Path = DEFAULT_OUTPUT
    """Where the Rerun recording is written; parent directories are created."""
    config_dir: Path = DEFAULT_CONFIG_DIR
    """Directory of `.pb.txt` configs; the loop-closure-fixed isaac profile by default."""
    work_dir: Path | None = None
    """cuSFM-layout workspace the run writes into; a directory under `data/bench` by default."""
    optimize_extrinsics: bool = False
    """Run stage 7b, the regularised rig-extrinsic refinement, and log its shift arrows."""
    loop_closure: bool = True
    """Run the retrieval loop-closure search. On by default here, unlike the pipeline:
    the whole point of stage 5's panel is the rejection funnel, and a disabled stage
    prints one line and produces none."""
    min_inter_frame_distance: float = 0.0
    """Keyframe displacement gate in metres; 0.0 keeps all 226 Galileo keyframes."""
    min_inter_frame_rotation_degrees: float = 5.0
    """Keyframe rotation gate in degrees, OR-combined with the distance gate."""
    use_gpu: bool = True
    """Run ALIKED and LightGlue on the GPU when one is usable."""
    num_sample_keyframes: int = 3
    """Keyframes shown with their keypoints in stage 2, spread over the sequence."""
    extrinsic_arrow_scale: float = DEFAULT_EXTRINSIC_ARROW_SCALE
    """How far stage 7b's shift arrows are exaggerated; the real shifts are sub-millimetre."""
    reference_run: Path | None = DEFAULT_REFERENCE_RUN
    """A blob reference run in the cuSFM layout, for the closing benchmark table."""
    robocap_loop_run: Path | None = DEFAULT_ROBOCAP_LOOP_RUN
    """A run whose loop closure found revisits, drawn beside Galileo's empty stage 5."""
    dataset: str = "galileo"
    """Label for this dataset; it names the recording and the benchmark report, nothing else."""

    @property
    def resolved_work_dir(self) -> Path:
        """The workspace the run will write into.

        Returns:
            `work_dir` when given, `DEFAULT_WORK_DIR` otherwise.
        """
        return DEFAULT_WORK_DIR if self.work_dir is None else self.work_dir

    def pipeline_options(self) -> PipelineOptions:
        """The options the pipeline stages are driven with.

        Returns:
            A `PipelineOptions` carrying this walkthrough's input, workspace and gates.
        """
        return PipelineOptions(
            input_dir=self.input_dir,
            output_dir=self.resolved_work_dir,
            config_dir=self.config_dir,
            min_inter_frame_distance=self.min_inter_frame_distance,
            min_inter_frame_rotation_degrees=self.min_inter_frame_rotation_degrees,
            optimize_extrinsics=self.optimize_extrinsics,
            loop_closure=self.loop_closure,
            use_gpu=self.use_gpu,
        )


@dataclass(frozen=True, slots=True)
class WalkthroughResult:
    """What one walkthrough produced."""

    rrd_path: Path
    """The recording written."""
    work_dir: Path
    """The cuSFM-layout workspace the stages wrote into."""
    stages: tuple[WalkthroughStage, ...]
    """The stages recorded, in timeline order."""
    seconds_by_stage: dict[str, float]
    """Wall-clock seconds per stage, the same rows as the workspace's `runtime.csv`."""


# ══════════════════════════════════════════════════════════════════════════════════════
# geometry and imagery helpers
# ══════════════════════════════════════════════════════════════════════════════════════


def transform_points(pose: pycolmap.Rigid3d, points_xyz: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
    """Apply a rigid transform to a batch of points.

    Args:
        pose: The transform, read as `dst_T_src`.
        points_xyz: Float64 points with shape `[n, 3]`, in the source frame.

    Returns:
        Float64 points with shape `[n, 3]`, in the destination frame.
    """
    matrix: Float64[ndarray, "3 4"] = np.asarray(pose.matrix(), dtype=np.float64)
    return points_xyz @ matrix[:3, :3].T + matrix[:3, 3]


def frustum_strips(world_T_cam: pycolmap.Rigid3d, camera: pycolmap.Camera, depth: float) -> list[Float64[ndarray, "n 3"]]:
    """The two polylines that draw one camera as a pyramid in the world frame.

    The image plane is placed at `depth` metres and its corners are back-projected
    through the camera's own calibration matrix, so a wide camera draws a wide
    frustum.

    Args:
        world_T_cam: The camera's pose in the world frame.
        camera: The COLMAP camera, for its pixel size and calibration matrix.
        depth: Distance to the drawn image plane, in metres.

    Returns:
        Two float64 polylines: the image rectangle, and the rays from the centre
        of projection to its four corners.
    """
    calibration: Float64[ndarray, "3 3"] = np.asarray(camera.calibration_matrix(), dtype=np.float64)
    focal_x: float = float(calibration[0, 0])
    focal_y: float = float(calibration[1, 1])
    principal_x: float = float(calibration[0, 2])
    principal_y: float = float(calibration[1, 2])
    corners_uv: Float64[ndarray, "4 2"] = np.array(
        [[0.0, 0.0], [float(camera.width), 0.0], [float(camera.width), float(camera.height)], [0.0, float(camera.height)]],
        dtype=np.float64,
    )
    corners_cam: Float64[ndarray, "4 3"] = np.stack(
        [
            depth * (corners_uv[:, 0] - principal_x) / focal_x,
            depth * (corners_uv[:, 1] - principal_y) / focal_y,
            np.full(4, depth, dtype=np.float64),
        ],
        axis=-1,
    )
    apex_and_corners: Float64[ndarray, "5 3"] = transform_points(
        world_T_cam, np.vstack([np.zeros((1, 3), dtype=np.float64), corners_cam])
    )
    apex: Float64[ndarray, "3"] = apex_and_corners[0]
    corners: Float64[ndarray, "4 3"] = apex_and_corners[1:]
    rectangle: Float64[ndarray, "5 3"] = np.vstack([corners, corners[:1]])
    rays: Float64[ndarray, "8 3"] = np.stack([np.tile(apex, (4, 1)), corners], axis=1).reshape(-1, 3)
    return [rectangle, rays]


def camera_frusta(
    frames_meta: FramesMeta, keyframes: Sequence[KeyframeMeta], depth: float
) -> list[Float64[ndarray, "n 3"]]:
    """Draw every keyframe of a collection as a frustum.

    Args:
        frames_meta: The collection the keyframes belong to, for its calibration.
        keyframes: The keyframes to draw.
        depth: Frustum depth in metres.

    Returns:
        Two polylines per keyframe, in keyframe order.
    """
    cameras: dict[int, pycolmap.Camera] = {
        camera_params_id: colmap_camera(camera) for camera_params_id, camera in frames_meta.cameras.items()
    }
    strips: list[Float64[ndarray, "n 3"]] = []
    for keyframe in keyframes:
        strips.extend(frustum_strips(keyframe.world_T_cam, cameras[keyframe.camera_params_id], depth))
    return strips


def camera_centres(keyframes: Sequence[KeyframeMeta]) -> Float64[ndarray, "n 3"]:
    """Every keyframe's centre of projection in the world frame.

    Args:
        keyframes: The keyframes to read.

    Returns:
        Float64 positions with shape `[n, 3]`, in metres.
    """
    return np.asarray([keyframe.world_T_cam.translation for keyframe in keyframes], dtype=np.float64).reshape(-1, 3)


def scene_extent(positions: Float64[ndarray, "n 3"]) -> float:
    """The diagonal of a point set's bounding box, as a scale for radii and frusta.

    Args:
        positions: Float64 positions with shape `[n, 3]`, in metres.

    Returns:
        The diagonal in metres, or 1.0 for fewer than two distinct positions, so a
        degenerate scene still draws something.
    """
    if len(positions) == 0:
        return 1.0
    diagonal: float = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    return diagonal if diagonal > 0.0 else 1.0


def polyline_segments(positions: Float64[ndarray, "n 3"]) -> Float64[ndarray, "m 2 3"]:
    """Turn a path into the two-point segments a `LineStrips3D` draws it with.

    Args:
        positions: Float64 path positions with shape `[n, 3]`.

    Returns:
        Float64 segments with shape `[n - 1, 2, 3]`; empty for fewer than two poses.
    """
    if len(positions) < 2:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.stack([positions[:-1], positions[1:]], axis=1)


def arc_strip(start_xyz: Float64[ndarray, "3"], end_xyz: Float64[ndarray, "3"]) -> Float64[ndarray, "n 3"]:
    """A polyline from `start` to `end` bulging upwards, so overlapping links stay readable.

    Args:
        start_xyz: Float64 start position with shape `[3]`.
        end_xyz: Float64 end position with shape `[3]`.

    Returns:
        Float64 polyline with shape `[ARC_POINTS, 3]`.
    """
    fraction: Float64[ndarray, "n"] = np.linspace(0.0, 1.0, ARC_POINTS)
    chord: Float64[ndarray, "n 3"] = start_xyz + (end_xyz - start_xyz) * fraction[:, None]
    lift: float = ARC_LIFT_FRACTION * float(np.linalg.norm(end_xyz - start_xyz))
    chord[:, 2] += lift * np.sin(np.pi * fraction)
    return chord


def load_rgb(path: Path, max_width: int) -> tuple[UInt8[ndarray, "h w 3"], float]:
    """Read one JPEG as RGB, scaled down to fit a width budget.

    Args:
        path: The image file.
        max_width: Largest width to return, in pixels.

    Returns:
        The uint8 RGB image and the scale factor that was applied to it.
    """
    with PillowImage.open(path) as opened:
        image = opened.convert("RGB")
        scale: float = min(1.0, max_width / float(image.width))
        if scale < 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)), PillowImage.BILINEAR)
        return np.asarray(image, dtype=np.uint8), scale


def rerun_keypoints(keypoints_xy: Float[ndarray, "n 2"]) -> Float64[ndarray, "n 2"]:
    """Convert COLMAP keypoints to Rerun's pixel convention.

    COLMAP puts the origin at the top-left *corner* of the top-left pixel and
    reports the pixel *centre* at (0.5, 0.5); Rerun puts the centre of that pixel
    at (0, 0). The half-pixel is the whole difference.

    Args:
        keypoints_xy: Keypoint pixel coordinates with shape `[n, 2]`.

    Returns:
        Float64 coordinates with shape `[n, 2]`, in Rerun's convention.
    """
    return np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2) - 0.5


def image_path_of(input_dir: Path, keyframe: KeyframeMeta) -> Path:
    """Where one keyframe's JPEG lives.

    Args:
        input_dir: The dataset's raw image root.
        keyframe: The keyframe naming it.

    Returns:
        `<input_dir>/<image_name>`.
    """
    return input_dir / keyframe.image_name


def spread_indices(count: int, wanted: int) -> list[int]:
    """Pick evenly spaced indices out of a sequence.

    Args:
        count: Length of the sequence to sample.
        wanted: How many indices to return.

    Returns:
        Ascending, deduplicated indices; empty when `count` is 0.
    """
    if count <= 0 or wanted <= 0:
        return []
    picks: Int64[ndarray, "k"] = np.unique(np.linspace(0, count - 1, min(wanted, count)).round().astype(np.int64))
    return [int(index) for index in picks]


# ══════════════════════════════════════════════════════════════════════════════════════
# the notes document
# ══════════════════════════════════════════════════════════════════════════════════════


def render_notes(stage: WalkthroughStage, rows: MarkdownRows, console: str, extra: str = "") -> str:
    """Render one stage's notes document.

    Args:
        stage: The stage being documented.
        rows: Name/value pairs measured on this run.
        console: Whatever the stage's own functions printed while they ran.
        extra: Further markdown appended after the console block; a stage with a
            table of its own puts it here rather than inside the code fence.

    Returns:
        Markdown holding the prose, the run's numbers, the stage's own log and
        whatever extra section it asked for.
    """
    card: StageCard = stage.card
    lines: list[str] = [
        f"# {card.title}",
        "",
        f"**Computes** — {card.computes}",
        "",
        f"**Replaces the cuSFM binary** — {card.replaces}",
        "",
        f"**Implemented by** — {card.module}",
        "",
        f"**Paper Figure 2** — {card.figure}",
        "",
        "## This run",
        "",
        "| quantity | value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    stripped: str = console.strip()
    if stripped:
        lines.extend(["", "## What the stage printed", "", "```", stripped, "```"])
    if extra:
        lines.extend(["", extra])
    return "\n".join(lines) + "\n"


def log_notes(stage: WalkthroughStage, rows: MarkdownRows, console: str, extra: str = "") -> None:
    """Log one stage's notes document at its own stage index.

    Args:
        stage: The stage being documented.
        rows: Name/value pairs measured on this run.
        console: Whatever the stage printed.
        extra: Further markdown appended after the console block.
    """
    rr.log(
        f"{stage.entity_root}/notes",
        rr.TextDocument(render_notes(stage, rows, console, extra), media_type="text/markdown"),
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# per-stage logging
# ══════════════════════════════════════════════════════════════════════════════════════


def _selection_verdict(num_input: int, num_kept: int) -> str:
    """Say in one line whether the selection gates actually removed anything.

    Args:
        num_input: Keyframes in the input collection.
        num_kept: Keyframes that survived.

    Returns:
        A sentence for the stage's notes table.
    """
    if num_kept == num_input:
        return (
            f"nothing was dropped — all {num_input} keyframes cleared the gates, so the grey and the blue frusta "
            "coincide. Pass `--min-inter-frame-distance 0.5` (the stock `feature_extractor_main` value) to watch the "
            "rule bite: on Galileo it keeps 34"
        )
    return f"dropped {num_input - num_kept} of {num_input} keyframes; the grey frusta with no blue frustum inside them are the ones that went"


def log_keyframe_selection(stage: WalkthroughStage, selection: KeyframeSelectionStageResult, console: str) -> None:
    """Stage 1: every input pose in grey, the kept ones in blue, and the rig track.

    Args:
        stage: This stage's timeline slot and prose.
        selection: What `run_keyframe_selection_stage` returned.
        console: What it printed.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    all_keyframes: tuple[KeyframeMeta, ...] = selection.frames_meta.keyframes
    kept: tuple[KeyframeMeta, ...] = selection.selected.keyframes
    input_centres: Float64[ndarray, "n 3"] = camera_centres(all_keyframes)
    extent: float = scene_extent(input_centres)
    rr.log(
        f"{stage.entity_root}/input/frusta",
        rr.LineStrips3D(
            camera_frusta(selection.frames_meta, all_keyframes, INPUT_FRUSTUM_FRACTION * extent),
            colors=INPUT_COLOR,
            radii=FRUSTUM_RADIUS_FRACTION * extent,
        ),
    )
    rr.log(
        f"{stage.entity_root}/input/centres",
        rr.Points3D(input_centres, colors=INPUT_COLOR, radii=0.5 * NODE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/selected/frusta",
        rr.LineStrips3D(
            camera_frusta(selection.selected, kept, SELECTED_FRUSTUM_FRACTION * extent),
            colors=SELECTED_COLOR,
            radii=FRUSTUM_RADIUS_FRACTION * extent,
        ),
    )
    rr.log(
        f"{stage.entity_root}/selected/centres",
        rr.Points3D(camera_centres(kept), colors=SELECTED_COLOR, radii=SOLVED_NODE_SHRINK * NODE_RADIUS_FRACTION * extent),
    )
    rig_positions: Float64[ndarray, "n 3"] = np.asarray(
        [rig_frame.world_T_vehicle.translation for rig_frame in selection.rig_frames], dtype=np.float64
    ).reshape(-1, 3)
    rr.log(
        f"{stage.entity_root}/selected/rig_track",
        rr.LineStrips3D(polyline_segments(rig_positions), colors=SOLVED_COLOR, radii=EDGE_RADIUS_FRACTION * extent),
    )
    gates = selection.config.keyframe_selection
    log_notes(
        stage,
        [
            ("input keyframes", str(len(all_keyframes))),
            ("kept keyframes", str(len(kept))),
            ("rig frames kept", str(len(selection.rig_frames))),
            ("cameras", str(len(selection.selected.cameras))),
            ("distance gate", f"{gates.min_inter_frame_distance_m:.3f} m"),
            ("rotation gate", f"{gates.min_inter_frame_rotation_degrees:.3f} deg"),
            ("rig sync window", f"{gates.sample_sync_threshold_microseconds} us"),
            ("selection rule", "keep a rig sample when it moved past the distance gate **or** turned past the rotation gate"),
            ("what the rule did here", _selection_verdict(len(all_keyframes), len(kept))),
        ],
        console,
    )


def sample_keyframes(selected: FramesMeta, wanted: int) -> list[KeyframeMeta]:
    """Pick a few keyframes spread over the sequence and over the cameras.

    Args:
        selected: The collection keyframe selection kept.
        wanted: How many samples to return.

    Returns:
        One keyframe per sampled rig frame, cycling the camera so the samples do
        not all show the same view.
    """
    rig_frames: tuple[RigFrame, ...] = selected.rig_frames()
    by_id: dict[int, KeyframeMeta] = selected.keyframe_by_id()
    samples: list[KeyframeMeta] = []
    for offset, position in enumerate(spread_indices(len(rig_frames), wanted)):
        members: tuple[int, ...] = rig_frames[position].keyframe_ids
        samples.append(by_id[members[offset % len(members)]])
    return samples


def log_feature_extraction(
    stage: WalkthroughStage,
    selected: FramesMeta,
    extraction: ExtractionReport,
    database_path: Path,
    input_dir: Path,
    num_samples: int,
    console: str,
) -> list[SamplePane]:
    """Stage 2: sample keyframes with their ALIKED keypoints, plus the count histogram.

    The keypoints are read back out of the database rather than kept from the
    extractor, so what is drawn is what the matcher will see.

    Args:
        stage: This stage's timeline slot and prose.
        selected: The collection keyframe selection kept.
        extraction: The report `run_feature_extraction_stage` returned.
        database_path: The database the stage wrote.
        input_dir: The dataset's raw image root.
        num_samples: How many keyframes to show.
        console: What the stage printed.

    Returns:
        One pane per keyframe that was drawn, so the blueprint can give each its
        own 2D view — stacked in one view, only the last image would be visible.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    panes: list[SamplePane] = []
    for keyframe in sample_keyframes(selected, num_samples):
        sensor_name: str = selected.cameras[keyframe.camera_params_id].sensor_name
        sample_path: str = f"{stage.entity_root}/samples/{sensor_name}_{keyframe.keyframe_id}"
        image_path: Path = image_path_of(input_dir, keyframe)
        if not image_path.is_file():
            continue
        keypoints_xy: KeypointsXY = read_keypoints(database_path, keyframe.keyframe_id)
        rr.log(f"{sample_path}/image", rr.EncodedImage(path=image_path))
        rr.log(
            f"{sample_path}/keypoints",
            rr.Points2D(rerun_keypoints(keypoints_xy), colors=SELECTED_COLOR, radii=KEYPOINT_RADIUS),
        )
        panes.append(SamplePane(entity_path=sample_path, name=f"{sensor_name} #{keyframe.keyframe_id} ({len(keypoints_xy)} kp)"))
    counts: dict[int, int] = extraction.keypoint_counts
    ordered: Int64[ndarray, "n"] = np.asarray([counts[key] for key in sorted(counts)], dtype=np.int64)
    rr.log(f"{stage.entity_root}/keypoint_counts", rr.BarChart(ordered, color=SELECTED_COLOR))
    log_notes(
        stage,
        [
            ("images extracted", str(len(counts))),
            ("keypoints total", f"{int(ordered.sum()):,}"),
            ("keypoints per image (min / median / max)", f"{int(ordered.min())} / {int(np.median(ordered))} / {int(ordered.max())}"),
            ("device", extraction.device),
            ("seconds", f"{extraction.elapsed_seconds:.2f}"),
            ("database", f"`{database_path}`"),
            ("samples drawn", ", ".join(pane.name for pane in panes) if panes else "none; the imagery is not on disk"),
        ],
        console,
    )
    return panes


def log_pair_selection(
    stage: WalkthroughStage,
    selected: FramesMeta,
    config: CusfmConfig,
    pairs: Sequence[ImagePair],
    console: str,
) -> None:
    """Stage 3: the consecutive and stereo pairs as lines between camera centres.

    Args:
        stage: This stage's timeline slot and prose.
        selected: The collection keyframe selection kept.
        config: The config profile, for `connected_keyframe_num`.
        pairs: The pairs `run_pair_selection_stage` returned.
        console: What it printed.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    by_id: dict[int, KeyframeMeta] = selected.keyframe_by_id()
    consecutive: list[ImagePair] = consecutive_pairs(selected, config.pose_graph.connected_keyframe_num)
    stereo: list[ImagePair] = stereo_pairs(selected)

    def segments(pairs: Sequence[ImagePair]) -> Float64[ndarray, "n 2 3"]:
        """Two-point segments joining each pair's camera centres."""
        if not pairs:
            return np.zeros((0, 2, 3), dtype=np.float64)
        return np.asarray(
            [[by_id[first].world_T_cam.translation, by_id[second].world_T_cam.translation] for first, second in pairs],
            dtype=np.float64,
        ).reshape(-1, 2, 3)

    centres: Float64[ndarray, "n 3"] = camera_centres(selected.keyframes)
    extent: float = scene_extent(centres)
    rr.log(f"{stage.entity_root}/centres", rr.Points3D(centres, colors=INPUT_COLOR, radii=0.5 * NODE_RADIUS_FRACTION * extent))
    rr.log(
        f"{stage.entity_root}/pairs/consecutive",
        rr.LineStrips3D(segments(consecutive), colors=SELECTED_COLOR, radii=EDGE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/pairs/stereo",
        rr.LineStrips3D(segments(stereo), colors=STEREO_COLOR, radii=EDGE_RADIUS_FRACTION * extent),
    )
    log_notes(
        stage,
        [
            ("pairs total", str(len(pairs))),
            ("consecutive pairs (blue)", str(len(consecutive))),
            ("stereo pairs (orange)", str(len(stereo))),
            ("connected_keyframe_num", str(config.pose_graph.connected_keyframe_num)),
            ("stereo declarations", str(len(selected.stereo_pairs))),
            ("loop pairs", "0 — they cannot exist before stage 5 has run"),
        ],
        console,
    )


def _pair_for_display(report_pairs: Mapping[ImagePair, PairMatchStats]) -> ImagePair | None:
    """Pick the pair whose inlier count is the median, so the panel shows a typical pair.

    Args:
        report_pairs: The matcher's per-pair statistics.

    Returns:
        The chosen pair, or None when nothing verified.
    """
    verified: list[ImagePair] = sorted(
        (pair for pair, stats in report_pairs.items() if stats.inlier_matches > 0),
        key=lambda pair: report_pairs[pair].inlier_matches,
    )
    return verified[len(verified) // 2] if verified else None


def log_matching(
    stage: WalkthroughStage,
    selected: FramesMeta,
    matching: MatchReport,
    database_path: Path,
    input_dir: Path,
    console: str,
) -> None:
    """Stage 4: one sample pair with its verified matches, plus the inlier distribution.

    Args:
        stage: This stage's timeline slot and prose.
        selected: The collection keyframe selection kept.
        matching: The report `run_matching_stage` returned.
        database_path: The database holding the keypoints and geometries.
        input_dir: The dataset's raw image root.
        console: What the stage printed.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    stats: dict[ImagePair, PairMatchStats] = matching.pair_stats
    inliers: Int64[ndarray, "n"] = np.asarray([stats[pair].inlier_matches for pair in sorted(stats)], dtype=np.int64)
    rr.log(f"{stage.entity_root}/inliers_per_pair", rr.BarChart(inliers, color=SELECTED_COLOR))
    histogram: Int64[ndarray, "bins"] = np.histogram(inliers, bins=INLIER_HISTOGRAM_BINS)[0].astype(np.int64)
    rr.log(f"{stage.entity_root}/inlier_histogram", rr.BarChart(histogram, color=STEREO_COLOR))

    rows: MarkdownRows = [
        ("pairs matched", str(len(stats))),
        ("pairs with no verified geometry", str(matching.empty_pairs)),
        ("median inliers per pair", f"{matching.median_inliers:.1f}"),
        ("median inlier retention", f"{100.0 * matching.median_retention:.1f} %"),
        ("device", matching.device),
        ("seconds", f"{matching.elapsed_seconds:.2f}"),
    ]
    chosen: ImagePair | None = _pair_for_display(stats)
    if chosen is not None:
        rows.extend(_log_match_pair(stage, selected, chosen, database_path, input_dir, stats[chosen]))
    log_notes(stage, rows, console)


def _log_match_pair(
    stage: WalkthroughStage,
    selected: FramesMeta,
    pair: ImagePair,
    database_path: Path,
    input_dir: Path,
    stats: PairMatchStats,
) -> MarkdownRows:
    """Draw one verified pair: each image with its keypoints, then the two concatenated.

    Args:
        stage: This stage's timeline slot.
        selected: The collection, for the keyframes' image names.
        pair: The image pair to draw.
        database_path: The database holding the keypoints and the geometry.
        input_dir: The dataset's raw image root.
        stats: The matcher's counters for this pair.

    Returns:
        Rows describing the pair, for the stage's notes table.
    """
    by_id: dict[int, KeyframeMeta] = selected.keyframe_by_id()
    left: KeyframeMeta = by_id[pair[0]]
    right: KeyframeMeta = by_id[pair[1]]
    left_path: Path = image_path_of(input_dir, left)
    right_path: Path = image_path_of(input_dir, right)
    if not left_path.is_file() or not right_path.is_file():
        return [("sample pair", "imagery missing on disk")]

    geometry: pycolmap.TwoViewGeometry = read_two_view_geometry(database_path, pair[0], pair[1])
    matches: Int64[ndarray, "m 2"] = np.asarray(geometry.inlier_matches, dtype=np.int64).reshape(-1, 2)
    left_keypoints: KeypointsXY = read_keypoints(database_path, pair[0])
    right_keypoints: KeypointsXY = read_keypoints(database_path, pair[1])
    left_matched: Float64[ndarray, "m 2"] = rerun_keypoints(left_keypoints[matches[:, 0]]) if len(matches) else np.zeros((0, 2))
    right_matched: Float64[ndarray, "m 2"] = rerun_keypoints(right_keypoints[matches[:, 1]]) if len(matches) else np.zeros((0, 2))

    for side, keyframe, path, matched in (("left", left, left_path, left_matched), ("right", right, right_path, right_matched)):
        side_path: str = f"{stage.entity_root}/pair/{side}"
        rr.log(f"{side_path}/image", rr.EncodedImage(path=path))
        rr.log(f"{side_path}/keypoints", rr.Points2D(matched, colors=SOLVED_COLOR, radii=KEYPOINT_RADIUS))
        rr.log(f"{side_path}/name", rr.TextDocument(f"{selected.cameras[keyframe.camera_params_id].sensor_name}\n{keyframe.image_name}"))

    left_rgb, scale = load_rgb(left_path, STACKED_MAX_WIDTH // 2)
    right_rgb, _ = load_rgb(right_path, STACKED_MAX_WIDTH // 2)
    height: int = min(left_rgb.shape[0], right_rgb.shape[0])
    stacked: UInt8[ndarray, "h w 3"] = np.hstack([left_rgb[:height], right_rgb[:height]])
    offset_x: float = float(left_rgb.shape[1])
    scaled_left: Float64[ndarray, "m 2"] = scale * left_matched
    scaled_right: Float64[ndarray, "m 2"] = scale * right_matched + np.array([offset_x, 0.0])
    drawn: list[int] = spread_indices(len(matches), MAX_MATCH_LINES)
    lines: Float64[ndarray, "k 2 2"] = np.stack([scaled_left[drawn], scaled_right[drawn]], axis=1) if drawn else np.zeros((0, 2, 2))
    rr.log(f"{stage.entity_root}/pair/stacked/image", rr.Image(stacked))
    rr.log(
        f"{stage.entity_root}/pair/stacked/keypoints",
        rr.Points2D(np.vstack([scaled_left, scaled_right]), colors=SOLVED_COLOR, radii=KEYPOINT_RADIUS),
    )
    rr.log(f"{stage.entity_root}/pair/stacked/matches", rr.LineStrips2D(lines, colors=STEREO_COLOR, radii=MATCH_LINE_RADIUS))
    return [
        ("sample pair", f"`{left.image_name}` ↔ `{right.image_name}`"),
        ("sample pair raw matches", str(stats.raw_matches)),
        ("sample pair verified inliers", str(stats.inlier_matches)),
        ("match lines drawn", f"{len(drawn)} of {len(matches)}"),
    ]


def _rig_id_by_keyframe(frames_meta: FramesMeta) -> dict[int, int]:
    """Index each keyframe's rig-frame position, for turning image pairs into rig pairs.

    Args:
        frames_meta: The collection to index.

    Returns:
        Rig-frame position (0-based, time ordered) keyed by keyframe id.
    """
    positions: dict[int, int] = {}
    for position, rig_frame in enumerate(frames_meta.rig_frames()):
        for keyframe_id in rig_frame.keyframe_ids:
            positions[keyframe_id] = position
    return positions


def _verified_revisit_pairs(database_path: Path, rig_position: Mapping[int, int]) -> dict[tuple[int, int], int]:
    """Every verified pair in a database that spans more than one rig frame.

    A pair whose two images sit in adjacent rig frames is a sequence link, which
    stage 3 already produced; anything further apart in time only reaches the
    database because the loop-closure stage matched it there.

    Args:
        database_path: The run's COLMAP database.
        rig_position: Rig-frame position per keyframe id.

    Returns:
        Summed inlier count keyed by the ordered rig-frame position pair.
    """
    with pycolmap.Database.open(database_path) as database:
        pair_ids, counts = database.read_two_view_geometry_num_inliers()
    revisits: dict[tuple[int, int], int] = {}
    for pair_id, count in zip(pair_ids, counts, strict=True):
        image_id_a, image_id_b = pycolmap.pair_id_to_image_pair(pair_id)
        first: int | None = rig_position.get(int(image_id_a))
        second: int | None = rig_position.get(int(image_id_b))
        if first is None or second is None or abs(first - second) < MIN_LOOP_RIG_GAP:
            continue
        key: tuple[int, int] = (min(first, second), max(first, second))
        revisits[key] = revisits.get(key, 0) + int(count)
    return revisits


def log_loop_closure(stage: WalkthroughStage, loops: LoopClosureStageResult, robocap_run: Path | None, console: str) -> None:
    """Stage 5: the rejection funnel, and a dataset that does have revisits.

    Galileo is a 0.66 m sweep, so every candidate the retrieval index proposes is
    rejected — the funnel in the notes is the whole result. To keep the stage from
    being an empty panel, a RoboCap run that *did* close loops is drawn beside it,
    in its own entity subtree and labelled as the other dataset.

    Args:
        stage: This stage's timeline slot and prose.
        loops: What `run_loop_closure_stage` returned.
        robocap_run: A RoboCap `.../cusfm` directory, or None to skip that half.
        console: What the stage printed — the funnel counters live here.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    rows: MarkdownRows = [
        ("loop edges kept on this dataset", str(len(loops.edges))),
        ("extra pairs this stage had to match", str(loops.num_pairs_matched)),
        (
            "which gate rejected what",
            (
                "read the funnel below. On Galileo every retrieved candidate falls to the **time** gate, not the "
                "metric one: the whole sweep lasts 0.93 s and a revisit has to be at least `min_time_gap_seconds` "
                "old, so no pair can qualify however well it matches"
            ),
        ),
    ]
    rows.extend(_log_robocap_loops(stage, robocap_run))
    log_notes(stage, rows, console)


def _log_robocap_loops(stage: WalkthroughStage, robocap_run: Path | None) -> MarkdownRows:
    """Draw a second dataset's revisit graph under `<stage>/robocap`.

    Args:
        stage: This stage's timeline slot.
        robocap_run: A RoboCap `.../cusfm` directory, or None.

    Returns:
        Rows describing the borrowed dataset, for the notes table.
    """
    if robocap_run is None:
        return [("second dataset", "not requested")]
    pose_graph_meta: Path = robocap_run / "pose_graph" / FRAMES_META_NAME
    database_path: Path = robocap_run / "database.db"
    if not pose_graph_meta.is_file() or not database_path.is_file():
        return [("second dataset", f"`{robocap_run}` is not on disk; nothing drawn")]

    frames_meta: FramesMeta = read_frames_meta(pose_graph_meta)
    rig_frames: tuple[RigFrame, ...] = frames_meta.rig_frames()
    positions: Float64[ndarray, "n 3"] = np.asarray(
        [rig_frame.world_T_vehicle.translation for rig_frame in rig_frames], dtype=np.float64
    ).reshape(-1, 3)
    revisits: dict[tuple[int, int], int] = _verified_revisit_pairs(database_path, _rig_id_by_keyframe(frames_meta))
    strongest: list[tuple[int, int]] = sorted(revisits, key=lambda key: revisits[key], reverse=True)[:MAX_LOOP_ARCS]

    extent: float = scene_extent(positions)
    rr.log(
        f"{stage.entity_root}/robocap/nodes",
        rr.Points3D(positions, colors=INPUT_COLOR, radii=0.5 * NODE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/robocap/trajectory",
        rr.LineStrips3D(polyline_segments(positions), colors=SELECTED_COLOR, radii=EDGE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/robocap/loop_edges",
        rr.LineStrips3D(
            [arc_strip(positions[first], positions[second]) for first, second in strongest],
            colors=LOOP_COLOR,
            radii=0.6 * EDGE_RADIUS_FRACTION * extent,
        ),
    )
    return [
        ("second dataset", f"RoboCap, `{robocap_run}` — **not** the dataset the other stages show"),
        ("RoboCap rig frames", str(len(rig_frames))),
        ("RoboCap verified revisit pairs", f"{len(revisits)} rig-frame pairs at least {MIN_LOOP_RIG_GAP} frames apart"),
        ("RoboCap arcs drawn", f"{len(strongest)}, the strongest by inlier count"),
        (
            "why the arcs are pairs and not edges",
            (
                "the run's `summary.json` counts the gated loop *edges* it kept, but `run_pipeline` never writes the "
                "edge list to disk, so what is drawn is the verified revisit pair graph its database still holds"
            ),
        ),
    ]


def log_pose_graph(stage: WalkthroughStage, pose_graph: PoseGraphStageResult, console: str) -> None:
    """Stage 6: rig nodes and constraints before and after the solve.

    Args:
        stage: This stage's timeline slot and prose.
        pose_graph: What `run_pose_graph_stage` returned.
        console: What it printed — the Ceres termination and costs live here.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    nodes: list[RigNode] = pose_graph.nodes
    before_by_id: dict[int, Float64[ndarray, "3"]] = {
        node.rig_id: np.asarray(node.world_T_rig.translation, dtype=np.float64) for node in nodes
    }
    after_by_id: dict[int, Float64[ndarray, "3"]] = {
        rig_frame.synced_sample_id: np.asarray(rig_frame.world_T_vehicle.translation, dtype=np.float64)
        for rig_frame in pose_graph.frames_meta.rig_frames()
    }

    def edge_segments(by_id: Mapping[int, Float64[ndarray, "3"]], kind: str) -> Float64[ndarray, "n 2 3"]:
        """Two-point segments for every constraint of one kind."""
        drawn: list[PoseGraphEdge] = [
            edge for edge in pose_graph.edges if edge.kind == kind and edge.source in by_id and edge.target in by_id
        ]
        if not drawn:
            return np.zeros((0, 2, 3), dtype=np.float64)
        return np.asarray([[by_id[edge.source], by_id[edge.target]] for edge in drawn], dtype=np.float64).reshape(-1, 2, 3)

    extent: float = scene_extent(np.asarray(list(before_by_id.values()), dtype=np.float64).reshape(-1, 3))
    # The solved nodes are drawn smaller, not just greener: on a graph with no loop
    # constraint the solve moves nothing, and equal radii would hide the input nodes
    # entirely behind them — which reads as "the input was not logged".
    for label, by_id, colour, shrink in (
        ("before", before_by_id, INPUT_COLOR, 1.0),
        ("after", after_by_id, SOLVED_COLOR, SOLVED_NODE_SHRINK),
    ):
        node_positions: Float64[ndarray, "n 3"] = np.asarray([by_id[node.rig_id] for node in nodes if node.rig_id in by_id]).reshape(-1, 3)
        rr.log(
            f"{stage.entity_root}/{label}/nodes",
            rr.Points3D(node_positions, colors=colour, radii=shrink * NODE_RADIUS_FRACTION * extent),
        )
        rr.log(
            f"{stage.entity_root}/{label}/edges",
            rr.LineStrips3D(edge_segments(by_id, "consecutive"), colors=colour, radii=shrink * EDGE_RADIUS_FRACTION * extent),
        )
        rr.log(
            f"{stage.entity_root}/{label}/loop_edges",
            rr.LineStrips3D(edge_segments(by_id, "loop"), colors=LOOP_COLOR, radii=shrink * EDGE_RADIUS_FRACTION * extent),
        )

    num_loop: int = sum(1 for edge in pose_graph.edges if edge.kind == "loop")
    log_notes(
        stage,
        [
            ("rig nodes", str(len(nodes))),
            ("constraints", str(len(pose_graph.edges))),
            ("of which loop constraints", str(num_loop)),
            ("largest rig position change", f"{1000.0 * pose_graph.translation_change_m:.4f} mm"),
            (
                "before vs after",
                (
                    "identical to numerical precision when the graph holds only sequential constraints — with no loop "
                    "closing it, the measurements are exactly the input relative poses and the input trajectory is "
                    "already the optimum"
                ),
            ),
        ],
        console,
    )


def log_reconstruction(stage: WalkthroughStage, mapping: MappingResult, console: str) -> None:
    """Stage 7: the mapped cloud, plus every bundle-adjustment round's statistics.

    Args:
        stage: This stage's timeline slot and prose.
        mapping: The result `run_reconstruction_stage` returned, read before the
            extrinsic refinement may adjust the same reconstruction in place.
        console: What it printed — the per-pass triangulation counts live here.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    positions: Float64[ndarray, "n 3"] = np.asarray(
        [point.xyz for point in mapping.reconstruction.points3D.values()], dtype=np.float64
    ).reshape(-1, 3)
    rr.log(f"{stage.entity_root}/points", rr.Points3D(positions, colors=SELECTED_COLOR, radii=POINT_RADIUS))

    rounds: tuple[RoundStats, ...] = mapping.rounds
    errors: Float64[ndarray, "r"] = np.asarray([stats.mean_reprojection_error_px for stats in rounds], dtype=np.float64)
    observations: Int64[ndarray, "r"] = np.asarray([stats.num_observations for stats in rounds], dtype=np.int64)
    rr.log(f"{stage.entity_root}/round_chart/mean_reprojection_error_px", rr.BarChart(errors, color=STEREO_COLOR))
    rr.log(f"{stage.entity_root}/round_chart/num_observations", rr.BarChart(observations, color=SELECTED_COLOR))

    # The per-round series lives on its own timeline: `stage` holds one index for
    # the whole stage, so five rounds would collapse onto a single point there.
    rr.reset_time()
    for stats in rounds:
        rr.set_time(ROUND_TIMELINE, sequence=stats.round_index)
        rr.log(f"{stage.entity_root}/rounds/mean_reprojection_error_px", rr.Scalars(stats.mean_reprojection_error_px))
        rr.log(f"{stage.entity_root}/rounds/num_observations", rr.Scalars(float(stats.num_observations)))
        rr.log(f"{stage.entity_root}/rounds/max_pixel_error", rr.Scalars(stats.max_pixel_error))
        rr.log(f"{stage.entity_root}/rounds/seconds", rr.Scalars(stats.seconds))
    rr.reset_time()
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)

    table: list[str] = [
        "## Rounds",
        "",
        "| round | gate px | merged | completed | filtered | observations | reprojection px | seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    table.extend(
        f"| {stats.round_index} | {stats.max_pixel_error:.2f} | {stats.num_merged} | {stats.num_completed} | "
        f"{stats.num_filtered} | {stats.num_observations} | {stats.mean_reprojection_error_px:.4f} | {stats.seconds:.2f} |"
        for stats in rounds
    )
    log_notes(
        stage,
        [
            ("registered images", str(mapping.num_registered_images)),
            ("images observing a point", str(mapping.num_images_with_observations)),
            ("points", f"{mapping.num_points3D:,}"),
            ("observations", f"{mapping.num_observations:,}"),
            ("mean reprojection error", f"{mapping.mean_reprojection_error_px:.4f} px"),
            ("mean track length", f"{mapping.mean_track_length:.3f}"),
            ("bundle-adjustment rounds", str(len(rounds))),
            ("triangulation / BA seconds", f"{mapping.triangulation_seconds:.2f} / {mapping.bundle_adjustment_seconds:.2f}"),
            (
                "per-round clouds",
                (
                    "not logged: `MappingResult` keeps one reconstruction and the per-round counters, so the cloud "
                    "can only be read after the last round. The `round` timeline carries the counters instead"
                ),
            ),
        ],
        console,
        "\n".join(table),
    )


def log_extrinsic_refinement(
    stage: WalkthroughStage,
    pose_graph_meta: FramesMeta,
    refinement: ExtrinsicRefinementStageResult,
    arrow_scale: float,
    console: str,
) -> None:
    """Stage 7b: how far the refinement moved each camera on the rig.

    The shifts are sub-millimetre, so the arrows are exaggerated by `arrow_scale`
    and the notes carry the real numbers.

    Args:
        stage: This stage's timeline slot and prose.
        pose_graph_meta: The collection the refinement started from.
        refinement: What `run_extrinsic_refinement_stage` returned.
        arrow_scale: Factor the drawn arrows are multiplied by.
        console: What the stage printed.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    rig_frames: tuple[RigFrame, ...] = pose_graph_meta.rig_frames()
    world_T_rig: pycolmap.Rigid3d = rig_frames[0].world_T_vehicle
    camera_ids: list[int] = sorted(pose_graph_meta.cameras)
    before: Float64[ndarray, "n 3"] = np.asarray(
        [(world_T_rig * pose_graph_meta.cameras[key].vehicle_T_cam).translation for key in camera_ids], dtype=np.float64
    ).reshape(-1, 3)
    after: Float64[ndarray, "n 3"] = np.asarray(
        [(world_T_rig * refinement.frames_meta.cameras[key].vehicle_T_cam).translation for key in camera_ids], dtype=np.float64
    ).reshape(-1, 3)
    labels: list[str] = [pose_graph_meta.cameras[key].sensor_name for key in camera_ids]

    # The rig, not the trajectory, sets the scale here: the cameras sit centimetres
    # apart on one vehicle, and a radius taken from the whole sweep would swallow them.
    extent: float = scene_extent(before)
    # The sensor names ride on the arrows, not on both: labelling the points too
    # prints every camera's name twice, on top of itself.
    rr.log(
        f"{stage.entity_root}/extrinsics/before",
        rr.Points3D(before, colors=INPUT_COLOR, radii=NODE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/extrinsics/after",
        rr.Points3D(after, colors=SOLVED_COLOR, radii=SOLVED_NODE_SHRINK * NODE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/extrinsics/shift",
        rr.Arrows3D(origins=before, vectors=arrow_scale * (after - before), colors=LOOP_COLOR, labels=labels),
    )

    table: list[str] = [
        "## Per-camera movement",
        "",
        "| camera | sensor | translation mm | rotation deg | rig origin |",
        "| --- | --- | --- | --- | --- |",
    ]
    table.extend(
        f"| {change.camera_params_id} | {change.sensor_name} | {change.translation_change_mm:.4f} | "
        f"{change.rotation_change_deg:.5f} | {'yes' if change.is_reference else ''} |"
        for change in refinement.changes
    )
    largest: ExtrinsicChange | None = max(refinement.changes, key=lambda change: change.translation_change_mm, default=None)
    log_notes(
        stage,
        [
            ("alternation rounds run", str(refinement.rounds_run)),
            ("cameras refined", str(len(refinement.changes))),
            ("largest translation change", "n/a" if largest is None else f"{largest.translation_change_mm:.4f} mm ({largest.sensor_name})"),
            ("arrow exaggeration", f"x{arrow_scale:g}"),
            ("points after refinement", f"{refinement.mapping.num_points3D:,}"),
            ("mean reprojection error", f"{refinement.mapping.mean_reprojection_error_px:.4f} px"),
        ],
        console,
        "\n".join(table),
    )


def log_run_as_rig(stage: WalkthroughStage, frames_meta: FramesMeta, index: int, *, label: str, colour: Rgb) -> int:
    """Draw one run as an exoego rig plus its trajectory, under this stage's subtree.

    `simplecv.log_rig_pose_stream` stamps only `frame_time`, so the rig's first
    pose is logged again on `stage`: without it the rig would have nowhere to
    stand while the walkthrough's own timeline is the active one.

    Args:
        stage: This stage's timeline slot.
        frames_meta: The run's collection, carrying its calibration and its poses.
        index: Rig entity index; 0 lands at `<stage>/rigs/rig_00`.
        label: Provenance string logged on the rig node.
        colour: Frustum and trajectory tint identifying the run.

    Returns:
        The number of rig frames drawn.
    """
    track: RigTrack = rig_track_from_frames_meta(frames_meta)
    if len(track) == 0:
        return 0
    rigs_path: str = f"{stage.entity_root.lstrip('/')}/rigs"
    rig: Rig = build_rig(frames_meta, index, track)
    log_rig_static(rig, world_path=rigs_path)
    log_rig_pose_stream(rig, timestamps_ns=1000 * track.timestamps_microseconds, world_path=rigs_path, timeline=TIMELINE)

    rig_path: str = f"/{rigs_path}/{entity_id('rig', index)}"
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    rr.log(rig_path, rr.Transform3D(translation=track.world_t_rig[0], mat3x3=track.world_R_rig[0]))
    rr.log(rig_path, rr.AnyValues(source=label, num_rig_frames=len(track)), static=True)
    for sensor in rig.calibration.cameras:
        rr.log(
            f"{rig_path}/{entity_id('cam', sensor.index)}/pinhole",
            rr.Pinhole.from_fields(image_plane_distance=IMAGE_PLANE_DISTANCE, color=(*colour, 255)),
            static=True,
        )
    rr.log(
        f"{stage.entity_root}/tracks/{entity_id('rig', index)}/trajectory",
        rr.LineStrips3D(
            polyline_segments(track.world_t_rig),
            colors=colour,
            radii=EDGE_RADIUS_FRACTION * scene_extent(track.world_t_rig),
        ),
    )
    return len(track)


def log_export(
    stage: WalkthroughStage,
    options: PipelineOptions,
    export: ExportStageResult,
    mapping: MappingResult,
    input_meta: FramesMeta,
    reference_run: Path | None,
    dataset: str,
    console: str,
) -> None:
    """Stage 8: the coloured cloud, the input and colsfm rigs, and the benchmark table.

    Args:
        stage: This stage's timeline slot and prose.
        options: The run's options, for the workspace and the input directory.
        export: What `run_export_stage` returned.
        mapping: The final mapping result, now coloured.
        input_meta: The dataset's own input collection, drawn as the reference rig.
        reference_run: A blob run in the cuSFM layout, or None to skip the comparison.
        dataset: Dataset label for the benchmark report heading.
        console: What the stage printed.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    positions: Float64[ndarray, "n 3"] = np.asarray(
        [point.xyz for point in mapping.reconstruction.points3D.values()], dtype=np.float64
    ).reshape(-1, 3)
    colours: Int64[ndarray, "n 3"] = np.asarray(
        [point.color for point in mapping.reconstruction.points3D.values()], dtype=np.int64
    ).reshape(-1, 3)
    rr.log(f"{stage.entity_root}/points", rr.Points3D(positions, colors=colours.astype(np.uint8), radii=POINT_RADIUS))

    log_run_as_rig(stage, input_meta, 0, label="input frames_meta.json", colour=INPUT_COLOR)
    log_run_as_rig(stage, export.optimised, 1, label="colsfm (this run)", colour=SELECTED_COLOR)

    rows: MarkdownRows = [
        ("points exported", f"{mapping.num_points3D:,}"),
        ("points that got a colour", f"{export.num_coloured:,}"),
        ("TUM trajectories written", str(len(export.pose_files))),
        ("workspace", f"`{options.output_dir}`"),
    ]
    rows.extend(_log_benchmark(stage, options, reference_run, dataset))
    log_notes(stage, rows, console)


def _log_benchmark(stage: WalkthroughStage, options: PipelineOptions, reference_run: Path | None, dataset: str) -> MarkdownRows:
    """Compare this run against a blob reference and log the report as the closing document.

    Args:
        stage: This stage's timeline slot.
        options: The run's options, for the workspace and the input directory.
        reference_run: A blob run in the cuSFM layout, or None.
        dataset: Dataset label for the report heading.

    Returns:
        Rows describing the comparison, for the notes table.
    """
    if reference_run is None or not (reference_run / "sparse").is_dir():
        rr.log(
            f"{stage.entity_root}/report",
            rr.TextDocument(
                f"# No reference run\n\nNothing to compare against: `{reference_run}` is not a cuSFM run directory.\n",
                media_type="text/markdown",
            ),
        )
        return [("benchmark", f"skipped; `{reference_run}` is not on disk")]

    blob: RunArtifacts = read_run(reference_run, "blob")
    colsfm: RunArtifacts = read_run(options.output_dir, "colsfm")
    comparison: Comparison = compare_runs(blob, colsfm, options.input_dir, dataset, AcceptanceBounds())
    rr.log(f"{stage.entity_root}/report", rr.TextDocument(render_markdown_report(comparison), media_type="text/markdown"))
    log_run_as_rig(stage, blob.frames_meta, 2, label=f"blob reference ({reference_run})", colour=REFERENCE_COLOR)
    return [
        ("benchmark reference", f"`{reference_run}`"),
        ("reference points", f"{comparison.run_a.reconstruction.num_points3D:,}"),
        ("this run's points", f"{comparison.run_b.reconstruction.num_points3D:,}"),
        ("acceptance bounds met", f"{sum(1 for check in comparison.acceptance if check.passed)}/{len(comparison.acceptance)}"),
    ]


# ══════════════════════════════════════════════════════════════════════════════════════
# the blueprint
# ══════════════════════════════════════════════════════════════════════════════════════


def _notes_view(stage: WalkthroughStage) -> rrb.TextDocumentView:
    """The notes pane every tab ends with.

    Args:
        stage: The stage the pane documents.

    Returns:
        A text view over `<stage>/notes`.
    """
    return rrb.TextDocumentView(origin=f"{stage.entity_root}/notes", name="Notes")


def _scene_view(stage: WalkthroughStage, name: str) -> rrb.Spatial3DView:
    """A 3D view over one stage's whole subtree, minus its text panes.

    Args:
        stage: The stage to show.
        name: Pane title.

    Returns:
        The view.
    """
    return rrb.Spatial3DView(
        origin=stage.entity_root,
        name=name,
        contents=["+ $origin/**", "- $origin/notes", "- $origin/report"],
    )


def stage_tab(stage: WalkthroughStage, samples: Sequence[SamplePane]) -> rrb.Container:
    """Lay out one stage's tab: the views that make that stage legible.

    Args:
        stage: The stage to lay out.
        samples: Stage 2's sample panes; ignored by every other stage.

    Returns:
        A container holding the stage's spatial, tabular and textual panes.
    """
    root: str = stage.entity_root
    name: str = stage.card.title
    if stage.name == "feature_extraction":
        # One view per sample: several images under one 2D origin all sit at the
        # same pixel coordinates, so only the last one logged would be visible.
        sample_views: list[rrb.View] = [
            rrb.Spatial2DView(origin=pane.entity_path, name=pane.name) for pane in samples
        ] or [rrb.Spatial2DView(origin=f"{root}/samples", name="Keyframes + ALIKED keypoints")]
        return rrb.Horizontal(
            rrb.Vertical(
                rrb.Horizontal(*sample_views),
                rrb.BarChartView(origin=f"{root}/keypoint_counts", name="Keypoints per image"),
                row_shares=[3.0, 1.0],
            ),
            _notes_view(stage),
            column_shares=[3.0, 2.0],
            name=name,
        )
    if stage.name == "matching":
        return rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{root}/pair/stacked", name="Verified matches"),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"{root}/pair/left", name="Left"),
                    rrb.Spatial2DView(origin=f"{root}/pair/right", name="Right"),
                ),
                rrb.Horizontal(
                    rrb.BarChartView(origin=f"{root}/inliers_per_pair", name="Inliers per pair"),
                    rrb.BarChartView(origin=f"{root}/inlier_histogram", name="Inlier histogram"),
                ),
                row_shares=[2.0, 2.0, 1.5],
            ),
            _notes_view(stage),
            column_shares=[3.0, 2.0],
            name=name,
        )
    if stage.name == "reconstruction":
        return rrb.Horizontal(
            _scene_view(stage, "Mapped cloud"),
            rrb.Vertical(
                rrb.BarChartView(origin=f"{root}/round_chart/mean_reprojection_error_px", name="Reprojection px per round"),
                rrb.BarChartView(origin=f"{root}/round_chart/num_observations", name="Observations per round"),
                rrb.TimeSeriesView(origin=f"{root}/rounds", name="Rounds (switch to the `round` timeline)"),
            ),
            _notes_view(stage),
            column_shares=[3.0, 2.0, 2.0],
            name=name,
        )
    if stage.name == "export":
        return rrb.Horizontal(
            _scene_view(stage, "Cloud, rigs and trajectories"),
            rrb.TextDocumentView(origin=f"{root}/report", name="Benchmark report"),
            _notes_view(stage),
            column_shares=[3.0, 2.0, 2.0],
            name=name,
        )
    return rrb.Horizontal(_scene_view(stage, name), _notes_view(stage), column_shares=[3.0, 2.0], name=name)


def build_blueprint(stages: Sequence[WalkthroughStage], samples: Sequence[SamplePane] = ()) -> rrb.Blueprint:
    """One tab per stage, with the time panel pinned to the `stage` timeline.

    Args:
        stages: The stages this recording holds, in order.
        samples: Stage 2's sample panes, which only exist once that stage has run.

    Returns:
        The blueprint the recording ships with.
    """
    return rrb.Blueprint(
        rrb.Tabs(*[stage_tab(stage, samples) for stage in stages], active_tab=0),
        rrb.TimePanel(state="expanded", timeline=STAGE_TIMELINE),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# the run
# ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class WalkthroughObserver(PipelineObserver):
    """Draws each stage of one `colsfm.pipeline` run into the active recording.

    An observer, not a second runner: `colsfm.pipeline.run_pipeline` sequences the
    stages, resolves the options, times them into `runtime.csv` and writes
    `summary.json`; this object is handed each stage's typed result *after* its
    timing window has closed. Nothing drawn here is timed, and no drawing can
    change what the run computes — which is what keeps the walkthrough's workspace
    byte-comparable with a plain `python -m colsfm run`.

    `stage_scope` is the one hook inside the timing window, and it only redirects
    stdout: the loop-closure funnel and the Ceres termination are *printed* by the
    stage functions rather than returned, and each notes panel quotes them verbatim.
    """

    config: WalkthroughConfig
    """The parsed command line: the sample count, the arrow scale and the reference runs."""
    options: PipelineOptions
    """The options `run_pipeline` is executing, for the database and image paths."""
    stages: tuple[WalkthroughStage, ...]
    """The timeline slots this run records, in stage order."""
    console: dict[str, str] = field(default_factory=dict)
    """Whatever each stage printed while it ran, keyed by stage name."""
    seconds_by_stage: dict[str, float] = field(default_factory=dict)
    """Wall-clock seconds per stage; filled from the run's own summary, not measured here."""
    samples: list[SamplePane] = field(default_factory=list)
    """Stage 2's sample panes. The blueprint needs them and nothing knows them up front."""
    selection: KeyframeSelectionStageResult | None = None
    """Stage 1's result; every later stage draws against the collection it selected."""
    pose_graph_meta: FramesMeta | None = None
    """Stage 6's re-posed collection, which stage 7b's before/after arrows start from."""

    def stage(self, name: StageName) -> WalkthroughStage:
        """Look up a stage's timeline slot by name.

        Args:
            name: The pipeline stage name.

        Returns:
            The walkthrough stage carrying its index and prose.

        Raises:
            KeyError: When this run does not record that stage.
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"{name} is not one of this walkthrough's stages")

    def _selected(self) -> KeyframeSelectionStageResult:
        """Stage 1's result, which every later panel needs.

        Returns:
            What `run_keyframe_selection_stage` produced.

        Raises:
            RuntimeError: When a later stage is observed before stage 1 has run.
        """
        if self.selection is None:
            raise RuntimeError("the walkthrough observed a later stage before keyframe selection")
        return self.selection

    @contextlib.contextmanager
    def _captured(self, stage: StageName) -> Iterator[None]:
        """Run the stage with its stdout diverted into `console[stage]`.

        Args:
            stage: The stage whose output to capture.

        Yields:
            Nothing; the stage body runs inside the redirection.
        """
        buffer: io.StringIO = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                yield
        finally:
            self.console[stage] = buffer.getvalue()

    def stage_scope(self, stage: StageName) -> contextlib.AbstractContextManager[None]:
        """Capture one stage's console output.

        Args:
            stage: The stage about to run.

        Returns:
            A context manager that redirects stdout for the duration of the stage.
        """
        return self._captured(stage)

    def run_started(self, options: PipelineOptions, stages: tuple[StageName, ...]) -> None:
        """Set the recording's world axes before stage 1.

        Args:
            options: The options the run will execute; already held as `options`.
            stages: The stages it will record; already held as `stages`.
        """
        rr.log("/", rr.ViewCoordinates.RFU, static=True)

    def on_keyframe_selection(self, result: KeyframeSelectionStageResult) -> None:
        """Draw stage 1.

        Args:
            result: The configuration read, the input collection and the selection.
        """
        self.selection = result
        log_keyframe_selection(self.stage("keyframe_selection"), result, self.console["keyframe_selection"])

    def on_feature_extraction(self, report: ExtractionReport) -> None:
        """Draw stage 2 and remember the sample panes the blueprint will need.

        Args:
            report: Per-image keypoint counts, the device used and the wall time.
        """
        self.samples = log_feature_extraction(
            self.stage("feature_extraction"),
            self._selected().selected,
            report,
            self.options.database_path,
            self.options.input_dir,
            self.config.num_sample_keyframes,
            self.console["feature_extraction"],
        )

    def on_pair_selection(self, pairs: Sequence[ImagePair]) -> None:
        """Draw stage 3.

        Args:
            pairs: The consecutive and stereo pairs the matcher will be given.
        """
        selection: KeyframeSelectionStageResult = self._selected()
        log_pair_selection(
            self.stage("pair_selection"), selection.selected, selection.config, pairs, self.console["pair_selection"]
        )

    def on_matching(self, report: MatchReport) -> None:
        """Draw stage 4.

        Args:
            report: Per-pair raw and inlier counts, the device used and the wall time.
        """
        log_matching(
            self.stage("matching"),
            self._selected().selected,
            report,
            self.options.database_path,
            self.options.input_dir,
            self.console["matching"],
        )

    def on_loop_closure(self, result: LoopClosureStageResult) -> None:
        """Draw stage 5.

        Args:
            result: The gated loop edges, the extra pairs matched and the diagnostics.
        """
        log_loop_closure(self.stage("loop_closure"), result, self.config.robocap_loop_run, self.console["loop_closure"])

    def on_pose_graph(self, result: PoseGraphStageResult) -> None:
        """Draw stage 6 and remember the collection it re-posed.

        Args:
            result: The nodes, the edges, the re-posed collection and the largest move.
        """
        self.pose_graph_meta = result.frames_meta
        log_pose_graph(self.stage("pose_graph"), result, self.console["pose_graph"])

    def on_reconstruction(self, mapping: MappingResult) -> None:
        """Draw stage 7, before stage 7b may adjust the same reconstruction in place.

        Args:
            mapping: The triangulated, bundle-adjusted model and its round statistics.
        """
        log_reconstruction(self.stage("reconstruction"), mapping, self.console["reconstruction"])

    def on_extrinsic_refinement(self, result: ExtrinsicRefinementStageResult) -> None:
        """Draw stage 7b.

        Args:
            result: The refined model, the re-calibrated collection and the movement.

        Raises:
            RuntimeError: When the refinement is observed before the pose graph.
        """
        if self.pose_graph_meta is None:
            raise RuntimeError("the walkthrough observed the extrinsic refinement before the pose graph")
        log_extrinsic_refinement(
            self.stage("extrinsic_refinement"),
            self.pose_graph_meta,
            result,
            self.config.extrinsic_arrow_scale,
            self.console["extrinsic_refinement"],
        )

    def on_export(self, result: ExportStageResult, mapping: MappingResult) -> None:
        """Draw stage 8.

        Args:
            result: The exported collection and what was written.
            mapping: The final mapping result the export was made from.
        """
        log_export(
            self.stage("export"),
            self.options,
            result,
            mapping,
            self._selected().frames_meta,
            self.config.reference_run,
            self.config.dataset,
            self.console["export"],
        )

    def run_finished(self, summary: PipelineSummary) -> None:
        """Take the run's timings and send the blueprint.

        Sent last, not first: stage 2's panes name the keyframes the sampler picked,
        which nothing knows before the run.

        Args:
            summary: What the run produced and how long each stage took.
        """
        self.seconds_by_stage = dict(summary.stage_seconds)
        rr.send_blueprint(build_blueprint(self.stages, self.samples), make_active=True, make_default=True)


def run_walkthrough(config: WalkthroughConfig) -> WalkthroughResult:
    """Run the dataset through `colsfm.pipeline` and save the walkthrough recording.

    Args:
        config: The parsed command line.

    Returns:
        The recording's path, the workspace, the stages and their timings.

    Raises:
        FileNotFoundError: When the input metadata or a config file is missing.
        ValueError: When a stage cannot proceed.
    """
    options: PipelineOptions = config.pipeline_options()
    config.output.parent.mkdir(parents=True, exist_ok=True)
    stages: tuple[WalkthroughStage, ...] = walkthrough_stages(config.optimize_extrinsics)
    observer: WalkthroughObserver = WalkthroughObserver(config=config, options=options, stages=stages)
    print(f"[walkthrough] input {config.input_dir} | workspace {options.output_dir} | recording {config.output}")

    # A scoped `RecordingStream` rather than `rr.init` + `rr.save`: leaving the
    # `with` block flushes *and* closes the file sink, so the `.rrd` gets its footer.
    # The extrinsics variant gets its own recording id: two `.rrd` files sharing an
    # application id *and* a recording id are one store to the viewer, so opening
    # both merges their entity trees and each shows the other's stages.
    recording_id: str = f"{config.dataset}_walkthrough" + ("_extrinsics" if config.optimize_extrinsics else "")
    recording: rr.RecordingStream = rr.RecordingStream(APPLICATION_ID, recording_id=recording_id)
    recording.save(str(config.output))
    with recording:
        run_pipeline(options, observer)
    recording.flush(timeout_sec=60.0)
    print(f"[walkthrough] wrote {config.output} ({config.output.stat().st_size / 1e6:.1f} MB)")
    return WalkthroughResult(
        rrd_path=config.output,
        work_dir=options.output_dir,
        stages=stages,
        seconds_by_stage=dict(observer.seconds_by_stage),
    )


def main() -> None:
    """Entry point for `python -m colsfm.walkthrough`."""
    run_walkthrough(tyro.cli(WalkthroughConfig))


if __name__ == "__main__":
    main()

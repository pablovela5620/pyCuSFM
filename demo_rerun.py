"""Run cuSFM on a posed multi-camera sequence and visualise it in Rerun.

Three datasets, one code path:

- ``galileo``  — the bundled ``data/r2b_galileo`` sample: 8 PINHOLE cameras on a
  ground robot, 226 keyframes, and a ``ground_truth.txt`` trajectory.
- ``robocap``  — a frozen RoboCap catalog segment: 6 Kannala-Brandt fisheye
  cameras on a head-worn rig, with a per-frame basalt VIO trajectory already
  in the recording. Only the 4 world-facing cameras are reconstructed; the two
  inward-facing eye-tracking cameras are shown as greyed frusta.
- ``kitti06`` — KITTI odometry sequence 06: a stereo pair on a car, 1101 frames
  over a 1233 m loop, with the benchmark's own ground truth. This is the paper's
  Table 4 experiment.

galileo and robocap already carry an initial trajectory, so cuVSLAM is skipped
(``--skip_cuvslam``) and cuSFM's global bundle adjustment *refines* the given
poses. The point of the demo is to make that refinement visible. KITTI ships no
poses at all, so there cuVSLAM runs first and its trajectory becomes the input.

Output follows the ``exoego:v2`` rig schema (see simplecv's
``packages/simplecv/docs/exoego_schema.md``), with **two rigs** under one world::

    /                                 ViewCoordinates.RFU (gravity-aligned Z-up)
      /world/rig_00                   AnyValues + world_T_rig(t)  <- input trajectory
        /cam_NN                       Transform3D (rig_T_cam, from_parent)
          /pinhole                    PinholeWithDistortion
            /image                    EncodedImage (JPEG)
      /world/rig_01                   AnyValues + world_T_rig(t)  <- cuSFM refined
        /cam_NN/pinhole               same calibration, refined motion
          /video                      VideoFrameReference -> rig_00's VideoStream (no copy)
      /world/points                   Points3D (cuSFM sparse cloud)

``rig_01`` is rigidly aligned to ``rig_00`` with a scale-free Umeyama fit, and the
residual RMSE is printed. Read that number as a **disagreement between two
estimates**, not as cuSFM's error: RoboCap's supplied trajectory is pure
multi-camera VIO (basalt) with no loop closure and no mapping, so it drifts too.
Only galileo ships an actual ground-truth trajectory, and only there is the word
"ATE" used. Run with ``--run.no-skip-cuvslam`` to get an independent third
trajectory from cuSFM's bundled cuVSLAM.

Note on fisheye frusta: Rerun draws a frustum from ``K`` alone, so a 150-degree+
fisheye renders as a much narrower cone than its true field of view. The
distortion coefficients are logged alongside (``PinholeWithDistortion``) and the
geometry is correct; only the drawn cone under-represents the FOV. This matches
exactly what the RoboCap catalog's own ``base`` layer displays.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, cast

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
import tyro
from jaxtyping import Bool, Float, Float64, Int
from numpy import ndarray
from scipy.spatial.transform import Rotation, Slerp
from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)
from simplecv.rerun_log_utils import RerunTyroConfig, mux_h264_to_mp4
from simplecv.rerun_rig_logger import log_rig_pose_stream, log_rig_static
from simplecv.rig import CameraSensor, Rig, RigCalibration, RigPoseStream, SensorKind, entity_id
from simplecv.rrd_query_utils import first_valid_value, unwrap_singleton_lists

import pycusfm.constants as _cusfm_constants

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TIMELINE: str = "video_time"
"""Duration timeline, shared with the RoboCap catalog's ``base``/``slam`` layers."""

INPUT_RIG_INDEX: int = 0
"""``rig_00`` carries the *input* trajectory (basalt VIO / galileo ego-motion)."""

CUSFM_RIG_INDEX: int = 1
"""``rig_01`` carries the cuSFM-refined trajectory."""

EXCLUDED_CAM_COLOR: tuple[int, int, int, int] = (110, 110, 110, 160)
"""Grey frustum tint for cameras present on the rig but excluded from the SfM."""

CUSFM_RIG_COLOR: tuple[int, int, int, int] = (255, 150, 40, 255)
"""Orange frustum tint distinguishing the cuSFM rig from the input rig."""

ROBOCAP_SFM_CAMERAS: tuple[str, ...] = ("left_front", "right_front", "left", "right")
"""World-facing RoboCap cameras. ``left_eye``/``right_eye`` point at the wearer's
eyes and carry no scene content, so they are excluded from the reconstruction."""

ROBOCAP_RIG_ADJACENCY_PAIRS: tuple[tuple[str, str], ...] = (
    ("left", "left_front"),
    ("left_front", "right_front"),
    ("right_front", "right"),
)
"""Rig *adjacency*, not rectifiable stereo. cuSFM expresses a cuVSLAM rig through
``stereo_pair`` entries, so covering all four world-facing cameras requires
declaring the ~90-degree neighbours too. Only use this when cuVSLAM must see the
whole rig: with the single true pair below it builds a 2-camera rig and loses
tracking on this fisheye sequence (`Tracking lost at frame 231` of 912)."""

ROBOCAP_STEREO_PAIR: tuple[str, str] = ("left_front", "right_front")
"""The only genuinely rectifiable RoboCap pair: 86.2 mm baseline, optical axes
2.6 degrees apart, baseline 1.3 degrees off the camera X axis. The other
combinations sit ~90 degrees apart (adjacent) or ~168 degrees (back-to-back)
and are *not* stereo pairs, despite having plausible-looking baselines."""

CHILD_FROM_PARENT: int = 2
"""``rr.components.TransformRelation.ChildFromParent``. exoego stores each
camera's static extrinsic this way, so the stored value is ``cam_T_rig``."""

# cuSFM's own workspace layout, re-exported from upstream rather than hardcoded.
# `pycusfm/constants.py` asks for exactly this ("Always import and use these
# constants instead of hardcoding directory or file names"), and it matters here:
# a renamed directory would otherwise degrade silently -- a missing `pose_graph/`
# reads as "no loop closures", a missing `output_poses/` as "no refined
# extrinsics". Reading frozen upstream is allowed; only editing it is not.
CUSFM_SPARSE_DIR: str = _cusfm_constants.kOPEN_MAP_DIR
CUSFM_POSE_GRAPH_DIR: str = _cusfm_constants.kPOSE_GRAPH_DIR
CUSFM_OUTPUT_POSES_DIR: str = _cusfm_constants.kOUTPUT_POSES_DIR
CUSFM_MERGED_POSE_FILE: str = _cusfm_constants.kMERGED_POSE_FILE
CUSFM_FRAME_META_FILE: str = _cusfm_constants.kFRAME_META_FILE
CUSFM_CUVSLAM_DIR: str = _cusfm_constants.kCUVSLAM_OUTPUT_DIR
CUSFM_SLAM_POSES_FILE: str = _cusfm_constants.kSLAM_POSES_FILE
CUSFM_ODOM_POSES_FILE: str = _cusfm_constants.kODOM_POSES_FILE

KITTI_CONFIG_DIR: Path = Path(__file__).resolve().parent / "data" / "kitti" / "config"
"""Upstream's KITTI configs, shipped in this repo and used unmodified.

They are not ``pycusfm/configs/isaac``: KITTI's are tuned for a forward-driving
stereo car (different keypoint counts, matching gates and pose-graph weights), and
running sequence 06 through the isaac profile is not the paper's experiment."""

LOOP_CLOSURE_CONFIG_DIR: Path = (
    Path(__file__).resolve().parent / "data" / "cusfm_configs" / "loop-closure-fixed"
)
"""Copy of ``pycusfm/configs/isaac`` with loop closure repaired. Upstream stays
frozen; this directory differs in five lines across three files.

Two independent defects made the shipped configs return zero loops on both
datasets we tried, and both had to be fixed — in single-track mode
``pose_graph_main`` runs its own loop search and never reads the association
stage's output.

1. ``max_mean_point_to_epipolarline_error: 10e-6`` (association *and* localizer)
   rejected every candidate that reached geometric verification. Instrumenting
   the binary showed 22091 BoW candidates narrowing to 180 stereo attempts, of
   which 168 failed on this threshold (observed residuals 2.8e-5 to 6.0e-4) and
   12 on ``scale_upper_limit``. Relaxing it to ``1e-4`` yields 7 associations;
   ``1e-3`` yields 51, at a higher false-loop risk.
2. ``loop_edge_translation_threshold_meters`` and
   ``loop_edge_rotation_threshold_degrees`` are absent from every shipped
   config, so proto3 defaults them to **0**. ``FindBestLoopCandidateForFrame``
   compares with a plain ``>`` and no zero special-case, so a 0 m / 0 degree
   limit rejects every nonzero relative pose. Setting 1 m / 10 degrees takes
   pose-graph optimisation from 0 to 11 loop constraints; combined with the
   localizer change above, 87.

``loop_edge_prune_threshold: 0`` is also written out, but only to document
intent: it is a *lower* cutoff, so zero is already permissive.

Caveat: these thresholds were validated only in that they produce loops. The
resulting edges were never checked geometrically and no ATE was measured, so
treat ``1e-4`` as the narrowest change that works on this sequence, not as a
calibration-independent default."""

ROBOCAP_SEGMENT: str = "robocap__f408193e6447b3b0__s00000021"
"""Segment frozen from the (in-memory) catalog into a local ``.rrd``."""

ROBOCAP_CATALOG_URL: str = "rerun+http://127.0.0.1:51235"
"""Catalog to pull the segment from when the local ``.rrd`` is absent."""


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RobocapConfig:
    """Reconstruct a frozen RoboCap catalog segment (fisheye, basalt VIO poses)."""

    rrd_path: Path = Path("data/robocap/s00000021.rrd")
    """Frozen catalog segment (``base`` + ``slam`` layers)."""
    trim_start_seconds: float = 2.0
    """Drop this much from the start. The capture opens on a blown-out white frame
    while exposure settles, which is textureless and pollutes both the vocabulary
    and the first keyframes."""
    trim_end_seconds: float = 2.0
    """Drop this much from the end, where the wearer's hand enters frame to stop
    the capture — a large moving occluder right where the loop should close."""
    frame_stride: int = 4
    """Use every Nth video frame. 4648 frames / 4 = 1132 samples per camera.

    Density dominates reconstruction quality on this sequence, so this is 4 rather
    than a faster coarser value. Measured, 20 against 4: vertical excursion on a
    flat floor 6.02 m against 0.73 m, BA point rejection 71 % against 0.5 %,
    ``left``/``right`` registration 59 %/48 % against 99.9 %/98.2 %, disagreement
    with the input trajectory 1214 mm against 499 mm. Stride 4 was also *faster*
    through bundle adjustment (130 s against 235 s), because the extra
    observations converge better -- the density is not simply bought with time.

    The loop-closure thresholds in ``LOOP_CLOSURE_CONFIG_DIR`` were validated at
    this stride, so raising it also runs those gates outside their tested regime.
    Expect ~20 min end to end; ``--dataset.frame-stride 20`` cuts that to roughly
    6 min at the cost of the reconstruction above."""
    max_samples: int | None = None
    """Cap on samples after striding (``None`` = all)."""
    cameras: tuple[str, ...] = ROBOCAP_SFM_CAMERAS
    """World-facing cameras fed to cuSFM."""
    rig_adjacency_pairs: bool = False
    """Declare all three adjacent camera pairs instead of only the true stereo
    pair, so cuSFM hands cuVSLAM a full four-camera rig. Needed for
    ``--run.no-skip-cuvslam``; leave off for bundle-adjustment-only runs, where a
    non-rectifiable pair is a false claim about the geometry."""
    image_plane_distance: float = 0.15
    """Frustum length in metres. The RoboCap catalog uses 0.025 to draw the
    headset at life size; over a 10 m reconstruction that is a speck, so the
    frusta are drawn larger here. Visual only — no effect on geometry."""


@dataclass
class GalileoConfig:
    """Reconstruct the bundled r2b_galileo sample (8 pinhole cameras, has GT)."""

    input_dir: Path = Path("data/r2b_galileo")
    """Upstream's sample directory, complete with ``frames_meta.json``."""
    image_stride: int = 1
    """Log JPEG imagery every Nth sample; 1 = exactly what cuSFM was given."""
    image_plane_distance: float = 0.25
    """Frustum length in metres (a ground robot, so larger than the headset)."""


@dataclass
class KittiConfig:
    """Reproduce the paper's KITTI odometry 06 experiment (stereo car, no input poses)."""

    input_dir: Path = Path("data/kitti/06")
    """Sequence directory: ``image_0/``, ``image_1/``, ``calib.txt``, ``times.txt``,
    and the ``frames_meta.json`` that ``data/kitti/get_framemeta_file_for_KITTI.py``
    produces. ``pixi run kitti06`` creates all of it; nothing here is committed."""
    result_dir: Path = Path("data/kitti/06_result_slam")
    """cuSFM workspace. Named for what it holds — a run initialised from cuVSLAM's
    *SLAM* poses, which is what the paper's baseline needs."""
    ground_truth: Path = Path("data/kitti/06/poses_gt_06.txt")
    """KITTI odometry ground truth (the benchmark's ``poses/06.txt``): 1101 rows of a
    row-major 3x4 ``world_T_cam0``. The upstream converter puts the vehicle frame on
    camera 0, so these poses need no frame change to be comparable."""
    ground_truth_tum: Path = Path("data/kitti/06/poses_gt_06.tum")
    """Where to write the ground truth in TUM form for ``evo_ape``.

    Timestamps go through the same ``int(seconds * 1e6)`` truncation the upstream
    converter applies to ``times.txt``, so they match cuSFM's TUM output exactly and
    evo associates every frame instead of silently dropping some to its 0.01 s
    matching window."""
    image_stride: int = 10
    """Log a stereo pair every Nth sample. KITTI's PNGs are ~270 KB each, so stride 1
    would put ~590 MB of imagery in the recording; 10 keeps it near 60 MB and still
    shows the drive. Visualisation only — cuSFM is given every frame either way."""
    image_plane_distance: float = 2.0
    """Frustum length in metres: a car over a 1233 m loop, so far larger than the
    ground robot and the headset."""


DatasetConfig: TypeAlias = (
    Annotated[
        RobocapConfig,
        tyro.conf.subcommand("robocap", description="RoboCap fisheye segment with basalt VIO poses"),
    ]
    | Annotated[
        GalileoConfig,
        tyro.conf.subcommand("galileo", description="Bundled r2b_galileo pinhole sample (has ground truth)"),
    ]
    | Annotated[
        KittiConfig,
        tyro.conf.subcommand("kitti06", description="KITTI odometry 06; cuVSLAM supplies the initial poses"),
    ]
)


@dataclass
class RunConfig:
    """cuSFM invocation knobs."""

    work_dir: Path = Path("data/cusfm_runs")
    """Root for extracted frames, the cuSFM workspace, and the COLMAP model."""
    feature_type: str = "aliked"
    """Feature extractor: ``aliked``, ``superpoint``, or ``sift_cv_cuda``."""
    ba_frame_type: Literal["vehicle_rig", "camera"] = "vehicle_rig"
    """``vehicle_rig`` keeps the rig rigid, which the exoego schema requires:
    a single ``world_T_rig(t)`` only exists if ``rig_T_cam`` stays fixed."""
    optimize_extrinsics: bool = False
    """Refining ``rig_T_cam`` would break the static-extrinsic contract above."""
    min_inter_frame_distance: float = 0.0
    """Keyframe spacing in metres; 0.0 keeps every supplied sample."""
    point_clip_percentile: float = 99.0
    """Drop sparse points outside this per-axis percentile band before logging.
    SfM clouds always carry a few far-flung outliers; left in, they inflate the
    3D view's bounds so far that the rig renders as a speck. Set to 100.0 to
    disable. Purely a visualisation filter — it never affects the reconstruction
    or any reported metric."""
    config_dir: Path | None = LOOP_CLOSURE_CONFIG_DIR
    """cuSFM's config directory. Defaults to our patched copy of
    `pycusfm/configs/isaac`, which repairs loop closure — see
    `LOOP_CLOSURE_CONFIG_DIR`. Point it at `pycusfm/configs/isaac` to reproduce
    the shipped behaviour, or at another copy to test settings without ever
    editing the frozen upstream files."""
    model_dir: Path | None = None
    """Override cuSFM's model root without editing the frozen `pycusfm/` tree."""
    skip_feature_extractor: bool = False
    """Reuse an existing `keyframes/` directory. Feature extraction is 260 s of the
    run and is unaffected by retrieval settings, so iterating on the vocabulary is
    ~4x faster with this on."""
    num_threads: int = 1
    """Left at cuSFM's default. Raising it to 32 was measured to change nothing
    (6.0 s either way on a 96-image sample): ALIKED runs on the GPU through
    TensorRT, so `--num_thread` only touches CPU-side loading."""
    feature_extractor_batch_size: int = 0
    """Must stay 0 for ALIKED. `--batch_size > 0` routes to a batched detector path
    that aborts with `Unsupported detector type: ALIKED_DETECTOR`."""
    feature_matching_batch_size: int = 0
    """Left at cuSFM's default; the shipped `lightglue_aliked_b16.onnx` hints batch
    16 is possible for the matcher, but that is untested here."""
    skip_data_association: bool = False
    """Run ``generate_association_main`` (the paper's §3.4 view-graph construction).

    cuSFM's own default is to SKIP it, which is easy to miss: the pose graph then
    falls back to `image_retrieval_config.pb.txt`, whose `query_result_number: 1`
    returns a single BoW candidate per query — almost always the temporally
    adjacent frame, never a loop. `association_config.pb.txt`, used only by this
    stage, asks for 20 candidates with a 10 s minimum loop interval. With it off,
    every run here found **0 loop constraints** despite the wearer repeatedly
    revisiting the same room."""
    skip_cuvslam: bool = True
    """Skip cuSFM's bundled cuVSLAM. Default True because both datasets already
    ship a trajectory. Set False to have cuVSLAM estimate its own from the stereo
    pair — it writes ``cuvslam_output/{odom,slam}_poses.tum``, giving an
    *independent* third trajectory. That matters because the supplied basalt
    trajectory is pure multi-camera VIO with no loop closure or mapping, so it
    drifts too and is not ground truth."""
    use_cuvslam_slam_pose: bool = False
    """Initialise cuSFM from cuVSLAM's loop-closed *SLAM* trajectory instead of its
    raw odometry. Only read when cuVSLAM actually runs, so it pairs with
    ``--run.no-skip-cuvslam``.

    Not cosmetic on KITTI 06: odometry alone scores 2.37 m Sim(3) ATE against the
    benchmark ground truth and SLAM 1.34 m, and cuSFM refines whatever it is handed
    rather than recovering from a bad initialisation."""
    downsampling_matches: bool = True
    """cuSFM's ``isaac`` profile downsamples which image pairs get matched
    (``max_keyframes_per_collection: 6`` per 10 s window). That suits a vehicle
    driving a line; a wearer revisiting the same room is starved of matches and
    the final bundle adjustment becomes under-constrained. Set False to match
    densely."""
    skip_pose_graph: bool = False
    """Skip appearance-based pose-graph loop closure. Worth setting when the
    supplied trajectory is already reliable and the scene has repetitive texture:
    an undertrained BoW vocabulary (cuSFM warns when its occupied ratio is below
    0.7) retrieves false loop closures that fold the trajectory."""
    variant: str = ""
    """Optional suffix on the run directory, so experiments do not overwrite
    each other (``data/cusfm_runs/<dataset><variant>/``)."""
    skip_reconstruction: bool = False
    """Re-log an existing COLMAP model without re-running cuSFM."""


@dataclass
class Config:
    """Top-level CLI."""

    dataset: DatasetConfig = field(default_factory=RobocapConfig)
    """Which sequence to reconstruct."""
    run: RunConfig = field(default_factory=RunConfig)
    """cuSFM settings."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun sink: spawn / save / connect / headless."""


# ─────────────────────────────────────────────────────────────────────────────
# SE3 helpers
# ─────────────────────────────────────────────────────────────────────────────


def compose(rotation: Float[ndarray, "3 3"], translation: Float[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """Build a 4x4 homogeneous transform from a rotation and a translation."""
    transform: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def to_axis_angle_dict(transform: Float[ndarray, "4 4"]) -> dict[str, dict[str, float]]:
    """Serialise a transform into cuSFM's ``axis_angle`` + ``translation`` JSON form.

    cuSFM stores the rotation as a unit axis plus an angle in *degrees*. A
    zero-rotation transform has no well-defined axis, so emit ``+Z`` with a zero
    angle rather than a NaN axis.
    """
    rotation_vector: Float64[ndarray, "3"] = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    angle_rad: float = float(np.linalg.norm(rotation_vector))
    axis: Float64[ndarray, "3"] = (
        rotation_vector / angle_rad if angle_rad > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    return {
        "axis_angle": {
            "x": float(axis[0]),
            "y": float(axis[1]),
            "z": float(axis[2]),
            "angle_degrees": float(np.degrees(angle_rad)),
        },
        "translation": {
            "x": float(transform[0, 3]),
            "y": float(transform[1, 3]),
            "z": float(transform[2, 3]),
        },
    }


def umeyama_rigid(
    source: Float[ndarray, "n 3"], target: Float[ndarray, "n 3"]
) -> tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3"], float, float]:
    """Least-squares rigid (SE3, **no scale**) fit mapping ``source`` onto ``target``.

    Kept local to preserve a float64 transform for millimetre-scale diagnostics;
    simplecv's canonical ``umeyama_transform`` returns float32.

    Scale is deliberately held at 1: if cuSFM rescales the trajectory we want to
    see that as alignment error, not absorb it into the fit. The scale that
    *would* have been fitted is returned as a diagnostic.

    Returns ``(rotation, translation, rmse, would_be_scale)``.
    """
    source_mean: Float64[ndarray, "3"] = source.mean(axis=0)
    target_mean: Float64[ndarray, "3"] = target.mean(axis=0)
    source_centred: Float64[ndarray, "n 3"] = source - source_mean
    target_centred: Float64[ndarray, "n 3"] = target - target_mean

    covariance: Float64[ndarray, "3 3"] = target_centred.T @ source_centred / len(source)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    sign_fix: Float64[ndarray, "3 3"] = np.eye(3)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0:
        sign_fix[2, 2] = -1.0
    rotation: Float64[ndarray, "3 3"] = u_matrix @ sign_fix @ vt_matrix

    source_variance: float = float((source_centred**2).sum() / len(source))
    would_be_scale: float = (
        float((singular_values * np.diag(sign_fix)).sum() / source_variance) if source_variance > 1e-15 else 1.0
    )

    translation: Float64[ndarray, "3"] = target_mean - rotation @ source_mean
    residual: Float64[ndarray, "n 3"] = (source @ rotation.T + translation) - target
    rmse: float = float(np.sqrt((residual**2).sum(axis=1).mean()))
    return rotation, translation, rmse, would_be_scale


def trajectory_length(positions: Float[ndarray, "n 3"]) -> float:
    """Total distance travelled along a polyline, in metres."""
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0


def sim3_ate(
    estimate: Float[ndarray, "n 3"], reference: Float[ndarray, "n 3"]
) -> tuple[float, float]:
    """Sim(3)-aligned absolute trajectory error (RMSE, metres) and the fitted scale.

    The rigid fit above deliberately holds scale at 1; this one does not, because
    the KITTI odometry benchmark and the cuSFM paper both report Sim(3)-aligned ATE.
    Mixing the two conventions is how a 1.3 m result is misread as a 3 m one, so the
    scale is returned and printed rather than absorbed silently. It is reused rather
    than reimplemented: ``umeyama_rigid`` already computes the optimal rotation and
    the scale it *would* have fitted, and only the translation has to be recomputed
    with the scale applied.
    """
    rotation, _, _, scale = umeyama_rigid(estimate, reference)
    translation: Float64[ndarray, "3"] = reference.mean(axis=0) - scale * rotation @ estimate.mean(axis=0)
    residual: Float64[ndarray, "n 3"] = (scale * (estimate @ rotation.T) + translation) - reference
    return float(np.sqrt((residual**2).sum(axis=1).mean())), scale


def match_timestamps(
    left_ns: Int[ndarray, "n"], right_ns: Int[ndarray, "m"]
) -> tuple[Int[ndarray, "k"], Int[ndarray, "k"]]:
    """Index pairs where two timelines name the same microsecond.

    Exact, not nearest-neighbour, and matched at microsecond rather than nanosecond
    granularity: every timeline joined here descends from the same
    ``int(seconds * 1e6)`` truncation that ``get_framemeta_file_for_KITTI.py``
    applies, and cuSFM prints those microseconds back out as decimal seconds. A
    tolerance would only hide it when that contract breaks.
    """
    lookup: dict[int, int] = {int(value) // 1000: index for index, value in enumerate(right_ns)}
    pairs: list[tuple[int, int]] = [
        (index, lookup[int(value) // 1000]) for index, value in enumerate(left_ns) if int(value) // 1000 in lookup
    ]
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    left_index, right_index = zip(*pairs, strict=True)
    return np.asarray(left_index, dtype=np.int64), np.asarray(right_index, dtype=np.int64)


def interpolate_poses(
    query_ns: Int[ndarray, "m"],
    source_ns: Int[ndarray, "n"],
    source_rotations: Rotation,
    source_translations: Float[ndarray, "n 3"],
) -> tuple[Float64[ndarray, "m 3 3"], Float64[ndarray, "m 3"]]:
    """Interpolate a pose stream onto ``query_ns`` (slerp + linear, clamped)."""
    clamped: Float64[ndarray, "m"] = np.clip(query_ns, source_ns[0], source_ns[-1]).astype(np.float64)
    slerp: Slerp = Slerp(source_ns.astype(np.float64), source_rotations)
    rotations: Float64[ndarray, "m 3 3"] = slerp(clamped).as_matrix()
    translations: Float64[ndarray, "m 3"] = np.stack(
        [np.interp(clamped, source_ns.astype(np.float64), source_translations[:, axis]) for axis in range(3)],
        axis=-1,
    )
    return rotations, translations


# ─────────────────────────────────────────────────────────────────────────────
# Prepared-sequence container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PreparedCamera:
    """One camera of a prepared sequence, in rig-relative form."""

    name: str
    """Human role label, e.g. ``left_front``."""
    rig_T_cam: Float64[ndarray, "4 4"]
    """Static extrinsic: camera pose in the rig frame."""
    k_matrix: Float64[ndarray, "3 3"]
    """Pinhole intrinsics."""
    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""
    distortion: tuple[float, ...] = ()
    """Kannala-Brandt ``k1..k4`` for fisheye cameras; empty for pinhole."""
    used_for_sfm: bool = True
    """False for cameras logged as frusta but withheld from the reconstruction."""
    kind: SensorKind = "rgb"
    """Image content as recorded upstream (RoboCap's cameras are ``grayscale``)."""


@dataclass
class PreparedSequence:
    """Everything cuSFM and Rerun need, independent of the source dataset."""

    name: str
    """Short dataset label used in entity metadata and run directories."""
    cameras: list[PreparedCamera]
    """All cameras on the rig, in entity order."""
    timestamps_ns: Int[ndarray, "n_samples"]
    """One canonical timestamp per synchronised sample."""
    world_T_rig: Float64[ndarray, "n_samples 4 4"]
    """Input trajectory: rig pose in world, per sample."""
    image_paths: dict[str, list[Path | None]]
    """Per-camera JPEG path per sample (``None`` where a frame is missing)."""
    input_dir: Path
    """Directory containing the per-camera image folders + ``frames_meta.json``."""
    frames_meta_path: Path
    """The ``frames_meta.json`` handed to cuSFM."""
    image_plane_distance: float
    """Frustum length in metres."""
    projection_model: str
    """``PINHOLE`` or ``OPENCV_FISHEYE``."""
    full_timestamps_ns: Int[ndarray, "n_frames"] | None = None
    """Every source frame timestamp (30 Hz), not just the sampled keyframes.
    Drives the continuous video + pose so the viewer shows the whole walk, while
    cuSFM still only ever receives the keyframes."""
    full_world_T_rig: Float64[ndarray, "n_frames 4 4"] | None = None
    """Rig pose at every ``full_timestamps_ns`` entry."""
    keyframe_mask: Bool[ndarray, "n_frames"] | None = None
    """True where a full-rate frame is one of the keyframes handed to cuSFM."""
    video_samples: dict[str, list[tuple[int, bytes]]] = field(default_factory=dict)
    """Per-camera raw H.264 samples, re-emitted verbatim as a ``VideoStream``."""
    reference_name: str = "imu_00"
    """True body-frame origin recorded in the rig metadata.

    We deliberately follow `COLMAP's rig convention <https://colmap.github.io/rigs.html>`_
    and give ``RigCalibration`` a camera reference sensor even though the physical
    origin is RoboCap's IMU or Galileo's vehicle frame. The viewer therefore tints
    ``cam_00`` as the reference camera; this value overrides the schema metadata
    with the true origin instead. Do not make the tint and metadata agree by moving
    the rig frame to the camera."""
    input_source: str = "ego-motion"
    """Provenance of ``world_T_rig``, shown on the rig in the viewer and printed.
    Never "ground truth": on every dataset here the input trajectory is an estimate."""
    base_dir: Path | None = None
    """Explicit cuSFM workspace. ``None`` uses ``run.work_dir/<name><variant>/cusfm``.
    KITTI pins its own so the run lands beside the sequence, the ground truth and the
    recording the task writes."""

    @property
    def sfm_cameras(self) -> list[PreparedCamera]:
        """The cameras cuSFM was actually given, in entity order.

        Distinct from ``cameras``, which is the whole rig as logged to Rerun. On
        RoboCap those differ -- six on the rig, four to cuSFM -- and conflating
        them is silent: ``len(cameras)`` was once used as the divisor for cuSFM
        keyframe ids, scaling every loop-closure chord by 4/6 with nothing
        visibly wrong. Prefer this property."""
        return [camera for camera in self.cameras if camera.used_for_sfm]


# ─────────────────────────────────────────────────────────────────────────────
# frames_meta.json
# ─────────────────────────────────────────────────────────────────────────────


def write_frames_meta(sequence: PreparedSequence, stereo_pairs: tuple[tuple[str, str], ...]) -> Path:
    """Write cuSFM's ``frames_meta.json`` input contract.

    ``camera_to_world`` is the **camera** pose in world (verified against
    galileo's ``ground_truth.txt``: ``camera_to_world @ inv(sensor_to_vehicle)``
    reproduces the GT vehicle pose to ~1.6 mm), and
    ``sensor_to_vehicle_transform`` is ``rig_T_cam``.
    """
    sfm_cameras: list[PreparedCamera] = sequence.sfm_cameras
    camera_params: dict[str, dict] = {}
    name_to_id: dict[str, str] = {}

    for camera_id, camera in enumerate(sfm_cameras):
        params_id: str = str(camera_id)
        name_to_id[camera.name] = params_id
        calibration: dict = {
            "image_width": camera.width,
            "image_height": camera.height,
        }
        if sequence.projection_model == "OPENCV_FISHEYE":
            calibration["camera_matrix"] = {
                "data": [float(value) for value in camera.k_matrix.reshape(-1)],
                "row_count": 3,
                "column_count": 3,
            }
            calibration["distortion_coefficients"] = {
                "data": [float(value) for value in camera.distortion],
                "row_count": 1,
                "column_count": len(camera.distortion),
            }
        else:
            projection: Float64[ndarray, "3 4"] = np.hstack([camera.k_matrix, np.zeros((3, 1))])
            calibration["projection_matrix"] = {
                "data": [float(value) for value in projection.reshape(-1)],
                "row_count": 3,
                "column_count": 4,
            }
        camera_params[params_id] = {
            "sensor_meta_data": {
                "sensor_id": camera_id,
                "sensor_type": "CAMERA",
                "sensor_name": camera.name,
                "sensor_to_vehicle_transform": to_axis_angle_dict(camera.rig_T_cam),
            },
            "calibration_parameters": calibration,
            "camera_projection_model_type": sequence.projection_model,
        }

    keyframes: list[dict] = []
    for sample_index, timestamp_ns in enumerate(sequence.timestamps_ns):
        for camera in sfm_cameras:
            image_path: Path | None = sequence.image_paths[camera.name][sample_index]
            if image_path is None:
                continue
            world_T_cam: Float64[ndarray, "4 4"] = sequence.world_T_rig[sample_index] @ camera.rig_T_cam
            keyframes.append(
                {
                    "id": str(len(keyframes)),
                    "camera_params_id": name_to_id[camera.name],
                    "timestamp_microseconds": str(int(timestamp_ns) // 1000),
                    "image_name": f"{camera.name}/{image_path.name}",
                    "camera_to_world": to_axis_angle_dict(world_T_cam),
                    "synced_sample_id": str(sample_index),
                }
            )

    pair_entries: list[dict] = []
    by_name: dict[str, PreparedCamera] = {camera.name: camera for camera in sfm_cameras}
    for left_name, right_name in stereo_pairs:
        if left_name not in name_to_id or right_name not in name_to_id:
            continue
        baseline: float = float(
            np.linalg.norm(by_name[left_name].rig_T_cam[:3, 3] - by_name[right_name].rig_T_cam[:3, 3])
        )
        pair_entries.append(
            {
                "left_camera_param_id": name_to_id[left_name],
                "right_camera_param_id": name_to_id[right_name],
                "baseline_meters": baseline,
            }
        )

    frames_meta: dict = {
        "keyframes_metadata": keyframes,
        "initial_pose_type": "EGO_MOTION",
        "camera_params_id_to_camera_params": camera_params,
        "stereo_pair": pair_entries,
    }
    sequence.frames_meta_path.parent.mkdir(parents=True, exist_ok=True)
    sequence.frames_meta_path.write_text(json.dumps(frames_meta, indent=2))
    print(
        f"  frames_meta.json: {len(keyframes)} keyframes, {len(camera_params)} cameras "
        f"({sequence.projection_model}), {len(pair_entries)} stereo pair(s)"
    )
    return sequence.frames_meta_path


# ─────────────────────────────────────────────────────────────────────────────
# RoboCap: frozen .rrd -> JPEG frames + rig geometry
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RrdCameraRecord:
    """Raw per-camera data pulled out of a frozen exoego recording."""

    name: str
    cam_T_rig: Float64[ndarray, "4 4"]
    k_matrix: Float64[ndarray, "3 3"]
    width: int
    height: int
    distortion: tuple[float, ...]
    kind: SensorKind = "rgb"
    samples: list[tuple[int, bytes]] = field(default_factory=list)


def fetch_robocap_segment(rrd_path: Path, catalog_url: str = ROBOCAP_CATALOG_URL) -> Path:
    """Freeze the RoboCap segment from a running catalog into a local ``.rrd``.

    The catalog this was developed against stores segments **in memory**
    (``memory:///store/...``), so a catalog restart loses them. Freezing to disk
    once makes the demo reproducible and removes the catalog from the hot path.
    """
    from rerun.catalog import CatalogClient

    print(f"[robocap] {rrd_path} missing — pulling {ROBOCAP_SEGMENT} from {catalog_url}")
    try:
        client = CatalogClient(catalog_url)
        dataset = client.get_dataset(name="robocap")
        recording = dataset.download_segment(ROBOCAP_SEGMENT)
    except Exception as error:
        raise RuntimeError(
            f"could not fetch {ROBOCAP_SEGMENT} from {catalog_url} ({error}). "
            "Start the catalog, or point --dataset.rrd-path at an existing .rrd."
        ) from error
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    recording.save(str(rrd_path))
    print(f"[robocap] wrote {rrd_path} ({rrd_path.stat().st_size / 1e6:.1f} MB)")
    return rrd_path


def read_robocap_rrd(rrd_path: Path) -> tuple[dict[int, RrdCameraRecord], Int[ndarray, "n"], Rotation, Float64[ndarray, "n 3"]]:
    """Read cameras, video samples, and the basalt trajectory from a frozen segment."""
    from rerun.experimental import RrdReader

    if not rrd_path.is_file():
        fetch_robocap_segment(rrd_path)

    cameras: dict[int, RrdCameraRecord] = {}
    pose_times: list[int] = []
    pose_quats: list[list[float]] = []
    pose_translations: list[list[float]] = []

    def camera_index(entity_path: str) -> int | None:
        parts: list[str] = entity_path.strip("/").split("/")
        for part in parts:
            if part.startswith("cam_"):
                return int(part.removeprefix("cam_"))
        return None

    store = RrdReader(str(rrd_path)).store()
    for chunk in store.stream():
        entity_path: str = str(chunk.entity_path)
        if not entity_path.startswith("/world/rig_00"):
            continue
        record_batch = chunk.to_record_batch()
        component_names: set[str] = set(record_batch.schema.names)

        if entity_path == "/world/rig_00":
            required: set[str] = {"Transform3D:quaternion", "Transform3D:translation", TIMELINE}
            if not required.issubset(component_names):
                continue
            quaternions: pa.Array = record_batch.column("Transform3D:quaternion")
            translations: pa.Array = record_batch.column("Transform3D:translation")
            times: pa.Array = record_batch.column(TIMELINE)
            for quaternion, translation, timestamp in zip(
                quaternions.to_pylist(), translations.to_pylist(), times.to_pylist()
            ):
                if quaternion is None or translation is None or timestamp is None:
                    continue
                pose_times.append(int(getattr(timestamp, "value", timestamp)))
                pose_quats.append(list(unwrap_singleton_lists(quaternion)))
                pose_translations.append(list(unwrap_singleton_lists(translation)))
            continue

        index: int | None = camera_index(entity_path)
        if index is None:
            continue

        if entity_path.endswith("/pinhole/video"):
            if "VideoStream:sample" not in component_names or TIMELINE not in component_names:
                continue
            samples: pa.Array = record_batch.column("VideoStream:sample")
            times = record_batch.column(TIMELINE)
            record: RrdCameraRecord | None = cameras.get(index)
            if record is None:
                record = RrdCameraRecord(f"cam_{index:02d}", np.eye(4), np.eye(3), 0, 0, ())
                cameras[index] = record
            # simplecv's canonical RRD video reader needs rerun-sdk[datafusion], absent from this frozen runtime; keep direct Arrow reads for the exo path.
            # Read the H.264 blobs straight out of the Arrow buffer. The obvious
            # `samples.to_pylist()` boxes every byte of every sample into Python
            # objects: profiled at 65.7 s for this segment versus 0.4 s here, a
            # 164x difference for byte-identical output, and it dominated the whole
            # Python side of the run (94.6 % of samples in py-spy).
            blobs = samples.combine_chunks() if hasattr(samples, "combine_chunks") else samples
            inner = blobs.flatten()  # list<list<u8>> -> list<u8>, one entry per sample
            offsets: Int[ndarray, "n+1"] = inner.offsets.to_numpy()
            byte_buffer: Int[ndarray, "n_bytes"] = inner.values.to_numpy(zero_copy_only=False)
            valid: Bool[ndarray, "n"] = inner.is_valid().to_numpy(zero_copy_only=False)
            stamps: list = times.to_pylist()
            for row, timestamp in enumerate(stamps):
                if timestamp is None or row >= len(inner) or not valid[row]:
                    continue
                raw: bytes = byte_buffer[offsets[row] : offsets[row + 1]].tobytes()
                record.samples.append((int(getattr(timestamp, "value", timestamp)), raw))
            continue

        record = cameras.setdefault(index, RrdCameraRecord(f"cam_{index:02d}", np.eye(4), np.eye(3), 0, 0, ()))

        if entity_path.endswith("/pinhole"):
            k_value: list[float] = first_valid_value(
                record_batch.column("Pinhole:image_from_camera"), component_name="Pinhole:image_from_camera"
            )
            # Rerun Mat3x3 is column-major.
            record.k_matrix = np.asarray(k_value, dtype=np.float64).reshape(3, 3).T
            resolution_value: list[float] = first_valid_value(
                record_batch.column("Pinhole:resolution"), component_name="Pinhole:resolution"
            )
            record.width = int(resolution_value[0])
            record.height = int(resolution_value[1])
            distortion_value: list[float] = first_valid_value(
                record_batch.column("simplecv.components.DistortionCoefficients"),
                component_name="simplecv.components.DistortionCoefficients",
            )
            # Fisheye62 stores 8 slots; RoboCap populates only k1..k4.
            record.distortion = tuple(float(c) for c in distortion_value[:4])
            continue

        # Bare camera node: name/kind and the static rig_T_cam.
        if "name" in component_names:
            name_value: list[str] = first_valid_value(record_batch.column("name"), component_name="name")
            record.name = str(name_value[0])
        if "kind" in component_names:
            kind_value: list[str] = first_valid_value(record_batch.column("kind"), component_name="kind")
            kind: str = str(kind_value[0])
            if kind not in ("rgb", "grayscale", "depth", "imu"):
                raise ValueError(f"{entity_path}: unsupported sensor kind {kind!r}")
            record.kind = cast(SensorKind, kind)
        transform_components: set[str] = {"Transform3D:mat3x3", "Transform3D:translation"}
        if transform_components.issubset(component_names):
            mat_value: list[float] = first_valid_value(
                record_batch.column("Transform3D:mat3x3"), component_name="Transform3D:mat3x3"
            )
            translation_value: list[float] = first_valid_value(
                record_batch.column("Transform3D:translation"), component_name="Transform3D:translation"
            )
            if "Transform3D:relation" in component_names:
                relation_value: list[int] = first_valid_value(
                    record_batch.column("Transform3D:relation"), component_name="Transform3D:relation"
                )
            else:
                relation_value = []
            if relation_value and int(relation_value[0]) != CHILD_FROM_PARENT:
                raise ValueError(
                    f"{entity_path}: expected a ChildFromParent extrinsic "
                    f"(got relation={int(relation_value[0])}); the stored value would be "
                    f"rig_T_cam, not cam_T_rig, and inverting it would flip every frustum."
                )
            rotation: Float64[ndarray, "3 3"] = np.asarray(mat_value, dtype=np.float64).reshape(3, 3).T
            record.cam_T_rig = compose(rotation, np.asarray(translation_value, dtype=np.float64))

    if not pose_times:
        raise RuntimeError(f"{rrd_path} has no world_T_rig stream — is the `slam` layer present?")

    order: Int[ndarray, "n"] = np.argsort(np.asarray(pose_times))
    times_ns: Int[ndarray, "n"] = np.asarray(pose_times, dtype=np.int64)[order]
    quaternions_xyzw: Float64[ndarray, "n 4"] = np.asarray(pose_quats, dtype=np.float64)[order]
    translations: Float64[ndarray, "n 3"] = np.asarray(pose_translations, dtype=np.float64)[order]
    for record in cameras.values():
        record.samples.sort(key=lambda item: item[0])

    print(f"  read {len(cameras)} cameras, {len(times_ns)} rig poses from {rrd_path.name}")
    return cameras, times_ns, Rotation.from_quat(quaternions_xyzw), translations


def extract_jpegs(
    mp4_path: Path, output_dir: Path, frame_indices: Sequence[int], timestamps_ns: list[int]
) -> list[Path]:
    """Extract exactly ``frame_indices`` as JPEGs named by their nanosecond timestamp.

    Takes the explicit index list rather than a stride so the ffmpeg selection and
    the output filenames cannot disagree. They previously could: the filter kept
    ``n % stride == 0`` (0, stride, 2*stride, ...) while the caller expects frames
    at ``first_ok + k*stride``, and those two sets coincide only when
    ``first_ok % stride == 0``. The shipped defaults satisfy that by luck
    (``first_ok`` 60, stride 20), but ``--dataset.trim-start-seconds 1.0`` makes
    every expected file missing, which surfaces as an empty ``frames_meta.json``
    and a reconstruction failure far downstream.

    CPU decode is used deliberately: measured on this host, ``h264_cuvid`` took
    4.16 s against 2.32 s for CPU on 233 frames, because the cost is JPEG
    *encoding* and the device-to-host copy, not H.264 decode.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    staging: Path = output_dir / "_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    # An explicit `eq(n,i)+eq(n,j)+...` term per frame: unambiguous, and it also
    # stops ffmpeg encoding the leading frames the trim discards.
    wanted: str = "+".join(rf"eq(n\,{index})" for index in frame_indices)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(mp4_path),
            "-vf", f"select='{wanted}'",
            # `-fps_mode passthrough` is the tested ffmpeg 8.x path. The project
            # keeps that major pinned because this extraction is untested on 9.x.
            "-fps_mode", "passthrough", "-q:v", "2",
            str(staging / "f_%06d.jpg"),
        ],
        check=True,
    )

    extracted: list[Path] = sorted(staging.glob("f_*.jpg"))
    if len(extracted) != len(frame_indices):
        print(
            f"    WARNING: asked ffmpeg for {len(frame_indices)} frames from "
            f"{mp4_path.name}, got {len(extracted)}"
        )
    paths: list[Path] = []
    for output_index, source_index in enumerate(frame_indices):
        if output_index >= len(extracted):
            break
        destination: Path = output_dir / f"{timestamps_ns[source_index]}.jpeg"
        shutil.move(str(extracted[output_index]), destination)
        paths.append(destination)
    shutil.rmtree(staging, ignore_errors=True)
    return paths


def prepare_robocap(config: RobocapConfig, run: RunConfig) -> PreparedSequence:
    """Turn a frozen RoboCap segment into cuSFM's on-disk input contract."""
    print(f"[robocap] reading {config.rrd_path}")
    start: float = time.time()
    records, pose_times_ns, pose_rotations, pose_translations = read_robocap_rrd(config.rrd_path)

    # Frames and remuxed video are shared across variants (extraction is the slow
    # part); frames_meta.json and cuSFM's stereo.edex are per-variant, because two
    # variants with different camera sets would otherwise overwrite each other.
    shared_dir: Path = run.work_dir / "robocap"
    work_dir: Path = run.work_dir / f"robocap{run.variant}"
    frames_dir: Path = shared_dir / "frames"
    input_dir: Path = work_dir / "input"
    video_dir: Path = shared_dir / "video"

    by_name: dict[str, RrdCameraRecord] = {record.name: record for record in records.values()}
    missing: list[str] = [name for name in config.cameras if name not in by_name]
    if missing:
        raise KeyError(f"cameras {missing} not in recording (have {sorted(by_name)})")

    reference_name: str = config.cameras[0]
    reference_samples: list[tuple[int, bytes]] = by_name[reference_name].samples
    all_times: Int[ndarray, "n"] = np.asarray([t for t, _ in reference_samples], dtype=np.int64)
    first_ok: int = int(np.searchsorted(all_times, all_times[0] + int(config.trim_start_seconds * 1e9)))
    last_ok: int = int(np.searchsorted(all_times, all_times[-1] - int(config.trim_end_seconds * 1e9)))
    if config.trim_start_seconds or config.trim_end_seconds:
        print(
            f"  trimmed {config.trim_start_seconds:g}s head / {config.trim_end_seconds:g}s tail: "
            f"frames {first_ok}..{last_ok} of {len(all_times)}"
        )
    sample_indices: list[int] = list(range(first_ok, last_ok, config.frame_stride))
    if config.max_samples is not None:
        sample_indices = sample_indices[: config.max_samples]
    canonical_times: Int[ndarray, "n_samples"] = np.asarray(
        [reference_samples[i][0] for i in sample_indices], dtype=np.int64
    )
    print(
        f"  {len(reference_samples)} frames -> {len(canonical_times)} samples "
        f"(stride {config.frame_stride}) across {len(config.cameras)} cameras"
    )

    input_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, list[Path | None]] = {}
    for name in config.cameras:
        record = by_name[name]
        mp4_path: Path = video_dir / f"{name}.mp4"
        if not mp4_path.is_file():
            mp4_path.parent.mkdir(parents=True, exist_ok=True)
            sample_times: pa.ChunkedArray = pa.chunked_array(
                [pa.array([timestamp for timestamp, _ in record.samples], type=pa.duration("ns"))]
            )
            sample_payload: bytes = b"".join(raw for _, raw in record.samples)
            sample_offsets: Int[ndarray, "n+1"] = np.zeros(len(record.samples) + 1, dtype=np.int32)
            np.cumsum([len(raw) for _, raw in record.samples], out=sample_offsets[1:])
            sample_array: pa.ListArray = pa.ListArray.from_arrays(
                pa.array(sample_offsets), pa.array(np.frombuffer(sample_payload, dtype=np.uint8))
            )
            mux_h264_to_mp4(sample_times, pa.chunked_array([sample_array]), str(mp4_path))
        camera_dir: Path = frames_dir / name
        timestamps: list[int] = [timestamp for timestamp, _ in record.samples]
        # MUST be the same indices as `sample_indices`, which are already trimmed.
        # Selecting from 0 here while the poses start at `first_ok` would pair every
        # image with a pose ~first_ok/stride samples away -- a silent misalignment.
        selected: list[int] = list(sample_indices)
        # Address frames by timestamp, never by directory order. Two traps otherwise:
        # `sorted()` is lexicographic, so `99621592000.jpeg` (t < 100 s) sorts AFTER
        # `100021636444.jpeg` and silently pairs samples with the wrong images; and the
        # frame cache is shared across strides, so a coarse run following a dense one
        # would find "enough" files and take the first N -- the opening seconds of video
        # rather than the strided samples.
        expected_paths: list[Path] = [camera_dir / f"{timestamps[i]}.jpeg" for i in selected]
        present: set[Path] = {path for path in expected_paths if path.is_file()}
        if len(present) != len(expected_paths):
            extract_jpegs(mp4_path, camera_dir, selected, timestamps)
            present = {path for path in expected_paths if path.is_file()}
        image_paths[name] = [
            expected_paths[i] if expected_paths[i] in present else None
            for i in range(len(canonical_times))
        ]
        # cuSFM resolves image_name relative to input_dir, so link the shared
        # frame directory into this variant's input directory.
        link: Path = input_dir / name
        # `exists()` follows the link, so a *dangling* symlink reports False and
        # `symlink_to` then raises FileExistsError. Test the link itself.
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(camera_dir.resolve(), target_is_directory=True)
        print(f"    {name:12s} {sum(p is not None for p in image_paths[name])} frames")

    # Full-rate stream: one pose per source frame, for continuous playback.
    full_times: Int[ndarray, "n_frames"] = all_times[first_ok:last_ok]
    full_rotations, full_translations = interpolate_poses(
        full_times, pose_times_ns, pose_rotations, pose_translations
    )
    full_world_T_rig: Float64[ndarray, "n_frames 4 4"] = np.stack(
        [compose(full_rotations[i], full_translations[i]) for i in range(len(full_times))]
    )
    keyframe_mask: Bool[ndarray, "n_frames"] = np.zeros(len(full_times), dtype=bool)
    keyframe_mask[np.asarray(sample_indices, dtype=np.int64) - first_ok] = True

    rig_rotations, rig_translations = interpolate_poses(
        canonical_times, pose_times_ns, pose_rotations, pose_translations
    )
    world_T_rig: Float64[ndarray, "n_samples 4 4"] = np.stack(
        [compose(rig_rotations[i], rig_translations[i]) for i in range(len(canonical_times))]
    )

    cameras: list[PreparedCamera] = []
    for index in sorted(records):
        record = records[index]
        cameras.append(
            PreparedCamera(
                name=record.name,
                rig_T_cam=np.linalg.inv(record.cam_T_rig),
                k_matrix=record.k_matrix,
                width=record.width,
                height=record.height,
                distortion=record.distortion,
                used_for_sfm=record.name in config.cameras,
                kind=record.kind,
            )
        )

    sequence: PreparedSequence = PreparedSequence(
        name="robocap",
        cameras=cameras,
        timestamps_ns=canonical_times,
        world_T_rig=world_T_rig,
        image_paths={c.name: image_paths.get(c.name, [None] * len(canonical_times)) for c in cameras},
        input_dir=input_dir,
        frames_meta_path=input_dir / CUSFM_FRAME_META_FILE,
        image_plane_distance=config.image_plane_distance,
        projection_model="OPENCV_FISHEYE",
        reference_name="imu_00",
        input_source="basalt VIO",
        full_timestamps_ns=full_times,
        full_world_T_rig=full_world_T_rig,
        keyframe_mask=keyframe_mask,
        video_samples={
            name: [
                (t, raw)
                for t, raw in by_name[name].samples
                if all_times[first_ok] <= t <= all_times[last_ok - 1]
            ]
            for name in config.cameras
        },
    )
    write_frames_meta(
        sequence,
        ROBOCAP_RIG_ADJACENCY_PAIRS if config.rig_adjacency_pairs else (ROBOCAP_STEREO_PAIR,),
    )
    print(f"[robocap] prepared in {time.time() - start:.1f}s")
    return sequence


# ─────────────────────────────────────────────────────────────────────────────
# Datasets already in cuSFM's frames_meta.json format: galileo and KITTI 06
# ─────────────────────────────────────────────────────────────────────────────


def from_axis_angle_dict(node: dict) -> Float64[ndarray, "4 4"]:
    """Inverse of :func:`to_axis_angle_dict`."""
    axis_angle: dict = node["axis_angle"]
    translation: dict = node["translation"]
    axis: Float64[ndarray, "3"] = np.array(
        [axis_angle["x"], axis_angle["y"], axis_angle["z"]], dtype=np.float64
    )
    norm: float = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    rotation: Float64[ndarray, "3 3"] = Rotation.from_rotvec(
        axis * np.radians(axis_angle["angle_degrees"])
    ).as_matrix()
    return compose(rotation, np.array([translation["x"], translation["y"], translation["z"]], dtype=np.float64))


def read_frames_meta_cameras(frames_meta: dict) -> tuple[list[PreparedCamera], dict[str, str]]:
    """Read ``camera_params_id_to_camera_params`` into prepared cameras.

    Shared by every dataset that already speaks cuSFM's format. Returns the cameras
    in ``camera_params_id`` order plus the id-to-sensor-name map that the keyframe
    records are indexed by.
    """
    camera_params: dict = frames_meta["camera_params_id_to_camera_params"]
    cameras: list[PreparedCamera] = []
    id_to_name: dict[str, str] = {}
    for params_id in sorted(camera_params, key=int):
        entry: dict = camera_params[params_id]
        sensor: dict = entry["sensor_meta_data"]
        calibration: dict = entry["calibration_parameters"]
        projection: Float64[ndarray, "3 4"] = np.asarray(
            calibration["projection_matrix"]["data"], dtype=np.float64
        ).reshape(3, 4)
        id_to_name[params_id] = sensor["sensor_name"]
        cameras.append(
            PreparedCamera(
                name=sensor["sensor_name"],
                rig_T_cam=from_axis_angle_dict(sensor["sensor_to_vehicle_transform"]),
                k_matrix=projection[:3, :3],
                width=int(calibration["image_width"]),
                height=int(calibration["image_height"]),
                used_for_sfm=True,
            )
        )
    return cameras, id_to_name


def prepare_galileo(config: GalileoConfig, run: RunConfig) -> PreparedSequence:
    """Read upstream's bundled ``frames_meta.json`` into the common container."""
    print(f"[galileo] reading {config.input_dir}")
    frames_meta: dict = json.loads((config.input_dir / CUSFM_FRAME_META_FILE).read_text())
    cameras, id_to_name = read_frames_meta_cameras(frames_meta)

    # Upstream's frames_meta.json is proto3-serialised, and proto3 drops
    # zero-valued scalar fields — so `camera_params_id: "0"` is simply absent
    # from every keyframe of camera 0 (29 of galileo's 226, all
    # back_stereo_camera_left). Default it back to "0", then cross-check against
    # the image directory, which equals the sensor name for every camera here.
    name_to_id: dict[str, str] = {name: params_id for params_id, name in id_to_name.items()}

    def keyframe_camera_id(keyframe: dict) -> str:
        params_id: str = keyframe.get("camera_params_id", "0")
        directory: str = keyframe["image_name"].split("/")[0]
        expected: str | None = name_to_id.get(directory)
        if expected is not None and expected != params_id:
            raise ValueError(
                f"{keyframe['image_name']}: camera_params_id {params_id!r} maps to "
                f"{id_to_name[params_id]!r}, which contradicts the image directory."
            )
        return params_id

    # Group keyframes by synced_sample_id -> one rig sample per group.
    grouped: dict[str, list[dict]] = {}
    for keyframe in frames_meta["keyframes_metadata"]:
        grouped.setdefault(keyframe["synced_sample_id"], []).append(keyframe)
    sample_ids: list[str] = sorted(grouped, key=int)

    timestamps: list[int] = []
    world_T_rig_list: list[Float64[ndarray, "4 4"]] = []
    image_paths: dict[str, list[Path | None]] = {camera.name: [] for camera in cameras}
    by_name: dict[str, PreparedCamera] = {camera.name: camera for camera in cameras}

    for sample_id in sample_ids:
        keyframes: list[dict] = grouped[sample_id]
        rig_estimates: list[Float64[ndarray, "4 4"]] = []
        present: dict[str, Path] = {}
        for keyframe in keyframes:
            name: str = id_to_name[keyframe_camera_id(keyframe)]
            world_T_cam: Float64[ndarray, "4 4"] = from_axis_angle_dict(keyframe["camera_to_world"])
            rig_estimates.append(world_T_cam @ np.linalg.inv(by_name[name].rig_T_cam))
            present[name] = config.input_dir / keyframe["image_name"]
        timestamps.append(int(keyframes[0]["timestamp_microseconds"]) * 1000)
        world_T_rig_list.append(rig_estimates[0])
        for camera in cameras:
            image_paths[camera.name].append(present.get(camera.name))

    sequence: PreparedSequence = PreparedSequence(
        name="galileo",
        cameras=cameras,
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        world_T_rig=np.stack(world_T_rig_list),
        image_paths=image_paths,
        input_dir=config.input_dir,
        frames_meta_path=config.input_dir / CUSFM_FRAME_META_FILE,
        image_plane_distance=config.image_plane_distance,
        projection_model="PINHOLE",
        reference_name="vehicle_00",
    )
    print(f"  {len(sequence.timestamps_ns)} samples, {len(cameras)} cameras (upstream frames_meta.json used as-is)")
    return sequence



def prepare_kitti(config: KittiConfig, run: RunConfig) -> PreparedSequence:
    """Read the KITTI 06 sequence that ``get_framemeta_file_for_KITTI.py`` converted.

    KITTI ships no trajectory, so ``world_T_rig`` is left at identity here and filled
    in by :func:`attach_cuvslam_trajectory` once cuVSLAM has run inside the pipeline.
    Nothing between the two reads it: cuSFM is driven by ``frames_meta.json``, and
    that file is regenerated with cuVSLAM's poses before feature extraction.

    The converter puts the vehicle frame on camera 0 (its
    ``sensor_to_vehicle_transform`` is a pure baseline offset along X), which is also
    the frame KITTI's ground truth is written in — so no frame change is ever needed
    between the ground truth, cuVSLAM's TUM output and cuSFM's vehicle-frame poses.
    """
    meta_path: Path = config.input_dir / CUSFM_FRAME_META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{meta_path} is missing. Run `pixi run kitti06`, which downloads the sequence and "
            f"converts it with data/kitti/get_framemeta_file_for_KITTI.py. That script always "
            f"writes keyframe_meta.json regardless of --output-name, so the task copies it."
        )
    print(f"[kitti06] reading {config.input_dir}")
    frames_meta: dict = json.loads(meta_path.read_text())
    cameras, id_to_name = read_frames_meta_cameras(frames_meta)

    grouped: dict[str, dict[str, str]] = {}
    timestamps_us: dict[str, int] = {}
    for keyframe in frames_meta["keyframes_metadata"]:
        sample_id: str = keyframe["synced_sample_id"]
        grouped.setdefault(sample_id, {})[id_to_name[keyframe["camera_params_id"]]] = keyframe["image_name"]
        timestamps_us[sample_id] = int(keyframe["timestamp_microseconds"])
    sample_ids: list[str] = sorted(grouped, key=int)

    image_paths: dict[str, list[Path | None]] = {
        camera.name: [
            config.input_dir / grouped[sample_id][camera.name] if camera.name in grouped[sample_id] else None
            for sample_id in sample_ids
        ]
        for camera in cameras
    }
    sequence: PreparedSequence = PreparedSequence(
        name="kitti06",
        cameras=cameras,
        timestamps_ns=np.asarray([timestamps_us[sample_id] * 1000 for sample_id in sample_ids], dtype=np.int64),
        world_T_rig=np.tile(np.eye(4, dtype=np.float64), (len(sample_ids), 1, 1)),
        image_paths=image_paths,
        input_dir=config.input_dir,
        frames_meta_path=meta_path,
        image_plane_distance=config.image_plane_distance,
        projection_model="PINHOLE",
        reference_name="vehicle_00",
        input_source="cuVSLAM (pending)",
        base_dir=config.result_dir,
    )
    print(f"  {len(sample_ids)} stereo samples, {len(cameras)} cameras (upstream frames_meta.json used as-is)")
    return sequence


def kitti_run_config(run: RunConfig) -> RunConfig:
    """Apply KITTI 06's published configuration to any knob still at its default.

    tyro picks the dataset through a subcommand but every dataset shares one
    ``RunConfig``, so ``dataset:kitti06`` would otherwise inherit the RoboCap
    defaults — fatally ``skip_cuvslam=True``, since KITTI carries no poses at all and
    cuSFM would be handed identity for every frame.

    Anything set explicitly is left alone, which is why this compares against a fresh
    ``RunConfig()`` instead of overwriting unconditionally: ``pixi run kitti06`` gets
    the paper's setup, and ``--run.no-use-cuvslam-slam-pose`` still means what it says.
    """
    defaults: RunConfig = RunConfig()
    published: dict[str, object] = {
        # Upstream's KITTI profile, unmodified. The loop-closure-repaired isaac copy
        # that the other two datasets default to is not this experiment.
        "config_dir": KITTI_CONFIG_DIR,
        # KITTI has no input trajectory; cuVSLAM has to produce one.
        "skip_cuvslam": False,
        # The paper's baseline row is cuVSLAM *SLAM*, not raw odometry.
        "use_cuvslam_slam_pose": True,
        # docs/tutorial.md: the paper's experiments used pose-graph optimisation
        # without the separate data-association stage.
        "skip_data_association": True,
        # cuSFM's own default keyframe gate; at 10 Hz and ~11 m/s it keeps almost
        # every frame, and leaving it at the demo's 0.0 changes which frames the
        # solver sees.
        "min_inter_frame_distance": 0.5,
    }
    changed: dict[str, object] = {
        name: value for name, value in published.items() if getattr(run, name) == getattr(defaults, name)
    }
    if changed:
        print("[kitti06] applying the published run configuration: " + ", ".join(sorted(changed)))
    return replace(run, **changed)


def attach_cuvslam_trajectory(sequence: PreparedSequence, base_dir: Path, use_slam_pose: bool) -> PreparedSequence:
    """Adopt the trajectory cuVSLAM produced during the run as the input trajectory.

    KITTI has no poses of its own, so ``rig_00`` shows whatever cuVSLAM estimated —
    ``slam_poses.tum`` (loop-closed) or ``odom_poses.tum`` (raw odometry), matching
    what cuSFM was initialised from.

    cuVSLAM does not pose every frame (it has no pose for frame 0, which it defines
    as the origin of nothing), so the sequence is narrowed to the samples it did
    pose. Leaving identity in for the rest would park a phantom rig at the origin and
    drag the Umeyama fit with it.
    """
    poses_file: Path = base_dir / CUSFM_CUVSLAM_DIR / (
        CUSFM_SLAM_POSES_FILE if use_slam_pose else CUSFM_ODOM_POSES_FILE
    )
    trajectory = read_tum_poses(poses_file)
    if trajectory is None:
        raise FileNotFoundError(
            f"{poses_file} is missing or empty, so KITTI has no input trajectory. "
            f"cuVSLAM must run for this dataset (--run.no-skip-cuvslam)."
        )
    times_ns, poses = trajectory
    sample_index, pose_index = match_timestamps(sequence.timestamps_ns, times_ns)
    if len(sample_index) < 3:
        raise RuntimeError(
            f"only {len(sample_index)} of {len(sequence.timestamps_ns)} samples matched a cuVSLAM pose "
            f"in {poses_file}; the timestamp contract between frames_meta.json and the TUM output broke."
        )
    label: str = f"cuVSLAM {'SLAM' if use_slam_pose else 'odometry'}"
    print(f"  input trajectory: {label}, {len(sample_index)}/{len(sequence.timestamps_ns)} samples posed")
    return replace(
        sequence,
        timestamps_ns=sequence.timestamps_ns[sample_index],
        world_T_rig=poses[pose_index],
        image_paths={
            name: [paths[index] for index in sample_index] for name, paths in sequence.image_paths.items()
        },
        input_source=label,
    )


def read_kitti_ground_truth(config: KittiConfig) -> tuple[Int[ndarray, "n"], Float64[ndarray, "n 4 4"]]:
    """Read ``poses/06.txt`` and pair it with ``times.txt``.

    Each row is a row-major 3x4 ``world_T_cam0``; the benchmark ships no timestamps
    with the poses, so row *i* is frame *i* and its time comes from ``times.txt``
    through the same truncation the upstream converter uses.
    """
    if not config.ground_truth.is_file():
        raise FileNotFoundError(
            f"{config.ground_truth} is missing. `pixi run kitti06` downloads it from the official "
            f"KITTI odometry archive and checks its md5; the image mirror carries no poses."
        )
    times_path: Path = config.input_dir / "times.txt"
    seconds: Float64[ndarray, "n"] = np.asarray(times_path.read_text().split(), dtype=np.float64)
    rows: Float64[ndarray, "n 3 4"] = np.asarray(
        [[float(value) for value in line.split()] for line in config.ground_truth.read_text().splitlines() if line.strip()],
        dtype=np.float64,
    ).reshape(-1, 3, 4)
    if len(rows) != len(seconds):
        raise ValueError(
            f"{config.ground_truth} has {len(rows)} poses but {times_path} has {len(seconds)} timestamps."
        )
    poses: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4, dtype=np.float64), (len(rows), 1, 1))
    poses[:, :3, :4] = rows
    # int(), not round(): this is exactly what get_framemeta_file_for_KITTI.py does to
    # times.txt, and the join has to agree with it to the microsecond.
    return (seconds * 1e6).astype(np.int64) * 1000, poses


def report_kitti_ground_truth(config: KittiConfig, sequence: PreparedSequence, base_dir: Path) -> None:
    """Score both trajectories against KITTI's ground truth and log it as a third run.

    Two rows of the reproduction table come out of one run: the cuVSLAM trajectory
    cuSFM was *initialised* with (``sequence.world_T_rig``) and cuSFM's own output
    (``output_poses/merged_pose_file.tum``, byte for byte the file ``evo_ape`` reads).

    Sim(3), matching the paper and the KITTI benchmark. The fitted scale is printed
    because it is a real diagnostic here — the stereo baseline makes the estimate
    metric, so a scale far from 1.0 would mean the reconstruction shrank, not that the
    alignment was free to choose.

    The ground truth is logged unaligned. It shares KITTI's camera-0 frame with
    cuVSLAM, which starts at the same origin, so the three lines are directly
    comparable and the drift between them is the thing worth seeing.
    """
    ground_truth_ns, ground_truth = read_kitti_ground_truth(config)
    write_tum_poses(config.ground_truth_tum, ground_truth_ns, ground_truth)

    print("\n─── KITTI 06 vs ground truth (Sim(3)-aligned ATE) ───")
    print(f"  ground truth      : {len(ground_truth)} poses, {trajectory_length(ground_truth[:, :3, 3]):.1f} m driven")

    input_index, truth_index = match_timestamps(sequence.timestamps_ns, ground_truth_ns)
    input_rmse, input_scale = sim3_ate(
        sequence.world_T_rig[input_index, :3, 3], ground_truth[truth_index, :3, 3]
    )
    print(
        f"  {sequence.input_source:18s}: {input_rmse:.3f} m RMSE over {len(input_index)} poses "
        f"(scale {input_scale:.5f})  <- what cuSFM was given"
    )

    refined = read_cusfm_vehicle_poses(base_dir / CUSFM_SPARSE_DIR)
    if refined is not None:
        refined_ns, refined_poses = refined
        refined_index, truth_index = match_timestamps(refined_ns, ground_truth_ns)
        refined_rmse, refined_scale = sim3_ate(
            refined_poses[refined_index, :3, 3], ground_truth[truth_index, :3, 3]
        )
        print(
            f"  {'cuSFM refined':18s}: {refined_rmse:.3f} m RMSE over {len(refined_index)} poses "
            f"(scale {refined_scale:.5f})  <- {CUSFM_MERGED_POSE_FILE}"
        )
    print(f"  {'paper, Table 4':18s}: CuVSLAM 1.202 m -> CuSfM-View Graph 0.783 m")
    print(f"  wrote {config.ground_truth_tum} for `evo_ape tum ... -as`")

    log_trajectory(
        "ground_truth",
        ground_truth[:, :3, 3],
        source=f"KITTI odometry benchmark ({len(ground_truth)} poses, unaligned)",
        hue=(140, 255, 140),
    )


# ─────────────────────────────────────────────────────────────────────────────
# cuSFM
# ─────────────────────────────────────────────────────────────────────────────


def run_cusfm(sequence: PreparedSequence, run: RunConfig) -> Path:
    """Run the cuSFM pipeline and return the COLMAP ``sparse/`` directory.

    On galileo and robocap cuVSLAM is skipped: both already carry a trajectory,
    fed in through ``frames_meta.json``'s ``camera_to_world`` under
    ``initial_pose_type: EGO_MOTION``, and cuSFM's global bundle adjustment refines
    it. On KITTI there is nothing to refine until cuVSLAM has run, so it runs first
    and rewrites ``frames_meta.json`` with its own poses.
    """
    from pycusfm.cusfm_runner import create_cusfm_runner

    base_dir: Path = (
        sequence.base_dir if sequence.base_dir is not None else run.work_dir / f"{sequence.name}{run.variant}" / "cusfm"
    )
    sparse_dir: Path = base_dir / CUSFM_SPARSE_DIR
    if run.skip_reconstruction:
        if not (sparse_dir / "images.txt").is_file() and not (sparse_dir / "images.bin").is_file():
            raise FileNotFoundError(f"--run.skip-reconstruction set but no model at {sparse_dir}")
        print(f"[cusfm] reusing existing model at {sparse_dir}")
        return sparse_dir

    base_dir.mkdir(parents=True, exist_ok=True)
    package_path: Path = Path(__file__).resolve().parent / "pycusfm"
    print(
        f"[cusfm] {sequence.name}{run.variant}: feature_type={run.feature_type} "
        f"ba_frame_type={run.ba_frame_type} skip_cuvslam={run.skip_cuvslam} "
        f"use_cuvslam_slam_pose={run.use_cuvslam_slam_pose} "
        f"skip_data_association={run.skip_data_association} "
        f"skip_pose_graph={run.skip_pose_graph} -> {base_dir}"
    )
    start: float = time.time()
    extra: dict[str, str] = {}
    if run.config_dir:
        extra["config_dir"] = str(run.config_dir)
    if run.model_dir:
        extra["model_dir"] = str(run.model_dir)
    runner = create_cusfm_runner(
        package_path=package_path,
        **extra,
        config_set="isaac",
        input_dir=str(sequence.input_dir),
        cusfm_base_dir=str(base_dir),
        binary_dir=str(package_path / "x86_cuda13" / "bin"),
        feature_type=run.feature_type,
        ba_frame_type=run.ba_frame_type,
        optimize_extrinsics=run.optimize_extrinsics,
        min_inter_frame_distance=run.min_inter_frame_distance,
        skip_cuvslam=run.skip_cuvslam,
        use_cuvslam_slam_pose=run.use_cuvslam_slam_pose,
        skip_feature_extractor=run.skip_feature_extractor,
        skip_data_association=run.skip_data_association,
        num_threads=run.num_threads,
        feature_extractor_batch_size=run.feature_extractor_batch_size,
        feature_matching_batch_size=run.feature_matching_batch_size,
        skip_pose_graph=run.skip_pose_graph,
        downsampling_matches=run.downsampling_matches,
        output_rgb=True,
    )
    runner.run_all()
    print(f"[cusfm] finished in {time.time() - start:.1f}s -> {sparse_dir}")
    return sparse_dir


@dataclass
class ColmapModel:
    """The parts of a COLMAP sparse model this demo needs."""

    world_T_cam: dict[str, Float64[ndarray, "4 4"]]
    """Image name (as written in ``frames_meta.json``) -> camera pose in world."""
    points_xyz: Float64[ndarray, "n_points 3"]
    """Sparse point positions."""
    points_rgb: Int[ndarray, "n_points 3"]
    """Per-point colour."""


def read_colmap_model(sparse_dir: Path) -> ColmapModel:
    """Load a COLMAP sparse model from ``images.txt`` + ``points3D.txt``.

    Deliberately parsed by hand rather than through ``pycolmap``: this
    environment is a delicately balanced CUDA-13 solve with a TensorRT
    dependency override, and pulling in pycolmap's ceres/CUDA stack risks
    perturbing it for the sake of two file formats that are a dozen lines each.
    cuSFM writes text by default (``--export_binary_colmap_files`` is opt-in).
    """
    images_path: Path = sparse_dir / "images.txt"
    points_path: Path = sparse_dir / "points3D.txt"
    if not images_path.is_file():
        if (sparse_dir / "images.bin").is_file():
            raise NotImplementedError(
                f"{sparse_dir} holds a binary COLMAP model; re-run cuSFM without "
                "--export_binary_colmap_files, or add pycolmap."
            )
        raise FileNotFoundError(f"no COLMAP model at {sparse_dir}")

    world_T_cam: dict[str, Float64[ndarray, "4 4"]] = {}
    # images.txt alternates: a pose line, then that image's 2D points (ignored).
    pose_lines: list[str] = [
        line for line in images_path.read_text().splitlines() if line and not line.startswith("#")
    ][::2]
    for line in pose_lines:
        fields: list[str] = line.split()
        if len(fields) < 10:
            continue
        quaternion_wxyz: Float64[ndarray, "4"] = np.asarray(fields[1:5], dtype=np.float64)
        cam_R_world: Float64[ndarray, "3 3"] = Rotation.from_quat(
            np.roll(quaternion_wxyz, -1)  # COLMAP stores wxyz; scipy wants xyzw
        ).as_matrix()
        cam_t_world: Float64[ndarray, "3"] = np.asarray(fields[5:8], dtype=np.float64)
        world_T_cam[" ".join(fields[9:])] = np.linalg.inv(compose(cam_R_world, cam_t_world))

    positions: list[list[float]] = []
    colours: list[list[int]] = []
    if points_path.is_file():
        for line in points_path.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 7:
                continue
            positions.append([float(value) for value in fields[1:4]])
            colours.append([int(value) for value in fields[4:7]])

    print(f"[colmap] {len(world_T_cam)} registered images, {len(positions)} points")
    return ColmapModel(
        world_T_cam=world_T_cam,
        points_xyz=np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        points_rgb=np.asarray(colours, dtype=np.uint8).reshape(-1, 3),
    )


def read_tum_poses(tum: Path) -> tuple[Int[ndarray, "n"], Float64[ndarray, "n 4 4"]] | None:
    """Parse a TUM trajectory (``t tx ty tz qx qy qz qw``) written by cuSFM or cuVSLAM.

    Returns ``None`` when the file is missing or empty, which is the normal case for
    stages that were skipped.

    cuSFM receives integer microseconds in ``frames_meta.json``, converts them to
    floating-point seconds, then prints many decimal places. Parse the text directly
    and recover that integer-microsecond contract: float64 cannot retain nanoseconds
    at Galileo's ~1.7e9-second epoch, while rounding the printed seconds to
    nanoseconds would preserve conversion noise instead of the input timestamp.
    """
    if not tum.is_file():
        return None
    rows: list[list[str]] = [line.split() for line in tum.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    times_ns: Int[ndarray, "n"] = np.asarray(
        [
            int((Decimal(row[0]) * Decimal(1_000_000)).to_integral_value(rounding=ROUND_HALF_UP))
            * 1000
            for row in rows
        ],
        dtype=np.int64,
    )
    raw: Float64[ndarray, "n 7"] = np.asarray(
        [[float(value) for value in row[1:8]] for row in rows], dtype=np.float64
    )
    poses: Float64[ndarray, "n 4 4"] = np.stack(
        [compose(Rotation.from_quat(row[3:7]).as_matrix(), row[0:3]) for row in raw]
    )
    return times_ns, poses


def write_tum_poses(tum: Path, times_ns: Int[ndarray, "n"], poses: Float[ndarray, "n 4 4"]) -> None:
    """Write a TUM trajectory, with timestamps printed the way cuSFM prints them.

    Nine decimal places on a value that is exact to the microsecond, so evo's
    timestamp association against cuSFM's own output is exact rather than nearest.
    """
    quaternions: Float64[ndarray, "n 4"] = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    lines: list[str] = [
        f"{1e-9 * float(timestamp):.9f} "
        + " ".join(f"{value:.9f}" for value in (*pose[:3, 3], *quaternion))
        for timestamp, pose, quaternion in zip(times_ns, poses, quaternions, strict=True)
    ]
    tum.write_text("\n".join(lines) + "\n")


def read_cusfm_vehicle_poses(sparse_dir: Path) -> tuple[Int[ndarray, "n"], Float64[ndarray, "n 4 4"]] | None:
    """Read cuSFM's own vehicle-frame (== rig-frame) trajectory, if it wrote one.

    ``extract_pose_from_map_main --export_pose_in_vehicle_frame=True`` emits
    ``output_poses/merged_pose_file.tum``. This is the correct source for
    ``world_T_rig`` once ``--run.optimize-extrinsics`` is on: deriving the rig pose
    from a camera pose and the *input* ``rig_T_cam`` would be wrong precisely
    because bundle adjustment has just changed that extrinsic. It is also the file
    ``evo_ape`` scores on KITTI, so the demo and evo read the same bytes.
    """
    return read_tum_poses(sparse_dir.parent / CUSFM_OUTPUT_POSES_DIR / CUSFM_MERGED_POSE_FILE)


def refined_extrinsics(
    sequence: PreparedSequence,
    model: ColmapModel,
    rig_times_ns: Int[ndarray, "n"],
    rig_poses: Float64[ndarray, "n 4 4"],
) -> tuple[dict[str, Float64[ndarray, "4 4"]], float]:
    """Recover each camera's ``rig_T_cam`` as bundle adjustment left it.

    ``rig_T_cam = world_T_rig(t)^-1 @ world_T_cam(t)``, averaged over every frame
    where both are known. Returns the per-camera extrinsics and the worst spread
    across frames — near zero means the rig stayed rigid, which the exoego schema
    requires.
    """
    lookup: dict[int, int] = {int(timestamp_ns): i for i, timestamp_ns in enumerate(rig_times_ns)}
    out: dict[str, Float64[ndarray, "4 4"]] = {}
    worst: float = 0.0
    matched_rows: int = 0
    for camera in sequence.sfm_cameras:
        estimates: list[Float64[ndarray, "4 4"]] = []
        for sample_index, timestamp in enumerate(sequence.timestamps_ns):
            image_path: Path | None = sequence.image_paths[camera.name][sample_index]
            if image_path is None:
                continue
            world_T_cam: Float64[ndarray, "4 4"] | None = model.world_T_cam.get(
                f"{camera.name}/{image_path.name}"
            )
            # Match the exact timestamp contract handed to cuSFM. RoboCap keeps
            # finer nanoseconds in its video names, but frames_meta.json carries
            # integer microseconds; Galileo already starts at that precision.
            timestamp_key_ns: int = (int(timestamp) // 1000) * 1000
            row: int | None = lookup.get(timestamp_key_ns)
            if world_T_cam is None or row is None:
                continue
            matched_rows += 1
            estimates.append(np.linalg.inv(rig_poses[row]) @ world_T_cam)
        if not estimates:
            continue
        stack: Float64[ndarray, "k 4 4"] = np.stack(estimates)
        translations: Float64[ndarray, "k 3"] = stack[:, :3, 3]
        worst = max(worst, float(np.linalg.norm(translations - translations.mean(axis=0), axis=1).max()))
        mean_rotation = Rotation.from_matrix(stack[:, :3, :3]).mean().as_matrix()
        out[camera.name] = compose(mean_rotation, translations.mean(axis=0))
    if matched_rows == 0:
        print(
            "  WARNING: refined-extrinsics timestamp join matched zero rows; "
            "no refined camera extrinsics can be recovered"
        )
    return out, worst


def cusfm_rig_trajectory(
    sequence: PreparedSequence, model: ColmapModel
) -> tuple[Int[ndarray, "m"], Float64[ndarray, "m 4 4"], float, dict[str, int]]:
    """Recover ``world_T_rig(t)`` from cuSFM's per-image camera poses.

    With ``ba_frame_type=vehicle_rig`` the rig stays rigid, so every camera in a
    sample yields the same ``world_T_rig = world_T_cam @ inv(rig_T_cam)``. The
    spread between those estimates is returned as a rigidity check — if it is
    not near zero, the rig was not treated as rigid and the exoego schema's
    single ``world_T_rig`` would be a lie.
    """
    by_name: dict[str, PreparedCamera] = {camera.name: camera for camera in sequence.cameras}
    per_camera_registered: dict[str, int] = {name: 0 for name in by_name}
    sample_indices: list[int] = []
    rig_poses: list[Float64[ndarray, "4 4"]] = []
    spreads: list[float] = []
    # Loop-invariant: four static extrinsics, otherwise re-inverted once per
    # (sample, camera) -- ~4500 redundant 4x4 inversions on a stride-4 run.
    cam_T_rig: dict[str, Float64[ndarray, "4 4"]] = {
        camera.name: np.linalg.inv(camera.rig_T_cam) for camera in sequence.sfm_cameras
    }

    for sample_index in range(len(sequence.timestamps_ns)):
        estimates: list[Float64[ndarray, "4 4"]] = []
        for camera in sequence.sfm_cameras:
            image_path: Path | None = sequence.image_paths[camera.name][sample_index]
            if image_path is None:
                continue
            key: str = f"{camera.name}/{image_path.name}"
            pose: Float64[ndarray, "4 4"] | None = model.world_T_cam.get(key)
            if pose is None:
                continue
            per_camera_registered[camera.name] += 1
            estimates.append(pose @ cam_T_rig[camera.name])
        if not estimates:
            continue
        translations: Float64[ndarray, "k 3"] = np.stack([estimate[:3, 3] for estimate in estimates])
        if len(estimates) > 1:
            spreads.append(float(np.linalg.norm(translations - translations.mean(axis=0), axis=1).max()))
        sample_indices.append(sample_index)
        rig_poses.append(estimates[0])

    max_spread: float = max(spreads) if spreads else 0.0
    return (
        np.asarray(sample_indices, dtype=np.int64),
        np.stack(rig_poses) if rig_poses else np.zeros((0, 4, 4)),
        max_spread,
        per_camera_registered,
    )


def clip_outliers(
    positions: Float[ndarray, "n 3"], percentile: float
) -> Bool[ndarray, "n"]:
    """Per-axis percentile band around the median; ``True`` marks points to keep."""
    if percentile >= 100.0 or len(positions) == 0:
        return np.ones(len(positions), dtype=bool)
    low: Float64[ndarray, "3"] = np.percentile(positions, 100.0 - percentile, axis=0)
    high: Float64[ndarray, "3"] = np.percentile(positions, percentile, axis=0)
    return np.all((positions >= low) & (positions <= high), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Rerun logging
# ─────────────────────────────────────────────────────────────────────────────


def build_rig(
    sequence: PreparedSequence,
    index: int,
    world_T_rig: Float64[ndarray, "n 4 4"],
    extrinsics_override: dict[str, Float64[ndarray, "4 4"]] | None = None,
) -> Rig:
    """Assemble an exoego ``Rig`` from prepared cameras plus a pose stream."""
    sensors: list[CameraSensor] = []
    for camera_index, camera in enumerate(sequence.cameras):
        intrinsics: Intrinsics = Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=camera.k_matrix,
            height=camera.height,
            width=camera.width,
        )
        # `Extrinsics`' "world" slot is the *parent* frame, which here is the rig.
        rig_T_cam: Float64[ndarray, "4 4"] = (
            extrinsics_override.get(camera.name, camera.rig_T_cam)
            if extrinsics_override
            else camera.rig_T_cam
        )
        extrinsics: Extrinsics = Extrinsics(
            world_R_cam=rig_T_cam[:3, :3], world_t_cam=rig_T_cam[:3, 3]
        )
        parameters: PinholeParameters | Fisheye62Parameters
        if camera.distortion:
            k1, k2, k3, k4 = (list(camera.distortion) + [0.0, 0.0, 0.0, 0.0])[:4]
            parameters = Fisheye62Parameters(
                name=camera.name,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                distortion=KannalaBrandtDistortion(k1=k1, k2=k2, k3=k3, k4=k4),
            )
        else:
            parameters = PinholeParameters(name=camera.name, extrinsics=extrinsics, intrinsics=intrinsics)
        sensors.append(
            CameraSensor(index=camera_index, name=camera.name, kind=camera.kind, pinhole=parameters)
        )

    return Rig(
        index=index,
        calibration=RigCalibration(cameras=sensors, reference_index=0),
        pose_stream=RigPoseStream(
            world_t_rig=world_T_rig[:, :3, 3].copy(),
            world_R_rig=world_T_rig[:, :3, :3].copy(),
        ),
        image_plane_distance=sequence.image_plane_distance,
    )


def log_rig(
    rig: Rig,
    sequence: PreparedSequence,
    timestamps_ns: Int[ndarray, "n"],
    *,
    label: str,
    frustum_color: tuple[int, int, int, int] | None,
) -> None:
    """Log a rig's static skeleton, its motion, and its provenance label."""
    log_rig_static(rig)
    log_rig_pose_stream(rig, timestamps_ns=timestamps_ns, timeline=TIMELINE)
    rig_path: str = f"world/{entity_id('rig', rig.index)}"
    # Deliberate COLMAP convention: cam_00 stays the camera reference (and gets
    # the viewer tint), while this override records the true IMU/vehicle origin.
    rr.log(
        rig_path,
        rr.AnyValues(source=label, dataset=sequence.name, reference=sequence.reference_name),
        static=True,
    )

    for camera_index, camera in enumerate(sequence.cameras):
        pinhole_path: str = f"{rig_path}/{entity_id('cam', camera_index)}/pinhole"
        if not camera.used_for_sfm:
            # Present on the rig but withheld from the SfM: grey it out so the
            # viewer shows the real hardware without implying it was used.
            rr.log(
                pinhole_path,
                rr.Pinhole.from_fields(
                    image_plane_distance=rig.image_plane_distance, color=EXCLUDED_CAM_COLOR
                ),
                static=True,
            )
        elif frustum_color is not None:
            rr.log(
                pinhole_path,
                rr.Pinhole.from_fields(
                    image_plane_distance=rig.image_plane_distance, color=frustum_color
                ),
                static=True,
            )


def log_images(sequence: PreparedSequence, rig_index: int, stride: int) -> int:
    """Log JPEG imagery under the input rig's cameras, strided for file size."""
    logged: int = 0
    for camera_index, camera in enumerate(sequence.cameras):
        if not camera.used_for_sfm:
            continue
        entity: str = f"world/{entity_id('rig', rig_index)}/{entity_id('cam', camera_index)}/pinhole/image"
        for sample_index in range(0, len(sequence.timestamps_ns), max(1, stride)):
            image_path: Path | None = sequence.image_paths[camera.name][sample_index]
            if image_path is None or not image_path.is_file():
                continue
            rr.set_time(TIMELINE, duration=1e-9 * float(sequence.timestamps_ns[sample_index]))
            rr.log(entity, rr.EncodedImage(path=image_path))
            logged += 1
    return logged


#: Endpoint colours, matching the RoboCap catalog's slam layer exactly.
TRAJECTORY_START_COLOR: tuple[int, int, int, int] = (0, 255, 0, 255)
TRAJECTORY_END_COLOR: tuple[int, int, int, int] = (255, 0, 0, 255)


def read_loop_closures(pose_graph_dir: Path) -> Int[ndarray, "m 2"] | None:
    """Extract the sample-index pairs that cuSFM's pose graph closed as loops.

    ``pose_graph.pb.txt`` is a flat list of ``pairs``, each a relative pose plus a
    ``type``. Only ``LOOP`` matters here; the run we ship also carries 4524
    ``CONSECUTIVE`` (odometry) and 13572 ``EXTRINSIC`` (inter-camera) pairs, both
    already implied by the trajectory and the rig, so drawing them would be noise.

    ``frame_id`` is a *global keyframe* index that strides by the number of cameras
    cuSFM was given, so dividing by that stride recovers the sample index the
    trajectory arrays use. The stride is measured from the ``CONSECUTIVE`` pairs
    rather than taken from the rig, because those two numbers differ: this rig
    logs six cameras to Rerun while cuSFM only ever received four, and using six
    silently scales every chord by 4/6 and lands it on the wrong pose.

    Loops are usually cross-camera -- frame 1168 -> 313 is sample 292 camera 0
    against sample 78 camera 1 -- but for *showing where the robot revisited a
    place* the rig position is the honest endpoint, so both ends collapse to their
    sample.

    Ids are read defensively: proto3 omits zero-valued scalars, so a pair against
    keyframe 0 has no ``source_frame_id`` line at all.
    """
    path: Path = pose_graph_dir / "pose_graph.pb.txt"
    if not path.is_file():
        return None
    loops: list[tuple[int, int]] = []
    strides: set[int] = set()
    source: int = 0
    target: int = 0
    depth: int = 0
    with path.open() as handle:
        for line in handle:
            stripped: str = line.strip()
            # Track real brace depth. A flat "am I inside frame_pair" flag breaks on
            # any nested submessage: the nested `}` clears it, so later `type:` lines
            # are attributed to whatever ids were last seen.
            if stripped.endswith("{"):
                depth += 1
                if depth == 2 and stripped.startswith("frame_pair"):
                    source, target = 0, 0
                continue
            if stripped == "}":
                depth -= 1
                continue
            if depth == 2 and stripped.startswith("source_frame_id:"):
                source = int(stripped.partition(":")[2])
            elif depth == 2 and stripped.startswith("target_frame_id:"):
                target = int(stripped.partition(":")[2])
            elif depth == 1 and stripped == "type: LOOP":
                loops.append((source, target))
            elif depth == 1 and stripped == "type: CONSECUTIVE":
                strides.add(abs(source - target))
    if not loops:
        return None
    # Every consecutive edge should span exactly one sample, i.e. the number of
    # cameras cuSFM was given. Disagreement means the keyframe ids are not a
    # uniform grid (a dropped image shifts them), and dividing by the minimum
    # would scale every chord. Refuse to guess rather than draw wrong chords.
    if len(strides) != 1 or 0 in strides:
        print(f"  WARNING: pose graph consecutive spans are not uniform ({sorted(strides)}); skipping loop chords")
        return None
    return np.asarray(loops, dtype=np.int64) // strides.pop()


def log_loop_closures(
    name: str,
    pairs: Int[ndarray, "m 2"],
    positions: Float[ndarray, "r 3"],
    sample_indices: Sequence[int],
) -> int:
    """Draw one chord per loop closure, straight across the trajectory.

    A loop closure is the one thing in the pose graph you cannot infer by looking
    at the path: it says "these two moments are the same place". Rendering each as
    a red segment between the two rig positions makes them literally visible as
    shortcuts across the walk, and the endpoint dots keep them findable when many
    chords overlap in a tight revisit.

    ``pairs`` holds *sample* indices while ``positions`` is indexed by *registered
    row*, and the two only coincide when every sample registered. ``sample_indices``
    is the row-to-sample map from ``cusfm_rig_trajectory``; it is required rather
    than optional because indexing one space with the other is silent -- the chords
    simply land on the wrong poses.

    Returns the number of chords drawn, which can be lower than ``len(pairs)`` when
    the pose graph references samples the reconstruction dropped.
    """
    row_of_sample: dict[int, int] = {sample: row for row, sample in enumerate(sample_indices)}
    rows: list[tuple[int, int]] = [
        (row_of_sample[a], row_of_sample[b])
        for a, b in pairs.tolist()
        if a in row_of_sample and b in row_of_sample
    ]
    if not rows:
        return 0
    kept: Int[ndarray, "k 2"] = np.asarray(rows, dtype=np.int64)
    segments: Float64[ndarray, "k 2 3"] = np.stack(
        [positions[kept[:, 0]], positions[kept[:, 1]]], axis=1
    )
    rr.log(
        f"world/runs/{name}/loop_closures",
        rr.LineStrips3D(segments, colors=LOOP_CLOSURE_COLOR, radii=LOOP_CLOSURE_RADIUS),
        static=True,
    )
    rr.log(
        f"world/runs/{name}/loop_closures/endpoints",
        rr.Points3D(
            positions=segments.reshape(-1, 3),
            colors=LOOP_CLOSURE_COLOR,
            radii=LOOP_CLOSURE_RADIUS * 2.0,
        ),
        static=True,
    )
    return len(kept)


def log_trajectory(
    name: str,
    positions: Float[ndarray, "n 3"],
    *,
    source: str,
    hue: tuple[int, int, int],
    keyframe_positions: Float[ndarray, "k 3"] | None = None,
) -> None:
    """Log a trajectory polyline the way the RoboCap catalog's slam layer does.

    Entity layout mirrors ``/world/runs/basalt`` in the source recording:
    ``runs/<name>`` carries provenance, ``runs/<name>/trajectory`` is a *static*
    ``LineStrips3D`` of ``n-1`` two-point segments (per-segment colour, so travel
    direction is readable), and ``runs/<name>/endpoints`` marks start and end.
    """
    if len(positions) < 2:
        return
    segments: Float64[ndarray, "n-1 2 3"] = np.stack([positions[:-1], positions[1:]], axis=1)
    # Fade from dark to full hue along the path, so direction of travel is visible.
    ramp: Float64[ndarray, "n-1"] = np.linspace(0.35, 1.0, len(segments))
    colors: Int[ndarray, "n-1 4"] = np.stack(
        [
            (ramp * hue[0]).astype(np.uint8),
            (ramp * hue[1]).astype(np.uint8),
            (ramp * hue[2]).astype(np.uint8),
            np.full(len(segments), 255, dtype=np.uint8),
        ],
        axis=-1,
    )
    rr.log(f"world/runs/{name}", rr.AnyValues(num_poses=len(positions), source=source), static=True)
    rr.log(
        f"world/runs/{name}/trajectory",
        rr.LineStrips3D(segments, colors=colors, radii=0.004),
        static=True,
    )
    rr.log(
        f"world/runs/{name}/endpoints",
        rr.Points3D(
            positions=np.stack([positions[0], positions[-1]]),
            colors=[TRAJECTORY_START_COLOR, TRAJECTORY_END_COLOR],
            labels=["start", "end"],
            radii=0.02,
        ),
        static=True,
    )
    if keyframe_positions is not None and len(keyframe_positions):
        # One dot per frame cuSFM was actually given. Against the continuous 30 fps
        # line this shows the *spatial* sampling pattern at a glance — dense where
        # the wearer paused, sparse through fast head motion — without scrubbing.
        rr.log(
            f"world/runs/{name}/keyframes",
            rr.Points3D(positions=keyframe_positions, colors=KEYFRAME_COLOR, radii=0.012),
            static=True,
        )


LOOP_CLOSURE_COLOR: tuple[int, int, int, int] = (235, 60, 70, 255)
"""Red, deliberately clashing with both trajectory hues (blue input, orange cuSFM)
so a chord never reads as part of the path it cuts across."""
LOOP_CLOSURE_RADIUS: float = 0.008
"""Twice the trajectory's 0.004 so the chords stay legible where they bundle."""

#: Frustum styling for frames cuSFM actually consumed, vs the ones it never saw.
KEYFRAME_COLOR: tuple[int, int, int, int] = (80, 230, 120, 255)
NON_KEYFRAME_COLOR: tuple[int, int, int, int] = (120, 124, 140, 110)
#: Frustum edge thickness as a FRACTION of image_plane_distance. `line_width` is
#: in **scene units**, not pixels: Rerun's own Pinhole example pairs
#: `image_plane_distance=0.1` with `line_width=0.003`. Absolute values here (0.6,
#: 2.5) drew edges metres thick and the frustum became an unreadable blob.
KEYFRAME_WIDTH_FRACTION: float = 0.040
NON_KEYFRAME_WIDTH_FRACTION: float = 0.012


def log_video_streams(sequence: PreparedSequence, rig_index: int) -> int:
    """Re-emit each camera's original H.264 samples as a ``VideoStream``.

    The samples are passed through verbatim — no re-encode — so the viewer shows
    the full 30 fps walk, not the keyframe slideshow. Far cheaper than JPEGs too:
    the whole segment is ~77 MB per camera against ~280 MB of per-sample JPEG.
    """
    total: int = 0
    for camera_index, camera in enumerate(sequence.cameras):
        samples: list[tuple[int, bytes]] = sequence.video_samples.get(camera.name, [])
        if not samples:
            continue
        entity: str = f"world/{entity_id('rig', rig_index)}/{entity_id('cam', camera_index)}/pinhole/video"
        rr.log(entity, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
        timestamps: Int[ndarray, "n"] = np.asarray([t for t, _ in samples], dtype=np.int64)
        rr.send_columns(
            entity,
            indexes=[rr.TimeColumn(TIMELINE, duration=1e-9 * timestamps.astype(np.float64))],
            columns=rr.VideoStream.columns(sample=[raw for _, raw in samples]),
        )
        total += len(samples)
    return total


def log_video_references(sequence: PreparedSequence, source_rig: int, target_rig: int) -> int:
    """Show ``source_rig``'s video under ``target_rig``'s cameras without copying it.

    Rerun 0.37.1 lets a ``VideoFrameReference`` point at a ``VideoStream`` on
    another entity, so the cuSFM rig gets the same imagery as the input rig for
    a few hundred KB of timestamps instead of another ~300 MB of H.264. The
    frames are geometrically valid on both rigs because nothing is rectified:
    both log the raw fisheye under a ``Pinhole`` with distortion coefficients.

    One reference per source sample, on the same timeline as the stream. Measured
    on 0.37.1: a ``VideoFrameReference`` into a stream shows the frame at the
    reference's *own* ``timestamp``, not at the viewer cursor -- a single static
    reference therefore pins one frame forever (and ``0`` clamps to the first
    sample). Per-sample references whose timestamps equal the stream's are what
    make the reference follow the cursor.

    Returns the number of references logged.
    """
    total: int = 0
    for camera_index, camera in enumerate(sequence.cameras):
        samples: list[tuple[int, bytes]] = sequence.video_samples.get(camera.name, [])
        if not samples:
            continue
        source: str = f"world/{entity_id('rig', source_rig)}/{entity_id('cam', camera_index)}/pinhole/video"
        target: str = f"world/{entity_id('rig', target_rig)}/{entity_id('cam', camera_index)}/pinhole/video"
        timestamps: Int[ndarray, "n"] = np.asarray([t for t, _ in samples], dtype=np.int64)
        rr.log(target, rr.VideoFrameReference.from_fields(video_reference=source), static=True)
        rr.send_columns(
            target,
            indexes=[rr.TimeColumn(TIMELINE, duration=1e-9 * timestamps.astype(np.float64))],
            columns=rr.VideoFrameReference.columns(timestamp=timestamps),
        )
        total += len(samples)
    return total


def log_keyframe_highlight(sequence: PreparedSequence, rig_index: int) -> None:
    """Tint each camera's frustum by whether the current frame is a cuSFM keyframe.

    A temporal partial update of ``Pinhole``'s ``color``/``line_width`` — the static
    intrinsics logged by ``log_pinhole`` stay untouched. Green and thick on the
    frames handed to the solver, dim and thin on the frames it never saw, so
    scrubbing shows exactly what cuSFM was given.
    """
    times: Int[ndarray, "n"] | None = sequence.full_timestamps_ns
    mask: Bool[ndarray, "n"] | None = sequence.keyframe_mask
    if times is None or mask is None:
        return
    colors: Int[ndarray, "n 4"] = np.where(
        mask[:, None], np.asarray(KEYFRAME_COLOR), np.asarray(NON_KEYFRAME_COLOR)
    ).astype(np.uint8)
    scale: float = sequence.image_plane_distance
    widths: Float64[ndarray, "n"] = np.where(
        mask, KEYFRAME_WIDTH_FRACTION * scale, NON_KEYFRAME_WIDTH_FRACTION * scale
    )
    for camera_index, camera in enumerate(sequence.cameras):
        if not camera.used_for_sfm:
            continue
        rr.send_columns(
            f"world/{entity_id('rig', rig_index)}/{entity_id('cam', camera_index)}/pinhole",
            indexes=[rr.TimeColumn(TIMELINE, duration=1e-9 * times.astype(np.float64))],
            columns=rr.Pinhole.columns(color=colors, line_width=widths.astype(np.float32)),
        )


def build_blueprint(sequence: PreparedSequence) -> rrb.Blueprint:
    """3D view of both rigs plus the cloud, with per-camera 2D views alongside."""
    sfm_cameras: list[tuple[int, PreparedCamera]] = [
        (index, camera) for index, camera in enumerate(sequence.cameras) if camera.used_for_sfm
    ]
    image_views: list[rrb.SpatialView2DView] = [
        rrb.Spatial2DView(
            origin=f"world/{entity_id('rig', INPUT_RIG_INDEX)}/{entity_id('cam', index)}/pinhole",
            name=camera.name,
        )
        for index, camera in sfm_cameras
    ]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                name="cuSFM reconstruction (rig_00 = input, rig_01 = cuSFM)",
            ),
            rrb.Grid(contents=image_views, name="cameras"),
            column_shares=[2.0, 1.0],
        ),
        rrb.TimePanel(state="expanded"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main(config: Config) -> None:
    """Prepare, reconstruct, align, and log."""
    dataset: DatasetConfig = config.dataset
    run: RunConfig = config.run
    sequence: PreparedSequence
    if isinstance(dataset, RobocapConfig):
        sequence = prepare_robocap(dataset, run)
    elif isinstance(dataset, KittiConfig):
        run = kitti_run_config(run)
        sequence = prepare_kitti(dataset, run)
    else:
        sequence = prepare_galileo(dataset, run)

    sparse_dir: Path = run_cusfm(sequence, run)
    if isinstance(dataset, KittiConfig):
        # Only now does an input trajectory exist: cuVSLAM produced it inside the run.
        sequence = attach_cuvslam_trajectory(sequence, sparse_dir.parent, run.use_cuvslam_slam_pose)
    model: ColmapModel = read_colmap_model(sparse_dir)
    sample_indices, cusfm_world_T_rig, rigidity_spread, per_camera = cusfm_rig_trajectory(sequence, model)

    print("\n─── cuSFM result ───")
    for name, count in per_camera.items():
        total: int = sum(1 for path in sequence.image_paths[name] if path is not None)
        if total:
            print(f"  {name:12s} {count:4d}/{total:4d} images registered")
    print(f"  rig rigidity spread: {1e3 * rigidity_spread:.2f} mm (should be ~0 with ba_frame_type=vehicle_rig)")

    if len(sample_indices) < 3:
        raise RuntimeError(f"only {len(sample_indices)} samples registered — reconstruction failed")

    input_world_T_rig: Float64[ndarray, "m 4 4"] = sequence.world_T_rig[sample_indices]
    rotation, translation, rmse, would_be_scale = umeyama_rigid(
        cusfm_world_T_rig[:, :3, 3], input_world_T_rig[:, :3, 3]
    )
    alignment: Float64[ndarray, "4 4"] = compose(rotation, translation)
    aligned_world_T_rig: Float64[ndarray, "m 4 4"] = alignment @ cusfm_world_T_rig
    travelled: float = trajectory_length(input_world_T_rig[:, :3, 3])
    print(f"  aligned {len(sample_indices)} rig poses to the input trajectory (scale held at 1.0)")
    # Not "ATE": the input trajectory is itself an estimate unless the dataset
    # ships ground truth (galileo does; RoboCap's basalt VIO does not).
    print(
        f"  disagreement RMSE : {1e3 * rmse:.1f} mm over {travelled:.2f} m travelled "
        f"(cuSFM vs input estimate, NOT ground truth)"
    )
    print(f"  would-be scale    : {would_be_scale:.5f} (1.0 = cuSFM preserved metric scale)")
    print(f"  sparse points     : {len(model.points_xyz)}")

    # ── log ──────────────────────────────────────────────────────────────────
    rr.log("/", rr.ViewCoordinates.RFU, static=True)
    rr.send_blueprint(build_blueprint(sequence))

    # The input rig plays at the source frame rate when the dataset provides it,
    # so the walk is continuous rather than a keyframe slideshow. cuSFM still only
    # ever received the keyframes; `log_keyframe_highlight` marks which those are.
    has_full: bool = sequence.full_timestamps_ns is not None and sequence.full_world_T_rig is not None
    input_times: Int[ndarray, "n"] = sequence.full_timestamps_ns if has_full else sequence.timestamps_ns
    input_poses: Float64[ndarray, "n 4 4"] = (
        sequence.full_world_T_rig if has_full else sequence.world_T_rig
    )
    input_rig: Rig = build_rig(sequence, INPUT_RIG_INDEX, input_poses)
    log_rig(
        input_rig,
        sequence,
        input_times,
        label=f"input trajectory ({sequence.input_source})",
        frustum_color=None,
    )

    # With --run.optimize-extrinsics, bundle adjustment changes rig_T_cam, so the
    # rig pose must come from cuSFM's own vehicle-frame output rather than being
    # derived from the (now stale) input extrinsics.
    refined: dict[str, Float64[ndarray, "4 4"]] | None = None
    cusfm_pose_times_ns: Int[ndarray, "n"] = sequence.timestamps_ns[sample_indices]
    cusfm_pose_stream: Float64[ndarray, "n 4 4"] = aligned_world_T_rig
    vehicle = read_cusfm_vehicle_poses(sparse_dir)
    if vehicle is not None and run.optimize_extrinsics:
        vehicle_times, vehicle_poses = vehicle
        refined, extrinsic_spread = refined_extrinsics(sequence, model, vehicle_times, vehicle_poses)
        cusfm_pose_times_ns = vehicle_times
        cusfm_pose_stream = alignment @ vehicle_poses
        print("\n─── extrinsic refinement ───")
        for name, matrix in refined.items():
            original: Float64[ndarray, "4 4"] = next(
                c.rig_T_cam for c in sequence.cameras if c.name == name
            )
            d_t: float = float(np.linalg.norm(matrix[:3, 3] - original[:3, 3]))
            d_r: float = float(
                np.degrees(
                    np.linalg.norm(Rotation.from_matrix(matrix[:3, :3] @ original[:3, :3].T).as_rotvec())
                )
            )
            print(f"  {name:12s} moved {1e3 * d_t:6.2f} mm, {d_r:5.2f} deg vs the factory calibration")
        print(f"  per-frame spread: {1e3 * extrinsic_spread:.2f} mm (near 0 = rig stayed rigid)")

    cusfm_rig: Rig = build_rig(
        sequence, CUSFM_RIG_INDEX, cusfm_pose_stream, extrinsics_override=refined
    )
    log_rig(
        cusfm_rig,
        sequence,
        cusfm_pose_times_ns,
        label=f"cuSFM refined (disagreement vs input {1e3 * rmse:.1f} mm RMSE)",
        frustum_color=CUSFM_RIG_COLOR,
    )

    aligned_points: Float64[ndarray, "n_points 3"] = (
        model.points_xyz @ rotation.T + translation if len(model.points_xyz) else model.points_xyz
    )
    keep: Bool[ndarray, "n_points"] = clip_outliers(aligned_points, run.point_clip_percentile)
    dropped: int = int((~keep).sum())
    if dropped:
        print(f"  clipped {dropped} outlier points ({100 * dropped / len(keep):.1f}%) for display only")
    rr.log(
        "world/points",
        rr.Points3D(positions=aligned_points[keep], colors=model.points_rgb[keep], radii=0.01),
        static=True,
    )

    log_trajectory(
        "input",
        input_poses[:, :3, 3],
        source=f"{sequence.input_source} ({len(input_times)} poses)",
        hue=(120, 200, 255),
        keyframe_positions=sequence.world_T_rig[:, :3, 3],
    )
    log_trajectory(
        "cusfm",
        aligned_world_T_rig[:, :3, 3],
        source=f"cuSFM global BA ({len(model.points_xyz)} points)",
        hue=(255, 150, 40),
    )

    if isinstance(dataset, KittiConfig):
        report_kitti_ground_truth(dataset, sequence, sparse_dir.parent)

    loops: Int[ndarray, "m 2"] | None = read_loop_closures(sparse_dir.parent / CUSFM_POSE_GRAPH_DIR)
    if loops is not None:
        drawn: int = log_loop_closures(
            "cusfm", loops, aligned_world_T_rig[:, :3, 3], sample_indices
        )
        spans: Int[ndarray, "m"] = np.abs(loops[:, 0] - loops[:, 1])
        print(
            f"  logged {drawn} loop closures (spanning {spans.min()}-{spans.max()} samples, "
            f"median {int(np.median(spans))})"
        )
        if drawn < len(loops):
            # Almost always a stride mismatch: the pose graph indexes the samples
            # cuSFM was given, so re-logging at a different --dataset.frame-stride
            # than the run used puts most loops past the end of the trajectory.
            print(
                f"  WARNING: dropped {len(loops) - drawn} of {len(loops)} loop closures - "
                f"their samples are outside the {len(sample_indices)} registered here. "
                f"Usually --dataset.frame-stride does not match the stride this run was "
                f"reconstructed at."
            )
    else:
        # Worth saying out loud: the shipped configs return zero loops, so a silent
        # absence here reads as "no revisits" when it actually means the gates are
        # rejecting them. See LOOP_CLOSURE_CONFIG_DIR.
        print("  no loop closures in the pose graph")

    if sequence.video_samples:
        frames: int = log_video_streams(sequence, INPUT_RIG_INDEX)
        references: int = log_video_references(sequence, INPUT_RIG_INDEX, CUSFM_RIG_INDEX)
        log_keyframe_highlight(sequence, INPUT_RIG_INDEX)
        print(
            f"  logged {frames} video frames across {len(sequence.video_samples)} cameras "
            f"(full rate); {int(sequence.keyframe_mask.sum())} marked as cuSFM keyframes; "
            f"{references} frame references give rig_01 the same imagery without copying it"
        )
    else:
        logged_images: int = log_images(sequence, INPUT_RIG_INDEX, dataset.image_stride)
        print(f"  logged {logged_images} images (every {dataset.image_stride} samples)")
    print()


if __name__ == "__main__":
    main(tyro.cli(Config))

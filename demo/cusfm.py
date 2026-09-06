"""Running cuSFM, and where it puts what it writes.

One function invokes the pipeline; everything else here is the workspace layout
it produces, re-exported from `pycusfm.constants` rather than hardcoded, and the
config directory the demo hands it. Reading those names out of upstream matters:
a renamed directory would otherwise degrade silently — a missing `pose_graph/`
reads as "no loop closures", a missing `output_poses/` as "no refined
extrinsics". Reading frozen upstream is allowed; only editing it is not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pycusfm.constants as _cusfm_constants
from demo.sequence import PreparedSequence

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

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
"""Repository root, so the frozen `pycusfm/` tree and the config copies resolve from any cwd."""

LOOP_CLOSURE_CONFIG_DIR: Path = REPO_ROOT / "data" / "cusfm_configs" / "loop-closure-fixed"
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



def run_cusfm(sequence: PreparedSequence, run: RunConfig) -> Path:
    """Run the cuSFM pipeline and return the COLMAP ``sparse/`` directory.

    cuVSLAM is skipped: both datasets already carry a trajectory, which is fed
    in through ``frames_meta.json``'s ``camera_to_world`` under
    ``initial_pose_type: EGO_MOTION``. cuSFM's global bundle adjustment then
    refines it.
    """
    from pycusfm.cusfm_runner import create_cusfm_runner

    base_dir: Path = run.work_dir / f"{sequence.name}{run.variant}" / "cusfm"
    sparse_dir: Path = base_dir / CUSFM_SPARSE_DIR
    if run.skip_reconstruction:
        if not (sparse_dir / "images.txt").is_file() and not (sparse_dir / "images.bin").is_file():
            raise FileNotFoundError(f"--run.skip-reconstruction set but no model at {sparse_dir}")
        print(f"[cusfm] reusing existing model at {sparse_dir}")
        return sparse_dir

    base_dir.mkdir(parents=True, exist_ok=True)
    package_path: Path = REPO_ROOT / "pycusfm"
    print(
        f"[cusfm] {sequence.name}{run.variant}: feature_type={run.feature_type} "
        f"ba_frame_type={run.ba_frame_type} skip_cuvslam={run.skip_cuvslam} "
        f"skip_pose_graph={run.skip_pose_graph}"
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

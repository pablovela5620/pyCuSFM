"""Stock COLMAP, rig-aware and GPU-solved, on the RoboCap fisheye sequence.

`tools.audit.colmap_rig_baseline_galileo` answers the same question on Galileo:
what does *standard* COLMAP do once it is told about the rig? RoboCap is the
harder half of that question — 4528 keyframes, 1132 rig frames, four
`OPENCV_FISHEYE` cameras and a walk that revisits itself — and it is the
sequence every `colsfm` number is quoted on, so the comparison needs a COLMAP
column measured with the same harness.

Everything here is COLMAP's own code. The four choices that make it a fair
opponent rather than a straw man:

1. **The calibrated rig.** `pycolmap.apply_rig_config` writes one rig and one
   frame per capture into the database, so the stock sequential matcher expands
   pairs across the rig (`expand_rig_images`, on by default) and the stock
   incremental mapper registers whole frames. `cam_from_rig` is
   `cam_i_T_vehicle * vehicle_T_cam_ref`, exactly what
   `colsfm.reconstruction.build_rig` writes.
2. **COLMAP's own loop detection.** `SequentialPairingOptions.loop_detection`
   with `vocab_tree_path` left empty, which makes COLMAP fetch the vocabulary
   tree its own `retrieval/resources.h` names for the feature type — for
   `ALIKED_N16ROT` that is
   `vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin` from the 3.13.0
   release, cached under `~/.cache/colmap/<sha256>-<name>`. **It is downloaded
   once**; a run whose cache already holds it needs no network. There is no
   offline vocabulary in the conda package, so the first run of this script on
   a fresh machine does need the network for that one file.
3. **COLMAP's bundled learned features.** `ALIKED_N16ROT` + `ALIKED_LIGHTGLUE`
   through pycolmap's own ONNX runtime, not colsfm's TensorRT or RaCo engines.
4. **CASPAR for the mapper's local bundle adjustment**, Ceres for the global
   one — `ba_local_backend = CASPAR`, `ba_global_backend` left at COLMAP's
   default. The `colsfm-caspar-fisheye` environment is the only build whose
   CASPAR carries an `OPENCV_FISHEYE` adapter (`docs/caspar-fisheye-adapter.md`);
   in any other build CASPAR silently *skips* every fisheye observation, which
   is why the run prints `caspar_supported_camera_models()` before it starts and
   greps its own COLMAP log for the skip warning afterwards.

**The one option CASPAR forces off.** `ba_refine_sensor_from_rig` defaults to
True, and CASPAR throws on a multi-sensor frame when it is
(`bundle_adjustment_caspar.cc:186`). The run therefore holds the extrinsics at
the calibration whenever the backend is CASPAR, and says so in the report; with
`--ba-backend ceres` the COLMAP default stands and the recovered extrinsics are
measured against the calibration the way the Galileo baseline measures them.

The output is a cuSFM-layout run — `cusfm/sparse/`, `cusfm/kpmap/keyframes/`,
`cusfm/output_poses/`, `cusfm/runtime.csv` — so `colsfm.bench_cli` compares it
against the blob reference and against a colsfm run directly:

    pixi run -e colsfm-caspar-fisheye python -m tools.audit.colmap_rig_baseline_robocap
    pixi run -e colsfm python -m colsfm.bench_cli --dataset robocap \\
        --run-a data/cusfm_runs/robocap_blobref/cusfm \\
        --run-b data/cusfm_runs/audit_colmap_rig_robocap/cusfm \\
        --input-dir data/cusfm_runs/robocap_blobref/input \\
        --name-a blob --name-b colmap-rig-caspar
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pycolmap
import tyro
from jaxtyping import Int64
from numpy import ndarray
from serde import serde

from colsfm.ba_backend import caspar_supported_camera_models
from colsfm.benchmark import ReconstructionMetrics, write_json_report
from colsfm.cameras import COLMAP_MODEL_BY_PROJECTION_MODEL, colmap_camera_parameters
from colsfm.export import KEYFRAME_METADATA_SUBPATH, colour_points_from_images, write_colmap_model, write_optimised_frames_meta, write_pose_files
from colsfm.frames_meta import FRAMES_META_NAME, CameraParams, FramesMeta, KeyframeMeta, read_frames_meta
from colsfm.reconstruction import RigReference, rig_reference
from tools.audit.colmap_baseline_galileo import (
    EXTRACTOR_BY_CHOICE,
    MATCHER_BY_CHOICE,
    FeatureChoice,
    ModelSummary,
    StageClock,
    _summarise,
)
from tools.audit.colmap_rig_baseline_galileo import ExtrinsicDelta

STAGED_IMAGE_DIR_NAME: str = "images"
"""Symlink tree written inside the run directory; see `stage_images`."""

CUSFM_DIR_NAME: str = "cusfm"
"""Subdirectory of the run holding the cuSFM-layout artifacts `colsfm.bench_cli` reads."""

COLMAP_LOG_NAME: str = "colmap.log"
"""COLMAP's own glog output, captured by redirecting the process's stderr."""

MILLIMETRES_PER_METRE: float = 1000.0
"""Extrinsic deltas are reported in millimetres, like every other length here."""

CASPAR_UNSUPPORTED_MODEL_MARKER: str = "unsupported camera model"
"""`bundle_adjustment_caspar.cc` logs this when it drops an observation whose camera
model the build has no adapter for. Zero of these is what proves the fisheye
observations went into the solve rather than round it."""

CASPAR_RAN_MARKER: str = "Caspar factor distribution"
"""Header of `LogFactorDistribution`, at VLOG(1): one line per CASPAR solve, and the
only unambiguous proof in the log that the GPU backend — not Ceres — solved."""

CASPAR_FISHEYE_MARKER: str = "model=5 variant="
"""A factor-distribution line for `CameraModelId::kOpenCVFisheye`, which is 5. CASPAR
groups its factors by camera model, so this is the line that says the fisheye
observations were built into kernels rather than dropped — the number to check against
`caspar_unsupported_model_warnings`."""

BaBackendChoice: TypeAlias = Literal["caspar", "ceres"]
"""Which backend solves the mapper's *local* bundle adjustment."""

BACKEND_BY_CHOICE: dict[BaBackendChoice, pycolmap.BundleAdjustmentBackend] = {
    "caspar": pycolmap.BundleAdjustmentBackend.CASPAR,
    "ceres": pycolmap.BundleAdjustmentBackend.CERES,
}
"""The pycolmap enum behind each choice."""


@dataclass(frozen=True)
class ColmapRigBaselineRobocapConfig:
    """One rig-enabled, loop-closing, GPU-solved COLMAP run on a cuSFM dataset."""

    input_dir: Path = Path("data/cusfm_runs/robocap_blobref/input")
    """Dataset directory holding `frames_meta.json` and the four camera folders."""
    output_root: Path = Path("data/cusfm_runs")
    """Runs are written to `<output_root>/<run_name>/`."""
    run_name: str = "audit_colmap_rig_robocap"
    """Directory name of the run, and the stem of its JSON report."""
    bench_dir: Path = Path("data/bench")
    """Where the JSON summary is written."""
    features: FeatureChoice = "aliked"
    """`aliked` is COLMAP's own ALIKED-n16rot + LightGlue ONNX stack; `sift` is the classic one."""
    overlap: int = 10
    """`SequentialPairingOptions.overlap`, left at COLMAP's default value."""
    loop_detection: bool = True
    """Turn on COLMAP's own vocabulary-tree loop detection inside the sequential matcher."""
    vocab_tree_path: Path | None = None
    """Vocabulary tree to load, or None to let COLMAP pick (and cache) the one for the feature type."""
    ba_backend: BaBackendChoice = "caspar"
    """Backend for the mapper's local bundle adjustment; the global one stays at COLMAP's default."""
    ba_global_caspar: bool = False
    """Also put the *global* bundle adjustment on CASPAR, instead of COLMAP's default Ceres."""
    max_num_rig_frames: int | None = None
    """Keep only the first N rig frames — the short slice used to validate the pipeline."""
    use_gpu: bool = True
    """Run extraction and matching on the GPU."""
    verbose_level: int = 1
    """COLMAP's VLOG level; 1 is what makes CASPAR announce itself in the log."""
    overwrite: bool = True
    """Delete any previous run directory first, so timings are cold-cache-fair."""


@serde
@dataclass(frozen=True)
class ColmapRigBaselineRobocapResult:
    """Everything one run measured."""

    run_dir: str
    """Where the artifacts were written."""
    cusfm_dir: str
    """The cuSFM-layout directory `colsfm.bench_cli` reads."""
    features: str
    """Extractor choice."""
    ba_local_backend: str
    """Backend that solved the mapper's local bundle adjustments."""
    ba_global_backend: str
    """Backend that solved the mapper's global bundle adjustments."""
    ba_refine_sensor_from_rig: bool
    """Whether the mapper was allowed to move the rig extrinsics."""
    caspar_supported_camera_models: tuple[str, ...]
    """Camera models this build's CASPAR carries an adapter for, measured by a probe solve."""
    num_caspar_solves: int
    """CASPAR solves COLMAP's log recorded, one `Caspar factor distribution` header each."""
    num_caspar_fisheye_factor_blocks: int
    """Factor-distribution lines built for the `OPENCV_FISHEYE` model, across those solves."""
    caspar_unsupported_model_warnings: int
    """`Skipping image ... with unsupported camera model` lines; 0 is the number that matters."""
    camera_model: str
    """COLMAP camera model every image was extracted with."""
    reference_camera_params_id: int
    """Camera the rig origin sits on: the camera of the lowest keyframe id."""
    total_images: int
    """Keyframes handed to COLMAP."""
    total_rig_frames: int
    """Captures the staged tree grouped the keyframes into."""
    loop_detection: bool
    """Whether COLMAP's vocabulary-tree loop detection ran inside the sequential matcher."""
    vocab_tree: str
    """The vocabulary tree used, or `default-for-feature-type` when COLMAP chose it."""
    num_sequential_pairs_proposed: int
    """Pairs a loop-detection-free sequential pass over the same database would propose."""
    num_image_pairs_matched: int
    """Pairs the matcher wrote any match for."""
    num_verified_image_pairs: int
    """Pairs with a verified two-view geometry."""
    num_verified_loop_pairs: int
    """Verified pairs the sequential pass would never have proposed, i.e. loop detection's."""
    num_models: int
    """How many disconnected reconstructions COLMAP returned."""
    models: tuple[ModelSummary, ...]
    """Every model, largest first."""
    stage_timings: tuple[tuple[str, float], ...]
    """Per-stage wall clock, in pipeline order."""
    total_runtime_seconds: float
    """Sum of the stage timings."""
    largest_model_fraction_registered: float
    """Registered images of the largest model over `total_images`."""
    num_rig_frames_registered: int
    """Rig frames the largest model gave a pose to."""
    extrinsic_deltas: tuple[ExtrinsicDelta, ...]
    """Recovered rig extrinsics against the calibration, per non-reference camera."""


@dataclass(frozen=True, slots=True)
class StagedImages:
    """The symlink tree the rig is grouped by, and the two maps back to the dataset."""

    image_name_by_staged: dict[str, str]
    """Staged image name to the dataset's own `image_name`."""
    keyframe_id_by_staged: dict[str, int]
    """Staged image name to the `keyframe_id` it came from."""
    num_rig_frames: int
    """Distinct captures the staged names group into."""


@contextmanager
def captured_stderr(path: Path) -> Iterator[None]:
    """Send this process's file descriptor 2 to a file for the duration of the block.

    COLMAP logs through glog from C++, so a Python-level redirect does not catch
    it; only `dup2` on the real descriptor does. The log is the run's evidence
    that CASPAR solved the fisheye problem rather than skipping it.

    Args:
        path: File to write; parents are created and any previous file truncated.

    Yields:
        Nothing; the block runs with stderr redirected.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stderr.flush()
    saved: int = os.dup(2)
    with path.open("w") as handle:
        os.dup2(handle.fileno(), 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)


def reference_camera_params_id(frames_meta: FramesMeta) -> int:
    """Pick the rig's reference camera: the one owning the lowest keyframe id.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        The reference `camera_params_id`.
    """
    return min(frames_meta.keyframes, key=lambda item: item.keyframe_id).camera_params_id


def stage_images(frames_meta: FramesMeta, input_dir: Path, staged_root: Path) -> StagedImages:
    """Build the symlink tree whose file names group into rig frames.

    COLMAP groups images into frames by the part of the path that follows
    `RigConfigCamera.image_prefix`, and RoboCap's per-camera file names are
    per-camera timestamps that differ inside one capture, so the raw tree would
    give 4528 one-image frames and the rig would do nothing. The staged name is
    `<camera_folder>/<synced_sample_id zero-padded>.jpeg`: identical across
    cameras, and **zero-padded** because both `apply_rig_config` and the
    sequential matcher order images by name, under which `10` sorts before `2`.

    Args:
        frames_meta: The parsed metadata.
        input_dir: Dataset directory the symlinks point into.
        staged_root: Directory to write the tree into; created.

    Returns:
        The tree's two maps back to the dataset and its rig-frame count.
    """
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    width: int = len(str(max(rig_frame.synced_sample_id for rig_frame in frames_meta.rig_frames())))
    image_name_by_staged: dict[str, str] = {}
    keyframe_id_by_staged: dict[str, int] = {}
    for rig_frame in frames_meta.rig_frames():
        for keyframe_id in rig_frame.keyframe_ids:
            keyframe: KeyframeMeta = keyframe_by_id[keyframe_id]
            folder: str = keyframe.image_name.split("/")[0]
            staged_name: str = f"{folder}/{rig_frame.synced_sample_id:0{width}d}{Path(keyframe.image_name).suffix}"
            link: Path = staged_root / staged_name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to((input_dir / keyframe.image_name).resolve())
            image_name_by_staged[staged_name] = keyframe.image_name
            keyframe_id_by_staged[staged_name] = keyframe_id
    return StagedImages(
        image_name_by_staged=image_name_by_staged,
        keyframe_id_by_staged=keyframe_id_by_staged,
        num_rig_frames=len(frames_meta.rig_frames()),
    )


def extract(config: ColmapRigBaselineRobocapConfig, frames_meta: FramesMeta, staged_root: Path, database_path: Path) -> str:
    """Extract features from the staged tree, one COLMAP camera per folder.

    Each folder gets its own camera carrying the dataset's *known* intrinsics —
    the same `OPENCV_FISHEYE` parameters `colsfm.cameras` hands the colsfm
    pipeline — so COLMAP is neither asked to guess a fisheye calibration nor
    given less than cuSFM gets.

    Args:
        config: The run's configuration.
        frames_meta: The parsed metadata, for the calibration.
        staged_root: The staged symlink tree.
        database_path: Database to write into.

    Returns:
        The COLMAP camera model name every image was extracted with.

    Raises:
        ValueError: When the dataset mixes camera models, which this staging cannot express.
    """
    models: set[str] = {COLMAP_MODEL_BY_PROJECTION_MODEL[camera.projection_model] for camera in frames_meta.cameras.values()}
    if len(models) != 1:
        raise ValueError(f"expected one COLMAP camera model across the rig, got {sorted(models)}")
    camera_model: str = models.pop()

    extraction_options: pycolmap.FeatureExtractionOptions = pycolmap.FeatureExtractionOptions()
    extraction_options.type = EXTRACTOR_BY_CHOICE[config.features]
    extraction_options.use_gpu = config.use_gpu
    device: pycolmap.Device = pycolmap.Device.cuda if config.use_gpu else pycolmap.Device.cpu

    for camera_params_id, camera in sorted(frames_meta.cameras.items()):
        folder: Path = staged_root / camera.sensor_name
        names: list[str] = sorted(f"{camera.sensor_name}/{item.name}" for item in folder.iterdir())
        reader_options: pycolmap.ImageReaderOptions = pycolmap.ImageReaderOptions()
        reader_options.camera_model = camera_model
        reader_options.camera_params = ",".join(repr(float(value)) for value in colmap_camera_parameters(camera))
        pycolmap.extract_features(
            database_path=database_path,
            image_path=staged_root,
            image_names=names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=device,
        )
        print(f"[audit-robocap] extracted {len(names)} images of camera {camera_params_id} ({camera.sensor_name})")
    return camera_model


def calibrated_cam_from_rig(frames_meta: FramesMeta, reference: RigReference) -> dict[int, pycolmap.Rigid3d]:
    """The calibration's `cam_from_rig` per camera, the reference at identity.

    Args:
        frames_meta: The parsed metadata.
        reference: The rig origin and its `vehicle_T_cam_ref` bridge.

    Returns:
        `cam_from_rig` per `camera_params_id`.
    """
    return {
        camera_params_id: camera.vehicle_T_cam.inverse() * reference.vehicle_T_reference
        for camera_params_id, camera in sorted(frames_meta.cameras.items())
    }


def rig_config(frames_meta: FramesMeta, reference: RigReference) -> pycolmap.RigConfig:
    """Describe the rig the way `apply_rig_config` wants it.

    Args:
        frames_meta: The parsed metadata.
        reference: The rig origin and its bridge.

    Returns:
        A config listing every camera by `image_prefix`, one of them the
        reference sensor and the rest carrying `cam_from_rig`.
    """
    calibrated: dict[int, pycolmap.Rigid3d] = calibrated_cam_from_rig(frames_meta, reference)
    # Reference first: `Rig::AddSensor` refuses to run before a reference sensor exists.
    ordered: list[int] = sorted(frames_meta.cameras, key=lambda item: (item != reference.camera_params_id, item))
    cameras: list[pycolmap.RigConfigCamera] = []
    for camera_params_id in ordered:
        camera: CameraParams = frames_meta.cameras[camera_params_id]
        entry: pycolmap.RigConfigCamera = pycolmap.RigConfigCamera()
        entry.image_prefix = f"{camera.sensor_name}/"
        entry.ref_sensor = camera_params_id == reference.camera_params_id
        if not entry.ref_sensor:
            entry.cam_from_rig = calibrated[camera_params_id]
        cameras.append(entry)
    config: pycolmap.RigConfig = pycolmap.RigConfig()
    config.cameras = cameras
    return config


def pairing_options(config: ColmapRigBaselineRobocapConfig, loop_detection: bool) -> pycolmap.SequentialPairingOptions:
    """COLMAP's sequential pairing, at its defaults but for the overlap and the loop switch.

    Args:
        config: The run's configuration.
        loop_detection: Whether to turn the vocabulary-tree pass on.

    Returns:
        The options; `vocab_tree_path` is left empty when the config names no
        tree, which is what makes COLMAP fetch the one its `retrieval/resources.h`
        pairs with the feature type.
    """
    options: pycolmap.SequentialPairingOptions = pycolmap.SequentialPairingOptions()
    options.overlap = config.overlap
    options.loop_detection = loop_detection
    if config.vocab_tree_path is not None:
        options.vocab_tree_path = config.vocab_tree_path
    return options


def match(config: ColmapRigBaselineRobocapConfig, database_path: Path) -> None:
    """Match features with COLMAP's stock sequential pairing.

    Args:
        config: The run's configuration.
        database_path: Database holding the extracted features and the rig.
    """
    matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    matching_options.type = MATCHER_BY_CHOICE[config.features]
    matching_options.use_gpu = config.use_gpu
    device: pycolmap.Device = pycolmap.Device.cuda if config.use_gpu else pycolmap.Device.cpu
    pycolmap.match_sequential(
        database_path=database_path,
        matching_options=matching_options,
        pairing_options=pairing_options(config, config.loop_detection),
        device=device,
    )


def pair_key(image_id1: int, image_id2: int) -> tuple[int, int]:
    """Order one image pair's ids, so a pair has one key whichever way round it came.

    Args:
        image_id1: One image id.
        image_id2: The other.

    Returns:
        The pair, smaller id first.
    """
    return (image_id1, image_id2) if image_id1 < image_id2 else (image_id2, image_id1)


def sequential_pair_keys(config: ColmapRigBaselineRobocapConfig, database_path: Path) -> set[tuple[int, int]]:
    """Every pair a loop-detection-free sequential pass would propose on this database.

    Run *after* matching, purely as a measurement: subtracting it from the
    database's verified pairs is what counts the pairs loop detection added.
    The generator is COLMAP's own, so the count needs no re-implementation of
    the quadratic overlap or of the rig expansion.

    Args:
        config: The run's configuration.
        database_path: The matched database.

    Returns:
        Ordered id pairs.
    """
    with pycolmap.Database.open(database_path) as database:
        generator: pycolmap.SequentialPairGenerator = pycolmap.SequentialPairGenerator(pairing_options(config, False), database)
        return {pair_key(int(first), int(second)) for first, second in generator.all_pairs()}


def verified_pair_keys(database_path: Path) -> set[tuple[int, int]]:
    """Every pair the database holds a verified two-view geometry for.

    `read_two_view_geometry_num_inliers` returns two parallel arrays — COLMAP's
    *packed* pair ids and the inlier counts — not a list of pairs, so the ids go
    back through `pair_id_to_image_pair` before they can be compared with the
    generator's output.

    Args:
        database_path: The matched database.

    Returns:
        Ordered id pairs.
    """
    with pycolmap.Database.open(database_path) as database:
        pair_ids: Int64[ndarray, "n_pairs"] = np.asarray(database.read_two_view_geometry_num_inliers()[0], dtype=np.int64)
    keys: set[tuple[int, int]] = set()
    for pair_id in pair_ids:
        first, second = pycolmap.pair_id_to_image_pair(int(pair_id))
        keys.add(pair_key(int(first), int(second)))
    return keys


def pipeline_options(config: ColmapRigBaselineRobocapConfig, refine_sensor_from_rig: bool) -> pycolmap.IncrementalPipelineOptions:
    """COLMAP's incremental mapper at its defaults, with the bundle-adjustment backend set.

    Args:
        config: The run's configuration.
        refine_sensor_from_rig: Whether the mapper may move the rig extrinsics;
            CASPAR throws on a multi-sensor frame when it may.

    Returns:
        The options handed to `pycolmap.incremental_mapping`.
    """
    options: pycolmap.IncrementalPipelineOptions = pycolmap.IncrementalPipelineOptions()
    options.ba_local_backend = BACKEND_BY_CHOICE[config.ba_backend]
    if config.ba_global_caspar:
        options.ba_global_backend = BACKEND_BY_CHOICE["caspar"]
    options.ba_use_gpu = config.use_gpu
    # A string, not an int: COLMAP takes a comma-separated GPU list here.
    options.ba_gpu_index = "0" if config.use_gpu else "-1"
    options.ba_refine_sensor_from_rig = refine_sensor_from_rig
    return options


def rotation_degrees(delta: pycolmap.Rigid3d) -> float:
    """Geodesic angle of a rigid transform's rotation.

    Args:
        delta: The transform.

    Returns:
        The angle in degrees.
    """
    trace: float = float(np.trace(np.asarray(delta.rotation.matrix(), dtype=np.float64)))
    return math.degrees(math.acos(max(-1.0, min(1.0, 0.5 * (trace - 1.0)))))


def extrinsic_deltas(reconstruction: pycolmap.Reconstruction, frames_meta: FramesMeta, reference: RigReference) -> tuple[ExtrinsicDelta, ...]:
    """Compare the model's rig extrinsics against the calibration.

    Args:
        reconstruction: The largest model.
        frames_meta: The parsed metadata.
        reference: The rig origin used to build the config.

    Returns:
        One delta per non-reference camera the model kept. Every entry is zero
        when `ba_refine_sensor_from_rig` was held off, which is the CASPAR path.
    """
    calibrated: dict[int, pycolmap.Rigid3d] = calibrated_cam_from_rig(frames_meta, reference)
    params_id_by_folder: dict[str, int] = {camera.sensor_name: camera_params_id for camera_params_id, camera in frames_meta.cameras.items()}
    params_id_by_camera_id: dict[int, int] = {
        image.camera_id: params_id_by_folder[image.name.split("/")[0]] for image in reconstruction.images.values()
    }
    rig: pycolmap.Rig = next(iter(reconstruction.rigs.values()))
    deltas: list[ExtrinsicDelta] = []
    for camera_id, camera_params_id in sorted(params_id_by_camera_id.items(), key=lambda item: item[1]):
        if camera_params_id == reference.camera_params_id:
            continue
        sensor_id: pycolmap.sensor_t = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)
        if not rig.has_sensor(sensor_id):
            continue
        difference: pycolmap.Rigid3d = rig.sensor_from_rig(sensor_id) * calibrated[camera_params_id].inverse()
        deltas.append(
            ExtrinsicDelta(
                camera_params_id=camera_params_id,
                sensor_name=frames_meta.cameras[camera_params_id].sensor_name,
                translation_millimeters=MILLIMETRES_PER_METRE * float(np.linalg.norm(np.asarray(difference.translation, dtype=np.float64))),
                rotation_degrees=rotation_degrees(difference),
            )
        )
    return tuple(deltas)


def rename_to_dataset_names(reconstruction: pycolmap.Reconstruction, staged: StagedImages) -> dict[int, int]:
    """Put the dataset's own `image_name`s back on a model built from the staged tree.

    Every downstream reader — `colsfm.benchmark.read_run`, the colour pass, the
    exported `frames_meta.json` — keys on the dataset's names, so this is what
    makes the run comparable with a blob run and a colsfm run.

    Args:
        reconstruction: The model to rewrite in place.
        staged: The maps `stage_images` returned.

    Returns:
        `keyframe_id` per COLMAP `image_id`, read before the names are replaced.
    """
    keyframe_id_by_image_id: dict[int, int] = {}
    for image_id, image in reconstruction.images.items():
        keyframe_id_by_image_id[image_id] = staged.keyframe_id_by_staged[image.name]
        image.name = staged.image_name_by_staged[image.name]
    return keyframe_id_by_image_id


def write_cusfm_run(
    cusfm_dir: Path,
    reconstruction: pycolmap.Reconstruction,
    frames_meta: FramesMeta,
    keyframe_id_by_image_id: dict[int, int],
    input_dir: Path,
) -> int:
    """Write the four artifacts a cuSFM run leaves behind, from a stock COLMAP model.

    Args:
        cusfm_dir: The run's `cusfm` directory.
        reconstruction: The largest model, already renamed to dataset image names.
        frames_meta: The input metadata, copied through except for the poses.
        keyframe_id_by_image_id: The map `rename_to_dataset_names` returned.
        input_dir: Image root the colour pass reads.

    Returns:
        Points that got a colour from the source imagery.
    """
    num_coloured: int = colour_points_from_images(reconstruction, input_dir)
    write_colmap_model(cusfm_dir, reconstruction)
    world_T_cam_by_keyframe_id: dict[int, pycolmap.Rigid3d] = {
        keyframe_id_by_image_id[image_id]: image.cam_from_world().inverse()
        for image_id, image in reconstruction.images.items()
        if image.has_pose
    }
    optimised: FramesMeta = write_optimised_frames_meta(cusfm_dir, frames_meta, world_T_cam_by_keyframe_id, "ALIGNMENT")
    write_pose_files(cusfm_dir, optimised)
    return num_coloured


def count_log_lines(path: Path, marker: str) -> int:
    """Count the lines of a log file containing a marker.

    Args:
        path: The log file; a missing file counts as zero.
        marker: Substring to look for.

    Returns:
        How many lines contain it.
    """
    if not path.is_file():
        return 0
    with path.open(errors="replace") as handle:
        return sum(1 for line in handle if marker in line)


def main(config: ColmapRigBaselineRobocapConfig) -> None:
    """Run one stock-COLMAP rig baseline and write a cuSFM-layout run beside its report.

    Args:
        config: The run's configuration.

    Raises:
        RuntimeError: When the mapper returns no reconstruction.
    """
    run_dir: Path = config.output_root / config.run_name
    if config.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    database_path: Path = run_dir / "database.db"
    models_dir: Path = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    staged_root: Path = run_dir / STAGED_IMAGE_DIR_NAME
    cusfm_dir: Path = run_dir / CUSFM_DIR_NAME
    log_path: Path = run_dir / COLMAP_LOG_NAME

    frames_meta: FramesMeta = read_frames_meta(config.input_dir / FRAMES_META_NAME)
    if config.max_num_rig_frames is not None:
        kept: list[int] = [
            keyframe_id for rig_frame in frames_meta.rig_frames()[: config.max_num_rig_frames] for keyframe_id in rig_frame.keyframe_ids
        ]
        frames_meta = frames_meta.filtered(kept)
    reference_id: int = reference_camera_params_id(frames_meta)
    reference: RigReference = rig_reference(frames_meta, reference_id)

    supported: frozenset[str] = caspar_supported_camera_models() if config.ba_backend == "caspar" else frozenset()
    # CASPAR throws on a multi-sensor frame when it may move the extrinsics
    # (`bundle_adjustment_caspar.cc:186`), so this is the one COLMAP default the
    # CASPAR path cannot keep.
    refine_sensor_from_rig: bool = config.ba_backend != "caspar"
    options: pycolmap.IncrementalPipelineOptions = pipeline_options(config, refine_sensor_from_rig)

    pycolmap.logging.verbose_level = config.verbose_level
    print(f"[audit-robocap] {len(frames_meta.keyframes)} images, {len(frames_meta.cameras)} cameras, {len(frames_meta.rig_frames())} rig frames")
    print(f"[audit-robocap] reference camera {reference_id} ({frames_meta.cameras[reference_id].sensor_name})")
    print(f"[audit-robocap] CASPAR camera models in this build: {sorted(supported) if supported else 'not probed (ceres)'}")
    print(f"[audit-robocap] loop detection: {config.loop_detection}; local BA: {config.ba_backend}; refine_sensor_from_rig: {refine_sensor_from_rig}")
    print(f"[audit-robocap] COLMAP log -> {log_path}")

    clock: StageClock = StageClock()
    with captured_stderr(log_path):
        with clock.stage("feature_extraction"):
            staged: StagedImages = stage_images(frames_meta, config.input_dir, staged_root)
            camera_model: str = extract(config, frames_meta, staged_root, database_path)
        with clock.stage("matching"):
            # Rig configuration is a sub-second database write with no stage of its own;
            # it is counted here, against the run, rather than left untimed.
            with pycolmap.Database.open(database_path) as database:
                pycolmap.apply_rig_config([rig_config(frames_meta, reference)], database)
            match(config, database_path)
        with clock.stage("incremental_mapping"):
            models: dict[int, pycolmap.Reconstruction] = pycolmap.incremental_mapping(
                database_path=database_path, image_path=staged_root, output_path=models_dir, options=options
            )

    sequential_keys: set[tuple[int, int]] = sequential_pair_keys(config, database_path)
    verified_keys: set[tuple[int, int]] = verified_pair_keys(database_path)
    with pycolmap.Database.open(database_path) as database:
        num_matched_pairs: int = int(database.num_matched_image_pairs())
        num_frames_in_database: int = int(database.num_frames())
    print(f"[audit-robocap] database frames: {num_frames_in_database}; matched pairs: {num_matched_pairs}; verified: {len(verified_keys)}")
    print(f"[audit-robocap] sequential proposals: {len(sequential_keys)}; verified loop pairs: {len(verified_keys - sequential_keys)}")

    summaries: list[ModelSummary] = sorted(
        (_summarise(model, index) for index, model in models.items()), key=lambda item: item.metrics.registered_images, reverse=True
    )
    print(f"[audit-robocap] {len(models)} model(s); registered per model: {[item.metrics.registered_images for item in summaries]}")
    if not summaries:
        raise RuntimeError("incremental_mapping returned no reconstruction")
    largest: pycolmap.Reconstruction = models[summaries[0].model_index]

    keyframe_id_by_image_id: dict[int, int] = rename_to_dataset_names(largest, staged)
    deltas: tuple[ExtrinsicDelta, ...] = extrinsic_deltas(largest, frames_meta, reference)
    with clock.stage("export"):
        num_coloured: int = write_cusfm_run(cusfm_dir, largest, frames_meta, keyframe_id_by_image_id, config.input_dir)
    clock.write_csv(cusfm_dir)
    print(f"[audit-robocap] export: coloured {num_coloured}/{summaries[0].metrics.num_points3D} points; run -> {cusfm_dir}")

    result: ColmapRigBaselineRobocapResult = ColmapRigBaselineRobocapResult(
        run_dir=str(run_dir),
        cusfm_dir=str(cusfm_dir),
        features=config.features,
        ba_local_backend=config.ba_backend,
        ba_global_backend="caspar" if config.ba_global_caspar else "ceres",
        ba_refine_sensor_from_rig=refine_sensor_from_rig,
        caspar_supported_camera_models=tuple(sorted(supported)),
        num_caspar_solves=count_log_lines(log_path, CASPAR_RAN_MARKER),
        num_caspar_fisheye_factor_blocks=count_log_lines(log_path, CASPAR_FISHEYE_MARKER),
        caspar_unsupported_model_warnings=count_log_lines(log_path, CASPAR_UNSUPPORTED_MODEL_MARKER),
        camera_model=camera_model,
        reference_camera_params_id=reference_id,
        total_images=len(frames_meta.keyframes),
        total_rig_frames=staged.num_rig_frames,
        loop_detection=config.loop_detection,
        vocab_tree=str(config.vocab_tree_path) if config.vocab_tree_path else "default-for-feature-type",
        num_sequential_pairs_proposed=len(sequential_keys),
        num_image_pairs_matched=num_matched_pairs,
        num_verified_image_pairs=len(verified_keys),
        num_verified_loop_pairs=len(verified_keys - sequential_keys),
        num_models=len(models),
        models=tuple(summaries),
        stage_timings=tuple((item.command, item.runtime_seconds) for item in clock.timings),
        total_runtime_seconds=clock.total_seconds(),
        largest_model_fraction_registered=summaries[0].metrics.registered_images / len(frames_meta.keyframes),
        num_rig_frames_registered=sum(1 for frame in largest.frames.values() if frame.has_pose),
        extrinsic_deltas=deltas,
    )
    output_json: Path = write_json_report(config.bench_dir / f"{config.run_name}.json", result)

    for timing in clock.timings:
        print(f"[audit-robocap] {timing.command:<22} {timing.runtime_seconds:8.2f} s")
    print(f"[audit-robocap] {'total':<22} {clock.total_seconds():8.2f} s")
    largest_metrics: ReconstructionMetrics = summaries[0].metrics
    print(
        f"[audit-robocap] largest model: {largest_metrics.registered_images}/{len(frames_meta.keyframes)} images, "
        f"{result.num_rig_frames_registered}/{staged.num_rig_frames} rig frames, "
        f"{largest_metrics.num_points3D} points, {largest_metrics.mean_reprojection_error_px:.3f} px"
    )
    print(
        f"[audit-robocap] CASPAR solves: {result.num_caspar_solves} ({result.num_caspar_fisheye_factor_blocks} OPENCV_FISHEYE factor blocks); "
        f"unsupported-model warnings: {result.caspar_unsupported_model_warnings}"
    )
    for delta in deltas:
        print(f"[audit-robocap] extrinsic {delta.sensor_name:<14} {delta.translation_millimeters:8.2f} mm  {delta.rotation_degrees:7.3f} deg")
    print(f"[audit-robocap] wrote {output_json}")
    print(f"[audit-robocap] optimised metadata at {cusfm_dir / KEYFRAME_METADATA_SUBPATH}")


if __name__ == "__main__":
    main(tyro.cli(ColmapRigBaselineRobocapConfig))

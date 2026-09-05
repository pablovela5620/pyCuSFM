"""ALIKED feature extraction — the Python replacement for `feature_extractor_main`'s per-image half.

COLMAP 4.2 ships ALIKED itself (`FeatureExtractorType.ALIKED_N16ROT`,
`ALIKED_N32`), so this stage is a thin, typed driver over
`pycolmap.extract_features` rather than a TensorRT runner. Two facts make that
work:

* **The pipeline owns the identifiers.** `colsfm.database.create_database`
  writes the cameras, rig, frames and images first; COLMAP's `ImageReader` then
  looks each name up, finds the row and keeps its `image_id`, `camera_id` and
  `frame_id`. Nothing is renumbered and no camera is invented.
* **The variant matches the blob.** `pycusfm/models/aliked_lightglue/aliked.onnx`
  carries a 32-channel SDDH offset convolution (`Conv[32, 128, 3, 3]` feeding a
  `Conv[32, 32, 1, 1]`), i.e. `M = 16` sampling positions, the ALIKED-n16
  architecture — corroborated by the RaCo export's own checkpoint,
  `~/.cache/torch/hub/checkpoints/aliked-n16.pth`. `ALIKED_N32` would be `M = 32`
  (`Conv[64, 128, 3, 3]`). So `ALIKED_N16ROT` is the closest COLMAP variant and
  the default here. Its weights are trained with rotation augmentation, so the
  descriptors are not the blob's bit for bit — the plan does not ask for that.

COLMAP downloads its ONNX models on first use from its own GitHub release
assets and caches them in `~/.cache/colmap/<sha256>-<filename>`; nothing ships
inside the conda package.

**Environment note.** ONNX Runtime's CUDA execution provider needs `libcudnn.so`
for the convolutions, and the `colsfm` pixi environment declares no cuDNN. The
failure is not an exception: ONNX Runtime throws inside a COLMAP worker thread
and the process aborts with SIGABRT. `resolve_device` therefore probes for cuDNN
*before* the call and falls back to the CPU provider, which is ~50x slower
(1.8 s per 1920x1200 image against 0.03 s on the GPU).
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias

import pycolmap

from colsfm.config import KeyframeSelectionConfig
from colsfm.database import create_database, keypoint_counts
from colsfm.frames_meta import FramesMeta
from colsfm.keyframe_selection import KeyframeSelection, apply_selection, select_keyframes

AlikedVariant: TypeAlias = Literal["ALIKED_N16ROT", "ALIKED_N32"]
"""The two ALIKED graphs COLMAP 4.2 can download and run."""

DeviceChoice: TypeAlias = Literal["auto", "cuda", "cpu"]
"""Requested compute device; `auto` picks CUDA when it is actually usable."""

ResolvedDevice: TypeAlias = Literal["cuda", "cpu"]
"""The device a stage really ran on."""

BLOB_DETECTOR_THRESHOLD: Final[float] = 0.005
"""`aliked_detector.detector_threshold` in `pycusfm/configs/isaac/keypoint_creation_config.pb.txt`."""

BLOB_MAX_KEYPOINTS: Final[int] = 2048
"""Keypoints the blob's ONNX detector head emits per image, which is what actually binds.

`max_key_points_for_tensorrt: 4000` never fires: the shipped graph's top-k is
fixed at 2048, so every Galileo keyframe holds 2047-2048 points
(feature_extractor_main.md §6.3).
"""

CUDNN_LIBRARY_NAME: Final[str] = "libcudnn.so"
"""What ONNX Runtime's CUDA provider dlopens before it will run a convolution."""


@dataclass(frozen=True, slots=True)
class FeatureOptions:
    """How ALIKED runs over one image set."""

    variant: AlikedVariant = "ALIKED_N16ROT"
    """ALIKED graph; `ALIKED_N16ROT` is the architecture the blob's ONNX uses."""
    max_num_features: int = BLOB_MAX_KEYPOINTS
    """Keypoint budget per image, matching the blob's fixed top-2048 head."""
    min_score: float = BLOB_DETECTOR_THRESHOLD
    """Detector score gate; the blob's `detector_threshold`, applied with `>=`."""
    max_image_size: int = -1
    """Longest side COLMAP resizes to before detection; -1 keeps the original size."""
    num_threads: int = -1
    """COLMAP extraction threads; -1 lets COLMAP choose."""
    device: DeviceChoice = "auto"
    """Compute device; `auto` falls back to the CPU provider when cuDNN is missing."""
    gpu_index: str = "-1"
    """CUDA device index as COLMAP's comma-separated string; `"-1"` lets COLMAP choose."""


DEFAULT_FEATURE_OPTIONS: Final[FeatureOptions] = FeatureOptions()
"""Shared immutable default, so the signatures below hold no constructor call."""

DEFAULT_KEYFRAME_SELECTION: Final[KeyframeSelectionConfig] = KeyframeSelectionConfig()
"""`feature_extractor_main`'s own gflag defaults (0.5 m / 5 deg / 50 us)."""


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """What one extraction run produced."""

    keypoint_counts: dict[int, int] = field(default_factory=dict)
    """Keypoints stored per `image_id`."""
    device: ResolvedDevice = "cpu"
    """The device the run actually used."""
    elapsed_seconds: float = 0.0
    """Wall time of the `pycolmap.extract_features` call."""

    def summary(self) -> str:
        """One line naming the image count, the keypoint range and the wall time.

        Returns:
            A printable summary; `"no images"` when nothing was extracted.
        """
        if not self.keypoint_counts:
            return f"extracted no images on {self.device} in {self.elapsed_seconds:.2f}s"
        counts: list[int] = sorted(self.keypoint_counts.values())
        return (
            f"extracted {len(counts)} images on {self.device} in {self.elapsed_seconds:.2f}s, "
            f"keypoints {counts[0]}-{counts[-1]} (median {counts[len(counts) // 2]})"
        )


def cudnn_is_available() -> bool:
    """Report whether ONNX Runtime will find cuDNN.

    Returns:
        True when `libcudnn.so` loads. ONNX Runtime dlopens it by that exact
        soname-less name, so a versioned `libcudnn.so.9` on its own is not enough.
    """
    try:
        ctypes.CDLL(CUDNN_LIBRARY_NAME)
    except OSError:
        return False
    return True


def resolve_device(device: DeviceChoice = "auto") -> ResolvedDevice:
    """Turn a device request into the device COLMAP's ONNX models can really use.

    Args:
        device: `auto`, `cuda` or `cpu`.

    Returns:
        `cuda` when a CUDA build, a CUDA device and cuDNN are all present, else `cpu`.

    Raises:
        RuntimeError: When `cuda` is requested and any of the three is missing.
            Letting the call through instead aborts the process: ONNX Runtime
            raises inside a COLMAP worker thread, which terminates.
    """
    if device == "cpu":
        return "cpu"
    reasons: list[str] = []
    if not pycolmap.has_cuda:
        reasons.append("pycolmap was built without CUDA")
    elif pycolmap.get_num_cuda_devices() < 1:
        reasons.append("no CUDA device is visible")
    if not cudnn_is_available():
        reasons.append(f"{CUDNN_LIBRARY_NAME} is not loadable, so ONNX Runtime's CUDA provider cannot run convolutions")
    if not reasons:
        return "cuda"
    if device == "cuda":
        raise RuntimeError("CUDA was requested but is unusable: " + "; ".join(reasons))
    print(f"colsfm: falling back to the CPU provider because {'; '.join(reasons)}")
    return "cpu"


def _extraction_options(options: FeatureOptions, device: ResolvedDevice) -> pycolmap.FeatureExtractionOptions:
    """Build the pycolmap options for one extraction run.

    Args:
        options: The typed settings.
        device: The already-resolved device.

    Returns:
        `FeatureExtractionOptions` with the ALIKED sub-options filled in. In
        COLMAP 4.2 `use_gpu`, `gpu_index`, `num_threads` and `max_image_size`
        live on this object, not on the per-extractor one.
    """
    extraction_options: pycolmap.FeatureExtractionOptions = pycolmap.FeatureExtractionOptions()
    extraction_options.type = getattr(pycolmap.FeatureExtractorType, options.variant)
    extraction_options.use_gpu = device == "cuda"
    extraction_options.gpu_index = options.gpu_index
    extraction_options.num_threads = options.num_threads
    extraction_options.max_image_size = options.max_image_size
    extraction_options.aliked.max_num_features = options.max_num_features
    extraction_options.aliked.min_score = options.min_score
    return extraction_options


def extract_features(
    database_path: Path,
    image_root: Path,
    image_names: Sequence[str],
    options: FeatureOptions = DEFAULT_FEATURE_OPTIONS,
) -> ExtractionReport:
    """Run ALIKED over the named images and store keypoints and descriptors.

    The database must already hold an image row per name (see
    `colsfm.database.create_database`); COLMAP reuses those rows and their
    cameras rather than creating its own.

    Args:
        database_path: An existing COLMAP database.
        image_root: Directory the `image_name` values are relative to, i.e. the
            raw data directory holding `<camera_name>/<timestamp>.jpeg`.
        image_names: Images to extract, as stored in the database.
        options: Extraction settings.

    Returns:
        Per-image keypoint counts, the device used and the wall time.

    Raises:
        FileNotFoundError: When the database or the image root is missing.
        RuntimeError: When `options.device` is `cuda` and CUDA is unusable.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"No image directory at {image_root}")

    device: ResolvedDevice = resolve_device(options.device)
    extraction_options: pycolmap.FeatureExtractionOptions = _extraction_options(options, device)
    started: float = time.perf_counter()
    pycolmap.extract_features(
        database_path,
        image_root,
        image_names=list(image_names),
        extraction_options=extraction_options,
        device=pycolmap.Device.cuda if device == "cuda" else pycolmap.Device.cpu,
    )
    elapsed_seconds: float = time.perf_counter() - started

    name_to_id: dict[str, int] = {
        image.name: image.image_id for image in pycolmap.Database.open(database_path).read_all_images()
    }
    wanted_ids: list[int] = [name_to_id[name] for name in image_names if name in name_to_id]
    report: ExtractionReport = ExtractionReport(
        keypoint_counts=keypoint_counts(database_path, wanted_ids),
        device=device,
        elapsed_seconds=elapsed_seconds,
    )
    print(f"colsfm.features: {report.summary()}")
    return report


@dataclass(frozen=True, slots=True)
class SelectedExtraction:
    """The full stage-1 result: which frames survived selection and what they yielded."""

    selection: KeyframeSelection
    """Kept keyframe ids and the dense 1-based rig renumbering."""
    frames_meta: FramesMeta
    """The filtered collection, i.e. what `feature_extractor_main` writes out."""
    report: ExtractionReport
    """Per-image keypoint counts and timing."""


def extract_selected(
    database_path: Path,
    image_root: Path,
    frames_meta: FramesMeta,
    selection_config: KeyframeSelectionConfig = DEFAULT_KEYFRAME_SELECTION,
    options: FeatureOptions = DEFAULT_FEATURE_OPTIONS,
    *,
    overwrite: bool = False,
) -> SelectedExtraction:
    """Select keyframes, create the database and extract features from the survivors.

    Keyframe selection is `colsfm.keyframe_selection`'s; this only wires it to
    the database and the extractor so no caller has to repeat the three steps.

    Args:
        database_path: Destination database.
        image_root: Directory the `image_name` values are relative to.
        frames_meta: The unfiltered input collection.
        selection_config: The `feature_extractor_main` gflags that drive selection.
        options: Extraction settings.
        overwrite: Replace an existing database.

    Returns:
        The selection, the filtered collection and the extraction report.

    Raises:
        FileExistsError: When the database exists and `overwrite` is False.
        RuntimeError: When `options.device` is `cuda` and CUDA is unusable.
    """
    selection: KeyframeSelection = select_keyframes(
        frames_meta,
        min_inter_frame_distance_m=selection_config.min_inter_frame_distance_m,
        min_inter_frame_rotation_degrees=selection_config.min_inter_frame_rotation_degrees,
        sample_sync_threshold_microseconds=selection_config.sample_sync_threshold_microseconds,
    )
    selected: FramesMeta = apply_selection(frames_meta, selection)
    print(f"colsfm.features: {len(selected.keyframes)} frames selected from {len(frames_meta.keyframes)} raw frames.")
    create_database(database_path, selected, overwrite=overwrite)
    report: ExtractionReport = extract_features(
        database_path,
        image_root,
        [keyframe.image_name for keyframe in selected.keyframes],
        options,
    )
    return SelectedExtraction(selection=selection, frames_meta=selected, report=report)

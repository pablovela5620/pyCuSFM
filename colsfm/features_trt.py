"""ALIKED feature extraction through the blob's own TensorRT engine.

The `tensorrt` half of `colsfm.features`. Where the default backend hands the
images to `pycolmap.extract_features` and lets COLMAP run its own ONNX ALIKED,
this one runs **the blob's graph**, `pycusfm/models/aliked_lightglue/aliked.onnx`,
through the engine the blob itself built
(`aliked_fp16_10_13_3_9_sm_12_0.engine`, accepted as a trusted prebuilt engine
when the TensorRT version and the GPU match — see
`colsfm.tensorrt_runtime.resolve_engine`). The
descriptors are therefore the blob's descriptors, not a rotation-augmented
lookalike, which is what makes a TensorRT LightGlue match meaningful
(`colsfm.matching_trt`).

**Preprocessing is the blob's, byte for byte** (feature_extractor_main.md §6.2):
decode with OpenCV, `cv2.resize` to the engine's input size with the default
`INTER_LINEAR` and **no** aspect-ratio preservation, split the BGR planes into
planar RGB, and divide by 255. A one-channel image is promoted to three by
`cv2.IMREAD_COLOR`, which is the same thing the blob's `cvtColor(GRAY -> BGR)`
does. `tools/raco_extract.prepare_frame` is the prior art.

**Post-processing is the blob's too** (§6.3): keep `scores >= detector_threshold`
(a `>=`, not a `>`), then map each normalised coordinate back to the **original**
image with `(k + 1) * 0.5 * (size - 1)`. The `max_key_points_for_tensorrt: 4000`
SSC branch is dead — the graph's own head emits a fixed top-2048 — so it is not
implemented; `FeatureOptions.max_num_features` truncates instead, which is a
no-op at the shipped 2048.

**Batch size is one, because the graph is.** `aliked.onnx` declares a static
`image [1, 3, 1200, 1920]` input, so there is no batch axis to widen and no
engine to rebuild; `tools/export_cusfm_raco.py --batched-extractor-path` is what
produces a batch-dynamic graph, and it lives in the `raco` environment. What is
overlapped instead is the CPU half: a bounded thread pool decodes and resizes
ahead of the GPU, which is where the blob's 6.3 ms per frame of preprocessing
goes. Measured on Galileo's 226 frames: 24.3 ms of engine execution per image,
7.01 s for the stage against the pycolmap backend's 7.19 s and the blob's 8.64 s.

**Every image keeps its identity.** The rows were written by
`colsfm.database.create_database`, and this backend only fills in the keypoint
and descriptor columns of images it looks up by name, so `image_id`, `camera_id`
and `frame_id` are untouched — the same contract the pycolmap backend gets from
COLMAP's `ImageReader`.

**Everything above from the tensor bindings inwards is
`colsfm.features_native`**, which `colsfm.features_raco` runs through as well.
What stays here is what is specific to the blob's graph: where it lives, what its
declared input size is, that its batch axis is one, and that its `scores` output
is a real detector response. The shared names this module used to define are
re-exported at the bottom, because `tools/audit/` imports several of them from
here.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from colsfm.database import keypoint_counts
from colsfm.features import TensorRTExtraction
from colsfm.features_native import (
    ALIKED_IMAGE_BINDING,
    DEFAULT_GPU_PREPROCESSING,
    DEFAULT_PREPROCESSING_WORKERS,
    DESCRIPTOR_TYPE,
    ExtractionGroup,
    FeatureBatch,
    ImageTask,
    NativeModel,
    NetworkImage,
    NormalizedKeypoints,
    PreparedImage,
    ResizedImageBGR,
    image_tasks,
    normalized_to_pixels,
    preprocess_image,
    run_native_extraction,
    to_network_bchw,
)
from colsfm.tensorrt_runtime import MODEL_DIR, resolve_engine

ALIKED_ONNX_PATH: Final[Path] = MODEL_DIR / "aliked.onnx"
"""The blob's own ALIKED graph; its engine is cached beside it."""

NETWORK_HEIGHT: Final[int] = 1200
"""`aliked.onnx`'s declared input height, and the blob's TensorRT profile `opt`."""

NETWORK_WIDTH: Final[int] = 1920
"""`aliked.onnx`'s declared input width."""

ALIKED_BATCH_SIZE: Final[int] = 1
"""Images per execution, which the graph's static `image [1, 3, 1200, 1920]` fixes.

Not a tuning knob: widening it means re-exporting the graph, which is what
`colsfm.features_raco` is."""

ALIKED_MODEL: Final[NativeModel] = NativeModel(
    name="tensorrt",
    image_binding=ALIKED_IMAGE_BINDING,
    descriptor_type=DESCRIPTOR_TYPE,
    score_meaning="detector_response",
)
"""The blob's graph as `colsfm.features_native` sees it.

`detector_response` is the load-bearing half: this graph's `scores` really is a
detector confidence, so `FeatureOptions.min_score` — the blob's 0.005
`detector_threshold` — gates it meaningfully. RaCo's does not; see
`colsfm.features_raco.RACO_MODEL`."""


def extract_tensorrt(
    database_path: Path,
    image_root: Path,
    image_names: Sequence[str],
    *,
    min_score: float,
    max_num_features: int,
    onnx_path: Path = ALIKED_ONNX_PATH,
    preprocessing_workers: int = DEFAULT_PREPROCESSING_WORKERS,
    gpu_preprocessing: bool = DEFAULT_GPU_PREPROCESSING,
) -> TensorRTExtraction:
    """Run the blob's ALIKED engine over the named images and store the features.

    One group, one engine, one image per execution, and the input size read off
    the engine's own `image` binding rather than restated here — so a re-exported
    graph at another size still preprocesses to what its engine wants.

    Args:
        database_path: An existing COLMAP database whose image rows are written.
        image_root: Directory the `image_names` are relative to.
        image_names: Images to extract, as stored in the database.
        min_score: Detector score gate, applied with `>=` as the blob does.
        max_num_features: Keypoint ceiling per image; the graph's own top-2048
            head normally binds first, so this only ever truncates.
        onnx_path: The ALIKED graph whose engine to use.
        preprocessing_workers: Decode-and-resize threads.
        gpu_preprocessing: Upload the decoded uint8 frame and let a TensorRT
            preprocessing engine transpose, widen and scale it. `False` does that
            arithmetic on the host, as this backend originally did.

    Returns:
        Keypoints stored per `image_id`, and the wall time of the whole pass
        including the engine load.

    Raises:
        FileNotFoundError: When the database, the image root or an image is missing.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"No image directory at {image_root}")

    started: float = time.perf_counter()
    tasks: list[ImageTask] = image_tasks(database_path, image_root, image_names)
    engine_path: Path = resolve_engine(onnx_path)
    run_native_extraction(
        database_path,
        [ExtractionGroup(engine_path=engine_path, tasks=tasks)],
        ALIKED_MODEL,
        min_score=min_score,
        max_num_features=max_num_features,
        batch_size=ALIKED_BATCH_SIZE,
        preprocessing_workers=preprocessing_workers,
        gpu_preprocessing=gpu_preprocessing,
    )
    elapsed_seconds: float = time.perf_counter() - started
    return keypoint_counts(database_path, [task.image_id for task in tasks]), elapsed_seconds


__all__: Final[list[str]] = [
    "ALIKED_BATCH_SIZE",
    "ALIKED_IMAGE_BINDING",
    "ALIKED_MODEL",
    "ALIKED_ONNX_PATH",
    "DEFAULT_GPU_PREPROCESSING",
    "DEFAULT_PREPROCESSING_WORKERS",
    "DESCRIPTOR_TYPE",
    "NETWORK_HEIGHT",
    "NETWORK_WIDTH",
    "ExtractionGroup",
    "FeatureBatch",
    "ImageTask",
    "NativeModel",
    "NetworkImage",
    "NormalizedKeypoints",
    "PreparedImage",
    "ResizedImageBGR",
    "extract_tensorrt",
    "image_tasks",
    "normalized_to_pixels",
    "preprocess_image",
    "run_native_extraction",
    "to_network_bchw",
]
"""The shared names `tools/audit/` and the tests still import from this module.

`colsfm.features_native` owns them now. Re-exporting rather than duplicating is
what keeps the audit tools — which are outside this package's public surface and
outside the pipeline — working against one implementation."""

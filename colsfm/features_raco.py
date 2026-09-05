"""RaCo-ALIKED feature extraction through a batched, dynamic-shape TensorRT engine.

The `raco` half of `colsfm.features`, and the third backend after `pycolmap`
(COLMAP's own ONNX ALIKED) and `tensorrt` (the blob's static batch-1 engine).

**What the model is.** RaCo-ALIKED is fabio-sim's rotation-and-scale-consistent
retraining of ALIKED, published with LightGlue-ONNX and the subject of the
"Discovering TensorRT Optimizations for RaCo-ALIKED-LightGlue+" write-up. The
graph this module runs is `data/cusfm_models/raco-aliked-b1-16.onnx`, exported
from that package by `tools/export_cusfm_raco.py --batched-extractor-path` in
the repo's `raco` Pixi environment (`lightglue-onnx` at `d12b4ba`). Nothing here
imports `onnx` or `torch`: the `colsfm` environment only parses the graph with
TensorRT's own `OnnxParser`, which is the same split `colsfm.tensorrt_runtime`
documents.

**What is different from `colsfm.features_trt`.** One thing, and it is the whole
point: the graph has a real batch axis, `image [batch, 3, 1200, 1920]` with
`batch` in 1..16, so the engine is built with an optimisation profile rather than
at the shipped static shape and `DEFAULT_BATCH_SIZE` images cross the PCIe bus
and the detector pyramid together. Everything either side of the engine is
unchanged and imported from `colsfm.features_trt`: the same OpenCV stretch
preprocessing, the same bounded decode-ahead thread pool, the same
`(k + 1) * 0.5 * (size - 1)` mapping back onto the original image, the same two
database columns written for image rows somebody else created.

**Scores are a selection order, not a detector response.** Upstream's boundary
ranker returns its top 2048 points already ordered but never materialises a
dense score map, so `tools/export_cusfm_raco.py` writes a strictly decreasing
`arange(2048, 0, -1) / 2048` in the `scores` output and stamps
`cusfm_scores=monotonic_upstream_selection_priority` into the ONNX metadata. The
smallest of those is `1/2048 = 0.00049`, which is *below* the blob's
`detector_threshold` of 0.005, so applying that gate here would throw away ten
points per image for no reason and for no meaning. `min_score` is therefore read
but compared against a rank, and `DEFAULT_MIN_SCORE` is 0: the honest reading is
that this graph has no score gate, and the matcher's SSC (which runs on
*LightGlue* scores, `colsfm.matching_raco`) is where selection actually happens.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray

from colsfm import REPO_ROOT
from colsfm.database import Descriptors, KeypointsXY, keypoint_counts
from colsfm.features import TensorRTExtraction
from colsfm.features_trt import (
    ALIKED_IMAGE_BINDING,
    DEFAULT_PREPROCESSING_WORKERS,
    DESCRIPTOR_TYPE,
    NETWORK_HEIGHT,
    NETWORK_WIDTH,
    ImageTask,
    PreparedImage,
    _image_tasks,
    _prepared_stream,
    normalized_to_pixels,
)
from colsfm.tensorrt_runtime import ShapeProfile, TensorRTSession, resolve_engine

NetworkBatch: TypeAlias = Float32[ndarray, "batch 3 network_height network_width"]
"""One batch of preprocessed images as the engine takes it: planar RGB in [0, 1]."""

RACO_ONNX_PATH: Final[Path] = REPO_ROOT / "data" / "cusfm_models" / "raco-aliked-b1-16.onnx"
"""The batch-dynamic RaCo-ALIKED graph; its engine is cached beside it.

Not committed — `data/cusfm_models` is gitignored and the graph is 8.9 MB of
weights. Produce it with
`pixi run -e raco raco-export --batched-extractor-path data/cusfm_models/raco-aliked-b1-16.onnx`."""

MINIMUM_BATCH_SIZE: Final[int] = 1
"""The optimisation profile's `min`, so a one-image run still executes."""

DEFAULT_BATCH_SIZE: Final[int] = 8
"""Images per execution, and the profile's `opt`; `tools/raco_extract.py`'s own default."""

MAXIMUM_BATCH_SIZE: Final[int] = 16
"""The profile's `max`, and the ceiling the ONNX graph itself was exported with."""


def raco_profile(network_height: int = NETWORK_HEIGHT, network_width: int = NETWORK_WIDTH) -> dict[str, ShapeProfile]:
    """The optimisation profile the batch-dynamic graph is built with.

    Args:
        network_height: The graph's input height.
        network_width: The graph's input width.

    Returns:
        One `(minimum, optimal, maximum)` shape for the `image` binding.
    """
    return {
        ALIKED_IMAGE_BINDING: (
            (MINIMUM_BATCH_SIZE, 3, network_height, network_width),
            (DEFAULT_BATCH_SIZE, 3, network_height, network_width),
            (MAXIMUM_BATCH_SIZE, 3, network_height, network_width),
        )
    }


def _batched(prepared: Iterator[PreparedImage], batch_size: int) -> Iterator[list[PreparedImage]]:
    """Group a prepared-image stream into fixed-size batches, the last one short.

    Args:
        prepared: Prepared images, in the order their results are wanted.
        batch_size: Images per batch.

    Yields:
        Non-empty lists of at most `batch_size` prepared images.
    """
    while batch := list(islice(prepared, max(batch_size, 1))):
        yield batch


def extract_raco(
    database_path: Path,
    image_root: Path,
    image_names: Sequence[str],
    *,
    min_score: float,
    max_num_features: int,
    onnx_path: Path = RACO_ONNX_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    preprocessing_workers: int = DEFAULT_PREPROCESSING_WORKERS,
) -> TensorRTExtraction:
    """Run the batched RaCo-ALIKED engine over the named images and store the features.

    Args:
        database_path: An existing COLMAP database whose image rows are written.
        image_root: Directory the `image_names` are relative to.
        image_names: Images to extract, as stored in the database.
        min_score: Gate on the graph's `scores` output. See the module docstring:
            that output is a selection rank, so anything above 0 truncates by
            rank rather than by detector response.
        max_num_features: Keypoint ceiling per image; the graph's own top-2048
            head normally binds first, so this only ever truncates.
        onnx_path: The batch-dynamic RaCo-ALIKED graph whose engine to use.
        batch_size: Images executed together, 1 to `MAXIMUM_BATCH_SIZE`.
        preprocessing_workers: Decode-and-resize threads.

    Returns:
        Keypoints stored per `image_id`, and the wall time of the whole pass
        including the engine load.

    Raises:
        FileNotFoundError: When the database, the image root, an image or the
            ONNX graph is missing.
        ValueError: When `batch_size` is outside the graph's profile.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"No image directory at {image_root}")
    if not MINIMUM_BATCH_SIZE <= batch_size <= MAXIMUM_BATCH_SIZE:
        raise ValueError(f"batch_size must be in [{MINIMUM_BATCH_SIZE}, {MAXIMUM_BATCH_SIZE}], got {batch_size}")

    started: float = time.perf_counter()
    tasks: list[ImageTask] = _image_tasks(database_path, image_root, image_names)
    engine_path: Path = resolve_engine(onnx_path, raco_profile())
    with TensorRTSession(engine_path) as session, pycolmap.Database.open(database_path) as database:
        declared_shape: tuple[int, ...] = tuple(session.engine.get_tensor_shape(ALIKED_IMAGE_BINDING))
        network_height: int = int(declared_shape[2])
        network_width: int = int(declared_shape[3])
        prepared: Iterator[PreparedImage] = _prepared_stream(tasks, preprocessing_workers, network_height, network_width)
        for batch in _batched(prepared, batch_size):
            images_bchw: NetworkBatch = np.ascontiguousarray(
                np.concatenate([item.network_bchw for item in batch], axis=0), dtype=np.float32
            )
            outputs: dict[str, ndarray] = session.run({ALIKED_IMAGE_BINDING: images_bchw})
            keypoints_bn2: Float32[ndarray, "batch num_keypoints 2"] = np.asarray(outputs["keypoints"], dtype=np.float32)
            descriptors_bnd: Float32[ndarray, "batch num_keypoints 128"] = np.asarray(outputs["descriptors"], dtype=np.float32)
            scores_bn: Float32[ndarray, "batch num_keypoints"] = np.asarray(outputs["scores"], dtype=np.float32)
            for index, item in enumerate(batch):
                scores: Float32[ndarray, " num_keypoints"] = scores_bn[index].reshape(-1)
                kept: Bool[ndarray, " num_keypoints"] = scores >= np.float32(min_score)
                if max_num_features > 0 and int(kept.sum()) > max_num_features:
                    strongest: Int64[ndarray, " num_selected"] = np.argsort(np.where(kept, scores, -np.inf))[::-1][
                        :max_num_features
                    ]
                    kept = np.zeros_like(kept)
                    kept[strongest] = True
                keypoints_xy: KeypointsXY = normalized_to_pixels(
                    keypoints_bn2[index].reshape(-1, 2)[kept], item.task.image_width, item.task.image_height
                )
                descriptors: Descriptors = np.ascontiguousarray(descriptors_bnd[index].reshape(len(scores), -1)[kept])
                database.write_keypoints(item.task.image_id, keypoints_xy)
                database.write_descriptors(
                    item.task.image_id,
                    pycolmap.FeatureDescriptorsFloat(data=descriptors, type=DESCRIPTOR_TYPE).to_bytes(),
                )
    elapsed_seconds: float = time.perf_counter() - started
    return keypoint_counts(database_path, [task.image_id for task in tasks]), elapsed_seconds

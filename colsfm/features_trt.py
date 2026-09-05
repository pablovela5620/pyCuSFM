"""ALIKED feature extraction through the blob's own TensorRT engine.

The `tensorrt` half of `colsfm.features`. Where the default backend hands the
images to `pycolmap.extract_features` and lets COLMAP run its own ONNX ALIKED,
this one runs **the blob's graph**, `pycusfm/models/aliked_lightglue/aliked.onnx`,
through the engine the blob itself built
(`aliked_fp16_10_13_3_9_sm_12_0.engine`, reused from the repo when the TensorRT
version and the GPU match — see `colsfm.tensorrt_runtime.engine_path_for`). The
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
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Final, TypeAlias

import cv2
import numpy as np
import pycolmap
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray

from colsfm.database import Descriptors, KeypointsXY, image_ids_by_name, keypoint_counts
from colsfm.features import TensorRTExtraction
from colsfm.tensorrt_runtime import MODEL_DIR, TensorRTSession, resolve_engine

NetworkImage: TypeAlias = Float32[ndarray, "1 3 network_height network_width"]
"""One preprocessed image as the engine takes it: planar RGB in [0, 1]."""

NormalizedKeypoints: TypeAlias = Float32[ndarray, "num_keypoints 2"]
"""ALIKED's own output coordinates, normalised to [-1, 1] over the network input."""

KeypointScores: TypeAlias = Float32[ndarray, " num_keypoints"]
"""ALIKED's detector response per keypoint."""

ALIKED_ONNX_PATH: Final[Path] = MODEL_DIR / "aliked.onnx"
"""The blob's own ALIKED graph; its engine is cached beside it."""

ALIKED_IMAGE_BINDING: Final[str] = "image"
"""The graph's single input."""

NETWORK_HEIGHT: Final[int] = 1200
"""`aliked.onnx`'s declared input height, and the blob's TensorRT profile `opt`."""

NETWORK_WIDTH: Final[int] = 1920
"""`aliked.onnx`'s declared input width."""

DESCRIPTOR_TYPE: Final[pycolmap.FeatureExtractorType] = pycolmap.FeatureExtractorType.ALIKED_N16ROT
"""What the descriptor blob is labelled as in the database.

The blob's `aliked.onnx` is ALIKED-n16 (`colsfm.features` module docstring), and
COLMAP's nearest first-class descriptor type is `ALIKED_N16ROT`. The label only
travels with the blob; nothing in this pipeline dispatches on it, and the
descriptors are recovered bit-exactly either way
(`docs/spec/pycolmap-capabilities.md` §4)."""

DEFAULT_PREPROCESSING_WORKERS: Final[int] = 8
"""Decode-and-resize threads, matching `tools/raco_extract.ExtractConfig`."""

PREPROCESSING_LOOKAHEAD: Final[int] = 2
"""Prepared frames held per worker, so the GPU never waits and memory stays bounded.

Each prepared frame is a 1920x1200x3 float32 tensor (27.6 MB) plus the decoded
source image, so an unbounded `ThreadPoolExecutor.map` over RoboCap's 4528 frames
would want ~150 GB. This window holds `workers * lookahead` of them."""


@dataclass(frozen=True, slots=True)
class ImageTask:
    """One image to extract, and where its row already lives."""

    image_id: int
    """The database `image_id`, owned by `colsfm.database.create_database`."""
    image_path: Path
    """Absolute path to the image file."""
    image_width: int
    """Original width in pixels, from the image's camera."""
    image_height: int
    """Original height in pixels, from the image's camera."""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """One decoded image and the tensor the engine takes."""

    task: ImageTask
    """Which image this is."""
    network_bchw: NetworkImage
    """Float32 planar RGB in [0, 1], stretched to the engine's input size."""


def preprocess_image(task: ImageTask, network_height: int = NETWORK_HEIGHT, network_width: int = NETWORK_WIDTH) -> PreparedImage:
    """Decode one image and reproduce the blob's OpenCV preprocessing.

    Args:
        task: The image to read and the size its camera declares.
        network_height: The engine's input height.
        network_width: The engine's input width.

    Returns:
        The image as the engine takes it.

    Raises:
        ValueError: When OpenCV cannot decode the file, or when the file's size
            disagrees with the camera the database holds for it — which would
            silently misplace every keypoint.
    """
    bgr_hwc: ndarray | None = cv2.imread(str(task.image_path), cv2.IMREAD_COLOR)
    if bgr_hwc is None:
        raise ValueError(f"OpenCV could not decode {task.image_path}")
    if bgr_hwc.shape[:2] != (task.image_height, task.image_width):
        raise ValueError(
            f"The database camera says {task.image_width}x{task.image_height}, but "
            f"{task.image_path} is {bgr_hwc.shape[1]}x{bgr_hwc.shape[0]}"
        )
    resized_bgr_hwc: ndarray = cv2.resize(bgr_hwc, (network_width, network_height), interpolation=cv2.INTER_LINEAR)
    network_chw: Float32[ndarray, "3 network_height network_width"] = np.ascontiguousarray(
        resized_bgr_hwc[..., ::-1].transpose(2, 0, 1), dtype=np.float32
    )
    network_chw /= np.float32(255.0)
    return PreparedImage(task=task, network_bchw=network_chw[None])


def _prepared_stream(tasks: Sequence[ImageTask], workers: int, network_height: int, network_width: int) -> Iterator[PreparedImage]:
    """Decode ahead of the GPU without letting the queue grow without bound.

    Args:
        tasks: The images, in the order their results are wanted.
        workers: Decode threads.
        network_height: The engine's input height.
        network_width: The engine's input width.

    Yields:
        Prepared images in `tasks` order.
    """
    window: int = max(workers, 1) * PREPROCESSING_LOOKAHEAD
    remaining: Iterator[ImageTask] = iter(tasks)
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        pending: deque[Future[PreparedImage]] = deque(
            pool.submit(preprocess_image, task, network_height, network_width) for task in islice(remaining, window)
        )
        while pending:
            yield pending.popleft().result()
            for task in islice(remaining, 1):
                pending.append(pool.submit(preprocess_image, task, network_height, network_width))


def normalized_to_pixels(normalized_xy: NormalizedKeypoints, image_width: int, image_height: int) -> KeypointsXY:
    """Map ALIKED's normalised coordinates onto the original image.

    `(k + 1) * 0.5 * (size - 1)`, the blob's own mapping
    (feature_extractor_main.md §6.3), which targets the **original** image size
    rather than the network's — the stretch is undone here and nowhere else.

    Args:
        normalized_xy: Float32 `[num_keypoints, 2]` coordinates in [-1, 1].
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        Float32 `[num_keypoints, 2]` xy pixel coordinates.
    """
    scale_xy: Float32[ndarray, " 2"] = np.asarray(
        [(image_width - 1) / 2.0, (image_height - 1) / 2.0], dtype=np.float32
    )
    return np.ascontiguousarray((normalized_xy + np.float32(1.0)) * scale_xy, dtype=np.float32)


def _image_tasks(database_path: Path, image_root: Path, image_names: Sequence[str]) -> list[ImageTask]:
    """Pair each requested image name with its database row and its camera size.

    Args:
        database_path: An existing database holding the image and camera rows.
        image_root: Directory the names are relative to.
        image_names: Images to extract, as stored in the database.

    Returns:
        One task per name, in the order given.

    Raises:
        KeyError: When a name has no image row.
        FileNotFoundError: When an image file is missing.
    """
    name_to_id: dict[str, int] = image_ids_by_name(database_path)
    with pycolmap.Database.open(database_path) as database:
        cameras: dict[int, pycolmap.Camera] = {camera.camera_id: camera for camera in database.read_all_cameras()}
        camera_id_by_image_id: dict[int, int] = {image.image_id: image.camera_id for image in database.read_all_images()}
    tasks: list[ImageTask] = []
    for name in image_names:
        if name not in name_to_id:
            raise KeyError(f"The database holds no image named {name}")
        image_id: int = name_to_id[name]
        camera: pycolmap.Camera = cameras[camera_id_by_image_id[image_id]]
        tasks.append(
            ImageTask(
                image_id=image_id,
                image_path=image_root / name,
                image_width=int(camera.width),
                image_height=int(camera.height),
            )
        )
    missing: list[Path] = [task.image_path for task in tasks if not task.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing; the first is {missing[0]}")
    return tasks


def extract_tensorrt(
    database_path: Path,
    image_root: Path,
    image_names: Sequence[str],
    *,
    min_score: float,
    max_num_features: int,
    onnx_path: Path = ALIKED_ONNX_PATH,
    preprocessing_workers: int = DEFAULT_PREPROCESSING_WORKERS,
) -> TensorRTExtraction:
    """Run the blob's ALIKED engine over the named images and store the features.

    Args:
        database_path: An existing COLMAP database whose image rows are written.
        image_root: Directory the `image_names` are relative to.
        image_names: Images to extract, as stored in the database.
        min_score: Detector score gate, applied with `>=` as the blob does.
        max_num_features: Keypoint ceiling per image; the graph's own top-2048
            head normally binds first, so this only ever truncates.
        onnx_path: The ALIKED graph whose engine to use.
        preprocessing_workers: Decode-and-resize threads.

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
    tasks: list[ImageTask] = _image_tasks(database_path, image_root, image_names)
    engine_path: Path = resolve_engine(onnx_path)
    with TensorRTSession(engine_path) as session, pycolmap.Database.open(database_path) as database:
        network_shape: tuple[int, ...] = tuple(session.engine.get_tensor_shape(ALIKED_IMAGE_BINDING))
        network_height: int = int(network_shape[2])
        network_width: int = int(network_shape[3])
        for prepared in _prepared_stream(tasks, preprocessing_workers, network_height, network_width):
            outputs: dict[str, ndarray] = session.run({ALIKED_IMAGE_BINDING: prepared.network_bchw})
            scores: KeypointScores = np.asarray(outputs["scores"], dtype=np.float32).reshape(-1)
            kept: Bool[ndarray, " num_keypoints"] = scores >= np.float32(min_score)
            if max_num_features > 0 and int(kept.sum()) > max_num_features:
                strongest: Int64[ndarray, " num_selected"] = np.argsort(np.where(kept, scores, -np.inf))[::-1][
                    :max_num_features
                ]
                kept = np.zeros_like(kept)
                kept[strongest] = True
            normalized_xy: NormalizedKeypoints = np.asarray(outputs["keypoints"], dtype=np.float32).reshape(-1, 2)[kept]
            descriptors: Descriptors = np.ascontiguousarray(
                np.asarray(outputs["descriptors"], dtype=np.float32).reshape(len(scores), -1)[kept]
            )
            keypoints_xy: KeypointsXY = normalized_to_pixels(
                normalized_xy, prepared.task.image_width, prepared.task.image_height
            )
            database.write_keypoints(prepared.task.image_id, keypoints_xy)
            database.write_descriptors(
                prepared.task.image_id,
                pycolmap.FeatureDescriptorsFloat(data=descriptors, type=DESCRIPTOR_TYPE).to_bytes(),
            )
    elapsed_seconds: float = time.perf_counter() - started
    return keypoint_counts(database_path, [task.image_id for task in tasks]), elapsed_seconds

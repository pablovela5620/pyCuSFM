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
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Final, TypeAlias

import cv2
import numpy as np
import pycolmap
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray

from colsfm.database import Descriptors, KeypointsXY, image_ids_by_name, keypoint_counts
from colsfm.features import TensorRTExtraction
from colsfm.tensorrt_runtime import MODEL_DIR, DeviceTensor, GpuPreprocessor, TensorRTSession, resolve_engine

NetworkImage: TypeAlias = Float32[ndarray, "1 3 network_height network_width"]
"""One preprocessed image as the engine takes it: planar RGB in [0, 1]."""

ResizedImageBGR: TypeAlias = UInt8[ndarray, "network_height network_width 3"]
"""One decoded image stretched to the engine's input size, still `cv2.imread`'s BGR HWC uint8."""

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

Each prepared frame is a 1920x1200x3 uint8 image (6.9 MB), or four times that on
the host-conversion path, so an unbounded `ThreadPoolExecutor.map` over RoboCap's
4528 frames would want tens of gigabytes. This window holds `workers * lookahead`
of them."""

DEFAULT_GPU_PREPROCESSING: Final[bool] = True
"""Whether the uint8-to-float32 conversion runs on the GPU rather than the host.

`False` restores the original host path — `[..., ::-1].transpose(2, 0, 1)` to
float32 and `/= 255` in the decode workers, then a pageable float32 upload — and
exists so the two can be measured against each other and so a machine whose
TensorRT will not build the four-layer preprocessing network still works."""


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
    """One decoded image, stretched to the engine's input size."""

    task: ImageTask
    """Which image this is."""
    resized_bgr_hwc: ResizedImageBGR
    """The decoded image at the engine's input size, still BGR HWC uint8.

    This is what the GPU path stages and uploads: 6.9 MB rather than the 27.6 MB
    of the float32 tensor, and no host arithmetic beyond the decode."""
    network_bchw: NetworkImage | None
    """Float32 planar RGB in [0, 1]; only the host-conversion path builds it."""

    def network_image(self) -> NetworkImage:
        """The float32 tensor, for callers that asked for host conversion.

        Returns:
            Float32 `[1, 3, height, width]` planar RGB in [0, 1].

        Raises:
            RuntimeError: When this image was prepared for the GPU path, which
                deliberately leaves the conversion to `GpuPreprocessor`.
        """
        if self.network_bchw is None:
            raise RuntimeError(f"{self.task.image_path} was prepared for the GPU path; no host tensor exists")
        return self.network_bchw


def to_network_bchw(resized_bgr_hwc: ResizedImageBGR) -> NetworkImage:
    """The blob's host-side conversion: BGR HWC uint8 to planar RGB float32 in [0, 1].

    Kept as its own function because it is the reference the GPU preprocessing
    engine is checked against, bit for bit
    (`tests/colsfm/test_features_trt_gpu_preprocess.py`).

    Args:
        resized_bgr_hwc: One decoded image at the engine's input size.

    Returns:
        Float32 `[1, 3, height, width]` planar RGB in [0, 1].
    """
    network_chw: Float32[ndarray, "3 network_height network_width"] = np.ascontiguousarray(
        resized_bgr_hwc[..., ::-1].transpose(2, 0, 1), dtype=np.float32
    )
    network_chw /= np.float32(255.0)
    return network_chw[None]


def preprocess_image(
    task: ImageTask,
    network_height: int = NETWORK_HEIGHT,
    network_width: int = NETWORK_WIDTH,
    convert_on_host: bool = False,
) -> PreparedImage:
    """Decode one image and reproduce the blob's OpenCV preprocessing.

    The `cv2.resize` is skipped when the file is already the engine's size, which
    every Galileo frame is; OpenCV's own resize does not shortcut that case and
    the copy costs 2.4 ms per image (docs/raco-speed-investigation.md §4).
    `INTER_LINEAR` at a 1:1 scale is an identity, so skipping it changes nothing.

    Args:
        task: The image to read and the size its camera declares.
        network_height: The engine's input height.
        network_width: The engine's input width.
        convert_on_host: Also build the float32 planar RGB tensor here, in this
            worker thread. The GPU path leaves it unset and uploads the uint8.

    Returns:
        The decoded image at the engine's input size.

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
    if bgr_hwc.shape[:2] == (network_height, network_width):
        resized_bgr_hwc: ResizedImageBGR = bgr_hwc
    else:
        resized_bgr_hwc = cv2.resize(bgr_hwc, (network_width, network_height), interpolation=cv2.INTER_LINEAR)
    return PreparedImage(
        task=task,
        resized_bgr_hwc=resized_bgr_hwc,
        network_bchw=to_network_bchw(resized_bgr_hwc) if convert_on_host else None,
    )


def _prepared_stream(
    tasks: Sequence[ImageTask], workers: int, network_height: int, network_width: int, convert_on_host: bool = False
) -> Iterator[PreparedImage]:
    """Decode ahead of the GPU without letting the queue grow without bound.

    Args:
        tasks: The images, in the order their results are wanted.
        workers: Decode threads.
        network_height: The engine's input height.
        network_width: The engine's input width.
        convert_on_host: Whether the workers also build the float32 tensor.

    Yields:
        Prepared images in `tasks` order.
    """
    window: int = max(workers, 1) * PREPROCESSING_LOOKAHEAD
    remaining: Iterator[ImageTask] = iter(tasks)
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        pending: deque[Future[PreparedImage]] = deque(
            pool.submit(preprocess_image, task, network_height, network_width, convert_on_host)
            for task in islice(remaining, window)
        )
        while pending:
            yield pending.popleft().result()
            for task in islice(remaining, 1):
                pending.append(pool.submit(preprocess_image, task, network_height, network_width, convert_on_host))


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
    gpu_preprocessing: bool = DEFAULT_GPU_PREPROCESSING,
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
    tasks: list[ImageTask] = _image_tasks(database_path, image_root, image_names)
    engine_path: Path = resolve_engine(onnx_path)
    with ExitStack() as stack:
        session: TensorRTSession = stack.enter_context(TensorRTSession(engine_path, reuse_output_buffers=True))
        database: pycolmap.Database = stack.enter_context(pycolmap.Database.open(database_path))
        network_shape: tuple[int, ...] = tuple(session.engine.get_tensor_shape(ALIKED_IMAGE_BINDING))
        network_height: int = int(network_shape[2])
        network_width: int = int(network_shape[3])
        preprocessor: GpuPreprocessor | None = (
            stack.enter_context(
                GpuPreprocessor(
                    height=network_height, width=network_width, max_batch=1, optimal_batch=1, stream=session.stream
                )
            )
            if gpu_preprocessing
            else None
        )
        for prepared in _prepared_stream(
            tasks, preprocessing_workers, network_height, network_width, not gpu_preprocessing
        ):
            network_input: NetworkImage | DeviceTensor
            if preprocessor is None:
                network_input = prepared.network_image()
            else:
                preprocessor.staging_bhwc[0] = prepared.resized_bgr_hwc
                network_input = preprocessor.run(1)
            outputs: dict[str, ndarray] = session.run({ALIKED_IMAGE_BINDING: network_input})
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

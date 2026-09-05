"""Measure the GPU preprocessing change on the colsfm TensorRT feature backends.

Runs the extraction stage over a whole dataset four ways — `raco` and `tensorrt`,
each with `gpu_preprocessing` on and off — and then breaks the per-image cost into
the parts that are left: the CPU decode and resize, the host-to-device transfer,
and the engine itself. The point is to answer one question with numbers rather
than logs: after the transpose, the widen and the `/255` move onto the stream,
what is the extraction stage actually made of?

Run it from the repo root:

```
PYTHONPATH=. pixi run -e colsfm python -m tools.audit.gpu_preprocess_bench
PYTHONPATH=. pixi run -e colsfm python -m tools.audit.gpu_preprocess_bench --dataset kitti --image-count 50
```

`docs/gpu-preprocessing.md` holds the numbers this produced on the RTX 5090.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
import tyro
from jaxtyping import Float32, UInt8
from numpy import ndarray

from colsfm import REPO_ROOT
from colsfm.database import create_database
from colsfm.features import TensorRTExtraction
from colsfm.features_raco import DEFAULT_BATCH_SIZE, RACO_ONNX_PATH, extract_raco, raco_profile
from colsfm.features_trt import (
    ALIKED_IMAGE_BINDING,
    ALIKED_ONNX_PATH,
    ImageTask,
    PreparedImage,
    preprocess_image,
    to_network_bchw,
)
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.tensorrt_runtime import DeviceTensor, GpuPreprocessor, TensorRTSession, resolve_engine

DatasetChoice: TypeAlias = Literal["galileo", "kitti"]
"""Which image set to time; `kitti` is the small-image case, upscaled to the engine."""

BackendChoice: TypeAlias = Literal["raco", "tensorrt"]
"""Which TensorRT feature backend to time."""

GALILEO_DIR: Path = REPO_ROOT / "data" / "r2b_galileo"
"""The 226-frame rig sample the pipeline is benchmarked on."""

KITTI_IMAGE_DIR: Path = REPO_ROOT / "data" / "kitti" / "06" / "image_0"
"""KITTI 06's left camera: 1226x370 PNGs, 5.1x smaller than the engine's input."""

MICRO_REPEATS: int = 3
"""Passes over the micro-benchmarks; the median is reported."""


@dataclass(frozen=True, slots=True)
class BenchConfig:
    """What to measure."""

    dataset: DatasetChoice = "galileo"
    """Which image set to run."""
    image_count: int = 0
    """How many images to use; 0 means all of them."""
    backends: tuple[BackendChoice, ...] = ("raco", "tensorrt")
    """Which backends to time end to end. Empty skips the stage runs."""
    batch_size: int = DEFAULT_BATCH_SIZE
    """Images per `extract_raco` execution."""
    repeats: int = 2
    """Stage runs per configuration; the fastest is reported.

    The machine shares its GPU with whatever else is running, and a stage that
    lost a scheduling slot is not evidence about this change. The minimum is the
    least contaminated sample, not the typical one, and it is reported as such."""


@dataclass(frozen=True, slots=True)
class StageResult:
    """One extraction-stage run."""

    backend: BackendChoice
    """Which backend ran."""
    gpu_preprocessing: bool
    """Whether the conversion ran on the stream."""
    seconds: float
    """Wall time of the stage, as the backend itself reports it."""
    images: int
    """Images extracted."""


def _galileo_tasks(image_count: int) -> tuple[FramesMeta, Path, list[str]]:
    """The Galileo keyframes, its image root and the image names.

    Args:
        image_count: How many keyframes to keep; 0 keeps all.

    Returns:
        The (possibly filtered) collection, the image root, and the names.
    """
    meta: FramesMeta = read_frames_meta(GALILEO_DIR / "frames_meta.json")
    if image_count:
        meta = meta.filtered([keyframe.keyframe_id for keyframe in meta.keyframes[:image_count]])
    return meta, GALILEO_DIR, [keyframe.image_name for keyframe in meta.keyframes]


def _image_tasks_for(dataset: DatasetChoice, image_count: int) -> list[ImageTask]:
    """Standalone decode tasks, for the micro-benchmarks that need no database.

    Args:
        dataset: Which image set.
        image_count: How many images; 0 means all.

    Returns:
        One task per image, with the file's own size as the camera size.
    """
    if dataset == "galileo":
        _meta, root, names = _galileo_tasks(image_count)
        paths: list[Path] = [root / name for name in names]
    else:
        paths = sorted(KITTI_IMAGE_DIR.glob("*.png"))
        if image_count:
            paths = paths[:image_count]
    tasks: list[ImageTask] = []
    for index, path in enumerate(paths):
        probe: ndarray | None = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if probe is None:
            raise ValueError(f"OpenCV could not decode {path}")
        tasks.append(
            ImageTask(image_id=index + 1, image_path=path, image_width=int(probe.shape[1]), image_height=int(probe.shape[0]))
        )
    return tasks


def run_stage(backend: BackendChoice, config: BenchConfig, *, gpu_preprocessing: bool) -> StageResult:
    """Run one extraction stage over the dataset and report its wall time.

    Args:
        backend: Which backend to run.
        config: What to measure.
        gpu_preprocessing: Whether the conversion runs on the stream.

    Returns:
        The stage's wall time and image count.
    """
    meta, root, names = _galileo_tasks(config.image_count)
    database_path: Path = Path(f"/tmp/gpu_preprocess_bench_{backend}_{gpu_preprocessing}.db")
    database_path.unlink(missing_ok=True)
    create_database(database_path, meta)
    if backend == "raco":
        extraction: TensorRTExtraction = extract_raco(
            database_path,
            root,
            names,
            min_score=0.0,
            max_num_features=0,
            batch_size=config.batch_size,
            gpu_preprocessing=gpu_preprocessing,
        )
    else:
        from colsfm.features_trt import extract_tensorrt

        extraction = extract_tensorrt(
            database_path, root, names, min_score=0.005, max_num_features=0, gpu_preprocessing=gpu_preprocessing
        )
    database_path.unlink(missing_ok=True)
    return StageResult(backend=backend, gpu_preprocessing=gpu_preprocessing, seconds=extraction[1], images=len(names))


def _median(values: list[float]) -> float:
    """The median of a short list.

    Args:
        values: The samples.

    Returns:
        The middle value after sorting.
    """
    return float(np.median(np.asarray(values, dtype=np.float64)))


def measure_host_costs(tasks: list[ImageTask], height: int, width: int) -> dict[str, float]:
    """Time the host half, single-threaded, in milliseconds per image.

    Args:
        tasks: The images to read.
        height: The engine's input height.
        width: The engine's input width.

    Returns:
        Milliseconds per image for the decode, the resize and the old float32
        conversion, plus the staging copy the GPU path does instead.
    """
    decode_seconds: float = 0.0
    resize_seconds: float = 0.0
    convert_seconds: float = 0.0
    copy_seconds: float = 0.0
    slot: UInt8[ndarray, "height width 3"] = np.zeros((height, width, 3), dtype=np.uint8)
    for task in tasks:
        started: float = time.perf_counter()
        bgr_hwc: ndarray | None = cv2.imread(str(task.image_path), cv2.IMREAD_COLOR)
        decode_seconds += time.perf_counter() - started
        if bgr_hwc is None:
            raise ValueError(f"OpenCV could not decode {task.image_path}")
        started = time.perf_counter()
        resized: UInt8[ndarray, "height width 3"] = (
            bgr_hwc
            if bgr_hwc.shape[:2] == (height, width)
            else cv2.resize(bgr_hwc, (width, height), interpolation=cv2.INTER_LINEAR)
        )
        resize_seconds += time.perf_counter() - started
        started = time.perf_counter()
        network: Float32[ndarray, "1 3 height width"] = to_network_bchw(resized)
        convert_seconds += time.perf_counter() - started
        assert network.shape[1] == 3
        started = time.perf_counter()
        np.copyto(slot, resized)
        copy_seconds += time.perf_counter() - started
    scale: float = 1000.0 / len(tasks)
    return {
        "decode": decode_seconds * scale,
        "resize": resize_seconds * scale,
        "host float32 conversion": convert_seconds * scale,
        "staging copy into pinned": copy_seconds * scale,
    }


def measure_device_costs(tasks: list[ImageTask], batch_size: int) -> dict[str, float]:
    """Time the transfer and the engine, in milliseconds per image.

    Args:
        tasks: Enough images to fill one batch; only the first `batch_size` are used.
        batch_size: Images per execution.

    Returns:
        Milliseconds per image for the uint8 upload plus preprocessing engine, the
        old float32 upload, and the detector engine on top of each.
    """
    engine_path: Path = resolve_engine(RACO_ONNX_PATH, raco_profile())
    prepared: list[PreparedImage] = [preprocess_image(task, convert_on_host=True) for task in tasks[:batch_size]]
    height: int = int(prepared[0].resized_bgr_hwc.shape[0])
    width: int = int(prepared[0].resized_bgr_hwc.shape[1])
    host_batch: Float32[ndarray, "batch 3 height width"] = np.ascontiguousarray(
        np.concatenate([item.network_image() for item in prepared], axis=0), dtype=np.float32
    )
    timings: dict[str, list[float]] = {"gpu path (upload + preprocess + engine)": [], "host path (upload + engine)": []}
    with TensorRTSession(engine_path, reuse_output_buffers=True) as session, GpuPreprocessor(
        height=height, width=width, max_batch=batch_size, optimal_batch=batch_size, stream=session.stream
    ) as preprocessor:
        for slot, item in enumerate(prepared):
            preprocessor.staging_bhwc[slot] = item.resized_bgr_hwc
        for repeat in range(MICRO_REPEATS + 1):
            started: float = time.perf_counter()
            device: DeviceTensor = preprocessor.run(batch_size)
            session.run({ALIKED_IMAGE_BINDING: device})
            gpu_seconds: float = time.perf_counter() - started
            started = time.perf_counter()
            session.run({ALIKED_IMAGE_BINDING: host_batch})
            host_seconds: float = time.perf_counter() - started
            if repeat == 0:
                continue
            timings["gpu path (upload + preprocess + engine)"].append(gpu_seconds)
            timings["host path (upload + engine)"].append(host_seconds)
    scale: float = 1000.0 / batch_size
    return {name: _median(samples) * scale for name, samples in timings.items()}


def main(config: BenchConfig) -> None:
    """Run the stage comparison and the breakdown, printing a table.

    Args:
        config: What to measure.
    """
    tasks: list[ImageTask] = _image_tasks_for(config.dataset, config.image_count)
    print(f"{config.dataset}: {len(tasks)} images, batch {config.batch_size}")

    if config.dataset == "galileo" and config.backends:
        if not RACO_ONNX_PATH.is_file():
            print(f"skipping the raco stage: {RACO_ONNX_PATH.name} is missing")
        if not ALIKED_ONNX_PATH.is_file():
            print(f"skipping the tensorrt stage: {ALIKED_ONNX_PATH.name} is missing")
        print("\nextraction stage, seconds")
        for backend in config.backends:
            before: float = min(
                run_stage(backend, config, gpu_preprocessing=False).seconds for _ in range(config.repeats)
            )
            after: float = min(
                run_stage(backend, config, gpu_preprocessing=True).seconds for _ in range(config.repeats)
            )
            print(f"  {backend:9s} host {before:6.2f} s   gpu {after:6.2f} s   {before / max(after, 1e-9):.2f}x")

    print("\nhost cost, ms/image, single-threaded")
    for name, milliseconds in measure_host_costs(tasks, 1200, 1920).items():
        print(f"  {name:28s} {milliseconds:6.2f}")

    if RACO_ONNX_PATH.is_file():
        print("\ndevice cost, ms/image, batch of eight")
        for name, milliseconds in measure_device_costs(tasks, config.batch_size).items():
            print(f"  {name:38s} {milliseconds:6.2f}")

    bytes_uint8: int = 1200 * 1920 * 3
    print(f"\nhost-to-device bytes per image: uint8 {bytes_uint8 / 2**20:.1f} MiB, float32 {4 * bytes_uint8 / 2**20:.1f} MiB")


if __name__ == "__main__":
    main(tyro.cli(BenchConfig))

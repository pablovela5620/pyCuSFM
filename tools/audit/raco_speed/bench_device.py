"""Device-time benchmark and stage profile for the colsfm RaCo TensorRT path.

Three questions, one script, all of it run in the `colsfm` environment because
that is where the TensorRT engines and `pycolmap` live:

* `engines` — median device time of our RaCo extractor engine (batch 1 and 8),
  our RaCo LightGlue+ matcher engine (one pair per call), and the blob's stock
  ALIKED and LightGlue engines, with the host copies separated from the compute.
* `transfer` — how much of the extractor's per-image cost is the pageable
  host-to-device copy of a float32 `[3, 1200, 1920]` tensor, and what pinned
  memory or a uint8 upload would save.
* `stage` — where the Galileo extraction stage's wall time actually goes:
  JPEG decode, resize, layout conversion, engine execution, post-processing and
  the database write.

Nothing here imports from a module another worker is editing beyond the frozen
helpers in `colsfm.features_trt`; the stage timings re-implement the loop rather
than instrument it in place.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import cv2
import numpy as np
import tensorrt as trt
import tyro
from cuda.bindings import runtime as cudart
from jaxtyping import Float32, UInt8
from numpy import ndarray

from colsfm.tensorrt_runtime import TensorRTSession, check_cuda

BenchMode: TypeAlias = Literal["engines", "transfer", "stage"]
"""Which of the three measurements to run."""

NetworkImage: TypeAlias = Float32[ndarray, "3 height width"]
"""One preprocessed image, planar RGB in [0, 1]."""

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
"""Repository root, three levels above `tools/audit/raco_speed`."""

RACO_EXTRACTOR_ENGINE: Final[Path] = REPO_ROOT / "data" / "cusfm_models" / "raco-aliked-b1-16_fp16_10_13_3_9_sm_12_0.engine"
"""Our batch-dynamic RaCo-ALIKED engine."""

RACO_MATCHER_ENGINE: Final[Path] = (
    REPO_ROOT / "data" / "cusfm_models" / "raco" / "aliked_lightglue" / "lightglue_aliked_fp16_10_13_3_9_sm_12_0.engine"
)
"""Our RaCo-trained LightGlue+ engine, one pair per call."""

STOCK_EXTRACTOR_ENGINE: Final[Path] = REPO_ROOT / "pycusfm" / "models" / "aliked_lightglue" / "aliked_fp16_10_13_3_9_sm_12_0.engine"
"""The blob's stock ALIKED engine, static batch one."""

STOCK_MATCHER_ENGINE: Final[Path] = (
    REPO_ROOT / "pycusfm" / "models" / "aliked_lightglue" / "lightglue_aliked_fp16_10_13_3_9_sm_12_0.engine"
)
"""The blob's stock LightGlue engine, one pair per call."""

NUM_KEYPOINTS: Final[int] = 2048
"""Keypoints per image, the budget both extractor graphs emit."""


@dataclass(slots=True)
class BenchConfig:
    """What to measure."""

    mode: BenchMode = "engines"
    """Which measurement to run."""
    image_root: Path = REPO_ROOT / "data" / "r2b_galileo"
    """Galileo sample root; every `.jpeg` beneath it is a candidate frame."""
    image_count: int = 226
    """Frames used by the stage profile, matching the pipeline's keyframe count."""
    repeats: int = 30
    """Timed executions per configuration."""
    warmups: int = 5
    """Untimed executions per configuration."""
    batch_size: int = 8
    """Images per extractor execution in the stage profile."""
    output_path: Path = Path("/tmp/raco_speed/colsfm.json")
    """Where the measurements are written."""


def _device_median_ms(
    session: TensorRTSession, inputs: dict[str, ndarray], repeats: int, warmups: int
) -> tuple[float, float]:
    """Time one engine with and without the host copies.

    Args:
        session: A live session over the engine.
        inputs: One contiguous host array per input binding.
        repeats: Timed executions.
        warmups: Untimed executions.

    Returns:
        `(median milliseconds including host copies, median milliseconds of
        `execute_async_v3` alone with the inputs already resident)`.
    """
    for _ in range(warmups):
        session.run(inputs)
    full_ms: list[float] = []
    for _ in range(repeats):
        started: float = time.perf_counter()
        session.run(inputs)
        full_ms.append((time.perf_counter() - started) * 1e3)
    compute_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        if not session.context.execute_async_v3(session.stream):
            raise RuntimeError("TensorRT execution failed")
        check_cuda(cudart.cudaStreamSynchronize(session.stream), "cudaStreamSynchronize")
        compute_ms.append((time.perf_counter() - started) * 1e3)
    return float(np.median(full_ms)), float(np.median(compute_ms))


def _extractor_inputs(session: TensorRTSession, batch_size: int) -> dict[str, ndarray]:
    """Build a random batch at the engine's declared spatial shape.

    Args:
        session: A live extractor session.
        batch_size: Images in the batch.

    Returns:
        The single `image` binding, float32 and C-contiguous.
    """
    name: str = session.input_names[0]
    shape: tuple[int, ...] = tuple(session.engine.get_tensor_shape(name))
    rng: np.random.Generator = np.random.default_rng(0)
    return {name: rng.random((batch_size, 3, int(shape[2]), int(shape[3])), dtype=np.float32)}


def _matcher_inputs(session: TensorRTSession) -> dict[str, ndarray]:
    """Build one pair of keypoint and descriptor tensors for a LightGlue engine.

    Args:
        session: A live matcher session.

    Returns:
        One contiguous host array per input binding.
    """
    rng: np.random.Generator = np.random.default_rng(0)
    inputs: dict[str, ndarray] = {}
    for name in session.input_names:
        channels: int = 2 if name.startswith("kpts") else 128
        inputs[name] = rng.random((1, NUM_KEYPOINTS, channels), dtype=np.float32) * np.float32(2.0) - np.float32(1.0)
    return inputs


def run_engines(config: BenchConfig) -> list[dict[str, object]]:
    """Measure every engine on this GPU.

    Args:
        config: Repeat and warm-up counts.

    Returns:
        One record per configuration.
    """
    records: list[dict[str, object]] = []
    for label, engine_path, batch_sizes in (
        ("raco extractor", RACO_EXTRACTOR_ENGINE, (1, 4, 8)),
        ("stock extractor", STOCK_EXTRACTOR_ENGINE, (1,)),
    ):
        if not engine_path.is_file():
            print(f"skipping {label}: no engine at {engine_path}")
            continue
        with TensorRTSession(engine_path) as session:
            for batch_size in batch_sizes:
                inputs: dict[str, ndarray] = _extractor_inputs(session, batch_size)
                full_ms, compute_ms = _device_median_ms(session, inputs, config.repeats, config.warmups)
                records.append(
                    {
                        "label": f"{label} B={batch_size}",
                        "ms_per_call": full_ms,
                        "ms_per_call_compute_only": compute_ms,
                        "ms_per_image": full_ms / batch_size,
                        "ms_per_image_compute_only": compute_ms / batch_size,
                        "ms_per_pair": 2.0 * full_ms / batch_size,
                    }
                )
                print(f"{records[-1]['label']:26s} call {full_ms:7.2f} ms  compute {compute_ms:7.2f} ms  per-image {full_ms / batch_size:6.2f} ms", flush=True)
    for label, engine_path in (("raco matcher", RACO_MATCHER_ENGINE), ("stock matcher", STOCK_MATCHER_ENGINE)):
        if not engine_path.is_file():
            print(f"skipping {label}: no engine at {engine_path}")
            continue
        with TensorRTSession(engine_path) as session:
            inputs = _matcher_inputs(session)
            full_ms, compute_ms = _device_median_ms(session, inputs, config.repeats, config.warmups)
            records.append(
                {
                    "label": f"{label} 1 pair",
                    "ms_per_call": full_ms,
                    "ms_per_call_compute_only": compute_ms,
                    "ms_per_pair": full_ms,
                    "ms_per_pair_compute_only": compute_ms,
                }
            )
            print(f"{records[-1]['label']:26s} call {full_ms:7.2f} ms  compute {compute_ms:7.2f} ms", flush=True)
    return records


def run_transfer(config: BenchConfig) -> list[dict[str, object]]:
    """Measure the host-to-device cost of one extractor batch three ways.

    Args:
        config: Repeat and warm-up counts, and the batch size.

    Returns:
        One record per transfer strategy.
    """
    height: int = 1200
    width: int = 1920
    batch_size: int = config.batch_size
    rng: np.random.Generator = np.random.default_rng(0)
    pageable: Float32[ndarray, "batch 3 height width"] = rng.random((batch_size, 3, height, width), dtype=np.float32)
    bytes_float32: int = pageable.nbytes
    bytes_uint8: int = batch_size * 3 * height * width
    stream: int = int(check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")[0])
    device_pointer: int = int(check_cuda(cudart.cudaMalloc(bytes_float32), "cudaMalloc")[0])
    pinned_pointer: int = int(check_cuda(cudart.cudaMallocHost(bytes_float32), "cudaMallocHost")[0])

    def _time(source: int, byte_count: int) -> float:
        for _ in range(config.warmups):
            check_cuda(
                cudart.cudaMemcpyAsync(device_pointer, source, byte_count, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync",
            )
            check_cuda(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
        samples_ms: list[float] = []
        for _ in range(config.repeats):
            started: float = time.perf_counter()
            check_cuda(
                cudart.cudaMemcpyAsync(device_pointer, source, byte_count, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync",
            )
            check_cuda(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            samples_ms.append((time.perf_counter() - started) * 1e3)
        return float(np.median(samples_ms))

    records: list[dict[str, object]] = [
        {
            "label": f"pageable float32 B={batch_size}",
            "megabytes": bytes_float32 / 2**20,
            "ms_per_call": _time(pageable.ctypes.data, bytes_float32),
        },
        {
            "label": f"pinned float32 B={batch_size}",
            "megabytes": bytes_float32 / 2**20,
            "ms_per_call": _time(pinned_pointer, bytes_float32),
        },
        {
            "label": f"pinned uint8 B={batch_size}",
            "megabytes": bytes_uint8 / 2**20,
            "ms_per_call": _time(pinned_pointer, bytes_uint8),
        },
    ]
    for record in records:
        record["ms_per_image"] = float(record["ms_per_call"]) / batch_size
        print(f"{record['label']:28s} {record['megabytes']:7.1f} MB  {record['ms_per_call']:7.2f} ms  per-image {record['ms_per_image']:5.2f} ms", flush=True)
    check_cuda(cudart.cudaFreeHost(pinned_pointer), "cudaFreeHost")
    check_cuda(cudart.cudaFree(device_pointer), "cudaFree")
    check_cuda(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")
    return records


def _galileo_images(image_root: Path, image_count: int) -> list[Path]:
    """The first `image_count` JPEGs under the Galileo sample, in a stable order.

    Args:
        image_root: The sample root.
        image_count: How many frames to take.

    Returns:
        Sorted image paths.

    Raises:
        FileNotFoundError: When the sample holds fewer frames than requested.
    """
    paths: list[Path] = sorted(image_root.rglob("*.jpeg"))
    if len(paths) < image_count:
        raise FileNotFoundError(f"{image_root} holds {len(paths)} JPEGs, fewer than the {image_count} requested")
    return paths[:image_count]


def run_stage(config: BenchConfig) -> list[dict[str, object]]:
    """Profile the Galileo extraction stage phase by phase, single threaded.

    The colsfm stage overlaps decode with execution across a thread pool, so
    these are the serial costs of each phase rather than the stage's wall time;
    what they answer is which phase dominates and by how much.

    Args:
        config: The image root, frame count and batch size.

    Returns:
        One record per phase, in milliseconds per image.
    """
    paths: list[Path] = _galileo_images(config.image_root, config.image_count)
    height: int = 1200
    width: int = 1920
    totals: dict[str, float] = {
        "decode": 0.0,
        "resize": 0.0,
        "layout": 0.0,
        "scale": 0.0,
        "concatenate": 0.0,
        "execute": 0.0,
        "postprocess": 0.0,
    }
    with TensorRTSession(RACO_EXTRACTOR_ENGINE) as session:
        name: str = session.input_names[0]
        rng: np.random.Generator = np.random.default_rng(0)
        session.run({name: rng.random((config.batch_size, 3, height, width), dtype=np.float32)})
        for start_index in range(0, len(paths), config.batch_size):
            chunk: list[Path] = paths[start_index : start_index + config.batch_size]
            prepared: list[NetworkImage] = []
            for path in chunk:
                started: float = time.perf_counter()
                bgr_hwc: UInt8[ndarray, "h w 3"] | None = cv2.imread(str(path), cv2.IMREAD_COLOR)
                totals["decode"] += time.perf_counter() - started
                if bgr_hwc is None:
                    raise ValueError(f"OpenCV could not decode {path}")
                started = time.perf_counter()
                resized_bgr_hwc: UInt8[ndarray, "height width 3"] = cv2.resize(
                    bgr_hwc, (width, height), interpolation=cv2.INTER_LINEAR
                )
                totals["resize"] += time.perf_counter() - started
                started = time.perf_counter()
                network_chw: NetworkImage = np.ascontiguousarray(
                    resized_bgr_hwc[..., ::-1].transpose(2, 0, 1), dtype=np.float32
                )
                totals["layout"] += time.perf_counter() - started
                started = time.perf_counter()
                network_chw /= np.float32(255.0)
                totals["scale"] += time.perf_counter() - started
                prepared.append(network_chw)
            started = time.perf_counter()
            batch: Float32[ndarray, "batch 3 height width"] = np.ascontiguousarray(
                np.concatenate([item[None] for item in prepared], axis=0), dtype=np.float32
            )
            totals["concatenate"] += time.perf_counter() - started
            started = time.perf_counter()
            outputs: dict[str, ndarray] = session.run({name: batch})
            totals["execute"] += time.perf_counter() - started
            started = time.perf_counter()
            keypoints_bn2: Float32[ndarray, "batch n 2"] = np.asarray(outputs["keypoints"], dtype=np.float32)
            descriptors_bnd: Float32[ndarray, "batch n 128"] = np.asarray(outputs["descriptors"], dtype=np.float32)
            for index in range(batch.shape[0]):
                _ = np.ascontiguousarray(descriptors_bnd[index].reshape(NUM_KEYPOINTS, -1))
                _ = np.ascontiguousarray((keypoints_bn2[index].reshape(-1, 2) + np.float32(1.0)) * np.float32(0.5))
            totals["postprocess"] += time.perf_counter() - started

    records: list[dict[str, object]] = [
        {"label": phase, "seconds_total": seconds, "ms_per_image": seconds / len(paths) * 1e3}
        for phase, seconds in totals.items()
    ]
    records.sort(key=lambda record: float(record["seconds_total"]), reverse=True)
    for record in records:
        print(f"{record['label']:14s} {record['seconds_total']:7.3f} s total  {record['ms_per_image']:7.2f} ms/image", flush=True)
    print(f"serial sum     {sum(totals.values()):7.3f} s over {len(paths)} images", flush=True)
    return records


def main(config: BenchConfig) -> None:
    """Run the requested measurement and write it as JSON.

    Args:
        config: What to measure and where to write it.
    """
    print(f"TensorRT {trt.__version__}, mode {config.mode}")
    if config.mode == "engines":
        records: list[dict[str, object]] = run_engines(config)
    elif config.mode == "transfer":
        records = run_transfer(config)
    else:
        records = run_stage(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(records, indent=2))
    print(f"wrote {config.output_path}")


if __name__ == "__main__":
    main(tyro.cli(BenchConfig))

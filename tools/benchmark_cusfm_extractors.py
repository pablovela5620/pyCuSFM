"""Build and benchmark cuSFM extractor ONNX models with TensorRT 10.13."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
import tensorrt as trt
import tyro
from cuda.bindings import runtime as cudart

from tools.trt_runtime import check_cuda, percentile, sha256_file

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
"""Repo root, so defaults resolve regardless of the working directory."""
from jaxtyping import Float32, UInt8
from numpy import ndarray
from numpy.typing import NDArray

ImageCHW: TypeAlias = Float32[ndarray, "3 h w"]
ImagesBCHW: TypeAlias = Float32[ndarray, "batch 3 h w"]
KeypointsN2: TypeAlias = Float32[ndarray, "num_keypoints 2"]
DescriptorsND: TypeAlias = Float32[ndarray, "num_keypoints descriptor_dim"]
ScoresN: TypeAlias = Float32[ndarray, "num_keypoints"]  # noqa: F821


@dataclass
class BenchmarkConfig:
    """TensorRT extractor benchmark settings."""

    baseline: Path = Path("pycusfm/models/aliked_lightglue/aliked.onnx")
    """Frozen cuSFM ALIKED ONNX model."""
    optimized: Path = Path("data/cusfm_models/raco/aliked_lightglue/aliked.onnx")
    """RaCo-ALIKED replacement ONNX model."""
    input_dir: Path = Path("data/cusfm_runs/robocap_full/input")
    """Image tree used by the measured RoboCap baseline."""
    engine_dir: Path = REPO_ROOT / "data" / "cusfm_models" / "engines"
    """Content-addressed TensorRT engine cache (gitignored; shared with raco_extract)."""
    output: Path = REPO_ROOT / "data" / "cusfm_models" / "extractor-benchmark.json"
    """Machine-readable benchmark result (gitignored)."""
    image_count: int = 4
    """Number of real images cycled through the timed iterations."""
    warmup_iterations: int = 30
    """Untimed executions before measurement."""
    iterations: int = 100
    """Timed executions used to compute the median."""
    workspace_gib: int = 8
    """TensorRT tactic workspace ceiling in GiB."""


@dataclass(frozen=True)
class ModelResult:
    """Steady-state batch-one TensorRT measurements and output diagnostics."""

    label: str
    """Model label."""
    onnx_path: str
    """Source ONNX path."""
    onnx_sha256: str
    """Source graph digest."""
    engine_path: str
    """Serialized TensorRT engine path."""
    engine_bytes: int
    """Serialized engine size."""
    precision: str
    """Requested TensorRT precision policy."""
    batch_size: int
    """Static batch size."""
    warmup_iterations: int
    """Warmup count."""
    timed_iterations: int
    """Measured count."""
    median_ms: float
    """Median device execution latency."""
    p10_ms: float
    """Tenth-percentile device execution latency."""
    p90_ms: float
    """Ninetieth-percentile device execution latency."""
    images_per_second: float
    """Throughput derived from median latency."""
    keypoint_min: float
    """Minimum normalized coordinate from the final execution."""
    keypoint_max: float
    """Maximum normalized coordinate from the final execution."""
    descriptor_norm_median: float
    """Median descriptor L2 norm from the final execution."""
    score_min: float
    """Minimum feature score from the final execution."""
    score_max: float
    """Maximum feature score from the final execution."""




def _collect_images(input_dir: Path, count: int) -> list[Path]:
    """Select real RoboCap images across the symlinked camera directories."""
    camera_directories: list[Path] = sorted(path for path in input_dir.iterdir() if path.is_dir())
    images: list[Path] = []
    for camera_directory in camera_directories:
        images.extend(sorted(camera_directory.glob("*.jpeg"))[:count])
    if len(images) < count:
        raise FileNotFoundError(f"Found only {len(images)} JPEG images below {input_dir}")
    stride: int = max(1, len(images) // count)
    return images[::stride][:count]


def _letterbox_rgb(path: Path, height: int = 1200, width: int = 1920) -> ImageCHW:
    """Load one image as cuSFM's float32 RGB letterbox tensor ``[3, H, W]``."""
    bgr_hwc: UInt8[ndarray, "source_h source_w 3"] | None = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr_hwc is None:
        raise ValueError(f"Failed to read {path}")
    source_height: int = bgr_hwc.shape[0]
    source_width: int = bgr_hwc.shape[1]
    scale: float = min(width / source_width, height / source_height)
    resized_width: int = round(source_width * scale)
    resized_height: int = round(source_height * scale)
    resized_bgr_hwc: UInt8[ndarray, "resized_h resized_w 3"] = cv2.resize(
        bgr_hwc, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    canvas_rgb_hwc: UInt8[ndarray, "height width 3"] = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset: int = (width - resized_width) // 2
    y_offset: int = (height - resized_height) // 2
    canvas_rgb_hwc[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized_bgr_hwc[..., ::-1]
    image_chw: ImageCHW = np.ascontiguousarray(canvas_rgb_hwc.transpose(2, 0, 1), dtype=np.float32)
    image_chw /= np.float32(255.0)
    return image_chw


def _build_engine(model_path: Path, engine_dir: Path, label: str, workspace_gib: int) -> Path:
    """Build a content-addressed FP16 TensorRT engine with parser errors preserved."""
    digest: str = sha256_file(model_path)
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path: Path = engine_dir / f"{label}-{digest[:16]}-trt-{trt.__version__}-fp16.engine"
    if engine_path.is_file():
        print(f"[{label}] reusing engine {engine_path}")
        return engine_path

    logger: trt.Logger = trt.Logger(trt.Logger.INFO)
    builder: trt.Builder = trt.Builder(logger)
    flags: int = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network: trt.INetworkDefinition = builder.create_network(flags)
    parser: trt.OnnxParser = trt.OnnxParser(network, logger)
    model_bytes: bytes = model_path.read_bytes()
    if not parser.parse(model_bytes):
        errors: list[str] = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT failed to parse {model_path}:\n" + "\n".join(errors))

    config: trt.IBuilderConfig = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    print(f"[{label}] building TensorRT {trt.__version__} FP16 engine from {model_path}")
    serialized: trt.IHostMemory | None = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {model_path}")
    engine_path.write_bytes(bytes(serialized))
    print(f"[{label}] wrote {engine_path} ({engine_path.stat().st_size / 2**20:.1f} MiB)")
    return engine_path



def _benchmark_engine(
    label: str,
    model_path: Path,
    engine_path: Path,
    images_bchw: ImagesBCHW,
    warmup_iterations: int,
    iterations: int,
) -> ModelResult:
    """Measure device-only execution with CUDA events and inspect final outputs."""
    logger: trt.Logger = trt.Logger(trt.Logger.WARNING)
    runtime: trt.Runtime = trt.Runtime(logger)
    engine: trt.ICudaEngine | None = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize {engine_path}")
    context: trt.IExecutionContext = engine.create_execution_context()

    input_names: list[str] = []
    output_names: list[str] = []
    device_pointers: dict[str, int] = {}
    output_arrays: dict[str, NDArray[Any]] = {}
    try:
        for index in range(engine.num_io_tensors):
            name: str = engine.get_tensor_name(index)
            mode: trt.TensorIOMode = engine.get_tensor_mode(name)
            shape: tuple[int, ...] = tuple(context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Unresolved TensorRT shape for {name}: {shape}")
            dtype: np.dtype[Any] = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            byte_count: int = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            pointer: int = int(check_cuda(cudart.cudaMalloc(byte_count), f"cudaMalloc({name})")[0])
            device_pointers[name] = pointer
            if not context.set_tensor_address(name, pointer):
                raise RuntimeError(f"Failed to bind TensorRT tensor {name}")
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
                output_arrays[name] = np.empty(shape, dtype=dtype)

        if input_names != ["image"]:
            raise RuntimeError(f"Expected extractor input ['image'], got {input_names}")
        input_pointer: int = device_pointers["image"]
        input_byte_count: int = images_bchw[0:1].nbytes
        stream: int = int(check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")[0])
        start_event: int = int(check_cuda(cudart.cudaEventCreate(), "cudaEventCreate(start)")[0])
        end_event: int = int(check_cuda(cudart.cudaEventCreate(), "cudaEventCreate(end)")[0])
        try:
            for iteration in range(warmup_iterations):
                image_bchw: ImagesBCHW = images_bchw[
                    iteration % len(images_bchw) : iteration % len(images_bchw) + 1
                ]
                check_cuda(
                    cudart.cudaMemcpy(
                        input_pointer,
                        image_bchw.ctypes.data,
                        input_byte_count,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    ),
                    "cudaMemcpy(H2D warmup)",
                )
                if not context.execute_async_v3(stream):
                    raise RuntimeError("TensorRT warmup execution failed")
            check_cuda(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)")

            latencies_ms: list[float] = []
            for iteration in range(iterations):
                image_index: int = iteration % len(images_bchw)
                image_bchw = images_bchw[image_index : image_index + 1]
                check_cuda(
                    cudart.cudaMemcpy(
                        input_pointer,
                        image_bchw.ctypes.data,
                        input_byte_count,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    ),
                    "cudaMemcpy(H2D timed input)",
                )
                check_cuda(cudart.cudaEventRecord(start_event, stream), "cudaEventRecord(start)")
                if not context.execute_async_v3(stream):
                    raise RuntimeError("TensorRT timed execution failed")
                check_cuda(cudart.cudaEventRecord(end_event, stream), "cudaEventRecord(end)")
                check_cuda(cudart.cudaEventSynchronize(end_event), "cudaEventSynchronize(end)")
                elapsed_ms: float = float(
                    check_cuda(cudart.cudaEventElapsedTime(start_event, end_event), "cudaEventElapsedTime")[0]
                )
                latencies_ms.append(elapsed_ms)

            for name in output_names:
                host_output: NDArray[Any] = output_arrays[name]
                check_cuda(
                    cudart.cudaMemcpy(
                        host_output.ctypes.data,
                        device_pointers[name],
                        host_output.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    ),
                    f"cudaMemcpy(D2H {name})",
                )
        finally:
            check_cuda(cudart.cudaEventDestroy(start_event), "cudaEventDestroy(start)")
            check_cuda(cudart.cudaEventDestroy(end_event), "cudaEventDestroy(end)")
            check_cuda(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")
    finally:
        for pointer in device_pointers.values():
            check_cuda(cudart.cudaFree(pointer), "cudaFree")

    keypoints_n2: KeypointsN2 = output_arrays["keypoints"]
    descriptors_nd: DescriptorsND = output_arrays["descriptors"]
    scores_n: ScoresN = output_arrays["scores"]
    for name, output in output_arrays.items():
        if not np.isfinite(output).all():
            raise RuntimeError(f"{label} produced non-finite values in {name}")
    descriptor_norms_n: ScoresN = np.linalg.norm(descriptors_nd, axis=1)
    median_ms: float = statistics.median(latencies_ms)
    result: ModelResult = ModelResult(
        label=label,
        onnx_path=str(model_path),
        onnx_sha256=sha256_file(model_path),
        engine_path=str(engine_path),
        engine_bytes=engine_path.stat().st_size,
        precision="fp16",
        batch_size=1,
        warmup_iterations=warmup_iterations,
        timed_iterations=iterations,
        median_ms=median_ms,
        p10_ms=percentile(latencies_ms, 0.1),
        p90_ms=percentile(latencies_ms, 0.9),
        images_per_second=1000.0 / median_ms,
        keypoint_min=float(keypoints_n2.min()),
        keypoint_max=float(keypoints_n2.max()),
        descriptor_norm_median=float(np.median(descriptor_norms_n)),
        score_min=float(scores_n.min()),
        score_max=float(scores_n.max()),
    )
    return result


def main(config: BenchmarkConfig) -> None:
    """Build all requested models, benchmark them, and write JSON results."""
    image_paths: list[Path] = _collect_images(config.input_dir, config.image_count)
    images: list[ImageCHW] = [_letterbox_rgb(path) for path in image_paths]
    images_bchw: ImagesBCHW = np.stack(images)
    print(f"Loaded {len(image_paths)} images: {[str(path) for path in image_paths]}")

    models: list[tuple[str, Path]] = [
        ("baseline_fp16", config.baseline),
        ("raco_fp16", config.optimized),
    ]

    results: list[ModelResult] = []
    for label, model_path in models:
        engine_path: Path = _build_engine(model_path, config.engine_dir, label, config.workspace_gib)
        result: ModelResult = _benchmark_engine(
            label,
            model_path,
            engine_path,
            images_bchw,
            config.warmup_iterations,
            config.iterations,
        )
        results.append(result)
        print(
            f"[{label}] median={result.median_ms:.3f} ms, "
            f"throughput={result.images_per_second:.2f} images/s, "
            f"p10/p90={result.p10_ms:.3f}/{result.p90_ms:.3f} ms"
        )

    payload: dict[str, object] = {
        "tensorrt_version": trt.__version__,
        "images": [str(path) for path in image_paths],
        "results": [asdict(result) for result in results],
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {config.output}")


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))

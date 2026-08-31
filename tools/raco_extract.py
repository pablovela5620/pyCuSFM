"""Extract batched RaCo-ALIKED features into cuSFM keyframe protobufs."""

# Ruff otherwise interprets jaxtyping shape strings as Python forward references.
# ruff: noqa: F821, UP037

from __future__ import annotations

import copy
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import batched
from pathlib import Path
from typing import Any, Self, TypeAlias, cast

import cv2
import numpy as np
import tensorrt as trt
import tyro
from cuda.bindings import runtime as cudart
from google.protobuf import (
    descriptor_pb2,
    descriptor_pool,
    json_format,
    message_factory,
)
from google.protobuf.message import Message
from jaxtyping import Float32, UInt8
from numpy import ndarray
from numpy.typing import NDArray

from tools.trt_runtime import check_cuda, percentile, sha256_file

ImageRGB: TypeAlias = UInt8[ndarray, "h w 3"]
ImageCHW: TypeAlias = Float32[ndarray, "3 network_h network_w"]
ImagesBCHW: TypeAlias = Float32[ndarray, "batch 3 network_h network_w"]
KeypointsBN2: TypeAlias = Float32[ndarray, "batch num_keypoints 2"]
DescriptorsBND: TypeAlias = Float32[ndarray, "batch num_keypoints descriptor_dim"]



REPO_ROOT: Path = Path(__file__).resolve().parent.parent
"""Repo root, so defaults resolve regardless of the working directory."""

@dataclass
class ExtractConfig:
    """Direct RaCo-ALIKED to cuSFM keyframe extraction settings."""

    input_dir: Path = Path("data/cusfm_runs/robocap_full/input")
    """Prepared cuSFM input containing camera image folders and frames_meta.json."""
    output_dir: Path = Path("data/cusfm_runs/robocap_raco_b8/cusfm/keyframes")
    """New cuSFM keyframe directory to populate without running feature_extractor_main."""
    onnx_path: Path = REPO_ROOT / "data" / "cusfm_models" / "raco-aliked-b1-16.onnx"
    """Batch-dynamic RaCo-ALIKED ONNX graph. Not committed (data/cusfm_models is
    gitignored); produce it with `pixi run -e raco raco-export
    --batched-extractor-path data/cusfm_models/raco-aliked-b1-16.onnx`."""
    descriptor_set_path: Path = REPO_ROOT / "data" / "cusfm_schema" / "cusfm_protos.fdset"
    """FileDescriptorSet holding cuSFM's own protobuf schema.

    Vendored in the repo rather than extracted per machine: this is the contract
    that makes writing cuSFM keyframes from Python legitimate, and defaulting it
    to a scratch path made both the extractor and its contract tests fail
    anywhere but the machine that first produced it. See
    ``data/cusfm_schema/README.md``."""
    engine_dir: Path = REPO_ROOT / "data" / "cusfm_models" / "engines"
    """Content-addressed TensorRT engine cache (gitignored; engines are per-GPU)."""
    result_path: Path = REPO_ROOT / "data" / "cusfm_models" / "extraction.json"
    """Machine-readable extraction or benchmark results (gitignored)."""
    batch_size: int = 8
    """Extraction batch size."""
    minimum_batch_size: int = 1
    """TensorRT profile minimum batch."""
    optimal_batch_size: int = 8
    """TensorRT profile optimal batch."""
    maximum_batch_size: int = 16
    """TensorRT profile maximum batch and benchmark ceiling."""
    network_width: int = 1920
    """RaCo network input width."""
    network_height: int = 1200
    """RaCo network input height; original images are stretched to this size."""
    workspace_gib: int = 8
    """TensorRT tactic workspace ceiling in GiB."""
    builder_optimization_level: int = 3
    """TensorRT build-search level from 0 (fastest build) through 5 (most tuning)."""
    preprocessing_workers: int = 8
    """Concurrent JPEG decode and resize workers."""
    progress_interval: int = 128
    """Print progress after this many newly written keyframes."""
    resume: bool = True
    """Keep atomically completed protobufs and extract only missing frames."""
    build_only: bool = False
    """Build or reuse the TensorRT engine, then exit."""
    benchmark_only: bool = False
    """Benchmark device execution without writing keyframes."""
    benchmark_batch_sizes: tuple[int, ...] = (1, 4, 8, 16)
    """Batch sizes measured in benchmark mode."""
    warmup_iterations: int = 30
    """Untimed executions per benchmark batch size."""
    timed_iterations: int = 100
    """Timed executions per benchmark batch size."""


@dataclass(frozen=True, slots=True)
class FrameTask:
    """One input image and its cuSFM keyframe destination."""

    frame_id: int
    """Metadata index used only for ordering and diagnostics."""
    image_path: Path
    """Original JPEG path."""
    output_path: Path
    """Atomic protobuf destination."""
    image_width: int
    """Original image width."""
    image_height: int
    """Original image height."""


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    """Decoded source pixels and the corresponding network tensor."""

    task: FrameTask
    """Source/destination metadata."""
    image_rgb_hwc: ImageRGB
    """Original-resolution UInt8 RGB image."""
    network_chw: ImageCHW
    """Stretched Float32 RGB network input in [0, 1]."""


@dataclass(frozen=True, slots=True)
class InferenceBatch:
    """TensorRT feature outputs for one image batch."""

    keypoints_bn2: KeypointsBN2
    """Normalized keypoints with shape [batch, 2048, 2]."""
    descriptors_bnd: DescriptorsBND
    """L2-normalized descriptors with shape [batch, 2048, 128]."""


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Steady-state device timing for one batch size."""

    batch_size: int
    """Images executed together."""
    warmup_iterations: int
    """Untimed execution count."""
    timed_iterations: int
    """Measured execution count."""
    median_batch_ms: float
    """Median whole-batch device latency."""
    median_ms_per_image: float
    """Median latency divided by batch size."""
    images_per_second: float
    """Throughput derived from median batch latency."""
    p10_batch_ms: float
    """Tenth-percentile batch latency."""
    p90_batch_ms: float
    """Ninetieth-percentile batch latency."""


def normalized_to_pixels(
    normalized_n2: Float32[ndarray, "n 2"], *, image_width: int, image_height: int
) -> Float32[ndarray, "n 2"]:
    """Map cuSFM ALIKED normalized coordinates to original-image pixels.

    Args:
        normalized_n2: Float32 normalized XY coordinates with shape ``[n, 2]``.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        Float32 pixel XY coordinates with shape ``[n, 2]``.
    """
    scale_xy: Float32[ndarray, "xy"] = np.asarray(
        [(image_width - 1) / 2.0, (image_height - 1) / 2.0], dtype=np.float32
    )
    pixels_n2: Float32[ndarray, "n 2"] = (normalized_n2 + np.float32(1.0)) * scale_xy
    return pixels_n2


def sample_keypoint_rgb(
    image_rgb_hwc: UInt8[ndarray, "h w 3"], keypoints_n2: Float32[ndarray, "n 2"]
) -> UInt8[ndarray, "n 3"]:
    """Sample source-image RGB at cuSFM keypoint coordinates.

    Args:
        image_rgb_hwc: UInt8 source RGB image with shape ``[h, w, 3]``.
        keypoints_n2: Float32 pixel XY coordinates with shape ``[n, 2]``.

    Returns:
        UInt8 RGB values with shape ``[n, 3]``.
    """
    rounded_n2: ndarray = np.rint(keypoints_n2)
    indices_n2: ndarray = rounded_n2.astype(np.int64)
    indices_n2[:, 0] = np.clip(indices_n2[:, 0], 0, image_rgb_hwc.shape[1] - 1)
    indices_n2[:, 1] = np.clip(indices_n2[:, 1], 0, image_rgb_hwc.shape[0] - 1)
    colors_rgb_n3: UInt8[ndarray, "n 3"] = image_rgb_hwc[indices_n2[:, 1], indices_n2[:, 0]]
    return colors_rgb_n3


def build_message_types(descriptor_set_path: Path) -> dict[str, type[Message]]:
    """Build cuSFM protobuf message classes from a descriptor set.

    Args:
        descriptor_set_path: FileDescriptorSet extracted from the cuSFM binaries.

    Returns:
        Runtime message classes needed by the extractor and metadata writer.
    """
    file_set: descriptor_pb2.FileDescriptorSet = descriptor_pb2.FileDescriptorSet.FromString(
        descriptor_set_path.read_bytes()
    )
    pool: descriptor_pool.DescriptorPool = descriptor_pool.DescriptorPool()
    remaining: list[descriptor_pb2.FileDescriptorProto] = list(file_set.file)
    while remaining:
        previous_count: int = len(remaining)
        for file_descriptor in remaining.copy():
            try:
                pool.Add(file_descriptor)
            except TypeError:
                continue
            remaining.remove(file_descriptor)
        if len(remaining) == previous_count:
            unresolved: list[str] = [file_descriptor.name for file_descriptor in remaining]
            raise ValueError(f"Could not resolve protobuf dependencies for {unresolved}")

    names: dict[str, str] = {
        "keyframe": "protos.visual.general.Keyframe",
        "metadata_collection": "protos.visual.general.KeyframesMetadataCollection",
    }
    classes: dict[str, type[Message]] = {}
    for label, full_name in names.items():
        classes[label] = cast(
            type[Message], message_factory.GetMessageClass(pool.FindMessageTypeByName(full_name))
        )
    return classes


def serialize_keyframe(
    *,
    message_types: dict[str, type[Message]],
    keypoints_n2: Float32[ndarray, "n 2"],
    descriptors_nd: Float32[ndarray, "n descriptor_dim"],
    colors_rgb_n3: UInt8[ndarray, "n 3"],
    image_width: int,
    image_height: int,
) -> bytes:
    """Serialize one cuSFM keyframe protobuf.

    Args:
        message_types: Runtime protobuf classes from :func:`build_message_types`.
        keypoints_n2: Float32 pixel XY coordinates with shape ``[n, 2]``.
        descriptors_nd: Float32 L2-normalized descriptors with shape ``[n, 128]``.
        colors_rgb_n3: UInt8 source-image colors with shape ``[n, 3]``.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        Serialized ``protos.visual.general.Keyframe`` bytes.
    """
    point_count: int = keypoints_n2.shape[0]
    if keypoints_n2.shape != (point_count, 2):
        raise ValueError(f"Expected keypoints [n, 2], got {keypoints_n2.shape}")
    if descriptors_nd.shape != (point_count, 128):
        raise ValueError(f"Expected descriptors [n, 128], got {descriptors_nd.shape}")
    if colors_rgb_n3.shape != (point_count, 3):
        raise ValueError(f"Expected colors [n, 3], got {colors_rgb_n3.shape}")

    keyframe: Message = message_types["keyframe"]()
    keyframe_dynamic: Any = keyframe
    keyframe_dynamic.inliers.extend([False] * point_count)
    keyframe_dynamic.is_valid = True
    vector: Any = keyframe_dynamic.keypoint_vector
    vector.x.extend(keypoints_n2[:, 0].tolist())
    vector.y.extend(keypoints_n2[:, 1].tolist())
    descriptors_le_nd: Float32[ndarray, "n descriptor_dim"] = np.ascontiguousarray(
        descriptors_nd, dtype="<f4"
    )
    vector.descriptors = descriptors_le_nd.tobytes()
    vector.descriptor_type = 4
    vector.bev_start_idx = -1
    vector.image_width = image_width
    vector.image_height = image_height
    vector.depth.extend([-1.0] * point_count)
    vector.r.extend(colors_rgb_n3[:, 0].tolist())
    vector.g.extend(colors_rgb_n3[:, 1].tolist())
    vector.b.extend(colors_rgb_n3[:, 2].tolist())
    return keyframe.SerializeToString()


def prepare_metadata_collection(
    message_types: dict[str, type[Message]], input_metadata: dict[str, Any]
) -> Message:
    """Convert input frames metadata to the feature-extractor output contract.

    Args:
        message_types: Runtime protobuf classes from :func:`build_message_types`.
        input_metadata: Parsed cuSFM input ``frames_meta.json`` object.

    Returns:
        Populated ``KeyframesMetadataCollection`` matching cuSFM extractor semantics.
    """
    output_metadata: dict[str, Any] = copy.deepcopy(input_metadata)
    entries: list[dict[str, Any]] = output_metadata["keyframes_metadata"]
    for entry in entries:
        synced_sample_id: int = int(entry.get("synced_sample_id", "0"))
        entry["synced_sample_id"] = str(synced_sample_id + 1)

    collection: Message = message_types["metadata_collection"]()
    json_format.ParseDict(output_metadata, collection)
    dynamic_collection: Any = collection
    dynamic_collection.detector_type = 3
    dynamic_collection.descriptor_type = 4
    return collection





def build_fp16_engine(config: ExtractConfig) -> Path:
    """Build or reuse a content-addressed FP16 TensorRT engine."""
    if not config.onnx_path.is_file():
        raise FileNotFoundError(config.onnx_path)
    if not (
        1 <= config.minimum_batch_size
        <= config.optimal_batch_size
        <= config.maximum_batch_size
    ):
        raise ValueError("Invalid TensorRT batch profile")
    digest: str = sha256_file(config.onnx_path)
    config.engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path: Path = config.engine_dir / (
        f"raco-aliked-{digest[:16]}-trt-{trt.__version__}-fp16-"
        f"b{config.minimum_batch_size}-{config.optimal_batch_size}-{config.maximum_batch_size}-"
        f"o{config.builder_optimization_level}.engine"
    )
    if engine_path.is_file():
        print(f"[engine] reusing {engine_path}")
        return engine_path

    logger: trt.Logger = trt.Logger(trt.Logger.INFO)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("Could not initialize TensorRT's standard plugin registry")
    builder: trt.Builder = trt.Builder(logger)
    flags: int = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network: trt.INetworkDefinition = builder.create_network(flags)
    parser: trt.OnnxParser = trt.OnnxParser(network, logger)
    if not parser.parse(config.onnx_path.read_bytes()):
        errors: list[str] = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    minimum_shape: tuple[int, int, int, int] = (
        config.minimum_batch_size,
        3,
        config.network_height,
        config.network_width,
    )
    optimal_shape: tuple[int, int, int, int] = (
        config.optimal_batch_size,
        3,
        config.network_height,
        config.network_width,
    )
    maximum_shape: tuple[int, int, int, int] = (
        config.maximum_batch_size,
        3,
        config.network_height,
        config.network_width,
    )
    input_tensor: trt.ITensor | None = network.get_input(0)
    if input_tensor is None:
        raise RuntimeError("TensorRT network has no image input")
    declared_shape: tuple[int, ...] = tuple(int(dimension) for dimension in input_tensor.shape)
    dynamic_input: bool = any(dimension < 0 for dimension in declared_shape)
    if not dynamic_input and declared_shape != maximum_shape:
        raise ValueError(
            f"Static ONNX input is {declared_shape}, but configured execution shape is {maximum_shape}"
        )

    builder_config: trt.IBuilderConfig = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, config.workspace_gib << 30)
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    builder_config.set_flag(trt.BuilderFlag.FP16)
    if not 0 <= config.builder_optimization_level <= 5:
        raise ValueError("TensorRT builder optimization level must be in [0, 5]")
    builder_config.builder_optimization_level = config.builder_optimization_level
    timing_cache_path: Path = config.engine_dir / f"tensorrt-{trt.__version__}.timing-cache"
    timing_cache_bytes: bytes = timing_cache_path.read_bytes() if timing_cache_path.is_file() else b""
    timing_cache: trt.ITimingCache = builder_config.create_timing_cache(timing_cache_bytes)
    if not builder_config.set_timing_cache(timing_cache, ignore_mismatch=False):
        raise RuntimeError(f"Could not attach TensorRT timing cache {timing_cache_path}")
    if dynamic_input:
        profile: trt.IOptimizationProfile = builder.create_optimization_profile()
        profile.set_shape("image", minimum_shape, optimal_shape, maximum_shape)
        builder_config.add_optimization_profile(profile)
    print(
        f"[engine] building TensorRT {trt.__version__} FP16 "
        + (
            f"profile {minimum_shape} / {optimal_shape} / {maximum_shape}"
            if dynamic_input
            else f"static shape {declared_shape}"
        )
    )
    serialized: trt.IHostMemory | None = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    temporary_path: Path = engine_path.with_suffix(engine_path.suffix + ".tmp")
    temporary_path.write_bytes(bytes(serialized))
    temporary_path.replace(engine_path)
    serialized_timing_cache: trt.IHostMemory = builder_config.get_timing_cache().serialize()
    temporary_cache_path: Path = timing_cache_path.with_suffix(timing_cache_path.suffix + ".tmp")
    temporary_cache_path.write_bytes(bytes(serialized_timing_cache))
    temporary_cache_path.replace(timing_cache_path)
    print(f"[engine] wrote {engine_path} ({engine_path.stat().st_size / 2**20:.1f} MiB)")
    return engine_path


class TensorRTExtractor:
    """Reusable TensorRT execution context and maximum-profile CUDA buffers."""

    def __init__(self, engine_path: Path, config: ExtractConfig) -> None:
        """Deserialize an engine and allocate buffers for its maximum batch."""
        logger: trt.Logger = trt.Logger(trt.Logger.WARNING)
        if not trt.init_libnvinfer_plugins(logger, ""):
            raise RuntimeError("Could not initialize TensorRT's standard plugin registry")
        self.runtime: trt.Runtime = trt.Runtime(logger)
        engine: trt.ICudaEngine | None = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}")
        self.engine: trt.ICudaEngine = engine
        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        configured_maximum_shape: tuple[int, int, int, int] = (
            config.maximum_batch_size,
            3,
            config.network_height,
            config.network_width,
        )
        declared_input_shape: tuple[int, ...] = tuple(self.engine.get_tensor_shape("image"))
        self.dynamic_input: bool = any(dimension < 0 for dimension in declared_input_shape)
        if self.dynamic_input:
            self.maximum_input_shape: tuple[int, ...] = configured_maximum_shape
            self.fixed_batch_size: int | None = None
            if not self.context.set_input_shape("image", self.maximum_input_shape):
                raise RuntimeError(f"Could not select maximum shape {self.maximum_input_shape}")
        else:
            self.maximum_input_shape = declared_input_shape
            self.fixed_batch_size = declared_input_shape[0]
            if declared_input_shape != configured_maximum_shape:
                raise ValueError(
                    f"Static engine input is {declared_input_shape}, configured {configured_maximum_shape}"
                )
        self.device_pointers: dict[str, int] = {}
        self.maximum_shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, np.dtype[Any]] = {}
        for index in range(self.engine.num_io_tensors):
            name: str = self.engine.get_tensor_name(index)
            shape: tuple[int, ...] = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Unresolved maximum shape for {name}: {shape}")
            dtype: np.dtype[Any] = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            byte_count: int = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            pointer: int = int(check_cuda(cudart.cudaMalloc(byte_count), f"cudaMalloc({name})")[0])
            self.device_pointers[name] = pointer
            self.maximum_shapes[name] = shape
            self.dtypes[name] = dtype
            if not self.context.set_tensor_address(name, pointer):
                raise RuntimeError(f"Could not bind {name}")
        self.stream: int = int(check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")[0])
        self.closed: bool = False

    def _select_shape(self, images_bchw: ImagesBCHW) -> None:
        """Select the current batch shape after validating the input tensor."""
        expected_tail: tuple[int, int, int] = self.maximum_input_shape[1:]
        if images_bchw.shape[1:] != expected_tail:
            raise ValueError(f"Expected image tail {expected_tail}, got {images_bchw.shape}")
        if images_bchw.dtype != self.dtypes["image"]:
            raise ValueError(f"Expected input dtype {self.dtypes['image']}, got {images_bchw.dtype}")
        if not images_bchw.flags.c_contiguous:
            raise ValueError("TensorRT input must be contiguous")
        current_shape: tuple[int, ...] = tuple(images_bchw.shape)
        if not self.dynamic_input and current_shape != self.maximum_input_shape:
            raise ValueError(f"Static TensorRT engine requires {self.maximum_input_shape}, got {current_shape}")
        if self.dynamic_input and not self.context.set_input_shape("image", current_shape):
            raise ValueError(f"TensorRT profile rejected {current_shape}")

    def _copy_input(self, images_bchw: ImagesBCHW) -> None:
        """Copy one selected batch to the reusable device buffer."""
        check_cuda(
            cudart.cudaMemcpyAsync(
                self.device_pointers["image"],
                images_bchw.ctypes.data,
                images_bchw.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync(H2D image)",
        )

    def _feature_output_names(self, prefix: str) -> list[str]:
        """Return one batched output or numerically ordered unrolled branch outputs."""
        if prefix in self.device_pointers:
            return [prefix]
        indexed_names: list[tuple[int, str]] = []
        for name in self.device_pointers:
            if not name.startswith(prefix + "_"):
                continue
            try:
                index: int = int(name.rsplit("_", maxsplit=1)[1])
            except ValueError:
                continue
            indexed_names.append((index, name))
        if not indexed_names:
            raise RuntimeError(f"TensorRT engine has no {prefix} output")
        return [name for _index, name in sorted(indexed_names)]

    def infer(self, images_bchw: ImagesBCHW) -> InferenceBatch:
        """Run one batch and return normalized keypoints and descriptors."""
        self._select_shape(images_bchw)
        self._copy_input(images_bchw)
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execution failed")
        output_arrays: dict[str, NDArray[Any]] = {}
        keypoint_names: list[str] = self._feature_output_names("keypoints")
        descriptor_names: list[str] = self._feature_output_names("descriptors")
        for name in (*keypoint_names, *descriptor_names):
            shape: tuple[int, ...] = tuple(self.context.get_tensor_shape(name))
            output: NDArray[Any] = np.empty(shape, dtype=self.dtypes[name])
            output_arrays[name] = output
            check_cuda(
                cudart.cudaMemcpyAsync(
                    output.ctypes.data,
                    self.device_pointers[name],
                    output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                f"cudaMemcpyAsync(D2H {name})",
            )
        check_cuda(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize(infer)")
        keypoints_bn2: KeypointsBN2 = np.asarray(
            np.concatenate([output_arrays[name] for name in keypoint_names], axis=0),
            dtype=np.float32,
        )
        descriptors_bnd: DescriptorsBND = np.asarray(
            np.concatenate([output_arrays[name] for name in descriptor_names], axis=0),
            dtype=np.float32,
        )
        if not np.isfinite(keypoints_bn2).all() or not np.isfinite(descriptors_bnd).all():
            raise RuntimeError("TensorRT produced non-finite features")
        return InferenceBatch(keypoints_bn2=keypoints_bn2, descriptors_bnd=descriptors_bnd)

    def benchmark(
        self,
        images_bchw: ImagesBCHW,
        batch_sizes: tuple[int, ...],
        warmup_iterations: int,
        timed_iterations: int,
    ) -> list[BenchmarkResult]:
        """Measure device-only execution latency with CUDA events."""
        results: list[BenchmarkResult] = []
        for batch_size in batch_sizes:
            if batch_size < 1 or batch_size > len(images_bchw):
                raise ValueError(f"Cannot benchmark batch {batch_size} with {len(images_bchw)} inputs")
            batch_bchw: ImagesBCHW = np.ascontiguousarray(images_bchw[:batch_size])
            self._select_shape(batch_bchw)
            self._copy_input(batch_bchw)
            for _ in range(warmup_iterations):
                if not self.context.execute_async_v3(self.stream):
                    raise RuntimeError(f"TensorRT warmup failed at batch {batch_size}")
            check_cuda(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize(warmup)")

            start_event: int = int(check_cuda(cudart.cudaEventCreate(), "cudaEventCreate(start)")[0])
            end_event: int = int(check_cuda(cudart.cudaEventCreate(), "cudaEventCreate(end)")[0])
            latencies_ms: list[float] = []
            try:
                for _ in range(timed_iterations):
                    check_cuda(cudart.cudaEventRecord(start_event, self.stream), "cudaEventRecord(start)")
                    if not self.context.execute_async_v3(self.stream):
                        raise RuntimeError(f"TensorRT timed execution failed at batch {batch_size}")
                    check_cuda(cudart.cudaEventRecord(end_event, self.stream), "cudaEventRecord(end)")
                    check_cuda(cudart.cudaEventSynchronize(end_event), "cudaEventSynchronize(end)")
                    elapsed_ms: float = float(
                        check_cuda(
                            cudart.cudaEventElapsedTime(start_event, end_event),
                            "cudaEventElapsedTime",
                        )[0]
                    )
                    latencies_ms.append(elapsed_ms)
            finally:
                check_cuda(cudart.cudaEventDestroy(start_event), "cudaEventDestroy(start)")
                check_cuda(cudart.cudaEventDestroy(end_event), "cudaEventDestroy(end)")
            median_batch_ms: float = statistics.median(latencies_ms)
            result: BenchmarkResult = BenchmarkResult(
                batch_size=batch_size,
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
                median_batch_ms=median_batch_ms,
                median_ms_per_image=median_batch_ms / batch_size,
                images_per_second=1000.0 * batch_size / median_batch_ms,
                p10_batch_ms=percentile(latencies_ms, 0.1),
                p90_batch_ms=percentile(latencies_ms, 0.9),
            )
            results.append(result)
            print(
                f"[benchmark] batch={batch_size:2d} median={median_batch_ms:.3f} ms/batch "
                f"{result.median_ms_per_image:.3f} ms/image {result.images_per_second:.2f} images/s"
            )
        return results

    def close(self) -> None:
        """Release CUDA allocations owned by this extractor."""
        if self.closed:
            return
        check_cuda(cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
        for pointer in self.device_pointers.values():
            check_cuda(cudart.cudaFree(pointer), "cudaFree")
        self.closed = True

    def __enter__(self) -> Self:
        """Return this extractor as a context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Release CUDA resources at context exit."""
        self.close()


def collect_frame_tasks(
    input_dir: Path, output_dir: Path, input_metadata: dict[str, Any]
) -> list[FrameTask]:
    """Resolve ordered cuSFM metadata entries to image and protobuf paths.

    Args:
        input_dir: Prepared cuSFM input directory.
        output_dir: Keyframe output directory.
        input_metadata: Parsed input frames metadata.

    Returns:
        Ordered frame extraction tasks across every camera.
    """
    camera_parameters: dict[str, dict[str, Any]] = input_metadata[
        "camera_params_id_to_camera_params"
    ]
    tasks: list[FrameTask] = []
    for entry in input_metadata["keyframes_metadata"]:
        frame_id: int = int(entry.get("id", "0"))
        camera_parameters_id: str = str(entry.get("camera_params_id", "0"))
        calibration: dict[str, Any] = camera_parameters[camera_parameters_id][
            "calibration_parameters"
        ]
        relative_image_path: Path = Path(entry["image_name"])
        image_path: Path = input_dir / relative_image_path
        output_path: Path = output_dir / relative_image_path.parent / f"{relative_image_path.stem}.pb"
        tasks.append(
            FrameTask(
                frame_id=frame_id,
                image_path=image_path,
                output_path=output_path,
                image_width=int(calibration["image_width"]),
                image_height=int(calibration["image_height"]),
            )
        )
    missing: list[Path] = [task.image_path for task in tasks if not task.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} input images; first is {missing[0]}")
    return tasks


def prepare_frame(task: FrameTask, network_width: int, network_height: int) -> PreparedFrame:
    """Decode one JPEG and reproduce cuSFM's OpenCV stretch preprocessing.

    Args:
        task: Source image and original dimensions.
        network_width: TensorRT input width.
        network_height: TensorRT input height.

    Returns:
        Original RGB pixels and Float32 network CHW tensor.
    """
    bgr_hwc: UInt8[ndarray, "h w 3"] | None = cv2.imread(str(task.image_path), cv2.IMREAD_COLOR)
    if bgr_hwc is None:
        raise ValueError(f"OpenCV could not decode {task.image_path}")
    if bgr_hwc.shape[:2] != (task.image_height, task.image_width):
        raise ValueError(
            f"Metadata says {task.image_width}x{task.image_height}, got "
            f"{bgr_hwc.shape[1]}x{bgr_hwc.shape[0]} for {task.image_path}"
        )
    resized_bgr_hwc: UInt8[ndarray, "network_h network_w 3"] = cv2.resize(
        bgr_hwc,
        (network_width, network_height),
        interpolation=cv2.INTER_LINEAR,
    )
    network_chw: ImageCHW = np.ascontiguousarray(
        resized_bgr_hwc[..., ::-1].transpose(2, 0, 1), dtype=np.float32
    )
    network_chw /= np.float32(255.0)
    image_rgb_hwc: ImageRGB = np.ascontiguousarray(bgr_hwc[..., ::-1])
    return PreparedFrame(task=task, image_rgb_hwc=image_rgb_hwc, network_chw=network_chw)



def _pad_images_to_batch(images_bchw: ImagesBCHW, target_batch_size: int) -> ImagesBCHW:
    """Repeat the final image until a partial batch satisfies a fixed engine."""
    current_batch_size: int = len(images_bchw)
    if current_batch_size < 1 or current_batch_size > target_batch_size:
        raise ValueError(
            f"Cannot pad batch of {current_batch_size} images to {target_batch_size}"
        )
    if current_batch_size == target_batch_size:
        return images_bchw
    padding: ImagesBCHW = np.repeat(
        images_bchw[-1:], target_batch_size - current_batch_size, axis=0
    )
    return np.ascontiguousarray(np.concatenate((images_bchw, padding), axis=0))


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    """Write a complete file by atomic rename so resumption never accepts a partial protobuf."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(payload)
    temporary_path.replace(path)


def write_frames_metadata(output_dir: Path, collection: Message) -> None:
    """Write feature-extractor-compatible frames_meta.json atomically."""
    payload: str = json_format.MessageToJson(
        collection,
        preserving_proto_field_name=True,
        indent=None,
    )
    _write_atomic_bytes(output_dir / "frames_meta.json", payload.encode())


def _prepare_batch(
    executor: ThreadPoolExecutor,
    tasks: list[FrameTask],
    network_width: int,
    network_height: int,
) -> list[PreparedFrame]:
    """Decode and stretch one batch concurrently while preserving task order."""
    prepared: list[PreparedFrame] = list(
        executor.map(
            prepare_frame,
            tasks,
            [network_width] * len(tasks),
            [network_height] * len(tasks),
        )
    )
    return prepared


def _write_feature_batch(
    prepared: list[PreparedFrame], outputs: InferenceBatch, message_types: dict[str, type[Message]]
) -> int:
    """Convert one TensorRT batch to atomically written cuSFM protobufs."""
    if len(prepared) != len(outputs.keypoints_bn2):
        raise ValueError("Prepared/output batch sizes differ")
    bytes_written: int = 0
    for index, frame in enumerate(prepared):
        keypoints_n2: Float32[ndarray, "num_keypoints 2"] = normalized_to_pixels(
            outputs.keypoints_bn2[index],
            image_width=frame.task.image_width,
            image_height=frame.task.image_height,
        )
        colors_rgb_n3: UInt8[ndarray, "num_keypoints 3"] = sample_keypoint_rgb(
            frame.image_rgb_hwc, keypoints_n2
        )
        descriptors_nd: Float32[ndarray, "num_keypoints descriptor_dim"] = np.ascontiguousarray(
            outputs.descriptors_bnd[index], dtype=np.float32
        )
        payload: bytes = serialize_keyframe(
            message_types=message_types,
            keypoints_n2=keypoints_n2,
            descriptors_nd=descriptors_nd,
            colors_rgb_n3=colors_rgb_n3,
            image_width=frame.task.image_width,
            image_height=frame.task.image_height,
        )
        _write_atomic_bytes(frame.task.output_path, payload)
        bytes_written += len(payload)
    return bytes_written


def run_benchmark(
    config: ExtractConfig,
    extractor: TensorRTExtractor,
    tasks: list[FrameTask],
    engine_path: Path,
) -> None:
    """Prepare real inputs and run the requested device-only batch benchmark."""
    largest_batch: int = max(config.benchmark_batch_sizes)
    if largest_batch > config.maximum_batch_size:
        raise ValueError("A benchmark batch exceeds the TensorRT profile maximum")
    with ThreadPoolExecutor(max_workers=config.preprocessing_workers) as executor:
        prepared: list[PreparedFrame] = _prepare_batch(
            executor,
            tasks[:largest_batch],
            config.network_width,
            config.network_height,
        )
    images_bchw: ImagesBCHW = np.stack([frame.network_chw for frame in prepared])
    results: list[BenchmarkResult] = extractor.benchmark(
        images_bchw,
        config.benchmark_batch_sizes,
        config.warmup_iterations,
        config.timed_iterations,
    )
    payload: dict[str, object] = {
        "onnx_path": str(config.onnx_path),
        "onnx_sha256": sha256_file(config.onnx_path),
        "engine_path": str(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "tensorrt_version": trt.__version__,
        "precision": "fp16",
        "images": [str(frame.task.image_path) for frame in prepared],
        "results": [asdict(result) for result in results],
    }
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[benchmark] wrote {config.result_path}")


def run_extraction(
    config: ExtractConfig,
    extractor: TensorRTExtractor,
    tasks: list[FrameTask],
    message_types: dict[str, type[Message]],
    engine_path: Path,
) -> None:
    """Extract every missing frame, write protobufs, and report wall throughput."""
    if config.resume:
        pending: list[FrameTask] = [
            task for task in tasks if not (task.output_path.is_file() and task.output_path.stat().st_size > 0)
        ]
    else:
        pending = tasks
    completed_before_start: int = len(tasks) - len(pending)
    print(
        f"[extract] total={len(tasks)} existing={completed_before_start} pending={len(pending)} "
        f"batch={config.batch_size} workers={config.preprocessing_workers}"
    )
    if extractor.fixed_batch_size is not None and extractor.fixed_batch_size != config.batch_size:
        raise ValueError(
            f"Static engine batch {extractor.fixed_batch_size} does not match extraction batch "
            f"{config.batch_size}"
        )
    start: float = time.perf_counter()
    processed: int = 0
    bytes_written: int = 0
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=config.preprocessing_workers) as executor:
        for batch_tasks in batched(pending, config.batch_size):
            prepared: list[PreparedFrame] = _prepare_batch(
                executor,
                batch_tasks,
                config.network_width,
                config.network_height,
            )
            images_bchw: ImagesBCHW = np.stack([frame.network_chw for frame in prepared])
            if extractor.fixed_batch_size is not None:
                images_bchw = _pad_images_to_batch(images_bchw, extractor.fixed_batch_size)
            outputs: InferenceBatch = extractor.infer(images_bchw)
            if len(outputs.keypoints_bn2) != len(prepared):
                outputs = InferenceBatch(
                    keypoints_bn2=outputs.keypoints_bn2[: len(prepared)],
                    descriptors_bnd=outputs.descriptors_bnd[: len(prepared)],
                )
            bytes_written += _write_feature_batch(prepared, outputs, message_types)
            processed += len(prepared)
            if processed % config.progress_interval < len(prepared) or processed == len(pending):
                elapsed: float = time.perf_counter() - start
                print(
                    f"[extract] {processed}/{len(pending)} new, "
                    f"{processed / elapsed:.2f} images/s, {elapsed:.1f}s"
                )
    wall_seconds: float = time.perf_counter() - start
    images_per_second: float = processed / wall_seconds if wall_seconds > 0.0 else 0.0
    payload: dict[str, object] = {
        "onnx_path": str(config.onnx_path),
        "onnx_sha256": sha256_file(config.onnx_path),
        "engine_path": str(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "precision": "fp16",
        "batch_size": config.batch_size,
        "total_keyframes": len(tasks),
        "existing_keyframes": completed_before_start,
        "new_keyframes": processed,
        "bytes_written": bytes_written,
        "wall_seconds": wall_seconds,
        "images_per_second": images_per_second,
        "ms_per_image": 1000.0 / images_per_second if images_per_second > 0.0 else None,
    }
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[extract] finished {processed} new keyframes in {wall_seconds:.3f}s "
        f"({images_per_second:.2f} images/s); wrote {config.result_path}"
    )


def main(config: ExtractConfig) -> None:
    """Build the engine, benchmark it, or write a complete cuSFM keyframe tree."""
    if config.batch_size < 1 or config.batch_size > config.maximum_batch_size:
        raise ValueError("batch_size must fall inside the TensorRT profile")
    engine_path: Path = build_fp16_engine(config)
    if config.build_only:
        return

    metadata_path: Path = config.input_dir / "frames_meta.json"
    input_metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    tasks: list[FrameTask] = collect_frame_tasks(config.input_dir, config.output_dir, input_metadata)
    message_types: dict[str, type[Message]] = build_message_types(config.descriptor_set_path)
    if config.benchmark_only:
        with TensorRTExtractor(engine_path, config) as extractor:
            run_benchmark(config, extractor, tasks, engine_path)
        return

    collection: Message = prepare_metadata_collection(message_types, input_metadata)
    write_frames_metadata(config.output_dir, collection)
    with TensorRTExtractor(engine_path, config) as extractor:
        run_extraction(config, extractor, tasks, message_types, engine_path)


if __name__ == "__main__":
    main(tyro.cli(ExtractConfig))

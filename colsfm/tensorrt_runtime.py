"""The TensorRT plumbing `colsfm.features_trt` and `colsfm.matching_trt` share.

cuSFM runs ALIKED and LightGlue as TensorRT engines built from the two ONNX
graphs in `pycusfm/models/aliked_lightglue/`, caching each engine next to its
graph under `<stem>_fp16_<trt-version>_sm_<arch>.engine`
(feature_matcher_main.md §5.2). This module owns that cache and the buffer
management around one execution context; the two backends above own the
pre- and post-processing.

`tools/raco_extract.py` is the working prior art — the same builder flags, the
same content-addressed timing cache, the same `set_tensor_address` +
`execute_async_v3` loop. Three things differ here:

* **The cache key is the blob's file name, not a content hash.** The point is to
  *reuse the engines that ship in the repo*: `aliked_fp16_10_13_3_9_sm_12_0.engine`
  and `lightglue_aliked_fp16_10_13_3_9_sm_12_0.engine` were built by the blob
  itself, deserialise under this TensorRT and this GPU, and cost about 205 s
  each to rebuild. A content hash would have missed both.
* **The session is generic.** `TensorRTExtractor` hard-codes ALIKED's
  `image`/`keypoints`/`descriptors` binding names; LightGlue has four inputs and
  two outputs, so `TensorRTSession` takes the bindings by name and grows its
  device buffers when a larger shape arrives.
* **It runs in the `colsfm` environment**, which carries `tensorrt-cu13` and
  `cuda-python` but not `onnx`; nothing here imports `onnx`, and the ONNX graph
  is parsed by TensorRT's own `OnnxParser`.

TensorRT and `cuda.bindings` are imported at module scope, so importing this
module is itself the availability probe: `colsfm.features` and
`colsfm.matching` import their TensorRT backends inside the dispatch branch, and
the tests skip on `ImportError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Self, TypeAlias

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from numpy import ndarray

from colsfm import REPO_ROOT

ShapeProfile: TypeAlias = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
"""One optimisation profile for one input: its minimum, optimal and maximum shape."""

MODEL_DIR: Final[Path] = REPO_ROOT / "pycusfm" / "models" / "aliked_lightglue"
"""Where the blob keeps `aliked.onnx`, `lightglue_aliked.onnx` and their engines."""

DEFAULT_WORKSPACE_GIB: Final[int] = 8
"""TensorRT tactic workspace ceiling, as `tools/raco_extract.py` sets it."""

DEFAULT_OPTIMIZATION_LEVEL: Final[int] = 3
"""TensorRT builder search level; 3 is the default and what the shipped engines used."""


def check_cuda(result: tuple[Any, ...] | Any, operation: str) -> tuple[Any, ...]:
    """Raise on a failed `cudart` call and return its payload.

    Every `cuda.bindings.runtime` entry point returns `(status, *values)`, so the
    status has to be stripped before the payload is usable; dropping it silently
    turns a failed allocation into a null pointer used several lines later. The
    same helper as `tools/trt_runtime.check_cuda`, repeated rather than imported
    because `tools/` is not on the `colsfm` environment's import path.

    Args:
        operation: What was attempted, for the error message.
        result: Whatever the `cudart` call returned.

    Returns:
        The payload after the status word.

    Raises:
        RuntimeError: When the status is not `cudaSuccess`.
    """
    values: tuple[Any, ...] = result if isinstance(result, tuple) else (result,)
    if values[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {values[0]}")
    return values[1:]


def compute_capability(device_index: int = 0) -> str:
    """The GPU's compute capability in the blob's engine-name spelling.

    Args:
        device_index: CUDA device to query.

    Returns:
        `"<major>_<minor>"`, e.g. `"12_0"` for an RTX 5090.
    """
    properties: Any = check_cuda(cudart.cudaGetDeviceProperties(device_index), "cudaGetDeviceProperties")[0]
    return f"{int(properties.major)}_{int(properties.minor)}"


def tensorrt_version_tag() -> str:
    """The TensorRT version in the blob's engine-name spelling.

    Returns:
        The four numeric components joined by underscores, e.g. `"10_13_3_9"`.
        The PyPI wheel reports `10.13.3.9.post1`; the `.post1` is a packaging
        suffix, not a TensorRT version, and the shipped engines do not carry it.
    """
    numeric: list[str] = [part for part in trt.__version__.split(".") if part.isdigit()]
    return "_".join(numeric[:4])


def engine_path_for(onnx_path: Path, *, device_index: int = 0) -> Path:
    """Where the engine built from one ONNX graph lives.

    Args:
        onnx_path: The graph, e.g. `pycusfm/models/aliked_lightglue/aliked.onnx`.
        device_index: CUDA device the engine is built for.

    Returns:
        `<onnx dir>/<stem>_fp16_<trt version>_sm_<arch>.engine`, which is the
        blob's own naming and therefore hits the two engines already in the repo.
    """
    return onnx_path.with_name(f"{onnx_path.stem}_fp16_{tensorrt_version_tag()}_sm_{compute_capability(device_index)}.engine")


def build_fp16_engine(
    onnx_path: Path,
    engine_path: Path,
    profiles: Mapping[str, ShapeProfile] | None = None,
    *,
    workspace_gib: int = DEFAULT_WORKSPACE_GIB,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
) -> Path:
    """Build one FP16 engine from an ONNX graph, writing it atomically.

    Args:
        onnx_path: The ONNX graph to parse.
        engine_path: Destination; a `.tmp` sibling is renamed onto it.
        profiles: Optimisation profile per dynamic input, as
            `(minimum, optimal, maximum)` shapes. Inputs absent from the mapping
            keep the shape the graph declares, which is what a fully static graph
            such as `aliked.onnx` needs.
        workspace_gib: Tactic workspace ceiling in GiB.
        optimization_level: TensorRT builder search level, 0 to 5.

    Returns:
        `engine_path`.

    Raises:
        FileNotFoundError: When the ONNX graph is missing.
        RuntimeError: When the parse or the build fails.
        ValueError: When `optimization_level` is outside 0 to 5.
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"No ONNX graph at {onnx_path}")
    if not 0 <= optimization_level <= 5:
        raise ValueError(f"TensorRT builder optimization level must be in [0, 5], got {optimization_level}")

    logger: trt.Logger = trt.Logger(trt.Logger.WARNING)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("Could not initialize TensorRT's standard plugin registry")
    builder: trt.Builder = trt.Builder(logger)
    network: trt.INetworkDefinition = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser: trt.OnnxParser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors: list[str] = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n" + "\n".join(errors))

    builder_config: trt.IBuilderConfig = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    builder_config.set_flag(trt.BuilderFlag.FP16)
    builder_config.builder_optimization_level = optimization_level
    timing_cache_path: Path = engine_path.with_name(f"tensorrt-{tensorrt_version_tag()}.timing-cache")
    timing_cache: trt.ITimingCache = builder_config.create_timing_cache(
        timing_cache_path.read_bytes() if timing_cache_path.is_file() else b""
    )
    builder_config.set_timing_cache(timing_cache, ignore_mismatch=False)
    if profiles:
        profile: trt.IOptimizationProfile = builder.create_optimization_profile()
        for name, (minimum, optimal, maximum) in profiles.items():
            profile.set_shape(name, minimum, optimal, maximum)
        builder_config.add_optimization_profile(profile)

    print(f"[colsfm] building TensorRT {trt.__version__} FP16 engine from {onnx_path.name}; this takes minutes")
    serialized: trt.IHostMemory | None = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = engine_path.with_suffix(engine_path.suffix + ".tmp")
    temporary_path.write_bytes(bytes(serialized))
    temporary_path.replace(engine_path)
    temporary_cache_path: Path = timing_cache_path.with_suffix(timing_cache_path.suffix + ".tmp")
    temporary_cache_path.write_bytes(bytes(builder_config.get_timing_cache().serialize()))
    temporary_cache_path.replace(timing_cache_path)
    print(f"[colsfm] wrote {engine_path} ({engine_path.stat().st_size / 2**20:.1f} MiB)")
    return engine_path


def resolve_engine(
    onnx_path: Path, profiles: Mapping[str, ShapeProfile] | None = None, *, device_index: int = 0
) -> Path:
    """Return the cached engine for one graph, building it when it is absent.

    Args:
        onnx_path: The ONNX graph.
        profiles: Optimisation profile per dynamic input; see `build_fp16_engine`.
        device_index: CUDA device the engine is built for.

    Returns:
        The engine file.
    """
    engine_path: Path = engine_path_for(onnx_path, device_index=device_index)
    if engine_path.is_file():
        return engine_path
    return build_fp16_engine(onnx_path, engine_path, profiles)


class TensorRTSession:
    """One deserialised engine, its execution context and its device buffers.

    Buffers are allocated lazily and only ever grow, so a run whose shapes change
    from pair to pair — LightGlue, whose keypoint counts differ per image — pays
    one allocation per new high-water mark rather than one per call.

    Use it as a context manager; `close` frees the device memory and the stream.
    """

    def __init__(self, engine_path: Path) -> None:
        """Deserialise one engine.

        Args:
            engine_path: A serialised TensorRT engine built for this GPU and
                this TensorRT version.

        Raises:
            FileNotFoundError: When the engine file is missing.
            RuntimeError: When TensorRT refuses to deserialise it.
        """
        if not engine_path.is_file():
            raise FileNotFoundError(f"No TensorRT engine at {engine_path}")
        logger: trt.Logger = trt.Logger(trt.Logger.ERROR)
        if not trt.init_libnvinfer_plugins(logger, ""):
            raise RuntimeError("Could not initialize TensorRT's standard plugin registry")
        self.runtime: trt.Runtime = trt.Runtime(logger)
        engine: trt.ICudaEngine | None = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}; it was built for another GPU or TensorRT")
        self.engine: trt.ICudaEngine = engine
        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        self.input_names: tuple[str, ...] = tuple(
            name
            for name in (self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_names: tuple[str, ...] = tuple(
            name
            for name in (self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        self.dtypes: dict[str, np.dtype[Any]] = {
            self.engine.get_tensor_name(index): np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.engine.get_tensor_name(index))))
            for index in range(self.engine.num_io_tensors)
        }
        self.device_pointers: dict[str, int] = {}
        self.capacities: dict[str, int] = {}
        self.stream: int = int(check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")[0])
        self.closed: bool = False

    def _ensure_capacity(self, name: str, byte_count: int) -> int:
        """Guarantee a device buffer of at least `byte_count` bytes for one binding.

        Args:
            name: The binding.
            byte_count: Bytes the current call needs.

        Returns:
            The device pointer, reallocated when the buffer had to grow.
        """
        wanted: int = max(byte_count, 1)
        if self.capacities.get(name, 0) >= wanted:
            return self.device_pointers[name]
        if name in self.device_pointers:
            check_cuda(cudart.cudaFree(self.device_pointers[name]), f"cudaFree({name})")
        pointer: int = int(check_cuda(cudart.cudaMalloc(wanted), f"cudaMalloc({name})")[0])
        self.device_pointers[name] = pointer
        self.capacities[name] = wanted
        return pointer

    def run(self, inputs: Mapping[str, ndarray]) -> dict[str, ndarray]:
        """Execute the engine once and return every output as a host array.

        Args:
            inputs: One contiguous host array per input binding, already in the
                binding's dtype.

        Returns:
            One host array per output binding, at the shape TensorRT resolved.

        Raises:
            RuntimeError: When the session is closed, when a profile rejects a
                shape, or when execution fails.
            ValueError: When an input is missing, mistyped or not contiguous.
        """
        if self.closed:
            raise RuntimeError("This TensorRTSession is closed")
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"TensorRT engine input {name} was not supplied")
            array: ndarray = inputs[name]
            if array.dtype != self.dtypes[name]:
                raise ValueError(f"Input {name} must be {self.dtypes[name]}, got {array.dtype}")
            if not array.flags.c_contiguous:
                raise ValueError(f"Input {name} must be C-contiguous")
            if not self.context.set_input_shape(name, tuple(array.shape)):
                raise RuntimeError(f"TensorRT profile rejected shape {tuple(array.shape)} for {name}")
        for name in self.input_names:
            array = inputs[name]
            pointer: int = self._ensure_capacity(name, array.nbytes)
            self.context.set_tensor_address(name, pointer)
            check_cuda(
                cudart.cudaMemcpyAsync(
                    pointer, array.ctypes.data, array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream
                ),
                f"cudaMemcpyAsync(H2D {name})",
            )
        outputs: dict[str, ndarray] = {}
        for name in self.output_names:
            shape: tuple[int, ...] = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"TensorRT left output {name} with an unresolved shape {shape}")
            host: ndarray = np.empty(shape, dtype=self.dtypes[name])
            outputs[name] = host
            self.context.set_tensor_address(name, self._ensure_capacity(name, max(host.nbytes, 1)))
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execution failed")
        for name, host in outputs.items():
            if host.nbytes == 0:
                continue
            check_cuda(
                cudart.cudaMemcpyAsync(
                    host.ctypes.data,
                    self.device_pointers[name],
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                f"cudaMemcpyAsync(D2H {name})",
            )
        check_cuda(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        return outputs

    def close(self) -> None:
        """Free the device buffers and the stream; safe to call twice."""
        if self.closed:
            return
        self.closed = True
        check_cuda(cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
        for name, pointer in self.device_pointers.items():
            check_cuda(cudart.cudaFree(pointer), f"cudaFree({name})")
        self.device_pointers.clear()
        self.capacities.clear()

    def __enter__(self) -> Self:
        """Return the session itself, so `with TensorRTSession(...) as session` works."""
        return self

    def __exit__(self, *_arguments: object) -> None:
        """Free the device buffers and the stream."""
        self.close()

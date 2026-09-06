"""The TensorRT plumbing `colsfm.features_trt` and `colsfm.matching_trt` share.

cuSFM runs ALIKED and LightGlue as TensorRT engines built from the two ONNX
graphs in `pycusfm/models/aliked_lightglue/`, caching each engine next to its
graph under `<stem>_fp16[_<profile tag>]_<identity>_<trt-version>_sm_<arch>.engine`
(feature_matcher_main.md §5.2). This module owns that cache and the buffer
management around one execution context; the two backends above own the
pre- and post-processing. `tools/raco_extract.py` builds through this cache too,
rather than keeping a second builder of its own.

**The cache key is a complete build identity.** `engine_build_identity` digests
every input that decides what the serialised bytes are: the graph's own SHA-256,
the optimisation profiles as `(min, opt, max)` per binding, the precision flag,
the tactic workspace ceiling, the builder optimisation level, the full TensorRT
runtime version and the target compute capability. Anything less would let a
changed input reuse a stale engine — which is exactly what a name built from the
file name alone did, and what a digest of the graph alone still does when the
caller re-tunes a profile.

**Prebuilt blob engines are accepted by name, explicitly.** `resolve_engine`
accepts the blob's undigested spelling for the two graphs it ships,
`aliked_fp16_10_13_3_9_sm_12_0.engine` and
`lightglue_aliked_fp16_10_13_3_9_sm_12_0.engine`: they were built by cuSFM
itself, deserialise under this TensorRT and this GPU, and cost about 205 s each
to rebuild. The allowance is narrow on purpose — `TRUSTED_PREBUILT_GRAPHS` names
the two stems, and only inside `MODEL_DIR` — so nothing else can pick up an
engine whose provenance is a file name.

**Publication is atomic and collision-free.** Every build writes to a unique
temporary file in the destination directory and renames it into place, so two
processes building the same engine cannot interleave into one file and a crashed
build leaves no half-written engine for the next run to deserialise.

**The session is generic.** LightGlue has four inputs and two outputs, so
`TensorRTSession` takes its bindings by name and grows its device buffers when a
larger shape arrives.

**It runs in the `colsfm` environment**, which carries `tensorrt-cu13` and
`cuda-python` but not `onnx`; nothing here imports `onnx`, and the ONNX graph is
parsed by TensorRT's own `OnnxParser`.

TensorRT and `cuda.bindings` are imported at module scope, so importing this
module is itself the availability probe: `colsfm.features` and
`colsfm.matching` import their TensorRT backends inside the dispatch branch, and
the tests skip on `ImportError`.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Self, TypeAlias

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from jaxtyping import UInt8
from numpy import ndarray

from colsfm import REPO_ROOT

ShapeProfile: TypeAlias = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
"""One optimisation profile for one input: its minimum, optimal and maximum shape."""

EnginePrecision: TypeAlias = Literal["fp16"]
"""The precisions this cache builds at.

One, today, and named rather than implied: the precision is part of an engine's
build identity, so it has to be a value the identity can carry rather than a
constant baked into a format string."""

MODEL_DIR: Final[Path] = REPO_ROOT / "pycusfm" / "models" / "aliked_lightglue"
"""Where the blob keeps `aliked.onnx`, `lightglue_aliked.onnx` and their engines."""

DEFAULT_WORKSPACE_GIB: Final[int] = 8
"""TensorRT tactic workspace ceiling, as `tools/raco_extract.py` sets it."""

DEFAULT_OPTIMIZATION_LEVEL: Final[int] = 3
"""TensorRT builder search level; 3 is the default and what the shipped engines used."""

PREPROCESS_IMAGE_U8_BINDING: Final[str] = "image_u8"
"""The preprocessing engine's input: `[batch, height, width, 3]` uint8 BGR, as `cv2.imread` gives it."""

PREPROCESS_IMAGE_BINDING: Final[str] = "image"
"""The preprocessing engine's output, named to match what both ALIKED graphs call their input."""

PREPROCESS_ENGINE_DIR: Final[Path] = REPO_ROOT / "data" / "cusfm_models"
"""Where the generated preprocessing engines are cached; gitignored, like the RaCo graphs."""

ONNX_DIGEST_CHARACTERS: Final[int] = 12
"""Hex characters of the build identity's SHA-256 that go into the engine's file name."""

PUBLISHED_FILE_MODE: Final[int] = 0o644
"""Permissions a published engine or timing cache carries.

`tempfile.mkstemp` opens at 0600; these are build products in a shared cache
directory, so they are made world-readable on the way out."""

DEFAULT_PRECISION: Final[EnginePrecision] = "fp16"
"""What `build_fp16_engine` builds at, and the `_fp16` the shipped engines are named with."""

TRUSTED_PREBUILT_GRAPHS: Final[frozenset[str]] = frozenset({"aliked", "lightglue_aliked"})
"""The graph stems whose cuSFM-built engines are accepted without a build identity.

Exactly the two `pycusfm/models/aliked_lightglue/` ships. Their engines carry no
digest, cost about 205 s each to rebuild, and deserialise under this TensorRT and
this GPU — so accepting them is a deliberate allowance for trusted prebuilt
blobs. Naming the stems, rather than accepting anything inside `MODEL_DIR`, is
what keeps that allowance from spreading to a graph somebody drops in later."""

BGR_TO_RGB_INDICES: Final[tuple[int, int, int]] = (2, 1, 0)
"""The channel permutation the host used to do with `resized_bgr_hwc[..., ::-1]`."""


@dataclass(frozen=True, slots=True)
class DeviceTensor:
    """One tensor that already lives in device memory, ready to bind as an input.

    The point is to chain two engines on one stream without a round trip through
    the host: `GpuPreprocessor.run` returns the device address of its float32
    output and `TensorRTSession.run` binds that address directly.
    """

    pointer: int
    """Device address, as `cudaMalloc` and `set_tensor_address` both spell it."""
    shape: tuple[int, ...]
    """The shape to declare for the binding this tensor is passed as."""
    dtype: np.dtype[Any]
    """Element type of the memory at `pointer`.

    Carried so that `TensorRTSession.run` can check a device input against the
    binding's dtype exactly as it checks a host array. Without it the device path was
    the one input the session bound unexamined, and a preprocessor rebuilt at another
    precision would have been read as whatever the engine expected."""
    capacity_bytes: int
    """Bytes the allocation behind `pointer` holds.

    The producer owns a fixed buffer sized for its largest batch and returns a view of
    the first `count` frames, so `shape` is normally smaller than this. The session
    checks `nbytes <= capacity_bytes`, which is the device equivalent of the
    contiguity check a host array gets."""
    stream: int
    """The CUDA stream the producing work was queued on.

    A device input is bound with no copy and no synchronisation, so it is only safe
    when the consumer executes on the same stream. That was a sentence in `run`'s
    docstring; it is a field the session can check."""

    @property
    def nbytes(self) -> int:
        """Bytes `shape` at `dtype` occupies.

        Returns:
            The size of the view this tensor declares, which the allocation must hold.
        """
        return int(np.prod(self.shape)) * self.dtype.itemsize


class PinnedHostBuffer:
    """One page-locked host allocation, viewed as an array.

    Page-locked staging is what makes `cudaMemcpyAsync` a DMA rather than a
    driver-side copy through an internal bounce buffer: measured on this host, a
    batch of eight 1920x1200x3 uint8 frames takes 1.4 ms pinned against 5.6 ms
    pageable (docs/raco-speed-investigation.md §4). Use it as a context manager;
    `close` returns the pages to the OS.
    """

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> None:
        """Allocate `shape` elements of `dtype` in page-locked memory.

        Args:
            shape: The array shape to expose.
            dtype: The element type.

        Raises:
            RuntimeError: When `cudaHostAlloc` fails.
        """
        self.shape: tuple[int, ...] = shape
        self.nbytes: int = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.pointer: int = int(
            check_cuda(cudart.cudaHostAlloc(self.nbytes, cudart.cudaHostAllocDefault), "cudaHostAlloc")[0]
        )
        buffer: Any = ctypes.cast(self.pointer, ctypes.POINTER(ctypes.c_ubyte * self.nbytes)).contents
        self.array: ndarray = np.frombuffer(buffer, dtype=dtype).reshape(shape)
        self.closed: bool = False

    def close(self) -> None:
        """Free the page-locked allocation; safe to call twice."""
        if self.closed:
            return
        self.closed = True
        del self.array
        check_cuda(cudart.cudaFreeHost(self.pointer), "cudaFreeHost")

    def __enter__(self) -> Self:
        """Return the buffer itself, so `with PinnedHostBuffer(...) as staging` works."""
        return self

    def __exit__(self, *_arguments: object) -> None:
        """Free the page-locked allocation."""
        self.close()


@dataclass(slots=True)
class DeviceBufferPool:
    """The device buffers one execution context owns, one per binding name.

    Buffers only ever grow, so a run whose shapes change from call to call pays
    one allocation per new high-water mark rather than one per call.

    Ownership is exception-safe in both directions. `ensure` allocates the
    replacement **before** it gives up the old one, so a failed `cudaMalloc`
    leaves the pool owning exactly what it owned before — still valid, still
    bindable — rather than a freed address recorded at its old capacity. `close`
    drops each record before freeing it, so no pointer can be freed twice even if
    one `cudaFree` fails.
    """

    pointers: dict[str, int] = field(default_factory=dict)
    """Device address per binding name; only the bindings that have a buffer."""
    capacities: dict[str, int] = field(default_factory=dict)
    """Bytes owned per binding name; always the same keys as `pointers`."""

    def ensure(self, name: str, byte_count: int) -> int:
        """Guarantee a buffer of at least `byte_count` bytes for one binding.

        Args:
            name: The binding.
            byte_count: Bytes the current call needs; zero is rounded up to one,
                because `cudaMalloc(0)` has no useful address to bind.

        Returns:
            The device pointer, reallocated when the buffer had to grow.

        Raises:
            RuntimeError: When `cudaMalloc` fails. The pool then still owns the
                buffer it had, unchanged.
        """
        wanted: int = max(byte_count, 1)
        if self.capacities.get(name, 0) >= wanted:
            return self.pointers[name]
        pointer: int = int(check_cuda(cudart.cudaMalloc(wanted), f"cudaMalloc({name})")[0])
        previous: int | None = self.pointers.get(name)
        self.pointers[name] = pointer
        self.capacities[name] = wanted
        if previous is not None:
            check_cuda(cudart.cudaFree(previous), f"cudaFree({name})")
        return pointer

    def close(self) -> None:
        """Free every buffer this pool owns; safe to call twice.

        Raises:
            RuntimeError: When a `cudaFree` fails. The buffers freed before it
                stay freed and forgotten; the ones after it are still owned.
        """
        while self.pointers:
            name: str = next(iter(self.pointers))
            pointer: int = self.pointers.pop(name)
            self.capacities.pop(name, None)
            check_cuda(cudart.cudaFree(pointer), f"cudaFree({name})")


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


def tensorrt_runtime_version() -> str:
    """The full TensorRT version string an engine is only valid under.

    Separate from `tensorrt_version_tag`, which drops the wheel's packaging
    suffix to reproduce the blob's engine-file spelling. The build identity wants
    the whole thing, because `10.13.3.9.post1` and `10.13.3.9` are different
    installations even when they name the same TensorRT.

    Returns:
        `tensorrt.__version__`.
    """
    return str(trt.__version__)


def onnx_digest(onnx_path: Path) -> str:
    """The full content hash of one ONNX graph.

    Args:
        onnx_path: The graph to digest.

    Returns:
        The hex SHA-256 of its bytes.

    Raises:
        FileNotFoundError: When the graph is missing.
    """
    return hashlib.sha256(onnx_path.read_bytes()).hexdigest()


def _canonical_profiles(profiles: Mapping[str, ShapeProfile] | None) -> str:
    """Spell one optimisation-profile mapping as a stable string.

    Sorted by binding name so two equal mappings built in different orders digest
    the same, and written out in full so a re-tuned `opt` shape — which changes
    which tactics TensorRT picks, and therefore the engine — is a different
    identity.

    Args:
        profiles: Optimisation profile per dynamic input, or `None` for a graph
            built at its declared shapes.

    Returns:
        `"none"`, or `name:min|opt|max` per binding joined by `;`.
    """
    if not profiles:
        return "none"
    return ";".join(
        f"{name}:{tuple(profiles[name][0])}|{tuple(profiles[name][1])}|{tuple(profiles[name][2])}"
        for name in sorted(profiles)
    )


def engine_build_identity(
    onnx_path: Path,
    profiles: Mapping[str, ShapeProfile] | None = None,
    *,
    precision: EnginePrecision = DEFAULT_PRECISION,
    workspace_gib: int = DEFAULT_WORKSPACE_GIB,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    device_index: int = 0,
) -> str:
    """Digest everything that decides what one engine's bytes are.

    Every input to `build_fp16_engine`, plus the two facts that make a serialised
    engine non-portable. Leaving any of them out is what lets a stale engine be
    reused: a name built from the file name alone survives replacing the graph,
    and a name built from the graph's digest alone survives re-tuning a profile
    or dropping the builder's optimisation level.

    Args:
        onnx_path: The graph to build from; its bytes are hashed.
        profiles: Optimisation profile per dynamic input.
        precision: The builder precision flag.
        workspace_gib: Tactic workspace ceiling in GiB.
        optimization_level: TensorRT builder search level.
        device_index: CUDA device the engine is built for.

    Returns:
        The first `ONNX_DIGEST_CHARACTERS` hex characters of the identity's SHA-256.

    Raises:
        FileNotFoundError: When the graph is missing.
    """
    fields: str = "\n".join(
        [
            f"graph={onnx_digest(onnx_path)}",
            f"profiles={_canonical_profiles(profiles)}",
            f"precision={precision}",
            f"workspace_gib={workspace_gib}",
            f"optimization_level={optimization_level}",
            f"tensorrt={tensorrt_runtime_version()}",
            f"architecture={compute_capability(device_index)}",
        ]
    )
    return hashlib.sha256(fields.encode()).hexdigest()[:ONNX_DIGEST_CHARACTERS]


def publish_atomically(path: Path, payload: bytes) -> None:
    """Write a complete file through a unique temporary and one rename.

    `path.with_suffix(".tmp")` is not enough: two processes building the same
    engine would write the same temporary, and the loser's partial bytes could be
    renamed over the winner's. A `mkstemp` name in the destination's own
    directory is unique per process and on the same filesystem, so the rename is
    atomic and a reader never sees a half-written engine.

    Args:
        path: Destination.
        payload: The whole file's bytes.

    Raises:
        OSError: When the write or the rename fails; the temporary is removed
            first, so a failure leaves the destination as it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    temporary_path: Path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        # `mkstemp` opens at 0600; an engine cache is not a secret, and a shared
        # `data/cusfm_models` has to stay readable by whoever built it second.
        temporary_path.chmod(PUBLISHED_FILE_MODE)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def blob_engine_path_for(onnx_path: Path, *, profile_tag: str = "", device_index: int = 0) -> Path:
    """The name cuSFM's own runtime gives the engine it builds for one graph.

    Only the prebuilt engines under `MODEL_DIR` carry it; `resolve_engine`
    accepts those and nothing else builds under this name any more.

    Args:
        onnx_path: The graph, e.g. `pycusfm/models/aliked_lightglue/aliked.onnx`.
        profile_tag: A short name for the optimisation profile, inserted after `_fp16`.
        device_index: CUDA device the engine is built for.

    Returns:
        `<onnx dir>/<stem>_fp16[_<profile tag>]_<trt version>_sm_<arch>.engine`.
    """
    tag: str = f"_{profile_tag}" if profile_tag else ""
    return onnx_path.with_name(f"{onnx_path.stem}_fp16{tag}_{tensorrt_version_tag()}_sm_{compute_capability(device_index)}.engine")


def engine_path_for(
    onnx_path: Path,
    profiles: Mapping[str, ShapeProfile] | None = None,
    *,
    profile_tag: str = "",
    device_index: int = 0,
    precision: EnginePrecision = DEFAULT_PRECISION,
    workspace_gib: int = DEFAULT_WORKSPACE_GIB,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
) -> Path:
    """Where the engine built from one ONNX graph and one set of build settings lives.

    The name carries `engine_build_identity`, so nothing that changes the engine's
    bytes — the graph, the profiles, the precision, the workspace, the
    optimisation level, the TensorRT version or the GPU architecture — can reuse
    an engine built from something else. A graph that is not on disk has nothing
    to digest — the prebuilt-engine case — and keeps the blob's own spelling.

    The human-readable `_<trt version>_sm_<arch>` suffix stays even though the
    identity already covers both: it is the blob's own spelling, and it makes an
    engine directory legible without recomputing a digest.

    Args:
        onnx_path: The graph, e.g. `pycusfm/models/aliked_lightglue/aliked.onnx`.
        profiles: Optimisation profile per dynamic input; part of the identity.
        profile_tag: A short readable name for the profile, inserted after the
            precision. Optional now that the profiles themselves are digested,
            but it is what tells two engines apart in a directory listing.
        device_index: CUDA device the engine is built for.
        precision: The builder precision flag.
        workspace_gib: Tactic workspace ceiling in GiB.
        optimization_level: TensorRT builder search level.

    Returns:
        `<onnx dir>/<stem>_<precision>[_<profile tag>]_<identity>_<trt version>_sm_<arch>.engine`,
        or `blob_engine_path_for`'s answer when the graph is missing.
    """
    if not onnx_path.is_file():
        return blob_engine_path_for(onnx_path, profile_tag=profile_tag, device_index=device_index)
    tag: str = f"_{profile_tag}" if profile_tag else ""
    identity: str = engine_build_identity(
        onnx_path,
        profiles,
        precision=precision,
        workspace_gib=workspace_gib,
        optimization_level=optimization_level,
        device_index=device_index,
    )
    machine: str = f"{tensorrt_version_tag()}_sm_{compute_capability(device_index)}"
    return onnx_path.with_name(f"{onnx_path.stem}_{precision}{tag}_{identity}_{machine}.engine")


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
        engine_path: Destination; `publish_atomically` renames a unique temporary onto it.
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
    publish_atomically(engine_path, bytes(serialized))
    publish_atomically(timing_cache_path, bytes(builder_config.get_timing_cache().serialize()))
    print(f"[colsfm] wrote {engine_path} ({engine_path.stat().st_size / 2**20:.1f} MiB)")
    return engine_path


def trusted_prebuilt_engine(onnx_path: Path, *, profile_tag: str = "", device_index: int = 0) -> Path | None:
    """The cuSFM-built engine for one of the blob's own graphs, when there is one.

    The one place an engine is accepted on a file name rather than on a build
    identity, and deliberately narrow: the graph has to be one of
    `TRUSTED_PREBUILT_GRAPHS`, it has to sit in `MODEL_DIR`, the engine has to
    exist, and its name still has to carry this TensorRT version and this GPU
    architecture. Those engines were serialised by cuSFM itself and cost about
    205 s each to rebuild.

    Args:
        onnx_path: The ONNX graph.
        profile_tag: A short name for the optimisation profile.
        device_index: CUDA device the engine is built for.

    Returns:
        The prebuilt engine, or `None` when this graph has no trusted one.
    """
    if onnx_path.parent != MODEL_DIR or onnx_path.stem not in TRUSTED_PREBUILT_GRAPHS:
        return None
    prebuilt: Path = blob_engine_path_for(onnx_path, profile_tag=profile_tag, device_index=device_index)
    return prebuilt if prebuilt.is_file() else None


def resolve_engine(
    onnx_path: Path,
    profiles: Mapping[str, ShapeProfile] | None = None,
    *,
    profile_tag: str = "",
    device_index: int = 0,
    precision: EnginePrecision = DEFAULT_PRECISION,
    workspace_gib: int = DEFAULT_WORKSPACE_GIB,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
) -> Path:
    """Return the cached engine for one graph and one set of build settings.

    Two names are accepted, in this order:

    1. The identity-named engine of `engine_path_for`, which is what any build
       here writes and the only name a locally generated graph — RaCo's, the
       preprocessing ones — can hit. Every input to the build is in that name, so
       a hit really is an engine built from these bytes with these settings.
    2. The blob's own spelling, for the two graphs `TRUSTED_PREBUILT_GRAPHS`
       names; see `trusted_prebuilt_engine`.

    Otherwise the engine is built, with the same settings the name was computed
    from — so the file the caller gets back always matches the name it was
    looked up under.

    Args:
        onnx_path: The ONNX graph.
        profiles: Optimisation profile per dynamic input; see `build_fp16_engine`.
        profile_tag: Names the profile in the cached engine's file name; see
            `engine_path_for`.
        device_index: CUDA device the engine is built for.
        precision: The builder precision flag.
        workspace_gib: Tactic workspace ceiling in GiB.
        optimization_level: TensorRT builder search level, 0 to 5.

    Returns:
        The engine file.
    """
    engine_path: Path = engine_path_for(
        onnx_path,
        profiles,
        profile_tag=profile_tag,
        device_index=device_index,
        precision=precision,
        workspace_gib=workspace_gib,
        optimization_level=optimization_level,
    )
    if engine_path.is_file():
        return engine_path
    prebuilt: Path | None = trusted_prebuilt_engine(onnx_path, profile_tag=profile_tag, device_index=device_index)
    if prebuilt is not None:
        return prebuilt
    return build_fp16_engine(
        onnx_path, engine_path, profiles, workspace_gib=workspace_gib, optimization_level=optimization_level
    )


class TensorRTSession:
    """One deserialised engine, its execution context and its device buffers.

    Buffers are allocated lazily and only ever grow, so a run whose shapes change
    from pair to pair — LightGlue, whose keypoint counts differ per image — pays
    one allocation per new high-water mark rather than one per call.

    Use it as a context manager; `close` frees the device memory and the stream.
    """

    def __init__(self, engine_path: Path, *, reuse_output_buffers: bool = False) -> None:
        """Deserialise one engine.

        Args:
            engine_path: A serialised TensorRT engine built for this GPU and
                this TensorRT version.
            reuse_output_buffers: Keep one host array per output binding and hand
                back views into it instead of allocating per call. Off by default
                because it invalidates the previous call's arrays; the feature
                backends turn it on because they copy what they keep out of the
                arrays before the next call.

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
        self.buffers: DeviceBufferPool = DeviceBufferPool()
        self.reuse_output_buffers: bool = reuse_output_buffers
        self.host_outputs: dict[str, ndarray] = {}
        self.stream: int = int(check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")[0])
        self.closed: bool = False

    def _host_output(self, name: str, shape: tuple[int, ...]) -> ndarray:
        """A host array of `shape` for one output binding, reused when allowed.

        Args:
            name: The output binding.
            shape: The shape TensorRT resolved for this call.

        Returns:
            A writable array. With `reuse_output_buffers` it is the same memory
            every call, so the previous call's view is invalidated; without it a
            fresh `np.empty`, whose first-touch page faults cost about 1 ms per
            megabyte of descriptors (docs/raco-speed-investigation.md §4).
        """
        if not self.reuse_output_buffers:
            return np.empty(shape, dtype=self.dtypes[name])
        cached: ndarray | None = self.host_outputs.get(name)
        if cached is None or cached.size < int(np.prod(shape)):
            cached = np.zeros(int(np.prod(shape)), dtype=self.dtypes[name])
            self.host_outputs[name] = cached
        return cached[: int(np.prod(shape))].reshape(shape)

    def run(self, inputs: Mapping[str, ndarray | DeviceTensor]) -> dict[str, ndarray]:
        """Execute the engine once and return every output as a host array.

        Args:
            inputs: Per input binding, either one contiguous host array already
                in the binding's dtype — which is copied to the device here — or
                a `DeviceTensor` that some earlier work on **this session's
                stream** already produced, which is bound in place with no copy.

        Returns:
            One host array per output binding, at the shape TensorRT resolved.
            With `reuse_output_buffers` these are views into session-owned memory
            and are only valid until the next `run`.

        Raises:
            RuntimeError: When the session is closed, when a profile rejects a
                shape, or when execution fails.
            ValueError: When an input is missing or mistyped, when a host array is
                not contiguous, or when a `DeviceTensor` was produced on another
                stream or declares more bytes than its allocation holds.
        """
        if self.closed:
            raise RuntimeError("This TensorRTSession is closed")
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"TensorRT engine input {name} was not supplied")
            supplied: ndarray | DeviceTensor = inputs[name]
            if supplied.dtype != self.dtypes[name]:
                raise ValueError(f"Input {name} must be {self.dtypes[name]}, got {supplied.dtype}")
            if isinstance(supplied, DeviceTensor):
                if supplied.stream != self.stream:
                    raise ValueError(f"Device input {name} was produced on another stream; binding it in place is unsafe")
                if supplied.nbytes > supplied.capacity_bytes:
                    raise ValueError(
                        f"Device input {name} declares {supplied.nbytes} bytes but its allocation holds {supplied.capacity_bytes}"
                    )
                input_shape: tuple[int, ...] = supplied.shape
            else:
                if not supplied.flags.c_contiguous:
                    raise ValueError(f"Input {name} must be C-contiguous")
                input_shape = tuple(supplied.shape)
            if not self.context.set_input_shape(name, input_shape):
                raise RuntimeError(f"TensorRT profile rejected shape {input_shape} for {name}")
        for name in self.input_names:
            supplied = inputs[name]
            if isinstance(supplied, DeviceTensor):
                self.context.set_tensor_address(name, supplied.pointer)
                continue
            pointer: int = self.buffers.ensure(name, supplied.nbytes)
            self.context.set_tensor_address(name, pointer)
            check_cuda(
                cudart.cudaMemcpyAsync(
                    pointer,
                    supplied.ctypes.data,
                    supplied.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                ),
                f"cudaMemcpyAsync(H2D {name})",
            )
        outputs: dict[str, ndarray] = {}
        for name in self.output_names:
            shape: tuple[int, ...] = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"TensorRT left output {name} with an unresolved shape {shape}")
            host: ndarray = self._host_output(name, shape)
            outputs[name] = host
            self.context.set_tensor_address(name, self.buffers.ensure(name, max(host.nbytes, 1)))
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execution failed")
        for name, host in outputs.items():
            if host.nbytes == 0:
                continue
            check_cuda(
                cudart.cudaMemcpyAsync(
                    host.ctypes.data,
                    self.buffers.pointers[name],
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
        self.buffers.close()

    def __enter__(self) -> Self:
        """Return the session itself, so `with TensorRTSession(...) as session` works."""
        return self

    def __exit__(self, *_arguments: object) -> None:
        """Free the device buffers and the stream."""
        self.close()


def preprocess_engine_path(height: int, width: int, max_batch: int, *, device_index: int = 0) -> Path:
    """Where the generated uint8 preprocessing engine for one input size lives.

    Args:
        height: The consumer engine's input height.
        width: The consumer engine's input width.
        max_batch: The largest batch the profile admits.
        device_index: CUDA device the engine is built for.

    Returns:
        A path under `PREPROCESS_ENGINE_DIR`, keyed by size, batch ceiling,
        TensorRT version and compute capability — the same four things that make
        a serialised engine non-portable.
    """
    tag: str = f"{height}x{width}_b{max_batch}_{tensorrt_version_tag()}_sm_{compute_capability(device_index)}"
    return PREPROCESS_ENGINE_DIR / f"preprocess_bgr_u8_{tag}.engine"


def build_preprocess_engine(
    engine_path: Path,
    *,
    height: int,
    width: int,
    max_batch: int,
    optimal_batch: int,
    workspace_gib: int = 1,
) -> Path:
    """Build the engine that does the host's old `preprocess_image` arithmetic.

    The network is authored with TensorRT's own graph API rather than parsed from
    ONNX, because the `colsfm` environment has no `onnx` package and because
    prepending nodes to `aliked.onnx` would force a fresh ~200 s build of the
    blob's shipped engine. Four layers, in the order the host used to do them:
    `Cast(uint8 -> float32)`, `Transpose(0, 3, 1, 2)`,
    `Gather(axis=1, [2, 1, 0])` for BGR to RGB, and `Div(255)`.

    FP16 is deliberately **not** enabled. Every step is exact in float32 —
    widening a uint8 loses nothing and `x / 255.0f` is one correctly-rounded IEEE
    division — so the tensor this engine hands the detector is bit-for-bit the
    one `numpy` used to build (`tests/colsfm/test_features_trt_gpu_preprocess.py`).

    Args:
        engine_path: Destination; `publish_atomically` renames a unique temporary onto it.
        height: Input height, matching the consumer engine.
        width: Input width, matching the consumer engine.
        max_batch: The profile's `max` batch.
        optimal_batch: The profile's `opt` batch.
        workspace_gib: Tactic workspace ceiling in GiB; four memory-bound layers
            need none of it.

    Returns:
        `engine_path`.

    Raises:
        RuntimeError: When TensorRT refuses to build the network.
        ValueError: When the batch bounds are not `1 <= optimal <= max`.
    """
    if not 1 <= optimal_batch <= max_batch:
        raise ValueError(f"Need 1 <= optimal_batch <= max_batch, got {optimal_batch} and {max_batch}")

    logger: trt.Logger = trt.Logger(trt.Logger.WARNING)
    builder: trt.Builder = trt.Builder(logger)
    network: trt.INetworkDefinition = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    images_u8: trt.ITensor = network.add_input(PREPROCESS_IMAGE_U8_BINDING, trt.uint8, (-1, height, width, 3))
    widened: trt.ITensor = network.add_cast(images_u8, trt.float32).get_output(0)
    planar: trt.IShuffleLayer = network.add_shuffle(widened)
    planar.first_transpose = trt.Permutation([0, 3, 1, 2])
    channel_order: trt.ITensor = network.add_constant(
        (3,), np.asarray(BGR_TO_RGB_INDICES, dtype=np.int32)
    ).get_output(0)
    rgb: trt.ITensor = network.add_gather(planar.get_output(0), channel_order, 1).get_output(0)
    denominator: trt.ITensor = network.add_constant((1, 1, 1, 1), np.asarray([[[[255.0]]]], dtype=np.float32)).get_output(0)
    scaled: trt.ITensor = network.add_elementwise(rgb, denominator, trt.ElementWiseOperation.DIV).get_output(0)
    scaled.name = PREPROCESS_IMAGE_BINDING
    network.mark_output(scaled)

    builder_config: trt.IBuilderConfig = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    profile: trt.IOptimizationProfile = builder.create_optimization_profile()
    profile.set_shape(
        PREPROCESS_IMAGE_U8_BINDING,
        (1, height, width, 3),
        (optimal_batch, height, width, 3),
        (max_batch, height, width, 3),
    )
    builder_config.add_optimization_profile(profile)

    serialized: trt.IHostMemory | None = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError(f"TensorRT could not build the uint8 preprocessing engine for {height}x{width}")
    publish_atomically(engine_path, bytes(serialized))
    return engine_path


class GpuPreprocessor:
    """The decode-to-tensor conversion, moved off the host and onto the stream.

    What the host used to do per image — widen 6.9 M uint8 samples to float32,
    transpose HWC to CHW, reverse the channels, divide by 255, then push 27.6 MB
    over PCIe — cost 16.5 ms of CPU and 2.15 ms of transfer per image
    (docs/raco-speed-investigation.md §4). This class pushes the 6.9 MB uint8
    frame instead and lets a four-layer TensorRT engine do the rest, on the same
    stream as the detector so the two never need a host synchronisation between
    them.

    Fill `staging_bhwc[:count]` with `cv2.imread`-order BGR frames, call `run`,
    and bind the `DeviceTensor` it returns as the detector's `image` input. The
    staging buffer may be refilled once the detector's own `run` has returned,
    because that call synchronises the shared stream.
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        max_batch: int,
        optimal_batch: int,
        stream: int,
        device_index: int = 0,
    ) -> None:
        """Load or build the preprocessing engine and allocate its buffers.

        Args:
            height: Input height, matching the consumer engine.
            width: Input width, matching the consumer engine.
            max_batch: The largest batch that will be passed to `run`.
            optimal_batch: The batch to tune the profile for.
            stream: The CUDA stream the consumer engine also executes on.
            device_index: CUDA device the engine is built for.

        Raises:
            RuntimeError: When the engine cannot be built or deserialised.
        """
        engine_path: Path = preprocess_engine_path(height, width, max_batch, device_index=device_index)
        if not engine_path.is_file():
            build_preprocess_engine(
                engine_path, height=height, width=width, max_batch=max_batch, optimal_batch=optimal_batch
            )
        logger: trt.Logger = trt.Logger(trt.Logger.ERROR)
        self.runtime: trt.Runtime = trt.Runtime(logger)
        engine: trt.ICudaEngine | None = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}; it was built for another GPU or TensorRT")
        self.engine: trt.ICudaEngine = engine
        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        self.height: int = height
        self.width: int = width
        self.max_batch: int = max_batch
        self.stream: int = stream
        self.closed: bool = False
        self.buffers: DeviceBufferPool = DeviceBufferPool()
        self.staging: PinnedHostBuffer = PinnedHostBuffer((max_batch, height, width, 3), np.dtype(np.uint8))
        self.staging_bhwc: UInt8[ndarray, "max_batch height width 3"] = self.staging.array
        self.output_nbytes: int = max_batch * 3 * height * width * np.dtype(np.float32).itemsize
        try:
            self.input_pointer: int = self.buffers.ensure(PREPROCESS_IMAGE_U8_BINDING, self.staging.nbytes)
            self.output_pointer: int = self.buffers.ensure(PREPROCESS_IMAGE_BINDING, self.output_nbytes)
        except BaseException:
            self.close()
            raise

    def run(self, count: int) -> DeviceTensor:
        """Convert the first `count` staged frames and leave the result on the device.

        Args:
            count: How many slots of `staging_bhwc` hold a frame, 1 to `max_batch`.

        Returns:
            The float32 planar RGB batch, `[count, 3, height, width]`, in device
            memory. Nothing is synchronised: the work is queued on the shared
            stream, ahead of whatever the caller enqueues next.

        Raises:
            RuntimeError: When the session is closed or execution fails.
            ValueError: When `count` is outside the profile.
        """
        if self.closed:
            raise RuntimeError("This GpuPreprocessor is closed")
        if not 1 <= count <= self.max_batch:
            raise ValueError(f"count must be in [1, {self.max_batch}], got {count}")
        input_shape: tuple[int, int, int, int] = (count, self.height, self.width, 3)
        if not self.context.set_input_shape(PREPROCESS_IMAGE_U8_BINDING, input_shape):
            raise RuntimeError(f"The preprocessing profile rejected shape {input_shape}")
        check_cuda(
            cudart.cudaMemcpyAsync(
                self.input_pointer,
                self.staging.pointer,
                count * self.height * self.width * 3,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync(H2D image_u8)",
        )
        self.context.set_tensor_address(PREPROCESS_IMAGE_U8_BINDING, self.input_pointer)
        self.context.set_tensor_address(PREPROCESS_IMAGE_BINDING, self.output_pointer)
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("The uint8 preprocessing engine failed to execute")
        return DeviceTensor(
            pointer=self.output_pointer,
            shape=(count, 3, self.height, self.width),
            dtype=np.dtype(np.float32),
            capacity_bytes=self.output_nbytes,
            stream=self.stream,
        )

    def close(self) -> None:
        """Free the device buffers and the page-locked staging; safe to call twice.

        Also the constructor's unwind path, so it releases whatever was acquired
        rather than assuming every allocation happened.
        """
        if self.closed:
            return
        self.closed = True
        del self.staging_bhwc
        self.staging.close()
        self.buffers.close()

    def __enter__(self) -> Self:
        """Return the preprocessor itself, so `with GpuPreprocessor(...)` works."""
        return self

    def __exit__(self, *_arguments: object) -> None:
        """Free the device buffers and the page-locked staging."""
        self.close()

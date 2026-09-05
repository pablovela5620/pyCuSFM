"""Benchmark GPU JPEG decoding (nvJPEG, through torchcodec) against the CPU decode path colsfm runs today.

**Why.** On Galileo the feature-extraction stage costs ~33 ms per image, of which
only ~11 ms is the TensorRT engine. The rest is `colsfm.features_trt.preprocess_image`:
`cv2.imread` -> `cv2.resize` -> BGR-to-planar-RGB float32 -> host-to-device copy.
This tool measures whether moving the decode onto the RTX 5090 with nvJPEG
removes that overhead, and whether the pixels that come out are the same pixels.

**The baseline is OpenCV, not PIL.** `colsfm/features_trt.py` decodes with
`cv2.imread(..., cv2.IMREAD_COLOR)` and resizes with `cv2.resize(...,
interpolation=cv2.INTER_LINEAR)` — no aspect-ratio preservation, a plain
stretch. `preprocess_image` is reproduced here byte for byte rather than
imported, so that this tool can run in its own `decode-bench` environment
without dragging in `pycolmap` and TensorRT.

**Two nvJPEG facts decide the shape of the benchmark.** First, torchcodec 0.16's
CUDA entry point is `decode_jpeg(..., device="cuda")`, not `decode_image`, which
takes no `device` at all. Second, `decode_jpeg` accepts a *list* of encoded
buffers and puts the whole list through one nvJPEG call — the release notes say
this is much faster than one image at a time, and method (d) is what tests it.
`decode_png` has no `device` parameter, so PNG has no CUDA path: nvJPEG is
JPEG-only and KITTI (`data/kitti/06/image_0/*.png`) is therefore measured on CPU
only, which the results table states rather than hides.

**Galileo's resize is an identity.** Its frames are already 1920x1200, exactly
the engine's input, so `cv2.resize` there is a copy and the pixel-equivalence
numbers isolate the decoders (libjpeg-turbo against nvJPEG) with no resampling
mixed in. KITTI's 1226x370 frames are genuinely upscaled, which is why both
datasets are reported.

Run it with `pixi run -e decode-bench decode-bench`.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Final, Literal, TypeAlias

import cv2
import numpy as np
import torch
import tyro
from jaxtyping import Float32, UInt8
from numpy import ndarray
from torch import Tensor
from torchcodec.decoders import decode_jpeg, decode_png

NETWORK_HEIGHT: Final[int] = 1200
"""`aliked.onnx`'s declared input height; mirrors `colsfm.features_trt.NETWORK_HEIGHT`."""

NETWORK_WIDTH: Final[int] = 1920
"""`aliked.onnx`'s declared input width; mirrors `colsfm.features_trt.NETWORK_WIDTH`."""

NetworkImage: TypeAlias = Float32[ndarray, "1 3 network_height network_width"]
"""One preprocessed image as the engine takes it: planar RGB in [0, 1]."""

NetworkBatchGPU: TypeAlias = Float32[Tensor, "batch 3 network_height network_width"]
"""One preprocessed batch on the GPU, in the engine's layout and dtype."""

EncodedImage: TypeAlias = UInt8[Tensor, " num_bytes"]
"""One image's file bytes, as torchcodec's decoders take them."""

DatasetName: TypeAlias = Literal["galileo", "kitti"]
"""Which image set a measurement belongs to."""

MethodName: TypeAlias = Literal[
    "a_cv2_cpu_full",
    "a_cv2_decode_only",
    "b_torchcodec_cpu_decode",
    "c_nvjpeg_cuda_single",
    "d_nvjpeg_cuda_batch",
    "e_nvjpeg_cuda_batch_threaded",
]
"""The decode strategies under test; see this module's docstring."""

BATCH_SIZES: Final[tuple[int, ...]] = (8, 16, 32)
"""nvJPEG batch sizes for methods (d) and (e); 8 is `colsfm.features_raco.DEFAULT_BATCH_SIZE`."""

READER_THREAD_COUNTS: Final[tuple[int, ...]] = (4, 8)
"""File-reader thread counts for method (e); file I/O is not GPU work."""

EQUIVALENCE_SAMPLE_SIZE: Final[int] = 5
"""Images decoded both ways for the pixel-difference check."""


@dataclass(frozen=True, slots=True)
class BenchConfig:
    """Which images to measure, and how hard."""

    galileo_dir: Path = Path("data/r2b_galileo")
    """Root of the Galileo JPEG sample; searched recursively for `*.jpeg`."""
    kitti_dir: Path = Path("data/kitti/06/image_0")
    """Directory of KITTI PNG frames; PNG has no nvJPEG path, so these run on CPU."""
    num_images: int = 226
    """Images per dataset, matching the Galileo extraction stage."""
    passes: int = 3
    """Timed passes per method; the reported figure is the median."""
    cpu_workers: int = 8
    """Threads for the baseline's decode-ahead pool, `colsfm.features_trt.DEFAULT_PREPROCESSING_WORKERS`."""
    docs_path: Path | None = None
    """When set, the results table is also written to this markdown file."""


@dataclass(frozen=True, slots=True)
class Measurement:
    """One method's cost on one dataset."""

    dataset: DatasetName
    """Which image set."""
    method: MethodName
    """Which decode strategy."""
    label: str
    """Human-readable variant, e.g. the batch size or thread count."""
    milliseconds_per_image: float
    """Median wall time per image over `BenchConfig.passes` passes."""
    substeps_ms: dict[str, float] = field(default_factory=dict)
    """Per-image milliseconds of each sub-step, when the method breaks down."""


@dataclass(frozen=True, slots=True)
class Equivalence:
    """How far nvJPEG's pixels sit from OpenCV's, in 0-255 units."""

    dataset: DatasetName
    """Which image set."""
    max_abs_difference: float
    """Largest absolute per-pixel difference across the sampled images."""
    mean_abs_difference: float
    """Mean absolute per-pixel difference across the sampled images."""
    fraction_above_one: float
    """Share of pixels differing by more than one 0-255 level."""


def resolve_device() -> Literal["cuda"]:
    """Fail loudly rather than silently benchmark a CPU.

    Returns:
        The string "cuda".

    Raises:
        RuntimeError: When no CUDA device is visible.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark measures nvJPEG; no CUDA device is visible")
    return "cuda"


def list_images(directory: Path, suffix: str, limit: int) -> list[Path]:
    """Collect a deterministic slice of a dataset's frames.

    Args:
        directory: Root to search recursively.
        suffix: File extension including the dot, e.g. ".jpeg".
        limit: Maximum number of paths to return.

    Returns:
        Up to `limit` sorted paths.

    Raises:
        FileNotFoundError: When the directory holds no matching file.
    """
    paths: list[Path] = sorted(path for path in directory.rglob(f"*{suffix}") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No {suffix} files under {directory}")
    return paths[:limit]


# ── (a) the path colsfm runs today ───────────────────────────────────────────


def preprocess_image_cv2(image_path: Path) -> NetworkImage:
    """Reproduce `colsfm.features_trt.preprocess_image` exactly.

    Args:
        image_path: The file to decode.

    Returns:
        Float32 `[1, 3, 1200, 1920]` planar RGB in [0, 1].

    Raises:
        ValueError: When OpenCV cannot decode the file.
    """
    bgr_hwc: ndarray | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr_hwc is None:
        raise ValueError(f"OpenCV could not decode {image_path}")
    resized_bgr_hwc: ndarray = cv2.resize(bgr_hwc, (NETWORK_WIDTH, NETWORK_HEIGHT), interpolation=cv2.INTER_LINEAR)
    network_chw: Float32[ndarray, "3 network_height network_width"] = np.ascontiguousarray(
        resized_bgr_hwc[..., ::-1].transpose(2, 0, 1), dtype=np.float32
    )
    network_chw /= np.float32(255.0)
    return network_chw[None]


def time_baseline_substeps(paths: Sequence[Path], device: Literal["cuda"]) -> dict[str, float]:
    """Time each stage of the current path separately, including the host-to-device copy.

    Args:
        paths: Images to walk, single-threaded so the stages do not overlap.
        device: The CUDA device the copy targets.

    Returns:
        Milliseconds per image for "decode", "resize", "to_chw_float" and "htod".
    """
    decode_s: float = 0.0
    resize_s: float = 0.0
    convert_s: float = 0.0
    htod_s: float = 0.0
    for image_path in paths:
        start: float = time.perf_counter()
        bgr_hwc: ndarray | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        decode_s += time.perf_counter() - start
        if bgr_hwc is None:
            raise ValueError(f"OpenCV could not decode {image_path}")

        start = time.perf_counter()
        resized_bgr_hwc: ndarray = cv2.resize(bgr_hwc, (NETWORK_WIDTH, NETWORK_HEIGHT), interpolation=cv2.INTER_LINEAR)
        resize_s += time.perf_counter() - start

        start = time.perf_counter()
        network_chw: Float32[ndarray, "3 network_height network_width"] = np.ascontiguousarray(
            resized_bgr_hwc[..., ::-1].transpose(2, 0, 1), dtype=np.float32
        )
        network_chw /= np.float32(255.0)
        convert_s += time.perf_counter() - start

        start = time.perf_counter()
        on_device: Tensor = torch.from_numpy(network_chw[None]).to(device=device, non_blocking=False)
        torch.cuda.synchronize()
        htod_s += time.perf_counter() - start
        del on_device
    count: int = len(paths)
    return {
        "decode": 1000.0 * decode_s / count,
        "resize": 1000.0 * resize_s / count,
        "to_chw_float": 1000.0 * convert_s / count,
        "htod": 1000.0 * htod_s / count,
    }


def run_baseline_full(paths: Sequence[Path], workers: int, device: Literal["cuda"]) -> None:
    """Run the current path end to end, with the same decode-ahead pool colsfm uses.

    Args:
        paths: Images to process.
        workers: Decode-and-resize threads.
        device: The CUDA device the copies target.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for network_bchw in pool.map(preprocess_image_cv2, paths):
            on_device: Tensor = torch.from_numpy(network_bchw).to(device=device, non_blocking=False)
            del on_device
    torch.cuda.synchronize()


# ── (b)-(e) torchcodec ───────────────────────────────────────────────────────


def read_bytes(image_path: Path) -> EncodedImage:
    """Read one file into the uint8 tensor torchcodec's decoders take.

    Args:
        image_path: The file to read.

    Returns:
        A writable uint8 tensor of the file's bytes.
    """
    return torch.frombuffer(bytearray(image_path.read_bytes()), dtype=torch.uint8)


def run_torchcodec_cpu(paths: Sequence[Path], is_jpeg: bool) -> None:
    """Decode every image on the CPU with torchcodec, one at a time.

    Args:
        paths: Images to decode.
        is_jpeg: Whether to use the JPEG decoder; PNG goes through `decode_png`.
    """
    decoder: Callable[[EncodedImage], Tensor] = decode_jpeg if is_jpeg else decode_png
    for image_path in paths:
        decoded: Tensor = decoder(read_bytes(image_path))
        del decoded


def run_nvjpeg_single(paths: Sequence[Path], device: Literal["cuda"]) -> None:
    """Decode every JPEG on the GPU one image per nvJPEG call.

    Args:
        paths: JPEG images to decode.
        device: The CUDA device.
    """
    for image_path in paths:
        decoded: Tensor = decode_jpeg(read_bytes(image_path), device=device)
        del decoded
    torch.cuda.synchronize()


def to_engine_batch(decoded_chw: Sequence[Tensor]) -> NetworkBatchGPU:
    """Turn nvJPEG's uint8 CHW tensors into the engine's input, entirely on the GPU.

    The resize matches `cv2.INTER_LINEAR`: bilinear with half-pixel centres
    (`align_corners=False`) and no antialiasing, which is what OpenCV does. The
    normalisation is the baseline's `/ 255`.

    Args:
        decoded_chw: uint8 `[3, h, w]` CUDA tensors, one per image.

    Returns:
        Float32 `[batch, 3, 1200, 1920]` planar RGB in [0, 1] on the GPU.
    """
    stacked: UInt8[Tensor, "batch 3 h w"] = torch.stack(list(decoded_chw), dim=0)
    as_float: Float32[Tensor, "batch 3 h w"] = stacked.to(dtype=torch.float32)
    if as_float.shape[-2:] != (NETWORK_HEIGHT, NETWORK_WIDTH):
        as_float = torch.nn.functional.interpolate(
            as_float, size=(NETWORK_HEIGHT, NETWORK_WIDTH), mode="bilinear", align_corners=False, antialias=False
        )
    return (as_float / 255.0).contiguous()


def _chunked(items: Sequence[Path], size: int) -> Iterator[list[Path]]:
    """Group paths into fixed-size batches, the last one short.

    Args:
        items: Paths in order.
        size: Batch size.

    Yields:
        Non-empty lists of at most `size` paths.
    """
    cursor: Iterator[Path] = iter(items)
    while chunk := list(islice(cursor, size)):
        yield chunk


def run_nvjpeg_batched(paths: Sequence[Path], batch_size: int, device: Literal["cuda"], reader_threads: int = 0) -> None:
    """Decode JPEGs in nvJPEG batches and build the engine's input on the GPU.

    Args:
        paths: JPEG images to decode.
        batch_size: Images per nvJPEG call.
        device: The CUDA device.
        reader_threads: When positive, read the file bytes with this many
            threads (method (e)); when zero, read them serially (method (d)).
    """
    pool: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=reader_threads) if reader_threads > 0 else None
    try:
        for chunk in _chunked(paths, batch_size):
            encoded: list[EncodedImage] = (
                list(pool.map(read_bytes, chunk)) if pool is not None else [read_bytes(path) for path in chunk]
            )
            decoded: list[Tensor] = decode_jpeg(encoded, device=device)
            batch: NetworkBatchGPU = to_engine_batch(decoded)
            del batch, decoded
    finally:
        if pool is not None:
            pool.shutdown()
    torch.cuda.synchronize()


# ── timing harness ───────────────────────────────────────────────────────────


def median_ms_per_image(run: Callable[[], None], count: int, passes: int) -> float:
    """Warm up once, then take the median of `passes` timed runs.

    Args:
        run: The pass to time; it must synchronise the GPU before returning.
        count: Images the pass covers, for the per-image divisor.
        passes: Timed passes.

    Returns:
        Median milliseconds per image.
    """
    run()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(passes):
        start: float = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append(1000.0 * (time.perf_counter() - start) / count)
    return statistics.median(samples)


def measure_pixel_equivalence(paths: Sequence[Path], dataset: DatasetName, device: Literal["cuda"]) -> Equivalence:
    """Compare the baseline's tensor with the nvJPEG tensor, in 0-255 units.

    Args:
        paths: JPEG images to decode both ways.
        dataset: Which image set, for the record.
        device: The CUDA device.

    Returns:
        Max, mean and above-one-level statistics of the absolute difference.
    """
    sample: list[Path] = list(paths[:EQUIVALENCE_SAMPLE_SIZE])
    baseline_255: Float32[ndarray, "n 3 network_height network_width"] = (
        np.concatenate([preprocess_image_cv2(path) for path in sample], axis=0) * 255.0
    )
    decoded: list[Tensor] = decode_jpeg([read_bytes(path) for path in sample], device=device)
    gpu_255: Float32[ndarray, "n 3 network_height network_width"] = (
        (to_engine_batch(decoded) * 255.0).cpu().numpy()
    )
    difference: Float32[ndarray, "n 3 network_height network_width"] = np.abs(baseline_255 - gpu_255)
    return Equivalence(
        dataset=dataset,
        max_abs_difference=float(difference.max()),
        mean_abs_difference=float(difference.mean()),
        fraction_above_one=float((difference > 1.0).mean()),
    )


def demonstrate_zero_copy_handoff(device: Literal["cuda"]) -> str:
    """Show that a torch CUDA tensor's `data_ptr()` is a plain device pointer a foreign runtime can read.

    TensorRT's `IExecutionContext.set_tensor_address(name, address)` takes an
    integer device address and nothing else — the same thing `cuda.bindings`
    hands it in `colsfm.tensorrt_runtime`. This proves the address torch gives
    out is usable outside torch by copying from it with `libcudart` reached
    through `ctypes`, i.e. by a consumer that knows nothing about torch, and
    checking the bytes against torch's own view.

    Args:
        device: The CUDA device.

    Returns:
        A one-line verdict for the report.
    """
    import ctypes
    import ctypes.util

    tensor: Float32[Tensor, "2 3 4 4"] = torch.rand((2, 3, 4, 4), device=device, dtype=torch.float32).contiguous()
    address: int = tensor.data_ptr()
    num_bytes: int = tensor.numel() * tensor.element_size()

    library_name: str | None = ctypes.util.find_library("cudart")
    cudart: ctypes.CDLL = ctypes.CDLL(library_name or "libcudart.so.13")
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpy.restype = ctypes.c_int

    host_buffer: Float32[ndarray, "2 3 4 4"] = np.empty(tuple(tensor.shape), dtype=np.float32)
    device_to_host: Final[int] = 2
    status: int = cudart.cudaMemcpy(
        host_buffer.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(address), num_bytes, device_to_host
    )
    if status != 0:
        return f"FAILED: cudaMemcpy from data_ptr() returned {status}"
    matches: bool = bool(np.array_equal(host_buffer, tensor.cpu().numpy()))
    return (
        f"OK: torch CUDA tensor data_ptr()=0x{address:x} ({num_bytes} bytes) was read by libcudart through ctypes, "
        f"outside torch, bytes identical={matches}. TensorRT's set_tensor_address takes exactly this integer."
    )


# ── driver ───────────────────────────────────────────────────────────────────


def benchmark_dataset(
    paths: Sequence[Path], dataset: DatasetName, is_jpeg: bool, config: BenchConfig, device: Literal["cuda"]
) -> list[Measurement]:
    """Run every applicable method over one dataset.

    Args:
        paths: The dataset's frames.
        dataset: Which image set.
        is_jpeg: Whether nvJPEG applies; PNG datasets get the CPU methods only.
        config: Pass counts and worker counts.
        device: The CUDA device.

    Returns:
        One measurement per method and variant.
    """
    count: int = len(paths)
    results: list[Measurement] = []

    substeps: dict[str, float] = time_baseline_substeps(paths, device)
    results.append(
        Measurement(
            dataset=dataset,
            method="a_cv2_decode_only",
            label="single-thread",
            milliseconds_per_image=substeps["decode"],
        )
    )
    results.append(
        Measurement(
            dataset=dataset,
            method="a_cv2_cpu_full",
            label=f"{config.cpu_workers} threads",
            milliseconds_per_image=median_ms_per_image(
                lambda: run_baseline_full(paths, config.cpu_workers, device), count, config.passes
            ),
            substeps_ms=substeps,
        )
    )
    results.append(
        Measurement(
            dataset=dataset,
            method="b_torchcodec_cpu_decode",
            label="single-thread",
            milliseconds_per_image=median_ms_per_image(
                lambda: run_torchcodec_cpu(paths, is_jpeg), count, config.passes
            ),
        )
    )
    if not is_jpeg:
        return results

    results.append(
        Measurement(
            dataset=dataset,
            method="c_nvjpeg_cuda_single",
            label="batch 1",
            milliseconds_per_image=median_ms_per_image(lambda: run_nvjpeg_single(paths, device), count, config.passes),
        )
    )
    for batch_size in BATCH_SIZES:
        results.append(
            Measurement(
                dataset=dataset,
                method="d_nvjpeg_cuda_batch",
                label=f"batch {batch_size}",
                milliseconds_per_image=median_ms_per_image(
                    lambda size=batch_size: run_nvjpeg_batched(paths, size, device), count, config.passes
                ),
            )
        )
    for threads in READER_THREAD_COUNTS:
        for batch_size in BATCH_SIZES:
            results.append(
                Measurement(
                    dataset=dataset,
                    method="e_nvjpeg_cuda_batch_threaded",
                    label=f"batch {batch_size}, {threads} readers",
                    milliseconds_per_image=median_ms_per_image(
                        lambda size=batch_size, workers=threads: run_nvjpeg_batched(paths, size, device, workers),
                        count,
                        config.passes,
                    ),
                )
            )
    return results


def format_table(results: Sequence[Measurement]) -> str:
    """Render the measurements as a markdown table.

    Args:
        results: Every measurement taken.

    Returns:
        A markdown table, one row per measurement.
    """
    lines: list[str] = ["| dataset | method | variant | ms/image |", "| --- | --- | --- | --- |"]
    for result in results:
        lines.append(f"| {result.dataset} | {result.method} | {result.label} | {result.milliseconds_per_image:.2f} |")
    return "\n".join(lines)


def main(config: BenchConfig) -> None:
    """Measure every decode path and print the report.

    Args:
        config: Which images to measure and how many passes to take.
    """
    device: Literal["cuda"] = resolve_device()
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  cuda {torch.version.cuda}")

    galileo_paths: list[Path] = list_images(config.galileo_dir, ".jpeg", config.num_images)
    kitti_paths: list[Path] = list_images(config.kitti_dir, ".png", config.num_images)
    print(f"galileo: {len(galileo_paths)} jpeg   kitti: {len(kitti_paths)} png")
    print("note: decode_png takes no `device` argument in torchcodec 0.16 — PNG has no nvJPEG path, KITTI is CPU only.")

    results: list[Measurement] = []
    results += benchmark_dataset(galileo_paths, "galileo", True, config, device)
    results += benchmark_dataset(kitti_paths, "kitti", False, config, device)

    print("\n" + format_table(results))

    baseline: Measurement = next(r for r in results if r.dataset == "galileo" and r.method == "a_cv2_cpu_full")
    print("\nbaseline sub-steps (galileo, ms/image, single-threaded):")
    for name, value in baseline.substeps_ms.items():
        print(f"  {name:14s} {value:6.2f}")

    equivalence: Equivalence = measure_pixel_equivalence(galileo_paths, "galileo", device)
    print(
        f"\npixel equivalence (galileo, {EQUIVALENCE_SAMPLE_SIZE} images, 0-255 units): "
        f"max={equivalence.max_abs_difference:.3f} mean={equivalence.mean_abs_difference:.4f} "
        f"fraction>1={equivalence.fraction_above_one:.6f}"
    )

    print("\nzero-copy handoff: " + demonstrate_zero_copy_handoff(device))

    if config.docs_path is not None:
        config.docs_path.write_text(format_table(results) + "\n")
        print(f"\nwrote {config.docs_path}")


if __name__ == "__main__":
    main(tyro.cli(BenchConfig))

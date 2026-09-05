"""Benchmark the v3.0 release RaCo-ALIKED + LightGlue+ pipeline graph on this GPU.

Runs `raco_aliked_lightglue_pipeline_k2048.onnx` (fabio-sim/LightGlue-ONNX
release v3.0, the graph the blog post measures) through ONNX Runtime's TensorRT
execution provider with FP16, at the blog's own square resolutions and at the
resolution `colsfm` deploys, so the blog's per-pair latency can be compared with
our two-engine split.

Run it in the `bench` environment: it is the only one carrying `onnxruntime-gpu`
with `TensorrtExecutionProvider`. That provider only loads because
`[feature.bench.activation.env]` puts the wheel's `tensorrt_libs`, `cudnn` and
`cu13` directories on `LD_LIBRARY_PATH`; without it ONNX Runtime advertises the
provider, fails to register it and falls through to CPU. The script therefore
refuses to report a number unless the session really came up on TensorRT.

**One configuration per process.** The TensorRT execution provider needed 69 GB
of host memory to build engines for four configurations in one process, so the
defaults measure a single (resolution, pair count) point and
`tools/audit/raco_speed/run_release_ort.sh` sequences the points, one process
each, with an RSS watchdog.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import onnxruntime as ort
import tyro
from jaxtyping import Float32
from numpy import ndarray

InterleavedPairs: TypeAlias = Float32[ndarray, "two_pairs 3 height width"]
"""Interleaved image pairs, the release pipeline graph's single `images` input."""

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
"""Repository root, three levels above `tools/audit/raco_speed`."""

TRT_PROVIDER: Final[str] = "TensorrtExecutionProvider"
"""The provider this benchmark exists to measure."""


def _peak_rss_gb() -> float:
    """Peak resident set size of this process so far.

    Returns:
        `VmHWM` from `/proc/self/status`, in gibibytes.
    """
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / (1024.0 * 1024.0)
    return float("nan")


@dataclass(slots=True)
class Measurement:
    """One (resolution, pair count) latency point."""

    label: str
    """Human-readable configuration name."""
    height: int
    """Network input height in pixels."""
    width: int
    """Network input width in pixels."""
    pair_count: int
    """Image pairs per execution."""
    median_ms_per_call: float
    """Median wall time of one warm execution, in milliseconds."""
    median_ms_per_pair: float
    """`median_ms_per_call / pair_count`."""
    min_ms_per_call: float
    """Fastest warm execution, in milliseconds."""
    max_ms_per_call: float
    """Slowest warm execution, in milliseconds."""
    build_seconds: float
    """Wall time of the first call, which builds and serialises the engines."""
    peak_rss_gb: float
    """Process peak RSS after the configuration finished, in gibibytes."""
    device_bound_inputs: bool
    """Whether the `images` input was already on the GPU (H2D excluded)."""
    device_bound_outputs: bool
    """Whether the outputs stayed on the GPU (D2H excluded)."""


@dataclass(slots=True)
class BenchConfig:
    """What to measure and where the graph lives."""

    model_path: Path = REPO_ROOT / "data" / "cusfm_models" / "raco_release" / "raco_aliked_lightglue_pipeline_k2048.onnx"
    """The release pipeline graph."""
    cache_dir: Path = Path("/tmp/ort_trt_cache")
    """TensorRT engine and timing cache directory, outside the repo."""
    height: int = 1024
    """Network input height in pixels; must be a multiple of 32."""
    width: int = 1024
    """Network input width in pixels; must be a multiple of 32."""
    pair_count: int = 1
    """Image pairs per execution."""
    repeats: int = 30
    """Timed executions, after warm-up."""
    warmups: int = 5
    """Untimed executions."""
    output_path: Path = Path("/tmp/raco_speed/release_ort.json")
    """Where the measurement is appended, one JSON object per line."""


def _providers(cache_dir: Path) -> list[tuple[str, dict[str, object]]]:
    """The TensorRT-first provider list, mirroring upstream's `infer` command.

    Args:
        cache_dir: Directory for the engine and timing caches.

    Returns:
        Provider specifications in ONNX Runtime's priority order.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    return [
        (
            TRT_PROVIDER,
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir / "engines"),
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": str(cache_dir / "timings"),
                "trt_fp16_enable": True,
                "trt_max_workspace_size": 8 << 30,
            },
        ),
        ("CUDAExecutionProvider", {}),
        ("CPUExecutionProvider", {}),
    ]


def measure(session: ort.InferenceSession, config: BenchConfig) -> Measurement:
    """Time one configuration of the release pipeline graph.

    Inputs are uploaded once and bound on the device, so the reported time is
    the GPU-side pipeline plus ONNX Runtime's dispatch, without the per-call
    host-to-device copy of the image tensor.

    Args:
        session: A live ONNX Runtime session over the pipeline graph.
        config: Shape, pair count, repeat and warm-up counts.

    Returns:
        The configuration's latency, engine build time and peak RSS.
    """
    rng: np.random.Generator = np.random.default_rng(0)
    images: InterleavedPairs = rng.random((2 * config.pair_count, 3, config.height, config.width), dtype=np.float32)
    input_name: str = session.get_inputs()[0].name
    output_names: list[str] = [output.name for output in session.get_outputs()]

    binding: ort.IOBinding = session.io_binding()
    device_input: ort.OrtValue = ort.OrtValue.ortvalue_from_numpy(images, "cuda", 0)
    binding.bind_ortvalue_input(input_name, device_input)
    device_outputs: bool = True
    try:
        for name in output_names:
            binding.bind_output(name, "cuda", 0)
    except Exception:  # noqa: BLE001 - fall back to host outputs if a branch is not on the GPU
        device_outputs = False
        binding = session.io_binding()
        binding.bind_ortvalue_input(input_name, device_input)
        for name in output_names:
            binding.bind_output(name)

    started: float = time.perf_counter()
    session.run_with_iobinding(binding)
    build_seconds: float = time.perf_counter() - started
    for _ in range(config.warmups):
        session.run_with_iobinding(binding)
    samples_ms: list[float] = []
    for _ in range(config.repeats):
        call_started: float = time.perf_counter()
        session.run_with_iobinding(binding)
        samples_ms.append((time.perf_counter() - call_started) * 1e3)
    median_ms: float = float(np.median(samples_ms))
    return Measurement(
        label=f"release-pipeline {config.height}x{config.width} B={config.pair_count}",
        height=config.height,
        width=config.width,
        pair_count=config.pair_count,
        median_ms_per_call=median_ms,
        median_ms_per_pair=median_ms / config.pair_count,
        min_ms_per_call=float(np.min(samples_ms)),
        max_ms_per_call=float(np.max(samples_ms)),
        build_seconds=build_seconds,
        peak_rss_gb=_peak_rss_gb(),
        device_bound_inputs=True,
        device_bound_outputs=device_outputs,
    )


def main(config: BenchConfig) -> None:
    """Measure one configuration and append it to the results file.

    Args:
        config: What to measure and where to write it.

    Raises:
        FileNotFoundError: If the release graph is missing.
        RuntimeError: If the session did not come up on the TensorRT provider.
    """
    if not config.model_path.is_file():
        raise FileNotFoundError(f"No release pipeline graph at {config.model_path}")
    available: set[str] = set(ort.get_available_providers())
    selected: list[tuple[str, dict[str, object]]] = [
        provider for provider in _providers(config.cache_dir) if provider[0] in available
    ]
    print(f"requested providers: {[name for name, _ in selected]}", flush=True)
    session: ort.InferenceSession = ort.InferenceSession(str(config.model_path), ort.SessionOptions(), selected)
    active: list[str] = session.get_providers()
    print(f"active providers: {active}", flush=True)
    if TRT_PROVIDER not in active:
        raise RuntimeError(f"TensorRT provider did not register; active providers are {active}")

    result: Measurement = measure(session, config)
    print(
        f"{result.label:34s} call {result.median_ms_per_call:8.2f} ms  "
        f"per-pair {result.median_ms_per_pair:7.2f} ms  "
        f"(build {result.build_seconds:.1f} s, peak RSS {result.peak_rss_gb:.1f} GB)",
        flush=True,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result)) + "\n")
    print(f"appended to {config.output_path}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(BenchConfig))

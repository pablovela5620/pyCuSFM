"""Benchmark the v3.0 release RaCo-ALIKED + LightGlue+ pipeline graph on this GPU.

Runs `raco_aliked_lightglue_pipeline_k2048.onnx` (fabio-sim/LightGlue-ONNX
release v3.0, the graph the blog post measures) through ONNX Runtime's TensorRT
execution provider with FP16, at the blog's own square resolutions and at the
resolution `colsfm` deploys, so the blog's per-pair latency can be compared with
our two-engine split.

Run it in the `bench` environment: it is the only one carrying `onnxruntime-gpu`
with `TensorrtExecutionProvider`.
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
    """Median wall time of one `session.run`, warm, in milliseconds."""
    median_ms_per_pair: float
    """`median_ms_per_call / pair_count`."""
    build_seconds: float
    """Wall time of the first (engine-building) call."""


@dataclass(slots=True)
class BenchConfig:
    """What to measure and where the graph lives."""

    model_path: Path = REPO_ROOT / "data" / "cusfm_models" / "raco_release" / "raco_aliked_lightglue_pipeline_k2048.onnx"
    """The release pipeline graph."""
    cache_dir: Path = REPO_ROOT / "data" / "cusfm_models" / "raco_release" / "trtcache"
    """TensorRT engine and timing cache directory; gitignored with the models."""
    shapes: tuple[tuple[int, int], ...] = ((1024, 1024),)
    """`(height, width)` pairs to measure; both must be multiples of 32.

    One configuration per process: the TensorRT execution provider needed 69 GB
    of host memory to build a single engine for this graph, so a sweep inside
    one process is not survivable on a shared machine."""
    pair_counts: tuple[int, ...] = (1,)
    """Image pairs per execution."""
    repeats: int = 20
    """Timed executions per configuration, after warm-up."""
    warmups: int = 3
    """Untimed executions per configuration."""
    output_path: Path = Path("/tmp/raco_speed/release_ort.json")
    """Where the measurements are written."""


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
            "TensorrtExecutionProvider",
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


def measure(session: ort.InferenceSession, config: BenchConfig, height: int, width: int, pair_count: int) -> Measurement:
    """Time one configuration of the release pipeline graph.

    Args:
        session: A live ONNX Runtime session over the pipeline graph.
        config: Repeat and warm-up counts.
        height: Network input height in pixels.
        width: Network input width in pixels.
        pair_count: Image pairs per execution.

    Returns:
        The median latency of the configuration.
    """
    rng: np.random.Generator = np.random.default_rng(0)
    images: InterleavedPairs = rng.random((2 * pair_count, 3, height, width), dtype=np.float32)
    input_name: str = session.get_inputs()[0].name
    started: float = time.perf_counter()
    session.run(None, {input_name: images})
    build_seconds: float = time.perf_counter() - started
    for _ in range(config.warmups):
        session.run(None, {input_name: images})
    samples_ms: list[float] = []
    for _ in range(config.repeats):
        call_started: float = time.perf_counter()
        session.run(None, {input_name: images})
        samples_ms.append((time.perf_counter() - call_started) * 1e3)
    median_ms: float = float(np.median(samples_ms))
    return Measurement(
        label=f"release-pipeline {height}x{width} B={pair_count}",
        height=height,
        width=width,
        pair_count=pair_count,
        median_ms_per_call=median_ms,
        median_ms_per_pair=median_ms / pair_count,
        build_seconds=build_seconds,
    )


def main(config: BenchConfig) -> None:
    """Measure every requested configuration and print a table.

    Args:
        config: What to measure and where to write it.
    """
    if not config.model_path.is_file():
        raise FileNotFoundError(f"No release pipeline graph at {config.model_path}")
    available: set[str] = set(ort.get_available_providers())
    selected: list[tuple[str, dict[str, object]]] = [
        provider for provider in _providers(config.cache_dir) if provider[0] in available
    ]
    print(f"providers: {[name for name, _ in selected]}")
    session_options: ort.SessionOptions = ort.SessionOptions()
    session: ort.InferenceSession = ort.InferenceSession(str(config.model_path), session_options, selected)
    results: list[Measurement] = []
    for height, width in config.shapes:
        for pair_count in config.pair_counts:
            result: Measurement = measure(session, config, height, width, pair_count)
            results.append(result)
            print(
                f"{result.label:38s} call {result.median_ms_per_call:8.2f} ms  "
                f"per-pair {result.median_ms_per_pair:7.2f} ms  (first call {result.build_seconds:.1f} s)",
                flush=True,
            )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps([asdict(result) for result in results], indent=2))
    print(f"wrote {config.output_path}")


if __name__ == "__main__":
    main(tyro.cli(BenchConfig))

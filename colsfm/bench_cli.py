"""Benchmark two cuSFM-layout runs and render the comparison in Rerun.

```
pixi run -e colsfm python -m colsfm.bench_cli \
    --dataset galileo \
    --run-a data/cusfm_runs/galileo_blobref/cusfm \
    --run-b data/cusfm_runs/galileo_colsfm/cusfm \
    --input-dir data/r2b_galileo \
    --save data/bench/galileo_compare.rrd \
    --report data/bench/galileo_compare.md
```

Run A is the reference (the NVIDIA blob) and lands at `/world/rig_00`; run B is
the candidate (the `colsfm` pipeline) and lands at `/world/rig_01`. The
acceptance bounds of `docs/open-pipeline-plan.md` are checked against B, and
only for Galileo — RoboCap ships no ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro

from colsfm.benchmark import (
    AcceptanceBounds,
    Comparison,
    DatasetName,
    RunArtifacts,
    compare_runs,
    print_comparison,
    read_ground_truth,
    read_run,
    rig_track_from_frames_meta,
    write_json_report,
    write_markdown_report,
)
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.rerun_log import TrajectorySources, save_comparison

INPUT_FRAMES_META_NAME: str = "frames_meta.json"
"""The dataset's input metadata, beside `ground_truth.txt` when one is shipped."""


@dataclass(frozen=True)
class BenchConfig:
    """Compare two cuSFM-layout runs of the same dataset."""

    run_a: Path
    """Reference run's `.../cusfm` directory — the NVIDIA blob; becomes `rig_00`."""
    run_b: Path
    """Candidate run's `.../cusfm` directory — the colsfm pipeline; becomes `rig_01`."""
    input_dir: Path
    """Dataset input directory holding `frames_meta.json` and maybe `ground_truth.txt`."""
    dataset: DatasetName = "galileo"
    """Which dataset these runs reconstruct; acceptance bounds apply to Galileo only."""
    name_a: str = "blob"
    """Short label for run A in the report and in the recording's metadata."""
    name_b: str = "colsfm"
    """Short label for run B."""
    save: Path | None = None
    """Write the Rerun recording here; skipped when omitted."""
    report: Path | None = None
    """Write the markdown report here; skipped when omitted."""
    json_report: Path | None = None
    """Write the pyserde JSON dump here; defaults to `report` with a `.json` suffix."""
    bounds: AcceptanceBounds = field(default_factory=AcceptanceBounds)
    """Acceptance bounds; the plan's values unless overridden on the command line."""


def run(config: BenchConfig) -> Comparison:
    """Read both runs, compute every metric, and write the requested artifacts.

    Args:
        config: Parsed command line.

    Returns:
        The comparison, so callers can assert on it.
    """
    print(f"[bench] A {config.name_a}: {config.run_a}")
    print(f"[bench] B {config.name_b}: {config.run_b}")
    run_a: RunArtifacts = read_run(config.run_a, config.name_a)
    run_b: RunArtifacts = read_run(config.run_b, config.name_b)
    comparison: Comparison = compare_runs(run_a, run_b, config.input_dir, config.dataset, config.bounds)
    print_comparison(comparison)

    if config.report is not None:
        print(f"[bench] report -> {write_markdown_report(config.report, comparison)}")
    json_path: Path | None = config.json_report or (config.report.with_suffix(".json") if config.report else None)
    if json_path is not None:
        print(f"[bench] json   -> {write_json_report(json_path, comparison)}")

    if config.save is not None:
        input_meta: FramesMeta = read_frames_meta(config.input_dir / INPUT_FRAMES_META_NAME)
        sources: TrajectorySources = TrajectorySources(
            input_track=rig_track_from_frames_meta(input_meta),
            ground_truth=read_ground_truth(config.input_dir),
        )
        print(f"[bench] rerun  -> {save_comparison(config.save, comparison, run_a, run_b, sources)}")

    failed: list[str] = [check.name for check in comparison.acceptance if not check.passed]
    if failed:
        print(f"[bench] acceptance FAILED: {', '.join(failed)}")
    elif comparison.acceptance:
        print(f"[bench] acceptance passed: {len(comparison.acceptance)}/{len(comparison.acceptance)} bounds met")
    return comparison


def main() -> None:
    """Entry point for `python -m colsfm.bench_cli`."""
    run(tyro.cli(BenchConfig))


if __name__ == "__main__":
    main()

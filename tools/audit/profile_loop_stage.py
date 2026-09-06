"""Profile colsfm's loop-closure stage alone, on a copy of a finished run's database.

The database must already hold the run's keypoints, descriptors and stage-4 matches; the
stage adds its own loop-pair matches, so point this at a *copy*. Writes `loop_stage.prof`
into `output_dir` and prints the cumulative-time table. Used for `docs/loop-stage-cost.md`.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from dataclasses import dataclass
from pathlib import Path

import tyro

from colsfm.matching import MatchingBackend, MatchingOptions
from colsfm.pipeline import (
    KeyframeSelectionStageResult,
    LoopClosureStageResult,
    PipelineOptions,
    run_keyframe_selection_stage,
    run_loop_closure_stage,
)


@dataclass(frozen=True)
class ProfileArgs:
    """Where the run lives and which matcher to profile."""

    input_dir: Path
    """Directory holding `frames_meta.json` and the imagery of the finished run."""
    database: Path
    """A copy of the finished run's `database.db`; the stage writes loop-pair matches into it."""
    output_dir: Path
    """Where `loop_stage.prof` and the stage's scratch files go."""
    backend: MatchingBackend = "raco"
    """Matching backend for the candidate pairs; the features backend is set to match it."""
    top_n: int = 45
    """How many rows of the cumulative-time table to print."""


def main(args: ProfileArgs) -> None:
    """Run the loop-closure stage once under cProfile and print where the time went."""
    options: PipelineOptions = PipelineOptions(
        input_dir=args.input_dir, output_dir=args.output_dir, matching_backend=args.backend, features_backend=args.backend
    )
    selection: KeyframeSelectionStageResult = run_keyframe_selection_stage(options.selection)
    verification = selection.config.matching_task_worker.verification
    matching_options: MatchingOptions = MatchingOptions(
        backend=args.backend,
        max_error_px=verification.max_pixel_error,
        confidence=verification.min_ransac_confidence,
        num_threads=-1,
        device=options.device,
        max_matches_per_pair=options.max_matches_per_pair,
        match_cap_mode=options.match_cap_mode,
    )
    profiler: cProfile.Profile = cProfile.Profile()
    started: float = time.perf_counter()
    profiler.enable()
    result: LoopClosureStageResult = run_loop_closure_stage(
        selection.selected, args.database, selection.config.pose_graph, matching_options, enabled=True
    )
    profiler.disable()
    elapsed: float = time.perf_counter() - started
    print(f"[profile] loop stage {elapsed:.1f}s, {len(result.edges)} edges, {result.num_pairs_matched} pairs matched")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(args.output_dir / "loop_stage.prof"))
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(args.top_n)


if __name__ == "__main__":
    main(tyro.cli(ProfileArgs))

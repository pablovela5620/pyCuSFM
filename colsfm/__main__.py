"""`python -m colsfm` — the tyro CLI over `colsfm.pipeline`.

Two subcommands:

```
python -m colsfm run   --input-dir data/r2b_galileo --output-dir data/cusfm_runs/galileo_colsfm/cusfm
python -m colsfm stage --input-dir data/r2b_galileo --stage pair-selection
```

`run` is the drop-in for a `pycusfm.cusfm_runner.CusfmRunner.run_all()` call: the
same input directory, the same output layout, the same `runtime.csv`. `stage`
runs one of the two metadata-only stages, which need neither images nor a GPU,
for looking at how a dataset would be selected and paired before paying for
feature extraction.

**`run` with no other flag is the fast full pipeline** (NOTES.md decision 17,
2026-09-06): RaCo-ALIKED features, LightGlue+ matching, loop closure, CASPAR
bundle adjustment with its closing Ceres polish, and the regularised rig-extrinsic
refinement. Every one of those is a flag with a `--no-` or a `pycolmap` spelling,
and `colsfm.run_config.PipelineOptions` — which is this CLI's help text — says what
each ablation costs. Two of them need something the repository does not ship: the
RaCo graphs under `data/cusfm_models/` (`pixi run -e raco raco-export`) and a
CASPAR-enabled pycolmap (`pixi run -e colsfm-caspar-fisheye ...`). A missing graph
stops the run and names the fix; a missing CASPAR build falls back to Ceres and
records the reason in `summary.json`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TypeAlias

import tyro

from colsfm.pipeline import StageResult, run_pipeline, run_stage
from colsfm.run_config import PipelineOptions, SelectionOptions
from colsfm.run_lifecycle import SUMMARY_NAME, CheapStageName
from colsfm.run_report import PipelineSummary


@dataclass(frozen=True, slots=True)
class RunCommand:
    """Run all eight stages over one dataset, in the default fast full configuration."""

    options: tyro.conf.OmitArgPrefixes[PipelineOptions]
    """Input, output, config directory and the per-stage knobs."""

    def execute(self) -> None:
        """Run the pipeline and print where the summary landed."""
        summary: PipelineSummary = run_pipeline(self.options)
        print(f"[colsfm] summary written to {summary.options.output_dir / SUMMARY_NAME}")


@dataclass(frozen=True, slots=True)
class StageCommand:
    """Run one metadata-only stage, without touching images or a GPU."""

    options: tyro.conf.OmitArgPrefixes[SelectionOptions]
    """Just what the two metadata-only stages read: the input directory, the config
    profile and the three selection gates. Not the full run's option set, which
    would demand an output directory neither stage writes into."""
    stage: CheapStageName = "keyframe_selection"
    """Which stage to run on its own."""

    def execute(self) -> None:
        """Run the stage and print its counts."""
        result: StageResult = run_stage(self.options, self.stage)
        print(f"[colsfm] {result.stage}: {result.num_selected_keyframes} keyframes, {result.num_pairs} pairs")


Command: TypeAlias = (
    Annotated[RunCommand, tyro.conf.subcommand("run", description="Run the whole pipeline")]
    | Annotated[StageCommand, tyro.conf.subcommand("stage", description="Run one metadata-only stage")]
)
"""The subcommands `python -m colsfm` offers."""


def main() -> None:
    """Parse the command line and run the chosen subcommand."""
    command: Command = tyro.cli(Command)
    command.execute()


if __name__ == "__main__":
    main()

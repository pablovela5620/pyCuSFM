# SPDX-License-Identifier: Apache-2.0
"""Figure for the RoboCap loop-stage rows of the colsfm walkthrough report.

`docs/loop-stage-cost.md` §4 breaks colsfm's 104.0 s loop stage on the full
RoboCap sequence into its parts, measured by one `cProfile` run of the stage
alone (`tools/audit/profile_loop_stage.py`, dump in
`data/bench/robocap_loop_stage_raco.prof`). This renders that table as a bar
chart in the report's own chart style, coloured by *who owns the code*: the two
parts the doc names as the next targets are colsfm's own Python, and everything
expensive besides them is already a GPU or pycolmap backend.

The parts come from the profiled run, in which the stage took 112.5 s, so they
sum a little above the 104.0 s the unprofiled stage clock reports; the chart
labels that rather than rescaling the measurement.

Run:

    pixi run -e colsfm python -m tools.audit.robocap_figures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import tyro

CHART_WIDTH: Final[float] = 760.0
"""viewBox width, matching the report's existing per-stage charts."""

AXIS_MAX_SECONDS: Final[float] = 50.0
"""Right-hand end of the seconds axis; the largest part is the 47 s matcher."""

COLOUR_BACKEND: Final[str] = "#3987e5"
"""Categorical slot 1: work already done by a GPU or pycolmap backend."""
COLOUR_HOT_PYTHON: Final[str] = "#d95926"
"""Categorical slot 2: the two colsfm Python parts the doc names as next targets."""
COLOUR_OTHER: Final[str] = "#2e353c"
"""`--line`: the rest of the stage's own Python."""


@dataclass(frozen=True, slots=True)
class StagePart:
    """One row of `docs/loop-stage-cost.md` §4."""

    label: str
    """Part name, as the doc names it."""
    seconds: float
    """Profiled wall clock."""
    colour: str
    """Fill, chosen by who owns the code."""
    note: str = ""
    """Sub-breakdown printed under the bar, if the doc gives one."""
    approximate: bool = False
    """Whether the doc reports the number as approximate."""


LOOP_STAGE_PARTS: Final[tuple[StagePart, ...]] = (
    StagePart(
        "vocabulary-tree retrieval index",
        31.6,
        COLOUR_HOT_PYTHON,
        note="all-pairs L1 score 14.7 s &#183; word assignment 10.2 s &#183; vocabulary training 5.7 s",
    ),
    StagePart("LightGlue+ over 8274 new candidate pairs", 47.0, COLOUR_BACKEND, note="about 5.7 ms per pair, on the GPU", approximate=True),
    StagePart("rig-resection measurement", 12.0, COLOUR_BACKEND, note="876 verified rig pairs, pycolmap generalized pose", approximate=True),
    StagePart("square-covering NMS, pure Python", 11.5, COLOUR_HOT_PYTHON, note="a Python set per pair, 15,000 calls"),
    StagePart("geometric verification, pycolmap", 10.5, COLOUR_BACKEND),
    StagePart("candidate bookkeeping and reads", 1.0, COLOUR_OTHER, approximate=True),
)
"""Transcribed from `docs/loop-stage-cost.md` §4, in descending order within owner."""

STAGE_CLOCK_SECONDS: Final[float] = 104.0
"""What the unprofiled stage reports on the full sequence, the number in the table."""
PROFILED_SECONDS: Final[float] = 112.5
"""What the same stage took under cProfile, which is what the parts add up to."""


def render_loop_stage_svg() -> str:
    """Render the loop-stage breakdown as an inline SVG fragment.

    Returns:
        An `<svg class="chart">` element using the report's own chart classes.
    """
    left: float = 8.0
    right: float = CHART_WIDTH - 8.0
    top: float = 40.0
    bar_height: float = 17.0
    plain_row: float = 44.0
    noted_row: float = 60.0

    heights: list[float] = [noted_row if part.note else plain_row for part in LOOP_STAGE_PARTS]
    bottom: float = top + sum(heights)
    height: float = bottom + 30.0

    def seconds_to_x(seconds: float) -> float:
        return left + seconds * (right - left) / AXIS_MAX_SECONDS

    description: str = (
        "colsfm's loop-closure stage on the full RoboCap sequence, broken into parts. The LightGlue+ matcher "
        "takes about 47 s on the GPU; the vocabulary-tree retrieval index takes 31.6 s and the pure-Python "
        "square-covering NMS 11.5 s, and those two are colsfm's own Python. Geometric verification takes 10.5 s "
        "and rig resection about 12 s, both in pycolmap."
    )
    parts: list[str] = [f'<svg class="chart" viewBox="0 0 {CHART_WIDTH:.0f} {height:.0f}" role="img" aria-label="{description}">']

    for tick in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0):
        tick_x: float = seconds_to_x(tick)
        parts.append(f'<line x1="{tick_x:.1f}" y1="34.0" x2="{tick_x:.1f}" y2="{bottom:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        anchor: str = "start" if tick == 0.0 else ("end" if tick == AXIS_MAX_SECONDS else "middle")
        parts.append(f'<text x="{tick_x:.1f}" y="26.0" class="ax" text-anchor="{anchor}">{tick:.0f}</text>')

    cursor: float = top
    for part, row_height in zip(LOOP_STAGE_PARTS, heights, strict=True):
        label_y: float = cursor + 12.0
        bar_y: float = cursor + 20.0
        prefix: str = "~" if part.approximate else ""
        parts.append(f'<text x="{left:.1f}" y="{label_y:.1f}" class="lab">{part.label}</text>')
        parts.append(
            f'<text x="{right:.1f}" y="{label_y:.1f}" class="val" text-anchor="end">{prefix}{part.seconds:.1f} s</text>'
        )
        parts.append(
            f"<g><title>{part.label}: {prefix}{part.seconds:.1f} s</title>"
            f'<rect x="{left:.1f}" y="{bar_y:.1f}" width="{seconds_to_x(part.seconds) - left:.1f}" '
            f'height="{bar_height:.1f}" fill="{part.colour}"/></g>'
        )
        if part.note:
            parts.append(f'<text x="{left:.1f}" y="{bar_y + bar_height + 13:.1f}" class="ax">{part.note}</text>')
        cursor += row_height

    parts.append(
        f'<text x="{right:.1f}" y="{height - 8:.1f}" class="ax" text-anchor="end">'
        f"seconds, full RoboCap (4528 keyframes, 1132 rig frames) &#8212; the stage clocks "
        f"{STAGE_CLOCK_SECONDS:.1f} s, {PROFILED_SECONDS:.1f} s under cProfile</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


@dataclass(frozen=True)
class RobocapFiguresConfig:
    """Where the report figures are written."""

    output_dir: Path = Path("/tmp/fleet-artifacts/pycusfm/figures")
    """Where the SVG is written; the report inlines it."""


def main(config: RobocapFiguresConfig) -> None:
    """Render the RoboCap loop-stage chart.

    Args:
        config: The output directory.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = config.output_dir / "robocap-loop-stage-chart.svg"
    output_path.write_text(render_loop_stage_svg())
    total: float = sum(part.seconds for part in LOOP_STAGE_PARTS)
    print(f"{output_path}: {len(LOOP_STAGE_PARTS)} parts, {total:.1f} s profiled against a {STAGE_CLOCK_SECONDS:.1f} s stage clock")


if __name__ == "__main__":
    main(tyro.cli(RobocapFiguresConfig))

# SPDX-License-Identifier: Apache-2.0
"""Figures for the KITTI odometry 06 rows of the colsfm walkthrough report.

The report's two KITTI tables are numbers only. This script renders the same
numbers as pictures, from the same files the tables were measured from, so the
figures cannot drift from the prose:

* a top-down overlay of the 1231 m loop — ground truth against the blob, the
  most accurate colsfm run (TensorRT features and matching, CASPAR) and the
  fastest one (RaCo native, CASPAR);
* per-pose position error against distance travelled for the same three runs.

Every estimated trajectory is joined to the ground truth on its timestamp and
Sim(3)-aligned before plotting, through `tools.kitti.evaluate_kitti` — the code
path that produced the ATE table — rather than through a second alignment
written here. KITTI's ground truth is `world_T_cam0` in the camera frame of
frame 0, so the ground plane is x (right) against z (forward).

Colours and the plot surface are the report's own: the categorical dark steps
already bound to `--series-colsfm` and `--series-blob`, plus the palette's third
slot for the third run, drawn on the page's `--surface`.

Run:

    pixi run -e colsfm python -m tools.audit.kitti_figures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import matplotlib
import numpy as np
import tyro
from jaxtyping import Float64, Int64
from numpy import ndarray

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from colsfm.benchmark import RigidAlignment, align_rigid, match_timestamps
from tools.kitti.evaluate_kitti import (
    Trajectory,
    apply_alignment,
    read_kitti_ground_truth,
    read_tum_trajectory,
)

Positions: TypeAlias = Float64[ndarray, "n 3"]
Errors: TypeAlias = Float64[ndarray, "n"]

SURFACE: Final[str] = "#0f1318"
"""The report's `--surface`, the background every figure is drawn on."""
INK: Final[str] = "#f1f4f6"
"""The report's `--ink`, for titles."""
INK_2: Final[str] = "#bec3c9"
"""The report's `--ink-2`, for tick labels and legends."""
INK_3: Final[str] = "#8e949b"
"""The report's `--ink-3`, for axis titles."""
LINE_SOFT: Final[str] = "#21272c"
"""The report's `--line-soft`, for the grid."""

SERIES_GROUND_TRUTH: Final[str] = "#c3c2b7"
"""Neutral, so the reference reads as a backdrop and not as a fourth competitor."""
SERIES_BLOB: Final[str] = "#d95926"
"""`--series-blob`, categorical slot 2."""
SERIES_TRT: Final[str] = "#3987e5"
"""`--series-colsfm`, categorical slot 1."""
SERIES_RACO: Final[str] = "#199e70"
"""Categorical slot 3 (aqua), the only added hue."""

FIGURE_DPI: Final[int] = 150
"""1800 px wide at the 12 in figure width — twice the report's content column."""

TOLERANCE_MICROSECONDS: Final[int] = 2000
"""`evaluate_kitti`'s default join window; KITTI frames are 100 ms apart."""


@dataclass(frozen=True, slots=True)
class AlignedRun:
    """One estimated trajectory after the Sim(3) fit onto the ground truth."""

    label: str
    """Legend entry, matching the report's table row."""
    colour: str
    """Hex colour, from the report's categorical palette."""
    aligned_xyz: Positions
    """Float64 estimated positions in the ground-truth frame, shape `[m, 3]`."""
    reference_xyz: Positions
    """Float64 ground-truth positions for the same samples, shape `[m, 3]`."""
    draw_order: int
    """Painting order: the most accurate run is drawn last so it stays visible."""

    @property
    def errors_meters(self) -> Errors:
        """Per-pose position error after alignment, shape `[m]`."""
        return np.linalg.norm(self.aligned_xyz - self.reference_xyz, axis=1)

    @property
    def distance_meters(self) -> Errors:
        """Ground-truth distance travelled at each sample, shape `[m]`."""
        steps: Errors = np.linalg.norm(np.diff(self.reference_xyz, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(steps)])


def align_run(*, label: str, colour: str, draw_order: int, tum_path: Path, ground_truth: Trajectory) -> AlignedRun:
    """Join one TUM trajectory to the ground truth and Sim(3)-align it.

    Args:
        label: Legend entry for the run.
        colour: Hex colour for the run.
        draw_order: Painting order; higher is drawn on top.
        tum_path: The run's `merged_pose_file.tum`.
        ground_truth: The KITTI reference trajectory.

    Returns:
        The aligned run, paired sample for sample with the ground truth.

    Raises:
        ValueError: When no pose associates with the ground truth.
    """
    estimate: Trajectory = read_tum_trajectory(tum_path)
    estimate_indices: Int64[ndarray, "m"]
    reference_indices: Int64[ndarray, "m"]
    estimate_indices, reference_indices = match_timestamps(
        estimate.timestamps_microseconds, ground_truth.timestamps_microseconds, TOLERANCE_MICROSECONDS
    )
    if len(estimate_indices) == 0:
        raise ValueError(f"{tum_path}: no pose matched the ground truth within {TOLERANCE_MICROSECONDS} us")
    source_xyz: Positions = estimate.take(estimate_indices).world_t_body
    target_xyz: Positions = ground_truth.take(reference_indices).world_t_body
    alignment: RigidAlignment = align_rigid(source_xyz, target_xyz)
    return AlignedRun(
        label=label,
        colour=colour,
        aligned_xyz=apply_alignment(mode="sim3", alignment=alignment, source_xyz=source_xyz, target_xyz=target_xyz),
        reference_xyz=target_xyz,
        draw_order=draw_order,
    )


def style_axes(axes: plt.Axes) -> None:
    """Paint one axes in the report's dark surface colours.

    Args:
        axes: The axes to restyle in place.
    """
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=LINE_SOFT, linewidth=0.8)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color(LINE_SOFT)
    axes.tick_params(colors=INK_3, labelsize=10)
    for label in (*axes.get_xticklabels(), *axes.get_yticklabels()):
        label.set_color(INK_2)


def draw_route(axes: plt.Axes, runs: list[AlignedRun], *, linewidth: float, label: bool) -> None:
    """Draw the ground truth and every estimate top-down, z across and x up.

    KITTI 06 is a 457 m corridor only 22 m wide, so the route is plotted with the
    long axis (z) horizontal.

    Args:
        axes: The axes to draw into.
        runs: The aligned runs, drawn over the ground truth in order.
        linewidth: Stroke width for the estimated trajectories.
        label: Whether to attach legend labels to the artists.
    """
    reference_xyz: Positions = runs[0].reference_xyz
    axes.plot(
        reference_xyz[:, 2],
        reference_xyz[:, 0],
        color=SERIES_GROUND_TRUTH,
        linewidth=linewidth * 3.0,
        alpha=0.55,
        solid_capstyle="round",
        label="ground truth (KITTI 06, 1231 m)" if label else None,
        zorder=1,
    )
    # Later runs are drawn on top and drawn thinner, so a run hidden under a run it
    # agrees with still shows as a fringe rather than disappearing.
    for run in sorted(runs, key=lambda candidate: candidate.draw_order):
        axes.plot(
            run.aligned_xyz[:, 2],
            run.aligned_xyz[:, 0],
            color=run.colour,
            linewidth=linewidth * (1.0 - 0.22 * run.draw_order),
            label=run.label if label else None,
            zorder=2 + run.draw_order,
        )
    axes.plot(
        reference_xyz[0, 2],
        reference_xyz[0, 0],
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgecolor=INK,
        markeredgewidth=1.5,
        linestyle="none",
        label="frame 0" if label else None,
        zorder=10,
    )


def plot_overlay(runs: list[AlignedRun], output_path: Path) -> None:
    """Draw the top-down overlay: the whole corridor, then both turnarounds true to scale.

    Args:
        runs: The aligned runs, drawn in order over the ground truth.
        output_path: Where the PNG is written.
    """
    figure: Figure = plt.figure(figsize=(13.0, 7.1), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.45), hspace=0.55, wspace=0.10, top=0.86)

    whole: plt.Axes = figure.add_subplot(grid[0, :])
    style_axes(whole)
    draw_route(whole, runs, linewidth=1.7, label=True)
    whole.set_xlabel("z, metres — along the corridor", color=INK_3, fontsize=10.5)
    whole.set_ylabel("x, metres\n(axis exaggerated)", color=INK_3, fontsize=10.5)
    whole.set_title(
        "the whole 1231 m route, cross-corridor axis stretched so the two carriageways separate",
        color=INK_2,
        fontsize=11,
        pad=8,
    )

    turnarounds: tuple[tuple[str, tuple[float, float], tuple[float, float]], ...] = (
        ("south turnaround, true to scale", (-162.0, -122.0), (-24.0, 4.0)),
        ("north turnaround, true to scale", (278.0, 306.0), (-24.0, 4.0)),
    )
    for column, (title, z_limits, x_limits) in enumerate(turnarounds):
        zoom: plt.Axes = figure.add_subplot(grid[1, column])
        style_axes(zoom)
        draw_route(zoom, runs, linewidth=2.2, label=False)
        zoom.set_xlim(*z_limits)
        zoom.set_ylim(*x_limits)
        zoom.set_aspect("equal", adjustable="box")
        zoom.set_title(title, color=INK_2, fontsize=11, pad=6)
        zoom.set_xlabel("z, metres", color=INK_3, fontsize=10.5)
        if column == 0:
            zoom.set_ylabel("x, metres", color=INK_3, fontsize=10.5)

    handles, labels = whole.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=3, frameon=False, fontsize=10.5, labelcolor=INK_2
    )
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.22)
    plt.close(figure)


def plot_error_profile(runs: list[AlignedRun], output_path: Path) -> None:
    """Draw per-pose error against distance travelled, one line per run.

    Args:
        runs: The aligned runs.
        output_path: Where the PNG is written.
    """
    figure: Figure = plt.figure(figsize=(13.0, 4.7), facecolor=SURFACE)
    axes: plt.Axes = figure.add_subplot(111)
    style_axes(axes)

    for run in runs:
        errors: Errors = run.errors_meters
        axes.plot(
            run.distance_meters,
            errors,
            color=run.colour,
            linewidth=1.8,
            zorder=2 + run.draw_order,
            label=f"{run.label} — RMSE {np.sqrt((errors**2).mean()):.3f} m",
        )
    axes.set_xlabel("distance travelled along the ground-truth route, metres", color=INK_3, fontsize=11)
    axes.set_ylabel("position error after\nSim(3) alignment, metres", color=INK_3, fontsize=11)
    axes.set_xlim(0.0, float(runs[0].distance_meters[-1]))
    axes.set_ylim(bottom=0.0)
    legend = axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=3, frameon=False, fontsize=10.5, labelcolor=INK_2)
    legend.set_zorder(20)
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.22)
    plt.close(figure)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime bar charts, rendered as SVG in the report's own chart style
# ─────────────────────────────────────────────────────────────────────────────

STAGE_COLOURS: Final[tuple[str, ...]] = (SERIES_TRT, SERIES_BLOB, SERIES_RACO, "#c98500", "#2e353c")
"""Categorical slots 1-4 for the four measured stages, `--line` for the remainder."""

STAGE_NAMES: Final[tuple[str, ...]] = (
    "feature extraction",
    "feature matching",
    "loop stage",
    "mapping (triangulation + BA)",
    "everything else",
)
"""Key entries, in stacking order."""


@dataclass(frozen=True, slots=True)
class RuntimeRow:
    """One row of the report's "where the seconds go per backend" table."""

    label: str
    """Run name, exactly as the table names it."""
    extraction_seconds: float
    """Feature extraction wall clock."""
    matching_seconds: float
    """Feature matching wall clock."""
    loop_seconds: float
    """Loop stage wall clock: retrieval plus pose graph."""
    mapping_seconds: float
    """Triangulation and bundle adjustment wall clock."""
    total_seconds: float
    """End-to-end wall clock; the remainder over the four stages is drawn as "everything else"."""
    rmse_meters: float
    """Sim(3)-aligned ATE RMSE for the same run."""

    @property
    def segments(self) -> tuple[float, ...]:
        """The five stacked segment lengths, summing exactly to `total_seconds`."""
        measured: tuple[float, ...] = (
            self.extraction_seconds,
            self.matching_seconds,
            self.loop_seconds,
            self.mapping_seconds,
        )
        return (*measured, max(0.0, self.total_seconds - sum(measured)))


KITTI_RUNTIME_ROWS: Final[tuple[RuntimeRow, ...]] = (
    RuntimeRow("blob cuSFM, SLAM init", 96.9, 32.6, 1306.8, 143.1, 1596.1, 1.328),
    RuntimeRow("colsfm TensorRT + loops, Ceres", 62.2, 25.2, 62.3, 209.9, 363.5, 0.736),
    RuntimeRow("colsfm TensorRT + loops, CASPAR + polish", 55.9, 21.3, 47.2, 62.3, 189.7, 0.732),
    RuntimeRow("colsfm cap 500 + loops, CASPAR + polish", 16.4, 74.7, 77.6, 41.3, 212.9, 0.904),
    RuntimeRow("colsfm RaCo native + loops, Ceres", 10.1, 21.8, 50.8, 117.6, 203.2, 0.878),
    RuntimeRow("colsfm RaCo native + loops, CASPAR + polish", 9.7, 20.2, 47.0, 63.0, 142.8, 0.937),
    RuntimeRow("colsfm RaCo native + blob LightGlue + loops, CASPAR + polish", 9.6, 21.6, 47.3, 65.7, 147.0, 0.826),
)
"""Transcribed from the report's KITTI 06 runtime table, itself each run's `runtime.csv`."""

BLOB_BOW_SECONDS: Final[float] = 106.8
"""The retrieval half of the blob's 1306.8 s loop stage; the rest is pose graph."""

CHART_WIDTH: Final[float] = 760.0
"""viewBox width, matching the report's existing per-stage chart."""
BREAK_SECONDS: Final[float] = 400.0
"""Where the seconds axis stops being linear, so 142.8 s and 363.5 s stay readable."""
BREAK_FRACTION: Final[float] = 0.74
"""Share of the plotting width given to the linear part of the axis."""
AXIS_MAX_SECONDS: Final[float] = 1600.0
"""Right-hand end of the compressed part of the axis."""


def seconds_to_x(seconds: float, *, left: float, right: float) -> float:
    """Map seconds to a pixel position on the broken axis.

    Linear to `BREAK_SECONDS`, then compressed about 7.7x so the blob's 1596.1 s
    fits beside bars a tenth its length.

    Args:
        seconds: The value to place.
        left: Pixel x of zero seconds.
        right: Pixel x of `AXIS_MAX_SECONDS`.

    Returns:
        The pixel x for `seconds`.
    """
    linear_width: float = (right - left) * BREAK_FRACTION
    if seconds <= BREAK_SECONDS:
        return left + seconds * linear_width / BREAK_SECONDS
    compressed_width: float = (right - left) - linear_width
    return left + linear_width + (seconds - BREAK_SECONDS) * compressed_width / (AXIS_MAX_SECONDS - BREAK_SECONDS)


def render_runtime_svg() -> str:
    """Render the KITTI 06 per-backend runtime chart as an inline SVG fragment.

    Returns:
        An `<svg class="chart">` element using the report's own chart classes.
    """
    left: float = 8.0
    right: float = CHART_WIDTH - 8.0
    row_height: float = 46.0
    top: float = 40.0
    bar_height: float = 17.0
    bottom: float = top + row_height * len(KITTI_RUNTIME_ROWS)
    height: float = bottom + 48.0

    description: str = (
        "KITTI odometry 06 wall clock per backend, stacked by stage. The NVIDIA blob spends 1596.1 s, "
        "1306.8 s of it in a loop stage that finds no loops; every colsfm run finishes between 142.8 s "
        "and 363.5 s. The seconds axis is linear to 400 s and compressed beyond it."
    )
    parts: list[str] = [f'<svg class="chart" viewBox="0 0 {CHART_WIDTH:.0f} {height:.0f}" role="img" aria-label="{description}">']
    for tick in (0.0, 100.0, 200.0, 300.0, 400.0, 800.0, 1200.0, 1600.0):
        tick_x: float = seconds_to_x(tick, left=left, right=right)
        parts.append(f'<line x1="{tick_x:.1f}" y1="34.0" x2="{tick_x:.1f}" y2="{bottom:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        anchor: str = "start" if tick == 0.0 else ("end" if tick == AXIS_MAX_SECONDS else "middle")
        parts.append(f'<text x="{tick_x:.1f}" y="26.0" class="ax" text-anchor="{anchor}">{tick:.0f}</text>')

    for index, row in enumerate(KITTI_RUNTIME_ROWS):
        label_y: float = top + index * row_height + 12.0
        bar_y: float = top + index * row_height + 20.0
        parts.append(f'<text x="{left:.1f}" y="{label_y:.1f}" class="lab">{row.label}</text>')
        parts.append(
            f'<text x="{right:.1f}" y="{label_y:.1f}" class="val" text-anchor="end">'
            f"{row.total_seconds:.1f} s total &#183; {row.rmse_meters:.3f} m RMSE</text>"
        )
        cursor: float = 0.0
        for segment, colour, name in zip(row.segments, STAGE_COLOURS, STAGE_NAMES, strict=True):
            if segment <= 0.0:
                cursor += segment
                continue
            start_x: float = seconds_to_x(cursor, left=left, right=right)
            end_x: float = seconds_to_x(cursor + segment, left=left, right=right)
            parts.append(
                f"<g><title>{row.label} — {name}: {segment:.1f} s</title>"
                f'<rect x="{start_x:.1f}" y="{bar_y:.1f}" width="{max(0.8, end_x - start_x):.1f}" '
                f'height="{bar_height:.1f}" fill="{colour}"/></g>'
            )
            cursor += segment
        if row.label.startswith("blob"):
            split_x: float = seconds_to_x(row.extraction_seconds + row.matching_seconds + BLOB_BOW_SECONDS, left=left, right=right)
            parts.append(
                f'<line x1="{split_x:.1f}" y1="{bar_y:.1f}" x2="{split_x:.1f}" y2="{bar_y + bar_height:.1f}" '
                'stroke="var(--page)" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{split_x + 7:.1f}" y="{bar_y + 12.5:.1f}" class="val" style="fill:#06120d">'
                "1200.0 s of pose graph, no loops found</text>"
            )

    break_x: float = seconds_to_x(BREAK_SECONDS, left=left, right=right)
    for offset in (-3.5, 3.5):
        parts.append(
            f'<path d="M{break_x + offset - 5:.1f} {bottom:.1f} L{break_x + offset + 5:.1f} 34.0" '
            'stroke="var(--page)" stroke-width="4" fill="none"/>'
        )
        parts.append(
            f'<path d="M{break_x + offset - 5:.1f} {bottom:.1f} L{break_x + offset + 5:.1f} 34.0" '
            'stroke="var(--line)" stroke-width="1" fill="none"/>'
        )
    parts.append(f'<text x="{break_x:.1f}" y="{bottom + 16:.1f}" class="ax" text-anchor="middle">scale break</text>')

    parts.append(
        f'<text x="{right:.1f}" y="{height - 8:.1f}" class="ax" text-anchor="end">'
        "seconds, KITTI odometry 06 (1101 stereo frames) &#8212; linear to 400 s, compressed after the break</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class GalileoTimingRow:
    """One timing row of the report's "TensorRT against pycolmap backends" table."""

    label: str
    """Stage name, as the table names it."""
    blob_seconds: float
    """The NVIDIA blob's wall clock."""
    pycolmap_seconds: float
    """colsfm on the pycolmap backend."""
    tensorrt_seconds: float
    """colsfm on the TensorRT backend."""


GALILEO_TIMING_ROWS: Final[tuple[GalileoTimingRow, ...]] = (
    GalileoTimingRow("feature extraction", 8.64, 7.19, 7.01),
    GalileoTimingRow("feature matching", 2.25, 8.28, 2.91),
    GalileoTimingRow("triangulation + BA", 6.69, 2.56, 7.81),
)
"""Transcribed from the report's Galileo backend table (NOTES.md, `colsfm/README.md`)."""


def render_galileo_svg() -> str:
    """Render the Galileo backend timing chart as an inline SVG fragment.

    Returns:
        An `<svg class="chart">` element using the report's own chart classes.
    """
    left: float = 190.0
    right: float = CHART_WIDTH - 46.0
    axis_max: float = 9.0
    group_height: float = 62.0
    top: float = 40.0
    bar_height: float = 14.0
    bottom: float = top + group_height * len(GALILEO_TIMING_ROWS)
    height: float = bottom + 26.0
    series: tuple[tuple[str, str], ...] = (("blob", SERIES_BLOB), ("colsfm pycolmap", SERIES_TRT), ("colsfm tensorrt", SERIES_RACO))

    def value_to_x(value: float) -> float:
        return left + value * (right - left) / axis_max

    description: str = (
        "Galileo stage timings for three backends. Feature extraction is within 1.6 s across all three; "
        "feature matching is where the pycolmap backend costs 8.28 s against the blob&rsquo;s 2.25 s and "
        "TensorRT&rsquo;s 2.91 s; triangulation and bundle adjustment run 2.56 s on pycolmap and 7.81 s on "
        "TensorRT, which maps four times the points."
    )
    parts: list[str] = [f'<svg class="chart" viewBox="0 0 {CHART_WIDTH:.0f} {height:.0f}" role="img" aria-label="{description}">']
    for tick in (0.0, 2.0, 4.0, 6.0, 8.0):
        tick_x: float = value_to_x(tick)
        parts.append(f'<line x1="{tick_x:.1f}" y1="34.0" x2="{tick_x:.1f}" y2="{bottom:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{tick_x:.1f}" y="26.0" class="ax" text-anchor="middle">{tick:.0f}</text>')

    for index, row in enumerate(GALILEO_TIMING_ROWS):
        group_top: float = top + index * group_height
        parts.append(f'<text x="{left - 12:.1f}" y="{group_top + 26:.1f}" class="lab" text-anchor="end">{row.label}</text>')
        values: tuple[float, ...] = (row.blob_seconds, row.pycolmap_seconds, row.tensorrt_seconds)
        for bar_index, (value, (name, colour)) in enumerate(zip(values, series, strict=True)):
            bar_y: float = group_top + bar_index * (bar_height + 3.0)
            parts.append(
                f"<g><title>{name} — {row.label}: {value:.2f} s</title>"
                f'<rect x="{left:.1f}" y="{bar_y:.1f}" width="{value_to_x(value) - left:.1f}" '
                f'height="{bar_height:.1f}" fill="{colour}"/></g>'
                f'<text x="{value_to_x(value) + 7:.1f}" y="{bar_y + bar_height - 2:.1f}" class="val">{value:.2f}</text>'
            )

    parts.append(f'<text x="{right:.1f}" y="{height - 6:.1f}" class="ax" text-anchor="end">'
                 "seconds, r2b_galileo (226 keyframes, 331 pairs, RTX 5090)</text>")
    parts.append("</svg>")
    return "".join(parts)


@dataclass(frozen=True)
class KittiFiguresConfig:
    """Inputs and output directory for the KITTI 06 report figures."""

    sequence_dir: Path = Path("data/kitti/06")
    """KITTI sequence directory holding `times.txt` and `poses_gt_06.txt`."""
    blob_tum: Path = Path("data/kitti/06_result_slam/output_poses/merged_pose_file.tum")
    """The NVIDIA blob's cuSFM output, SLAM init."""
    trt_tum: Path = Path("/tmp/colsfm_runs/kitti_trt_caspar/cusfm/output_poses/merged_pose_file.tum")
    """The most accurate colsfm run: TensorRT features and matching, CASPAR + polish."""
    raco_tum: Path = Path("/tmp/colsfm_runs/kitti_raco_caspar/cusfm/output_poses/merged_pose_file.tum")
    """The fastest colsfm run: RaCo native features, CASPAR + polish."""
    output_dir: Path = Path("/tmp/fleet-artifacts/pycusfm/figures")
    """Where the PNGs are written; the report inlines them as base64."""


def main(config: KittiFiguresConfig) -> None:
    """Render both KITTI 06 figures and print what each one measured.

    Args:
        config: Trajectory paths and the output directory.
    """
    ground_truth: Trajectory = read_kitti_ground_truth(
        poses_path=config.sequence_dir / f"poses_gt_{config.sequence_dir.name}.txt",
        times_path=config.sequence_dir / "times.txt",
    )
    runs: list[AlignedRun] = [
        align_run(
            label="blob cuSFM, SLAM init",
            colour=SERIES_BLOB,
            draw_order=0,
            tum_path=config.blob_tum,
            ground_truth=ground_truth,
        ),
        align_run(
            label="colsfm TensorRT + loops, CASPAR + polish",
            colour=SERIES_TRT,
            draw_order=2,
            tum_path=config.trt_tum,
            ground_truth=ground_truth,
        ),
        align_run(
            label="colsfm RaCo native + loops, CASPAR + polish",
            colour=SERIES_RACO,
            draw_order=1,
            tum_path=config.raco_tum,
            ground_truth=ground_truth,
        ),
    ]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plot_overlay(runs, config.output_dir / "kitti06-trajectory-overlay.png")
    plot_error_profile(runs, config.output_dir / "kitti06-error-profile.png")
    (config.output_dir / "kitti06-runtime-chart.svg").write_text(render_runtime_svg())
    (config.output_dir / "galileo-backend-chart.svg").write_text(render_galileo_svg())
    for run in runs:
        errors: Errors = run.errors_meters
        print(
            f"{run.label}: {len(errors)} matched poses, "
            f"RMSE {np.sqrt((errors**2).mean()):.3f} m, max {errors.max():.3f} m, "
            f"route {run.distance_meters[-1]:.1f} m"
        )


if __name__ == "__main__":
    main(tyro.cli(KittiFiguresConfig))

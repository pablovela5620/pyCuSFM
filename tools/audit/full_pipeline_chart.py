"""Chart every full-pipeline run (loop closure + extrinsic refinement) per dataset.

Reads the runs' own `runtime.csv` files, groups stages into the plan's eight buckets and
draws one stacked horizontal bar per run, one panel per dataset, with the accuracy figure
printed at the bar's end. Written for the 2026-09-05 "full chart" request; the numbers it
draws are whatever the run directories hold, nothing is typed in.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCKETS: tuple[str, ...] = ("extraction", "loop + pose graph", "matching", "mapping", "extrinsics", "export")
"""Stage buckets, in drawing order."""
COLOURS: dict[str, str] = {
    "extraction": "#4f8fd6",
    "loop + pose graph": "#e8a33d",
    "matching": "#7bc47f",
    "mapping": "#d9635d",
    "extrinsics": "#b07cd8",
    "export": "#8c8c8c",
}
"""One colour per bucket."""
COLSFM_BUCKET: dict[str, str] = {
    "keyframe_selection": "extraction",
    "feature_extraction": "extraction",
    "pair_selection": "matching",
    "matching": "matching",
    "loop_closure": "loop + pose graph",
    "pose_graph": "loop + pose graph",
    "reconstruction": "mapping",
    "extrinsic_refinement": "extrinsics",
    "export": "export",
}
"""colsfm `runtime.csv` stage name to bucket."""
BLOB_BUCKET: dict[str, str] = {
    "feature_extractor_main": "extraction",
    "generate_bow_vocabulary_main": "loop + pose graph",
    "generate_bow_index_main": "loop + pose graph",
    "generate_association_main": "loop + pose graph",
    "pose_graph_main": "loop + pose graph",
    "feature_matcher_task_builder_main": "matching",
    "feature_matcher_main": "matching",
    "keypoints_mapper_main": "mapping",
    "kpmap_to_colmap": "export",
    "extract_pose_from_map_main": "export",
    "cuvslam_api_launcher": "extraction",
}
"""Blob binary name to bucket. The blob's second mapper pass (extrinsics) is folded into
`mapping` because its `runtime.csv` records one `keypoints_mapper_main` line per pass and
this script does not tell them apart; the caption says so."""


@dataclass(frozen=True)
class Row:
    """One bar."""

    dataset: str
    """Panel the bar belongs to."""
    label: str
    """Bar label."""
    runtime_csv: Path
    """The run's `runtime.csv`."""
    accuracy: str
    """Text printed after the bar, e.g. `ATE 2.69 mm`."""
    is_blob: bool = False
    """Whether `runtime_csv` is the blob's (binary names) or colsfm's (stage names)."""


def bucket_seconds(row: Row) -> dict[str, float]:
    """Sum a run's `runtime.csv` into the buckets.

    Args:
        row: The run.

    Returns:
        Seconds per bucket, zero for absent buckets.
    """
    totals: dict[str, float] = dict.fromkeys(BUCKETS, 0.0)
    with row.runtime_csv.open() as handle:
        for record in csv.DictReader(handle):
            command: str = record["command"].split(" ")[0]
            seconds: float = float(record["runtime_seconds"])
            bucket: str = (BLOB_BUCKET if row.is_blob else COLSFM_BUCKET).get(command, "export")
            totals[bucket] += seconds
    return totals


def draw(rows: list[Row], output: Path, title: str) -> None:
    """Render the panels.

    Args:
        rows: Every bar, grouped by `dataset` in order of first appearance.
        output: PNG path.
        title: Figure title.
    """
    datasets: list[str] = list(dict.fromkeys(row.dataset for row in rows))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 1.1 * len(rows) + 1.6 * len(datasets)), facecolor="#0b0e14")
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets, strict=True):
        panel: list[Row] = [row for row in rows if row.dataset == dataset]
        axis.set_facecolor("#0b0e14")
        for spine in axis.spines.values():
            spine.set_color("#2a2f3a")
        axis.tick_params(colors="#cfd3dc")
        for index, row in enumerate(panel):
            left: float = 0.0
            seconds: dict[str, float] = bucket_seconds(row)
            for bucket in BUCKETS:
                axis.barh(index, seconds[bucket], left=left, color=COLOURS[bucket], height=0.62, label=bucket if index == 0 else None)
                left += seconds[bucket]
            axis.text(left * 1.01, index, f"{left:.0f} s   {row.accuracy}", va="center", color="#cfd3dc", fontsize=9)
        axis.set_yticks(range(len(panel)))
        axis.set_yticklabels([row.label for row in panel], color="#e6e9ef", fontsize=9)
        axis.invert_yaxis()
        axis.set_title(dataset, loc="left", color="#e6e9ef", fontsize=11)
        axis.set_xlim(0, max(sum(bucket_seconds(row).values()) for row in panel) * 1.45)
        axis.grid(axis="x", color="#2a2f3a", linewidth=0.6)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8, facecolor="#141821", edgecolor="#2a2f3a", labelcolor="#cfd3dc")
    fig.suptitle(title, color="#e6e9ef", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    print(f"[chart] wrote {output}")


def main(spec: Path, output: Path, title: str = "Full pipeline: loop closure + extrinsic refinement") -> None:
    """Draw the chart from a JSON list of rows.

    Args:
        spec: JSON file holding a list of `Row` dicts.
        output: PNG path.
        title: Figure title.
    """
    import json

    rows: list[Row] = [Row(**{**item, "runtime_csv": Path(item["runtime_csv"])}) for item in json.loads(spec.read_text())]
    draw(rows, output, title)


if __name__ == "__main__":
    tyro.cli(main)

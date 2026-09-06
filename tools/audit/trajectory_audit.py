"""C12: does cuSFM's refinement improve Galileo's *input* trajectory?

`data/bench/galileo_compare.md` reports the blob at 5.00 mm and `colsfm` at
4.33 mm ATE against `ground_truth.txt`. Neither report scores the **input**
trajectory the same way, and NOTES.md quotes ~4.29 mm for it in passing. If that
holds under the identical join and the identical alignment, the blob made this
sequence slightly worse, not better — which is the opposite of the paper's
§6.3.2 claim. This script settles it by scoring all three trajectories through
one code path:

* the same rig-pose rule (`FramesMeta.rig_frames`, lowest keyframe id),
* the same timestamp join (nearest within 20 microseconds),
* the same rigid alignment with the scale held at 1.0,
* the same sample set — restricted, per comparison, to the rig frames the run
  registered, and additionally to the intersection of all three so that no run
  is scored on a different set of samples than another.

Run:

    pixi run -e colsfm python -m tools.audit.trajectory_audit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import tyro
from jaxtyping import Int64
from numpy import ndarray
from serde import serde

from colsfm.benchmark import RigTrack, read_run, rig_track_from_frames_meta, write_json_report
from colsfm.trajectory import TrajectoryScore, indices_of_timestamps, score_track_against_ground_truth
from tools.audit.galileo_metrics import GalileoReference, read_galileo_reference


@dataclass(frozen=True)
class TrajectoryAuditConfig:
    """Inputs and output path for the C12 measurement."""

    input_dir: Path = Path("data/r2b_galileo")
    """Dataset directory holding `frames_meta.json` and `ground_truth.txt`."""
    blob_run: Path = Path("data/cusfm_runs/galileo_blobref/cusfm")
    """The NVIDIA blob's run directory (run A of `data/bench/galileo_compare.md`)."""
    colsfm_run: Path = Path("data/cusfm_runs/galileo_colsfm/cusfm")
    """The `colsfm` run directory (run B of the same report)."""
    output_json: Path = Path("data/bench/audit_galileo_trajectory.json")
    """Where the machine-readable result is written."""


SampleSet: TypeAlias = Literal["own", "common"]
"""Which rig frames a trajectory is scored on: its own registered ones, or the
intersection of all three trajectories' so that no run is scored on a different set."""


@serde
@dataclass(frozen=True)
class ScoredTrajectory:
    """One trajectory's ATE against ground truth, on one sample set."""

    name: str
    """`input`, `blob` or `colsfm`."""
    sample_set: SampleSet
    """Which rig frames it was scored on."""
    score: TrajectoryScore
    """Every number the scorer produced, embedded whole rather than transcribed —
    the previous shape restated five of its seven fields and dropped two."""


@serde
@dataclass(frozen=True)
class TrajectoryAuditResult:
    """Everything C12 measured."""

    input_dir: str
    """Dataset the numbers came from."""
    scored: tuple[ScoredTrajectory, ...]
    """Every trajectory, on every sample set."""
    verdict: str
    """One sentence stating whether refinement improved the input trajectory."""


def _score_on_timestamps(track: RigTrack, reference: GalileoReference, keep_microseconds: set[int] | None) -> TrajectoryScore:
    """Score a rig track against ground truth, optionally on a fixed sample set.

    Args:
        track: The trajectory to score.
        reference: The Galileo reference bundle.
        keep_microseconds: When given, only samples with these timestamps are used.

    Returns:
        The score.
    """
    selected: Int64[ndarray, "m"] = indices_of_timestamps(track, keep_microseconds)
    return score_track_against_ground_truth(track.take(selected), reference.ground_truth)


def main(config: TrajectoryAuditConfig) -> None:
    """Measure input, blob and `colsfm` trajectories against Galileo ground truth.

    Args:
        config: Paths for the three trajectories and the output report.
    """
    reference: GalileoReference = read_galileo_reference(config.input_dir)
    # `read_run` is the benchmark's own reader, so these tracks are built from the same
    # sparse model, the same optimised metadata and the same registered-image rule the
    # report's numbers come from.
    tracks: dict[str, RigTrack] = {
        "input": rig_track_from_frames_meta(reference.frames_meta),
        "blob": read_run(config.blob_run, "blob").track,
        "colsfm": read_run(config.colsfm_run, "colsfm").track,
    }

    common: set[int] = set.intersection(*({int(stamp) for stamp in track.timestamps_microseconds} for track in tracks.values()))
    sample_sets: tuple[tuple[SampleSet, set[int] | None], ...] = (("own", None), ("common", common))
    scored: list[ScoredTrajectory] = [
        ScoredTrajectory(name=name, sample_set=sample_set, score=_score_on_timestamps(track, reference, keep))
        for sample_set, keep in sample_sets
        for name, track in tracks.items()
    ]

    by_key: dict[tuple[str, SampleSet], ScoredTrajectory] = {(item.name, item.sample_set): item for item in scored}
    input_rmse: float = by_key[("input", "common")].score.rigid_rmse_millimeters
    blob_rmse: float = by_key[("blob", "common")].score.rigid_rmse_millimeters
    colsfm_rmse: float = by_key[("colsfm", "common")].score.rigid_rmse_millimeters
    verdict: str = (
        f"on the {len(common)} common rig frames: input {input_rmse:.2f} mm, blob {blob_rmse:.2f} mm "
        f"({blob_rmse - input_rmse:+.2f} mm), colsfm {colsfm_rmse:.2f} mm ({colsfm_rmse - input_rmse:+.2f} mm)"
    )

    print(f"{'trajectory':<10} {'set':<8} {'matched':>8} {'RMSE mm':>9} {'max mm':>9} {'scale':>9} {'SIM3 mm':>9}")
    for item in scored:
        print(
            f"{item.name:<10} {item.sample_set:<8} {item.score.num_matched:>8d} "
            f"{item.score.rigid_rmse_millimeters:>9.3f} {item.score.rigid_max_millimeters:>9.3f} "
            f"{item.score.would_be_scale:>9.5f} {item.score.similarity_rmse_millimeters:>9.3f}"
        )
    print(f"\n{verdict}")

    result: TrajectoryAuditResult = TrajectoryAuditResult(input_dir=str(config.input_dir), scored=tuple(scored), verdict=verdict)
    print(f"wrote {write_json_report(config.output_json, result)}")


if __name__ == "__main__":
    main(tyro.cli(TrajectoryAuditConfig))

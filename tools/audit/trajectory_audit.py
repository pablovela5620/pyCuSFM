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

import numpy as np
import pycolmap
import tyro
from jaxtyping import Int64
from numpy import ndarray
from serde import serde
from serde.json import to_json

from colsfm.benchmark import GROUND_TRUTH_TOLERANCE_MICROSECONDS, RigTrack, match_timestamps, rig_track_from_frames_meta
from colsfm.frames_meta import FramesMeta, read_frames_meta
from tools.audit.galileo_metrics import GalileoReference, TrajectoryScore, read_galileo_reference, score_positions


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


@serde
@dataclass(frozen=True)
class ScoredTrajectory:
    """One trajectory's ATE against ground truth, on one sample set."""

    name: str
    """`input`, `blob` or `colsfm`."""
    sample_set: str
    """`own` (the run's own registered samples) or `common` (the intersection)."""
    num_matched: int
    """Rig frames that joined ground truth."""
    rigid_rmse_millimeters: float
    """RMSE after a rigid fit with the scale held at 1.0."""
    rigid_max_millimeters: float
    """Largest residual of that fit."""
    would_be_scale: float
    """Scale a similarity fit would have chosen."""
    similarity_rmse_millimeters: float
    """RMSE after a scale-free SIM(3) fit, for reference only."""


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


def _track_of_run(run_dir: Path) -> RigTrack:
    """Rig trajectory of a cuSFM-layout run, restricted to its registered images.

    Args:
        run_dir: A `.../cusfm` directory.

    Returns:
        The optimised rig trajectory.
    """
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction(str(run_dir / "sparse"))
    registered_names: set[str] = {image.name for image in reconstruction.images.values() if image.num_points3D > 0}
    frames_meta: FramesMeta = read_frames_meta(run_dir / "kpmap" / "keyframes" / "frames_meta.json")
    return rig_track_from_frames_meta(frames_meta, keep_image_names=registered_names)


def _score_on_timestamps(track: RigTrack, reference: GalileoReference, keep_microseconds: set[int] | None) -> TrajectoryScore:
    """Score a rig track against ground truth, optionally on a fixed sample set.

    Args:
        track: The trajectory to score.
        reference: The Galileo reference bundle.
        keep_microseconds: When given, only samples with these timestamps are used.

    Returns:
        The score.
    """
    selected: Int64[ndarray, "m"] = (
        np.arange(len(track), dtype=np.int64)
        if keep_microseconds is None
        else np.asarray([index for index, stamp in enumerate(track.timestamps_microseconds) if int(stamp) in keep_microseconds], dtype=np.int64)
    )
    subset: RigTrack = track.take(selected)
    track_indices, reference_indices = match_timestamps(
        subset.timestamps_microseconds, reference.ground_truth.timestamps_microseconds, GROUND_TRUTH_TOLERANCE_MICROSECONDS
    )
    return score_positions(subset.world_t_rig[track_indices], reference.ground_truth.world_t_rig[reference_indices])


def main(config: TrajectoryAuditConfig) -> None:
    """Measure input, blob and `colsfm` trajectories against Galileo ground truth.

    Args:
        config: Paths for the three trajectories and the output report.
    """
    reference: GalileoReference = read_galileo_reference(config.input_dir)
    tracks: dict[str, RigTrack] = {
        "input": rig_track_from_frames_meta(reference.frames_meta),
        "blob": _track_of_run(config.blob_run),
        "colsfm": _track_of_run(config.colsfm_run),
    }

    common: set[int] = set.intersection(*({int(stamp) for stamp in track.timestamps_microseconds} for track in tracks.values()))
    scored: list[ScoredTrajectory] = []
    for sample_set, keep in (("own", None), ("common", common)):
        for name, track in tracks.items():
            score: TrajectoryScore = _score_on_timestamps(track, reference, keep)
            scored.append(
                ScoredTrajectory(
                    name=name,
                    sample_set=sample_set,
                    num_matched=score.num_matched,
                    rigid_rmse_millimeters=score.rigid_rmse_millimeters,
                    rigid_max_millimeters=score.rigid_max_millimeters,
                    would_be_scale=score.would_be_scale,
                    similarity_rmse_millimeters=score.similarity_rmse_millimeters,
                )
            )

    by_key: dict[tuple[str, str], ScoredTrajectory] = {(item.name, item.sample_set): item for item in scored}
    input_rmse: float = by_key[("input", "common")].rigid_rmse_millimeters
    blob_rmse: float = by_key[("blob", "common")].rigid_rmse_millimeters
    colsfm_rmse: float = by_key[("colsfm", "common")].rigid_rmse_millimeters
    verdict: str = (
        f"on the {len(common)} common rig frames: input {input_rmse:.2f} mm, blob {blob_rmse:.2f} mm "
        f"({blob_rmse - input_rmse:+.2f} mm), colsfm {colsfm_rmse:.2f} mm ({colsfm_rmse - input_rmse:+.2f} mm)"
    )

    print(f"{'trajectory':<10} {'set':<8} {'matched':>8} {'RMSE mm':>9} {'max mm':>9} {'scale':>9} {'SIM3 mm':>9}")
    for item in scored:
        print(
            f"{item.name:<10} {item.sample_set:<8} {item.num_matched:>8d} {item.rigid_rmse_millimeters:>9.3f} "
            f"{item.rigid_max_millimeters:>9.3f} {item.would_be_scale:>9.5f} {item.similarity_rmse_millimeters:>9.3f}"
        )
    print(f"\n{verdict}")

    result: TrajectoryAuditResult = TrajectoryAuditResult(input_dir=str(config.input_dir), scored=tuple(scored), verdict=verdict)
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(to_json(result))
    print(f"wrote {config.output_json}")


if __name__ == "__main__":
    main(tyro.cli(TrajectoryAuditConfig))

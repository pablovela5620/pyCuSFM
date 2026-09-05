"""C8: how far did cuSFM's extrinsic refinement actually move the calibration?

Paper §4.4 and §6.4.1 claim that refining `sensor_from_rig` jointly with the rig
poses improves accuracy. `data/cusfm_runs/galileo/cusfm` was produced with
`--optimize_extrinsics=True`, so its `kpmap/keyframes/frames_meta.json` carries
refined `sensor_to_vehicle_transform` values while `data/r2b_galileo/frames_meta.json`
carries the input calibration. This script reports the movement two ways:

* **absolute** — each camera's `vehicle_T_cam` against its input value. A rigid
  gauge freedom lives here: bundle adjustment may rotate the whole rig frame,
  which moves every camera without changing the rig's shape.
* **relative** — each ordered camera pair's `cam_i_T_cam_j` against its input
  value. That is gauge-free, so it is the number that says whether the *shape*
  of the calibration changed.

Run:

    pixi run -e colsfm python -m tools.audit.extrinsics_audit
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pycolmap
import tyro
from serde import serde

from colsfm.bench_report import number as format_number
from colsfm.benchmark import write_json_report
from colsfm.extrinsic_refinement import extrinsic_deltas
from colsfm.frames_meta import CameraParams, FramesMeta, read_frames_meta


@dataclass(frozen=True)
class ExtrinsicsAuditConfig:
    """Which two calibrations to compare."""

    input_meta: Path = Path("data/r2b_galileo/frames_meta.json")
    """The input calibration handed to cuSFM."""
    refined_meta: Path = Path("data/cusfm_runs/galileo/cusfm/kpmap/keyframes/frames_meta.json")
    """A run written with `--optimize_extrinsics=True`."""
    output_json: Path = Path("data/bench/audit_galileo_extrinsics.json")
    """Where the machine-readable result is written."""


@serde
@dataclass(frozen=True)
class ExtrinsicDelta:
    """How far one transform moved."""

    label: str
    """Camera name, or `"cam_i -> cam_j"` for a relative extrinsic."""
    translation_millimeters: float
    """Norm of the translation difference."""
    rotation_degrees: float
    """Geodesic angle between the two rotations."""


@serde
@dataclass(frozen=True)
class ExtrinsicsAuditResult:
    """Everything C8 measured."""

    input_meta: str
    """Path of the input calibration."""
    refined_meta: str
    """Path of the refined calibration."""
    absolute: tuple[ExtrinsicDelta, ...]
    """Per-camera `vehicle_T_cam` movement; carries the rig-frame gauge."""
    relative: tuple[ExtrinsicDelta, ...]
    """Per-pair `cam_i_T_cam_j` movement; gauge-free."""
    max_absolute_translation_millimeters: float
    """Largest absolute translation movement."""
    max_absolute_rotation_degrees: float
    """Largest absolute rotation movement."""
    max_relative_translation_millimeters: float
    """Largest relative translation movement — the headline number."""
    max_relative_rotation_degrees: float
    """Largest relative rotation movement — the headline number."""


def _delta(label: str, before: pycolmap.Rigid3d, after: pycolmap.Rigid3d) -> ExtrinsicDelta:
    """Measure the movement between two poses.

    Goes through `colsfm.extrinsic_refinement.extrinsic_deltas` on a one-camera
    mapping, so this audit, the refinement's own per-round stopping test and
    `colsfm.pipeline.extrinsic_changes` all report the same two quantities from
    the same code rather than from three angle formulas.

    Args:
        label: Name for the report.
        before: The input transform.
        after: The refined transform.

    Returns:
        Translation in millimetres and rotation in degrees.
    """
    translation_millimeters, rotation_degrees = extrinsic_deltas({0: before}, {0: after})
    return ExtrinsicDelta(label=label, translation_millimeters=translation_millimeters, rotation_degrees=rotation_degrees)


def _print_table(title: str, deltas: Sequence[ExtrinsicDelta]) -> None:
    """Print one movement table to stdout.

    Args:
        title: Column heading for the label column.
        deltas: The rows, in the order they should be printed.
    """
    print(f"\n{title:<46} {'mm':>8} {'deg':>8}")
    for item in deltas:
        print(f"{item.label:<46} {format_number(item.translation_millimeters, 3):>8} {format_number(item.rotation_degrees, 3):>8}")


def main(config: ExtrinsicsAuditConfig) -> None:
    """Compare an `--optimize_extrinsics` run's calibration against its input.

    Args:
        config: Paths for the two calibrations and the output report.
    """
    before_meta: FramesMeta = read_frames_meta(config.input_meta)
    after_meta: FramesMeta = read_frames_meta(config.refined_meta)
    shared: list[int] = sorted(set(before_meta.cameras) & set(after_meta.cameras))

    absolute: list[ExtrinsicDelta] = []
    for camera_params_id in shared:
        before: CameraParams = before_meta.cameras[camera_params_id]
        after: CameraParams = after_meta.cameras[camera_params_id]
        absolute.append(_delta(before.sensor_name, before.vehicle_T_cam, after.vehicle_T_cam))

    relative: list[ExtrinsicDelta] = []
    for first, second in combinations(shared, 2):
        before_pair: pycolmap.Rigid3d = before_meta.cameras[first].vehicle_T_cam.inverse() * before_meta.cameras[second].vehicle_T_cam
        after_pair: pycolmap.Rigid3d = after_meta.cameras[first].vehicle_T_cam.inverse() * after_meta.cameras[second].vehicle_T_cam
        label: str = f"{before_meta.cameras[first].sensor_name} -> {before_meta.cameras[second].sensor_name}"
        relative.append(_delta(label, before_pair, after_pair))

    # A single shared camera yields no relative pair, and `max` over an empty
    # sequence raises; an unmeasured maximum is 0.0, not a crash.
    result: ExtrinsicsAuditResult = ExtrinsicsAuditResult(
        input_meta=str(config.input_meta),
        refined_meta=str(config.refined_meta),
        absolute=tuple(absolute),
        relative=tuple(relative),
        max_absolute_translation_millimeters=max((item.translation_millimeters for item in absolute), default=0.0),
        max_absolute_rotation_degrees=max((item.rotation_degrees for item in absolute), default=0.0),
        max_relative_translation_millimeters=max((item.translation_millimeters for item in relative), default=0.0),
        max_relative_rotation_degrees=max((item.rotation_degrees for item in relative), default=0.0),
    )

    _print_table("absolute vehicle_T_cam", absolute)
    _print_table(
        "relative cam_i_T_cam_j (gauge-free)",
        sorted(relative, key=lambda entry: entry.translation_millimeters, reverse=True),
    )
    print(
        f"\nmax absolute {format_number(result.max_absolute_translation_millimeters)} mm / "
        f"{format_number(result.max_absolute_rotation_degrees)} deg; "
        f"max relative {format_number(result.max_relative_translation_millimeters)} mm / "
        f"{format_number(result.max_relative_rotation_degrees)} deg"
    )
    print(f"wrote {write_json_report(config.output_json, result)}")


if __name__ == "__main__":
    main(tyro.cli(ExtrinsicsAuditConfig))

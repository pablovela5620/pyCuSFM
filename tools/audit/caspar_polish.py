# SPDX-License-Identifier: Apache-2.0
"""One Ceres global bundle adjustment on a finished model, with nothing else moved.

`docs/caspar-build.md` § "Verdict: not a stopping-criterion problem" leaves two
explanations of KITTI 06's 0.4 m gap between the Ceres mapper (0.895 m Sim(3)
ATE) and the CASPAR one (1.299 m): either both solvers sit in the same basin and
CASPAR merely stops at a different point of it (a gauge/conditioning story), or
Ceres agrees that CASPAR's answer is a minimum and the trajectories parted
earlier, over which tracks survived round 0's 25 px gate (a mapper-schedule
story).

This tool decides between them. It takes a *finished* model off disk and runs a
single global bundle adjustment with `colsfm.mapping.bundle_adjustment_options`
— the same Cauchy loss, the same tolerances, the same 200 iterations, the same
one constant rig frame `run_mapping` would pick, extrinsics fixed, intrinsics
fixed — and then stops. No re-triangulation, no merge, no complete, no filter:
the observation set that goes in is the observation set that comes out, so any
change in the score is the solve's alone.

The polished model is re-exported with `colsfm.export` (`sparse/`,
`kpmap/keyframes/frames_meta.json`, `output_poses/*.tum`) using the source run's
own `frames_meta.json` as the template, which is what makes the result scoreable
with `tools/kitti/evaluate_kitti.py` exactly as the mapper's own output is.

Usage:
    pixi run -e colsfm python -m tools.audit.caspar_polish \
        --sparse-dir data/kitti/06_colsfm_caspar/cusfm/sparse \
        --frames-meta data/kitti/06_colsfm_caspar/cusfm/kpmap/keyframes/frames_meta.json \
        --output-dir data/kitti/06_colsfm_caspar_polish/cusfm
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pycolmap
import tyro
from jaxtyping import Float64
from numpy import ndarray
from serde import serde
from serde.json import to_json

from colsfm.alignment import RigidAlignment, RigTrack, align_rigid, rig_track_from_frames_meta
from colsfm.config import BundleAdjustmentConfig, CusfmConfig, LossFunctionType, read_config_directory
from colsfm.export import write_colmap_model, write_optimised_frames_meta, write_pose_files
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.mapping import (
    MappingOptions,
    bundle_adjustment_options,
    registered_image_ids,
    solve_bundle_adjustment,
)
from colsfm.solver_report import SolverReport, parse_brief_report
from colsfm.stages import optimised_camera_poses

Positions: TypeAlias = Float64[ndarray, "n 3"]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
"""Repository root, so the defaults can name paths the way the docs do."""


@dataclass(frozen=True)
class PolishConfig:
    """Inputs of one polish run."""

    sparse_dir: Path = REPO_ROOT / "data/kitti/06_colsfm_caspar/cusfm/sparse"
    """Text-format COLMAP model to adjust; the model a finished run left behind."""
    frames_meta: Path | None = None
    """Template `frames_meta.json`; `<sparse_dir>/../kpmap/keyframes/frames_meta.json` when None."""
    config_dir: Path = REPO_ROOT / "data/kitti/config"
    """cuSFM config profile the solve's settings come from."""
    output_dir: Path = REPO_ROOT / "data/kitti/06_colsfm_caspar_polish/cusfm"
    """Workspace root the polished model, metadata and TUM files are written to."""
    label: str = "caspar-polish"
    """Name for this run in the printed report and in `polish.json`."""
    num_threads: int | None = None
    """Ceres threads; the config's own `num_threads` when None."""
    loss_type: LossFunctionType | None = None
    """Robust loss to solve with; the config's own (`CAUCHY` on isaac) when None.

    `TRIVIAL` is the ablation that separates "CASPAR stopped short of the
    minimum" from "the Cauchy reweighting is what moved the model", since CASPAR
    applies no robust loss at all."""


@serde
@dataclass(frozen=True, slots=True)
class SolveReport:
    """What the single bundle adjustment cost and how far it moved the model."""

    label: str
    """The run this polish was applied to."""
    loss_type: LossFunctionType
    """Robust loss the solve used."""
    sparse_dir: str
    """Model that went in."""
    output_dir: str
    """Workspace the polished model was written to."""
    num_registered_images: int
    """Images with a pose, i.e. every image the solve parameterised."""
    num_points3D: int
    """Points in the model; unchanged by definition, since nothing filters."""
    num_observations: int
    """Observations in the model; likewise unchanged."""
    gauge_frame_id: int
    """The one frame whose `rig_from_world` was held constant."""
    ba_num_iterations: int | None
    """Ceres iterations, from `brief_report()`; None when the solver published none."""
    ba_initial_cost: float | None
    """Cost of the model as it came off disk; None when the solver published none."""
    ba_final_cost: float | None
    """Cost after the solve; None when the solver published none."""
    ba_termination: str
    """`BundleAdjustmentSummary.termination_type`."""
    solve_seconds: float
    """Wall-clock seconds inside `BundleAdjuster.solve()`."""
    mean_reprojection_error_before_px: float
    """Mean reprojection error before the solve, over the same observations."""
    mean_reprojection_error_after_px: float
    """Mean reprojection error after the solve."""
    rig_displacement_rmse_m: float
    """Root-mean-square rig-position change the polish produced, in the model's own frame."""
    rig_displacement_max_m: float
    """Largest single rig-position change, metres."""
    rig_residual_after_sim3_rmse_m: float
    """Rig displacement left after a similarity fit of the old track onto the new one.

    Separates a global gauge motion (which a Sim(3) fit absorbs) from a genuine
    change of trajectory shape (which it cannot)."""
    rig_residual_after_sim3_max_m: float
    """Largest residual of that similarity fit, metres."""
    rig_sim3_scale: float
    """Scale that fit chose; 1.0 means the polish did not resize the trajectory."""


def load_reconstruction(sparse_dir: Path) -> pycolmap.Reconstruction:
    """Read a text-format model and refresh its per-point reprojection errors.

    `Reconstruction.compute_mean_reprojection_error` averages the stored
    `Point3D.error`, which a text model carries from whenever it was written;
    recomputing makes the "before" number the state of *this* model rather than
    the state its writer happened to record.

    Args:
        sparse_dir: Directory holding `cameras.txt`, `images.txt`, `points3D.txt`,
            `rigs.txt` and `frames.txt`.

    Returns:
        The loaded reconstruction, with fresh point errors.

    Raises:
        FileNotFoundError: When the directory holds no model.
    """
    if not (sparse_dir / "images.txt").is_file():
        raise FileNotFoundError(f"No COLMAP text model in {sparse_dir}")
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    reconstruction.update_point_3d_errors()
    return reconstruction


def rig_track_of(reconstruction: pycolmap.Reconstruction, template: FramesMeta) -> RigTrack:
    """The rig trajectory a reconstruction implies, through the metadata it exports into.

    Args:
        reconstruction: A posed model.
        template: The collection the run exported, used only for its rig geometry
            and timestamps; its poses are replaced by the reconstruction's.

    Returns:
        One `world_T_rig` per synchronised sample, in time order.
    """
    posed: FramesMeta = template.with_camera_to_world(optimised_camera_poses(reconstruction))
    return rig_track_from_frames_meta(posed)


def polish(config: PolishConfig) -> SolveReport:
    """Run the single solve, export the result and measure what moved.

    Args:
        config: Paths and solver knobs.

    Returns:
        The report, also written to `<output_dir>/polish.json`.

    Raises:
        FileNotFoundError: When the model or the template metadata is missing.
        ValueError: When the model has no registered image.
    """
    frames_meta_path: Path = (
        config.sparse_dir.parent / "kpmap" / "keyframes" / "frames_meta.json" if config.frames_meta is None else config.frames_meta
    )
    reconstruction: pycolmap.Reconstruction = load_reconstruction(config.sparse_dir)
    template: FramesMeta = read_frames_meta(frames_meta_path)

    image_ids: list[int] = registered_image_ids(reconstruction)
    if not image_ids:
        raise ValueError(f"{config.sparse_dir} has no registered images")
    # `run_mapping`'s own choice: the frame owning the lowest registered image id.
    gauge_frame_id: int = reconstruction.image(min(image_ids)).frame_id

    cusfm_config: CusfmConfig = read_config_directory(config.config_dir)
    ba_config: BundleAdjustmentConfig = cusfm_config.vision_mapping.bundle_adjustment
    if config.loss_type is not None:
        ba_config = replace(ba_config, loss_type=config.loss_type)
    # The default plan is Ceres with no polish, which is exactly what this tool solves.
    options: MappingOptions = MappingOptions(num_threads=config.num_threads, verbose=False)
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(ba_config, options)

    before_reprojection_px: float = reconstruction.compute_mean_reprojection_error()
    before_track: RigTrack = rig_track_of(reconstruction, template)
    num_observations: int = reconstruction.compute_num_observations()

    print(
        f"[polish] {config.label}: {len(image_ids)} images, {reconstruction.num_points3D()} points, "
        f"{num_observations} observations, gauge frame {gauge_frame_id}, "
        f"loss {ba_config.loss_type} scale {ba_options.ceres.loss_function_scale:.3f}, "
        f"max {ba_config.max_num_iterations} iterations"
    )
    started: float = time.perf_counter()
    summary, solve_seconds = solve_bundle_adjustment(reconstruction, ba_options, gauge_frame_id, None)
    print(f"[polish] solve took {solve_seconds:.1f} s ({time.perf_counter() - started:.1f} s including setup)")
    print(summary.brief_report())

    reconstruction.update_point_3d_errors()
    after_reprojection_px: float = reconstruction.compute_mean_reprojection_error()
    after_track: RigTrack = rig_track_of(reconstruction, template)

    displacement_m: Float64[ndarray, "n"] = np.linalg.norm(after_track.world_t_rig - before_track.world_t_rig, axis=1)
    fit: RigidAlignment = align_rigid(before_track.world_t_rig, after_track.world_t_rig, estimate_scale=True)
    solver: SolverReport = parse_brief_report(summary.brief_report())

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_colmap_model(config.output_dir, reconstruction)
    optimised: FramesMeta = write_optimised_frames_meta(config.output_dir, template, optimised_camera_poses(reconstruction), "ALIGNMENT")
    written: list[Path] = write_pose_files(config.output_dir, optimised)
    print(f"[polish] wrote sparse/, kpmap/keyframes/frames_meta.json and {len(written)} TUM files under {config.output_dir}")

    report: SolveReport = SolveReport(
        label=config.label,
        loss_type=ba_config.loss_type,
        sparse_dir=str(config.sparse_dir),
        output_dir=str(config.output_dir),
        num_registered_images=len(image_ids),
        num_points3D=reconstruction.num_points3D(),
        num_observations=num_observations,
        gauge_frame_id=gauge_frame_id,
        ba_num_iterations=solver.num_iterations,
        ba_initial_cost=solver.initial_cost,
        ba_final_cost=solver.final_cost,
        ba_termination=summary.termination_type.name,
        solve_seconds=solve_seconds,
        mean_reprojection_error_before_px=before_reprojection_px,
        mean_reprojection_error_after_px=after_reprojection_px,
        rig_displacement_rmse_m=float(np.sqrt(np.mean(displacement_m**2))),
        rig_displacement_max_m=float(displacement_m.max()),
        rig_residual_after_sim3_rmse_m=fit.rmse_meters,
        rig_residual_after_sim3_max_m=fit.max_error_meters,
        rig_sim3_scale=fit.scale,
    )
    (config.output_dir / "polish.json").write_text(to_json(report, indent=2))
    return report


NOT_REPORTED: Final[str] = "not reported"
"""Table entry for a statistic the solver did not publish; never a zero."""


def format_optional(value: float | None, spec: str) -> str:
    """Render a statistic that the solver may not have published.

    Args:
        value: The number, or None when the solver reported none.
        spec: Format specification applied when the number exists.

    Returns:
        The formatted number, or `NOT_REPORTED`.
    """
    return NOT_REPORTED if value is None else format(value, spec)


def main(config: PolishConfig) -> None:
    """Polish one model and print the numbers the experiment needs.

    Args:
        config: Parsed command-line configuration.
    """
    report: SolveReport = polish(config)
    solver: SolverReport = SolverReport(
        num_iterations=report.ba_num_iterations, initial_cost=report.ba_initial_cost, final_cost=report.ba_final_cost
    )
    print()
    print(f"| metric | {report.label} |")
    print("|---|---:|")
    iterations: str = NOT_REPORTED if report.ba_num_iterations is None else str(report.ba_num_iterations)
    print(f"| Ceres iterations | {iterations} |")
    print(f"| initial cost | {format_optional(report.ba_initial_cost, '.6g')} |")
    print(f"| final cost | {format_optional(report.ba_final_cost, '.6g')} |")
    print(f"| cost reduction | {format_optional(solver.cost_reduction, '.4%')} |")
    print(f"| termination | {report.ba_termination} |")
    print(f"| solve seconds | {report.solve_seconds:.1f} |")
    print(f"| mean reprojection before (px) | {report.mean_reprojection_error_before_px:.4f} |")
    print(f"| mean reprojection after (px) | {report.mean_reprojection_error_after_px:.4f} |")
    print(f"| rig displacement RMSE (m) | {report.rig_displacement_rmse_m:.4f} |")
    print(f"| rig displacement max (m) | {report.rig_displacement_max_m:.4f} |")
    print(f"| residual after Sim(3) fit RMSE (m) | {report.rig_residual_after_sim3_rmse_m:.4f} |")
    print(f"| residual after Sim(3) fit max (m) | {report.rig_residual_after_sim3_max_m:.4f} |")
    print(f"| Sim(3) scale of the displacement | {report.rig_sim3_scale:.6f} |")
    print(f"\nwrote {config.output_dir / 'polish.json'}")


if __name__ == "__main__":
    main(tyro.cli(PolishConfig))

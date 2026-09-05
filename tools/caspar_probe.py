"""Time COLMAP's CASPAR GPU bundle adjustment against Ceres, and compare the answers.

CASPAR is COLMAP's experimental GPU bundle-adjustment backend.  It is compiled in
only when libcolmap is built with `-DCASPAR_ENABLED=ON`, which no distributed
binary does; the `colsfm-caspar` environment supplies a from-source build that
does (see `docs/caspar-build.md`).  The conda-forge build binds the
`BundleAdjustmentBackend.CASPAR` enum regardless, so the enum is not a capability
probe — running a solve is.

Two problems are measured:

* the synthetic two-camera rig from `tests/colsfm_probes/synthetic.py`, perturbed
  away from its exact solution, which is small enough that the answer is known;
* a real reconstruction on disk (by default the 226-image Galileo model), with
  every frame free except one gauge frame, points free and intrinsics fixed.

Both are run under CASPAR's constraints: PINHOLE cameras, no extrinsic
refinement, and `refine_focal_length == refine_extra_params`.

Usage:
    pixi run -e colsfm-caspar python -m tools.caspar_probe
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pycolmap
import tyro
from jaxtyping import Float
from numpy import ndarray

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
# `tests/colsfm_probes/synthetic.py` is the suite's scene builder and is imported
# rather than duplicated.  The tests package is not installed, so its parent goes
# on `sys.path` exactly as `tests/colsfm_probes/conftest.py` arranges for pytest.
sys.path.insert(0, str(REPO_ROOT / "tests"))

from colsfm_probes.synthetic import SyntheticScene, add_ground_truth_points, build_scene

Backend: TypeAlias = Literal["ceres", "caspar"]
"""Selector for the bundle-adjustment implementation under test."""

_BACKENDS: dict[Backend, pycolmap.BundleAdjustmentBackend] = {
    "ceres": pycolmap.BundleAdjustmentBackend.CERES,
    "caspar": pycolmap.BundleAdjustmentBackend.CASPAR,
}


@dataclass(slots=True)
class SolveResult:
    """What one bundle-adjustment run cost and where it ended up."""

    backend: Backend
    """Which implementation produced this result."""
    seconds: float
    """Wall-clock time of `BundleAdjuster.solve()` alone, excluding setup."""
    reprojection_error_px: float
    """Mean reprojection error over all points after the solve."""
    termination: str
    """`BundleAdjustmentSummary.termination_type`, as a string."""
    frame_translations: Float[ndarray, "n_frames 3"]
    """Per-frame `rig_from_world` translation, ordered by frame id."""
    frame_rotations: Float[ndarray, "n_frames 3 3"]
    """Per-frame `rig_from_world` rotation matrix, ordered by frame id."""
    points_xyz: Float[ndarray, "n_points 3"]
    """Point positions, ordered by point3D id."""


@dataclass(slots=True)
class ProbeConfig:
    """Inputs of the CASPAR-versus-Ceres comparison."""

    galileo_sparse: Path = REPO_ROOT / "data/cusfm_runs/galileo_colsfm/cusfm/sparse"
    """Text-format COLMAP model to use for the real-problem comparison."""
    synthetic_frames: int = 5
    """Number of rig frames in the synthetic scene."""
    perturbation_m: float = 0.05
    """Standard deviation of the point perturbation applied to the synthetic scene."""
    gpu_index: str = "0"
    """CUDA device CASPAR runs on; "-1" lets COLMAP pick."""
    skip_galileo: bool = False
    """Run only the synthetic comparison, for a fast smoke test."""


def _caspar_safe_options(
    backend: Backend, gpu_index: str, refine_intrinsics: bool
) -> pycolmap.BundleAdjustmentOptions:
    """Build options both backends accept, so only the solver differs.

    CASPAR throws when `refine_sensor_from_rig` is true on a multi-sensor frame
    and when `refine_focal_length != refine_extra_params`, so both constraints
    are honoured for Ceres too — otherwise the two runs would not be solving the
    same problem.

    Args:
        backend: Which implementation to select.
        gpu_index: CUDA device index for CASPAR, as a string.
        refine_intrinsics: Whether focal length and extra params are free.

    Returns:
        Configured `BundleAdjustmentOptions`.
    """
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    options.backend = _BACKENDS[backend]
    options.print_summary = False
    options.refine_sensor_from_rig = False
    options.refine_focal_length = refine_intrinsics
    options.refine_extra_params = refine_intrinsics
    options.refine_principal_point = False
    options.refine_rig_from_world = True
    options.refine_points3D = True
    options.caspar.gpu_index = gpu_index
    return options


def _snapshot(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[Float[ndarray, "n_frames 3"], Float[ndarray, "n_frames 3 3"], Float[ndarray, "n_points 3"]]:
    """Read out the poses and points of a model in a stable, comparable order.

    Args:
        reconstruction: The model to read.

    Returns:
        Frame translations, frame rotation matrices and point positions.
    """
    frame_ids: list[int] = sorted(reconstruction.frames)
    point_ids: list[int] = sorted(reconstruction.point3D_ids())
    translations: Float[ndarray, "n_frames 3"] = np.array(
        [reconstruction.frame(i).rig_from_world.translation for i in frame_ids], dtype=np.float64
    )
    rotations: Float[ndarray, "n_frames 3 3"] = np.array(
        [reconstruction.frame(i).rig_from_world.rotation.matrix() for i in frame_ids],
        dtype=np.float64,
    )
    points: Float[ndarray, "n_points 3"] = np.array(
        [reconstruction.point3D(i).xyz for i in point_ids], dtype=np.float64
    )
    return translations, rotations, points


def _solve(
    reconstruction: pycolmap.Reconstruction,
    config: pycolmap.BundleAdjustmentConfig,
    backend: Backend,
    gpu_index: str,
    refine_intrinsics: bool,
) -> SolveResult:
    """Run one bundle adjustment and time the solve call itself.

    Args:
        reconstruction: Model to adjust, modified in place.
        config: Which images, points and blocks participate.
        backend: Implementation to use.
        gpu_index: CUDA device index for CASPAR.
        refine_intrinsics: Whether intrinsics are free.

    Returns:
        The timing, final error and geometry of the run.
    """
    options: pycolmap.BundleAdjustmentOptions = _caspar_safe_options(
        backend, gpu_index, refine_intrinsics
    )
    adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(
        options, config, reconstruction
    )
    start: float = time.perf_counter()
    summary: pycolmap.BundleAdjustmentSummary = adjuster.solve()
    seconds: float = time.perf_counter() - start

    reconstruction.update_point_3d_errors()
    translations, rotations, points = _snapshot(reconstruction)
    return SolveResult(
        backend=backend,
        seconds=seconds,
        reprojection_error_px=float(reconstruction.compute_mean_reprojection_error()),
        termination=str(summary.termination_type),
        frame_translations=translations,
        frame_rotations=rotations,
        points_xyz=points,
    )


def _make_perturbed_synthetic(
    num_frames: int, perturbation_m: float
) -> tuple[pycolmap.Reconstruction, pycolmap.BundleAdjustmentConfig, SyntheticScene]:
    """Build the synthetic rig scene and knock it off its exact solution.

    The scene is rebuilt from scratch for each backend rather than copied:
    `build_scene` is deterministic, so both backends start from bit-identical
    state, and `pycolmap.Reconstruction` has no deep-copy that preserves rigs.

    Args:
        num_frames: Number of rig frames.
        perturbation_m: Standard deviation of the point displacement, in metres.

    Returns:
        The perturbed model, its BA config, and the scene it came from.
    """
    scene: SyntheticScene = build_scene(num_frames=num_frames)
    add_ground_truth_points(scene)
    reconstruction: pycolmap.Reconstruction = scene.reconstruction

    rng: np.random.Generator = np.random.default_rng(7)
    for point3D_id in reconstruction.point3D_ids():
        point: pycolmap.Point3D = reconstruction.point3D(point3D_id)
        point.xyz = point.xyz + rng.normal(0.0, perturbation_m, 3)

    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images):
        config.add_image(image_id)
    # Gauge: freeze the first frame, otherwise the whole scene is free to drift.
    config.set_constant_rig_from_world_pose(min(reconstruction.frames))
    for camera_id in sorted(reconstruction.cameras):
        config.set_constant_cam_intrinsics(camera_id)
    return reconstruction, config, scene


def _make_galileo(sparse_dir: Path) -> tuple[pycolmap.Reconstruction, pycolmap.BundleAdjustmentConfig]:
    """Load a text COLMAP model and set up a global BA over it.

    Args:
        sparse_dir: Directory holding cameras/images/points3D/rigs/frames .txt.

    Returns:
        The model and a config with every frame free except one gauge frame.
    """
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reconstruction.read_text(str(sparse_dir))
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images):
        config.add_image(image_id)
    config.set_constant_rig_from_world_pose(min(reconstruction.frames))
    for camera_id in sorted(reconstruction.cameras):
        config.set_constant_cam_intrinsics(camera_id)
    return reconstruction, config


def _report(title: str, results: dict[Backend, SolveResult]) -> None:
    """Print the timings and the disagreement between two solutions.

    Args:
        title: Name of the problem being reported.
        results: One `SolveResult` per backend.
    """
    print(f"\n=== {title} ===")
    for result in results.values():
        print(
            f"  {result.backend:>6}: {result.seconds:8.3f} s  "
            f"reproj {result.reprojection_error_px:.6f} px  ({result.termination})"
        )
    if len(results) < 2:
        return
    ceres: SolveResult = results["ceres"]
    caspar: SolveResult = results["caspar"]
    print(f"  speedup (ceres/caspar): {ceres.seconds / caspar.seconds:.2f}x")

    translation_delta: Float[ndarray, " n_frames"] = np.linalg.norm(
        ceres.frame_translations - caspar.frame_translations, axis=1
    )
    rotation_delta_deg: Float[ndarray, " n_frames"] = np.degrees(
        np.arccos(
            np.clip(
                (np.trace(ceres.frame_rotations @ caspar.frame_rotations.transpose(0, 2, 1), axis1=1, axis2=2) - 1.0)
                / 2.0,
                -1.0,
                1.0,
            )
        )
    )
    point_delta: Float[ndarray, " n_points"] = np.linalg.norm(
        ceres.points_xyz - caspar.points_xyz, axis=1
    )
    extent: float = float(np.ptp(ceres.frame_translations, axis=0).max())
    print(
        f"  max pose change: {translation_delta.max():.6f} m "
        f"({100.0 * translation_delta.max() / max(extent, 1e-9):.4f}% of the {extent:.2f} m trajectory extent), "
        f"{rotation_delta_deg.max():.6f} deg"
    )
    print(
        f"  point agreement: median {np.median(point_delta):.6f} m, max {point_delta.max():.6f} m"
    )


def main(config: ProbeConfig) -> None:
    """Run the synthetic and real CASPAR-versus-Ceres comparisons.

    Args:
        config: Probe inputs.
    """
    print(f"pycolmap {pycolmap.__version__}  has_cuda={pycolmap.has_cuda}")

    # Creating the CUDA context and loading CASPAR's kernels costs ~1 s and
    # happens inside the first solve.  Burn it on a throwaway problem, otherwise
    # whichever CASPAR run goes first is timed against Ceres plus a GPU boot.
    warmup_reconstruction, warmup_config, _ = _make_perturbed_synthetic(
        config.synthetic_frames, config.perturbation_m
    )
    warmup: SolveResult = _solve(
        warmup_reconstruction, warmup_config, "caspar", config.gpu_index, refine_intrinsics=False
    )
    print(f"[warmup] first CASPAR solve (CUDA context + kernel load): {warmup.seconds:.3f} s")

    synthetic_results: dict[Backend, SolveResult] = {}
    for backend in ("ceres", "caspar"):
        reconstruction, ba_config, scene = _make_perturbed_synthetic(
            config.synthetic_frames, config.perturbation_m
        )
        reconstruction.update_point_3d_errors()
        before: float = reconstruction.compute_mean_reprojection_error()
        result: SolveResult = _solve(
            reconstruction, ba_config, backend, config.gpu_index, refine_intrinsics=False
        )
        truth_delta: float = float(
            np.abs(
                np.sort(result.points_xyz, axis=0) - np.sort(scene.points_xyz, axis=0)
            ).max()
        )
        print(
            f"[synthetic/{backend}] reproj {before:.4f} -> {result.reprojection_error_px:.6f} px, "
            f"max |point - truth| {truth_delta:.2e} m"
        )
        synthetic_results[backend] = result
    _report(
        f"synthetic rig ({config.synthetic_frames} frames, 2 cameras, PINHOLE)", synthetic_results
    )

    if config.skip_galileo:
        return
    if not config.galileo_sparse.is_dir():
        print(f"\n[skip] no model at {config.galileo_sparse}")
        return

    galileo_results: dict[Backend, SolveResult] = {}
    for backend in ("ceres", "caspar"):
        reconstruction, ba_config = _make_galileo(config.galileo_sparse)
        reconstruction.update_point_3d_errors()
        before = reconstruction.compute_mean_reprojection_error()
        print(
            f"[galileo/{backend}] {reconstruction.num_images()} images, "
            f"{reconstruction.num_frames()} frames, {reconstruction.num_points3D()} points, "
            f"reproj in {before:.6f} px"
        )
        galileo_results[backend] = _solve(
            reconstruction, ba_config, backend, config.gpu_index, refine_intrinsics=False
        )
    _report(f"global BA on {config.galileo_sparse}", galileo_results)


if __name__ == "__main__":
    main(tyro.cli(ProbeConfig))

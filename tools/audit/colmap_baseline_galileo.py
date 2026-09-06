"""C10/C11: run COLMAP's own pipeline on Galileo and time it against cuSFM.

The paper (§6.2) claims an order-of-magnitude total speedup over COLMAP and a
20x faster mapping stage, and (§6.3.1) that COLMAP "was unable to reconstruct
the entire trajectory within a single model". Both claims are made on KITTI,
which is not available here. Galileo is: 226 images, 8 rectified pinhole
cameras, a 0.66 m sweep, with ground truth — and both the NVIDIA blob (40.07 s)
and `colsfm` (19.35 s) have already been measured on it by
`colsfm.benchmark`. This script supplies the missing third column.

What it runs, all through `pycolmap` 4.2 in the `colsfm` environment:

1. **Feature extraction**, once per camera folder so that each of the eight
   folders gets its own COLMAP camera with the dataset's *known* rectified
   PINHOLE intrinsics and `has_prior_focal_length` set. That is deliberately
   generous to COLMAP: cuSFM is handed the same calibration, so withholding it
   from COLMAP would inflate the gap.
2. **Matching**, either `match_sequential` (the paper's COLMAP configuration,
   "sequential matcher mode", loop detection off) or `match_exhaustive`.
   Sequential matching orders images by name, and Galileo's names are
   `<camera_folder>/<timestamp>.jpeg`, so a sequential pass walks one camera at
   a time and only touches another camera at a folder boundary. Exhaustive is
   therefore also run: it is the configuration under which COLMAP has a real
   chance on an eight-camera rig, and reporting only the sequential number
   would be a straw man.
3. **Incremental mapping** with default `IncrementalPipelineOptions`.

What it reports: wall clock per stage, the number of reconstructions COLMAP
produced (§6.3.1's fragmentation claim), and — for the largest model — the
registered image count, point count, mean reprojection error, and ATE against
`ground_truth.txt` under both a rigid fit with the scale held at 1.0 (the
`colsfm.benchmark` metric, for a like-for-like number against 5.00/4.33 mm) and
a scale-free SIM(3) fit (the only fair metric for a reconstruction with no
metric scale).

Run:

    pixi run -e colsfm python -m tools.audit.colmap_baseline_galileo \\
        --features sift --matcher sequential
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import pycolmap
import tyro
from jaxtyping import Float64
from numpy import ndarray
from serde import serde

from colsfm.benchmark import ReconstructionMetrics, RigTrack, reconstruction_metrics, write_json_report
from colsfm.export import RUNTIME_CSV_NAME, RuntimeRecord, append_runtime_record
from colsfm.frames_meta import CameraParams, KeyframeMeta
from tools.audit.galileo_metrics import (
    GalileoReference,
    TrajectoryScore,
    image_center_score,
    read_galileo_reference,
    rig_track_from_reconstruction,
    score_track_against_ground_truth,
)

FeatureChoice: TypeAlias = Literal["sift", "aliked"]
"""Which COLMAP feature stack to run: SIFT-GPU + brute force, or ALIKED + LightGlue."""

MatcherChoice: TypeAlias = Literal["sequential", "exhaustive"]
"""Which COLMAP pairing strategy to run."""

EXTRACTOR_BY_CHOICE: dict[FeatureChoice, pycolmap.FeatureExtractorType] = {
    "sift": pycolmap.FeatureExtractorType.SIFT,
    "aliked": pycolmap.FeatureExtractorType.ALIKED_N16ROT,
}
"""COLMAP extractor for each choice; ALIKED-n16rot is the closest variant to cuSFM's engine."""

BaselineStage: TypeAlias = Literal["feature_extraction", "matching", "incremental_mapping", "export"]
"""The stages a COLMAP baseline times; the names it writes into `runtime.csv`.

`colsfm.runtime.classify_stage` maps them onto `extraction`, `matching`,
`mapping` and `export`, so a baseline run drops into the benchmark's runtime
table beside a blob run and a colsfm run. This module itself times only the
first three; `export` is timed by `tools.audit.colmap_rig_baseline_robocap`,
which writes a full cuSFM-layout run and shares `StageClock` with this one."""

MATCHER_BY_CHOICE: dict[FeatureChoice, pycolmap.FeatureMatcherType] = {
    "sift": pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE,
    "aliked": pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE,
}
"""COLMAP matcher paired with each extractor."""


@dataclass(frozen=True)
class ColmapBaselineConfig:
    """One COLMAP baseline run on the Galileo dataset."""

    features: FeatureChoice = "sift"
    """`sift` reproduces the paper's COLMAP configuration; `aliked` isolates the mapper."""
    matcher: MatcherChoice = "sequential"
    """`sequential` is the paper's configuration; `exhaustive` is the fair one for a rig."""
    overlap: int = 10
    """`SequentialPairingOptions.overlap`; ignored by the exhaustive matcher."""
    input_dir: Path = Path("data/r2b_galileo")
    """Dataset directory holding `frames_meta.json`, `ground_truth.txt` and the image folders."""
    output_root: Path = Path("data/cusfm_runs")
    """Runs are written to `<output_root>/audit_colmap_<features>_<matcher>/`."""
    bench_dir: Path = Path("data/bench")
    """Where the JSON summary is written, as `audit_colmap_<features>_<matcher>.json`."""
    use_gpu: bool = True
    """Run extraction and matching on the GPU."""
    overwrite: bool = True
    """Delete any previous run directory before starting, so timings are cold-cache-fair."""


@serde
@dataclass(frozen=True)
class ModelSummary:
    """One reconstruction COLMAP produced."""

    model_index: int
    """COLMAP's index for the model inside the output directory."""
    metrics: ReconstructionMetrics
    """The same counters `colsfm.benchmark` reports for a cuSFM-layout run, so the
    baseline's column and the benchmark's columns are produced by one function."""


@serde
@dataclass(frozen=True)
class ColmapBaselineResult:
    """Everything one baseline run measured."""

    features: str
    """Extractor choice."""
    matcher: str
    """Pairing choice."""
    run_dir: str
    """Where the artifacts were written."""
    total_images: int
    """Keyframes handed to COLMAP."""
    num_image_pairs_matched: int
    """Pairs with at least one verified match in the database."""
    num_models: int
    """How many disconnected reconstructions COLMAP returned (paper §6.3.1)."""
    models: tuple[ModelSummary, ...]
    """Every model, largest first."""
    stage_timings: tuple[RuntimeRecord, ...]
    """Per-stage wall clock, in pipeline order; `runtime.csv`'s own row type."""
    total_runtime_seconds: float
    """Sum of the stage timings."""
    largest_model_fraction_registered: float
    """Registered images of the largest model over `total_images`."""
    similarity_scale: float
    """Scale the SIM(3) fit of the largest model onto ground truth chose."""
    image_center_rigid_rmse_millimeters: float
    """Per-image camera-centre ATE, rigid fit, scale held at 1.0."""
    image_center_similarity_rmse_millimeters: float
    """Per-image camera-centre ATE after a scale-free SIM(3) fit."""
    rig_rigid_rmse_millimeters: float
    """Rig-frame ATE after rescaling by `similarity_scale`; comparable to 5.00/4.33 mm."""
    rig_similarity_rmse_millimeters: float
    """Rig-frame ATE under a further scale-free fit."""
    num_rig_frames: int
    """Rig frames the largest model covers."""


@dataclass
class StageClock:
    """Collects `time.perf_counter` spans, in the order they were measured."""

    timings: list[RuntimeRecord] = field(default_factory=list)
    """One entry per completed stage."""

    @contextmanager
    def stage(self, name: BaselineStage) -> Iterator[None]:
        """Time a block and record it under `name`.

        Args:
            name: Stage label for `runtime.csv`.

        Yields:
            Nothing; the block runs inside the timer.
        """
        started: float = time.perf_counter()
        try:
            yield
        finally:
            self.timings.append(RuntimeRecord(command=name, runtime_seconds=time.perf_counter() - started))

    def total_seconds(self) -> float:
        """Sum of every recorded stage.

        Returns:
            Total wall clock in seconds.
        """
        return float(sum(item.runtime_seconds for item in self.timings))

    def write_csv(self, output_dir: Path) -> None:
        """Append the timings to `<output_dir>/runtime.csv`.

        Goes through `colsfm.export.append_runtime_record`, so the header, the
        quoting and the dialect are the ones `read_runtime_records` expects — an
        unquoted hand-rolled row breaks the moment a command contains a comma.

        Args:
            output_dir: Directory holding the log; parents are created.
        """
        # One run, one log: the writer appends, so a reused directory would otherwise
        # accumulate blocks the way the blob's own runner does.
        (output_dir / RUNTIME_CSV_NAME).unlink(missing_ok=True)
        for record in self.timings:
            append_runtime_record(output_dir, record)


def _pinhole_params(camera: CameraParams) -> str:
    """Format a rectified camera's intrinsics as COLMAP `PINHOLE` parameters.

    Args:
        camera: A Galileo camera; its `projection_matrix` carries `fx, fy, cx, cy`.

    Returns:
        `"fx,fy,cx,cy"`.

    Raises:
        ValueError: When the camera ships no rectified projection matrix.
    """
    projection: Float64[ndarray, "3 4"] | None = camera.projection_matrix
    if projection is None:
        raise ValueError(f"camera {camera.sensor_name} has no projection_matrix")
    return f"{projection[0, 0]},{projection[1, 1]},{projection[0, 2]},{projection[1, 2]}"


def _extract(config: ColmapBaselineConfig, reference: GalileoReference, database_path: Path) -> None:
    """Extract features for all 226 keyframes, one COLMAP camera per folder.

    Args:
        config: The run's configuration.
        reference: The Galileo reference bundle, for the calibration and keyframe list.
        database_path: Database to write into; created on the first call.
    """
    names_by_camera: dict[int, list[str]] = {}
    for keyframe in reference.frames_meta.keyframes:
        names_by_camera.setdefault(keyframe.camera_params_id, []).append(keyframe.image_name)

    extraction_options: pycolmap.FeatureExtractionOptions = pycolmap.FeatureExtractionOptions()
    extraction_options.type = EXTRACTOR_BY_CHOICE[config.features]
    extraction_options.use_gpu = config.use_gpu
    device: pycolmap.Device = pycolmap.Device.cuda if config.use_gpu else pycolmap.Device.cpu

    for camera_params_id in sorted(names_by_camera):
        camera: CameraParams = reference.frames_meta.cameras[camera_params_id]
        reader_options: pycolmap.ImageReaderOptions = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "PINHOLE"
        reader_options.camera_params = _pinhole_params(camera)
        pycolmap.extract_features(
            database_path=database_path,
            image_path=config.input_dir,
            image_names=sorted(names_by_camera[camera_params_id]),
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=device,
        )


def _match(config: ColmapBaselineConfig, database_path: Path) -> None:
    """Match features with the configured pairing strategy.

    Args:
        config: The run's configuration.
        database_path: Database holding the extracted features.
    """
    matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    matching_options.type = MATCHER_BY_CHOICE[config.features]
    matching_options.use_gpu = config.use_gpu
    device: pycolmap.Device = pycolmap.Device.cuda if config.use_gpu else pycolmap.Device.cpu

    if config.matcher == "sequential":
        pairing_options: pycolmap.SequentialPairingOptions = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = config.overlap
        pairing_options.loop_detection = False
        pairing_options.expand_rig_images = False
        pycolmap.match_sequential(database_path=database_path, matching_options=matching_options, pairing_options=pairing_options, device=device)
    else:
        pycolmap.match_exhaustive(database_path=database_path, matching_options=matching_options, device=device)


def _count_verified_pairs(database_path: Path) -> int:
    """Count image pairs that survived geometric verification.

    Args:
        database_path: A COLMAP database written by the matching stage.

    Returns:
        Number of two-view geometries with at least one inlier.
    """
    with pycolmap.Database.open(database_path) as database:
        return int(database.num_verified_image_pairs())


def _summarise(reconstruction: pycolmap.Reconstruction, model_index: int) -> ModelSummary:
    """Reduce one reconstruction to the report's columns.

    Args:
        reconstruction: A model, already loaded.
        model_index: COLMAP's directory index for it.

    Returns:
        The summary.
    """
    reconstruction.update_point_3d_errors()
    return ModelSummary(model_index=model_index, metrics=reconstruction_metrics(reconstruction))


def main(config: ColmapBaselineConfig) -> None:
    """Run one COLMAP baseline on Galileo and write its timings and accuracy.

    Args:
        config: The run's configuration.
    """
    run_dir: Path = config.output_root / f"audit_colmap_{config.features}_{config.matcher}"
    if config.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    database_path: Path = run_dir / "database.db"
    sparse_dir: Path = run_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    reference: GalileoReference = read_galileo_reference(config.input_dir)
    keyframes: tuple[KeyframeMeta, ...] = reference.frames_meta.keyframes
    print(f"[audit] {config.features} + {config.matcher}: {len(keyframes)} images, {len(reference.frames_meta.cameras)} cameras")

    clock: StageClock = StageClock()
    with clock.stage("feature_extraction"):
        _extract(config, reference, database_path)
    with clock.stage("matching"):
        _match(config, database_path)
    num_pairs: int = _count_verified_pairs(database_path)
    print(f"[audit] verified image pairs: {num_pairs}")

    with clock.stage("incremental_mapping"):
        models: dict[int, pycolmap.Reconstruction] = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=config.input_dir,
            output_path=sparse_dir,
            options=pycolmap.IncrementalPipelineOptions(),
        )
    clock.write_csv(run_dir)

    # Materialised before sorting: `_summarise` calls `update_point_3d_errors`, and a
    # generator that mutates its inputs while `sorted` drains it is a trap, not a saving.
    measured: list[ModelSummary] = [_summarise(model, index) for index, model in models.items()]
    summaries: list[ModelSummary] = sorted(measured, key=lambda item: item.metrics.registered_images, reverse=True)
    print(f"[audit] {len(models)} model(s); registered per model: {[item.metrics.registered_images for item in summaries]}")

    if not summaries:
        raise RuntimeError("incremental_mapping returned no reconstruction")
    largest: pycolmap.Reconstruction = models[summaries[0].model_index]

    center_score, alignment = image_center_score(largest, reference)
    rig_track: RigTrack = rig_track_from_reconstruction(largest, reference.frames_meta, alignment)
    rig_score: TrajectoryScore = score_track_against_ground_truth(rig_track, reference.ground_truth)

    result: ColmapBaselineResult = ColmapBaselineResult(
        features=config.features,
        matcher=config.matcher,
        run_dir=str(run_dir),
        total_images=len(keyframes),
        num_image_pairs_matched=num_pairs,
        num_models=len(models),
        models=tuple(summaries),
        stage_timings=tuple(clock.timings),
        total_runtime_seconds=clock.total_seconds(),
        largest_model_fraction_registered=summaries[0].metrics.registered_images / len(keyframes),
        similarity_scale=alignment.scale,
        image_center_rigid_rmse_millimeters=center_score.rigid_rmse_millimeters,
        image_center_similarity_rmse_millimeters=center_score.similarity_rmse_millimeters,
        rig_rigid_rmse_millimeters=rig_score.rigid_rmse_millimeters,
        rig_similarity_rmse_millimeters=rig_score.similarity_rmse_millimeters,
        num_rig_frames=len(rig_track),
    )

    output_json: Path = write_json_report(config.bench_dir / f"audit_colmap_{config.features}_{config.matcher}.json", result)

    for timing in clock.timings:
        print(f"[audit] {timing.command:<22} {timing.runtime_seconds:8.2f} s")
    print(f"[audit] {'total':<22} {clock.total_seconds():8.2f} s")
    largest_metrics: ReconstructionMetrics = summaries[0].metrics
    print(
        f"[audit] largest model: {largest_metrics.registered_images}/{len(keyframes)} images, "
        f"{largest_metrics.num_points3D} points, {largest_metrics.mean_reprojection_error_px:.3f} px"
    )
    print(f"[audit] SIM(3) scale onto ground truth: {alignment.scale:.5f}")
    print(f"[audit] per-image ATE: rigid {center_score.rigid_rmse_millimeters:.2f} mm, SIM(3) {center_score.similarity_rmse_millimeters:.2f} mm")
    print(
        f"[audit] rig-frame ATE ({len(rig_track)} frames): rigid {rig_score.rigid_rmse_millimeters:.2f} mm, "
        f"SIM(3) {rig_score.similarity_rmse_millimeters:.2f} mm"
    )
    print(f"[audit] wrote {output_json}")


if __name__ == "__main__":
    main(tyro.cli(ColmapBaselineConfig))

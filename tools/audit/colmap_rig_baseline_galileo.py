"""C10/C11 addendum: run *standard* COLMAP with rig support on Galileo.

`tools.audit.colmap_baseline_galileo` runs COLMAP the way the paper describes
it — eight independent cameras, no rig — and COLMAP fragments the sweep.
COLMAP has shipped rig support since 3.12 (https://colmap.github.io/rigs.html):
`pycolmap.apply_rig_config` writes a rig and one frame per capture into the
database, after which the *stock* sequential matcher and the *stock* incremental
mapper are rig-aware. This script runs that configuration, so the audit can say
whether COLMAP's fragmentation is a mapper weakness or only a missing rig.

Two details make it work on Galileo:

1. **Frame grouping is by file name.** COLMAP groups images into frames by the
   part of the path that follows `image_prefix`, and Galileo's per-camera
   timestamps differ by a few microseconds inside one capture
   (`...246532.jpeg` vs `...247532.jpeg`), so the raw tree would give 226
   one-image frames and the rig would do nothing. The run therefore stages a
   symlink tree `images/<camera_folder>/<synced_sample_id>.jpeg`, which makes
   the suffix identical across cameras. Image names are mapped back to the
   dataset's own `image_name`s before scoring, so every metric is computed by
   the same `tools.audit.galileo_metrics` code as the rig-free baseline.
2. **`cam_from_rig` is `cam_i_T_cam_ref`.** The metadata's
   `sensor_to_vehicle_transform` is `vehicle_T_cam`, so the config carries
   `cam_i_T_vehicle * vehicle_T_cam_ref`, exactly what
   `colsfm.reconstruction.build_rig` writes. The reference sensor is the camera
   of the lowest keyframe id, the same rule `FramesMeta.rig_frames` uses.

Everything else is left at COLMAP's defaults on purpose: sequential pairing with
`overlap = 10` and loop detection off, and `IncrementalPipelineOptions()` with
`ba_refine_sensor_from_rig` at its default — **on**. The mapper is therefore free
to move the rig extrinsics, and the report measures how far it moved them.

Run:

    pixi run -e colsfm python -m tools.audit.colmap_rig_baseline_galileo \\
        --features aliked --matcher sequential
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pycolmap
import tyro
from serde import serde

from colsfm.benchmark import ReconstructionMetrics, RigTrack, write_json_report
from colsfm.frames_meta import FramesMeta, KeyframeMeta
from colsfm.reconstruction import RigReference, rig_reference
from colsfm.rig_calibration import ExtrinsicDelta, extrinsic_deltas, reference_camera_params_id, rig_config
from tools.audit.colmap_baseline_galileo import (
    EXTRACTOR_BY_CHOICE,
    MATCHER_BY_CHOICE,
    FeatureChoice,
    MatcherChoice,
    ModelSummary,
    StageClock,
    _pinhole_params,
    _summarise,
)
from tools.audit.galileo_metrics import (
    GalileoReference,
    TrajectoryScore,
    image_center_score,
    read_galileo_reference,
    rig_track_from_reconstruction,
    score_track_against_ground_truth,
)

STAGED_IMAGE_DIR_NAME: str = "images"
"""Symlink tree written inside the run directory; see the module docstring."""

@dataclass(frozen=True)
class ColmapRigBaselineConfig:
    """One rig-enabled COLMAP run on the Galileo dataset."""

    features: FeatureChoice = "aliked"
    """`aliked` is the COLMAP stack closest to cuSFM's own; `sift` is the paper's."""
    matcher: MatcherChoice = "sequential"
    """`sequential` is rig-aware once `apply_rig_config` has run; `exhaustive` is the fallback."""
    overlap: int = 10
    """`SequentialPairingOptions.overlap`, left at COLMAP's default value."""
    input_dir: Path = Path("data/r2b_galileo")
    """Dataset directory holding `frames_meta.json`, `ground_truth.txt` and the image folders."""
    output_root: Path = Path("data/cusfm_runs")
    """Runs are written to `<output_root>/audit_colmap_rig_<features>_<matcher>/`."""
    bench_dir: Path = Path("data/bench")
    """Where the JSON summary is written."""
    use_gpu: bool = True
    """Run extraction and matching on the GPU."""
    overwrite: bool = True
    """Delete any previous run directory first, so timings are cold-cache-fair."""


@serde
@dataclass(frozen=True)
class ColmapRigBaselineResult:
    """Everything one rig-enabled run measured."""

    features: str
    """Extractor choice."""
    matcher: str
    """Pairing choice."""
    run_dir: str
    """Where the artifacts were written."""
    reference_camera_params_id: int
    """Camera the rig origin sits on: the camera of the lowest keyframe id."""
    total_images: int
    """Keyframes handed to COLMAP."""
    total_rig_frames: int
    """Captures the staged tree grouped the keyframes into."""
    num_image_pairs_matched: int
    """Pairs with at least one verified match in the database."""
    num_models: int
    """How many disconnected reconstructions COLMAP returned (paper §6.3.1)."""
    models: tuple[ModelSummary, ...]
    """Every model, largest first."""
    stage_timings: tuple[tuple[str, float], ...]
    """Per-stage wall clock, in pipeline order."""
    total_runtime_seconds: float
    """Sum of the stage timings."""
    largest_model_fraction_registered: float
    """Registered images of the largest model over `total_images`."""
    similarity_scale: float
    """Scale the SIM(3) fit of the largest model onto ground truth chose."""
    would_be_scale: float
    """Scale the rigid fit would have picked had it been free to."""
    image_center_rigid_rmse_millimeters: float
    """Per-image camera-centre ATE, rigid fit, scale held at 1.0."""
    image_center_similarity_rmse_millimeters: float
    """Per-image camera-centre ATE after a scale-free SIM(3) fit."""
    rig_rigid_rmse_millimeters: float
    """Rig-frame ATE after rescaling by `similarity_scale`."""
    rig_similarity_rmse_millimeters: float
    """Rig-frame ATE under a further scale-free fit."""
    num_rig_frames_scored: int
    """Rig frames the largest model covers."""
    extrinsic_deltas: tuple[ExtrinsicDelta, ...]
    """Recovered rig extrinsics against the calibration, per non-reference camera."""


def _stage_images(frames_meta: FramesMeta, input_dir: Path, staged_root: Path) -> dict[str, str]:
    """Build the symlink tree whose file names group into rig frames.

    Args:
        frames_meta: The parsed metadata.
        input_dir: Dataset directory the symlinks point into.
        staged_root: Directory to write the tree into; created.

    Returns:
        Staged image name (`<camera_folder>/<synced_sample_id>.jpeg`) to the
        dataset's own `image_name`.
    """
    original_by_staged: dict[str, str] = {}
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    for rig_frame in frames_meta.rig_frames():
        for keyframe_id in rig_frame.keyframe_ids:
            keyframe: KeyframeMeta = keyframe_by_id[keyframe_id]
            folder: str = keyframe.image_name.split("/")[0]
            suffix: str = Path(keyframe.image_name).suffix
            staged_name: str = f"{folder}/{rig_frame.synced_sample_id}{suffix}"
            link: Path = staged_root / staged_name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to((input_dir / keyframe.image_name).resolve())
            original_by_staged[staged_name] = keyframe.image_name
    return original_by_staged


def _extract(config: ColmapRigBaselineConfig, reference: GalileoReference, staged_root: Path, database_path: Path) -> None:
    """Extract features from the staged tree, one COLMAP camera per folder.

    Args:
        config: The run's configuration.
        reference: The Galileo reference bundle, for the calibration.
        staged_root: The staged symlink tree.
        database_path: Database to write into.
    """
    extraction_options: pycolmap.FeatureExtractionOptions = pycolmap.FeatureExtractionOptions()
    extraction_options.type = EXTRACTOR_BY_CHOICE[config.features]
    extraction_options.use_gpu = config.use_gpu
    device: pycolmap.Device = pycolmap.Device.cuda if config.use_gpu else pycolmap.Device.cpu

    for camera_params_id, camera in sorted(reference.frames_meta.cameras.items()):
        folder: Path = staged_root / camera.sensor_name
        names: list[str] = sorted(f"{camera.sensor_name}/{item.name}" for item in folder.iterdir())
        reader_options: pycolmap.ImageReaderOptions = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "PINHOLE"
        reader_options.camera_params = _pinhole_params(camera)
        pycolmap.extract_features(
            database_path=database_path,
            image_path=staged_root,
            image_names=names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=device,
        )


def _match(config: ColmapRigBaselineConfig, database_path: Path) -> None:
    """Match features with COLMAP's stock pairing, left at its defaults.

    Args:
        config: The run's configuration.
        database_path: Database holding the extracted features and the rig.
    """
    matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    matching_options.type = MATCHER_BY_CHOICE[config.features]
    matching_options.use_gpu = config.use_gpu
    device: pycolmap.Device = pycolmap.Device.cuda if config.use_gpu else pycolmap.Device.cpu

    if config.matcher == "sequential":
        # Defaults throughout, including `expand_rig_images = True`: that is the
        # switch that makes a sequential pass over a rig match consecutive *frames*.
        pairing_options: pycolmap.SequentialPairingOptions = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = config.overlap
        pycolmap.match_sequential(
            database_path=database_path, matching_options=matching_options, pairing_options=pairing_options, device=device
        )
    else:
        pycolmap.match_exhaustive(database_path=database_path, matching_options=matching_options, device=device)


def _rename_to_dataset_names(reconstruction: pycolmap.Reconstruction, original_by_staged: dict[str, str]) -> None:
    """Put the dataset's own `image_name`s back on a model built from the staged tree.

    Every metric in `tools.audit.galileo_metrics` keys on `image_name`, so this is
    what lets the rig run and the rig-free run be scored by the same code.

    Args:
        reconstruction: The model to rewrite in place.
        original_by_staged: Map from `_stage_images`.
    """
    for image in reconstruction.images.values():
        image.name = original_by_staged[image.name]


def main(config: ColmapRigBaselineConfig) -> None:
    """Run one rig-enabled COLMAP baseline on Galileo and write its timings and accuracy.

    Args:
        config: The run's configuration.
    """
    run_dir: Path = config.output_root / f"audit_colmap_rig_{config.features}_{config.matcher}"
    if config.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    database_path: Path = run_dir / "database.db"
    sparse_dir: Path = run_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    staged_root: Path = run_dir / STAGED_IMAGE_DIR_NAME

    reference: GalileoReference = read_galileo_reference(config.input_dir)
    frames_meta: FramesMeta = reference.frames_meta
    reference_params_id: int = reference_camera_params_id(frames_meta)
    rig_ref: RigReference = rig_reference(frames_meta, reference_params_id)
    original_by_staged: dict[str, str] = _stage_images(frames_meta, config.input_dir, staged_root)
    num_rig_frames: int = len({name.split("/")[1] for name in original_by_staged})
    print(
        f"[audit-rig] {config.features} + {config.matcher}: {len(frames_meta.keyframes)} images, "
        f"{len(frames_meta.cameras)} cameras, {num_rig_frames} rig frames, "
        f"reference camera {reference_params_id} ({frames_meta.cameras[reference_params_id].sensor_name})"
    )

    clock: StageClock = StageClock()
    with clock.stage("feature_extraction"):
        _extract(config, reference, staged_root, database_path)
    with clock.stage("matching"):
        # Rig configuration is a sub-second database write with no stage of its own in
        # `runtime.csv`; it is counted here, against the run, rather than left untimed.
        with pycolmap.Database.open(database_path) as database:
            pycolmap.apply_rig_config([rig_config(frames_meta, rig_ref)], database)
        _match(config, database_path)
    with pycolmap.Database.open(database_path) as database:
        num_pairs: int = int(database.num_verified_image_pairs())
        num_frames_in_database: int = int(database.num_frames())
    print(f"[audit-rig] database frames: {num_frames_in_database}; verified image pairs: {num_pairs}")

    with clock.stage("incremental_mapping"):
        models: dict[int, pycolmap.Reconstruction] = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=staged_root,
            output_path=sparse_dir,
            options=pycolmap.IncrementalPipelineOptions(),
        )
    clock.write_csv(run_dir)

    for model in models.values():
        _rename_to_dataset_names(model, original_by_staged)
    measured: list[ModelSummary] = [_summarise(model, index) for index, model in models.items()]
    summaries: list[ModelSummary] = sorted(measured, key=lambda item: item.metrics.registered_images, reverse=True)
    print(f"[audit-rig] {len(models)} model(s); registered per model: {[item.metrics.registered_images for item in summaries]}")
    if not summaries:
        raise RuntimeError("incremental_mapping returned no reconstruction")
    largest: pycolmap.Reconstruction = models[summaries[0].model_index]

    center_score, alignment = image_center_score(largest, reference)
    rig_track: RigTrack = rig_track_from_reconstruction(largest, frames_meta, alignment)
    rig_score: TrajectoryScore = score_track_against_ground_truth(rig_track, reference.ground_truth)
    deltas: tuple[ExtrinsicDelta, ...] = extrinsic_deltas(largest, frames_meta, rig_ref)

    result: ColmapRigBaselineResult = ColmapRigBaselineResult(
        features=config.features,
        matcher=config.matcher,
        run_dir=str(run_dir),
        reference_camera_params_id=reference_params_id,
        total_images=len(frames_meta.keyframes),
        total_rig_frames=num_rig_frames,
        num_image_pairs_matched=num_pairs,
        num_models=len(models),
        models=tuple(summaries),
        stage_timings=tuple((item.command, item.runtime_seconds) for item in clock.timings),
        total_runtime_seconds=clock.total_seconds(),
        largest_model_fraction_registered=summaries[0].metrics.registered_images / len(frames_meta.keyframes),
        similarity_scale=alignment.scale,
        would_be_scale=center_score.would_be_scale,
        image_center_rigid_rmse_millimeters=center_score.rigid_rmse_millimeters,
        image_center_similarity_rmse_millimeters=center_score.similarity_rmse_millimeters,
        rig_rigid_rmse_millimeters=rig_score.rigid_rmse_millimeters,
        rig_similarity_rmse_millimeters=rig_score.similarity_rmse_millimeters,
        num_rig_frames_scored=len(rig_track),
        extrinsic_deltas=deltas,
    )
    output_json: Path = write_json_report(config.bench_dir / f"audit_colmap_rig_{config.features}_{config.matcher}.json", result)

    for timing in clock.timings:
        print(f"[audit-rig] {timing.command:<22} {timing.runtime_seconds:8.2f} s")
    print(f"[audit-rig] {'total':<22} {clock.total_seconds():8.2f} s")
    largest_metrics: ReconstructionMetrics = summaries[0].metrics
    print(
        f"[audit-rig] largest model: {largest_metrics.registered_images}/{len(frames_meta.keyframes)} images, "
        f"{largest_metrics.num_points3D} points, {largest_metrics.mean_reprojection_error_px:.3f} px"
    )
    print(f"[audit-rig] SIM(3) scale onto ground truth: {alignment.scale:.5f}; would-be scale of the rigid fit: {center_score.would_be_scale:.5f}")
    print(f"[audit-rig] per-image ATE: rigid {center_score.rigid_rmse_millimeters:.2f} mm, SIM(3) {center_score.similarity_rmse_millimeters:.2f} mm")
    print(
        f"[audit-rig] rig-frame ATE ({len(rig_track)} frames): rigid {rig_score.rigid_rmse_millimeters:.2f} mm, "
        f"SIM(3) {rig_score.similarity_rmse_millimeters:.2f} mm"
    )
    for delta in deltas:
        print(f"[audit-rig] extrinsic {delta.sensor_name:<28} {delta.translation_millimeters:8.2f} mm  {delta.rotation_degrees:7.3f} deg")
    print(f"[audit-rig] wrote {output_json}")


if __name__ == "__main__":
    main(tyro.cli(ColmapRigBaselineConfig))

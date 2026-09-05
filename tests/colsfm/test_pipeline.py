"""The end-to-end pipeline, run once on Galileo at the isaac default keyframe spacing.

The seam is `colsfm.pipeline.run_pipeline`: one call in, one `PipelineSummary`
out plus a workspace on disk. Everything asserted here is observed through that
public surface — the files a cuSFM run leaves behind, the COLMAP model as
`pycolmap` reads it back, `runtime.csv` as `colsfm.export` reads it back, and
`summary.json` through pyserde — never through the pipeline's internals.

The fixture uses the stock `feature_extractor_main` gates (0.5 m / 5 deg), which
keep 34 of Galileo's 226 keyframes: enough for a real reconstruction, fast
enough for a smoke test. The 226-frame run is the benchmark, not a unit test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from serde.json import from_json

from colsfm.benchmark import rig_rigidity_spread_millimeters
from colsfm.export import KEYFRAME_METADATA_SUBPATH, RUNTIME_CSV_NAME, RuntimeRecord, read_runtime_records
from colsfm.frames_meta import FRAMES_META_NAME, FramesMeta, read_frames_meta
from colsfm.geometry import MILLIMETRES_PER_METRE
from colsfm.pipeline import (
    ALL_STAGE_NAMES,
    STAGE_NAMES,
    SUMMARY_NAME,
    ExtrinsicChange,
    PipelineOptions,
    PipelineSummary,
    run_pipeline,
    stage_names,
)

SMOKE_MIN_INTER_FRAME_DISTANCE_M: float = 0.5
"""`feature_extractor_main`'s own default gate; keeps 34 of Galileo's 226 keyframes."""

MIN_REGISTERED_IMAGES: int = 30
"""Of the 34 selected keyframes, at least this many must end up registered."""

RIG_RIGIDITY_TOLERANCE_MM: float = 1e-6
"""How far the eight cameras of one rig instant may disagree on `world_T_vehicle`.

Pure algebra once the export is self-consistent: every pose is derived from the
same rig, so the only spread is float rounding."""

MIN_OBSERVING_FRACTION: float = 0.7
"""Fraction of the selected keyframes that must observe at least one point.

Deliberately below the registered count: 0.5 m spacing leaves five rig samples
across Galileo's 0.66 m sweep, and the cameras facing away from the overlap
legitimately see nothing. Measured 26 of 34."""


@pytest.fixture(scope="module")
def galileo_run(galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> PipelineSummary:
    """Run the whole pipeline once on Galileo into a temporary workspace.

    Args:
        galileo_input_dir: The r2b_galileo input directory, from the shared conftest.
        tmp_path_factory: pytest's per-module temporary directory factory.

    Returns:
        The summary the run produced.
    """
    input_dir: Path = galileo_input_dir
    if not (input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {input_dir / FRAMES_META_NAME}")
    options: PipelineOptions = PipelineOptions(
        input_dir=input_dir,
        output_dir=tmp_path_factory.mktemp("galileo_colsfm"),
        min_inter_frame_distance=SMOKE_MIN_INTER_FRAME_DISTANCE_M,
    )
    return run_pipeline(options)


def test_the_run_writes_every_artifact_a_cusfm_run_leaves_behind(galileo_run: PipelineSummary) -> None:
    """Each of cuSFM's output paths exists after the run."""
    output_dir: Path = galileo_run.options.output_dir
    expected: list[Path] = [
        output_dir / "keyframes" / FRAMES_META_NAME,
        output_dir / "database.db",
        output_dir / "pose_graph" / FRAMES_META_NAME,
        output_dir / "pose_graph" / "vehicle_pose.tum",
        output_dir / "sparse" / "cameras.txt",
        output_dir / "sparse" / "images.txt",
        output_dir / "sparse" / "points3D.txt",
        output_dir / KEYFRAME_METADATA_SUBPATH,
        output_dir / "output_poses" / "merged_pose_file.tum",
        output_dir / RUNTIME_CSV_NAME,
        output_dir / SUMMARY_NAME,
    ]
    missing: list[str] = [str(path) for path in expected if not path.exists()]
    assert not missing, f"pipeline did not write {missing}"


def test_the_sparse_model_loads_and_registers_the_selected_keyframes(galileo_run: PipelineSummary) -> None:
    """`sparse/` is a real COLMAP text model with most of the selected images in it."""
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reconstruction.read_text(str(galileo_run.options.output_dir / "sparse"))
    print(
        f"[colsfm] galileo smoke | registered {len(reconstruction.reg_image_ids())} "
        f"| points {reconstruction.num_points3D()} "
        f"| reprojection {reconstruction.compute_mean_reprojection_error():.4f} px"
    )
    assert len(reconstruction.reg_image_ids()) >= MIN_REGISTERED_IMAGES
    assert reconstruction.num_points3D() > 0
    assert reconstruction.num_points3D() == galileo_run.mapping.num_points3D
    assert galileo_run.mapping.num_registered_images == galileo_run.num_selected_keyframes
    assert galileo_run.mapping.num_images_with_observations >= MIN_OBSERVING_FRACTION * galileo_run.num_selected_keyframes


def test_the_runtime_log_holds_one_row_per_stage(galileo_run: PipelineSummary) -> None:
    """`runtime.csv` is cuSFM's two-column log, one row per stage, in stage order."""
    records: list[RuntimeRecord] = read_runtime_records(galileo_run.options.output_dir / RUNTIME_CSV_NAME)
    assert [record.command for record in records] == list(STAGE_NAMES)
    assert list(galileo_run.stage_seconds) == list(STAGE_NAMES)
    assert all(record.runtime_seconds >= 0.0 for record in records)


def test_the_summary_round_trips_through_pyserde(galileo_run: PipelineSummary) -> None:
    """`summary.json` deserialises back into the very summary the run returned."""
    restored: PipelineSummary = from_json(PipelineSummary, (galileo_run.options.output_dir / SUMMARY_NAME).read_text())
    assert restored == galileo_run


def test_the_pose_graph_stage_writes_a_pose_per_rig_frame(galileo_run: PipelineSummary) -> None:
    """`pose_graph/vehicle_pose.tum` carries one TUM line per rig frame."""
    lines: list[str] = (galileo_run.options.output_dir / "pose_graph" / "vehicle_pose.tum").read_text().splitlines()
    assert len(lines) == galileo_run.num_rig_frames
    assert all(len(line.split()) == 8 for line in lines)


# ── `--optimize-extrinsics`: the blob's second mapping pass ──────────────────────────


@pytest.fixture(scope="module")
def galileo_refined_run(galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> PipelineSummary:
    """The same smoke run with `--optimize-extrinsics`, into its own workspace.

    Args:
        galileo_input_dir: The r2b_galileo input directory, from the shared conftest.
        tmp_path_factory: pytest's per-module temporary directory factory.

    Returns:
        The summary the run produced.
    """
    input_dir: Path = galileo_input_dir
    if not (input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {input_dir / FRAMES_META_NAME}")
    options: PipelineOptions = PipelineOptions(
        input_dir=input_dir,
        output_dir=tmp_path_factory.mktemp("galileo_colsfm_ext"),
        min_inter_frame_distance=SMOKE_MIN_INTER_FRAME_DISTANCE_M,
        optimize_extrinsics=True,
    )
    return run_pipeline(options)


def test_the_refined_run_records_a_second_mapping_pass(galileo_refined_run: PipelineSummary) -> None:
    """`runtime.csv` gains one row, `extrinsic_refinement`, between mapping and export."""
    expected: list[str] = list(stage_names(optimize_extrinsics=True))
    assert expected == list(ALL_STAGE_NAMES) == [*STAGE_NAMES[:-1], "extrinsic_refinement", "export"]
    records: list[RuntimeRecord] = read_runtime_records(galileo_refined_run.options.output_dir / RUNTIME_CSV_NAME)
    assert [record.command for record in records] == expected
    assert list(galileo_refined_run.stage_seconds) == expected


def test_the_refinement_moves_the_extrinsics_and_says_by_how_much(galileo_refined_run: PipelineSummary) -> None:
    """`summary.json` carries one movement record per camera, and they are not all zero."""
    assert galileo_refined_run.options.optimize_extrinsics is True
    changes: tuple[ExtrinsicChange, ...] = galileo_refined_run.extrinsic_changes
    assert len(changes) > 1
    print(
        "[colsfm] galileo smoke refinement: "
        + ", ".join(
            f"{change.sensor_name}={change.translation_change_mm:.2f} mm/{change.rotation_change_deg:.3f} deg"
            for change in changes
        )
    )
    references: list[ExtrinsicChange] = [change for change in changes if change.is_reference]
    assert len(references) == 1, "exactly one camera is the rig origin"
    assert references[0].translation_change_mm == pytest.approx(0.0, abs=1e-9)
    assert references[0].rotation_change_deg == pytest.approx(0.0, abs=1e-9)
    moved: list[ExtrinsicChange] = [change for change in changes if not change.is_reference]
    assert all(change.translation_change_mm > 0.0 for change in moved), "every free extrinsic moved"


def test_the_refined_extrinsics_reach_the_exported_metadata(galileo_refined_run: PipelineSummary) -> None:
    """`kpmap/keyframes/frames_meta.json` carries the refined `sensor_to_vehicle_transform`.

    Read back through `colsfm.frames_meta` rather than out of the summary, so
    this is the file a downstream consumer would see.
    """
    exported: FramesMeta = read_frames_meta(galileo_refined_run.options.output_dir / KEYFRAME_METADATA_SUBPATH)
    source: FramesMeta = read_frames_meta(galileo_refined_run.options.output_dir / "pose_graph" / FRAMES_META_NAME)
    by_camera_params_id: dict[int, ExtrinsicChange] = {
        change.camera_params_id: change for change in galileo_refined_run.extrinsic_changes
    }
    for camera_params_id, camera in exported.cameras.items():
        written_mm: float = MILLIMETRES_PER_METRE * float(
            np.linalg.norm(
                np.asarray(camera.vehicle_T_cam.translation)
                - np.asarray(source.cameras[camera_params_id].vehicle_T_cam.translation)
            )
        )
        assert written_mm == pytest.approx(by_camera_params_id[camera_params_id].translation_change_mm, abs=1e-9)


def test_the_exported_camera_poses_agree_with_the_refined_extrinsics(
    galileo_refined_run: PipelineSummary,
) -> None:
    """`camera_to_world` and `sensor_to_vehicle_transform` in the export describe one rig.

    Every camera of a `synced_sample_id` must imply the same `world_T_vehicle`;
    if the export wrote refined extrinsics next to poses derived from the input
    ones, that spread would open up.
    """
    exported: FramesMeta = read_frames_meta(galileo_refined_run.options.output_dir / KEYFRAME_METADATA_SUBPATH)
    spread_mm: float = rig_rigidity_spread_millimeters(exported)
    print(f"[colsfm] galileo smoke refinement: rig rigidity spread {spread_mm:.4f} mm")
    assert spread_mm < RIG_RIGIDITY_TOLERANCE_MM


def test_the_refined_summary_round_trips_through_pyserde(galileo_refined_run: PipelineSummary) -> None:
    """The nested `ExtrinsicChange` records survive `summary.json` and come back."""
    restored: PipelineSummary = from_json(
        PipelineSummary, (galileo_refined_run.options.output_dir / SUMMARY_NAME).read_text()
    )
    assert restored == galileo_refined_run
    assert restored.extrinsic_changes == galileo_refined_run.extrinsic_changes


def test_a_run_without_the_flag_records_nothing_about_extrinsics(galileo_run: PipelineSummary) -> None:
    """With the flag off the summary and the stage list are exactly what they were."""
    assert galileo_run.options.optimize_extrinsics is False
    assert galileo_run.extrinsic_changes == ()
    assert stage_names(optimize_extrinsics=False) == STAGE_NAMES

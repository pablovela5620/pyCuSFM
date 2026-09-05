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

import pycolmap
import pytest
from serde.json import from_json

from colsfm.export import RUNTIME_CSV_NAME, RuntimeRecord, read_runtime_records
from colsfm.pipeline import STAGE_NAMES, PipelineOptions, PipelineSummary, run_pipeline

SMOKE_MIN_INTER_FRAME_DISTANCE_M: float = 0.5
"""`feature_extractor_main`'s own default gate; keeps 34 of Galileo's 226 keyframes."""

MIN_REGISTERED_IMAGES: int = 30
"""Of the 34 selected keyframes, at least this many must end up registered."""

MIN_OBSERVING_FRACTION: float = 0.7
"""Fraction of the selected keyframes that must observe at least one point.

Deliberately below the registered count: 0.5 m spacing leaves five rig samples
across Galileo's 0.66 m sweep, and the cameras facing away from the overlap
legitimately see nothing. Measured 26 of 34."""


@pytest.fixture(scope="module")
def galileo_run(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> PipelineSummary:
    """Run the whole pipeline once on Galileo into a temporary workspace.

    Args:
        repo_root: Repository root, from the shared conftest.
        tmp_path_factory: pytest's per-module temporary directory factory.

    Returns:
        The summary the run produced.
    """
    input_dir: Path = repo_root / "data" / "r2b_galileo"
    if not (input_dir / "frames_meta.json").is_file():
        pytest.skip(f"missing {input_dir / 'frames_meta.json'}")
    options: PipelineOptions = PipelineOptions(
        input_dir=input_dir,
        output_dir=tmp_path_factory.mktemp("galileo_colsfm"),
        min_inter_frame_distance=SMOKE_MIN_INTER_FRAME_DISTANCE_M,
    )
    return run_pipeline(options)


def test_the_run_writes_every_artifact_a_cusfm_run_leaves_behind(galileo_run: PipelineSummary) -> None:
    """Each of cuSFM's output paths exists after the run."""
    output_dir: Path = galileo_run.output_dir
    expected: list[Path] = [
        output_dir / "keyframes" / "frames_meta.json",
        output_dir / "database.db",
        output_dir / "pose_graph" / "frames_meta.json",
        output_dir / "pose_graph" / "vehicle_pose.tum",
        output_dir / "sparse" / "cameras.txt",
        output_dir / "sparse" / "images.txt",
        output_dir / "sparse" / "points3D.txt",
        output_dir / "kpmap" / "keyframes" / "frames_meta.json",
        output_dir / "output_poses" / "merged_pose_file.tum",
        output_dir / RUNTIME_CSV_NAME,
        output_dir / "summary.json",
    ]
    missing: list[str] = [str(path) for path in expected if not path.exists()]
    assert not missing, f"pipeline did not write {missing}"


def test_the_sparse_model_loads_and_registers_the_selected_keyframes(galileo_run: PipelineSummary) -> None:
    """`sparse/` is a real COLMAP text model with most of the selected images in it."""
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reconstruction.read_text(str(galileo_run.output_dir / "sparse"))
    print(
        f"[colsfm] galileo smoke | registered {len(reconstruction.reg_image_ids())} "
        f"| points {reconstruction.num_points3D()} "
        f"| reprojection {reconstruction.compute_mean_reprojection_error():.4f} px"
    )
    assert len(reconstruction.reg_image_ids()) >= MIN_REGISTERED_IMAGES
    assert reconstruction.num_points3D() > 0
    assert reconstruction.num_points3D() == galileo_run.num_points3D
    assert galileo_run.num_registered_images == galileo_run.num_selected_keyframes
    assert galileo_run.num_images_with_observations >= MIN_OBSERVING_FRACTION * galileo_run.num_selected_keyframes


def test_the_runtime_log_holds_one_row_per_stage(galileo_run: PipelineSummary) -> None:
    """`runtime.csv` is cuSFM's two-column log, one row per stage, in stage order."""
    records: list[RuntimeRecord] = read_runtime_records(galileo_run.output_dir / RUNTIME_CSV_NAME)
    assert [record.command for record in records] == list(STAGE_NAMES)
    assert list(galileo_run.stage_seconds) == list(STAGE_NAMES)
    assert all(record.runtime_seconds >= 0.0 for record in records)


def test_the_summary_round_trips_through_pyserde(galileo_run: PipelineSummary) -> None:
    """`summary.json` deserialises back into the very summary the run returned."""
    restored: PipelineSummary = from_json(PipelineSummary, (galileo_run.output_dir / "summary.json").read_text())
    assert restored == galileo_run


def test_the_pose_graph_stage_writes_a_pose_per_rig_frame(galileo_run: PipelineSummary) -> None:
    """`pose_graph/vehicle_pose.tum` carries one TUM line per rig frame."""
    lines: list[str] = (galileo_run.output_dir / "pose_graph" / "vehicle_pose.tum").read_text().splitlines()
    assert len(lines) == galileo_run.num_rig_frames
    assert all(len(line.split()) == 8 for line in lines)

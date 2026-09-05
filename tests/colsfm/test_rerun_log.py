"""The benchmark recording, checked by reading a saved `.rrd` back.

Every assertion goes through `rerun.experimental.RrdReader`, so what is checked
is what landed in the file, not what the logging calls were handed. NOTES.md
gotcha 9 applies: chunk record batches carry **bare** component names
(`Transform3D:translation`), not the entity-path-prefixed names the catalog
dataframe API returns, so the column lookups here are deliberately unprefixed.

The two runs under test are the synthetic cuSFM-layout fixtures from
`test_benchmark`, which pytest imports by module name because `tests/colsfm`
has no `__init__.py` and is therefore on `sys.path`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pycolmap
import pytest
import rerun.blueprint as rrb
import rerun.experimental as rrx
from jaxtyping import Float64, Int64, UInt8, UInt32
from numpy import ndarray
from test_benchmark import BLOB_RUNTIMES, COLSFM_RUNTIMES, SYNTHETIC_ERROR_PX, transform_rig_trajectory, write_synthetic_run

from colsfm.benchmark import Comparison, RigTrack, RunArtifacts, compare_runs, read_ground_truth, read_run, rig_track_from_frames_meta
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.rerun_log import (
    RUN_A_COLOR,
    RUN_A_INDEX,
    RUN_B_INDEX,
    TIMELINE,
    TrajectorySources,
    build_blueprint,
    log_points,
    save_comparison,
)

NANOSECONDS_PER_MICROSECOND: int = 1000
"""`log_rig_pose_stream` converts microsecond keyframe times to nanoseconds."""


@pytest.fixture(scope="module")
def galileo_input_dir(repo_root: Path) -> Path:
    """The r2b_galileo input directory: `frames_meta.json` plus `ground_truth.txt`."""
    return repo_root / "data" / "r2b_galileo"


@pytest.fixture(scope="module")
def galileo_input(galileo_input_dir: Path) -> FramesMeta:
    """The 226-keyframe input metadata."""
    return read_frames_meta(galileo_input_dir / "frames_meta.json")


@pytest.fixture(scope="module")
def three_samples(galileo_input: FramesMeta) -> FramesMeta:
    """The first three synchronised samples, enough for a two-segment polyline."""
    keep: list[int] = [keyframe_id for rig_frame in galileo_input.rig_frames()[:3] for keyframe_id in rig_frame.keyframe_ids]
    return galileo_input.filtered(keep)


@pytest.fixture(scope="module")
def runs(three_samples: FramesMeta, tmp_path_factory: pytest.TempPathFactory) -> tuple[RunArtifacts, RunArtifacts]:
    """Two synthetic runs, B offset from A by a known rigid transform."""
    root: Path = tmp_path_factory.mktemp("runs")
    offset: pycolmap.Rigid3d = pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([0.05, 0.0, 0.0]))
    run_a: RunArtifacts = read_run(
        write_synthetic_run(root / "a", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES), "blob"
    )
    run_b: RunArtifacts = read_run(
        write_synthetic_run(
            root / "b",
            transform_rig_trajectory(three_samples, rig_transform=offset),
            error_px=SYNTHETIC_ERROR_PX,
            runtimes=COLSFM_RUNTIMES,
        ),
        "colsfm",
    )
    return run_a, run_b


@pytest.fixture(scope="module")
def comparison(runs: tuple[RunArtifacts, RunArtifacts], galileo_input_dir: Path) -> Comparison:
    """The comparison the recording is built from."""
    return compare_runs(runs[0], runs[1], galileo_input_dir, "galileo")


@pytest.fixture(scope="module")
def sources(galileo_input: FramesMeta, galileo_input_dir: Path) -> TrajectorySources:
    """The input and ground-truth trajectories drawn alongside the two runs."""
    return TrajectorySources(
        input_track=rig_track_from_frames_meta(galileo_input), ground_truth=read_ground_truth(galileo_input_dir)
    )


@pytest.fixture(scope="module")
def recording_path(
    comparison: Comparison,
    runs: tuple[RunArtifacts, RunArtifacts],
    sources: TrajectorySources,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A saved `.rrd` holding the whole comparison."""
    path: Path = tmp_path_factory.mktemp("rrd") / "compare.rrd"
    return save_comparison(path, comparison, runs[0], runs[1], sources)


@pytest.fixture(scope="module")
def chunks(recording_path: Path) -> list[rrx.Chunk]:
    """Every chunk of the saved recording."""
    reader: rrx.RrdReader = rrx.RrdReader(str(recording_path))
    return list(reader.stream(store=reader.recordings()[0]).to_chunks())


def entity_paths(chunks: list[rrx.Chunk]) -> set[str]:
    """Every entity path present in a recording."""
    return {str(chunk.entity_path) for chunk in chunks}


def columns_at(chunks: list[rrx.Chunk], path: str) -> set[str]:
    """Every column name logged at one entity path (bare names — gotcha 9)."""
    names: set[str] = set()
    for chunk in chunks:
        if str(chunk.entity_path) == path:
            names.update(chunk.to_record_batch().schema.names)
    return names


def column_at(chunks: list[rrx.Chunk], path: str, column: str) -> pa.Array:
    """The first occurrence of one column at one entity path."""
    for chunk in chunks:
        if str(chunk.entity_path) != path:
            continue
        batch: pa.RecordBatch = chunk.to_record_batch()
        if column in batch.schema.names:
            return batch.column(column)
    raise AssertionError(f"no column {column!r} at {path!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Entity layout
# ─────────────────────────────────────────────────────────────────────────────


def test_both_runs_are_logged_as_rigs(chunks: list[rrx.Chunk], runs: tuple[RunArtifacts, RunArtifacts]) -> None:
    """`/world/rig_00` is run A, `/world/rig_01` is run B, each with its cameras."""
    paths: set[str] = entity_paths(chunks)

    assert "/world/rig_00" in paths
    assert "/world/rig_01" in paths
    num_cameras: int = len(runs[0].frames_meta.cameras)
    for rig_index in (RUN_A_INDEX, RUN_B_INDEX):
        for camera_index in range(num_cameras):
            assert f"/world/rig_{rig_index:02d}/cam_{camera_index:02d}/pinhole" in paths


def test_both_sparse_clouds_are_logged(chunks: list[rrx.Chunk]) -> None:
    """Each run's cloud sits in the world frame under its own `rig_NN` leaf."""
    paths: set[str] = entity_paths(chunks)

    assert "/world/points/rig_00" in paths
    assert "/world/points/rig_01" in paths
    assert "Points3D:positions" in columns_at(chunks, "/world/points/rig_00")


def test_every_trajectory_is_logged_as_a_polyline(chunks: list[rrx.Chunk]) -> None:
    """Both runs plus the input and the ground truth, each with endpoints."""
    paths: set[str] = entity_paths(chunks)

    for name in ("rig_00", "rig_01", "input", "ground_truth"):
        assert f"/world/runs/{name}/trajectory" in paths, name
        assert f"/world/runs/{name}/endpoints" in paths, name


def test_the_report_rides_along_as_a_text_document(chunks: list[rrx.Chunk]) -> None:
    """`/report` carries the same markdown the report file holds."""
    assert "/report" in entity_paths(chunks)

    text: str = str(column_at(chunks, "/report", "TextDocument:text")[0][0])

    assert "# cuSFM benchmark — galileo" in text
    assert "| Stage | A (s) | B (s) | B/A |" in text


# ─────────────────────────────────────────────────────────────────────────────
# Time
# ─────────────────────────────────────────────────────────────────────────────


def test_the_rig_poses_ride_the_frame_time_timeline(chunks: list[rrx.Chunk], runs: tuple[RunArtifacts, RunArtifacts]) -> None:
    """One `world_T_rig` per keyframe timestamp, on `frame_time` and nowhere else."""
    pose_chunks: list[rrx.Chunk] = [
        chunk for chunk in chunks if str(chunk.entity_path) == "/world/rig_00" and not chunk.is_static
    ]
    assert pose_chunks

    for chunk in pose_chunks:
        assert list(chunk.timeline_names) == [TIMELINE]
    assert sum(chunk.num_rows for chunk in pose_chunks) == len(runs[0].track)
    assert "Transform3D:translation" in columns_at(chunks, "/world/rig_00")


def test_everything_else_is_static(chunks: list[rrx.Chunk]) -> None:
    """Only the two rig pose streams are temporal; clouds and polylines are static."""
    temporal: set[str] = {str(chunk.entity_path) for chunk in chunks if not chunk.is_static}

    assert temporal == {"/world/rig_00", "/world/rig_01"}


def test_the_pose_timestamps_are_the_keyframe_timestamps(chunks: list[rrx.Chunk], runs: tuple[RunArtifacts, RunArtifacts]) -> None:
    """`frame_time` values are the metadata's microseconds, to within a microsecond.

    Not exact: `log_rig_pose_stream` converts nanoseconds to float64 *seconds*
    before handing them to `rr.TimeColumn`, and at Galileo's 1.7e9-second epoch
    a float64 second holds only ~380 ns of resolution. The join that produces
    every metric works on the integer microseconds in the metadata, so this
    rounding is display-only — but it is why the assertion has a tolerance.
    """
    track: RigTrack = runs[0].track
    logged: list[int] = []
    for chunk in chunks:
        if str(chunk.entity_path) == "/world/rig_00" and not chunk.is_static:
            # A duration timeline reads back as pandas `Timedelta`; `.value` is
            # the nanosecond count the SDK stored.
            logged.extend(int(value.value) for value in chunk.to_record_batch().column(TIMELINE).to_pylist())

    expected: Int64[ndarray, "n"] = track.timestamps_microseconds * NANOSECONDS_PER_MICROSECOND
    assert len(logged) == len(expected)
    assert np.abs(np.sort(np.asarray(logged, dtype=np.int64)) - np.sort(expected)).max() < NANOSECONDS_PER_MICROSECOND


# ─────────────────────────────────────────────────────────────────────────────
# Colours and geometry
# ─────────────────────────────────────────────────────────────────────────────


def test_a_polyline_has_one_segment_per_consecutive_pose(chunks: list[rrx.Chunk], runs: tuple[RunArtifacts, RunArtifacts]) -> None:
    """`n` poses become `n - 1` two-point segments, as the catalog slam layer draws them."""
    strips: pa.Array = column_at(chunks, "/world/runs/rig_00/trajectory", "LineStrips3D:strips")

    assert len(strips[0]) == len(runs[0].track) - 1


def test_a_black_cloud_falls_back_to_the_runs_flat_colour(runs: tuple[RunArtifacts, RunArtifacts], tmp_path: Path) -> None:
    """A run written without `--output_rgb` is all-black; it must not render invisible."""
    import rerun as rr

    black: RunArtifacts = RunArtifacts(
        name=runs[0].name,
        run_dir=runs[0].run_dir,
        reconstruction=runs[0].reconstruction,
        frames_meta=runs[0].frames_meta,
        track=runs[0].track,
        points_xyz=runs[0].points_xyz,
        points_rgb=np.zeros_like(runs[0].points_rgb),
        runtime_seconds_by_stage=runs[0].runtime_seconds_by_stage,
    )
    path: Path = tmp_path / "black.rrd"
    recording: rr.RecordingStream = rr.RecordingStream("colsfm_bench_black", recording_id="black")
    recording.save(str(path))
    with recording:
        assert log_points(black, RUN_A_INDEX, RUN_A_COLOR) == len(black.points_xyz)
    recording.flush(timeout_sec=30.0)

    reader: rrx.RrdReader = rrx.RrdReader(str(path))
    saved: list[rrx.Chunk] = list(reader.stream(store=reader.recordings()[0]).to_chunks())
    # Rerun packs a colour into one uint32 as 0xRRGGBBAA.
    packed: UInt32[ndarray, "n"] = column_at(saved, "/world/points/rig_00", "Points3D:colors")[0].values.to_numpy()
    channels: UInt8[ndarray, "n 4"] = np.stack([(packed >> shift) & 0xFF for shift in (24, 16, 8, 0)], axis=-1).astype(np.uint8)

    assert len(channels) == len(black.points_xyz)
    assert {tuple(int(value) for value in row) for row in channels[:, :3]} == {RUN_A_COLOR}


def test_the_cloud_positions_survive_the_round_trip(chunks: list[rrx.Chunk], runs: tuple[RunArtifacts, RunArtifacts]) -> None:
    """The world-frame positions read back are the reconstruction's own points."""
    positions: Float64[ndarray, "n 3"] = np.asarray(
        column_at(chunks, "/world/points/rig_00", "Points3D:positions")[0].values.to_pylist(), dtype=np.float64
    ).reshape(-1, 3)

    assert len(positions) == len(runs[0].points_xyz)
    assert np.allclose(np.sort(positions, axis=0), np.sort(runs[0].points_xyz, axis=0), atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Blueprint
# ─────────────────────────────────────────────────────────────────────────────


def test_the_blueprint_shows_the_scene_the_trajectories_and_the_report(comparison: Comparison) -> None:
    """Three panes, and the trajectory pane is scoped so the clouds cannot bury it."""
    blueprint: rrb.Blueprint = build_blueprint(comparison)
    row = blueprint.root_container
    assert isinstance(row, rrb.Horizontal)
    assert len(row.contents) == 3

    scene, trajectories, report = row.contents
    assert isinstance(scene, rrb.Spatial3DView)
    assert isinstance(trajectories, rrb.Spatial3DView)
    assert isinstance(report, rrb.TextDocumentView)
    assert trajectories.contents == ["+ /world/runs/**"]
    assert str(report.origin) == "/report"

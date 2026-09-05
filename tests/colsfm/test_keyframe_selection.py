"""Keyframe selection must reproduce `feature_extractor_main` frame for frame.

The reference outputs are real `feature_extractor_main` runs over
`data/r2b_galileo/frames_meta.json` with the flags in
`data/cusfm_re/analysis/sweep_extractor.sh`, kept under
`data/cusfm_re/runs/sweep_extractor/`. Each case pins both the surviving frame
ids and the renumbered rig ids, so a right-count / wrong-frames port fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.keyframe_selection import KeyframeSelection, SyncedSamples, apply_selection, select_keyframes, synchronise_samples

SWEEP_CASES: list[tuple[str, float, float]] = [
    ("d0_r5", 0.0, 5.0),
    ("d05_r5", 0.5, 5.0),
    ("d01_r5", 0.1, 5.0),
    ("d002_r5", 0.02, 5.0),
    ("d05_r0", 0.5, 0.0),
    ("d05_r05", 0.5, 0.5),
    ("d99_r99", 99.0, 999.0),
]
"""`(sweep directory, --min_inter_frame_distance, --min_inter_frame_rotation_degrees)`."""


@pytest.fixture(scope="module")
def galileo_input(galileo_input_meta: Path) -> FramesMeta:
    """The 226-keyframe Galileo input metadata."""
    return read_frames_meta(galileo_input_meta)


def test_sample_sync_reproduces_the_input_grouping_as_a_dense_rank(galileo_input: FramesMeta) -> None:
    """At the shipped 100 us the recomputed grouping equals the input's, renumbered 1..29.

    feature_extractor_main.md §4: Galileo's samples span at most 19 us while the
    gaps between them are 33 ms, so the threshold cannot change the partition.
    """
    samples: SyncedSamples = synchronise_samples(galileo_input, 100)
    assert sorted(samples.keyframe_ids_by_synced_sample_id) == list(range(1, 30))
    input_rank: dict[int, int] = {
        original: rank for rank, original in enumerate(sorted({k.synced_sample_id for k in galileo_input.keyframes}), start=1)
    }
    for keyframe in galileo_input.keyframes:
        assert samples.synced_sample_id_by_keyframe_id[keyframe.keyframe_id] == input_rank[keyframe.synced_sample_id]


@pytest.mark.parametrize(("case_name", "min_distance_m", "min_rotation_degrees"), SWEEP_CASES)
def test_selection_reproduces_the_blob_sweep(
    galileo_input: FramesMeta, repo_root: Path, case_name: str, min_distance_m: float, min_rotation_degrees: float
) -> None:
    """Every swept flag combination gives the blob's exact frame and rig id sets."""
    reference: FramesMeta = read_frames_meta(repo_root / "data" / "cusfm_re" / "runs" / "sweep_extractor" / case_name / "frames_meta.json")
    selection: KeyframeSelection = select_keyframes(
        galileo_input,
        min_inter_frame_distance_m=min_distance_m,
        min_inter_frame_rotation_degrees=min_rotation_degrees,
        sample_sync_threshold_microseconds=100,
    )
    expected_ids: list[int] = sorted(keyframe.keyframe_id for keyframe in reference.keyframes)
    expected_samples: list[int] = sorted({keyframe.synced_sample_id for keyframe in reference.keyframes})
    assert sorted(selection.kept_keyframe_ids) == expected_ids
    assert list(selection.kept_synced_sample_ids) == expected_samples


def test_isaac_defaults_keep_thirty_four_frames_over_five_rig_samples(galileo_input: FramesMeta) -> None:
    """The upstream defaults (0.5 m, 5 deg) keep 34 of 226 frames (NOTES.md decision 6)."""
    selection: KeyframeSelection = select_keyframes(
        galileo_input, min_inter_frame_distance_m=0.5, min_inter_frame_rotation_degrees=5.0
    )
    assert len(selection.kept_keyframe_ids) == 34
    assert selection.kept_synced_sample_ids == (1, 8, 15, 22, 29)


def test_zero_distance_variant_keeps_all_226(galileo_input: FramesMeta, galileo_run_dir: Path) -> None:
    """The demo's `--min_inter_frame_distance 0.0` keeps every frame, matching the shipped run."""
    selection: KeyframeSelection = select_keyframes(
        galileo_input, min_inter_frame_distance_m=0.0, min_inter_frame_rotation_degrees=5.0
    )
    assert len(selection.kept_keyframe_ids) == 226
    shipped: FramesMeta = read_frames_meta(galileo_run_dir / "cusfm" / "keyframes" / "frames_meta.json")
    assert sorted(selection.kept_keyframe_ids) == sorted(keyframe.keyframe_id for keyframe in shipped.keyframes)
    assert selection.kept_synced_sample_ids == tuple(range(1, 30))


def test_applying_a_selection_writes_the_extractor_style_collection(galileo_input: FramesMeta, repo_root: Path) -> None:
    """The filtered collection matches the blob's own output, keyframe by keyframe."""
    reference: FramesMeta = read_frames_meta(repo_root / "data" / "cusfm_re" / "runs" / "sweep_extractor" / "d05_r5" / "frames_meta.json")
    selection: KeyframeSelection = select_keyframes(
        galileo_input, min_inter_frame_distance_m=0.5, min_inter_frame_rotation_degrees=5.0
    )
    selected: FramesMeta = apply_selection(galileo_input, selection)
    assert [keyframe.keyframe_id for keyframe in selected.keyframes] == [keyframe.keyframe_id for keyframe in reference.keyframes]
    for ours, theirs in zip(selected.keyframes, reference.keyframes, strict=True):
        assert ours.synced_sample_id == theirs.synced_sample_id
        assert ours.image_name == theirs.image_name
        assert ours.camera_params_id == theirs.camera_params_id
        assert ours.timestamp_microseconds == theirs.timestamp_microseconds

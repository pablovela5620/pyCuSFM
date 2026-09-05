"""Pair selection against the pair lists `feature_matcher_task_builder_main` wrote."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from colsfm.database import ImagePair, create_database
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.pairs import consecutive_pairs, read_task_pairs, select_pairs, stereo_pairs, write_pair_list

GALILEO_LOOP_PAIRS: Final[tuple[ImagePair, ...]] = (
    (2362, 2402),
    (2373, 2409),
    (2377, 2394),
    (2463, 2482),
    (2521, 2547),
    (2534, 2562),
    (2537, 2569),
    (2550, 2573),
)
"""The eight LOOP edges of the shipped 226-keyframe Galileo pose graph.

They are the exact difference between the blob's 339-pair task file and the 331
pairs the default mode derives from the metadata alone, so they stand in for the
loop-closure stage this module does not own.
"""


@pytest.fixture(scope="module")
def galileo_full_run(repo_root: Path) -> Path:
    """The 226-keyframe cuSFM Galileo run (`min_inter_frame_distance 0.0`)."""
    return repo_root / "data" / "cusfm_runs" / "galileo" / "cusfm"


@pytest.fixture(scope="module")
def galileo_full_meta(galileo_full_run: Path) -> FramesMeta:
    """The metadata the blob's task builder was pointed at."""
    return read_frames_meta(galileo_full_run / "pose_graph" / "frames_meta.json")


@pytest.fixture(scope="module")
def galileo_selected_meta(galileo_run_dir: Path) -> FramesMeta:
    """The 32-keyframe run's metadata (`min_inter_frame_distance 0.5`)."""
    return read_frames_meta(galileo_run_dir / "pose_graph" / "frames_meta.json")


def test_galileo_default_mode_reproduces_the_blob_minus_loops(
    galileo_full_meta: FramesMeta, galileo_full_run: Path
) -> None:
    """331 pairs: 218 consecutive same-camera links plus 113 declared stereo pairs.

    The blob's task file holds 339, the extra eight being LOOP edges contributed
    by the pose graph rather than by the task builder's own rules
    (feature_matcher_task_builder_main.md §4.2).
    """
    pairs: list[ImagePair] = select_pairs(galileo_full_meta, connected_keyframe_num=1)
    assert len(pairs) == 331
    assert len(consecutive_pairs(galileo_full_meta, connected_keyframe_num=1)) == 218
    assert len(stereo_pairs(galileo_full_meta)) == 113

    blob: set[ImagePair] = set(read_task_pairs(galileo_full_run / "matches" / "tasks" / "matching_task_000.pb"))
    assert len(blob) == 339
    assert set(pairs) <= blob
    assert blob - set(pairs) == set(GALILEO_LOOP_PAIRS)


def test_galileo_with_loop_pairs_matches_the_blob_exactly(galileo_full_meta: FramesMeta, galileo_full_run: Path) -> None:
    """The task builder's pairs plus the pose graph's LOOP edges are the 339-pair task file.

    `select_pairs` deliberately does not take the loop edges — they only exist after
    retrieval and verification, and `colsfm.pipeline` matches them into the same
    database afterwards. Merging them here is what the pipeline's two stages add up to.
    """
    pairs: set[ImagePair] = set(select_pairs(galileo_full_meta, connected_keyframe_num=1)) | set(GALILEO_LOOP_PAIRS)
    blob: list[ImagePair] = sorted(read_task_pairs(galileo_full_run / "matches" / "tasks" / "matching_task_000.pb"))
    assert sorted(pairs) == blob


def test_selected_galileo_run_matches_the_blob_exactly(galileo_selected_meta: FramesMeta, galileo_run_dir: Path) -> None:
    """The 32-keyframe run has no loop edges, so the default mode is the whole answer."""
    pairs: list[ImagePair] = select_pairs(galileo_selected_meta, connected_keyframe_num=1)
    blob: list[ImagePair] = sorted(read_task_pairs(galileo_run_dir / "matches" / "tasks" / "matching_task_000.pb"))
    assert pairs == blob
    assert len(pairs) == 40


def test_pairs_are_normalised_deduplicated_and_sorted(galileo_selected_meta: FramesMeta) -> None:
    """`(min, max)`, unique, ascending — the invariants `UpdateMatchingTaskAndSave` keeps.

    The consecutive and stereo rules overlap wherever a stereo partner is also the
    next rig frame's same-camera successor, so the deduplication is load-bearing.
    """
    pairs: list[ImagePair] = select_pairs(galileo_selected_meta, connected_keyframe_num=1)
    assert pairs == sorted(set(pairs))
    assert all(first < second for first, second in pairs)
    assert set(pairs) == set(consecutive_pairs(galileo_selected_meta, 1)) | set(stereo_pairs(galileo_selected_meta))


def test_connected_keyframe_num_adds_further_rig_hops(galileo_selected_meta: FramesMeta) -> None:
    """A larger `connected_keyframe_num` links each rig frame to more successors.

    The shipped isaac `pose_graph_config.pb.txt` uses 1; 2 must be a strict
    superset, since the extra edges only ever add same-camera links.
    """
    near: set[ImagePair] = set(select_pairs(galileo_selected_meta, connected_keyframe_num=1))
    far: set[ImagePair] = set(select_pairs(galileo_selected_meta, connected_keyframe_num=2))
    assert near < far


def test_connected_keyframe_num_must_be_positive(galileo_selected_meta: FramesMeta) -> None:
    """`RunPoseGraph` rejects anything below 1, and so does this."""
    with pytest.raises(ValueError):
        select_pairs(galileo_selected_meta, connected_keyframe_num=0)


def test_write_pair_list_emits_the_image_names_colmap_expects(
    galileo_selected_meta: FramesMeta, tmp_path: Path
) -> None:
    """pycolmap addresses pairs by image name, so the list is written from the database."""
    database_path: Path = tmp_path / "pairs.db"
    create_database(database_path, galileo_selected_meta)
    pairs: list[ImagePair] = select_pairs(galileo_selected_meta, connected_keyframe_num=1)
    pair_list_path: Path = tmp_path / "pairs.txt"
    write_pair_list(pair_list_path, pairs, database_path)

    lines: list[str] = pair_list_path.read_text().splitlines()
    assert len(lines) == len(pairs)
    keyframe_by_id = galileo_selected_meta.keyframe_by_id()
    first, second = pairs[0]
    assert lines[0] == f"{keyframe_by_id[first].image_name} {keyframe_by_id[second].image_name}"


def test_write_pair_list_rejects_an_unknown_image(galileo_selected_meta: FramesMeta, tmp_path: Path) -> None:
    """A pair naming an image the database does not hold is an error, not a silent skip."""
    database_path: Path = tmp_path / "unknown.db"
    create_database(database_path, galileo_selected_meta)
    with pytest.raises(KeyError):
        write_pair_list(tmp_path / "pairs.txt", [(999_999, 999_998)], database_path)

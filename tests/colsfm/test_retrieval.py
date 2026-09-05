"""Tests for `colsfm.retrieval`, the Python replacement for cuSFM's BoW stage.

Seams under test (all public):

* `build_retrieval_index` — turning per-image descriptors into a similarity matrix.
* `RetrievalIndex.query` / `.score` — ranked candidates, self and temporal exclusion.
* `read_descriptors_from_database` — the pycolmap read path.

The Galileo test reads the shipped blob keyframe protos directly rather than running the
feature extractor, so it is offline and needs no GPU. Its reader lives in this file
because the feature/database modules belong to other workers.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float32, Int32
from numpy import ndarray

from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.retrieval import (
    ALIKED_DESCRIPTOR_DIM,
    Candidate,
    Descriptors,
    RetrievalConfig,
    RetrievalIndex,
    build_retrieval_index,
    read_descriptors_from_database,
)
from colsfm.schema import KEYFRAME, load_schema

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
"""Repo root, so data paths resolve regardless of the working directory."""

GALILEO_KEYFRAMES: Final[Path] = REPO_ROOT / "data/cusfm_runs/galileo/cusfm/keyframes"
"""The 226-keyframe blob run; each `<camera>/<ts>.pb` carries 2048 ALIKED descriptors."""

FRAME_PERIOD_US: Final[int] = 100_000
"""Synthetic capture period, so a temporal gate can be expressed in whole frames."""


# --------------------------------------------------------------------------------------
# synthetic corpus
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticCorpus:
    """A descriptor corpus with a known revisit structure."""

    descriptors: dict[int, Descriptors]
    """Descriptors per image id; ids are `0 .. n_images - 1`."""
    timestamps_us: dict[int, int]
    """Capture time per image id, one `FRAME_PERIOD_US` apart."""
    loop_offset: int
    """Image `i` and image `i + loop_offset` share `loop_fraction` of their descriptors."""
    loop_fraction: float
    """Fraction of descriptors an image shares with its revisit partner."""
    neighbour_fraction: float
    """Fraction an image shares with its immediate temporal neighbour."""


def make_synthetic_corpus(
    n_images: int = 100,
    descriptors_per_image: int = 64,
    loop_offset: int = 50,
    loop_fraction: float = 0.6,
    neighbour_fraction: float = 0.45,
    seed: int = 0,
) -> SyntheticCorpus:
    """Build a corpus where temporal neighbours look alike and `i`/`i + 50` are a revisit.

    Every image observes `descriptors_per_image` landmarks. Image `i` inherits its leading
    45 % of slots from the trailing 45 % of image `i - 1`, so consecutive frames overlap
    the way real video does while frames two apart share nothing. Image `i + loop_offset`
    then copies the trailing 60 % of image `i`'s slots — the revisit the retrieval must
    find. The two overlaps are deliberately different so the revisit has a unique best
    score and the test cannot pass on a tie.

    Args:
        n_images: How many images to synthesise; must be at least `2 * loop_offset`.
        descriptors_per_image: Landmarks observed per image.
        loop_offset: Distance in frames between an image and its revisit partner.
        loop_fraction: Fraction of slots a revisit pair shares.
        neighbour_fraction: Fraction of slots consecutive frames share.
        seed: Seed for the landmark directions and the observation noise.

    Returns:
        The corpus, with unit-norm float32 descriptors.

    Raises:
        ValueError: When the corpus is too short to hold a revisit pair.
    """
    if n_images < 2 * loop_offset:
        raise ValueError(f"n_images {n_images} must be at least 2 * loop_offset {loop_offset}")
    generator: np.random.Generator = np.random.default_rng(seed)
    loop_slots: int = round(loop_fraction * descriptors_per_image)
    neighbour_slots: int = round(neighbour_fraction * descriptors_per_image)

    landmark_ids: Int32[ndarray, "n_images descriptors_per_image"] = np.arange(
        n_images * descriptors_per_image, dtype=np.int32
    ).reshape(n_images, descriptors_per_image)
    for image_id in range(1, n_images):
        landmark_ids[image_id, :neighbour_slots] = landmark_ids[image_id - 1, -neighbour_slots:]
    for image_id in range(loop_offset, 2 * loop_offset):
        landmark_ids[image_id, -loop_slots:] = landmark_ids[image_id - loop_offset, -loop_slots:]

    landmarks: Float32[ndarray, "n_landmarks 128"] = generator.standard_normal(
        (n_images * descriptors_per_image, ALIKED_DESCRIPTOR_DIM)
    ).astype(np.float32)
    landmarks /= np.linalg.norm(landmarks, axis=1, keepdims=True)

    descriptors: dict[int, Descriptors] = {}
    for image_id in range(n_images):
        observed: Float32[ndarray, "descriptors_per_image 128"] = landmarks[landmark_ids[image_id]].copy()
        observed += generator.standard_normal(observed.shape).astype(np.float32) * np.float32(0.02)
        descriptors[image_id] = np.ascontiguousarray(observed / np.linalg.norm(observed, axis=1, keepdims=True))

    return SyntheticCorpus(
        descriptors=descriptors,
        timestamps_us={image_id: image_id * FRAME_PERIOD_US for image_id in range(n_images)},
        loop_offset=loop_offset,
        loop_fraction=loop_slots / descriptors_per_image,
        neighbour_fraction=neighbour_slots / descriptors_per_image,
    )


@pytest.fixture(scope="module")
def synthetic_corpus() -> SyntheticCorpus:
    """The shared synthetic corpus."""
    return make_synthetic_corpus()


@pytest.fixture(scope="module")
def synthetic_index(synthetic_corpus: SyntheticCorpus) -> RetrievalIndex:
    """An index over the synthetic corpus, built with the default settings."""
    return build_retrieval_index(synthetic_corpus.descriptors)


# --------------------------------------------------------------------------------------
# a. synthetic retrieval
# --------------------------------------------------------------------------------------


def test_revisit_partner_is_the_top_candidate_once_neighbours_are_excluded(
    synthetic_corpus: SyntheticCorpus, synthetic_index: RetrievalIndex
) -> None:
    """With the temporal gate on, `i + 50` outranks everything for every query."""
    gate_us: int = 5 * FRAME_PERIOD_US
    for query_id in range(synthetic_corpus.loop_offset):
        candidates: list[Candidate] = synthetic_index.query(
            query_id, top_k=20, min_time_gap_us=gate_us, timestamps=synthetic_corpus.timestamps_us
        )
        assert candidates, f"image {query_id} retrieved nothing"
        assert candidates[0].image_id == query_id + synthetic_corpus.loop_offset
        assert candidates[0].score > 0.1


def test_query_excludes_itself_and_its_temporal_neighbours(
    synthetic_corpus: SyntheticCorpus, synthetic_index: RetrievalIndex
) -> None:
    """The gate removes the query and every image within `min_time_gap_us` of it."""
    gate_us: int = 5 * FRAME_PERIOD_US
    candidates: list[Candidate] = synthetic_index.query(
        10, top_k=20, min_time_gap_us=gate_us, timestamps=synthetic_corpus.timestamps_us
    )
    returned: set[int] = {candidate.image_id for candidate in candidates}
    assert returned.isdisjoint(set(range(6, 15)))
    assert 10 not in returned


def test_the_gate_is_what_removes_the_neighbours(synthetic_corpus: SyntheticCorpus, synthetic_index: RetrievalIndex) -> None:
    """Unfiltered, the query's neighbours rank second and third; the gate is what drops them.

    This is the reason `docs/spec/bow.md` §9.2 lists neighbour exclusion under "must keep"
    — the pose graph already carries those links as sequential edges, so a candidate that
    close in time adds no information.
    """
    unfiltered: list[Candidate] = synthetic_index.query(10, top_k=5)
    assert {candidate.image_id for candidate in unfiltered[1:3]} == {9, 11}
    gated: list[Candidate] = synthetic_index.query(
        10, top_k=5, min_time_gap_us=5 * FRAME_PERIOD_US, timestamps=synthetic_corpus.timestamps_us
    )
    assert {9, 11}.isdisjoint({candidate.image_id for candidate in gated})


def test_scores_are_symmetric_and_match_the_known_overlap(
    synthetic_corpus: SyntheticCorpus, synthetic_index: RetrievalIndex
) -> None:
    """Mutual-nearest-neighbour votes recover the fraction of shared landmarks."""
    assert synthetic_index.score(10, 60) == pytest.approx(synthetic_corpus.loop_fraction, abs=0.05)
    assert synthetic_index.score(60, 10) == synthetic_index.score(10, 60)
    assert synthetic_index.score(10, 11) == pytest.approx(synthetic_corpus.neighbour_fraction, abs=0.05)
    assert synthetic_index.score(10, 10) == 1.0
    assert synthetic_index.score(0, 99) < 0.05


def test_unrelated_images_score_near_zero() -> None:
    """Random descriptor sets must not clear a `good_score_threshold`-style gate."""
    generator: np.random.Generator = np.random.default_rng(7)
    descriptors: dict[int, Descriptors] = {}
    for image_id in range(8):
        rows: Float32[ndarray, "64 128"] = generator.standard_normal((64, ALIKED_DESCRIPTOR_DIM)).astype(np.float32)
        descriptors[image_id] = np.ascontiguousarray(rows / np.linalg.norm(rows, axis=1, keepdims=True))
    index: RetrievalIndex = build_retrieval_index(descriptors)
    off_diagonal: Float32[ndarray, "n n"] = index.similarity - np.eye(8, dtype=np.float32)
    assert float(off_diagonal.max()) < 0.1


def test_query_rejects_an_unknown_image_and_a_missing_timestamp_map(synthetic_index: RetrievalIndex) -> None:
    """Bad arguments fail loudly rather than silently returning nothing."""
    with pytest.raises(KeyError):
        synthetic_index.query(10_000)
    with pytest.raises(ValueError):
        synthetic_index.query(10, min_time_gap_us=1)
    with pytest.raises(ValueError):
        synthetic_index.query(10, top_k=0)


def test_build_rejects_a_single_image() -> None:
    """An index over one image can answer nothing."""
    with pytest.raises(ValueError):
        build_retrieval_index({0: np.zeros((4, ALIKED_DESCRIPTOR_DIM), dtype=np.float32)})


# --------------------------------------------------------------------------------------
# a. runtime
# --------------------------------------------------------------------------------------


def _random_corpus(n_images: int, descriptors_per_image: int, seed: int) -> dict[int, Descriptors]:
    """Build `n_images` sets of unit-norm random descriptors.

    Args:
        n_images: How many images to synthesise.
        descriptors_per_image: Descriptors per image.
        seed: Seed for the generator.

    Returns:
        Float32 descriptors per image id.
    """
    generator: np.random.Generator = np.random.default_rng(seed)
    corpus: dict[int, Descriptors] = {}
    for image_id in range(n_images):
        rows: Float32[ndarray, "n 128"] = generator.standard_normal(
            (descriptors_per_image, ALIKED_DESCRIPTOR_DIM), dtype=np.float32
        )
        corpus[image_id] = np.ascontiguousarray(rows / np.linalg.norm(rows, axis=1, keepdims=True))
    return corpus


@pytest.mark.parametrize(("n_images", "budget_seconds"), [(226, 20.0), (900, 60.0)])
def test_build_stays_inside_the_runtime_budget(n_images: int, budget_seconds: float) -> None:
    """226 x 2048 in a few seconds and 900 x 2048 under a minute, on CPU NumPy.

    The full brute force over all 2048 descriptors per image is 5.5e13 FLOP for Galileo
    and 8.7e14 for 900 images, i.e. ~370 s and ~5900 s at this machine's measured 147
    GFLOP/s — so `RetrievalConfig.max_total_descriptors` is what makes the budget
    reachable. See the module docstring of `colsfm.retrieval` for the recall it costs.
    """
    corpus: dict[int, Descriptors] = _random_corpus(n_images, 2048, seed=n_images)
    started: float = time.perf_counter()
    index: RetrievalIndex = build_retrieval_index(corpus)
    elapsed: float = time.perf_counter() - started
    index.query(0, top_k=20)
    print(
        f"retrieval build: {n_images} images x 2048 descriptors -> "
        f"{index.descriptors_per_image} kept/image, {elapsed:.1f} s (budget {budget_seconds:.0f} s)"
    )
    assert elapsed < budget_seconds


# --------------------------------------------------------------------------------------
# b. real Galileo descriptors
# --------------------------------------------------------------------------------------


def read_blob_descriptors(keyframe_dir: Path, frames_meta: FramesMeta) -> dict[int, Descriptors]:
    """Read ALIKED descriptors out of the blob's `Keyframe` protos.

    `keypoint_vector.descriptors` is `n * 128` little-endian float32, L2-normalised
    (`docs/spec/feature_extractor_main.md` §4). One file per keyframe, at
    `<camera_name>/<image_stem>.pb`.

    Args:
        keyframe_dir: Directory holding `frames_meta.json` and the per-camera folders.
        frames_meta: The parsed metadata of the same directory.

    Returns:
        Float32 descriptors per keyframe id.
    """
    keyframe_class = load_schema().message_class(KEYFRAME)
    camera_names: dict[int, str] = {camera_id: camera.sensor_name for camera_id, camera in frames_meta.cameras.items()}
    descriptors: dict[int, Descriptors] = {}
    for keyframe in frames_meta.keyframes:
        path: Path = keyframe_dir / camera_names[keyframe.camera_params_id] / f"{Path(keyframe.image_name).stem}.pb"
        message = keyframe_class()
        message.ParseFromString(path.read_bytes())
        flat: Float32[ndarray, "n_values"] = np.frombuffer(message.keypoint_vector.descriptors, dtype=np.float32)
        descriptors[keyframe.keyframe_id] = np.ascontiguousarray(flat.reshape(-1, ALIKED_DESCRIPTOR_DIM))
    return descriptors


@pytest.fixture(scope="module")
def galileo_frames_meta() -> FramesMeta:
    """Metadata of the shipped 226-keyframe Galileo run."""
    if not GALILEO_KEYFRAMES.is_dir():
        pytest.skip(f"Galileo keyframes are not present at {GALILEO_KEYFRAMES}")
    return read_frames_meta(GALILEO_KEYFRAMES / "frames_meta.json")


@pytest.fixture(scope="module")
def galileo_descriptors(galileo_frames_meta: FramesMeta) -> dict[int, Descriptors]:
    """ALIKED descriptors of all 226 Galileo keyframes, straight from the blob protos."""
    return read_blob_descriptors(GALILEO_KEYFRAMES, galileo_frames_meta)


@pytest.fixture(scope="module")
def galileo_index(galileo_descriptors: Mapping[int, Descriptors]) -> RetrievalIndex:
    """A retrieval index over the real Galileo descriptors."""
    return build_retrieval_index(galileo_descriptors)


def test_galileo_descriptors_are_128d_and_unit_norm(galileo_descriptors: Mapping[int, Descriptors]) -> None:
    """The blob writes 2048 x 128 float32 rows whose norm is 1 to within 1e-3.

    The values are quantised (they come back through a float16-precision pipeline), so the
    norm drifts by up to 5e-4 — which is why `build_retrieval_index` renormalises.
    """
    assert len(galileo_descriptors) == 226
    sample: Descriptors = galileo_descriptors[min(galileo_descriptors)]
    assert sample.shape == (2048, ALIKED_DESCRIPTOR_DIM)
    assert np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-3)


def test_galileo_top_candidate_is_a_nearby_view(galileo_index: RetrievalIndex, galileo_frames_meta: FramesMeta) -> None:
    """Every keyframe's best match is the same camera or its stereo partner, and nearly always a nearby rig frame.

    Galileo is a 0.66 m sweep over 29 rig frames of 8 cameras, so there is no true
    revisit in the data: the only "loops" it can offer are near-duplicate views. That
    bounds what this test can assert. What the data *does* support is that the ranking is
    geometric rather than random:

    * the top candidate always comes from the query's own camera or its stereo partner
      (226/226) — cross-rig, cross-direction views never win;
    * the top candidate is within 2 rig frames for 194/226 keyframes, and one of the top 3
      is within 2 rig frames for 222/226.

    The thresholds below sit under the measured values with a margin, so the test fails on
    a real regression rather than on noise.
    """
    stereo_partner: dict[int, int] = {}
    for pair in galileo_frames_meta.stereo_pairs:
        stereo_partner[pair.left_camera_params_id] = pair.right_camera_params_id
        stereo_partner[pair.right_camera_params_id] = pair.left_camera_params_id
    keyframe_by_id = galileo_frames_meta.keyframe_by_id()
    rig_order: dict[int, int] = {
        rig.synced_sample_id: position for position, rig in enumerate(galileo_frames_meta.rig_frames())
    }

    same_camera_family: int = 0
    top1_within_two_rigs: int = 0
    top3_within_two_rigs: int = 0
    for keyframe_id in galileo_index.image_ids:
        candidates: list[Candidate] = galileo_index.query(keyframe_id, top_k=3)
        assert candidates, f"keyframe {keyframe_id} retrieved nothing"
        query_camera: int = keyframe_by_id[keyframe_id].camera_params_id
        query_rig: int = rig_order[keyframe_by_id[keyframe_id].synced_sample_id]
        best_camera: int = keyframe_by_id[candidates[0].image_id].camera_params_id
        same_camera_family += best_camera in (query_camera, stereo_partner[query_camera])
        offsets: list[int] = [abs(rig_order[keyframe_by_id[c.image_id].synced_sample_id] - query_rig) for c in candidates]
        top1_within_two_rigs += offsets[0] <= 2
        top3_within_two_rigs += min(offsets) <= 2

    n_keyframes: int = len(galileo_index.image_ids)
    print(
        f"galileo retrieval: build {galileo_index.build_seconds:.1f} s, "
        f"{galileo_index.descriptors_per_image} descriptors/image, "
        f"top1 same-or-stereo camera {same_camera_family}/{n_keyframes}, "
        f"top1 within 2 rigs {top1_within_two_rigs}/{n_keyframes}, "
        f"top3 within 2 rigs {top3_within_two_rigs}/{n_keyframes}"
    )
    assert same_camera_family == n_keyframes
    assert top1_within_two_rigs >= 0.80 * n_keyframes
    assert top3_within_two_rigs >= 0.95 * n_keyframes


def test_galileo_scores_separate_near_views_from_far_ones(galileo_index: RetrievalIndex) -> None:
    """The best score of a keyframe clears 0.1, while the median pair sits far below it."""
    best_per_row: Float32[ndarray, "n_images"] = (galileo_index.similarity - np.eye(len(galileo_index.image_ids), dtype=np.float32)).max(
        axis=1
    )
    assert float(np.median(best_per_row)) > 0.1
    assert float(np.median(galileo_index.similarity)) < 0.02


# --------------------------------------------------------------------------------------
# the pycolmap read path
# --------------------------------------------------------------------------------------


def test_descriptors_round_trip_through_a_colmap_database(tmp_path: Path, synthetic_corpus: SyntheticCorpus) -> None:
    """`read_descriptors_from_database` recovers float32 ALIKED blocks bit-exactly.

    COLMAP 4.2 stores descriptors as an opaque uint8 blob, so a 128-d float32 row is a
    512-byte uint8 row on disk and only `FeatureDescriptorsFloat.to_bytes()` /
    `FeatureDescriptors.to_float()` reinterpret it correctly
    (`docs/spec/pycolmap-capabilities.md` §4).
    """
    database_path: Path = tmp_path / "database.db"
    database: pycolmap.Database = pycolmap.Database.open(str(database_path))
    camera: pycolmap.Camera = pycolmap.Camera(camera_id=1, model="PINHOLE", width=640, height=480, params=[500.0, 500.0, 320.0, 240.0])
    database.write_camera(camera, use_camera_id=True)
    wanted: list[int] = [0, 1, 2]
    for image_id in wanted:
        database.write_image(pycolmap.Image(image_id=image_id + 1, name=f"{image_id}.png", camera_id=1), use_image_id=True)
        database.write_descriptors(
            image_id + 1,
            pycolmap.FeatureDescriptorsFloat(
                data=synthetic_corpus.descriptors[image_id], type=pycolmap.FeatureExtractorType.ALIKED_N32
            ).to_bytes(),
        )
    database.close()

    recovered: dict[int, Descriptors] = read_descriptors_from_database(database_path)
    assert sorted(recovered) == [1, 2, 3]
    for image_id in wanted:
        np.testing.assert_array_equal(recovered[image_id + 1], synthetic_corpus.descriptors[image_id])

    index: RetrievalIndex = build_retrieval_index(recovered, RetrievalConfig(max_descriptors_per_image=64))
    assert index.image_ids == (1, 2, 3)


def test_read_descriptors_rejects_a_missing_database(tmp_path: Path) -> None:
    """A missing database is an error, not an empty result."""
    with pytest.raises(FileNotFoundError):
        read_descriptors_from_database(tmp_path / "absent.db")

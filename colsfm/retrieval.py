"""Image retrieval for loop closure — the Python replacement for cuSFM's BoW stage.

`generate_bow_vocabulary_main` + `generate_bow_index_main` exist only to answer "which
other keyframes look like this keyframe?" so that `generate_association_main` and
`pose_graph_main` can propose loop pairs without all-pairs matching
(`docs/spec/bow.md` §1, §9.1). Nothing downstream reads the word ids, the tree, the IDF
values or the file formats, so this module drops all of them and implements **Option A**
of `docs/spec/bow.md` §9.4: brute-force cosine similarity on the ALIKED descriptors
themselves, aggregated into a per-image-pair score.

Score definition
----------------

For a pair of images the score is the **mutual-nearest-neighbour vote ratio**:

    score(a, b) = |{ i : nn_b(i) = j and nn_a(j) = i and sim(i, j) >= floor }| / n_per_image

where `nn_b(i)` is the descriptor of image `b` most similar to descriptor `i` of image
`a`. Mutual NN was chosen over max-pooled similarity because it is what the downstream
verifier does anyway (LightGlue + essential matrix on mutual matches), so the score
predicts the inlier count rather than merely "these two images contain a similar patch".
It lands in `[0, 1]` by construction, a frame retrieves itself with 1.0, and the
similarity floor keeps unrelated pairs near 0 — so a `good_score_threshold`-style gate
(`docs/spec/bow.md` §9.2) still means something. Measured on the 226 Galileo keyframes:
median best score 0.15 with the default floor, versus <0.01 for unrelated pairs.

Cost, and why the descriptors are subsampled
--------------------------------------------

The full brute force is `(n_images * n_descriptors)^2 * 128` inner products. Galileo is
226 x 2048 = 462 848 descriptors, i.e. 5.5e13 FLOP; this machine's NumPy runs float32
GEMM at a measured 147 GFLOP/s, so the full pass takes ~370 s and a 900-image sequence
~5900 s. Neither meets the "a few seconds / under a minute" budget, and no torch or
faiss is available in the `colsfm` environment. The fix is the one
`docs/spec/bow.md` §9.4 anticipates for Option A: keep only
`min(max_descriptors_per_image, max_total_descriptors // n_images)` descriptors per
image, which makes the pass quadratically cheaper. Measured on the real Galileo
descriptors (top-1 candidate within 2 rig frames / top-1 in the same or the stereo-paired
camera / a top-3 candidate within 2 rig frames):

| descriptors per image | build | within 2 rigs | same-or-stereo camera | top-3 within 2 rigs |
|---|---|---|---|---|
| 128 | 2.8 s | 164/226 | 226/226 | 216/226 |
| 256 (default) | 9.9 s | 194/226 | 226/226 | 222/226 |
| 512 | 52.6 s | 207/226 | 226/226 | 221/226 |

So subsampling costs recall roughly logarithmically while costing runtime
quadratically. 256 is the knee.

Note that `docs/spec/bow.md` §9.4's "a couple of seconds on CPU" for Option A assumes a
`faiss.IndexFlatIP`. faiss is **not** in the `colsfm` environment and neither is torch, so
that figure does not transfer; the subsampling above is what replaces the index.

A k-means visual-word index (Option C-lite) was measured as the alternative and **not
adopted**, but not because it fails. On the real Galileo descriptors, 1024 words trained
on a 50 000-descriptor sample plus one assignment pass over all 462 848 descriptors costs
1.6 s — cheaper than the 9.4 s spent here — and the vocabulary is healthy rather than
degenerate: the median word occurs in 87 of 226 images, so the IDF weight
`log(n_images / document_frequency)` has a median of 0.955 and nothing collapses. (K =
8192 gives a median document frequency of 24/226 and a median IDF of 2.24, at 35 s for a
full-corpus fit.) It is left out because `docs/spec/bow.md` §9.4 recommends Option A at
this scale, because brute force has no vocabulary to tune and no training set to overfit,
and because Option A's recall on the real data is measured above while Option C-lite's is
not. It is the obvious next step once a sequence is long enough for the quadratic term to
dominate; the score would have to be re-tuned, since a DBoW2 L1 score and a
mutual-nearest-neighbour vote ratio are not on the same scale.

Extension point for Option B (global descriptors)
-------------------------------------------------

`docs/spec/bow.md` §9.4 Option B replaces the descriptor stack with one global vector per
image (NetVLAD / DINOv2 / SALAD). It plugs in here without touching the callers: build an
index whose `similarity` matrix is `global_descriptors @ global_descriptors.T` on
L2-normalised rows and whose `descriptors_per_image` is 1, then `query` works unchanged.
It is deliberately not implemented — it needs a model download and a forward pass per
keyframe, and the acceptance bound is on registered images / ATE / reprojection error,
not on retrieval recall.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Float32, Int32
from numpy import ndarray

Descriptors: TypeAlias = Float32[ndarray, "n_descriptors descriptor_dim"]
"""One image's local descriptors, one row each. ALIKED emits 128-d L2-normalised rows."""

DescriptorScores: TypeAlias = Float32[ndarray, "n_descriptors"]
"""Per-descriptor detector responses, used to pick which descriptors survive subsampling."""

SimilarityMatrix: TypeAlias = Float32[ndarray, "n_images n_images"]
"""Symmetric image-to-image scores in `[0, 1]`, with 1.0 on the diagonal."""

ALIKED_DESCRIPTOR_DIM: Final[int] = 128
"""Descriptor width `feature_extractor_main` writes (`docs/spec/feature_extractor_main.md` §4)."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One retrieval hit for a query image."""

    image_id: int
    """Database image id of the retrieved image; the cuSFM keyframe id."""
    score: float
    """Mutual-nearest-neighbour vote ratio in `[0, 1]`; 1.0 means identical descriptor sets."""


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Knobs for the brute-force descriptor index."""

    max_descriptors_per_image: int = 256
    """Upper bound on descriptors kept per image. The measured knee of the recall/runtime
    curve on Galileo; see the module docstring's table."""
    max_total_descriptors: int = 100_000
    """Global cap, so long sequences stay inside the runtime budget. The per-image count
    becomes `min(max_descriptors_per_image, max_total_descriptors // n_images)`: 256 for
    Galileo's 226 images (9.4 s), 111 for a 900-image sequence (37 s)."""
    min_descriptor_similarity: float = 0.9
    """Cosine floor a mutual pair must clear to vote. Measured on Galileo: dropping the
    floor moves top-1-within-2-rig-frames from 194/226 down to 146/226, because unrelated
    images always produce a few mutual pairs by chance."""
    similarity_chunk_elements: int = 40_000_000
    """Roughly how many float32 similarities to hold at once (~160 MB). Only affects peak
    memory and cache behaviour, never the result."""

    def __post_init__(self) -> None:
        """Reject settings that would produce an empty or meaningless index."""
        if self.max_descriptors_per_image < 1:
            raise ValueError(f"max_descriptors_per_image must be at least 1, got {self.max_descriptors_per_image}")
        if self.max_total_descriptors < 1:
            raise ValueError(f"max_total_descriptors must be at least 1, got {self.max_total_descriptors}")
        if not -1.0 <= self.min_descriptor_similarity <= 1.0:
            raise ValueError(f"min_descriptor_similarity must be a cosine in [-1, 1], got {self.min_descriptor_similarity}")
        if self.similarity_chunk_elements < 1:
            raise ValueError(f"similarity_chunk_elements must be at least 1, got {self.similarity_chunk_elements}")


@dataclass(frozen=True, slots=True)
class RetrievalIndex:
    """Precomputed image-to-image similarity over a fixed set of images.

    The index is built over the same frames that query it, so a frame always retrieves
    itself with score 1.0 (`docs/spec/bow.md` §9.2); `query` skips it.
    """

    image_ids: tuple[int, ...]
    """Indexed image ids, ascending. Row/column `i` of `similarity` is `image_ids[i]`."""
    similarity: SimilarityMatrix
    """Symmetric mutual-nearest-neighbour vote ratios in `[0, 1]`."""
    descriptors_per_image: int
    """How many descriptors per image survived subsampling; the score's denominator."""
    config: RetrievalConfig
    """Settings the index was built with."""
    build_seconds: float
    """Wall time the build took, so pipelines can report a per-stage runtime."""

    def query(
        self,
        image_id: int,
        top_k: int = 20,
        min_time_gap_us: int | None = None,
        timestamps: Mapping[int, int] | None = None,
    ) -> list[Candidate]:
        """Rank the other indexed images by similarity to `image_id`.

        `query_result_number = 20` is the blob's own top-N (`docs/spec/bow.md` §9.1). The
        query image is always excluded, and so are images closer in time than
        `min_time_gap_us` — the pose graph already has sequential edges, so a loop
        candidate must be separated in time to add information
        (`docs/spec/bow.md` §9.2, `docs/spec/generate_association_main.md` §6.4).

        Args:
            image_id: Query image id; must be in `image_ids`.
            top_k: Maximum number of candidates to return.
            min_time_gap_us: Minimum `|dt|` in microseconds a candidate must be from the
                query, or None to apply no temporal gate.
            timestamps: Capture time in microseconds per image id. Required when
                `min_time_gap_us` is given.

        Returns:
            Candidates with a non-zero score, best first, at most `top_k` of them.

        Raises:
            KeyError: When `image_id` is not indexed, or a needed timestamp is missing.
            ValueError: When `min_time_gap_us` is given without `timestamps`, or
                `top_k` is not positive.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        if min_time_gap_us is not None and timestamps is None:
            raise ValueError("min_time_gap_us needs timestamps to apply the temporal gate")
        position: int = self._position_of(image_id)

        scores: Float32[ndarray, "n_images"] = self.similarity[position]
        order: Int32[ndarray, "n_images"] = np.argsort(-scores, kind="stable").astype(np.int32)
        query_timestamp_us: int = 0 if timestamps is None else timestamps[image_id]

        candidates: list[Candidate] = []
        for rank in order:
            if len(candidates) == top_k:
                break
            score: float = float(scores[rank])
            if score <= 0.0:
                break
            other_id: int = self.image_ids[int(rank)]
            if other_id == image_id:
                continue
            if min_time_gap_us is not None and timestamps is not None and abs(timestamps[other_id] - query_timestamp_us) < min_time_gap_us:
                continue
            candidates.append(Candidate(image_id=other_id, score=score))
        return candidates

    def score(self, image_id_a: int, image_id_b: int) -> float:
        """Look up the stored score of one pair.

        Args:
            image_id_a: First image id.
            image_id_b: Second image id.

        Returns:
            The mutual-nearest-neighbour vote ratio in `[0, 1]`.

        Raises:
            KeyError: When either id is not indexed.
        """
        return float(self.similarity[self._position_of(image_id_a), self._position_of(image_id_b)])

    def _position_of(self, image_id: int) -> int:
        """Row index of an image id.

        Args:
            image_id: The image id to locate.

        Returns:
            Its position in `image_ids`.

        Raises:
            KeyError: When the id is not indexed.
        """
        position: int = int(np.searchsorted(np.asarray(self.image_ids), image_id))
        if position >= len(self.image_ids) or self.image_ids[position] != image_id:
            raise KeyError(f"image id {image_id} is not in the retrieval index")
        return position


def read_descriptors_from_database(database_path: Path, image_ids: Sequence[int] | None = None) -> dict[int, Descriptors]:
    """Read float32 descriptors out of a COLMAP database.

    COLMAP 4.2 stores descriptors as an opaque uint8 blob, so an Nx128 float32 ALIKED
    block comes back as Nx512 uint8 and must be reinterpreted with
    `FeatureDescriptors.to_float()` — a reinterpret, not a cast
    (`docs/spec/pycolmap-capabilities.md` §4).

    Args:
        database_path: Path to `database.db`.
        image_ids: Images to read, or None for every image in the database.

    Returns:
        Descriptors per image id. Images with no descriptor row are skipped.

    Raises:
        FileNotFoundError: When the database file does not exist.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    database: pycolmap.Database = pycolmap.Database.open(str(database_path))
    try:
        wanted: list[int] = (
            [int(image.image_id) for image in database.read_all_images()] if image_ids is None else [int(image_id) for image_id in image_ids]
        )
        descriptors_by_image: dict[int, Descriptors] = {}
        for image_id in wanted:
            if not database.exists_descriptors(image_id):
                continue
            stored: pycolmap.FeatureDescriptors = database.read_descriptors(image_id)
            descriptors_by_image[image_id] = np.ascontiguousarray(stored.to_float().data, dtype=np.float32)
        return descriptors_by_image
    finally:
        database.close()


def _subsample(descriptors: Descriptors, keep: int, scores: DescriptorScores | None) -> Descriptors:
    """Reduce one image's descriptors to `keep` rows.

    With detector responses the strongest rows win, which is what
    `docs/spec/bow.md` §9.4 suggests; without them an even stride is used. A stride is
    unbiased because the blob's `Keyframe` proto carries no per-keypoint score
    (`docs/spec/feature_extractor_main.md` §4), so there is nothing to rank by, and
    taking a prefix would bias towards whatever order the detector happened to emit.

    Args:
        descriptors: Float32 descriptors with shape `[n_descriptors, descriptor_dim]`.
        keep: How many rows to keep; must not exceed `n_descriptors`.
        scores: Float32 per-descriptor responses with shape `[n_descriptors]`, or None.

    Returns:
        Float32 descriptors with shape `[keep, descriptor_dim]`.
    """
    if len(descriptors) == keep:
        return descriptors
    if scores is not None:
        strongest: Int32[ndarray, "keep"] = np.argsort(-scores, kind="stable")[:keep].astype(np.int32)
        return descriptors[np.sort(strongest)]
    stride: int = len(descriptors) // keep
    return descriptors[:: max(1, stride)][:keep]


def _stack_descriptors(
    descriptors_by_image: Mapping[int, Descriptors],
    image_ids: tuple[int, ...],
    keep: int,
    scores_by_image: Mapping[int, DescriptorScores] | None,
) -> Float32[ndarray, "n_images keep descriptor_dim"]:
    """Subsample every image to `keep` descriptors and L2-normalise them.

    The blob's ALIKED descriptors come out with a norm of 1.0002 rather than exactly 1,
    so they are renormalised here: the score is a cosine and must not inherit that drift.

    Args:
        descriptors_by_image: Descriptors per image id.
        image_ids: Image ids in the index's order.
        keep: Descriptors to keep per image.
        scores_by_image: Detector responses per image id, or None.

    Returns:
        Float32 array with shape `[n_images, keep, descriptor_dim]`, rows unit-norm.

    Raises:
        ValueError: When the descriptor widths disagree.
    """
    dim: int = int(descriptors_by_image[image_ids[0]].shape[1])
    rows: list[Descriptors] = []
    for image_id in image_ids:
        descriptors: Descriptors = np.asarray(descriptors_by_image[image_id], dtype=np.float32)
        if descriptors.shape[1] != dim:
            raise ValueError(f"image {image_id} has {descriptors.shape[1]}-d descriptors, expected {dim}")
        scores: DescriptorScores | None = None if scores_by_image is None else np.asarray(scores_by_image[image_id], dtype=np.float32)
        rows.append(_subsample(descriptors, keep, scores))
    stacked: Float32[ndarray, "n_images keep descriptor_dim"] = np.stack(rows).astype(np.float32)
    norms: Float32[ndarray, "n_images keep 1"] = np.linalg.norm(stacked, axis=2, keepdims=True)
    return stacked / np.maximum(norms, np.float32(1e-12))


def build_retrieval_index(
    descriptors_by_image: Mapping[int, Descriptors],
    config: RetrievalConfig | None = None,
    scores_by_image: Mapping[int, DescriptorScores] | None = None,
) -> RetrievalIndex:
    """Build the brute-force mutual-nearest-neighbour index (`docs/spec/bow.md` §9.4 Option A).

    One chunked `float32` GEMM over the whole subsampled descriptor stack gives, for every
    ordered pair of images, the forward and backward nearest neighbour of every descriptor;
    a pair votes when the two agree and the cosine clears
    `RetrievalConfig.min_descriptor_similarity`. The result is symmetric by construction.

    Args:
        descriptors_by_image: Local descriptors per image id, `[n_descriptors, dim]` each.
            Images may carry different counts; the smallest count caps the whole index.
        config: Subsampling and chunking settings; the defaults are the measured knee.
        scores_by_image: Optional detector responses per image id, used to pick the
            strongest descriptors instead of an even stride.

    Returns:
        The index, with its wall-clock build time recorded.

    Raises:
        ValueError: When there are fewer than two images, an image has no descriptors, or
            the descriptor widths disagree.
    """
    settings: RetrievalConfig = RetrievalConfig() if config is None else config
    image_ids: tuple[int, ...] = tuple(sorted(descriptors_by_image))
    n_images: int = len(image_ids)
    if n_images < 2:
        raise ValueError(f"a retrieval index needs at least two images, got {n_images}")
    smallest: int = min(int(descriptors_by_image[image_id].shape[0]) for image_id in image_ids)
    if smallest < 1:
        raise ValueError("every image must carry at least one descriptor")

    started: float = time.perf_counter()
    keep: int = min(settings.max_descriptors_per_image, smallest, max(1, settings.max_total_descriptors // n_images))
    stacked: Float32[ndarray, "n_images keep descriptor_dim"] = _stack_descriptors(
        descriptors_by_image, image_ids, keep, scores_by_image
    )
    flat: Float32[ndarray, "n_total descriptor_dim"] = np.ascontiguousarray(stacked.reshape(n_images * keep, -1))

    votes: SimilarityMatrix = np.zeros((n_images, n_images), dtype=np.float32)
    own_index: Int32[ndarray, "keep"] = np.arange(keep, dtype=np.int32)
    chunk: int = max(1, settings.similarity_chunk_elements // (keep * n_images * keep))
    floor: np.float32 = np.float32(settings.min_descriptor_similarity)
    for first in range(0, n_images, chunk):
        last: int = min(first + chunk, n_images)
        block: Float32[ndarray, "block descriptor_dim"] = flat[first * keep : last * keep]
        similarity: Float32[ndarray, "block keep n_images keep"] = (block @ flat.T).reshape(last - first, keep, n_images, keep)
        forward: Int32[ndarray, "block keep n_images"] = np.argmax(similarity, axis=3).astype(np.int32)
        backward: Int32[ndarray, "block n_images keep"] = np.argmax(similarity, axis=1).astype(np.int32)
        forward_by_image: Int32[ndarray, "block n_images keep"] = np.ascontiguousarray(forward.transpose(0, 2, 1))
        round_trip: Int32[ndarray, "block n_images keep"] = np.take_along_axis(backward, forward_by_image, axis=2)
        mutual: np.ndarray = round_trip == own_index[None, None, :]
        best_similarity: Float32[ndarray, "block n_images keep"] = np.ascontiguousarray(
            np.take_along_axis(similarity, forward[:, :, :, None], axis=3)[:, :, :, 0].transpose(0, 2, 1)
        )
        mutual &= best_similarity >= floor
        votes[first:last] = mutual.sum(axis=2, dtype=np.float32)

    scores: SimilarityMatrix = votes / np.float32(keep)
    np.fill_diagonal(scores, np.float32(1.0))
    return RetrievalIndex(
        image_ids=image_ids,
        similarity=scores,
        descriptors_per_image=keep,
        config=settings,
        build_seconds=time.perf_counter() - started,
    )

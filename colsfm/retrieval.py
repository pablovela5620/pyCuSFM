"""Image retrieval for loop closure — the Python replacement for cuSFM's BoW stage.

`generate_bow_vocabulary_main` + `generate_bow_index_main` exist only to answer "which
other keyframes look like this keyframe?" so that `generate_association_main` and
`pose_graph_main` can propose loop pairs without all-pairs matching
(`docs/spec/bow.md` §1, §9.1). Nothing downstream reads the word ids, the tree, the IDF
values or the file formats, so this module drops all of them and keeps only the ranked
candidate list.

Two backends, chosen by sequence length
---------------------------------------

`RetrievalConfig.backend` is `"auto"`: **brute force** at or below
`brute_force_max_images` (500) images, **vocab** above it. Both produce the same
`RetrievalIndex` — a symmetric `[0, 1]` score matrix over the indexed images — so
`query`, `score` and every consumer are backend-agnostic. What differs is the score's
definition, and therefore its calibration; see `default_good_score_threshold`.

**Brute force** (`docs/spec/bow.md` §9.4 Option A) scores a pair by the
**mutual-nearest-neighbour vote ratio**:

    score(a, b) = |{ i : nn_b(i) = j and nn_a(j) = i and sim(i, j) >= floor }| / n_per_image

Mutual NN was chosen over max-pooled similarity because it is what the downstream verifier
does anyway (LightGlue + essential matrix on mutual matches), so the score predicts the
inlier count rather than merely "these two images contain a similar patch". It lands in
`[0, 1]`, a frame retrieves itself with 1.0, and the similarity floor keeps unrelated pairs
near 0.

**Vocab** (`docs/spec/bow.md` §9.4 Option C, minus the parts §9.3 says to drop) trains a
hierarchical k-means vocabulary, encodes each image as an L1-normalised TF-IDF vector and
scores a pair with the blob's own DBoW2 L1 score (§7.4, §9.5).

Why brute force does not scale, measured
----------------------------------------

The brute force is `(n_images * n_descriptors)^2 * 128` inner products, so
`RetrievalConfig.max_total_descriptors` caps the per-image count at
`max_total_descriptors // n_images`. That is the knee at Galileo's 226 images (256
descriptors each, 9.4 s) and it **breaks down** on RoboCap: 4528 images leave 22
descriptors per image, the largest off-diagonal score is 0.227, and after the loop stage's
temporal gate nothing clears `good_score_threshold` at all — zero loop candidates on a
124 m walk with 90 genuine revisits. Raising the cap is not a fix: giving those 4528 images
Galileo's 256 descriptors each costs 3.4e14 FLOP, about 40 minutes at this machine's
measured 147 GFLOP/s float32 GEMM, and using every descriptor — which is what the vocab
path does — would be 2.2e16 FLOP, about 41 hours. The quadratic term is why the vocab
backend exists.

Measured on Galileo (226 images x 2048 descriptors), top-1 candidate within 2 rig frames /
top-1 in the same or the stereo-paired camera / a top-3 candidate within 2 rig frames:

| descriptors per image | build | within 2 rigs | same-or-stereo camera | top-3 within 2 rigs |
|---|---|---|---|---|
| 128 | 2.8 s | 164/226 | 226/226 | 216/226 |
| 256 (default) | 9.9 s | 194/226 | 226/226 | 222/226 |
| 512 | 52.6 s | 207/226 | 226/226 | 221/226 |

Note that `docs/spec/bow.md` §9.4's "a couple of seconds on CPU" for Option A assumes a
`faiss.IndexFlatIP`. faiss is **not** in the `colsfm` environment and neither is torch, so
that figure does not transfer.

The vocabulary tree, and why it is not the blob's
-------------------------------------------------

`docs/spec/bow.md` §6 builds `k = 9`, `depth = 7` — 5.4 M reserved nodes for 65 533 to
436 358 actual words, i.e. **about one descriptor per leaf**. §6.8 measures the occupancy
at 0.014 and 0.091 against the blob's own 0.7 maturity bar, and §9.4 concludes the tree is
"degenerate ... slower and no better" than a real vocabulary. So the shape here is chosen
by measurement instead. On the 4528 RoboCap images (9 253 392 ALIKED descriptors,
everything the extractor produced, no subsampling):

| tree | words | build | doc freq (median/max) | median IDF | top-1 within 2 rig frames |
|---|---|---|---|---|---|
| k=10, depth 4 | 10 000 | 67.3 s | 650 / 2094 | 1.94 | 4518/4528 |
| k=10, depth 5 | 100 000 | 42.2 s | 36 / 1996 | 4.83 | 4517/4528 |

Depth 5 is both faster and healthier: the all-pairs scoring cost is `sum_w df_w^2`, so ten
times more words makes the words ten times rarer and the scoring nearly ten times cheaper,
while the IDF stops collapsing towards zero. The 42.2 s breaks down as 6.0 s training,
12.4 s assignment, 0.6 s TF-IDF and 23.2 s all-pairs scoring; the 4528 `query` calls that
follow cost 1.4 s together, because the scoring already happened. For comparison the blob
spends 162.4 s on the vocabulary, 81.0 s on the index and 75.7 s on the association stage
(`docs/open-pipeline-plan.md`).

On Galileo's 226 images the vocab backend is also the **more accurate** of the two — top-1
within 2 rig frames for 222/226 keyframes against the brute force's 194/226, and 226/226 in
the same or the stereo-paired camera either way. `brute_force_max_images` is therefore about
cost and about which path has a recall figure measured on the shipped data, not about
quality: below 500 images the brute force has no vocabulary to over-fit to a short sequence
and its descriptor budget still buys 256 rows per image, so it stays the default there.

**Descriptors are not subsampled on the vocab path**, and that is a measured decision, not
an oversight. Keeping 512 per image cuts the build to 20.3 s but moves the median gated
best score from 0.153 to 0.069 — the L1 score counts *shared* words, so thinning the
descriptors thins the overlap and silently rescales the score out from under
`good_score_threshold`. Using every descriptor is also what the blob does (§6.3: "no
sampling, no capping"), which is what keeps its 0.1 threshold meaningful here.

What is deliberately not copied from `docs/spec/bow.md`
-------------------------------------------------------

* The integer-division IDF of §6.7 (`logf(N // df)`), reproduced only for bit-comparison
  (§9.3). The plain `log(n_images / df)` is used instead. Its one visible consequence —
  every word with `df > N/2` collapsing to weight 0 — happens naturally here, since
  `log(N/df)` goes to 0 as `df` goes to `N`, and the zero rows are dropped by
  `eliminate_zeros`.
* `min_shared_word_number = 50` (§9.1, and `docs/spec/generate_association_main.md` §7.1
  item 4, which lists it as a "must keep"). The score already subsumes it: since every
  term of `s = sum_w min(q_w, d_w)` is bounded by `q_w`, a pair that shares few words can
  only score high if those few words carry most of the query's L1 mass — which is the
  strong evidence the shared-word count was a proxy for. Keeping the count as well would
  add a second threshold to tune for no measured gain.
* The `VocabularyNode` slab format, BIRCH incremental building, `apply_merge_before_split`
  and the 2 GB shard split (§9.3).

Extension point for Option B (global descriptors)
-------------------------------------------------

`docs/spec/bow.md` §9.4 Option B replaces the descriptor stack with one global vector per
image (NetVLAD / DINOv2 / SALAD). It plugs in here without touching the callers: build an
index whose `similarity` matrix is `global_descriptors @ global_descriptors.T` on
L2-normalised rows, then `query` works unchanged. It is deliberately not implemented — it
needs a model download and a forward pass per keyframe, and the vocab backend already
covers all 90 of the blob's RoboCap loop pairs.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float32, Int32, Int64
from numpy import ndarray
from scipy import sparse
from scipy.cluster.vq import kmeans2

Descriptors: TypeAlias = Float32[ndarray, "n_descriptors descriptor_dim"]
"""One image's local descriptors, one row each. ALIKED emits 128-d L2-normalised rows."""

DescriptorScores: TypeAlias = Float32[ndarray, "n_descriptors"]
"""Per-descriptor detector responses, used to pick which descriptors survive subsampling."""

SimilarityMatrix: TypeAlias = Float32[ndarray, "n_images n_images"]
"""Symmetric image-to-image scores in `[0, 1]`, with 1.0 on the diagonal."""

RetrievalBackend: TypeAlias = Literal["auto", "brute_force", "vocab"]
"""How `build_retrieval_index` builds the score matrix; `auto` picks by image count."""

ResolvedBackend: TypeAlias = Literal["brute_force", "vocab"]
"""A `RetrievalBackend` after `auto` has been resolved against the corpus."""

ALIKED_DESCRIPTOR_DIM: Final[int] = 128
"""Descriptor width `feature_extractor_main` writes (`docs/spec/feature_extractor_main.md` §4)."""

GOOD_SCORE_THRESHOLD: Final[Mapping[ResolvedBackend, float]] = {"brute_force": 0.1, "vocab": 0.1}
"""Retrieval score a candidate must clear, per backend. See `default_good_score_threshold`."""


def default_good_score_threshold(backend: ResolvedBackend) -> float:
    """The measured `good_score_threshold` for one backend.

    The two backends' scores are different quantities and had to be calibrated separately,
    even though both land on 0.1:

    * **brute_force** — a mutual-nearest-neighbour vote ratio. 0.1 is the blob's number
      reused; on Galileo the median keyframe's best score is 0.15 and the median pair is
      below 0.02, so the gate separates near views from far ones.
    * **vocab** — the blob's own DBoW2 L1 score, so the blob's own 0.1
      (`docs/spec/generate_association_main.md` §7.1 item 4) transfers by construction, and
      it was re-measured rather than assumed. On RoboCap's 4528 images, bucketing every
      top-20 hit that survives the loop stage's 12.1 s temporal gate by the distance
      between the two rig frames:

      | score | hits | median distance | within 1 m | within 3 m |
      |---|---|---|---|---|
      | [0.00, 0.05) | 742 | 2.84 m | 12 % | 55 % |
      | [0.05, 0.10) | 31 251 | 0.95 m | 52 % | 84 % |
      | [0.10, 0.15) | 35 555 | 0.54 m | 88 % | 99 % |
      | [0.15, 0.20) | 16 622 | 0.43 m | 95 % | 100 % |
      | [0.20, 1.00) | 6 390 | 0.37 m | 99 % | 100 % |

      0.1 is where precision jumps from 52 % to 88 %, so it is the knee, not an inherited
      constant. Raise it to 0.15 to cut the pairs handed to the matcher roughly in half at
      95 % precision; the recall headroom is there, because the gated top-20 covers all 90
      of the blob's RoboCap loop pairs at either setting.

    Args:
        backend: The resolved backend of the index that produced the scores.

    Returns:
        The threshold in `[0, 1]`.
    """
    return GOOD_SCORE_THRESHOLD[backend]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One retrieval hit for a query image."""

    image_id: int
    """Database image id of the retrieved image; the cuSFM keyframe id."""
    score: float
    """Similarity in `[0, 1]`; 1.0 means identical. Its definition depends on the backend
    that built the index — a mutual-nearest-neighbour vote ratio for `brute_force`, the
    DBoW2 L1 score for `vocab`."""


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Knobs for both retrieval backends."""

    backend: RetrievalBackend = "auto"
    """Which index to build. `auto` takes `brute_force` at or below
    `brute_force_max_images` and `vocab` above it, because brute force is quadratic in the
    image count and its descriptor budget collapses on a long sequence; see the module
    docstring."""
    brute_force_max_images: int = 500
    """Image count at which `auto` switches to the vocab backend. Galileo's 226 images stay
    on brute force, whose recall is measured there; RoboCap's 4528 do not, because at that
    length `max_total_descriptors` leaves 22 descriptors per image and every score
    collapses."""
    seed: int = 0
    """Seed for the vocabulary's training subsample and k-means. The vocab backend is
    deterministic given this seed; the blob's is not, because it seeds libc `rand()` with
    nothing (`docs/spec/bow.md` §6.5)."""
    max_descriptors_per_image: int = 256
    """`brute_force` only. Upper bound on descriptors kept per image; the measured knee of
    the recall/runtime curve on Galileo. The vocab backend uses every descriptor, because
    thinning them rescales the L1 score (module docstring)."""
    max_total_descriptors: int = 100_000
    """`brute_force` only. Global cap, so the quadratic pass stays inside the runtime
    budget. The per-image count becomes
    `min(max_descriptors_per_image, max_total_descriptors // n_images)`."""
    min_descriptor_similarity: float = 0.9
    """`brute_force` only. Cosine floor a mutual pair must clear to vote. Measured on
    Galileo: dropping the floor moves top-1-within-2-rig-frames from 194/226 down to
    146/226, because unrelated images always produce a few mutual pairs by chance."""
    similarity_chunk_elements: int = 40_000_000
    """`brute_force` only. Roughly how many float32 similarities to hold at once (~160 MB).
    Only affects peak memory and cache behaviour, never the result."""
    vocab_branching: int = 10
    """`vocab` only. Children per vocabulary-tree node. The blob uses 9
    (`docs/spec/bow.md` §3); 10 is the same thing with round word counts."""
    vocab_depth: int = 5
    """`vocab` only. Tree levels, so the vocabulary holds `vocab_branching ** vocab_depth`
    words — 100 000 here. Measured against depth 4 on RoboCap: same top-1 recall, 1.6x
    faster to build and a median IDF of 4.83 instead of 1.94. The blob's depth 7 is
    degenerate (`docs/spec/bow.md` §6.8) and is not copied. Lowered automatically when the
    training sample cannot fill the tree."""
    vocab_training_descriptors: int = 200_000
    """`vocab` only. Descriptors drawn (seeded, evenly across images) to train the tree.
    200 000 trains in 6.0 s against 13.6 s for 500 000, with no measurable recall
    difference on RoboCap."""
    vocab_kmeans_iterations: int = 10
    """`vocab` only. Lloyd iterations per node, matching
    `weighted_kmeans_max_number_of_iterations` (`docs/spec/bow.md` §6.5)."""
    vocab_assign_chunk_descriptors: int = 1_000_000
    """`vocab` only. Descriptors held in memory per assignment pass (~500 MB at 128-d).
    Only affects peak memory, never the result."""

    def __post_init__(self) -> None:
        """Reject settings that would produce an empty or meaningless index.

        Raises:
            ValueError: When any bound is below its smallest useful value, or
                `min_descriptor_similarity` is not a cosine.
        """
        if self.max_descriptors_per_image < 1:
            raise ValueError(f"max_descriptors_per_image must be at least 1, got {self.max_descriptors_per_image}")
        if self.max_total_descriptors < 1:
            raise ValueError(f"max_total_descriptors must be at least 1, got {self.max_total_descriptors}")
        if not -1.0 <= self.min_descriptor_similarity <= 1.0:
            raise ValueError(f"min_descriptor_similarity must be a cosine in [-1, 1], got {self.min_descriptor_similarity}")
        if self.similarity_chunk_elements < 1:
            raise ValueError(f"similarity_chunk_elements must be at least 1, got {self.similarity_chunk_elements}")
        if self.brute_force_max_images < 1:
            raise ValueError(f"brute_force_max_images must be at least 1, got {self.brute_force_max_images}")
        if self.vocab_branching < 2:
            raise ValueError(f"vocab_branching must be at least 2, got {self.vocab_branching}")
        if self.vocab_depth < 1:
            raise ValueError(f"vocab_depth must be at least 1, got {self.vocab_depth}")
        if self.vocab_training_descriptors < self.vocab_branching:
            raise ValueError(f"vocab_training_descriptors must be at least vocab_branching, got {self.vocab_training_descriptors}")
        if self.vocab_kmeans_iterations < 1:
            raise ValueError(f"vocab_kmeans_iterations must be at least 1, got {self.vocab_kmeans_iterations}")
        if self.vocab_assign_chunk_descriptors < 1:
            raise ValueError(f"vocab_assign_chunk_descriptors must be at least 1, got {self.vocab_assign_chunk_descriptors}")

    def resolve_backend(self, n_images: int) -> ResolvedBackend:
        """Turn `auto` into a concrete backend for a corpus of this size.

        Args:
            n_images: How many images the index will hold.

        Returns:
            `"brute_force"` or `"vocab"`.
        """
        if self.backend != "auto":
            return self.backend
        return "brute_force" if n_images <= self.brute_force_max_images else "vocab"


@dataclass(frozen=True, slots=True)
class VocabularyTree:
    """A trained hierarchical k-means vocabulary, one centroid array per level.

    Level `l` holds `branching ** (l + 1)` centroid rows; the children of node `n` are rows
    `n * branching` to `(n + 1) * branching`. A leaf's index in the last level is its word
    id, so word ids need no separate table — unlike the blob, which numbers words by their
    rank among childless nodes in flat-array order (`docs/spec/bow.md` §6.6).
    """

    branching: int
    """Children per node."""
    centroids: tuple[Float32[ndarray, "n_children descriptor_dim"], ...]
    """Centroid rows per level, `[branching ** (level + 1), descriptor_dim]` each."""
    biases: tuple[Float32[ndarray, "n_children"], ...]
    """`-0.5 * ||centroid||^2` per row, so the descent is one `argmax` of
    `descriptor @ centroids.T + bias`. Rows of nodes that were never populated hold
    `-inf`, which is what keeps an all-zero centroid from winning a descent."""

    @property
    def depth(self) -> int:
        """How many levels the tree has.

        Returns:
            The level count; the word ids run over `branching ** depth`.
        """
        return len(self.centroids)

    @property
    def n_words(self) -> int:
        """Size of the vocabulary.

        Returns:
            `branching ** depth`, including leaves no descriptor ever reaches.
        """
        return self.branching**self.depth


@dataclass(frozen=True, slots=True)
class RetrievalIndex:
    """Precomputed image-to-image similarity over a fixed set of images.

    The index is built over the same frames that query it, so a frame always retrieves
    itself with score 1.0 (`docs/spec/bow.md` §9.2); `query` skips it.
    """

    image_ids: tuple[int, ...]
    """Indexed image ids, ascending. Row/column `i` of `similarity` is `image_ids[i]`."""
    similarity: SimilarityMatrix
    """Symmetric scores in `[0, 1]`; see `Candidate.score` for what they mean per backend."""
    descriptors_per_image: int
    """Descriptors each image contributed: the subsample size for `brute_force` (and the
    score's denominator), the mean count for `vocab`, which subsamples nothing."""
    config: RetrievalConfig
    """Settings the index was built with."""
    build_seconds: float
    """Wall time the build took, so pipelines can report a per-stage runtime."""
    backend: ResolvedBackend = "brute_force"
    """Which backend produced `similarity`. Feeds `default_good_score_threshold`."""
    n_words: int = 0
    """Vocabulary size for `vocab`, 0 for `brute_force`."""

    @property
    def good_score_threshold(self) -> float:
        """The measured score gate for this index's backend.

        Returns:
            The value `default_good_score_threshold` gives for `backend`.
        """
        return default_good_score_threshold(self.backend)

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
            The similarity in `[0, 1]`.

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


# ======================================================================================
# brute force (docs/spec/bow.md §9.4 Option A)
# ======================================================================================


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


def _brute_force_similarity(
    descriptors_by_image: Mapping[int, Descriptors],
    image_ids: tuple[int, ...],
    keep: int,
    scores_by_image: Mapping[int, DescriptorScores] | None,
    config: RetrievalConfig,
) -> SimilarityMatrix:
    """Mutual-nearest-neighbour vote ratios over the subsampled descriptor stack.

    One chunked `float32` GEMM gives, for every ordered pair of images, the forward and
    backward nearest neighbour of every descriptor; a pair votes when the two agree and the
    cosine clears `RetrievalConfig.min_descriptor_similarity`.

    Args:
        descriptors_by_image: Descriptors per image id.
        image_ids: Image ids in the index's order.
        keep: Descriptors kept per image.
        scores_by_image: Detector responses per image id, or None.
        config: The retrieval settings.

    Returns:
        The symmetric score matrix, 1.0 on the diagonal.
    """
    n_images: int = len(image_ids)
    stacked: Float32[ndarray, "n_images keep descriptor_dim"] = _stack_descriptors(descriptors_by_image, image_ids, keep, scores_by_image)
    flat: Float32[ndarray, "n_total descriptor_dim"] = np.ascontiguousarray(stacked.reshape(n_images * keep, -1))

    votes: SimilarityMatrix = np.zeros((n_images, n_images), dtype=np.float32)
    own_index: Int32[ndarray, "keep"] = np.arange(keep, dtype=np.int32)
    chunk: int = max(1, config.similarity_chunk_elements // (keep * n_images * keep))
    floor: np.float32 = np.float32(config.min_descriptor_similarity)
    for first in range(0, n_images, chunk):
        last: int = min(first + chunk, n_images)
        block: Float32[ndarray, "block descriptor_dim"] = flat[first * keep : last * keep]
        similarity: Float32[ndarray, "block keep n_images keep"] = (block @ flat.T).reshape(last - first, keep, n_images, keep)
        forward: Int32[ndarray, "block keep n_images"] = np.argmax(similarity, axis=3).astype(np.int32)
        backward: Int32[ndarray, "block n_images keep"] = np.argmax(similarity, axis=1).astype(np.int32)
        forward_by_image: Int32[ndarray, "block n_images keep"] = np.ascontiguousarray(forward.transpose(0, 2, 1))
        round_trip: Int32[ndarray, "block n_images keep"] = np.take_along_axis(backward, forward_by_image, axis=2)
        mutual: Bool[ndarray, "block n_images keep"] = round_trip == own_index[None, None, :]
        best_similarity: Float32[ndarray, "block n_images keep"] = np.ascontiguousarray(
            np.take_along_axis(similarity, forward[:, :, :, None], axis=3)[:, :, :, 0].transpose(0, 2, 1)
        )
        mutual &= best_similarity >= floor
        votes[first:last] = mutual.sum(axis=2, dtype=np.float32)

    scores: SimilarityMatrix = votes / np.float32(keep)
    np.fill_diagonal(scores, np.float32(1.0))
    return scores


# ======================================================================================
# vocabulary tree (docs/spec/bow.md §9.4 Option C)
# ======================================================================================


def _effective_depth(config: RetrievalConfig, n_training_descriptors: int) -> int:
    """Deepest tree the training sample can fill, capped at `config.vocab_depth`.

    A level whose nodes hold fewer than one training descriptor each cannot be clustered
    and produces empty leaves, which cost memory and buy nothing. The bound is therefore
    `branching ** depth <= n_training_descriptors`.

    Args:
        config: The retrieval settings.
        n_training_descriptors: Descriptors the tree will be trained on.

    Returns:
        A depth of at least 1.
    """
    depth: int = 1
    while depth < config.vocab_depth and config.vocab_branching ** (depth + 1) <= n_training_descriptors:
        depth += 1
    return depth


def _training_sample(
    descriptors_by_image: Mapping[int, Descriptors],
    image_ids: tuple[int, ...],
    target: int,
    seed: int,
) -> Float32[ndarray, "n_sample descriptor_dim"]:
    """Draw a seeded subsample of descriptors, spread evenly over the images.

    Drawing a per-image quota rather than a global random subset keeps the vocabulary from
    over-fitting whichever part of the sequence happens to be densest, and makes the sample
    independent of how many descriptors each image carries.

    Args:
        descriptors_by_image: Descriptors per image id.
        image_ids: Image ids in the index's order.
        target: How many descriptors to draw in total.
        seed: Seed for the draw.

    Returns:
        Float32 descriptors with shape `[n_sample, descriptor_dim]`. Every image
        contributes the same quota, so `n_sample` is `target` rounded up to a whole
        number of images; truncating back to `target` would silently drop the tail of
        the sequence from the training set.
    """
    generator: np.random.Generator = np.random.default_rng(seed)
    quota: int = max(1, -(-target // len(image_ids)))
    blocks: list[Descriptors] = []
    for image_id in image_ids:
        descriptors: Descriptors = descriptors_by_image[image_id]
        if len(descriptors) <= quota:
            blocks.append(np.asarray(descriptors, dtype=np.float32))
            continue
        chosen: Int64[ndarray, "quota"] = np.sort(generator.choice(len(descriptors), quota, replace=False))
        blocks.append(np.asarray(descriptors[chosen], dtype=np.float32))
    return np.ascontiguousarray(np.concatenate(blocks, axis=0))


def _train_vocabulary(sample: Float32[ndarray, "n_sample descriptor_dim"], depth: int, config: RetrievalConfig) -> VocabularyTree:
    """Train a hierarchical k-means vocabulary on a descriptor sample.

    One `scipy.cluster.vq.kmeans2` call per node, k-means++ seeded from a per-node draw of
    a seeded generator, so the whole tree is reproducible. The recursion is breadth-first
    over levels rather than depth-first over nodes, which lets the level's descriptors be
    partitioned with one sort instead of one gather per node.

    Nodes holding fewer descriptors than `branching` become their own centroids and leave
    the remaining child slots dead (`-inf` bias); the blob drops empty clusters the same
    way (`docs/spec/bow.md` §6.4).

    Args:
        sample: Float32 training descriptors with shape `[n_sample, descriptor_dim]`.
        depth: Levels to build.
        config: The retrieval settings.

    Returns:
        The trained tree.
    """
    branching: int = config.vocab_branching
    dim: int = int(sample.shape[1])
    generator: np.random.Generator = np.random.default_rng(config.seed)
    centroids_per_level: list[Float32[ndarray, "n_children descriptor_dim"]] = []
    biases_per_level: list[Float32[ndarray, "n_children"]] = []

    node_of_sample: Int64[ndarray, "n_sample"] = np.zeros(len(sample), dtype=np.int64)
    n_nodes: int = 1
    for _level in range(depth):
        centroids: Float32[ndarray, "n_children descriptor_dim"] = np.zeros((n_nodes * branching, dim), dtype=np.float32)
        live: Bool[ndarray, "n_children"] = np.zeros(n_nodes * branching, dtype=bool)
        order: Int64[ndarray, "n_sample"] = np.argsort(node_of_sample, kind="stable")
        bounds: Int64[ndarray, "n_nodes_plus_one"] = np.searchsorted(node_of_sample[order], np.arange(n_nodes + 1))
        child_of_sample: Int64[ndarray, "n_sample"] = np.zeros(len(sample), dtype=np.int64)
        for node in range(n_nodes):
            members: Int64[ndarray, "n_members"] = order[bounds[node] : bounds[node + 1]]
            base: int = node * branching
            if len(members) == 0:
                continue
            block: Float32[ndarray, "n_members descriptor_dim"] = sample[members]
            if len(members) < branching:
                centroids[base : base + len(members)] = block
                live[base : base + len(members)] = True
                child_of_sample[members] = base + np.arange(len(members))
                continue
            # A node whose descriptors are all duplicates of one another makes k-means++
            # divide by a zero squared-distance sum and leaves clusters empty, which scipy
            # reports as a UserWarning and NumPy as a RuntimeWarning. Both are expected
            # here and harmless: the duplicate rows collapse onto one centroid, which is
            # what the blob's own "empty clusters are dropped" does
            # (`docs/spec/bow.md` §6.4). Silenced around this call only.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                book, labels = kmeans2(
                    block,
                    branching,
                    iter=config.vocab_kmeans_iterations,
                    minit="++",
                    seed=int(generator.integers(1 << 31)),
                    check_finite=False,
                )
            centroids[base : base + branching] = np.asarray(book, dtype=np.float32)
            live[base : base + branching] = True
            child_of_sample[members] = base + np.asarray(labels, dtype=np.int64)
        biases: Float32[ndarray, "n_children"] = np.where(
            live, -0.5 * np.sum(centroids * centroids, axis=1), -np.inf
        ).astype(np.float32)
        centroids_per_level.append(centroids)
        biases_per_level.append(biases)
        node_of_sample = child_of_sample
        n_nodes *= branching
    return VocabularyTree(branching=branching, centroids=tuple(centroids_per_level), biases=tuple(biases_per_level))


def _assign_words(descriptors: Float32[ndarray, "n_descriptors descriptor_dim"], tree: VocabularyTree) -> Int32[ndarray, "n_descriptors"]:
    """Greedily descend every descriptor to its leaf word.

    Pure greedy descent, no beam and no backtracking, exactly as
    `VisualVocabulary::SearchFeature` does (`docs/spec/bow.md` §7.1). Nearest centroid by
    squared L2 is `argmax(descriptor @ centroids.T - 0.5 * ||centroid||^2)`, since
    `||descriptor||^2` is the same for every candidate — which is why the descriptors need
    no renormalising here.

    Args:
        descriptors: Float32 descriptors with shape `[n_descriptors, descriptor_dim]`.
        tree: The trained vocabulary.

    Returns:
        Int32 word ids with shape `[n_descriptors]`, in `[0, tree.n_words)`.
    """
    branching: int = tree.branching
    node: Int64[ndarray, "n_descriptors"] = np.zeros(len(descriptors), dtype=np.int64)
    for centroids, biases in zip(tree.centroids, tree.biases, strict=True):
        n_nodes: int = len(centroids) // branching
        order: Int64[ndarray, "n_descriptors"] = np.argsort(node, kind="stable")
        bounds: Int64[ndarray, "n_nodes_plus_one"] = np.searchsorted(node[order], np.arange(n_nodes + 1))
        child: Int64[ndarray, "n_descriptors"] = np.zeros(len(descriptors), dtype=np.int64)
        for parent in range(n_nodes):
            members: Int64[ndarray, "n_members"] = order[bounds[parent] : bounds[parent + 1]]
            if len(members) == 0:
                continue
            base: int = parent * branching
            candidates: Float32[ndarray, "branching descriptor_dim"] = centroids[base : base + branching]
            affinity: Float32[ndarray, "n_members branching"] = descriptors[members] @ candidates.T + biases[base : base + branching]
            child[members] = base + np.argmax(affinity, axis=1)
        node = child
    return node.astype(np.int32)


def _bow_matrix(
    word_of_descriptor: Int32[ndarray, "n_descriptors"],
    image_of_descriptor: Int32[ndarray, "n_descriptors"],
    n_images: int,
    n_words: int,
) -> sparse.csr_matrix:
    """Turn word assignments into L1-normalised TF-IDF rows.

    `docs/spec/bow.md` §7.2: the BoW vector is `v_w = tf_w * idf_w / sum_u (tf_u * idf_u)`
    with raw term counts, and words of zero weight are dropped. The IDF is
    `log(n_images / df_w)`, not the blob's integer-division variant (§6.7, §9.3): the
    quirk's only visible effect is that a word appearing in more than half the images
    collapses to weight 0, and `log(n_images / df)` already decays to 0 as `df` approaches
    `n_images`.

    Args:
        word_of_descriptor: Int32 word id per descriptor, shape `[n_descriptors]`.
        image_of_descriptor: Int32 image row per descriptor, shape `[n_descriptors]`.
        n_images: Rows of the result.
        n_words: Columns of the result.

    Returns:
        A `[n_images, n_words]` CSR matrix whose non-negative rows each sum to 1, except
        rows whose every word was dropped, which are empty.
    """
    counts: sparse.csr_matrix = sparse.coo_matrix(
        (np.ones(len(word_of_descriptor), dtype=np.float32), (image_of_descriptor, word_of_descriptor)),
        shape=(n_images, n_words),
    ).tocsr()
    counts.sum_duplicates()
    document_frequency: Int64[ndarray, "n_words"] = np.asarray((counts > 0).sum(axis=0)).ravel()
    inverse_document_frequency: Float32[ndarray, "n_words"] = np.zeros(n_words, dtype=np.float32)
    seen: Bool[ndarray, "n_words"] = document_frequency > 0
    inverse_document_frequency[seen] = np.log(n_images / document_frequency[seen]).astype(np.float32)

    weighted: sparse.csr_matrix = counts.multiply(inverse_document_frequency[None, :]).tocsr()
    weighted.eliminate_zeros()
    row_sums: Float32[ndarray, "n_images"] = np.asarray(weighted.sum(axis=1)).ravel()
    row_sums[row_sums == 0.0] = 1.0
    return (sparse.diags(1.0 / row_sums) @ weighted).tocsr().astype(np.float32)


def l1_score(query: Mapping[int, float], document: Mapping[int, float]) -> float:
    """The DBoW2 L1 score of two L1-normalised BoW vectors (`docs/spec/bow.md` §9.5).

    `s = -0.5 * sum_{w in q & d} (|q_w - d_w| - |q_w| - |d_w|)`, which lands in `[0, 1]`
    with 1.0 for identical vectors. This is the reference implementation the vectorised
    inverted index in `_all_pairs_l1_score` is checked against.

    Args:
        query: Word id to weight for the query image; weights must be non-negative.
        document: Word id to weight for the candidate image.

    Returns:
        The score in `[0, 1]`.
    """
    shared: set[int] = set(query) & set(document)
    return -0.5 * sum(abs(query[word] - document[word]) - abs(query[word]) - abs(document[word]) for word in shared)


def _all_pairs_l1_score(bow: sparse.csr_matrix) -> SimilarityMatrix:
    """Score every image pair through the inverted index.

    For non-negative weights `-0.5 * (|a - b| - a - b)` is exactly `min(a, b)`, so the
    DBoW2 L1 score of `docs/spec/bow.md` §7.4 is the histogram intersection
    `s(q, d) = sum_w min(q_w, d_w)`. Written that way it needs no per-query pass: walking
    the inverted index once and adding the outer minimum of each word's posting list into
    the score matrix scores every pair at the same time, which is why `query` is free
    afterwards.

    The cost is `sum_w df_w^2`, quadratic in the image count at a fixed vocabulary size —
    the reason `RetrievalConfig.vocab_depth` matters as much as it does. Measured on
    RoboCap's 4528 images: 23.2 s at 100 000 words against 44.2 s at 10 000.

    Args:
        bow: `[n_images, n_words]` CSR matrix of L1-normalised non-negative rows.

    Returns:
        The symmetric score matrix, 1.0 on the diagonal.
    """
    n_images: int = bow.shape[0]
    scores: SimilarityMatrix = np.zeros((n_images, n_images), dtype=np.float32)
    columns: sparse.csc_matrix = bow.tocsc()
    for word in range(bow.shape[1]):
        start: int = int(columns.indptr[word])
        end: int = int(columns.indptr[word + 1])
        if end - start < 2:
            continue
        documents: Int32[ndarray, "df"] = columns.indices[start:end]
        weights: Float32[ndarray, "df"] = columns.data[start:end]
        scores[np.ix_(documents, documents)] += np.minimum(weights[:, None], weights[None, :])
    np.fill_diagonal(scores, np.float32(1.0))
    return np.clip(scores, 0.0, 1.0, out=scores)


def _assignment_batches(counts: Sequence[int], max_descriptors: int) -> list[tuple[int, int]]:
    """Split the images into contiguous runs of roughly `max_descriptors` descriptors.

    Assignment is a pure per-descriptor function, so batching only bounds peak memory: a
    whole RoboCap corpus is 9.3 M descriptors, i.e. 4.7 GB at 128-d float32, and stacking
    it in one array on top of the caller's own copy is what the batches avoid.

    Args:
        counts: Descriptor count per image, in the index's order.
        max_descriptors: Soft upper bound on a batch; one image is never split.

    Returns:
        Half-open `[start, stop)` image ranges covering every image exactly once.
    """
    batches: list[tuple[int, int]] = []
    start: int = 0
    running: int = 0
    for position, count in enumerate(counts):
        running += count
        if running >= max_descriptors:
            batches.append((start, position + 1))
            start = position + 1
            running = 0
    if start < len(counts):
        batches.append((start, len(counts)))
    return batches


def _vocab_similarity(
    descriptors_by_image: Mapping[int, Descriptors],
    image_ids: tuple[int, ...],
    counts: Sequence[int],
    config: RetrievalConfig,
) -> tuple[SimilarityMatrix, int]:
    """Train a vocabulary, encode every image and score every pair.

    Args:
        descriptors_by_image: Descriptors per image id.
        image_ids: Image ids in the index's order.
        counts: Descriptor count per image, in the same order.
        config: The retrieval settings.

    Returns:
        The symmetric score matrix and the vocabulary size.
    """
    n_images: int = len(image_ids)
    sample: Float32[ndarray, "n_sample descriptor_dim"] = _training_sample(
        descriptors_by_image, image_ids, config.vocab_training_descriptors, config.seed
    )
    tree: VocabularyTree = _train_vocabulary(sample, _effective_depth(config, len(sample)), config)
    del sample

    word_batches: list[Int32[ndarray, "n_batch"]] = []
    image_batches: list[Int32[ndarray, "n_batch"]] = []
    for start, stop in _assignment_batches(counts, config.vocab_assign_chunk_descriptors):
        block: Float32[ndarray, "n_batch descriptor_dim"] = np.ascontiguousarray(
            np.concatenate([np.asarray(descriptors_by_image[image_id], dtype=np.float32) for image_id in image_ids[start:stop]], axis=0)
        )
        word_batches.append(_assign_words(block, tree))
        image_batches.append(np.repeat(np.arange(start, stop, dtype=np.int32), counts[start:stop]))

    bow: sparse.csr_matrix = _bow_matrix(np.concatenate(word_batches), np.concatenate(image_batches), n_images, tree.n_words)
    return _all_pairs_l1_score(bow), tree.n_words


# ======================================================================================
# the stage
# ======================================================================================


def build_retrieval_index(
    descriptors_by_image: Mapping[int, Descriptors],
    config: RetrievalConfig | None = None,
    scores_by_image: Mapping[int, DescriptorScores] | None = None,
) -> RetrievalIndex:
    """Score every image pair, with the backend `config.backend` selects.

    Args:
        descriptors_by_image: Local descriptors per image id, `[n_descriptors, dim]` each.
            Images may carry different counts; on the brute-force path the smallest count
            caps the whole index, on the vocab path every descriptor is used.
        config: Backend choice and its settings; the defaults are the measured ones.
        scores_by_image: Optional detector responses per image id, used by the brute-force
            path to keep the strongest descriptors instead of an even stride. The vocab
            path subsamples nothing and ignores it.

    Returns:
        The index, with its wall-clock build time and its resolved backend recorded.

    Raises:
        ValueError: When there are fewer than two images, an image has no descriptors, or
            the descriptor widths disagree.
    """
    settings: RetrievalConfig = RetrievalConfig() if config is None else config
    image_ids: tuple[int, ...] = tuple(sorted(descriptors_by_image))
    n_images: int = len(image_ids)
    if n_images < 2:
        raise ValueError(f"a retrieval index needs at least two images, got {n_images}")
    counts: list[int] = [int(descriptors_by_image[image_id].shape[0]) for image_id in image_ids]
    if min(counts) < 1:
        raise ValueError("every image must carry at least one descriptor")
    backend: ResolvedBackend = settings.resolve_backend(n_images)

    started: float = time.perf_counter()
    if backend == "vocab":
        similarity, n_words = _vocab_similarity(descriptors_by_image, image_ids, counts, settings)
        descriptors_per_image: int = sum(counts) // n_images
    else:
        n_words = 0
        descriptors_per_image = min(settings.max_descriptors_per_image, min(counts), max(1, settings.max_total_descriptors // n_images))
        similarity = _brute_force_similarity(descriptors_by_image, image_ids, descriptors_per_image, scores_by_image, settings)
    build_seconds: float = time.perf_counter() - started

    print(
        f"colsfm.retrieval: {backend} index over {n_images} images, "
        f"{descriptors_per_image} descriptors/image, {n_words} words, {build_seconds:.1f}s"
    )
    return RetrievalIndex(
        image_ids=image_ids,
        similarity=similarity,
        descriptors_per_image=descriptors_per_image,
        config=settings,
        build_seconds=build_seconds,
        backend=backend,
        n_words=n_words,
    )

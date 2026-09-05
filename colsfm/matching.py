"""LightGlue matching plus geometric verification — the replacement for `feature_matcher_main`.

COLMAP 4.2 ships `FeatureMatcherType.ALIKED_LIGHTGLUE` and downloads the ONNX
graph itself into `~/.cache/colmap/`, so the whole stage is a
`pycolmap.match_image_pairs` call over a pair list, with verification running
inside COLMAP. Its graph is not the blob's — see `MatchingOptions.model_path`.

The blob's own pipeline is LightGlue -> score gate -> spatial NMS -> essential
matrix RANSAC (feature_matcher_main.md §5). Three of those four map straight
across; the fourth does not:

| blob | here |
|---|---|
| `mscores0 > match_threshold` (0.3) | `LightGlueONNXMatchingOptions.min_score` |
| SSC NMS down to `match_top_k: 500` | `max_matches_per_pair` — a score-free grid subsample, see below |
| `cv::findEssentialMat`, threshold `0.5*(err/f0 + err/f1)` on bearings | `TwoViewGeometryOptions.ransac.max_error = 4.0` px on a real camera model |
| `n_inliers < 10` empties the pair | `TwoViewGeometryOptions.min_num_inliers = 10` |

**The SSC divergence.** The blob thins each pair's matches to a spatially uniform
top-500 using the LightGlue score as the keypoint response, which is what turns
its ~931 raw matches per Galileo pair into ~430. pycolmap exposes no per-match
score: `Database.read_matches` and `FeatureMatcher.match` both return bare
`uint32[m, 2]` index pairs, so neither the sort key nor SSC's "strongest per
cell" rule exists on this side. What the blob is *buying* with SSC — spatial
uniformity and a bounded track density — is reachable without a score, so
`subsample_matches_by_coverage` lays a fixed grid of about `match_top_k` cells
over image 0 and keeps one match per occupied cell, ties broken by keypoint
index. It runs **after** verification, so every survivor is an inlier of the same
RANSAC; only the count changes.

What binds is the number of *occupied* cells, not the cap: ALIKED keypoints
cluster on texture, so a 1920x1200 Galileo pair fills about 160 of its 476 cells
and the survivors come out well under 500. Measured over the 331 Galileo pairs:

| | mean matches per pair | mean reprojection | ATE vs ground truth |
|---|---|---|---|
| uncapped | 1300 | 1.80 px | 5.39 mm |
| `max_matches_per_pair = 500` | 156 | 1.33 px | 4.32 mm |
| blob (SSC, `match_top_k: 500`) | 430 | 1.55 px | 5.00 mm |

So the cap costs nothing in accuracy — it *buys* it, by refusing to pile a
thousand near-collinear observations from one image region onto one track — and
it takes Galileo's bundle adjustment from 27.7 s to 2.8 s.

**Two phases, not one.** `match_image_pairs` can verify inline, but when a pair
fails verification COLMAP stores neither the geometry *nor the raw matches* —
measured: a pair with 400 LightGlue matches and `min_num_inliers = 1e6` leaves an
empty `matches` row. Reporting "raw matches and inliers per pair", which is what
tells a stale threshold apart from a genuinely textureless pair, therefore needs
matching and verification split: `match_image_pairs` with
`skip_geometric_verification`, then COLMAP's own batch `verify_matches` over the
same pair file. That is also the blob's order. One wrinkle in between: the
skipped-verification pass still writes an `UNDEFINED` two-view geometry row per
pair, and `verify_matches` silently skips any pair that already has one, so the
placeholders are deleted first.

**The score threshold.** The blob's 0.3 is calibrated for its own LightGlue
engine. COLMAP's graph scores differently, and at 0.3 the low-texture
`left_stereo_*` pairs collapse: 22 of 331 Galileo pairs come back empty (6.6 %)
against the blob's 8 (2.4 %). The default here is COLMAP's own 0.1, measured at
13 empty pairs (3.9 %) with a 10th-percentile match count of 116 against the
blob's 118. `BLOB_MATCH_THRESHOLD` is kept for parity runs.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Int64
from numpy import ndarray

from colsfm.database import (
    ImagePair,
    Keypoints,
    KeypointsXY,
    MatchIndices,
    delete_two_view_geometries,
    pair_inlier_counts,
    raw_match_counts,
    read_keypoints_batch,
)
from colsfm.features import DeviceChoice, ResolvedDevice, resolve_device
from colsfm.pairs import write_pair_list

MatcherVariant: TypeAlias = Literal["ALIKED_LIGHTGLUE", "ALIKED_BRUTEFORCE"]
"""The ALIKED matchers COLMAP 4.2 offers; the pipeline uses LightGlue."""

DEFAULT_MIN_SCORE: Final[float] = 0.1
"""COLMAP's own `LightGlueONNXMatchingOptions.min_score`, measured best on Galileo."""

BLOB_MATCH_THRESHOLD: Final[float] = 0.3
"""`lightglue_params.match_threshold` in `matching_task_worker_config.pb.txt`.

Applied by the blob with a strict `>`. Calibrated for the blob's own engine; on
COLMAP's graph it empties 6.6 % of Galileo's pairs, so it is not the default.
"""

BLOB_MIN_NUM_INLIERS: Final[int] = 10
"""`SelectMatchesByEpipolarGeometry` empties a pair below this, rather than trimming it."""

BLOB_MAX_PIXEL_ERROR: Final[float] = 4.0
"""`verification_config.max_pixel_error`; COLMAP takes it in pixels directly."""

BLOB_RANSAC_CONFIDENCE: Final[float] = 0.98
"""`verification_config.min_ransac_confidence`; COLMAP's own default is 0.999."""

BLOB_MATCH_TOP_K: Final[int] = 500
"""`lightglue_params.match_top_k`; the match budget the blob's SSC NMS aims at."""


@dataclass(frozen=True, slots=True)
class MatchingOptions:
    """How LightGlue and the two-view verifier run over a pair list."""

    variant: MatcherVariant = "ALIKED_LIGHTGLUE"
    """Matcher COLMAP runs; LightGlue is what the blob uses."""
    min_score: float = DEFAULT_MIN_SCORE
    """LightGlue score gate; see the module docstring for why this is not the blob's 0.3."""
    model_path: Path | None = None
    """Explicit LightGlue ONNX file; None lets COLMAP download and cache its own.

    Not a slot for the blob's `pycusfm/models/aliked_lightglue/lightglue_aliked.onnx`:
    that graph takes four inputs (`kpts0/1`, `desc0/1`) and returns `matches0` as
    an int32 `[num_matches, 2]` index matrix, while COLMAP's own
    `aliked-lightglue.onnx` takes six (adding `image_size0/1`, so normalisation
    happens inside the graph) and returns an int64 `[num_keypoints0]` assignment
    vector. `CreateLightGlueONNXFeatureMatcher` rejects the mismatch by throwing
    inside a worker thread, which aborts the process.
    """
    max_num_matches: int = 32768
    """COLMAP's cap on matches per pair; never binds at 2048 keypoints per image."""
    min_num_inliers: int = BLOB_MIN_NUM_INLIERS
    """Inlier count below which the pair is emptied, matching the blob's rule."""
    max_error_px: float = BLOB_MAX_PIXEL_ERROR
    """RANSAC inlier threshold in pixels."""
    confidence: float = BLOB_RANSAC_CONFIDENCE
    """RANSAC success probability driving the trial count."""
    skip_image_pairs_in_same_frame: bool = False
    """Must stay False: the stereo pairs of one rig frame are exactly what cuSFM matches."""
    device: DeviceChoice = "auto"
    """Compute device; `auto` falls back to the CPU provider when cuDNN is missing."""
    gpu_index: str = "-1"
    """CUDA device index as COLMAP's comma-separated string."""
    num_threads: int = -1
    """COLMAP matching threads; -1 lets COLMAP choose."""
    max_matches_per_pair: int | None = BLOB_MATCH_TOP_K
    """Verified matches to keep per pair, spread over the image; None keeps every inlier.

    The score-free stand-in for the blob's SSC spatial NMS — see
    `subsample_matches_by_coverage` and the module docstring."""


DEFAULT_MATCHING_OPTIONS: Final[MatchingOptions] = MatchingOptions()
"""Shared immutable default, so the signatures below hold no constructor call."""


@dataclass(frozen=True, slots=True)
class PairMatchStats:
    """What one image pair produced."""

    raw_matches: int
    """Matches LightGlue kept after the score gate, before verification."""
    inlier_matches: int
    """Matches surviving two-view verification; 0 when the pair was emptied."""

    @property
    def retention(self) -> float:
        """Inliers as a fraction of raw matches.

        Returns:
            The ratio, or 0.0 when the pair had no raw matches. The blob's median
            on Galileo is 0.985 — verification is a light filter, not a gate.
        """
        return 0.0 if self.raw_matches == 0 else self.inlier_matches / self.raw_matches


@dataclass(frozen=True, slots=True)
class MatchReport:
    """What one matching run produced, per pair and in aggregate."""

    pair_stats: dict[ImagePair, PairMatchStats] = field(default_factory=dict)
    """Raw and inlier counts keyed by the pair as given."""
    device: ResolvedDevice = "cpu"
    """The device the run actually used."""
    elapsed_seconds: float = 0.0
    """Wall time of both phases: LightGlue matching and two-view verification."""

    @property
    def empty_pairs(self) -> int:
        """How many pairs came back with no verified match.

        Returns:
            The count of pairs whose inlier list is empty.
        """
        return sum(1 for stats in self.pair_stats.values() if stats.inlier_matches == 0)

    @property
    def median_inliers(self) -> float:
        """Median verified match count over all pairs.

        Returns:
            The median, or 0.0 when nothing was matched.
        """
        if not self.pair_stats:
            return 0.0
        return float(median(stats.inlier_matches for stats in self.pair_stats.values()))

    @property
    def median_retention(self) -> float:
        """Median inlier fraction over the pairs that produced any raw match.

        Returns:
            The median retention, or 0.0 when no pair produced a raw match.
        """
        retentions: list[float] = [stats.retention for stats in self.pair_stats.values() if stats.raw_matches > 0]
        return float(median(retentions)) if retentions else 0.0

    def summary(self) -> str:
        """One line naming the pair count, the inlier median and the wall time.

        Returns:
            A printable summary; `"no pairs"` when nothing was matched.
        """
        if not self.pair_stats:
            return f"matched no pairs on {self.device} in {self.elapsed_seconds:.2f}s"
        empty_fraction: float = 100.0 * self.empty_pairs / len(self.pair_stats)
        return (
            f"matched {len(self.pair_stats)} pairs on {self.device} in {self.elapsed_seconds:.2f}s, "
            f"median inliers {self.median_inliers:.0f}, retention {self.median_retention:.3f}, "
            f"{self.empty_pairs} empty ({empty_fraction:.1f}%)"
        )


def _matching_options(options: MatchingOptions, device: ResolvedDevice) -> pycolmap.FeatureMatchingOptions:
    """Build the pycolmap matcher options.

    Args:
        options: The typed settings.
        device: The already-resolved device.

    Returns:
        `FeatureMatchingOptions` with the ALIKED LightGlue sub-options filled in.
    """
    matching_options: pycolmap.FeatureMatchingOptions = pycolmap.FeatureMatchingOptions()
    matching_options.type = getattr(pycolmap.FeatureMatcherType, options.variant)
    matching_options.use_gpu = device == "cuda"
    matching_options.gpu_index = options.gpu_index
    matching_options.num_threads = options.num_threads
    matching_options.max_num_matches = options.max_num_matches
    matching_options.skip_image_pairs_in_same_frame = options.skip_image_pairs_in_same_frame
    matching_options.aliked.lightglue.min_score = options.min_score
    if options.model_path is not None:
        matching_options.aliked.lightglue.model_path = str(options.model_path)
    return matching_options


def verification_options(options: MatchingOptions) -> pycolmap.TwoViewGeometryOptions:
    """Build the two-view geometry options that mirror cuSFM's verifier.

    COLMAP works in pixels against the real camera model, so the blob's
    `MapErrorToNormalizedPlane` conversion (`0.5*(err/f0 + err/f1)` on bearing
    vectors) is not needed: `max_error` takes the 4.0 px straight from the
    config.

    Args:
        options: The typed settings.

    Returns:
        `TwoViewGeometryOptions` carrying the blob's thresholds.
    """
    geometry_options: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    geometry_options.min_num_inliers = options.min_num_inliers
    geometry_options.ransac.max_error = options.max_error_px
    geometry_options.ransac.confidence = options.confidence
    return geometry_options


def subsample_matches_by_coverage(
    keypoints_xy: KeypointsXY,
    matches: MatchIndices,
    image_width: int,
    image_height: int,
    max_matches: int,
) -> MatchIndices:
    """Thin a pair's matches to one per grid cell, so the survivors cover the image.

    The score-free replacement for the blob's SSC non-maximum suppression
    (feature_matcher_main.md §5.3). SSC sorts the matched keypoints by their
    LightGlue score and binary-searches a suppression radius until roughly
    `match_top_k` survive; pycolmap exposes no per-match score, so neither the
    sort key nor the "strongest per cell" rule is reachable. What *is* reachable
    is the property the blob is buying with it — spatial uniformity — so this
    lays a fixed grid over image 0 and keeps one match per occupied cell.

    The grid is `num_columns = floor(sqrt(max_matches * width / height))` by
    `num_rows = floor(max_matches / num_columns)`, i.e. cells of roughly
    `sqrt(width * height / max_matches)` on a side, rounded so the cell count
    never exceeds `max_matches`. That makes the cap a real maximum rather than
    the blob's plus-or-minus 30 % target band.

    Ties inside a cell are broken by the image-0 keypoint index, which makes the
    result a pure function of the inputs; nothing here depends on match order or
    on a random seed.

    Args:
        keypoints_xy: Float32 `[num_keypoints, 2]` xy pixel coordinates of the
            pair's **first** image, as `colsfm.database.read_keypoints` returns them.
        matches: Int64 `[num_matches, 2]` keypoint index pairs, first image first.
        image_width: Width of the first image in pixels.
        image_height: Height of the first image in pixels.
        max_matches: Target match count, i.e. roughly the cell count.

    Returns:
        Int64 `[num_kept, 2]` subset of `matches`, in ascending image-0 keypoint
        index order. `matches` is returned unchanged when it already holds
        `max_matches` rows or fewer, when the cap is not positive, or when the
        image has no area.
    """
    if len(matches) <= max_matches or max_matches <= 0 or image_width <= 0 or image_height <= 0:
        return np.asarray(matches, dtype=np.int64)

    order: Int64[ndarray, "num_matches"] = np.lexsort((matches[:, 1], matches[:, 0]))
    ordered: MatchIndices = np.asarray(matches, dtype=np.int64)[order]
    positions_xy: KeypointsXY = keypoints_xy[ordered[:, 0]]
    num_columns: int = max(int(math.sqrt(max_matches * image_width / image_height)), 1)
    num_rows: int = max(max_matches // num_columns, 1)
    columns: Int64[ndarray, "num_matches"] = np.clip(
        (positions_xy[:, 0] * num_columns / image_width).astype(np.int64), 0, num_columns - 1
    )
    rows: Int64[ndarray, "num_matches"] = np.clip(
        (positions_xy[:, 1] * num_rows / image_height).astype(np.int64), 0, num_rows - 1
    )
    cells: Int64[ndarray, "num_matches"] = rows * num_columns + columns
    # `np.unique` returns the first index of each value because `cells` is scanned in
    # the sorted keypoint order above, so the winner per cell is the lowest index.
    _, first_in_cell = np.unique(cells, return_index=True)
    return ordered[np.sort(first_in_cell)]


def _image_sizes(database: pycolmap.Database, image_ids: Sequence[int]) -> dict[int, tuple[int, int]]:
    """Look up the pixel size of each image through its camera.

    Args:
        database: An open database.
        image_ids: Images to size.

    Returns:
        `(width, height)` per `image_id`.
    """
    cameras: dict[int, pycolmap.Camera] = {camera.camera_id: camera for camera in database.read_all_cameras()}
    images: dict[int, pycolmap.Image] = {image.image_id: image for image in database.read_all_images()}
    sizes: dict[int, tuple[int, int]] = {}
    for image_id in image_ids:
        camera: pycolmap.Camera = cameras[images[image_id].camera_id]
        sizes[image_id] = (int(camera.width), int(camera.height))
    return sizes


def cap_verified_matches(database_path: Path, pairs: Sequence[ImagePair], max_matches_per_pair: int) -> int:
    """Rewrite every over-full two-view geometry with a spatially spread subset.

    Runs **after** verification, so the survivors are inliers of the same
    essential-matrix RANSAC the blob verifies with; only the count changes. The
    geometry's `E`/`F`/`H`, its configuration and its triangulation angle are kept
    as verification estimated them.

    Args:
        database_path: An existing COLMAP database holding verified geometries.
        pairs: Pairs to cap, as `(min(image_id), max(image_id))`.
        max_matches_per_pair: Matches to keep per pair.

    Returns:
        Total inlier matches removed across every pair.
    """
    removed: int = 0
    first_image_ids: list[int] = sorted({pair[0] for pair in pairs})
    # `read_keypoints_batch` owns the "leading xy columns" rule; spelling it a second time
    # here is how the two readers drifted to different dtypes in the first place.
    keypoints_by_image_id: dict[int, Keypoints] = read_keypoints_batch(database_path, first_image_ids)
    with pycolmap.Database.open(database_path) as database:
        sizes: dict[int, tuple[int, int]] = _image_sizes(database, first_image_ids)
        for image_id1, image_id2 in pairs:
            geometry: pycolmap.TwoViewGeometry = database.read_two_view_geometry(image_id1, image_id2)
            inliers: MatchIndices = np.asarray(geometry.inlier_matches, dtype=np.int64)
            if len(inliers) <= max_matches_per_pair:
                continue
            width, height = sizes[image_id1]
            kept: MatchIndices = subsample_matches_by_coverage(
                keypoints_by_image_id[image_id1], inliers, width, height, max_matches_per_pair
            )
            removed += len(inliers) - len(kept)
            geometry.inlier_matches = kept.astype(np.uint32)
            database.update_two_view_geometry(image_id1, image_id2, geometry)
    return removed


def match_pairs(
    database_path: Path,
    pairs: Sequence[ImagePair],
    options: MatchingOptions = DEFAULT_MATCHING_OPTIONS,
) -> MatchReport:
    """Match and verify a list of image pairs, writing the results into the database.

    Both images of every pair must already carry keypoints and descriptors (see
    `colsfm.features.extract_features`). Matching and verification both run
    inside COLMAP, in two phases so that the raw match count survives a pair that
    fails verification; the database ends up with both tables filled.

    Args:
        database_path: An existing COLMAP database.
        pairs: Image-id pairs, normalised and deduplicated by `colsfm.pairs`.
        options: Matching and verification settings.

    Returns:
        Per-pair raw and inlier counts, the device used and the wall time.

    Raises:
        FileNotFoundError: When the database is missing.
        KeyError: When a pair names an image the database does not hold.
        RuntimeError: When `options.device` is `cuda` and CUDA is unusable.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    if not pairs:
        return MatchReport(device=resolve_device(options.device))

    device: ResolvedDevice = resolve_device(options.device)
    with tempfile.TemporaryDirectory(prefix="colsfm-pairs-") as scratch:
        pair_list_path: Path = Path(scratch) / "pairs.txt"
        write_pair_list(pair_list_path, pairs, database_path)
        pairing_options: pycolmap.ImportedPairingOptions = pycolmap.ImportedPairingOptions()
        pairing_options.match_list_path = pair_list_path
        matching_options: pycolmap.FeatureMatchingOptions = _matching_options(options, device)
        matching_options.skip_geometric_verification = True
        geometry_options: pycolmap.TwoViewGeometryOptions = verification_options(options)
        started: float = time.perf_counter()
        pycolmap.match_image_pairs(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            verification_options=geometry_options,
            device=pycolmap.Device.cuda if device == "cuda" else pycolmap.Device.cpu,
        )
        raw: dict[ImagePair, int] = raw_match_counts(database_path, pairs)
        delete_two_view_geometries(database_path, pairs)
        pycolmap.verify_matches(database_path, pair_list_path, geometry_options)
        if options.max_matches_per_pair is not None:
            dropped: int = cap_verified_matches(database_path, pairs, options.max_matches_per_pair)
            print(f"colsfm.matching: spatial cap at {options.max_matches_per_pair}/pair dropped {dropped} inlier matches")
        elapsed_seconds: float = time.perf_counter() - started

    inliers: dict[ImagePair, int] = pair_inlier_counts(database_path, pairs)
    report: MatchReport = MatchReport(
        pair_stats={pair: PairMatchStats(raw_matches=raw[pair], inlier_matches=inliers[pair]) for pair in pairs},
        device=device,
        elapsed_seconds=elapsed_seconds,
    )
    print(f"colsfm.matching: {report.summary()}")
    return report

"""The `raco` feature and matching backends: batched RaCo-ALIKED and LightGlue+.

Same skip discipline as `tests/colsfm/test_tensorrt_backends.py`, plus one more
thing to be missing: the two RaCo ONNX graphs live under `data/cusfm_models/`,
which is gitignored, so a fresh clone has the code and not the weights. Rebuild
them with `pixi run -e raco raco-export` and
`pixi run -e raco raco-export --batched-extractor-path
data/cusfm_models/raco-aliked-b1-16.onnx`.

The bands are the blob's own Galileo statistics again (feature_matcher_main.md
§6): 2048 keypoints per keyframe, median 430 verified matches per pair. RaCo is
a *different* extractor, so this asserts that it lands in the same neighbourhood
rather than on the same number — that is the "near-parity" claim, and a band is
what near-parity means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

from colsfm.database import ImagePair, KeypointsXY, create_database, read_descriptors_from_database, read_keypoints
from colsfm.features import BLOB_MAX_KEYPOINTS, ExtractionReport, FeatureOptions, extract_features
from colsfm.frames_meta import CameraParams, FramesMeta, KeyframeMeta
from colsfm.matching import MatchingOptions, MatchReport, match_pairs
from colsfm.pairs import select_pairs
from colsfm.pipeline import PipelineOptions, PipelineSummary, run_pipeline

pytest.importorskip("tensorrt", reason="the raco backend needs the tensorrt package")
pytest.importorskip("cuda.bindings.runtime", reason="the raco backend allocates through cuda-python")

BLOB_DESCRIPTOR_DIM: int = 128
"""RaCo-ALIKED's descriptor width, and LightGlue+'s `desc0`/`desc1` binding width."""

MIN_MEDIAN_INLIERS: int = 250
"""Lower band on the median verified matches per pair; the blob's Galileo median is 430."""

MAX_MEDIAN_INLIERS: int = 650
"""Upper band on the same, i.e. SSC's target plus its 30 % tolerance."""

SMOKE_MIN_INTER_FRAME_DISTANCE_M: float = 0.5
"""`feature_extractor_main`'s own default gate; keeps 34 of Galileo's 226 keyframes.

The same subset `tests/colsfm/test_pipeline.py` and the TensorRT smoke run on."""

SMOKE_MIN_REGISTERED_IMAGES: int = 30
"""Of the 34 selected keyframes, at least this many must end up registered."""

SMOKE_MIN_MEAN_MATCHES: int = 40
"""Lower band on the mean verified matches per pair over the 34-frame selection."""

SMOKE_MAX_MEAN_MATCHES: int = 650
"""Upper band on the same: SSC's target plus its tolerance."""


def _raco_models_available() -> bool:
    """Whether both RaCo graphs and a usable CUDA device are present.

    Returns:
        True when the extractor and matcher graphs exist and
        `cudaGetDeviceProperties` answers, so an engine can be loaded or built.
    """
    from colsfm.features_raco import RACO_ONNX_PATH
    from colsfm.matching_raco import RACO_LIGHTGLUE_ONNX_PATH
    from colsfm.tensorrt_runtime import compute_capability

    if not RACO_ONNX_PATH.is_file() or not RACO_LIGHTGLUE_ONNX_PATH.is_file():
        return False
    try:
        compute_capability()
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _raco_models_available(), reason="the RaCo ONNX graphs or a CUDA device are missing"
)


def test_the_matcher_frame_is_the_exact_inverse_of_the_extractor_mapping() -> None:
    """`normalize_keypoints_raco` undoes `normalized_to_pixels` to float32 precision.

    This is the whole contract between the two RaCo modules: the extractor emits
    normalised coordinates, writes pixels, and the matcher has to hand the graph
    back exactly what the extractor produced. A mismatch here does not raise
    anywhere — it just moves every keypoint by a fraction of the image and quietly
    costs matches — so it is asserted directly rather than through match counts.
    """
    from colsfm.features_trt import normalized_to_pixels
    from colsfm.matching_raco import normalize_keypoints_raco

    generator: np.random.Generator = np.random.default_rng(0)
    normalized_xy: Float32[ndarray, "num_keypoints 2"] = generator.uniform(-1.0, 1.0, size=(512, 2)).astype(np.float32)
    pixels_xy: KeypointsXY = normalized_to_pixels(normalized_xy, 1920, 1200)
    recovered: Float32[ndarray, "1 num_keypoints 2"] = normalize_keypoints_raco(pixels_xy, 1920, 1200)
    assert recovered.shape == (1, 512, 2)
    assert np.allclose(recovered[0], normalized_xy, atol=1e-6)


def test_the_blob_normalisation_would_not_have_been_the_inverse() -> None:
    """The two graphs really do want different frames, so the split is not cosmetic.

    Guards the reason `KeypointNormalizer` exists: on a 1920x1200 image the blob's
    normalisation agrees with RaCo's in x and disagrees in y by `H / W`. If a
    refactor ever collapsed the two, this is what would fail.
    """
    from colsfm.matching_raco import normalize_keypoints_raco
    from colsfm.matching_trt import normalize_keypoints

    keypoints_xy: KeypointsXY = np.asarray([[0.0, 0.0], [1919.0, 1199.0], [960.0, 600.0]], dtype=np.float32)
    blob: Float32[ndarray, "1 3 2"] = normalize_keypoints(keypoints_xy, 1920, 1200)
    raco: Float32[ndarray, "1 3 2"] = normalize_keypoints_raco(keypoints_xy, 1920, 1200)
    assert np.allclose(blob[0, :, 0], raco[0, :, 0], atol=2e-3)
    assert not np.allclose(blob[0, :, 1], raco[0, :, 1], atol=1e-2)


def test_the_batch_size_is_checked_against_the_graphs_profile(tmp_path: Path, galileo_input: FramesMeta) -> None:
    """A batch outside 1..16 is refused before an engine is touched.

    The optimisation profile is built for that range; TensorRT would otherwise
    reject the shape several seconds later, after an engine load.
    """
    from colsfm.features_raco import extract_raco

    database_path: Path = tmp_path / "batch.db"
    create_database(database_path, galileo_input.filtered(list(galileo_input.rig_frames()[0].keyframe_ids)))
    with pytest.raises(ValueError, match="batch_size"):
        extract_raco(database_path, tmp_path, [], min_score=0.0, max_num_features=BLOB_MAX_KEYPOINTS, batch_size=32)


@pytest.fixture(scope="module")
def raco_stereo_frame(
    galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, list[ImagePair], ExtractionReport, MatchReport]:
    """Extract and match one Galileo rig frame through both RaCo backends.

    Args:
        galileo_input: The 226-keyframe Galileo input collection.
        galileo_input_dir: Where its images live.
        tmp_path_factory: pytest's per-module temporary directory factory.

    Returns:
        The database, the pairs matched, and both reports.
    """
    keyframe_ids: list[int] = list(galileo_input.rig_frames()[0].keyframe_ids)
    subset: FramesMeta = galileo_input.filtered(keyframe_ids)
    database_path: Path = tmp_path_factory.mktemp("raco") / "galileo.db"
    create_database(database_path, subset)
    extraction: ExtractionReport = extract_features(
        database_path,
        galileo_input_dir,
        [keyframe.image_name for keyframe in subset.keyframes],
        FeatureOptions(backend="raco"),
    )
    pairs: list[ImagePair] = select_pairs(subset, connected_keyframe_num=1)
    matching: MatchReport = match_pairs(database_path, pairs, MatchingOptions(backend="raco"))
    return database_path, pairs, extraction, matching


def test_the_batched_graph_stores_2048_keypoints_per_image(
    raco_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
) -> None:
    """RaCo's top-k head is fixed at 2048 and the batching does not lose the tail.

    A batch that mis-slices its outputs would show up as a short last image or as
    every image holding the first image's points; the count is the cheap half of
    that check and `test_the_keypoints_land_inside_the_original_image` the rest.
    """
    _database_path, _pairs, extraction, _matching = raco_stereo_frame
    counts: list[int] = list(extraction.keypoint_counts.values())
    assert counts
    assert all(count == BLOB_MAX_KEYPOINTS for count in counts), counts
    assert extraction.device == "cuda"
    assert extraction.elapsed_seconds > 0.0


def test_the_keypoints_land_inside_the_original_image(
    raco_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
    galileo_input: FramesMeta,
) -> None:
    """The stretch to 1920x1200 is undone against each camera's own size."""
    database_path, _pairs, extraction, _matching = raco_stereo_frame
    keyframe_by_id: dict[int, KeyframeMeta] = galileo_input.keyframe_by_id()
    for image_id in extraction.keypoint_counts:
        keypoints_xy: KeypointsXY = read_keypoints(database_path, image_id)
        camera: CameraParams = galileo_input.cameras[keyframe_by_id[image_id].camera_params_id]
        assert keypoints_xy.dtype == np.float32
        assert keypoints_xy.shape[1] == 2
        assert keypoints_xy.min() >= 0.0
        assert keypoints_xy[:, 0].max() <= camera.image_width
        assert keypoints_xy[:, 1].max() <= camera.image_height


def test_the_descriptors_round_trip_as_l2_normalised_floats(
    raco_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
) -> None:
    """RaCo's descriptors are 128 L2-normalised float32 columns, unchanged by the database."""
    database_path, _pairs, extraction, _matching = raco_stereo_frame
    descriptors = read_descriptors_from_database(database_path, list(extraction.keypoint_counts))
    assert set(descriptors) == set(extraction.keypoint_counts)
    for image_id, block in descriptors.items():
        assert block.dtype == np.float32
        assert block.shape == (extraction.keypoint_counts[image_id], BLOB_DESCRIPTOR_DIM)
        norms = np.linalg.norm(block, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-2), f"image {image_id}: norms {norms.min()}-{norms.max()}"


def test_lightglue_plus_matches_in_the_blobs_band(
    raco_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
) -> None:
    """Median verified matches land near the blob's 430, on a different extractor.

    This is the near-parity claim: RaCo-ALIKED plus LightGlue+ is not the blob's
    ALIKED plus the blob's LightGlue, so the number is allowed to move — but if
    the descriptors, the normalised frame or the score gate were wrong it would
    not move, it would collapse.
    """
    _database_path, pairs, _extraction, matching = raco_stereo_frame
    assert set(matching.pair_stats) == set(pairs)
    assert matching.device == "cuda"
    assert matching.empty_pairs == 0
    assert MIN_MEDIAN_INLIERS <= matching.median_inliers <= MAX_MEDIAN_INLIERS
    assert matching.median_retention >= 0.9


def test_the_whole_pipeline_runs_on_the_raco_backend(
    galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The 34-frame Galileo selection registers and matches in the same band as the others.

    Deliberately the same assertions and the same bands as
    `test_tensorrt_backends.test_the_whole_pipeline_runs_on_either_backend`, so
    the three backends are compared rather than each graded on its own curve.
    """
    options: PipelineOptions = PipelineOptions(
        input_dir=galileo_input_dir,
        output_dir=tmp_path_factory.mktemp("galileo_raco"),
        min_inter_frame_distance=SMOKE_MIN_INTER_FRAME_DISTANCE_M,
        features_backend="raco",
        matching_backend="raco",
    )
    summary: PipelineSummary = run_pipeline(options)
    mean_matches: float = summary.mapping.num_observations / max(summary.num_pairs, 1)
    print(
        f"[colsfm] raco end to end | registered {summary.mapping.num_registered_images} "
        f"| points {summary.mapping.num_points3D} | {mean_matches:.0f} observations per pair "
        f"| {summary.mapping.mean_reprojection_error_px:.3f} px"
    )
    assert summary.mapping.num_registered_images >= SMOKE_MIN_REGISTERED_IMAGES
    assert summary.mapping.num_points3D > 0
    assert SMOKE_MIN_MEAN_MATCHES <= mean_matches <= SMOKE_MAX_MEAN_MATCHES

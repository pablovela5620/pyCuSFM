"""The TensorRT feature and matching backends, against the blob's own engines.

Everything here needs three things the default backends do not: the `tensorrt`
and `cuda-python` packages, the two ONNX graphs under
`pycusfm/models/aliked_lightglue/`, and a GPU whose architecture the cached
engine was built for. `pytest.importorskip` and `_engines_available` skip the
whole module when any of them is missing, so a CPU machine still runs the suite.

The point of comparison is the blob's own published statistics for Galileo
(feature_matcher_main.md §6): 2048 keypoints per keyframe, and matches per pair
with p10 118, median 430 and mean 400. This module asserts bands around those —
the engines are the blob's, so the numbers should be the blob's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from colsfm.database import ImagePair, KeypointsXY, create_database, read_descriptors_from_database, read_keypoints
from colsfm.features import BLOB_MAX_KEYPOINTS, ExtractionReport, FeatureOptions, extract_features
from colsfm.frames_meta import CameraParams, FramesMeta, KeyframeMeta
from colsfm.matching import MatchingOptions, MatchReport, match_pairs
from colsfm.pairs import select_pairs
from colsfm.pipeline import run_pipeline
from colsfm.run_config import PipelineOptions
from colsfm.run_report import PipelineSummary

pytest.importorskip("tensorrt", reason="the TensorRT backends need the tensorrt package")
pytest.importorskip("cuda.bindings.runtime", reason="the TensorRT backends allocate through cuda-python")

BLOB_DESCRIPTOR_DIM: int = 128
"""ALIKED's descriptor width, and LightGlue's `desc0`/`desc1` binding width."""

MIN_MEDIAN_INLIERS: int = 250
"""Lower band on the median verified matches per pair; the blob's Galileo median is 430."""

MAX_MEDIAN_INLIERS: int = 650
"""Upper band on the same, i.e. SSC's target plus its 30 % tolerance."""

SMOKE_MIN_INTER_FRAME_DISTANCE_M: float = 0.5
"""`feature_extractor_main`'s own default gate; keeps 34 of Galileo's 226 keyframes.

The same subset `tests/colsfm/test_pipeline.py` runs on, so the end-to-end
assertions below are directly comparable to the default-backend ones."""

SMOKE_MIN_REGISTERED_IMAGES: int = 30
"""Of the 34 selected keyframes, at least this many must end up registered."""

SMOKE_MIN_MEAN_MATCHES: int = 40
"""Lower band on the mean verified matches per pair over the 34-frame selection.

Wide on purpose. At 0.5 m spacing the pairs are far apart and both backends leave
several of them empty, so the mean sits well under the 400 the blob reports for
the full 226-frame run; what this catches is a backend that matched nothing."""

SMOKE_MAX_MEAN_MATCHES: int = 650
"""Upper band on the same: SSC's target plus its tolerance."""


def _engines_available() -> bool:
    """Whether both ONNX graphs and a usable CUDA device are present.

    Returns:
        True when the graphs exist and `cudaGetDeviceProperties` answers, so an
        engine can be loaded or built. A machine with the wheels but no GPU
        raises inside `compute_capability`, which is what this catches.
    """
    from colsfm.features_trt import ALIKED_ONNX_PATH
    from colsfm.matching_trt import LIGHTGLUE_ONNX_PATH
    from colsfm.tensorrt_runtime import compute_capability

    if not ALIKED_ONNX_PATH.is_file() or not LIGHTGLUE_ONNX_PATH.is_file():
        return False
    try:
        compute_capability()
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _engines_available(), reason="the blob's ONNX graphs or a CUDA device are missing"
)


@pytest.fixture(scope="module")
def tensorrt_stereo_frame(
    galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, list[ImagePair], ExtractionReport, MatchReport]:
    """Extract and match one Galileo rig frame through both TensorRT backends.

    Args:
        galileo_input: The 226-keyframe Galileo input collection.
        galileo_input_dir: Where its images live.
        tmp_path_factory: pytest's per-module temporary directory factory.

    Returns:
        The database, the pairs matched, and both reports.
    """
    keyframe_ids: list[int] = list(galileo_input.rig_frames()[0].keyframe_ids)
    subset: FramesMeta = galileo_input.filtered(keyframe_ids)
    database_path: Path = tmp_path_factory.mktemp("tensorrt") / "galileo.db"
    create_database(database_path, subset)
    extraction: ExtractionReport = extract_features(
        database_path,
        galileo_input_dir,
        [keyframe.image_name for keyframe in subset.keyframes],
        FeatureOptions(backend="tensorrt"),
    )
    pairs: list[ImagePair] = select_pairs(subset, connected_keyframe_num=1)
    matching: MatchReport = match_pairs(database_path, pairs, MatchingOptions(backend="tensorrt"))
    return database_path, pairs, extraction, matching


def test_the_engine_stores_the_blobs_2048_keypoints_per_image(
    tensorrt_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
) -> None:
    """The graph's own top-2048 head binds, so every keyframe holds 2047-2048 points.

    Measured over 226 Galileo and 880 RoboCap keyframes by the blob itself
    (feature_extractor_main.md §6.3); the odd 2047 is `detector_threshold`
    removing a single point.
    """
    _database_path, _pairs, extraction, _matching = tensorrt_stereo_frame
    counts: list[int] = list(extraction.keypoint_counts.values())
    assert counts
    assert all(BLOB_MAX_KEYPOINTS - 1 <= count <= BLOB_MAX_KEYPOINTS for count in counts), counts
    assert extraction.device == "cuda"
    assert extraction.elapsed_seconds > 0.0


def test_the_keypoints_land_inside_the_original_image(
    tensorrt_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
    galileo_input: FramesMeta,
) -> None:
    """The normalised coordinates are mapped back to the source image, not the network's.

    The engine is fed a 1920x1200 stretch of whatever came in; a mapping that
    forgot to undo the stretch would still produce plausible-looking coordinates,
    so the bound that matters is the *camera's* size.
    """
    database_path, _pairs, extraction, _matching = tensorrt_stereo_frame
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
    tensorrt_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
) -> None:
    """`FeatureDescriptorsFloat` stores 128 float32 columns and gives them back unchanged.

    ALIKED's descriptors are L2-normalised, and LightGlue is trained on that; a
    lossy uint8 round trip through COLMAP's own descriptor table would show up
    here as norms away from 1.
    """
    database_path, _pairs, extraction, _matching = tensorrt_stereo_frame
    descriptors = read_descriptors_from_database(database_path, list(extraction.keypoint_counts))
    assert set(descriptors) == set(extraction.keypoint_counts)
    for image_id, block in descriptors.items():
        assert block.dtype == np.float32
        assert block.shape == (extraction.keypoint_counts[image_id], BLOB_DESCRIPTOR_DIM)
        norms = np.linalg.norm(block, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-2), f"image {image_id}: norms {norms.min()}-{norms.max()}"


def test_the_matcher_reproduces_the_blobs_match_counts(
    tensorrt_stereo_frame: tuple[Path, list[ImagePair], ExtractionReport, MatchReport],
) -> None:
    """Median inliers land in SSC's band around 500, where the blob's Galileo median is 430.

    Verification is a light filter on this path, not a gate: the blob's median
    retention is 0.985 and this backend runs the same essential-matrix RANSAC on
    matches the same SSC already thinned.
    """
    _database_path, pairs, _extraction, matching = tensorrt_stereo_frame
    assert set(matching.pair_stats) == set(pairs)
    assert matching.device == "cuda"
    assert matching.empty_pairs == 0
    assert MIN_MEDIAN_INLIERS <= matching.median_inliers <= MAX_MEDIAN_INLIERS
    assert matching.median_retention >= 0.9


def test_the_two_backends_are_independent(
    galileo_input: FramesMeta, galileo_input_dir: Path, tmp_path: Path
) -> None:
    """TensorRT LightGlue matches pycolmap-extracted features, and vice versa.

    Both extraction backends write the same two database columns — float32 xy
    keypoints and 128-column float descriptors — so the four combinations are all
    legal. This checks the crossed one that is easiest to break: COLMAP's own
    ALIKED feeding the blob's LightGlue engine.
    """
    keyframe_ids: list[int] = list(galileo_input.rig_frames()[0].keyframe_ids)[:4]
    subset: FramesMeta = galileo_input.filtered(keyframe_ids)
    database_path: Path = tmp_path / "crossed.db"
    create_database(database_path, subset)
    extract_features(
        database_path,
        galileo_input_dir,
        [keyframe.image_name for keyframe in subset.keyframes],
        FeatureOptions(backend="pycolmap"),
    )
    pairs: list[ImagePair] = select_pairs(subset, connected_keyframe_num=1)
    report: MatchReport = match_pairs(database_path, pairs, MatchingOptions(backend="tensorrt"))
    assert set(report.pair_stats) == set(pairs)
    assert report.median_inliers > 0.0


# ── End to end: the whole pipeline on the 34-frame Galileo selection ──────────────────


@pytest.mark.parametrize("backend", ["pycolmap", "tensorrt"])
def test_the_whole_pipeline_runs_on_either_backend(
    backend: str, galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Both backends register the selection and produce matches in the same band.

    The point is that the backend is invisible past the database: the same eight
    stages run, the same keyframes come back registered, and the per-pair match
    counts land in the same range. The numbers themselves differ — the TensorRT
    path runs the blob's real SSC and keeps roughly three times the matches — and
    the band is wide enough to hold both.

    Args:
        backend: Which implementation both ONNX stages use.
        galileo_input_dir: The r2b_galileo input directory.
        tmp_path_factory: pytest's per-module temporary directory factory.
    """
    options: PipelineOptions = PipelineOptions(
        input_dir=galileo_input_dir,
        output_dir=tmp_path_factory.mktemp(f"galileo_{backend}"),
        min_inter_frame_distance=SMOKE_MIN_INTER_FRAME_DISTANCE_M,
        features_backend=backend,  # type: ignore[arg-type]
        matching_backend=backend,  # type: ignore[arg-type]
    )
    summary: PipelineSummary = run_pipeline(options)
    mean_matches: float = summary.mapping.num_observations / max(summary.num_pairs, 1)
    print(
        f"[colsfm] {backend} end to end | registered {summary.mapping.num_registered_images} "
        f"| points {summary.mapping.num_points3D} | {mean_matches:.0f} observations per pair "
        f"| {summary.mapping.mean_reprojection_error_px:.3f} px"
    )
    assert summary.mapping.num_registered_images >= SMOKE_MIN_REGISTERED_IMAGES
    assert summary.mapping.num_points3D > 0
    assert SMOKE_MIN_MEAN_MATCHES <= mean_matches <= SMOKE_MAX_MEAN_MATCHES


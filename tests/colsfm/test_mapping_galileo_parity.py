"""Galileo parity: the blob's own keyframes and matches, so only the mapper differs.

The integration half of the mapping suite, and its own file for that reason: every
test here needs a shipped cuSFM run on disk, rewrites that run's matches into a
COLMAP database, and maps all 226 keyframes. It is the slowest mapping module by
an order of magnitude and the only one that skips when the data is missing.

Everything upstream is held fixed — the blob's keyframes, its matches and its
input poses — so any difference between cuSFM's numbers and ours is the mapper's.
The reference numbers come from the run's own `sparse/` export, not from its log.
"""

from __future__ import annotations

import dataclasses
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from conftest import pose_delta, read_blob_keyframes
from google.protobuf.message import Message
from jaxtyping import Float64, UInt32
from mapping_helpers import quiet_options, write_database
from numpy import ndarray
from scipy.spatial import cKDTree

from colsfm.config import CusfmConfig, VisionMappingConfig
from colsfm.export import read_runtime_records
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.mapping import MappingResult, load_correspondences, run_mapping
from colsfm.reconstruction import build_reconstruction, gauge_rig_frame
from colsfm.schema import load_schema


@pytest.fixture(autouse=True, scope="module")
def _quiet_glog() -> None:
    """Silence COLMAP's own logging so the tests' own numbers stay readable."""
    pycolmap.logging.minloglevel = 2

BLOB_RUN_DIRS: dict[str, Path] = {
    "galileo-32": Path("data") / "cusfm_runs" / "galileo",
    "galileo-226": Path("data") / "cusfm_runs" / "galileo_blobref" / "cusfm",
}
"""The shipped cuSFM runs parity is measured against, relative to the repo root.

`galileo-32` is the 32-keyframe run the mapper spec documents throughout;
`galileo-226` is the acceptance run of `docs/open-pipeline-plan.md`. Both are
plain `VEHICLE_RIG` invocations with `--optimize_extrinsics` off. The
226-keyframe run under `data/cusfm_runs/galileo/cusfm` is deliberately not used:
its artifacts were last written with `--optimize_extrinsics=True`, which adds
777 `relative_extrinsic` and 8 `absolute_extrinsic` residual blocks that
pycolmap's bundle adjuster cannot express.
"""

DROPPED_PAIRS_PATTERN: str = r"Filter image pairs: by min matches=\d+: (\d+)"
"""The blob's own count of pairs rejected under `min_num_matches_per_pair`."""


def _blob_keypoints(keyframe_dir: Path, frames_meta: FramesMeta) -> dict[int, Float64[ndarray, "num_keypoints 2"]]:
    """Take the extractor's keypoint pixel columns out of the blob's own keyframes.

    Only `keypoint_vector.x` and `.y` are read; the descriptors are what the matcher
    needed and the mapper never sees them.

    Args:
        keyframe_dir: The run's `keyframes` directory.
        frames_meta: The collection naming the images.

    Returns:
        Keypoints per keyframe id, in the file's own order — which is what the
        match files index into.
    """
    return {
        keyframe_id: np.stack(
            [
                np.asarray(message.keypoint_vector.x, dtype=np.float64),
                np.asarray(message.keypoint_vector.y, dtype=np.float64),
            ],
            axis=1,
        )
        for keyframe_id, message in read_blob_keyframes(keyframe_dir, frames_meta).items()
    }


def _blob_matches(matches_dir: Path) -> dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]]:
    """Read every `FrameMatches` result the blob's matcher wrote.

    `FramePairMatch.frames` names the two **keyframe ids** and each
    `FeatureMatch` is a pair of indices into those keyframes' keypoint vectors
    (feature_matcher_main.md §4.2). One `FramePairMatch` is written per input
    pair, empty ones included. COLMAP keys a pair by the lower image id, so the
    pairs are normalised here rather than relying on the database to swap them.

    Args:
        matches_dir: The run's `matches` directory.

    Returns:
        Matches per ordered image pair, empty pairs included.
    """
    frame_matches_class = load_schema().message_class("protos.visual.cusfm.FrameMatches")
    matches: dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]] = {}
    for path in sorted(matches_dir.glob("*_result.pb")):
        message: Message = frame_matches_class()
        message.ParseFromString(path.read_bytes())
        for pair in message.feature_matches:
            source_id: int = int(pair.frames.source_frame_id)
            target_id: int = int(pair.frames.target_frame_id)
            columns: UInt32[ndarray, "num_matches 2"] = np.array(
                [[int(match.source_point_id), int(match.target_point_id)] for match in pair.matches], dtype=np.uint32
            ).reshape(-1, 2)
            if source_id > target_id:
                source_id, target_id = target_id, source_id
                columns = columns[:, ::-1].copy()
            matches[(source_id, target_id)] = columns
    return matches


@dataclass(frozen=True, slots=True)
class BlobRun:
    """One shipped cuSFM run, loaded as a mapper input and a reference output."""

    name: str
    """Key in `BLOB_RUN_DIRS`."""
    input_meta: FramesMeta
    """`pose_graph/frames_meta.json`: the poses the mapper was given."""
    output_meta: FramesMeta
    """`kpmap/keyframes/frames_meta.json`: the poses the mapper produced."""
    database_path: Path
    """The blob's keypoints and matches, rewritten as a COLMAP database."""
    reference: pycolmap.Reconstruction
    """`sparse/`, the blob's exported model, with point errors computed."""
    num_match_pairs: int
    """Image pairs the matcher wrote, empty ones included."""
    num_dropped_pairs: int
    """Pairs the blob's own log says it dropped under `min_num_matches_per_pair`."""
    runtime_seconds: float
    """What `runtime.csv` recorded for `keypoints_mapper_main`."""


@pytest.fixture(scope="module", params=sorted(BLOB_RUN_DIRS))
def galileo_blob(request: pytest.FixtureRequest, repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> BlobRun:
    """A shipped run, with its matches rewritten into a COLMAP database.

    Args:
        request: Carries the run name.
        repo_root: Repository root.
        tmp_path_factory: Where the database goes.

    Returns:
        The run's input, its reference output and a database of its matches.
    """
    name: str = str(request.param)
    run_dir: Path = repo_root / BLOB_RUN_DIRS[name]
    if not (run_dir / "sparse" / "images.txt").is_file():
        pytest.skip(f"No cuSFM reference run at {run_dir}")

    input_meta: FramesMeta = read_frames_meta(run_dir / "pose_graph" / "frames_meta.json")
    matches: dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]] = _blob_matches(run_dir / "matches")
    database_path: Path = tmp_path_factory.mktemp(name) / "blob_matches.db"
    write_database(database_path, input_meta, _blob_keypoints(run_dir / "keyframes", input_meta), matches)

    reference: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reference.read_text(str(run_dir / "sparse"))
    reference.update_point_3d_errors()
    log_text: str = (run_dir / "kpmap" / "keypoints_mapper_main.txt").read_text()
    dropped_match: re.Match[str] | None = re.search(DROPPED_PAIRS_PATTERN, log_text)
    assert dropped_match is not None, "the blob's log always reports how many pairs it dropped"
    runtime_seconds: float = next(
        record.runtime_seconds
        for record in reversed(read_runtime_records(run_dir / "runtime.csv"))
        if record.command.startswith("keypoints_mapper_main")
    )
    return BlobRun(
        name=name,
        input_meta=input_meta,
        output_meta=read_frames_meta(run_dir / "kpmap" / "keyframes" / "frames_meta.json"),
        database_path=database_path,
        reference=reference,
        num_match_pairs=len(matches),
        num_dropped_pairs=int(dropped_match.group(1)),
        runtime_seconds=runtime_seconds,
    )


@dataclass(frozen=True, slots=True)
class ParityRun:
    """One `run_mapping` call over a blob run's own matches."""

    result: MappingResult
    """What the mapper produced."""
    elapsed_seconds: float
    """Wall-clock seconds the call took, for the runtime comparison."""


@pytest.fixture(scope="module")
def parity(galileo_blob: BlobRun, isaac_config: CusfmConfig) -> ParityRun:
    """Map a blob run's own matches with the isaac configuration.

    Args:
        galileo_blob: The reference run.
        isaac_config: The profile the blob ran with.

    Returns:
        The mapping result and its wall-clock time.
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    started: float = time.perf_counter()
    result: MappingResult = run_mapping(
        build_reconstruction(galileo_blob.input_meta), galileo_blob.database_path, mapping_config, quiet_options()
    )
    return ParityRun(result=result, elapsed_seconds=time.perf_counter() - started)


def test_the_blob_database_keeps_the_pairs_the_blob_kept(galileo_blob: BlobRun, isaac_config: CusfmConfig) -> None:
    """The loader and `min_num_matches_per_pair` agree with the blob's own log.

    `Filter image pairs: by min matches=15: N` is the blob's count of rejected
    pairs, so the pairs the correspondence graph ends up with must be the pairs
    written minus that N. This is what makes the comparison a mapper comparison:
    both sides start from the same graph.

    An image whose every pair was dropped never enters `DatabaseCache` at all,
    so the graph can hold fewer images than the metadata — one fewer on the
    32-keyframe run. Those images stay registered and simply observe nothing,
    which is also why the blob exports 28 images out of 32.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(galileo_blob.input_meta).reconstruction
    correspondences = load_correspondences(
        reconstruction, galileo_blob.database_path, isaac_config.vision_mapping.min_num_matches_per_pair
    )
    expected_pairs: int = galileo_blob.num_match_pairs - galileo_blob.num_dropped_pairs
    print(
        f"[colsfm] {galileo_blob.name}: {galileo_blob.num_match_pairs} pairs written, "
        f"{galileo_blob.num_dropped_pairs} dropped by the blob, {correspondences.num_image_pairs} in the graph"
    )
    assert correspondences.num_image_pairs == expected_pairs
    assert galileo_blob.reference.num_images() <= correspondences.num_images <= len(galileo_blob.input_meta.keyframes)


def test_galileo_parity_against_the_blob(galileo_blob: BlobRun, parity: ParityRun) -> None:
    """The whole mapper, on the blob's own matches, against the blob's own output.

    Everything upstream is held fixed — the blob's keyframes, its matches and its
    input poses — so any difference is the mapper's. The reference numbers come
    from `sparse/`, not from the log.

    The point count is **not** compared directly. COLMAP's LO-RANSAC
    triangulation does a local optimisation and a final refit that cuSFM's plain
    MSAC does not (§9.3), and cuSFM grows one disjoint track per connected
    component of the match graph while COLMAP grows one per reference
    observation. The result is that colsfm keeps weak three-view tracks cuSFM
    rejects: a superset, at a lower reprojection error. What is asserted instead
    is that none of the blob's structure is lost.
    """
    result: MappingResult = parity.result
    blob_reprojection_px: float = galileo_blob.reference.compute_mean_reprojection_error()
    print(
        f"[colsfm] {galileo_blob.name} parity | images {result.num_images_with_observations} vs "
        f"{galileo_blob.reference.num_images()} | points {result.num_points3D} vs "
        f"{galileo_blob.reference.num_points3D()} | observations {result.num_observations} vs "
        f"{galileo_blob.reference.compute_num_observations()} | reprojection "
        f"{result.mean_reprojection_error_px:.4f} vs {blob_reprojection_px:.4f} px | track "
        f"{result.mean_track_length:.3f} vs {galileo_blob.reference.compute_mean_track_length():.3f} | "
        f"{parity.elapsed_seconds:.2f} s vs {galileo_blob.runtime_seconds:.2f} s "
        f"({parity.elapsed_seconds / galileo_blob.runtime_seconds:.2f}x)"
    )

    assert result.num_registered_images == len(galileo_blob.input_meta.keyframes)
    assert abs(result.num_images_with_observations - galileo_blob.reference.num_images()) <= 2
    assert abs(result.mean_reprojection_error_px - blob_reprojection_px) < 0.3
    # `docs/open-pipeline-plan.md` asks for a per-stage runtime within 2x of the
    # blob's; the benchmark harness is where that bound is enforced. This guard
    # only catches a gross regression (an order of magnitude), because the test
    # measures wall clock on a machine that other runs share: standalone the stage
    # takes ~7 s, under a concurrent RoboCap run it measured 28 s against a 6.7 s
    # blob reference, and a 3x bound failed for reasons unrelated to the code.
    assert parity.elapsed_seconds < 10.0 * galileo_blob.runtime_seconds
    assert result.num_points3D >= 0.9 * galileo_blob.reference.num_points3D()

    blob_points: Float64[ndarray, "num_blob_points 3"] = np.array(
        [galileo_blob.reference.point3D(point_id).xyz for point_id in galileo_blob.reference.point3D_ids()]
    )
    our_points: Float64[ndarray, "num_points 3"] = np.array(
        [result.reconstruction.point3D(point_id).xyz for point_id in result.reconstruction.point3D_ids()]
    )
    nearest_m: Float64[ndarray, " num_blob_points"] = cKDTree(our_points).query(blob_points)[0]
    coverage: float = float(np.mean(nearest_m < 0.05))
    print(f"[colsfm] {galileo_blob.name} parity | {coverage:.1%} of the blob's points have one of ours within 5 cm")
    assert coverage >= 0.95


def test_galileo_rig_poses_match_the_blob(galileo_blob: BlobRun, parity: ParityRun) -> None:
    """Every rig pose lands within a fraction of the correction the blob applied.

    The gauge frame is the one holding the lowest keyframe id, which cuSFM also
    holds constant (§6.3), so the two trajectories are comparable directly with
    no alignment step to hide an error. The yardstick is how far the blob moved
    the poses it was given — about 12 mm and half a degree on the last rig frame.
    """
    blob_correction_m: float = 0.0
    blob_correction_deg: float = 0.0
    worst_translation_m: float = 0.0
    worst_rotation_deg: float = 0.0
    input_by_sample: dict[int, pycolmap.Rigid3d] = {
        rig_frame.synced_sample_id: rig_frame.world_T_vehicle for rig_frame in galileo_blob.input_meta.rig_frames()
    }
    for rig_frame in galileo_blob.output_meta.rig_frames():
        correction_m, correction_deg = pose_delta(
            input_by_sample[rig_frame.synced_sample_id], rig_frame.world_T_vehicle
        )
        blob_correction_m = max(blob_correction_m, correction_m)
        blob_correction_deg = max(blob_correction_deg, correction_deg)
        translation_m, rotation_deg = pose_delta(
            parity.result.reconstruction.frame(rig_frame.synced_sample_id).rig_from_world.inverse(),
            rig_frame.world_T_vehicle,
        )
        worst_translation_m = max(worst_translation_m, translation_m)
        worst_rotation_deg = max(worst_rotation_deg, rotation_deg)

    print(
        f"[colsfm] {galileo_blob.name} parity | pose delta {worst_translation_m * 1e3:.3f} mm / "
        f"{worst_rotation_deg:.4f} deg, against a blob correction of {blob_correction_m * 1e3:.3f} mm / "
        f"{blob_correction_deg:.4f} deg"
    )
    assert worst_translation_m < 5e-3
    assert worst_rotation_deg < 0.15
    assert worst_translation_m < blob_correction_m
    assert worst_rotation_deg < blob_correction_deg

    # Both sides hold the same rig frame constant, so it cannot have moved.
    gauge_sample_id: int = gauge_rig_frame(galileo_blob.input_meta).synced_sample_id
    gauge_translation_m, gauge_rotation_deg = pose_delta(
        parity.result.reconstruction.frame(gauge_sample_id).rig_from_world.inverse(),
        next(
            rig_frame
            for rig_frame in galileo_blob.output_meta.rig_frames()
            if rig_frame.synced_sample_id == gauge_sample_id
        ).world_T_vehicle,
    )
    assert gauge_translation_m < 1e-9
    assert gauge_rotation_deg < 1e-9


def test_the_cauchy_scale_is_the_sigma_whitening_equivalent(galileo_blob: BlobRun, isaac_config: CusfmConfig) -> None:
    """A loss scale equal to sigma is what actually reproduces the blob's poses.

    cuSFM whitens the residual by `reprojection_error_standard_deviation = 4.0`
    and then applies `CauchyLoss(1.0)`; COLMAP does not whiten, so the port
    carries the whitening in the loss scale (§9.2). That is an argument, not a
    measurement — this is the measurement. On the 32-keyframe run the worst pose
    difference against the blob is 5.51 mm at scale 1.0, 2.06 mm at 2.0,
    **1.26 mm at 4.0** and 2.63 mm at 8.0: the minimum sits exactly at sigma.
    """
    if galileo_blob.name != "galileo-32":
        pytest.skip("one run is enough to locate the minimum")
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    deltas_mm: dict[float, float] = {}
    for sigma in (1.0, 4.0, 8.0):
        scaled_config: VisionMappingConfig = dataclasses.replace(
            mapping_config,
            bundle_adjustment=dataclasses.replace(
                mapping_config.bundle_adjustment, reprojection_error_standard_deviation=sigma
            ),
        )
        result: MappingResult = run_mapping(
            build_reconstruction(galileo_blob.input_meta),
            galileo_blob.database_path,
            scaled_config,
            quiet_options(),
        )
        deltas_mm[sigma] = 1e3 * max(
            pose_delta(
                result.reconstruction.frame(rig_frame.synced_sample_id).rig_from_world.inverse(),
                rig_frame.world_T_vehicle,
            )[0]
            for rig_frame in galileo_blob.output_meta.rig_frames()
        )
    print(f"[colsfm] pose delta by Cauchy scale: {', '.join(f'{k}: {v:.3f} mm' for k, v in deltas_mm.items())}")
    assert deltas_mm[4.0] < deltas_mm[1.0]
    assert deltas_mm[4.0] < deltas_mm[8.0]

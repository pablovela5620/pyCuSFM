"""The top-level CLI, and the run it describes.

One sequence of decisions, each delegated: pick a dataset adapter, hand its
`PreparedSequence` to cuSFM, read the model back, fit the result onto the input
trajectory, and log both. The arithmetic that stays here is the alignment and
the numbers printed about it — the one thing the demo exists to report, and the
one thing no other module owns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, TypeAlias

import numpy as np
import rerun as rr
import tyro
from jaxtyping import Bool, Float64, Int
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.rig import Rig

from colsfm.colmap_text_model import ColmapModel, read_colmap_model
from demo.cusfm import CUSFM_POSE_GRAPH_DIR, RunConfig, run_cusfm
from demo.galileo import GalileoConfig, prepare_galileo
from demo.model_io import cusfm_rig_trajectory, read_cusfm_vehicle_poses, read_loop_closures, refined_extrinsics
from demo.poses import compose, umeyama_rigid
from demo.robocap import RobocapConfig, prepare_robocap
from demo.schema import CUSFM_RIG_INDEX, INPUT_RIG_INDEX
from demo.sequence import PreparedSequence
from demo.visualise import (
    CUSFM_RIG_COLOR,
    build_blueprint,
    build_rig,
    clip_outliers,
    log_images,
    log_keyframe_highlight,
    log_loop_closures,
    log_rig,
    log_trajectory,
    log_video_references,
    log_video_streams,
)

DatasetConfig: TypeAlias = (
    Annotated[
        RobocapConfig,
        tyro.conf.subcommand("robocap", description="RoboCap fisheye segment with basalt VIO poses"),
    ]
    | Annotated[
        GalileoConfig,
        tyro.conf.subcommand("galileo", description="Bundled r2b_galileo pinhole sample (has ground truth)"),
    ]
)



@dataclass
class Config:
    """Top-level CLI."""

    dataset: DatasetConfig = field(default_factory=RobocapConfig)
    """Which sequence to reconstruct."""
    run: RunConfig = field(default_factory=RunConfig)
    """cuSFM settings."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun sink: spawn / save / connect / headless."""



def main(config: Config) -> None:
    """Prepare, reconstruct, align, and log."""
    dataset: DatasetConfig = config.dataset
    if isinstance(dataset, RobocapConfig):
        sequence: PreparedSequence = prepare_robocap(dataset, config.run)
    else:
        sequence = prepare_galileo(dataset, config.run)

    sparse_dir: Path = run_cusfm(sequence, config.run)
    model: ColmapModel = read_colmap_model(sparse_dir)
    sample_indices, cusfm_world_T_rig, rigidity_spread, per_camera = cusfm_rig_trajectory(sequence, model)

    print("\n─── cuSFM result ───")
    for name, count in per_camera.items():
        total: int = sum(1 for path in sequence.image_paths[name] if path is not None)
        if total:
            print(f"  {name:12s} {count:4d}/{total:4d} images registered")
    print(f"  rig rigidity spread: {1e3 * rigidity_spread:.2f} mm (should be ~0 with ba_frame_type=vehicle_rig)")

    if len(sample_indices) < 3:
        raise RuntimeError(f"only {len(sample_indices)} samples registered — reconstruction failed")

    input_world_T_rig: Float64[ndarray, "m 4 4"] = sequence.world_T_rig[sample_indices]
    rotation, translation, rmse, would_be_scale = umeyama_rigid(
        cusfm_world_T_rig[:, :3, 3], input_world_T_rig[:, :3, 3]
    )
    alignment: Float64[ndarray, "4 4"] = compose(rotation, translation)
    aligned_world_T_rig: Float64[ndarray, "m 4 4"] = alignment @ cusfm_world_T_rig
    trajectory_length: float = float(
        np.linalg.norm(np.diff(input_world_T_rig[:, :3, 3], axis=0), axis=1).sum()
    )
    print(f"  aligned {len(sample_indices)} rig poses to the input trajectory (scale held at 1.0)")
    # Not "ATE": the input trajectory is itself an estimate unless the dataset
    # ships ground truth (galileo does; RoboCap's basalt VIO does not).
    print(
        f"  disagreement RMSE : {1e3 * rmse:.1f} mm over {trajectory_length:.2f} m travelled "
        f"(cuSFM vs input estimate, NOT ground truth)"
    )
    print(f"  would-be scale    : {would_be_scale:.5f} (1.0 = cuSFM preserved metric scale)")
    print(f"  sparse points     : {len(model.points_xyz)}")

    # ── log ──────────────────────────────────────────────────────────────────
    rr.log("/", rr.ViewCoordinates.RFU, static=True)
    rr.send_blueprint(build_blueprint(sequence))

    # The input rig plays at the source frame rate when the dataset provides it,
    # so the walk is continuous rather than a keyframe slideshow. cuSFM still only
    # ever received the keyframes; `log_keyframe_highlight` marks which those are.
    has_full: bool = sequence.full_timestamps_ns is not None and sequence.full_world_T_rig is not None
    input_times: Int[ndarray, "n"] = sequence.full_timestamps_ns if has_full else sequence.timestamps_ns
    input_poses: Float64[ndarray, "n 4 4"] = (
        sequence.full_world_T_rig if has_full else sequence.world_T_rig
    )
    input_rig: Rig = build_rig(sequence, INPUT_RIG_INDEX, input_poses)
    log_rig(
        input_rig,
        sequence,
        input_times,
        label="input trajectory (basalt VIO)" if sequence.name == "robocap" else "input trajectory (ego-motion)",
        frustum_color=None,
    )

    # With --run.optimize-extrinsics, bundle adjustment changes rig_T_cam, so the
    # rig pose must come from cuSFM's own vehicle-frame output rather than being
    # derived from the (now stale) input extrinsics.
    refined: dict[str, Float64[ndarray, "4 4"]] | None = None
    cusfm_pose_times_ns: Int[ndarray, "n"] = sequence.timestamps_ns[sample_indices]
    cusfm_pose_stream: Float64[ndarray, "n 4 4"] = aligned_world_T_rig
    vehicle = read_cusfm_vehicle_poses(sparse_dir)
    if vehicle is not None and config.run.optimize_extrinsics:
        vehicle_times, vehicle_poses = vehicle
        refined, extrinsic_spread = refined_extrinsics(sequence, model, vehicle_times, vehicle_poses)
        cusfm_pose_times_ns = vehicle_times
        cusfm_pose_stream = alignment @ vehicle_poses
        print("\n─── extrinsic refinement ───")
        for name, matrix in refined.items():
            original: Float64[ndarray, "4 4"] = next(
                c.rig_T_cam for c in sequence.cameras if c.name == name
            )
            d_t: float = float(np.linalg.norm(matrix[:3, 3] - original[:3, 3]))
            d_r: float = float(
                np.degrees(
                    np.linalg.norm(Rotation.from_matrix(matrix[:3, :3] @ original[:3, :3].T).as_rotvec())
                )
            )
            print(f"  {name:12s} moved {1e3 * d_t:6.2f} mm, {d_r:5.2f} deg vs the factory calibration")
        print(f"  per-frame spread: {1e3 * extrinsic_spread:.2f} mm (near 0 = rig stayed rigid)")

    cusfm_rig: Rig = build_rig(
        sequence, CUSFM_RIG_INDEX, cusfm_pose_stream, extrinsics_override=refined
    )
    log_rig(
        cusfm_rig,
        sequence,
        cusfm_pose_times_ns,
        label=f"cuSFM refined (disagreement vs input {1e3 * rmse:.1f} mm RMSE)",
        frustum_color=CUSFM_RIG_COLOR,
    )

    aligned_points: Float64[ndarray, "n_points 3"] = (
        model.points_xyz @ rotation.T + translation if len(model.points_xyz) else model.points_xyz
    )
    keep: Bool[ndarray, "n_points"] = clip_outliers(aligned_points, config.run.point_clip_percentile)
    dropped: int = int((~keep).sum())
    if dropped:
        print(f"  clipped {dropped} outlier points ({100 * dropped / len(keep):.1f}%) for display only")
    rr.log(
        "world/points",
        rr.Points3D(positions=aligned_points[keep], colors=model.points_rgb[keep], radii=0.01),
        static=True,
    )

    input_source: str = "basalt VIO" if sequence.name == "robocap" else "ego-motion"
    log_trajectory(
        "input",
        input_poses[:, :3, 3],
        source=f"{input_source} ({len(input_times)} poses)",
        hue=(120, 200, 255),
        keyframe_positions=sequence.world_T_rig[:, :3, 3],
    )
    log_trajectory(
        "cusfm",
        aligned_world_T_rig[:, :3, 3],
        source=f"cuSFM global BA ({len(model.points_xyz)} points)",
        hue=(255, 150, 40),
    )

    loops: Int[ndarray, "m 2"] | None = read_loop_closures(sparse_dir.parent / CUSFM_POSE_GRAPH_DIR)
    if loops is not None:
        drawn: int = log_loop_closures(
            "cusfm", loops, aligned_world_T_rig[:, :3, 3], sample_indices
        )
        spans: Int[ndarray, "m"] = np.abs(loops[:, 0] - loops[:, 1])
        print(
            f"  logged {drawn} loop closures (spanning {spans.min()}-{spans.max()} samples, "
            f"median {int(np.median(spans))})"
        )
        if drawn < len(loops):
            # Almost always a stride mismatch: the pose graph indexes the samples
            # cuSFM was given, so re-logging at a different --dataset.frame-stride
            # than the run used puts most loops past the end of the trajectory.
            print(
                f"  WARNING: dropped {len(loops) - drawn} of {len(loops)} loop closures - "
                f"their samples are outside the {len(sample_indices)} registered here. "
                f"Usually --dataset.frame-stride does not match the stride this run was "
                f"reconstructed at."
            )
    else:
        # Worth saying out loud: the shipped configs return zero loops, so a silent
        # absence here reads as "no revisits" when it actually means the gates are
        # rejecting them. See LOOP_CLOSURE_CONFIG_DIR.
        print("  no loop closures in the pose graph")

    if sequence.video_samples:
        frames: int = log_video_streams(sequence, INPUT_RIG_INDEX)
        references: int = log_video_references(sequence, INPUT_RIG_INDEX, CUSFM_RIG_INDEX)
        log_keyframe_highlight(sequence, INPUT_RIG_INDEX)
        print(
            f"  logged {frames} video frames across {len(sequence.video_samples)} cameras "
            f"(full rate); {int(sequence.keyframe_mask.sum())} marked as cuSFM keyframes; "
            f"{references} frame references give rig_01 the same imagery without copying it"
        )
    else:
        logged_images: int = log_images(sequence, INPUT_RIG_INDEX, dataset.image_stride)
        print(f"  logged {logged_images} images (every {dataset.image_stride} samples)")
    print()



if __name__ == "__main__":
    main(tyro.cli(Config))

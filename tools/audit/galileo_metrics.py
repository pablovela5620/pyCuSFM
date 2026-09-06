"""The Galileo dataset's own reference bundle: what the audit scripts score against.

A plain COLMAP reconstruction carries no `kpmap/keyframes/frames_meta.json` and
no rig, so `colsfm.benchmark`'s run-to-run comparison cannot reach it. What it
can be scored against is a **per-image** ground-truth camera centre: for every
keyframe the dataset's ground truth gives `world_T_vehicle` at that timestamp,
and composing with the calibrated `vehicle_T_cam` gives a camera centre. Reading
that out of `data/r2b_galileo` is all this module now does.

The scoring itself — both alignments, the timestamp join, the rig track a
rig-free model implies — is `colsfm.trajectory`, so the audit's numbers and the
benchmark report's numbers come from the same code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from jaxtyping import Float64, Int64
from numpy import ndarray

from colsfm.benchmark import (
    GROUND_TRUTH_TOLERANCE_MICROSECONDS,
    RigTrack,
    match_timestamps,
    read_ground_truth,
)
from colsfm.frames_meta import FRAMES_META_NAME, CameraParams, FramesMeta, KeyframeMeta, read_frames_meta
from colsfm.geometry import rigid3d_from_matrix


@dataclass(frozen=True, slots=True)
class GalileoReference:
    """Everything the audit needs about the Galileo dataset, read once."""

    frames_meta: FramesMeta
    """The input collection: 226 keyframes, 8 cameras, the prior trajectory."""
    ground_truth: RigTrack
    """The shipped `ground_truth.txt` rig trajectory."""
    gt_center_by_image_name: dict[str, Float64[ndarray, "3"]]
    """Ground-truth camera centre per keyframe image name, in metres."""


def read_galileo_reference(input_dir: Path) -> GalileoReference:
    """Read the Galileo input metadata and expand ground truth to per-image poses.

    The ground truth is a vehicle-frame TUM trajectory sampled far faster than the
    keyframes, so each keyframe timestamp is joined to the nearest ground-truth
    sample within `GROUND_TRUTH_TOLERANCE_MICROSECONDS`, and the calibrated
    `vehicle_T_cam` turns that into a camera pose.

    Args:
        input_dir: Directory holding `frames_meta.json` and `ground_truth.txt`.

    Returns:
        The reference bundle.

    Raises:
        FileNotFoundError: When the dataset ships no ground truth.
    """
    frames_meta: FramesMeta = read_frames_meta(input_dir / FRAMES_META_NAME)
    ground_truth: RigTrack | None = read_ground_truth(input_dir)
    if ground_truth is None:
        raise FileNotFoundError(f"no ground_truth.txt under {input_dir}")

    keyframes: tuple[KeyframeMeta, ...] = tuple(sorted(frames_meta.keyframes, key=lambda item: item.timestamp_microseconds))
    query_microseconds: Int64[ndarray, "n"] = np.asarray([item.timestamp_microseconds for item in keyframes], dtype=np.int64)
    keyframe_indices, ground_truth_indices = match_timestamps(
        query_microseconds, ground_truth.timestamps_microseconds, GROUND_TRUTH_TOLERANCE_MICROSECONDS
    )

    centers: dict[str, Float64[ndarray, "3"]] = {}
    for keyframe_index, ground_truth_index in zip(keyframe_indices, ground_truth_indices, strict=True):
        keyframe: KeyframeMeta = keyframes[int(keyframe_index)]
        camera: CameraParams = frames_meta.cameras[keyframe.camera_params_id]
        world_T_vehicle: pycolmap.Rigid3d = rigid3d_from_matrix(
            ground_truth.world_R_rig[int(ground_truth_index)], ground_truth.world_t_rig[int(ground_truth_index)]
        )
        world_T_cam: pycolmap.Rigid3d = world_T_vehicle * camera.vehicle_T_cam
        centers[keyframe.image_name] = np.asarray(world_T_cam.translation, dtype=np.float64)

    return GalileoReference(frames_meta=frames_meta, ground_truth=ground_truth, gt_center_by_image_name=centers)

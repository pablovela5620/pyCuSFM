"""`frames_meta.json` to the rig types `colsfm.loop_pose` measures against.

`colsfm.loop_pose` deliberately knows nothing about cuSFM metadata: it takes a `RigGeometry`
(cameras, `cam_T_vehicle` extrinsics, stereo partnerships) and a `RigFrameIndex` (rig frames
in time order, their prior poses, their images) and nothing else, so it can be driven from a
synthetic scene in a test as easily as from a real run. This module is the one adapter that
turns a parsed `FramesMeta` into those two, and it lives apart from `colsfm.loop_closure`
because the conversion is not part of the loop-closure funnel — the pose-graph and
extrinsics work needs the same rig.
"""

from __future__ import annotations

import pycolmap

from colsfm.cameras import colmap_cameras
from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame
from colsfm.loop_pose import RigFrameIndex, RigGeometry, build_rig_geometry


def calibrated_cameras(frames_meta: FramesMeta) -> dict[int, pycolmap.Camera]:
    """Build the `pycolmap.Camera`s the generalized estimators need.

    `Camera.has_prior_focal_length` defaults to **False**, and with it false COLMAP takes
    the uncalibrated path: a pinhole pair degrades to a fundamental matrix with no relative
    pose, and a fisheye pair comes back `DEGENERATE` with zero inliers, silently
    (`docs/spec/pycolmap-capabilities.md` §5). `colsfm.cameras.colmap_camera` sets it on
    every camera it builds, so this is `colmap_cameras` and the name is kept only to say
    at the call site which property the estimators depend on.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        Calibrated cameras keyed by `camera_params_id`.
    """
    return colmap_cameras(frames_meta)


def rig_geometry(frames_meta: FramesMeta) -> RigGeometry:
    """The rig the metric estimator resections against.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        Calibrated cameras, their `cam_T_vehicle` extrinsics and the declared stereo
        partnerships, in `colsfm.loop_pose`'s parallel-tuple form.
    """
    partners: dict[int, int] = {}
    for pair in frames_meta.stereo_pairs:
        partners[pair.left_camera_params_id] = pair.right_camera_params_id
        partners[pair.right_camera_params_id] = pair.left_camera_params_id
    return build_rig_geometry(
        calibrated_cameras(frames_meta),
        {camera_id: camera.vehicle_T_cam for camera_id, camera in frames_meta.cameras.items()},
        partners,
    )


def rig_frame_index(frames_meta: FramesMeta) -> RigFrameIndex:
    """The rig frames, their prior poses and their images, in time order.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        The index `colsfm.loop_pose` walks to find a rig frame's neighbours and images.
    """
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    rig_frames: tuple[RigFrame, ...] = frames_meta.rig_frames()
    keyframe_by_rig_camera: dict[tuple[int, int], int] = {}
    for rig in rig_frames:
        for keyframe_id in rig.keyframe_ids:
            keyframe_by_rig_camera.setdefault((rig.synced_sample_id, keyframe_by_id[keyframe_id].camera_params_id), keyframe_id)
    return RigFrameIndex(
        sequence=tuple(rig.synced_sample_id for rig in rig_frames),
        world_T_rig={rig.synced_sample_id: rig.world_T_vehicle for rig in rig_frames},
        keyframe_by_rig_camera=keyframe_by_rig_camera,
    )

"""A `pycolmap.Reconstruction` built from cuSFM's `frames_meta.json`.

This is the data-model half of `keypoints_mapper_main`: the rig, the frames and
the images the triangulator and the bundle adjuster then work on
(keypoints_mapper_main.md §9.1). No triangulation happens here — see
`colsfm.mapping`.

## The rig frame is the vehicle, not a camera

cuSFM's `VEHICLE_RIG` mode carries one pose per rig instant (`synced_sample_id`)
in the **vehicle** FLU frame, plus one shared `sensor_to_vehicle_transform` per
camera (§5.2, §6.2). COLMAP expresses exactly that with a `Rig` whose
`sensor_from_rig` per camera is `cam_T_vehicle` — the **inverse** of the
metadata's `vehicle_T_cam` — and a `Frame` per rig instant whose
`rig_from_world` is `vehicle_T_world`.

COLMAP forces the rig's **reference** sensor to identity
(`rig.h:200 Check failed: sensor_id != ref_sensor_id_`), so no camera can be the
reference without moving the rig origin onto that camera. The vehicle body is
therefore registered as a non-camera reference sensor,
`sensor_t(SensorType.IMU, 0)`, which owns no data. Measured: triangulation,
`ObservationManager` and `create_default_bundle_adjuster` all accept a rig whose
reference sensor has no images.

The alternative — making the lowest-id keyframe's camera the reference and
folding the extrinsic into every other sensor — gives the same bundle adjustment
problem up to a fixed change of basis, but then `frame.rig_from_world` is a
camera pose and every comparison against cuSFM's vehicle poses needs an extra
composition. The vehicle rig keeps the two models directly comparable.

## Identifiers are cuSFM's own, verbatim

| cuSFM | COLMAP |
|---|---|
| `KeyframeMetaData.id` | `image_id` |
| `camera_params_id` | `camera_id` |
| `synced_sample_id` | `frame_id` |

All three are zero-based in RoboCap and COLMAP accepts 0 for each, so nothing is
renumbered. That is what lets a database written by the feature and matching
stages address the same images by id.
"""

from __future__ import annotations

from typing import Final

import pycolmap

from colsfm.cameras import colmap_cameras
from colsfm.frames_meta import FramesMeta, RigFrame

RIG_ID: Final[int] = 1
"""The single rig every reconstruction carries; cuSFM has no multi-rig concept."""

VEHICLE_SENSOR_ID: Final[int] = 0
"""Sensor id of the vehicle body, the rig's reference sensor."""


def vehicle_sensor_id() -> pycolmap.sensor_t:
    """The rig's reference sensor: the vehicle body, which owns no images.

    Returns:
        A non-camera `sensor_t` standing for the vehicle FLU frame.
    """
    return pycolmap.sensor_t(pycolmap.SensorType.IMU, VEHICLE_SENSOR_ID)


def camera_sensor_id(camera_params_id: int) -> pycolmap.sensor_t:
    """The rig sensor of one cuSFM camera.

    Args:
        camera_params_id: cuSFM's `camera_params_id`, used verbatim as the COLMAP camera id.

    Returns:
        The camera's `sensor_t`.
    """
    return pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_params_id)


def build_rig(frames_meta: FramesMeta) -> pycolmap.Rig:
    """Build the vehicle rig: one sensor per `camera_params_id`, extrinsics inverted.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        A rig whose reference sensor is the vehicle body and whose camera sensors
        each carry `sensor_from_rig = cam_T_vehicle`.
    """
    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = RIG_ID
    rig.add_ref_sensor(vehicle_sensor_id())
    for camera_params_id, camera in sorted(frames_meta.cameras.items()):
        rig.add_sensor(camera_sensor_id(camera_params_id), camera.vehicle_T_cam.inverse())
    return rig


def build_reconstruction(
    frames_meta: FramesMeta,
    cameras: dict[int, pycolmap.Camera] | None = None,
) -> pycolmap.Reconstruction:
    """Build a registered, pose-carrying reconstruction with no 3D points yet.

    Every rig frame is registered, so `IncrementalTriangulator` will triangulate
    against the supplied trajectory rather than re-estimating it. Cameras get
    `has_prior_focal_length = True`: the calibration comes from the metadata, and
    leaving the flag false silently degrades every downstream two-view geometry
    (pycolmap-capabilities.md §5).

    Args:
        frames_meta: The parsed `frames_meta.json`.
        cameras: COLMAP cameras keyed by `camera_params_id`; derived from the
            metadata by `colsfm.cameras.colmap_cameras` when None.

    Returns:
        A reconstruction with one rig, one frame per `synced_sample_id` and one
        image per keyframe, all frames registered.

    Raises:
        KeyError: When a keyframe names a `camera_params_id` the cameras lack.
    """
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    resolved_cameras: dict[int, pycolmap.Camera] = colmap_cameras(frames_meta) if cameras is None else cameras
    for camera_params_id in sorted(resolved_cameras):
        camera: pycolmap.Camera = resolved_cameras[camera_params_id]
        camera.has_prior_focal_length = True
        reconstruction.add_camera(camera)
    reconstruction.add_rig(build_rig(frames_meta))

    keyframe_by_id = frames_meta.keyframe_by_id()
    for rig_frame in frames_meta.rig_frames():
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = rig_frame.synced_sample_id
        frame.rig_id = RIG_ID
        frame.rig_from_world = rig_frame.world_T_vehicle.inverse()
        for keyframe_id in rig_frame.keyframe_ids:
            camera_params_id: int = keyframe_by_id[keyframe_id].camera_params_id
            frame.add_data_id(pycolmap.data_t(camera_sensor_id(camera_params_id), keyframe_id))
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)

    for rig_frame in frames_meta.rig_frames():
        for keyframe_id in rig_frame.keyframe_ids:
            keyframe = keyframe_by_id[keyframe_id]
            image: pycolmap.Image = pycolmap.Image(
                name=keyframe.image_name, camera_id=keyframe.camera_params_id, image_id=keyframe_id
            )
            image.frame_id = rig_frame.synced_sample_id
            reconstruction.add_image(image)
    return reconstruction


def gauge_rig_frame(frames_meta: FramesMeta) -> RigFrame:
    """The rig frame whose pose bundle adjustment holds constant.

    cuSFM fixes a single keyframe — in Galileo the lowest id in the map — and
    through the shared rig that fixes its whole rig frame
    (keypoints_mapper_main.md §6.3), which is why rig frame 1 moves 0.0000 mm
    while later frames move millimetres.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        The rig frame containing the lowest keyframe id.

    Raises:
        ValueError: When the collection holds no keyframes.
    """
    rig_frames: tuple[RigFrame, ...] = frames_meta.rig_frames()
    if not rig_frames:
        raise ValueError("Cannot choose a gauge rig frame: the metadata holds no keyframes")
    lowest_keyframe_id: int = min(keyframe.keyframe_id for keyframe in frames_meta.keyframes)
    return next(rig_frame for rig_frame in rig_frames if lowest_keyframe_id in rig_frame.keyframe_ids)

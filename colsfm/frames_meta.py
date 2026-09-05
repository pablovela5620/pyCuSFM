"""Typed access to cuSFM's `frames_meta.json` (`KeyframesMetadataCollection`).

The file is proto3 JSON with `preserve_proto_field_names`, which means **fields
equal to their default are omitted**: 29 of Galileo's 226 keyframes carry no
`camera_params_id` because theirs is 0 (NOTES.md gotcha 8). Parsing therefore
goes through protobuf's own `json_format` against the schema in `colsfm.schema`,
so defaults are supplied by protobuf rather than by hand.

`FramesMeta` keeps the parsed message alongside the typed view. Writers copy that
message and replace only what changed, so fields the pipeline never models —
`pose_covariance`, `car_mask_polygon`, `chroma`, `raw_image_cropping` — survive a
round trip untouched, exactly as `update_keyframe_pose_main` preserves them
(export.md §5.2).

Conventions (export.md §2, keypoints_mapper_main.md §4.3):

- `camera_to_world` **is** `world_T_cam`; axis-angle in degrees, translation in metres.
- `sensor_meta_data.sensor_to_vehicle_transform` **is** `vehicle_T_cam`.
- `synced_sample_id` groups the cameras of one rig instant.
- `world_T_vehicle(sample) = world_T_cam(ref) * vehicle_T_cam(ref)^-1`, where `ref`
  is the **lowest keyframe id** in the sample (pose_graph_main.md §4).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from google.protobuf import json_format
from google.protobuf.message import Message
from jaxtyping import Float64
from numpy import ndarray

from colsfm.geometry import AxisAngleDegrees, axis_angle_degrees_from_rigid3d, rigid3d_from_axis_angle_degrees
from colsfm.schema import KEYFRAMES_METADATA_COLLECTION, CusfmSchema, load_schema

CameraProjectionModel: TypeAlias = Literal["PINHOLE", "DISTORTED_PINHOLE", "FTHETA_WINDSHIELD", "OPENCV_FISHEYE"]
"""cuSFM's four camera projection models (`CameraProjectionModelType`)."""

InitialPoseType: TypeAlias = Literal["ALIGNMENT", "EGO_MOTION", "SESSION_EGO_MOTION", "GPS_IMU", "MAP_POSE"]
"""Where the poses in the collection came from; the mapper flips `EGO_MOTION` to `ALIGNMENT`."""

ProjectionMatrix: TypeAlias = Float64[ndarray, "3 4"]
"""Rectified 3x4 projection matrix, row-major, as `calibration_parameters.projection_matrix`."""

CameraMatrix: TypeAlias = Float64[ndarray, "3 3"]
"""Intrinsic 3x3 matrix, row-major, as `calibration_parameters.camera_matrix`."""

DistortionCoefficients: TypeAlias = Float64[ndarray, "num_coefficients"]
"""Distortion coefficients in OpenCV order, however many the calibration carries."""

JSON_INDENT: Final[int] = 2
"""Indent cuSFM's own writer uses, so our output diffs cleanly against the blob's."""


@dataclass(frozen=True, slots=True)
class KeyframeMeta:
    """One camera keyframe: identity, timing and pose."""

    keyframe_id: int
    """`id`; the COLMAP `IMAGE_ID`. Sparse and not 1-based."""
    camera_params_id: int
    """Sensor id; the COLMAP `CAMERA_ID`. Omitted from JSON when 0."""
    timestamp_microseconds: int
    """Capture time in microseconds since the Unix epoch."""
    image_name: str
    """`<camera_name>/<timestamp_nanoseconds>.jpeg`, relative to the raw data directory."""
    world_T_cam: pycolmap.Rigid3d
    """`camera_to_world`: the camera's pose in the world frame."""
    synced_sample_id: int
    """Rig frame id; every camera of one synchronised capture shares it."""
    track_id: int
    """Session segment id."""


@dataclass(frozen=True, slots=True)
class CameraParams:
    """One entry of `camera_params_id_to_camera_params`: a `CameraSensor`."""

    camera_params_id: int
    """Key in the map; the COLMAP `CAMERA_ID`."""
    sensor_name: str
    """Camera folder name, e.g. `front_stereo_camera_left`."""
    model_name: str
    """Hardware model string; empty in both shipped datasets."""
    sensor_id: int
    """Sensor id inside `sensor_meta_data`; not the map key."""
    frequency_hz: float
    """Capture rate in hertz."""
    vehicle_T_cam: pycolmap.Rigid3d
    """`sensor_to_vehicle_transform`: the rig extrinsic, camera to vehicle (FLU)."""
    image_width: int
    """Image width in pixels."""
    image_height: int
    """Image height in pixels."""
    projection_matrix: ProjectionMatrix | None
    """Rectified 3x4 projection matrix, or None when the calibration omits it."""
    camera_matrix: CameraMatrix | None
    """Unrectified 3x3 intrinsic matrix, or None when the calibration omits it."""
    distortion_coefficients: DistortionCoefficients
    """Distortion coefficients; empty when the calibration omits them."""
    projection_model: CameraProjectionModel
    """Which of cuSFM's four projection models this camera uses."""
    rolling_shutter_delay_microseconds: int
    """Row-to-row readout delay; 0 when global shutter or unset."""


@dataclass(frozen=True, slots=True)
class StereoPair:
    """A declared rectified stereo pair."""

    left_camera_params_id: int
    """`camera_params_id` of the left camera. Omitted from JSON when 0."""
    right_camera_params_id: int
    """`camera_params_id` of the right camera."""
    baseline_meters: float
    """Baseline in metres, despite what the surrounding units suggest."""


@dataclass(frozen=True, slots=True)
class RigFrame:
    """One rig instant: the cameras that fired together and their shared pose."""

    synced_sample_id: int
    """The rig frame id all member keyframes carry."""
    keyframe_ids: tuple[int, ...]
    """Member keyframe ids, ascending."""
    reference_keyframe_id: int
    """Lowest member keyframe id; the one the rig pose is derived from."""
    world_T_vehicle: pycolmap.Rigid3d
    """Rig pose in the world frame, in the vehicle FLU frame."""
    timestamp_microseconds: int
    """Timestamp of the reference keyframe."""


@dataclass(frozen=True, slots=True)
class FramesMeta:
    """A parsed `frames_meta.json`, with the source message kept for lossless writes."""

    keyframes: tuple[KeyframeMeta, ...]
    """Camera keyframes in file order."""
    cameras: dict[int, CameraParams]
    """Calibration and rig extrinsic per `camera_params_id`."""
    stereo_pairs: tuple[StereoPair, ...]
    """Declared stereo pairs; empty when the metadata declares none."""
    initial_pose_type: InitialPoseType
    """Provenance of `camera_to_world` for the whole collection."""
    detector_type: str
    """Keypoint detector the keyframes were produced with, e.g. `ALIKED_DETECTOR`."""
    descriptor_type: str
    """Descriptor type, e.g. `ALIKED_DESCRIPTOR`."""
    session_name_by_camera_params_id: dict[int, str]
    """`camera_params_id_to_session_name`; the TUM output subdirectory per camera."""
    track_name_by_track_id: dict[int, str]
    """`track_id_to_track_name`; empty in both shipped datasets."""
    bag_name_by_track_id: dict[int, str]
    """`track_id_to_bag_name`; empty in both shipped datasets."""
    keypoint_feature_has_depth: bool
    """Whether keypoints carry a depth channel (RGBD or stereo-as-RGBD modes)."""
    local_sector_key: str
    """Map sector key; empty in both shipped datasets."""
    message: Message
    """The parsed `KeyframesMetadataCollection`, so writers can copy it verbatim."""

    def keyframe_by_id(self) -> dict[int, KeyframeMeta]:
        """Index the keyframes by their `id`.

        Returns:
            Keyframes keyed by `keyframe_id`.
        """
        return {keyframe.keyframe_id: keyframe for keyframe in self.keyframes}

    def world_T_vehicle(self, keyframe: KeyframeMeta) -> pycolmap.Rigid3d:
        """Derive the vehicle pose implied by one keyframe.

        `world_T_vehicle = world_T_cam * cam_T_vehicle`, the composition
        `extract_pose_from_map_main` writes to its TUM files (export.md §4.2).

        Args:
            keyframe: A keyframe of this collection.

        Returns:
            The rig pose in the world frame.

        Raises:
            KeyError: When the keyframe's `camera_params_id` has no calibration.
        """
        vehicle_T_cam: pycolmap.Rigid3d = self.cameras[keyframe.camera_params_id].vehicle_T_cam
        return keyframe.world_T_cam * vehicle_T_cam.inverse()

    def rig_frames(self) -> tuple[RigFrame, ...]:
        """Group the keyframes into rig frames by `synced_sample_id`.

        The rig pose comes from the **lowest keyframe id** of the sample, not from
        an average and not from the lowest `camera_params_id`: those two rules
        disagree on 28 of Galileo's 29 samples, and only the lowest id reproduces
        cuSFM's own `vehicle_frames_meta.json` (to 1.4e-16 m, against 1.4e-5 m for
        the other cameras of the same sample). See pose_graph_main.md §4.

        Returns:
            Rig frames ordered by `synced_sample_id`, which is also time order.
        """
        members: dict[int, list[KeyframeMeta]] = {}
        for keyframe in self.keyframes:
            members.setdefault(keyframe.synced_sample_id, []).append(keyframe)
        rig_frames: list[RigFrame] = []
        for synced_sample_id in sorted(members):
            group: list[KeyframeMeta] = sorted(members[synced_sample_id], key=lambda keyframe: keyframe.keyframe_id)
            reference: KeyframeMeta = group[0]
            rig_frames.append(
                RigFrame(
                    synced_sample_id=synced_sample_id,
                    keyframe_ids=tuple(keyframe.keyframe_id for keyframe in group),
                    reference_keyframe_id=reference.keyframe_id,
                    world_T_vehicle=self.world_T_vehicle(reference),
                    timestamp_microseconds=reference.timestamp_microseconds,
                )
            )
        return tuple(rig_frames)

    def to_json(self) -> str:
        """Serialise back to proto3 JSON in cuSFM's own layout.

        Field names stay snake_case and default-valued fields stay omitted, so the
        output diffs against a blob-written file down to map entry order (protobuf
        maps have none) and the enum values cuSFM prints even at their default.

        Returns:
            The JSON text, without a trailing newline.
        """
        return json_format.MessageToJson(self.message, indent=JSON_INDENT, preserving_proto_field_name=True)

    def with_camera_to_world(
        self,
        world_T_cam_by_keyframe_id: Mapping[int, pycolmap.Rigid3d],
        initial_pose_type: InitialPoseType | None = None,
    ) -> FramesMeta:
        """Replace keyframe poses, keeping every other field byte-identical.

        This is what `keypoints_mapper_main` writes after bundle adjustment: the
        input collection with `camera_to_world` swapped and `initial_pose_type`
        moved from `EGO_MOTION` to `ALIGNMENT` (keypoints_mapper_main.md §4.4).
        Keyframes missing from the mapping keep their original pose.

        Args:
            world_T_cam_by_keyframe_id: New camera poses keyed by keyframe id.
            initial_pose_type: New provenance, or None to keep the current one.

        Returns:
            A new `FramesMeta`.
        """
        message: Message = copy.deepcopy(self.message)
        for entry in message.keyframes_metadata:
            pose: pycolmap.Rigid3d | None = world_T_cam_by_keyframe_id.get(int(entry.id))
            if pose is None:
                continue
            _write_rigid_transform(entry.camera_to_world, pose)
        if initial_pose_type is not None:
            message.initial_pose_type = message.DESCRIPTOR.fields_by_name["initial_pose_type"].enum_type.values_by_name[
                initial_pose_type
            ].number
        return parse_message(message)

    def filtered(
        self,
        keyframe_ids: Sequence[int] | set[int],
        synced_sample_id_by_keyframe_id: Mapping[int, int] | None = None,
    ) -> FramesMeta:
        """Keep a subset of keyframes, optionally renumbering their rig frame ids.

        `feature_extractor_main` writes exactly this: the surviving keyframes with
        `synced_sample_id` rewritten to a dense 1-based rank in time order
        (feature_extractor_main.md §4 and §8.1).

        Args:
            keyframe_ids: Keyframe ids to keep.
            synced_sample_id_by_keyframe_id: New `synced_sample_id` per kept keyframe
                id, or None to keep the existing ids.

        Returns:
            A new `FramesMeta` holding only the kept keyframes.
        """
        kept: set[int] = set(keyframe_ids)
        message: Message = copy.deepcopy(self.message)
        if synced_sample_id_by_keyframe_id is not None:
            for entry in message.keyframes_metadata:
                if int(entry.id) in kept:
                    entry.synced_sample_id = synced_sample_id_by_keyframe_id[int(entry.id)]
        surviving: list[bytes] = [entry.SerializeToString() for entry in message.keyframes_metadata if int(entry.id) in kept]
        del message.keyframes_metadata[:]
        for encoded in surviving:
            message.keyframes_metadata.add().ParseFromString(encoded)
        return parse_message(message)


def _rigid_transform(transform: Message) -> pycolmap.Rigid3d:
    """Convert a `RigidTransform3d` message into a `Rigid3d`.

    Args:
        transform: A `protos.common.geometry.RigidTransform3d`.

    Returns:
        The equivalent rigid transform; identity when the message is empty.
    """
    return rigid3d_from_axis_angle_degrees(
        axis_xyz=np.array([transform.axis_angle.x, transform.axis_angle.y, transform.axis_angle.z]),
        angle_degrees=float(transform.axis_angle.angle_degrees),
        translation_xyz=np.array([transform.translation.x, transform.translation.y, transform.translation.z]),
    )


def _write_rigid_transform(transform: Message, pose: pycolmap.Rigid3d) -> None:
    """Overwrite a `RigidTransform3d` message in place from a `Rigid3d`.

    Args:
        transform: A `protos.common.geometry.RigidTransform3d` to mutate.
        pose: The pose to store, encoded as axis-angle in degrees plus metres.
    """
    axis_angle: AxisAngleDegrees = axis_angle_degrees_from_rigid3d(pose)
    transform.axis_angle.x = float(axis_angle.axis_xyz[0])
    transform.axis_angle.y = float(axis_angle.axis_xyz[1])
    transform.axis_angle.z = float(axis_angle.axis_xyz[2])
    transform.axis_angle.angle_degrees = axis_angle.angle_degrees
    transform.translation.x = float(axis_angle.translation_xyz[0])
    transform.translation.y = float(axis_angle.translation_xyz[1])
    transform.translation.z = float(axis_angle.translation_xyz[2])


def _matrix(matrix_message: Message, row_count: int, column_count: int) -> Float64[ndarray, "rows cols"] | None:
    """Reshape a `MatrixD` into a numpy array when it holds the expected size.

    Args:
        matrix_message: A `protos.common.geometry.MatrixD`.
        row_count: Expected number of rows.
        column_count: Expected number of columns.

    Returns:
        The row-major matrix, or None when the field is absent or the wrong size.
    """
    data: list[float] = list(matrix_message.data)
    if len(data) != row_count * column_count:
        return None
    return np.array(data, dtype=np.float64).reshape(row_count, column_count)


def _camera_params(camera_params_id: int, camera_sensor: Message, schema: CusfmSchema) -> CameraParams:
    """Convert one `CameraSensor` message into a typed `CameraParams`.

    Args:
        camera_params_id: Map key the sensor was stored under.
        camera_sensor: A `protos.common.sensor.CameraSensor`.
        schema: Loaded cuSFM schema, used to name the projection-model enum value.

    Returns:
        The typed camera parameters.
    """
    calibration: Message = camera_sensor.calibration_parameters
    projection_model: str = schema.enum_value_name(
        "protos.common.sensor.CameraProjectionModelType", camera_sensor.camera_projection_model_type
    )
    return CameraParams(
        camera_params_id=camera_params_id,
        sensor_name=camera_sensor.sensor_meta_data.sensor_name,
        model_name=camera_sensor.sensor_meta_data.model_name,
        sensor_id=int(camera_sensor.sensor_meta_data.sensor_id),
        frequency_hz=float(camera_sensor.sensor_meta_data.frequency),
        vehicle_T_cam=_rigid_transform(camera_sensor.sensor_meta_data.sensor_to_vehicle_transform),
        image_width=int(calibration.image_width),
        image_height=int(calibration.image_height),
        projection_matrix=_matrix(calibration.projection_matrix, 3, 4),
        camera_matrix=_matrix(calibration.camera_matrix, 3, 3),
        distortion_coefficients=np.array(list(calibration.distortion_coefficients.data), dtype=np.float64),
        projection_model=projection_model,
        rolling_shutter_delay_microseconds=int(calibration.rolling_shutter_delay_microseconds),
    )


def parse_message(message: Message, schema: CusfmSchema | None = None) -> FramesMeta:
    """Build the typed view over an already-parsed `KeyframesMetadataCollection`.

    Args:
        message: The parsed collection.
        schema: Loaded cuSFM schema; the vendored one by default.

    Returns:
        The typed metadata, holding `message` for lossless re-serialisation.
    """
    resolved_schema: CusfmSchema = load_schema() if schema is None else schema
    keyframes: tuple[KeyframeMeta, ...] = tuple(
        KeyframeMeta(
            keyframe_id=int(entry.id),
            camera_params_id=int(entry.camera_params_id),
            timestamp_microseconds=int(entry.timestamp_microseconds),
            image_name=entry.image_name,
            world_T_cam=_rigid_transform(entry.camera_to_world),
            synced_sample_id=int(entry.synced_sample_id),
            track_id=int(entry.track_id),
        )
        for entry in message.keyframes_metadata
    )
    cameras: dict[int, CameraParams] = {
        int(camera_params_id): _camera_params(int(camera_params_id), camera_sensor, resolved_schema)
        for camera_params_id, camera_sensor in message.camera_params_id_to_camera_params.items()
    }
    stereo_pairs: tuple[StereoPair, ...] = tuple(
        StereoPair(
            left_camera_params_id=int(pair.left_camera_param_id),
            right_camera_params_id=int(pair.right_camera_param_id),
            baseline_meters=float(pair.baseline_meters),
        )
        for pair in message.stereo_pair
    )
    initial_pose_type: str = resolved_schema.enum_value_name("protos.visual.general.InitialPoseType", message.initial_pose_type)
    return FramesMeta(
        keyframes=keyframes,
        cameras=cameras,
        stereo_pairs=stereo_pairs,
        initial_pose_type=initial_pose_type,
        detector_type=resolved_schema.enum_value_name("protos.common.image.DetectorType", message.detector_type),
        descriptor_type=resolved_schema.enum_value_name("protos.common.image.DescriptorType", message.descriptor_type),
        session_name_by_camera_params_id={int(key): value for key, value in message.camera_params_id_to_session_name.items()},
        track_name_by_track_id={int(key): value for key, value in message.track_id_to_track_name.items()},
        bag_name_by_track_id={int(key): value for key, value in message.track_id_to_bag_name.items()},
        keypoint_feature_has_depth=bool(message.keypoint_feature_has_depth),
        local_sector_key=message.local_sector_key,
        message=message,
    )


def read_frames_meta(path: Path, schema: CusfmSchema | None = None) -> FramesMeta:
    """Parse a `frames_meta.json` file.

    Args:
        path: The JSON file.
        schema: Loaded cuSFM schema; the vendored one by default.

    Returns:
        The typed metadata.

    Raises:
        json_format.ParseError: When the file has fields the schema does not know.
    """
    resolved_schema: CusfmSchema = load_schema() if schema is None else schema
    collection_class = resolved_schema.message_class(KEYFRAMES_METADATA_COLLECTION)
    return parse_message(json_format.Parse(path.read_text(), collection_class()), resolved_schema)


def write_frames_meta(path: Path, frames_meta: FramesMeta) -> None:
    """Write a `frames_meta.json` file.

    Args:
        path: Destination; parent directories are created.
        frames_meta: The metadata to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frames_meta.to_json() + "\n")

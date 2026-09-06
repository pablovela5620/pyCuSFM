"""The dataset-independent container, and the input contract cuSFM is handed.

`PreparedSequence` is what every dataset adapter must produce and what every
later stage consumes: one rig, its cameras in entity order, one canonical
timestamp per synchronised sample, a pose per sample, and a JPEG path per camera
per sample. Once a dataset is in this shape, nothing downstream needs to know
which dataset it was.

`write_frames_meta` is the other half of that contract — the same container
serialised into the `frames_meta.json` cuSFM reads. It lives here because the
two must change together: a field added to the container that never reaches the
file is a field cuSFM does not see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float64, Int
from numpy import ndarray
from simplecv.rig import SensorKind

from demo.poses import to_axis_angle_dict


@dataclass
class PreparedCamera:
    """One camera of a prepared sequence, in rig-relative form."""

    name: str
    """Human role label, e.g. ``left_front``."""
    rig_T_cam: Float64[ndarray, "4 4"]
    """Static extrinsic: camera pose in the rig frame."""
    k_matrix: Float64[ndarray, "3 3"]
    """Pinhole intrinsics."""
    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""
    distortion: tuple[float, ...] = ()
    """Kannala-Brandt ``k1..k4`` for fisheye cameras; empty for pinhole."""
    used_for_sfm: bool = True
    """False for cameras logged as frusta but withheld from the reconstruction."""
    kind: SensorKind = "rgb"
    """Image content as recorded upstream (RoboCap's cameras are ``grayscale``)."""


@dataclass
class PreparedSequence:
    """Everything cuSFM and Rerun need, independent of the source dataset."""

    name: str
    """Short dataset label used in entity metadata and run directories."""
    cameras: list[PreparedCamera]
    """All cameras on the rig, in entity order."""
    timestamps_ns: Int[ndarray, "n_samples"]
    """One canonical timestamp per synchronised sample."""
    world_T_rig: Float64[ndarray, "n_samples 4 4"]
    """Input trajectory: rig pose in world, per sample."""
    image_paths: dict[str, list[Path | None]]
    """Per-camera JPEG path per sample (``None`` where a frame is missing)."""
    input_dir: Path
    """Directory containing the per-camera image folders + ``frames_meta.json``."""
    frames_meta_path: Path
    """The ``frames_meta.json`` handed to cuSFM."""
    image_plane_distance: float
    """Frustum length in metres."""
    projection_model: str
    """``PINHOLE`` or ``OPENCV_FISHEYE``."""
    full_timestamps_ns: Int[ndarray, "n_frames"] | None = None
    """Every source frame timestamp (30 Hz), not just the sampled keyframes.
    Drives the continuous video + pose so the viewer shows the whole walk, while
    cuSFM still only ever receives the keyframes."""
    full_world_T_rig: Float64[ndarray, "n_frames 4 4"] | None = None
    """Rig pose at every ``full_timestamps_ns`` entry."""
    keyframe_mask: Bool[ndarray, "n_frames"] | None = None
    """True where a full-rate frame is one of the keyframes handed to cuSFM."""
    video_samples: dict[str, list[tuple[int, bytes]]] = field(default_factory=dict)
    """Per-camera raw H.264 samples, re-emitted verbatim as a ``VideoStream``."""
    reference_name: str = "imu_00"
    """True body-frame origin recorded in the rig metadata.

    We deliberately follow `COLMAP's rig convention <https://colmap.github.io/rigs.html>`_
    and give ``RigCalibration`` a camera reference sensor even though the physical
    origin is RoboCap's IMU or Galileo's vehicle frame. The viewer therefore tints
    ``cam_00`` as the reference camera; this value overrides the schema metadata
    with the true origin instead. Do not make the tint and metadata agree by moving
    the rig frame to the camera."""

    @property
    def sfm_cameras(self) -> list[PreparedCamera]:
        """The cameras cuSFM was actually given, in entity order.

        Distinct from ``cameras``, which is the whole rig as logged to Rerun. On
        RoboCap those differ -- six on the rig, four to cuSFM -- and conflating
        them is silent: ``len(cameras)`` was once used as the divisor for cuSFM
        keyframe ids, scaling every loop-closure chord by 4/6 with nothing
        visibly wrong. Prefer this property."""
        return [camera for camera in self.cameras if camera.used_for_sfm]



def write_frames_meta(sequence: PreparedSequence, stereo_pairs: tuple[tuple[str, str], ...]) -> Path:
    """Write cuSFM's ``frames_meta.json`` input contract.

    ``camera_to_world`` is the **camera** pose in world (verified against
    galileo's ``ground_truth.txt``: ``camera_to_world @ inv(sensor_to_vehicle)``
    reproduces the GT vehicle pose to ~1.6 mm), and
    ``sensor_to_vehicle_transform`` is ``rig_T_cam``.
    """
    sfm_cameras: list[PreparedCamera] = sequence.sfm_cameras
    camera_params: dict[str, dict] = {}
    name_to_id: dict[str, str] = {}

    for camera_id, camera in enumerate(sfm_cameras):
        params_id: str = str(camera_id)
        name_to_id[camera.name] = params_id
        calibration: dict = {
            "image_width": camera.width,
            "image_height": camera.height,
        }
        if sequence.projection_model == "OPENCV_FISHEYE":
            calibration["camera_matrix"] = {
                "data": [float(value) for value in camera.k_matrix.reshape(-1)],
                "row_count": 3,
                "column_count": 3,
            }
            calibration["distortion_coefficients"] = {
                "data": [float(value) for value in camera.distortion],
                "row_count": 1,
                "column_count": len(camera.distortion),
            }
        else:
            projection: Float64[ndarray, "3 4"] = np.hstack([camera.k_matrix, np.zeros((3, 1))])
            calibration["projection_matrix"] = {
                "data": [float(value) for value in projection.reshape(-1)],
                "row_count": 3,
                "column_count": 4,
            }
        camera_params[params_id] = {
            "sensor_meta_data": {
                "sensor_id": camera_id,
                "sensor_type": "CAMERA",
                "sensor_name": camera.name,
                "sensor_to_vehicle_transform": to_axis_angle_dict(camera.rig_T_cam),
            },
            "calibration_parameters": calibration,
            "camera_projection_model_type": sequence.projection_model,
        }

    keyframes: list[dict] = []
    for sample_index, timestamp_ns in enumerate(sequence.timestamps_ns):
        for camera in sfm_cameras:
            image_path: Path | None = sequence.image_paths[camera.name][sample_index]
            if image_path is None:
                continue
            world_T_cam: Float64[ndarray, "4 4"] = sequence.world_T_rig[sample_index] @ camera.rig_T_cam
            keyframes.append(
                {
                    "id": str(len(keyframes)),
                    "camera_params_id": name_to_id[camera.name],
                    "timestamp_microseconds": str(int(timestamp_ns) // 1000),
                    "image_name": f"{camera.name}/{image_path.name}",
                    "camera_to_world": to_axis_angle_dict(world_T_cam),
                    "synced_sample_id": str(sample_index),
                }
            )

    pair_entries: list[dict] = []
    by_name: dict[str, PreparedCamera] = {camera.name: camera for camera in sfm_cameras}
    for left_name, right_name in stereo_pairs:
        if left_name not in name_to_id or right_name not in name_to_id:
            continue
        baseline: float = float(
            np.linalg.norm(by_name[left_name].rig_T_cam[:3, 3] - by_name[right_name].rig_T_cam[:3, 3])
        )
        pair_entries.append(
            {
                "left_camera_param_id": name_to_id[left_name],
                "right_camera_param_id": name_to_id[right_name],
                "baseline_meters": baseline,
            }
        )

    frames_meta: dict = {
        "keyframes_metadata": keyframes,
        "initial_pose_type": "EGO_MOTION",
        "camera_params_id_to_camera_params": camera_params,
        "stereo_pair": pair_entries,
    }
    sequence.frames_meta_path.parent.mkdir(parents=True, exist_ok=True)
    sequence.frames_meta_path.write_text(json.dumps(frames_meta, indent=2))
    print(
        f"  frames_meta.json: {len(keyframes)} keyframes, {len(camera_params)} cameras "
        f"({sequence.projection_model}), {len(pair_entries)} stereo pair(s)"
    )
    return sequence.frames_meta_path


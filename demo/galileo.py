"""Galileo: upstream's bundled sample, already in cuSFM's own format.

Eight PINHOLE cameras on a ground robot, 226 keyframes, and a `ground_truth.txt`
trajectory — the only one of the two datasets that ships a real reference. The
`frames_meta.json` is used exactly as upstream wrote it, so this adapter only
reads: it never rewrites the file cuSFM is given.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from demo.cusfm import CUSFM_FRAME_META_FILE, RunConfig
from demo.poses import from_axis_angle_dict
from demo.sequence import PreparedCamera, PreparedSequence


@dataclass
class GalileoConfig:
    """Reconstruct the bundled r2b_galileo sample (8 pinhole cameras, has GT)."""

    input_dir: Path = Path("data/r2b_galileo")
    """Upstream's sample directory, complete with ``frames_meta.json``."""
    image_stride: int = 1
    """Log JPEG imagery every Nth sample; 1 = exactly what cuSFM was given."""
    image_plane_distance: float = 0.25
    """Frustum length in metres (a ground robot, so larger than the headset)."""



def prepare_galileo(config: GalileoConfig, run: RunConfig) -> PreparedSequence:
    """Read upstream's bundled ``frames_meta.json`` into the common container."""
    print(f"[galileo] reading {config.input_dir}")
    frames_meta: dict = json.loads((config.input_dir / CUSFM_FRAME_META_FILE).read_text())
    camera_params: dict = frames_meta["camera_params_id_to_camera_params"]

    cameras: list[PreparedCamera] = []
    id_to_name: dict[str, str] = {}
    for params_id in sorted(camera_params, key=int):
        entry: dict = camera_params[params_id]
        sensor: dict = entry["sensor_meta_data"]
        calibration: dict = entry["calibration_parameters"]
        projection: Float64[ndarray, "3 4"] = np.asarray(
            calibration["projection_matrix"]["data"], dtype=np.float64
        ).reshape(3, 4)
        id_to_name[params_id] = sensor["sensor_name"]
        cameras.append(
            PreparedCamera(
                name=sensor["sensor_name"],
                rig_T_cam=from_axis_angle_dict(sensor["sensor_to_vehicle_transform"]),
                k_matrix=projection[:3, :3],
                width=int(calibration["image_width"]),
                height=int(calibration["image_height"]),
                used_for_sfm=True,
            )
        )

    # Upstream's frames_meta.json is proto3-serialised, and proto3 drops
    # zero-valued scalar fields — so `camera_params_id: "0"` is simply absent
    # from every keyframe of camera 0 (29 of galileo's 226, all
    # back_stereo_camera_left). Default it back to "0", then cross-check against
    # the image directory, which equals the sensor name for every camera here.
    name_to_id: dict[str, str] = {name: params_id for params_id, name in id_to_name.items()}

    def keyframe_camera_id(keyframe: dict) -> str:
        params_id: str = keyframe.get("camera_params_id", "0")
        directory: str = keyframe["image_name"].split("/")[0]
        expected: str | None = name_to_id.get(directory)
        if expected is not None and expected != params_id:
            raise ValueError(
                f"{keyframe['image_name']}: camera_params_id {params_id!r} maps to "
                f"{id_to_name[params_id]!r}, which contradicts the image directory."
            )
        return params_id

    # Group keyframes by synced_sample_id -> one rig sample per group.
    grouped: dict[str, list[dict]] = {}
    for keyframe in frames_meta["keyframes_metadata"]:
        grouped.setdefault(keyframe["synced_sample_id"], []).append(keyframe)
    sample_ids: list[str] = sorted(grouped, key=int)

    timestamps: list[int] = []
    world_T_rig_list: list[Float64[ndarray, "4 4"]] = []
    image_paths: dict[str, list[Path | None]] = {camera.name: [] for camera in cameras}
    by_name: dict[str, PreparedCamera] = {camera.name: camera for camera in cameras}

    for sample_id in sample_ids:
        keyframes: list[dict] = grouped[sample_id]
        rig_estimates: list[Float64[ndarray, "4 4"]] = []
        present: dict[str, Path] = {}
        for keyframe in keyframes:
            name: str = id_to_name[keyframe_camera_id(keyframe)]
            world_T_cam: Float64[ndarray, "4 4"] = from_axis_angle_dict(keyframe["camera_to_world"])
            rig_estimates.append(world_T_cam @ np.linalg.inv(by_name[name].rig_T_cam))
            present[name] = config.input_dir / keyframe["image_name"]
        timestamps.append(int(keyframes[0]["timestamp_microseconds"]) * 1000)
        world_T_rig_list.append(rig_estimates[0])
        for camera in cameras:
            image_paths[camera.name].append(present.get(camera.name))

    sequence: PreparedSequence = PreparedSequence(
        name="galileo",
        cameras=cameras,
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        world_T_rig=np.stack(world_T_rig_list),
        image_paths=image_paths,
        input_dir=config.input_dir,
        frames_meta_path=config.input_dir / CUSFM_FRAME_META_FILE,
        image_plane_distance=config.image_plane_distance,
        projection_model="PINHOLE",
        reference_name="vehicle_00",
    )
    print(f"  {len(sequence.timestamps_ns)} samples, {len(cameras)} cameras (upstream frames_meta.json used as-is)")
    return sequence


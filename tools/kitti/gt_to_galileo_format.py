# SPDX-License-Identifier: Apache-2.0
"""Convert KITTI odometry ground truth into the TUM `ground_truth.txt` the harness reads.

`colsfm.benchmark.read_ground_truth` looks for a TUM file named `ground_truth.txt`
beside a dataset's `frames_meta.json`, whose body frame is the vehicle frame of
the same world the input `camera_to_world` poses live in. KITTI ships
`poses_gt_<seq>.txt` instead: one flattened 3x4 `world_T_cam0` per frame, with
the timestamps in a separate `times.txt`. For the KITTI `frames_meta.json` that
`data/kitti/get_framemeta_file_for_KITTI.py` generates, camera 0 has an identity
`sensor_to_vehicle_transform`, so vehicle == camera 0 and the conversion is a
pure reformat: no frame change, only rotation-matrix to quaternion and seconds
to microseconds.

The rotation goes through `scipy.spatial.transform.Rotation`, which
re-orthonormalises; KITTI's printed matrices are only accurate to seven digits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm.geometry import format_tum_line

QuaternionXYZW: TypeAlias = Float64[ndarray, "4"]

MICROSECONDS_PER_SECOND: Final[float] = 1e6
"""`times.txt` holds seconds; cuSFM keys everything by integer microseconds."""

KITTI_POSE_COLUMNS: Final[int] = 12
"""A KITTI ground-truth row is a 3x4 matrix flattened row-major."""


@dataclass(frozen=True)
class Config:
    """Write a KITTI sequence's ground truth as a TUM `ground_truth.txt`."""

    sequence_dir: Path
    """KITTI sequence directory holding `times.txt` and `poses_gt_<seq>.txt`."""
    output: Path
    """Destination TUM file, normally `<input dir>/ground_truth.txt`."""
    ground_truth_name: str | None = None
    """Ground-truth file name; defaults to `poses_gt_<sequence dir name>.txt`."""


def convert(*, poses_path: Path, times_path: Path, output: Path) -> int:
    """Write the KITTI ground truth as TUM lines.

    Args:
        poses_path: `poses_gt_<seq>.txt`, one flattened 3x4 `world_T_cam0` per line.
        times_path: `times.txt`, one timestamp in seconds per line.
        output: Destination TUM file; parent directories are created.

    Returns:
        The number of poses written.

    Raises:
        ValueError: When the pose file and the timestamp file differ in length.
    """
    pose_rows: Float64[ndarray, "n 12"] = np.loadtxt(poses_path, dtype=np.float64).reshape(-1, KITTI_POSE_COLUMNS)
    seconds: Float64[ndarray, "n"] = np.loadtxt(times_path, dtype=np.float64).reshape(-1)
    if len(pose_rows) != len(seconds):
        raise ValueError(f"{poses_path} has {len(pose_rows)} rows but {times_path} has {len(seconds)}")

    world_T_cam0: Float64[ndarray, "n 3 4"] = pose_rows.reshape(-1, 3, 4)
    lines: list[str] = []
    for index in range(len(world_T_cam0)):
        rotation_matrix: Float64[ndarray, "3 3"] = world_T_cam0[index, :, :3]
        translation_xyz: Float64[ndarray, "3"] = world_T_cam0[index, :, 3]
        quaternion_xyzw: QuaternionXYZW = Rotation.from_matrix(rotation_matrix).as_quat()
        # Truncation, not rounding: get_framemeta_file_for_KITTI.py uses int(seconds * 1e6).
        timestamp_microseconds: int = int(seconds[index] * MICROSECONDS_PER_SECOND)
        world_T_body: pycolmap.Rigid3d = pycolmap.Rigid3d(pycolmap.Rotation3d(quaternion_xyzw), translation_xyz)
        lines.append(format_tum_line(timestamp_microseconds, world_T_body))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{line}\n" for line in lines))
    return len(lines)


def main(config: Config) -> None:
    """Convert one sequence and report where the file landed.

    Args:
        config: Parsed command-line configuration.
    """
    ground_truth_name: str = config.ground_truth_name or f"poses_gt_{config.sequence_dir.name}.txt"
    written: int = convert(
        poses_path=config.sequence_dir / ground_truth_name,
        times_path=config.sequence_dir / "times.txt",
        output=config.output,
    )
    print(f"wrote {written} TUM poses to {config.output}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))

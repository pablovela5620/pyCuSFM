"""Reading back what cuSFM wrote, and what those files imply about the rig.

Four artifacts, four readers, plus the two derivations that turn per-image
camera poses into the rig quantities the exoego schema wants:

- the COLMAP sparse model, through `colsfm.colmap_text_model` — the one parser,
  shared with the pycolmap side but needing none of pycolmap;
- `output_poses/merged_pose_file.tum`, cuSFM's own vehicle-frame trajectory;
- `pose_graph/pose_graph.pb.txt`, for the loop closures it found;
- and from the model, `world_T_rig(t)` and the refined `rig_T_cam`, each with
  the spread across frames that says whether the rig stayed rigid.

Nothing here logs anything; nothing here runs cuSFM.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm.colmap_text_model import ColmapImage, ColmapModel, parse_colmap_images_text, read_colmap_model
from demo.cusfm import CUSFM_MERGED_POSE_FILE, CUSFM_OUTPUT_POSES_DIR
from demo.poses import compose
from demo.sequence import PreparedCamera, PreparedSequence

__all__ = [
    "ColmapImage",
    "ColmapModel",
    "cusfm_rig_trajectory",
    "parse_colmap_images_text",
    "read_colmap_model",
    "read_cusfm_vehicle_poses",
    "read_loop_closures",
    "refined_extrinsics",
]


def read_cusfm_vehicle_poses(sparse_dir: Path) -> tuple[Int[ndarray, "n"], Float64[ndarray, "n 4 4"]] | None:
    """Read cuSFM's own vehicle-frame (== rig-frame) trajectory, if it wrote one.

    ``extract_pose_from_map_main --export_pose_in_vehicle_frame=True`` emits
    ``output_poses/merged_pose_file.tum``. This is the correct source for
    ``world_T_rig`` once ``--run.optimize-extrinsics`` is on: deriving the rig pose
    from a camera pose and the *input* ``rig_T_cam`` would be wrong precisely
    because bundle adjustment has just changed that extrinsic.
    """
    tum: Path = sparse_dir.parent / CUSFM_OUTPUT_POSES_DIR / CUSFM_MERGED_POSE_FILE
    if not tum.is_file():
        return None
    rows: list[list[str]] = [line.split() for line in tum.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    # cuSFM receives integer microseconds in frames_meta.json, converts them to
    # floating-point seconds, then prints many decimal places in TUM. Parse the
    # text directly and recover that integer-microsecond contract: float64 cannot
    # retain nanoseconds at Galileo's ~1.7e9-second epoch, while rounding the
    # printed seconds to nanoseconds would preserve conversion noise instead of
    # the input timestamp.
    times_ns: Int[ndarray, "n"] = np.asarray(
        [
            int((Decimal(row[0]) * Decimal(1_000_000)).to_integral_value(rounding=ROUND_HALF_UP))
            * 1000
            for row in rows
        ],
        dtype=np.int64,
    )
    raw: Float64[ndarray, "n 7"] = np.asarray(
        [[float(value) for value in row[1:8]] for row in rows], dtype=np.float64
    )
    poses: Float64[ndarray, "n 4 4"] = np.stack(
        [compose(Rotation.from_quat(row[3:7]).as_matrix(), row[0:3]) for row in raw]
    )
    return times_ns, poses


def refined_extrinsics(
    sequence: PreparedSequence,
    model: ColmapModel,
    rig_times_ns: Int[ndarray, "n"],
    rig_poses: Float64[ndarray, "n 4 4"],
) -> tuple[dict[str, Float64[ndarray, "4 4"]], float]:
    """Recover each camera's ``rig_T_cam`` as bundle adjustment left it.

    ``rig_T_cam = world_T_rig(t)^-1 @ world_T_cam(t)``, averaged over every frame
    where both are known. Returns the per-camera extrinsics and the worst spread
    across frames — near zero means the rig stayed rigid, which the exoego schema
    requires.
    """
    lookup: dict[int, int] = {int(timestamp_ns): i for i, timestamp_ns in enumerate(rig_times_ns)}
    out: dict[str, Float64[ndarray, "4 4"]] = {}
    worst: float = 0.0
    matched_rows: int = 0
    for camera in sequence.sfm_cameras:
        estimates: list[Float64[ndarray, "4 4"]] = []
        for sample_index, timestamp in enumerate(sequence.timestamps_ns):
            image_path: Path | None = sequence.image_paths[camera.name][sample_index]
            if image_path is None:
                continue
            world_T_cam: Float64[ndarray, "4 4"] | None = model.world_T_cam.get(
                f"{camera.name}/{image_path.name}"
            )
            # Match the exact timestamp contract handed to cuSFM. RoboCap keeps
            # finer nanoseconds in its video names, but frames_meta.json carries
            # integer microseconds; Galileo already starts at that precision.
            timestamp_key_ns: int = (int(timestamp) // 1000) * 1000
            row: int | None = lookup.get(timestamp_key_ns)
            if world_T_cam is None or row is None:
                continue
            matched_rows += 1
            estimates.append(np.linalg.inv(rig_poses[row]) @ world_T_cam)
        if not estimates:
            continue
        stack: Float64[ndarray, "k 4 4"] = np.stack(estimates)
        translations: Float64[ndarray, "k 3"] = stack[:, :3, 3]
        worst = max(worst, float(np.linalg.norm(translations - translations.mean(axis=0), axis=1).max()))
        mean_rotation = Rotation.from_matrix(stack[:, :3, :3]).mean().as_matrix()
        out[camera.name] = compose(mean_rotation, translations.mean(axis=0))
    if matched_rows == 0:
        print(
            "  WARNING: refined-extrinsics timestamp join matched zero rows; "
            "no refined camera extrinsics can be recovered"
        )
    return out, worst


def cusfm_rig_trajectory(
    sequence: PreparedSequence, model: ColmapModel
) -> tuple[Int[ndarray, "m"], Float64[ndarray, "m 4 4"], float, dict[str, int]]:
    """Recover ``world_T_rig(t)`` from cuSFM's per-image camera poses.

    With ``ba_frame_type=vehicle_rig`` the rig stays rigid, so every camera in a
    sample yields the same ``world_T_rig = world_T_cam @ inv(rig_T_cam)``. The
    spread between those estimates is returned as a rigidity check — if it is
    not near zero, the rig was not treated as rigid and the exoego schema's
    single ``world_T_rig`` would be a lie.
    """
    by_name: dict[str, PreparedCamera] = {camera.name: camera for camera in sequence.cameras}
    per_camera_registered: dict[str, int] = {name: 0 for name in by_name}
    sample_indices: list[int] = []
    rig_poses: list[Float64[ndarray, "4 4"]] = []
    spreads: list[float] = []
    # Loop-invariant: four static extrinsics, otherwise re-inverted once per
    # (sample, camera) -- ~4500 redundant 4x4 inversions on a stride-4 run.
    cam_T_rig: dict[str, Float64[ndarray, "4 4"]] = {
        camera.name: np.linalg.inv(camera.rig_T_cam) for camera in sequence.sfm_cameras
    }

    for sample_index in range(len(sequence.timestamps_ns)):
        estimates: list[Float64[ndarray, "4 4"]] = []
        for camera in sequence.sfm_cameras:
            image_path: Path | None = sequence.image_paths[camera.name][sample_index]
            if image_path is None:
                continue
            key: str = f"{camera.name}/{image_path.name}"
            pose: Float64[ndarray, "4 4"] | None = model.world_T_cam.get(key)
            if pose is None:
                continue
            per_camera_registered[camera.name] += 1
            estimates.append(pose @ cam_T_rig[camera.name])
        if not estimates:
            continue
        translations: Float64[ndarray, "k 3"] = np.stack([estimate[:3, 3] for estimate in estimates])
        if len(estimates) > 1:
            spreads.append(float(np.linalg.norm(translations - translations.mean(axis=0), axis=1).max()))
        sample_indices.append(sample_index)
        rig_poses.append(estimates[0])

    max_spread: float = max(spreads) if spreads else 0.0
    return (
        np.asarray(sample_indices, dtype=np.int64),
        np.stack(rig_poses) if rig_poses else np.zeros((0, 4, 4)),
        max_spread,
        per_camera_registered,
    )



def read_loop_closures(pose_graph_dir: Path) -> Int[ndarray, "m 2"] | None:
    """Extract the sample-index pairs that cuSFM's pose graph closed as loops.

    ``pose_graph.pb.txt`` is a flat list of ``pairs``, each a relative pose plus a
    ``type``. Only ``LOOP`` matters here; the run we ship also carries 4524
    ``CONSECUTIVE`` (odometry) and 13572 ``EXTRINSIC`` (inter-camera) pairs, both
    already implied by the trajectory and the rig, so drawing them would be noise.

    ``frame_id`` is a *global keyframe* index that strides by the number of cameras
    cuSFM was given, so dividing by that stride recovers the sample index the
    trajectory arrays use. The stride is measured from the ``CONSECUTIVE`` pairs
    rather than taken from the rig, because those two numbers differ: this rig
    logs six cameras to Rerun while cuSFM only ever received four, and using six
    silently scales every chord by 4/6 and lands it on the wrong pose.

    Loops are usually cross-camera -- frame 1168 -> 313 is sample 292 camera 0
    against sample 78 camera 1 -- but for *showing where the robot revisited a
    place* the rig position is the honest endpoint, so both ends collapse to their
    sample.

    Ids are read defensively: proto3 omits zero-valued scalars, so a pair against
    keyframe 0 has no ``source_frame_id`` line at all.
    """
    path: Path = pose_graph_dir / "pose_graph.pb.txt"
    if not path.is_file():
        return None
    loops: list[tuple[int, int]] = []
    strides: set[int] = set()
    source: int = 0
    target: int = 0
    depth: int = 0
    with path.open() as handle:
        for line in handle:
            stripped: str = line.strip()
            # Track real brace depth. A flat "am I inside frame_pair" flag breaks on
            # any nested submessage: the nested `}` clears it, so later `type:` lines
            # are attributed to whatever ids were last seen.
            if stripped.endswith("{"):
                depth += 1
                if depth == 2 and stripped.startswith("frame_pair"):
                    source, target = 0, 0
                continue
            if stripped == "}":
                depth -= 1
                continue
            if depth == 2 and stripped.startswith("source_frame_id:"):
                source = int(stripped.partition(":")[2])
            elif depth == 2 and stripped.startswith("target_frame_id:"):
                target = int(stripped.partition(":")[2])
            elif depth == 1 and stripped == "type: LOOP":
                loops.append((source, target))
            elif depth == 1 and stripped == "type: CONSECUTIVE":
                strides.add(abs(source - target))
    if not loops:
        return None
    # Every consecutive edge should span exactly one sample, i.e. the number of
    # cameras cuSFM was given. Disagreement means the keyframe ids are not a
    # uniform grid (a dropped image shifts them), and dividing by the minimum
    # would scale every chord. Refuse to guess rather than draw wrong chords.
    if len(strides) != 1 or 0 in strides:
        print(f"  WARNING: pose graph consecutive spans are not uniform ({sorted(strides)}); skipping loop chords")
        return None
    return np.asarray(loops, dtype=np.int64) // strides.pop()


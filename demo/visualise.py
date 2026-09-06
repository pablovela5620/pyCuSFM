"""Everything logged to Rerun, and the blueprint that lays it out.

Two rigs under one world, following the exoego:v2 schema: `rig_00` carries the
input trajectory and the imagery, `rig_01` the cuSFM-refined one. The colours,
radii and frustum widths here are the whole visual language of the recording —
which frames the solver was given, which cameras it never saw, which way the
walk ran, and where it closed a loop — so they live beside the calls that use
them rather than at the top of a 2000-line file.

`clip_outliers` is in this module for the same reason: it is a display filter
and nothing else. It never touches the reconstruction or a reported metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float, Float64, Int
from numpy import ndarray
from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)
from simplecv.rerun_rig_logger import log_rig_pose_stream, log_rig_static
from simplecv.rig import CameraSensor, Rig, RigCalibration, RigPoseStream, entity_id

from demo.schema import INPUT_RIG_INDEX, TIMELINE
from demo.sequence import PreparedCamera, PreparedSequence

EXCLUDED_CAM_COLOR: tuple[int, int, int, int] = (110, 110, 110, 160)
"""Grey frustum tint for cameras present on the rig but excluded from the SfM."""

CUSFM_RIG_COLOR: tuple[int, int, int, int] = (255, 150, 40, 255)
"""Orange frustum tint distinguishing the cuSFM rig from the input rig."""

#: Endpoint colours, matching the RoboCap catalog's slam layer exactly.
TRAJECTORY_START_COLOR: tuple[int, int, int, int] = (0, 255, 0, 255)
TRAJECTORY_END_COLOR: tuple[int, int, int, int] = (255, 0, 0, 255)

LOOP_CLOSURE_COLOR: tuple[int, int, int, int] = (235, 60, 70, 255)
"""Red, deliberately clashing with both trajectory hues (blue input, orange cuSFM)
so a chord never reads as part of the path it cuts across."""
LOOP_CLOSURE_RADIUS: float = 0.008
"""Twice the trajectory's 0.004 so the chords stay legible where they bundle."""

#: Frustum styling for frames cuSFM actually consumed, vs the ones it never saw.
KEYFRAME_COLOR: tuple[int, int, int, int] = (80, 230, 120, 255)
NON_KEYFRAME_COLOR: tuple[int, int, int, int] = (120, 124, 140, 110)
#: Frustum edge thickness as a FRACTION of image_plane_distance. `line_width` is
#: in **scene units**, not pixels: Rerun's own Pinhole example pairs
#: `image_plane_distance=0.1` with `line_width=0.003`. Absolute values here (0.6,
#: 2.5) drew edges metres thick and the frustum became an unreadable blob.
KEYFRAME_WIDTH_FRACTION: float = 0.040
NON_KEYFRAME_WIDTH_FRACTION: float = 0.012


def clip_outliers(
    positions: Float[ndarray, "n 3"], percentile: float
) -> Bool[ndarray, "n"]:
    """Per-axis percentile band around the median; ``True`` marks points to keep."""
    if percentile >= 100.0 or len(positions) == 0:
        return np.ones(len(positions), dtype=bool)
    low: Float64[ndarray, "3"] = np.percentile(positions, 100.0 - percentile, axis=0)
    high: Float64[ndarray, "3"] = np.percentile(positions, percentile, axis=0)
    return np.all((positions >= low) & (positions <= high), axis=1)


def build_rig(
    sequence: PreparedSequence,
    index: int,
    world_T_rig: Float64[ndarray, "n 4 4"],
    extrinsics_override: dict[str, Float64[ndarray, "4 4"]] | None = None,
) -> Rig:
    """Assemble an exoego ``Rig`` from prepared cameras plus a pose stream."""
    sensors: list[CameraSensor] = []
    for camera_index, camera in enumerate(sequence.cameras):
        intrinsics: Intrinsics = Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=camera.k_matrix,
            height=camera.height,
            width=camera.width,
        )
        # `Extrinsics`' "world" slot is the *parent* frame, which here is the rig.
        rig_T_cam: Float64[ndarray, "4 4"] = (
            extrinsics_override.get(camera.name, camera.rig_T_cam)
            if extrinsics_override
            else camera.rig_T_cam
        )
        extrinsics: Extrinsics = Extrinsics(
            world_R_cam=rig_T_cam[:3, :3], world_t_cam=rig_T_cam[:3, 3]
        )
        parameters: PinholeParameters | Fisheye62Parameters
        if camera.distortion:
            k1, k2, k3, k4 = (list(camera.distortion) + [0.0, 0.0, 0.0, 0.0])[:4]
            parameters = Fisheye62Parameters(
                name=camera.name,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                distortion=KannalaBrandtDistortion(k1=k1, k2=k2, k3=k3, k4=k4),
            )
        else:
            parameters = PinholeParameters(name=camera.name, extrinsics=extrinsics, intrinsics=intrinsics)
        sensors.append(
            CameraSensor(index=camera_index, name=camera.name, kind=camera.kind, pinhole=parameters)
        )

    return Rig(
        index=index,
        calibration=RigCalibration(cameras=sensors, reference_index=0),
        pose_stream=RigPoseStream(
            world_t_rig=world_T_rig[:, :3, 3].copy(),
            world_R_rig=world_T_rig[:, :3, :3].copy(),
        ),
        image_plane_distance=sequence.image_plane_distance,
    )


def log_rig(
    rig: Rig,
    sequence: PreparedSequence,
    timestamps_ns: Int[ndarray, "n"],
    *,
    label: str,
    frustum_color: tuple[int, int, int, int] | None,
) -> None:
    """Log a rig's static skeleton, its motion, and its provenance label."""
    log_rig_static(rig)
    log_rig_pose_stream(rig, timestamps_ns=timestamps_ns, timeline=TIMELINE)
    rig_path: str = f"world/{entity_id('rig', rig.index)}"
    # Deliberate COLMAP convention: cam_00 stays the camera reference (and gets
    # the viewer tint), while this override records the true IMU/vehicle origin.
    rr.log(
        rig_path,
        rr.AnyValues(source=label, dataset=sequence.name, reference=sequence.reference_name),
        static=True,
    )

    for camera_index, camera in enumerate(sequence.cameras):
        pinhole_path: str = f"{rig_path}/{entity_id('cam', camera_index)}/pinhole"
        if not camera.used_for_sfm:
            # Present on the rig but withheld from the SfM: grey it out so the
            # viewer shows the real hardware without implying it was used.
            rr.log(
                pinhole_path,
                rr.Pinhole.from_fields(
                    image_plane_distance=rig.image_plane_distance, color=EXCLUDED_CAM_COLOR
                ),
                static=True,
            )
        elif frustum_color is not None:
            rr.log(
                pinhole_path,
                rr.Pinhole.from_fields(
                    image_plane_distance=rig.image_plane_distance, color=frustum_color
                ),
                static=True,
            )


def log_images(sequence: PreparedSequence, rig_index: int, stride: int) -> int:
    """Log JPEG imagery under the input rig's cameras, strided for file size."""
    logged: int = 0
    for camera_index, camera in enumerate(sequence.cameras):
        if not camera.used_for_sfm:
            continue
        entity: str = f"world/{entity_id('rig', rig_index)}/{entity_id('cam', camera_index)}/pinhole/image"
        for sample_index in range(0, len(sequence.timestamps_ns), max(1, stride)):
            image_path: Path | None = sequence.image_paths[camera.name][sample_index]
            if image_path is None or not image_path.is_file():
                continue
            rr.set_time(TIMELINE, duration=1e-9 * float(sequence.timestamps_ns[sample_index]))
            rr.log(entity, rr.EncodedImage(path=image_path))
            logged += 1
    return logged



def log_loop_closures(
    name: str,
    pairs: Int[ndarray, "m 2"],
    positions: Float[ndarray, "r 3"],
    sample_indices: Sequence[int],
) -> int:
    """Draw one chord per loop closure, straight across the trajectory.

    A loop closure is the one thing in the pose graph you cannot infer by looking
    at the path: it says "these two moments are the same place". Rendering each as
    a red segment between the two rig positions makes them literally visible as
    shortcuts across the walk, and the endpoint dots keep them findable when many
    chords overlap in a tight revisit.

    ``pairs`` holds *sample* indices while ``positions`` is indexed by *registered
    row*, and the two only coincide when every sample registered. ``sample_indices``
    is the row-to-sample map from ``cusfm_rig_trajectory``; it is required rather
    than optional because indexing one space with the other is silent -- the chords
    simply land on the wrong poses.

    Returns the number of chords drawn, which can be lower than ``len(pairs)`` when
    the pose graph references samples the reconstruction dropped.
    """
    row_of_sample: dict[int, int] = {sample: row for row, sample in enumerate(sample_indices)}
    rows: list[tuple[int, int]] = [
        (row_of_sample[a], row_of_sample[b])
        for a, b in pairs.tolist()
        if a in row_of_sample and b in row_of_sample
    ]
    if not rows:
        return 0
    kept: Int[ndarray, "k 2"] = np.asarray(rows, dtype=np.int64)
    segments: Float64[ndarray, "k 2 3"] = np.stack(
        [positions[kept[:, 0]], positions[kept[:, 1]]], axis=1
    )
    rr.log(
        f"world/runs/{name}/loop_closures",
        rr.LineStrips3D(segments, colors=LOOP_CLOSURE_COLOR, radii=LOOP_CLOSURE_RADIUS),
        static=True,
    )
    rr.log(
        f"world/runs/{name}/loop_closures/endpoints",
        rr.Points3D(
            positions=segments.reshape(-1, 3),
            colors=LOOP_CLOSURE_COLOR,
            radii=LOOP_CLOSURE_RADIUS * 2.0,
        ),
        static=True,
    )
    return len(kept)


def log_trajectory(
    name: str,
    positions: Float[ndarray, "n 3"],
    *,
    source: str,
    hue: tuple[int, int, int],
    keyframe_positions: Float[ndarray, "k 3"] | None = None,
) -> None:
    """Log a trajectory polyline the way the RoboCap catalog's slam layer does.

    Entity layout mirrors ``/world/runs/basalt`` in the source recording:
    ``runs/<name>`` carries provenance, ``runs/<name>/trajectory`` is a *static*
    ``LineStrips3D`` of ``n-1`` two-point segments (per-segment colour, so travel
    direction is readable), and ``runs/<name>/endpoints`` marks start and end.
    """
    if len(positions) < 2:
        return
    segments: Float64[ndarray, "n-1 2 3"] = np.stack([positions[:-1], positions[1:]], axis=1)
    # Fade from dark to full hue along the path, so direction of travel is visible.
    ramp: Float64[ndarray, "n-1"] = np.linspace(0.35, 1.0, len(segments))
    colors: Int[ndarray, "n-1 4"] = np.stack(
        [
            (ramp * hue[0]).astype(np.uint8),
            (ramp * hue[1]).astype(np.uint8),
            (ramp * hue[2]).astype(np.uint8),
            np.full(len(segments), 255, dtype=np.uint8),
        ],
        axis=-1,
    )
    rr.log(f"world/runs/{name}", rr.AnyValues(num_poses=len(positions), source=source), static=True)
    rr.log(
        f"world/runs/{name}/trajectory",
        rr.LineStrips3D(segments, colors=colors, radii=0.004),
        static=True,
    )
    rr.log(
        f"world/runs/{name}/endpoints",
        rr.Points3D(
            positions=np.stack([positions[0], positions[-1]]),
            colors=[TRAJECTORY_START_COLOR, TRAJECTORY_END_COLOR],
            labels=["start", "end"],
            radii=0.02,
        ),
        static=True,
    )
    if keyframe_positions is not None and len(keyframe_positions):
        # One dot per frame cuSFM was actually given. Against the continuous 30 fps
        # line this shows the *spatial* sampling pattern at a glance — dense where
        # the wearer paused, sparse through fast head motion — without scrubbing.
        rr.log(
            f"world/runs/{name}/keyframes",
            rr.Points3D(positions=keyframe_positions, colors=KEYFRAME_COLOR, radii=0.012),
            static=True,
        )


def log_video_streams(sequence: PreparedSequence, rig_index: int) -> int:
    """Re-emit each camera's original H.264 samples as a ``VideoStream``.

    The samples are passed through verbatim — no re-encode — so the viewer shows
    the full 30 fps walk, not the keyframe slideshow. Far cheaper than JPEGs too:
    the whole segment is ~77 MB per camera against ~280 MB of per-sample JPEG.
    """
    total: int = 0
    for camera_index, camera in enumerate(sequence.cameras):
        samples: list[tuple[int, bytes]] = sequence.video_samples.get(camera.name, [])
        if not samples:
            continue
        entity: str = f"world/{entity_id('rig', rig_index)}/{entity_id('cam', camera_index)}/pinhole/video"
        rr.log(entity, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
        timestamps: Int[ndarray, "n"] = np.asarray([t for t, _ in samples], dtype=np.int64)
        rr.send_columns(
            entity,
            indexes=[rr.TimeColumn(TIMELINE, duration=1e-9 * timestamps.astype(np.float64))],
            columns=rr.VideoStream.columns(sample=[raw for _, raw in samples]),
        )
        total += len(samples)
    return total


def log_video_references(sequence: PreparedSequence, source_rig: int, target_rig: int) -> int:
    """Show ``source_rig``'s video under ``target_rig``'s cameras without copying it.

    Rerun 0.37.1 lets a ``VideoFrameReference`` point at a ``VideoStream`` on
    another entity, so the cuSFM rig gets the same imagery as the input rig for
    a few hundred KB of timestamps instead of another ~300 MB of H.264. The
    frames are geometrically valid on both rigs because nothing is rectified:
    both log the raw fisheye under a ``Pinhole`` with distortion coefficients.

    One reference per source sample, on the same timeline as the stream. Measured
    on 0.37.1: a ``VideoFrameReference`` into a stream shows the frame at the
    reference's *own* ``timestamp``, not at the viewer cursor -- a single static
    reference therefore pins one frame forever (and ``0`` clamps to the first
    sample). Per-sample references whose timestamps equal the stream's are what
    make the reference follow the cursor.

    Returns the number of references logged.
    """
    total: int = 0
    for camera_index, camera in enumerate(sequence.cameras):
        samples: list[tuple[int, bytes]] = sequence.video_samples.get(camera.name, [])
        if not samples:
            continue
        source: str = f"world/{entity_id('rig', source_rig)}/{entity_id('cam', camera_index)}/pinhole/video"
        target: str = f"world/{entity_id('rig', target_rig)}/{entity_id('cam', camera_index)}/pinhole/video"
        timestamps: Int[ndarray, "n"] = np.asarray([t for t, _ in samples], dtype=np.int64)
        rr.log(target, rr.VideoFrameReference.from_fields(video_reference=source), static=True)
        rr.send_columns(
            target,
            indexes=[rr.TimeColumn(TIMELINE, duration=1e-9 * timestamps.astype(np.float64))],
            columns=rr.VideoFrameReference.columns(timestamp=timestamps),
        )
        total += len(samples)
    return total


def log_keyframe_highlight(sequence: PreparedSequence, rig_index: int) -> None:
    """Tint each camera's frustum by whether the current frame is a cuSFM keyframe.

    A temporal partial update of ``Pinhole``'s ``color``/``line_width`` — the static
    intrinsics logged by ``log_pinhole`` stay untouched. Green and thick on the
    frames handed to the solver, dim and thin on the frames it never saw, so
    scrubbing shows exactly what cuSFM was given.
    """
    times: Int[ndarray, "n"] | None = sequence.full_timestamps_ns
    mask: Bool[ndarray, "n"] | None = sequence.keyframe_mask
    if times is None or mask is None:
        return
    colors: Int[ndarray, "n 4"] = np.where(
        mask[:, None], np.asarray(KEYFRAME_COLOR), np.asarray(NON_KEYFRAME_COLOR)
    ).astype(np.uint8)
    scale: float = sequence.image_plane_distance
    widths: Float64[ndarray, "n"] = np.where(
        mask, KEYFRAME_WIDTH_FRACTION * scale, NON_KEYFRAME_WIDTH_FRACTION * scale
    )
    for camera_index, camera in enumerate(sequence.cameras):
        if not camera.used_for_sfm:
            continue
        rr.send_columns(
            f"world/{entity_id('rig', rig_index)}/{entity_id('cam', camera_index)}/pinhole",
            indexes=[rr.TimeColumn(TIMELINE, duration=1e-9 * times.astype(np.float64))],
            columns=rr.Pinhole.columns(color=colors, line_width=widths.astype(np.float32)),
        )


def build_blueprint(sequence: PreparedSequence) -> rrb.Blueprint:
    """3D view of both rigs plus the cloud, with per-camera 2D views alongside."""
    sfm_cameras: list[tuple[int, PreparedCamera]] = [
        (index, camera) for index, camera in enumerate(sequence.cameras) if camera.used_for_sfm
    ]
    image_views: list[rrb.SpatialView2DView] = [
        rrb.Spatial2DView(
            origin=f"world/{entity_id('rig', INPUT_RIG_INDEX)}/{entity_id('cam', index)}/pinhole",
            name=camera.name,
        )
        for index, camera in sfm_cameras
    ]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                name="cuSFM reconstruction (rig_00 = input, rig_01 = cuSFM)",
            ),
            rrb.Grid(contents=image_views, name="cameras"),
            column_shares=[2.0, 1.0],
        ),
        rrb.TimePanel(state="expanded"),
    )


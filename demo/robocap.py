"""RoboCap: a frozen catalog segment turned into cuSFM's on-disk input.

Six Kannala-Brandt fisheye cameras on a head-worn rig, with a per-frame basalt
VIO trajectory already in the recording. This adapter owns everything specific
to that: which cameras are world-facing, which pair is genuinely stereo, how the
H.264 samples are pulled out of the `.rrd`, how they become JPEGs on disk, and
how the trim and stride settle which frames cuSFM ever sees.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Float64, Int
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.rerun_log_utils import mux_h264_to_mp4
from simplecv.rig import SensorKind
from simplecv.rrd_query_utils import first_valid_value, unwrap_singleton_lists

from demo.cusfm import CUSFM_FRAME_META_FILE, RunConfig
from demo.poses import compose, interpolate_poses
from demo.schema import CHILD_FROM_PARENT, TIMELINE
from demo.sequence import PreparedCamera, PreparedSequence, write_frames_meta

ROBOCAP_SFM_CAMERAS: tuple[str, ...] = ("left_front", "right_front", "left", "right")
"""World-facing RoboCap cameras. ``left_eye``/``right_eye`` point at the wearer's
eyes and carry no scene content, so they are excluded from the reconstruction."""

ROBOCAP_RIG_ADJACENCY_PAIRS: tuple[tuple[str, str], ...] = (
    ("left", "left_front"),
    ("left_front", "right_front"),
    ("right_front", "right"),
)
"""Rig *adjacency*, not rectifiable stereo. cuSFM expresses a cuVSLAM rig through
``stereo_pair`` entries, so covering all four world-facing cameras requires
declaring the ~90-degree neighbours too. Only use this when cuVSLAM must see the
whole rig: with the single true pair below it builds a 2-camera rig and loses
tracking on this fisheye sequence (`Tracking lost at frame 231` of 912)."""

ROBOCAP_STEREO_PAIR: tuple[str, str] = ("left_front", "right_front")
"""The only genuinely rectifiable RoboCap pair: 86.2 mm baseline, optical axes
2.6 degrees apart, baseline 1.3 degrees off the camera X axis. The other
combinations sit ~90 degrees apart (adjacent) or ~168 degrees (back-to-back)
and are *not* stereo pairs, despite having plausible-looking baselines."""

ROBOCAP_SEGMENT: str = "robocap__f408193e6447b3b0__s00000021"
"""Segment frozen from the (in-memory) catalog into a local ``.rrd``."""

ROBOCAP_CATALOG_URL: str = "rerun+http://127.0.0.1:51235"
"""Catalog to pull the segment from when the local ``.rrd`` is absent."""


@dataclass
class RobocapConfig:
    """Reconstruct a frozen RoboCap catalog segment (fisheye, basalt VIO poses)."""

    rrd_path: Path = Path("data/robocap/s00000021.rrd")
    """Frozen catalog segment (``base`` + ``slam`` layers)."""
    trim_start_seconds: float = 2.0
    """Drop this much from the start. The capture opens on a blown-out white frame
    while exposure settles, which is textureless and pollutes both the vocabulary
    and the first keyframes."""
    trim_end_seconds: float = 2.0
    """Drop this much from the end, where the wearer's hand enters frame to stop
    the capture — a large moving occluder right where the loop should close."""
    frame_stride: int = 4
    """Use every Nth video frame. 4648 frames / 4 = 1132 samples per camera.

    Density dominates reconstruction quality on this sequence, so this is 4 rather
    than a faster coarser value. Measured, 20 against 4: vertical excursion on a
    flat floor 6.02 m against 0.73 m, BA point rejection 71 % against 0.5 %,
    ``left``/``right`` registration 59 %/48 % against 99.9 %/98.2 %, disagreement
    with the input trajectory 1214 mm against 499 mm. Stride 4 was also *faster*
    through bundle adjustment (130 s against 235 s), because the extra
    observations converge better -- the density is not simply bought with time.

    The loop-closure thresholds in ``LOOP_CLOSURE_CONFIG_DIR`` were validated at
    this stride, so raising it also runs those gates outside their tested regime.
    Expect ~20 min end to end; ``--dataset.frame-stride 20`` cuts that to roughly
    6 min at the cost of the reconstruction above."""
    max_samples: int | None = None
    """Cap on samples after striding (``None`` = all)."""
    cameras: tuple[str, ...] = ROBOCAP_SFM_CAMERAS
    """World-facing cameras fed to cuSFM."""
    rig_adjacency_pairs: bool = False
    """Declare all three adjacent camera pairs instead of only the true stereo
    pair, so cuSFM hands cuVSLAM a full four-camera rig. Needed for
    ``--run.no-skip-cuvslam``; leave off for bundle-adjustment-only runs, where a
    non-rectifiable pair is a false claim about the geometry."""
    image_plane_distance: float = 0.15
    """Frustum length in metres. The RoboCap catalog uses 0.025 to draw the
    headset at life size; over a 10 m reconstruction that is a speck, so the
    frusta are drawn larger here. Visual only — no effect on geometry."""



@dataclass
class RrdCameraRecord:
    """Raw per-camera data pulled out of a frozen exoego recording."""

    name: str
    cam_T_rig: Float64[ndarray, "4 4"]
    k_matrix: Float64[ndarray, "3 3"]
    width: int
    height: int
    distortion: tuple[float, ...]
    kind: SensorKind = "rgb"
    samples: list[tuple[int, bytes]] = field(default_factory=list)


def fetch_robocap_segment(rrd_path: Path, catalog_url: str = ROBOCAP_CATALOG_URL) -> Path:
    """Freeze the RoboCap segment from a running catalog into a local ``.rrd``.

    The catalog this was developed against stores segments **in memory**
    (``memory:///store/...``), so a catalog restart loses them. Freezing to disk
    once makes the demo reproducible and removes the catalog from the hot path.
    """
    from rerun.catalog import CatalogClient

    print(f"[robocap] {rrd_path} missing — pulling {ROBOCAP_SEGMENT} from {catalog_url}")
    try:
        client = CatalogClient(catalog_url)
        dataset = client.get_dataset(name="robocap")
        recording = dataset.download_segment(ROBOCAP_SEGMENT)
    except Exception as error:
        raise RuntimeError(
            f"could not fetch {ROBOCAP_SEGMENT} from {catalog_url} ({error}). "
            "Start the catalog, or point --dataset.rrd-path at an existing .rrd."
        ) from error
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    recording.save(str(rrd_path))
    print(f"[robocap] wrote {rrd_path} ({rrd_path.stat().st_size / 1e6:.1f} MB)")
    return rrd_path


def read_robocap_rrd(rrd_path: Path) -> tuple[dict[int, RrdCameraRecord], Int[ndarray, "n"], Rotation, Float64[ndarray, "n 3"]]:
    """Read cameras, video samples, and the basalt trajectory from a frozen segment."""
    from rerun.experimental import RrdReader

    if not rrd_path.is_file():
        fetch_robocap_segment(rrd_path)

    cameras: dict[int, RrdCameraRecord] = {}
    pose_times: list[int] = []
    pose_quats: list[list[float]] = []
    pose_translations: list[list[float]] = []

    def camera_index(entity_path: str) -> int | None:
        parts: list[str] = entity_path.strip("/").split("/")
        for part in parts:
            if part.startswith("cam_"):
                return int(part.removeprefix("cam_"))
        return None

    store = RrdReader(str(rrd_path)).store()
    for chunk in store.stream():
        entity_path: str = str(chunk.entity_path)
        if not entity_path.startswith("/world/rig_00"):
            continue
        record_batch = chunk.to_record_batch()
        component_names: set[str] = set(record_batch.schema.names)

        if entity_path == "/world/rig_00":
            required: set[str] = {"Transform3D:quaternion", "Transform3D:translation", TIMELINE}
            if not required.issubset(component_names):
                continue
            quaternions: pa.Array = record_batch.column("Transform3D:quaternion")
            translations: pa.Array = record_batch.column("Transform3D:translation")
            times: pa.Array = record_batch.column(TIMELINE)
            for quaternion, translation, timestamp in zip(
                quaternions.to_pylist(), translations.to_pylist(), times.to_pylist()
            ):
                if quaternion is None or translation is None or timestamp is None:
                    continue
                pose_times.append(int(getattr(timestamp, "value", timestamp)))
                pose_quats.append(list(unwrap_singleton_lists(quaternion)))
                pose_translations.append(list(unwrap_singleton_lists(translation)))
            continue

        index: int | None = camera_index(entity_path)
        if index is None:
            continue

        if entity_path.endswith("/pinhole/video"):
            if "VideoStream:sample" not in component_names or TIMELINE not in component_names:
                continue
            samples: pa.Array = record_batch.column("VideoStream:sample")
            times = record_batch.column(TIMELINE)
            record: RrdCameraRecord | None = cameras.get(index)
            if record is None:
                record = RrdCameraRecord(f"cam_{index:02d}", np.eye(4), np.eye(3), 0, 0, ())
                cameras[index] = record
            # simplecv's canonical RRD video reader needs rerun-sdk[datafusion], absent from this frozen runtime; keep direct Arrow reads for the exo path.
            # Read the H.264 blobs straight out of the Arrow buffer. The obvious
            # `samples.to_pylist()` boxes every byte of every sample into Python
            # objects: profiled at 65.7 s for this segment versus 0.4 s here, a
            # 164x difference for byte-identical output, and it dominated the whole
            # Python side of the run (94.6 % of samples in py-spy).
            blobs = samples.combine_chunks() if hasattr(samples, "combine_chunks") else samples
            inner = blobs.flatten()  # list<list<u8>> -> list<u8>, one entry per sample
            offsets: Int[ndarray, "n+1"] = inner.offsets.to_numpy()
            byte_buffer: Int[ndarray, "n_bytes"] = inner.values.to_numpy(zero_copy_only=False)
            valid: Bool[ndarray, "n"] = inner.is_valid().to_numpy(zero_copy_only=False)
            stamps: list = times.to_pylist()
            for row, timestamp in enumerate(stamps):
                if timestamp is None or row >= len(inner) or not valid[row]:
                    continue
                raw: bytes = byte_buffer[offsets[row] : offsets[row + 1]].tobytes()
                record.samples.append((int(getattr(timestamp, "value", timestamp)), raw))
            continue

        record = cameras.setdefault(index, RrdCameraRecord(f"cam_{index:02d}", np.eye(4), np.eye(3), 0, 0, ()))

        if entity_path.endswith("/pinhole"):
            k_value: list[float] = first_valid_value(
                record_batch.column("Pinhole:image_from_camera"), component_name="Pinhole:image_from_camera"
            )
            # Rerun Mat3x3 is column-major.
            record.k_matrix = np.asarray(k_value, dtype=np.float64).reshape(3, 3).T
            resolution_value: list[float] = first_valid_value(
                record_batch.column("Pinhole:resolution"), component_name="Pinhole:resolution"
            )
            record.width = int(resolution_value[0])
            record.height = int(resolution_value[1])
            distortion_value: list[float] = first_valid_value(
                record_batch.column("simplecv.components.DistortionCoefficients"),
                component_name="simplecv.components.DistortionCoefficients",
            )
            # Fisheye62 stores 8 slots; RoboCap populates only k1..k4.
            record.distortion = tuple(float(c) for c in distortion_value[:4])
            continue

        # Bare camera node: name/kind and the static rig_T_cam.
        if "name" in component_names:
            name_value: list[str] = first_valid_value(record_batch.column("name"), component_name="name")
            record.name = str(name_value[0])
        if "kind" in component_names:
            kind_value: list[str] = first_valid_value(record_batch.column("kind"), component_name="kind")
            kind: str = str(kind_value[0])
            if kind not in ("rgb", "grayscale", "depth", "imu"):
                raise ValueError(f"{entity_path}: unsupported sensor kind {kind!r}")
            record.kind = cast(SensorKind, kind)
        transform_components: set[str] = {"Transform3D:mat3x3", "Transform3D:translation"}
        if transform_components.issubset(component_names):
            mat_value: list[float] = first_valid_value(
                record_batch.column("Transform3D:mat3x3"), component_name="Transform3D:mat3x3"
            )
            translation_value: list[float] = first_valid_value(
                record_batch.column("Transform3D:translation"), component_name="Transform3D:translation"
            )
            if "Transform3D:relation" in component_names:
                relation_value: list[int] = first_valid_value(
                    record_batch.column("Transform3D:relation"), component_name="Transform3D:relation"
                )
            else:
                relation_value = []
            if relation_value and int(relation_value[0]) != CHILD_FROM_PARENT:
                raise ValueError(
                    f"{entity_path}: expected a ChildFromParent extrinsic "
                    f"(got relation={int(relation_value[0])}); the stored value would be "
                    f"rig_T_cam, not cam_T_rig, and inverting it would flip every frustum."
                )
            rotation: Float64[ndarray, "3 3"] = np.asarray(mat_value, dtype=np.float64).reshape(3, 3).T
            record.cam_T_rig = compose(rotation, np.asarray(translation_value, dtype=np.float64))

    if not pose_times:
        raise RuntimeError(f"{rrd_path} has no world_T_rig stream — is the `slam` layer present?")

    order: Int[ndarray, "n"] = np.argsort(np.asarray(pose_times))
    times_ns: Int[ndarray, "n"] = np.asarray(pose_times, dtype=np.int64)[order]
    quaternions_xyzw: Float64[ndarray, "n 4"] = np.asarray(pose_quats, dtype=np.float64)[order]
    translations: Float64[ndarray, "n 3"] = np.asarray(pose_translations, dtype=np.float64)[order]
    for record in cameras.values():
        record.samples.sort(key=lambda item: item[0])

    print(f"  read {len(cameras)} cameras, {len(times_ns)} rig poses from {rrd_path.name}")
    return cameras, times_ns, Rotation.from_quat(quaternions_xyzw), translations


def extract_jpegs(
    mp4_path: Path, output_dir: Path, frame_indices: Sequence[int], timestamps_ns: list[int]
) -> list[Path]:
    """Extract exactly ``frame_indices`` as JPEGs named by their nanosecond timestamp.

    Takes the explicit index list rather than a stride so the ffmpeg selection and
    the output filenames cannot disagree. They previously could: the filter kept
    ``n % stride == 0`` (0, stride, 2*stride, ...) while the caller expects frames
    at ``first_ok + k*stride``, and those two sets coincide only when
    ``first_ok % stride == 0``. The shipped defaults satisfy that by luck
    (``first_ok`` 60, stride 20), but ``--dataset.trim-start-seconds 1.0`` makes
    every expected file missing, which surfaces as an empty ``frames_meta.json``
    and a reconstruction failure far downstream.

    CPU decode is used deliberately: measured on this host, ``h264_cuvid`` took
    4.16 s against 2.32 s for CPU on 233 frames, because the cost is JPEG
    *encoding* and the device-to-host copy, not H.264 decode.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    staging: Path = output_dir / "_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    # An explicit `eq(n,i)+eq(n,j)+...` term per frame: unambiguous, and it also
    # stops ffmpeg encoding the leading frames the trim discards.
    wanted: str = "+".join(rf"eq(n\,{index})" for index in frame_indices)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(mp4_path),
            "-vf", f"select='{wanted}'",
            # `-fps_mode passthrough` is the tested ffmpeg 8.x path. The project
            # keeps that major pinned because this extraction is untested on 9.x.
            "-fps_mode", "passthrough", "-q:v", "2",
            str(staging / "f_%06d.jpg"),
        ],
        check=True,
    )

    extracted: list[Path] = sorted(staging.glob("f_*.jpg"))
    if len(extracted) != len(frame_indices):
        print(
            f"    WARNING: asked ffmpeg for {len(frame_indices)} frames from "
            f"{mp4_path.name}, got {len(extracted)}"
        )
    paths: list[Path] = []
    for output_index, source_index in enumerate(frame_indices):
        if output_index >= len(extracted):
            break
        destination: Path = output_dir / f"{timestamps_ns[source_index]}.jpeg"
        shutil.move(str(extracted[output_index]), destination)
        paths.append(destination)
    shutil.rmtree(staging, ignore_errors=True)
    return paths


def prepare_robocap(config: RobocapConfig, run: RunConfig) -> PreparedSequence:
    """Turn a frozen RoboCap segment into cuSFM's on-disk input contract."""
    print(f"[robocap] reading {config.rrd_path}")
    start: float = time.time()
    records, pose_times_ns, pose_rotations, pose_translations = read_robocap_rrd(config.rrd_path)

    # Frames and remuxed video are shared across variants (extraction is the slow
    # part); frames_meta.json and cuSFM's stereo.edex are per-variant, because two
    # variants with different camera sets would otherwise overwrite each other.
    shared_dir: Path = run.work_dir / "robocap"
    work_dir: Path = run.work_dir / f"robocap{run.variant}"
    frames_dir: Path = shared_dir / "frames"
    input_dir: Path = work_dir / "input"
    video_dir: Path = shared_dir / "video"

    by_name: dict[str, RrdCameraRecord] = {record.name: record for record in records.values()}
    missing: list[str] = [name for name in config.cameras if name not in by_name]
    if missing:
        raise KeyError(f"cameras {missing} not in recording (have {sorted(by_name)})")

    reference_name: str = config.cameras[0]
    reference_samples: list[tuple[int, bytes]] = by_name[reference_name].samples
    all_times: Int[ndarray, "n"] = np.asarray([t for t, _ in reference_samples], dtype=np.int64)
    first_ok: int = int(np.searchsorted(all_times, all_times[0] + int(config.trim_start_seconds * 1e9)))
    last_ok: int = int(np.searchsorted(all_times, all_times[-1] - int(config.trim_end_seconds * 1e9)))
    if config.trim_start_seconds or config.trim_end_seconds:
        print(
            f"  trimmed {config.trim_start_seconds:g}s head / {config.trim_end_seconds:g}s tail: "
            f"frames {first_ok}..{last_ok} of {len(all_times)}"
        )
    sample_indices: list[int] = list(range(first_ok, last_ok, config.frame_stride))
    if config.max_samples is not None:
        sample_indices = sample_indices[: config.max_samples]
    canonical_times: Int[ndarray, "n_samples"] = np.asarray(
        [reference_samples[i][0] for i in sample_indices], dtype=np.int64
    )
    print(
        f"  {len(reference_samples)} frames -> {len(canonical_times)} samples "
        f"(stride {config.frame_stride}) across {len(config.cameras)} cameras"
    )

    input_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, list[Path | None]] = {}
    for name in config.cameras:
        record = by_name[name]
        mp4_path: Path = video_dir / f"{name}.mp4"
        if not mp4_path.is_file():
            mp4_path.parent.mkdir(parents=True, exist_ok=True)
            sample_times: pa.ChunkedArray = pa.chunked_array(
                [pa.array([timestamp for timestamp, _ in record.samples], type=pa.duration("ns"))]
            )
            sample_payload: bytes = b"".join(raw for _, raw in record.samples)
            sample_offsets: Int[ndarray, "n+1"] = np.zeros(len(record.samples) + 1, dtype=np.int32)
            np.cumsum([len(raw) for _, raw in record.samples], out=sample_offsets[1:])
            sample_array: pa.ListArray = pa.ListArray.from_arrays(
                pa.array(sample_offsets), pa.array(np.frombuffer(sample_payload, dtype=np.uint8))
            )
            mux_h264_to_mp4(sample_times, pa.chunked_array([sample_array]), str(mp4_path))
        camera_dir: Path = frames_dir / name
        timestamps: list[int] = [timestamp for timestamp, _ in record.samples]
        # MUST be the same indices as `sample_indices`, which are already trimmed.
        # Selecting from 0 here while the poses start at `first_ok` would pair every
        # image with a pose ~first_ok/stride samples away -- a silent misalignment.
        selected: list[int] = list(sample_indices)
        # Address frames by timestamp, never by directory order. Two traps otherwise:
        # `sorted()` is lexicographic, so `99621592000.jpeg` (t < 100 s) sorts AFTER
        # `100021636444.jpeg` and silently pairs samples with the wrong images; and the
        # frame cache is shared across strides, so a coarse run following a dense one
        # would find "enough" files and take the first N -- the opening seconds of video
        # rather than the strided samples.
        expected_paths: list[Path] = [camera_dir / f"{timestamps[i]}.jpeg" for i in selected]
        present: set[Path] = {path for path in expected_paths if path.is_file()}
        if len(present) != len(expected_paths):
            extract_jpegs(mp4_path, camera_dir, selected, timestamps)
            present = {path for path in expected_paths if path.is_file()}
        image_paths[name] = [
            expected_paths[i] if expected_paths[i] in present else None
            for i in range(len(canonical_times))
        ]
        # cuSFM resolves image_name relative to input_dir, so link the shared
        # frame directory into this variant's input directory.
        link: Path = input_dir / name
        # `exists()` follows the link, so a *dangling* symlink reports False and
        # `symlink_to` then raises FileExistsError. Test the link itself.
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(camera_dir.resolve(), target_is_directory=True)
        print(f"    {name:12s} {sum(p is not None for p in image_paths[name])} frames")

    # Full-rate stream: one pose per source frame, for continuous playback.
    full_times: Int[ndarray, "n_frames"] = all_times[first_ok:last_ok]
    full_rotations, full_translations = interpolate_poses(
        full_times, pose_times_ns, pose_rotations, pose_translations
    )
    full_world_T_rig: Float64[ndarray, "n_frames 4 4"] = np.stack(
        [compose(full_rotations[i], full_translations[i]) for i in range(len(full_times))]
    )
    keyframe_mask: Bool[ndarray, "n_frames"] = np.zeros(len(full_times), dtype=bool)
    keyframe_mask[np.asarray(sample_indices, dtype=np.int64) - first_ok] = True

    rig_rotations, rig_translations = interpolate_poses(
        canonical_times, pose_times_ns, pose_rotations, pose_translations
    )
    world_T_rig: Float64[ndarray, "n_samples 4 4"] = np.stack(
        [compose(rig_rotations[i], rig_translations[i]) for i in range(len(canonical_times))]
    )

    cameras: list[PreparedCamera] = []
    for index in sorted(records):
        record = records[index]
        cameras.append(
            PreparedCamera(
                name=record.name,
                rig_T_cam=np.linalg.inv(record.cam_T_rig),
                k_matrix=record.k_matrix,
                width=record.width,
                height=record.height,
                distortion=record.distortion,
                used_for_sfm=record.name in config.cameras,
                kind=record.kind,
            )
        )

    sequence: PreparedSequence = PreparedSequence(
        name="robocap",
        cameras=cameras,
        timestamps_ns=canonical_times,
        world_T_rig=world_T_rig,
        image_paths={c.name: image_paths.get(c.name, [None] * len(canonical_times)) for c in cameras},
        input_dir=input_dir,
        frames_meta_path=input_dir / CUSFM_FRAME_META_FILE,
        image_plane_distance=config.image_plane_distance,
        projection_model="OPENCV_FISHEYE",
        reference_name="imu_00",
        full_timestamps_ns=full_times,
        full_world_T_rig=full_world_T_rig,
        keyframe_mask=keyframe_mask,
        video_samples={
            name: [
                (t, raw)
                for t, raw in by_name[name].samples
                if all_times[first_ok] <= t <= all_times[last_ok - 1]
            ]
            for name in config.cameras
        },
    )
    write_frames_meta(
        sequence,
        ROBOCAP_RIG_ADJACENCY_PAIRS if config.rig_adjacency_pairs else (ROBOCAP_STEREO_PAIR,),
    )
    print(f"[robocap] prepared in {time.time() - start:.1f}s")
    return sequence


"""The COLMAP database that carries cuSFM's frames into pycolmap.

`create_database` writes the four tables COLMAP's own feature extractor and
matcher expect to find already populated — cameras, rig, frames, images — using
cuSFM's identifiers verbatim (`camera_params_id` -> `camera_id`,
`KeyframeMetaData.id` -> `image_id`, `synced_sample_id` -> `frame_id`), the same
mapping `colsfm.reconstruction` uses.

Three things about COLMAP 4.2 shape this module:

* **Everything is `write_*`, not `add_*`.** pycolmap 0.x's `Database.add_camera`
  and friends do not exist (pycolmap-capabilities.md §4).
* **`Camera.has_prior_focal_length` defaults to False**, and that silently
  selects the uncalibrated (fundamental-matrix) path in two-view geometry: a
  PINHOLE pair comes back `UNCALIBRATED`, an `OPENCV_FISHEYE` pair comes back
  `DEGENERATE` with zero inliers (§5). Every camera written here sets it True.
* **`extract_features` reuses a pre-existing image row** rather than inserting a
  second one: COLMAP's `ImageReader` looks the name up first, keeps the
  `camera_id`, `image_id` and `frame_id` already stored, and leaves the rig and
  frame tables untouched. That is what lets the pipeline own the identifiers and
  still use COLMAP's own ALIKED extractor. Measured on Galileo; the log line
  `Focal Length: 852.92px (Prior)` is the visible confirmation.

Keypoints written by COLMAP are Nx6 (xy plus a 2x2 affine block), so
`read_keypoints` slices the two columns the rest of the pipeline wants.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Float32
from numpy import ndarray

from colsfm.cameras import colmap_cameras
from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame
from colsfm.reconstruction import RIG_ID, build_rig, camera_sensor_id

ImagePair: TypeAlias = tuple[int, int]
"""An undirected image pair as `(min(image_id), max(image_id))`."""

KeypointsXY: TypeAlias = Float32[ndarray, "num_keypoints 2"]
"""Keypoint pixel coordinates in the original image, COLMAP's float32 storage type."""


def create_database(
    database_path: Path,
    frames_meta: FramesMeta,
    cameras: Mapping[int, pycolmap.Camera] | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Write cameras, the rig, the frames and the images of a collection.

    Nothing else is written: keypoints, descriptors, matches and two-view
    geometries are filled in later by `colsfm.features` and `colsfm.matching`.

    Args:
        database_path: Destination `.db` file; parent directories are created.
        frames_meta: The collection to import, already filtered to the keyframes
            that should be reconstructed (see `colsfm.keyframe_selection`).
        cameras: COLMAP cameras keyed by `camera_params_id`; derived from
            `frames_meta` with `colsfm.cameras.colmap_cameras` when None.
        overwrite: Delete an existing database first instead of failing.

    Raises:
        FileExistsError: When the file exists and `overwrite` is False.
    """
    if database_path.exists():
        if not overwrite:
            raise FileExistsError(f"Database {database_path} already exists; pass overwrite=True to replace it")
        database_path.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_cameras: Mapping[int, pycolmap.Camera] = colmap_cameras(frames_meta) if cameras is None else cameras
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    rig_frames: tuple[RigFrame, ...] = frames_meta.rig_frames()

    database: pycolmap.Database = pycolmap.Database.open(database_path)
    try:
        for camera_params_id, camera in sorted(resolved_cameras.items()):
            camera.camera_id = camera_params_id
            camera.has_prior_focal_length = True
            database.write_camera(camera, True)
        database.write_rig(build_rig(frames_meta), True)
        for rig_frame in rig_frames:
            database.write_frame(_colmap_frame(rig_frame, keyframe_by_id), True)
        for rig_frame in rig_frames:
            for keyframe_id in rig_frame.keyframe_ids:
                database.write_image(_colmap_image(keyframe_by_id[keyframe_id]), True)
    finally:
        database.close()


def _colmap_frame(rig_frame: RigFrame, keyframe_by_id: Mapping[int, KeyframeMeta]) -> pycolmap.Frame:
    """Build the COLMAP frame for one rig instant.

    Args:
        rig_frame: The rig instant, from `FramesMeta.rig_frames`.
        keyframe_by_id: Keyframes of the collection, keyed by keyframe id.

    Returns:
        A frame carrying one `data_t` per member image. The pose stays unset —
        database frames carry the grouping only (pycolmap-capabilities.md §4).
    """
    frame: pycolmap.Frame = pycolmap.Frame()
    frame.frame_id = rig_frame.synced_sample_id
    frame.rig_id = RIG_ID
    for keyframe_id in rig_frame.keyframe_ids:
        sensor: pycolmap.sensor_t = camera_sensor_id(keyframe_by_id[keyframe_id].camera_params_id)
        frame.add_data_id(pycolmap.data_t(sensor, keyframe_id))
    return frame


def _colmap_image(keyframe: KeyframeMeta) -> pycolmap.Image:
    """Build the COLMAP image row for one keyframe.

    Args:
        keyframe: The keyframe to import.

    Returns:
        An image whose `image_id`, `camera_id` and `frame_id` are cuSFM's own
        ids and whose `name` is `image_name` relative to the raw image root.
    """
    image: pycolmap.Image = pycolmap.Image(name=keyframe.image_name, camera_id=keyframe.camera_params_id)
    image.image_id = keyframe.keyframe_id
    image.frame_id = keyframe.synced_sample_id
    return image


def image_ids_by_name(database_path: Path) -> dict[str, int]:
    """Index the database's images by name.

    COLMAP's pair lists and `verify_matches` address images by name, so this is
    what turns keyframe-id pairs into something pycolmap will accept.

    Args:
        database_path: An existing database.

    Returns:
        `image_id` keyed by `name`.
    """
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    try:
        return {image.name: image.image_id for image in database.read_all_images()}
    finally:
        database.close()


def read_keypoints(database_path: Path, image_id: int) -> KeypointsXY:
    """Read one image's keypoints as pixel coordinates.

    Args:
        database_path: An existing database.
        image_id: The image to read.

    Returns:
        Float32 `[num_keypoints, 2]` xy pixel coordinates. COLMAP's ALIKED
        extractor stores Nx6 rows; the trailing affine block is dropped.
    """
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    try:
        stored: Float32[ndarray, "num_keypoints num_columns"] = database.read_keypoints(image_id)
    finally:
        database.close()
    return np.ascontiguousarray(stored[:, :2])


def keypoint_counts(database_path: Path, image_ids: Iterable[int] | None = None) -> dict[int, int]:
    """Count the keypoints stored per image.

    Args:
        database_path: An existing database.
        image_ids: Images to count; every image in the database by default.

    Returns:
        Keypoint count keyed by `image_id`.
    """
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    try:
        wanted: list[int] = (
            [image.image_id for image in database.read_all_images()] if image_ids is None else list(image_ids)
        )
        return {image_id: int(database.read_keypoints(image_id).shape[0]) for image_id in wanted}
    finally:
        database.close()


def read_two_view_geometry(database_path: Path, image_id1: int, image_id2: int) -> pycolmap.TwoViewGeometry:
    """Read the verified two-view geometry of one pair.

    Args:
        database_path: An existing database.
        image_id1: First image of the pair.
        image_id2: Second image of the pair.

    Returns:
        The stored geometry; an empty one when the pair was never verified.
    """
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    try:
        return database.read_two_view_geometry(image_id1, image_id2)
    finally:
        database.close()


def pair_inlier_counts(database_path: Path, pairs: Sequence[ImagePair]) -> dict[ImagePair, int]:
    """Count the verified inlier matches of many pairs in one pass.

    Args:
        database_path: An existing database.
        pairs: Image pairs to look up, in any order.

    Returns:
        Inlier count keyed by the pair as given; 0 for a pair with no geometry.
    """
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    try:
        return {
            pair: len(database.read_two_view_geometry(pair[0], pair[1]).inlier_matches) for pair in pairs
        }
    finally:
        database.close()


def delete_two_view_geometries(database_path: Path, pairs: Sequence[ImagePair]) -> None:
    """Drop the two-view geometry rows of the given pairs.

    `pycolmap.verify_matches` skips any pair that already has a row, and COLMAP's
    matcher writes an `UNDEFINED` placeholder row whenever
    `skip_geometric_verification` is set — so the placeholders have to go before
    verification will do anything.

    Args:
        database_path: An existing database.
        pairs: Image pairs whose geometries should be removed.
    """
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    try:
        for image_id1, image_id2 in pairs:
            database.delete_two_view_geometry(image_id1, image_id2)
    finally:
        database.close()


def raw_match_counts(database_path: Path, pairs: Sequence[ImagePair]) -> dict[ImagePair, int]:
    """Count the unverified matches of many pairs in one pass.

    Args:
        database_path: An existing database.
        pairs: Image pairs to look up, in any order.

    Returns:
        Raw match count keyed by the pair as given; 0 for an unmatched pair.
    """
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    try:
        return {pair: len(database.read_matches(pair[0], pair[1])) for pair in pairs}
    finally:
        database.close()

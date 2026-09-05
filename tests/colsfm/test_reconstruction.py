"""`build_reconstruction`, checked against the metadata the two shipped datasets carry.

The seam is `colsfm.reconstruction.build_reconstruction(frames_meta)` and the
`pycolmap.Reconstruction` it returns: poses read back through COLMAP's own
accessors, never through the module's internals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame, read_frames_meta
from colsfm.geometry import relative_rotation_degrees
from colsfm.reconstruction import RIG_ID, RigReference, build_reconstruction, camera_sensor_id, rig_reference

POSE_TOLERANCE: float = 1e-9
"""Round-trip tolerance in metres and in the matrix norm; this is pure algebra."""


@pytest.fixture(scope="module")
def galileo_meta(galileo_run_dir: Path) -> FramesMeta:
    """The 226-keyframe, 8-camera PINHOLE Galileo poses, after the pose graph.

    This is what the mapper is actually handed (`--keyframe_metadata_file`), and
    unlike the raw cuVSLAM input it is rig-exact — see
    `test_a_rig_inexact_input_snaps_to_the_rig`.
    """
    return read_frames_meta(galileo_run_dir / "cusfm" / "pose_graph" / "frames_meta.json")


@pytest.fixture(scope="module")
def galileo_raw_meta(galileo_input_meta: Path) -> FramesMeta:
    """The raw 226-keyframe cuVSLAM input, which is not rig-exact."""
    return read_frames_meta(galileo_input_meta)


@pytest.fixture(scope="module")
def fisheye_meta(repo_root: Path) -> FramesMeta:
    """A real OPENCV_FISHEYE collection: the 32-keyframe, 4-camera RoboCap smoke run."""
    return read_frames_meta(repo_root / "data" / "cusfm_runs" / "robocap_raco_smoke" / "input" / "frames_meta.json")


def _pose_delta(actual: pycolmap.Rigid3d, expected: pycolmap.Rigid3d) -> tuple[float, float]:
    """Translation and rotation difference between two poses.

    Args:
        actual: Pose under test.
        expected: Reference pose.

    Returns:
        Translation difference in metres and rotation difference in degrees.
    """
    translation_m: float = float(np.linalg.norm(np.asarray(actual.translation) - np.asarray(expected.translation)))
    return translation_m, relative_rotation_degrees(actual, expected)


def test_galileo_poses_round_trip_through_the_reconstruction(galileo_meta: FramesMeta) -> None:
    """Every keyframe's `camera_to_world` comes back out of COLMAP unchanged.

    `image.cam_from_world()` is composed by COLMAP from `sensor_from_rig` and
    the frame's `rig_from_world`, so this exercises the whole rig construction:
    if the extrinsic were stored the right way up instead of inverted, or the
    rig frame were the reference camera rather than the vehicle, the composition
    would not return the metadata's own pose.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(galileo_meta)

    assert reconstruction.num_cameras() == 8
    assert reconstruction.num_images() == len(galileo_meta.keyframes) == 226
    assert reconstruction.num_frames() == len(galileo_meta.rig_frames()) == 29
    assert reconstruction.num_reg_frames() == 29
    assert reconstruction.num_rigs() == 1

    worst_translation_m: float = 0.0
    worst_rotation_deg: float = 0.0
    for keyframe in galileo_meta.keyframes:
        image: pycolmap.Image = reconstruction.image(keyframe.keyframe_id)
        assert image.name == keyframe.image_name
        assert image.camera_id == keyframe.camera_params_id
        assert image.frame_id == keyframe.synced_sample_id
        translation_m, rotation_deg = _pose_delta(image.cam_from_world().inverse(), keyframe.world_T_cam)
        worst_translation_m = max(worst_translation_m, translation_m)
        worst_rotation_deg = max(worst_rotation_deg, rotation_deg)
    print(f"[colsfm] Galileo pose round trip: {worst_translation_m:.2e} m, {worst_rotation_deg:.2e} deg")
    assert worst_translation_m < POSE_TOLERANCE
    assert worst_rotation_deg < POSE_TOLERANCE


def test_rig_frames_carry_the_vehicle_pose(galileo_meta: FramesMeta) -> None:
    """`frame.rig_from_world` is `vehicle_T_world`, the inverse of `world_T_vehicle`.

    That is the property the mapper's pose comparison against the blob relies
    on, and it is what makes one BA pose per rig frame equal cuSFM's
    `VEHICLE_RIG` parameterisation.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(galileo_meta)
    for rig_frame in galileo_meta.rig_frames():
        frame: pycolmap.Frame = reconstruction.frame(rig_frame.synced_sample_id)
        translation_m, rotation_deg = _pose_delta(frame.rig_from_world.inverse(), rig_frame.world_T_vehicle)
        assert translation_m < POSE_TOLERANCE
        assert rotation_deg < POSE_TOLERANCE
        assert sorted(data_id.id for data_id in frame.data_ids) == sorted(rig_frame.keyframe_ids)


def test_sensor_from_rig_is_the_inverted_extrinsic(galileo_meta: FramesMeta) -> None:
    """`sensor_from_rig` is `cam_T_vehicle`, the inverse of `sensor_to_vehicle_transform`."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(galileo_meta)
    rig: pycolmap.Rig = reconstruction.rig(RIG_ID)
    assert rig.num_sensors() == 9, "eight cameras plus the vehicle body as the reference sensor"
    for camera_params_id, camera in galileo_meta.cameras.items():
        sensor_from_rig: pycolmap.Rigid3d = rig.sensor_from_rig(camera_sensor_id(camera_params_id))
        translation_m, rotation_deg = _pose_delta(sensor_from_rig, camera.vehicle_T_cam.inverse())
        assert translation_m < POSE_TOLERANCE
        assert rotation_deg < POSE_TOLERANCE


def test_cameras_declare_a_prior_focal_length(galileo_meta: FramesMeta, fisheye_meta: FramesMeta) -> None:
    """Every camera carries `has_prior_focal_length`.

    pycolmap defaults it to False, which silently downgrades two-view geometry
    to a fundamental matrix for PINHOLE and to `DEGENERATE` with zero inliers
    for OPENCV_FISHEYE (pycolmap-capabilities.md §5). The calibration is known
    here, so the flag is always true.
    """
    for frames_meta in (galileo_meta, fisheye_meta):
        reconstruction: pycolmap.Reconstruction = build_reconstruction(frames_meta)
        for camera_id in sorted(reconstruction.cameras):
            assert reconstruction.camera(camera_id).has_prior_focal_length is True


def test_fisheye_poses_round_trip_through_the_reconstruction(fisheye_meta: FramesMeta) -> None:
    """The same round trip on a four-camera OPENCV_FISHEYE rig with zero-based ids.

    RoboCap numbers its keyframes, cameras and rig frames from zero, so this
    also proves that COLMAP accepts id 0 for a camera, an image and a frame.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(fisheye_meta)

    assert reconstruction.num_cameras() == 4
    assert reconstruction.num_images() == 32
    assert reconstruction.num_frames() == 8
    assert {camera.model.name for camera in reconstruction.cameras.values()} == {"OPENCV_FISHEYE"}
    assert 0 in reconstruction.cameras and 0 in reconstruction.images and 0 in reconstruction.frames

    for keyframe in fisheye_meta.keyframes:
        translation_m, rotation_deg = _pose_delta(
            reconstruction.image(keyframe.keyframe_id).cam_from_world().inverse(), keyframe.world_T_cam
        )
        assert translation_m < POSE_TOLERANCE
        assert rotation_deg < POSE_TOLERANCE


def test_supplied_cameras_are_used_verbatim(galileo_meta: FramesMeta) -> None:
    """A caller-supplied camera map replaces the derived one, ids and all.

    The matcher and the mapper have to agree on the camera parameters they put
    in the database, so the cameras are an argument rather than always derived.
    """
    cameras: dict[int, pycolmap.Camera] = {}
    for camera_params_id in sorted(galileo_meta.cameras):
        camera: pycolmap.Camera = pycolmap.Camera.create_from_model_id(
            camera_params_id, pycolmap.CameraModelId.PINHOLE, 700.0, 1920, 1200
        )
        cameras[camera_params_id] = camera
    reconstruction: pycolmap.Reconstruction = build_reconstruction(galileo_meta, cameras)
    for camera_params_id in sorted(galileo_meta.cameras):
        assert reconstruction.camera(camera_params_id).params[0] == pytest.approx(700.0)


def test_a_rig_inexact_input_snaps_to_the_rig(galileo_input_meta: Path) -> None:
    """The raw cuVSLAM metadata is not rig-exact, and the rig model rounds it off.

    `data/r2b_galileo/frames_meta.json` carries an independent `camera_to_world`
    per image, and the vehicle poses those imply disagree across the eight
    cameras of one `synced_sample_id` by 13.7 um and 4.5e-4 deg — measured
    straight off the file, with no reconstruction involved. A single rig pose per
    frame cannot reproduce all eight, so the round trip is exact only for the
    rig-exact files cuSFM's own stages write (`pose_graph/`, `kpmap/keyframes/`).
    The residual is three orders of magnitude below the 5 mm the mapper moves
    poses by, so nothing downstream notices.
    """
    frames_meta: FramesMeta = read_frames_meta(galileo_input_meta)
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    worst_spread_m: float = max(
        float(
            np.linalg.norm(
                np.asarray(frames_meta.world_T_vehicle(keyframe_by_id[keyframe_id]).translation)
                - np.asarray(frames_meta.world_T_vehicle(keyframe_by_id[rig_frame.reference_keyframe_id]).translation)
            )
        )
        for rig_frame in frames_meta.rig_frames()
        for keyframe_id in rig_frame.keyframe_ids
    )
    assert 1e-6 < worst_spread_m < 1e-4

    reconstruction: pycolmap.Reconstruction = build_reconstruction(frames_meta)
    worst_translation_m: float = max(
        _pose_delta(reconstruction.image(keyframe.keyframe_id).cam_from_world().inverse(), keyframe.world_T_cam)[0]
        for keyframe in frames_meta.keyframes
    )
    print(f"[colsfm] raw Galileo input: rig spread {worst_spread_m * 1e6:.2f} um, snap {worst_translation_m * 1e6:.2f} um")
    # The snap is the spread carried through each camera's lever arm, so it is
    # the same size but not the same number.
    assert worst_spread_m < worst_translation_m < 1e-4


def test_the_reference_keyframe_is_the_lowest_id_of_the_first_rig_frame(galileo_meta: FramesMeta) -> None:
    """The gauge frame the mapper fixes is the one holding the lowest keyframe id.

    cuSFM puts exactly one keyframe in `constant_keyframes_` — in Galileo the
    lowest id — and through the rig that fixes its whole rig frame
    (keypoints_mapper_main.md §6.3).
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(galileo_meta)
    lowest_keyframe: KeyframeMeta = min(galileo_meta.keyframes, key=lambda keyframe: keyframe.keyframe_id)
    rig_frame: RigFrame = next(
        frame for frame in galileo_meta.rig_frames() if lowest_keyframe.keyframe_id in frame.keyframe_ids
    )
    assert reconstruction.image(lowest_keyframe.keyframe_id).frame_id == rig_frame.synced_sample_id


REFERENCE_CAMERA_PARAMS_ID: int = 0
"""The camera the extrinsic-refinement tests put the rig origin on."""


@pytest.mark.parametrize("frames_meta_name", ["galileo_meta", "galileo_raw_meta"])
def test_a_camera_referenced_rig_round_trips_the_same_poses(
    request: pytest.FixtureRequest, frames_meta_name: str
) -> None:
    """Putting the rig origin on a camera changes nothing an image pose can see.

    `sensor_from_rig` and `rig_from_world` both move — the first becomes
    `cam_i_T_cam_ref`, the second `cam_ref_T_world` — but their composition is
    still `cam_T_world`, so every `camera_to_world` comes back out unchanged.
    """
    frames_meta: FramesMeta = request.getfixturevalue(frames_meta_name)
    reconstruction: pycolmap.Reconstruction = build_reconstruction(
        frames_meta, reference_camera_params_id=REFERENCE_CAMERA_PARAMS_ID
    )
    reference: pycolmap.Reconstruction = build_reconstruction(frames_meta)

    rig: pycolmap.Rig = reconstruction.rig(RIG_ID)
    assert rig.ref_sensor_id == camera_sensor_id(REFERENCE_CAMERA_PARAMS_ID)
    assert rig.num_sensors() == len(frames_meta.cameras), "no vehicle sensor: a camera is the reference"

    worst_translation_m: float = 0.0
    worst_rotation_deg: float = 0.0
    for keyframe in frames_meta.keyframes:
        image: pycolmap.Image = reconstruction.image(keyframe.keyframe_id)
        translation_m, rotation_deg = _pose_delta(
            image.cam_from_world(), reference.image(keyframe.keyframe_id).cam_from_world()
        )
        worst_translation_m = max(worst_translation_m, translation_m)
        worst_rotation_deg = max(worst_rotation_deg, rotation_deg)
    print(f"[colsfm] camera-referenced rig, {frames_meta_name}: {worst_translation_m:.2e} m, {worst_rotation_deg:.2e} deg")
    assert worst_translation_m < POSE_TOLERANCE
    assert worst_rotation_deg < POSE_TOLERANCE


@pytest.mark.parametrize("reference_camera_params_id", [None, REFERENCE_CAMERA_PARAMS_ID])
def test_the_rig_reference_recovers_the_vehicle_frame_quantities(
    galileo_meta: FramesMeta, reference_camera_params_id: int | None
) -> None:
    """`RigReference` inverts the bookkeeping for both reference choices.

    Given the reconstruction and the input `vehicle_T_cam_ref`, every camera's
    `sensor_to_vehicle_transform` and every frame's `world_T_vehicle` come back
    as the metadata declared them.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(
        galileo_meta, reference_camera_params_id=reference_camera_params_id
    )
    reference: RigReference = rig_reference(galileo_meta, reference_camera_params_id)

    vehicle_T_cam: dict[int, pycolmap.Rigid3d] = reference.vehicle_T_cam_by_camera_params_id(reconstruction)
    assert sorted(vehicle_T_cam) == sorted(galileo_meta.cameras)
    for camera_params_id, camera in galileo_meta.cameras.items():
        translation_m, rotation_deg = _pose_delta(vehicle_T_cam[camera_params_id], camera.vehicle_T_cam)
        assert translation_m < POSE_TOLERANCE
        assert rotation_deg < POSE_TOLERANCE

    world_T_vehicle: dict[int, pycolmap.Rigid3d] = reference.world_T_vehicle_by_frame_id(reconstruction)
    for rig_frame in galileo_meta.rig_frames():
        translation_m, rotation_deg = _pose_delta(
            world_T_vehicle[rig_frame.synced_sample_id], rig_frame.world_T_vehicle
        )
        assert translation_m < POSE_TOLERANCE
        assert rotation_deg < POSE_TOLERANCE


def test_the_reference_camera_carries_the_identity_extrinsic(galileo_meta: FramesMeta) -> None:
    """The reference camera's `sensor_from_rig` is identity, and its `vehicle_T_cam` the input one.

    That is what makes the reference camera the gauge: bundle adjustment has no
    parameter block for it, so it is the fixed bridge back to the vehicle frame.
    """
    reference: RigReference = rig_reference(galileo_meta, REFERENCE_CAMERA_PARAMS_ID)
    assert reference.camera_params_id == REFERENCE_CAMERA_PARAMS_ID
    translation_m, rotation_deg = _pose_delta(
        reference.vehicle_T_reference, galileo_meta.cameras[REFERENCE_CAMERA_PARAMS_ID].vehicle_T_cam
    )
    assert translation_m < POSE_TOLERANCE
    assert rotation_deg < POSE_TOLERANCE

    reconstruction: pycolmap.Reconstruction = build_reconstruction(
        galileo_meta, reference_camera_params_id=REFERENCE_CAMERA_PARAMS_ID
    )
    assert reconstruction.rig(RIG_ID).is_ref_sensor(camera_sensor_id(REFERENCE_CAMERA_PARAMS_ID))


def test_an_unknown_reference_camera_is_rejected(galileo_meta: FramesMeta) -> None:
    """Naming a camera the metadata does not carry fails loudly."""
    with pytest.raises(KeyError):
        build_reconstruction(galileo_meta, reference_camera_params_id=999)

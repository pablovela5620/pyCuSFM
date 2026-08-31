"""Behavior tests for the standalone RaCo cuSFM keyframe writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from google.protobuf.message import Message
from jaxtyping import Float32, UInt8
from numpy import ndarray

from tools.raco_extract import (
    ExtractConfig,
    FrameTask,
    _pad_images_to_batch,
    build_message_types,
    collect_frame_tasks,
    normalized_to_pixels,
    prepare_metadata_collection,
    sample_keypoint_rgb,
    serialize_keyframe,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DESCRIPTOR_SET: Path = REPO_ROOT / "data" / "cusfm_schema" / "cusfm_protos.fdset"
REFERENCE_INPUT_METADATA: Path = Path("data/cusfm_runs/robocap_full/input/frames_meta.json")
REFERENCE_OUTPUT_METADATA: Path = Path("data/cusfm_runs/robocap_full/cusfm/keyframes/frames_meta.json")
BATCHED_MODEL: Path = REPO_ROOT / "data" / "cusfm_models" / "raco-aliked-b1-16.onnx"
"""Exported by `pixi run -e raco raco-export --batched-extractor-path
data/cusfm_models/raco-aliked-b1-16.onnx`; absent on a fresh clone, so the
dependent test skips."""

#: Tests needing an artifact a fresh clone does not have, rather than the
#: vendored schema, skip instead of failing.
requires_batched_model = pytest.mark.skipif(
    not BATCHED_MODEL.is_file(), reason=f"needs an exported model at {BATCHED_MODEL}"
)
requires_reference_run = pytest.mark.skipif(
    not REFERENCE_OUTPUT_METADATA.is_file(),
    reason="needs a completed cuSFM run under data/cusfm_runs/robocap_full",
)


def test_extract_config_exposes_precision_policy_at_cli_seam() -> None:
    """The CLI distinguishes plain FP16 from FP8 Q/DQ with FP16 fallback."""
    default_config: ExtractConfig = ExtractConfig()
    fp8_config: ExtractConfig = ExtractConfig(precision="fp8_qdq_with_fp16_fallback")

    assert default_config.precision == "fp16"
    assert fp8_config.precision == "fp8_qdq_with_fp16_fallback"


def test_normalized_to_pixels_matches_cusfm_align_corners_mapping() -> None:
    """Normalized extrema and center map to cuSFM's original-image coordinates."""
    normalized_n2: Float32[ndarray, "3 2"] = np.asarray(
        [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32
    )
    expected_n2: Float32[ndarray, "3 2"] = np.asarray(
        [[0.0, 0.0], [1919.0, 1079.0], [959.5, 539.5]], dtype=np.float32
    )

    actual_n2: Float32[ndarray, "3 2"] = normalized_to_pixels(
        normalized_n2, image_width=1920, image_height=1080
    )

    np.testing.assert_array_equal(actual_n2, expected_n2)


def test_sample_keypoint_rgb_uses_nearest_original_pixel() -> None:
    """RGB comes from nearest-integer coordinates in the unscaled source image."""
    image_rgb_hwc: UInt8[ndarray, "2 3 3"] = np.asarray(
        [
            [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
            [[40, 41, 42], [50, 51, 52], [60, 61, 62]],
        ],
        dtype=np.uint8,
    )
    keypoints_n2: Float32[ndarray, "2 2"] = np.asarray(
        [[0.51, 0.49], [1.51, 0.51]], dtype=np.float32
    )
    expected_rgb_n3: UInt8[ndarray, "2 3"] = np.asarray(
        [[20, 21, 22], [60, 61, 62]], dtype=np.uint8
    )

    actual_rgb_n3: UInt8[ndarray, "2 3"] = sample_keypoint_rgb(image_rgb_hwc, keypoints_n2)

    np.testing.assert_array_equal(actual_rgb_n3, expected_rgb_n3)


def test_static_batch_padding_repeats_last_image() -> None:
    """A resumed final batch can execute a fixed engine without creating extra files."""
    images_bchw: Float32[ndarray, "3 3 2 2"] = np.arange(
        3 * 3 * 2 * 2, dtype=np.float32
    ).reshape(3, 3, 2, 2)

    padded_bchw: Float32[ndarray, "8 3 2 2"] = _pad_images_to_batch(images_bchw, 8)

    assert padded_bchw.shape == (8, 3, 2, 2)
    np.testing.assert_array_equal(padded_bchw[:3], images_bchw)
    np.testing.assert_array_equal(padded_bchw[3:], np.repeat(images_bchw[-1:], 5, axis=0))


def test_serialize_keyframe_matches_cusfm_wire_contract() -> None:
    """A serialized keyframe exposes every populated cuSFM field at the file seam."""
    message_types: dict[str, type[Message]] = build_message_types(DESCRIPTOR_SET)
    keypoints_n2: Float32[ndarray, "2 2"] = np.asarray(
        [[2.25, 3.75], [4.5, 5.5]], dtype=np.float32
    )
    descriptors_nd: Float32[ndarray, "2 128"] = np.zeros((2, 128), dtype=np.float32)
    descriptors_nd[0, 0] = 1.0
    descriptors_nd[1, 1] = 1.0
    colors_rgb_n3: UInt8[ndarray, "2 3"] = np.asarray(
        [[1, 2, 3], [4, 5, 6]], dtype=np.uint8
    )

    payload: bytes = serialize_keyframe(
        message_types=message_types,
        keypoints_n2=keypoints_n2,
        descriptors_nd=descriptors_nd,
        colors_rgb_n3=colors_rgb_n3,
        image_width=1920,
        image_height=1080,
    )
    keyframe: Message = message_types["keyframe"]()
    keyframe.ParseFromString(payload)
    dynamic_keyframe: Any = keyframe
    vector: Any = dynamic_keyframe.keypoint_vector

    assert list(dynamic_keyframe.inliers) == [False, False]
    assert dynamic_keyframe.is_valid is True
    assert list(vector.x) == [2.25, 4.5]
    assert list(vector.y) == [3.75, 5.5]
    np.testing.assert_array_equal(
        np.frombuffer(vector.descriptors, dtype="<f4").reshape(2, 128), descriptors_nd
    )
    assert vector.descriptor_type == 4
    assert vector.bev_start_idx == -1
    assert (vector.image_width, vector.image_height) == (1920, 1080)
    assert list(vector.depth) == [-1.0, -1.0]
    assert list(vector.r) == [1, 4]
    assert list(vector.g) == [2, 5]
    assert list(vector.b) == [3, 6]


@requires_reference_run
def test_prepare_metadata_collection_matches_cusfm_reference_semantics() -> None:
    """Generated frames metadata is protobuf-equivalent to cuSFM's real output."""
    message_types: dict[str, type[Message]] = build_message_types(DESCRIPTOR_SET)
    input_metadata: dict[str, Any] = json.loads(REFERENCE_INPUT_METADATA.read_text())
    expected_json: str = REFERENCE_OUTPUT_METADATA.read_text()

    actual: Message = prepare_metadata_collection(message_types, input_metadata)
    expected: Message = message_types["metadata_collection"]()
    from google.protobuf import json_format

    json_format.Parse(expected_json, expected)

    assert actual == expected


@requires_reference_run
def test_collect_frame_tasks_covers_all_four_cameras_in_metadata_order() -> None:
    """The writer resolves all 4,528 images and timestamp-based destinations."""
    input_metadata: dict[str, Any] = json.loads(REFERENCE_INPUT_METADATA.read_text())

    tasks: list[FrameTask] = collect_frame_tasks(
        REFERENCE_INPUT_METADATA.parent,
        Path("data/cusfm_runs/robocap_raco_b8/cusfm/keyframes"),
        input_metadata,
    )

    assert len(tasks) == 4528
    assert [task.image_path.parent.name for task in tasks[:4]] == [
        "left_front",
        "right_front",
        "left",
        "right",
    ]
    assert tasks[0].output_path.as_posix().endswith("left_front/101622069778.pb")
    assert (tasks[0].image_width, tasks[0].image_height) == (1920, 1080)


@requires_batched_model
def test_dynamic_extractor_onnx_covers_benchmark_batch_range() -> None:
    """The exported writer model preserves one dynamic batch on every tensor."""
    model: onnx.ModelProto = onnx.load(BATCHED_MODEL)
    onnx.checker.check_model(model, full_check=True)

    inputs: list[tuple[str, list[int | str]]] = [
        (
            value.name,
            [dimension.dim_value or dimension.dim_param for dimension in value.type.tensor_type.shape.dim],
        )
        for value in model.graph.input
    ]
    outputs: list[tuple[str, list[int | str]]] = [
        (
            value.name,
            [dimension.dim_value or dimension.dim_param for dimension in value.type.tensor_type.shape.dim],
        )
        for value in model.graph.output
    ]

    assert inputs == [("image", ["batch", 3, 1200, 1920])]
    assert outputs == [
        ("keypoints", ["batch", 2048, 2]),
        ("descriptors", ["batch", 2048, 128]),
        ("scores", ["batch", 2048]),
    ]

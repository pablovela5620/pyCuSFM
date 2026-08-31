"""Compatibility tests for the cuSFM RaCo-ALIKED model replacement."""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto

from tools.export_cusfm_raco import CusfmExtractor, ExportConfig, static_output_names

MODEL_DIRECTORIES: tuple[Path, ...] = (
    Path("data/cusfm_models/raco/aliked_lightglue"),
    Path("data/cusfm_models/raco_fp8/aliked_lightglue"),
)
"""FP16 and FP8 directories below the cuSFM model-root compatibility layer."""


def test_extractor_can_select_native_deformable_convolution() -> None:
    """The direct-writer export can use TensorRT's native DeformConv path."""
    module: CusfmExtractor = CusfmExtractor(ExportConfig(portable_deform_conv=False))

    portable_flags: list[bool] = [
        child.portable
        for child in module.modules()
        if hasattr(child, "portable")
    ]

    assert portable_flags == [False, False, False, False]


def test_export_config_accepts_a_fixed_unrolled_batch() -> None:
    """A static writer graph can avoid TensorRT's dynamic batch fusion."""
    config: ExportConfig = ExportConfig(static_batch_size=8)

    assert config.static_batch_size == 8


def test_static_output_names_keep_branches_separate() -> None:
    """Per-image graph outputs prevent TensorRT from fusing the final concatenation."""
    assert static_output_names(2) == [
        "keypoints_0",
        "descriptors_0",
        "scores_0",
        "keypoints_1",
        "descriptors_1",
        "scores_1",
    ]


def _dimensions(value: onnx.ValueInfoProto) -> tuple[int | str | None, ...]:
    """Return concrete and symbolic dimensions from an ONNX value declaration."""
    dimensions: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return tuple(dimensions)


@pytest.mark.parametrize("model_directory", MODEL_DIRECTORIES)
def test_extractor_matches_cusfm_contract(model_directory: Path) -> None:
    """cuSFM can load the replacement extractor without an adapter."""
    if not (model_directory / "aliked.onnx").is_file():
        pytest.skip(f"no exported model at {model_directory}; run `pixi run -e raco raco-export`")
    model: onnx.ModelProto = onnx.load(model_directory / "aliked.onnx")
    onnx.checker.check_model(model, full_check=True)

    assert [(value.name, _dimensions(value), value.type.tensor_type.elem_type) for value in model.graph.input] == [
        ("image", (1, 3, 1200, 1920), TensorProto.FLOAT)
    ]
    assert [(value.name, _dimensions(value), value.type.tensor_type.elem_type) for value in model.graph.output] == [
        ("keypoints", (2048, 2), TensorProto.FLOAT),
        ("descriptors", (2048, 128), TensorProto.FLOAT),
        ("scores", (2048,), TensorProto.FLOAT),
    ]

    operation_types: list[str] = [node.op_type for node in model.graph.node]
    assert "DeformConv" not in operation_types
    assert "GridSample" in operation_types
    assert "BatchNormalization" not in operation_types
    assert operation_types.count("TopK") >= 3


@pytest.mark.parametrize("model_directory", MODEL_DIRECTORIES)
def test_matcher_matches_cusfm_contract(model_directory: Path) -> None:
    """cuSFM can load the RaCo-trained LightGlue matcher without an adapter."""
    if not (model_directory / "lightglue_aliked.onnx").is_file():
        pytest.skip(f"no exported model at {model_directory}; run `pixi run -e raco raco-export`")
    model: onnx.ModelProto = onnx.load(model_directory / "lightglue_aliked.onnx")
    onnx.checker.check_model(model, full_check=True)

    input_contract: list[tuple[str, tuple[int | str | None, ...], int]] = [
        (value.name, _dimensions(value), value.type.tensor_type.elem_type) for value in model.graph.input
    ]
    assert [name for name, _shape, _dtype in input_contract] == ["kpts0", "kpts1", "desc0", "desc1"]
    assert [shape[0] for _name, shape, _dtype in input_contract] == [1, 1, 1, 1]
    assert [shape[-1] for _name, shape, _dtype in input_contract] == [2, 2, 128, 128]
    assert all(dtype == TensorProto.FLOAT for _name, _shape, dtype in input_contract)
    assert all(isinstance(shape[1], str) for _name, shape, _dtype in input_contract)

    output_contract: list[tuple[str, tuple[int | str | None, ...], int]] = [
        (value.name, _dimensions(value), value.type.tensor_type.elem_type) for value in model.graph.output
    ]
    assert [name for name, _shape, _dtype in output_contract] == ["matches0", "mscores0"]
    assert output_contract[0][1][0] == input_contract[1][1][1]
    assert output_contract[1][1][0] == input_contract[1][1][1]
    assert output_contract[0][1][-1] == 2
    assert output_contract[0][2] == TensorProto.INT32
    assert len(output_contract[1][1]) == 1
    assert output_contract[1][2] == TensorProto.FLOAT

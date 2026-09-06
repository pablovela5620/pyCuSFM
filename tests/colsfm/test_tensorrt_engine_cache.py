"""What names a cached TensorRT engine, and which cached engine a graph accepts.

The seam is `engine_path_for` and `resolve_engine`. Neither needs a device here:
the two machine-dependent parts of the name, the TensorRT version and the compute
capability, are pinned to fixed strings so the tests assert the naming rule rather
than this host's GPU. `colsfm.tensorrt_runtime` still imports `tensorrt` and
`cuda-python` at module scope, hence the two `importorskip` calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorrt", reason="colsfm.tensorrt_runtime imports tensorrt at module scope")
pytest.importorskip("cuda.bindings.runtime", reason="colsfm.tensorrt_runtime allocates through cuda-python")

from colsfm import tensorrt_runtime
from colsfm.tensorrt_runtime import blob_engine_path_for, engine_path_for, resolve_engine


@pytest.fixture(autouse=True)
def fixed_machine_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the TensorRT version and the GPU architecture the engine name carries.

    Args:
        monkeypatch: pytest's patcher.
    """
    monkeypatch.setattr(tensorrt_runtime, "compute_capability", lambda device_index=0: "12_0")
    monkeypatch.setattr(tensorrt_runtime, "tensorrt_version_tag", lambda: "10_13_3_9")


def test_two_graphs_at_one_path_cannot_share_one_cached_engine(tmp_path: Path) -> None:
    """The engine name follows the graph's bytes, not just its file name.

    Replacing an ONNX graph in place left the stale engine matching by name, so
    the next run silently executed the old graph.
    """
    onnx_path: Path = tmp_path / "aliked.onnx"

    onnx_path.write_bytes(b"first graph")
    first: Path = engine_path_for(onnx_path)
    onnx_path.write_bytes(b"second graph")
    second: Path = engine_path_for(onnx_path)
    onnx_path.write_bytes(b"first graph")
    again: Path = engine_path_for(onnx_path)

    assert first != second
    assert first == again
    assert first.parent == tmp_path
    assert first.name.startswith("aliked_fp16_")
    assert first.name.endswith("_10_13_3_9_sm_12_0.engine")


def test_the_profile_tag_still_separates_engines_built_from_one_graph(tmp_path: Path) -> None:
    """Two profiles over the same bytes stay two engines; the digest does not merge them."""
    onnx_path: Path = tmp_path / "raco-aliked-dyn.onnx"
    onnx_path.write_bytes(b"one graph")

    assert engine_path_for(onnx_path, profile_tag="b1-8") != engine_path_for(onnx_path, profile_tag="b1-16")


def test_a_graph_that_is_not_there_keeps_the_blobs_own_engine_name(tmp_path: Path) -> None:
    """With no graph to digest, the name stays the one cuSFM's own runtime writes."""
    assert engine_path_for(tmp_path / "aliked.onnx").name == "aliked_fp16_10_13_3_9_sm_12_0.engine"


def test_a_prebuilt_blob_engine_is_accepted_rather_than_rebuilt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The engines cuSFM itself built next to its own graphs stay usable.

    They cost about 205 s each to rebuild and carry no digest in their name, so
    `resolve_engine` accepts that spelling explicitly — but only inside the blob's
    own model directory.
    """
    monkeypatch.setattr(tensorrt_runtime, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(tensorrt_runtime, "build_fp16_engine", _refuse_to_build)
    onnx_path: Path = tmp_path / "aliked.onnx"
    onnx_path.write_bytes(b"the blob's own graph")
    blob_engine: Path = blob_engine_path_for(onnx_path)
    blob_engine.write_bytes(b"a serialised engine")

    assert resolve_engine(onnx_path) == blob_engine


def test_a_stale_undigested_engine_elsewhere_is_not_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside the blob's model directory only the digest-named engine counts."""
    monkeypatch.setattr(tensorrt_runtime, "MODEL_DIR", tmp_path / "blob")
    built: list[tuple[Path, Path]] = []

    def record_build(onnx_path: Path, engine_path: Path, profiles: object = None, **_keywords: object) -> Path:
        """Stand in for the TensorRT build, recording what it was asked for.

        Args:
            onnx_path: The graph to build from.
            engine_path: Where the engine would be written.
            profiles: The optimisation profiles, unused here.
            **_keywords: The builder settings, unused here.

        Returns:
            `engine_path`, as the real builder does.
        """
        built.append((onnx_path, engine_path))
        engine_path.write_bytes(b"a freshly built engine")
        return engine_path

    monkeypatch.setattr(tensorrt_runtime, "build_fp16_engine", record_build)
    onnx_path: Path = tmp_path / "raco-aliked-dyn.onnx"
    onnx_path.write_bytes(b"a locally generated graph")
    blob_engine_path_for(onnx_path).write_bytes(b"an engine built from an older graph")

    resolved: Path = resolve_engine(onnx_path)

    assert built == [(onnx_path, engine_path_for(onnx_path))]
    assert resolved == engine_path_for(onnx_path)
    assert resolved != blob_engine_path_for(onnx_path)


def test_a_cached_digest_named_engine_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The digest name is a cache, not a rebuild: an existing one is returned as is."""
    monkeypatch.setattr(tensorrt_runtime, "build_fp16_engine", _refuse_to_build)
    onnx_path: Path = tmp_path / "raco-aliked-dyn.onnx"
    onnx_path.write_bytes(b"a locally generated graph")
    cached: Path = engine_path_for(onnx_path)
    cached.write_bytes(b"a serialised engine")

    assert resolve_engine(onnx_path) == cached


def _refuse_to_build(onnx_path: Path, engine_path: Path, profiles: object = None, **_keywords: object) -> Path:
    """Fail the test rather than spend minutes in TensorRT.

    Args:
        onnx_path: The graph the caller wanted built.
        engine_path: Where it wanted the engine.
        profiles: The optimisation profiles, unused here.
        **_keywords: The builder settings, unused here.

    Returns:
        Never; it always raises.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError(f"resolve_engine rebuilt {onnx_path} into {engine_path} instead of using the cached engine")

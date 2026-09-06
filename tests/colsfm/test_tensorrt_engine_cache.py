"""What names a cached TensorRT engine, and which cached engine a graph accepts.

The seam is `engine_build_identity`, `engine_path_for` and `resolve_engine`.
Nothing here needs a device: the three machine-dependent parts of the name — the
TensorRT version tag, the full runtime version and the compute capability — are
pinned to fixed strings, so the tests assert the naming rule rather than this
host's GPU. `colsfm.tensorrt_runtime` still imports `tensorrt` and `cuda-python`
at module scope, hence the two `importorskip` calls.

Caching is asserted by counting builds, never by object identity or elapsed
time: a `resolve_engine` that rebuilt would be a 205-second regression, and the
only honest way to see it is to count how often the builder was called.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorrt", reason="colsfm.tensorrt_runtime imports tensorrt at module scope")
pytest.importorskip("cuda.bindings.runtime", reason="colsfm.tensorrt_runtime allocates through cuda-python")

from colsfm import tensorrt_runtime
from colsfm.tensorrt_runtime import (
    ShapeProfile,
    blob_engine_path_for,
    engine_build_identity,
    engine_path_for,
    publish_atomically,
    resolve_engine,
    trusted_prebuilt_engine,
)

ONE_PROFILE: dict[str, ShapeProfile] = {"image": ((1, 3, 256, 256), (8, 3, 768, 1024), (8, 3, 1216, 1920))}
"""A profile the tests vary one shape of at a time."""


@pytest.fixture(autouse=True)
def fixed_machine_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every machine-dependent input to the engine's name and identity.

    Args:
        monkeypatch: pytest's patcher.
    """
    monkeypatch.setattr(tensorrt_runtime, "compute_capability", lambda device_index=0: "12_0")
    monkeypatch.setattr(tensorrt_runtime, "tensorrt_version_tag", lambda: "10_13_3_9")
    monkeypatch.setattr(tensorrt_runtime, "tensorrt_runtime_version", lambda: "10.13.3.9.post1")


class BuildCounter:
    """A stand-in for `build_fp16_engine` that records every call.

    Object identity and wall clocks say nothing useful about a cache whose miss
    costs minutes; the number of builds says everything.
    """

    def __init__(self) -> None:
        """Start with nothing built."""
        self.calls: list[tuple[Path, Path]] = []
        """One `(onnx_path, engine_path)` per build, in order."""

    def __call__(self, onnx_path: Path, engine_path: Path, profiles: object = None, **_keywords: object) -> Path:
        """Record the build and write a placeholder engine.

        Args:
            onnx_path: The graph to build from.
            engine_path: Where the engine would be written.
            profiles: The optimisation profiles, unused here.
            **_keywords: The builder settings, unused here.

        Returns:
            `engine_path`, as the real builder does.
        """
        self.calls.append((onnx_path, engine_path))
        engine_path.write_bytes(b"a freshly built engine")
        return engine_path


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> BuildCounter:
    """Replace the TensorRT build with a counter.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        The counter every `resolve_engine` build lands in.
    """
    counter: BuildCounter = BuildCounter()
    monkeypatch.setattr(tensorrt_runtime, "build_fp16_engine", counter)
    return counter


# ── What goes into the name ──────────────────────────────────────────────────


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


def test_a_retuned_profile_names_a_different_engine(tmp_path: Path) -> None:
    """Profiles decide which tactics TensorRT picks, so they decide the engine.

    This is the hole a graph-only digest left: `raco_profile()` carries no tag,
    so widening its batch ceiling used to reuse the engine built for the old one
    and fail at the first `set_input_shape` — minutes into a run, not at the
    cache lookup.
    """
    onnx_path: Path = tmp_path / "raco-aliked-b1-16.onnx"
    onnx_path.write_bytes(b"one graph")
    wider: dict[str, ShapeProfile] = {"image": (ONE_PROFILE["image"][0], ONE_PROFILE["image"][1], (16, 3, 1216, 1920))}

    assert engine_path_for(onnx_path, ONE_PROFILE) != engine_path_for(onnx_path, wider)
    assert engine_path_for(onnx_path, ONE_PROFILE) != engine_path_for(onnx_path)


def test_the_profile_mapping_digests_the_same_whatever_order_it_was_built_in(tmp_path: Path) -> None:
    """Two equal profile mappings are one identity, so insertion order cannot split the cache."""
    onnx_path: Path = tmp_path / "two-input.onnx"
    onnx_path.write_bytes(b"one graph")
    shape: ShapeProfile = ((1,), (2,), (3,))
    forwards: dict[str, ShapeProfile] = {"image": shape, "mask": shape}
    backwards: dict[str, ShapeProfile] = {"mask": shape, "image": shape}

    assert engine_path_for(onnx_path, forwards) == engine_path_for(onnx_path, backwards)


def test_the_builder_settings_are_part_of_the_identity(tmp_path: Path) -> None:
    """Workspace and optimisation level change the tactics, so they change the engine."""
    onnx_path: Path = tmp_path / "raco-aliked-dyn.onnx"
    onnx_path.write_bytes(b"one graph")

    baseline: str = engine_build_identity(onnx_path)

    assert engine_build_identity(onnx_path, workspace_gib=1) != baseline
    assert engine_build_identity(onnx_path, optimization_level=5) != baseline


def test_the_runtime_version_and_the_architecture_are_part_of_the_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A serialised engine is valid for one TensorRT and one architecture, and says so."""
    onnx_path: Path = tmp_path / "raco-aliked-dyn.onnx"
    onnx_path.write_bytes(b"one graph")
    baseline: str = engine_build_identity(onnx_path)

    monkeypatch.setattr(tensorrt_runtime, "tensorrt_runtime_version", lambda: "10.14.0.0")
    assert engine_build_identity(onnx_path) != baseline

    monkeypatch.setattr(tensorrt_runtime, "tensorrt_runtime_version", lambda: "10.13.3.9.post1")
    monkeypatch.setattr(tensorrt_runtime, "compute_capability", lambda device_index=0: "8_9")
    assert engine_build_identity(onnx_path) != baseline


def test_the_profile_tag_still_separates_engines_built_from_one_graph(tmp_path: Path) -> None:
    """The readable tag stays, so a directory listing tells two profiles apart."""
    onnx_path: Path = tmp_path / "raco-aliked-dyn.onnx"
    onnx_path.write_bytes(b"one graph")

    assert engine_path_for(onnx_path, profile_tag="b1-8") != engine_path_for(onnx_path, profile_tag="b1-16")
    assert "_b1-8_" in engine_path_for(onnx_path, profile_tag="b1-8").name


def test_a_graph_that_is_not_there_keeps_the_blobs_own_engine_name(tmp_path: Path) -> None:
    """With no graph to digest, the name stays the one cuSFM's own runtime writes."""
    assert engine_path_for(tmp_path / "aliked.onnx").name == "aliked_fp16_10_13_3_9_sm_12_0.engine"


# ── Which engine is accepted ─────────────────────────────────────────────────


def test_a_prebuilt_blob_engine_is_accepted_rather_than_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builds: BuildCounter
) -> None:
    """The engines cuSFM itself built next to its own graphs stay usable.

    They cost about 205 s each to rebuild and carry no build identity, so
    `resolve_engine` accepts that spelling explicitly — but only for the two
    graphs the blob ships, and only inside its own model directory.
    """
    monkeypatch.setattr(tensorrt_runtime, "MODEL_DIR", tmp_path)
    onnx_path: Path = tmp_path / "aliked.onnx"
    onnx_path.write_bytes(b"the blob's own graph")
    blob_engine: Path = blob_engine_path_for(onnx_path)
    blob_engine.write_bytes(b"a serialised engine")

    assert resolve_engine(onnx_path) == blob_engine
    assert builds.calls == []


def test_an_untrusted_graph_in_the_model_directory_gets_no_prebuilt_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builds: BuildCounter
) -> None:
    """The allowance names two graphs; sitting in the right directory is not enough.

    Otherwise dropping any ONNX beside the blob's, next to a same-named engine,
    would run an engine whose only provenance is a file name.
    """
    monkeypatch.setattr(tensorrt_runtime, "MODEL_DIR", tmp_path)
    onnx_path: Path = tmp_path / "somebody-elses.onnx"
    onnx_path.write_bytes(b"a graph the blob never shipped")
    blob_engine_path_for(onnx_path).write_bytes(b"an engine of unknown provenance")

    assert trusted_prebuilt_engine(onnx_path) is None
    assert resolve_engine(onnx_path) == engine_path_for(onnx_path)
    assert builds.calls == [(onnx_path, engine_path_for(onnx_path))]


def test_a_stale_undigested_engine_elsewhere_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builds: BuildCounter
) -> None:
    """Outside the blob's model directory only the identity-named engine counts."""
    monkeypatch.setattr(tensorrt_runtime, "MODEL_DIR", tmp_path / "blob")
    onnx_path: Path = tmp_path / "raco-aliked-dyn.onnx"
    onnx_path.write_bytes(b"a locally generated graph")
    blob_engine_path_for(onnx_path).write_bytes(b"an engine built from an older graph")

    resolved: Path = resolve_engine(onnx_path)

    assert builds.calls == [(onnx_path, engine_path_for(onnx_path))]
    assert resolved == engine_path_for(onnx_path)
    assert resolved != blob_engine_path_for(onnx_path)


def test_a_cached_engine_is_reused_and_the_builder_is_never_called_twice(
    tmp_path: Path, builds: BuildCounter
) -> None:
    """The identity name is a cache: one build, then hits.

    Counted rather than timed. A wall-clock guard here would measure this
    machine; the call count measures the contract.
    """
    onnx_path: Path = tmp_path / "raco-aliked-dyn.onnx"
    onnx_path.write_bytes(b"a locally generated graph")

    first: Path = resolve_engine(onnx_path, ONE_PROFILE, profile_tag="b1-8")
    second: Path = resolve_engine(onnx_path, ONE_PROFILE, profile_tag="b1-8")

    assert first == second
    assert len(builds.calls) == 1


def test_the_engine_a_build_writes_is_the_one_the_name_was_computed_from(
    tmp_path: Path, builds: BuildCounter
) -> None:
    """Look-up and build share one set of settings, so a hit is never a different engine.

    Passing the settings to `engine_path_for` but not to `build_fp16_engine`
    would write an engine at a name that promised something else, and every later
    run would trust it.
    """
    onnx_path: Path = tmp_path / "raco-aliked-b1-16.onnx"
    onnx_path.write_bytes(b"a locally generated graph")

    resolved: Path = resolve_engine(onnx_path, ONE_PROFILE, workspace_gib=1, optimization_level=5)

    assert builds.calls == [
        (onnx_path, engine_path_for(onnx_path, ONE_PROFILE, workspace_gib=1, optimization_level=5))
    ]
    assert resolved.is_file()
    assert resolve_engine(onnx_path, ONE_PROFILE, workspace_gib=1, optimization_level=5) == resolved
    assert len(builds.calls) == 1


# ── Publication ──────────────────────────────────────────────────────────────


def test_publication_renames_a_unique_temporary_and_leaves_none_behind(tmp_path: Path) -> None:
    """Two publications of one destination must not share a temporary name."""
    destination: Path = tmp_path / "nested" / "engine.engine"

    publish_atomically(destination, b"first")
    publish_atomically(destination, b"second")

    assert destination.read_bytes() == b"second"
    assert sorted(path.name for path in destination.parent.iterdir()) == ["engine.engine"]


def test_a_failed_publication_leaves_the_previous_file_intact(tmp_path: Path) -> None:
    """A build that dies mid-write must not replace a good engine with a stub."""
    destination: Path = tmp_path / "engine.engine"
    destination.write_bytes(b"a good engine")

    with pytest.raises(TypeError):
        publish_atomically(destination, "not bytes")  # type: ignore[arg-type]

    assert destination.read_bytes() == b"a good engine"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["engine.engine"]

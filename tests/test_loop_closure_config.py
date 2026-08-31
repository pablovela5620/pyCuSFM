"""Guard the loop-closure config copy against silent drift from upstream.

``data/cusfm_configs/loop-closure-fixed`` is a full copy of
``pycusfm/configs/isaac`` because upstream is frozen and must not be edited. The
cost of that isolation is that the intentional five-line delta is invisible
inside ~312 KB of duplicate, and an upstream version bump would silently keep
stale values for the fifteen files nobody meant to change.

These tests pin the delta exactly, so any drift -- in either direction -- fails
loudly instead of changing reconstruction behaviour unnoticed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
UPSTREAM_DIR: Path = REPO_ROOT / "pycusfm" / "configs" / "isaac"
PATCHED_DIR: Path = REPO_ROOT / "data" / "cusfm_configs" / "loop-closure-fixed"

# The complete intended difference. Anything outside this set is drift.
#   1. The epipolar gate rejected every candidate that reached geometric
#      verification: 22091 BoW candidates narrowed to 180 stereo attempts, of
#      which 168 failed on this threshold (residuals 2.8e-5 .. 6.0e-4).
#   2. `loop_edge_*` are absent upstream, so proto3 defaults them to 0 and
#      `FindBestLoopCandidateForFrame` compares with a plain `>` -- a 0 m / 0 deg
#      limit rejects every nonzero relative pose.
EXPECTED_CHANGED_FILES: frozenset[str] = frozenset(
    {"association_config.pb.txt", "localizer_config.pb.txt", "pose_graph_config.pb.txt"}
)
EPIPOLAR_FILES: frozenset[str] = frozenset(
    {"association_config.pb.txt", "localizer_config.pb.txt"}
)
PATCHED_EPIPOLAR: float = 1e-4
LOOP_EDGE_SETTINGS: dict[str, float] = {
    "loop_edge_translation_threshold_meters": 1.0,
    "loop_edge_rotation_threshold_degrees": 10.0,
}

pytestmark = pytest.mark.skipif(
    not UPSTREAM_DIR.is_dir() or not PATCHED_DIR.is_dir(),
    reason="requires both the frozen upstream config tree and the patched copy",
)


def _epipolar_values(text: str) -> list[float]:
    return [float(v) for v in re.findall(r"max_mean_point_to_epipolarline_error:\s*(\S+)", text)]


def test_patched_tree_has_the_same_files_as_upstream() -> None:
    """A missing file would make cuSFM fall back to a default, not error."""
    upstream: set[str] = {path.name for path in UPSTREAM_DIR.glob("*.pb.txt")}
    patched: set[str] = {path.name for path in PATCHED_DIR.glob("*.pb.txt")}
    assert patched == upstream


def test_exactly_three_files_differ_from_upstream() -> None:
    """Pins the blast radius: fifteen files must stay byte-identical."""
    differing: set[str] = {
        path.name
        for path in PATCHED_DIR.glob("*.pb.txt")
        if path.read_bytes() != (UPSTREAM_DIR / path.name).read_bytes()
    }
    assert differing == set(EXPECTED_CHANGED_FILES)


@pytest.mark.parametrize("filename", sorted(EPIPOLAR_FILES))
def test_epipolar_gate_is_relaxed(filename: str) -> None:
    """Upstream ships `10e-6` (=1e-5), tighter than any residual we measured."""
    upstream_values: list[float] = _epipolar_values((UPSTREAM_DIR / filename).read_text())
    patched_values: list[float] = _epipolar_values((PATCHED_DIR / filename).read_text())
    assert upstream_values, f"{filename} upstream should declare the epipolar gate"
    assert len(patched_values) == len(upstream_values)
    assert all(value == pytest.approx(PATCHED_EPIPOLAR) for value in patched_values)
    assert all(value < PATCHED_EPIPOLAR for value in upstream_values)


@pytest.mark.parametrize(("field", "expected"), sorted(LOOP_EDGE_SETTINGS.items()))
def test_loop_edge_thresholds_are_set_explicitly(field: str, expected: float) -> None:
    """Absent upstream, so proto3 would default them to a reject-everything 0."""
    upstream_text: str = (UPSTREAM_DIR / "pose_graph_config.pb.txt").read_text()
    patched_text: str = (PATCHED_DIR / "pose_graph_config.pb.txt").read_text()
    assert field not in upstream_text, f"upstream now sets {field}; revisit this patch"
    found: list[str] = re.findall(rf"^{field}:\s*(\S+)", patched_text, re.MULTILINE)
    assert len(found) == 1
    assert float(found[0]) == pytest.approx(expected)
    assert float(found[0]) > 0.0, "a zero threshold rejects every candidate"

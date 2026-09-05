"""Tests for `colsfm.pose_graph`, the Python replacement for cuSFM `pose_graph_main`.

Seams under test (all public):

* `sequential_edges` / `loop_edge_information` / `gate_loop_edges` — graph construction.
* `solve_pose_graph` — the Ceres problem and its result.

The Galileo parity tests read the shipped cuSFM artifacts directly with the small
readers in this file; they deliberately do not import `colsfm.frames_meta`, which is
owned by another worker and not a dependency of the pose-graph stage.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float
from scipy.spatial.transform import Rotation

from colsfm.pose_graph import (
    PoseGraphEdge,
    PoseGraphSolveOptions,
    RelativePoseCost,
    RigNode,
    apply_rig_update,
    default_information,
    gate_loop_edges,
    loop_edge_information,
    relative_pose_error,
    sequential_edges,
    solve_pose_graph,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
# The two archived galileo runs differ in more than size: the 4-rig one ran the stock isaac
# config and logs "0 loop constraints", while the 29-rig one under `cusfm/` used a locally
# patched config with the loop gates opened and produced 8 loop edges
# (docs/spec/pose_graph_main.md §1, §9.2). Its output matches the §9.2 `pg_loop` rerun to
# 4.8e-15 m, so the committed copy is used here and `pg_loop` only as a fallback.
GALILEO_NO_LOOP_KEYFRAMES: Path = REPO_ROOT / "data/cusfm_runs/galileo/keyframes/frames_meta.json"
GALILEO_NO_LOOP_RUN: Path = REPO_ROOT / "data/cusfm_runs/galileo/pose_graph"
GALILEO_LOOP_KEYFRAMES: Path = REPO_ROOT / "data/cusfm_runs/galileo/cusfm/keyframes/frames_meta.json"
GALILEO_LOOP_RUN: Path = REPO_ROOT / "data/cusfm_runs/galileo/cusfm/pose_graph"
GALILEO_STOCK_RERUN: Path = REPO_ROOT / "data/cusfm_re/runs/pg_stock"
ISAAC_POSE_GRAPH_CONFIG: Path = REPO_ROOT / "pycusfm/configs/isaac/pose_graph_config.pb.txt"

TextProto: TypeAlias = dict[str, list["TextProto | float | int | str"]]


# --------------------------------------------------------------------------------------
# test-only readers for the shipped cuSFM artifacts
# --------------------------------------------------------------------------------------


def read_text_proto(text: str) -> TextProto:
    """Parse a protobuf text-format message into nested dicts of repeated fields.

    Every field maps to a list, because the text format repeats keys for repeated
    fields; scalars therefore read as `msg["field"][0]`. Enough for the cuSFM
    `pose_graph.pb.txt` and `pose_graph_config.pb.txt` files, which use only nested
    messages, numbers, enum names and quoted strings.

    Args:
        text: The whole text-proto file.

    Returns:
        The parsed top-level message.
    """
    tokens: list[str] = re.findall(r'"[^"]*"|[{}]|[^\s{}]+', re.sub(r"#[^\n]*", "", text))

    def parse_block(start: int) -> tuple[TextProto, int]:
        message: TextProto = {}
        index: int = start
        while index < len(tokens):
            token: str = tokens[index]
            if token == "}":
                return message, index + 1
            key: str = token.rstrip(":")
            if tokens[index + 1] == "{":
                value, index = parse_block(index + 2)
            else:
                raw: str = tokens[index + 1]
                index += 2
                try:
                    value = float(raw) if re.search(r"[.eE]", raw) else int(raw)
                except ValueError:
                    value = raw.strip('"')
            message.setdefault(key, []).append(value)
        return message, index

    parsed, _ = parse_block(0)
    return parsed


def rigid_from_axis_angle(message: dict[str, object]) -> pycolmap.Rigid3d:
    """Build a `Rigid3d` from a `protos.common.geometry.RigidTransform3d` JSON object.

    The proto stores a unit axis plus an angle in DEGREES, and omits zero-valued
    fields (proto3), so every component defaults to 0.

    Args:
        message: The decoded JSON object with `axis_angle` and `translation`.

    Returns:
        The transform, with the rotation converted to radians.
    """
    axis_angle: dict[str, float] = message.get("axis_angle", {})
    translation_json: dict[str, float] = message.get("translation", {})
    axis: Float[np.ndarray, "3"] = np.array(
        [axis_angle.get("x", 0.0), axis_angle.get("y", 0.0), axis_angle.get("z", 0.0)], dtype=np.float64
    )
    angle_rad: float = np.deg2rad(axis_angle.get("angle_degrees", 0.0))
    translation: Float[np.ndarray, "3"] = np.array(
        [translation_json.get("x", 0.0), translation_json.get("y", 0.0), translation_json.get("z", 0.0)], dtype=np.float64
    )
    quat_xyzw: Float[np.ndarray, "4"] = Rotation.from_rotvec(axis * angle_rad).as_quat()
    return pycolmap.Rigid3d(pycolmap.Rotation3d(quat_xyzw), translation)


def rigid_from_text_proto(message: TextProto) -> pycolmap.Rigid3d:
    """Same as `rigid_from_axis_angle` but for a text-proto `RigidTransform3d`."""
    axis_angle: TextProto = message["axis_angle"][0]
    translation_proto: TextProto = message["translation"][0]
    plain: dict[str, object] = {
        "axis_angle": {key: float(values[0]) for key, values in axis_angle.items()},
        "translation": {key: float(values[0]) for key, values in translation_proto.items()},
    }
    return rigid_from_axis_angle(plain)


def read_rig_nodes(frames_meta_path: Path) -> list[RigNode]:
    """Group camera keyframes into rig nodes, exactly as `ComputeVehicleFrames` does.

    The rig pose is not stored in the file. Per `docs/spec/pose_graph_main.md` §4 it is
    derived from the LOWEST-id camera keyframe of the synced sample:
    `world_T_rig = world_T_camera(ref) * sensor_to_vehicle_transform(ref)^-1`.

    Args:
        frames_meta_path: Path to a `keyframes/frames_meta.json`.

    Returns:
        One `RigNode` per `synced_sample_id`, sorted by rig id.
    """
    document: dict[str, object] = json.loads(frames_meta_path.read_text())
    rig_T_cam: dict[int, pycolmap.Rigid3d] = {
        int(camera_id): rigid_from_axis_angle(params["sensor_meta_data"]["sensor_to_vehicle_transform"])
        for camera_id, params in document["camera_params_id_to_camera_params"].items()
    }
    grouped: dict[int, list[dict[str, object]]] = {}
    for keyframe in document["keyframes_metadata"]:
        grouped.setdefault(int(keyframe["synced_sample_id"]), []).append(keyframe)
    nodes: list[RigNode] = []
    for rig_id in sorted(grouped):
        reference: dict[str, object] = min(grouped[rig_id], key=lambda entry: int(entry["id"]))
        world_T_cam: pycolmap.Rigid3d = rigid_from_axis_angle(reference["camera_to_world"])
        camera_id: int = int(reference.get("camera_params_id", 0))
        nodes.append(
            RigNode(
                rig_id=rig_id,
                world_T_rig=world_T_cam * rig_T_cam[camera_id].inverse(),
                timestamp_us=int(reference["timestamp_microseconds"]),
            )
        )
    return nodes


def read_vehicle_poses(vehicle_frames_meta_path: Path) -> dict[int, pycolmap.Rigid3d]:
    """Read `vehicle_frames_meta.json`, the blob's optimised rig poses, keyed by rig id."""
    document: dict[str, object] = json.loads(vehicle_frames_meta_path.read_text())
    return {int(entry["id"]): rigid_from_axis_angle(entry["camera_to_world"]) for entry in document["keyframes_metadata"]}


def read_vehicle_pose_graph(pose_graph_path: Path) -> list[PoseGraphEdge]:
    """Read a `vehicle_pose_graph.pb.txt` into `PoseGraphEdge`s.

    `pose_covariance` is consumed as an INFORMATION matrix despite the field name
    (`docs/spec/pose_graph_main.md` §5); an absent matrix means identity.

    Args:
        pose_graph_path: Path to `vehicle_pose_graph.pb.txt`.

    Returns:
        The edges in file order.
    """
    graph: TextProto = read_text_proto(pose_graph_path.read_text())
    edges: list[PoseGraphEdge] = []
    for pair in graph.get("pairs", []):
        frame_pair: TextProto = pair["frame_pair"][0]
        constraint_type: str = str(pair.get("type", ["CONSECUTIVE"])[0])
        if "pose_covariance" in pair:
            values: list[float] = [float(value) for value in pair["pose_covariance"][0]["data"]]
            information: Float[np.ndarray, "6 6"] = np.asarray(values, dtype=np.float64).reshape(6, 6)
        else:
            information = np.eye(6, dtype=np.float64)
        edges.append(
            PoseGraphEdge(
                source=int(frame_pair.get("source_frame_id", [0])[0]),
                target=int(frame_pair.get("target_frame_id", [0])[0]),
                source_T_target=rigid_from_text_proto(pair["pose_source_to_target"][0]),
                information=information,
                kind="loop" if constraint_type == "LOOP" else "consecutive",
            )
        )
    return edges


def isaac_connected_keyframe_num() -> int:
    """Read `connected_keyframe_num` out of the shipped isaac pose-graph config."""
    config: TextProto = read_text_proto(ISAAC_POSE_GRAPH_CONFIG.read_text())
    return int(config["connected_keyframe_num"][0])


# --------------------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SquareLoop:
    """A synthetic closed-square trajectory with ground truth and drifting odometry."""

    truth: list[RigNode]
    """Ground-truth rig poses, node 0 at the identity."""
    odometry: list[RigNode]
    """Dead-reckoned poses: the ground-truth relative motions plus a rotation bias."""


def make_square_loop(n_nodes: int = 200, rotation_bias_deg: float = 0.02, seed: int = 7) -> SquareLoop:
    """Build a closed square trajectory and a drifting odometry estimate of it.

    Args:
        n_nodes: Number of rig samples around the square.
        rotation_bias_deg: Systematic yaw error added to every odometry step, degrees.
        seed: Seed for the small zero-mean noise on each step.

    Returns:
        Ground truth and odometry, both starting at the same identity pose.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    side: int = n_nodes // 4
    truth_poses: list[pycolmap.Rigid3d] = [pycolmap.Rigid3d()]
    odometry_poses: list[pycolmap.Rigid3d] = [pycolmap.Rigid3d()]
    for step in range(1, n_nodes):
        turn_rad: float = np.pi / 2.0 / 1.0 if step % side == 0 else 0.0
        true_step: pycolmap.Rigid3d = pycolmap.Rigid3d(
            pycolmap.Rotation3d(Rotation.from_rotvec([0.0, 0.0, turn_rad]).as_quat()), np.array([0.25, 0.0, 0.0])
        )
        drift_rotvec: Float[np.ndarray, "3"] = np.array([0.0, 0.0, np.deg2rad(rotation_bias_deg)]) + rng.normal(scale=1e-4, size=3)
        drifted_step: pycolmap.Rigid3d = pycolmap.Rigid3d(
            pycolmap.Rotation3d((Rotation.from_quat(true_step.rotation.quat) * Rotation.from_rotvec(drift_rotvec)).as_quat()),
            true_step.translation + rng.normal(scale=1e-4, size=3),
        )
        truth_poses.append(truth_poses[-1] * true_step)
        odometry_poses.append(odometry_poses[-1] * drifted_step)
    return SquareLoop(
        truth=[RigNode(rig_id=index, world_T_rig=pose, timestamp_us=index * 100_000) for index, pose in enumerate(truth_poses)],
        odometry=[RigNode(rig_id=index, world_T_rig=pose, timestamp_us=index * 100_000) for index, pose in enumerate(odometry_poses)],
    )


def position_errors(estimate: dict[int, pycolmap.Rigid3d], truth: list[RigNode]) -> Float[np.ndarray, "n"]:
    """Per-node absolute position error in metres, no alignment (node 0 is the anchor)."""
    return np.array([float(np.linalg.norm(estimate[node.rig_id].translation - node.world_T_rig.translation)) for node in truth])


def pose_differences(
    estimate: dict[int, pycolmap.Rigid3d], reference: dict[int, pycolmap.Rigid3d]
) -> tuple[Float[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    """Per-node translation (m) and rotation (deg) differences between two pose sets."""
    rig_ids: list[int] = sorted(reference)
    translation: Float[np.ndarray, "n"] = np.array(
        [float(np.linalg.norm(estimate[rig_id].translation - reference[rig_id].translation)) for rig_id in rig_ids]
    )
    rotation_deg: Float[np.ndarray, "n"] = np.array(
        [float(np.rad2deg((estimate[rig_id].rotation.inverse() * reference[rig_id].rotation).angle())) for rig_id in rig_ids]
    )
    return translation, rotation_deg


# --------------------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------------------


def test_sequential_edges_link_each_rig_to_the_next_n_in_time_order() -> None:
    loop: SquareLoop = make_square_loop(n_nodes=8)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=2)

    assert [(edge.source, edge.target) for edge in edges] == [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5),
        (4, 5), (4, 6), (5, 6), (5, 7), (6, 7),
    ]  # fmt: skip
    assert {edge.kind for edge in edges} == {"consecutive"}


def test_sequential_edge_measurement_is_source_inverse_times_target() -> None:
    loop: SquareLoop = make_square_loop(n_nodes=6)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=1)

    for edge in edges:
        world_T_source: pycolmap.Rigid3d = loop.odometry[edge.source].world_T_rig
        world_T_target: pycolmap.Rigid3d = loop.odometry[edge.target].world_T_rig
        composed: pycolmap.Rigid3d = world_T_source * edge.source_T_target
        assert np.allclose(composed.translation, world_T_target.translation, atol=1e-12)
        assert np.rad2deg((composed.rotation.inverse() * world_T_target.rotation).angle()) < 1e-9


def test_sequential_edges_default_to_identity_information() -> None:
    loop: SquareLoop = make_square_loop(n_nodes=4)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=1)

    assert np.array_equal(edges[0].information, np.eye(6))


def test_sequential_edges_use_sigmas_when_given() -> None:
    loop: SquareLoop = make_square_loop(n_nodes=4)
    edges: list[PoseGraphEdge] = sequential_edges(
        loop.odometry, connected_keyframe_num=1, relative_translation_sigma_m=0.05, relative_rotation_sigma_deg=2.0
    )

    expected: Float[np.ndarray, "6 6"] = default_information(relative_rotation_sigma_deg=2.0, relative_translation_sigma_m=0.05)
    assert np.allclose(edges[0].information, expected)
    assert np.isclose(expected[0, 0], 1.0 / np.deg2rad(2.0) ** 2)
    assert np.isclose(expected[3, 3], 1.0 / 0.05**2)


def test_sequential_edges_reject_a_connection_count_below_one() -> None:
    loop: SquareLoop = make_square_loop(n_nodes=4)
    with pytest.raises(ValueError, match="connected_keyframe_num"):
        sequential_edges(loop.odometry, connected_keyframe_num=0)


def test_loop_edge_information_reproduces_the_archived_blob_weight() -> None:
    """The 6->3 loop edge of the §9.2 rerun has diagonal 3.6021773815155029.

    `docs/spec/pose_graph_main.md` §6.4 reads that as `score * loop_residual_weight`
    with `loop_residual_weight = 0.25`, i.e. a float32 score of 14.408709526062012.
    """
    information: Float[np.ndarray, "6 6"] = loop_edge_information(score=14.408709526062012, loop_residual_weight=0.25)

    assert np.allclose(information, np.eye(6) * 3.6021773815155029)


def test_gate_loop_edges_rejects_everything_at_the_shipped_zero_thresholds() -> None:
    """The shipped isaac/av configs leave both thresholds at the proto3 default 0.0."""
    loop: SquareLoop = make_square_loop(n_nodes=8)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=1)
    edges.append(
        PoseGraphEdge(
            source=1,
            target=0,
            source_T_target=loop.truth[1].world_T_rig.inverse() * loop.truth[0].world_T_rig,
            information=np.eye(6),
            kind="loop",
        )
    )

    kept: list[PoseGraphEdge] = gate_loop_edges(edges, max_translation_m=0.0, max_rotation_deg=0.0)

    assert [edge.kind for edge in kept] == ["consecutive"] * 7
    assert len(gate_loop_edges(edges, max_translation_m=5.0, max_rotation_deg=60.0)) == 8


def test_gate_loop_edges_applies_translation_and_rotation_separately() -> None:
    identity_edge: PoseGraphEdge = PoseGraphEdge(
        source=1, target=0, source_T_target=pycolmap.Rigid3d(), information=np.eye(6), kind="loop"
    )
    far_edge: PoseGraphEdge = PoseGraphEdge(
        source=2, target=0, source_T_target=pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([4.0, 0.0, 0.0])), information=np.eye(6), kind="loop"
    )
    turned_edge: PoseGraphEdge = PoseGraphEdge(
        source=3,
        target=0,
        source_T_target=pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_rotvec([0.0, 0.0, np.deg2rad(45.0)]).as_quat()), np.zeros(3)),
        information=np.eye(6),
        kind="loop",
    )
    edges: list[PoseGraphEdge] = [identity_edge, far_edge, turned_edge]

    assert [edge.source for edge in gate_loop_edges(edges, max_translation_m=3.0, max_rotation_deg=60.0)] == [1, 3]
    assert [edge.source for edge in gate_loop_edges(edges, max_translation_m=5.0, max_rotation_deg=30.0)] == [1, 2]


# --------------------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------------------


def test_the_analytic_jacobians_agree_with_numeric_differentiation() -> None:
    """`RelativePoseCost` reports AMBIENT Jacobians; Ceres forms `J_ambient @ PlusJacobian`.

    Both factors are checked here against a central difference taken in the manifold's own
    local coordinates, which is the only product Ceres ever uses.
    """
    rng: np.random.Generator = np.random.default_rng(11)
    measured: pycolmap.Rigid3d = pycolmap.Rigid3d(
        pycolmap.Rotation3d(Rotation.from_rotvec([0.05, -0.02, 0.31]).as_quat()), np.array([0.4, -0.1, 0.05])
    )
    information: Float[np.ndarray, "6 6"] = np.diag([9.0, 4.0, 1.0, 25.0, 16.0, 100.0])
    cost = RelativePoseCost(measured, information)

    blocks: list[Float[np.ndarray, "_"]] = [
        Rotation.from_rotvec(rng.normal(scale=0.4, size=3)).as_quat(),
        rng.normal(scale=0.5, size=3),
        Rotation.from_rotvec(rng.normal(scale=0.4, size=3)).as_quat(),
        rng.normal(scale=0.5, size=3),
    ]
    residuals: Float[np.ndarray, "6"] = np.zeros(6)
    jacobians: list[Float[np.ndarray, "_"]] = [np.zeros(24), np.zeros(18), np.zeros(24), np.zeros(18)]
    cost.Evaluate(blocks, residuals, jacobians)

    expected_residual: Float[np.ndarray, "6"] = np.linalg.cholesky(information).T @ relative_pose_error(
        pycolmap.Rigid3d(pycolmap.Rotation3d(blocks[0]), blocks[1]),
        pycolmap.Rigid3d(pycolmap.Rotation3d(blocks[2]), blocks[3]),
        measured,
    )
    assert np.allclose(residuals, expected_residual, atol=1e-12)

    step: float = 1e-6
    for block_index, ambient_size in ((0, 4), (1, 3), (2, 4), (3, 3)):
        local_size: int = 3
        numeric: Float[np.ndarray, "6 3"] = np.zeros((6, local_size))
        for axis in range(local_size):
            delta: Float[np.ndarray, "3"] = np.zeros(3)
            delta[axis] = step
            forward: list[Float[np.ndarray, "_"]] = [block.copy() for block in blocks]
            backward: list[Float[np.ndarray, "_"]] = [block.copy() for block in blocks]
            if ambient_size == 4:
                forward[block_index] = (Rotation.from_quat(blocks[block_index]) * Rotation.from_rotvec(delta)).as_quat()
                backward[block_index] = (Rotation.from_quat(blocks[block_index]) * Rotation.from_rotvec(-delta)).as_quat()
            else:
                forward[block_index] = blocks[block_index] + delta
                backward[block_index] = blocks[block_index] - delta
            plus: Float[np.ndarray, "6"] = np.zeros(6)
            minus: Float[np.ndarray, "6"] = np.zeros(6)
            cost.Evaluate(forward, plus, None)
            cost.Evaluate(backward, minus, None)
            numeric[:, axis] = (plus - minus) / (2.0 * step)

        ambient: Float[np.ndarray, "6 _"] = jacobians[block_index].reshape(6, ambient_size)
        if ambient_size == 4:
            quat: Float[np.ndarray, "4"] = blocks[block_index]
            plus_jacobian: Float[np.ndarray, "4 3"] = np.zeros((4, 3))
            plus_jacobian[:3, :] = 0.5 * (
                quat[3] * np.eye(3) + np.array([[0.0, -quat[2], quat[1]], [quat[2], 0.0, -quat[0]], [-quat[1], quat[0], 0.0]])
            )
            plus_jacobian[3, :] = -0.5 * quat[:3]
            analytic: Float[np.ndarray, "6 3"] = ambient @ plus_jacobian
        else:
            analytic = ambient
        assert np.allclose(analytic, numeric, atol=1e-6), f"block {block_index}"


def test_consistent_odometry_without_loops_leaves_the_trajectory_untouched() -> None:
    """The blob's headline behaviour: with 0 loop constraints the input is the optimum."""
    loop: SquareLoop = make_square_loop(n_nodes=200)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=1)

    result = solve_pose_graph(loop.odometry, edges)

    displacement: Float[np.ndarray, "n"] = np.array(
        [float(np.linalg.norm(result.world_T_rig[node.rig_id].translation - node.world_T_rig.translation)) for node in loop.odometry]
    )
    assert displacement.max() < 1e-9
    assert result.edge_residual_norms.max() < 1e-9


def test_one_loop_closure_cuts_the_drift_error_by_more_than_five_times() -> None:
    loop: SquareLoop = make_square_loop(n_nodes=200)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=1)
    closure: PoseGraphEdge = PoseGraphEdge(
        source=199,
        target=0,
        source_T_target=loop.truth[199].world_T_rig.inverse() * loop.truth[0].world_T_rig,
        information=loop_edge_information(score=100.0, loop_residual_weight=1.0),
        kind="loop",
    )

    before: Float[np.ndarray, "n"] = position_errors({node.rig_id: node.world_T_rig for node in loop.odometry}, loop.truth)
    result = solve_pose_graph(loop.odometry, [*edges, closure])
    after: Float[np.ndarray, "n"] = position_errors(result.world_T_rig, loop.truth)

    assert before.max() > 0.1, "the synthetic odometry must actually drift"
    assert after.max() * 5.0 < before.max()
    assert result.edge_residual_norms[-1] < 0.05
    assert result.termination == "CONVERGENCE"


def test_the_anchor_defaults_to_the_source_of_the_first_edge_and_stays_fixed() -> None:
    loop: SquareLoop = make_square_loop(n_nodes=20)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=1)
    edges.append(
        PoseGraphEdge(
            source=19,
            target=0,
            source_T_target=loop.truth[19].world_T_rig.inverse() * loop.truth[0].world_T_rig,
            information=np.eye(6) * 100.0,
            kind="loop",
        )
    )

    result = solve_pose_graph(loop.odometry, edges)
    assert result.anchor_rig_id == 0
    assert np.allclose(result.world_T_rig[0].translation, loop.odometry[0].world_T_rig.translation, atol=1e-15)

    anchored = solve_pose_graph(loop.odometry, edges, anchor_rig_id=10)
    assert anchored.anchor_rig_id == 10
    assert np.allclose(anchored.world_T_rig[10].translation, loop.odometry[10].world_T_rig.translation, atol=1e-15)


def test_both_sparse_linear_solvers_reach_the_same_optimum() -> None:
    """The blob hard-codes SPARSE_SCHUR; a pose graph has no Schur structure to exploit."""
    loop: SquareLoop = make_square_loop(n_nodes=60)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=1)
    edges.append(
        PoseGraphEdge(
            source=59,
            target=0,
            source_T_target=loop.truth[59].world_T_rig.inverse() * loop.truth[0].world_T_rig,
            information=np.eye(6) * 100.0,
            kind="loop",
        )
    )

    cholesky = solve_pose_graph(loop.odometry, edges, options=PoseGraphSolveOptions(linear_solver="SPARSE_NORMAL_CHOLESKY"))
    schur = solve_pose_graph(loop.odometry, edges, options=PoseGraphSolveOptions(linear_solver="SPARSE_SCHUR"))

    translation_m, rotation_deg = pose_differences(cholesky.world_T_rig, schur.world_T_rig)
    assert translation_m.max() < 1e-9
    assert rotation_deg.max() < 1e-7


def test_apply_rig_update_replaces_poses_and_keeps_identity_and_time() -> None:
    loop: SquareLoop = make_square_loop(n_nodes=12)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=1)
    edges.append(
        PoseGraphEdge(
            source=11,
            target=0,
            source_T_target=loop.truth[11].world_T_rig.inverse() * loop.truth[0].world_T_rig,
            information=np.eye(6) * 100.0,
            kind="loop",
        )
    )
    result = solve_pose_graph(loop.odometry, edges)

    updated: list[RigNode] = apply_rig_update(loop.odometry, result.world_T_rig)

    assert [node.rig_id for node in updated] == [node.rig_id for node in loop.odometry]
    assert [node.timestamp_us for node in updated] == [node.timestamp_us for node in loop.odometry]
    assert np.allclose(updated[7].world_T_rig.translation, result.world_T_rig[7].translation)
    with pytest.raises(ValueError, match="no optimised pose"):
        apply_rig_update(loop.odometry, {0: result.world_T_rig[0]})


def test_solve_rejects_an_empty_graph_and_unknown_endpoints() -> None:
    loop: SquareLoop = make_square_loop(n_nodes=4)
    with pytest.raises(ValueError, match="empty"):
        solve_pose_graph(loop.odometry, [])
    stray: PoseGraphEdge = PoseGraphEdge(source=0, target=99, source_T_target=pycolmap.Rigid3d(), information=np.eye(6), kind="loop")
    with pytest.raises(ValueError, match="99"):
        solve_pose_graph(loop.odometry, [stray])


def test_solve_rejects_an_information_matrix_that_is_not_positive_definite() -> None:
    loop: SquareLoop = make_square_loop(n_nodes=4)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=1)
    broken: PoseGraphEdge = PoseGraphEdge(
        source=edges[0].source,
        target=edges[0].target,
        source_T_target=edges[0].source_T_target,
        information=np.zeros((6, 6)),
        kind="loop",
    )
    with pytest.raises(ValueError, match="positive definite"):
        solve_pose_graph(loop.odometry, [*edges, broken])


# --------------------------------------------------------------------------------------
# Galileo parity against the cuSFM blob
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not GALILEO_NO_LOOP_KEYFRAMES.exists(), reason=f"missing {GALILEO_NO_LOOP_KEYFRAMES}")
def test_galileo_no_loop_solve_reproduces_the_blob_rig_poses() -> None:
    """Stock isaac config: 0 loop constraints, so the optimum is the input trajectory.

    `data/cusfm_runs/galileo/pose_graph/pose_graph_main.txt` logs "Perform pose graph
    optimization with 0 loop constraints", "total_constraint_number: 3, anchored keyframe
    id: 1" — 4 rig samples, one odometry chain, one fixed anchor.
    """
    nodes: list[RigNode] = read_rig_nodes(GALILEO_NO_LOOP_KEYFRAMES)
    assert len(nodes) == 4

    edges: list[PoseGraphEdge] = sequential_edges(nodes, connected_keyframe_num=isaac_connected_keyframe_num())
    assert len(edges) == 3

    result = solve_pose_graph(nodes, edges)
    assert result.anchor_rig_id == 1
    reference: dict[int, pycolmap.Rigid3d] = read_vehicle_poses(GALILEO_NO_LOOP_RUN / "vehicle_frames_meta.json")
    translation_m, rotation_deg = pose_differences(result.world_T_rig, reference)

    assert translation_m.max() < 1e-4, f"max {translation_m.max():.3e} m"
    assert rotation_deg.max() < 1e-3, f"max {rotation_deg.max():.3e} deg"


@pytest.mark.skipif(
    not (GALILEO_STOCK_RERUN / "vehicle_frames_meta.json").exists(),
    reason=f"stock-config rerun not present: {GALILEO_STOCK_RERUN} (docs/spec/pose_graph_main.md §9.2)",
)
def test_galileo_no_loop_solve_matches_the_29_rig_stock_rerun() -> None:
    """Same check on the larger 29-rig sequence, against the §9.2 stock-config rerun."""
    nodes: list[RigNode] = read_rig_nodes(GALILEO_LOOP_KEYFRAMES)
    edges: list[PoseGraphEdge] = sequential_edges(nodes, connected_keyframe_num=isaac_connected_keyframe_num())
    assert (len(nodes), len(edges)) == (29, 28)

    result = solve_pose_graph(nodes, edges)
    reference: dict[int, pycolmap.Rigid3d] = read_vehicle_poses(GALILEO_STOCK_RERUN / "vehicle_frames_meta.json")
    translation_m, rotation_deg = pose_differences(result.world_T_rig, reference)

    assert translation_m.max() < 1e-4, f"max {translation_m.max():.3e} m"
    assert rotation_deg.max() < 1e-3, f"max {rotation_deg.max():.3e} deg"


@pytest.mark.skipif(
    not (GALILEO_LOOP_RUN / "vehicle_pose_graph.pb.txt").exists(),
    reason=f"archived loop-gated run missing: {GALILEO_LOOP_RUN} (see docs/spec/pose_graph_main.md §9.2)",
)
def test_galileo_loop_solve_reproduces_the_blob_rig_poses() -> None:
    """The archived 29-rig run with the loop gates opened: 8 loop edges, 12.3 mm of motion.

    The blob's solver overwrites every stored 6x6 with the default information matrix
    unless `relative_constraint_use_pose_covariance` is set, and no shipped config sets
    it — so the score-based loop weights that `pose_graph_main` WRITES into
    `vehicle_pose_graph.pb.txt` are not the weights it SOLVES with. Both variants are
    exercised here; only the identity-weighted one is blob parity, which is itself the
    evidence for that reading of spec §7.
    """
    nodes: list[RigNode] = read_rig_nodes(GALILEO_LOOP_KEYFRAMES)
    archived: list[PoseGraphEdge] = read_vehicle_pose_graph(GALILEO_LOOP_RUN / "vehicle_pose_graph.pb.txt")
    loop_edges: list[PoseGraphEdge] = [edge for edge in archived if edge.kind == "loop"]
    assert len(archived) == 36
    assert [(edge.source, edge.target) for edge in loop_edges] == [(6, 3), (7, 2), (8, 3), (17, 14), (24, 21), (26, 23), (27, 24), (28, 25)]

    reference: dict[int, pycolmap.Rigid3d] = read_vehicle_poses(GALILEO_LOOP_RUN / "vehicle_frames_meta.json")
    no_loop: dict[int, pycolmap.Rigid3d] = {node.rig_id: node.world_T_rig for node in nodes}
    loop_effect_m: float = pose_differences(no_loop, reference)[0].max()
    assert loop_effect_m > 1e-2, "the archived loops must move the trajectory by ~12.3 mm"

    identity_weighted: list[PoseGraphEdge] = sequential_edges(nodes, connected_keyframe_num=isaac_connected_keyframe_num())
    identity_weighted += [PoseGraphEdge(edge.source, edge.target, edge.source_T_target, np.eye(6), "loop") for edge in loop_edges]
    result = solve_pose_graph(nodes, identity_weighted)
    translation_m, rotation_deg = pose_differences(result.world_T_rig, reference)
    assert translation_m.max() < 1e-3, f"max {translation_m.max():.3e} m"
    assert rotation_deg.max() < 0.05, f"max {rotation_deg.max():.3e} deg"

    score_weighted: list[PoseGraphEdge] = sequential_edges(nodes, connected_keyframe_num=isaac_connected_keyframe_num()) + loop_edges
    score_result = solve_pose_graph(nodes, score_weighted)
    score_translation_m, _ = pose_differences(score_result.world_T_rig, reference)
    assert score_translation_m.max() < 3e-3
    assert score_translation_m.max() > translation_m.max(), "the score weights are not what the blob solved with"


@pytest.mark.skipif(not (GALILEO_LOOP_RUN / "vehicle_pose_graph.pb.txt").exists(), reason="archived loop run missing")
def test_archived_loop_edge_weights_follow_the_documented_score_convention() -> None:
    archived: list[PoseGraphEdge] = read_vehicle_pose_graph(GALILEO_LOOP_RUN / "vehicle_pose_graph.pb.txt")
    first_loop: PoseGraphEdge = next(edge for edge in archived if edge.kind == "loop")

    score: float = float(first_loop.information[0, 0]) / 0.25
    assert np.allclose(first_loop.information, loop_edge_information(score=score, loop_residual_weight=0.25))
    assert np.float32(score) == np.float32(14.408709526062012)


# --------------------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("PIXI_DEV_MODE") == "1",
    reason="PIXI_DEV_MODE=1 puts beartype's claw on every call in the Ceres inner loop; timing is meaningless there",
)
def test_a_thousand_node_graph_solves_within_one_second() -> None:
    """Budget from the task: the blob solves 1132 nodes in 6.5 ms; we allow 1.0 s."""
    loop: SquareLoop = make_square_loop(n_nodes=1000, rotation_bias_deg=0.004)
    edges: list[PoseGraphEdge] = sequential_edges(loop.odometry, connected_keyframe_num=1)
    assert len(edges) == 999
    for index in range(50):
        source: int = 999 - index
        target: int = index
        edges.append(
            PoseGraphEdge(
                source=source,
                target=target,
                source_T_target=loop.truth[source].world_T_rig.inverse() * loop.truth[target].world_T_rig,
                information=np.eye(6) * 10.0,
                kind="loop",
            )
        )

    before: Float[np.ndarray, "n"] = position_errors({node.rig_id: node.world_T_rig for node in loop.odometry}, loop.truth)
    started: float = time.perf_counter()
    result = solve_pose_graph(loop.odometry, edges, options=PoseGraphSolveOptions())
    elapsed_s: float = time.perf_counter() - started
    after: Float[np.ndarray, "n"] = position_errors(result.world_T_rig, loop.truth)

    print(f"solve_pose_graph: {len(loop.odometry)} nodes, {len(edges)} edges, {result.iterations} iterations, {elapsed_s * 1e3:.1f} ms")
    assert result.termination == "CONVERGENCE"
    assert after.max() * 5.0 < before.max(), "the solve must actually close the loop, not just terminate"
    assert elapsed_s < 1.0, f"{elapsed_s:.3f} s"

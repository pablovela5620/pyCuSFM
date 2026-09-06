"""Poses as 4x4 matrices, cuSFM's axis-angle JSON, and the two fits the demo needs.

Everything here is float64 numpy and scipy: no Rerun, no cuSFM, no pycolmap. It
is the layer that translates between three representations of the same rigid
motion — a homogeneous matrix, the axis-plus-degrees form `frames_meta.json`
stores, and a scipy `Rotation` stream sampled at arbitrary times.

`umeyama_rigid` stays local rather than calling `colsfm.alignment.align_rigid`:
that module reaches for pycolmap, which this environment deliberately does not
carry, and simplecv's canonical `umeyama_transform` returns float32, which loses
the millimetre-scale diagnostics the demo prints.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Float64, Int
from numpy import ndarray
from scipy.spatial.transform import Rotation, Slerp


def compose(rotation: Float[ndarray, "3 3"], translation: Float[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """Build a 4x4 homogeneous transform from a rotation and a translation."""
    transform: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def to_axis_angle_dict(transform: Float[ndarray, "4 4"]) -> dict[str, dict[str, float]]:
    """Serialise a transform into cuSFM's ``axis_angle`` + ``translation`` JSON form.

    cuSFM stores the rotation as a unit axis plus an angle in *degrees*. A
    zero-rotation transform has no well-defined axis, so emit ``+Z`` with a zero
    angle rather than a NaN axis.
    """
    rotation_vector: Float64[ndarray, "3"] = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    angle_rad: float = float(np.linalg.norm(rotation_vector))
    axis: Float64[ndarray, "3"] = (
        rotation_vector / angle_rad if angle_rad > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    return {
        "axis_angle": {
            "x": float(axis[0]),
            "y": float(axis[1]),
            "z": float(axis[2]),
            "angle_degrees": float(np.degrees(angle_rad)),
        },
        "translation": {
            "x": float(transform[0, 3]),
            "y": float(transform[1, 3]),
            "z": float(transform[2, 3]),
        },
    }


def umeyama_rigid(
    source: Float[ndarray, "n 3"], target: Float[ndarray, "n 3"]
) -> tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3"], float, float]:
    """Least-squares rigid (SE3, **no scale**) fit mapping ``source`` onto ``target``.

    Kept local to preserve a float64 transform for millimetre-scale diagnostics;
    simplecv's canonical ``umeyama_transform`` returns float32.

    Scale is deliberately held at 1: if cuSFM rescales the trajectory we want to
    see that as alignment error, not absorb it into the fit. The scale that
    *would* have been fitted is returned as a diagnostic.

    Returns ``(rotation, translation, rmse, would_be_scale)``.
    """
    source_mean: Float64[ndarray, "3"] = source.mean(axis=0)
    target_mean: Float64[ndarray, "3"] = target.mean(axis=0)
    source_centred: Float64[ndarray, "n 3"] = source - source_mean
    target_centred: Float64[ndarray, "n 3"] = target - target_mean

    covariance: Float64[ndarray, "3 3"] = target_centred.T @ source_centred / len(source)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    sign_fix: Float64[ndarray, "3 3"] = np.eye(3)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0:
        sign_fix[2, 2] = -1.0
    rotation: Float64[ndarray, "3 3"] = u_matrix @ sign_fix @ vt_matrix

    source_variance: float = float((source_centred**2).sum() / len(source))
    would_be_scale: float = (
        float((singular_values * np.diag(sign_fix)).sum() / source_variance) if source_variance > 1e-15 else 1.0
    )

    translation: Float64[ndarray, "3"] = target_mean - rotation @ source_mean
    residual: Float64[ndarray, "n 3"] = (source @ rotation.T + translation) - target
    rmse: float = float(np.sqrt((residual**2).sum(axis=1).mean()))
    return rotation, translation, rmse, would_be_scale


def interpolate_poses(
    query_ns: Int[ndarray, "m"],
    source_ns: Int[ndarray, "n"],
    source_rotations: Rotation,
    source_translations: Float[ndarray, "n 3"],
) -> tuple[Float64[ndarray, "m 3 3"], Float64[ndarray, "m 3"]]:
    """Interpolate a pose stream onto ``query_ns`` (slerp + linear, clamped)."""
    clamped: Float64[ndarray, "m"] = np.clip(query_ns, source_ns[0], source_ns[-1]).astype(np.float64)
    slerp: Slerp = Slerp(source_ns.astype(np.float64), source_rotations)
    rotations: Float64[ndarray, "m 3 3"] = slerp(clamped).as_matrix()
    translations: Float64[ndarray, "m 3"] = np.stack(
        [np.interp(clamped, source_ns.astype(np.float64), source_translations[:, axis]) for axis in range(3)],
        axis=-1,
    )
    return rotations, translations



def from_axis_angle_dict(node: dict) -> Float64[ndarray, "4 4"]:
    """Inverse of :func:`to_axis_angle_dict`."""
    axis_angle: dict = node["axis_angle"]
    translation: dict = node["translation"]
    axis: Float64[ndarray, "3"] = np.array(
        [axis_angle["x"], axis_angle["y"], axis_angle["z"]], dtype=np.float64
    )
    norm: float = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    rotation: Float64[ndarray, "3 3"] = Rotation.from_rotvec(
        axis * np.radians(axis_angle["angle_degrees"])
    ).as_matrix()
    return compose(rotation, np.array([translation["x"], translation["y"], translation["z"]], dtype=np.float64))


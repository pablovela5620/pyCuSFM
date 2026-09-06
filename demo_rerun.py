"""Run cuSFM on a posed multi-camera sequence and visualise it in Rerun.

Two datasets, one code path:

- ``galileo``  — the bundled ``data/r2b_galileo`` sample: 8 PINHOLE cameras on a
  ground robot, 226 keyframes, and a ``ground_truth.txt`` trajectory.
- ``robocap``  — a frozen RoboCap catalog segment: 6 Kannala-Brandt fisheye
  cameras on a head-worn rig, with a per-frame basalt VIO trajectory already
  in the recording. Only the 4 world-facing cameras are reconstructed; the two
  inward-facing eye-tracking cameras are shown as greyed frusta.

Both datasets already carry an initial trajectory, so cuVSLAM is skipped
(``--skip_cuvslam``) and cuSFM's global bundle adjustment *refines* the given
poses. The point of the demo is to make that refinement visible.

Output follows the ``exoego:v2`` rig schema (see simplecv's
``packages/simplecv/docs/exoego_schema.md``), with **two rigs** under one world::

    /                                 ViewCoordinates.RFU (gravity-aligned Z-up)
      /world/rig_00                   AnyValues + world_T_rig(t)  <- input trajectory
        /cam_NN                       Transform3D (rig_T_cam, from_parent)
          /pinhole                    PinholeWithDistortion
            /image                    EncodedImage (JPEG)
      /world/rig_01                   AnyValues + world_T_rig(t)  <- cuSFM refined
        /cam_NN/pinhole               same calibration, refined motion
          /video                      VideoFrameReference -> rig_00's VideoStream (no copy)
      /world/points                   Points3D (cuSFM sparse cloud)

``rig_01`` is rigidly aligned to ``rig_00`` with a scale-free Umeyama fit, and the
residual RMSE is printed. Read that number as a **disagreement between two
estimates**, not as cuSFM's error: RoboCap's supplied trajectory is pure
multi-camera VIO (basalt) with no loop closure and no mapping, so it drifts too.
Only galileo ships an actual ground-truth trajectory, and only there is the word
"ATE" used. Run with ``--run.no-skip-cuvslam`` to get an independent third
trajectory from cuSFM's bundled cuVSLAM.

Note on fisheye frusta: Rerun draws a frustum from ``K`` alone, so a 150-degree+
fisheye renders as a much narrower cone than its true field of view. The
distortion coefficients are logged alongside (``PinholeWithDistortion``) and the
geometry is correct; only the drawn cone under-represents the FOV. This matches
exactly what the RoboCap catalog's own ``base`` layer displays.

The code lives in the ``demo`` package beside this file, split by owner: dataset
adapters (``demo.robocap``, ``demo.galileo``), the shared container and cuSFM's
input contract (``demo.sequence``), the cuSFM invocation (``demo.cusfm``),
reading its output back (``demo.model_io``), the Rerun logging
(``demo.visualise``), and the run itself (``demo.app``). This file is the entry
point and nothing else.
"""

from __future__ import annotations

import tyro

from colsfm.colmap_text_model import ColmapImage, ColmapModel, parse_colmap_images_text, read_colmap_model
from demo.app import Config, main

# `read_colmap_model` and its parser moved to `colsfm.colmap_text_model`, which
# needs numpy and scipy and nothing else. They are re-exported here because the
# probe suite still reaches for them through this module.
__all__ = ["ColmapImage", "ColmapModel", "Config", "main", "parse_colmap_images_text", "read_colmap_model"]


if __name__ == "__main__":
    main(tyro.cli(Config))

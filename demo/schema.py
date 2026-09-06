"""The exoego:v2 recording conventions this demo reads and writes.

Both the RoboCap catalog's own recordings and everything logged here follow one
layout: a duration timeline shared across layers, two rigs under one world, and
static camera extrinsics stored child-from-parent. These are the names and
numbers that layout is made of, kept in one place because the reader
(`demo.robocap`) and the writer (`demo.visualise`) must agree on them exactly.
"""

from __future__ import annotations

TIMELINE: str = "video_time"
"""Duration timeline, shared with the RoboCap catalog's ``base``/``slam`` layers."""

INPUT_RIG_INDEX: int = 0
"""``rig_00`` carries the *input* trajectory (basalt VIO / galileo ego-motion)."""

CUSFM_RIG_INDEX: int = 1
"""``rig_01`` carries the cuSFM-refined trajectory."""

CHILD_FROM_PARENT: int = 2
"""``rr.components.TransformRelation.ChildFromParent``. exoego stores each
camera's static extrinsic this way, so the stored value is ``cam_T_rig``."""

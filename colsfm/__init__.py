"""colsfm — an open Python + pycolmap replacement for NVIDIA cuSFM.

The package mirrors cuSFM's stages (keyframe selection, feature extraction and
matching, pose-graph optimisation, triangulation and bundle adjustment, export),
takes the same `frames_meta.json` in and writes the same outputs: a COLMAP
`sparse/` model, `kpmap/keyframes/frames_meta.json` with optimised poses, and TUM
pose files. See `docs/open-pipeline-plan.md` for the plan of record and
`docs/spec/` for the stage-by-stage reverse-engineering specs this code follows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
"""Repo root, so vendored data and default config directories resolve from any cwd."""

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()

"""The Rerun demo: one cuSFM run on a posed multi-camera sequence, made visible.

`demo_rerun.py` at the repository root is the entry point; everything it does
lives here, split by who owns each decision:

- `demo.schema` — the exoego:v2 recording conventions both rigs are written to.
- `demo.poses` — 4x4 poses, cuSFM's axis-angle JSON encoding, and the two fits.
- `demo.sequence` — the dataset-independent container, and cuSFM's input contract.
- `demo.robocap`, `demo.galileo` — one dataset adapter each, config included.
- `demo.cusfm` — the cuSFM invocation and its workspace layout.
- `demo.model_io` — reading back what cuSFM wrote, and what it implies.
- `demo.visualise` — everything logged to Rerun, and the blueprint.
- `demo.app` — the top-level CLI and the run itself.

Nothing here is part of `colsfm`; the demo imports it read-only, and shares its
pycolmap-free COLMAP text reader (`colsfm.colmap_text_model`) so this
environment never needs pycolmap's ceres/CUDA stack.
"""

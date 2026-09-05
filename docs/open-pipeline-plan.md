# Open cuSFM pipeline on pycolmap — plan of record

Agreed 2026-09-04. This file is the shared understanding that every worker reads first.

## Goal

A Python + pycolmap pipeline (`colsfm/`) that mirrors cuSFM's stages, takes the same
`frames_meta.json` in, and writes the same outputs (`sparse/` COLMAP model,
`kpmap/keyframes/frames_meta.json` with optimised poses, TUM pose files). Similar runtime and
accuracy to the NVIDIA binaries ("fast happy path"); bit-exactness is not a goal.

- Triangulation, bundle adjustment, geometric verification, database, model I/O: **pycolmap**.
- Pose graph optimisation: **reimplemented in Python** (pycolmap has none). This stage is the
  priority.
- Features: ALIKED + LightGlue via the TensorRT engines already in `pycusfm/models` and the
  TensorRT code in `tools/` (same models as the blob, so accuracy is comparable).
- cuVSLAM: out of scope. Poses come from `frames_meta.json` as today.

## Facts that shape the design

- The 21 executables in `pycusfm/x86_cuda13/bin` are **unstripped** with embedded source paths.
  None contains CUDA kernels; GPU work is TensorRT (open ONNX models) and CV-CUDA preprocessing.
  `keypoints_mapper_main` (triangulation + BA) and `bundle_adjustment_runner` are **Ceres on CPU**.
- Python reaches the blob only by subprocess through `pycusfm/command_runner.py`; the runner
  takes one `--binary_dir`. An overlay bin dir (symlinks to unreplaced blob binaries + replaced
  entries) makes stage-by-stage swapping possible with no Python change.
- Configs are protobuf text (`pycusfm/configs/isaac/*.pb.txt`). The 19 embedded `.proto`
  files are vendored as `data/cusfm_schema/cusfm_protos.fdset`.
- gflags help for every binary is dumped at `data/cusfm_re/help/<binary>.txt` (gitignored).
- `NOTES.md` localises all trajectory change to `keypoints_mapper_main`; the pose graph is a
  no-op when data association is skipped (the default). With loop closure on, PGO matters.
- Licence: repo is Apache 2.0, no separate EULA for the binaries. Direct reading of the
  disassembly is fine; no clean-room split.

## Stages and their spec owners

| # | cuSFM binary | Spec file | Python replacement |
|---|---|---|---|
| 1 | feature_extractor_main | docs/spec/feature_extractor_main.md | ALIKED TensorRT + keyframe selection |
| 2 | generate_bow_vocabulary_main / generate_bow_index_main | docs/spec/bow.md | retrieval for loop closure |
| 3 | generate_association_main | docs/spec/generate_association_main.md | loop-closure candidates + verification |
| 4 | pose_graph_main / pose_graph_association_main | docs/spec/pose_graph_main.md | **Python PGO** |
| 5 | feature_matcher_task_builder_main | docs/spec/feature_matcher_task_builder_main.md | pair selection |
| 6 | feature_matcher_main | docs/spec/feature_matcher_main.md | LightGlue TensorRT + pycolmap verification |
| 7 | keypoints_mapper_main | docs/spec/keypoints_mapper_main.md | pycolmap triangulation + BA |
| 8 | kpmap_to_colmap, extract_pose_from_map_main, update_keyframe_pose_main | docs/spec/export.md | pycolmap write + TUM |

## Acceptance (Galileo, `data/r2b_galileo`, ground truth shipped)

Blob reference: 224/226 registered, 4938 points, ATE 3.9 mm vs ground truth, 1.54 px mean
reprojection. Bound: registered >= 220, ATE <= 5 mm, reprojection <= 1.7 px, per-stage runtime
within 2x of the blob. RoboCap (`data/robocap/s00000021.rrd`, from the catalog): stride 20 for
iteration, stride 4 for the final run, disagreement vs basalt reported, no ground truth.

Benchmark output: per-stage runtime table, metrics table, one Rerun recording with blob and
Python reconstructions as two rigs (pattern from `demo_rerun.py`), validated with the
rerun-viewer-validation skill (pixel evidence).

## Engineering rules

- pixi only. New work lives in the `colsfm` environment/feature; the default environment's
  dependencies are not touched (fragile CUDA 13 + TensorRT solve).
- Python follows the `python-conventions` skill (jaxtyping + beartype, tyro CLIs, pyserde,
  dataclass field docstrings, TypeAlias).
- Branch: `feat/pycolmap-open-pipeline`. Specs in `docs/spec/`, scratch in `data/cusfm_re/`.
- Workers: Opus 5 for reverse engineering, specs, implementation, and review (user decision).

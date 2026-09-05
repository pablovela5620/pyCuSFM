# colsfm module map

`colsfm` is an open Python and pycolmap replacement for the NVIDIA cuSFM binaries. It takes
the same `frames_meta.json` in and writes the same outputs. `docs/open-pipeline-plan.md` is
the plan of record, `docs/spec/` holds the reverse-engineered contract for each blob stage,
and the "colsfm" section of `NOTES.md` holds the decisions, the deviations and the measured
results.

## Modules

| Module | What it does |
|---|---|
| `__init__.py` | Package docstring and `REPO_ROOT`. Turns on the beartype claw when `PIXI_DEV_MODE=1`. |
| `__main__.py` | The tyro CLI. Two subcommands: `run` (all eight stages) and `stage` (one metadata-only stage, no images and no GPU). |
| `pipeline.py` | The eight stages wired into one run. The replacement for `pycusfm.cusfm_runner.CusfmRunner.run_all`, down to the `runtime.csv` layout. |
| `schema.py` | Runtime protobuf message classes built from the vendored `data/cusfm_schema/cusfm_protos.fdset`, so cuSFM protos are read by protobuf itself and not by guesswork. |
| `config.py` | cuSFM's protobuf-text configs as typed frozen dataclasses. |
| `frames_meta.py` | Typed access to `frames_meta.json` (`KeyframesMetadataCollection`), including the grouping of keyframes into rig frames by `synced_sample_id`. |
| `geometry.py` | SE(3) helpers on top of `pycolmap.Rigid3d`, in cuSFM's conventions. |
| `cameras.py` | cuSFM camera parameters to `pycolmap.Camera`. |
| `keyframe_selection.py` | cuSFM's keyframe selection, as `feature_extractor_main` performs it. |
| `database.py` | The COLMAP database that carries cuSFM's cameras, rig, frames and images into pycolmap, keeping cuSFM's own identifiers. |
| `features.py` | ALIKED feature extraction over `pycolmap.extract_features`. The replacement for `feature_extractor_main`'s per-image half. |
| `pairs.py` | Which image pairs get matched. The replacement for `feature_matcher_task_builder_main`. |
| `matching.py` | LightGlue matching plus COLMAP's geometric verification, then the spatial subsample. The replacement for `feature_matcher_main`. |
| `retrieval.py` | Image retrieval for loop closure, brute force or vocabulary tree. The replacement for `generate_bow_vocabulary_main` and `generate_bow_index_main`. |
| `loop_closure.py` | Loop-closure candidate selection, gating and rig-edge assembly. The replacement for `generate_association_main`'s `RetrievalLoopAssociations`. |
| `loop_pose.py` | The metric rig-to-rig relative pose a loop edge carries: a local triangulated map in the source rig frame, then generalized resection of the target rig. The replacement for `StereoPoseEstimator`. |
| `pose_graph.py` | Rig-level pose graph optimisation on pyceres with a Python residual. The replacement for `pose_graph_main`. |
| `ceres_pose.py` | The pyceres plumbing `pose_graph` and `extrinsic_refinement` share: the linear-solver vocabulary, quaternion priming, the manifold, `SolverOptions` and summary unpacking. |
| `reconstruction.py` | A `pycolmap.Reconstruction` built from `frames_meta.json`: the rig, the frames and the images. The rig's reference sensor is the vehicle body by default, or a named camera when `reference_camera_params_id` is given — which is what extrinsic refinement needs, because COLMAP freezes every `sensor_from_rig` of a rig whose reference sensor owns no images. `RigReference` inverts that change of basis. |
| `mapping.py` | Triangulation and global bundle adjustment. The replacement for `keypoints_mapper_main`. Takes the `PosedModel` `reconstruction` builds, so the reconstruction and its `RigReference` can never be mispaired. With `optimize_extrinsics` it also refines the rig extrinsics through pycolmap's rig BA, holding the gauge camera fixed, and reports them as `MappingResult.refined_extrinsics` — the unregularised path, kept for comparison behind `--no-regularised-extrinsics`. |
| `extrinsic_refinement.py` | Regularised rig-extrinsic refinement: cuSFM's absolute (Eq. 14) and inter-camera relative (Eq. 6) extrinsic priors in pyceres, alternated with pycolmap's own bundle adjustment. What `--optimize-extrinsics` runs by default, because pycolmap's rig BA carries no prior term. |
| `export.py` | Writers for the four artifacts a cuSFM run leaves behind: the `sparse/` model, `kpmap/keyframes/frames_meta.json`, the TUM pose files and `runtime.csv`. The replacement for `kpmap_to_colmap`, `extract_pose_from_map_main` and `update_keyframe_pose_main`. |
| `benchmark.py` | Metrics that compare two runs in the cuSFM output layout, plus the acceptance bounds. It reads both runs the same way and knows nothing about which producer wrote which. Re-exports the names `alignment` and `runtime` own, so one import still covers a whole comparison. |
| `alignment.py` | Umeyama alignment (rigid, or scale-estimating), rig trajectories, the ground-truth reader and the timestamp join every trajectory metric is built on. |
| `runtime.py` | `runtime.csv` classification: the closed `colsfm.pipeline` stage vocabulary as an exhaustive table, the blob's binary names as ordered substring rules, and the "keep only the last appended run" rule. |
| `bench_report.py` | The markdown rendering of a comparison. |
| `bench_cli.py` | The tyro CLI over `benchmark`. Writes the markdown report, the JSON report and the Rerun recording. |
| `rerun_log.py` | Logs a two-run comparison into one Rerun recording: two exoego rigs, two clouds, the trajectory polylines and the report. |

## Stage order

`pipeline.run_pipeline` runs these eight stages in this order, and writes one `runtime.csv`
row per stage under these names:

| # | Stage | Module | Blob stage it replaces |
|---|---|---|---|
| 1 | `keyframe_selection` | `keyframe_selection`, `frames_meta` | `feature_extractor_main` (selection half) |
| 2 | `feature_extraction` | `database`, `cameras`, `features` | `feature_extractor_main` (per-image half) |
| 3 | `pair_selection` | `pairs` | `feature_matcher_task_builder_main` |
| 4 | `matching` | `matching` | `feature_matcher_main` |
| 5 | `loop_closure` | `retrieval`, `loop_closure`, `loop_pose` | `generate_bow_*_main` + `generate_association_main` |
| 6 | `pose_graph` | `pose_graph` | `pose_graph_main` |
| 7 | `reconstruction` | `reconstruction`, `mapping` | `keypoints_mapper_main` |
| 7b | `extrinsic_refinement` | `reconstruction`, `mapping`, `extrinsic_refinement` | `keypoints_mapper_main --optimize_extrinsics=True` (only with `--optimize-extrinsics`) |
| 8 | `export` | `export` | `kpmap_to_colmap`, `extract_pose_from_map_main`, `update_keyframe_pose_main` |

Each stage is a `run_<stage>_stage` function returning a small result dataclass, and
`run_pipeline` only composes them and times each one, so a caller that wants to step through
a dataset stage by stage runs the same code the full pipeline runs.

Three ordering notes matter when reading the code. Pair selection cannot see loop pairs:
stage 3 emits only the consecutive and stereo pairs it can derive from the metadata, and loop
pairs first exist after stage 5, which matches them into the same database. Stage 5 therefore
runs the search twice, once with a recording `match_fn` that collects the candidate pairs and
returns no matches, then one batch match over those pairs, then the real search reading the
matches back. Loop closure is off by default. And `--optimize-extrinsics` maps **once**:
stage 7 builds the camera-referenced reconstruction the refinement needs, and stage 7b
refines the extrinsics of the model stage 7 left behind rather than mapping the scene again.

## Tests

| Directory | What it holds |
|---|---|
| `tests/colsfm` | One test file per module, plus `conftest.py` with the Galileo fixtures, the shared geometry helpers (`pose_delta`, `project_and_mask`, `read_blob_keyframes`) and the synthetic cuSFM-layout run builder. These are the pipeline's own tests. Run with `pixi run colsfm-test`. |
| `tests/colsfm_probes` | Runnable probes that establish what pycolmap 4.2.0 offers. Each one builds its own synthetic scene (`synthetic.py`) and asserts the behaviour that `docs/spec/pycolmap-capabilities.md` documents: database, two-view geometry, triangulation, bundle adjustment, pose graph, camera models, model I/O, GPU. beartype is on unconditionally here, because the point of the suite is to check what the bindings accept. Run with `pixi run colsfm-probe`. |

The probes are the reason the specs can say "measured". Change one, and the corresponding
claim in `docs/spec/pycolmap-capabilities.md` needs re-checking.

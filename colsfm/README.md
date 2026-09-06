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
| `pipeline.py` | The one runner, and `PipelineObserver` — the seam a viewer or a notebook watches a run through, instead of sequencing the stages itself. The replacement for `pycusfm.cusfm_runner.CusfmRunner.run_all`, down to the `runtime.csv` layout. |
| `run_config.py` | `PipelineOptions` as the command line spells it, `SelectionOptions` for the two metadata-only stages, and the `ResolvedRun` every stage actually takes. Resolution happens once, before the workspace exists, and validates as it goes. |
| `run_lifecycle.py` | The stage vocabulary, the `StageLedger` that `runtime.csv` and `summary.json` both come from, and the staging workspace a run publishes across with `RunWorkspace.publish`, so a failed rerun cannot leave a directory holding two runs. |
| `run_report.py` | `summary.json` and `loop_edges.json`: what a finished run says about itself, as data a benchmark or a viewer can read. |
| `stages.py` | One `run_*_stage` function per stage and the typed result each returns. They carry no timing and write no `runtime.csv` row, so a caller stepping through a dataset one stage at a time reuses exactly the code a full run executes. |
| `schema.py` | Runtime protobuf message classes built from the vendored `data/cusfm_schema/cusfm_protos.fdset`, so cuSFM protos are read by protobuf itself and not by guesswork. |
| `config.py` | cuSFM's protobuf-text configs as typed frozen dataclasses. |
| `frames_meta.py` | Typed access to `frames_meta.json` (`KeyframesMetadataCollection`), including the grouping of keyframes into rig frames by `synced_sample_id`. |
| `geometry.py` | SE(3) helpers on top of `pycolmap.Rigid3d`, in cuSFM's conventions. |
| `cameras.py` | cuSFM camera parameters to `pycolmap.Camera`. |
| `keyframe_selection.py` | cuSFM's keyframe selection, as `feature_extractor_main` performs it. |
| `database.py` | The COLMAP database that carries cuSFM's cameras, rig, frames and images into pycolmap, keeping cuSFM's own identifiers. |
| `features.py` | ALIKED feature extraction over `pycolmap.extract_features`, and the backend switch. The replacement for `feature_extractor_main`'s per-image half. |
| `tensorrt_runtime.py` | The TensorRT plumbing both `_trt` backends share: the blob's engine-name cache, the FP16 builder, and one generic execution session with growing device buffers. |
| `features_trt.py` | The same stage through the blob's own `aliked.onnx` and its FP16 engine, with the blob's preprocessing and pixel mapping. `--features-backend tensorrt`. |
| `features_raco.py` | The same stage through fabio-sim's RaCo-ALIKED on a batch-dynamic FP16 engine, 8 images per execution; reuses `features_trt`'s preprocessing and pixel mapping unchanged. `--features-backend raco`. |
| `pairs.py` | Which image pairs get matched. The replacement for `feature_matcher_task_builder_main`. |
| `matching.py` | LightGlue matching plus COLMAP's geometric verification, then the spatial subsample. The replacement for `feature_matcher_main`. Also owns the blob's real SSC (`select_by_square_covering`) and `MatchLimitPolicy`, the one answer to how many matches a pair may keep. |
| `matching_trt.py` | The same stage through the blob's own `lightglue_aliked.onnx` and its FP16 engine. The only path with the per-match score, so it runs the real SSC before verification instead of the score-free grid subsample. `--matching-backend tensorrt`. |
| `matching_raco.py` | The same stage through LightGlue+, the matcher fabio-sim trained against RaCo-ALIKED. `match_pairs_tensorrt` does the work; this module supplies the graph and the normalised keypoint frame it wants. `--matching-backend raco`. |
| `retrieval.py` | Image retrieval for loop closure, brute force or vocabulary tree. The replacement for `generate_bow_vocabulary_main` and `generate_bow_index_main`. |
| `loop_shortlist.py` | The retrieval half of the loop funnel — the gates, the `|dt|` banding, the rig-pair deduplication and the counters they fill. It reads no match at all, which is why the loop stage no longer has to run itself twice to learn its pair list. |
| `loop_closure.py` | Loop-closure candidate selection, gating and rig-edge assembly. The replacement for `generate_association_main`'s `RetrievalLoopAssociations`. `plan_loop_search` decides which rig pairs to measure and which image pairs that needs *without matching anything*; `verify_loop_plan` is the only pass that reads a match. |
| `loop_pose.py` | The metric rig-to-rig relative pose a loop edge carries: a local triangulated map in the source rig frame, then generalized resection of the target rig. The replacement for `StereoPoseEstimator`. |
| `pose_graph.py` | Rig-level pose graph optimisation on pyceres with a Python residual. The replacement for `pose_graph_main`. |
| `ceres_pose.py` | The pyceres plumbing `pose_graph` and `extrinsic_refinement` share: the linear-solver vocabulary, quaternion priming, the manifold, `SolverOptions` and summary unpacking. |
| `reconstruction.py` | A `pycolmap.Reconstruction` built from `frames_meta.json`: the rig, the frames and the images. The rig's reference sensor is the vehicle body by default, or a named camera when `reference_camera_params_id` is given — which is what extrinsic refinement needs, because COLMAP freezes every `sensor_from_rig` of a rig whose reference sensor owns no images. `RigReference` inverts that change of basis. |
| `mapping.py` | Triangulation and global bundle adjustment. The replacement for `keypoints_mapper_main`. Takes the `PosedModel` `reconstruction` builds, so the reconstruction and its `RigReference` can never be mispaired. With `optimize_extrinsics` it also refines the rig extrinsics through pycolmap's rig BA, holding the gauge camera fixed, and reports them as `MappingResult.refined_extrinsics` — the unregularised path, kept for comparison behind `--no-regularised-extrinsics`. `--ba-backend caspar` swaps Ceres for COLMAP's GPU bundle adjustment, which needs the `colsfm-caspar` environment and falls back to Ceres, loudly, when the build, the camera model or `--optimize-extrinsics` rules it out — which camera models are in is measured from the build at run time, so the OPENCV_FISHEYE adapter of `colsfm-caspar-fisheye` is used where it exists (`docs/caspar-fisheye-adapter.md`); a CASPAR run then ends on one Ceres global bundle adjustment over the final model (`--no-caspar-ceres-polish` to ablate), which is what buys Ceres' accuracy at a third of its mapping time. |
| `ba_backend.py` | Which bundle-adjustment backend a run gets, and what this build's CASPAR can do. `CasparCapability` distinguishes an absent build from a failed probe from a measured one; `BaExecutionPlan` records the effective backend and the reason for any fallback, so `summary.json` can say why a run that asked for the GPU solved on the CPU. |
| `correspondences.py` | Reading the database's verified two-view geometries into the graph the triangulator walks; the mapper's input. |
| `point_filters.py` | The guards that run after a solve: a point outside the world, a point inside a camera, a projection that is not finite. |
| `mapping_result.py` | What a mapping pass reports: its rounds, its closing Ceres polish and its final model. |
| `solver_report.py` | The one reader of Ceres' brief report. Unavailable statistics stay `None` rather than becoming a zero-cost solve — which is the normal outcome on the CASPAR backend, since it writes no Ceres line at all. |
| `extrinsic_observations.py` | The observation graph the refinement solves over: which image saw which point and where, indexed once per stage and folded in per round. |
| `extrinsic_costs.py` | What Ceres gets from one residual block: the rig reprojection cost with its analytic pose Jacobian, and the repeated-Cauchy loss. |
| `extrinsic_solve.py` | Problem assembly: `ExtrinsicRefinementOptions`, the three block builders and `solve_extrinsics`. |
| `extrinsic_refinement.py` | Regularised rig-extrinsic refinement: cuSFM's absolute (Eq. 14) and inter-camera relative (Eq. 6) extrinsic priors in pyceres, alternated with pycolmap's own bundle adjustment. What `--optimize-extrinsics` runs by default, because pycolmap's rig BA carries no prior term. |
| `export.py` | Writers for the four artifacts a cuSFM run leaves behind: the `sparse/` model, `kpmap/keyframes/frames_meta.json`, the TUM pose files and `runtime.csv`. The replacement for `kpmap_to_colmap`, `extract_pose_from_map_main` and `update_keyframe_pose_main`. |
| `benchmark.py` | Metrics that compare two runs in the cuSFM output layout, plus the acceptance bounds. It reads both runs the same way and knows nothing about which producer wrote which. Re-exports the names `alignment` and `runtime` own, so one import still covers a whole comparison. |
| `alignment.py` | Umeyama alignment (rigid, or scale-estimating), rig trajectories, the ground-truth reader and the timestamp join every trajectory metric is built on. |
| `runtime.py` | `runtime.csv` classification: the closed `colsfm.pipeline` stage vocabulary as an exhaustive table, the blob's binary names as ordered substring rules, and the "keep only the last appended run" rule. |
| `bench_report.py` | The markdown rendering of a comparison. |
| `bench_cli.py` | The tyro CLI over `benchmark`. Writes the markdown report, the JSON report and the Rerun recording. |
| `rerun_log.py` | Logs a two-run comparison into one Rerun recording: two exoego rigs, two clouds, the trajectory polylines and the report. |
| `walkthrough.py` | The stage-by-stage Rerun walkthrough's CLI and its `WalkthroughObserver`, which draws a `run_pipeline` run rather than running one of its own. |
| `walkthrough_stages.py` | What a walkthrough stage *is*: its prose, its timeline index, its entity root, and the palette every panel shares. |
| `walkthrough_draw.py` | The drawing primitives — frusta, arcs, imagery — and the Markdown notes document each panel closes with. |
| `walkthrough_panels.py` | One `log_*` per stage: what that stage produced, drawn into the recording. |
| `walkthrough_blueprint.py` | The tab-per-stage layout, sent once at the end because stage 2's sample panes are not known before the run. |

## Stage order

`pipeline.run_pipeline` runs these eight stages in this order, and writes one `runtime.csv`
row per stage under these names:

| # | Stage | Module | Blob stage it replaces |
|---|---|---|---|
| 1 | `keyframe_selection` | `keyframe_selection`, `frames_meta` | `feature_extractor_main` (selection half) |
| 2 | `feature_extraction` | `database`, `cameras`, `features`, `features_trt`, `features_raco` | `feature_extractor_main` (per-image half) |
| 3 | `pair_selection` | `pairs` | `feature_matcher_task_builder_main` |
| 4 | `matching` | `matching`, `matching_trt`, `matching_raco` | `feature_matcher_main` |
| 5 | `loop_closure` | `retrieval`, `loop_shortlist`, `loop_closure`, `loop_pose` | `generate_bow_*_main` + `generate_association_main` |
| 6 | `pose_graph` | `pose_graph` | `pose_graph_main` |
| 7 | `reconstruction` | `reconstruction`, `mapping`, `correspondences`, `point_filters`, `ba_backend` | `keypoints_mapper_main` |
| 7b | `extrinsic_refinement` | `reconstruction`, `mapping`, `extrinsic_refinement`, `extrinsic_observations`, `extrinsic_costs`, `extrinsic_solve` | `keypoints_mapper_main --optimize_extrinsics=True` (only with `--optimize-extrinsics`) |
| 8 | `export` | `export` | `kpmap_to_colmap`, `extract_pose_from_map_main`, `update_keyframe_pose_main` |

Each stage is a `run_<stage>_stage` function returning a small result dataclass, and
`run_pipeline` only composes them and times each one, so a caller that wants to step through
a dataset stage by stage runs the same code the full pipeline runs.

## The two ONNX stages have two backends each

Stages 2 and 4 both run a neural network, and each can run it two ways. The default,
`--features-backend pycolmap --matching-backend pycolmap`, hands the work to COLMAP 4.2's
own ALIKED and LightGlue, which COLMAP downloads into `~/.cache/colmap/` and runs through
ONNX Runtime. The alternative, `tensorrt`, runs **the blob's graphs through the blob's
engines** (`pycusfm/models/aliked_lightglue/`), reusing the `.engine` files already in the
repo when the TensorRT version and the GPU architecture match.

They are chosen independently, and all four combinations work: both write float32 xy
keypoints and 128-column float descriptors into the same database rows, and both leave the
matches and two-view geometries COLMAP's verifier produced. Measured on Galileo's 226
frames (RTX 5090); the blob column is the binary this replaces:

| | blob | `pycolmap` | `tensorrt` |
|---|---:|---:|---:|
| feature extraction (s) | 8.64 | 7.19 | 7.01 |
| matching (s) | 2.25 | 8.28 | **2.91** |
| median verified matches per pair | 430 | 161 | 433 |
| mean reprojection error (px) | 1.550 | 1.332 | 1.414 |
| ATE vs ground truth (mm) | 5.00 | 4.35 | 5.19 |
| rig poses vs the blob (mm RMSE / deg RMSE) | — | 0.99 / 1.669 | **0.38 / 0.040** |

So `tensorrt` is the faster matcher by 2.8x and the closer reproduction of the blob, and
`pycolmap` is the more accurate one on this dataset and the one that needs no TensorRT.
The default stays `pycolmap` because it needs neither the `tensorrt` and `cuda-python`
packages nor an engine built for the local GPU.

Three ordering notes matter when reading the code. Pair selection cannot see loop pairs:
stage 3 emits only the consecutive and stereo pairs it can derive from the metadata, and loop
pairs first exist after stage 5, which matches them into the same database. Stage 5 therefore
runs the search twice, once with a recording `match_fn` that collects the candidate pairs and
returns no matches, then one batch match over those pairs, then the real search reading the
matches back. Loop closure is on by default, as in the blob (`--no-loop-closure` is the
ablation). And `--optimize-extrinsics` maps **once**:
stage 7 builds the camera-referenced reconstruction the refinement needs, and stage 7b
refines the extrinsics of the model stage 7 left behind rather than mapping the scene again.

## Tests

| Directory | What it holds |
|---|---|
| `tests/colsfm` | One test file per module, plus `conftest.py` with the Galileo fixtures, the shared geometry helpers (`pose_delta`, `project_and_mask`, `read_blob_keyframes`) and the synthetic cuSFM-layout run builder. These are the pipeline's own tests. Run with `pixi run colsfm-test`. |
| `tests/colsfm_probes` | Runnable probes that establish what pycolmap 4.2.0 offers. Each one builds its own synthetic scene (`synthetic.py`) and asserts the behaviour that `docs/spec/pycolmap-capabilities.md` documents: database, two-view geometry, triangulation, bundle adjustment, pose graph, camera models, model I/O, GPU. beartype is on unconditionally here, because the point of the suite is to check what the bindings accept. Run with `pixi run colsfm-probe`. |

The probes are the reason the specs can say "measured". Change one, and the corresponding
claim in `docs/spec/pycolmap-capabilities.md` needs re-checking.

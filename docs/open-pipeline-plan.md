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

## Blob reference runs (2026-09-05, RTX 5090, `demo_rerun.py --run.variant _blobref`)

The demo runs the blob with `data/cusfm_configs/loop-closure-fixed` (isaac with loop closure
repaired) and `skip_data_association=False`, so all eight stages run. Outputs:
`data/cusfm_runs/{galileo,robocap}_blobref/cusfm/`, recordings `data/bench/*_blob.rrd`.

| | Galileo (226 frames, 8 cams) | RoboCap stride 4 (4528 frames, 4 cams) |
|---|---|---|
| registered / points | 224 / 5065 | 4526 / 168 874 |
| vs input trajectory | 1.8 mm RMSE over 0.66 m | 334 mm RMSE over 124.6 m |
| loop closures in pose graph | 0 | 90 |
| feature_extractor_main | 8.6 s | 183.9 s |
| bow vocabulary + index | 7.5 + 4.2 s | 162.4 + 81.0 s |
| generate_association_main | 3.2 s | 75.7 s |
| pose_graph_main | 3.6 s | 386.6 s |
| task builder + matcher | 1.9 + 2.2 s | 5.6 + 52.5 s |
| keypoints_mapper_main | 6.7 s | 178.7 s |
| kpmap_to_colmap + pose export | 2.1 s | 7.1 s |
| **total cuSFM** | **40.1 s** | **1133.5 s** |

## Acceptance (Galileo, `data/r2b_galileo`, ground truth shipped)

**Revised 2026-09-05: the bounds are now relative to run A, measured by the same harness.**

The original bounds were absolute — registered >= 220, ATE <= 5 mm, reprojection <= 1.7 px —
and the ATE and reprojection figures came from an older NOTES table (3.9 mm, 1.54 px). When
`colsfm.benchmark` measures the blob reference itself it gets **5.00 mm** and **1.550 px**, so
an absolute 5 mm bound demanded that the port beat the binary it reproduces, and the harness
would have failed a bit-perfect clone. Absolute figures also break on a new machine or a
re-run of A. Every bound is therefore expressed against run A as this harness measures it:

| Check | Bound |
|---|---|
| registered images | `>= A - 4` |
| mean reprojection error | `<= 1.10 x A` |
| ATE vs ground truth (Galileo only) | `<= 1.10 x A` |
| total runtime | `<= 2.0 x A` |

Implemented in `colsfm.benchmark.AcceptanceBounds` / `check_acceptance`; the report prints the
resolved absolute value beside each ratio. RoboCap (`data/robocap/s00000021.rrd`, from the
catalog): stride 20 for iteration, stride 4 for the final run, disagreement vs basalt
reported, no ground truth and therefore no acceptance table.

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

## Status (2026-09-05)

**Done.** All eight stages run end to end from `python -m colsfm run`, over the same
`--input-dir` and into the same output layout as `cusfm_cli`, with `runtime.csv`,
`summary.json` and TUM poses. The benchmark harness (`colsfm.bench_cli`) reads two runs in
that layout, recomputes every metric, checks the acceptance bounds and writes one two-rig
Rerun recording. Specs for the eight stages are in `docs/spec/`. The pose graph is
reimplemented on pyceres with a Python residual and matches the blob's archived 8-loop
Galileo run to 0.28 mm. Retrieval has two backends and the vocabulary tree recovers all 90
of the blob's RoboCap loop pairs.

`--optimize-extrinsics` runs the blob's second `keypoints_mapper_main` pass — the same
matches and the same pose-graph poses again, with `sensor_from_rig` free — and writes the
refined `sensor_to_vehicle_transform` into `kpmap/keyframes/frames_meta.json` and into
`summary.json`. It is **regularised**: `colsfm.extrinsic_refinement` carries cuSFM's absolute
(Eq. 14) and inter-camera relative (Eq. 6) extrinsic priors, which pycolmap's bundle adjuster
cannot express, in a pyceres solve alternated with pycolmap's bundle adjustment. Measured on
Galileo (`data/bench/galileo_ext_compare.md`, NOTES.md deviation 9): **0.898 px** reprojection
and **4.28 mm** ATE, against 1.334 px / 4.33 mm with the extrinsics fixed and 0.861 px /
5.70 mm unregularised — the priors buy back all of the trajectory accuracy the free
extrinsics were costing, and keep the extrinsics inside 2.8 mm where the unregularised solve
walked 234 mm. **4/4 acceptance bounds**, 3.2 s for the stage. `--no-regularised-extrinsics`
keeps the plain pycolmap path for comparison. The flag stays off by default: the gain over a
fixed-extrinsic run is 0.4 px of reprojection and 0.02 mm of ATE.

`--ba-backend caspar` runs the bundle adjustments on COLMAP's GPU CASPAR solver instead of
Ceres, in the `colsfm-caspar` environment's from-source pycolmap (`docs/caspar-build.md`),
falling back to Ceres with a message wherever the build, the camera model or
`--optimize-extrinsics` rules it out. Measured in NOTES.md "CASPAR backend": on KITTI 06
with loops it takes the mapping stage from **148.5 s to 31.5 s** (4.7x, total 338.0 s to
208.6 s) and the Sim(3) ATE from 0.895 m to 1.299 m; an ablation puts that on CASPAR's
float32 solve rather than on the robust loss it drops, since Ceres without that loss scores
0.899 m. On Galileo it is a wash (1.76 s against 1.82 s). Off by default.

**Acceptance, Galileo** (`data/bench/galileo_compare.md`), B relative to A:

| Check | Bound | Value | Result |
|---|---|---|---|
| registered images | >= A - 4 = 220 | 225.000 | PASS |
| mean reprojection error (px) | <= 1.10 x A = 1.704 | 1.334 | PASS |
| ATE vs ground truth (mm) | <= 1.10 x A = 5.495 | 4.327 | PASS |
| total runtime ratio | <= 2.0 x A | 0.483 | PASS |

**4/4 bounds met.** RoboCap stride 4 (`data/bench/robocap_compare.md`) has no ground truth
and therefore no acceptance table: 4528/4528 registered, 1.501 px against the blob's 1.486,
421.79 s against 1133.53 s (0.37x), disagreement with the basalt input 461.25 mm against the
blob's 334.10 mm.

**Remaining.**

1. **The loop-edge estimator.** Retrieval and pair selection are solved: our 488 RoboCap
   edges cover all 90 of the blob's LOOP pairs, the blob's own edges through
   `solve_pose_graph` land 135.0 mm from its result, and oracle poses on our pairs land at
   39.9 mm. The measurement is the fault: a two-view essential matrix plus prior-pose scale
   gives 609.0 mm, worse than the 460.8 mm of the input trajectory alone. Stereo
   triangulation plus PnP was measured at 434.5 mm. Closing the rest needs the blob's
   `StereoPoseRefineSolver` and its occupied-area and inlier gates.
2. **Matching is about 3x the blob.** 3.63x on Galileo (8.15 s against 2.25 s) and 2.89x on
   RoboCap (152.03 s against 52.53 s). It is the only stage that is slower, and it is now the
   largest single term in a colsfm run.
3. **Extrinsic priors: done, with one gap left.** `colsfm.extrinsic_refinement` carries both
   priors at the config's own sigmas (`extrinsic_error_meters: 0.01`,
   `extrinsic_error_degrees: 2`, identified by reproducing the blob's logged group costs to
   four significant figures) and reproduces its 777-block relative-constraint arithmetic. What
   is left is magnitude parity on the weakest-constrained cameras: the blob moves the front
   stereo pair 10.4-10.9 mm against camera 0 and colsfm moves it 1.5-2.8 mm. At the solution
   colsfm sits at 8 % of the blob's absolute-prior cost and 21 % of its relative-prior cost, so
   the priors are not what holds it back — with 25 363 observations against the blob's ~42 600,
   colsfm's reprojection term simply does not ask for a larger extrinsic. The block-coordinate
   alternation is also only linearly convergent (19 rounds to reach 0.1 mm / 0.005 deg on
   Galileo); folding the rig poses into the pyceres solve would close that half.
4. **RoboCap with loop closure: measured.** With the rig-resection estimator (74 edges),
   colsfm's disagreement with the input trajectory drops from 461 mm to 313 mm (blob: 334 mm)
   and its rig poses land 146 mm RMSE from the blob's, at 1.52x the blob's runtime. Loop
   closure is on by default since 2026-09-05 (it was off until then; the blob always runs
   it, and comparisons must pay for the stage on both sides); the estimator's rotation gap
   (item 1) and the loop stage's cost (721 s, mostly matching) are what remain.

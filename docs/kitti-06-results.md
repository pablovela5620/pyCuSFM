# KITTI odometry sequence 06 — reproducing the cuSFM paper's Table 4

What was measured, and against what. Sequence 06 of the KITTI odometry benchmark
(1101 stereo frames, 114.29 s, 1231.7 m of ground-truth path) was processed twice
by the NVIDIA PyCuSFM blob and twice by `colsfm`, the open Python + pycolmap port,
and every trajectory was scored against the KITTI ground truth.

The paper's setup is followed as the tutorial describes it: `--config_dir
data/kitti/config`, initial poses from cuVSLAM, and `--skip_data_association`,
because "the experiments in the cuSFM paper were conducted using pose graph
optimization without data association" (`docs/tutorial.md`, KITTI Dataset Example).
`data/kitti/config/match_pair_select_config.pb.txt` sets `search_method:
POSE_GRAPH_ASSOCIATION`, so this is the paper's **view-graph** variant, not the
radius one.

## 1. What was run

| Step | Command | Output |
|---|---|---|
| metadata | `pixi run -- python data/kitti/get_framemeta_file_for_KITTI.py data/kitti/06` | `data/kitti/06/keyframe_meta.json`, copied to `frames_meta.json` |
| blob, odom init | `pixi run -- python -m pycusfm.cusfm_cli --input_dir data/kitti/06 --cusfm_base_dir data/kitti/06_result --config_dir data/kitti/config --skip_data_association` | `data/kitti/06_result/`, log `run.log` |
| blob, SLAM init | the same plus `--use_cuvslam_slam_pose`, `--cusfm_base_dir data/kitti/06_result_slam` | `data/kitti/06_result_slam/` |
| colsfm, odom init | `python -m colsfm run --input-dir data/kitti/06_colsfm_input --output-dir data/kitti/06_colsfm/cusfm --config-dir data/kitti/config --no-loop-closure --min-inter-frame-distance 0.5` | `data/kitti/06_colsfm/` |
| colsfm, SLAM init | the same with `--input-dir data/kitti/06_colsfm_input_slam --output-dir data/kitti/06_colsfm_slam/cusfm` | `data/kitti/06_colsfm_slam/` |
| scoring | `pixi run -e colsfm python -m tools.kitti.evaluate_kitti --sequence-dir data/kitti/06 --trajectories <label> <tum> ...` | `data/bench/kitti_06_ate.json` |
| head-to-head | `python -m colsfm.bench_cli --run-a ... --run-b ... --dataset robocap` | `data/bench/kitti_06_compare.md`, `data/bench/kitti_06_slam_compare.md` |

`data/kitti/get_framemeta_file_for_KITTI.py` always writes `keyframe_meta.json`
and ignores its own `--output-name`; the pipeline reads `frames_meta.json`
(`pycusfm/constants.py:kFRAME_META_FILE`), so the file was copied. It produced
**2202 keyframes over 2 cameras** with one stereo pair (baseline 0.5371506532679237 m)
and **no initial poses** — `world_T_cam` comes back as the identity for every
keyframe, which is what cuVSLAM then fills in.

`colsfm` has no `--override-frames-meta-file`, so its input directories are new
trees that symlink `image_0/` and `image_1/` and carry the blob's
`cuvslam_output/frames_meta_cuvslam.json` renamed to `frames_meta.json`. Grayscale
PNGs caused no trouble anywhere: COLMAP read all 2156 images as `PINHOLE`, and
`colsfm`'s export coloured 97656/97656 points from the same grayscale imagery.

**Frame convention.** Both producers write TUM poses in the *vehicle* frame
(`docs/spec/export.md` §4.2). The generated KITTI metadata gives camera 0 an
identity `sensor_to_vehicle_transform`, so the vehicle frame **is** camera 0 and no
frame change is needed against the KITTI ground truth, which is also camera 0.
`tools/kitti/evaluate_kitti.py --check-frames-meta` asserts this on the actual
metadata and reports `vehicle frame == camera 0: True`.

## 2. Which alignment matches the paper

The paper says ATE came from the EVO toolkit but never states the alignment, so all
three are reported. `evo_ape -a` is Umeyama with the scale held at 1 (SE(3)); `-as`
lets the scale float (Sim(3)).

The answer is **Sim(3)** — `evo_ape ... -as`. Our cuVSLAM SLAM trajectory, which is
the same quantity the paper's `CuVSLAM` row measures, lands at RMSE 1.299 / mean
1.172 / median 1.127 / **std 0.561** / min 0.309 / **max 2.802** against the paper's
1.202 / 1.055 / 0.907 / **0.577** / 0.234 / **2.855**. The standard deviation agrees
to 3 %, the maximum to 2 %, and the shape of the whole row matches. Unaligned is far
too large (3.461) and SE(3) is noticeably larger (1.600) than any published number.

## 3. ATE against KITTI ground truth

Every "ours" row is `tools/kitti/evaluate_kitti.py` over
`data/kitti/06/poses_gt_06.txt` (1101 poses), associated on timestamp with a 2000 µs
tolerance; the raw numbers are in `data/bench/kitti_06_ate.json`. Paper rows are
Table 4 and Table 5 of `/tmp/cusfm_paper.txt` for sequence 06, quoted for reference
only. All values in metres.

### Sim(3)-aligned (`evo_ape -as`) — the mode that matches the paper

| Method | RMSE | Mean | Median | STD | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| *paper* CuVSLAM | 1.202 | 1.055 | 0.907 | 0.577 | 0.234 | 2.855 |
| *paper* CuSfM-View Graph (cuVSLAM init) | 0.783 | 0.741 | 0.738 | 0.255 | 0.112 | 1.488 |
| *paper* CuSfM (ORB-SLAM2 init, Table 5) | 0.436 | 0.436 | 0.387 | 0.175 | 0.009 | 0.788 |
| ours cuVSLAM odom | 2.136 | 1.937 | 1.832 | 0.900 | 0.128 | 3.861 |
| ours cuVSLAM slam | 1.299 | 1.172 | 1.127 | 0.561 | 0.309 | 2.802 |
| ours blob cuSFM, odom init | 2.134 | 1.936 | 1.877 | 0.897 | 0.018 | 3.858 |
| ours colsfm, odom init | 2.181 | 2.005 | 2.029 | 0.859 | 0.453 | 4.242 |
| ours blob cuSFM, SLAM init | **1.328** | 1.183 | 1.131 | 0.603 | 0.304 | 3.039 |
| ours colsfm, SLAM init | 1.596 | 1.444 | 1.334 | 0.679 | 0.522 | 3.652 |

### SE(3)-aligned (`evo_ape -a`)

| Method | RMSE | Mean | Median | STD | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| ours cuVSLAM odom | 2.375 | 2.243 | 1.950 | 0.780 | 1.311 | 4.454 |
| ours cuVSLAM slam | 1.600 | 1.530 | 1.527 | 0.469 | 0.642 | 2.991 |
| ours blob cuSFM, odom init | 2.345 | 2.217 | 1.884 | 0.764 | 1.266 | 4.367 |
| ours colsfm, odom init | 2.277 | 2.162 | 2.066 | 0.716 | 1.259 | 4.329 |
| ours blob cuSFM, SLAM init | 1.605 | 1.525 | 1.476 | 0.498 | 0.737 | 3.071 |
| ours colsfm, SLAM init | 1.774 | 1.665 | 1.727 | 0.613 | 0.705 | 3.581 |

The SE(3) column is independently confirmed by `colsfm.bench_cli`, which fits the
same rigid transform in millimetres: 2344.69 mm for the blob and 2277.39 mm for
`colsfm` on the odom pair (`data/bench/kitti_06_compare.md`), 1604.74 mm and
1774.44 mm on the SLAM pair (`data/bench/kitti_06_slam_compare.md`). Two
independent implementations, same numbers.

### Unaligned

| Method | RMSE | Mean | Median | STD | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| ours cuVSLAM odom | 3.764 | 3.439 | 3.193 | 1.531 | 0.865 | 6.754 |
| ours cuVSLAM slam | 3.461 | 3.002 | 2.675 | 1.722 | 0.701 | 6.752 |
| ours blob cuSFM, odom init | 3.720 | 3.395 | 3.182 | 1.521 | 0.801 | 6.637 |
| ours colsfm, odom init | 4.139 | 3.555 | 2.882 | 2.121 | 0.998 | 7.885 |
| ours blob cuSFM, SLAM init | 3.481 | 2.992 | 2.594 | 1.779 | 0.741 | 6.801 |
| ours colsfm, SLAM init | 3.918 | 3.150 | 2.077 | 2.330 | 0.248 | 8.155 |

The Sim(3) scale is 0.9925 (cuVSLAM odom) to 0.9952 (`colsfm`), i.e. every
trajectory is metric to within 0.8 %; the alignment is not rescuing a broken scale.

## 4. Reconstruction

From `data/bench/kitti_06_compare.md` and `data/bench/kitti_06_slam_compare.md`
(`colsfm.bench_cli`, which recomputes the reprojection error from the tracks because
`kpmap_to_colmap` hardcodes `ERROR = 2.0`).

| Metric | blob (odom) | colsfm (odom) | blob (SLAM) | colsfm (SLAM) |
|---|---:|---:|---:|---:|
| registered / total images | 2154 / 2154 | 2156 / 2156 | 2154 / 2154 | 2156 / 2156 |
| rig frames registered | 1077 | 1078 | 1077 | 1078 |
| 3D points | 120252 | 97656 | 120097 | 97655 |
| observations | 738528 | 529169 | 737199 | 529105 |
| mean reprojection error (px) | 0.455 | 0.599 | 0.454 | 0.599 |
| track length mean / median | 6.14 / 5.00 | 5.42 / 4.00 | 6.14 / 5.00 | 5.42 / 4.00 |
| point colours present | no | yes | no | yes |

The two-image / one-sample difference is keyframe selection: the blob's
`feature_extractor_main` kept 2154 of 2200 keyframes, `colsfm` kept 2156 with the
same `--min-inter-frame-distance 0.5`. `colsfm` also finds 19 % fewer points and a
0.14 px worse reprojection error, both of which trace to its 500-match-per-pair
spatial cap: `colsfm.matching: spatial cap at 500/pair dropped 3782937 inlier
matches` (`data/kitti/06_colsfm/run.log`), a retention of 0.201.

## 5. Runtime

Per-stage wall clock, blob from `runtime.csv` (written by the runner), `colsfm` from
its own `runtime.csv`. Both are the odom-init runs; the SLAM-init runs are in
`data/bench/kitti_06_slam_compare.md`.

| Stage | blob (s) | colsfm (s) | colsfm / blob |
|---|---:|---:|---:|
| feature extraction + keyframe selection | 123.01 | 19.16 | 0.16 |
| BoW vocabulary + index | 103.51 | — | — |
| loop-closure association | — | 0.00 | — |
| pose graph optimisation | 1098.72 | 0.21 | 0.00 |
| match pair selection | 6.31 | 0.01 | 0.00 |
| feature matching | 21.02 | 87.89 | 4.18 |
| triangulation + bundle adjustment | 113.33 | 149.88 | 1.32 |
| COLMAP + TUM export | 6.72 | 3.57 | 0.53 |
| **total** | **1472.62** | **260.71** | **0.18** |

Whole-process wall clock, `/usr/bin/time -v`: the blob run took **24:41.87**
(odom) and **26:46.62** (SLAM); `colsfm` took **4:21.83** and **5:05.26**. cuVSLAM
itself is not in the blob's `runtime.csv` — it ran in about 9 s, between the
`keyframe_metadata_to_edex_main` and `feature_extractor_main` log lines of
`data/kitti/06_result/run.log`.

**The blob's pose-graph stage costs 1098.72 s and buys nothing here.** It spends
18 minutes querying the BoW index for loop candidates and then logs `Perform pose
graph optimization with 0 loop constraints` (`data/kitti/06_result/pose_graph/pose_graph_main.txt`),
optimising 1076 relative-pose constraints plus one position anchor. The saved graph
confirms it: `data/kitti/06_result/pose_graph/vehicle_pose_graph.pb.txt` holds 1076
`CONSECUTIVE` edges and **zero `LOOP` edges**; the per-camera
`pose_graph/pose_graph.pb.txt` holds 2152 `CONSECUTIVE` and 2152 `EXTRINSIC` edges
and again no loops. The SLAM-init run is identical in this respect (1076
`CONSECUTIVE`, 0 loops, 1200.01 s). `colsfm` was run with `--no-loop-closure`, so
its pose graph is the same chain and converges in 0.21 s with a cost of 5e-25 —
the relative constraints are derived from the input poses, so the input already
satisfies them exactly.

Against the paper's Table 2 (seconds per 100 frames, on 1077 samples): feature
extraction 11.4 s/100 against the paper's 4.996, matching 1.95 s/100 against 37.370,
mapping 10.5 s/100 against 16.458. Matching is 19x faster here and extraction 2.3x
slower, which is what a 3229-pair view graph on an RTX 5090 versus an RTX 6000 Ada
looks like; the paper does not say how many pairs its view-graph mode built.

## 6. What matches the paper and what does not

The cuVSLAM baseline reproduces. Our cuVSLAM SLAM trajectory (Sim(3) RMSE 1.299,
STD 0.561, max 2.802) is within a few per cent of the paper's `CuVSLAM` row (1.202 /
0.577 / 2.855) on every statistic except the minimum, which is a single-pose
quantity and not stable. That is the strongest evidence that the alignment is Sim(3)
and that the data, calibration and frame conventions here are right.

The refinement does not reproduce. The paper's `CuSfM-View Graph` improves cuVSLAM's
1.202 to 0.783, a 35 % reduction. Ours improves 1.299 to 1.328 — no improvement at
all, in fact 2 % worse; `colsfm` goes to 1.596, 23 % worse. The reason is visible in
the logs rather than inferred: **the pose graph found zero loop constraints**, so the
only thing the blob's pose-graph stage can do is re-solve a chain that its own input
already satisfies, and the only correction left comes from the mapper's bundle
adjustment, which is anchored by `relative_pose_translation_error_meters: 0.1` and
`absolute_pose_translation_error_meters: 1` priors on the same input poses
(`data/kitti/config/vision_mapping_config.pb.txt`). Sequence 06 is a loop — the
vehicle drives a circuit and revisits its start — so a loop closure exists in the
data and was not found. With `loop_closure_min_words: 1`, `num_votes: 1` and
`good_loop_threshold: 150` in `data/kitti/config/pose_graph_config.pb.txt`, the
candidate search ran (18 minutes of it) and every candidate was rejected. Whether
that is a threshold mismatch for this cuVSLAM version, a BoW vocabulary built from
too few images, or a behaviour difference between the 0.1.3-era binaries and the
paper's 0.1.8 is not established by this run.

The initial-pose source matters more than anything else measured here. The blob
defaults to `use_cuvslam_slam_pose=False`, i.e. cuVSLAM's **odometry** poses, which
score 2.136 — far from the paper's 1.202. Only `--use_cuvslam_slam_pose` reaches the
paper's baseline. Anyone reproducing Table 4 has to pass that flag; the tutorial
command does not. This is consistent with the paper's own observation that
"refinement performance exhibits dependency on initial values".

`colsfm` tracks the blob closely and neither beats it. On the odom pair the two rig
trajectories differ by 960 mm RMSE and 1.49° RMSE, on the SLAM pair by 735 mm and
1.11°; `colsfm` is slightly better than the blob under SE(3) with odom init (2.277
against 2.345) and slightly worse with SLAM init (1.774 against 1.605). Given that
both are anchored to the same input poses and neither closes a loop, that spread is
the mapper's, not the trajectory's.

**Hardware and version differences.** The paper used an Intel i9-9820X with an RTX
6000 Ada (48 GB), Ubuntu 24.04, driver 560.35.03, CUDA 12.6, PyCuSfM v0.1.8. This
run used an AMD Ryzen 9 9950X3D (32 threads) with an RTX 5090 (32 GB), driver
580.173.02, CUDA 13.0, the CUDA-13 binaries in `pycusfm/x86_cuda13`. The repo
declares `__version__ = "0.1.8"` (`pycusfm/__init__.py`) while `NOTES.md` records
the fork as frozen at upstream `0c97a67`, "Release 0.1.3"; the binaries are
therefore not certainly the paper's. The cuVSLAM build is whatever
`cuvslam_api_launcher` in that same tree contains — it reports no version string.
`colsfm` runs on pycolmap 4.2.0 with CUDA enabled.

## 7. `colsfm` bug hit on the way

The working tree of `colsfm/` is mid-refactor and its pipeline does not run:

```
File "colsfm/pipeline.py", line 712, in run_pipeline
    create_database(options.database_path, selected, colmap_cameras(selected), overwrite=True)
TypeError: create_database() takes 2 positional arguments but 3 positional arguments (and 1 keyword-only argument) were given
```

`colsfm/database.py` is modified in the working tree and `create_database` lost its
optional third parameter — at `HEAD` the signature is
`create_database(database_path, frames_meta, cameras=None, *, overwrite=False)`,
in the working tree it is `create_database(database_path, frames_meta, *, overwrite=False)`.
`colsfm/pipeline.py` is *not* modified and still passes `colmap_cameras(selected)`
positionally. Every `colsfm` run reported here therefore used the last committed
state, `274b5c50b4e196c7c5bed1a6343b201b67b650ec`, extracted with
`git archive HEAD colsfm` into a scratch directory outside the repo and run from
there; nothing under `colsfm/` was edited. `colsfm.bench_cli` was run from the same
snapshot so that every number comes from one code state.

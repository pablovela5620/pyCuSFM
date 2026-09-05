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

## 8. Follow-up: the match cap, loop closure and the TensorRT backends

Section 4 blamed `colsfm`'s 19 % point deficit and its 0.14 px worse reprojection
error on the fixed 500-match spatial cap, which discarded 3 782 937 verified
inliers here (retention 0.201). This section tests that: five `colsfm` runs from
the same cuVSLAM **SLAM** initialisation, differing only in the cap mode, in
whether loop closure ran, and in which ALIKED and LightGlue implementation ran.

### 8.1 What was run

Every run is the same command with different flags, on the same host as section 5
(Ryzen 9 9950X3D, RTX 5090):

```bash
python -m colsfm run \
    --input-dir data/kitti/06_colsfm_input_slam \
    --config-dir data/kitti/config \
    --min-inter-frame-distance 0.5 \
    --output-dir <run>/cusfm \
    <flags>
```

| Label | Flags |
|---|---|
| cap 500 | `--no-loop-closure --match-cap-mode fixed` (the section 3 baseline) |
| cap off | `--no-loop-closure --match-cap-mode off` |
| image_area | `--no-loop-closure --match-cap-mode image_area` |
| cap 500 + loops | `--loop-closure --match-cap-mode fixed` |
| cap off + loops | `--loop-closure --match-cap-mode off` |
| TensorRT + loops | `--loop-closure --match-cap-mode fixed --features-backend tensorrt --matching-backend tensorrt` |

`--match-cap-mode` and the two `--backend` flags are new; see NOTES.md decisions
13 and 14. `image_area` keeps the density the 500 was calibrated at — one kept
match per 4608 px of image — which on KITTI's 1241x376 frames is a cap of 101.
The TensorRT run keeps `fixed`, because on that backend the cap is the target of
the blob's *real* SSC spatial NMS rather than a post-verification grid subsample.

Scored exactly as section 3 was:

```bash
pixi run -e colsfm python -m tools.kitti.evaluate_kitti \
    --sequence-dir data/kitti/06 --trajectories <label> <run>/cusfm/output_poses/merged_pose_file.tum ...
```

### 8.2 The cap: 500 is already the right number

Sim(3)-aligned ATE against `data/kitti/06/poses_gt_06.txt`, metres. The two
reference rows are re-scored here so every number in this table comes from one
evaluation run.

| Run | RMSE | Mean | Median | STD | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| cuVSLAM SLAM (input) | 1.336 | 1.191 | 1.145 | 0.604 | 0.305 | 2.948 |
| blob cuSFM, SLAM init | **1.328** | 1.183 | 1.131 | 0.603 | 0.304 | 3.039 |
| colsfm, cap 500 | **1.552** | 1.408 | 1.299 | 0.653 | 0.503 | 3.459 |
| colsfm, cap off | 1.822 | 1.667 | 1.659 | 0.735 | 0.463 | 4.148 |
| colsfm, image_area (101) | 2.103 | 1.890 | 1.560 | 0.923 | 0.301 | 5.236 |

SE(3)-aligned, same runs:

| Run | RMSE | Mean | Median | STD | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| cuVSLAM SLAM (input) | 1.648 | 1.566 | 1.534 | 0.512 | 0.712 | 3.051 |
| blob cuSFM, SLAM init | 1.605 | 1.525 | 1.476 | 0.498 | 0.737 | 3.071 |
| colsfm, cap 500 | **1.732** | 1.635 | 1.653 | 0.571 | 0.668 | 3.427 |
| colsfm, cap off | 1.917 | 1.806 | 1.713 | 0.644 | 0.563 | 4.202 |
| colsfm, image_area (101) | 2.243 | 2.053 | 1.882 | 0.905 | 0.405 | 5.165 |

Reconstruction and per-stage runtime:

| Metric | cap 500 | cap off | image_area |
|---|---:|---:|---:|
| registered images | 2156 | 2156 | 2156 |
| 3D points | 97 654 | 221 545 | 28 484 |
| observations | 529 101 | 2 523 036 | 131 926 |
| mean reprojection (px) | 0.599 | 0.922 | 0.485 |
| mean track length | 5.42 | 11.39 | 4.63 |
| feature extraction (s) | 17.9 | 17.7 | 17.2 |
| matching (s) | 94.4 | 73.9 | 102.4 |
| pose graph (s) | 0.2 | 0.2 | 0.2 |
| triangulation + BA (s) | 151.7 | 513.5 | 68.0 |
| export (s) | 3.6 | 4.3 | 3.4 |
| **total (s)** | **268.0** | **609.9** | **191.3** |

**The cap was not the problem.** Section 4 read the log line `spatial cap at
500/pair dropped 3782937 inlier matches` as a loss. It is not: removing the cap
gives the mapper 4.8x the observations and 2.3x the points, and the trajectory
gets **worse**, from 1.552 to 1.822 Sim(3) RMSE, while the run takes 2.3x as long
and the reprojection error rises from 0.599 px to 0.922 px. Scaling the cap down
with the frame area (`image_area`, 101 matches per pair here) is worse again, at
2.103. 500 sits between two worse answers.

That is the same shape decision 7 measured on Galileo — uncapped 1300 matches per
pair at 1.80 px and 5.39 mm ATE against 156 matches at 1.33 px and 4.32 mm — so
the cap is doing the same job on both datasets: it refuses to pile a thousand
near-collinear observations from one image region onto one track. The 80 % figure
in section 4 is a red herring; Galileo discards 88 % and gains from it.

`match_cap_mode` therefore stays at `fixed` by default. It exists because the
question was worth answering, and now it is answered with numbers rather than
with a log line.

### 8.3 Loop closure is worth 0.8 m, and it is not the pose graph that pays

Two more runs add `--loop-closure`, and two more open the pose-graph loop gates
that `data/kitti/config` leaves at 0.0. Sim(3)-aligned ATE, metres, every row
from one `evaluate_kitti` invocation (`/tmp/colsfm_runs/kitti_followup_ate.json`):

| Run | loop edges | RMSE | Mean | Median | STD | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| *paper* CuSfM-View Graph | — | 0.783 | 0.741 | 0.738 | 0.255 | 0.112 | 1.488 |
| cuVSLAM SLAM (input) | — | 1.336 | 1.191 | 1.145 | 0.604 | 0.305 | 2.948 |
| blob cuSFM, SLAM init | 0 | 1.328 | 1.183 | 1.131 | 0.603 | 0.304 | 3.039 |
| colsfm, cap 500 | — | 1.552 | 1.408 | 1.299 | 0.653 | 0.503 | 3.459 |
| colsfm, cap off | — | 1.822 | 1.667 | 1.659 | 0.735 | 0.463 | 4.148 |
| colsfm, image_area | — | 2.103 | 1.890 | 1.560 | 0.923 | 0.301 | 5.236 |
| colsfm, cap 500 + loops | 0 | 0.895 | 0.781 | 0.728 | 0.437 | 0.045 | 1.846 |
| colsfm, cap off + loops | 0 | 0.817 | 0.731 | 0.681 | 0.363 | 0.103 | 1.503 |
| **colsfm, TensorRT + loops** | 0 | **0.736** | 0.624 | 0.458 | 0.390 | 0.106 | 1.690 |
| colsfm, cap 500 + loops, gates open | 279 | 1.428 | 1.303 | 1.301 | 0.585 | 0.431 | 2.774 |
| colsfm, TensorRT + loops, gates open | 283 | 1.893 | 1.649 | 1.455 | 0.930 | 0.214 | 4.552 |

SE(3)-aligned, same runs:

| Run | RMSE | Mean | Median | STD | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| cuVSLAM SLAM (input) | 1.648 | 1.566 | 1.534 | 0.512 | 0.712 | 3.051 |
| blob cuSFM, SLAM init | 1.605 | 1.525 | 1.476 | 0.498 | 0.737 | 3.071 |
| colsfm, cap 500 | 1.732 | 1.635 | 1.653 | 0.571 | 0.668 | 3.427 |
| colsfm, cap off | 1.917 | 1.806 | 1.713 | 0.644 | 0.563 | 4.202 |
| colsfm, image_area | 2.243 | 2.053 | 1.882 | 0.905 | 0.405 | 5.165 |
| colsfm, cap 500 + loops | 1.118 | 1.075 | 0.969 | 0.308 | 0.666 | 1.823 |
| **colsfm, cap off + loops** | **1.029** | 1.001 | 0.932 | 0.239 | 0.646 | 1.477 |
| colsfm, TensorRT + loops | 1.111 | 1.038 | 1.092 | 0.396 | 0.295 | 1.861 |
| colsfm, cap 500 + loops, gates open | 1.589 | 1.517 | 1.452 | 0.473 | 0.746 | 2.838 |
| colsfm, TensorRT + loops, gates open | 1.989 | 1.795 | 1.660 | 0.857 | 0.570 | 4.611 |

Reconstruction and per-stage runtime for all seven `colsfm` runs:

| Metric | cap 500 | cap off | image_area | 500+loops | off+loops | TRT+loops | 500+loops gated | TRT+loops gated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| registered images | 2156 | 2156 | 2156 | 2156 | 2156 | 2156 | 2156 | 2156 |
| 3D points | 97 654 | 221 545 | 28 484 | 96 691 | 174 339 | 174 319 | 97 674 | 176 927 |
| observations | 529 101 | 2 523 036 | 131 926 | 713 124 | 2 707 266 | 1 214 490 | 712 275 | 1 217 074 |
| mean reprojection (px) | 0.599 | 0.922 | 0.485 | 0.674 | 0.987 | 0.534 | 0.674 | 0.533 |
| mean track length | 5.42 | 11.39 | 4.63 | 7.38 | 15.53 | 6.97 | 7.29 | 6.88 |
| loop edges in the pose graph | 0 | 0 | 0 | 0 | 0 | 0 | 279 | 283 |
| extra loop pairs matched | 0 | 0 | 0 | 1344 | 1344 | 1320 | 1344 | 1320 |
| feature extraction (s) | 17.9 | 17.7 | 17.2 | 16.9 | 17.5 | **62.2** | 16.7 | 62.2 |
| matching (s) | 94.4 | 73.9 | 102.4 | 76.3 | 84.2 | **25.2** | 77.1 | 25.2 |
| loop closure (s) | 0.0 | 0.0 | 0.0 | 93.0 | 97.6 | 62.3 | 92.6 | 63.5 |
| pose graph (s) | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 1.9 | 1.5 |
| triangulation + BA (s) | 151.7 | 513.5 | 68.0 | 148.5 | 533.8 | 209.9 | 145.3 | 244.1 |
| export (s) | 3.6 | 4.3 | 3.4 | 2.9 | 3.4 | 3.5 | 3.6 | 2.9 |
| **total (s)** | 268.0 | 609.9 | 191.3 | **338.0** | 736.8 | 363.5 | 337.5 | 399.6 |

**The loop is there, and finding it is worth 0.8 m.** `--loop-closure` takes
`colsfm` from 1.552 to 0.895 (cap 500), 0.817 (cap off) and **0.736** (TensorRT
backends) Sim(3) RMSE. Section 6 said "sequence 06 is a loop and the loop closure
was not found"; it is found, and closing it puts `colsfm` past the blob's 1.328
and past the paper's own 0.783 for its `CuSfM-View Graph` row, from the same
cuVSLAM SLAM initialisation. The cost is 70 s of extra runtime — 338.0 s against
268.0 s — because 1344 of the 2386 retrieval candidates still had to be matched.

**It is not the pose graph that closes it.** Every one of those runs reports
`0 loop edges`: `data/kitti/config/pose_graph_config.pb.txt` sets neither
`loop_edge_translation_threshold_meters` nor `loop_edge_rotation_threshold_degrees`,
so `colsfm.pose_graph.gate_loop_edges` rejects all 275-290 verified candidates
and the pipeline prints its own warning about it. What closes the loop is that
the loop-closure stage **matches its candidate pairs into the same database**, and
`colsfm.mapping.load_correspondences` reads every verified two-view geometry —
so the loop pairs reach triangulation and bundle adjustment as ordinary tracks
even though no pose-graph constraint survived. Observations rise from 529 101 to
713 124 and the mean track length from 5.42 to 7.38 while the point count barely
moves: those are the same points, seen again from the other side of the loop.

**Opening the gates makes it worse.** A `data/kitti/config` copy with
`loop_edge_translation_threshold_meters: 5` and
`loop_edge_rotation_threshold_degrees: 60` — the values
`colsfm.pose_graph.gate_loop_edges` documents for its own A/B — admits 279 and 283
loop edges and the pose graph converges (cost 77.7 -> 1.56, 30 iterations). The
trajectory then degrades from 0.895 to 1.428 and from 0.736 to 1.893. The
measurement each edge carries comes from `colsfm.loop_pose`'s generalized
resection, and NOTES.md deviation 5 already records that it lands 408 mm from the
blob's own loop poses on RoboCap where the blob reaches 135 mm; on a 1231 m car
sequence that error is large enough that 279 such constraints pull the chain off
the answer bundle adjustment would otherwise have found. **The recommendation for
KITTI-like data is `--loop-closure` with the pose-graph loop gates left closed**,
which is what the shipped config already does.

**The TensorRT extractor is the wrong tool for KITTI.** Feature extraction is
62.2 s against 17.9 s, because `aliked.onnx` takes a fixed 1920x1200 input and a
1241x376 KITTI frame is stretched *up* to it, while COLMAP's own ALIKED runs at
the native size. Matching goes the other way, 25.2 s against 94.4 s. The net is a
wash on total runtime (363.5 s against 338.0 s) and a clear win on accuracy
(0.736 against 0.895), which is the blob's own descriptors and the blob's own SSC
doing their job.

### 8.4 What this means against the paper's 0.783

| Method | Sim(3) RMSE | Improvement over its own cuVSLAM input |
|---|---:|---:|
| *paper* CuSfM-View Graph | 0.783 | -35 % (from 1.202) |
| blob cuSFM, SLAM init | 1.328 | +2 % (from 1.336) |
| colsfm, cap 500, no loops | 1.552 | +16 % |
| **colsfm, TensorRT + loops** | **0.736** | **-45 % (from 1.336)** |

Section 6 concluded that "the refinement does not reproduce" and that the reason
was a loop that existed in the data and was never found. That conclusion holds
for the blob and is now explained for `colsfm`: with the loop found, the open port
improves its cuVSLAM input by 45 %, against the paper's 35 %, and lands at 0.736
against the paper's 0.783. The remaining differences from the paper — a different
GPU, a different cuVSLAM build, COLMAP's LO-RANSAC triangulation against cuSFM's
MSAC — are the ones section 6 already lists, and none of them is now hiding a
missing loop.

Two caveats on that number. It is one sequence, and the run that produces it uses
both TensorRT backends, so it depends on `pycusfm/models/aliked_lightglue/` being
present. The pycolmap-backend equivalent, `cap 500 + loops`, is 0.895 — still past
the blob and still a 33 % improvement on its input, with no TensorRT anywhere.

**What changed in the code.** `--match-cap-mode` (NOTES.md decision 14) and the
two `--*-backend` flags (decision 13). Nothing else: the loop-closure stage, the
pose graph and the mapper are the same code section 3 measured, and the 1.552
baseline reproduces section 3's 1.596 to within the run-to-run spread of a
threaded Ceres solve.


## 9. RaCo backend at scale

Section 8 measured the blob's own TensorRT graphs on KITTI 06. This section runs
the third backend — RaCo-ALIKED plus LightGlue+, NOTES.md "RaCo backend" — on the
same 2156 images, from the same cuVSLAM SLAM initialisation, with the flags of
the `TensorRT + loops` run and nothing else changed:

```bash
pixi run -e colsfm python -m colsfm run \
    --input-dir data/kitti/06_colsfm_input_slam --config-dir data/kitti/config \
    --min-inter-frame-distance 0.5 --output-dir data/kitti/06_colsfm_raco_loops/cusfm \
    --loop-closure --match-cap-mode fixed \
    --features-backend raco --matching-backend raco
```

Stage times from `data/kitti/06_colsfm_raco_loops/cusfm/runtime.csv`, reconstruction
counts from that run's `summary.json`, ATE from one `evaluate_kitti` invocation
that re-scores every row (`data/bench/kitti_06_raco_ate.json`), blob columns from
`data/kitti/06_result_slam/runtime.csv` by way of
`data/bench/kitti_06_raco_compare.md`. The three earlier `colsfm` rows are their
own `/tmp/colsfm_runs/kitti/<run>/cusfm/summary.json`.

| Run | extraction (s) | matching (s) | loop stage (s) | mapping (s) | total (s) | registered | points | reprojection (px) | Sim(3) RMSE (m) | SE(3) RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *paper* CuSfM-View Graph | — | — | — | — | — | — | — | — | 0.783 | — |
| blob cuSFM, SLAM init | 96.85 | 32.61 | 106.75 BoW + 1200.01 pose graph | 143.14 | 1596.08 | 2154 | 120 097 | 0.454 | 1.328 | 1.605 |
| colsfm pycolmap + loops (cap 500) | 16.9 | 76.3 | 93.0 | 148.5 | 338.0 | 2156 | 96 691 | 0.674 | 0.895 | 1.118 |
| colsfm pycolmap + loops (cap off) | 17.5 | 84.2 | 97.6 | 533.8 | 736.8 | 2156 | 174 339 | 0.987 | 0.817 | 1.029 |
| colsfm TensorRT + loops | 62.2 | 25.2 | 62.3 | 209.9 | 363.5 | 2156 | 174 319 | 0.534 | 0.736 | 1.111 |
| **colsfm RaCo + loops** | **42.3** | **23.8** | **55.0** | **170.8** | **295.0** | 2156 | 139 976 | 0.507 | 0.902 | 1.419 |

The blob's reprojection error is `colsfm.bench_cli`'s recomputed one; the `colsfm`
rows are each run's own `mean_reprojection_error_px`, which for this run agrees with
the harness to 0.001 px (0.5068 against 0.507). The RaCo run is the fastest of every
loop-closing run here — 295.0 s, 18 % of the blob's 1596.08 s, and below even the
`cap 500` run that closes no loops at all (268.0 s) once its 55 s loop stage is
subtracted — and it
matched 3232 view-graph pairs plus 1376 loop pairs (`summary.json`), against the
TensorRT run's 3232 + 1320.

**The engine was already built, and that build is not free.** `resolve_engine`
found `data/cusfm_models/raco-aliked-b1-16_fp16_10_13_3_9_sm_12_0.engine` from an
earlier session, so the 42.3 s extraction stage is steady state with only the
engine *load* inside it; the run log contains no build. NOTES.md records the one-time
cost of producing that file as **19 min on this RTX 5090**, and KITTI does not pay it
again for a second reason: the graph's spatial dimensions are static, so the same
engine serves both datasets.

**RaCo is not dynamic in height and width, only in batch.** `raco_profile` builds the
optimisation profile at `NETWORK_HEIGHT` x `NETWORK_WIDTH` = 1200 x 1920 for all three
of min/opt/max, and `extract_raco` reads the height and width back out of the built
engine, so KITTI 06's 1226x370 frames are stretched up to 1920x1200 exactly as
section 8.3 describes for the blob's `aliked.onnx`. The RaCo win over that backend is
therefore the engine itself and batch 8, not a smaller network input: extraction falls
from 62.2 s to 42.3 s (28.8 to **19.6 ms per image**), while COLMAP's own ALIKED at
the native size still wins at 16.9 s (7.8 ms per image).

**At 19.6 ms per image the stage is roughly half GPU.** Decode, resize to 1920x1200
and the CHW float conversion of all 2202 images through the same 8-worker pool cost
**15.73 s, 7.1 ms per image** measured directly on this host, against the 11.0 ms per
image of device time NOTES.md measured for this engine at batch 8. That is the
inversion of Galileo: there the stage was 8.30 s over 226 frames, 36.7 ms per image,
of which only ~11 ms was the GPU, because 1920x1200 JPEGs cost more to decode than the
engine costs to run. KITTI's frames are a fifth of that area and PNG, so the CPU half
shrinks to 7.1 ms and the fixed-shape engine — which cannot get cheaper, because the
input it sees is 1920x1200 either way — becomes the larger term. The remaining ~1.5 ms
per image is the host-to-device copy, the keypoint rescale and the two database writes.

**Matching scales, and it is the cheapest of the four.** 23.8 s over 3232 pairs is
**7.4 ms per pair**, against TensorRT's 7.8 (25.2 s) and pycolmap's 23.6 (76.3 s); the
loop stage, which matches 1376 extra pairs on top of retrieval, is 55.0 s against 62.3
and 93.0. This reproduces at 14x the pair count what Galileo showed at 2.31 s: LightGlue+
is the stage the RaCo pair actually makes faster, and it is faster than the blob's own
LightGlue engine.

**Accuracy held against the pycolmap backend, not against the blob's descriptors.**
Sim(3) RMSE is **0.902 m**, statistically the `cap 500 + loops` number (0.895) and well
past the blob's 1.328, but short of the TensorRT run's 0.736. SE(3) is 1.419 m, the worst
of the three `colsfm` loop runs (1.118, 1.029, 1.111) — a Sim(3) scale of 0.99198 against
TensorRT's 0.99390 says RaCo's tracks drift slightly more in scale, which SE(3) alignment
cannot absorb. The reconstruction is otherwise the healthiest of the four: 2156/2156
registered, 139 976 points, 1 073 575 observations, a mean track length of 7.67 (the
longest here) and a mean reprojection error of 0.507 px, the best of any `colsfm` run.
Against the blob, `colsfm.bench_cli` passes three of its four acceptance bounds and misses
reprojection error by 0.007 px (0.507 against a 0.500 ceiling, `data/bench/kitti_06_raco_compare.md`);
the TensorRT run's 0.534 would have missed the same bound by more.

**What this means.** RaCo is the right backend when total wall clock matters — it is
1.15x faster than the pycolmap backend (338.0 s) and 1.23x faster than TensorRT (363.5 s)
on this sequence, and it beats the blob at both GPU stages by the widest margin of the
three backends (2.3x on extraction, 1.4x on matching). It is not the
right backend when the 0.736 matters: on KITTI the blob's own ALIKED descriptors still
produce the better trajectory, and the honest summary is that RaCo buys the pycolmap
backend's accuracy at TensorRT's speed, plus 20 s of extraction over the pycolmap backend
that a dynamic-shape export would remove.


## 10. RaCo at native resolution, Ceres against CASPAR mapping

Section 9's RaCo run stretched every 1226x370 frame to 1920x1200. NOTES.md "RaCo
backend" records the dynamic-shape export that removed the stretch (extraction 42.3 s
to 10.1 s, total 295.0 s to 203.2 s, Sim(3) 0.902 m to 0.878 m). This section stacks
the GPU bundle adjuster from `docs/caspar-build.md` on top of that run, in the
`colsfm-caspar-fisheye` environment (KITTI is PINHOLE, so the stock adapter serves;
the fisheye build is simply the CASPAR build that was installed):

```bash
pixi run -e colsfm-caspar-fisheye python -m colsfm run \
    --input-dir data/kitti/06_colsfm_input_slam --config-dir data/kitti/config \
    --output-dir /tmp/colsfm_runs/kitti_raco_caspar/cusfm \
    --min-inter-frame-distance 0.5 --loop-closure --match-cap-mode fixed \
    --features-backend raco --matching-backend raco --ba-backend caspar
```

Stage times from each run's `runtime.csv`, counts from `summary.json`, ATE from one
`evaluate_kitti` invocation over all six rows (`data/bench/kitti_06_raco_caspar_ate.json`),
the blob head-to-head in `data/bench/kitti_06_raco_caspar_compare.md`. The CASPAR row
includes the default Ceres polish (9.7 s, 20 iterations, reprojection 0.599 to 0.582 px).

| Run | extraction (s) | matching (s) | loop stage (s) | mapping (s) | total (s) | points | reprojection (px) | Sim(3) RMSE (m) | SE(3) RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blob cuSFM, SLAM init | 96.85 | 32.61 | 1306.76 | 143.14 | 1596.08 | 120 097 | 0.454 | 1.328 | 1.605 |
| colsfm TensorRT + loops | 62.2 | 25.2 | 62.3 | 209.9 | 363.5 | 174 319 | 0.534 | 0.736 | 1.111 |
| colsfm pycolmap + loops, CASPAR + polish | 16.4 | 74.7 | 77.6 | 41.3 | 212.9 | 96 691 | 0.674 | 0.904 | 1.128 |
| colsfm RaCo native + loops, Ceres | 10.1 | 21.8 | 50.8 | 117.6 | 203.2 | 133 784 | 0.580 | 0.878 | 1.423 |
| **colsfm RaCo native + loops, CASPAR + polish** | **9.7** | **20.2** | **47.0** | **63.0** | **142.8** | 134 884 | 0.582 | 0.937 | 1.461 |

**Fastest run of the whole study.** 142.8 s is 8.9 % of the blob's 1596.08 s and 0.70x
the previous best (203.2 s). Mapping fell from 117.6 s to 63.0 s; the other stages are
unchanged to within run-to-run noise (extraction 10.1 vs 9.7 s, matching 21.8 vs 20.2 s).
The mapping stage is now the same size as the loop stage, and the sum of the three
GPU-bound stages (extraction, matching, loop association) at 77 s is larger than mapping.

**The CASPAR mapping is 1.87x faster here, not the 3.6x of section 8's pycolmap run.**
The pycolmap run's 148.5 s to 41.3 s had 96 691 points; RaCo's map has 134 884 points
and 1 085 213 observations (1.5x), and the 9.7 s polish is a fixed Ceres cost on top.
CASPAR's own share of the 63.0 s is therefore about 53 s, or 2.2x over Ceres on the same
map.

**Accuracy cost: 0.878 to 0.937 m Sim(3), 1.423 to 1.461 m SE(3).** That is the same
direction as section 8's CASPAR + polish result on the pycolmap backend (0.895 to 0.904 m),
but a larger step (+0.059 m against +0.009 m). The reconstruction statistics do not show
it: 2156/2156 registered, reprojection 0.582 against 0.580 px, track length 8.05 against
8.18, and the Sim(3) scale is identical to five digits (0.99180). The difference is in
the trajectory shape after loop closure, where fp32 CASPAR plus one Ceres round lands in a
slightly different basin than five rounds of Ceres. Against the blob the run still passes
three of four acceptance bounds and fails the same one every `colsfm` KITTI run fails,
reprojection error (0.582 against the 0.500 ceiling that 1.1x the blob's 0.454 sets).

**Where this leaves the backends on KITTI 06.** For wall clock, RaCo native + CASPAR at
142.8 s and 0.937 m. For accuracy, TensorRT (the blob's own ALIKED descriptors) + Ceres
at 363.5 s and 0.736 m. RaCo native + Ceres (203.2 s, 0.878 m) is the middle point. The
RaCo descriptor gap to the blob's ALIKED (0.878/0.937 against 0.736) remains the open
accuracy question; CASPAR does not change it.

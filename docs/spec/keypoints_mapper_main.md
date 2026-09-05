# `keypoints_mapper_main` — triangulation and global bundle adjustment

Implementation spec for the cuSFM mapping stage and its Python + pycolmap replacement.
Stage 7 of `docs/open-pipeline-plan.md`. `bundle_adjustment_runner` is covered in
[§10](#10-bundle_adjustment_runner).

Evidence classes used throughout: **CONFIRMED** (read off a real artifact, a run log, the
extracted protobuf schema, or a symbol/decompilation), **INFERRED** (deduced, consistent with
evidence but not directly observed), **UNKNOWN**.

---

## 1. Purpose and pipeline position

`keypoints_mapper_main` is the only stage that changes the trajectory in the default Isaac
profile. It takes per-pair feature matches and an initial pose per keyframe, builds
multi-view correspondence tracks, triangulates them, and then runs an alternating
triangulation / bundle-adjustment / outlier-removal loop. It writes the map
(`map_keypoints.pb`) and an updated keyframe metadata file with the optimised poses.

```
feature_matcher_main            pose_graph_main
   matches/*.pb                 pose_graph/frames_meta.json
            \                    /
             v                  v
        keypoints_mapper_main  --raw_data_dir (images; rolling shutter only)
             |
             +--> <output_map_dir>/map_keypoints.pb
             +--> <output_map_dir>/keyframes/frames_meta.json   (optimised poses)
                        |
                        v
              kpmap_to_colmap / extract_pose_from_map_main   (docs/spec/export.md)
```

The binary is invoked twice by `pycusfm/cusfm_runner.py`: once as `map_keypoints()` and,
when extrinsic refinement is requested, again as `refine_extrinsics()` with
`--optimize_extrinsics=true`. The second call is off by default.

**CONFIRMED.** `pycusfm/cusfm_runner.py:768-830` (`map_keypoints`), `:832-880`
(`refine_extrinsics`).

---

## 2. CLI flags used by `cusfm_runner.py`

From `data/cusfm_re/help/keypoints_mapper_main.txt` (source
`tools/visual/cusfm/keypoints_mapper_main.cc`) and the two call sites.

| Flag | Default | Passed by the runner | Meaning |
|---|---|---|---|
| `--mapping_config_file` | `""` | always, `vision_mapping_config.pb.txt` | **Required.** The `VisionMappingConfig` text proto. |
| `--keyframe_dir` | `""` | always | Directory of keyframe `.pb` files (keypoints + descriptors). |
| `--matches_dir` | `""` | always | Directory of `matching_task_*_result.pb` files. |
| `--output_map_dir` | `""` | always | Where `map_keypoints.pb` and `keyframes/frames_meta.json` are written. |
| `--raw_data_dir` | `""` | always | Raw images. Used **only** for rolling shutter and the vehicle trajectory — **not** for point colour (§5.7). |
| `--use_vehicle_trajectory` | `false` | always, as `False` | Use a vehicle trajectory for pose constraints. |
| `--keyframe_metadata_file` | `""` | when `pose_graph/frames_meta.json` exists | Initial poses. Overrides the per-keyframe poses. |
| `--ba_frame_type` | `"config"` | only if set (default `None`, so not passed) | `config` / `camera_frame` / `vehicle_frame` / `vehicle_rig`. `config` takes `ba_frame_type` from the config file. |
| `--optimize_extrinsics` | `false` | only if truthy | Refine `sensor_from_rig`. CLI rejects it unless `ba_frame_type=vehicle_rig`. |
| `--optimize_intrinsics` | `false` | only if truthy | Refine camera intrinsics. |
| `--do_rolling_shutter_correction` | `false` | only if truthy | Per-row pose interpolation. |
| `--fixed_camera_name` | `"front_stereo_camera_left"` | never | Camera whose extrinsics stay fixed when refining extrinsics. |
| `--fixed_track_name` | `""` | AV data only (`--anchor_track`) | Track folder(s) held fixed. |
| `--previous_cusfm_ws` | `""` | patch mode only | Previously built map. |
| `--debug_dir` | `""` | `--enable_debug` only | Debug dumps. |
| `--frame_to_pixel_per_row_dir` | `""` | never | Rolling-shutter row-time tables. |

Runner defaults (`pycusfm/cusfm_runner.py:75-95`): `optimize_extrinsics=False`,
`optimize_intrinsics=False`, `ba_frame_type=None`, `use_vehicle_trajectory=False`,
`anchor_track=""`. So the default Galileo invocation is:

```
keypoints_mapper_main \
  --mapping_config_file .../isaac/vision_mapping_config.pb.txt \
  --keyframe_dir <ws>/keyframes --matches_dir <ws>/matches \
  --output_map_dir <ws>/kpmap --raw_data_dir <input> \
  --use_vehicle_trajectory=False \
  --keyframe_metadata_file <ws>/pose_graph/frames_meta.json
```

**CONFIRMED.**

---

## 3. Config fields consumed

The schema is **not** in the vendored `data/cusfm_schema/cusfm_protos.fdset`. It was
recovered from the binary's embedded descriptors into
`data/cusfm_re/schema/extracted.fdset` and rendered at
`data/cusfm_re/schema/config_protos.txt` (extractor:
`data/cusfm_re/schema/extract_desc.py`). `protos/visual/cusfm/vision_mapping_config.proto`
and `protos/visual/geometry/bundle_adjustment_config.proto` are proto3.

### 3.1 `VisionMappingConfig` (isaac profile)

Values from `pycusfm/configs/isaac/vision_mapping_config.pb.txt`; the AV profile differs
where noted (`pycusfm/configs/av/vision_mapping_config.pb.txt`).

| Field | isaac | AV | Role | Confidence |
|---|---|---|---|---|
| `min_num_matches_per_pair` | 15 | 15 | Drop an image pair with fewer matches. Rejected 4 pairs in the shipped run. | CONFIRMED |
| `depth_threshold` | 300 | 300 | **World-frame Z cap**, one axis, one-sided, metres. Not a camera depth. | CONFIRMED |
| `search_depth` | 15 | 15 | Max **BFS hop count** when growing one track over the match graph. 2557 match sets built. | CONFIRMED |
| `min_correspondences` | 3 | 3 | Minimum track length. Two-view tracks are dropped. | CONFIRMED |
| `initial_max_pixel_error` | 25.0 | 40.0 | Start of the **linear** per-round gate schedule (§5.6). | CONFIRMED |
| `final_max_pixel_error` | 5.0 | 8.0 | End of the linear gate schedule: 25, 20, 15, 10, 5. | CONFIRMED |
| `min_triangulation_deg` | 2.0 | 1.0 | Min triangulation angle. **All pairs** inside RANSAC; **max over pairs** in the post-BA filter (§5.5). | CONFIRMED |
| `max_ransac_iteration` | 1000 | 1000 | RANSAC trial cap. The **only** `RansacConfig` field taken from config; the rest are hard-coded (§5.3). | CONFIRMED |
| `min_num_samples` | 3 | 3 | RANSAC minimal sample size = **3 views**. | CONFIRMED |
| `iterative_triangulation_times` | 10 | 10 | Max inner triangulation passes; exits early when a pass adds 0 points (1..8 observed). | CONFIRMED |
| `maximum_cache_frame_size` | 10000 | 10000 | Keyframe LRU cache size. Performance only. | INFERRED |
| `num_ba_iterations` | 5 | 5 | Outer triangulate/BA/filter rounds. Logged `Bundle Adjustment Iterative number: 5`. | CONFIRMED |
| `max_observation_change` | 0.0005 | 0.0005 | Early-exit threshold on the outer loop. | CONFIRMED |
| `num_threads` | 8 | 12 | Ceres and OpenMP triangulation threads. **Must be 1 for deterministic triangulation** — a re-run at 8 gave 1281 points against the shipped 1275. | CONFIRMED |
| `random_seed` | 1 | 1 | Seeds the **global libc `srand`**. Does not make multi-threaded triangulation deterministic. | CONFIRMED |
| `triangulation_method` | *unset* -> `NORMALIZED_COORDINATES` | `MIDPOINT` | See §5. | CONFIRMED |
| `pose_constrains_type` | `AUTO` | `POSITION_CONSTRAINS` | See §7. | CONFIRMED |
| `use_camera_extrinsic_constraint` | true | true | Extrinsic prior term. Inactive here — see §7. | CONFIRMED |
| `use_relative_pose_constraint` | true | true | Relative pose prior. Inactive here — see §7. | CONFIRMED |
| `use_absolute_position_constraints` | *unset* -> false | true | Absolute position prior. | CONFIRMED |
| `use_absolute_pose_constraints` | *unset* -> false | true | Absolute pose prior. | CONFIRMED |
| `extrinsic_max_timestamp_diff_microseconds` | 100 | 1000 | Max timestamp skew for two images to count as one rig frame. | CONFIRMED |
| `relative_pose_measurement_min_timestamp_gap_us` | 100000 | 100000 | Minimum gap between frames joined by a relative constraint (0.1 s > 3 frames at 30 Hz). | CONFIRMED |
| `localization_mode` | `FIX` | `FIX` | Prior-map handling. `NONE/FIX/ADJUST/PRE_BUILD`. | CONFIRMED |
| `refine_ba` | false | **true** | Enable the second, stringent BA stage. See §8. | CONFIRMED |
| `image_pair_min_shared_points` | 1 | 1 | Min shared points for a pair to enter the refine stage. | INFERRED |
| `refine_ba_use_fixed_max_pixel_error` | true | true | Refine stage uses a fixed, not decayed, pixel gate. | CONFIRMED |
| `refine_ba_avoid_loss_function` | true | true | Refine stage drops the robust loss. | CONFIRMED |
| `fixed_camera_name` | *from flag* | — | Camera with fixed extrinsics. | CONFIRMED |
| `fixed_track_name` | *from flag* | — | Fixed track(s). | CONFIRMED |
| `keypoint_creation_config` | unset | unset | Only used when the mapper re-detects. | UNKNOWN |

Fields present in the schema but unset in both profiles: `alignment_id`, `track_ids`,
`keyframe_dir`, `raw_data_dir`, `matches_dir`, `output_map_dir`, `debug_dir`,
`keyframe_metadata_file`, `previous_cusfm_ws` — all overridden by the CLI flags.

### 3.2 `BundleAdjustmentConfig` (isaac profile)

| Field | isaac | AV | Role | Confidence |
|---|---|---|---|---|
| `loss_type` | `CAUCHY` | `CAUCHY` | `TRIVIAL=0, SOFT_L1=1, CAUCHY=2, HUBER=3`. | CONFIRMED |
| `loss_function_scale` | 1.0 | 1.0 | Loss scale, applied to the **whitened** residual. See §6.4. | CONFIRMED |
| `minimizer_progress_to_stdout` | true | true | Ceres per-iteration table. | CONFIRMED |
| `max_num_iterations` | 200 | 200 | Ceres cap. Round 0 used 20. | CONFIRMED |
| `max_invalid_steps` | 10 | 10 | Ceres `max_num_consecutive_invalid_steps`. | CONFIRMED |
| `max_linear_solver_iterations` | 100 | 100 | Ceres linear solver cap. Irrelevant for a direct sparse solver. | CONFIRMED |
| `num_threads` | 8 | 12 | Given and used, per the Ceres summary. | CONFIRMED |
| `reprojection_error_standard_deviation` | 4.0 | 4.0 | Pixel sigma; whitens the reprojection residual. See §6.4. | CONFIRMED |
| `absolute_position_constraint_error_meters` | 1 | 5 | Sigma of the absolute position prior. | CONFIRMED |
| `absolute_pose_translation_error_meters` | 1 | 1 | Sigma of the absolute pose prior, translation. | CONFIRMED |
| `absolute_pose_rotation_error_degrees` | 5 | 5 | Sigma of the absolute pose prior, rotation. | CONFIRMED |
| `relative_pose_translation_error_meters` | 0.1 | 1 | Sigma of the relative pose prior, translation. | CONFIRMED |
| `relative_pose_rotation_error_degrees` | 5 | 10 | Sigma of the relative pose prior, rotation. | CONFIRMED |
| `relative_constraint_use_pose_covariance` | false | false | Use `pose_covariance` from the metadata instead of the fixed sigmas. | CONFIRMED |
| `extrinsic_error_meters` | 0.01 | 0.05 | Sigma of the extrinsic prior, translation. | CONFIRMED |
| `extrinsic_error_degrees` | 2 | 2 | Sigma of the extrinsic prior, rotation. | CONFIRMED |
| `use_cudss_solver` | true | true | Use the cuDSS GPU sparse Cholesky. | CONFIRMED |
| `do_rolling_shutter_correction` | false | **true** | Per-row pose interpolation in the residual. | CONFIRMED |
| `ba_frame_type` | **`VEHICLE_RIG`** | `VEHICLE_FRAME` | `CAMERA_FRAME=0, VEHICLE_FRAME=1, VEHICLE_RIG=2`. See §6.3. | CONFIRMED |
| `verbose` | true | true | Extra `[BA COST]` group reporting. | CONFIRMED |

Schema fields unset in both profiles: `refine_extrinsic_params` (19), `optimize_intrinsics`
(25), `reprojection_error_standard_deviation_3d` (26), `use_fast_reproject_cost` (27). The
first two are shadowed by the CLI flags.

---

## 4. Input and output files

### 4.1 Inputs

| Path | Format | Notes |
|---|---|---|
| `--keyframe_dir/**/*.pb` | `protos.visual.general.Keyframe` | Keypoints as a `KeypointVector` struct-of-arrays: parallel `x`, `y` (float, pixels), packed `descriptors`, optional `depth`, optional `r`/`g`/`b`, plus `image_width`/`image_height`. |
| `--keyframe_dir/frames_meta.json` or `--keyframe_metadata_file` | proto3 JSON of `KeyframesMetadataCollection` | Poses, camera calibration, rig extrinsics, stereo pairs. |
| `--matches_dir/matching_task_*_result.pb` | `protos.visual.cusfm.FrameMatches` | Per-pair `FeatureMatch` lists. |
| `--matches_dir/map_keypoints.pb` | `MapKeyPoints` | Optional; patch mode. Absent in the shipped run (`Map keypoints file doesn't exist`). |
| `--raw_data_dir/<image_name>` | JPEG | Rolling shutter only. **Never read for point colour** (§5.7). |

### 4.2 `KeyframesMetadataCollection` — the fields that matter

`protos/visual/general/keyframe_metadata.proto`, rendered in
`data/cusfm_re/schema/data_protos.txt`.

`KeyframeMetaData`:

| Field | # | Units / meaning |
|---|---|---|
| `id` | 1 | Keyframe id. Referenced by `MapKeyPoints` observations. |
| `track_id` | 3 | Track (session segment) id. |
| `camera_params_id` | 6 | **Sensor id.** Indexes `camera_params_id_to_camera_params`. This is the rig sensor. |
| `timestamp_microseconds` | 7 | Microseconds. |
| `image_name` | 8 | `"<camera_name>/<timestamp_ns>.jpeg"`, stored, relative to `--raw_data_dir`. |
| `camera_to_world` | 9 | `RigidTransform3d` = **`world_T_cam`**. |
| `pose_covariance` | 10 | `MatrixD`, optional; used when `relative_constraint_use_pose_covariance`. |
| `synced_sample_id` | 12 | **Rig frame id.** All images captured at one rig instant share it. |
| `start_camera_to_world` | 15 | Pose at the first row; written when rolling shutter correction is on. |

`RigidTransform3d` = `AxisAngleDegree axis_angle` + `Translation3d translation`.
`AxisAngleDegree` = `(x, y, z, angle_degrees)`: a rotation axis (normalised in practice) and
an angle **in degrees**. Translation is in metres.

Camera calibration lives in `camera_params_id_to_camera_params[<id>]`, a
`protos.common.sensor.CameraSensor`, whose `sensor_meta_data.sensor_to_vehicle_transform` is
the **`vehicle_T_cam` rig extrinsic** and whose `calibration_parameters` is a
`MonoCalibrationParameters`. `camera_projection_model_type` is one of exactly four values:
`PINHOLE=0, DISTORTED_PINHOLE=1, FTHETA_WINDSHIELD=2, OPENCV_FISHEYE=3`.

**CONFIRMED**, schema plus `data/cusfm_runs/galileo/kpmap/keyframes/frames_meta.json`.

### 4.3 Conventions

- Camera frame: **RDF** (x right, y down, z forward), OpenCV.
- Vehicle frame: **FLU** (x forward, y left, z up).
- World frame: the vehicle frame of the first frame.
- `camera_to_world` is `world_T_cam`; `sensor_to_vehicle_transform` is `vehicle_T_cam`.
- Rotations in the metadata are axis-angle in **degrees**; COLMAP wants quaternions and the
  **inverse** (`cam_T_world`). See `docs/spec/export.md`.

**CONFIRMED**, `docs/camera.md` §Coordinate Systems.

### 4.4 Outputs

`<output_map_dir>/map_keypoints.pb`, a `protos.visual.cusfm.MapKeyPoints`, is a
struct-of-arrays:

```proto
message MapKeyPoints {
  repeated float points_x = 1;              // world coordinates, metres
  repeated float points_y = 2;
  repeated float points_z = 3;
  repeated int32 points_num_observations = 4;
  repeated PointObservations observations = 5;   // per point: list of (keyframe_id, keypoint_id)
  repeated protos.visual.general.Keyframe frames = 6;
  repeated uint32 points_r = 7;             // 0..255
  repeated uint32 points_g = 8;
  repeated uint32 points_b = 9;
}
```

There is **no per-point error field**, so the COLMAP `points3D.txt` `ERROR` column must be
recomputed or defaulted at export time.

`<output_map_dir>/keyframes/frames_meta.json` is the input collection with
`camera_to_world` replaced by the optimised poses. Measured against the input
(`data/cusfm_re/notes/rig_check_output.txt`): the camera calibration and the rig extrinsics
are numerically unchanged (translation delta 0.000 um, rotation delta <= 1.2 microdegrees,
which is axis-angle JSON rounding), and `initial_pose_type` changes `EGO_MOTION` ->
`ALIGNMENT`.

**CONFIRMED.**

---

## 5. Algorithm, step by step

The run log `data/cusfm_runs/galileo/kpmap/keypoints_mapper_main.txt` prints the whole
control flow with source file and line. The shipped Galileo artifact is a **small** run —
32 keyframes, 8 cameras, 4 rig frames, 1275 exported points — not the 224-image acceptance
run in `docs/open-pipeline-plan.md`. Absolute counts below are from that small run.

### 5.1 Ingestion and track building

1. Read the config, the keyframe metadata, and every `*.pb` in `--matches_dir`
   (`keypoints_mapper.cc:1014`).
2. Drop image pairs with fewer than `min_num_matches_per_pair` matches, **before any edge is
   added**: `Filter image pairs: by min matches=15: 4` (`keypoints_mapper.cc:1081`) — 4 of
   40 pairs in Galileo.
3. Build a `KeypointMatchGraph`: vertices are `(frame_id, keypoint_id)`, one edge per
   `FeatureMatch`.
4. Build **match sets** (tracks). `GetNeighbors(v, depth, out)` is a **level-synchronous
   BFS**, `for (level = 0; level <= search_depth; ++level)` with a `used` bitset — so
   `search_depth: 15` is the **maximum hop count when growing one track**, not a transitivity
   count. `GenerateMatches` seeds greedily, rejects components smaller than
   `min_correspondences: 3`, then calls `MarkAsUsed`, so **tracks are disjoint**.
   `Generate match sets with search_depth=15: 2557` (`keypoints_mapper.cc:1102`).
5. Track length distribution (`keypoints_mapper.cc:1128`): `Min:3 Max:13 Median:4 90%:7 99%:9`.

The builder has **no explicit same-frame conflict check**, but the shipped map has 0 of 1275
tracks with a duplicate keyframe — the disjoint-component construction makes conflicts rare
rather than impossible.

The paper calls this a depth-first search; the decompilation shows a breadth-first one. The
bound is what matters.

**CONFIRMED** (decompilation + log).

### 5.2 Rig frame construction (VEHICLE_RIG only)

Keyframes are grouped into vehicle/rig frames. The grouping key is `synced_sample_id`;
`extrinsic_max_timestamp_diff_microseconds` (100 us) bounds the timestamp skew inside a
group. The log prints `FRAME MODE: VEHICLE_RIG` (`keypoints_mapper.cc:624`) and
`vehicle frame size: 4` (`:661`).

Verified numerically in `data/cusfm_re/notes/rig_check.py`: for all 4 groups of 8 images,
`world_T_cam * (vehicle_T_cam)^-1` is identical across the 8 cameras to
**0.00000 mm and 2e-6 deg**, both before and after BA. The rig constraint is exact, not
approximate.

**CONFIRMED.**

### 5.3 Triangulation

Config: `triangulation_method` unset in the isaac profile, so
`NORMALIZED_COORDINATES` (enum value 0). The AV profile uses `MIDPOINT`. The enum is
`NORMALIZED_COORDINATES=0, PIXEL_COORDINATES=1, MIDPOINT=2, NORMALIZED_COORDINATES_WITH_DEPTH=3`.

The paper describes a DLT: minimise `sum_j || P_j X - lambda_j p_j ||^2`, which "can operate
in both image and normalized camera coordinates" — that is exactly the
`NORMALIZED_COORDINATES` / `PIXEL_COORDINATES` pair — plus a separate midpoint method.

The initial triangulator logs its effective settings (`iterative_triangulator.cc:42`):

```
    min_triangulation_deg: 2
    triangulation_method: NORMALIZED_COORDINATES
```

and dispatches to `MultiViewsTriangulation<double>` (`triangulation_ransac.cc:247`), i.e. all
sampled rays are solved at once, not pairwise.

**The estimator is COLMAP's `TriangulateMultiViewPoint` verbatim.** `MultiViewsTriangulation`
builds the 4x4 normal matrix and solves it with
`Eigen::internal::tridiagonalization_inplace<Matrix4d>` + `tridiagonal_qr_step`, i.e. a
`SelfAdjointEigenSolver<Matrix4d>` — exactly COLMAP's formulation. The two-view path uses
`real_2x2_jacobi_svd` on a 4x4, which is COLMAP's `TriangulatePoint`. **No Ceres is involved
in triangulation.**

#### RANSAC

`RansacConfig` is **not** in the mapping config; only `max_iterations` comes from
`max_ransac_iteration: 1000`. The rest are **hard-coded in the binary**:

| `RansacConfig` field | Value | Source |
|---|---|---|
| `max_iterations` | 1000 | config `max_ransac_iteration` |
| `desired_success_probability` | **0.999** | hard-coded |
| `min_success_probability` | **0.99** | hard-coded |
| `min_inlier_percentage` | **0.0** (disabled) | hard-coded |
| `max_hypothesis_generation_attempts` | **10** | hard-coded |
| `max_inlier_threshold` | derived, see below | per call |

- Sample size is **3 views** (`min_num_samples: 3`).
- Scoring is **truncated L1 (MSAC)**, not an inlier count.
- Adaptive stop: `n = ceil( log(1 - 0.999) / log(1 - w^3) )`.
- **Not LO-RANSAC, and there is no final refit on the inlier set.**
- Inlier threshold: for `NORMALIZED_COORDINATES` the residual is in normalised camera
  coordinates, so the worker sets
  `max_inlier_threshold = max_pixel_error / projector[0]->focal_length()`. The effective gate
  is `max_pixel_error` **pixels** either way.
- Every pair among the 3 sampled views must satisfy `min_triangulation_deg` —
  `CheckTriangulationAngle` returns false on the **first** failing pair.

**Determinism warning.** `random_seed: 1` drives the global libc `srand`, and
`TriangulatePoints` is a `GOMP_parallel` region with `num_threads: 8`. A re-run on identical
inputs produced **1281** points against the shipped **1275**. The config comment
("use num_threads=1 to get deterministic triangulation results") is accurate and necessary.

#### The inner loop

The `Iterative number: 0, 2, 4` sequence is a **logging artefact** — `LOG_EVERY_N(INFO, 2)`.
The inner loop actually runs 1..8 passes out of a maximum
`iterative_triangulation_times: 10` and exits early when a pass adds zero points.

Result of the initial pass (`keypoints_mapper.cc:341-343`):

```
Triangulation Finished
Number point3D: 1435 Valid 3d points: 1435 Invalid 3d points: 0
Re-projection Error: 3.80096 Mean length: 4.5338
```

**CONFIRMED** (decompilation, `objdump`, and a `--v=3` re-run at
`data/cusfm_re/runs/tri_v3/v3.log`).

### 5.4 The outer loop

`keypoints_mapper.cc:361-435`:

```
Start iterative Bundle Adjustment...
Bundle Adjustment Iterative number: 5
use num_ba_iterations to stop BA
BA iterative: 0 .. 4
```

Each round performs, in order:

1. **Merge** — `[TryMergeCorrespondence] Merged 8 correspondences, 4 points.`
   (`iterative_triangulator.cc:598`). Merges tracks that resolve to the same 3D point.
2. **Complete** — extend tracks through transitive matches.
3. **Filter** — `[CompleteAndFilterCorrespondences] Deleted points by
   min_correspondences=3: 95, by small_triangulation_angle=2: 0`
   (`iterative_triangulator.cc:546`) and `Deleted correspondences: 230, completed
   correspondences: 0` (`:553`). Individual drops are logged as
   `Point 1403 has 2 correspondences, min required: 3` (`:528`).
4. **Bundle adjustment** — §6.
5. **Report** — `Number point3D: 1336 ... Re-projection Error: 2.05731 Mean length: 4.6976
   Max error: 14.3158 Min error: 0.0763973` (`keypoints_mapper.cc:435`).

The `Merge` / `Complete` / `Filter` triple is the same design as COLMAP's
`IncrementalTriangulator::MergeAllTracks` / `CompleteAllTracks` and
`ObservationManager::FilterAllPoints3D`.

Convergence bookkeeping (`keypoints_mapper.cc:407-415`):

```
Num total correspondences: 10979, num observed correspondences: 6276, percent: 57%
Num merged correspondences: 8, num completed correspondences: 0,
  num filtered correspondences: 230, observation change: 0.0379222
```

The convergence statistic is exactly

```
observation_change = (num_merged + num_completed + num_filtered) / num_observed_correspondences
```

verified against all five rounds of the shipped log:

| Round | merged | completed | filtered | sum | observed | computed | logged |
|---|---|---|---|---|---|---|---|
| 0 | 8 | 0 | 230 | 238 | 6276 | 0.03792224 | 0.03792220 |
| 1 | 0 | 16 | 24 | 40 | 6268 | 0.00638162 | 0.00638162 |
| 2 | 0 | 1 | 15 | 16 | 6254 | 0.00255836 | 0.00255836 |
| 3 | 0 | 0 | 61 | 61 | 6193 | 0.00984983 | 0.00984983 |
| 4 | 0 | 0 | 358 | 358 | 5835 | 0.06135390 | 0.06135390 |

It is compared against `max_observation_change: 0.0005`. Note it is **not monotonic** — it
rises again in rounds 3 and 4 as the pixel gate tightens to 10 and 5 px — so it is a
"nothing changed this round" test, not a convergence measure. In this run it never reached
the threshold and the loop ran the full 5 rounds
(`use num_ba_iterations to stop BA`, `BA iteration number reaches max number: 5`).

Trajectory over the 5 rounds: 1435 -> 1336 -> 1335 -> ... -> 1275 points; mean reprojection
error 3.80 -> 2.06 -> 2.02 -> 1.93 -> 1.70 px; mean track length 4.53 -> 4.70.

**CONFIRMED.**

### 5.5 Point filters

| Filter | Threshold | Semantics | Effect in the shipped run |
|---|---|---|---|
| Track length | `min_correspondences: 3` | Minimum observations per point | Dominant: 95 points in round 0, then 1, 2, ... |
| Reprojection error | the decaying gate, §5.6 | Per observation, in pixels | Drives the correspondence deletions (230, 24, 15, ...) |
| Triangulation angle | `min_triangulation_deg: 2.0` | **max over all observing pairs** (`HasLargeEnoughTriangulateAngle`) | **0 points in almost every round** — effectively inert |
| Depth | `depth_threshold: 300` | **world-frame Z cap, one axis, one-sided, metres** | Not exercised |

Two details that matter for a port:

- **The angle test is asymmetric.** Inside RANSAC, *every* pair among the 3 sampled views must
  clear 2 deg. In the post-BA filter, only the **maximum** over all observing pairs must clear
  it. That asymmetry is why the post-BA angle filter deletes nothing while triangulation still
  respects the threshold — `min_correspondences` does all the visible work.
  `GetTriangulationAngleRad` is COLMAP's `CalculateTriangulationAngle` verbatim.
- **`depth_threshold` is not a camera depth.** Disassembly at `0x1c9702` compares the
  triangulated point's **world** `z()` (the `Eigen::Vector3d&` output of
  `TriangulateWithRansac`, `lea -0x110(%rbp),%r9`) against `(double)opt.depth_threshold` and
  rejects when greater. One-sided, one axis, in metres. A port that implements it as a
  camera-frame depth check will diverge.

**CONFIRMED** (decompilation + disassembly + log).

### 5.6 The pixel-error schedule — linear across the outer loop

`--decay_max_pixel_error` defaults true. The decay lives in the **outer** loop, not the
triangulator's inner loop, and it is **linear**:

```
max_pixel_error(k) = initial + (final - initial) * k / (num_ba_iterations - 1),  k = 0..4
```

A `--v=3` re-run prints `Set triangulator max_pixel_error=` **25, 20, 15, 10, 5** across the
five BA rounds — exactly the linear interpolation from `initial_max_pixel_error: 25.0` to
`final_max_pixel_error: 5.0`. The AV profile would give 40, 32, 24, 16, 8.

**CONFIRMED** (`data/cusfm_re/runs/tri_v3/v3.log`).

---

### 5.7 Point colour — the mapper never assigns it

`MapKeyPoints` has `points_r/g/b`, and `kpmap_to_colmap` reads them, but **the mapper does not
fill them from imagery**. The binary has no `Rgb8`, `CvtColor` or `imread` symbols;
`--raw_data_dir` is used only for rolling shutter and the vehicle trajectory. Re-running the
mapper **with** `--raw_data_dir` still produced all-zero colours.

Colour is meant to flow from `KeypointVector.r/g/b`, which the **feature extractor** sets per
keypoint. All 32 Galileo keyframes have zero `r/g/b` (though non-zero `depth`), so the whole
chain yields black points and `kpmap_to_colmap` logs
`No RGB colors found in map, using default (0,0,0)`.

**Consequence for the Python port:** to get coloured points, sample the image at each keypoint
in the **extractor** stage and populate `KeypointVector.r/g/b`. Doing it in the mapper would
not match cuSFM, and matching cuSFM here means shipping black points.

**CONFIRMED.**

---

## 6. Bundle adjustment

### 6.1 Cost functors

16 BA cost functors exist. All are `ceres::AutoDiffCostFunction` except one. The reprojection
residual is always **2-D**; pose is always **two** parameter blocks, a quaternion `q(4)` and a
translation `t(3)` — never a single 6- or 7-vector.

| Functor (template args) | Parameter blocks | Cost group |
|---|---|---|
| `OptimizePointFixedPoseReprojectionCost<2,3>` | point only (pose baked in) | `reprojection_constant_frame` |
| `OptimizeRigPoseAndPointReprojectionCost<2,4,3,4,3,3>` | `veh_q, veh_t, extr_q, extr_t, point` | `reprojection_vehicle_frame` |
| `OptimizePoseAndPointReprojectionCost<2,4,3,3>` | `cam_q, cam_t, point` | `reprojection_camera_frame` |
| `OptimizeRigPoseFixedPointReprojectionCost<2,4,3,4,3>` | rig pose + extrinsic, point constant | — |
| `NormalizedReprojectionCost<2,4,3,4,3,3>` | as rig, residual in normalised rays | `use_fast_reproject_cost` |
| `OptimizeIntrinsicsFixedPointPoseReprojectionCost<2,N>`, `N in {8,9,12,17,18}` | intrinsics only | — |
| `OptimizePointAndIntrinsicsFixedPose...<2,3,N>` | point + intrinsics | — |
| `OptimizePoseRelativeConstraintCost<6,4,3,4,3>` | two poses | relative prior |
| `OptimizePoseAbsoluteConstraintCost<6,4,3>` | one pose | absolute pose prior |
| `OptimizePoseAbsolutePositionConstraintCost<3,4,3>` | one pose | absolute position prior |
| `VehicleCameraReprojectionCost3D` — `NumericDiffCostFunction<CENTRAL,3,4,3,4,3,3>` | rig pose + extrinsic + point | `reprojection_vehicle_frame_3d` |

Rolling-shutter twins exist for each reprojection variant (`..._rs` groups).

`N in {9, 12, 18}` are the distorted-pinhole intrinsic counts for BASIC / RATIONAL /
COMPLETE; `N = 17` is F-Theta. The single `NumericDiff` functor is a **3-D residual** for
keypoints that carry depth, weighted by `reprojection_error_standard_deviation_3d`.

**CONFIRMED** (symbol table + decompiled dispatch).

### 6.2 Parameterisation — derived exactly from the Ceres summary

Full derivation in `data/cusfm_re/notes/ba_parameterisation_from_ceres_summary.md`. The Ceres
header for round 0 of the shipped run is:

```
Parameter blocks    1360   1342        Residual blocks   6276   6276
Parameters          4092   4029        Residuals        12552  12552
Effective params    4080   4026
```

A single model reproduces **all six** counters exactly, in two independent rounds:

- Each **3D point** is one block of **3** parameters, variable. `P = 1336` in round 0.
- Each **pose entity** is **two** blocks: a **quaternion** (4 parameters, **3 effective**
  under a manifold) and a **translation** (3 parameters).
- There are **12 pose entities** = **4 rig poses** + **8 `sensor_from_rig` extrinsics**. The
  extrinsics come from the `CameraParam` record keyed by `camera_params_id`, so there is
  **one shared q/t pair per camera across all frames** — a true rig.
- **9** entities are constant and vanish from the `Reduced` column: the 8 extrinsics
  (`--optimize_extrinsics=false`) plus **1 fixed rig pose**.
- There are **no intrinsics blocks** (`--optimize_intrinsics=false`).

Check: `1336 + 12*2 = 1360`; `3*1336 + 12*7 = 4092`; `3*1336 + 12*6 = 4080`; subtracting
`9*2`, `9*7`, `9*6` gives `1342`, `4029`, `4026`. Round 1 with `P = 1335` gives
`1359 / 4089 / 4077 / 1341 / 4026 / 4023`. Every number matches. This is exactly the block
layout of `OptimizeRigPoseAndPointReprojectionCost<2,4,3,4,3,3>`.

The residual under `ba_frame_type: VEHICLE_RIG` is

```
r = pi_c( K_c * sensor_from_rig[c] * rig_from_world[f] * X_p ) - x_obs        (2-vector, px)
```

matching the paper: "Each camera's pose is decomposed into two components: the vehicle pose in
the world coordinate frame `T_v^w` and the camera-to-vehicle extrinsic transformation
`T_c^v`".

**Rotation manifold.** The only manifold vtable in either binary is
`ceres::EigenQuaternionManifold`, so quaternion blocks are in **Eigen order `[x, y, z, w]`**,
not Ceres' `[w, x, y, z]`. Quaternions are normalised in place before each residual is added.

**Pose storage.** `SE3Transform<double>{q @ +0x20, t @ +0x40}`, holding **`world_T_cam`** (the
proto field is `camera_to_world`). `ColmapManager` inverts on export.

**CONFIRMED.**

### 6.3 Which blocks are constant

There are exactly **two** `SetParameterBlockConstant` call sites:

1. **`AddConstantFrameConstraints`.** For every id in `data.constant_keyframes_`: if that
   camera frame has a vehicle frame, it fixes the **vehicle** frame's `q` and `t`; otherwise
   the **camera** frame's. Ends with `"Added N constant frame constraints to problem"`.
2. **`SetCameraParametersManifold`.** Fixes **both** extrinsic blocks when
   `!refine_extrinsic_params`.

| `ba_frame_type` | Optimised pose | Extrinsics |
|---|---|---|
| `CAMERA_FRAME` (0) | one `cam_from_world` per image | not a parameter |
| `VEHICLE_FRAME` (1) | vehicle pose per image | shared, fixed |
| `VEHICLE_RIG` (2) | one `rig_from_world` per rig frame (`synced_sample_id`) | shared across all frames, fixed unless `--optimize_extrinsics` |

The mode is chosen **per observation**, by whether the frame has a vehicle frame, not by the
enum directly. `--optimize_extrinsics` is `CHECK`ed to require `VEHICLE_RIG`.

#### The constant frame is one *keyframe*, not one rig frame

`Added 1 constant frame constraints to problem`, and the cost groups split
`13 / 6263` — not `1151 / 4684`, which is what the rig-frame-1 observation count would give.
So `constant_keyframes_` holds a **single camera keyframe**. In Galileo that is
**keyframe 2359** (`left_stereo_camera_left`, rig frame 1, the **lowest id** in the map); its
observation count decays 13, 13, 13, 11, 1 across the five rounds, and the map's final state
gives keyframe 2359 exactly **1** observation — matching the round-4 group size of 1 block.

Because that keyframe **has** a vehicle frame, the constraint fixes **rig frame 1's vehicle
pose**, which through the shared rig fixes all 8 of its cameras. That is why the measured
motion of rig frame 1 is exactly 0.0000 mm / 0.000000 deg while frames 8, 15, 22 move
4.97 / 5.54 / 11.27 mm (`data/cusfm_re/notes/rig_check_output.txt`). The growth with time is
a drift correction anchored at the first frame.

Observations **of** the constant keyframe use `OptimizePointFixedPoseReprojectionCost<2,3>` —
the pose is baked into the functor and only the point is a parameter. They constrain points,
not poses.

`--fixed_camera_name` (default `front_stereo_camera_left`) pins one camera's extrinsics when
extrinsics are refined. `--fix_first_camera` in `bundle_adjustment_runner` is **inert**:
toggling it changed neither the block count nor the constant-frame group.

**CONFIRMED.**

### 6.4 Residual weighting and robust loss

The reprojection cost is

```
C = 0.5 * sum_obs  rho( | w * (pi(...) - x_obs) |^2 ),     w = 1 / sigma,  sigma = 4.0 px
```

- **`weight = sqrt(1/sigma^2) = 1/sigma`**, a plain scalar multiplying the 2-vector. Measured
  cost scaling across `sigma = 1, 2, 4, 8` is exactly `1/sigma^2`. **CONFIRMED**
  (decompilation plus a four-point live sweep).
- Independently corroborated by inversion of the shipped log: `s = exp(2C/n_obs) - 1`,
  `|r| = 4*sqrt(s)` gives an implied RMS pixel error a stable **1.13x** above the mapper's
  logged **mean** in all five rounds (2.314 vs 2.057, 2.292 vs 2.024, 2.271 vs 1.999,
  2.197 vs 1.925, 1.951 vs 1.699). With no whitening the implied error would be 0.58 px
  against a logged 2.06 px — wrong by 3.5x.

> **`reprojection_constant_frame` residuals are NOT weighted.** Their group cost was
> identical at `sigma = 1, 2, 4, 8`, and the functor has no weight member — it stores only
> `{quat(4), trans(3), obs(2), projector}`. Almost certainly an upstream bug, but a faithful
> port must reproduce it: constant-frame observations enter at weight 1.0 while everything
> else is at `1/sigma`. In Galileo this is 13 of 6276 residual blocks, so the practical
> effect is negligible.

Robust loss: `TrivialLoss` / `SoftLOneLoss` / `CauchyLoss` / `HuberLoss(scale)`. **No
`ScaledLoss` anywhere.** One loss instance is created per *constraint builder* and shared by
all of that builder's blocks, so the same loss also wraps the pose, position and extrinsic
constraints — not just reprojection. Isaac uses `CAUCHY` with scale 1.0.

`use_fast_reproject_cost` switches to `NormalizedReprojectionCost`, whose weight is
`sigma / focal_length`.

Pose-prior weighting: `DefaultPoseInformationMatrix(rotation_deg, translation_m)` builds
`diag(1/(deg * pi/180)^2, 1/m^2)` with **rotation in residual slots 0-2**, runs an LDLT
definiteness check, then an **LLT to get `sqrt_information`**; the residual is
`L * [rot_err; trans_err]`.

> **Likely upstream bug.** `AddAbsolutePoseConstraints` passes
> `(translation_m, rotation_deg)` where the helper expects `(rotation_deg, translation_m)`.
> The relative-constraint path passes them in the right order. Irrelevant to Isaac, where the
> absolute path never runs (§7).

The logged `Initial cost` / `Final cost` is **not** the Ceres cost: it is
`sqrt(ceres_total_cost / num_residuals)`, exact to six decimals in all ten values across the
five rounds. `[BA COST] total_cost` is the raw Ceres cost. **CONFIRMED** twice, independently.

### 6.5 Solver settings

Ceres **2.3.0**, built `-cuda-(13000)-cudss-(0.7.1)`, `no_lapack`, **no SuiteSparse**.

Only **7** `Solver::Options` fields are overridden. Everything else is a Ceres default —
notably `function_tolerance = 1e-6`, `gradient_tolerance = 1e-10`,
`parameter_tolerance = 1e-8`, all untouched.

| Setting | Value |
|---|---|
| Minimizer | `TRUST_REGION` |
| Trust region strategy | `LEVENBERG_MARQUARDT` |
| Linear solver, **default** | `SPARSE_SCHUR` |
| Linear solver, `use_cudss_solver: true` | **replaced** by `SPARSE_NORMAL_CHOLESKY` + `CUDA_SPARSE` — no Schur at all on the GPU path |
| Linear solver, CPU fallback | `SPARSE_SCHUR` + `EIGEN_SPARSE` + `AMD` ordering (verified live) |
| Ordering | `AUTOMATIC` |
| Threads | 8 given, 8 used |
| `max_num_iterations` | 200 (round 0 used 20, all successful) |
| `max_num_consecutive_invalid_steps` | 10 |
| Termination | `CONVERGENCE (Function tolerance reached)` |

The mapper logs `Using: CUDSS_SPARSE + SPARSE_NORMAL_CHOLESKY`, and the summary reports
`Sparse linear algebra library CUDA_SPARSE`. The paper agrees: "All evaluated systems utilize
Ceres-Solver with SPARSE_NORMAL_CHOLESKY linear solver type for Bundle Adjustment
optimization."

`config.minimizer_progress_to_stdout` is **dead code** — both the progress table and the
summary are gated on `verbose`.

**CONFIRMED.**

---

## 7. Pose constraints — why they are all inactive in the Isaac profile

This resolves a contradiction in `NOTES.md`, which states that BA "is not unconstrained"
because `use_relative_pose_constraint: true`, and reads the 1 m
`absolute_pose_translation_error_meters` as an active leash on the solution.

The shipped run log says otherwise:

```
keypoints_mapper.cc:1516]         Num vehicle relative pose constraints: 0
bundle_adjustment_solver.cc:731]  Not using absolute pose constraints
bundle_adjustment_solver.cc:840]  Not using position constraints
bundle_adjustment_solver.cc:910]  Not using extrinsic constraints
bundle_adjustment_solver.cc:1268] Not using relative pose constraints
```

and the cost breakdown contains only reprojection groups: `6263 + 13 = 6276`, the total
residual block count. There is no room for a constraint term.

**Mechanism.** The BA solver has **no `use_*_constraint` flags at all**. All four
"Not using ..." lines are pure **emptiness tests** on `BundleAdjustmentData` containers. The
real gate is **`--use_vehicle_trajectory`, default `false`** (and it additionally `CHECK`s
`--raw_data_dir`), so the mapper never populates them. The data agrees:
`frames_meta.json` has `"vehicle_trajectory_files": []`.

**What `pose_constrains_type: AUTO` would resolve to.** A literal jump table at
`.rodata:0x2bf4c0` = `{1, 2, 2, 1, 3}` maps `KeyframesMetadataCollection.initial_pose_type`:

| `initial_pose_type` | resolves to |
|---|---|
| `ALIGNMENT` (0) | `POSITION_CONSTRAINS` |
| `EGO_MOTION` (1) | `RELATIVE_CONSTRAINS` |
| `SESSION_EGO_MOTION` (2) | `RELATIVE_CONSTRAINS` |
| `GPS_IMU` (3) | `POSITION_CONSTRAINS` |
| `MAP_POSE` (4) | `CONSTANT_CONSTRAINS` |

Returning `AUTO` is the failure path (`"Failed to get pose constrains type!"`).

**Which pairs would get a relative constraint.** Per `(track, camera)`, the first forward
frame whose timestamp gap exceeds
`relative_pose_measurement_min_timestamp_gap_us: 100000`, capped at **3 per frame** in the
vehicle variant.

**Practical consequence for the Python port.** On the Isaac/Galileo profile the bundle
adjustment is a **pure reprojection problem with one gauge-fixing constant keyframe**. The
prior sigmas are dead configuration. The `colsfm` implementation needs no pose-prior term to
reproduce Galileo; it does need one for the AV profile, which sets
`pose_constrains_type: POSITION_CONSTRAINS`, `use_absolute_pose_constraints: true` and
`use_absolute_position_constraints: true`.

**CONFIRMED.**

---

## 8. The two-stage design (`refine_ba`)

The paper: "The first stage adopts relaxed outlier thresholds and robust loss functions to
ensure rapid convergence. The second stage applies stringent outlier criteria without loss
functions to achieve high-precision results."

`refine_ba` gates a **second `RunOuterIteration`**:

| Config field | isaac | AV | Effect in stage 2 |
|---|---|---|---|
| `refine_ba` | **false** | true | Run the second stage at all. |
| `refine_ba_use_fixed_max_pixel_error` | true | true | Use a **constant `final_max_pixel_error / 2`** (2.5 px isaac, 4 px AV) instead of a ramp; when false, a **halved decay ramp**. Also forces iteration-count termination, with an explicit warning. |
| `refine_ba_avoid_loss_function` | true | true | Writes `loss_type = 0` (`TRIVIAL`) into the BA config. |
| `image_pair_min_shared_points` | 1 | 1 | Gates `KeepImagePair`, dropping pairs below the threshold before matches are re-loaded on non-first passes. |

**The Isaac/Galileo profile runs stage 1 only** (`refine_ba: false`). A port targeting Galileo
can ignore stage 2; a port targeting AV cannot.

**CONFIRMED.**

---

## 9. pycolmap mapping

Probed against the environment's real build: **pycolmap 4.2.0, `has_cuda=True`**
(`data/cusfm_re/notes/pycolmap_probe.txt`). This version has native `Rig` and `Frame`
objects, which is what makes the `VEHICLE_RIG` mode expressible.

### 9.1 Data model

| cuSFM | pycolmap 4.2.0 |
|---|---|
| `synced_sample_id` (rig frame) | `pycolmap.Frame` — `frame.rig_from_world` |
| `camera_params_id` (sensor) | a sensor in `pycolmap.Rig`; `rig.sensor_from_rig(sensor_id)` |
| `sensor_to_vehicle_transform` = `vehicle_T_cam` | `rig.set_sensor_from_rig(...)` with the **inverse**, `cam_T_vehicle` |
| `camera_to_world` = `world_T_cam` | `frame.sensor_from_world(...)` returns `cam_T_world` |
| `MapKeyPoints` point | `pycolmap.Point3D` |
| `PointObservations` | `pycolmap.Track` of `(image_id, point2D_idx)` |

Build with `reconstruction.add_rig`, `add_frame`, `add_camera`, `add_image`,
`register_frame`, or `set_rigs_and_frames`.

### 9.2 Bundle adjustment

`pycolmap.BundleAdjustmentOptions` maps almost one-to-one:

| cuSFM behaviour | pycolmap setting |
|---|---|
| One pose per rig frame | native: a `Frame` holds `rig_from_world`; set `refine_rig_from_world = True` |
| Extrinsics fixed (`--optimize_extrinsics=false`) | `refine_sensor_from_rig = False` |
| Extrinsics refined | `refine_sensor_from_rig = True`, plus `config.set_constant_sensor_from_rig_pose(...)` for `--fixed_camera_name` |
| Intrinsics fixed (`--optimize_intrinsics=false`) | `refine_focal_length = False`, `refine_principal_point = False`, `refine_extra_params = False` (defaults are `True/False/True` — all three must be set) |
| Points variable | `refine_points3D = True` (default) |
| Constant keyframe -> constant rig pose (§6.3) | `BundleAdjustmentConfig.set_constant_rig_from_world_pose(frame_id)` on the rig frame containing the lowest-id keyframe |
| `loss_type: CAUCHY`, scale 1.0 | `options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY`, `options.ceres.loss_function_scale = 1.0` |
| `refine_ba_avoid_loss_function` (stage 2) | `LossFunctionType.TRIVIAL` |
| `SPARSE_NORMAL_CHOLESKY` (the `use_cudss_solver` path) | already the pycolmap default, **but** `options.ceres.auto_select_solver_type = True` by default will override it — set it to `False` to force the solver. cuSFM's *CPU* fallback is `SPARSE_SCHUR` + `EIGEN_SPARSE`, which is also what pycolmap auto-selects for small problems; pick deliberately. |
| cuDSS GPU solve | `options.ceres.use_gpu = True`, `options.ceres.gpu_index = "0"` |
| `num_threads: 8` | `options.ceres.solver_options.num_threads = 8` |
| `max_num_iterations: 200` | `options.ceres.solver_options.max_num_iterations = 200` (pycolmap default 100) |
| `max_invalid_steps: 10` | `solver_options.max_num_consecutive_invalid_steps = 10` (already 10) |
| `max_linear_solver_iterations: 100` | `solver_options.max_linear_solver_iterations = 100` (pycolmap default 200) |
| `minimizer_progress_to_stdout: true` | `solver_options.minimizer_progress_to_stdout = True` |
| LEVENBERG_MARQUARDT | already the default |
| Ceres tolerances | cuSFM leaves them at Ceres defaults: `function_tolerance = 1e-6`, `gradient_tolerance = 1e-10`, `parameter_tolerance = 1e-8`. **pycolmap overrides them to `0.0 / 1e-4 / 0.0`** — set all three explicitly or the stopping behaviour will differ. |
| `EigenQuaternionManifold`, Eigen `[x,y,z,w]` order | pycolmap uses its own `Rigid3d` internally; only matters when converting quaternions at the boundary |
| `reprojection_error_standard_deviation: 4.0` | **no equivalent** — COLMAP does not whiten. Compensate exactly through the loss scale: `CauchyLoss(a)` is `a^2 log(1 + s/a^2)`, so scale `a = 1.0` on a residual divided by `sigma = 4.0` equals scale `a = sigma = 4.0` on the raw pixel residual, up to a constant factor `sigma^2` that does not move the minimum. **Set `loss_function_scale = 4.0`, not 1.0.** |

`constant_rig_from_world_rotation` and `min_track_length` have no cuSFM counterpart; leave
them at their defaults (`False`, `0`).

### 9.3 Triangulation and filtering

| cuSFM | pycolmap |
|---|---|
| `MultiViewsTriangulation` | `pycolmap.triangulate_multi_view_point(cams_from_world, cam_rays)` — **the same algorithm**: cuSFM uses `SelfAdjointEigenSolver<Matrix4d>` on the 4x4 normal matrix, exactly COLMAP's `TriangulateMultiViewPoint` |
| `MIDPOINT` (AV) | `pycolmap.triangulate_mid_point(cam2_from_cam1, ray1, ray2)` |
| RANSAC triangulation | `pycolmap.estimate_triangulation(points, cams_from_world, cameras, options)` |
| `max_ransac_iteration: 1000` | `ransac.max_num_trials = 1000` (default 10000) |
| hard-coded `desired_success_probability = 0.999` | `ransac.confidence = 0.999` (default 0.9999) |
| hard-coded `min_inlier_percentage = 0.0` | `ransac.min_inlier_ratio = 0.0` (default 0.02) |
| MSAC truncated-L1 scoring, no LO, no final refit | COLMAP's `LORANSAC` **does** local optimisation and a final refit. Expect slightly different, generally better, points. No option disables it. |
| `random_seed: 1` | `ransac.random_seed = 1` (default -1) |
| pixel-error inlier test | `options.residual_type = TriangulationResidualType.REPROJECTION_ERROR` — **the default is `ANGULAR_ERROR` with `max_error` 0.0349 rad**, so this must be changed, and `ransac.max_error` set in pixels |
| `min_triangulation_deg: 2.0` (all sampled pairs) | `EstimateTriangulationOptions.min_tri_angle = 2.0` (default 0.0) |
| post-BA angle filter (**max** over pairs) | `filter_points3D_with_small_triangulation_angle` also uses the max over pairs — matching semantics |
| Merge / Complete step | `IncrementalTriangulator.merge_all_tracks(opts)`, `complete_all_tracks(opts)` |
| `search_depth: 15` (max BFS hops when growing a track) | `IncrementalTriangulatorOptions.max_transitivity` / `complete_max_transitivity` (defaults 1 / 5). **Different semantics** — cuSFM grows disjoint connected components to 15 hops, COLMAP expands transitively per image. Treat 15 as an upper bound and tune. |
| `min_correspondences: 3` | `IncrementalTriangulatorOptions.ignore_two_view_tracks = True` (default), plus `ObservationManager.filter_points3D_with_short_tracks(3)` |
| Reprojection filter | `ObservationManager.filter_all_points3D(max_reproj_error, min_tri_angle)` or `filter_points3D_with_large_reprojection_error(max_error, ids, ReprojectionErrorType.PIXEL)` |
| Triangulation-angle filter | `filter_points3D_with_small_triangulation_angle(min_tri_angle, ids)` |
| Cheirality | `filter_observations_with_negative_depth()` |
| Decaying pixel gate 25, 20, 15, 10, 5 (linear) | no equivalent; drive `filter_all_points3D(max_reproj_error=g_k, min_tri_angle=2.0)` yourself once per outer round with `g_k = 25 + (5-25)*k/4` |
| `depth_threshold: 300` (world **Z** cap) | no equivalent; filter by hand on `point3D.xyz[2] > 300` |
| `observation change` early exit | no equivalent; compute `(merged + completed + filtered) / observed` yourself from the return values of `merge_all_tracks`, `complete_all_tracks` and the filter calls |

`IncrementalTriangulatorOptions` defaults that differ from cuSFM and must be overridden:
`min_angle = 1.5` (cuSFM 2.0), `merge_max_reproj_error = 4.0`,
`complete_max_reproj_error = 4.0`, `random_seed = -1`.

### 9.4 Camera models

| cuSFM `CameraProjectionModelType` | COLMAP model |
|---|---|
| `PINHOLE` | `PINHOLE` (fx, fy, cx, cy) |
| `DISTORTED_PINHOLE` | `FULL_OPENCV` (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6) |
| `OPENCV_FISHEYE` | `OPENCV_FISHEYE` (fx, fy, cx, cy, k1, k2, k3, k4) |
| `FTHETA_WINDSHIELD` | none — falls back to `PINHOLE` from the projection matrix |

**CONFIRMED**, `docs/camera.md`. Details and the parameter extraction are in
`docs/spec/export.md`.

### 9.5 What has no pycolmap equivalent

1. **Residual whitening** by `reprojection_error_standard_deviation`. Fold it into the loss
   scale.
2. **The decaying pixel-error schedule** (`initial_max_pixel_error` -> `final_max_pixel_error`).
   Drive it from Python around `filter_all_points3D`.
3. **The `observation change` convergence test.** Compute it from the observation set.
4. **Relative / absolute / extrinsic pose priors.** pycolmap's bundle adjuster has no prior
   terms. Needed only for the AV profile; would require `pyceres`. Not needed for Galileo
   (§7).
5. **`depth_threshold`** (a world-Z cap) and the `max_pose_gap` style guards.
6. **cuDSS specifically** — pycolmap exposes `use_gpu`, which selects its own GPU solver.
7. **The unweighted constant-frame residuals** (§6.4). pycolmap applies one loss and one
   weighting uniformly. Reproducing cuSFM's inconsistency is neither possible nor desirable;
   in Galileo it is 13 of 6276 blocks.
8. **The 3-D depth residual** (`VehicleCameraReprojectionCost3D`,
   `reprojection_error_standard_deviation_3d`) for keypoints carrying depth. pycolmap's
   bundle adjuster is 2-D only; this would need `pyceres`. Unused in Galileo
   (`keypoint_feature_has_depth: false`).

---

## 10. `bundle_adjustment_runner`

A standalone re-optimiser that reads an existing **COLMAP** sparse model instead of a cuSFM
kpmap. It shares `bundle_adjustment_solver.cc`, `bundle_adjustment_data.cc` and
`iterative_triangulator.cc` with `keypoints_mapper_main`; the difference is only the I/O
front end and the fact that every knob is a gflag rather than a config field.

Flags (`data/cusfm_re/help/bundle_adjustment_runner.txt`, source
`tools/visual/cusfm/bundle_adjustment_runner_main.cc`). Every one mirrors a
`BundleAdjustmentConfig` or `VisionMappingConfig` field, which makes this help text the
single best dictionary for the config semantics:

| Flag | Default | Config counterpart |
|---|---|---|
| `--colmap_sparse_dir` | required | — (input model) |
| `--output_dir` | required | — |
| `--output_binary` | false | — (`.bin` vs `.txt`) |
| `--loss_function` | `CAUCHY` | `loss_type` |
| `--loss_function_scale` | 1 | `loss_function_scale` |
| `--reprojection_error_std_dev` | 4 | `reprojection_error_standard_deviation` |
| `--max_iterations` | 200 | `max_num_iterations` |
| `--max_invalid_steps` | 10 | `max_invalid_steps` |
| `--max_linear_solver_iterations` | 100 | `max_linear_solver_iterations` |
| `--num_threads` | 8 | `num_threads` |
| `--use_cudss_solver` | true | `use_cudss_solver` |
| `--minimizer_progress_to_stdout` | true | `minimizer_progress_to_stdout` |
| `--verbose` | true | `verbose` |
| `--fix_first_camera` | **true** | inert in practice (§10) — the constant frame comes from `constant_keyframes_` (§6.3) |
| `--iterative_mode` | false | the outer loop |
| `--num_ba_iterations` | 5 | `num_ba_iterations` |
| `--iterative_triangulation_times` | 10 | `iterative_triangulation_times` |
| `--initial_max_pixel_error` | 25 | `initial_max_pixel_error` |
| `--final_max_pixel_error` | 5 | `final_max_pixel_error` |
| `--decay_max_pixel_error` | **true** | the 25 -> 5 schedule |
| `--min_triangulation_deg` | 2 | `min_triangulation_deg` |
| `--max_observation_change` | 0.0005 | `max_observation_change` |
| `--redo_triangulation` | false | re-triangulate and drop RGB from the output |
| `--do_rolling_shutter_correction` | false | `do_rolling_shutter_correction` |
| `--relative_pose_translation_error_meters` | 0.1 | same |
| `--relative_pose_rotation_error_degrees` | 5 | same |
| `--relative_constraint_use_pose_covariance` | false | same |
| `--absolute_pose_translation_error_meters` | 1 | same |
| `--absolute_pose_rotation_error_degrees` | 5 | same |
| `--absolute_position_constraint_error_meters` | 1 | same |
| `--extrinsic_error_meters` | 0.01 | same |
| `--extrinsic_error_degrees` | 2 | same |

Notes specific to the runner, all verified by live runs under `data/cusfm_re/runs/`:

- `--decay_max_pixel_error` defaults to **true**; the schedule is the same linear ramp as
  §5.6.
- **`--fix_first_camera` is inert.** Despite the description "Fix the first camera pose as
  reference", toggling it changed neither the parameter-block count nor the constant-frame
  cost group. Do not rely on it.
- The runner hard-codes `use_num_ba_iterations = true`, so **`--max_observation_change` is
  dead code there**. The two stopping rules are mutually exclusive.
- Reading a COLMAP model gives one pose per **image**, so the runner exercises the
  `CAMERA_FRAME` cost path (`OptimizePoseAndPointReprojectionCost<2,4,3,3>`) rather than the
  rig path, unless the model carries rig/frame information.
- `--redo_triangulation` re-triangulates and drops RGB from the output.

**CONFIRMED.**

---

## 11. Open questions

1. **Which keyframe enters `constant_keyframes_`, and by what rule.** In Galileo it is the
   lowest id (2359), and the block-count trajectory 13, 13, 13, 11, 1 matches that keyframe's
   observation count. The selection rule itself was not read out of the code. Low risk: any
   single keyframe fixes the gauge.
2. **The `"Point is 1000m away from camera"` guard** appears as a string but was not located
   in any decompiled body. Separate from `depth_threshold: 300`.
3. **`HasLargeEnoughTriangulateAngle` using the max over pairs** is inferred from a running
   maximum and a degree-valued out-parameter rather than read cleanly from the control flow.
4. **`maximum_cache_frame_size: 10000`** is assumed to be a keyframe LRU bound with no effect
   on results.
5. **Stage-2 (`refine_ba`) behaviour end to end** was read from the code but never executed —
   `refine_ba: false` in the Isaac profile.
6. The shipped Galileo artifact is a **32-image** run, not the 224-image acceptance run in
   `docs/open-pipeline-plan.md`, so all absolute counts here are small-run figures.

### Resolved during this work

- The pixel-error decay is **linear across the outer loop**: 25, 20, 15, 10, 5 (§5.6).
- `Iterative number: 0, 2, 4` was a `LOG_EVERY_N(INFO, 2)` artefact (§5.3).
- The reprojection residual is weighted by `1/sigma`, `sigma = 4.0` — measured cost scaling
  exactly `1/sigma^2` — but **constant-frame residuals are unweighted** (§6.4).
- `Initial cost` / `Final cost` = `sqrt(ceres_total_cost / num_residuals)` (§6.4), derived
  twice independently.
- `observation change = (merged + completed + filtered) / observed`, exact on all 5 rounds,
  and **not monotonic** (§5.4).
- The RANSAC constants are hard-coded, scoring is MSAC, no LO and no final refit (§5.3).
- Rotation uses `EigenQuaternionManifold`, so quaternion blocks are `[x,y,z,w]` (§6.2).
- The default linear solver is `SPARSE_SCHUR`; `use_cudss_solver` **replaces** it with
  `SPARSE_NORMAL_CHOLESKY` + `CUDA_SPARSE` (§6.5).
- The "constant frame" is a single **keyframe**, which through the rig fixes one **rig pose**
  (§6.3) — the 13/6263 cost split, not the 1151/4684 a whole rig frame would give.
- All pose-constraint families are gated on `--use_vehicle_trajectory` (default false), not on
  the `use_*_constraint` config fields, which the solver does not read (§7).
- `pose_constrains_type: AUTO` resolves through a jump table on `initial_pose_type` (§7).
- `depth_threshold` is a **world Z** cap, not a camera depth (§5.5).
- `search_depth` is a **BFS hop bound** on disjoint components (§5.1).
- The mapper never assigns point colour (§5.7).
- `--fix_first_camera` in `bundle_adjustment_runner` is inert (§10).

---

## 12. Evidence index

| Artifact | What it establishes |
|---|---|
| `data/cusfm_runs/galileo/kpmap/keypoints_mapper_main.txt` | The whole control flow with source file:line; Ceres summaries; cost groups; filter counts |
| `data/cusfm_re/notes/ba_parameterisation_from_ceres_summary.md` | The exact parameter-block model (six counters, two rounds); the sigma=4 whitening; the cost normalisation |
| `data/cusfm_re/notes/triangulation.md` | Track building, the triangulation estimator, the RANSAC constants, the filters, the colour finding |
| `data/cusfm_re/notes/ba_construction.md` | The 16 cost functors, the weighting sweep, the manifold, the constraint gating, the `AUTO` jump table, the solver options |
| `data/cusfm_re/runs/{ba_probe1,sigma8,loss_trivial,triv_s*,nofix,cauchy_scale2}/` | Live `bundle_adjustment_runner` ablations: the `1/sigma^2` cost scaling, the unweighted constant frame, the inert `--fix_first_camera`, the CPU solver fallback |
| `data/cusfm_re/decomp/tri/`, `data/cusfm_re/decomp/ba/` | Decompiled `iterative_triangulator.cc`, `triangulation_ransac.cc`, `ransac.cc`, `bundle_adjustment_*.cc` functions |
| `data/cusfm_re/runs/tri_v3/v3.log` | A fresh `--v=3` re-run: the `max_pixel_error` schedule, the inner-loop counts, the non-determinism |
| `data/cusfm_re/notes/rig_check.py`, `rig_check_output.txt` | Rig exactness (2e-6 deg), the fixed first frame (0.0000 mm), constant extrinsics and intrinsics |
| `data/cusfm_re/schema/config_protos.txt`, `extracted.fdset` | `VisionMappingConfig`, `BundleAdjustmentConfig`, `RansacConfig` and their enums, recovered from the binary |
| `data/cusfm_re/schema/data_protos.txt` | `MapKeyPoints`, `KeyframeMetaData`, `CameraSensor`, `RigidTransform3d` |
| `data/cusfm_re/help/bundle_adjustment_runner.txt` | Flag-level dictionary for the BA config, with defaults |
| `data/cusfm_re/notes/pycolmap_probe.txt` | pycolmap 4.2.0 option defaults, rig/frame API, camera models |
| `pycusfm/configs/isaac/*.pb.txt`, `pycusfm/configs/av/*.pb.txt` | The two profiles |
| `docs/camera.md` | Camera models, projection maths, frame conventions |
| arXiv 2510.15271 | Track building, DLT and midpoint triangulation, the rig decomposition, the two-stage BA design, SPARSE_NORMAL_CHOLESKY |

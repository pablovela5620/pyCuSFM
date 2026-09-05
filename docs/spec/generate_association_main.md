# `generate_association_main` — implementation spec

Reverse-engineered from `pycusfm/x86_cuda13/bin/generate_association_main` (unstripped, source
paths `/home/jryu/workspaces/visual_mapping/src/...`). Core module:
`src/visual/cusfm/association_worker.cc`, class
`nvidia::isaac::visual::cusfm::AssociationWorker`. Geometry lives in
`src/visual/cuvgl/stereo_pose_estimator.cc` (`StereoPoseEstimator`) and
`src/visual/loop_closing/loop_closure_index.cc` (`LoopClosureIndex`).

Confidence tags: **CONFIRMED** (read in decompiled code, disassembly, strings, or measured on
artifacts), **LIKELY** (strongly implied but one inference step away), **INFERRED** (reasoned
from types, names, or config comments only).

---

## 1. Purpose and position in the pipeline

`generate_association_main` decides **which pairs of keyframes are related** and, in one of its
two modes, **what the relative pose between them is**. It writes one `ConstraintPair` protobuf
per pair plus a summary `pose_graph.pb.txt`.

```
feature_extractor_main  ->  keyframes/            (frames_meta.json + per-image keypoint .pb)
generate_bow_*_main     ->  cuvgl_map/            (vocabulary/ + bow_index.pb)
                                  |
                                  v
                    generate_association_main  ->  associations/
                                  |                     |
                                  |                     +-> pose_graph_association_main / pose_graph_main
                                  +-------------------------> feature_matcher_task_builder_main
```

It produces three kinds of edge:

| type | meaning | count on `robocap_full` |
|---|---|---|
| `EXTRINSIC` (3) | two cameras of the same rig at the same instant | 6792 = 1132 rigs x C(4,2) |
| `CONSECUTIVE` (1) | the same camera in rig *k* and rig *k+1* | 4524 = 1131 x 4 |
| `LOOP` (2) | revisit found by BoW retrieval | 0 (pose-graph mode) / 231 (feature-matcher mode) |

CONFIRMED from `data/cusfm_runs/robocap_full/cusfm/associations/associations_main.txt` and from
counting the 11316 `association/*.pb` files.

### 1.1 Two modes — this is the single most important fact

`--skip_pose_graph` does **not** merely skip a step; it switches the whole loop-closure branch
to a different function.

| | `--skip_pose_graph=false` (default) | `--skip_pose_graph=true` |
|---|---|---|
| loop routine | `AssociationWorker::RetrievalLoopAssociationsForFrame` | `AssociationWorker::RetrievalLoopAssociationsForFeatureMatcher` |
| frames queried | left cameras that have a stereo partner only (1132 of 4528) | every keyframe (4528) |
| candidate filter | BoW score, `|dt|` gate, LightGlue + `StereoPoseEstimator` verification | BoW score, session+`|dt|` gate, **pose-prior** gate (`pose_translation_threshold_meters` / `pose_rotation_threshold_degrees`) |
| LOOP pose written | measured relative pose from two-view geometry | **identity** (pair list only) |
| LOOP covariance | `num_inliers * loop_residual_weight * I6` | `I6` |
| LOOP `matches` / `matches_inliers` | populated | empty |
| after retrieval | `GetMaxConnectedGraph` -> `UpdateAssociationStateByValidKeyFrame` -> `RecoordinateKeyFrames` | all three skipped |

CONFIRMED: `AssociationWorker::RetrievalLoopAssociations` dispatches on `this[0x300]`, the flag
stored by `Init(..., bool skip_pose_graph)`; `RetrievalAssociations` guards the post-processing
with the same flag. Measured: my two runs
`data/cusfm_re/runs/assoc_v3/verbose_run.log` (skip=true, 231 LOOP, identity poses) and
`data/cusfm_re/runs/assoc_pg_v3/verbose_run.log` (skip=false, 0 LOOP).

`pycusfm/cusfm_runner.py` defaults to `skip_pose_graph=False` (line 66, line 1061); the `av`
config preset sets it to `True` (line ~1139).

---

## 2. Command line

Built by `CuSFMRunner.generate_associations` (`pycusfm/cusfm_runner.py:557-...`).

| flag | required | meaning | value the runner passes |
|---|---|---|---|
| `--keyframe_dir` | yes | directory holding `frames_meta.json` and the per-image keypoint protos | `<out>/cusfm/keyframes` |
| `--gl_map_dir` | yes | directory holding `vocabulary/` and `bow_index.pb` | `<out>/cusfm/cuvgl_map` |
| `--output_dir` | yes | where `association/` and `pose_graph.pb.txt` are written | `<out>/cusfm/associations` |
| `--association_worker_config` | yes | text proto `protos.visual.cusfm.AssociationWorkConfig` | `configs/isaac/association_config.pb.txt` |
| `--matching_worker_config` | yes | text proto `protos.visual.cusfm.MatchingTaskWorkerConfig` | `configs/isaac/matching_task_worker_config.pb.txt` |
| `--model_dir` | no (default `models`) | TensorRT engine / ONNX directory for LightGlue | `pycusfm/models` |
| `--image_dir` | no | if non-empty, `SaveDebugAssociations` writes match overlay images | only with `enable_debug` |
| `--skip_pose_graph` | no (default false) | see 1.1 | `true` only for the `av` preset |

**`--matching_worker_config` is parsed, validated, and then thrown away.** CONFIRMED by
disassembly of `CheckInputValidity` at `0x3dcc3-0x3dd52`: it constructs a stack-local
`MatchingTaskWorkerConfig`, calls `FileUtils::ReadProtoFileByExtension(FLAGS_matching_worker_config, &local)`,
`CHECK`s the status, then reads `FLAGS_association_worker_config` into the real config and
destroys the local. Every matching / verification parameter the binary actually uses comes from
`AssociationWorkConfig.match_worker_config`. The file must exist and parse or the process dies.

`gflags` dump: `data/cusfm_re/help/generate_association_main.txt`.

---

## 3. Inputs

### 3.1 `keyframe_dir/frames_meta.json`

A JSON object (`keyframe_metadata.cc:103` loads it):

```
{
  "keyframes_metadata": [                 # index in this list == frame id (uint64)
    { "timestamp_microseconds": "101622069",
      "image_name": "left_front/101622069778.jpeg",
      "camera_params_id": "1",            # absent means 0
      "synced_sample_id": "1",            # rig / vehicle-frame grouping key
      "camera_to_world": { "axis_angle": {x,y,z,angle_degrees},
                           "translation": {x,y,z} } },
    ...
  ],
  "initial_pose_type": "EGO_MOTION",
  "camera_params_id_to_camera_params": { "0": {...}, "1": {...}, ... },
  "stereo_pair": [ { "right_camera_param_id": "1", "baseline_meters": 0.08619758 } ],
  "detector_type": ..., "descriptor_type": ...
}
```

CONFIRMED by reading the file for `robocap_full` (4528 entries, 1132 distinct
`synced_sample_id`, 4 `camera_params_id`).

Consumed as:
* `camera_to_world` is the pose used for every EXTRINSIC and CONSECUTIVE constraint and for the
  pose-prior gate in feature-matcher mode.
* `synced_sample_id` groups frames into *vehicle frames* (rigs);
  `KeyframesMetadataCollection::ComputeVehicleFrames()` is called from `AssociationWorker::Init`
  (CONFIRMED, disassembly `0x5131e`) and fills `frames_in_rig_` / `cameras_in_rig_`.
* `stereo_pair` gives, per left camera id, the right camera id and the metric baseline. `Init`
  calls `GetLeftToRightCameraIDMap()` and builds an `unordered_map<camera_params_id, double
  baseline>` used by `CalculateRelativePoseViaStereoEstimator` ("Baseline not found for camera
  params ID: " otherwise). CONFIRMED (disassembly `0x5137a`, decompiled
  `CalculateRelativePoseViaStereoEstimator`).
* Per-frame keypoints and descriptors come from the per-image protos under
  `keyframe_dir/<camera>/<ts>.pb`, loaded lazily by `KeyframeCacher` (`KeyframeCacher::LoadFrame`
  in the profile timers; 25677 loads on `robocap_full`).

### 3.2 `gl_map_dir`

* `gl_map_dir/vocabulary/bow_vocabulary.pb.txt` — BoW tree (`visual_vocabulary.cc:663`).
* `gl_map_dir/bow_index.pb` — inverted index (`loop_closure_index.cc:46`).

Both are only used by `LoopClosureIndex::DetectCandidates`. See `docs/spec/bow.md` for the
vocabulary format and scoring.

---

## 4. Outputs

### 4.1 `output_dir/association/<source>_<target>.pb`

One binary `protos.visual.cusfm.ConstraintPair` per association, for **every** association in
the graph, good or not. CONFIRMED: `SaveGraphToFile` iterates the whole
`map<FramePairID, AssociationState>` and calls `SaveAssociationToFile`, which builds
`JoinPath(output_dir, "association")` then `to_string(source) + "_" + to_string(target) + ".pb"`.

```protobuf
message ConstraintPair {
  FramePair        frame_pair             = 1;  // {source_frame_id, target_frame_id}
  repeated FeatureMatch matches           = 2;  // {source_point_id, target_point_id} (uint32)
  RigidTransform3d pose_source_to_target  = 3;  // axis_angle (DEGREES) + translation (metres)
  MatrixD          pose_covariance        = 4;  // 6x6 row-major, row_count=column_count=6
  ConstraintType   type                   = 5;  // UNKNOWN/CONSECUTIVE/LOOP/EXTRINSIC
  repeated bool    matches_inliers        = 6;  // parallel to `matches`
  int32            matched_pairs_num      = 7;
  bool             is_good                = 8;
  double           time_delta_source_to_target = 9;  // seconds
}
```

Field semantics, all CONFIRMED from `AssociationState::ToProto` plus measurement:

* **`frame_pair` is always sorted**: `source_frame_id < target_frame_id` (`std::min`/`std::max`
  in all three producers).
* **`pose_source_to_target` is `T_target_from_source`.** Verified numerically on
  `robocap_full` (`data/cusfm_re/tools/verify_pose.py`): for every sampled pair,
  `pose_source_to_target == inv(camera_to_world[target]) @ camera_to_world[source]`
  to ~1e-15, while `inv(T_source) @ T_target` is off by 0.17-2.0. So the matrix maps a point
  expressed in the **source** camera frame into the **target** camera frame. The name reads
  "transform *from* source *to* target", not "pose of source relative to target".
* `axis_angle.angle_degrees` is **degrees**; the axis is a unit vector.
  `QuaternionToAxisAndAngleDegree` is the converter (CONFIRMED in `ToProto`).
* `pose_covariance` is stored row-major, `data[r*6+c]`; `AssociationState` holds it column-major
  (Eigen) and `ToProto` transposes while copying. Despite the name it is used as an
  **information / weight** matrix, not a covariance: EXTRINSIC and CONSECUTIVE get `I6`, LOOP
  (pose-graph mode) gets `num_inliers * loop_residual_weight * I6`. An "empty/identity" 6x6
  therefore means *unit weight*, not *unit uncertainty*.
* `time_delta_source_to_target = t_source - t_target` in seconds. Measured `-0.133366` for
  consecutive pairs (7.5 Hz rig) and `0.0` for extrinsic pairs; matches `frames_meta.json`
  exactly.
* **`matched_pairs_num` is uninitialised garbage for EXTRINSIC and CONSECUTIVE.** CONFIRMED:
  `AddAssociationToGraph` only calls `AssociationState::DetermineValidityViaInlierMatches`
  (which writes `state+0x1e8`) on the LOOP path; on the other path it writes
  `is_good = true` (`state+0x1ec`) directly and leaves `state+0x1e8` untouched. Measured: every
  EXTRINSIC pair in `robocap_full` carries the constant `1055136869` (the same stale stack slot),
  CONSECUTIVE pairs carry random values such as `-582370477`, `112889247`, `1136103105`.
  **Do not read this field for non-LOOP edges.**
* `is_good`: `true` unconditionally for EXTRINSIC; for LOOP (pose-graph mode)
  `num_inliers > min_matches_num && num_inliers / num_matches > min_matches_ratio`
  (CONFIRMED, `DetermineValidityViaInlierMatches`); it is then cleared by
  `UpdateAssociationStateByValidKeyFrame` if either endpoint keyframe is invalid.
* `matches` / `matches_inliers` are empty for EXTRINSIC and CONSECUTIVE (measured: 0 matches on
  all 40 sampled files) and for feature-matcher-mode LOOP.

### 4.2 `output_dir/pose_graph.pb.txt`

Text proto `protos.visual.cusfm.PoseGraph { repeated ConstraintPair pairs }`. CONFIRMED from
`SaveAsPoseGraph`: it emits **only `is_good` associations** and **only three fields** —
`frame_pair`, `pose_source_to_target`, `type`. No matches, no covariance, no `is_good`, no
`time_delta`. A type outside 1..3 logs "Association type is not supported." and aborts the save.

Measured on `robocap_full`: 11316 `pairs`, zero occurrences of `pose_covariance`, `matches`,
`is_good`, `time_delta`.

### 4.3 Debug output (only with `--image_dir`)

`SaveDebugAssociations` -> `SaveDebugAssociationImage` writes match overlays. Not used by the
pipeline.

### 4.4 What is *not* written

`RecoordinateKeyFrames` rewrites keyframe poses **in memory only**. `SaveKeyFramesToFile` exists
but `main` never calls it. Measured: `keyframes/frames_meta.json` mtime is unchanged by the
association run. So this stage never modifies its input.

---

## 5. Configuration reference

All defaults below are from `pycusfm/configs/isaac/association_config.pb.txt` unless noted.
`AssociationWorkConfig` is copied wholesale into the worker (`Init` -> `CopyFrom`).

`pycusfm/configs/av/association_config.pb.txt` differs in only four places (CONFIRMED by diff):
`loop_interval_threshold_in_seconds: 5`, `good_score_threshold: 0.01`,
`lightglue_params.match_threshold: 0.8`, and no `aliked_tensorrt_config` block. The whole
`stereo_estimator_config` block is identical.

### 5.1 `AssociationWorkConfig` (top level)

| field | type | isaac value (file:line) | effect | conf |
|---|---|---|---|---|
| `consecutive_frame_count` | uint32 | `1` (`association_config.pb.txt:7`) | how many *successful* forward rig hops to add per frame; the loop over later rigs stops once this many `ProcessConsecutiveCandidates` calls returned true | CONFIRMED (`RetrievalConsecutiveAssociations` compares the success counter with `this+0x208`) |
| `init_pose_mode` | enum | `GIVEN` (`:1`) | `GIVEN` / `STEREO` / `MONO`. With `GIVEN` the consecutive and extrinsic poses are taken from `camera_to_world`. No other branch was observed to execute. | LIKELY (enum read; only the GIVEN behaviour was exercised) |
| `use_frame_selection` | bool | `false` (`:5`) | when true, `RetrievalAssociations` calls `KeyframeCacher::UpdateKeyFrameValidityByInlierRatio(0.25)` (**hard-coded 0.25**) then `UpdateAssociationStateByValidKeyFrame` before the connected-graph pass | CONFIRMED (decompiled `RetrievalAssociations`) |
| `loop_interval_threshold_in_seconds` | double | `10` (`:2`) | temporal gate on loop candidates; also the bucket width in `SelectBestCandidates` | CONFIRMED |
| `min_matches_num` | uint32 | `30` (`:3`) | LOOP `is_good` needs `inliers > 30` | CONFIRMED |
| `min_matches_ratio` | float | `0.25` (`:4`) | LOOP `is_good` needs `inliers / matches > 0.25` | CONFIRMED |
| `loop_residual_weight` | float | `0.25` (`:6`) | LOOP information matrix = `inliers * 0.25 * I6` | CONFIRMED |
| `match_worker_config` | msg | see 5.3 | matcher + essential-matrix verification parameters | CONFIRMED |
| `stereo_estimator_config` | msg | see 5.4 | two-view geometry gates | CONFIRMED |
| `image_retrieval_config` | msg | see 5.2 | BoW retrieval gates | CONFIRMED |

### 5.2 `protos.visual.loop_closing.ImageRetrievalConfig`

Only five fields are read by this binary.

| field | type | isaac value (`association_config.pb.txt`) | effect | conf |
|---|---|---|---|---|
| `min_shared_word_number` | uint32 | `50` (`:9`) | 2nd arg of `LoopClosureIndex::DetectCandidates`; raw candidates with fewer shared words are dropped | CONFIRMED (`DetectCandidates` filters on `node+0x2c`, the shared-word count) |
| `query_result_number` | uint32 | `20` (`:10`) | 3rd arg; `CalculateCandidatesScore` sorts by score and truncates the candidate list to this many | CONFIRMED |
| `good_score_threshold` | double | `0.1` (`:11`) | candidate kept only if `score >= good_score_threshold` | CONFIRMED (both loop routines test `*(double*)(cfg+0x40) <= candidate.score`) |
| `pose_translation_threshold_meters` | float | `1` (`:12`) | **feature-matcher mode only**: `||t(inv(T_query)*T_cand)|| <= 1 m` | CONFIRMED |
| `pose_rotation_threshold_degrees` | float | `10` (`:13`) | **feature-matcher mode only**: `2*acos(q.w) <= 10 deg` | CONFIRMED (log: "Skip candidate N due to large pose difference: trans=..m, rot=..rad", all rejected values above 0.1745 rad) |

Every other `ImageRetrievalConfig` field (tree geometry, `min_inlier_number`,
`max_reprojection_error`, ...) belongs to vocabulary building or to `DetectCandidatesForMapper`
and is unused here (INFERRED: no reference to offsets 0x10-0x3c or 0x30 in either loop routine).

### 5.3 `MatchingTaskWorkerConfig` (as `match_worker_config`)

| field | type | isaac value | effect | conf |
|---|---|---|---|---|
| `max_pixel_error` | double | `4.0` (`:16`) | RANSAC threshold passed to `DataSelector::SelectMatchesByOpenCVEssentialMat` inside `AddAssociationToGraph` when the caller supplies no inlier mask | CONFIRMED (`*(double*)(cfg+0x20)`) |
| `min_ransac_confidence` | double | `0.95` (`:17`) | RANSAC confidence for the same call | CONFIRMED (`*(double*)(cfg+0x28)`) |
| `matching_params.lightglue_params.match_threshold` | float | `0.3` | LightGlue score threshold | CONFIRMED (log line `lightglue_matcher.cc:38` prints the sibling fields) |
| `matching_params.lightglue_params.match_top_k` | int | `500` | at most 500 matches kept, via `SUPPRESSION_VIA_SQUARE_COVERING` non-max | CONFIRMED (log: "top_k: 500") |
| `matching_params.lightglue_params.num_points_tolerance_fraction` | float | `0.3` | LightGlue batching tolerance | CONFIRMED (log) |
| `matching_params.lightglue_params.aliked_tensorrt_config` | msg | fp16, opt 3200 pts, max 5500 | engine selection / dynamic shape profile | CONFIRMED (engine path in the log) |
| `verification_config.*` | msg | present in the file | **not read by this binary** — `AddAssociationToGraph` uses the top-level `max_pixel_error`/`min_ransac_confidence` at offsets 0x20/0x28, not the nested message at 0x18 | LIKELY |
| `matching_params.cv_cuda_params.*`, `general_matching_params.*` | msg | present | only relevant when the matcher is not LightGlue; the log says "Matcher will use lightglue" | LIKELY |

### 5.4 `protos.visual.cuvgl.StereoPoseEstimatorConfig` (as `stereo_estimator_config`)

Used only by the pose-graph-mode loop path.

| field | type | isaac value (`association_config.pb.txt`) | effect | conf |
|---|---|---|---|---|
| `ransac_max_pixel_error` | double | `1` (`:143`) | mapped to normalised-plane units by `geometry::MapErrorToNormalizedPlane(proj0, proj1, err_px)` and passed as the `cv::findEssentialMat` threshold | CONFIRMED |
| `ransac_confidence` | double | `0.999` (`:145`) | `cv::findEssentialMat` probability | CONFIRMED |
| `distance_3d_threshold` | double | `1e9` (`:147`) | passed to `PoseEstimateFromTwoViewNormalized` as `cv::recoverPose`'s `distanceThresh` (effectively disabled) | LIKELY |
| `scale_upper_limit` | double | `4` (`:149`) | (a) initial scale must be in `[0, 4]` or the fallback/early-return fires; (b) after refinement `||t|| <= 4` or "Scale too large after pose refinement" | CONFIRMED |
| `min_inlier_number` | uint32 | `25` (`:150`) | three separate gates: raw left-to-candidate matches, raw candidate-to-right matches, and each two-view inlier count | CONFIRMED |
| `min_occupied_area_ratio` | double | `0.10` (`:154`) | `min(area_left_cand, area_cand_right) >= 0.10`, where area is `GetMatchedPointDistribution` in 0..1 | CONFIRMED ("Occupied area (0~1) is less than threshold") |
| `max_mean_point_to_epipolarline_error` | double | `10e-6` (`:156`) | after `StereoPoseRefineSolver::SolveProblem`, the mean point-to-epipolar-line error (normalised plane) must be `<= 1e-5`. **This is the gate that rejects every loop candidate on RoboCap.** | CONFIRMED (measured, see 6.4) |
| `enable_homography_check` | bool | `false` (`:157`) | 8th arg of `PoseEstimateFromTwoViewNormalized`; enables the H-vs-F comparison ("Homography is better than Fundamental") | LIKELY |
| `use_estimated_baseline_direction` | bool | `false` (`:158`) | when false the left->right translation direction is hard-set to `(-1, 0, 0)` | LIKELY (the `(-1,0,0)` store is in the `else` branch of `this[0x45]`) |
| `best_candidate_use_max_inliers` | bool | `false` (`:159`) | multi-camera candidate ranking in `LocalizeByStereoFrames`; not on this path | INFERRED |
| `keep_duplicate_sift_match` | bool | `false` (`:160`) | when false and `descriptor_type == 2` (SIFT), `RemoveDuplicateSIFTKeypoints` runs. ALIKED is not type 2, so this never fires here. | CONFIRMED (branch `(*(int*)(kpv+0x24) == 2) && !cfg[0x47]`) |
| `online_calibrate_left2right_rot` | bool | `false` (`:161`) | not exercised | INFERRED |
| `early_return_invalid_scale` | bool | `false` (`:162`) | when false, an out-of-range initial scale is silently replaced by the magic constant **0.47**; when true the candidate is rejected with "Inital scale wrong ..., initial scale unknown!" | CONFIRMED (the literal `0x3fde147ae147ae14 == 0.47` is stored in the `else` branch) |
| `vote_translation_error_meter_tolerance` | double | `0.5` (`:165`) | multi-camera voting in `LocalizeByStereoFrames`; **not used** by `EstimateRelativePose` | LIKELY |
| `vote_rotation_error_degree_tolerance` | double | `5` (`:166`) | same | LIKELY |
| `init_scale_use_linear_search` | bool | `false` (`:169`) | false -> `EstimateScaleByLinearFitting`; true -> `EstimateScaleLinearSearch(..., scale_upper_limit, step)` | CONFIRMED |
| `init_scale_linear_search_step` | float | `0.05` (`:170`) | step of the linear search | CONFIRMED |
| `allow_best_single_inlier` | bool | `false` (`:175`) | `LocalizeByStereoFrames` only | INFERRED |
| `allow_single_camera_result` | bool | `false` (`:182`) | `LocalizeByStereoFrames` only | INFERRED |
| `relative_pose_translation_threshold_meters` | float | unset -> `0` | post-refinement `||t||` gate; `<= 0` disables it | CONFIRMED ("relative pose translation too large: X, threshold:Y") |
| `relative_pose_rotation_threshold_degrees` | float | unset -> `0` | post-refinement `2*acos(q.w)` gate in degrees; `<= 0` disables it | CONFIRMED |
| `use_cudss_solver` | bool | unset | solver choice inside `StereoPoseRefineSolver` | INFERRED |

---

## 6. Algorithm

`main` (`tools/visual/cusfm/generate_association_main.cc`):

```
ParseCommandLineFlags
CheckInputValidity            # reads both config files, keeps only the association one
                              # keyframe_cache_manager->Init(FLAGS_keyframe_dir)
                              # image_retriever->Init(gl_map_dir/vocabulary, gl_map_dir/bow_index.pb)
LOG "Initialization completed."                       # generate_association_main.cc:81
AssociationWorker::Init(cacher, retriever, output_dir, cfg, model_dir, skip_pose_graph)
    - CopyFrom(config)
    - build KeypointMatcher (LightGlue, engine from model_dir)
    - KeyframesMetadataCollection::ComputeVehicleFrames()
    - StereoPoseEstimator config copy; build camera_id -> baseline map
AssociationWorker::RetrievalAssociations(...)          # section 6.1
AssociationWorker::SaveAssociationsToFile()            # SaveGraphToFile + SaveAsPoseGraph
[--image_dir] SaveDebugAssociations()
```

### 6.1 `RetrievalAssociations` (association_worker.cc:~120-135)

```
RetrievalExtrinsicAssociations()      || return false
RetrievalConsecutiveAssociations()    || return false
RetrievalLoopAssociations()           || return false
if (config.use_frame_selection):
    KeyframeCacher::UpdateKeyFrameValidityByInlierRatio(0.25)
    UpdateAssociationStateByValidKeyFrame()
if (!skip_pose_graph):
    root = GetMaxConnectedGraph()
    UpdateAssociationStateByValidKeyFrame()
    RecoordinateKeyFrames(root)
LOG "Retrieval N associations."                        # association_worker.cc:132
```

CONFIRMED (decompiled `RetrievalAssociations`).

### 6.2 Step 1 — extrinsic associations (`association_worker.cc:396-454`)

```
cam2cam = KeyframesMetadataCollection::GetCameraToCameraTransforms()   # map<FramePairID(camA,camB), SE3>
for rig in vehicle_frames_:                       # ordered map, one entry per synced_sample_id
    CHECK(frames_in_rig_.size() == cameras_in_rig_.size())
    LOG every 10% "Processing extrinsic associations: i / N"
    for i in range(len(rig)):
      for j in range(i+1, len(rig)):
        if cameras[i] == cameras[j]: continue
        key = (min(cam_i, cam_j), max(cam_i, cam_j))
        T = cam2cam[key]  or  LOG "No camera to camera transform found for frame pair: a and b"; continue
        pair = (min(frame_i, frame_j), max(frame_i, frame_j))
        if the camera order and the frame order disagree:  T = T.Inverse()
        AddAssociationToGraph(pair, T, EXTRINSIC, I6, {}, {}, 0)
LOG "N extrinsic associations are retrieved."
```

Notes:
* The relative pose is a **fixed per-camera-pair calibration**, not re-derived per rig.
  Measured: `0_1.pb` and `1000_1001.pb` are bit-identical, and both equal
  `inv(T_target) @ T_source` from `frames_meta.json` to 1e-15 (the given poses are rigidly
  consistent inside a rig because they were generated from a rig pose plus extrinsics).
* Weight matrix is exactly `I6`; `matches`, `matches_inliers` empty; `matched_pairs_num` arg `0`
  (and left uninitialised in the proto, see 4.1).
* Cost on `robocap_full`: 3.6 s for 6792 edges (no image data touched).

### 6.3 Step 2 — consecutive associations (`association_worker.cc:464-504`)

```
for rig_k in vehicle_frames_:                    # ordered by synced_sample_id
    for frame in rig_k.frames_in_rig_:
        successes = 0; offset = 1
        while successes < consecutive_frame_count:
            rig_next = rig_k advanced by `offset`
            if rig_next past the end: break
            if rig_next.<field at +8> != rig_k.<field at +8>: break     # same session/vehicle
            if ProcessConsecutiveCandidates(frame, rig_next.frames_in_rig_): successes += 1
            offset += 1
LOG "N consecutive associations are retrieved."
```

`ProcessConsecutiveCandidates(frame_id, candidate_frames)`:

```
kf = cacher[frame_id]
for cand in candidate_frames:
    kc = cacher[cand]; if !kc: break
    if kc.camera_params_id != kf.camera_params_id: break     # same camera only
    pair  = (min(frame_id, cand), max(frame_id, cand))
    T     = inv(camera_to_world[pair.target]) * camera_to_world[pair.source]
    AddAssociationToGraph(pair, T, CONSECUTIVE, I6, {}, {}, 0)
```

CONFIRMED (decompiled; the six `0x3ff0000000000000` stores on the stack are the `I6` diagonal at
byte offsets 0, 56, 112, 168, 224, 280 of a 6x6 row-major double matrix). Measured: 4524 pairs,
all stride 4 (= rig size), all with `time_delta_source_to_target == -0.1333` matching
`frames_meta.json`.

`consecutive_frame_count: 1` therefore yields exactly one edge per camera per rig hop. No image
data is read; cost 5.8 s.

### 6.4 Step 3a — loop associations, **pose-graph mode** (`RetrievalLoopAssociationsForFrame`, association_worker.cc:836-1020)

Per query frame `q` (all 4528 keyframes are iterated; most exit immediately):

```
r = GetStereoPairFrameId(q)
if r == npos:                     -> "Skip, due to invalid frame id or cannot read keyframes: q"
kf_q = cacher[q]; kf_r = cacher[r]
if !IsLeftCamera(kf_q.camera_params_id):
                                  -> "Skip, due to frame q is not a left camera frame"

cands = LoopClosureIndex::DetectCandidates(kf_q.keypoints,
                                           min_shared_word_number,   # 50
                                           query_result_number)      # 20
    # DetectRawCandidates -> map<frame, (score, shared_words)>
    # drop shared_words < 50
    # CalculateCandidatesScore: sort by score, truncate to 20

n = len(cands); half = n/2 + 1; near_in_time = 0
infos = []
for i, c in enumerate(cands):
    if c.score < good_score_threshold: continue          # 0.1
    if c.frame_id == q: continue
    kf_c = cacher[c.frame_id]  or  "Skip, can't read keyframe"
    dt = abs(kf_q.timestamp_us - kf_c.timestamp_us) / 1e6
    if dt < loop_interval_threshold_in_seconds:          # 10 s
        near_in_time += 1
        if i <= half and near_in_time >= half: break     # early abort for this frame
        continue
    ok, T_cand_from_q, res = CalculateRelativePoseViaStereoEstimator(kf_q, kf_r, kf_c)
    if !ok: continue                                     # "Skip, cannot estimate relative pose"
    infos.append(LoopCandidateInfo{c, res, T_cand_from_q, dt})

best = SelectBestCandidates(infos)
for k in best:
    info = infos[k]
    pair = (min(q, info.frame_id), max(q, info.frame_id))
    T    = info.pose;  if info.frame_id == pair.source: T = T.Inverse()
    W    = res.num_inliers * loop_residual_weight * I6
    AddAssociationToGraph(pair, T, LOOP, W, res.matches, res.inlier_mask, res.num_inliers)
LOG "N loop associations are retrieved."
```

`SelectBestCandidates` (CONFIRMED, decompiled): builds `tuple<index, dt, res.num_inliers>`,
sorts **ascending by `dt`**, then walks the sorted list greedily:

```
last_dt = -inf; last = 0
for e in sorted:
    if e.dt - last_dt > loop_interval_threshold_in_seconds:
        emit(e.index); last = e.index; last_dt = e.dt
    else:                                   # same 10-second bucket as the last emitted one
        if e.num_inliers > infos[last].num_inliers                    -> replace last emit
        elif e.num_inliers == infos[last].num_inliers and e.bow_score > infos[last].bow_score
                                                                      -> replace last emit
```

So it keeps at most one candidate per 10-second band of `|dt|`, ranked first by geometric inlier
count and second by BoW score.

Only `q` for which a stereo partner exists and which is the left camera reach retrieval.
Measured on `robocap_full`: 1132 processed, 1132 "not a left camera", 2264 "invalid frame id"
(cameras 2 and 3 have no `stereo_pair` entry).

#### `CalculateRelativePoseViaStereoEstimator(kf_left, kf_right, kf_cand)`

```
projectors for all three frames        -> "Skip, invalid keyframe with projector."
m_lc = GetKeyPointMatches(kf_left, kf_cand)   # LightGlue, no epipolar prior
m_cr = GetKeyPointMatches(kf_cand, kf_right)
if either fails                       -> "Skip, cannot get keypoint matches."
baseline = baseline_map[kf_left.camera_params_id]
                                      -> "Baseline not found for camera params ID: N"
StereoPoseEstimator::EstimateRelativePose(kf_left, kf_right, kf_cand, m_lc, m_cr, baseline, &T, &res)
```

`GetKeyPointMatches` copies both `KeypointVector`s and calls
`KeypointMatcher::MatchKeypoints(kp0, kp1, &matches, nullptr /*no fundamental prior*/, {})`.
Two calls per candidate — the profile shows exactly 360 `GetKeyPointMatches` for 180 candidates.

#### `StereoPoseEstimator::EstimateRelativePose` (stereo_pose_estimator.cc, gates in order)

1. `len(m_lc) >= min_inlier_number && len(m_cr) >= min_inlier_number`
   -> "Max Inlier Number is less than threshold, left matches: A, right matches: B, threshold: T"
2. `NormalizedPointAndComputePose(kf_left, kf_cand, m_lc)`:
   * optional `RemoveDuplicateSIFTKeypoints` (SIFT only)
   * unproject both keypoint sets through their camera projectors to the normalised plane
   * `thresh = MapErrorToNormalizedPlane(proj_left, proj_cand, ransac_max_pixel_error)`
   * `PoseEstimateFromTwoViewNormalized(p0, p1, &inliers, &T, thresh, ransac_confidence,
      distance_3d_threshold, enable_homography_check)`:
     - needs >= 5 correspondences ("Two view verification need at least 5 matched point.")
     - `cv::findEssentialMat(p0, p1, K=I3, method, prob=ransac_confidence, threshold=thresh)`
     - `cv::recoverPose(E, p0, p1, K=I3, R, t, distanceThresh, mask)` -> `T = cand_T_left`
   * `GetMatchedPointDistribution(...)` -> occupied-area fraction in 0..1
3. `len(inliers_lc) >= min_inlier_number`
   -> "Geometry check: inlier is less than threshold, left to cand pairs: N, threshold: T"
4. left->right baseline direction: `(-1, 0, 0)` unless `use_estimated_baseline_direction`
5. `NormalizedPointAndComputePose(kf_cand, kf_right, m_cr)` and
   `len(inliers_cr) >= min_inlier_number`
   -> "candidate to right pairs is less than threshold, candidate has: N, threshold is: T"
6. `min(area_lc, area_cr) >= min_occupied_area_ratio`
   -> "Occupied area (0~1) is less than threshold, occupied area is A, threshold is T"
7. scale: `EstimateScaleByLinearFitting(t_lc_rot, t_cr, baseline * dir, &scale, &err)`
   (or `EstimateScaleLinearSearch` if `init_scale_use_linear_search`).
   If `scale > scale_upper_limit || scale < 0`:
   `early_return_invalid_scale ? reject : scale = 0.47`
8. `t_lc *= scale`
9. `RelativePoseRefine(res, refine_data)` when `max_mean_point_to_epipolarline_error >= 0`:
   * `StereoPoseRefineSolver::SolveProblem(...)` (Ceres) -> "Pose refine solve failure"
   * `||t_refined|| <= scale_upper_limit`
     -> "Scale too large after pose refinement, estimated: X, greater than threshold: Y"
   * `mean_epipolar_error <= max_mean_point_to_epipolarline_error`
     -> "Mean point to epipolar line error is X, greater than threshold Y"
10. optional `||t|| <= relative_pose_translation_threshold_meters` (disabled, 0)
11. optional `2*acos(q.w)*180/pi <= relative_pose_rotation_threshold_degrees` (disabled, 0)

**Measured failure histogram** (`data/cusfm_re/runs/assoc_pg_v3/verbose_run.log`, `--v=3`,
robocap_full): 180 candidates reached step 9; **168 failed on the mean-epipolar-line-error gate,
12 on "Scale too large after pose refinement"**; 0 loop associations survived. Sample:
`Mean point to epipolar line error is 0.000599097, greater than threshold 1e-05`. The measured
errors sit two orders of magnitude above `10e-6`. The same result (0 loop associations) appears
in `robocap_full`, `robocap_v4` (tiny vocabulary, 2555 candidates) and `robocap_assoc`.

Interpretation: `max_mean_point_to_epipolarline_error: 10e-6` is expressed in normalised-plane
units, i.e. roughly `1e-5 * f ~ 0.004 px` for a 400 px focal length. On this dataset nothing can
pass it. **A Python reimplementation must not copy this number.** Anything in the
`1e-4 .. 1e-3` normalised range (≈ 0.04–0.4 px) is a defensible starting point.

### 6.5 Step 3b — loop associations, **feature-matcher mode** (`RetrievalLoopAssociationsForFeatureMatcher`, association_worker.cc:~740-800)

```
for every keyframe q:
    kf_q = cacher[q]                            -> "Skip, cannot read keyframe: q"
    cands = DetectCandidates(kf_q.keypoints, min_shared_word_number, query_result_number)
    for c in cands:
        if c.score < good_score_threshold or c.frame_id == q: continue
        kf_c = cacher[c.frame_id]
        if kf_q.session_id == kf_c.session_id and
           abs(dt) < loop_interval_threshold_in_seconds:   continue     # same-session time gate
        rel = inv(camera_to_world[q]) * camera_to_world[c]
        if ||rel.t|| > pose_translation_threshold_meters or
           2*acos(rel.q.w) > radians(pose_rotation_threshold_degrees):
            LOG(V2) "Skip candidate C due to large pose difference: trans=Xm, rot=Yrad"; continue
        pair = (min(q, c), max(q, c))
        AddAssociationToGraph(pair, IDENTITY, LOOP, I6, {}, {}, 0)
```

CONFIRMED (decompiled; the SE3 handed to `AddAssociationToGraph` is explicitly reset to
translation `(0,0,0)` and quaternion `(0,0,0,1)` immediately before the call). Measured: all 231
LOOP entries in `data/cusfm_re/runs/assoc_v3/pose_graph.pb.txt` carry
`axis_angle { x: 1 }` (angle 0) and an empty translation.

There is no `SelectBestCandidates` here — every surviving candidate becomes a pair. The output is
a **matching task list**, which is exactly what `feature_matcher_task_builder_main` wants.

### 6.6 `AddAssociationToGraph(pair, T, type, W6x6, matches, inlier_mask, matched_num)`

```
state = AssociationState{ source, target, dt = t_source - t_target,
                          pose = T, covariance = W, type }
if type == LOOP and !skip_pose_graph:
    matches_ = matches (or GetKeyPointMatches(kf_src, kf_tgt) if empty)
    if an association already exists for `pair` with >= matched count: return   # dedup
    if inlier_mask empty or matched_num == 0:
        matched_num = DataSelector::SelectMatchesByOpenCVEssentialMat(
            kp_src, kp_tgt, proj_src, proj_tgt, matches_, &inlier_mask,
            match_worker_config.max_pixel_error,        # 4.0 px
            match_worker_config.min_ransac_confidence)  # 0.95
    DetermineValidityViaInlierMatches(matched_num, min_matches_num, min_matches_ratio)
        # state.matched_pairs_num = matched_num
        # state.is_good = (type == EXTRINSIC) ? true
        #                 : (matched_num > min_matches_num &&
        #                    matched_num / matches_.size() > min_matches_ratio)
else:
    state.is_good = true          # state.matched_pairs_num left UNINITIALISED
AssociationGraph::AddAssociationState(state)
```

CONFIRMED (decompiled `AddAssociationToGraph`, `DetermineValidityViaInlierMatches`).

### 6.7 Step 4 — connected-graph analysis (`GetMaxConnectedGraph`, association_worker.cc:1100-1140)

Only when `!skip_pose_graph`.

```
adjacency, visited = GetAdjacencyMapAndVisitMap()     # only is_good associations, undirected
components = []
for node in adjacency:
    if keyframe(node) is invalid: LOG "N frame is invalid, but in adjecency map. Should not happen!!!"
    elif not visited[node]: components.append(VectorUtils::BFS(node, visited, adjacency))
LOG "Found K connected graphs"                        # :1130
for c in components: LOG "Graph size: |c|"             # :1132
largest = argmax_size(components)
for every frame id in the metadata:
    keyframe.valid = (id in largest)
return representative_node(largest)                   # BFS root for RecoordinateKeyFrames
```

Then `UpdateAssociationStateByValidKeyFrame()` walks the association map and clears `is_good`
on any edge whose source or target keyframe is now invalid (or missing). Those edges **stay** in
`association/` (with `is_good: false`) but are **dropped** from `pose_graph.pb.txt`.

Measured on `robocap_full`: `Found 1 connected graphs`, `Graph size: 4528` — nothing dropped.
Cost 9.5 s, the second most expensive stage after loop retrieval.

### 6.8 Step 5 — `RecoordinateKeyFrames(root)`

BFS from `root` over the good-association adjacency map; for each newly reached neighbour,
compose the edge's SE3 with the current frame's pose and overwrite the neighbour's pose, marking
it recoordinated. Frames never reached log
"Frame N is not recoordinated, which should not happen."

This makes the in-memory poses exactly consistent with the constraint spanning tree.
**It has no observable effect**: `main` never persists keyframes and `frames_meta.json` is not
modified (measured by mtime). CONFIRMED. A Python reimplementation can skip it entirely.

### 6.9 Runtime profile (robocap_full, 4528 keyframes, one GPU)

| stage | wall time |
|---|---|
| `RetrievalExtrinsicAssociations` | 3.6 s |
| `RetrievalConsecutiveAssociations` | 5.8 s |
| `RetrievalLoopAssociations` | 45.4 s (10.0 ms per frame x 4528) |
| `GetMaxConnectedGraph` | 9.5 s |
| `UpdateAssociationStateByValidKeyFrame` | 4.7 s |
| `RecoordinateKeyFrames` | 2.0 s |
| save 11316 protos + `pose_graph.pb.txt` | ~13 s |
| **total** | ~75 s |

Per candidate: `GetKeyPointMatches` 9.9 ms (LightGlue TensorRT end-to-end 8.7 ms),
`PoseEstimateFromTwoViewNormalized` 45.4 ms, `RelativePoseRefine` 12.4 ms,
`CalculateRelativePoseViaStereoEstimator` 123 ms.

---

## 7. Fast happy path for a Python reimplementation

### 7.1 Must keep

1. **The output contract.** Sorted `frame_pair`; `pose_source_to_target = T_target_from_source`;
   axis-angle in **degrees**; 6x6 row-major weight matrix; `pose_graph.pb.txt` containing only
   good pairs and only three fields; per-pair files named `<src>_<tgt>.pb` under
   `association/`. Downstream (`pose_graph_association_main`,
   `feature_matcher_task_builder_main`) reads exactly these.
2. **Extrinsic and consecutive edges are pure bookkeeping.** `inv(T_tgt) @ T_src` from
   `camera_to_world`, weight `I6`, no matching, no image reads. 11316 of the 11316 edges on
   RoboCap come from here. This is ~10 lines of numpy and reproduces the blob bit-for-bit.
3. **`consecutive_frame_count` semantics**: count *successful* hops, not rig offsets.
4. **BoW candidate generation gates**: `min_shared_word_number = 50`,
   `query_result_number = 20`, `good_score_threshold = 0.1`, plus the
   `loop_interval_threshold_in_seconds = 10` temporal gate.
5. **`SelectBestCandidates`**: one candidate per 10 s band of `|dt|`, ranked by inlier count then
   BoW score. This is what stops a single revisit producing 20 near-duplicate loop edges.
6. **The `is_good` rule for LOOP edges**: `inliers > 30 && inliers/matches > 0.25`.
7. **The information-matrix convention**: `inliers * loop_residual_weight * I6` for LOOP,
   `I6` otherwise. `docs/spec/pose_graph_main.md` needs this to weight residuals the same way.
8. **Largest-connected-component pruning**, if the PGO stage assumes a single component.

### 7.2 Can drop or replace

| blob behaviour | recommendation |
|---|---|
| `RecoordinateKeyFrames` | drop — no observable effect |
| `--matching_worker_config` | drop — read and discarded |
| the 4-view stereo estimator (left + right + candidate, baseline-locked scale, `StereoPoseRefineSolver`) | replace with a plain 2-view essential-matrix estimate between query and candidate; recover metric scale afterwards from the given poses or leave the loop edge scale-free and let the PGO handle it |
| `max_mean_point_to_epipolarline_error: 10e-6` | **do not copy** — it rejects 168/180 candidates on RoboCap. Use ~`1e-4`..`1e-3` normalised, or express the gate in pixels (0.5–2 px mean symmetric epipolar distance) |
| `scale = 0.47` fallback when the initial scale is out of range | drop; reject the candidate instead (i.e. behave as `early_return_invalid_scale: true`) |
| `matched_pairs_num` for non-LOOP edges | write `0` instead of leaving garbage; nothing downstream should read it, but garbage is worse |
| SIFT duplicate removal, homography check, multi-camera voting, cuDSS solver | never exercised with the isaac config |

### 7.3 Concrete Python building blocks

* **Retrieval**: reuse the BoW index reader from `docs/spec/bow.md`; the score threshold and the
  shared-word gate are trivially vectorisable with scipy sparse.
* **Matching**: LightGlue via the existing TensorRT engine in `pycusfm/models/aliked_lightglue/`
  (`tools/` already wraps it), or `lightglue` on torch. Keep `match_threshold=0.3` and the
  top-500 cap so the inlier ratios stay comparable to the blob.
* **Two-view geometry**: `pycolmap.estimate_essential_matrix` /
  `pycolmap.TwoViewGeometry` with `pycolmap.TwoViewGeometryOptions` (ransac
  `max_error` in pixels, `confidence=0.999`), or
  `cv2.findEssentialMat(p0, p1, np.eye(3), cv2.USAC_MAGSAC, prob=0.999, threshold=thr_norm)`
  followed by `cv2.recoverPose`. The blob uses exactly `findEssentialMat` + `recoverPose` on
  **normalised** points with `K = I3`, so a like-for-like port is `cv2`, and the threshold must
  be divided by the focal length.
  For fisheye keyframes use the camera model to unproject to bearing vectors first
  (`cv2.fisheye.undistortPoints`) — the blob does the same through
  `BaseCameraProjector` / `MapErrorToNormalizedPlane`.
* **Scale**: with `camera_to_world` available (`init_pose_mode: GIVEN`), take the loop edge
  translation direction from `recoverPose` and the magnitude from the prior poses, or run a
  small `scipy.optimize.least_squares` on the reprojection of the triangulated inliers. The
  blob's linear fit against the known stereo baseline is only needed because it deliberately
  refuses to trust the prior pose.
* **Refinement**: `scipy.optimize.least_squares` on the 5-dof epipolar residual
  (Sampson distance) over the inliers reproduces `StereoPoseRefineSolver` closely enough; gate on
  the mean Sampson distance in pixels.
* **Connected components**: `scipy.sparse.csgraph.connected_components`.
* **Rotations**: `scipy.spatial.transform.Rotation`; remember the proto stores **degrees**.

### 7.4 Suggested milestone order

1. Extrinsic + consecutive writer, byte-compared against
   `data/cusfm_runs/robocap_full/cusfm/associations/` (should match to float rounding).
2. Feature-matcher-mode loop pairs (BoW + pose-prior gate) — compare the 231-pair set in
   `data/cusfm_re/runs/assoc_v3/`.
3. Pose-graph-mode loop edges with a *sane* epipolar gate, evaluated by whether the PGO stage
   actually reduces ATE on RoboCap. There is no blob reference to match here, because the blob
   produces none.

---

## 8. Evidence index

Decompiled sources (Ghidra 12.1.3, output under
`/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish/data/cusfm_re/decomp/generate_association_main/`):

| claim | file |
|---|---|
| stage order, `use_frame_selection` / `skip_pose_graph` guards | `nvidia__isaac__visual__cusfm__AssociationWorker__RetrievalAssociations.c` |
| rig pair enumeration, `I6`, EXTRINSIC | `..._RetrievalExtrinsicAssociations.c` |
| forward hop counting | `..._RetrievalConsecutiveAssociations.c` |
| same-camera pairing, `inv(T_tgt)*T_src` | `..._ProcessConsecutiveCandidates.c` |
| BoW gates, time gate, early abort, LOOP weight `inliers*w*I6` | `..._RetrievalLoopAssociationsForFrame.c` |
| feature-matcher mode, pose-prior gate, identity pose | `..._RetrievalLoopAssociationsForFeatureMatcher.c` |
| `|dt|` bucketing and inlier/score ranking | `..._SelectBestCandidates.c` |
| two LightGlue calls, baseline lookup | `..._CalculateRelativePoseViaStereoEstimator.c` |
| `MatchKeypoints` with no epipolar prior | `..._GetKeyPointMatches.c` |
| dedup, essential-matrix fallback, `is_good` | `..._AddAssociationToGraph.c` |
| `is_good` formula, `matched_pairs_num` write | `nvidia__isaac__visual__general__AssociationState__DetermineValidityViaInlierMatches.c` |
| proto field mapping, degrees, row-major covariance | `nvidia__isaac__visual__general__AssociationState__ToProto.c` |
| only good pairs, only 3 fields | `..._SaveAsPoseGraph.c` |
| `<src>_<tgt>.pb` under `association/` | `..._SaveAssociationToFile.c` |
| BFS components, keyframe validity | `..._GetMaxConnectedGraph.c`, `..._GetAdjacencyMapAndVisitMap.c` |
| in-memory pose rewrite | `..._RecoordinateKeyFrames.c` |
| the full gate ladder | `nvidia__isaac__visual__cuvgl__StereoPoseEstimator__EstimateRelativePose__at_0x15c8e0.c` |
| `findEssentialMat` + `recoverPose` on normalised points | `..._PoseEstimateFromTwoViewNormalized.c` |
| `MapErrorToNormalizedPlane`, occupied area | `..._NormalizedPointAndComputePose__at_0x15ab00.c` |
| scale + epipolar-error gates | `..._RelativePoseRefine.c` |
| shared-word filter, top-k truncation | `nvidia__isaac__visual__loop_closing__LoopClosureIndex__DetectCandidates.c`, `..._CalculateCandidatesScore__at_0x18d120.c` |
| `--matching_worker_config` discarded | `objdump -d --start-address=0x3dca0 --stop-address=0x3dd60` |
| matcher / rig / baseline setup | `objdump -d --start-address=0x51010 --stop-address=0x51400` |

Log-string anchors (`association_worker.cc`): `:28` "One of the input pointers is null.",
`:132` "Retrieval N associations.", `:146` "Failed to save pose graph for feature matcher!",
`:152` "Saving all the association in graph to the folder", `:396`/`:464` "No vehicle frames
found.", `:406` "Processing extrinsic associations", `:454` "N extrinsic associations are
retrieved.", `:472` "Processing consecutive associations", `:504` "N consecutive associations
are retrieved.", `:513` "No frames found.", `:520` "Processing loop associations", `:532` "N
loop associations are retrieved.", `:793` "Skip candidate N due to large pose difference",
`:836` "Skip, due to invalid frame id or cannot read keyframes", `:843` "Skip, due to frame N is
not a left camera frame", `:881` "Skip, can't read keyframe", `:978` "Skip, invalid keyframe with
projector.", `:992` "Skip, cannot get keypoint matches.", `:999` "Baseline not found for camera
params ID", `:1014` "Skip, cannot estimate relative pose", `:1116` "frame is invalid, but in
adjecency map. Should not happen!!!", `:1130` "Found N connected graphs", `:1132` "Graph size:",
`:1288` "Association type is not supported.", `:1312` "Pose graph relationships are saved to".

Log-string anchors (`stereo_pose_estimator.cc`): `:503` "Two view verification need at least 5
matched point.", `:680` "get relative_transform failed.", `:684` "Inlier match indices size:",
`:846` "Pose refine solve failure", `:852` "Scale too large after pose refinement, estimated:",
`:860` "Mean point to epipolar line error is".

Artifacts and measurements:

* `data/cusfm_runs/robocap_full/cusfm/associations/` — 11316 pairs, `associations_main.txt`
  (stage order + profile timers), `pose_graph.pb.txt`.
* `data/cusfm_runs/robocap_v4/…`, `data/cusfm_runs/robocap_assoc/…` — same zero-loop result with
  a different vocabulary (2555 stereo-estimator calls, still 0 loop edges).
* `data/cusfm_re/runs/assoc_v3/` — my `--skip_pose_graph=true --v=3` rerun: 231 LOOP pairs with
  identity poses, `verbose_run.log`.
* `data/cusfm_re/runs/assoc_pg_v3/` — my `--v=3` pose-graph-mode rerun: 168 mean-epipolar-error
  rejections, 12 scale rejections, 0 LOOP, `verbose_run.log`.
* `data/cusfm_re/tools/parse_assoc.py`, `scan_assoc.py`, `verify_pose.py`, `verify2.py` —
  the scripts used for the proto and pose-convention measurements
  (`pixi run -e raco python <script>`).
* Proto schema: `data/cusfm_re/protos/generate_association_main.fdset`, rendered with
  `data/cusfm_re/tools/fdset_to_proto.py`.
* Ghidra recipe: project `/tmp/cusfm_ghidra_assoc/ga`, scripts
  `data/cusfm_re/ghidra_scripts/DecompMatching.java` (name substrings) and
  `data/cusfm_re/ghidra_scripts/DecompAt.java` (entry addresses, added here to disambiguate
  overloaded symbols such as the two `EstimateRelativePose`).

---

## 9. Open questions

1. **Where do the camera-to-camera extrinsics come from?**
   `KeyframesMetadataCollection::GetCameraToCameraTransforms()` returns a ready-made
   `map<FramePairID(camA,camB), SE3>`. On RoboCap the values equal `inv(T_tgt) @ T_src` of the
   first rig exactly, so either the map is built from the first rig or the poses were themselves
   generated from a calibration. Which one matters if a dataset ever has non-rigid per-rig
   poses. Not resolved.
2. **`init_pose_mode: STEREO` / `MONO`.** Only `GIVEN` was exercised. The other modes presumably
   estimate the consecutive pose from imagery instead of the prior, which would change stage 2
   substantially. Not investigated.
3. **The `<rig field at +8>` session key** used to break the consecutive walk and to gate the
   feature-matcher-mode time check is `keyframe+0x50` / `vehicle_frame+0x8`. It is almost
   certainly a session or sequence id from a multi-session map. Untested here (single session).
4. **`StereoPoseRefineSolver::SolveProblem` cost function.** I confirmed its two output gates but
   not the exact residual (it takes both match sets, the baseline, and a rotation, and returns a
   refined pose plus a mean epipolar error). A faithful port would need it; the recommendation in
   7.2 is to substitute a Sampson-distance refinement instead.
5. **Is `matched_pairs_num` read anywhere downstream?** If `pose_graph_association_main` or the
   task builder reads it from the per-pair `.pb` files, the uninitialised values for
   EXTRINSIC/CONSECUTIVE are a latent bug in the blob. `pose_graph.pb.txt` does not carry the
   field, so the risk is limited to consumers of `association/*.pb`.
6. **Match index orientation on LOOP edges.** `res.matches` holds
   `(query_left_kp_idx, candidate_kp_idx)`, but the emitted `frame_pair` is sorted, so when the
   candidate id is smaller than the query id the pose is inverted while the match indices are
   not. Either the blob has an index-order bug on those edges or `FeatureMatch` is interpreted
   symmetrically downstream. Unverified — no LOOP edge with matches was ever produced on the
   available data.
7. **Was the `10e-6` epipolar threshold ever exercised?** Three independent runs produce zero
   loop edges, and `configs/av/association_config.pb.txt` carries the *same*
   `stereo_estimator_config` block (diffed: the only differences are
   `loop_interval_threshold_in_seconds: 5`, `good_score_threshold: 0.01`,
   `lightglue_params.match_threshold: 0.8`, and no `aliked_tensorrt_config`). Since the `av`
   preset also sets `skip_pose_graph: True`, the pose-graph-mode loop path may never run in any
   shipped configuration. Whether NVIDIA ever produced a LOOP edge with a measured pose is
   unresolved.

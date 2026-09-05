# `pose_graph_main` and `pose_graph_association_main` — implementation spec

Status: reverse-engineered from `pycusfm/x86_cuda13/bin/pose_graph_main` and
`pycusfm/x86_cuda13/bin/pose_graph_association_main` (unstripped ELF, C++ symbols, embedded
source paths `/home/jryu/workspaces/visual_mapping/src/...`), the shipped configs, and the
completed runs under `data/cusfm_runs/`. Evidence and confidence are marked per claim:
**[C]** confirmed (code, decompiled body, string literal, or exact numeric match against a real
artifact), **[L]** likely, **[I]** inferred.

This spec is meant to be sufficient to reimplement the stage in Python without further RE.

---

## 1. Purpose and position in the pipeline

`pose_graph_main` is stage 4. It takes the keyframe metadata produced by
`feature_extractor_main` / `keyframe_aggregation_main` (`kpmap/keyframes/frames_meta.json`,
poses from cuVSLAM ego-motion), optionally searches for loop closures using the BoW vocabulary
and index from stage 2, builds a **rig-level (vehicle-frame) pose graph**, optimises it with
Ceres, and writes an updated `frames_meta.json` that the downstream matcher
(`feature_matcher_task_builder_main`) and mapper (`keypoints_mapper_main`) consume.

```
keyframes/frames_meta.json  ─┐
cuvgl_map/vocabulary/*       ├─► pose_graph_main ─► pose_graph/frames_meta.json  (camera poses, updated)
cuvgl_map/bow_index.pb      ─┤                     pose_graph/pose_graph.pb.txt         (camera-frame edges)
configs/<profile>/          ─┘                     pose_graph/vehicle_frames_meta.json  (rig poses)
                                                   pose_graph/vehicle_pose_graph.pb.txt (rig edges)
                                                   pose_graph/vehicle_pose.tum
```

`pose_graph_association_main` is the **multi-track** variant. It skips its own loop search and
instead consumes the `ConstraintPair` files written by `generate_association_main`, converts
them from camera frames to vehicle frames, and runs the identical optimiser. Selection in
`pycusfm/cusfm_runner.py:1015-1019`:

```python
if not self.skip_pose_graph:
    if self.multi_track_input:
        self.association_graph()      # pose_graph_association_main
    else:
        self.pose_graph()             # pose_graph_main
```
**[C]**

### Headline finding for the Python port

With the **shipped** `isaac` and `av` configs, `pose_graph_main` **never produces a loop edge**:
`loop_edge_translation_threshold_meters` and `loop_edge_rotation_threshold_degrees` are unset
(proto3 default `0.0`) and the acceptance test rejects any candidate whose relative-pose
magnitude *exceeds* the threshold (§6.4). Consequently the graph is a pure odometry chain with
one fully-fixed anchor node, and the optimum reproduces the input trajectory exactly.

This was **confirmed dynamically**, not just read off the disassembly (§9.2): re-running the blob
on `data/cusfm_runs/galileo/cusfm/keyframes` (226 keyframes, 29 rig frames) with the stock
`pycusfm/configs/isaac` produces `Perform pose graph optimization with 0 loop constraints`, while
the same run with only `loop_edge_translation_threshold_meters: 5.0` and
`loop_edge_rotation_threshold_degrees: 60.0` appended produces 8 loop edges. **[C]**

(The archived `data/cusfm_runs/galileo/cusfm/pose_graph/pose_graph_main.txt` shows 8 loops —
that run used a locally patched config, not the shipped one. My rerun with the two thresholds set
reproduces its loop set and its constraint counts exactly, line for line.)

It is *not* quite a pass-through, though: even with zero loops the stage **rigidifies the rig**.
Camera poses are re-derived from the optimised rig pose, so any per-camera inconsistency in the
input (calibration noise between the cuVSLAM camera poses and the nominal
`sensor_to_vehicle_transform`) is snapped away. On the 226-frame galileo set that moves camera
positions by at most 15 µm (mean 2.6 µm) — negligible, but not zero. **[C]**

A Python port therefore has two tiers:
* **Tier 1 (parity with the blob, ~30 lines):** derive each rig pose, re-emit `frames_meta.json`
  with rigidified camera poses, and write the four extra output files. This reproduces the blob
  to ~15 µm.
* **Tier 2 (the useful stage):** implement the real PGO described below so that loop closures
  actually help. That is what the rest of this document specifies.

---

## 2. CLI flags

### `pose_graph_main` (`pycusfm/cusfm_runner.py:596-618`) **[C]**

| Flag | Passed as | Meaning |
|---|---|---|
| `--keyframe_directory` | `<cusfm>/kpmap/keyframes` | Directory holding `frames_meta.json` and the per-frame keyframe protos. Required. |
| `--match_directory` | `<cusfm>/matches` | Directory of association relationships. Required by gflags; unused on the default path (matches do not exist yet at this point in the pipeline). |
| `--voc_directory` | `<cusfm>/cuvgl_map` | Contains `vocabulary/` and `bow_index.pb`. Required. |
| `--config_directory` | `pycusfm/configs/<profile>` | **Four** protos are read from here: `pose_graph_config.pb.txt` (`InitGraph` joins the literal `"/pose_graph_config.pb.txt"`), `localizer_config.pb.txt`, `matching_config.pb.txt` and `keypoint_creation_config.pb.txt`. See §3.1. |
| `--pose_graph_directory` | `<cusfm>/pose_graph` | Output directory. |
| `--model_dir` | `pycusfm/models` | TensorRT/ONNX models for the LightGlue matcher used during loop verification. |
| `--debug_dir` | `<pose_graph>/debug` (debug only) | When set, per-iteration `vehicle_pose_graph.pb.txt` / `vehicle_frames_meta.json` dumps are written with the constraint count in the name. |
| `--raw_dir` | `<input_dir>` (debug only) | Source images for debug match visualisations. |
| `--max_buffer_size` | not passed (default 2000) | Keyframe LRU cache size (`KeyframeCacher`). |

### `pose_graph_association_main` (`pycusfm/cusfm_runner.py:623-641`) **[C]**

| Flag | Passed as | Meaning |
|---|---|---|
| `--association_directory` | `<cusfm>/associations` | Directory written by `generate_association_main`; the per-pair `ConstraintPair` files live in its `association/` subfolder. |
| `--pose_graph_directory` | `<cusfm>/pose_graph` | Output directory (same layout as above). |
| `--pose_graph_config` | `pycusfm/configs/<profile>/pose_graph_config.pb.txt` | The `PoseGraphConfig` proto, passed as a file rather than a directory. |
| `--debug_directory` | `<pose_graph>/debug` (debug only) | Debug dumps. |
| `--max_buffer_size` | not passed (default 2000) | Keyframe cache size. |

---

## 3. Config: `protos.visual.cusfm.PoseGraphConfig`

Schema recovered from the binary's `protodesc_cold` section
(`data/cusfm_re/protos/pose_graph_main.fdset`, rendered by
`data/cusfm_re/tools/fdset_to_proto.py`). Defaults below are the shipped
`pycusfm/configs/isaac/pose_graph_config.pb.txt`; `av` differences are noted. **[C]**

```proto
message PoseGraphConfig {
  int32  connected_keyframe_num                 = 1;
  int32  loop_closure_min_words                 = 2;
  int32  loop_closure_query_count               = 3;
  double loop_closure_interval_ratio            = 4;
  protos.visual.geometry.BundleAdjustmentConfig optimization_config = 5;
  bool   use_incremental                        = 6;
  double loop_residual_weight                   = 7;
  double good_loop_threshold                    = 8;
  int32  num_votes                              = 9;
  bool   use_switchable_pgo                     = 10;
  double confidence_to_pose_error_ratio         = 11;
  double loop_edge_prune_threshold              = 12;
  double point3d_max_error                      = 13;
  double loop_edge_translation_threshold_meters = 14;
  double loop_edge_rotation_threshold_degrees   = 15;
  bool   use_sequential_edge_pose_estimation    = 16;
}
```

| Field | isaac (`pose_graph_config.pb.txt`) | av | Effect | Conf |
|---|---|---|---|---|
| `connected_keyframe_num` | 1 (L1) | 1 | Number of *following* rig frames each rig frame is linked to by a sequential edge. `RunPoseGraph` aborts with `"ERROR: Keyframe_connection_count is N, which should be larger than 1."` when `< 1`. | **[C]** |
| `loop_closure_min_words` | 50 (L2) | 50 | `min_shared_words` argument to `LoopClosureIndex::DetectCandidates`. | **[C]** |
| `loop_closure_query_count` | 20 (L3) | 20 | `top_k` argument to `LoopClosureIndex::DetectCandidates`. | **[C]** |
| `loop_closure_interval_ratio` | 0.08 (L5) | 0.05 | Multiplied by the session duration (s) to give the minimum time gap between query and candidate. | **[C]** |
| `optimization_config` | see §7 | same | Ceres settings. | **[C]** |
| `use_incremental` | true (L6) | true | `RunIncrementalPoseGraph` vs `RunGlobalPoseGraph`. | **[C]** |
| `loop_residual_weight` | 0.25 (L7) | 0.25 | Scales the loop edge's information matrix: `Λ = I₆ · (score · loop_residual_weight)`. | **[C]** |
| `good_loop_threshold` | 150 (L8) | 100 | Minimum two-view **inlier count** for a loop candidate to be accepted. | **[C]** |
| `num_votes` | 1 (L9) | 1 | How many loop edges to keep per rig frame after cross-camera voting. | **[C]** |
| `use_switchable_pgo` | unset → false | unset → false | Enables Sünderhauf switchable constraints on loop edges + a second solve after pruning. | **[C]** |
| `confidence_to_pose_error_ratio` | unset → 0.0 | unset → 0.0 | Weight `Ξ` of the switch prior; `sqrt(Ξ)` multiplies `(1 − s)`. Logged as `GetConfidenceToPoseErrorRatio:0`. | **[C]** |
| `loop_edge_prune_threshold` | unset → 0.0 | unset → 0.0 | After the first solve, loop edges with switch value `< threshold` are dropped. Logged `total pruned N loop edges at threshold T`. | **[C]** |
| `point3d_max_error` | unset | unset | Not used on this path. | **[I]** |
| `loop_edge_translation_threshold_meters` | **unset → 0.0** | unset → 0.0 | Loop candidate rejected if `‖t‖ > threshold`. With 0.0 this rejects everything. | **[C]** |
| `loop_edge_rotation_threshold_degrees` | **unset → 0.0** | unset → 0.0 | Loop candidate rejected if `angle_deg > threshold`. Same problem. | **[C]** |
| `use_sequential_edge_pose_estimation` | unset → false | unset → false | When true, sequential edges are re-estimated from images (`EsimateSequentialEdgePose`) instead of using the input odometry. | **[C]** |

Offsets were pinned by decoding `protos::visual::cusfm::PoseGraphConfig::_InternalSerialize`,
which maps field numbers to member offsets; the embedded `PoseGraphConfig` starts at
`PoseGraph + 0x2c0` (`InitGraph` writes the parsed proto to `this + 0x2c0`). Every offset used
in the decompiled logic (`0x2d8` → field 1, `0x2ec` → field 9, `0x2f8` → field 8, `0x300` →
field 11, `0x308` → field 12, `0x318` → field 14, `0x320` → field 15, `0x328/0x329/0x32a` →
fields 6/10/16, `0x2d0` → field 5) matches that table. **[C]**

### 3.1 The other three config files

The string table contains exactly four `*.pb.txt` names joined against `--config_directory`:
`/pose_graph_config.pb.txt`, `/localizer_config.pb.txt`, `/matching_config.pb.txt`,
`/keypoint_creation_config.pb.txt`. **[C]**

* **`localizer_config.pb.txt`** → `protos.visual.cuvgl.KeypointsLocalizerConfig`.
  `PoseGraph::InitStereoEstimator` reads the whole file and copies only the
  `stereo_pose_estimator_config` submessage into the `StereoPoseEstimator`, then calls
  `InitCalibParam`. **[C]** Everything else in that file (PnP, the localiser's own BA) is unused
  here. The thresholds that therefore govern **loop verification** are, from
  `pycusfm/configs/isaac/localizer_config.pb.txt:56-96`:

  | Field | isaac | Effect |
  |---|---|---|
  | `ransac_max_pixel_error` | 1 | Essential-matrix RANSAC threshold, px. |
  | `ransac_confidence` | 0.999 | RANSAC confidence. |
  | `distance_3d_threshold` | 1e9 | Effectively disabled. |
  | `scale_upper_limit` | 4 | Reject relative poses whose recovered scale exceeds this. |
  | `min_inlier_number` | 25 | Minimum inliers inside the estimator (note: the *pose-graph* gate is the much stricter `good_loop_threshold` = 150). |
  | `min_occupied_area_ratio` | 0.10 | Reject when matches concentrate in a small image region. |
  | `max_mean_point_to_epipolarline_error` | 10e-6 | Refinement convergence bound. |
  | `vote_translation_error_meter_tolerance` | 0.5 | Cross-camera agreement, metres. |
  | `vote_rotation_error_degree_tolerance` | 5 | Cross-camera agreement, degrees. |
  | `allow_best_single_inlier` / `allow_single_camera_result` | false / false | A result from a single camera is rejected. |

  The same message is inlined in `association_config.pb.txt` for
  `generate_association_main`, with identical values — see
  `docs/spec/generate_association_main.md`.
* **`matching_config.pb.txt`** → `protos.common.image.KeypointMatchingParams`: the LightGlue
  matcher used during loop verification (`match_threshold: 0.3`, `match_top_k: 500`,
  `global_non_max_method: SUPPRESSION_VIA_SQUARE_COVERING`,
  `num_points_tolerance_fraction: 0.3`, ALIKED TensorRT profile). The run log confirms it:
  `lightglue_matcher.cc:38 LightGlueMatcher initialized with global non-max method:
  SUPPRESSION_VIA_SQUARE_COVERING, top_k: 500, num_points_tolerance_fraction: 0.3`. **[C]**
* **`keypoint_creation_config.pb.txt`** → `protos.common.image.KeypointCreationParams`. Read for
  the detector metadata; no keypoints are extracted at this stage (they come from the keyframe
  protos). **[L]**

`image_retrieval_config.pb.txt` is **not** read by `pose_graph_main` — the only
`ImageRetrievalConfig` instances in the shipped configs are nested inside
`association_config.pb.txt` and `query_map_config.pb.txt`, neither of which this binary opens.
Verified dynamically: appending `pose_translation_threshold_meters: 100` and
`pose_rotation_threshold_degrees: 100` to `image_retrieval_config.pb.txt` leaves the loop set
unchanged at 8 edges. The BoW query parameters come from `PoseGraphConfig`
(`loop_closure_query_count`, `loop_closure_min_words`). **[C]**

---

## 4. Input format

### `frames_meta.json` — `protos.visual.general.KeyframesMetadataCollection` in protobuf-JSON

Top-level keys observed in `data/cusfm_runs/galileo/keyframes/frames_meta.json`: **[C]**

* `keyframes_metadata[]` — one entry per **camera** keyframe:
  * `id` (uint64 as string) — global keyframe id.
  * `camera_params_id` (uint32 as string, **omitted when 0**) — index into
    `camera_params_id_to_camera_params`.
  * `timestamp_microseconds` (uint64 as string).
  * `image_name` — `<camera_folder>/<timestamp_ns>.jpeg`.
  * `camera_to_world` — `protos.common.geometry.RigidTransform3d`:
    `axis_angle {x, y, z, angle_degrees}` + `translation {x, y, z}`. **Units: axis is a unit
    vector, the angle is in DEGREES, translation in metres.** The transform is
    `world_T_camera` — it maps camera coordinates into world coordinates (verified numerically,
    §9). **[C]**
  * `synced_sample_id` (uint64 as string) — the **rig / vehicle frame id**. All cameras of one
    synchronised capture share it. This is the node id of the pose graph.
* `initial_pose_type` — `EGO_MOTION` for the shipped runs.
* `camera_params_id_to_session_name`, `camera_params_id_to_camera_params` — per-camera
  `sensor_meta_data` (with `sensor_to_vehicle_transform`, again a `RigidTransform3d` in degrees,
  i.e. `vehicle_T_camera`) and `calibration_parameters` (image size, 3×4 projection matrix).
* `detector_type` / `descriptor_type` — `ALIKED_DETECTOR` / `ALIKED_DESCRIPTOR`.
* `stereo_pair[]` — `{left_camera_param_id (omitted when 0), right_camera_param_id,
  baseline_meters}`.

`KeyframesMetadataCollection::ComputeVehicleFrames()` groups camera keyframes by
`synced_sample_id` into `VehicleFrame` objects. Each `VehicleFrame` carries `frames_in_rig_`
(the camera keyframe ids) and `cameras_in_rig_` (their `camera_params_id`s), with a `CHECK` that
the two have equal size (`include/visual/general/frame.h:185`). **[C]**

**The rig pose is not stored in the JSON.** It is derived from the **lowest-id camera keyframe**
of the sample:

```
ref = min(frames_in_rig(s))
world_T_vehicle(s) = world_T_camera(ref) · vehicle_T_camera(camera_params_id(ref))⁻¹
```

Verified on all four galileo rigs: the lowest-id frame reproduces the exported
`vehicle_frames_meta.json` pose to 1e-16, while the other cameras of the same rig disagree by
7e-7 … 1.4e-5 m (per-camera calibration noise). The stereo partner of the reference frame
usually agrees to 1e-16 as well, since the pair shares a rectified extrinsic. **[C]** for the
rule, **[L]** for "lowest id" versus "first in insertion order" (they coincide in every shipped
artifact). The inverse relation — camera poses re-derived from the rig pose — is verified exactly
in §9.1. **[C]**

### Association files (for `pose_graph_association_main`)

`<association_directory>/association/<source_id>_<target_id>.pb` — one **binary**
`protos.visual.cusfm.ConstraintPair` each; see §5 for the message. Loaded by
`AssociationGraph::LoadAssociationGraphFromFiles`. **[C]**

---

## 5. Output format

All paths relative to `--pose_graph_directory`.

| File | Writer | Content |
|---|---|---|
| `frames_meta.json` | `PoseGraph::ExportUpdatedKeyframes` | The full input `KeyframesMetadataCollection` with every `camera_to_world` replaced by `world_T_vehicle(synced_sample_id) · vehicle_T_camera(camera_params_id)`. Everything else is byte-identical to the input. Logged `pose_graph.cc:806 Keyframes in pose graph are saved to ...`. **[C]** |
| `vehicle_pose.tum` | `PoseSerializer::ExportPoseToTumFile` (called from `ExportUpdatedKeyframes`) | One row per distinct keyframe timestamp: `t_seconds tx ty tz qx qy qz qw`, holding the **vehicle** pose of that rig sample. Timestamps are `timestamp_microseconds / 1e6` printed with 16 decimals. **[C]** |
| `pose_graph.pb.txt` | `PoseGraph::ExportPoseGraphInCameraFrame` | `protos.visual.cusfm.PoseGraph` text proto of the **camera-frame** edge set (CONSECUTIVE + EXTRINSIC, and LOOP when any). Logged `pose_graph.cc:744`. **[C]** |
| `vehicle_frames_meta.json` | `PoseGraph::ExportVehicleFrames` | A `KeyframesMetadataCollection` whose entries are the rig nodes: `id` = `synced_sample_id`, `camera_to_world` = the optimised `world_T_vehicle`. No timestamps or image names. Filename = prefix `"vehicle_"` + `"frames_meta.json"`. **[C]** |
| `vehicle_pose_graph.pb.txt` | `PoseGraph::ExportPoseGraphInVehicleFrame` | The **vehicle-frame** edge set actually fed to Ceres. Prefix `"vehicle_"` + `"pose_graph.pb.txt"`. **[C]** |

### `protos.visual.cusfm.ConstraintPair`

```proto
enum ConstraintType { UNKNOWN = 0; CONSECUTIVE = 1; LOOP = 2; EXTRINSIC = 3; }

message ConstraintPair {
  FramePair       frame_pair                 = 1;  // { source_frame_id, target_frame_id }
  repeated FeatureMatch matches              = 2;  // not written by pose_graph_main
  RigidTransform3d pose_source_to_target     = 3;
  MatrixD          pose_covariance           = 4;  // row_count=column_count=6, row-major
  ConstraintType   type                      = 5;
  repeated bool    matches_inliers           = 6;
  int32            matched_pairs_num         = 7;
  bool             is_good                   = 8;
  double           time_delta_source_to_target = 9;
}
message PoseGraph { repeated ConstraintPair pairs = 1; }
```

Conventions, all **[C]**:

* `pose_source_to_target` is `source_T_target`, i.e.
  `world_T_target = world_T_source · pose_source_to_target`. Verified to 6e-17 m on all three
  galileo vehicle edges (§9.1). `axis_angle.angle_degrees` is in **degrees**; translation in metres.
  **Caveat for the association path:** `docs/spec/generate_association_main.md` states the
  opposite convention (`inv(T_target) @ T_source`) for the edges written by
  `generate_association_main`. I could not settle that independently: the association
  `pose_graph.pb.txt` is written *after* `RecoordinateKeyFrames`, so its edges are ~4.8 m
  inconsistent with the on-disk `keyframes/frames_meta.json` in either direction, and its
  EXTRINSIC edges do not match the composition of the `sensor_to_vehicle_transform`s either
  (which is that spec's own open question 1). A Python port should **assert** the convention at
  load time — take one CONSECUTIVE edge, compose both ways against the node poses, and keep the
  one that closes — rather than hard-code it. **[C]** for `pose_graph_main`, **open** for
  `generate_association_main`.
* `pose_covariance` is written by `ConvertEigenMatrixToMatrixD` from a 6×6 `Eigen::Matrix<double,6,6>`
  and is consumed as an **information** matrix (see §7). Despite the name it is *not* a
  covariance in the code path that matters. It is left empty (`row_count = 0`) for CONSECUTIVE
  and EXTRINSIC edges written by `pose_graph_main`; the loader substitutes `I₆`.
* `pose_graph_main` writes only fields 1, 3, 5 (and 4 for loop edges). `generate_association_main`
  additionally writes 4, 7, 8, 9 — and `matched_pairs_num` is left uninitialised there (observed
  value `29316958`), so do not trust it.
* Proto3 zero fields are omitted in the text format, so a pair with `source_frame_id: 0` shows
  only `target_frame_id`.

### What is in the camera-frame `pose_graph.pb.txt`

For galileo (32 camera keyframes = 4 rig samples × 8 cameras): 24 `CONSECUTIVE` + 168
`EXTRINSIC`, no covariance fields. **[C]**
* `CONSECUTIVE` = 8 cameras × 3 rig links.
* `EXTRINSIC` = C(8,2)=28 per `AddExtrinsicCameraFrame` call; the incremental loop calls it for
  *both* endpoints of every sequential link, so the first and last rig sample get 28 edges and
  the interior ones 56 → 28+56+56+28 = 168. This exactly reproduces the artifact. **[C]**
* `LOOP` edges are mirrored into the camera-frame graph by
  `UpdatePoseGraphInCameraFrameLoops` after the solve. The 29-rig / 8-loop rerun of §9.2 yields
  218 CONSECUTIVE + 1533 EXTRINSIC + 8 LOOP. **[C]**

---

## 6. Algorithm

### 6.0 Data model

Two graphs are maintained side by side:

* `pose_graph_` (vehicle) — `PoseGraph` proto at `PoseGraph + 0x260`. Holds CONSECUTIVE and LOOP
  edges **between rig nodes**. This is the graph that is optimised.
* `camera_pose_graph_` — `PoseGraph` proto at `PoseGraph + 0x290`. Holds CONSECUTIVE and
  EXTRINSIC edges **between camera keyframes**. Purely a report/export artefact; it is never
  optimised. **[C]**

### 6.1 `main`

`ParseCommandLineFlags` → `InitGoogleLogging` → `LOG "Starting pose graph processing."` →
construct `PoseGraph` (which calls `InitGraph(...)`: read
`<config_directory>/pose_graph_config.pb.txt`, load `frames_meta.json`, build the camera
projectors, initialise the LightGlue `KeypointMatcher`, load `vocabulary/bow_vocabulary.pb.txt`
+ `bow_index.pb`) → `PoseGraph::RunPoseGraph()` → `LOG "Pose graph processing completed."` **[C]**

### 6.2 `RunPoseGraph`

```
if connected_keyframe_num < 1: log error, return 0
return use_incremental ? RunIncrementalPoseGraph(vehicle_frames)
                       : RunGlobalPoseGraph(vehicle_frames)
```
**[C]**

### 6.3 `RunIncrementalPoseGraph` (the default; `use_incremental: true`)

Decompiled body, `pose_graph.cc:320-424`:

```
clear pose_graph_                       # vehicle edge set
processed = set()                       # rig ids already visited
had_loop = false; total_loops = 0; seq_fallbacks = 0

for v in vehicle_frames (ascending rig id):
    normalise v.pose quaternion; flip sign so w >= 0
    ok = FindBestLoopCandidateForVehicle(v.id, processed, &ids, &poses, &covs)
    if not ok:
        if had_loop:                                   # a loop closed on the previous frame
            LOG "Perform pose graph optimization with %d loop constraints." % total_loops
            MultiOptimizePoseGraph()                   # re-solve now, mid-stream
        had_loop = false
    else:
        had_loop = true
        for k in range(len(ids)):
            AddConstraintPair(pose_graph_, v.id, ids[k], poses[k], covs[k], LOOP)
            LOG "Added loop constraint pair: %d -> %d"
        total_loops += len(ids)

    for j in 1 ..= connected_keyframe_num:             # the NEXT j rig frames
        w = vehicle_frames.iterator(v) advanced j times
        if w == end: break
        normalise w.pose quaternion
        if use_sequential_edge_pose_estimation:
            ok, rel = EsimateSequentialEdgePose(w, v)
            if not ok: rel = odometry relative pose; seq_fallbacks += 1
        else:
            rel = v.pose⁻¹ · w.pose                    # source_T_target from the input poses
        AddConstraintPair(pose_graph_, v.id, w.id, rel, empty, CONSECUTIVE)
        AddTimeConsecutiveCameraFrame(v.id, w.id, rel) # expand to per-camera edges
        AddExtrinsicCameraFrame(v.id)
        AddExtrinsicCameraFrame(w.id)

    processed.add(v.id)

LOG "Perform pose graph optimization with %d loop constraints. and seq link use input pose: %d"
MultiOptimizePoseGraph()
```
**[C]** (the mid-stream re-solve, the `AddExtrinsicCameraFrame` double call, and the
`_Rb_tree_increment` direction are all confirmed against the decompiled body and reproduce the
galileo edge counts exactly).

`RunGlobalPoseGraph` is the alternative: it adds all sequential edges first
(`"Added N continuous constraints based on M search depth."`), then all loop edges
(`"Added N loop constraints."`, `"Total added N loop constraints."`), then optimises once. **[C]**

#### `AddTimeConsecutiveCameraFrame(v1, v2, v1_T_v2)`

For every camera keyframe `a` in rig `v1` and `b` in rig `v2` **with the same
`camera_params_id`**, emit
`a_T_b = (vehicle_T_cam_b)⁻¹ · v1_T_v2 · (vehicle_T_cam_a)`
as a CONSECUTIVE edge in `camera_pose_graph_` with an empty covariance. **[C]**

#### `AddExtrinsicCameraFrame(v)`

For every unordered pair `(i < j)` of camera keyframes inside rig `v` with distinct
`camera_params_id`, look up the cached camera-to-camera transform and emit it as an EXTRINSIC
edge in `camera_pose_graph_` with an empty covariance. Missing entries log
`"No camera to camera transform found for frame pair: A and B"`. **[C]**

### 6.4 Loop closure search

#### `FindBestLoopCandidateForVehicle(rig_id, processed, &ids, &poses, &covs)`

Loops over the camera keyframes of the rig, calls `FindBestLoopCandidateForFrame` on each, and
then calls `LoopCandidateVoting` to reduce the per-camera results. **[C]**

#### `FindBestLoopCandidateForFrame(query_frame, processed, ...)`

1. `GetStereoPairFrameId(query)` and `IsLeftCamera(query)` — only the **left** camera of a stereo
   pair is used as a query; otherwise `"Skip, due to frame N is not a left camera frame"`. **[C]**
2. `KeyframeCacher::GetKeyFrame(query)`; failure → `"Skip, due to invalid frame id or cannot read
   keyframes"`. **[C]**
3. `LoopClosureIndex::DetectCandidates(query_keypoints, loop_closure_query_count,
   loop_closure_min_words, ...)` → up to `query_count` candidate frames sharing at least
   `min_words` vocabulary words. **[C]**
4. Time gate: `min_interval = loop_closure_interval_ratio · session_duration_seconds`, where the
   session duration is `max(timestamp_us) − min(timestamp_us)` over all keyframes, in seconds
   (`GetSessionDurationInSecond`, logged as `session duration is 150.842 seconds`). A candidate
   closer in time than `min_interval` is dropped with `"Skip, candidate frame too close to the
   query frame"`; if more than half the candidates are dropped this way the loop aborts early
   (`"Skip, more than half of the candidates are too close in time, exiting loop."`). **[C]**
5. `"Skip, candidate out of candidate search range"` — additional range gate on the candidate's
   rig id relative to the already-processed set. **[L]**
6. Two-view pose estimation: `StereoPoseEstimator::EstimateRelativePose(query_stereo_pair,
   candidate_stereo_pair, ...)` → LightGlue matches + essential-matrix RANSAC + scale from the
   stereo baseline + `RelativePoseRefine`. Returns a `StereoLocalizeResult` with an
   `inlier_number`. Failure → skip. **[C]**
7. **Acceptance gates**, applied in this order (all `[C]`, read straight from the disassembly at
   `0x5b338`–`0x5b3c4`):
   * `inlier_number >= good_loop_threshold`               (150 isaac / 100 av)
   * `‖t‖ <= loop_edge_translation_threshold_meters`      — `SE3Transform::Magnitude` first output
   * `angle_rad · 57.29577951308232 <= loop_edge_rotation_threshold_degrees` — second output,
     the constant is exactly `180/π`
   * `1.0 / ‖t‖ > best_score_so_far` — i.e. the score is `1/‖t‖` and only the best candidate for
     this query frame is kept ("FindBest…").
   With the shipped configs the middle two gates have threshold `0.0` and reject every candidate.
8. The accepted candidate's information matrix is
   `Λ = I₆ · (best_score · loop_residual_weight)` — the scalar is written to the six diagonal
   slots of a row-major 6×6 `MatrixD` (stack offsets `0x2e0, 0x318, 0x350, 0x388, 0x3c0, 0x3f8`,
   stride 0x38 = 7 doubles). Verified on a real loop edge from the §9.2 rerun
   (`data/cusfm_re/runs/pg_loop/vehicle_pose_graph.pb.txt`, edge 6→3): every diagonal entry is
   `3.6021773815155029` and every off-diagonal is `0`, with `row_count = column_count = 6`.
   Dividing by `loop_residual_weight = 0.25` gives `score = 14.408709526062012`, which is exactly
   a `float32` — matching the `vcvtss2sd` load of the score. **[C]**
   Note `1/score = 0.06940 m` while the *rig-frame* edge translation is `0.07345 m`: the score is
   `1/‖t‖` of the **camera-frame** estimate, computed before the rig conversion of step 9. **[C]**
   **Beware:** `generate_association_main` weights *its* loop edges differently —
   `Λ = num_inliers · loop_residual_weight · I₆` (see `docs/spec/generate_association_main.md`
   §1.1). So the same `pose_covariance` field carries an inlier-count-based weight on the
   association path and an inverse-distance weight on this one. A Python port should pick one
   scheme deliberately; the inlier-count version is the more standard choice.
9. The camera-frame relative pose is converted to the rig frame via
   `GetCameraToVehicleTransform` on both ends. **[C]**

#### `LoopCandidateVoting(ids, poses, covs)`

`score(k) = GetScoreFromPoseCovariance(covs[k]) = covs[k].data[0]` (the first element, i.e.
`best_score · loop_residual_weight`). **[C]**

* If `num_votes != 0`: push all candidates into a max-heap on `score` and pop `num_votes` of
  them; those indices are the accepted loop edges. With `num_votes: 1` this is "keep the single
  best loop candidate across all cameras of the rig". **[C]**
* If `num_votes == 0`: group candidates by target rig id, log
  `"Get multiple constraints between the same vehicle pairs."` for duplicates, and keep the
  max-score index per target. **[C]**

### 6.5 `MultiOptimizePoseGraph`

```
anchor_id = pose_graph_.pairs[0].frame_pair.source_frame_id
ok = OptimizePoseGraph()
if use_switchable_pgo:
    n_pruned = UpdatePoseGraphFromOpt()
    if n_pruned != 0:
        ok = OptimizePoseGraph()          # solve again without the pruned loop edges
return ok
```
**[C]**

**Quirk — constraints accumulate across mid-stream re-solves.** `PoseGraphData` lives at
`PoseGraph + 0x330` and is a *member*, and `OptimizePoseGraph` appends to it without clearing.
`ClearRelativeConstraint()` is only reached from `UpdatePoseGraphFromOpt`, i.e. only when
`use_switchable_pgo` is on. So with the default (`use_switchable_pgo` off) every mid-stream
re-solve re-adds the whole current edge set on top of the previous one. Observed in the 8-loop
rerun (§9.2): the graph sizes reported by `OptimizePoseGraph` are 11, 21, 29, 36, 36, while Ceres
receives 11, 32, 61, 97, 133 residual blocks — exactly the running sums. Earlier edges are
silently duplicated and over-weighted. Turning `use_switchable_pgo` on makes the counts match
(36 and 36), which confirms the mechanism. A Python port should **not** reproduce this; do a
single global solve. **[C]** (both halves measured, §9.3)

`UpdatePoseGraphFromOpt` rebuilds `pose_graph_` from the solver's `RelativePose` list: a
non-switchable constraint is re-added as CONSECUTIVE; a switchable one is re-added as LOOP only
if its optimised switch value `s >= loop_edge_prune_threshold`, otherwise it is dropped. Logs
`output loop_edge_confidence_weight:<s>` (VLOG 1) and
`total pruned N loop edges at threshold T`. **[C]**

### 6.6 `OptimizePoseGraph` (`pose_graph.cc:526-590`)

```
if pose_graph_.pairs.empty(): log "Pose graph is empty."; return 0
data = PoseGraphData<VehicleFrame>()
for pair in pose_graph_.pairs:
    data.AddFrame(pair.source_frame_id, vehicle_frames[source])
    data.AddFrame(pair.target_frame_id, vehicle_frames[target])
    rp = RelativePose{
        source_id, target_id,
        pose  = SE3(AxisAngleDegreeToQuaternion(pair.pose_source_to_target.axis_angle),
                    pair.pose_source_to_target.translation),
        info  = pair.has_pose_covariance ? MatrixD→Eigen6x6 : Identity6,
        switchable = (pair.type == LOOP) and use_switchable_pgo,
        switch_value = 1.0,
    }
    data.AddRelativeConstraint(rp)
data.SetPositionConstraintFrame(anchor_id)
data.SetConfidenceToPoseErrorRatio(confidence_to_pose_error_ratio)
LOG "Pose graph optimization will be performed, total_constraint_number: N, anchored keyframe id: A"
solver = BundleAdjustmentSolver(optimization_config)
ok = solver.SolvePoseGraphProblem<VehicleFrame>(data)
if ok: write the optimised poses back into the VehicleFrame objects
```
**[C]**

`RelativePose` is a 448-byte (0x1C0) POD: `{u64 source_id; u64 target_id; SE3Transform pose;
Matrix6d information @ +0x80; bool switchable @ +0x1A0; double switch_value @ +0x1A8}`. **[C]**

When `--debug_dir` is set, `ExportPoseGraphInVehicleFrame` and `ExportVehicleFrames` are called
before the solve with a filename containing the constraint count. **[C]**

### 6.7 `pose_graph_association_main` — `RunAssociationBasedPoseGraph`

```
clear pose_graph_
for cp in AssociationGraph::LoadAssociationGraphFromFiles(<association_directory>):
    if cp.type == EXTRINSIC: count it, skip           # intra-rig, nothing to optimise
    elif cp.type in (CONSECUTIVE, LOOP):
        v_a = vehicle_frame_of(cp.frame_pair.source_frame_id)
        v_b = vehicle_frame_of(cp.frame_pair.target_frame_id)
        va_T_vb = vehicle_T_cam(source) · cp.pose_source_to_target · vehicle_T_cam(target)⁻¹
        AddConstraintPair(pose_graph_, v_a, v_b, va_T_vb, cp.pose_covariance, cp.type)
    else: LOG ERROR "Unknown association type."
LOG "Added %d loop constraints. %d continuous constraints. %d extrinsic constraints."
MultiOptimizePoseGraph()
```
The transform composition is confirmed by the call sequence
`GetCameraToVehicleTransformByFrameId ×2 → Inverse ×2 → operator* ×2 → Inverse → AddConstraintPair`.
**[C]** for the calls, **[L]** for the exact left/right placement of the inverses.

Output files are identical to `pose_graph_main`'s. Multi-track handling is entirely upstream:
`generate_association_main` produces cross-track associations, and the tracks are already in a
common frame because `cusfm_runner` aligns each non-anchor track to the anchor's cuVGL map
before feature extraction (`align_track_to_cuvglmap`). **[C]**

---

## 7. The Ceres problem — `BundleAdjustmentSolver::SolvePoseGraphProblem<VehicleFrame>`

Address `0x102860`, size 0x164c. Disassembly saved at
`data/cusfm_re/disasm/SolvePoseGraphProblem_VehicleFrame.asm`.

### Parameterisation

Each rig node contributes two parameter blocks: a **unit quaternion** (4 doubles, at
`VehicleFrame + 0x20`) and a **translation** (3 doubles, at `VehicleFrame + 0x40`), holding
`world_T_vehicle`. `AddParameterBlock(q, 4, manifold)` is called with a shared `ceres::Manifold`
— a quaternion manifold (`SetVehicleframesQuaternionManifold` exists as a sibling symbol). The
quaternion storage order matches Ceres' `QuaternionManifold` (w first) or
`EigenQuaternionManifold` (w last); the surrounding code normalises and sign-flips so that
`w >= 0`. **[C]** for the block sizes and the manifold, **[L]** for which of the two orders.

### Default information matrix

```
Λ_default = I₆
if relative_pose_rotation_error_degrees > 0 and relative_pose_translation_error_meters > 0:
    Λ_default = DefaultPoseInformationMatrix(rot_deg, trans_m)
             = blockdiag( I₃ / (rot_deg·π/180)² , I₃ / trans_m² )
```
Confirmed by decompiling `DefaultPoseInformationMatrix` (rotation block first, translation block
second; `0.017453292519943295` = π/180). Neither field is set in the shipped configs, so
`Λ = I₆`. **[C]**

Per edge: if `relative_constraint_use_pose_covariance` is **false** (the shipped default), the
edge's stored 6×6 is **overwritten** with `Λ_default`; if true, the stored matrix is used (via an
`LDLT`/`LLT` path). **[C]** for the branch, **[L]** for exactly what the true-branch does with it.

### Residual — `OptimizePoseRelativeConstraintCost`

`ceres::AutoDiffCostFunction<OptimizePoseRelativeConstraintCost, 6, 4, 3, 4, 3>`, parameter
blocks in the order `[q_source, t_source, q_target, t_target]`. **[C]**

Constructor `OptimizePoseRelativeConstraintCost(const Matrix4d& T_meas, const Matrix6d& info)`:
* `CHECK T_meas.block<3,3>(0,0).isUnitary(1e-6)` and `|det(R) − 1| < 1e-6`, else
  `throw std::runtime_error("Relative transform must be a valid SE(3) matrix")`. **[C]**
* `LDLT(info)`; if not positive definite,
  `throw std::runtime_error("Information matrix must be positive definite")`. **[C]**
* stores `R_meas`, `t_meas`, and `sqrt_information = LLT(info).matrixL()` (6×6). **[C]**

Error function `ComputeRelativePoseError<T>(R_meas, t_meas, q_a, t_a, q_b, t_b, e)`:
```
meas = SE3(normalize(quat(R_meas)), t_meas)
A    = SE3(normalize(q_a), t_a)          # world_T_source
B    = SE3(normalize(q_b), t_b)          # world_T_target
E    = meas⁻¹ · (A⁻¹ · B)                # identity when the measurement is consistent
e[0:3] = rotation vector of E.rotation, in RADIANS
         (angle = 2·atan2(‖q.vec‖, |q.w|), sign-corrected for q.w < 0; zero when ‖q.vec‖ < 2.2e-16)
e[3:6] = E.translation, in METRES
```
**[C]** for the structure and the units (the `atan2(norm, |w|)·2` rotation-vector extraction, the
`2.220446049250313e-16` guard, and the residual layout are all explicit in the decompiled body);
**[L]** for the exact left/right placement of `meas⁻¹` — the zero set is identical either way.

`residual = sqrt_information · e`, a 6×6 by 6×1 product. Whether the stored factor is `L` or
`Lᵀ` only matters for non-diagonal information matrices; every shipped config uses a scaled
identity, where they coincide. In Python use
`S = np.linalg.cholesky(Λ).T` so that `‖S·e‖² = eᵀΛe`. **[C]/[L]**

### Robust loss

`AddResidualBlock(cost, /*loss=*/nullptr, params, 4)` — the second argument register is zeroed
(`xor %edx,%edx` at `0x102dda`). **There is no robust loss on pose-graph relative constraints.**
`BundleAdjustmentSolver::CreateLossFunction` (which maps `loss_type` TRIVIAL/SOFT_L1/CAUCHY/HUBER
to `ceres::TrivialLoss` / `SoftLOneLoss` / `CauchyLoss` / `HuberLoss` with
`loss_function_scale`) is only reached from `AddRelativePoseConstraints`, the **bundle
adjustment** path — not from `SolvePoseGraphProblem`. So the `loss_type: CAUCHY,
loss_function_scale: 1.0` in `pose_graph_config.pb.txt` has **no effect on PGO**. **[C]**

Robustness instead comes from (a) the hard gates in §6.4 and (b) switchable constraints.

### Switchable constraints (`use_switchable_pgo`, off by default)

For an edge with `switchable == true`:
* `ceres::AutoDiffCostFunction<SwitchableRelativeConstraintCost, 6, 4, 3, 4, 3, 1>` with
  parameter blocks `[q_a, t_a, q_b, t_b, s]` and **no loss function**. The residual is
  `s · (sqrt_information · e)` — an Eigen `scalar_constant_op` scaling of the 6×6 factor before
  the product, with the scalar read from `*params[4]`. **[C]**
* `ceres::AutoDiffCostFunction<LoopEdgeConfidenceCost, 1, 1>` on `s` alone, no loss:
  `residual = (1 − s) · sqrt(confidence_to_pose_error_ratio)`, analytic Jacobian
  `−sqrt(Ξ)`. **[C]** (decompiled `Evaluate` reads literally
  `*param_2 = (1.0 - s) * w; **param_3 = -w;` with `w = sqrt(Ξ)` allocated in the caller.)
* `AddParameterBlock(&s, 1)`, `SetParameterLowerBound(&s, 0, 0.0)`,
  `SetParameterUpperBound(&s, 0, 1.0)`; `s` is initialised to `1.0`. **[C]**

This is Sünderhauf & Protzel's switchable constraints with `Ξ = confidence_to_pose_error_ratio`
and `γ = 1`.

### Gauge fixing

`SetPositionConstraintFrame(anchor_id)` where `anchor_id` is the `source_frame_id` of the first
edge in the graph (rig id `1` in both real runs). In the solver this becomes **two**
`ceres::Problem::SetParameterBlockConstant` calls — on the anchor's quaternion *and* its
translation. The node is fully fixed; there is no soft prior. Logged as
`Added 1 position constraint frame constraints to problem`. **[C]**

### Solver options

`BundleAdjustmentSolver` is constructed directly from the `BundleAdjustmentConfig` proto, and
`Solve()` overrides `ceres::Solver::Options` from it. **[C]** unless noted:

| Setting | Value | Source |
|---|---|---|
| `linear_solver_type` | `SPARSE_SCHUR` (enum 4) | hard-coded default **[C]** |
| `linear_solver_type` when `use_cudss_solver` | `SPARSE_NORMAL_CHOLESKY` (2) + `sparse_linear_algebra_library_type = CUDA_SPARSE` (3), logged `Using: CUDSS_SPARSE + SPARSE_NORMAL_CHOLESKY` | **[C]** |
| `max_num_iterations` | `optimization_config.max_num_iterations` = 200 | **[C]** |
| `max_linear_solver_iterations` | `optimization_config.max_linear_solver_iterations` = 100 | **[L]** |
| `num_threads` | `optimization_config.num_threads` = 8 | **[L]** |
| `minimizer_progress_to_stdout` | from config (`true`) | **[L]** |
| trust region / tolerances | Ceres defaults (LM, `function_tolerance` 1e-6, `gradient_tolerance` 1e-10, `parameter_tolerance` 1e-8) — the option block is initialised with the stock defaults and only the fields above are overwritten | **[L]** |
| guard | `CHECK(solver_options.IsValid(&err))`; aborts if 0 residuals with `"0 residual in bundle adjustment problem. Please Check!"` | **[C]** |
| result | `Summary::IsSolutionUsable()`; failure logs `"Optimization failed: <message>"` and `"Termination type: N"` | **[C]** |
| reporting | `verbose` gates `Ceres summary`, `FullReport`, `Time`, `Initial cost`, `Final cost` (costs printed as `sqrt(cost / num_residuals)`) | **[C]** |

---

## 8. Fast happy path for the Python reimplementation

### Must keep

1. **Optimise the rig, not the cameras.** Nodes are `synced_sample_id` groups; camera poses are
   *derived* as `world_T_vehicle · vehicle_T_camera` on export. Optimising per-camera poses
   would (a) change the parameter count 8×, (b) let the rig deform. Getting this wrong is the
   single biggest accuracy risk.
2. **Edge convention** `source_T_target = world_T_source⁻¹ · world_T_target`, degrees on disk,
   radians internally.
3. **One fully-fixed anchor** (the source node of the first edge). No soft prior.
4. **Sequential edges from the input odometry**, `connected_keyframe_num` links forward.
5. **Residual** `[log(R_err); t_err]` whitened by `chol(Λ)ᵀ`, **no robust loss**.
6. **Loop edge weighting** `Λ = I₆ · score · loop_residual_weight` with `score = 1/‖t_rel‖`.
7. **Export all five files** — `feature_matcher_task_builder_main` reads
   `pose_graph/frames_meta.json` (`cusfm_runner.py:655-657`), and the pose exporters read the
   vehicle files.

### Can drop

* The camera-frame `pose_graph.pb.txt` and the EXTRINSIC edge expansion — nothing downstream
  reads them (they are a debug artefact). Emit an empty or minimal file if you want the layout.
* Switchable constraints and edge pruning (`use_switchable_pgo` is off; `Ξ = 0` makes the switch
  prior vanish anyway, which would let every `s` collapse to 0 — do not enable it without also
  setting `confidence_to_pose_error_ratio`).
* CUDSS, rolling shutter, absolute/GPS position constraints, intrinsics refinement — all live in
  the BA path, not here.
* The mid-stream `MultiOptimizePoseGraph` re-solves. A single global solve at the end is
  equivalent to within the tolerance budget and much cheaper. Keep the incremental variant only
  if you need the blob's exact iterate sequence.
* `EsimateSequentialEdgePose` (`use_sequential_edge_pose_estimation` is off). Measured: turning
  it on for galileo shifts camera positions by up to 110 mm (mean 73 mm) on a 1.37 m
  trajectory and logs one fallback to the odometry pose — far outside the 5 mm ATE budget.
  Leave it off. **[C]** (§9.4)

### Should fix relative to the blob

Set `loop_edge_translation_threshold_meters` and `loop_edge_rotation_threshold_degrees` to real
values (e.g. 3.0 m and 30°) or the loop path is dead. **But measure before shipping it:** on
galileo, turning the gates on (5.0 m / 60°) adds 8 loop edges and moves camera positions by up to
12.7 mm (mean 4.6 mm) on a 1.37 m trajectory — the same order as the whole ATE budget
(blob reference 3.9 mm, acceptance bound 5 mm). On a short, low-drift sequence the loops can
easily hurt. Gate them behind a flag and validate per dataset; they should matter much more on
RoboCap. **[C]** (measured from the §9.2 A/B outputs.)

Also consider setting
`relative_pose_translation_error_meters` / `relative_pose_rotation_error_degrees` so sequential
edges get a physically meaningful `Λ` instead of `I₆` — with `I₆` a 1 rad rotation error costs
the same as a 1 m translation error.

### Suggested building blocks

* **Solver:** `pyceres` if it is in the `colsfm` environment (it gives `QuaternionManifold`,
  `SPARSE_SCHUR`, and the same convergence behaviour, so the parity story is clean).
  Otherwise `scipy.optimize.least_squares(method="trf", jac_sparsity=...)` over a
  `(N−1)×6` local-parameterisation vector, or `g2o-python`. The problem is tiny — 1132 nodes and
  1131 edges solved in 6.5 ms in the blob — so almost anything works.
* **SE(3):** `scipy.spatial.transform.Rotation` (`as_rotvec` / `from_rotvec` /
  `from_quat` with `scalar_first=`) or `pycolmap.Rigid3d`, whose `*` and `.inverse()` match the
  `world_T_x` convention here directly.
* **Loop retrieval:** see `docs/spec/bow.md`. Any retrieval that returns time-separated
  candidates works; the verification gates are what matter.
* **Loop verification:** `pycolmap.estimate_and_refine_two_view_geometry` or
  `cv2.recoverPose` with USAC, plus stereo-baseline scale recovery. The blob's
  `StereoPoseEstimator` path is specified in `docs/spec/generate_association_main.md`.
* **Axis-angle-degrees ↔ quaternion** for the on-disk format:
  `R.from_rotvec(np.deg2rad(angle) * axis)`.

### Reference skeleton

```python
# nodes: dict[rig_id, Rigid3d]  (world_T_vehicle, from the lowest-id camera of the rig)
# edges: list[(src, dst, Rigid3d source_T_target, np.ndarray (6,6) information)]
# anchor: rig id, held constant

def residual(T_s, T_t, meas, S):           # S = cholesky(info).T
    E = meas.inverse() * (T_s.inverse() * T_t)
    e = np.concatenate([Rotation.from_matrix(E.rotation).as_rotvec(), E.translation])
    return S @ e
```

---

## 9. Verification

### 9.1 Numerical checks against `data/cusfm_runs/galileo`

All checks re-runnable; they use only `numpy` and the shipped artifacts.

| Claim | Result |
|---|---|
| `world_T_target == world_T_source · pose_source_to_target` for the three vehicle edges | translation error ≤ 5.9e-17 m, rotation Frobenius error ≤ 2.5e-16 |
| the reversed convention `world_T_source == world_T_target · pose_source_to_target` | translation error 0.33 m — rejected |
| `world_T_camera(f) == world_T_vehicle(synced_sample_id) · sensor_to_vehicle_transform(camera_params_id)` for frames 2417 (cam 4), 2415 (cam 3), 2423 (cam 0, `camera_params_id` omitted → 0) | translation error ≤ 6.2e-17 m, rotation error ≤ 6.7e-16 |
| `pose_graph/frames_meta.json` vs `keyframes/frames_meta.json` | identical structure and key set; the reference camera of each rig is unchanged to 1e-16, the other cameras move by up to 1.4e-5 m — the rigidification of §1, not an optimisation |
| camera-frame edge counts | 24 CONSECUTIVE + 168 EXTRINSIC, matching 8×3 and 28+56+56+28 |
| rig pose comes from the lowest-id camera of the sample | that frame reproduces `vehicle_frames_meta.json` to ≤ 2.3e-16; the other seven cameras disagree by 7e-7 … 1.4e-5 m |

Run logs used: `data/cusfm_runs/galileo/pose_graph/pose_graph_main.txt` (4 rig frames, 3
constraints, anchor 1, 0 loops) and
`data/cusfm_runs/robocap_full/cusfm/pose_graph/pose_graph_main.txt` (1132 rig frames, 1131
constraints, anchor 1, 0 loops, 307 s in loop search vs 6.5 ms in Ceres).

### 9.2 Dynamic A/B on the loop gates

Two reruns of the unmodified blob on `data/cusfm_runs/galileo/cusfm/keyframes` (226 keyframes,
29 rig frames, session duration 0.933314 s), writing only into `data/cusfm_re/runs/`:

```
pixi run -- ./pycusfm/x86_cuda13/bin/pose_graph_main \
  --keyframe_directory data/cusfm_runs/galileo/cusfm/keyframes \
  --match_directory    data/cusfm_runs/galileo/cusfm/matches \
  --voc_directory      data/cusfm_runs/galileo/cusfm/cuvgl_map \
  --config_directory   data/cusfm_re/runs/cfg_stock \
  --pose_graph_directory data/cusfm_re/runs/pg_stock \
  --model_dir pycusfm/models --alsologtostderr
```

| Config | Result |
|---|---|
| `cfg_stock` = verbatim `pycusfm/configs/isaac` | `0 loop constraints`, one solve, 28 constraints (29 rig frames − 1), anchor 1 |
| `cfg_loop` = the same plus `loop_edge_translation_threshold_meters: 5.0` and `loop_edge_rotation_threshold_degrees: 60.0` | 8 loop edges: `6→3, 7→2, 8→3, 17→14, 24→21, 26→23, 27→24, 28→25`; four mid-stream re-solves at graph sizes 11/21/29/36 with 11/32/61/97 residual blocks, then the final solve at 36/133 |

Effect on the trajectory (`pg_stock/frames_meta.json` vs `pg_loop/frames_meta.json`, 226 camera
poses, trajectory extent 1.367 m):

| Comparison | max Δposition | mean Δposition |
|---|---|---|
| stock output vs input `frames_meta.json` | 1.51e-05 m | 2.63e-06 m (rig rigidification only) |
| loops-on output vs input | 1.27e-02 m | 4.6e-03 m |
| stock vs loops-on | 1.27e-02 m | 4.6e-03 m |

This pins, in one experiment: the sign of both magnitude gates and that `0.0` means "reject
everything"; that fields 14/15 of `PoseGraphConfig` are the two gates; `num_votes: 1` → at most
one loop edge per rig frame; loop edges point from the current rig frame **back** to an earlier
one (`source` = current, `target` = older); the time gate (`0.08 × 0.933 s = 0.075 s`, i.e. about
three rig frames at ~32 ms spacing — the closest accepted loop is 3 frames back); and the
duplicate-accumulation quirk of the mid-stream re-solves.

The graph sizes reconcile exactly with the pseudocode in §6.3: a mid-stream solve fires on the
first rig frame *after* a run of loop-closing frames, and at that point the graph holds
`(rig_index − 1)` sequential edges plus the loops so far — 8+3 = 11 at rig 9, 17+4 = 21 at rig 18,
24+5 = 29 at rig 25, and 28+8 = 36 both at rig 29 and at the final solve. **[C]**

---

### 9.3 Dynamic check of the switchable path

Third rerun, `cfg_loop` plus `use_switchable_pgo: true`, `confidence_to_pose_error_ratio: 10.0`,
`loop_edge_prune_threshold: 0.5` (output in `data/cusfm_re/runs/pg_switch/`, `--v=1`):

```
bundle_adjustment_solver.cc:1167] GetConfidenceToPoseErrorRatio:10
bundle_adjustment_solver.cc:1235] Added 36 relative pose constraints to problem
pose_graph.cc:615] output loop_edge_confidence_weight:1        (× 8)
pose_graph.cc:642] total pruned 0 loop edges at threshold 0.5
```

Confirms: `confidence_to_pose_error_ratio` reaches the solver; exactly **8** switch variables
exist, i.e. one per LOOP edge and none for CONSECUTIVE edges; all 8 converge to `s = 1`, so the
galileo loops are all geometrically consistent and nothing is pruned; and the residual-block
count is 36 rather than the 97/133 of the non-switchable run, because `UpdatePoseGraphFromOpt`
calls `ClearRelativeConstraint()` — which is exactly the accumulation mechanism described in
§6.5. **[C]**

### 9.4 Dynamic check of `use_sequential_edge_pose_estimation`

Fourth rerun, stock `isaac` plus `use_sequential_edge_pose_estimation: true`
(`data/cusfm_re/runs/pg_seq/`): still 0 loops and 28 constraints, but the log ends
`... and seq link use input pose: 1` — one of the 28 sequential edges failed re-estimation and
fell back to the odometry pose. Camera positions differ from the stock run by up to 0.1102 m
(mean 0.0728 m). The image-based sequential edges are therefore *much* noisier than cuVSLAM's
ego-motion on this dataset. **[C]**

## 10. Evidence index

Symbols (all from `nm -C --defined-only pycusfm/x86_cuda13/bin/pose_graph_main`):

* `nvidia::isaac::visual::cusfm::PoseGraph::` `RunPoseGraph`, `RunIncrementalPoseGraph`,
  `RunGlobalPoseGraph`, `RunAssociationBasedPoseGraph`, `MultiOptimizePoseGraph`,
  `OptimizePoseGraph`, `UpdatePoseGraphFromOpt`, `UpdatePoseGraphInCameraFrameLoops`,
  `AddConstraintPair`, `AddTimeConsecutiveCameraFrame`, `AddExtrinsicCameraFrame`,
  `FindBestLoopCandidateForVehicle`, `FindBestLoopCandidateForFrame`, `LoopCandidateVoting`,
  `GetScoreFromPoseCovariance`, `GetSessionDurationInSecond`, `EsimateSequentialEdgePose` (sic),
  `InitGraph`, `InitGraphWithAssociation`, `ExportUpdatedKeyframes`, `ExportVehicleFrames`,
  `ExportPoseGraphInVehicleFrame`, `ExportPoseGraphInCameraFrame`.
* `nvidia::isaac::visual::geometry::` `BundleAdjustmentSolver::SolvePoseGraphProblem<VehicleFrame>`
  (`0x102860`), `BundleAdjustmentSolver::Solve`, `BundleAdjustmentSolver::CreateLossFunction`,
  `DefaultPoseInformationMatrix` (`0xea600`), `ComputeRelativePoseError<double>` (`0x20ae40`),
  `OptimizePoseRelativeConstraintCost::Create` / `::OptimizePoseRelativeConstraintCost`,
  `PoseGraphDataTemplate<VehicleFrame>::{AddFrame,AddRelativeConstraint,ClearRelativeConstraint,
  SetPositionConstraintFrame,SetConfidenceToPoseErrorRatio,GetConfidenceToPoseErrorRatio}`.
* Ceres templates: `AutoDiffCostFunction<OptimizePoseRelativeConstraintCost, 6, 4, 3, 4, 3>`,
  `AutoDiffCostFunction<SwitchableRelativeConstraintCost, 6, 4, 3, 4, 3, 1>`,
  `AutoDiffCostFunction<LoopEdgeConfidenceCost, 1, 1>`,
  `AutoDiffCostFunction<OptimizePoseAbsolutePositionConstraintCost, 3, 4, 3>` (BA path only).
* `nvidia::isaac::visual::loop_closing::LoopClosureIndex::DetectCandidates`.
* `nvidia::isaac::visual::cuvgl::StereoPoseEstimator::{EstimateRelativePose,RelativePoseRefine,
  PoseEstimateFromTwoViewNormalized}`.

Source files named in string literals:
`src/visual/cusfm/pose_graph.cc`, `src/visual/geometry/bundle_adjustment_solver.cc`,
`tools/visual/cusfm/pose_graph_main.cc`, `tools/visual/cusfm/pose_graph_association_main.cc`,
`src/visual/general/association_data.cc`, `include/visual/general/frame.h`.

Key line numbers (from glog call sites):
`pose_graph.cc` — `:114` "session duration is N seconds" (constructor), `:199`
connection-count check, `:320` "Vehicle frames are empty.", `:324` "Using incremental pose graph
model.", `:355` "Added loop constraint pair", `:361` mid-stream "Perform pose graph optimization
with N loop constraints.", `:422` final "Perform pose graph optimization … and seq link use input
pose: M", `:526` "Pose graph is empty.", `:575` "Pose graph optimization will be performed,
total_constraint_number: N, anchored keyframe id: A", `:615` "output
loop_edge_confidence_weight", `:642` "total pruned N loop edges at threshold", `:705` "loop edge
don't exist in the loop_edges_ impossible", `:744` camera pose graph saved, `:806` keyframes
saved, `:965` session-duration error branch, `:1372` "Get multiple constraints between the same
vehicle pairs." (in `LoopCandidateVoting`), `:1434` "Association type is not supported.",
`:1546` "No camera to camera transform found for frame pair".

`bundle_adjustment_solver.cc` — `:32` 0-residual guard, `:52` "Using: CUDSS_SPARSE +
SPARSE_NORMAL_CHOLESKY", `:69` `CHECK(solver_options.IsValid(...))`, `:73`-`:79` summary/report/
time/cost logging, `:87`-`:88` "Optimization failed" / "Termination type", `:111` "Unsupported
loss type, using trivial loss instead.", `:1145` `CHECK(!problem_)` "Cannot use the same solver
multiple times", `:1148` "frame relative constrain is empty.", `:1167`
`GetConfidenceToPoseErrorRatio:`, `:1235` "Added N relative pose constraints to problem", `:1252`
"Added N position constraint frame constraints to problem", `:1374` (BA path) "Added N relative
pose constraints to problem, M vehicle relative pose constraints".

Artifacts and scratch produced during this analysis (all under `data/cusfm_re/`, gitignored):
`protos/*.fdset` and `protos/*.proto.txt` (schema extracted from `protodesc_cold`),
`tools/extract_protodesc.py`, `tools/fdset_to_proto.py`,
`ghidra_scripts/DecompMatching.java`, `decomp/pose_graph_main/*.c` (389 functions),
`disasm/SolvePoseGraphProblem_VehicleFrame.asm`,
`runs/cfg_stock/` + `runs/pg_stock/` and `runs/cfg_loop/` + `runs/pg_loop/` (the §9.2 A/B).

---

## 11. Open questions

1. **Quaternion storage order** in the parameter block — Ceres `QuaternionManifold` (w first) vs
   `EigenQuaternionManifold` (w last). `SetVehicleframesQuaternionManifold` exists but was not
   decompiled. Irrelevant to a from-scratch Python port; relevant only for byte-level parity.
2. **Left vs right composition** of `meas⁻¹` in `ComputeRelativePoseError`, and whether the
   stored Cholesky factor is `L` or `Lᵀ`. Both are no-ops for the shipped diagonal information
   matrices.
3. ~~The float compared at `0x2c(%rsp)` in the loop score gate.~~ **Resolved**: it is an internal
   running "best score so far", not a config value — `pose_graph_main` reads no
   `ImageRetrievalConfig` file, and forcing large values into `image_retrieval_config.pb.txt`
   changes nothing (§3.1). Its initial value (0.0, so the first candidate always passes) was not
   read out of the prologue, but the semantics are unambiguous.
4. **`"Skip, candidate out of candidate search range"`** — the exact range predicate on the
   candidate rig id was not decoded.
5. **`--match_directory`** is required by gflags but no read of it was found on the default
   path. Possibly used only when associations already exist.
6. **Where `pose_graph_association_main` finds `frames_meta.json`** — `InitGraphWithAssociation`
   joins one of its string arguments with the literal `"keyframes"`, but the runner passes only
   `--association_directory`, and no `keyframes/` subfolder exists in the recorded association
   outputs (that path was never exercised: all shipped runs are single-track). Needs a
   multi-track run to settle.
7. **Solver option offsets** — `max_linear_solver_iterations`, `num_threads` and
   `minimizer_progress_to_stdout` were mapped by elimination against the `BundleAdjustmentConfig`
   field list rather than by decoding each `ceres::Solver::Options` slot.

# `feature_matcher_main` — implementation spec

Status: reverse-engineered from `pycusfm/x86_cuda13/bin/feature_matcher_main` (unstripped ELF,
C++ symbols, embedded source paths `/home/jryu/workspaces/visual_mapping/src/...`), Ghidra
decompilation under `data/cusfm_re/decomp/matcher/`, the shipped configs, the completed runs
under `data/cusfm_runs/galileo/`, and **11 re-runs of the binary with mutated configs** under
`data/cusfm_re/runs/sweep_matcher/`. Evidence and confidence per claim: **[C]** confirmed
(decompiled body, string literal, or exact numeric match against a real or re-run artifact),
**[L]** likely, **[I]** inferred.

---

## 1. Purpose and position in the pipeline

Stage 6. For every `FramePair` in one task file it runs **LightGlue** on the ALIKED descriptors
already stored in the keyframe protos, applies a spatial non-maximum suppression to the matched
keypoints, geometrically verifies the pair with an **essential-matrix RANSAC**, and writes the
surviving matches.

```
keyframes/<camera>/<ts>.pb        ─┐
pose_graph/frames_meta.json       ─┤
matches/tasks/matching_task_NNN.pb├─► feature_matcher_main ─► matches/matching_task_NNN_result.pb
models/aliked_lightglue/*.engine  ─┤                          matches/map_keypoints.pb
configs/…/matching_task_worker…   ─┘
```

The runner launches **one process per task file** in parallel (`--num_thread 1` each),
`pycusfm/cusfm_runner.py:709-766`. Each process re-loads the engine (~0.4 s) and the whole
keyframe pool. **[C]**

No image pixels are read in the default configuration; the matcher works purely on the keypoints
and descriptors stored in the keyframe protos. `--current_raw_image_dir` is only needed for
`--use_rolling_shutter_correction` or `--debug_dir`. **[C]**

### Headline findings for the Python port

1. **Matching**: LightGlue TensorRT, `matches0`/`mscores0` filtered by `score > match_threshold`
   (**strictly greater**, `match_threshold: 0.3` in the isaac worker config). **[C]**
2. **Then** spatial NMS (SSC, "Suppression via Square Covering") over the *matched* keypoints of
   image 0, targeting `match_top_k: 500` matches with `num_points_tolerance_fraction: 0.3`. This
   is what turns ~931 raw matches per pair into ~400. **[C]**
3. **Then** verification: `cv::findEssentialMat` with `cv::RANSAC` on **normalized (bearing)
   coordinates** with an identity camera matrix, threshold =
   `0.5·(max_pixel_error/f₀ + max_pixel_error/f₁)`, probability = `min_ransac_confidence`. **[C]**
4. **A pair with fewer than 10 inliers is emptied entirely** — not just trimmed. This is the
   single most surprising rule and it fires on 8 of 339 Galileo pairs. **[C]**
5. The written matches are the **RANSAC inliers only**; verified as a strict subset of the
   unverified run for all 339 pairs. **[C]**
6. `FramePairMatch.source_to_target` is **never populated** by this stage (0 of 339). The
   relative pose is not estimated here. **[C]**
7. `pycusfm/configs/isaac/matching_config.pb.txt` (with `match_threshold: 0.99`) is **not read by
   this binary** — only by `export_lightglue_engine`. The live config is
   `matching_task_worker_config.pb.txt`. **[C]**

The cuSfM paper (arXiv:2510.15271) corroborates the design: §3.1 states "ALIKED is selected as
the default feature extraction method and LightGlue as the default matching method", reports
"TensorRT-optimized engines based on C++ ... reducing inference time to approximately 20 ms", and
§4.1 describes the two-view check in terms of **essential** matrices. The paper gives no
thresholds; all numbers in this spec come from the binary and the shipped configs. **[C]**

---

## 2. CLI flags

From `data/cusfm_re/help/feature_matcher_main.txt` and `pycusfm/cusfm_runner.py:715-740`. **[C]**

| Flag | Type | Default | isaac pipeline | Meaning |
|---|---|---|---|---|
| `--current_keyframe_directory` | string | "" | `keyframes/` | Keyframe root |
| `--output_dir` | string | "" | `matches/` | Result + `map_keypoints.pb` destination |
| `--task_file` | string | "" | `matches/tasks/matching_task_NNN.pb` | If empty, the binary builds the task itself (`NO task is provided, building task on the fly`) |
| `--matching_worker_config` | string | `configs/matching_task_worker_config.pb.txt` | yes | `MatchingTaskWorkerConfig` text proto |
| `--match_pair_select_config` | string | `configs/keymatch_pair_select_config.pb.txt` | yes | Only used on the build-on-the-fly path |
| `--model_dir` | string | "" | `pycusfm/models` | **Parent** of `aliked_lightglue/`; the binary appends the subdirectory itself (passing the leaf dir segfaults) |
| `--keyframe_metadata_file` | string | "" | `pose_graph/frames_meta.json` | Overrides the metadata path |
| `--num_thread` | int32 | 1 | 1 | Worker threads inside one process |
| `--batch_size` | int32 | 0 | 0 (runner default) | `>0` switches to `PerformBatchMatchingTask` (`Using GPU-accelerated batch matching with batch size: N`) |
| `--max_buffer_size` | int32 | 2000 | default | Keyframe LRU cache size |
| `--use_rolling_shutter_correction` | bool | false | false | Selects `TwoViewVerificationRollingShutter`; needs `--current_raw_image_dir` |
| `--debug_dir` / `--debug_interval` | string/int | ""/500 | off | Writes match visualisations |
| `--pose_graph_association_dir`, `--previous_cusfm_ws`, `--*_raw_image_dir`, `--map_image_dir` | | "" | off | Patch / localisation / debug modes |

---

## 3. Config: `MatchingTaskWorkerConfig`

`data/cusfm_re/proto/matching_config.proto`, decoded from
`data/cusfm_schema/cusfm_protos.fdset`. **[C]**

```proto
message GeometryVerificationConfig {
  optional bool   enable_geometry_verification      = 1;   // struct +0x30
  optional double max_pixel_error                   = 2;   // +0x10
  optional double min_ransac_confidence             = 3;   // +0x18
  optional double point3d_max_error                 = 4;   // RGBD path only
  optional double depth_check_max_reproj_pixel_error= 5;   // RGBD path only
}
message MatchingTaskWorkerConfig {
  optional KeypointMatchingParams     matching_params     = 1;
  optional double                     max_pixel_error     = 2;   // LEGACY, unread
  optional double                     min_ransac_confidence = 3; // LEGACY, unread
  optional GeometryVerificationConfig verification_config = 4;   // THE live one
}
message LightGlueMatcherParams {
  optional TensorRTParams superpoint_tensorrt_config = 1;
  optional TensorRTParams aliked_tensorrt_config     = 2;
  optional float  match_threshold                    = 3;
  optional int32  match_top_k                        = 4;
  optional GlobalNonMaxMethod global_non_max_method  = 5;   // NONE=0, SUPPRESSION_VIA_SQUARE_COVERING=1, TOP_M=2
  optional float  num_points_tolerance_fraction      = 6;
}
```

`TwoViewVerification` reads its two thresholds out of the **`verification_config`** copy
(`GeometryVerificationConfig` at `+0x10`/`+0x18`, enable flag at `+0x30`), never the top-level
duplicates. **[C]**
(`data/cusfm_re/decomp/matcher/…TwoViewVerification_00153580.c:42-49,123`)

Shipped isaac values, `pycusfm/configs/isaac/matching_task_worker_config.pb.txt`:

```
verification_config { enable_geometry_verification: true; max_pixel_error: 4.0;   # :1-5
                      min_ransac_confidence: 0.98 }
matching_params.lightglue_params { match_threshold: 0.3; match_top_k: 500;        # :27-30
                                   global_non_max_method: SUPPRESSION_VIA_SQUARE_COVERING;
                                   num_points_tolerance_fraction: 0.3 }
matching_params.cv_cuda_params { … }          # SIFT path only — dead with ALIKED
matching_params.general_matching_params { … } # brute-force/epipolar path only — dead with ALIKED
```
The `av` profile differs only in `min_ransac_confidence: 0.99`. **[C]**

Matcher selection, `KeypointMatcher::InitMatcher`: if `lightglue_params` is set →
`Matcher will use lightglue.`; else `use_bf_matcher` → `cv::BFMatcher(NORM_L2, crossCheck=true)`;
else `cv_cuda_params` → CV-CUDA pairwise matcher; else `general_matching_params` → epipolar /
general matcher. With the shipped ALIKED config only the LightGlue branch runs. **[C]**

---

## 4. Input / output formats

### 4.1 Keyframe input

`<keyframe_dir>/<camera_name>/<image_stem>.pb`, a `protos.visual.general.Keyframe`, where the
stem is the JPEG basename (nanosecond timestamp). Only `keypoint_vector` is used:
`x`, `y` (float32 pixels in the **original** image), `descriptors` (`n·128` little-endian
float32, L2-normalised), `descriptor_type = ALIKED_DESCRIPTOR(4)`, `image_width`, `image_height`.
**[C]** — see `docs/spec/feature_extractor_main.md` §5 and `tools/raco_extract.py`.

### 4.2 Result output

`<output_dir>/<task-file-stem>_result.pb`, binary `protos.visual.cusfm.FrameMatches`. **[C]**

```proto
message FeatureMatch   { optional uint32 source_point_id = 1; optional uint32 target_point_id = 2; }
message FramePairMatch { optional FramePair frames = 1; repeated FeatureMatch matches = 2;
                         optional RigidTransform3d source_to_target = 3; }   // never set here
message FrameMatches   { repeated FramePairMatch feature_matches = 1; }
```

Point ids are **indices into the `keypoint_vector` of each keyframe** (0-based). One
`FramePairMatch` is written for **every** input pair, including pairs whose match list is empty.
**[C]**

### 4.3 `map_keypoints.pb`

`<output_dir>/map_keypoints.pb`, a `protos.visual.cusfm.MapKeyPoints` written once per process by
`SaveMapPoint2DFile` (only when `--task_file` is *not* given, or by the last worker —
`main_00138d30.c:190`). On the real Galileo run it holds:

* `frames`: 226 `Keyframe` messages — `frame_id`, `is_valid`, `inliers` (2048 bools, all false at
  this stage) and `keypoint_vector` with `x`, `y`, `depth`, `r`, `g`, `b`, `image_width`,
  `image_height` but **`descriptors` stripped** (0 bytes).
* `points_x/y/z`, `observations`, `points_num_observations`: all empty — the 3D fields are filled
  later by `keypoints_mapper_main`. **[C]**

So this file is the mapper's cache of 2D observations, not a product of matching.

---

## 5. Algorithm

### 5.1 Per-pair driver

`FeatureMatcher::ComputeMatches(worker_config, keyframe_a, keyframe_b)`
(`data/cusfm_re/decomp/matcher/…ComputeMatches_001539c0.c`): **[C]**

```
if kf_a.keypoint_vector empty or kf_b.keypoint_vector empty:
    LOG("Skipping frame pair due to empty keypoints"); return
if bev_start_idx valid on both:   NormalMatchAndExtraMatchForBEV(...)     # BEV path, unused
else:                             KeypointMatcher::MatchKeypoints(kv_a, kv_b, &matches,
                                                                  fundamental=nullptr,
                                                                  constraint_fn=empty)
if use_rolling_shutter_correction:  TwoViewVerificationRollingShutter(...)
elif keypoints carry depth (RGBD):  RGBDGeometryVerification(...)
else:                               TwoViewVerification(...)
```

### 5.2 LightGlue inference

`LightGlueMatcher::MatchKeypoints` (`…LightGlueMatcher__MatchKeypoints_001fc850.c`): **[C]**

```
KeypointVector::ToOpenCv(kv0) -> points0 (cv::Point2f), descriptors0
KeypointVector::ToOpenCv(kv1) -> points1,               descriptors1
RunWithTensorrt(points0, points1, desc0, desc1, size0=(w0,h0), size1=(w1,h1), &matches)
ApplyNonMaximumSuppression(kv0, kv1, &matches)
```

**Keypoint normalisation** — `LightGlueTensorRT::NormalizedKeyPoints(points, w, h, &out)`
(`…NormalizedKeyPoints_00207a80.c`): shift by half the image size, divide by
`max(w, h) / 2`, i.e. the standard LightGlue `normalize_keypoints`:

```
scale = max(w, h) / 2
kpts_norm = (kpts - [w/2, h/2]) / scale
```
**[L]** — the shift/scale arithmetic is visible but Ghidra's SSE modelling of the packed
`[w, h]` load is imperfect; the scale term `max(w,h)/2` is unambiguous.

**Engine bindings** (`pycusfm/models/aliked_lightglue/lightglue_aliked.onnx`): **[C]**

| idx | name | shape | dtype |
|---|---|---|---|
| 0 | `kpts0` | `[1, N0, 2]` | float32 (normalised) |
| 1 | `kpts1` | `[1, N1, 2]` | float32 |
| 2 | `desc0` | `[1, N0, 128]` | float32 |
| 3 | `desc1` | `[1, N1, 128]` | float32 |
| 4 | `matches0` | `[M, 2]` | int32 (index0, index1) |
| 5 | `mscores0` | `[M]` | float32 |

The engine is FP16 (`use_fp16: true`), profile min/opt/max keypoint counts
`1 / 3200 / 5500` (`matching_task_worker_config.pb.txt:35-78`). It is cached as
`lightglue_aliked_fp16_<trt-version>_sm_<arch>.engine` next to the ONNX and rebuilt when
missing. **[C]**

**Post-processing** (`LightGlueAlikedTensorRT::PostProcess(&matches, match_threshold)`,
`…PostProcess_00200260.c`): copy outputs to host, then for `i` in `0..M`:

```c
if (mscores0[i] > match_threshold)                     // strictly greater
    matches.push_back({matches0[i][0], matches0[i][1]});
```
**[C]**

### 5.3 Spatial NMS on the matches (SSC)

`LightGlueMatcher::ApplyNonMaximumSuppression`
(`…ApplyNonMaximumSuppression_001fbdb0.c`): **[C]**

```
if match_top_k <= 0 or matches.size() < match_top_k:  return          # no-op
if global_non_max_method != SUPPRESSION_VIA_SQUARE_COVERING: (other methods)
build cv::KeyPoint list from the image-0 keypoints of each match, response = LightGlue score
GlobalNonMaximumSuppressionViaSquareCovering(keypoints, &kept,
    target      = match_top_k,
    width       = kv0.image_width,
    height      = kv1.image_height,          # NOTE: height taken from the *second* vector
    sorted      = false,
    tolerance   = num_points_tolerance_fraction)
matches = matches[kept]
```

The `width` comes from `kv0` (+0x44) but `height` from `kv1` (+0x48) — harmless while both images
are the same size, which is the case for every rig in `data/`. **[C]** (latent bug)

`GlobalNonMaximumSuppressionViaSquareCovering` sorts keypoints by strength, then runs the
**SSC algorithm** (Bailo et al. 2018, *Efficient adaptive non-maximal suppression algorithms for
homogeneous spatial keypoint distribution*): binary-search a suppression square width `w` between
`low = floor(sqrt(n / target))` and `high = round((exp1 ± exp3) / (2·target − 2))` with
`exp1 = width + height + 2·target`, mark a grid of cell size `w/2`, keep the strongest keypoint
per uncovered cell, and stop as soon as the surviving count falls inside
`[round(target − target·tolerance), round(target + target·tolerance)]`. **[C]**
(`…GlobalNonMaximumSuppressionViaSquareCoveringForSortedPoints_…0021e370.c:116-160`)

Behavioural confirmation on the 339 Galileo pairs (mean matches per pair): **[C]**

| variant | mean matches/pair |
|---|---|
| shipped (`top_k 500`, tol 0.3, NMS on) | **400** |
| `global_non_max_method: NONE` | 931 |
| `match_top_k: 3000` (NMS effectively off) | 931 |
| `match_top_k: 100` | 84 |
| `num_points_tolerance_fraction: 0.9` | 161 |

### 5.4 Geometric verification

`FeatureMatcher::TwoViewVerification(worker_config, kf_a, kf_b, &matches)`
(`…TwoViewVerification_00153580.c`): **[C]**

```
cfg = worker_config.verification_config
if not cfg.enable_geometry_verification:  return
if matches.size() < 5:
    VLOG("Fewer than 5 points in current pair, skip two view verification. Current point number: N")
    return                                                # matches kept unfiltered
proj0 = KeyframeCacherPool::GetCameraProjector(kf_a.camera_params_id)
proj1 = KeyframeCacherPool::GetCameraProjector(kf_b.camera_params_id)
indices = DataSelector::SelectMatchesByEpipolarGeometry(kv0, kv1, proj0, proj1, matches,
                                                       cfg.max_pixel_error,
                                                       cfg.min_ransac_confidence)
matches = matches[indices]
```

`DataSelector::SelectMatchesByEpipolarGeometry` (`…SelectMatchesByEpipolarGeometry_00189ed0.c`):
**[C]**

```
n_inliers, mask = SelectMatchesByOpenCVEssentialMat(...)
if n_inliers < 10:   return []          # <-- drops the ENTIRE pair
return [i for i where mask[i]]
```

`DataSelector::SelectMatchesByOpenCVEssentialMat`
(`…SelectMatchesByOpenCVEssentialMat_00188f30.c`): **[C]**

```
pts0 = [kv0.xy[m.src] for m in matches];  pts1 = [kv1.xy[m.dst] for m in matches]
bear0 = proj0.BackProjectNormalized(pts0)      # normalised image plane (x/z, y/z)
bear1 = proj1.BackProjectNormalized(pts1)
thresh = MapErrorToNormalizedPlane(proj0, proj1, max_pixel_error)
       = 0.5 * (max_pixel_error / f0 + max_pixel_error / f1)
E, mask = cv::findEssentialMat(bear0, bear1,
                               cameraMatrix = cv::Mat::eye(3, 3),
                               method       = cv::RANSAC,
                               prob         = min_ransac_confidence,
                               threshold    = thresh)
if E empty: return 0
return popcount(mask), mask
```

`MapErrorToNormalizedPlane` is `(err/f0 + err/f1) * 0.5` where `f` is each projector's focal
length (`BaseCameraProjector` vtable +0x48). **[C]**
(`…MapErrorToNormalizedPlane_001a2f40.c`)

No fundamental matrix, no homography, no `recoverPose`: `cv::findEssentialMat` is the **only**
OpenCV geometry call linked into the binary. **[C]**

Behavioural confirmation (mean matches per pair, 339 Galileo pairs, `two view verifier`
invocations in brackets): **[C]**

| variant | mean matches/pair |
|---|---|
| shipped (`max_pixel_error 4.0`, `min_ransac_confidence 0.98`) | **400** (333 verified) |
| `enable_geometry_verification: false` | 412 (0) |
| `max_pixel_error: 1.0` | 345 (333) |
| `max_pixel_error: 16.0` | 411 (333) |
| `min_ransac_confidence: 0.5` | 382 (333) |
| `match_threshold: 0.01` | 407 (339 — every pair now has ≥5 matches) |
| `match_threshold: 0.9` | 312 (306) |

333 of 339 pairs get verified; the other 6 have fewer than 5 matches and pass through untouched.

Alternate verifiers (not used by the isaac pipeline): **[C]**
* `TwoViewVerificationRollingShutter` — `--use_rolling_shutter_correction`, undistorts
  per-row using `rolling_shutter_delay_microseconds` before the same essential-matrix test.
* `RGBDGeometryVerification` — when keypoints carry depth; `Fewer than 4 points in current pair,
  skip RGBD verification`, then `cuvgl::estimateRigidTransformationRANSAC` with
  `point3d_max_error` / `depth_check_max_reproj_pixel_error`.

### 5.5 Result assembly

Matches are converted with `FeatureMatcher::KeypointMatchToProto` into `FeatureMatch`
(`source_point_id`, `target_point_id`) and appended to one `FrameMatches`, written to
`<task-stem>_result.pb`. `source_to_target` is left unset. **[C]**

---

## 6. Real-artifact statistics (Galileo, `data/cusfm_runs/galileo/cusfm/matches`)

Parsed with `data/cusfm_re/analysis/parse_matches.py` and
`data/cusfm_re/analysis/compare_results.py`. **[C]**

* 339 pairs, 226 keyframes, 2048 ALIKED keypoints per keyframe.
* Matches per pair: min 0, p10 118, median 430, p90 566, max 643; **135 645 matches total**
  (mean 400).
* 8 pairs with 0 matches — all of them had 5-10 raw matches and were emptied by the
  `n_inliers < 10` rule (confirmed by diffing against the `enable_geometry_verification: false`
  run, where their counts are 5,6,6,7,8,9,9,10).
* Inlier retention per pair (verified / unverified): min 0.0, p10 0.897, median **0.985**, p90 1.0.
  Verification is a light filter, not a hard gate.
* `source_to_target` present on 0 of 339 pairs.
* Runtime: 3.4 s total, 8.5 ms per pair (0.4 s of it engine load; LightGlue end-to-end 8.5 ms,
  inference 6.4 ms, verification 0.5 ms).

---

## 7. Python replacement

### 7.1 What already exists in this repo

* `tools/trt_runtime.py` — CUDA helpers (`check_cuda`, `percentile`, `sha256_file`).
* `tools/raco_extract.py` — a complete, working example of building a TensorRT engine from ONNX,
  a content-addressed engine cache, host/device buffer management and dynamic-shape selection
  (`_select_shape`, `_copy_input`, `infer`). The LightGlue runner is the same pattern with four
  inputs and two outputs.
* `tools/export_cusfm_raco.py` — already exports the LightGlue ONNX
  (`pycusfm/models/aliked_lightglue/lightglue_aliked.onnx`, and a batched `…_b16.onnx`).
* Nothing implements NMS, verification or the result protos.

### 7.2 Mapping, behaviour by behaviour

| cuSFM behaviour | Python replacement |
|---|---|
| LightGlue TensorRT inference | New `LightGlueRunner`, modelled on `RacoEngine` in `tools/raco_extract.py:423-628`; bindings `kpts0/kpts1/desc0/desc1 → matches0/mscores0` |
| Keypoint normalisation | `(kpts - [w/2, h/2]) / (max(w, h) / 2)` |
| `score > match_threshold` | `matches0[mscores0 > 0.3]` |
| SSC NMS to `match_top_k` | New ~60-line function; port the SSC binary search. Or accept a divergence and use `np.argsort(scores)[:500]` (`TOP_M`) — this changes which matches survive, so measure before adopting |
| Essential-matrix RANSAC | `pycolmap.estimate_two_view_geometry` / `pycolmap.verify_matches` with `TwoViewGeometryOptions(ransac=RANSACOptions(max_error=…, confidence=0.98))`. pycolmap works in **pixels** with a real camera model, so pass `max_error = 4.0` px directly instead of converting to the normalized plane |
| `n_inliers < 10 → drop pair` | pycolmap's `TwoViewGeometryOptions.min_num_inliers` defaults to 15; set it to 10 for parity |
| `matches.size() < 5 → skip verification, keep all` | pycolmap always verifies; either special-case pairs with <5 matches or accept the divergence (6 of 339 pairs on Galileo, all with ≤4 matches, i.e. useless for triangulation) |
| Result proto | `pycolmap.Database.write_matches(image_id1, image_id2, matches_Nx2)` and `write_two_view_geometry(...)`; keep the `FrameMatches` writer only if the blob mapper must still consume the output |
| `map_keypoints.pb` | `pycolmap.Database.write_keypoints(image_id, xy_Nx2)`; only needed for the blob's `keypoints_mapper_main` |

### 7.3 pycolmap notes

* `pycolmap.Database.write_keypoints` expects `float32 [N, 2]` (or `[N, 4]`/`[N, 6]` with
  scale/orientation); cuSFM keypoints have no scale, so write the 2-column form.
* `pycolmap.Database.write_descriptors` expects `uint8 [N, 128]`. **ALIKED descriptors are
  float32 and L2-normalised**; they do not fit COLMAP's descriptor table. Since LightGlue
  matching happens outside COLMAP, write only keypoints + matches and skip descriptors entirely.
* Camera models: the rig cameras in `frames_meta.json` carry a 3×4 `projection_matrix`
  (`fx, 0, cx, 0; 0, fy, cy, 0; 0, 0, 1, 0`) plus distortion; `camera_projector_factory.cc:18`
  logs `Creating pinhole camera projector` for both Galileo and RoboCap, so `PINHOLE`
  (`fx, fy, cx, cy`) is the right pycolmap model for the isaac path. **[C]**
* `pycolmap.verify_matches(database_path, pair_list_path, options)` is the batch entry point;
  `pycolmap.estimate_two_view_geometry(camera1, points1, camera2, points2, matches, options)` is
  the per-pair one and is easier to drive from a Python pair loop.

### 7.4 Cost model

The blob spends 8.5 ms per pair single-threaded, of which 6.4 ms is TensorRT inference on
2048×2048 keypoints. A batched Python runner (the `lightglue_aliked_b16.onnx` already exported by
`tools/export_cusfm_raco.py`) should beat it, since the blob pays a full engine load per task
file. Budget: 339 pairs in <3 s to stay inside the 2× rule of `docs/open-pipeline-plan.md`.

---

## 8. Evidence index

| Claim | Evidence |
|---|---|
| Per-pair driver | `data/cusfm_re/decomp/matcher/nvidia__isaac__visual__cusfm__FeatureMatcher__ComputeMatches_001539c0.c` |
| LightGlue call chain | `…LightGlueMatcher__MatchKeypoints_001fc850.c`, `…LightGlueMatcher__RunWithTensorrt_001fb2a0.c` |
| Score threshold (strict `>`) | `…LightGlueAlikedTensorRT__PostProcess_00200260.c:150-175` |
| Keypoint normalisation | `…LightGlueTensorRT__NormalizedKeyPoints_00207a80.c` |
| NMS gate + SSC call | `…LightGlueMatcher__ApplyNonMaximumSuppression_001fbdb0.c:105-108,263-265` |
| SSC algorithm | `…GlobalNonMaximumSuppressionViaSquareCoveringForSortedPoints_…_0021e370.c` |
| Verification gate + thresholds | `…FeatureMatcher__TwoViewVerification_00153580.c` |
| `n_inliers < 10` rule | `…DataSelector__SelectMatchesByEpipolarGeometry_00189ed0.c:62-68` |
| `cv::findEssentialMat` call | `…DataSelector__SelectMatchesByOpenCVEssentialMat_00188f30.c` |
| Pixel→normalized threshold | `…MapErrorToNormalizedPlane_001a2f40.c` |
| Only OpenCV geometry call | `strings feature_matcher_main | grep '^_ZN2cv' | c++filt` — `findEssentialMat` only |
| ONNX bindings | `pycusfm/models/aliked_lightglue/lightglue_aliked.onnx` |
| Config sweep | `data/cusfm_re/analysis/sweep_matcher.sh`, outputs in `data/cusfm_re/runs/sweep_matcher/` |
| Result statistics | `data/cusfm_re/analysis/parse_matches.py`, `data/cusfm_re/analysis/compare_results.py` |

## 9. Open questions

1. **Exact SSC parity.** The binary search bounds were read off the decompiler and match the
   published SSC formulation, but I did not re-derive them symbolically. If the Python NMS keeps
   a different number of matches than 400/pair on Galileo, that is the first place to look. **[L]**
2. **`RunWithTensorrt` input padding.** With `--batch_size 0` the engine runs one pair at a time
   with dynamic shapes; the batched path (`BatchMatchInference`) pads to the profile max and logs
   `inefficient BatchMatchInference input batch size not equal max batch size`. The padding rule
   for uneven keypoint counts inside a batch was not decompiled. **[I]**
3. **`GeneralFeatureMatcher::MatchKeypoints` constraint callback.** `ComputeMatches` passes an
   empty `std::function<bool(uint,uint)>` and a null fundamental matrix, so
   `match_radius_pixels`, `epipolar_threshold_pixels` and `max_abs_level_delta` are inert on the
   LightGlue path. Confirmed by inspection, not by sweep. **[L]**
4. **`SaveMapPoint2DFile` under parallel workers.** With multiple task files every worker writes
   `map_keypoints.pb` to the same path; the last writer wins. Harmless because the content does
   not depend on the task, but a Python port should write it once. **[I]**

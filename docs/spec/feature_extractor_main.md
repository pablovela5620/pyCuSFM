# `feature_extractor_main` — implementation spec

Status: reverse-engineered from `pycusfm/x86_cuda13/bin/feature_extractor_main` (unstripped ELF,
C++ symbols, embedded source paths `/home/jryu/workspaces/visual_mapping/src/...`), Ghidra
decompilation under `data/cusfm_re/decomp/extractor/`, the shipped configs, the completed runs
under `data/cusfm_runs/`, and **9 re-runs of the binary with mutated flags** under
`data/cusfm_re/runs/sweep_extractor/`. Evidence and confidence per claim: **[C]** confirmed
(decompiled body, string literal, or exact numeric match against a real or re-run artifact),
**[L]** likely, **[I]** inferred.

---

## 1. Purpose and position in the pipeline

Stage 1. It reads the input trajectory (`frames_meta.json`), **selects a subset of synchronised
samples as keyframes**, runs ALIKED on each selected image, and writes one keyframe protobuf per
image plus a filtered `frames_meta.json`.

```
<input_dir>/<camera>/<ts>.jpeg  ─┐
<input_dir>/frames_meta.json    ─┼─► feature_extractor_main ─► keyframes/<camera>/<ts>.pb
models/aliked_lightglue/*.onnx  ─┤                             keyframes/frames_meta.json
configs/…/keypoint_creation…    ─┘                             keyframes/logs.txt
```

Call site: `pycusfm/cusfm_runner.py:243-309`. **[C]**

### What this repo already covers

**`tools/raco_extract.py` already implements the entire per-image half of this stage.** Its
protos are built from cuSFM's own `FileDescriptorSet`, and parsing a cuSFM-produced keyframe and
re-serialising it reproduces the file byte for byte (`data/cusfm_schema/README.md`,
`tests/test_cusfm_raco_contract.py`) — so the *layout* is pinned; the *values* come from a
different (RaCo) ALIKED graph and are not expected to match the blob bit for bit.

| Behaviour | Covered by `tools/raco_extract.py`? |
|---|---|
| JPEG decode, stretch resize to the network size, RGB planar, `/255` | **yes** (`prepare_frame`, lines 671-700) — same operations as cuSFM's `cv::resize` + `cv::split` + `convertTo(1/255)` |
| TensorRT FP16 engine build + content-addressed cache | **yes** (`build_fp16_engine`, `RacoEngine`) |
| Normalised keypoints → original-image pixels | **yes** (`normalized_to_pixels`) — same formula as the blob |
| Per-keypoint RGB sampling (`--output_rgb`) | **yes** (`sample_keypoint_rgb`) |
| `Keyframe` proto layout (`keypoint_vector`, descriptor bytes, `depth = -1`, `bev_start_idx = -1`) | **yes** (`serialize_keyframe`) |
| `frames_meta.json` output with `detector_type`/`descriptor_type` | **partly** (`prepare_metadata_collection`) — see §7.2 for the `synced_sample_id` bug |
| Batching + threading | **yes** (`run_extraction`) |
| **Keyframe selection** (`min_inter_frame_distance` / `min_inter_frame_rotation_degrees`) | **no** — raco_extract keeps every input frame |
| Sample synchronisation (`sample_sync_threshold_microseconds`) | **no** |
| Detector score threshold + detector-side NMS | **no** (moot with the shipped ONNX — see §6.3) |
| Masks, RGBD/stereo depth | **no** (unused by the isaac pipeline) |

So the only genuinely missing piece for the open pipeline is **§5, keyframe selection**, which is
now fully specified and reproduced exactly in `data/cusfm_re/analysis/keyframe_selection.py`.

---

## 2. CLI flags

From `data/cusfm_re/help/feature_extractor_main.txt` and `pycusfm/cusfm_runner.py:257-309`. **[C]**

| Flag | Type | Default | isaac demo value | Meaning |
|---|---|---|---|---|
| `--input_image_directory` | string | "" | `<ws>/input` | Root holding `<camera_name>/<ts>.jpeg` |
| `--frames_meta_file` | string | "" | `<input>/frames_meta.json` | Input trajectory + calibration |
| `--output_dir` | string | "" | `keyframes/` | Keyframe protos + filtered metadata |
| `--keypoint_creation_config` | string | `configs/isaac/keypoint_creation_config.pb.txt` | same | `KeypointCreationParams` text proto |
| `--feature_type` | string | "" (**required**) | `aliked` | `aliked` \| `superpoint` \| `sift` |
| `--model_dir` | string | "" | `pycusfm/models` | **Parent** of `aliked_lightglue/` |
| `--min_inter_frame_distance` | double | 0.5 | **0.0** (demo) | Metres; see §5 |
| `--min_inter_frame_rotation_degrees` | double | 5 | 5 | Degrees; see §5 |
| `--sample_sync_threshold_microseconds` | int64 | 50 | 100 | Sample grouping window; see §4 |
| `--stereo_pair_non_baseline_max_distance` | double | 0 | 0.001 | Auto-derive stereo pairs when the metadata has none |
| `--num_thread` | int32 | 1 | from runner | Extraction threads |
| `--batch_size` | int32 | 0 | 0 | `0` → `Batch size is not set. Using single thread.`; `>0` → `BatchProcessFrames` |
| `--max_keypoints_num` | int32 | -1 | -1 | Overrides `max_key_points_for_tensorrt` when `> 0` |
| `--output_rgb` | bool | false | **true** (demo) | Sample per-keypoint RGB into `keypoint_vector.r/g/b` |
| `--rgbd_mode` | int32 | 0 | 0 | `1` RGBD, `2` stereo-as-RGBD (fills `keypoint_vector.depth`) |
| `--input_image_mask_directory` | string | "" | off | `FilterKeypointsByMask` |
| `--debug_dir` / `--debug_image_interval` | string/int | ""/100 | off | Keypoint overlays |

`--min_inter_frame_distance 0.0` is the demo's one functional deviation from upstream defaults
(`NOTES.md:19`): with 0.5 only 34 of Galileo's 226 frames survive. **[C]**

---

## 3. Input format (`frames_meta.json`)

JSON encoding of `protos.visual.general.KeyframesMetadataCollection`
(`data/cusfm_re/proto/keyframe_metadata.proto`). Fields the extractor consumes: **[C]**

```jsonc
{
  "keyframes_metadata": [ {
      "id": "2356",                                   // uint64, unique frame id
      "camera_params_id": "6",                        // uint64; ABSENT means 0 (proto default)
      "timestamp_microseconds": "1707938736136246",
      "image_name": "right_stereo_camera_left/1707938736136246532.jpeg",
      "camera_to_world": { "axis_angle": {"x":…,"y":…,"z":…,"angle_degrees":…},
                           "translation": {"x":…,"y":…,"z":…} },
      "synced_sample_id": "318"                       // recomputed by the extractor, see §4
  } ],
  "camera_params_id_to_camera_params": { "3": { "sensor_meta_data": {
        "sensor_name": "front_stereo_camera_right", "frequency": 30,
        "sensor_to_vehicle_transform": { "axis_angle": …, "translation": … } },
      "calibration_parameters": { "image_width": 1920, "image_height": 1200,
        "projection_matrix": { "data": [fx,0,cx,0, 0,fy,cy,0, 0,0,1,0], "row_count":3, "column_count":4 } } } },
  "camera_params_id_to_session_name": { … },
  "stereo_pair": [ {"left_camera_param_id":"2","right_camera_param_id":"3","baseline_meters":0.15} ],
  "initial_pose_type": "EGO_MOTION"
}
```

`camera_params_id: 0` is omitted from the JSON (proto2/3 default suppression). A Python reader
**must** default it to 0 — 29 of Galileo's 226 entries have no `camera_params_id`. **[C]**

Conventions: `camera_to_world` is the camera→world rigid transform (axis-angle + metres);
`sensor_to_vehicle_transform` is camera→vehicle. Timestamps are microseconds; image basenames are
nanoseconds. `camera_projector_factory.cc:18` logs `Creating pinhole camera projector` for both
shipped datasets. **[C]**

---

## 4. Sample synchronisation

`keyframe_utils.cc:122` logs `Adding sync info based on threshold: N us` on **every** run and
rewrites `synced_sample_id` from scratch: frames are sorted by timestamp and grouped into
*samples* whose timestamps fall within `sample_sync_threshold_microseconds`; groups are then
numbered **1..N in time order**. **[C]**

Evidence: Galileo's input carries `synced_sample_id` 318…346 and the output carries 1…29;
raising the threshold to 100 000 µs merges neighbouring samples and changes the group count
(`data/cusfm_re/runs/sweep_extractor/sync100000/`). **[C]**

At the shipped 100 µs the recomputed grouping is identical to the input grouping on both Galileo
(samples span ≤ 19 µs, inter-sample gaps 33 ms) and RoboCap. **[C]**

`keyframe_utils.cc:200` logs `Already have stereo pairs, not populating new stereo pairs` when
`stereo_pair` is present. When absent, pairs are derived from the extrinsics using
`stereo_pair_non_baseline_max_distance` (two cameras form a stereo pair when their y/z
translation difference in the vehicle frame is below it, per the flag help text). **[L]**

---

## 5. Keyframe selection — **CONFIRMED, exactly reproduced**

`FeatureExtractor::ExtractSelectedFrameIndices` (profile timer name; the selection is logged as
`Total N frames selected from M raw frames.`, `feature_extractor.cc:732`). **[C]**

```
samples   = group frames by synced_sample_id (§4), ordered by sample id (== time order)
ref(s)    = vehicle_to_world of sample s
          = camera_to_world(frame) ∘ sensor_to_vehicle(camera_of_frame)⁻¹
            evaluated on ANY frame of the sample (all frames of a sample agree, being one rig pose)
kept      = [samples[0]]                       # the first sample is always kept
R_k, t_k  = ref(samples[0])
for s in samples[1:]:
    R, t = ref(s)
    if ‖t − t_k‖ ≥ min_inter_frame_distance                       # straight-line displacement, metres
       or angle(R_kᵀ · R) ≥ min_inter_frame_rotation_degrees:     # relative rotation, degrees
        kept.append(s); R_k, t_k = R, t
selected_frames = every frame of every kept sample
```

Key properties: **selection is at the sample (rig) level, never per camera**; the criteria are
**OR**-combined; the reference pose is the **vehicle/rig frame**, not any single camera;
and the distance is **straight-line displacement from the last kept sample**, *not* accumulated
path length. **[C]**

Reproduced exactly by `data/cusfm_re/analysis/keyframe_selection.py`:

| dataset | flags | blob | Python | identical frame-id set |
|---|---|---|---|---|
| Galileo (226 frames, 29 samples) | `d=0.5 r=5` | 34 | 34 | yes |
| Galileo | `d=0.1 r=5` | 46 | 46 | yes |
| Galileo | `d=0.02 r=5` | 226 | 226 | yes |
| Galileo | `d=0.5 r=0` | 226 | 226 | yes (rotation ≥ 0 is always true) |
| Galileo | `d=0.5 r=0.5` | 226 | 226 | yes |
| Galileo | `d=99 r=999` | 6 | 6 | yes (only the first sample, which has 6 frames) |
| Galileo | `d=0.0 r=5` | 226 | 226 | yes |
| **RoboCap (4528 frames, 1132 samples, 123 m path)** | `d=0.5 r=999` | **880** | **880** | **yes** |

The RoboCap case is what settles displacement-vs-path-length: on that curving trajectory
"traveled distance" predicts 896 frames / 224 samples with 432 wrong ids, straight-line
displacement predicts 880 / 220 with a set-identical result. **[C]**

Two gotchas a naive port hits:
* using one camera's pose instead of the vehicle frame gives the right *count* on Galileo but the
  wrong *samples* (318,323,**327**,332,… instead of 318,323,**328**,333,…);
* `camera_params_id` missing from JSON means 0, not "absent".

Rotation angle convention: `angle = acos((trace(R_kᵀ·R) − 1) / 2)` in degrees; equivalently
`2·atan2(‖vec(q)‖, |w|)` on the relative quaternion, which is the form the binary uses elsewhere
(`DataSelector::FilterByAngularDistance`). **[C]**

---

## 6. Feature extraction (ALIKED)

### 6.1 Config

`pycusfm/configs/isaac/keypoint_creation_config.pb.txt:15-41`. `KeypointCreationParams` /
`LearningBasedDetectorParams` (`data/cusfm_re/proto/keypoint_creation_params.proto`): **[C]**

```
aliked_detector {
  max_key_points_for_tensorrt: 4000                        # never binds — see §6.3
  detector_threshold: 0.005                                # LIVE (barely)
  num_points_tolerance_fraction: 0.3                       # detector NMS tolerance — dead
  global_non_max_method: SUPPRESSION_VIA_SQUARE_COVERING   # dead
  tensorrt_config { model_id: 0  model_name: "aliked"  use_fp16: true
    input_dimension { min: [1,3,100,100]  opt: [1,3,1200,1920]  max: [1,3,2400,4000] } }
}
```

The `av` profile only differs in `max_key_points_for_tensorrt: 1024` and no `use_fp16`. **[C]**

### 6.2 Preprocessing

`AlikedTensorRT::Inference` (`data/cusfm_re/decomp/extractor/…Inference_00222d80.c:158-335`):
**[C]**

```
orig_size = (image.cols, image.rows)                 # stashed for the pixel mapping
if image size != engine input size (h, w from the TRT profile):
    cv::resize(image, resized, Size(w, h))           # default INTER_LINEAR, aspect ratio NOT kept
if 1-channel: cv::cvtColor(GRAY -> BGR)
cv::split(resized, ch)                               # ch = [B, G, R]
plane0 <- ch[2].convertTo(CV_32F, 1/255)             # R
plane1 <- ch[1].convertTo(CV_32F, 1/255)             # G
plane2 <- ch[0].convertTo(CV_32F, 1/255)             # B
```

So the network sees **planar RGB float32 in [0, 1], stretched to the profile's `opt` size
(1200×1920 for isaac)**. No mean/std normalisation, no letterboxing. `tools/raco_extract.py:690-698`
already reproduces this exactly. All work is CPU OpenCV — CV-CUDA in this binary is only linked
for the SIFT detector and the pairwise matcher (`cvcudaSIFTCreate`, `cvcudaPairwiseMatcherCreate`
are the only CV-CUDA entry points). **[C]**

### 6.3 ONNX bindings and post-processing

`pycusfm/models/aliked_lightglue/aliked.onnx`: **[C]**

| dir | name | shape | dtype |
|---|---|---|---|
| in | `image` | `[1, 3, 1200, 1920]` | float32 |
| out | `keypoints` | `[K, 2]` | float32, normalised to `[-1, 1]` |
| out | `descriptors` | `[K, 128]` | float32, L2-normalised |
| out | `scores` | `[K]` | float32 |

Post-processing (`…Inference_00222d80.c:636-700`): **[C]**

```c
for i in 0..K:
    if (scores[i] >= detector_threshold) {               // >= , not >
        x = (kx[i] + 1) * 0.5 * (orig_width  - 1);
        y = (ky[i] + 1) * 0.5 * (orig_height - 1);
        keep(i, x, y, response = scores[i]);
    }
if (kept.size() > max_key_points_for_tensorrt)
    ApplyNonMaximumSuppression(kept)   // SSC, target = max_key_points_for_tensorrt,
                                       // tolerance = num_points_tolerance_fraction
descriptor_dim = 128
```

The pixel mapping is exactly `tools/raco_extract.py:163-181`
(`(normalized + 1) * [(W−1)/2, (H−1)/2]`) and maps back to the **original** image size, not the
network size. **[C]**

**`max_key_points_for_tensorrt: 4000` never binds** with the shipped ONNX: the graph's own
detector head emits a fixed top-2048, so the SSC branch is dead. Measured on 226 Galileo
keyframes (all exactly 2048) and 880 RoboCap keyframes (2047 or 2048 — the odd one is the
`detector_threshold` filter removing a single point). **[C]**

### 6.4 Optional post-steps

* `--input_image_mask_directory` → `FilterKeypointsByMask` drops keypoints landing on masked
  pixels (`car_mask_polygon` is also supported via `cv::fillConvexPoly`). **[L]**
* `--rgbd_mode 1/2` → `stereo_depth_estimator.cc` fills `keypoint_vector.depth` (and
  `KeyframesMetadataCollection.keypoint_feature_has_depth`); with `rgbd_mode 0` every depth is
  `-1.0`. **[C]**
* `--output_rgb` → per-keypoint RGB sampled from the **original** image at the rounded keypoint
  coordinate (`tools/raco_extract.py:183-201`). The demo enables it; every shipped run has
  `r/g/b` populated. **[C]**

---

## 7. Output format

### 7.1 `keyframes/<camera_name>/<image_stem>.pb`

`protos.visual.general.Keyframe`, binary. `<camera_name>` and `<image_stem>` come straight from
`image_name` (`right_stereo_camera_left/1707938736136246532.jpeg` →
`right_stereo_camera_left/1707938736136246532.pb`). **[C]**

Fields actually written by the extractor, verified against a real Galileo keyframe and a
`tools/raco_extract.py` keyframe (both 2048/2038 points, ~1.08 MB): **[C]**

| field | value |
|---|---|
| `is_valid` | `true` |
| `keypoint_vector.x`, `.y` | float32 pixels in the original image |
| `keypoint_vector.descriptors` | `n · 128` little-endian float32, L2-normalised |
| `keypoint_vector.descriptor_type` | `ALIKED_DESCRIPTOR` (4) |
| `keypoint_vector.bev_start_idx` | `-1` |
| `keypoint_vector.image_width` / `image_height` | **original** image size (1920×1200 Galileo, 1920×1080 RoboCap) |
| `keypoint_vector.depth` | `n × -1.0` when `rgbd_mode 0` |
| `keypoint_vector.r/g/b` | `n` uint32 each when `--output_rgb` |
| `keypoints`, `keypoint3d_indices`, `metadata`, `frame_id`, `inliers` | **not set** at this stage |

`frame_id` and `inliers` are filled later (`matches/map_keypoints.pb` carries `frame_id` and an
all-false `inliers` array of length `n`). **[C]**

### 7.2 `keyframes/frames_meta.json`

The input collection, filtered to the selected frames, with three changes: **[C]**

1. `synced_sample_id` renumbered densely **1..N** in time order (Galileo: 318…346 → 1…29).
2. `detector_type: ALIKED_DETECTOR` added.
3. `descriptor_type: ALIKED_DESCRIPTOR` added.

Everything else (`camera_params_id_to_camera_params`, `camera_params_id_to_session_name`,
`stereo_pair`, `initial_pose_type`, and each entry's `id` / `camera_params_id` /
`timestamp_microseconds` / `image_name` / `camera_to_world`) is copied verbatim.

> **Bug in the existing Python code.** `tools/raco_extract.py:293-320`
> (`prepare_metadata_collection`) renumbers with `synced_sample_id = int(input) + 1`. That happens
> to be right for RoboCap (input ids start at 0 and are contiguous) but is wrong for Galileo
> (318 → 319 instead of 1). Downstream stages only use `synced_sample_id` for grouping, so it is
> currently harmless — but it should become "dense rank in time order, 1-based". **[C]**

---

## 8. Python replacement

### 8.1 Work plan

1. **Reuse `tools/raco_extract.py` unchanged** for preprocessing, inference, keypoint mapping,
   RGB sampling and proto serialisation.
2. **Add a selection pass** in front of `collect_frame_tasks` — port
   `data/cusfm_re/analysis/keyframe_selection.py` (≈40 lines: group by `synced_sample_id`, build
   vehicle poses from `camera_to_world ∘ sensor_to_vehicle⁻¹`, greedy OR-threshold walk).
3. **Add sample synchronisation** if the input ever lacks `synced_sample_id` (both shipped
   datasets have it).
4. **Fix the `synced_sample_id` renumbering** to a dense 1-based rank.

### 8.2 Mapping to pycolmap

This stage has no pycolmap equivalent — COLMAP has no keyframe-selection concept and its feature
extractors (SIFT) are not what cuSFM runs. The pycolmap-facing outputs are:

| cuSFM output | pycolmap |
|---|---|
| selected frames + `camera_to_world` | `Database.add_image` + `Image.cam_from_world` (note: COLMAP stores **world→camera**, so invert) |
| `camera_params_id_to_camera_params` | `Database.add_camera(model="PINHOLE", width, height, params=[fx, fy, cx, cy])` |
| `keypoint_vector.x/y` | `Database.write_keypoints(image_id, xy_float32_Nx2)` |
| `keypoint_vector.descriptors` | **no fit** — COLMAP descriptors are `uint8[N, 128]`; keep ALIKED descriptors in a side array (`.npy`) for the LightGlue matcher |
| `stereo_pair` / `sensor_to_vehicle_transform` | `pycolmap.Rig` / `RigConfig` (rig-frame BA, matching cuSFM's `ba_frame_type: VEHICLE_RIG`) |

### 8.3 Cost model

Blob: 226 Galileo frames in 12.6 s (55 ms/frame) single-threaded, of which ALIKED network execute
is 40 ms and preprocessing 6.3 ms; RoboCap 880 frames in 36.8 s (42 ms/frame). Plus a one-off
~205 s TensorRT engine build. `tools/raco_extract.py` already benchmarks the batched path
(`--benchmark-only`) and should beat this comfortably at batch 8-16. **[C]**

---

## 9. Evidence index

| Claim | Evidence |
|---|---|
| Selection log line and counts | `data/cusfm_runs/galileo/cusfm/keyframes/logs.txt`, `feature_extractor.cc:732` |
| Selection rule (exact reproduction) | `data/cusfm_re/analysis/keyframe_selection.py`, sweep in `data/cusfm_re/analysis/sweep_extractor.sh`, outputs in `data/cusfm_re/runs/sweep_extractor/` |
| Vehicle-frame reference | brute-force over all reference choices; only `camera_to_world ∘ sensor_to_vehicle⁻¹` reproduces the blob on both datasets |
| Displacement, not path length | RoboCap `d=0.5 r=999`: 880 (displacement, exact) vs 896 (path length, 432 wrong ids) |
| Sample regrouping | `keyframe_utils.cc:122`; `data/cusfm_re/runs/sweep_extractor/sync100000/frames_meta.json` |
| Preprocessing (resize / split / 1÷255 / RGB order) | `data/cusfm_re/decomp/extractor/nvidia__isaac__common__image__feature__AlikedTensorRT__Inference_00222d80.c:158-335` |
| Score threshold and pixel mapping | same file, lines 636-700 |
| Detector NMS never fires | keypoint counts over 226 Galileo + 880 RoboCap keyframes (`data/cusfm_re/analysis/keyframe_stats.py`; all 2047-2048 against a 4000 target) |
| Keyframe proto field set | `data/cusfm_schema/cusfm_protos.fdset` → `data/cusfm_re/proto/keyframe.proto`, `keypoint.proto`; real artifacts under `data/cusfm_runs/galileo/cusfm/keyframes/` |
| `frames_meta.json` diffs | `data/r2b_galileo/frames_meta.json` vs `data/cusfm_runs/galileo/cusfm/keyframes/frames_meta.json` |
| ONNX bindings | `pycusfm/models/aliked_lightglue/aliked.onnx` |

> Scratch note: `data/cusfm_re/runs/sweep_extractor/*/` keeps each run's `frames_meta.json`
> (which is what the selection evidence rests on) plus 8 sample keyframe protos; the remaining
> `.pb` files were deleted after measurement to reclaim 2 GB. Re-run
> `data/cusfm_re/analysis/sweep_extractor.sh` to regenerate them.

## 10. Open questions

1. **Sample-grouping rule at large thresholds.** At 100 µs the grouping is unambiguous and
   matches the input. At 100 000 µs the observed groups (spans 33 ms and 67 ms) fit neither a
   pure "within threshold of the group's first frame" nor a pure chaining rule. Irrelevant for
   the shipped configs. **[I]**
2. **`--sample_sync_threshold_microseconds 0`** made the binary exit without writing
   `frames_meta.json`; not investigated. **[I]**
3. **Which frame of a sample supplies the reference pose.** All frames of one sample give the
   same vehicle pose up to calibration noise, so any of them reproduces the blob; the tie-break
   (lowest `camera_params_id` vs first in file order) is unobservable. **[I]**
4. **Batched path (`--batch_size > 0`).** `BatchProcessFrames` / `BatchDetectAndCompute` were not
   decompiled; the demo uses `--batch_size 0`. `tools/raco_extract.py` already batches
   independently. **[I]**
5. **Mask semantics** (`FilterKeypointsByMask`, `car_mask_polygon`) — not exercised by any
   shipped dataset. **[I]**

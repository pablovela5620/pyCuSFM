# Export — `kpmap_to_colmap`, `extract_pose_from_map_main`, `update_keyframe_pose_main`

Implementation spec for the cuSFM export stage and its Python + pycolmap replacement.
Stage 8 of `docs/open-pipeline-plan.md`. Upstream stage: `docs/spec/keypoints_mapper_main.md`.

Evidence classes: **CONFIRMED** (read off a real artifact, a verbose re-run, a numeric check,
or a decompilation), **INFERRED**, **UNKNOWN**.

> **Reproduction status.** Running `kpmap_to_colmap --map_dir data/cusfm_runs/galileo/kpmap
> --output_dir <scratch>` with **no other flags** reproduces `images.txt` and `points3D.txt`
> **byte-identical** to the shipped `data/cusfm_runs/galileo/sparse/`, and the log matches
> line for line. `cameras.txt` has the same 8 lines in a different order
> (`unordered_map` iteration; the sorted diff is empty). Everything marked CONFIRMED below
> rests on that reproduction, a decompilation, or a numeric check.

---

## 1. Purpose and pipeline position

Three small tools turn the cuSFM map into portable formats. None of them optimises anything.

```
kpmap/map_keypoints.pb  +  kpmap/keyframes/frames_meta.json
        |                                |
        |                                +--> extract_pose_from_map_main --> output_poses/*.tum
        |                                                                          |
        +--> kpmap_to_colmap --> sparse/{cameras,images,points3D}.txt               v
                                                              update_keyframe_pose_main
                                                                 --> output_meta/<track>/frames_meta.json
```

Call sites: `pycusfm/cusfm_runner.py` `convert_map()` (kpmap_to_colmap) and `convert_poses()`
(extract_pose_from_map_main, then update_keyframe_pose_main per track when the input is
multi-track).

**CONFIRMED.**

---

## 2. Shared conventions

| Concept | cuSFM | COLMAP |
|---|---|---|
| Stored pose | `camera_to_world` = **`world_T_cam`** | `cam_T_world` |
| Rotation encoding | `AxisAngleDegree` — unit axis `(x,y,z)` + `angle_degrees` | quaternion `QW QX QY QZ` |
| Translation | metres | metres |
| Timestamps | `timestamp_microseconds`, uint64 | — |
| Camera frame | RDF (right, down, forward), OpenCV | same |
| Vehicle frame | FLU (forward, left, up) | — |
| World frame | vehicle FLU frame at the **first keyframe** | — |
| Rig extrinsic | `sensor_meta_data.sensor_to_vehicle_transform` = **`vehicle_T_cam`** | `sensor_from_rig` = `cam_T_rig` (the inverse) |
| Rig frame grouping | `synced_sample_id` | `pycolmap.Frame` |
| Sensor identity | `camera_params_id` | `CAMERA_ID` |

The RDF-to-FLU axis swap is **baked into `sensor_to_vehicle_transform`**; there is no separate
fixed rotation anywhere in the code path. `back_stereo_camera_right`'s extrinsic is
`axis=(-0.5784, -0.5768, 0.5768), angle=119.877 deg` — a ~120 deg rotation about a body
diagonal, i.e. exactly the RDF->FLU permutation composed with the camera's yaw. So
`world_T_cam * (vehicle_T_cam)^-1` lands in FLU automatically.

The world frame is **not** ENU: `reference_latlngalt` is `{0,0,0}`, the merged vehicle TUM
starts at `t ~ (1.4e-5, -6.1e-7, -1.1e-7)` with `q ~ identity`, and the camera positions at
`t0` equal the `sensor_to_vehicle_transform` translations to 1e-5 m.

**CONFIRMED**, `docs/camera.md` §Coordinate Systems plus numeric checks.

---

## 3. `kpmap_to_colmap`

### 3.1 Flags

| Flag | Default | Passed by the runner | Meaning |
|---|---|---|---|
| `--map_dir` | `""` | always | **Required.** Holds `map_keypoints.pb` and `keyframes/`. |
| `--output_dir` | `""` | always | **Required.** Destination for the sparse model. |
| `--binary` | false | if `export_binary_colmap_files` | Write `.bin` instead of `.txt`. |
| `--keyframe_dir` | `""` | never | Override the `keyframes` subdirectory. |
| `--vocabulary_dir` | `""` | if the directory exists | Loads the BoW vocabulary. |
| `--bow_index_file` | `""` | if the file exists | Loads the BoW index. |
| `--skip_images_without_keypoints` | **true** | never | Omit images with no 2-D keypoints. |

### 3.2 Inputs and outputs

Reads exactly **two** files under `--map_dir`:

- `map_keypoints.pb` -> `MapKeyPoints` (geometry, observations, colours, **and the
  keyframes themselves**).
- `keyframes/frames_meta.json` -> `KeyframesMetadataCollection` (poses and calibration).
  `--keyframe_dir` overrides the subdirectory name.

It does **not** read the per-camera `keyframes/<cam>/*.pb` keyframe files. All keypoints come
from `MapKeyPoints.frames[].keypoint_vector`. Log evidence:
`keyframe_cacher.cc:50 Keyframe manager load from map_keypoint_file frame number:32`.
**CONFIRMED.**

`--vocabulary_dir` and `--bow_index_file` have **no effect on the output**. Verified by
running with and without: `images.txt` and `points3D.txt` byte-identical, no extra files, no
`database.db`. They only trigger the BoW load inside `KeyframesKeypointsMap`, a class other
tools share. A reimplementation can ignore both. **CONFIRMED.**

Writes exactly three files — `cameras`, `images`, `points3D` — with `.txt` or `.bin`. **No
`database.db` and no `project.ini`.** `ColmapManager::WriteMapperIni`,
`WriteFeatureExtractorIni`, `WriteSequentialFeatureMatcherIni`, `WriteImageList` and
`PrepareProjectFiles` all exist in the binary but `ExportSparseFolder` never calls them.
**CONFIRMED.**

Text formatting: `ofs.precision(15)`, default (non-fixed) float format, and a **trailing
space after every field**. **CONFIRMED** (byte-identical reproduction).

`--skip_images_without_keypoints=true` drops images whose keypoint vector is absent or empty.
Galileo: 32 -> **28** images; the four dropped are ids 2471, 2487, 2530, 2531, all
`left_stereo_camera_*`. The mapper's own log flags one of them:
`Num points in frame: Min:0 pt(frame 2531)`. With `=false` they are emitted with a pose,
camera and name plus an **empty second line**, and the header reads
`# Number of images: 32`. `points3D.txt` is unchanged either way, and **all 8 cameras are
written regardless**. **CONFIRMED.**

### 3.3 `cameras.txt` — camera model mapping

One COLMAP camera per **`camera_params_id`**, not per image.

| `CameraProjectionModelType` | COLMAP MODEL | Source | PARAMS order |
|---|---|---|---|
| `PINHOLE` (0) | `PINHOLE` | `projection_matrix` (3x4, row-major) | `fx=P[0] fy=P[5] cx=P[2] cy=P[6]` |
| `DISTORTED_PINHOLE` (1) | `FULL_OPENCV` | `camera_matrix` (3x3) + `distortion_coefficients` | `fx=K[0] fy=K[4] cx=K[2] cy=K[5]`, then 8 coefficients `k1 k2 p1 p2 k3 k4 k5 k6` |
| `FTHETA_WINDSHIELD` (2) | `PINHOLE` | `projection_matrix` | `fx fy cx cy`, plus `LOG(WARNING) "Camera parameter N is FTHETA_WINDSHIELD, thus the output in rectified model may be inaccurate."` |
| `OPENCV_FISHEYE` (3) | `OPENCV_FISHEYE` | `camera_matrix` + `distortion_coefficients` | `fx fy cx cy k1 k2 k3 k4` |
| anything else | — | — | `LOG(FATAL) "Unsupported camera projection model type: <name>"` |

The enum has only these four values, so the table is complete. Corroboration: only three
COLMAP model-name strings exist anywhere in the binary — `PINHOLE`, `FULL_OPENCV`,
`OPENCV_FISHEYE`, at rodata `0x244b0c`, `0x244b14`, `0x244b20`.

Note the asymmetry: models 0 and 2 read the 3x4 **`projection_matrix`**; models 1 and 3 read
the 3x3 **`camera_matrix`**. A missing `projection_matrix` for models 0/2 is
`LOG(FATAL) "Projection matrix is not set for camera parameter N"`.

Distortion policy: more than 8 (resp. 4) coefficients ->
`LOG(WARNING) "Too many distortion coefficients for camera N. Only writing the first 8"` and
truncate; fewer -> `"Too few ... Will pad with zeros."`; absent ->
`"No distortion coefficients found for camera N. Will pad with zeros."`

**FTheta is exported as a rectified PINHOLE, not rejected.** `docs/camera.md` line 392 agrees
("falls back to PINHOLE using the projection matrix if available"), though its summary table
row says "Not supported" — **the table is misleading**; the binary emits a PINHOLE line. This
is the one genuinely lossy conversion: the F-Theta polynomial and any windshield model are
discarded.

The shipped model (note the trailing space on every line):

```
# Number of cameras: 8
7 PINHOLE 1920 1200 852.918823242188 852.918823242188 933.233764648438 557.628784179688 
2 PINHOLE 1920 1200 841.564514160156 841.564514160156 936.249633789062 566.368103027344 
...
```

Cameras 0/1, 2/3, 4/5, 6/7 are the left/right halves of the four rectified stereo pairs,
hence the identical intrinsics and `fx == fy`. Line **order is non-deterministic**; only the
set is stable.

**CONFIRMED** (`colmap_manager.cc` decompiled, plus the shipped file).

### 3.4 `images.txt` — pose conversion

Chain, from the decompiled `ColmapManager::WriteImagesFile`:

1. Read `camera_to_world` = `{AxisAngleDegree{x,y,z,angle_degrees}, Translation3d{x,y,z}}`,
   i.e. `world_T_cam`.
2. `AxisAngleDegreeToQuaternion` — Rodrigues with `theta = radians(angle_degrees)` about the
   (already unit) axis -> `R = world_R_cam`.
3. `SE3Transform<double>::Inverse()` -> `R' = R^T`, `t' = -R^T t`.
4. Normalise the quaternion; **if `w < 0`, negate all four components** (canonical `w >= 0`).
5. Write `QW QX QY QZ TX TY TZ`. cuSFM stores Eigen `(x,y,z,w)` order internally and reorders
   on write.

Numeric verification (`data/cusfm_re/analysis/export_verify.py`), image 2559:

```
camera_to_world: axis=(-0.6497364711905401, -0.5354993856596588, 0.5395210153117233)
                 angle_degrees=113.49233777718922
                 t=(-0.19074984903419873, 0.1490971270102635, 0.350389792773112)
colmap  q(wxyz) = [0.548349146992837  0.543341794014345  0.447811089263254 -0.451174175017098]
derived q(wxyz) = [0.548349146992837  0.543341794014345  0.447811089263254 -0.451174175017098]
colmap  t = [-0.110030856003945  0.348453786441002 -0.218773020527753]
derived t = [-0.110030856003945  0.348453786441002 -0.218773020527753]
|dq|max = 4.44e-16   |dt|max = 4.03e-16
```

**Worst case over all 28 images: `|dt|max = 6.4e-16 m`, `|dq|max = 1e-16`.** Sign
canonicalisation confirmed independently: the minimum `QW` across all 28 lines is
`+0.000175198162354528`, never negative. **CONFIRMED.**

### 3.5 `images.txt` — identifiers and ordering

Verified 28/28 rows for each:

| Column | Value |
|---|---|
| `IMAGE_ID` | **`KeyframeMetaData.id`, verbatim.** Not 1-based, not reassigned; Galileo ids run 2359..2609 with gaps. |
| `CAMERA_ID` | **`camera_params_id`, verbatim.** Not remapped to a dense range — it is dense here only because the source keys are 0..7. |
| `NAME` | **`image_name`, verbatim.** Not synthesised. Upstream stages set it as `<camera_name>/<timestamp_nanoseconds>.jpeg`, e.g. `back_stereo_camera_right/1707938736636230528.jpeg` — note **nanoseconds in the filename** against **microseconds** in `timestamp_microseconds`. |

Line order is `unordered_map` bucket order (`2559, 2544, 2424, 2423, 2528, 2415, 2609, ...`),
neither ascending nor descending, but reproducible run to run. `points3D.txt` likewise.
**A reimplementation should not try to match the order** — COLMAP readers do not care.

**CONFIRMED.**

### 3.6 `images.txt` — the POINTS2D block

Written from `vector<Keypoint2DWith3DId>{double x; double y; int64 point3d_id}`.

- Only keypoints **in the map** are written; `MapKeyPoints.frames[].keypoint_vector` already
  contains only mapped keypoints. Verified `len(POINTS2D) == len(keypoint_vector.x)` for all
  28 images (image 2559: 200 keypoints).
- `X, Y` are `keypoint_vector.x[k]`, `.y[k]` **bit-for-bit** (`max|dx| = max|dy| = 0.0`; under
  a `+0.5` hypothesis the error is exactly `0.5`). These are **raw distorted pixel
  coordinates** — no undistortion, **no half-pixel shift**.
- `POINT3D_ID = keypoint3d_indices[k]` exactly, strictly ascending within a line.
  **0 occurrences of `-1`** across all 5835 observations. The `-1` path exists (the id is
  written through the signed `long` inserter, `0xFFFFFFFFFFFFFFFF` in binary) but cannot fire
  on kpmap input.

**Pixel-centre caveat.** COLMAP places the centre of the top-left pixel at `(0.5, 0.5)`;
ALIKED and OpenCV place it at `(0, 0)`. cuSFM copies **both** the keypoints and `cx, cy` in
the OpenCV convention, so the exported model is internally self-consistent — reprojection
errors are unaffected — but it sits half a pixel off a COLMAP-native model. **Do not mix
these keypoints with COLMAP-detected features.** CONFIRMED for the offset, INFERRED for the
consequence.

### 3.7 `points3D.txt`

Format `POINT3D_ID X Y Z R G B ERROR TRACK[(IMAGE_ID, POINT2D_IDX)...]`.

| Column | Source |
|---|---|
| `X Y Z` | `MapKeyPoints.points_x/y/z[i]` exactly (max delta 4.9e-15, float->double widening only). World frame = vehicle FLU at the first keyframe. Metres. |
| `R G B` | `MapKeyPoints.points_r/g/b[i]` (uint8); `0 0 0` if the index is out of range |
| `ERROR` | **hardcoded `2.0`**, not computed — `local_90 = 0x4000000000000000` at the `Point3D` construction site in `ColmapManager::InitFromKpMap` |
| `TRACK` | `MapKeyPoints.observations[i].keyframe_point2d_id` = `(keyframe_id, keypoint_id)`, **re-sorted ascending by `IMAGE_ID`** |

`MapKeyPoints` has **no error field at all**, so `ERROR` cannot be carried through; cuSFM
writes the constant instead of recomputing. All 1275 rows read `2`.
`ColmapManager::GetMaxReprojectionError()` takes the max over this same field, so it always
returns 2.0 — it is a meaningless quantity here.

The track re-sort is real: point 1274 is `[(2375,253),(2370,281),(2425,460)]` in the proto and
`2370 281 2375 253 2425 460` in the file. 0 of 1275 tracks violate ascending order.
`len(track) == points_num_observations[i]` for every point.

**In Galileo all 1275 points are `0 0 0`**, because the proto colour arrays are all zero —
despite the log line `keyframes_keypoints_map.cc:112 Loaded RGB colors for 1275 map points`.

**CONFIRMED.**

### 3.8 `--binary`

Writes `cameras.bin`, `images.bin`, `points3D.bin` in the standard COLMAP binary layout and
**nothing else**. Verified by parsing: `cameras.bin` is `uint64 n` then
`{uint32 id, int32 model_id, uint64 w, uint64 h, double params[]}` with `model_id = 1`
(PINHOLE); the other two match the text content exactly (point 1274: `xyz`, `rgb=(0,0,0)`,
`err=2.0`, track `[(2370,281),(2375,253),(2425,460)]`).

`ExportSparseFolder(dir, binary, skip_no_kp)` builds the three names from the literals
`"cameras"`, `"images"`, `"points3D"` plus `".txt"` / `".bin"`, and the log tail switches
between `"in text format"` and `"in binary format"`. **CONFIRMED.**

### 3.9 Shipped-model statistics

| Metric | Value |
|---|---|
| cameras | 8, all `PINHOLE 1920x1200` |
| images | 28 (32 keyframes, 4 skipped) |
| points3D | 1275 |
| total 2-D observations | 5835 (= sum of track lengths) |
| observations per image | min 1, mean 208.4, max 474 |
| track length | min 3, median 4, mean 4.576, max 8; histogram `{3:395, 4:350, 5:208, 6:144, 7:78, 8:100}` |
| `ERROR` | **2.0 for every point** |
| colour | **(0,0,0) for every point** |
| XYZ bbox (m) | min `[-5.172, -3.590, -0.111]`, max `[3.201, 1.337, 3.321]`, centroid `[-0.263, -0.612, 0.312]` |
| images per `CAMERA_ID` | 0:4, 1:4, 2:4, 3:4, **4:2, 5:2**, 6:4, 7:4 |

The track-length minimum of 3 is the mapper's `min_correspondences: 3`.

---

## 4. `extract_pose_from_map_main`

### 4.1 Flags

| Flag | Default | Passed by the runner |
|---|---|---|
| `--input_keyframe_metadata_file` | `""` | `<map_dir>/keyframes/frames_meta.json` |
| `--output_pose_dir` | `""` | `<ws>/output_poses` |
| `--output_pose_file_type` | `"tum"` | `"tum"`; empty is rejected by a `CHECK` |
| `--export_pose_in_vehicle_frame` | true | `True` |
| `--also_merge_poses` | true | passed as a bare flag |
| `--main_camera_name` | `""` | never |

### 4.2 The pose that is written

It is **derived**, not stored:

```
world_T_vehicle = SE3(camera_to_world) * SE3(sensor_to_vehicle_transform)^-1
```

Verified for keyframe 2609 against `output_poses/merged_pose_file.tum`:

```
tum     t = [ 0.50030147 -0.06450697  0.00376314]   q(xyzw) = [0.00168844 0.00502812 -0.14284095 0.98973144]
derived t = [ 0.50030147 -0.06450697  0.00376314]   q(xyzw) = [0.00168844 0.00502812 -0.14284095 0.98973144]
```

**Worst case over all 32 keyframes: `|dt|max = 3.6e-16 m`.** With
`--export_pose_in_vehicle_frame=false` the per-camera file is `camera_to_world` verbatim
(verified to `5.6e-17 m` over 32 poses). **CONFIRMED.**

### 4.3 TUM format

`PoseSerializer::TumRecord(ofstream&, uint64 timestamp_us, const SE3Transform&)`:
`os.precision(0x10)` and the float format forced to `fixed`, i.e. **`std::fixed`, 16
decimals**. Line:

```
timestamp tx ty tz qx qy qz qw
```

single-space separated, no trailing space, `\n` and a flush per line.

- **Timestamp is `(double)timestamp_microseconds * 1e-06` — a multiplication by the
  reciprocal, not a division by `1e6`.** Observable: for 4 of the 15 Galileo timestamps the
  two forms differ by one ULP, and the file always matches the multiply. `1707938736169573`
  is written `1707938736.1695728302001953`, whereas `us / 1e6` would print
  `1707938736.1695730686187744`. A Python port must multiply.
- Quaternion order is `qx qy qz qw` (Eigen coefficient order) — **the opposite end-placement
  from COLMAP's `QW QX QY QZ`**.
- **No `w >= 0` canonicalisation here**, unlike `images.txt`. The raw quaternion is written.
- The pose is `world_T_body` (body = vehicle or camera, per the flag).

**CONFIRMED.**

### 4.4 Ordering, deduplication and file names

Poses are collected into `std::map<uint64 microseconds, SE3>`, so lines come out **sorted
ascending by timestamp** and **duplicate microsecond timestamps collapse** (last write wins).
Galileo: 32 keyframes -> 15 unique timestamps -> 15 lines in `merged_pose_file.tum`.

| Flags | Files produced |
|---|---|
| default (vehicle, tum, merge) | `<dir>/<session>/camera_name-<sensor_name>_pose_file.tum` x8, plus `<dir>/merged_pose_file.tum` |
| `--also_merge_poses=false` | per-camera files only |
| `--output_pose_file_type=ego_motion` | the same names with `.pb` (an `EgoMotionTrajectory` proto) |
| `--main_camera_name=<name>` | **`<dir>/<session>.tum`** (i.e. `0.tum`) — no subdirectory, single camera |

The `0` subdirectory is the **session name** from `camera_params_id_to_session_name` (all 8
cameras map to `"0"` in Galileo). The per-camera stem is
`StringPrintf("camera_name-%s_pose_file", sensor_name)`; the extension is appended by
`ExportPoses` from `--output_pose_file_type` (rodata `.pb` @0xb95d0, `.tum` @0xb95d4);
an unknown type gives `LOG(ERROR) "Unsupported pose file type: <t>"`. The merged file lands
in `--output_pose_dir` itself, not the session subdirectory.

**CONFIRMED** (each flag re-run).

> **Trap.** With `--export_pose_in_vehicle_frame=false`, `merged_pose_file.tum` is
> meaningless: it mixes eight different cameras' world poses into one timestamp-keyed map and
> collapses the collisions. Only the per-camera files are usable in camera mode.

---

## 5. `update_keyframe_pose_main`

### 5.1 Flags

| Flag | Default | Meaning |
|---|---|---|
| `--input_file` | `""` | **Required.** Input `KeyframesMetadataCollection`. |
| `--output_file` | `""` | **Required.** Output collection. |
| `--tum_pose_file` | `""` | Vehicle poses, TUM, single session. |
| `--tum_pose_dir` | `""` | One TUM file per session name. |
| `--max_pose_gap_seconds` | 0.2 | Maximum interpolation gap. |

`--tum_pose_file` and `--tum_pose_dir` are mutually exclusive and at least one is required
(`CHECK(file.empty() || dir.empty())` and `CHECK(!(both empty))`).

### 5.2 What is written

```
camera_to_world = interp(world_T_vehicle, timestamp_microseconds) * sensor_to_vehicle_transform
```

Note `sensor_to_vehicle_transform` (= `vehicle_T_cam`) is applied on the right and **not**
inverted here; the inverse lives on the `extract_pose` side. The extrinsic per
`camera_params_id` is built once via `AxisAngleDegreeToQuaternion`, and the result is
converted back with `QuaternionToAxisAndAngleDegree` into the output `AxisAngleDegree`.
A missing id gives `CHECK` / `"Can't find camera-params-id N in calibration map"`.

Verified by re-running the tool on `pose_graph/frames_meta.json` +
`output_poses/merged_pose_file.tum` with `--max_pose_gap_seconds=10` and reimplementing the
composition in Python (linear + SLERP): **`|dt|max = 3.6e-08 m`, rotation residual
`1.7e-06 deg`** over all 32 keyframes. The residual is entirely the TUM text round-trip
described in §5.4, not a model error. **CONFIRMED.**

Field-by-field diff of output against input over 32 keyframes: **only `camera_to_world`
differs (32/32); every other field is byte-identical** — `id`, `track_id`,
`camera_params_id`, `timestamp_microseconds`, `image_name`, `synced_sample_id`,
`frame_to_pixel_per_row_name`, `alignment_id`, `sample_id`, `camera_id`, `mask_image_name`.
Collection-level fields are unchanged and keyframe order is preserved. Mechanically:
`KeyframeMetaData::CopyFrom(input)` then `RigidTransform3d::InternalSwap` on the new pose.
**CONFIRMED.**

If `camera_params_id_to_session_name` is empty it logs
`"camera_params_id_to_session_name is empty, generating fake session names"`.

### 5.3 The interpolator and `--max_pose_gap_seconds`

`FLAGS_max_pose_gap_seconds * 1000000.0` is stored as the maximum interval width **in
microseconds** in `LinearInterpolatorHelper<uint64>`. `GetLerpInterval(query_us)` does a
`lower_bound` binary search over the sorted sample timestamps, then:

- exact hit -> ratio 0, **valid** (short-circuits before the gap test);
- `lower_bound == begin` (query before the first sample) -> brackets `[0]`,`[1]` and
  **extrapolates backwards** with a negative ratio;
- otherwise -> brackets `[i-1]`,`[i]`, ratio `(q - t[i-1]) / (t[i] - t[i-1])`.

Invalid if **either** `max_gap != 0 && max_gap < (t_hi - t_lo)`, **or** the ratio exceeds a
stored maximum-extrapolation bound. `GetInterpolatedPose` returns `{SE3, bool valid}`;
translation is **linear** and rotation is **SLERP** (`SO3::ComputeSlerpAndJacobian<double>`
is linked in).

**A keyframe whose bracket exceeds the gap is DROPPED, not kept with its old pose:**

```
W update_keyframe_pose_main.cc:192 No pose available for keyframe 2415 in session 0 at timestamp 1707938736402913
W update_keyframe_pose_main.cc:192 No pose available for keyframe 2420 in session 0 at timestamp 1707938736402913
I update_keyframe_pose_main.cc:207 Keyframe metadata size updates from 32 to 30
```

With `--max_pose_gap_seconds=10` the same run gives `32 to 32`; with `0.0001` it still gives
`30`, because the 30 survivors are exact hits. **CONFIRMED.**

Other parse behaviour: `"Invalid row with timestamp = 0, skip it"`, a `CHECK` failure on an
unreadable TUM, and `"No pose file found for session: <s>"` in `--tum_pose_dir` mode.

### 5.4 The TUM microsecond round-trip is lossy — a real bug

`extract_pose_from_map_main` writes `us * 1e-6` at 16 fixed decimals;
`update_keyframe_pose_main` reads it back with a **truncating** `* 1e6`. So
`1707938736402913` round-trips to `1707938736402912`. The exact-hit lookup then fails, the
query brackets `[1707938736402912, 1707938736636230]` = 233 ms > 200 ms, and the frame is
silently discarded. That is exactly what dropped 2 of 32 frames above. The round-trip also
collapses 15 written samples to 13 unique ones.

**A reimplementation should carry microseconds as integers, or round rather than truncate on
parse.** **CONFIRMED.**

---

## 6. `frames_meta.json` as actually used

Proto-JSON of `KeyframesMetadataCollection`. **JSON keys are the proto field names verbatim**
(snake_case, i.e. `preserve_proto_field_names`). `uint64` fields are **strings**; `uint32` and
`double` are numbers; enums are name strings; **fields equal to their default are omitted**
(e.g. `camera_params_id` vanishes when it is 0 — a trap when parsing).

Collection level, all 16 keys present in Galileo:

| Key | Type | Value / units |
|---|---|---|
| `reference_latlngalt` | LatLngAlt | `{0,0,0}` — no geo anchor |
| `keyframes_metadata` | repeated | 32 entries |
| `initial_pose_type` | enum | `"ALIGNMENT"` in the kpmap output, `"EGO_MOTION"` in the pose_graph input |
| `camera_params_id_to_session_name` | map<uint64,string> | all `"0"` |
| `camera_params_id_to_camera_params` | map<uint64,CameraSensor> | 8 entries, keys `"0"`..`"7"` |
| `detector_type` / `descriptor_type` | enum | `"ALIKED_DETECTOR"` / `"ALIKED_DESCRIPTOR"` |
| `stereo_pair` | repeated StereoPair | 4 x `{left_camera_param_id, right_camera_param_id, baseline_meters ~ 0.15}` — **metres** |
| `camera_params_id_to_distorted_row_indices` | map | `{}` |
| `vehicle_trajectory_files`, `imu_recording_files` | repeated string | `[]` |
| `track_id_to_track_name`, `track_id_to_bag_name` | map | `{}` |
| `keypoint_feature_has_depth` | bool | `false` |
| `camera_params` | repeated CameraSensor | `[]` (legacy; the map is used) |
| `local_sector_key` | string | `""` |

Per keyframe, all 12 keys present on all 32:

```json
{
 "id": "2609",                                   // uint64 as string; == COLMAP IMAGE_ID
 "track_id": "0",
 "camera_params_id": "1",                        // uint64 as string; == COLMAP CAMERA_ID
 "timestamp_microseconds": "1707938736869567",   // uint64, MICROSECONDS since the Unix epoch
 "image_name": "back_stereo_camera_right/1707938736869567528.jpeg",  // == COLMAP NAME; filename is NANOseconds
 "camera_to_world": {                            // world_T_cam
  "axis_angle": {"x": -0.6860096450048235,       // unit axis, dimensionless
                 "y": -0.5124644716589568,
                 "z":  0.516498724342725,
                 "angle_degrees": 110.43680406216905},   // DEGREES
  "translation": {"x": -0.009022440313658842,    // METRES
                  "y":  0.16313734597729856,
                  "z":  0.3540842947261688}
 },
 "synced_sample_id": "22",                       // uint64 as string; the RIG FRAME id
 "frame_to_pixel_per_row_name": "",
 "alignment_id": "0",
 "sample_id": "0",
 "camera_id": 0,                                 // uint32 -> plain NUMBER, not a string
 "mask_image_name": ""
}
```

Optional and absent here: `pose_covariance` (MatrixD), `start_camera_to_world`
(written when rolling shutter correction is on).

`CameraSensor` holds `sensor_meta_data{sensor_id, sensor_type "CAMERA", model_name,
sensor_name, frequency (Hz), sensor_to_vehicle_transform}`, `chroma`,
`calibration_parameters{image_width, image_height, projection_matrix{data[12], row_count 3,
column_count 4}, rolling_shutter_delay_microseconds}`,
`horizontal/vertical_field_of_view_degrees`, `raw_pixel_aspect_ratio`, `car_mask_polygon`,
`camera_projection_model_type`.

**CONFIRMED.**

---

## 7. What bundle adjustment actually moved

Comparing the BA input `data/cusfm_runs/galileo/pose_graph/frames_meta.json` against the BA
output `data/cusfm_runs/galileo/kpmap/keyframes/frames_meta.json`. Same 32 ids, **frame count
unchanged (32 -> 32)**.

Per **camera** (all 32 keyframes):

| Statistic | Translation (mm) | Rotation (deg) |
|---|---|---|
| min | 0.0000 | 0.0000 |
| median | 6.0194 | 0.2946 |
| mean | 6.4090 | 0.2894 |
| p95 | 14.1401 | 0.5686 |
| max | 14.4514 | 0.5686 |

Per **rig frame** (`data/cusfm_re/notes/rig_check.py`, the rig origin rather than the camera
centres):

| Rig frame | Translation | Rotation |
|---|---|---|
| 1 (first) | **0.0000 mm** | **0.000000 deg** |
| 8 | 4.9744 mm | 0.215421 deg |
| 15 | 5.5435 mm | 0.373728 deg |
| 22 | 11.2740 mm | 0.568586 deg |

Best-fit global transform (Kabsch on the 32 camera centres, pre -> post):

- **rigid**: rotation 0.7232 deg, translation 6.5123 mm; residual mean **3.4763 mm**, max 7.8747 mm
- **similarity**: scale 1.00318441 (+0.32%); residual mean **3.2784 mm**, max 7.9252 mm

**Interpretation.** Roughly half the motion is a global rigid drift and half is genuine
non-rigid refinement: a single rigid transform pulls the mean displacement from 6.41 mm down
to 3.48 mm, explaining the bulk but leaving a comparable residual. Adding scale buys almost
nothing (3.48 -> 3.28 mm), so the +0.3% scale is noise, not scale drift. Rig frame 1 moves
**exactly** zero because BA holds one keyframe constant (Galileo: keyframe 2359, the lowest
id) and, through the shared rig, that pins its whole rig pose — see
`docs/spec/keypoints_mapper_main.md` §6.3. The motion grows monotonically with time, the
signature of a drift correction anchored at the first frame. Absolute magnitudes are
millimetres and sub-degree.

Camera intrinsics and rig extrinsics are numerically unchanged (translation delta 0.000 um,
rotation delta <= 1.2 microdegrees, which is axis-angle JSON rounding). `initial_pose_type`
changes `EGO_MOTION` -> `ALIGNMENT`.

**CONFIRMED** for the numbers; **INFERRED** for the rigid-versus-non-rigid split, given
n = 32.

---

## 8. pycolmap mapping

Probed against the environment's real build: **pycolmap 4.2.0, `has_cuda=True`**
(`data/cusfm_re/notes/pycolmap_probe.txt`).

```python
rec = pycolmap.Reconstruction()
rec.add_camera(pycolmap.Camera(camera_id=cpid, model="PINHOLE",
                               width=w, height=h, params=[fx, fy, cx, cy]))
img = pycolmap.Image(image_id=kf_id, name=kf.image_name, camera_id=cpid,
                     points2D=[pycolmap.Point2D(xy, point3D_id) for ...])
img.cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(quat_xyzw), tvec)  # cam_T_world
rec.add_image(img)
rec.add_point3D(xyz, pycolmap.Track([(image_id, point2D_idx), ...]), color=np.uint8([r, g, b]))
rec.write_text(dir)     # cameras.txt / images.txt / points3D.txt
rec.write_binary(dir)   # .bin
```

Direct equivalents (all three model names confirmed present in `pycolmap.CameraModelId`, with
identical parameter order, so §3.3 transfers 1:1):

| cuSFM | pycolmap |
|---|---|
| `PINHOLE` / `FULL_OPENCV` / `OPENCV_FISHEYE` | same names, same parameter order |
| `SE3(camera_to_world).Inverse()` | `Rigid3d.inverse()` on a `world_T_cam` |
| hardcoded `ERROR = 2.0` | `Point3D.error` is settable |
| `POINT3D_ID = -1` | `pycolmap.INVALID_POINT3D_ID` |
| rig / `synced_sample_id` | `pycolmap.Rig` + `pycolmap.Frame` (native in 4.2.0) |
| write text / binary | `Reconstruction.write_text` / `write_binary` |

**Watch the quaternion order**: pycolmap's `Rotation3d` takes `xyzw`, COLMAP text writes
`wxyz`, and cuSFM TUM writes `qx qy qz qw`. Three conventions in one pipeline.

### What has no pycolmap equivalent

1. **Byte-identical output.** `write_text` uses its own precision, sorts ids, and writes no
   trailing space; cuSFM uses `precision(15)`, a trailing space after every field, and
   unordered-map line order. The spec target is **semantic equivalence, not byte
   equivalence**.
2. **The FTheta -> PINHOLE fallback** and the truncate / zero-pad distortion policy. Pure
   application logic.
3. **The `w >= 0` quaternion canonicalisation.** pycolmap does not force it.
4. **Everything in §4 and §5.** TUM read/write, the SE3 interpolator, the `max_pose_gap` rule,
   the extrinsic composition and the whole `KeyframesMetadataCollection` round-trip. Use
   protobuf plus numpy/scipy (`Rotation.from_rotvec`, `Slerp`) directly.
5. **`GetMaxReprojectionError`** — meaningless here, always 2.0.

Note that cuSFM's own `sparse/` export **flattens the rig away**: it writes one independent
pose per image. Emitting a real `Rig`/`Frame` from Python is therefore an *improvement*, not a
fidelity requirement, and it will not reproduce the shipped file.

---

## 9. Two traps worth repeating

1. **The TUM microsecond round-trip is lossy** (§5.4). Odd-microsecond timestamps come back
   one microsecond low, silently turning exact hits into interpolations that can breach
   `--max_pose_gap_seconds` and drop frames. Carry microseconds as integers.
2. **`points3D.txt` carries no real photometry and no real error** (§3.7). Colour is
   `(0,0,0)` and `ERROR` is a hardcoded `2.0` for every point. Any consumer treating `ERROR`
   as a quality signal, or expecting coloured points, will be misled.

---

## 10. Open questions

1. ~~Whether `MapKeyPoints.points_r/g/b` is ever populated.~~ **Resolved.** The mapper never
   samples imagery for colour — it has no `Rgb8`, `CvtColor` or `imread` symbols, and
   `--raw_data_dir` serves only rolling shutter and the vehicle trajectory. Colour is meant to
   arrive in `KeypointVector.r/g/b` from the **feature extractor**; all 32 Galileo keyframes
   have zero `r/g/b`, so the map is black end to end. To get coloured points in the Python
   port, sample at each keypoint in the extractor stage. See
   `docs/spec/keypoints_mapper_main.md` §5.7.
2. `--output_pose_file_type=ego_motion` produces an `EgoMotionTrajectory` proto; its contents
   were not decoded.
3. The maximum-extrapolation-ratio bound in `LinearInterpolatorHelper` was seen in the
   decompilation but its value was not read out.
4. The shipped Galileo artifact is a **32-keyframe / 28-image** run, not the 224-image
   acceptance run in `docs/open-pipeline-plan.md`; all absolute counts here are small-run
   figures.

---

## 11. Evidence index

| Artifact | What it establishes |
|---|---|
| `data/cusfm_re/notes/export.md` | Decompilation notes, the byte-identical reproduction, the camera-model table, the hardcoded `ERROR`, the TUM formatting |
| `data/cusfm_re/decomp/export/` | Decompiled `colmap_manager.cc`, `pose_serializer.cc`, `transform_interpolator.cc` functions |
| `data/cusfm_re/analysis/export_verify.py`, `export_kpmap.py`, `export_poses.py` | The numeric verifications quoted throughout |
| `data/cusfm_re/runs/` | Verbose scratch re-runs of all three tools under varied flags |
| `data/cusfm_re/notes/rig_check.py`, `rig_check_output.txt` | Per-rig-frame BA deltas, rig exactness, constant extrinsics |
| `data/cusfm_runs/galileo/sparse/` | The shipped model and its `kpmap_to_colmap.txt` log |
| `data/cusfm_re/schema/data_protos.txt`, `extracted.fdset` | `MapKeyPoints`, `KeyframeMetaData`, `CameraSensor`, `RigidTransform3d` |
| `data/cusfm_re/notes/pycolmap_probe.txt` | pycolmap 4.2.0 camera models, `Rig`/`Frame`, writers |
| `docs/camera.md` | Camera models, projection maths, frame conventions |

# pycolmap capabilities for the open cuSFM pipeline

What pycolmap 4.2.0 gives the `colsfm` pipeline, what it does not, and what the
`colsfm` pixi environment had to do to get there. Every claim here is asserted by
a probe in `tests/colsfm_probes/`, runnable with:

```
pixi run -e colsfm colsfm-probe        # 60 probes, ~37 s
pixi run -e colsfm colsfm-lint
```

Probe letters (a)-(h) below match `docs/open-pipeline-plan.md`'s task list.

## 1. Build and versions

| Item | Value |
|---|---|
| pycolmap | `4.2.0`, conda-forge build `cuda_130hc321d7d_0` |
| COLMAP runtime | `colmap 4.2.0`, build `cuda_130hea678e2_0` |
| `pycolmap.COLMAP_build` | `Commit Unknown on Unknown with CUDA` |
| `pycolmap.has_cuda` | `True`, `get_num_cuda_devices() == 1` (RTX 5090) |
| pyceres | `2.6` (conda-forge `py312h54b85e1_3`), Ceres `2.2.0` |
| ceres-solver | `2.2.0`, conda-forge build `gpugplhf7c42c3_213` — the GPU/GPL build |
| Python | 3.12 |

Ceres' dense linear algebra libraries include `CUDA` alongside `EIGEN` and
`LAPACK`, so `BundleAdjustmentOptions.ceres.use_gpu` has something to reach.
COLMAP 4.2 also ships a second, non-Ceres GPU bundle adjuster: the `CASPAR`
backend (`BundleAdjustmentBackend.CASPAR`, `CasparBundleAdjustmentOptions`).

## 2. The pixi environment, and every gotcha the solve produced

The `colsfm` feature is `no-default-feature` with its own solve group, so the
default environment's CUDA 13 + TensorRT solve is untouched. Verified: the
`environments.default`, `environments.raco` and `environments.bench` blocks of
`pixi.lock` are byte-identical to `HEAD`, and no package was removed from the
lockfile — only additions.

### Gotcha A — `cuda-version = "13.0.*"` does **not** select a CUDA build

pycolmap 4.2.0 has `cpu_*`, `cuda_129_*` and `cuda_130_*` builds on conda-forge.
The `cpu_*` build declares no `cuda-version` constraint at all, so it satisfies a
CUDA-13 environment and the solver takes it. The first solve here silently
installed `pycolmap-4.2.0-cpu_h874a1db_0` and `pycolmap.has_cuda` was `False`.
The build string is what actually selects it:

```toml
pycolmap = { version = "4.2.0.*", build = "cuda_130*" }
```

`test_gpu.py::test_build_reports_cuda` is the regression guard.

### Gotcha B — pixi will not feed `__cuda` to the solver unless asked

With the build string pinned, the solve failed with
`pycolmap 4.2.0 would require __cuda >=13.0, for which no candidates were found`
even though `pixi info` reports `__cuda=13.0=0`. pixi only supplies the CUDA
virtual package when the environment declares it:

```toml
[feature.colsfm.system-requirements]
cuda = "13.0"
```

pixi 0.77.1 prints a deprecation warning for that table and points at
`platforms = [{ name = ..., platform = ..., cuda = ... }]`, but that form is
(a) rejected by 0.77.1's own parser (`expected a string, found table`) and
(b) workspace-scoped, so it would re-solve the default environment. The warning
is accepted deliberately.

### Gotcha C — a bare `pyserde` breaks the PyPI solve

`simplecv 0.7.2` requires `pyserde>=0.31.2,<0.32`. With `pyserde = "*"` the conda
solver pins 0.32.1 first and the PyPI solve then has no solution
(`Because simplecv==0.7.2 depends on pyserde>=0.31.2,<0.32 and pyserde==0.32.1,
we can conclude that simplecv==0.7.2 cannot be used`). Declared as
`pyserde = ">=0.31.2,<0.32"`. This is the same shape as `NOTES.md` gotcha 5,
which caps `typing-extensions` at `<4.16` for the same reason; both caps are
declared conda-side so one solver owns the constraint.

### Gotcha D — `colmap` needs no separate entry

pycolmap depends on `colmap >=4.2.0,<4.3.0a0`, so the C++ runtime arrives
transitively. Listing `colmap` explicitly only adds a second chance to pick a
mismatched build.

### Gotcha E — `pyyaml` is needed only for the interop probe

`tests/colsfm_probes/test_model_io.py` imports `demo_rerun.read_colmap_model`,
and `demo_rerun` imports `pycusfm`, which imports `yaml` at module scope. Adding
`pyyaml` to the feature is what makes that probe possible without touching
`demo_rerun.py`.

### Gotcha F — the beartype claw makes per-point pycolmap calls quadratic

Not a solve gotcha, but it dominated the suite: with `beartype_this_package()`
active, a loop of `Image.project_point(xyz)` calls carrying jaxtyping-annotated
locals leaks ~15 uncollectable objects per call. Repeated `build_scene()` calls
went 0.54 s, 1.60 s, 2.64 s, 4.05 s, ... and the whole suite took **367 s**.
Batching through `Camera.img_from_cam(points_in_cam)` (which has an `[m, 3] ->
[m, 2]` overload) removed it: `build_scene()` is now 0.004 s and the suite is
**37 s**. Rule for `colsfm/`: never call a pycolmap per-item accessor inside a
Python loop when a batched overload exists.

### Reproducible facts

```
$ pixi run -e colsfm python -c "import pycolmap, rerun; print(pycolmap.__version__, rerun.__version__)"
4.2.0 0.37.1
```

The TensorRT and Rerun halves of the environment were checked too, since the
plan needs both in the same process as pycolmap: `trt.Builder(trt.Logger())`
constructs (TensorRT 10.13.3.9.post1), `rerun.catalog` imports (so the
`catalog` extra's datafusion is present), and `simplecv`, `serde`, `tyro`,
`einops`, `pyarrow`, `av` and `google.protobuf` (7.35.1) all import cleanly.

## 3. Probe results

| # | Probe file | Tests | Result |
|---|---|---|---|
| a | `test_database.py` | 11 | pass |
| b | `test_two_view_geometry.py` | 8 | pass |
| c | `test_triangulation.py` | 5 | pass |
| d | `test_bundle_adjustment.py` | 8 | pass |
| e | `test_pose_graph.py` | 5 | pass |
| f | `test_camera_models.py` | 13 | pass |
| g | `test_model_io.py` | 7 | pass |
| h | `test_gpu.py` | 4 | pass |
| | **total** | **60** | **60 pass, 0 fail** |

## 4. (a) Database

### API renames — everything is `write_*`, not `add_*`

COLMAP 4.2 has **no** `Database.add_camera` / `add_image` / `add_keypoints` /
`add_descriptors` / `add_matches` / `add_two_view_geometry`. Code carried over
from pycolmap 0.x raises `AttributeError`. The current names:

```python
Database.open(path) -> Database                    # creates the file if absent
db.write_camera(camera, use_camera_id=False) -> int
db.write_rig(rig, use_rig_id=False) -> int
db.write_frame(frame, use_frame_id=False) -> int
db.write_image(image, use_image_id=False) -> int
db.write_keypoints(image_id, keypoints)            # float32 [m, n]
db.write_descriptors(image_id, descriptors)        # FeatureDescriptors, NOT ndarray
db.write_matches(image_id1, image_id2, matches)    # uint32 [m, 2]
db.write_two_view_geometry(image_id1, image_id2, two_view_geometry)
db.write_pose_prior(pose_prior, use_pose_prior_id=False) -> int
```

Readers mirror them (`read_camera`, `read_all_rigs`, `read_two_view_geometry`,
`read_all_matches`, ...), plus counters (`num_rigs`, `num_frames`,
`num_verified_image_pairs`, ...) and `clear_*` per table.

### Descriptors: uint8 storage, arbitrary width, ALIKED first-class

The database column is an opaque uint8 blob, so `write_descriptors` refuses a
bare ndarray (`TypeError`) and takes a `pycolmap.FeatureDescriptors`:

```python
FeatureDescriptors(data=UInt8[ndarray, "m k"], type=FeatureExtractorType)
FeatureDescriptorsFloat(data=Float32[ndarray, "m k"], type=FeatureExtractorType)
FeatureDescriptorsFloat.to_bytes() -> FeatureDescriptors    # reinterpret, not cast
FeatureDescriptors.to_float()      -> FeatureDescriptorsFloat
```

Both take **keyword arguments only**. So the answer to "must descriptors be
uint8 Nx128 (SIFT)?" is **no**: an Nx128 float32 ALIKED block is stored as an
Nx512 uint8 blob and recovered bit-exactly. Probed at 64, 128 and 256
dimensions. `FeatureExtractorType` already has `SIFT`, `ALIKED_N16ROT`,
`ALIKED_N32`, `LOMA_B`, `LOMA_B128` — ALIKED descriptors are a supported
first-class type, not a hack.

### Keypoints: float32, and the width is **not** validated

`write_keypoints` casts to float32 (so float64 sub-pixel precision is lost:
`123.456789012345` comes back as `123.45679`) and stores whatever column count
it is handed. COLMAP itself writes Nx2 (xy), Nx4 (xy + scale + orientation) or
Nx6 (xy + 2x2 affine), but Nx1, Nx3, Nx5 and Nx8 all round-trip unchanged. A
malformed width is a silent corruption downstream, not an error at write time.

### Rigs, frames and pose priors all have database tables

COLMAP 4.2 stores rigs and frames in the database:

- `Rig`: `rig_id`, `add_ref_sensor(sensor_t)`, `add_sensor(sensor_t, Rigid3d|None)`,
  `sensor_from_rig(sensor_t) -> Rigid3d|None`, `set_sensor_from_rig`,
  `is_ref_sensor`, `num_sensors()`, `non_ref_sensors`.
- `Frame`: `frame_id`, `rig_id`, `add_data_id(data_t)`, `data_ids`, `image_ids`,
  `rig_from_world`, `sensor_from_world(sensor_t)`, `set_cam_from_world`.
- Identifiers are `sensor_t(SensorType.CAMERA, camera_id)` and
  `data_t(sensor_t, image_id)`.

This is the direct home for cuSFM's `sensor_to_vehicle_transform` (`rig_T_cam`)
and its per-timestamp multi-camera grouping.

**Trap:** `Frame.has_pose` is a *method*, `Image.has_pose` is a *property*.
`assert not frame.has_pose` is always false. Database frames carry the grouping
only — the pose lives in the `Reconstruction`.

`PosePrior` fields: `corr_data_id` (`data_t`), `position`, `position_covariance`,
`gravity`, `coordinate_system` (`UNDEFINED` / `WGS84` / `CARTESIAN`),
`pose_prior_id`. **Position and gravity only — there is no rotation prior.**
`read_pose_prior(prior_id, is_deprecated_image_prior=True)`; pass `False` when
reading back an id returned by `write_pose_prior`.

## 5. (b) Two-view geometry

```python
estimate_two_view_geometry(camera1, points1, camera2, points2,
                           matches=None, options=TwoViewGeometryOptions())
    -> TwoViewGeometry
estimate_calibrated_two_view_geometry(...)   # same signature, forces the E path
estimate_two_view_geometry_pose(camera1, points1, camera2, points2, geometry) -> bool
verify_matches(database_path, pairs_path, options, cancellation_token=None) -> None
geometric_verification(database_path, verifier_options, pairing_options,
                       two_view_geometry_options, cancellation_token=None) -> None
```

`points1`/`points2` are float64 `[m, 2]` pixel coordinates, `matches` is uint32
`[m, 2]`. `verify_matches` takes a text file of `image_name1 image_name2` lines,
reads raw matches from the database and writes verified `TwoViewGeometry` rows
back; it returns nothing. Measured: 45 of 45 synthetic pairs verified.

### `TwoViewGeometryOptions` defaults

| Field | Default |
|---|---|
| `min_num_inliers` | 15 |
| `min_inlier_ratio` | 0.0 |
| `min_E_F_inlier_ratio` | 0.95 |
| `max_H_inlier_ratio` | 0.8 |
| `force_H_use` | False |
| `compute_relative_pose` | **False** |
| `multiple_models` | False |
| `use_degensac` | False |
| `use_sampson_refinement` | True |
| `detect_watermark` | True |
| `filter_stationary_matches` | False |
| `ransac.max_error` | 4.0 px |
| `ransac.min_inlier_ratio` | 0.25 |
| `ransac.confidence` | 0.999 |
| `ransac.min_num_trials` / `max_num_trials` | 100 / 10000 |
| `ransac.num_threads` | 1 |

Note that the nested `RANSACOptions` defaults inside `TwoViewGeometryOptions`
differ from a bare `RANSACOptions()` (`confidence 0.9999`, `min_inlier_ratio
0.01`, `min_num_trials 1000`, `max_num_trials 100000`).

### The trap: `Camera.has_prior_focal_length` selects the calibrated path

It defaults to **False** on a freshly created camera, and nothing warns.

| Camera | `has_prior_focal_length` | Result |
|---|---|---|
| PINHOLE | True | `CALIBRATED` (2), pose recovered, rotation < 0.5 deg, translation direction dot > 0.999 |
| PINHOLE | False | `UNCALIBRATED` (3) — a fundamental matrix, `cam2_from_cam1 is None` |
| OPENCV_FISHEYE | True | `CALIBRATED` (2), pose recovered |
| OPENCV_FISHEYE | False | **`DEGENERATE` (1), zero inliers** — F cannot model fisheye |

Cameras imported from `frames_meta.json` must have `has_prior_focal_length = True`
set explicitly, or fisheye pairs silently produce nothing.

Also note: when the inlier count falls below `min_num_inliers`, the result is
`DEGENERATE` (1), **not** `UNDEFINED` (0).

RANSAC threshold behaviour, with 25 % gross outliers injected into 195 matches:
`ransac.max_error = 1.0` keeps 147 inliers, `max_error = 200.0` keeps 195.

## 6. (c) Triangulation with known poses

```python
triangulate_points(reconstruction, database_path, image_path, output_path,
                   clear_points=True,
                   options=IncrementalPipelineOptions(),
                   refine_intrinsics=False,
                   cancellation_token=None) -> Reconstruction
```

**It is pose-preserving.** COLMAP runs a global bundle adjustment at the end of
this call (its log says `Retriangulation and Global bundle adjustment`), but with
every `rig_from_world` held constant. Measured: a frame pushed 6 cm off its true
pose comes back **byte-identical** (max change 0.0 over all frames), while the
model's mean reprojection error rises to 0.68 px — so the call really did run and
was not a no-op. This is exactly the contract cuSFM needs: the supplied
trajectory is a prior, not an initialisation.

On the clean synthetic scene: 5 frames x 2 cameras, 72 points, all 72
triangulated, mean reprojection error 8e-6 px, worst point < 1e-3 m from ground
truth. `image_path` must contain readable images (colour extraction runs); flat
grey PNGs are enough.

`triangulate_points` writes **five** files, not three: `cameras.bin`,
`images.bin`, `points3D.bin`, `rigs.bin`, `frames.bin`.

### `IncrementalTriangulator` — triangulation with nothing else

```python
graph = pycolmap.CorrespondenceGraph()
graph.add_image(image_id, num_points2D)
graph.add_two_view_geometry(image_id1, image_id2, two_view_geometry)
graph.finalize()
observations = pycolmap.ObservationManager(reconstruction, graph)
triangulator = pycolmap.IncrementalTriangulator(graph, reconstruction, observations)
triangulator.triangulate_image(options, image_id) -> int
triangulator.complete_all_tracks(options) -> int
triangulator.merge_all_tracks(options) -> int
triangulator.retriangulate(options) -> int
```

Both the graph and the reconstruction must outlive the triangulator. No database
and no image files are needed. Poses are provably untouched (matrices compared
with `np.array_equal`).

### `IncrementalTriangulatorOptions` — the full field list with defaults

| Field | Default | Meaning |
|---|---|---|
| `min_angle` | 1.5 | Minimum pairwise triangulation angle, degrees |
| `max_transitivity` | 1 | Transitivity when searching correspondences |
| `create_max_angle_error` | 2.0 | Max angular error to create a triangulation |
| `continue_max_angle_error` | 2.0 | Max angular error to continue one |
| `merge_max_reproj_error` | 4.0 | Max reprojection error to merge tracks |
| `complete_max_reproj_error` | 4.0 | Max reprojection error to complete a track |
| `complete_max_transitivity` | 5 | Transitivity for track completion |
| `re_max_angle_error` | 5.0 | Max angular error to re-triangulate |
| `re_min_ratio` | 0.2 | Under-reconstruction ratio that triggers re-triangulation |
| `re_max_trials` | 1 | Re-triangulation attempts per pair |
| `min_focal_length_ratio` | 0.1 | Degenerate-intrinsics filter |
| `max_focal_length_ratio` | 10.0 | Degenerate-intrinsics filter |
| `max_extra_param` | 1.0 | Degenerate-intrinsics filter |
| `ignore_two_view_tracks` | True | Drop tracks seen in only two views |
| `random_seed` | -1 | PRNG seed |

The field is `re_max_angle_error`, **not** `re_max_angle`.

### Standalone triangulation

```python
estimate_triangulation(points, cams_from_world, cameras,
                       options=EstimateTriangulationOptions()) -> dict | None
```

LO-RANSAC over N views; returns `{"xyz": ..., ...}` or `None`. Defaults:
`min_tri_angle = 0.0`, `residual_type = ANGULAR_ERROR`,
`ransac.max_error = 0.0349` (radians, because the residual is angular),
`ransac.min_inlier_ratio = 0.02`. Also available without RANSAC:
`triangulate_point`, `triangulate_mid_point`, `triangulate_multi_view_point`.

## 7. (d) Bundle adjustment

```python
bundle_adjustment(reconstruction, options=BundleAdjustmentOptions(),
                  cancellation_token=None) -> None
create_default_bundle_adjuster(options, config, reconstruction) -> BundleAdjuster
create_default_ceres_bundle_adjuster(options, config, reconstruction) -> CeresBundleAdjuster
create_pose_prior_bundle_adjuster(options, prior_options, config,
                                  pose_priors, reconstruction) -> BundleAdjuster
create_pose_prior_ceres_bundle_adjuster(...) -> CeresBundleAdjuster
BundleAdjuster.solve() -> BundleAdjustmentSummary
```

`BundleAdjustmentSummary` exposes `termination_type`
(`CONVERGENCE` / `NO_CONVERGENCE` / `FAILURE` / `USER_SUCCESS` / `USER_FAILURE`),
`num_residuals`, `is_solution_usable`, `brief_report()`.
`CeresBundleAdjuster` additionally exposes `.problem`, but that object cannot
cross into the standalone `pyceres` (see section 8).

### What can be held constant

Per-group flags on `BundleAdjustmentOptions`:

| Field | Default |
|---|---|
| `refine_focal_length` | True |
| `refine_principal_point` | **False** |
| `refine_extra_params` | True |
| `refine_rig_from_world` | True |
| `refine_sensor_from_rig` | True |
| `refine_points3D` | True |
| `constant_rig_from_world_rotation` | False (only when `refine_rig_from_world`) |
| `min_track_length` | 0 |
| `print_summary` | True |
| `backend` | `BundleAdjustmentBackend.CERES` |

Per-entity freezing on `BundleAdjustmentConfig`:

```python
config.add_image(image_id)
config.set_constant_rig_from_world_pose(frame_id)      # per FRAME, not per image
config.set_constant_sensor_from_rig_pose(sensor_t)     # rig extrinsics, per sensor
config.set_constant_cam_intrinsics(camera_id)
config.add_constant_point(point3D_id) / add_variable_point(point3D_id)
config.fix_gauge(BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD | THREE_POINTS)
```

`BundleAdjustmentGauge` has `UNSPECIFIED`, `TWO_CAMS_FROM_WORLD`, `THREE_POINTS`.

### Measured behaviour

| Scenario | Result |
|---|---|
| All poses + intrinsics + rig extrinsics constant, points perturbed by sigma 5 cm | mean reprojection error **5.6528 -> 0.000000 px**; every pose matrix and every intrinsics vector byte-identical; every point back within 1e-4 m |
| One frame perturbed (rotation ~0.03 rad, translation 8/5/6 cm), points constant | frame error **0.1181 -> 7.8e-13** |
| Rig extrinsic perturbed 3 cm, `refine_sensor_from_rig = True` | recovered to **1.8e-10 m** |
| Cauchy loss, one point moved 5.2 m, poses constant | worst inlier point error **9.0e-15 m** |
| Position priors on every frame, one frame shifted 20 cm | residual **1.6e-12 m** |

### Robust loss and solver options

`options.ceres` is a `CeresBundleAdjustmentOptions`:

| Field | Default |
|---|---|
| `loss_function_type` | `LossFunctionType.TRIVIAL` |
| `loss_function_scale` | 1.0 |
| `use_gpu` | False |
| `gpu_index` | `"-1"` |
| `auto_select_solver_type` | True |
| `min_num_images_gpu_solver` | 50 |
| `min_num_residuals_for_cpu_multi_threading` | 50000 |
| `max_num_images_direct_dense_cpu_solver` | 50 |
| `max_num_images_direct_sparse_cpu_solver` | 1000 |
| `max_num_images_direct_dense_gpu_solver` | 200 |
| `max_num_images_direct_sparse_gpu_solver` | 4000 |

`LossFunctionType` has exactly four members: `TRIVIAL`, `SOFT_L1`, `CAUCHY`,
`HUBER`. No Tukey, no Geman-McClure.

`options.ceres.solver_options` is a live reference — in-place mutation sticks.
Defaults that matter: `max_num_iterations = 100`, `num_threads = -1`,
`linear_solver_type = SPARSE_NORMAL_CHOLESKY`, `preconditioner_type = JACOBI`,
`function_tolerance = 0.0`, `gradient_tolerance = 1e-4`,
`parameter_tolerance = 0.0`, `logging_type = SILENT`.
Setting `linear_solver_type` explicitly requires `auto_select_solver_type = False`.

GPU: `options.ceres.use_gpu = True` is accepted and the solve still succeeds, but
Ceres only picks a GPU solver past `min_num_images_gpu_solver` (50 images), so a
small problem stays on the CPU. Not exercised end-to-end at scale here.

### Pose priors in BA

`PosePriorBundleAdjustmentOptions`:

| Field | Default |
|---|---|
| `prior_position_fallback_stddev` | 1.0 |
| `ceres.prior_position_loss_function_type` | `TRIVIAL` |
| `ceres.prior_position_loss_scale` | 2.7954834829151074 (chi2 3-DoF 95 % = 7.815) |
| `alignment_ransac` | `RANSACOptions(max_error=0.0, min_inlier_ratio=0.1, confidence=0.99, max_num_trials=2^31-1)` |

**Priors are 3-DoF position only** (plus an optional gravity direction). There is
no 6-DoF absolute-pose prior and no relative-pose prior in the BA path. cuSFM's
`use_relative_pose_constraint` / `relative_pose_translation_error_meters` has no
direct pycolmap equivalent — it has to be built with pyceres, or approximated
with position priors and `prior_position_fallback_stddev`.

Covariance is available separately: `estimate_ba_covariance`,
`estimate_ba_covariance_from_problem`, `BACovarianceOptions`.

## 8. (e) Pose graph optimisation — the backend decision

### pycolmap has no pose-graph optimiser

`pycolmap.PoseGraph` is a **container** for the incremental mapper's view graph:
`add_edge`, `get_edge`, `delete_edge`, `load(correspondence_graph)`,
`largest_connected_frame_component`, `set_valid_edge`. There is no `optimize`,
`solve` or `run`. `PoseGraphEdge` carries only `cam2_from_cam1`, `num_matches`
and `valid` — **no information matrix**, so an edge cannot even be weighted.

The related global-SfM entry points (`run_rotation_averaging`,
`run_global_positioning`, `GlobalPipeline`) solve a different problem: they
estimate poses from scratch, they do not refine a supplied trajectory against
loop constraints.

### `pycolmap.cost_functions` exists and is unusable

`pycolmap.cost_functions` advertises exactly the right residuals:
`RelativePosePriorCost`, `AbsolutePosePriorCost`,
`AbsolutePosePositionPriorCost`, `ReprojErrorCost`, `RigReprojErrorCost`,
`SampsonErrorCost`, `Point3DAlignmentCost`, each with a covariance overload.
**Every one of them raises on construction:**

```
TypeError: Unable to convert function return value to a Python type!
           The signature was (i_from_j_prior: pycolmap._core.Rigid3d) -> ceres::CostFunction
  caused by: TypeError: Unregistered type :
           ceres::AutoDiffCostFunction<colmap::RelativePosePriorCostFunctor, 6, 7, 7>
```

The cause: conda-forge's pycolmap embeds a **cut-down** pyceres at
`pycolmap._core.pyceres` (mirrored as `pycolmap.pyceres`, deliberately excluded
from `pycolmap.__all__`) that binds only `SolverOptions`, `SolverSummary` and the
enums. It has no `Problem`, no `CostFunction`, no `Manifold`, so
`ceres::CostFunction` is registered in neither pybind11 registry. Import order
does not help — the standalone conda `pyceres` extension has its own registry.

Scalar enums *do* cross the boundary (pybind11 converts them through their
integer value), so `options.ceres.solver_options.linear_solver_type` accepts
either module's `LinearSolverType`. Only the opaque C++ object types are
incompatible.

**Consequence:** the pose-graph residual has to be written in Python. So does any
custom BA residual that needs `pycolmap.cost_functions`.

### Measured: pyceres vs scipy, 1000 nodes

Synthetic loop: 1000 poses on a circle, 999 odometry edges plus one loop closure,
per-edge noise sigma 0.002 rad and 0.01 m, initial guess from integrating the
noisy odometry (0.855 m mean node error, 0.518 m loop seam), node 0 held fixed.
Identical residual in both cases; single-threaded; RTX 5090 host, CPU only.

| Backend | Wall | Work | Final cost | Loop seam | Converged? |
|---|---|---|---|---|---|
| **pyceres 2.6** | **24.6 s** | 29 residual + 29 Jacobian evaluations | 9.563e+00 -> **4.704e-06** | 0.518 -> **0.070 m** | **yes** (`CONVERGENCE`) |
| scipy `least_squares` (`trf`, `jac_sparsity`) | 10.3 s | nfev 40, njev 31 (capped) | 3.642e-03 | 0.518 -> 0.059 m | **no** (`status == 0`, `max_nfev` hit) |

Left to run to `xtol = ftol = gtol = 1e-12`, the scipy solve did not finish in
**over 40 minutes** and was killed.

The decisive number is the pyceres breakdown:

```
jacobian evaluation 22.92 s    (Python-side numeric differencing)
residual evaluation  1.55 s
linear solver        0.04 s
```

**99.8 % of pyceres' time is the Python residual, and the sparse Cholesky solve
for 1000 SE(3) nodes costs 40 ms.** The backend has ~600x of headroom the moment
the residual gets analytic Jacobians (or is vectorised into a batched cost
function); scipy has no equivalent headroom, because its cost *is* the
finite-difference sweeps of a residual that is already fully vectorised.

### Decision: **pyceres**

- It converges; scipy does not, inside any budget worth spending.
- The solver itself is effectively free (40 ms), so the whole optimisation
  budget is under our control in the residual.
- `pyceres.EigenQuaternionManifold` handles the SO(3) parameterisation
  correctly, and `pyceres.Problem.set_parameter_block_constant` fixes the gauge.
- `pycolmap.Rigid3d`'s memory layout is directly usable: `rotation.quat` is a
  writable float64 xyzw view at a stable address, `translation` a writable
  float64 3-vector. A `Rigid3d` can be handed to `pyceres.Problem` without a
  round trip through numpy. `Sim3d` is there too (`params` is 8-vector:
  rotation xyzw, translation, scale) for similarity problems.

### pyceres API notes

- Subclass `pyceres.CostFunction`, call `set_num_residuals(n)` and
  `set_parameter_block_sizes([...])` in `__init__`.
- **The override must be named `Evaluate` (capital E)**, even though the bound
  read-only method on the class is `evaluate`. A lowercase override is never
  found and Ceres aborts with
  `RuntimeError: Tried to call pure virtual function "Ceres::CostFunction::Evaluate"`.
- `problem.add_residual_block(cost, loss, [ndarray, ...])`; the arrays must be
  contiguous float64 and are mutated in place.
- Losses: `TrivialLoss`, `HuberLoss`, `CauchyLoss`, `SoftLOneLoss`.
- `pyceres.SolverSummary` has **no** `iterations` list; use
  `num_residual_evaluations`, `num_jacobian_evaluations`,
  `*_time_in_seconds`, `BriefReport()`.
- `pyceres.factors` contains only `NormalPrior` and `NormalError`.

## 9. (f) Camera models

`Camera.params_info` was checked against `docs/camera.md`'s cuSFM -> COLMAP
mapping table. **The table is correct.**

| cuSFM model | COLMAP model | id | `params_info` |
|---|---|---|---|
| `PINHOLE` | `PINHOLE` | 1 | `fx, fy, cx, cy` |
| `DISTORTED_PINHOLE` | `FULL_OPENCV` | 6 | `fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6` |
| `OPENCV_FISHEYE` | `OPENCV_FISHEYE` | 5 | `fx, fy, cx, cy, k1, k2, k3, k4` |
| `FTHETA_WINDSHIELD` | — | — | no equivalent; nothing in `CameraModelId` matches `*THETA*` |

cuSFM's `distortion_coefficients` array is OpenCV rational order
`[k1, k2, p1, p2, k3, k4, k5, k6]`, which is exactly COLMAP's `params[4:12]` — a
straight copy, no permutation. RoboCap's `Fisheye62` populates `k1..k4` only, and
`OPENCV_FISHEYE` has exactly four extra parameters, so that mapping is lossless.

Parameter groups (what the `refine_*` flags act on):

| Model | `focal_length_idxs` | `principal_point_idxs` | `extra_params_idxs` |
|---|---|---|---|
| `PINHOLE` | `[0, 1]` | `[2, 3]` | `[]` |
| `FULL_OPENCV` | `[0, 1]` | `[2, 3]` | `[4..11]` |
| `OPENCV_FISHEYE` | `[0, 1]` | `[2, 3]` | `[4..7]` |

Model ids are the classic COLMAP values, verified 0-10:
`SIMPLE_PINHOLE 0, PINHOLE 1, SIMPLE_RADIAL 2, RADIAL 3, OPENCV 4,
OPENCV_FISHEYE 5, FULL_OPENCV 6, FOV 7, SIMPLE_RADIAL_FISHEYE 8,
RADIAL_FISHEYE 9, THIN_PRISM_FISHEYE 10`.

`img_from_cam` and `cam_ray_from_img` invert each other to 1e-6 for all three
models with non-zero distortion, which is the independent check that the
parameter ordering is right.

COLMAP 4.2 adds models COLMAP 3.x lacked — `FISHEYE`, `SIMPLE_FISHEYE`, `EUCM`,
`DIVISION`, `SIMPLE_DIVISION`, `RAD_TAN_THIN_PRISM_FISHEYE`, `EQUIRECTANGULAR` —
worth revisiting if a better fisheye fit than `OPENCV_FISHEYE` is ever wanted.

Constructors: `Camera.create_from_model_id(camera_id, CameraModelId, focal, w, h)`
and `Camera.create_from_model_name(camera_id, "PINHOLE", focal, w, h)`.
`Camera.create` is deprecated. `verify_params()` checks the parameter count.

## 10. (g) Model I/O

```python
Reconstruction.write_text(path)      # cameras/images/points3D/rigs/frames .txt
Reconstruction.write_binary(path)    # ... .bin
Reconstruction.write(path)           # alias for write_binary
Reconstruction.read(path)            # sniffs whichever pair is present
Reconstruction.read_text(path) / read_binary(path)
Reconstruction.export_PLY(path)
```

Both formats round-trip cameras, rigs, frames, images and points, with poses
equal to 1e-9. **COLMAP 4.2 writes five files, not three** — anything that
globs `sparse/*.txt` or asserts a three-file set needs updating.

`images.txt` is unchanged from COLMAP 3.x:
`IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`, then a `POINTS2D[]` line of
`(X, Y, POINT3D_ID)` triples. The quaternion is `cam_from_world` in **wxyz**
order (pycolmap's `Rotation3d.quat` is xyzw, hence the `np.roll`).
`points3D.txt` is `POINT3D_ID X Y Z R G B ERROR TRACK[]`.

### Interop with `demo_rerun.read_colmap_model`

A fully observed pycolmap model parses **correctly**: all 10 images and 72 points
recovered, poses matching to 1e-6.

**But there is a real failure mode.** COLMAP writes two lines per image; when an
image has no 2D points the `POINTS2D` line is *empty*. `read_colmap_model`
filters empty lines before taking every second line, so from that point on the
pose/points alternation shifts and `POINTS2D` lines are parsed as poses. It does
not raise. On a three-image model where image 2 has no observations, the parser
returned:

```
['234.28571428571428 211.42857142857142 4 189.23076923076923 247.69230769230768 5',
 'img_1.png', 'img_2.png']
```

— `img_3.png`'s real pose is gone and a nonsense entry with a 201 m translation
took its place. `pycolmap.Reconstruction.read_text` reads the same file
correctly (3 registered images).

**Action for `colsfm`:** either guarantee every written image has at least one
observation, or read models with `pycolmap.Reconstruction.read` instead of the
hand-rolled parser. `demo_rerun.py` is not modified here (out of scope); this is
recorded so the exporter can be written not to trip it. `NOTES.md` decision 7
("COLMAP model parsed by hand") was taken to avoid pulling pycolmap into the
default environment — that reason no longer applies inside `colsfm`.

## 11. (h) GPU

- `pycolmap.has_cuda == True`, `get_num_cuda_devices() == 1`,
  `COLMAP_build` contains `with CUDA`. The `cpu_*` conda build reports
  `has_cuda == False`, which is why the build string is pinned (gotcha A).
- **`use_gpu` moved.** In COLMAP 4.2 it lives on `FeatureExtractionOptions` /
  `FeatureMatchingOptions` (with `gpu_index`, `num_threads`, `max_image_size`,
  `type`), not on `SiftExtractionOptions` / `SiftMatchingOptions`, which now hold
  only per-extractor tuning. COLMAP 3.x code raises `AttributeError`.
  `pycolmap.Device` members are lowercase: `auto`, `cpu`, `cuda`.
- Two GPU bundle adjustment paths: Ceres' CUDA linear algebra
  (`options.ceres.use_gpu`, gated by `min_num_images_gpu_solver = 50`) and the
  `CASPAR` backend (`options.caspar`, `gpu_index`, `solver_iter_max = 200`).
- `FeatureMatchingOptions` is rig-aware: `rig_verification`,
  `skip_image_pairs_in_same_frame`.

### COLMAP 4.2 already ships ALIKED + LightGlue

Discovered while probing, and directly relevant to stages 1 and 6 of the plan:

- `FeatureExtractorType`: `SIFT`, `ALIKED_N16ROT`, `ALIKED_N32`, `LOMA_B`,
  `LOMA_B128`.
- `FeatureMatcherType`: `SIFT_BRUTEFORCE`, `SIFT_LIGHTGLUE`,
  `ALIKED_BRUTEFORCE`, `ALIKED_LIGHTGLUE`, `LOMA_*`.
- `AlikedExtractionOptions`: `max_num_features = 2048`, `min_score = 0.2`, and
  model paths given as `url;filename;sha256` download specs pointing at COLMAP
  release assets, e.g.
  `https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n32.onnx;aliked-n32.onnx;a077728a...`.
  Nothing ships in the conda package, so COLMAP's own ALIKED needs network access
  on first use.
- `LightGlueONNXMatchingOptions`: `model_path` defaults to `""` (the caller must
  supply the ONNX file), `min_score = 0.1`.

This does **not** change the plan — the pipeline uses the TensorRT engines from
`pycusfm/models`, which are the same models the blob uses, so accuracy stays
comparable. It does mean COLMAP's ONNX path is available as a cross-check.

## 12. Summary of what is missing

1. **No pose-graph optimiser, and the cost functions that would build one are
   unreachable.** `PoseGraph` is a container without an `optimize`, and every
   `pycolmap.cost_functions` factory raises `TypeError: Unregistered type`
   against the standalone `pyceres`. The PGO is written in Python on
   `pyceres.Problem` (section 8).
2. **No relative-pose or 6-DoF absolute-pose prior in bundle adjustment.**
   `PosePrior` is position (+ gravity) only, and `PoseGraphEdge` has no
   information matrix. cuSFM's `use_relative_pose_constraint` with its
   `relative_pose_translation_error_meters` sigma has no direct equivalent.
3. **`Camera.has_prior_focal_length` defaults to False and silently downgrades
   two-view geometry** — to a fundamental matrix for PINHOLE, and to
   `DEGENERATE` with zero inliers for OPENCV_FISHEYE. Nothing warns.

Runner-up limitations worth knowing: keypoint column widths are not validated on
write; keypoints are stored as float32; `Frame.has_pose` is a method while
`Image.has_pose` is a property; `compute_mean_reprojection_error()` silently
returns 0.0 until `update_point_3d_errors()` has been called.

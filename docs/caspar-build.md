# CASPAR: COLMAP's GPU bundle adjustment, built from source

CASPAR is COLMAP's experimental GPU bundle-adjustment backend. COLMAP's FAQ
("Speedup bundle adjustment") describes it as one to two orders of magnitude
faster than the Ceres CUDA solver on medium-to-large problems, and says it is
available only when the library is compiled with `-DCASPAR_ENABLED=ON`.

No distributed binary sets that flag. The conda-forge `pycolmap 4.2.0
cuda_130*` package that the `colsfm` environment installs *does* expose
`pycolmap.BundleAdjustmentBackend.CASPAR` and `BundleAdjustmentOptions.caspar`,
because the enum and the option struct are bound unconditionally. Selecting the
backend and calling `solve()` then aborts:

    Caspar BA backend selected but COLMAP was built without CASPAR_ENABLED;
    rebuild with -DCASPAR_ENABLED=ON

So **the enum is not a capability probe**. The only reliable test is to run a
solve. This document covers the from-source build that makes that solve work,
what it measured, and what colsfm has to respect to use it.

## What was built

`packages/pycolmap-caspar/` is a local pixi-build package: a `pixi.toml` that
names the `pixi-build-rattler-build` backend plus a `recipe.yaml` that
rattler-build compiles on demand. It is consumed by exactly one environment,
`colsfm-caspar`, through a path dependency.

| | |
|---|---|
| Source | `github.com/colmap/colmap`, tag `4.2.0` = `be5e29168d4aff238409d60424812df66aac919f` |
| Package | `pycolmap 4.2.0` (the conda package name is `pycolmap`, so it shadows the conda-forge one inside `colsfm-caspar` only) |
| CUDA | 13.0 (`cuda-nvcc 13.0.*` pinned build-side so nvcc and the CCCL headers match) |
| GPU arch | `CMAKE_CUDA_ARCHITECTURES=120` (sm_120, RTX 5090 / Blackwell) |
| Flags | `-DCUDA_ENABLED=ON -DCASPAR_ENABLED=ON -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF -DFETCH_ONNX=OFF` |
| Precision | float32 — `CASPAR_USE_DOUBLE` left at its default `OFF` |

Two things about the recipe are load-bearing:

1. **The build is two steps.** `pycolmap`'s `pyproject.toml` sets
   `cmake.source-dir = "python/"`, so the bindings link a *separately built*
   libcolmap — and `CASPAR_ENABLED` lives in libcolmap, not in the bindings. The
   recipe therefore configures, builds and installs libcolmap into `$PREFIX`
   first, then runs `pip install . --no-build-isolation` with
   `CMAKE_PREFIX_PATH=$PREFIX` so the bindings find that install.
2. **`-DFETCH_ONNX=OFF` plus a `CPATH` shim.** COLMAP's default
   `FETCH_ONNX=ON` downloads a prebuilt *CUDA 12* onnxruntime whose CUDA
   provider needs `libcublasLt.so.12`, which does not exist in a CUDA 13
   environment; ALIKED then falls back to CPU or aborts the worker thread. With
   the flag off, COLMAP `find_package`s conda's `onnxruntime-cpp * *_cuda130`
   instead. conda nests that header at
   `include/onnxruntime/core/session/onnxruntime_cxx_api.h` while COLMAP
   includes it flat, hence the `export CPATH=...` line at the top of the build
   script.

### The sm_120 risk did not materialise

CASPAR's kernels are Symforce-generated, and their `CMakeLists.txt` hardcodes
`75 80 86 89 75-virtual` — but only inside `if(NOT DEFINED
CMAKE_CUDA_ARCHITECTURES)`
(`src/thirdparty/Symforce-Caspar/generated/f{32,64}/CMakeLists.txt:10-11`).
Passing `-DCMAKE_CUDA_ARCHITECTURES=120` explicitly is honoured, and the kernels
compile and launch on the 5090. **No patch is needed.** COLMAP 4.2.0 also
hard-errors when `CASPAR_ENABLED` is on and `CMAKE_CUDA_ARCHITECTURES` is
`native` / `all` / `all-major` (`cmake/FindDependencies.cmake:311-339`), which is
why the recipe spells the value out rather than leaving it to CMake.

### Dependency list

Derived from the conda-forge `colmap` / `pycolmap` 4.2.0 `cuda_130` packages
actually installed in this repo's `colsfm` environment, minus the Qt/xorg GUI
side. It deliberately drops the MKL / faiss-cuda / `fmt 12.1.*` pins that an
earlier version of this recipe carried in the `examples-monorepo` worktree:
those existed only to co-solve with a CUDA-13 PyTorch, which `colsfm-caspar` has
no reason to install. One `fmt` entry survives, unpinned, because
`OpenImageIOConfig.cmake` does `find_dependency(fmt)` and the very first
`find_package(OpenImageIO)` fails without it. Relative to 4.1.0, COLMAP 4.2.0
moves to eigen 5 (`eigen-abi >=5.0.1.80`) and adds PoseLib.

One correction the build log surfaced: COLMAP 4.2.0 vendors **PoseLib, faiss and
Caspar itself via `FetchContent`** rather than `find_package`, so the `poselib`
and `libfaiss` entries in the recipe are inert — those libraries end up statically
linked from the fetched sources. They are harmless (conda-forge's own `pycolmap`
does declare them) and were left in place rather than triggering a full rebuild,
but a future revision of the recipe can drop them.

## The environment

`[feature.colsfm-caspar]` plus the `colsfm-caspar` environment
(`no-default-feature`, own solve group) is a copy of `[feature.colsfm]` with one
line changed:

```toml
# colsfm
pycolmap = { version = "4.2.0.*", build = "cuda_130*" }
# colsfm-caspar
pycolmap = { path = "packages/pycolmap-caspar" }
```

The dependency list is duplicated rather than composed. Composing `colsfm` into
`colsfm-caspar` would put two `pycolmap` specs in one solve, and would drag the
frozen `colsfm` lock into the new solve — which is exactly what must not happen.
`colsfm`, `default`, `raco` and `bench` are untouched. Keep the two lists in
sync by hand.

`[workspace] preview = ["pixi-build"]` had to be added; path dependencies on a
pixi-build package are gated behind it.

## What the probe measured

`tools/caspar_probe.py` (run it with `pixi run -e colsfm-caspar caspar-probe`)
compares the two backends on the same problem, from the same starting state,
under options both accept. It reports the solve time, the final mean
reprojection error, and how far the two answers ended up apart.

The first CASPAR solve in a process pays 0.2-0.5 s for CUDA context creation
and kernel loading. The probe burns that on a throwaway warm-up so the reported
numbers are steady-state.

### Build

`pixi install -e colsfm-caspar` on a 32-core machine, from a cold cache:
**5 min 16 s** end to end (libcolmap ~4.5 min at 662 ninja targets, the pybind11
bindings ~40 s). The resulting artifact is a 96 MB
`pycolmap-4.2.0-hb0f4dca_0.conda` under `.pixi/artifacts-v0/`. The forecast of
20-60 minutes was pessimistic — but only after one fix: scikit-build-core drives
the bindings through Unix Makefiles with a **single** job by default, which cost
~25 minutes on its own until the recipe started exporting
`CMAKE_BUILD_PARALLEL_LEVEL`.

`pixi.lock` grew by **1913 lines** (4 removed, all of them a re-serialisation of
one `cuda-cudart` entry). The `environments:` block of the lock is purely
additive: `colsfm`, `default`, `raco` and `bench` have not one changed line.

### Synthetic rig — 5 frames, 2 PINHOLE cameras, 72 points

Points perturbed by sigma = 0.05 m from their exact positions; first frame held as
the gauge; intrinsics fixed.

| backend | solve | final reproj | max \|point − truth\| |
|---|---:|---:|---:|
| ceres | 0.002 s | 0.000000 px | 3.2e-09 m |
| caspar | 0.16 s | 0.000109 px | 6.7e-06 m |

**CASPAR is ~60x slower here, and that is the expected answer.** The problem is
far too small to amortise a GPU kernel launch, let alone a PCG iteration on the
device. Both backends recover the ground truth; CASPAR's float32 arithmetic
lands 1e-4 px from Ceres' double, and the two solutions agree to 8 um in
translation and 8e-5 degrees in rotation.

### Galileo — 226 images, 29 rig frames, 6194 points, PINHOLE

`data/cusfm_runs/galileo_colsfm/cusfm/sparse`, read with
`Reconstruction.read_text`. One global BA from the converged state: every frame
free except frame 1, points free, intrinsics fixed, no extrinsic refinement.
Starting reprojection error 1.332403 px. Four repeats:

| backend | solve (4 runs) | final reproj |
|---|---:|---:|
| ceres | 0.119 / 0.222 / 0.151 / 0.245 s | 1.362713 px |
| caspar | 0.040 / 0.100 / 0.134 / 0.109 s | 1.36235-1.36238 px |

**CASPAR is roughly 1.1-3.0x faster, median about 2x.** Agreement between the
two solutions: **max pose change 0.72 mm (0.037% of the 1.96 m trajectory
extent) and 0.0099 degrees**; points agree to a median of 0.17 mm, worst case
37 mm on a single point. Final reprojection error differs by 4e-4 px, in
CASPAR's favour.

Two caveats on those numbers, both important:

1. **Galileo is small.** COLMAP's FAQ claims one to two orders of magnitude on
   medium-to-large problems; 6194 points is neither. An earlier measurement in
   the `examples-monorepo` worktree, on a 300-image / 24k-point kitchen rig,
   saw 2.3-13.6x over Ceres-CPU with identical accuracy — the win grows with the
   problem. On Galileo, both solves finish in well under a quarter second, so
   the absolute saving is ~0.1 s per global BA.
2. **The Ceres baseline here is 32-core CPU Ceres.** A first measurement taken
   while the build was still occupying all 32 cores reported 13.2x for CASPAR;
   that number is an artifact of a starved Ceres, not a property of the solver,
   and is not the one to quote. The table above was taken on an idle machine.

### Probe suite

`pixi run -e colsfm-caspar python -m pytest tests/colsfm_probes -q` →
**60 passed**, byte-identical to the same suite in the stock `colsfm`
environment (also 60 passed). Nothing regresses under the from-source build.


## Constraints that matter for colsfm

CASPAR is not a drop-in replacement for Ceres. These are the limits, all
verified against the 4.2.0 source:

- **Camera model must be `PINHOLE` or `SIMPLE_RADIAL`.** Observations from any
  other model are *silently skipped* with a `LOG(WARNING)` — the solve
  succeeds and quietly ignores part of the problem. The Galileo model is
  PINHOLE, and the synthetic probe rig is PINHOLE, so both are fine; a fisheye
  or OPENCV run is not.
- **`refine_sensor_from_rig=True` is a hard `LOG(FATAL_THROW)`** whenever a
  non-reference sensor appears in a multi-sensor frame — which is every frame of
  the Galileo rig. CASPAR still optimises the per-frame `rig_from_world` pose,
  the points and the intrinsics; it just holds the *inter-camera* extrinsics
  fixed. For a calibrated rig that is no loss. For a rig whose extrinsics are
  being derived, it is a real semantic change.
- **`refine_focal_length` must equal `refine_extra_params`**, or the solve
  throws. CASPAR merges focal length and distortion into one parameter block.
- **Robust losses are dropped.** CASPAR solves plain L2 regardless of the loss
  the options ask for. Prefer it for the *global* solve, where colsfm is already
  running a trivial loss, and keep Ceres for any pass that relies on SOFT_L1 to
  reject outliers.
- **float32.** `CASPAR_USE_DOUBLE` is off, so CASPAR's arithmetic is single
  precision where Ceres is double. The measured difference on the Galileo model
  is well below the noise floor of the reconstruction (see the table above), but
  it is a real difference and it will grow on longer chains.
- **`refine_points3D=false` appears to be ignored** — there is no corresponding
  term in CASPAR's `IsPointVariable`. If points must stay fixed, mark them
  constant on the `BundleAdjustmentConfig` instead of relying on the option.
- **`IsSolutionUsable()` is not trustworthy.** CASPAR can report
  `CONVERGED_DIAG_EXIT` as a success. `BundleAdjustmentOptions.Check()`
  validates only the Ceres parameters.

**The regularised pyceres pass is unaffected.** `colsfm/extrinsic_refinement.py`
builds its own problem in standalone pyceres rather than going through
`pycolmap.BundleAdjuster`, so none of the above reaches it — the backend selector
only ever touches COLMAP's own bundle adjuster.

That is a convenient fit rather than a coincidence. CASPAR's one real limitation
for colsfm is that it holds `sensor_from_rig` fixed; colsfm already refines the
extrinsics in a separate pyceres stage with the rig poses and points frozen. So
the natural composition is: CASPAR for the heavy global rig BA (extrinsics
fixed), then the existing pyceres extrinsic pass. No new Ceres polish stage is
needed.

## Recommended wiring for `colsfm/mapping.py`

Not implemented here — `mapping.py` belongs to another worker. The shape that
fits the constraints above:

```python
@dataclass(slots=True)
class MappingOptions:
    ...
    ba_backend: Literal["ceres", "caspar"] = "ceres"
    """Bundle-adjustment implementation for the global solve.

    `caspar` is COLMAP's GPU backend and needs a `CASPAR_ENABLED` build (the
    `colsfm-caspar` environment); it falls back to `ceres` with a warning when
    the reconstruction's cameras are not all PINHOLE or SIMPLE_RADIAL.
    """
```

At the point where the options object is built (this was the original design sketch;
the shipped implementation in `colsfm/mapping.py` replaces the constant with a cached
runtime probe, `caspar_supported_camera_models()`, that solves a tiny reconstruction per
candidate model and reads `num_residuals`, so a build with the fisheye adapter is detected
automatically — see docs/caspar-fisheye-adapter.md §9):

```python
CASPAR_CAMERA_MODELS: frozenset[pycolmap.CameraModelId] = frozenset(
    {pycolmap.CameraModelId.PINHOLE, pycolmap.CameraModelId.SIMPLE_RADIAL}
)

backend = options.ba_backend
if backend == "caspar":
    models = {reconstruction.camera(i).model for i in reconstruction.cameras}
    unsupported = models - CASPAR_CAMERA_MODELS
    if unsupported:
        # Not an error: CASPAR would *silently drop* those observations.
        logger.warning(
            "CASPAR supports only PINHOLE and SIMPLE_RADIAL; %s present. Using Ceres.",
            sorted(m.name for m in unsupported),
        )
        backend = "ceres"

ba_options.backend = (
    pycolmap.BundleAdjustmentBackend.CASPAR
    if backend == "caspar"
    else pycolmap.BundleAdjustmentBackend.CERES
)
if backend == "caspar":
    ba_options.refine_sensor_from_rig = False          # otherwise FATAL_THROW
    ba_options.refine_extra_params = ba_options.refine_focal_length  # merged block
    ba_options.caspar.gpu_index = "0"
```

Three further points for whoever wires it:

1. **Do not gate on `hasattr(pycolmap.BundleAdjustmentBackend, "CASPAR")`** — it
   is always true. If a runtime capability check is wanted, catch the exception
   from a tiny solve once and cache the answer.
2. **Only the global solve should use CASPAR.** The incremental bootstrap's
   local BA relies on a robust loss, which CASPAR drops.
3. **If extrinsics need refining**, run CASPAR first with
   `refine_sensor_from_rig=False`, then a short Ceres pass with it True. CASPAR
   for the many iterations, Ceres for the finish.

## Reproducing

```bash
pixi install -e colsfm-caspar          # builds the package; see the timing above
pixi run -e colsfm-caspar caspar-probe # CASPAR vs Ceres, synthetic + Galileo
pixi run -e colsfm-caspar caspar-test  # the probe suite against the from-source build
```

## § Double precision

`CASPAR_USE_DOUBLE` is the second experimental flag COLMAP 4.2.0 exposes next to
`CASPAR_ENABLED`:

```cmake
# CMakeLists.txt:78
option(CASPAR_USE_DOUBLE "Use double precision in Caspar solver" OFF)
```

It switches two things, and only these two:

1. **The scalar type of the solver.** With the flag on, CMake adds the
   `CASPAR_USE_DOUBLE` compile definition (`CMakeLists.txt:88-92`), and
   `src/colmap/estimators/bundle_adjustment_caspar.h:42-48` typedefs
   `StorageType` to `double` instead of `float`. Every factor, residual and
   Jacobian in the CASPAR problem changes width.
2. **Which pre-generated kernel tree is compiled.**
   `src/thirdparty/CMakeLists.txt:100-106` selects
   `src/thirdparty/Symforce-Caspar/generated/f64` instead of `.../f32`. Both
   trees ship inside the 4.2.0 tag (4.5 MB and 4.3 MB), so nothing has to be
   regenerated — `generate_caspar.py` is not needed.

Upstream treats the two precisions as materially different accuracies. Its own
CASPAR test hardcodes one tolerance per precision
(`src/colmap/estimators/bundle_adjustment_caspar_test.cc:457-465`): focal length
**1.0 px under fp64 against 20.0 px under fp32**, principal point 1.0 against
10.0, extra params 1e-4 against 1.5e-2. That asymmetry is what made float32 the
prime suspect for the KITTI 06 accuracy loss.

### The build

`packages/pycolmap-caspar64/` is a copy of `packages/pycolmap-caspar/` with
`-DCASPAR_USE_DOUBLE=ON` added to **both** cmake invocations — the libcolmap
configure and the bindings' `CMAKE_ARGS` — plus an explicit build string
`caspar_f64_cuda130_py312`, so the fp64 artifact can never collide with the fp32
`pycolmap 4.2.0` in pixi's artifact cache. `[feature.colsfm-caspar64]` and the
`colsfm-caspar64` environment mirror the fp32 pair with one line changed:

```toml
# colsfm-caspar
pycolmap = { path = "packages/pycolmap-caspar" }
# colsfm-caspar64
pycolmap = { path = "packages/pycolmap-caspar64" }
```

`pixi install -e colsfm-caspar64` on the same 32-core machine: **5 min 30 s**,
against 5 min 16 s for fp32. Double precision costs essentially nothing at
compile time — the same 662 ninja targets. The configure log confirms the switch
landed:

```
-- Caspar precision:  f64
-- Caspar source dir: $SRC_DIR/src/thirdparty/Symforce-Caspar/generated/f64
```

and the packaged headers under `include/colmap/thirdparty/Symforce-Caspar/generated/`
are the `f64` set only. `pixi.lock` grew by **763 lines with zero deletions**;
the only new key in the `environments:` block is `colsfm-caspar64`, so `colsfm`,
`colsfm-caspar`, `default`, `raco`, `bench` and `decode-bench` are untouched.

### The probe: fp64 CASPAR reproduces Ceres to machine precision

`pixi run -e colsfm-caspar64 caspar-probe`.

Synthetic rig — 5 frames, 2 PINHOLE cameras, 72 points, perturbed by sigma = 0.05 m:

| backend | solve | final reproj | max \|point − truth\| | agreement with Ceres |
|---|---:|---:|---:|---|
| ceres | 0.002 s | 0.000000 px | 3.2e-09 m | — |
| caspar fp32 | 0.16 s | 0.000109 px | 6.7e-06 m | 8 um / 8e-5 deg |
| **caspar fp64** | 0.22 s | **0.000000 px** | **5.3e-15 m** | **0.000000 m / 0.000000 deg** |

Galileo — 226 images, 29 rig frames, 6194 points, one global BA from the
converged state, starting reprojection 1.332403 px:

| backend | solve | final reproj | termination | max pose change vs Ceres |
|---|---:|---:|---|---:|
| ceres | 0.113 s | 1.362713 px | CONVERGENCE | — |
| caspar fp32 | 0.04-0.13 s | 1.36235-1.36238 px | — | 0.72 mm / 0.0099 deg |
| **caspar fp64** | 1.012 s | **1.362712 px** | NO_CONVERGENCE | **0.003 mm / 0.000044 deg** |

**Numerically the flag does exactly what it promises.** fp64 CASPAR lands 5e-15 m
from ground truth on the synthetic rig — tighter than Ceres' own 3e-09 — and
agrees with Ceres to 3 micrometres on Galileo against fp32's 0.72 mm, a 240x
improvement. Its final reprojection error matches Ceres to the sixth decimal.

**The cost on a GeForce card is exactly the 1/64 FP64 penalty you would expect.**
On Galileo, fp64 CASPAR takes 1.012 s where fp32 took about 0.1 s and Ceres takes
0.113 s: fp64 is roughly 10x slower than fp32 CASPAR and **9x slower than CPU
Ceres**. On this problem size the GPU backend stops being a win at all.

### KITTI 06 — the accuracy does not come back

The §8 `cap 500 + loops` configuration, from the cuVSLAM SLAM initialisation,
with the backend as the only change:

```bash
pixi run -e colsfm-caspar64 python -m colsfm run \
    --input-dir data/kitti/06_colsfm_input_slam \
    --config-dir data/kitti/config \
    --output-dir data/kitti/06_colsfm_caspar64/cusfm \
    --min-inter-frame-distance 0.5 --loop-closure \
    --match-cap-mode fixed --ba-backend caspar
```

`summary.json` records `mapping.ba_backend = "caspar"`, so no fallback fired.
All three ATE rows come from one `tools/kitti/evaluate_kitti.py` invocation
against `data/kitti/06/poses_gt_06.txt` (`/tmp/caspar64_ate.json`):

| Metric | ceres (CAUCHY) | caspar fp32 | **caspar fp64** |
|---|---:|---:|---:|
| mapping stage (s) | 148.5 | **31.5** | 46.1 † |
| total (s) | 338.0 | 208.6 | 249.5 † |
| registered images | 2156 | 2156 | 2156 |
| 3D points | 96 691 | 97 224 | 96 644 |
| observations | 713 124 | 709 950 | 712 232 |
| mean reprojection (px) | 0.674 | 0.715 | 0.714 |
| Sim(3) ATE RMSE (m) | **0.895** | 1.299 | 1.285 |
| SE(3) ATE RMSE (m) | **1.118** | 1.601 | 1.555 |

† **The two fp64 runtimes are contended and are upper bounds.** Another worker
held the GPU for the whole run — 31 samples at 10 s intervals, mean utilisation
64 %, every one of them with a competing process. The four non-BA stages, whose
cost is independent of the BA backend, inflated by a mean factor of **1.31**
against the fp32 run (feature extraction 1.39x, matching 1.17x, loop closure
1.07x, export 1.61x). Dividing the mapping stage by that same factor puts the
uncontended fp64 estimate near **35 s**, close to fp32's 31.5 s. Treat 35-46 s as
the range and do not quote 46.1 s as a clean measurement.

**Double precision does not recover the accuracy.** Sim(3) ATE moves from
1.299 m to 1.285 m — 1.1 %, against the 0.404 m gap to Ceres' 0.895 m. SE(3)
moves 1.601 to 1.555, 2.9 %. Mean reprojection error is unchanged at 0.714 px
against Ceres' 0.674 px. **The float32 hypothesis is falsified.**

That is a genuinely informative negative, because the probe proves the arithmetic
is now exact: fp64 CASPAR and Ceres agree to 3 micrometres on a single global BA,
yet the full pipeline still lands 0.39 m apart. So the remaining gap is not
rounding, and — per the existing `ceres, loss TRIVIAL` ablation at 0.899 m — not
the dropped robust loss either. Both candidate explanations are now eliminated by
measurement. The evidence points instead at **convergence behaviour**: CASPAR
terminates `NO_CONVERGENCE` even on the small Galileo solve, and colsfm runs it
across five triangulate / filter / bundle-adjust rounds, so an under-converged
solve compounds. The persistent 0.04 px reprojection deficit in both CASPAR
columns, identical across precisions, is the signature of a solver stopping
short rather than one computing the wrong answer. That is where the next
investigation should look — CASPAR's PCG tolerance and iteration budget, not its
number format.

### Verdict

**No. fp64 CASPAR is not both faster than Ceres and as accurate.** It is faster
than Ceres on KITTI — roughly 3.2x on the mapping stage as measured, 4.2x if the
contention correction holds — but it keeps essentially all of fp32's 0.4 m
accuracy loss while giving back a third of fp32's speed advantage. It is the
worst of the three on the speed/accuracy trade-off: it costs 1.5x fp32's mapping
time and buys 1.1 % of ATE. On smaller problems it is worse still, losing to CPU
Ceres outright (9x slower on Galileo), because a GeForce card runs FP64 at 1/64
rate.

**Recommendation for pinhole datasets: colsfm should keep defaulting to `colsfm`
with Ceres, and `colsfm-caspar64` should not become anyone's default.** The
choice is between the two ends, not the middle. Where trajectory accuracy is the
deliverable, use Ceres — 0.895 m against 1.285-1.299 m is not a rounding
difference on a 1231 m sequence. Where mapping throughput is the deliverable and
0.4 m of ATE is acceptable, use `colsfm-caspar` (fp32), which is 4.7x on the
stage. `colsfm-caspar64` earns its keep only as a **diagnostic**: it is the
instrument that proved precision was not the problem, and it is the environment
to reach for when a future CASPAR question needs the arithmetic held exact while
something else varies. Keep the package and the environment for that purpose;
do not put a pipeline on it.

## § Convergence sweep

The fp64 experiment above left one hypothesis standing: CASPAR stops short. This
section tests it by moving CASPAR's stopping rules from the command line —
`--caspar-option name=value`, threaded through `PipelineOptions.caspar_option` to
`MappingOptions.caspar_options` and applied to `BundleAdjustmentOptions.caspar`
in `colsfm.mapping.bundle_adjustment_options`. The flag repeats:

```bash
python -m colsfm run ... --ba-backend caspar \
    --caspar-option solver_iter_max=1000 pcg_iter_max=80
```

### What CASPAR can stop on at all

`pycolmap.BundleAdjustmentOptions().caspar.todict()` on the fp32 build:

| knob | default |
|---|---:|
| `solver_iter_max` | 200 |
| `pcg_iter_max` | 20 |
| `pcg_rel_error_exit` | 1e-4 |
| `pcg_rel_score_exit` | -1.0 (disabled) |
| `pcg_rel_decrease_min` | -1.0 (disabled) |
| `solver_rel_decrease_min` | 1.0 |
| `score_exit_value` | 0.0 |
| `diag_init` / `diag_min` | 1.0 / 1e-12 |
| `diag_scaling_up` / `diag_scaling_down` | 2.0 / 0.333333 |
| `diag_exit_value` | 1e3 |
| `gpu_index` | "-1" |

The header is more informative than the defaults.
`thirdparty/Symforce-Caspar/generated/f32/solver.h` declares **three** exit
reasons and no others:

```cpp
enum class ExitReason { MAX_ITERATIONS, CONVERGED_SCORE_THRESHOLD, CONVERGED_DIAG_EXIT };
```

So CASPAR has **no gradient, parameter or function tolerance** — nothing
corresponding to the 1e-6 / 1e-10 / 1e-8 triple `colsfm.mapping` hands Ceres.
With `score_exit_value = 0.0` the score test can never fire on a non-zero
residual, which leaves exactly two ways out: the damping `diag` climbing past
1e3 (reported as convergence — the LM step has stopped helping), or the
iteration cap (reported as `NO_CONVERGENCE`). "Converged" from CASPAR therefore
means "damping ran away", not "the gradient is small".

### Galileo, from the converged state

`data/cusfm_runs/galileo_colsfm/cusfm/sparse`, 29 rig frames, 6194 points, one
global BA with one gauge frame, intrinsics fixed, TRIVIAL loss on both sides and
colsfm's own Ceres settings (SPARSE_SCHUR, 200 iterations, 1e-6 / 1e-10 / 1e-8).
Starting reprojection 1.332403 px. Pose deltas are against the Ceres row.

| setting | termination | reproj px | vs ceres | solve s | max abs dt mm | rms dt mm | max dR deg |
|---|---|---:|---:|---:|---:|---:|---:|
| ceres | CONVERGENCE | 1.3627 | — | 0.06 | — | — | — |
| caspar default | CONVERGENCE | 1.3623 | -0.0003 | 0.26 | 0.688 | 0.548 | 0.0095 |
| `solver_iter_max=1000` | CONVERGENCE | 1.3624 | -0.0003 | 0.23 | 0.648 | 0.517 | 0.0091 |
| `pcg_iter_max=80` | CONVERGENCE | 1.3624 | -0.0003 | 0.80 | 0.650 | 0.538 | 0.0087 |
| `pcg_rel_error_exit=1e-5` | CONVERGENCE | 1.3623 | -0.0004 | 0.35 | 0.724 | 0.575 | 0.0099 |
| `pcg_rel_error_exit=1e-6` | CONVERGENCE | 1.3624 | -0.0003 | 0.68 | 0.610 | 0.489 | 0.0086 |
| `score_exit_value=1e-6` | CONVERGENCE | 1.3624 | -0.0003 | 0.53 | 0.612 | 0.492 | 0.0087 |
| `diag_exit_value=1e9` | CONVERGENCE | 1.3624 | -0.0003 | 0.51 | 0.661 | 0.535 | 0.0091 |
| `solver_iter_max=1000 pcg_iter_max=80` | CONVERGENCE | 1.3626 | -0.0001 | 1.57 | **0.288** | **0.241** | 0.0045 |
| ... `+ pcg_rel_error_exit=1e-6` | CONVERGENCE | 1.3624 | -0.0003 | 0.77 | 0.666 | 0.547 | 0.0089 |
| `pcg_rel_decrease_min=1e-3` | CONVERGENCE | 1.3324 | -0.0303 | 0.03 | 1.139 | 0.692 | 0.0234 |
| `pcg_rel_score_exit=1e-3` | CONVERGENCE | 1.3324 | -0.0303 | 0.04 | 1.139 | 0.692 | 0.0234 |
| `solver_rel_decrease_min=1e-2` | CONVERGENCE | 1.3324 | -0.0303 | 0.05 | 1.139 | 0.692 | 0.0234 |
| `solver_rel_decrease_min=1e-4` | CONVERGENCE | 1.3324 | -0.0303 | 0.04 | 1.139 | 0.692 | 0.0234 |

Three corrections to the paragraph above this section:

1. **fp32 CASPAR does not report `NO_CONVERGENCE` on Galileo.** Every fp32 run
   in the table terminates `CONVERGENCE`. The `NO_CONVERGENCE` row in the fp64
   probe table is an fp64 observation, and it does not generalise.
2. **There is no 0.04 px deficit on Galileo either.** fp32 CASPAR lands 0.0003 px
   *below* Ceres. The 0.04 px figure (0.715 against 0.674) is a KITTI-only,
   five-round, whole-pipeline number, not a per-solve one.
3. **Four of the knobs are traps.** Enabling `pcg_rel_decrease_min` or
   `pcg_rel_score_exit`, or moving `solver_rel_decrease_min` off its 1.0 default,
   returns the input model **unchanged** (1.3324 px is the starting value) in
   0.03-0.05 s. Those three are early-exit ratios that fire on the first
   iteration; do not put them in a sweep expecting more work, they buy none.

The converged model is a weak discriminator: one solve moves the mean
reprojection by 0.03 px, so every legitimate setting looks alike, and the widest
disagreement with Ceres is 0.7 mm on a multi-metre scene.

### Galileo, from a state far from the optimum

KITTI's mapper hands CASPAR a freshly re-triangulated and filtered model, not a
converged one. The same sweep, with poses perturbed by sigma = 0.01 m / 0.3 deg
and points by sigma = 0.02 m (starting reprojection 82.5 px):

| setting | termination | reproj px | solve s | max abs dt mm vs ceres | rms dt mm | max dR deg |
|---|---|---:|---:|---:|---:|---:|
| ceres | CONVERGENCE | (diverged points) | 0.16 | — | — | — |
| caspar default | CONVERGENCE | 1.3640 | 10.3 | 10.2 | 5.13 | 0.206 |
| `solver_iter_max=1000` | CONVERGENCE | 1.3655 | 13.2 | 10.3 | 5.15 | 0.205 |
| `pcg_iter_max=80` | NO_CONVERGENCE | 1.3641 | 15.0 | 10.8 | 5.08 | 0.193 |
| `pcg_iter_max=200` | NO_CONVERGENCE | 1.3643 | 6.7 | 10.6 | 7.38 | 0.261 |
| `pcg_rel_error_exit=1e-6` | NO_CONVERGENCE | 1.3645 | 0.8 | 10.0 | 5.22 | 0.210 |
| `pcg_rel_error_exit=1e-8` | NO_CONVERGENCE | 1.3644 | 6.0 | 10.1 | 5.21 | 0.208 |
| `diag_exit_value=1e9` | NO_CONVERGENCE | 1.3647 | 1.0 | 9.9 | 5.23 | 0.213 |
| `solver_iter_max=1000 pcg_iter_max=80` | CONVERGENCE | 1.3651 | 7.4 | 10.8 | 5.11 | 0.190 |
| max effort (1000 / 200 / 1e-8) | CONVERGENCE | 1.3644 | 70.8 | 11.1 | 5.02 | 0.187 |

Ceres' own mean reprojection is not usable here — from 82 px it walks a handful
of weakly-constrained points into their own cameras, which is exactly what
`filter_degenerate_points` and `filter_projection_failures` exist for inside the
mapper and which this bare probe does not run. Its poses are still the reference.

Two things are visible even so. `NO_CONVERGENCE` **is** reachable on fp32, and
what reaches it is *tightening* the PCG side: the default's `CONVERGENCE` is the
damping blow-up, and a more accurate inner solve keeps LM alive until the
iteration cap. And every setting, from the default to a 70 s max-effort run,
lands within 1.3640-1.3655 px and 5.0-7.4 mm rms of Ceres. **A 9x change in
solver budget moves the answer by less than the spread between two of its own
settings.**

### KITTI 06, the question that was asked

The three settings the Galileo tables leave standing — more PCG iterations, more
LM iterations, and both plus a tighter inner tolerance — on the full sequence.
Same command as § Double precision with `--ba-backend caspar` plus
`--caspar-option`, scored by `tools/kitti/evaluate_kitti.py` against
`data/kitti/06/poses_gt_06.txt` (1078 matched poses):

```bash
pixi run -e colsfm-caspar python -m colsfm run \
    --input-dir data/kitti/06_colsfm_input_slam --output-dir <out> \
    --config-dir data/kitti/config --min-inter-frame-distance 0.5 \
    --loop-closure --match-cap-mode fixed --ba-backend caspar \
    --caspar-option solver_iter_max=1000 pcg_iter_max=80
```

| run | Sim(3) RMSE m | SE(3) RMSE m | final reproj px | mapping s |
|---|---:|---:|---:|---:|
| Ceres (Cauchy, the baseline) | **0.895** | **1.118** | 0.6741 | 148.5 |
| Ceres, TRIVIAL loss | 0.899 | 1.097 | 0.7078 | 319.7 |
| CASPAR default | 1.299 | 1.601 | 0.7147 | 31.5 |
| `pcg_iter_max=80` | 1.287 | 1.562 | 0.7217 | 75.6 † |
| `solver_iter_max=1000 pcg_iter_max=80` | 1.286 | 1.518 | 0.7276 | 120.3 † |
| `solver_iter_max=1000 pcg_iter_max=200 pcg_rel_error_exit=1e-8` | 1.279 | 1.501 | 0.7264 | 176.3 |

† `nvidia-smi` showed another worker holding 1.2-7.4 GB on the same 5090 for the
whole of these two runs, so their mapping seconds are upper bounds. The default,
the Ceres rows and the max-effort row ran on an idle GPU. ATE is unaffected by
contention either way.

**No setting closes the gap.** A 5.6x mapping-time increase buys 0.020 m of the
0.404 m Sim(3) deficit — 5 % — and the last three rows agree with each other to
0.008 m, which is the run-to-run spread of the mapper itself. SE(3) moves more
(0.100 m of 0.483 m, 21 %) but lands nowhere near Ceres. Spending more on the
solver also makes the *reprojection* slightly worse, not better (0.7147 →
0.7276), which is the first sign that the two solvers are not racing along the
same path to the same point.

### Where the gap is actually made

The per-round line of each run (`[colsfm] round N gate … reprojection … px`)
shows the deficit is born in round 0 and then merely carried:

| round | gate px | Ceres Cauchy | Ceres TRIVIAL | CASPAR default | CASPAR max effort |
|---:|---:|---:|---:|---:|---:|
| 0 | 25 | 0.8626 | 0.9060 | 0.9251 | 0.9243 |
| 1 | 20 | 0.8357 | 0.9056 | 0.9197 | 0.9923 |
| 2 | 15 | 0.8014 | 0.8763 | 0.8884 | 0.8880 |
| 3 | 10 | 0.7536 | 0.8202 | 0.8299 | 0.8341 |
| 4 | 5 | 0.6741 | 0.7078 | 0.7147 | 0.7264 |

Two readings, and they kill the stopping-criterion hypothesis outright:

1. **The solver budget does not touch round 0.** 0.9251 px at the default and
   0.9243 px at 1000 LM iterations / 200 PCG iterations / 1e-8 — a 0.0008 px
   difference for roughly six times the work. CASPAR is *already at* its solution
   after the default budget; it is not stopping short.
2. **Compared like with like, CASPAR reproduces Ceres.** The right Ceres column is
   the TRIVIAL one, because CASPAR applies no robust loss. Against it, CASPAR is
   0.019 px behind at round 0 and **0.007 px** behind at the end. The "0.04 px
   deficit" of the earlier sections is almost entirely the Cauchy loss, which
   CASPAR cannot honour — and the Cauchy/TRIVIAL Ceres pair scores 0.895 against
   0.899 m, i.e. the loss is worth 0.004 m of ATE, not 0.4 m.

So CASPAR minimises the same objective to within 1 % of Ceres' own value and
still lands 0.4 m away on the ground.

### Verdict: not a stopping-criterion problem

**The 0.4 m gap on KITTI 06 is not a stopping-criterion problem.** Every knob
CASPAR exposes was moved, one at a time and in combination, on a converged
Galileo solve, on a Galileo solve started 82 px from its optimum, and on the full
KITTI sequence. Nothing recovered more than 5 % of the Sim(3) deficit, and the
setting that did (176 s of mapping against the default's 31.5 s) gave up
CASPAR's only advantage to get it. Two of the three signatures the hypothesis
rested on do not survive contact:

- `NO_CONVERGENCE` is not a symptom of stopping short. fp32 CASPAR reports
  `CONVERGENCE` on every Galileo solve from the converged state, and reaches
  `NO_CONVERGENCE` only when the PCG tolerance is *tightened* — the default's
  "convergence" is `CONVERGED_DIAG_EXIT`, the damping blow-up, since
  `score_exit_value = 0` makes the score test unreachable and CASPAR has no
  gradient, step or function tolerance at all.
- The 0.04 px reprojection deficit is the Cauchy loss, not lost accuracy. Held
  to the same TRIVIAL objective, CASPAR and Ceres finish 0.007 px apart, and
  Ceres scores the same ATE under either loss.

**The remaining hypothesis is gauge freedom — the solvers agree on the cost and
disagree on the point.** The perturbed-Galileo table is the direct evidence: from
the same start, every CASPAR setting lands within 1.3640-1.3655 px, a spread
smaller than its own run-to-run noise, yet 5.0-7.4 mm rms and up to 0.26° away
from Ceres' poses. The disagreement does not shrink with budget, so it is not
truncation; it is displacement along directions the reprojection objective barely
penalises. colsfm pins the gauge with a single
`set_constant_rig_from_world_pose` on the lowest frame id, which fixes six
degrees of freedom of a 1078-frame, 1.2 km chain and leaves scale and the
low-curvature bending modes to the data alone. On Galileo — 29 frames, a few
metres across — that costs 0.7 mm and is invisible. On KITTI it is the whole
0.4 m, and it is set in round 0, before the five-round schedule has any chance to
correct it.

**The one experiment that would test it.** Take the *final* CASPAR model
(`data/kitti/06_colsfm_caspar/cusfm/sparse`), run a single Ceres global BA on it
with colsfm's own settings and no re-triangulation and no filtering, re-export
the TUM poses and re-score. It costs one solve, and it separates the two
remaining explanations cleanly:

- ATE snaps back towards 0.9 m → both solvers see the same basin and Ceres simply
  sits at a different point in it. The gap is gauge/conditioning, and the fix is
  to constrain it — more constant frames, or a scale prior from the rig baseline,
  or Ceres' `use_inner_iterations` equivalent.
- ATE stays near 1.29 m → Ceres agrees this is a minimum, and the divergence was
  decided by which tracks round 0's 25 px gate and `filter_degenerate_points`
  kept, not by the solve. The fix is then in the mapper's schedule, not in
  CASPAR's options at all.

Either way, do not spend more on CASPAR's solver: the sweep above shows that
budget buys nothing.

## § Ceres polish experiment

The experiment the section above asked for, run: take the finished CASPAR fp32
KITTI 06 model, hand it to Ceres for **one** global bundle adjustment with
colsfm's own settings — `data/kitti/config`'s Cauchy loss at scale 4.0, 200
iterations, `SPARSE_SCHUR`, extrinsics and intrinsics fixed, and the single
constant rig frame `run_mapping` picks (the frame owning the lowest registered
image id, frame 1 here) — and then stop. No re-triangulation, no merge, no
complete, no filter: 2156 images, 97 224 points and 709 950 observations go in
and the same ones come out, so everything below is the solve's doing.

```bash
pixi run -e colsfm python -m tools.audit.caspar_polish --label caspar-polish
pixi run -e colsfm python -m tools.audit.caspar_polish --label ceres-polish \
    --sparse-dir /tmp/colsfm_runs/kitti/cap500_loops/cusfm/sparse \
    --output-dir data/kitti/06_colsfm_loops_polish/cusfm
pixi run -e colsfm python -m tools.kitti.evaluate_kitti --sequence-dir data/kitti/06 \
    --trajectories caspar <...>/06_colsfm_caspar/cusfm/output_poses/merged_pose_file.tum \
                   caspar-polished <...>/06_colsfm_caspar_polish/cusfm/output_poses/merged_pose_file.tum \
                   ceres /tmp/colsfm_runs/kitti/cap500_loops/cusfm/output_poses/merged_pose_file.tum \
                   ceres-polished <...>/06_colsfm_loops_polish/cusfm/output_poses/merged_pose_file.tum
```

`tools/audit/caspar_polish.py` re-exports the polished model the way the pipeline
does — `sparse/`, `kpmap/keyframes/frames_meta.json` off the source run's own
collection as the template, and the TUM trajectories — so the scoring is
`evaluate_kitti` exactly as §8 of `docs/kitti-06-results.md` runs it, against
`data/kitti/06/poses_gt_06.txt`, from the cuVSLAM-SLAM initialisation, 1078 of
1078 rig poses matched.

| Model | Sim(3) RMSE (m) | SE(3) RMSE (m) | mean reproj (px) | Ceres iters | initial cost | final cost | cost cut | solve (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CASPAR fp32, as the mapper left it | 1.299 | 1.601 | 0.7147 | — | — | — | — | — |
| **CASPAR + one Ceres BA (Cauchy)** | **0.890** | **1.117** | 0.6924 | 22 | 3.3279e5 | 3.2115e5 | 3.50 % | 18.2 |
| CASPAR + one Ceres BA (TRIVIAL, ablation) | 0.903 | 1.115 | 0.7050 | 18 | 3.9627e5 | 3.8720e5 | 2.29 % | 12.7 |
| Ceres cap 500 + loops (the baseline) | 0.895 | 1.118 | 0.6741 | — | — | — | — | — |
| Ceres + one Ceres BA (the control) | 0.895 | 1.118 | 0.6741 | 1 | 2.9909e5 | 2.9909e5 | 0.00 % | 2.0 |

What the polish moved, in the model's own frame:

| Model | rig displacement RMSE (m) | max (m) | residual after a Sim(3) fit, RMSE (m) | max (m) | fitted scale |
|---|---:|---:|---:|---:|---:|
| CASPAR + Ceres BA | 2.381 | 5.302 | 0.927 | 2.184 | 0.99810 |
| Ceres + Ceres BA (control) | 2.5e-14 | 3.3e-13 | 2.5e-14 | 3.3e-13 | 1.000000 |

**The control is exact.** Ceres' own model is a fixed point of Ceres: one
iteration, no cost change at all, and the rig moves 25 femtometres. The
instrument does not manufacture motion, so every number in the CASPAR row is
CASPAR's.

**ATE snaps back.** 1.299 m to **0.890 m** Sim(3) and 1.601 m to **1.117 m**
SE(3) — the Ceres baseline, 0.895 m and 1.118 m, to within 5 mm. One solve, 18 s,
recovers the whole 0.4 m gap. CASPAR and Ceres are in the same basin, the
correspondences CASPAR mapped with are the ones Ceres wants, and round 0's 25 px
gate did not decide anything: had the tracks parted, no amount of solving on
CASPAR's own observation set could have reached Ceres' score, and it reaches it
exactly.

**But it is not gauge freedom either.** The gauge hypothesis of the section above
predicted that Ceres would agree on the cost and only disagree on the point; it
does not agree on the cost. From CASPAR's answer Ceres finds another 3.50 % of
Cauchy cost (2.29 % of plain least squares, which is the objective CASPAR
actually minimises) and 0.022 px of reprojection error, and it does so in 22
iterations from a `CONVERGENCE` start. CASPAR's stopping point is not a
stationary point of its own objective. The displacement confirms the reading: of
the 2.381 m rms the rig moves, a Sim(3) fit absorbs the part that is pure gauge
and **0.927 m rms of shape change survives it** — and Sim(3)-aligned ATE only
sees that surviving part, which is why the score moves.

**The TRIVIAL ablation kills the last alternative.** Polishing under `TRIVIAL`
— the objective with no robust loss, the one CASPAR itself solves — lands at
0.903 m, statistically the Cauchy polish's 0.890 m. So the recovery is not the
Cauchy reweighting doing something CASPAR could not do; it is plain optimisation
that CASPAR left on the table.

**This re-reads the sweep's own evidence.** §"Where the gap is actually made"
measured CASPAR as 0.007 px behind Ceres on the TRIVIAL objective and concluded
the objectives agreed. The polish moves CASPAR by 0.0097 px on that same
objective and 0.409 m on the ground: at 1078 frames over 1.2 km, one hundredth of
a pixel of residual suboptimality *is* 0.4 m of trajectory. A cost gap of 2 % is
not a rounding difference on this problem, and "CASPAR minimises the same
objective to within 1 %" was never evidence that it lands in the same place.

### What to do with this

1. **Ship the polish.** `--ba-backend caspar` plus one Ceres global BA is 31.5 s
   of mapping + 18 s of polish against the Ceres mapper's 148.5 s, at 0.890 m
   against 0.895 m. That is the accuracy of Ceres at 1/3 of its bundle-adjustment
   time, and it needs no CASPAR change at all. Worth wiring into `run_mapping` as
   a final round on the Ceres backend when `ba_backend == "caspar"`.
2. **Do not spend the effort on more gauge constraints.** Two constant frames or
   a scale prior would remove the 2.4 m of gauge motion the polish also produces,
   but the ATE is measured after a Sim(3) fit, which already removes it; the
   0.927 m that the fit cannot absorb came from solving further, not from
   anchoring differently.
3. **If CASPAR itself is to be fixed, the target is now specific.** It is not the
   iteration budget (the sweep bought nothing with 176 s), it is that
   `CONVERGED_DIAG_EXIT` — the damping blow-up — fires while 2 % of the cost is
   still reachable, and CASPAR has no gradient, step or function tolerance to fire
   instead. The test is one line of instrumentation: log the gradient norm at
   CASPAR's exit on this model, and compare it against the norm at Ceres'
   converged point.

## § Shipped: CASPAR + Ceres polish

The polish of the section above is now the default behaviour of the mapper, not a
separate tool. When `MappingOptions.ba_backend` **resolves** to `caspar` — every
fallback ends on Ceres, which is already at its own fixed point —
`colsfm.mapping.run_mapping` runs the five outer rounds on CASPAR and then one
Ceres global bundle adjustment over the final model:
`ceres_polish_options(ba_config, options, reconstruction)` is the round options
with the backend swapped, so the config's Cauchy loss at scale 4, its 200
iterations, `SPARSE_SCHUR` and the frozen extrinsics and intrinsics all carry
over, and the gauge stays the single constant frame the rounds used. Nothing is
merged, completed or filtered around it. `MappingResult.polish` (a `PolishStats`)
records the iterations, the costs, the seconds and the reprojection error either
side; `summary.json` carries `mapping.polish_seconds` and
`mapping.polish_iterations`. The seconds stay inside the `reconstruction` row of
`runtime.csv`, because the polish *is* mapping. `--no-caspar-ceres-polish`
switches it off for an ablation.

Measured on the 5090, one run each, `pixi run -e colsfm-caspar`, one idle Rerun
viewer on the same GPU:

```bash
# Galileo 226, three backends into /tmp/colsfm_runs/polish/
pixi run -e colsfm-caspar python -m colsfm run --input-dir data/r2b_galileo \
    --output-dir /tmp/colsfm_runs/polish/galileo_caspar_polish/cusfm --ba-backend caspar
pixi run -e colsfm python -m colsfm.bench_cli --dataset galileo \
    --run-a data/cusfm_runs/galileo_blobref/cusfm \
    --run-b /tmp/colsfm_runs/polish/galileo_caspar_polish/cusfm \
    --input-dir data/r2b_galileo --name-a blob --name-b caspar-polish

# KITTI 06, cap 500 + loops, the § Convergence sweep flags
pixi run -e colsfm-caspar python -m colsfm run \
    --input-dir data/kitti/06_colsfm_input_slam --config-dir data/kitti/config \
    --output-dir /tmp/colsfm_runs/polish/kitti_caspar_polish/cusfm \
    --min-inter-frame-distance 0.5 --loop-closure --match-cap-mode fixed --ba-backend caspar
pixi run -e colsfm python -m tools.kitti.evaluate_kitti --sequence-dir data/kitti/06 \
    --trajectories ceres <...> caspar <...> caspar-polish <...>
```

**KITTI 06 — the whole gap comes back for 9.8 s.** ATE against
`data/kitti/06/poses_gt_06.txt`, 1078 of 1078 rig poses matched, from one
`evaluate_kitti` run. The `ceres` and `caspar` columns are the existing runs of
NOTES.md § CASPAR backend, re-scored here unchanged.

| Metric | ceres | caspar | **caspar + polish** |
|---|---:|---:|---:|
| mapping stage (s) | 148.5 | 31.5 | **41.3**, of which 9.8 is the polish |
| total (s) | 338.0 | 208.6 | 212.9 |
| registered images | 2156 | 2156 | 2156 |
| 3D points | 96 691 | 97 224 | 97 223 |
| observations | 713 124 | 709 950 | 710 169 |
| mean reprojection (px) | 0.674 | 0.715 | 0.692 |
| Sim(3) ATE RMSE (m) | **0.895** | 1.299 | **0.904** |
| SE(3) ATE RMSE (m) | 1.118 | 1.601 | 1.128 |
| polish iterations / cost cut | — | — | 22 / 3.39 % |

The polish reproduces `tools/audit/caspar_polish.py` exactly where it should: 22
Ceres iterations, 3.39 % of Cauchy cost against the offline tool's 3.50 %, and
0.6923 px against 0.6924 px. It lands 9 mm from the Ceres mapper on a 1231.7 m
sequence at **3.6x less mapping time** (41.3 s against 148.5 s), and the
observation set it hands Ceres is CASPAR's own — 710 169 observations in, 710 169
out. The 9.8 s here is half the audit's 18.2 s because the audit re-reads the
model off disk and re-derives the point errors first.

**Galileo 226 — nothing to fix, nothing broken.** Three runs, same environment,
same host; ATE is `colsfm.bench_cli` against the blob reference.

| Metric | ceres | caspar | **caspar + polish** |
|---|---:|---:|---:|
| mapping stage (s) | 1.87 | 2.10 | 2.58, of which 0.19 is the polish |
| total (s) | 16.71 | 16.78 | 17.93 |
| registered images | 225 / 226 | 225 / 226 | 225 / 226 |
| 3D points | 6190 | 6283 | 6298 |
| mean reprojection (px) | 1.334 | 1.423 | 1.390 |
| ATE vs ground truth (mm) | 4.32 | 4.56 | **4.30** |
| acceptance bounds | 4/4 | 4/4 | 4/4 |
| polish iterations | — | — | 20 |

29 rig frames and 6298 points is far too small a problem for the gap CASPAR opens
on KITTI to exist at all — CASPAR and Ceres agree to 0.28 mm here — so the polish
has almost nothing to do, does it in 0.19 s, and still ends one hundredth of a
millimetre ahead of the Ceres run. The cost is 1.2 s of a 17.9 s run.

**Verdict: on by default.** The polish never makes a model worse (Ceres is a
fixed point of Ceres, measured to 25 femtometres in § Ceres polish experiment),
it costs a fraction of what the CASPAR rounds save, and it is what turns
`--ba-backend caspar` from a debug-loop convenience into the default recommendation
for pinhole datasets.

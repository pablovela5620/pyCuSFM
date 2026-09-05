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

At the point where the options object is built:

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

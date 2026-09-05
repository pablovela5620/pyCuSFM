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

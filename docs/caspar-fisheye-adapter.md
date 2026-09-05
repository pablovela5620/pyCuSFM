# An `OPENCV_FISHEYE` adapter for COLMAP's CASPAR GPU bundle adjuster

Follow-on to `docs/caspar-fisheye-feasibility.md`, which estimated 1–2 days for
a working fork. **It took under three hours.** This document records what was
built, what it measures, and what an upstream submission would still need.

Everything here is local. There is no GitHub fork, no branch pushed anywhere,
and no published artifact.

---

## 1. What exists now

| Thing | Where |
|---|---|
| COLMAP fork | `/home/pablo/0Dev/forks/colmap-caspar-fisheye`, branch `caspar-opencv-fisheye`, commit `267dd0fd`, branched from tag `4.2.0` (`be5e29168d4aff238409d60424812df66aac919f`) |
| Symforce generator environment | `/home/pablo/0Dev/forks/colmap-caspar-fisheye/pixi.toml` (gitignored in the fork) — `symforce == 0.12.0` from PyPI, plus `clang-format` |
| conda package | `packages/pycolmap-caspar-fisheye/{recipe.yaml,pixi.toml}` — build string `caspar_fisheye_cuda130_py312` |
| pixi environment | `colsfm-caspar-fisheye` (own solve group, `no-default-feature`) |
| validation probe | `tools/caspar_fisheye_probe.py` |

### The fork diff

```
 doc/faq.rst                                        |     4 +-
 src/colmap/estimators/bundle_adjustment_caspar.cc  |     9 +
 src/colmap/estimators/caspar/caspar_model_adapter.h|   465 +-
 src/thirdparty/Symforce-Caspar/caspar_generate.py  |   170 +
 ...generated/{f32,f64}/*                           |  448 new files
 461 files changed, 124440 insertions(+), 17240 deletions(-)
```

Generated-tree growth, per precision: **488 → 712 files**, 86k → **137k lines**
(f32) / 146k lines (f64), 6.6 MB. Exactly 224 new files per tree — 112 `.cu`
plus 112 `.h` — which is the number the feasibility study predicted.

The hand-written work is the four non-generated files above: **~650 lines**, of
which ~460 are the mechanically-derived C++ adapter.

---

## 2. Generator reproduction: what the shipped tree actually is

Before changing anything, `caspar_generate.py` was run **unchanged** against
`symforce 0.12.0` and diffed against COLMAP's shipped `generated/f32`. It does
*not* reproduce byte-for-byte. The 488 files split three ways:

| Class | Files | Verdict |
|---|---:|---|
| Reproduce exactly **after applying COLMAP's own `.clang-format`** | 395 | ✅ identical |
| Reproduce exactly **raw, unformatted** (the 9 static sources Caspar copies from `symforce/caspar/source/`: `memops.cuh`, `shared_indices.*`, `solver_params.h`, `solver_tools.*`, `sort_indices.*`, `__init__.py`) | 9 | ✅ identical — COLMAP does **not** reformat these |
| Differ | 84 | ⚠️ see below |

**404 / 488 = 82.8 % byte-for-byte.** The 84 that differ are:

1. **82 factor kernels** (`kernel_*_res_jac.cu`, `*_res_jac_first.cu`,
   `*_score.cu`; every `*_jtjnjtr_direct.cu` matches). They differ only in
   register allocation and instruction scheduling — 0.12.0 uses one register
   *fewer* on `kernel_pinhole_score.cu` (45 vs 46). Same math, different
   schedule out of Caspar's adaptive-reordering pass.
2. **`solver.cc`**, by exactly one line. 0.12.0 emits
   `result.initial_score = score_best;`; **the shipped tree does not.**
   `bundle_adjustment_caspar.cc:1010` reads `caspar_summary.initial_score`,
   and `SolveResult::initial_score` has no in-class initialiser — so stock
   COLMAP 4.2.0 reports an *uninitialised* CASPAR initial score. Regenerating
   fixes that bug.
3. **`CMakeLists.txt`**: 0.12.0 emits `CASPAR_MIN_ARCH` and
   `75 80-real 86-real 89-real`; the shipped file has `75 80 86 89 75-virtual`.
   Immaterial here — the recipe passes `-DCMAKE_CUDA_ARCHITECTURES=120`
   explicitly, and the file's guard is `if(NOT DEFINED ...)`.

**Where the shipped tree came from.** Not from any public symforce.
`symforce 0.12.0` is the newest PyPI release, and `git diff v0.12.0..main --
symforce/caspar` in `symforce-org/symforce` is **empty** — main and 0.12.0 are
identical for Caspar. Codegen is deterministic (two consecutive runs produced
byte-identical trees). So COLMAP 4.2.0 vendored a Caspar snapshot that predates
the public 0.12.0 release, and 0.12.0 is strictly the newer of the two. This is
worth stating in any upstream PR, because it means *nobody* can currently
regenerate COLMAP's tree from a released dependency.

### Consequence for the merge

Adding the model regenerates only **4 pre-existing files** per tree
(`solver.{h,cc}`, `caspar_mappings.{h,cu}`); every pre-existing *kernel* file is
byte-identical whether or not `opencv_fisheye` is registered (verified by
diffing a with-fisheye run against a without-fisheye run). So the fork keeps
COLMAP's shipped kernels untouched and installs only the 224 new files plus the
4 aggregates, formatted with COLMAP's `.clang-format`.

**PINHOLE and SIMPLE_RADIAL therefore cannot regress: their kernels did not
change.** No mapping function was removed or renamed
(checked by name-set diff on `caspar_mappings.*`).

---

## 3. What was added

### 3.1 `caspar_generate.py` (+170)

Node classes `OpenCVFisheye{Pose,Calib,PrincipalPoint,FocalAndExtra}` and their
`Const` twins, plus `ConstOpenCVFisheyeSensorFromRig`. Merged calib is `V8`
`[fx, fy, k1, k2, k3, k4, cx, cy]`; split is `V6` focal-and-extra + `V2`
principal point. Identical arity to `OPENCV`, i.e. the layout PR #4611 already
ran on hardware — the shared-memory objection in issue #4371 targets 12-param
`THIN_PRISM_FISHEYE` and does not apply. Codegen emitted no shared-memory
warning.

The residual, transcribed from `OpenCVFisheyeCameraModel` in `sensor/models.h`:

```python
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = p.norm(sf.epsilon())          # sqrt(r^2 + eps) -- see below
    theta = sf.atan(r)
    theta2 = theta * theta
    radial = theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
    scale = theta * (1 + radial) / r
    return sf.V2([fx * scale * p[0] + cx, fy * scale * p[1] + cy]) - pixel
```

`models.h:429-438` guards the `theta/r` division with `if (r > epsilon)`. A
branch does not survive symbolic codegen, so the core uses
`Matrix.norm(sf.epsilon())` = `sqrt(r² + ε)`, which is finite everywhere and
tends to the same limit (`scale → 1`) as `r → 0`. ε is `1e-6` for f32 and
`1e-15` for f64, set at module load.

**Both trees were regenerated.** PR #4611 shipped f32 only, so a
`CASPAR_USE_DOUBLE=ON` build of that branch would not compile.

### 3.2 `caspar_model_adapter.h` (+465)

`OpenCVFisheyeAdapter` with `FocalAndExtraSize() == 6`,
`PrincipalPointSize() == 2`, `CalibSize() == 8`; the 15-variant
`SetVariantFactors` switch; 17 new `CasparSolverSizing` fields; the factory
case; and the `CreateSolver` positional-argument surgery.

The feasibility study flagged `CreateSolver` as the one genuinely hazardous
edit ("Argument order is opaque and bug-prone… silent mis-ordering yields wrong
sizing, not a compile error"). It was **not** inferred. Both orders were read
directly off the generated constructor signature in
`generated/f32/solver.h`:

- Node counts, alphabetical: `OpenCVFisheyeCalib`, `OpenCVFisheyeFocalAndExtra`,
  `OpenCVFisheyePose`, `OpenCVFisheyePrincipalPoint`, then `PinholeCalib`, …
  — so **every pre-existing node-count argument shifts by four positions**.
- Factor counts, registration order:
  `simple_radial (4) → pinhole (4) → opencv_fisheye (4) →
  simple_radial_split (11) → pinhole_split (11) → opencv_fisheye_split (11)`.

Note the two spellings Caspar generates and the trap they set: node accessors
use the *class* name (`SetOpenCVFisheyePoseNodesFromStackedHost`) while factor
accessors use the CamelCased *snake* model name
(`SetOpencvFisheyeNum`, lowercase `cv`).

### 3.3 `bundle_adjustment_caspar.cc` (+9) and `doc/faq.rst`

Pose-pool and calib sizing hookup; supported-model sentence.

---

## 4. Build

`pixi install -e colsfm-caspar-fisheye`, logged to
`/tmp/caspar-fisheye-build.log`:

| | |
|---|---|
| Wall time | **5 m 25 s** (75 min CPU across 32 threads) |
| Result | `pycolmap-4.2.0-caspar_fisheye_cuda130_py312.conda` |
| `libcaspar_lib_core.a` | 27.20 MiB |
| Extra CUDA TUs vs `pycolmap-caspar` | +112 |

Effectively the same 5-minute build as the stock CASPAR package, despite 45 %
more CUDA in the Caspar library.

`pixi.lock` was diffed before and after: the `bench`, `colsfm`,
`colsfm-caspar`, `colsfm-caspar64`, `decode-bench`, `default` and `raco`
environment blocks are **byte-identical**. Only the new
`colsfm-caspar-fisheye` block was added.

---

## 5. Validation

All from `pixi run -e colsfm-caspar-fisheye python -m tools.caspar_fisheye_probe`
on an RTX 5090 (sm_120), pycolmap 4.2.0, `has_cuda=True`.

RoboCap is not on disk (`data/cusfm_runs/` holds only Galileo and KITTI), so the
distortion coefficients are the hand-picked realistic set
`k = (0.02, -0.01, 0.005, -0.001)` rather than measured ones.

### 5.1 The hazard check — **PASS**

The inherited hazard: while validating PR #4611 its author found that the
generated OpenCV split-variant *score* kernels read an uninitialised per-thread
accumulator, which decoded as NaN and produced a **false convergence — a clean
3-iteration exit with the poses unchanged**. Every "did the solver succeed?"
assertion passes that. So the probe re-solves the *same* perturbed state with
`solver_iter_max` capped at 1, 2, … 60 and watches the score.

| cap | score | | cap | score |
|---:|---:|---|---:|---:|
| 0 (input) | 2.127967e+04 | | 10 | 9.839e-03 |
| 1 | 3.585110e+03 | | 20 | 9.411e-05 |
| 2 | 3.526279e+03 | | 30 | 9.330e-05 |
| 3 | 4.960600e+02 | | 40 | 8.567e-05 |
| 5 | 2.010126e+01 | | 60 | 9.250e-05 |

- iterations swept: **60**
- monotone non-increasing over the signal region: **True**
  (worst rise there: **−5.114e-02**, i.e. no rise at all)
- score fell below half the input: **True** (by 8 orders of magnitude)
- max pose translation change: **0.015813 m** — the output moved
- **VERDICT: PASS**

Monotonicity is asserted only while the score is above `1e-6 ×` the input,
which still spans the six orders of magnitude where the solve happens. Below
that the float32 residual floor (~1e-4) makes successive caps wobble by a few
times 1e-4 from rounding alone. The #4611 failure mode is not hidden by this:
it produces a *flat* curve pinned at the input value, which fails the
`decreased` test at any tolerance.

**One harness bug worth recording**, because it counterfeits the very bug being
hunted. Setting `caspar.solver_rel_decrease_min = -1.0` — the obvious reading of
"disable the early-exit criterion" — produced a textbook false convergence:
60 iterations, score unchanged at `2.127968e+04`, **zero pose change**. Caspar
accepts a step only when `score_new < score_best * solver_rel_decrease_min`
(`generated/f32/solver.cc:3449`), so a negative value makes the right-hand side
negative and rejects *every* step. The default `1.0` already means "accept any
decrease". Left at the default, the sweep is the table above.

### 5.2 Synthetic 4-camera `OPENCV_FISHEYE` rig — **PASS**

6 frames × 4 cameras, 800×800, f = 260 px (~114° diagonal), 147 points, exact
projections, then perturbed 1 cm / 0.3° on the poses and 2 cm on the points.
Intrinsics fixed, one gauge frame, both backends from bit-identical state.

| backend | time | final reproj | score | termination | max &#124;pose − truth&#124; | max &#124;point − truth&#124; |
|---|---:|---:|---:|---|---:|---:|
| ceres | 0.020 s | 0.000000 px | 6.92e-16 | CONVERGENCE | 9.34e-11 m | 1.02e-09 m |
| caspar (fisheye) | 0.063 s | **0.000143 px** | 8.77e-05 | CONVERGENCE | 3.25e-06 m | 3.70e-05 m |

Agreement between the two: max pose change **0.000003 m** (0.0002 % of the
1.75 m trajectory extent) and **0.000027°**; point agreement median
**0.000017 m**, max **0.000037 m**.

Ceres wins on wall clock at this size — 147 points is far below the scale where
a GPU pays off, and CASPAR carries per-solve setup. The result to read here is
accuracy, not speed.

### 5.3 Galileo, 226 images — **PASS**

`data/cusfm_runs/galileo_colsfm/cusfm/sparse`, 226 images / 29 frames.

Two notes on method, because the brief's framing needs correcting:

1. **`OPENCV_FISHEYE` with `k1..k4 = 0` does not reproduce `PINHOLE`.** It is an
   *equidistant* projection (`θ = atan(r)`), not a perspective one. Swapping the
   model id on Galileo's cameras and keeping the stored keypoints leaves the
   model ~60 px inconsistent, and bundle-adjusting from there is an ill-posed
   problem that measures divergence, not the adapter. So the probe **re-renders
   every observation through the new model** from the existing 3D points, making
   the converted model exactly self-consistent, and then applies the same
   perturbation as the synthetic case. The testable statement is: *the fisheye
   adapter agrees with Ceres as closely as the shipped PINHOLE adapter does.*
2. **852 of 6194 tracks were pruned.** Real reconstructions carry points nearly
   in a camera's principal plane; re-rendering puts their keypoints astronomically
   off-sensor (mean reprojection errors of 1e150 px were measured), which
   saturates CASPAR's float32 accumulators so badly that it fails to solve even
   the **shipped PINHOLE** model. Dropping them makes the problem well-posed for
   both backends rather than special-casing one. 5342 tracks remain.

| case | model | adapter | CASPAR reproj | CASPAR score | Ceres score | max pose Δ (CASPAR vs Ceres) | point Δ median / max |
|---|---|---|---:|---:|---:|---|---:|
| `galileo-pinhole` | PINHOLE | **shipped** | 0.016826 px | 1.361e+01 | 1.75e-16 | 0.000090 m / 0.001187° | 0.000022 / 0.000745 m |
| `galileo-zero-k` | OPENCV_FISHEYE, k=0 | **new** | 0.010762 px | 5.949e+00 | 9.40e-18 | 0.000070 m / 0.000892° | 0.000014 / 0.000691 m |
| `galileo-small-k` | OPENCV_FISHEYE, k=(0.02,−0.01,0.005,−0.001) | **new** | 0.014026 px | 1.055e+01 | 9.69e-18 | 0.000091 m / 0.001078° | 0.000018 / 0.000507 m |

All three terminated with `CONVERGENCE`. Pose deltas are ≤ 0.005 % of the 1.96 m
trajectory extent.

**The fisheye adapter is as accurate as the shipped PINHOLE adapter on the same
model, in the same environment, at both zero and non-zero distortion.** The
residual gap in all three rows is the float32-versus-double gap CASPAR always
has, not a fisheye-specific error. Solve times: 0.23–0.31 s CASPAR vs
0.05–0.06 s Ceres — again, 5.3k points is below the GPU crossover.

### 5.4 No regression on PINHOLE / SIMPLE_RADIAL — **PASS**

`tools/caspar_probe.py` (the pre-existing probe, untouched) run in both
environments. Same scenes, same seeds, same options; only the pycolmap differs.

| problem | metric | `colsfm-caspar` (stock) | `colsfm-caspar-fisheye` (fork) |
|---|---|---:|---:|
| synthetic rig, 5 frames × 2 cam, PINHOLE | CASPAR final reproj | 0.000096 px | 0.000106 px |
| | max pose change vs Ceres | 0.000004 m / 0.000061° | 0.000005 m / 0.000067° |
| | point agreement median / max | 0.000002 / 0.000006 m | 0.000002 / 0.000005 m |
| Galileo global BA, 226 images | Ceres final reproj | 1.362713 px | 1.362713 px |
| | **CASPAR final reproj** | **1.362384 px** | **1.362338 px** |
| | max pose change vs Ceres | 0.000657 m / 0.009167° | 0.000725 m / 0.009998° |
| | point agreement median / max | 0.000164 / 0.029255 m | 0.000169 / 0.043621 m |
| | speedup (ceres/caspar) | 2.25× | 2.56× |

The two CASPAR results agree to **4.6e-05 px** on a 226-image model. That
residual is float32 run-to-run variation, not a behaviour change: the
pre-existing kernels are byte-identical between the two builds (§2), so nothing
about the PINHOLE path could have changed. Both terminated with `CONVERGENCE`.

### 5.5 Probe suite — **PASS**

`pixi run -e colsfm-caspar-fisheye python -m pytest tests/colsfm_probes -q`:
**60 passed** in 26.21 s (one unrelated `pkg_resources` DeprecationWarning from
`pycusfm/cusfm_runner.py`).

---

## 6. How to point colsfm at it

```bash
# The validation probe (this document's §5):
pixi run -e colsfm-caspar-fisheye caspar-fisheye-probe

# Or explicitly, with knobs:
pixi run -e colsfm-caspar-fisheye python -m tools.caspar_fisheye_probe \
    --distortion 0.02 -0.01 0.005 -0.001 \
    --hazard-iterations 60 \
    --gpu-index 0

# The pycolmap capability suite in this environment:
pixi run -e colsfm-caspar-fisheye caspar-fisheye-test
```

Once RoboCap is available again, run the colsfm pipeline on it with the GPU
backend — the cameras must be `OPENCV_FISHEYE` in the model, which stock CASPAR
would skip and this build solves natively:

```bash
pixi run -e colsfm-caspar-fisheye python -m colsfm run \
    --input-dir data/cusfm_runs/robocap_blobref/input \
    --output-dir data/cusfm_runs/robocap_colsfm_caspar/cusfm \
    --ba-backend caspar
```

(§9.3 has this command with the short-slice variant the user prefers for quick
iteration; the pipeline has no frame-limit flag, so the slice is a filtered
`frames_meta.json` plus symlinked images.)

Two constraints CASPAR imposes on any such run, fisheye or not:
`refine_focal_length` must equal `refine_extra_params` (the merged
focal-and-extra node cannot be split), and `refine_sensor_from_rig` must be
false on multi-sensor frames. A pre-calibrated rig like RoboCap satisfies both
by holding intrinsics and extrinsics fixed.

Keep the Ceres polish pass regardless: this is a float32 backend, and §5.3 shows
CASPAR landing 1e+01 in score where Ceres lands 1e-16.

---

## 7. Status at the time box

Everything in §5 completed and passed inside the 3-hour box. Raw output at
`/tmp/caspar-fisheye-build.log` (build), `/tmp/regress.log` (§5.4) and
`/tmp/pytest-fisheye.log` (§5.5).

Not attempted: a `CASPAR_USE_DOUBLE=ON` build of the fork. The f64 tree is
regenerated and committed, but no package builds it, so the f64 kernels are
**generated but not compiled**. `packages/pycolmap-caspar64/recipe.yaml`
retargeted at the fork path would test it in ~6 minutes.

---

## 8. What an upstream PR would need

The slot is publicly unclaimed: on #4611 the author said OPENCV_FISHEYE is "a
natural next step… if someone else wants to pick it up this PR's
`caspar_generate.py` changes are a reasonable template to follow."

**Scope.** ~650 hand-written lines plus ~124k lines of machine output across 461
files. That size is the problem, not the content: #4611 (+54k), #4463 (+18k) and
#4657 (+30k) have all been open for months, and #4463's author named the cause
outright — "SymForce autogenerates hundreds of files… bloating the apparent size
of this PR. Wondering if there's a cleaner way to do Caspar PRs in the future."
Upstream has no policy for reviewing generated diffs, and that unresolved
question is plausibly what stalls all three. Do not schedule around a merge.

**What this fork has that the stalled PRs do not:**

1. **Both precision trees.** #4611 regenerated f32 only; `CASPAR_USE_DOUBLE=ON`
   would not compile on that branch. Both are here — though the f64 tree is
   generated, not yet *compiled* (§7). Building it is a prerequisite for
   submission.
2. **The arity-equivalence argument, stated explicitly**, so the #4371
   shared-memory objection is pre-empted rather than re-litigated:
   `OPENCV_FISHEYE` is 8 params, byte-for-byte the node layout of `OPENCV`;
   `THIN_PRISM_FISHEYE`, the model tordnat said does not fit, is 12.
3. **A false-convergence check, not a clean-exit check** (§5.1), which is the
   one test that would have caught the #4611 NaN bug.
4. **A minimal generated diff.** Only the 224 new files and 4 aggregates change
   per tree; pre-existing kernels are untouched, which makes "did this regress
   PINHOLE?" answerable by inspection.

**What it would still need:**

- A C++ unit test alongside `bundle_adjustment_caspar_test.cc`, which already
  encodes the precision split (focal tolerance 1.0 px fp64 vs 20.0 px fp32).
- The `CreateSolver` argument-order comment, kept honest — it now says both
  orders were read off the generated signature. Upstream should probably assert
  this rather than comment it.
- A decision on §2's finding, which is a genuine upstream problem independent of
  this change: **COLMAP 4.2.0's `generated/` tree cannot be reproduced from any
  released symforce**, and its `solver.cc` never assigns
  `SolveResult::initial_score`, which `bundle_adjustment_caspar.cc:1010` then
  reads uninitialised. Either regenerate against 0.12.0 (which fixes it, at the
  cost of an 84-file scheduling churn) or pin the exact Caspar revision used.
  A PR that quietly changed those 84 files would look like unexplained noise;
  raising it as its own issue first is the cheaper path.

---

## 9. Wiring it into colsfm: the capability probe

§6's `--dataset robocap` snippet is a placeholder; the commands that actually
run are in §9.3 below.

`--ba-backend caspar` has always had a pre-flight check that falls back to Ceres
when the model carries a camera CASPAR would skip. Until this adapter existed
that check was a constant, `{PINHOLE, SIMPLE_RADIAL}`, so this environment's
fisheye build would have been refused the very rigs it was made for. The set is
now a property of the build, measured at run time.

### 9.1 How the probe knows

`colsfm.mapping.caspar_supported_camera_models()` builds a minimal two-frame,
sixteen-point reconstruction per candidate model — PINHOLE, SIMPLE_RADIAL,
OPENCV_FISHEYE, OPENCV, FULL_OPENCV — solves each on CASPAR, and reads
`BundleAdjustmentSummary.num_residuals`:

| | supported model | unsupported model |
|---|---|---|
| COLMAP log | — | `Skipping image N with unsupported camera model` |
| `num_residuals` | `2 x observations` | `0` |
| `termination_type` | `CONVERGENCE` | `USER_FAILURE` |

`num_residuals` is the signal rather than the termination type or a
did-the-poses-move test: it is bound by pycolmap, it is exactly zero when every
image was skipped, and it does not confuse "CASPAR refused this model" with
"CASPAR failed for some other reason". A build without `CASPAR_ENABLED` throws
on the first `create_default_bundle_adjuster`, which the probe reports as the
empty set — and an empty set disables the camera check, because there is no
half-problem to protect against and the existing solve-time fallback already
names the missing build.

Measured on the 5090, one process each:

| environment | probe answers | probe cost |
|---|---|---|
| `colsfm` (stock pycolmap) | `{}` | 0.001 s |
| `colsfm-caspar` | `{PINHOLE, SIMPLE_RADIAL}` | 0.194 s |
| `colsfm-caspar-fisheye` | `{OPENCV_FISHEYE, PINHOLE, SIMPLE_RADIAL}` | 0.244 s |

The probe is `functools.cache`d, so a run pays for it once, and it is reached
only when `--ba-backend caspar` was asked for.
`MappingOptions.caspar_supported_models` overrides it with a stated set — that
is what the unit tests use, so the rule "model not in the set means Ceres" is
asserted identically on every environment, and it is also how to force a model
through a build whose adapter is not yet trusted.

### 9.2 What it changes

A synthetic two-camera OPENCV_FISHEYE rig, 30 frames, 1495 recovered points,
mapped end to end through `run_mapping` with the isaac config, `--ba-backend
caspar` in both environments:

| | `colsfm-caspar` | `colsfm-caspar-fisheye` |
|---|---|---|
| backend that ran | `ceres` (fallback) | **`caspar`** |
| bundle adjustment (s) | 0.061 | 0.135 (0.045 of it polish, 2 iterations) |
| total mapping (s) | 0.249 | 0.331 |
| points / reprojection | 1495 / 0.3570 px | 1495 / 0.3570 px |

Against a plain `--ba-backend ceres` run of the same rig in
`colsfm-caspar-fisheye` (0.063 s of BA, same 1495 points, same 0.3570 px), the
CASPAR-plus-polish poses agree to **0.037 mm / 0.0000 deg worst case, 0.020 mm
median** over the 30 rig frames. The GPU is *slower* here, as on Galileo, and
for the same reason: a fifteen-hundred-point problem does not pay for the kernel
launches. What the measurement establishes is agreement, not speed; the speed
case is KITTI-scale (NOTES.md "CASPAR backend": 148.5 s to 41.3 s on 2156
images).

Galileo, 226 pinhole images, `--ba-backend caspar` in `colsfm-caspar-fisheye`,
benched against the blob reference: **225/226 images, 6301 points, 1.389 px,
4.313 mm ATE, 4/4 acceptance bounds**, mapping 2.61 s of a 16.91 s total, polish
21 iterations in 0.198 s, `mapping.ba_backend == "caspar"` in `summary.json`.
That is `colsfm-caspar`'s 4.30 mm / 4-of-4 to within the run-to-run noise, so
the extra adapters cost the pinhole path nothing.

### 9.3 The RoboCap run, for when the data is back

RoboCap is the fisheye rig this whole adapter is for, and it was off disk while
this was written; nothing below has been executed against it.

```bash
# Full segment, GPU bundle adjustment, Ceres polish on (the default):
pixi run -e colsfm-caspar-fisheye python -m colsfm run \
    --input-dir data/cusfm_runs/robocap_blobref/input \
    --output-dir data/cusfm_runs/robocap_colsfm_caspar/cusfm \
    --ba-backend caspar
```

Confirm it really went to the GPU rather than falling back — the run prints
`this CASPAR build projects [...] and this model has [...]` when it did not, and
`summary.json` records what actually ran:

```bash
python -c "import json;print(json.load(open('data/cusfm_runs/robocap_colsfm_caspar/cusfm/summary.json'))['mapping'])"
# expect: 'ba_backend': 'caspar', with 'polish_seconds' and 'polish_iterations' set
```

**A short slice, for quick iteration.** The pipeline has no frame-limit flag —
`--min-inter-frame-distance` and `--min-inter-frame-rotation-degrees` thin the
keyframes but never truncate the sequence — so trim `frames_meta.json` and leave
the imagery where it is. `FramesMeta.filtered` keeps a subset of keyframe ids,
and `rig_frames()` groups them by synchronised sample, so N rig frames is N
synchronised samples of every camera:

```bash
cat > /tmp/robocap_slice.py <<'PY'
"""Write a short-slice input directory: the first N rig frames, images symlinked."""
from __future__ import annotations
from pathlib import Path
import tyro
from colsfm.frames_meta import FRAMES_META_NAME, FramesMeta, read_frames_meta, write_frames_meta


def main(input_dir: Path, output_dir: Path, num_rig_frames: int = 40) -> None:
    """Link the imagery of `input_dir` into `output_dir` under a trimmed `frames_meta.json`."""
    meta: FramesMeta = read_frames_meta(input_dir / FRAMES_META_NAME)
    keep: list[int] = [kid for rig_frame in meta.rig_frames()[:num_rig_frames] for kid in rig_frame.keyframe_ids]
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(input_dir.iterdir()):
        if entry.name == FRAMES_META_NAME:
            continue
        link: Path = output_dir / entry.name
        if not link.exists():
            link.symlink_to(entry.resolve())
    write_frames_meta(output_dir / FRAMES_META_NAME, meta.filtered(keep))
    print(f"[slice] {len(keep)} of {len(meta.keyframes)} keyframes, {num_rig_frames} rig frames -> {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
PY

pixi run -e colsfm-caspar-fisheye python /tmp/robocap_slice.py \
    --input-dir data/cusfm_runs/robocap_blobref/input \
    --output-dir /tmp/robocap_slice --num-rig-frames 40

pixi run -e colsfm-caspar-fisheye python -m colsfm run \
    --input-dir /tmp/robocap_slice \
    --output-dir /tmp/colsfm_runs/robocap_slice_caspar/cusfm \
    --ba-backend caspar
```

The recipe was validated on Galileo, which stands in for RoboCap's directory
layout: 10 rig frames of 226 keyframes gave a 78-image input that mapped in
6.98 s against the full run's 16.91 s.

The comparison worth making once the full run exists is the same one §5 makes on
the synthetic rig — `--ba-backend caspar` against `--ba-backend ceres` on
identical input — because on a real fisheye rig the fallback is no longer
available as a control: in `colsfm-caspar` the caspar run *is* the ceres run.

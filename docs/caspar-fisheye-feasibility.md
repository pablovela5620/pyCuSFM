# Feasibility: an `OPENCV_FISHEYE` adapter for COLMAP's CASPAR GPU bundle adjuster

Read-only study. Source of truth: COLMAP 4.2.0, unpacked at `/tmp/colmap420`
(tag `4.2.0` = `be5e29168d4aff238409d60424812df66aac919f`, the same revision
`packages/pycolmap-caspar/recipe.yaml` builds). All line numbers below refer to
that tree.

## TL;DR

The blocker everyone assumes — "the Symforce kernels are pre-generated blobs, we
cannot make new ones" — **is false**. The generator is in the COLMAP tree
(`src/thirdparty/Symforce-Caspar/caspar_generate.py`, 396 lines, Apache-2.0) and
the `symforce.caspar` extension it imports is open source and released
(symforce-org/symforce, `symforce/caspar/`, present in tag `v0.12.0`; PyPI
`symforce` 0.12.0).

Better still, the work has a near-exact template already on GitHub: **PR #4611,
"Add OPENCV camera model support to Caspar GPU bundle adjustment"** (open since
2026-08-04). `OPENCV` and `OPENCV_FISHEYE` have *identical parameter arity* — both
are `(2 focal, 2 principal point, 4 extra) = 8 params`
(`src/colmap/sensor/models.h:536` and `:554`). Only the distortion expression
differs, and that expression is ~10 lines of Symforce.

**Estimate: 1–2 days to a working fork, 3–5 days to something upstreamable.**

---

## 1. Where do the generated kernels come from?

### The generator is in-tree

```
src/thirdparty/Symforce-Caspar/
├── LICENSE                  (Apache-2.0)
├── caspar_generate.py       ← 396 lines: the symbolic model definitions + codegen driver
├── msvc_compact.h
├── msvc_cuda_compact.h
└── generated/{f32,f64}/     ← 488 files, 86,133 lines each (f32 measured)
```

That is the **entire** non-generated content: 4 files. There is no git submodule
(`/tmp/colmap420/.gitmodules` does not exist), so nothing is hidden behind a
checkout step.

`caspar_generate.py` is invoked as `python caspar_generate.py <out_dir> [f32|f64]`
and imports:

```python
import symforce.symbolic as sf
from symforce.caspar import CasparLibrary
from symforce.caspar import memory as mem
```

It defines the node types, the residual functions, and the fixed-variable variant
matrix, then calls `caslib.generate(out_dir, use_symlinks=False,
python_bindings=False)`. The two residual cores are ~20 lines each —
`simple_radial_core` and `pinhole_core` — and Symforce does every Jacobian.

The COLMAP build even tells you the file is the source of truth when the
generated tree is missing (`src/thirdparty/CMakeLists.txt:109-114`):

```
"Caspar: pre-generated code not found for precision '${_caspar_precision}'.\n"
"Expected: ${CASPAR_GEN_DIR}/CMakeLists.txt\n"
"Re-run generate_caspar.py to regenerate."
```

### The upstream dependency is public

- Paper: `doc/bibliography.rst:11-14` — Martens, Emil; Miller, Aaron; Varnum,
  Matias; Stahl, Annette. *"Caspar: CUDA Accelerator for Symbolic Programming
  with Adaptive Reordering."* ICRA 2026.
- Code: `symforce/caspar/` in <https://github.com/symforce-org/symforce>, present
  in released tag `v0.12.0` (subdirs: `code_formulation/`, `code_generation/`,
  `memory/`, `source/`, `examples/`). Its README documents `CasparLibrary`,
  `add_kernel`, `add_factor` and the memory annotations (`TunableShared`,
  `ConstantSequential`, …) that `caspar_generate.py` uses.
- PyPI `symforce` is at 0.12.0, so the toolchain is `pixi add symforce`-able.

**Answer: both the generator and its upstream are available. Nothing is
proprietary or missing.**

---

## 2. What does one adapter consist of?

Taking `SimpleRadial` as the unit of work. Five distinct surfaces:

### 2.1 The Symforce definition — `src/thirdparty/Symforce-Caspar/caspar_generate.py`

| Piece | Lines | Notes |
|---|---|---|
| Node classes | 8 classes | `SimpleRadialPose` / `Const…`, `SimpleRadialCalib` (V4) / `Const…`, `SimpleRadialPrincipalPoint` (V2) / `Const…`, `SimpleRadialFocalAndExtra` (V2) / `Const…`, plus `ConstSimpleRadialSensorFromRig` |
| `simple_radial_core` | ~20 | the merged-calib residual |
| `simple_radial_split_core` | ~15 | delegates to the merged core |
| `FIXABLE_SIMPLE_RADIAL`, `FIXABLE_SIMPLE_RADIAL_SPLIT` | ~12 | which params may be held fixed |
| 2 × `register_camera_model(...)` | ~10 | merged (`include_all_fixed=True`) + split (`must_fix_one_of=…`) |

**~65–70 lines of Python per camera model.** PR #4611's diff on this file is
`+151` for OPENCV, which includes shared scaffolding.

Node types are deliberately per-model — the comment at `caspar_generate.py:53-56`
says *"Pose and Calib nodes are camera-model-specific to prevent cross-model
batching"*. That is why a new model cannot reuse the pinhole pose pool.

### 2.2 The generated kernels — `generated/{f32,f64}/`

The variant matrix (`caspar_generate.py:41-50`): **4 merged + 11 split = 15
variants per camera model**, mirrored in C++ by `#define CASPAR_NUM_VARIANTS 15`
(`src/colmap/estimators/bundle_adjustment_caspar.h:52`) and the
`enum class FactorVariant` at `bundle_adjustment_caspar.h:53-73`. The variant
RoboCap actually needs is named in that enum:

```cpp
FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT,  // calibrated camera
```
— `bundle_adjustment_caspar.h:65`

Measured in `generated/f32/`:

| Kind | Files | Lines |
|---|---:|---:|
| Factor kernels `kernel_simple_radial*` (15 variants × 4 kernels × {.cu,.h}) | 120 | 26,275 |
| Node kernels `kernel_SimpleRadial{Calib,FocalAndExtra,Pose,PrincipalPoint}_*` (4 node types × 13 solver ops × {.cu,.h}) | 104 | 6,091 |
| **Per model, per precision** | **224** | **~32,400** |
| Shared infra (`solver.*`, `caspar_mappings.*`, `Point*`, CMakeLists, …) | 40 | remainder |
| f32 tree total (2 models) | 488 | 86,133 |

The 4 factor kernels per variant are `res_jac`, `res_jac_first`,
`jtjnjtr_direct`, `score`. The 13 node ops are `retract`, `normalize`,
`update_p`, `update_r`, `update_r_first`, `update_step`, `update_step_first`,
`update_Mp`, `start_w`, `start_w_contribute`, `alpha_numerator_denominator`,
`alpha_denominator_or_beta_numerator`, `pred_decrease_times_two`.

So a new model adds roughly **~224 files and ~32k lines per precision tree**,
i.e. ~450 files / ~65k lines if both `f32` and `f64` are regenerated. This is
entirely machine output — it is diff noise, not work.

### 2.3 CMake registration — **none required**

`generated/f32/CMakeLists.txt:17-20`:

```cmake
file(GLOB CUDA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cu" "*.cc")
list(FILTER CUDA_SOURCES EXCLUDE REGEX "pybind.*")
file(GLOB CUDA_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cuh" "*.h")
```

A glob. New kernels are picked up with zero CMake edits. `src/thirdparty/CMakeLists.txt:116-122`
pulls the directory in via `FetchContent_Declare(caspar SOURCE_DIR ${CASPAR_GEN_DIR})`,
also model-agnostic.

### 2.4 The C++ adapter — `src/colmap/estimators/caspar/caspar_model_adapter.h` (986 lines)

| Location | Content |
|---|---|
| `:19-50` | `CasparSolverSizing` SimpleRadial fields — **17 `size_t` members**: 1 pose pool, 1 calib pool, 15 variant counts |
| `:73-135` | `ICasparModelAdapter` — the 18-method pure-virtual interface every adapter implements |
| `:137-527` | `class SimpleRadialAdapter` — **389 lines** |
| `:528-902` | `class PinholeAdapter` — 375 lines |
| `:904-914` | `CreateCasparAdapter` factory — a 2-line `case` per model |
| `:916-960` | `CreateSolver` — the positional-argument list, see warning below |

The interface is already size-parameterized, which is what makes an 8-param model
possible without touching it:

```cpp
virtual size_t FocalAndExtraSize() const = 0;   // SimpleRadial: 2   OPENCV_FISHEYE: 6
virtual size_t PrincipalPointSize() const = 0;  //              2                   2
virtual size_t CalibSize() const = 0;           //              4                   8
```
— `caspar_model_adapter.h:80-86`

The bulk of the 389 lines is `SetVariantFactors`, a 15-way `switch` where each
arm makes 5–6 `s.SetSimpleRadial…FromHost(...)` calls into the generated solver.
Pure boilerplate, name-mangled per variant.

The one genuinely hazardous edit is `CreateSolver`, whose own comment says
(`caspar_model_adapter.h:916-921`):

```
// WARNING: Argument order is opaque and bug-prone and will change in a future
// Caspar release. Order:
//   1. Node type counts, alphabetical by type name
//   2. Factor counts, in registration order from caspar_generate.py
```

Adding `OpenCVFisheye*` node types shifts alphabetical positions (`OpenCVFisheyeCalib`
sorts before `PinholeCalib`), so every existing argument moves. Silent
mis-ordering here yields wrong sizing, not a compile error.

### 2.5 Dispatch — `src/colmap/estimators/bundle_adjustment_caspar.cc` (1044 lines)

Almost nothing. `GetAdapter` (`:43-52`) caches whatever `CreateCasparAdapter`
returns; `nullptr` means "unsupported", producing the two skip warnings:

```cpp
LOG(WARNING) << "Skipping image " << image_id
             << " with unsupported camera model: " << camera.ModelName();
```
— `:65-69` (and the external-observation twin at `:253-259`)

The only model-specific code is the sizing hookup at `:876-901`, which
hard-codes `CameraModelId::kSimpleRadial` and `kPinhole`. PR #4611's diff on this
whole file is **+9 lines**.

### 2.6 Documentation

`doc/faq.rst:664-669` carries the supported-model list and would need editing:

> Caspar is experimental and currently supports only the ``SIMPLE_RADIAL`` and
> ``PINHOLE`` camera models; observations using other camera models are skipped.

---

## 3. The `OPENCV_FISHEYE` adapter itself

### The math COLMAP already specifies

`OPENCV_FISHEYE` derives from `BasePerspectiveFisheyeCameraModel`
(`src/colmap/sensor/models.h:551-556`), params `fx, fy, cx, cy, k1, k2, k3, k4`.
The projection is three pieces:

1. `FisheyeFromNormal` — `models.h:429-438`:
   ```cpp
   const T r = ceres::sqrt(u * u + v * v);
   if (r > T(std::numeric_limits<double>::epsilon())) {
     const T theta = ceres::atan(r);
     *uu *= theta / r;  *vv *= theta / r;
   }
   ```
2. `Distortion` — `models.h:1660-1675`. Note COLMAP works in `(uu, vv)` where
   `theta2 = uu*uu + vv*vv` **is already θ²**, so this is the Kannala–Brandt
   polynomial written additively:
   ```cpp
   const T radial = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
   *du = u * radial;  *dv = v * radial;
   ```
3. `ImgFromFisheye` — `models.h:1601-1611`: `x = f1*uu + c1`, `y = f2*vv + c2`.

Composed in `ImgFromCam` (`models.h:1625-1647`). Equivalent to the brief's
θ_d = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸), pixel = f·θ_d·(x,y)/r + c.

### The Symforce core, written out

```python
def opencv_fisheye_core(
    pose, sensor_from_rig,
    calib,   # V8: [fx, fy, k1, k2, k3, k4, cx, cy]
    point, pixel,
) -> sf.V2:
    cam_T_world = sensor_from_rig * pose
    fx, fy, k1, k2, k3, k4, cx, cy = calib
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = sf.sqrt(p.squared_norm() + sf.epsilon())      # epsilon guards r -> 0
    theta = sf.atan(r)
    t2 = theta * theta
    scale = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) / r
    return sf.V2([fx * scale * p[0] + cx, fy * scale * p[1] + cy]) - pixel
```

~14 lines. Symforce differentiates it; nobody hand-writes a Jacobian.

Note the epsilon handling: COLMAP's `if (r > eps)` branch has no Symforce
equivalent (branches do not survive symbolic codegen). The established pattern in
the existing cores is `sf.epsilon()` / `sf.sign_no_zero`
(`caspar_generate.py`, `simple_radial_core`), and `symforce.set_epsilon_to_number`
is already called at module load with `1e-6` for f32 / `1e-15` for f64. This is
the one place the port needs numerical care rather than transcription.

### The shared-memory question is already answered — and it answers *for* us

Issue **#4371** ("Questions about adding THIN_PRISM_FISHEYE support to the Caspar
backend"), answered by `tordnat` (the Caspar contributor):

> In the current version, most consumer grade GPUs do not have enough shared
> memory to support all the shared variables needed for THIN_PRISM_FISHEYE. This
> is a known limitation we intend to address… I would recommend looking at [`caspar_generate.py`]
> for hints as to what is needed to implement a new camera type… The dispatch C++
> code would also need to be updated (which is a lot of boilerplate), but this
> should be trivial for an LLM to do provided that we already have two examples.

That objection **does not apply to `OPENCV_FISHEYE`**. Parameter arity decides
the shared-memory footprint, and:

| Model | `models.h` | focal / pp / extra | total params | merged calib node |
|---|---|---|---:|---|
| `SIMPLE_RADIAL` | shipped | 1 / 2 / 1 | 4 | V4 |
| `PINHOLE` | shipped | 2 / 2 / 0 | 4 | V4 |
| `OPENCV` | `:536` | 2 / 2 / 4 | **8** | V8 — **shipped in PR #4611** |
| **`OPENCV_FISHEYE`** | `:554` | 2 / 2 / 4 | **8** | **V8 — same size as OPENCV** |
| `THIN_PRISM_FISHEYE` | `:651` | 2 / 2 / 8 | 12 | V12 — the model tordnat says does not fit |

`OPENCV_FISHEYE` is byte-for-byte the same node layout as `OPENCV`, and PR #4611
demonstrates that layout building and running on real hardware. The 48 KB limit
is not a blocker here.

### Estimates

| Path | Work | Estimate |
|---|---|---|
| **(a) Regenerate with the Symforce generator** *(the generator IS available)* | `caspar_generate.py` +~150; run for f32 **and** f64; `caspar_model_adapter.h` +~450; `bundle_adjustment_caspar.cc` +9; `CreateSolver` arg-order surgery; `doc/faq.rst`; numerical validation vs Ceres on the RoboCap rig | **1–2 days** to a working fork build; **3–5 days** to upstreamable (both precisions, tests, real-data validation) |
| **(b) Hand-port without the generator** | Hand-write and hand-differentiate ~32k lines of register-scheduled CUDA across 15 variants × 4 kernels × 13 node ops, in two precisions | **4–8 weeks**, and the result would be slower and less trustworthy than the generated code. **Do not do this.** It is moot anyway — the generator is available. |
| **(c) Generic bearing-vector adapter** (covers all calibrated models at once) | See below | **2–4 days**, but with a semantic cost |

#### On (c), the bearing-vector adapter

**Is CASPAR's factor structure amenable? Yes — and there is proof.** PR #4463
("Add EQUIRECTANGULAR support to the Caspar GPU bundle adjuster", open) ships a
`SphericalAdapter` whose *"calib / focal / principal-point methods are no-ops"*
and gates the refine toggles on `FocalAndExtraSize() / PrincipalPointSize() > 0`.
That is exactly the shape a bearing adapter needs: no calib node, only
`Pose × Point × <constant observation>`.

A bearing adapter would be *smaller* than a normal one, not larger:

- Replace `ConstPixel` (V2) with a constant unit-bearing node (V3, or a V2
  tangent-plane observation plus a stored basis).
- Drop `Calib`, `FocalAndExtra`, `PrincipalPoint` node types entirely — that
  removes 3 of the 4 node kernel groups (~78 of 104 node kernel files).
- The variant matrix collapses from **15 to 3** (`BASE`, `FIXED_POSE`,
  `FIXED_POINT`) because there are no intrinsics to fix — ~24 factor kernel
  files instead of 120.

Total generated footprint: roughly **1/5** of a per-model adapter, and it would
serve `OPENCV_FISHEYE`, `THIN_PRISM_FISHEYE`, `RAD_TAN_THIN_PRISM_FISHEYE`,
`FULL_OPENCV`, `FOV`, every fisheye variant — one adapter, all models.

**The cost, and it is real:**

1. **Intrinsics cannot be refined.** The host unprojects observations with the
   true model once; `refine_focal_length`, `refine_extra_params`,
   `refine_principal_point` become inexpressible. Any of them set → must error,
   not silently skip.
2. **The residual changes units.** Angular/tangent-plane error, not pixels.
   COLMAP's loss thresholds, `max_reproj_error` filtering, and every downstream
   consumer of BA residuals are calibrated in pixels. A per-observation 2×2
   whitening matrix (the pixel-space Jacobian of the projection at the
   observation) recovers approximate pixel scaling, but exactness needs it
   recomputed each iteration — impossible for a *constant* node. So the
   robustifier's cut point drifts against the Ceres path.
3. It is an architectural change to a backend that has no such abstraction today,
   so upstream review is a much bigger conversation than "one more model".

For RoboCap specifically — a pre-calibrated rig where intrinsics are held fixed —
limitation (1) is free and (2) is a tolerable approximation. It is a genuinely
attractive option *if* the goal is "all fisheye models on GPU"; it is the wrong
tool if the goal is "self-calibrating fisheye BA".

---

## 4. Would upstream accept it?

**Yes, on current evidence — but do not schedule around the merge.**

Prior art and precedent (queried 2026-09-05):

| # | Title | State |
|---|---|---|
| **#4611** | Add **OPENCV** camera model support to Caspar GPU bundle adjustment | **OPEN** (2026-08-04) — `+54,489 / −9,486`, 231 files |
| #4463 | Add EQUIRECTANGULAR support to the Caspar GPU bundle adjuster | OPEN — `+18,429 / −63`, 116 files |
| #4657 | Add fixed-calibration **THIN_PRISM_FISHEYE** support for Caspar BA | DRAFT — `+30,005 / −13,260`, 99 files |
| #4655, #4656 | earlier THIN_PRISM_FISHEYE attempts | CLOSED |
| #4371 | Questions about adding THIN_PRISM_FISHEYE support to the Caspar backend | OPEN |
| #4385 | Add rig support to Caspar BA | MERGED |
| #4441 | Add Spherical Camera Model | MERGED |

Signals in favour:

- The maintainer-side answer in #4371 is an explicit invitation: *"I would
  recommend looking at [`caspar_generate.py`] for hints as to what is needed to
  implement a new camera type."*
- On #4611 a user asked verbatim **"Are you planning to add support for
  OPENCV_FISHEYE?"** and the author replied: *"Not yet, but it's a natural next
  step and should follow the same recipe as this PR's OPENCV support — happy to
  take a stab at it if there's interest, or if someone else wants to pick it up
  this PR's `caspar_generate.py` changes are a reasonable template to follow."*
  The slot is open and publicly unclaimed.
- Maintainer `ahojnnes` engaged constructively on #4611, asking only for a
  scope split (HIP support moved out to #4618) — a process note, not a
  rejection.
- Camera-model additions to Caspar are an accepted category, not a novelty.

Signals against, i.e. why not to *depend* on it:

- #4611 has been open a month with no merge; #4463 likewise; three
  THIN_PRISM_FISHEYE attempts have gone nowhere. **No third camera model has
  landed in Caspar yet.** Merge latency here is measured in months.
- These PRs are 100k-line diffs of machine output. #4463's author says it
  plainly: *"SymForce autogenerates hundreds of files and nearly 20,000 lines,
  bloating the apparent size of this PR. Wondering if there's a cleaner way to do
  Caspar PRs in the future."* Upstream has no policy for this yet, and that
  unresolved question is plausibly what is stalling all of them.

### One concrete hazard to inherit

While validating #4611 on real hardware the author found a **correctness bug in
the generated kernels**, not in the adapter:

> the two OpenCV split-variant score kernels … read their per-thread score
> accumulator unconditionally in `SumStore()`, but only ever assign it inside a
> `problem_size` guard. For any thread beyond `problem_size` … that's an
> uninitialized-variable read, which reliably decoded as NaN for these two
> kernels specifically (larger, more register-pressured than SimpleRadial's /
> Pinhole's equivalents) and corrupted the whole reduction from iteration 0.

Symptom: *"a false 3-iteration convergence exit with unchanged output"* — BA
reports success and silently does nothing. An `OPENCV_FISHEYE` adapter uses the
same generator and the same register-pressure regime, so it is likely to hit the
same class of bug. **Budget for it: any validation must assert that the score
actually decreases over 50+ iterations, never merely that the solver exits
cleanly.**

Second inherited gap: PR #4611 regenerated **only `f32`** (97 f32 files, 0 f64).
A `CASPAR_USE_DOUBLE=ON` build of that branch would not compile. Regenerate both
trees. `packages/pycolmap-caspar/recipe.yaml` builds f32 by default, so this
would not bite us immediately, but it is a correctness trap and an upstream
review comment waiting to happen.

---

## 5. Recommendation

**Ship the shadow-pinhole + Ceres-polish workaround now. Start path (a) in a
fork in parallel. Offer it upstream, but treat the merge as a bonus.**

Reasoning:

1. **Path (a) is cheap and de-risked.** 1–2 days for a working fork, against a
   working template (#4611) for a model of *identical* arity, with the generator
   in-tree and the toolchain on PyPI. The C++ side is, in the maintainer's own
   words, *"a lot of boilerplate… trivial for an LLM to do."*
2. **The known objection does not apply.** tordnat's shared-memory concern in
   #4371 targets 12-param `THIN_PRISM_FISHEYE`. `OPENCV_FISHEYE` is 8 params —
   the same footprint as `OPENCV`, which #4611 has already run on hardware.
3. **Do it in a fork first, because merge latency is real.** Three camera-model
   PRs are stalled, plausibly on the unresolved question of how to review 100k-line
   generated diffs. We already build COLMAP from source via
   `packages/pycolmap-caspar/recipe.yaml`, which pins a git rev — pointing that
   `source:` at a fork branch is a one-line change. There is no packaging
   obstacle to carrying this ourselves.
4. **Upstream it anyway, framed as a follow-on to #4611.** Cite #4611 as the
   template, state the arity-equivalence argument explicitly so the #4371
   shared-memory objection is pre-empted, ship both `f32` and `f64`, and include
   the score-decrease validation. That is a materially stronger submission than
   the three currently stalled, and it costs a day on top of the fork work.
5. **Keep the Ceres polish regardless.** Even with a native adapter, f32 GPU BA
   on a fisheye model warrants a double-precision final pass; and the polish path
   is the only cross-check that would have caught the #4611 NaN bug.
6. **Do not pursue (b).** Moot — the generator exists.
7. **Hold (c) in reserve.** The bearing-vector adapter is the right answer if the
   requirement ever broadens to "every fisheye model on GPU" and intrinsics stay
   fixed. #4463's `SphericalAdapter` proves the calib-free factor shape works.
   But it changes residual units and forecloses self-calibration, so it is not
   the right first move for one known model.

### Definition of done for path (a)

- [ ] `caspar_generate.py`: `OpenCVFisheye*` node types, `opencv_fisheye_core`,
      `opencv_fisheye_split_core`, two `register_camera_model` calls.
- [ ] Regenerate **both** `generated/f32` and `generated/f64`.
- [ ] `caspar_model_adapter.h`: `OpenCVFisheyeAdapter`
      (`FocalAndExtraSize()==6`, `PrincipalPointSize()==2`, `CalibSize()==8`),
      17 new `CasparSolverSizing` fields, factory case, and a careful
      re-ordering of `CreateSolver`'s positional arguments.
- [ ] `bundle_adjustment_caspar.cc:876-901`: sizing hookup.
- [ ] `doc/faq.rst:664-669`: update the supported-model sentence.
- [ ] Validation on the RoboCap rig: assert monotone score decrease over ≥50
      iterations and final residuals within tolerance of the Ceres reference —
      **not** merely a clean solver exit.

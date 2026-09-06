# Codex adversarial review of the CASPAR OPENCV_FISHEYE adapter patch (2026-09-05)

Model gpt-6-astra, xhigh, read-only. Question: was the adapter done the best way; can the same be achieved with less. Kept verbatim.

**The adapter follows CASPAR’s existing design, but the 6.3 MB patch is unnecessary.** The same native fisheye implementation can be carried as roughly **38 KB of handwritten changes plus controlled build-time generation**. A blanket regeneration would introduce extra changes, and the current implementation still has a correctness hazard.

I inspected commit `267dd0fd` against its parent, confirmed the stated base commit, and compared the saved generator outputs with the fork. I did not modify files, regenerate files, build, or run GPU tests.

1. **High: the new kernels retain the known uninitialized padding-thread read.**

   In the fixed-intrinsics score kernel, `r4` is declared without initialization, assigned inside `global_thread_idx < problem_size`, then passed **by value** to `SumStore` outside that guard. Inactive threads therefore read an uninitialized value. `SumStore` masks it inside the function, after argument evaluation. See [score kernel:48](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/generated/f32/kernel_opencv_fisheye_split_fixed_focal_and_extra_fixed_principal_point_score.cu:48), [assignment and call:238](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/generated/f32/kernel_opencv_fisheye_split_fixed_focal_and_extra_fixed_principal_point_score.cu:238), and [memops.cuh:331](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/generated/f32/memops.cuh:331).

   This is the same source-level hazard reported in [upstream PR #4611](https://github.com/colmap/colmap/pull/4611#issuecomment-3210129449). I did not reproduce a GPU failure here; the defect is present in the source regardless of whether the current compiler masks it successfully.

   The cause is regenerable too: Symforce’s [AddSum.write_template:539](/home/pablo/0Dev/forks/colmap-caspar-fisheye/.pixi/envs/default/lib/python3.11/site-packages/symforce/caspar/memory/accessors.py:539) emits that call. Fix the generator template so the argument is initialized or conditionally evaluated **before** the call. Passing a convergence probe on one GPU is insufficient.

2. **High: the validation can conceal projection failures and does not establish full adapter parity.**

   The probe computes its score with `np.nansum(residual**2)`. A projection that becomes NaN disappears from the score instead of failing the test. Its result also discards the solver’s residual count. See [probe:442](/home/pablo/0Dev/forks/pyCuSFM/.claude/worktrees/agent-a11c3b12911104e01/tools/caspar_fisheye_probe.py:442) and [probe:524](/home/pablo/0Dev/forks/pyCuSFM/.claude/worktrees/agent-a11c3b12911104e01/tools/caspar_fisheye_probe.py:524).

   There are **no test changes in the fork commit**. The external probe fixes all intrinsics, so it does not validate the new calibration-refinement variants. The recipe tests only CUDA availability, an options attribute, and an enum that also exists in stock COLMAP. See [probe options:418](/home/pablo/0Dev/forks/pyCuSFM/.claude/worktrees/agent-a11c3b12911104e01/tools/caspar_fisheye_probe.py:418) and [recipe tests:186](/home/pablo/0Dev/forks/pyCuSFM/.claude/worktrees/agent-a11c3b12911104e01/packages/pycolmap-caspar-fisheye/recipe.yaml:186).

   The notes explicitly say the fork’s **fp64 code was generated but never compiled**. [Design notes:396](/home/pablo/0Dev/forks/pyCuSFM/.claude/worktrees/agent-a11c3b12911104e01/docs/caspar-fisheye-adapter.md:396)

3. **Medium: the fork is a deliberate mixture of stock and regenerated code. Plain regeneration changes that mixture.**

   My comparisons found:

   | Check | Result |
   |---|---|
   | Generated files added | 448: 224 per precision |
   | Existing generated files replaced | Eight: `solver.{h,cc}` and `caspar_mappings.{h,cu}` per precision |
   | Existing PINHOLE/SIMPLE_RADIAL kernels changed by this commit | **Zero** |
   | Existing kernels that differ after a complete regeneration and COLMAP formatting | **82 fp32; 61 fp64** |
   | New/replaced files matching saved generation after formatting | **456/456** |
   | Saved repeated baseline fp32 generation | **488/488 byte-identical** |

   Thus, generating everything directly over `generated/` would also replace existing-model kernels. The observed differences include register allocation and instruction ordering; these can affect rounding, register pressure, and performance even when the symbolic objective is unchanged. They require separate regression validation.

   Regeneration also brings in a real stock bug fix: `result.initial_score = score_best`. I confirmed this is the **only difference in baseline-regenerated `solver.cc` after formatting**, for both precisions. The field otherwise has no initializer, and COLMAP reads it. See [SolveResult:30](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/generated/f32/solver.h:30), [fp32 assignment:3344](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/generated/f32/solver.cc:3344), [fp64 assignment:3374](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/generated/f64/solver.cc:3374), and [summary consumer:1019](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/estimators/bundle_adjustment_caspar.cc:1019).

   This fixes reported initial-score behavior; it does not itself change step selection. Both stock recipes retain the vendored code. The notes’ claim that existing models “cannot regress” is too strong because shared solver code changes. Their claim about the exact historical origin of the vendored snapshot is also stronger than the evidence: **output mismatch is verified; the original generator revision is not identified.**

4. **Medium: the projection is correct on its intended domain, but it does not mirror COLMAP exactly.**

   The adapter packs and unpacks parameters correctly:

   - COLMAP: `[fx, fy, cx, cy, k1, k2, k3, k4]`.
   - CASPAR merged calibration: `[fx, fy, k1, k2, k3, k4, cx, cy]`.
   - Split calibration: six focal/distortion values plus two principal-point values.

   The extraction and write-back are inverses. The variant dispatch follows `SimpleRadialAdapter`; after normalizing model names and whitespace, the substantive differences are the expected dimensions and parameter accesses. See [adapter:932](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/estimators/caspar/caspar_model_adapter.h:932) and [packing:1016](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/estimators/caspar/caspar_model_adapter.h:1016).

   Its polynomial is the right one:

   \[
   \theta_d=\theta(1+k_1\theta^2+k_2\theta^4+k_3\theta^6+k_4\theta^8).
   \]

   Angles are radians, distortion coefficients are dimensionless, and residuals are pixels. Pose composition is `sensor_from_rig * rig_from_world`. Symforce generates the tangent-space Jacobians; no fisheye Jacobian is handwritten. See [projection:358](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/caspar_generate.py:358), [COLMAP projection:1625](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/sensor/models.h:1625), and [Jacobian generation:371](/home/pablo/0Dev/forks/colmap-caspar-fisheye/.pixi/envs/default/lib/python3.11/site-packages/symforce/caspar/code_generation/factor.py:371).

   Two deviations matter:

   - **Near the axis:** `sqrt(r² + ε)` keeps derivatives finite, but the scale limit is approximately `1 + (k1 − 1/3)ε`, not exactly one as the comment claims. The center pixel remains correct; its derivative is slightly biased.
   - **Depth and cheirality:** CASPAR uses `z + ε sign(z)` and accepts negative depth. COLMAP’s default projection rejects nonprojectable depth, and its Ceres residual becomes zero when projection fails. The depth epsilon also breaks exact scale invariance. For `X=Z`, `Y=0`, zero distortion, and `fx=1000`, the fp32 epsilon alone causes about **0.005 px error at Z=0.1**, and **0.5 px at Z=0.001**, before float rounding.

   These depth conventions are inherited from the existing CASPAR cores, but remain limits of an “exact COLMAP adapter” claim. See [existing cores:284](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/caspar_generate.py:284), [COLMAP axis handling:429](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/sensor/models.h:429), and [Ceres failure handling:237](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/estimators/cost_functions/reprojection_error.h:237).

   There is also an inherited mixed-rig limitation: the same frame gets separate pose variables for different camera models, then both are written back to one frame pose. An all-fisheye rig avoids this; a mixed PINHOLE/fisheye rig needs explicit validation or rejection. [Pose allocation:407](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/estimators/bundle_adjustment_caspar.cc:407), [write-back:781](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/estimators/bundle_adjustment_caspar.cc:781)

5. **Medium: the consuming recipe is not yet reproducible from the exported patch.**

   At the supplied path it still uses `source.path: /home/pablo/0Dev/forks/colmap-caspar-fisheye`, with no `patches:` entry. The declared base revision does not control that source. This matches the notes’ description of patch-based packaging as future work. [Recipe:65](/home/pablo/0Dev/forks/pyCuSFM/.claude/worktrees/agent-a11c3b12911104e01/packages/pycolmap-caspar-fisheye/recipe.yaml:65)

The handwritten/generated split is:

| File or component | Handwritten change |
|---|---:|
| [caspar_generate.py](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/caspar_generate.py:123): node types, projection, split wrapper, registration | +170 |
| [caspar_model_adapter.h](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/estimators/caspar/caspar_model_adapter.h:932): adapter, sizing, factory, constructor arguments | +463/−2 |
| [bundle_adjustment_caspar.cc](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/estimators/bundle_adjustment_caspar.cc:884): sizing registration | +9 |
| FAQ and `.gitignore` | +7/−2 |
| CMake changes | **None** |
| Test changes | **None** |
| Everything changed under `generated/` | Generated output; all matched saved generation after formatting |

The operative three-file patch is **37,780 bytes, +642/−2 lines**. Including FAQ and `.gitignore`, it is **38,936 bytes, +649/−4**.

Build-time generation is practical. The driver takes an output directory and precision, imports Symforce, and emits kernels, mappings, solver files, build files, and copied runtime support. It does not need a GPU to generate code. CMake merely consumes the resulting tree and globs its sources. [Generator:17](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/caspar_generate.py:17), [output call:564](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/caspar_generate.py:564), [CMake:106](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/CMakeLists.txt:106)

It needs a **build-only** Python/Symforce environment, not a runtime dependency. Pin Python, Symforce 0.12.0, its transitive dependencies, and the formatter. The existing environment uses Python 3.11; I identified clang-format **21.1.2** and used it through stdin for the comparisons. Formatting depends on the output path’s ancestor `.clang-format`. The current generator manifest and lock are gitignored. [Manifest](/home/pablo/0Dev/forks/colmap-caspar-fisheye/pixi.toml:1), [formatter behavior:17](/home/pablo/0Dev/forks/colmap-caspar-fisheye/.pixi/envs/default/lib/python3.11/site-packages/symforce/codegen/format_util.py:17)

Use a committed Pixi tool environment or packaged build toolchain; adding an unavailable conda package name to rattler’s build requirements is not enough. The saved baseline runs took **15.5 seconds for fp32 and 11.7 seconds for fp64**. Full fisheye generation was not separately timed in the evidence I inspected. CUDA compilation remains required either way. [fp32 timing](/tmp/gen_base_f32.log:3), [fp64 timing](/tmp/gen_base_f64.log:3)

Pre-undistortion can remove all COLMAP changes, **but preserves rays rather than the optimization objective**. Unproject each measured fisheye observation, project that ray through a fixed virtual PINHOLE camera, solve a temporary reconstruction, and copy back poses and points. Preserve the original observations for filtering, evaluation, and Ceres polish. All three intrinsic-refinement flags are already false in this pipeline. [Pipeline options:853](/home/pablo/0Dev/forks/pyCuSFM/.claude/worktrees/agent-a11c3b12911104e01/colsfm/mapping.py:853)

For a zero-noise consistent problem, this preserves the zero-residual solution. With noise, it changes residual weighting. Even for zero-coefficient fisheye with the same focal length, radial error is multiplied by `sec²(theta)` and tangential error by `tan(theta)/theta`. At 60°, radial squared-error weight becomes **16×**. Near 90°, the pinhole coordinates become badly conditioned. A single focal length or loss threshold cannot correct that spatially varying, anisotropic change.

Consequently, pre-undistortion may be close enough for restricted fields of view and good initial poses, particularly with native Ceres polish. It is **not established as close enough for this wide-angle pipeline**. Evaluate it in original fisheye pixels on the same observations. Also, `k1…k4=0` is equidistant fisheye, not pinhole.

A useful Hypothesis suite would assert:

- **Identical problem:** clone the same observations, topology, initial state, variable blocks, and gauge. Use Ceres’s `TRIVIAL` loss for solver parity: CASPAR ignores Ceres robust-loss settings, while the pipeline normally requests Cauchy. Test pipeline behavior separately. [Loss handling](/home/pablo/0Dev/forks/pyCuSFM/.claude/worktrees/agent-a11c3b12911104e01/colsfm/mapping.py:124)
- **Exact coverage:** `num_residuals == 2 × eligible observations`, with no missing model, dropped track, or changed ID. Account explicitly for fully constant factors and minimum track length. Assert finite projections before summation; never use `nansum`. [Count implementation:863](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/colmap/estimators/bundle_adjustment_caspar.cc:863)
- **Projection/Jacobian agreement:** sample unequal `fx/fy`, off-center principal points, individual nonzero coefficients, axis limits, peripheral rays, and varying depths. Compare directional derivatives using matching pose perturbation conventions. Treat zero/negative depth as explicit expected-domain tests.
- **Actual improvement:** perturbed, solvable cases must reduce independently computed pixel error and move the intended variables. Frozen intrinsics, poses, and points must remain fixed. Exercise factor counts around **1024-thread boundaries**, including 1023/1024/1025, and repeat cases to expose the padding hazard.
- **Geometric agreement:** compare camera centers and geodesic rotation error after a common, fully constrained gauge. One fixed monocular pose leaves scale free in this backend. Include all 15 variants if claiming general adapter support; otherwise explicitly restrict the supported contract.
- **Bounded error:** use an absolute floor plus a relative bound, such as `RMSE_CASPAR ≤ 1.05 × RMSE_Ceres + δ`. A ratio alone fails when Ceres reaches nearly zero.

For normalized, well-conditioned synthetic scenes, **δ≈0.001 px**, center error around **10⁻⁴ of scene extent**, and rotation error around **0.002°** are reasonable initial test targets, informed by the recorded probe results—not universal fp32 guarantees. Larger recorded cases already need roughly a **0.02 px** floor. A 5% or 10% allowance is an engineering acceptance budget; fp32 machine epsilon does not justify it by itself. Conditioning, scene scale, the depth epsilon, fast math, and solver stopping rules all matter. [Recorded accuracy](/home/pablo/0Dev/forks/pyCuSFM/.claude/worktrees/agent-a11c3b12911104e01/docs/caspar-fisheye-adapter.md:263), [fast-math build flag](/home/pablo/0Dev/forks/colmap-caspar-fisheye/src/thirdparty/Symforce-Caspar/generated/f32/CMakeLists.txt:24)

**Recommendation: handwritten patch plus build-time generation.** Generate into a staging directory, then install only the fisheye files and four shared aggregates per precision, preserving stock kernels and CMake. Record output hashes and test independent generation runs. Make the initial-score fix explicit, and fix the padding-read defect at its generator source.

Budget **about 40–55 KB** for the source patch, generation control, and focused tests, versus 6.3 MB today. This reduces carried source by roughly two orders of magnitude while retaining the native pixel objective. The main risks are reproducible toolchain packaging, untested refinement variants/fp64, and inherited CASPAR numerical limitations. Keep pre-undistortion as a measured alternative, not an assumed equivalent.
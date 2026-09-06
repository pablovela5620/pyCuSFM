# pycolmap-caspar-fisheye

pycolmap 4.2.0 built from upstream COLMAP `be5e2916` (tag 4.2.0) plus the
patches below, by the `pixi-build-rattler-build` backend. Consumed only by the
`colsfm-caspar-fisheye` environment.

Stock CASPAR — COLMAP's GPU bundle adjuster — skips every OPENCV_FISHEYE
observation with "unsupported camera model". This build solves them natively,
in fisheye pixels. See `docs/caspar-fisheye-adapter.md`.

## Upstream

- Repo: <https://github.com/colmap/colmap>, pinned
  `be5e29168d4aff238409d60424812df66aac919f`
- Patches cut from fork `/home/pablo/0Dev/forks/colmap-caspar-fisheye`,
  branch `caspar-opencv-fisheye`, commit
  `c03b1f388e8872196e84ba08297a1c67b7d8b526`

## Patches

| Patch | Purpose |
|---|---|
| `add-opencv-fisheye-caspar-adapter.patch` | Hand-written, 4 files: `OpenCVFisheyeAdapter` and its sizing/factory in `caspar_model_adapter.h`, the dispatch in `bundle_adjustment_caspar.cc`, the `opencv_fisheye` residual **and the SumStore padding-thread fix** in `Symforce-Caspar/caspar_generate.py`, one FAQ line |
| `add-opencv-fisheye-generated-f32.patch` | Symforce output, 228 files: 224 new fp32 fisheye kernels plus the 4 regenerated aggregates (`solver.{h,cc}`, `caspar_mappings.{h,cu}`). Not hand-editable |
| `SHA256SUMS` | One hash per patched file, checked by the recipe before it compiles anything |

The generated CUDA travels inside the patch, so symforce is not a build
dependency. The vendored PINHOLE and SIMPLE_RADIAL kernels are byte-identical
to upstream; only files this branch adds, plus the four shared aggregates, are
replaced. The fork's fp64 tree is deliberately not shipped: no package compiles
it, and it would add 3.3 MiB.

## Two fixes that are easy to miss

- **Padding-thread read.** symforce 0.12.0 emits the `SumStore` call outside the
  `global_thread_idx < problem_size` guard while its data argument is assigned
  inside it, so inactive threads read an uninitialised register (upstream COLMAP
  PR #4611). The fix is in the generator, not in the emitted `.cu` files:
  `caspar_generate.py` replaces `AddSum.write_template` so the argument is
  evaluated conditionally. All 58 `SumStore` calls in the fp32 fisheye kernels
  are guarded.
- **`initial_score`.** Regenerating `solver.cc` also brings in
  `result.initial_score = score_best`, which the solver vendored in COLMAP 4.2.0
  never assigns even though `CasparBundleAdjustmentSummary::Create` reads it.
  This is wanted, and both ends now say so in a comment. Initial scores from
  this build are therefore not comparable with a stock CASPAR build's, where the
  number is indeterminate.

## Regenerating

1. In the fork: edit, run the generator, commit on `caspar-opencv-fisheye`
   (any number of commits, as long as `be5e2916` stays the merge base). The
   generator environment is `generator/pixi.{toml,lock}` here — copy both to the
   fork root (they are gitignored there) and read the header comment.
2. `bash tools/caspar_fisheye_patch.sh regen`. It refuses a dirty fork or a
   wrong base, regenerates fp32 and requires the committed tree to match byte
   for byte, cuts both patches and `SHA256SUMS`, and proves that applying them
   to a pristine `be5e2916` worktree reproduces the fork commit.
3. If the per-tree fisheye file count changed, edit the `-eq 224` line in the
   recipe's Step 0.
4. Update the fork SHA above and in `docs/caspar-fisheye-adapter.md`.
   `pixi install -e colsfm-caspar-fisheye` rebuilds, because `patches/**/*` is in
   `extra-input-globs`.
5. `pixi run -e colsfm-caspar-fisheye caspar-fisheye-test` and
   `caspar-fisheye-probe`.

Without the fork: `bash tools/caspar_fisheye_patch.sh check` clones upstream at
the pinned revision, applies the committed patches and verifies every hash.

`git diff` prints "Binary files differ" for the generated patch — `.gitattributes`
marks it `-diff`. Read it with `git diff --text`.

## Why this differs from the asmk recipe in examples-monorepo

Same shape — `source: git:/rev:/patches:` with commented `.patch` files — with
three deviations, all of them because the generated patch is 3 MiB of CUDA that
no human reviews:

- The backend is pinned to `0.4.6` rather than `"*"`. It bundles flickzeug
  0.5.2, whose "already applied" heuristic can skip a hunk with a *warning* and
  exit 0 (fixed upstream in flickzeug 0.5.3/0.5.4, not yet on conda-forge). The
  pin makes the applier a known quantity.
- `SHA256SUMS` plus the recipe's Step 0 turn that warning into a build failure.
  Nothing else proves a 17,240-deletion patch landed completely.
- The patches are cut by a script, not hand-authored, and
  `extra-input-globs = ["patches/**/*"]` makes an edit to any of them rebuild.

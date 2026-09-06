# pixi-build packaging of the CASPAR pycolmap variants — design (Opus workflow, 2026-09-05)

Produced by an 11-agent workflow: 4 readers (pixi-build docs, our recipes, the examples-monorepo asmk convention, the fork diff), 3 designs, 2 judges, synthesis, critic. The Codex adversarial review of the adapter patch (docs/reviews/2026-09-05-codex-fisheye-patch-review.md) argues for build-time generation instead of committing generated patches; symforce is not on conda-forge (checked 2026-09-05), which is why committed generated patches remain the reproducible option. The decision is recorded in NOTES.md when taken.

---

# Design: pixi-build packaging of the CASPAR pycolmap variants, fisheye adapter as committed patches

Repo: `/home/pablo/.paseo/worktrees/2unuhqu0/grey-lionfish`. Facts below marked **(verified)** were re-checked on this machine on 2026-09-05 against the fork at `267dd0fd` and the four installed `colsfm*` environments.

## 1. Decision

Keep three sibling `pixi-build-rattler-build` packages (`packages/pycolmap-caspar{,64,-fisheye}`), each consumed by one environment in its own solve group, as today. The one structural change is in `packages/pycolmap-caspar-fisheye/recipe.yaml`: replace `source: path: /home/pablo/0Dev/forks/colmap-caspar-fisheye` with the same pinned upstream source the siblings use (`https://github.com/colmap/colmap.git` @ `be5e29168d4aff238409d60424812df66aac919f`, tag 4.2.0) plus rattler-build's native `source.patches:` listing three committed files in `packages/pycolmap-caspar-fisheye/patches/`: one hand-written adapter patch (4 files, 38,659 B, reviewable) and two Symforce-generated kernel-tree patches (f32 3,149,146 B; f64 3,336,904 B). This is the asmk pattern from rerun-io/examples-monorepo and the only patch mechanism the pixi-build docs confirm (pixi#6918 is open; the cmake/python backends have no hook). The patches are produced by `tools/caspar_fisheye_patch.sh`, which reads the base rev from the recipe, refuses a dirty fork, and proves `git apply --check` against a pristine `be5e2916` tree. The fisheye build gains two real proofs that the fork was built: a Step 0 source assertion before cmake, and a `tests:` grep for `OpenCVFisheyeAdapter` in the extension module (0 hits in every stock build, 2 in the fisheye build, **verified**). Hardening that rides along: backend pinned to `0.4.6`, `extra-input-globs = ["patches/**/*"]`, `requires-pixi = ">=0.77"`, `*.patch -text` in `.gitattributes`. No `[workspace.build-variants]`, no shared recipe template, no feature composition: the docs confirm no way for an environment to select a custom variant key, and each build costs about 5.5 min on 32 cores, so the duplication is cheaper than any mechanism that removes it.

## 2. Directory layout

```
pixi.toml                                   # + requires-pixi; + default task caspar-fisheye-patch; comments
.gitattributes                              # + 2 lines (see §3)
packages/
  pycolmap-caspar/
    pixi.toml                               # backend pinned 0.4.6 (was "*")           [step 5]
    recipe.yaml                             # unchanged
  pycolmap-caspar64/
    pixi.toml                               # backend pinned 0.4.6                      [step 5]
    recipe.yaml                             # unchanged
  pycolmap-caspar-fisheye/
    pixi.toml                               # backend pinned 0.4.6 + extra-input-globs
    recipe.yaml                             # source: git+rev+patches; Step 0; real test
    README.md                               # NEW: Patch|Purpose table, SHAs, regen steps
    patches/
      add-opencv-fisheye-caspar-adapter.patch     # NEW  4 files, 644+/4-, hand-written
      add-opencv-fisheye-generated-f32.patch      # NEW  228 files under generated/f32
      add-opencv-fisheye-generated-f64.patch      # NEW  228 files under generated/f64
tools/
  caspar_fisheye_patch.sh                   # NEW: regen | check
docs/
  caspar-fisheye-adapter.md                 # edit §1 table rows, lines 7-8, §4 reproduce
  caspar-build.md                           # note backend pin
```

The absolute fork path appears only in the README, the patch headers, and the script's `CASPAR_FORK` default. The fork's `.gitignore` hunk (+5, hides its generator env) is excluded from the patches on purpose.

## 3. Manifests

### 3.1 Workspace `pixi.toml`, lines 1-10 become

```toml
[workspace]
name = "pycusfm"
version = "0.1.8"
description = "Pixi environment for the CUDA 13 pyCuSFM binaries"
channels = ["conda-forge"]
platforms = ["linux-64"]
# `packages/pycolmap-caspar*` are local pixi-build packages (rattler-build
# recipes compiled on demand), each consumed by exactly one environment.
# Path dependencies on a pixi-build package need this preview flag.
preview = ["pixi-build"]
# 0.77.1 stopped mtime-only source changes from rebuilding source packages
# (pixi #6703); a git checkout touches every mtime. Do not go below it.
requires-pixi = ">=0.77"
```

### 3.2 Workspace `pixi.toml`, default `[tasks]` table (line 471), add

```toml
# Maintainer-only. Lives in the DEFAULT feature on purpose: putting it in
# colsfm-caspar-fisheye would install (build) that env with the stale patch
# before regenerating it. Runs equally well as `bash tools/caspar_fisheye_patch.sh`.
caspar-fisheye-patch = { cmd = "bash tools/caspar_fisheye_patch.sh regen", description = "Re-cut packages/pycolmap-caspar-fisheye/patches/*.patch from the local colmap-caspar-fisheye fork and apply-check them against the pinned upstream rev" }
caspar-fisheye-patch-check = { cmd = "bash tools/caspar_fisheye_patch.sh check", description = "Apply-check the committed fisheye patches against a fresh blob-less clone of upstream COLMAP at the pinned rev (no fork needed)" }
```

### 3.3 Workspace `pixi.toml`, fisheye feature comment block (lines ~399-415), replace the prose with

```toml
# ── colsfm-caspar-fisheye: fp32 CASPAR + OPENCV_FISHEYE adapter ─────────────
# pycolmap comes from `packages/pycolmap-caspar-fisheye`, which builds upstream
# COLMAP 4.2.0 (be5e2916) with the three committed patches in
# packages/pycolmap-caspar-fisheye/patches/ applied by rattler-build. Stock
# CASPAR skips OPENCV_FISHEYE observations ("unsupported camera model"); this
# build solves them. See docs/caspar-fisheye-adapter.md and the package README.
#
# A fourth duplicated dependency list, for the same reason the second and third
# exist: each `pycolmap` spec has to be alone in its own solve, and `colsfm`
# (conda-forge pycolmap) must never be re-solved. Keep all four in sync by hand.
```

The dependency line `pycolmap = { path = "packages/pycolmap-caspar-fisheye" }` (line 415), the 26 other entries, pypi tables, activation, tasks and `[environments]` are unchanged.

### 3.4 `packages/pycolmap-caspar-fisheye/pixi.toml` (whole file)

```toml
[package]
# Named `pycolmap` so it satisfies colsfm's `import pycolmap` and shadows the
# conda-forge pycolmap inside the `colsfm-caspar-fisheye` environment only.
# The version matches the upstream COLMAP tag the patches apply to.
name = "pycolmap"
version = "4.2.0"

[package.build]
# Pinned: pixi.lock records no backend version and the local cache holds both
# 0.4.4 (pixi-build-api-version 5) and 0.4.6 (api-version 7). Bump on purpose.
backend = { name = "pixi-build-rattler-build", version = "0.4.6" }

[package.build.config]
# The patches are outside the recipe's default input globs; list them so an
# edit to a patch invalidates the cached build (the backend doc recommends this).
extra-input-globs = ["patches/**/*"]
```

`packages/pycolmap-caspar/pixi.toml` and `packages/pycolmap-caspar64/pixi.toml`: change only line 9 to `backend = { name = "pixi-build-rattler-build", version = "0.4.6" }` (step 5 of the migration). No `extra-input-globs` (no `patches/`).

### 3.5 `packages/pycolmap-caspar-fisheye/recipe.yaml` (whole file)

Lines marked `# (base)` must stay byte-identical to `packages/pycolmap-caspar/recipe.yaml`; the requirements block is copied from it verbatim including comments.

```yaml
# rattler-build recipe for a CASPAR pycolmap 4.2.0 with an OPENCV_FISHEYE
# Caspar adapter (CUDA 13 / sm_120).
#
# This is packages/pycolmap-caspar/recipe.yaml plus three patches on the source,
# a Step 0 source assertion, a distinct build string, and one real test. The
# two-step build, the CUDA 13 / sm_120 pins and the dependency list are
# byte-identical to the base recipe (its header explains why). Diff the two files
# before editing either.
#
# THE PATCHES (packages/pycolmap-caspar-fisheye/patches/, applied by
# rattler-build to the fresh upstream clone, in list order):
#   add-opencv-fisheye-caspar-adapter.patch  OpenCVFisheyeAdapter in
#       caspar_model_adapter.h, its dispatch in bundle_adjustment_caspar.cc, the
#       opencv_fisheye residual in Symforce-Caspar/caspar_generate.py, doc/faq.rst.
#   add-opencv-fisheye-generated-f32.patch   224 new kernel files + 4 regenerated
#       aggregates (caspar_mappings.{cu,h}, solver.{cc,h}) under generated/f32.
#   add-opencv-fisheye-generated-f64.patch   same for generated/f64 (not compiled
#       by this package; kept so the tree equals the fork commit).
# The generated CUDA is inside the patches, so symforce is NOT a build dependency.
# Cut from fork commit 267dd0fd (see README.md) with tools/caspar_fisheye_patch.sh.
#
# BUILD COST: 712 CUDA translation units in generated/f32 vs 488 stock; measured
# 5 min 25 s wall (75 CPU-min) on 32 threads, docs/caspar-fisheye-adapter.md §4.
context:
  # tag 4.2.0 -> be5e29168d4aff238409d60424812df66aac919f.
  # tools/caspar_fisheye_patch.sh reads this line as the patch base.
  colmap_rev: be5e29168d4aff238409d60424812df66aac919f
  cuda_arch: "120"

package:
  name: pycolmap
  version: "4.2.0"

source:
  git: https://github.com/colmap/colmap.git
  rev: ${{ colmap_rev }}
  # Paths resolve relative to this recipe directory. Each file starts with a
  # `# Patch:` prose header (git apply and GNU patch skip text before the first
  # `diff --git`). Order: hand-written first, then generated trees (disjoint paths).
  patches:
    - patches/add-opencv-fisheye-caspar-adapter.patch
    - patches/add-opencv-fisheye-generated-f32.patch
    - patches/add-opencv-fisheye-generated-f64.patch

build:
  number: 0
  # Distinguishes this artifact from the fp32/fp64 stock-source builds, which
  # share the name and version `pycolmap 4.2.0`.
  string: caspar_fisheye_cuda130_py312
  script:
    # --- Step 0: prove the patches landed before spending 5 min compiling ---
    # No Python symbol distinguishes the adapter build from upstream
    # (CameraModelId.OPENCV_FISHEYE exists in stock pycolmap), so assert on the
    # patched tree. The 224 new files per tree are 104 kernel_OpenCVFisheye* plus
    # 120 kernel_opencv_fisheye_*, hence the case-insensitive count.
    - grep -q "class OpenCVFisheyeAdapter" src/colmap/estimators/caspar/caspar_model_adapter.h
    - grep -q "opencv_fisheye" src/thirdparty/Symforce-Caspar/caspar_generate.py
    - test "$(ls src/thirdparty/Symforce-Caspar/generated/f32 | grep -ci fisheye)" -eq 224
    - test "$(ls src/thirdparty/Symforce-Caspar/generated/f64 | grep -ci fisheye)" -eq 224
    # conda's onnxruntime-cpp nests its headers at include/onnxruntime/core/session/,   # (base)
    # but COLMAP includes <onnxruntime_cxx_api.h> flat (the prebuilt-release layout).  # (base)
    # Put the session dir on the compiler's include search path so it resolves.        # (base)
    - export CPATH="$PREFIX/include/onnxruntime/core/session${CPATH:+:$CPATH}"
    # --- Step 1: build + install libcolmap WITH caspar (the flag lives here) ---
    - >-
      cmake -S . -B build_colmap -GNinja
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=$PREFIX
      -DCUDA_ENABLED=ON
      -DCASPAR_ENABLED=ON
      -DGUI_ENABLED=OFF
      -DTESTS_ENABLED=OFF
      -DFETCH_ONNX=OFF
      -DCMAKE_CUDA_ARCHITECTURES=${{ cuda_arch }}
      -DCMAKE_PREFIX_PATH=$PREFIX
    - cmake --build build_colmap --parallel ${{ CPU_COUNT }}
    - cmake --install build_colmap
    # --- Step 2: build the pycolmap bindings against our caspar libcolmap ---
    # GENERATE_STUBS=OFF keeps pybind11-stubgen (a git dependency of the
    # pyproject build-system table) out of the picture under --no-build-isolation.
    - export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"
    # scikit-build-core drives the bindings through Unix Makefiles and, unlike
    # step 1's `cmake --build --parallel`, defaults to a SINGLE job.  ~120
    # pybind11 translation units at that rate cost ~25 minutes; this cuts it to
    # about one.  scikit-build-core reads CMAKE_BUILD_PARALLEL_LEVEL from the
    # environment.
    - export CMAKE_BUILD_PARALLEL_LEVEL=${{ CPU_COUNT }}
    - export CMAKE_ARGS="-DCASPAR_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=${{ cuda_arch }} -DGENERATE_STUBS=OFF -DFETCH_ONNX=OFF -DCMAKE_PREFIX_PATH=$PREFIX"
    - python -m pip install . -vv --no-build-isolation --no-deps
    # --- Provenance inside the package, readable by probes and `pixi run` ---
    - mkdir -p "$PREFIX/etc/pycolmap-caspar"
    - printf 'fisheye_f32\n' > "$PREFIX/etc/pycolmap-caspar/variant"
    - printf '%s\n' "${{ colmap_rev }}" > "$PREFIX/etc/pycolmap-caspar/colmap_rev"

requirements:
  # (base) -- copy lines 79-152 of packages/pycolmap-caspar/recipe.yaml verbatim.
  build:
    - ${{ compiler('cxx') }}
    - ${{ compiler('c') }}
    - cuda-nvcc 13.0.*
    - cuda-version 13.0.*
    - cmake
    - ninja
    - make
    - git
    - libgomp
    - patchelf
  host:
    - python 3.12.*
    - pip
    - scikit-build-core
    - pybind11 >=3.0,<4
    - numpy
    - libboost-devel
    - eigen
    - ceres-solver
    - poselib
    - cgal-cpp
    - sqlite
    - flann
    - glog
    - gflags
    - freeimage
    - metis
    - gmp
    - lz4-c
    - libfaiss * *_cuda
    - openimageio
    - fmt
    - onnxruntime-cpp * *_cuda130
    - libgl-devel
    - libopengl-devel
    - libglx-devel
    - libegl-devel
    - glew
    - cuda-version 13.0.*
    - cuda-cudart-dev
    - libcurand-dev
    - libcublas-dev
    - libcusparse-dev
    - libcusolver-dev
  run:
    - python 3.12.*
    - numpy
    - libopenimageio
    - libfaiss
    - ceres-solver
    - poselib
    - onnxruntime-cpp * *_cuda130
  run_constraints:
    - __cuda >=13.0

tests:
  - script:
      - python -c "import pycolmap; assert pycolmap.has_cuda, 'built without CUDA'"
      - python -c "import pycolmap; assert hasattr(pycolmap.BundleAdjustmentOptions(), 'caspar')"
      # The two asserts above also pass on stock pycolmap. This one does not:
      # the adapter's class name survives in the extension module's symbols
      # (0 hits in the stock, fp32 and fp64 builds; 2 in this one).
      - python -c "import pathlib, pycolmap; p = pathlib.Path(pycolmap.__file__).parent; assert any(b'OpenCVFisheyeAdapter' in f.read_bytes() for f in p.glob('_core*.so')), 'OpenCVFisheyeAdapter not linked into pycolmap'"
      - test "$(cat "$PREFIX/etc/pycolmap-caspar/variant")" = fisheye_f32

about:
  homepage: https://colmap.github.io/
  license: BSD-3-Clause
  summary: pycolmap 4.2.0 (colmap be5e2916 + committed OPENCV_FISHEYE Caspar adapter patches) with CASPAR_ENABLED, CUDA 13 / sm_120.
```

Do not use `b'OpenCVFisheye'` as the discriminator: stock `_core*.so` contains it 18-27 times (`OpenCVFisheyeCameraModel`) **(verified)**. `OpenCVFisheyeAdapter` (0 vs 2) and `kernel_OpenCVFisheye` (0 vs 968) are the real ones.

### 3.6 `.gitattributes` (append to the existing file)

```
# Patch files must reach rattler-build byte-for-byte; never normalise line endings.
*.patch -text
# ~6.2 MiB of Symforce-generated CUDA; collapse in GitHub review.
packages/pycolmap-caspar-fisheye/patches/add-opencv-fisheye-generated-*.patch linguist-generated=true -diff
```

The existing `*.json filter=lfs` line does not match `.patch`; nothing else in the file interferes.

### 3.7 `packages/pycolmap-caspar-fisheye/README.md` (new)

```markdown
# pycolmap-caspar-fisheye

pycolmap 4.2.0 built from upstream COLMAP `be5e2916` (tag 4.2.0) plus the patches
below, by the `pixi-build-rattler-build` backend. Consumed only by the
`colsfm-caspar-fisheye` environment.

## Upstream
- Repo: https://github.com/colmap/colmap, pinned `be5e29168d4aff238409d60424812df66aac919f`
- Patches cut from fork `/home/pablo/0Dev/forks/colmap-caspar-fisheye`,
  branch `caspar-opencv-fisheye`, commit `267dd0fd2f30b56b8e23275d6e3ed5a54cb79359`

## Patches
| Patch | Purpose |
|---|---|
| `add-opencv-fisheye-caspar-adapter.patch` | `OpenCVFisheyeAdapter`, its dispatch, the generator residual, FAQ line (4 files, hand-written) |
| `add-opencv-fisheye-generated-f32.patch` | Symforce-generated fp32 kernels + regenerated aggregates (compiled here) |
| `add-opencv-fisheye-generated-f64.patch` | Same for fp64 (not compiled by any package yet) |

Generated CUDA travels inside the patches; symforce is not a build dependency.

## Regenerating
1. Commit in the fork (clean tree, single commit on top of `be5e2916`).
2. `pixi run caspar-fisheye-patch` (or `bash tools/caspar_fisheye_patch.sh regen`).
3. Update the fork SHA above. `pixi install -e colsfm-caspar-fisheye` rebuilds
   (the changed patch is in `extra-input-globs`).
4. `pixi run -e colsfm-caspar-fisheye caspar-fisheye-probe` and `caspar-fisheye-test`.

Without the fork: `pixi run caspar-fisheye-patch-check` apply-checks the committed
patches against a fresh upstream clone.
```

### 3.8 The variant mechanism

Three directories, one variant each, selected by the path dependency in each environment's own feature and solve group:

| dir | source | extra cmake flag | build string | env |
|---|---|---|---|---|
| `packages/pycolmap-caspar` | colmap `be5e2916` | none (fp32) | hashed `hb0f4dca_0` (unchanged) | `colsfm-caspar` |
| `packages/pycolmap-caspar64` | colmap `be5e2916` | `-DCASPAR_USE_DOUBLE=ON` (both cmake calls) | `caspar_f64_cuda130_py312` | `colsfm-caspar64` |
| `packages/pycolmap-caspar-fisheye` | colmap `be5e2916` + 3 patches | none (fp32) | `caspar_fisheye_cuda130_py312` | `colsfm-caspar-fisheye` |

Not used: `[workspace.build-variants]` (docs confirm custom keys reach the recipe as Jinja/env vars, but confirm no way for an environment to select a non-dependency variant value, and the fisheye variant is a source difference, not a flag); feature composition (allowed in principle, since the repo comments only forbid composing with `colsfm`, but it saves 27 lines per feature at the cost of a lock re-solve of all three caspar groups; not worth it now). Guard: `diff packages/pycolmap-caspar/recipe.yaml packages/pycolmap-caspar-fisheye/recipe.yaml` must show only the header, `source.patches`, `build.string`, Step 0, the provenance lines, the test, and `about.summary`.

## 4. Patch workflow

### 4.1 Generate

`tools/caspar_fisheye_patch.sh` (new, `chmod +x`):

```bash
#!/usr/bin/env bash
# regen: cut the three fisheye patches from the local fork and apply-check them.
# check: apply-check the committed patches against a fresh upstream clone (no fork needed).
set -euo pipefail
MODE="${1:-regen}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/packages/pycolmap-caspar-fisheye"
PATCHES=(add-opencv-fisheye-caspar-adapter.patch add-opencv-fisheye-generated-f32.patch add-opencv-fisheye-generated-f64.patch)
UPSTREAM=https://github.com/colmap/colmap.git

# Single source of truth for the base rev: the recipe's context.colmap_rev.
BASE="$(grep -oE '^  colmap_rev: [0-9a-f]{40}' "$PKG/recipe.yaml" | awk '{print $2}')"
[ -n "$BASE" ] || { echo "colmap_rev not found in $PKG/recipe.yaml" >&2; exit 1; }

apply_check() {  # $1 = dir containing a pristine checkout of $BASE
  git -C "$1" rev-parse HEAD | grep -qx "$BASE"
  local args=(); for p in "${PATCHES[@]}"; do args+=("$PKG/patches/$p"); done
  git -C "$1" apply --check "${args[@]}"
  echo "APPLY-CHECK: OK against $BASE"
}

case "$MODE" in
regen)
  FORK="${CASPAR_FORK:-$HOME/0Dev/forks/colmap-caspar-fisheye}"
  [ -z "$(git -C "$FORK" status --porcelain)" ] || { echo "fork has uncommitted changes; commit first" >&2; exit 1; }
  [ "$(git -C "$FORK" merge-base HEAD "$BASE")" = "$BASE" ] || { echo "fork HEAD is not based on $BASE" >&2; exit 1; }
  HEAD_SHA="$(git -C "$FORK" rev-parse HEAD)"
  BRANCH="$(git -C "$FORK" rev-parse --abbrev-ref HEAD)"
  COMMIT_LINE="$(git -C "$FORK" log -1 --format='%s | %an <%ae> | %ad' HEAD)"
  DIFF=(git -C "$FORK" diff --no-color --no-ext-diff --no-renames --full-index "$BASE" "$HEAD_SHA" --)
  mkdir -p "$PKG/patches"
  header() {  # $1 = one-line "what"
    printf '# Patch: %s\n' "$1"
    printf '# Problem: stock CASPAR 4.2.0 skips OPENCV_FISHEYE observations ("unsupported camera model").\n'
    printf '# Base:    colmap/colmap %s (tag 4.2.0)\n' "$BASE"
    printf '# Fork:    %s, branch %s @ %s\n' "$FORK" "$BRANCH" "$HEAD_SHA"
    printf '# Commit:  %s\n' "$COMMIT_LINE"
    printf '# Applied: rattler-build source.patches in packages/pycolmap-caspar-fisheye/recipe.yaml.\n'
    printf '# Regen:   tools/caspar_fisheye_patch.sh regen  -- do not hand-edit.\n\n'
  }
  { header "OpenCVFisheyeAdapter for Caspar GPU BA (caspar_model_adapter.h, bundle_adjustment_caspar.cc), the opencv_fisheye residual in caspar_generate.py, doc/faq.rst. Hand-written."
    "${DIFF[@]}" src/colmap doc/faq.rst src/thirdparty/Symforce-Caspar/caspar_generate.py
  } > "$PKG/patches/${PATCHES[0]}"
  { header "Symforce-generated fp32 OpenCVFisheye kernels and regenerated aggregates under generated/f32 (symforce 0.12.0, clang-formatted in the fork). Compiled by this package. Not hand-editable."
    "${DIFF[@]}" src/thirdparty/Symforce-Caspar/generated/f32
  } > "$PKG/patches/${PATCHES[1]}"
  { header "Symforce-generated fp64 OpenCVFisheye kernels and regenerated aggregates under generated/f64. Not compiled by any package yet. Not hand-editable."
    "${DIFF[@]}" src/thirdparty/Symforce-Caspar/generated/f64
  } > "$PKG/patches/${PATCHES[2]}"
  # Every fork change must be covered by the three path sets (only .gitignore is excluded on purpose).
  UNCOVERED="$(git -C "$FORK" diff --name-only "$BASE" "$HEAD_SHA" -- . ':!src/colmap' ':!doc/faq.rst' ':!src/thirdparty/Symforce-Caspar' ':!.gitignore')"
  [ -z "$UNCOVERED" ] || { echo "fork changes not covered by any patch:"; echo "$UNCOVERED"; exit 1; } >&2
  TMP="$(mktemp -d)"; trap 'git -C "$FORK" worktree remove --force "$TMP"' EXIT
  git -C "$FORK" worktree add --detach "$TMP" "$BASE" >/dev/null
  apply_check "$TMP"
  for p in "${PATCHES[@]}"; do printf '%10d bytes  %4d files  %s\n' "$(wc -c <"$PKG/patches/$p")" "$(grep -c '^diff --git' "$PKG/patches/$p")" "$p"; done
  echo "cut from $HEAD_SHA; update the fork SHA in $PKG/README.md"
  ;;
check)
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
  git clone -q --filter=blob:none --no-checkout "$UPSTREAM" "$TMP"
  git -C "$TMP" checkout -q "$BASE"
  apply_check "$TMP"
  ;;
*) echo "usage: $0 [regen|check]" >&2; exit 2 ;;
esac
```

Determinism: `git diff` between two commits with `--full-index --no-color --no-ext-diff` is byte-stable for a given git version; the header has no date except the fork commit's own author date, so a re-run on an unchanged fork is a no-op and does not invalidate the build cache. `--no-renames` keeps plain add/modify hunks (the fork has 448 A, 13 M, 0 renames, 0 binaries). Expected output today **(verified)**: adapter 38,659 B / 4 files; f32 3,149,146 B / 228 files; f64 3,336,904 B / 228 files; three-patch `git apply --check` with a `#` header on the first file passes on a pristine `be5e2916` worktree; GNU `patch -p1 --dry-run` also passes on both the header file and the 448 no-trailing-newline files.

### 4.2 Store

`packages/pycolmap-caspar-fisheye/patches/*.patch`, committed. `.gitattributes` per §3.6. Fork SHA recorded in the patch headers and README. The fork is provenance, not an input.

### 4.3 Apply at build time

rattler-build (via the pinned backend) clones upstream at `rev`, applies `source.patches` in list order relative to the recipe dir, then runs `build.script`. Step 0 asserts the patched tree before cmake, so a skipped or partial apply fails within the clone time rather than after a 5-minute compile. No `git apply`/`patch` in the script, no `$RECIPE_DIR`, no symforce. Fallback if rattler-build's own applier rejects the header or a 3 MiB file (unconfirmed, see §8): move the three files to the first script lines as `git apply --check "$RECIPE_DIR/patches/<p>" && git apply "$RECIPE_DIR/patches/<p>"` (mast3r pattern; `git` is already a build requirement). Patch files unchanged either way.

### 4.4 Regenerate when the fork changes

1. In the fork: edit; if `caspar_generate.py` changed, run the generator (`python src/thirdparty/Symforce-Caspar/caspar_generate.py <out> f32|f64` in the fork's gitignored symforce 0.12.0 env) and clang-format; commit on `caspar-opencv-fisheye`, keep it one commit on top of `be5e2916` (squash), or the merge-base check still passes but the header shows a different SHA than `git describe`.
2. `pixi run caspar-fisheye-patch` (or `bash tools/caspar_fisheye_patch.sh regen`). It refuses a dirty tree, a wrong base, or an uncovered path.
3. If the per-tree file count changed, edit the `-eq 224` lines in Step 0.
4. Update the SHA in `README.md`; `pixi install -e colsfm-caspar-fisheye` (rebuilds via `extra-input-globs`); `pixi run -e colsfm-caspar-fisheye caspar-fisheye-probe`; `caspar-fisheye-test`.
5. Base bump: change `colmap_rev` in the recipe, rebase the fork, rerun regen. The script reads the rev from the recipe, so they cannot drift.

## 5. Reproducibility

Pinned in the repo:
- Source: `colmap_rev: be5e2916...` (40-char) in each recipe, used by `source.git/rev`; for fisheye plus three committed patches that apply cleanly to that rev.
- Toolchain: `cuda-nvcc 13.0.*`, `cuda-version 13.0.*` (build + host), `python 3.12.*`, `pybind11 >=3.0,<4`, `cuda_arch: "120"`, `onnxruntime-cpp * *_cuda130`, `libfaiss * *_cuda`.
- Backend: `pixi-build-rattler-build 0.4.6`. The lock records no backend version (**verified**: no `backend` in `pixi.lock`); the cache holds 0.4.4 (`pixi-build-api-version >=5,<6`) and 0.4.6 (`>=7,<8`). Which one built the current artifacts is not recorded; 0.4.6 is chosen because it is the newest present and the only one whose API range matches the two most recent cache entries. Not "the version that built the artifacts".
- pixi: `requires-pixi = ">=0.77"`, `preview = ["pixi-build"]`.
- Artifact identity: `build.string` on f64 and fisheye; fp32 stays hashed (see §7).
- Consumer env: `pixi.lock` pins every conda/PyPI dep of each env and records `conda_source: pycolmap[<hash>] @ packages/pycolmap-caspar-fisheye` (lock lines 1249, 11274) with `depends:`.
- Provenance file `$CONDA_PREFIX/etc/pycolmap-caspar/{variant,colmap_rev}` in the fisheye build (and in the other two once step 5 lands, if wanted).

Not pinned, stated plainly: unversioned `requirements.build/host` entries (compilers via `compiler()`, cmake, ninja, ceres-solver, eigen, glog, poselib, boost, flann, openimageio...) are re-solved from conda-forge at build time. Since pixi 0.62.0 the lock stores no source input hash and the package record carries only `variants: target_platform` + `depends:`; nothing in the brief says the build/host env of a source package is locked. A fresh clone months later compiles identical source with the same patches but may link newer libraries: source-identical, not bit-identical. If bit-identity is wanted, add exact pins or `build-variants-files` in a separate change.

Rebuild triggers on pixi 0.77.1 (from the brief): an edit to `recipe.yaml` or to relevant parts of the package `pixi.toml` (0.49.0); a change in the package dependencies (0.50.0, 0.54.2); an edit under `patches/**/*` (only because of `extra-input-globs`); `pixi clean --build`. Not triggers: mtime-only changes (0.77.1); edits to the other two packages; edits to the fork on disk (no longer read). Unconfirmed: whether the backend version pin alone rebuilds; whether reverting a patch to identical bytes reuses the earlier artifact. Work dirs on this pixi are `.pixi/bld/pycolmap/<id>/{work,debug,output}` (**verified**; the brief's `.pixi/build/work/...` path does not exist here); four dirs today, 1.1-4.8 GB each.

Fresh-clone verification (no access to `/home/pablo/0Dev/forks`):

```bash
git clone <repo> /tmp/pycusfm-fresh && cd /tmp/pycusfm-fresh
pixi --version                                   # >= 0.77
pixi run caspar-fisheye-patch-check              # APPLY-CHECK: OK (clones upstream, no fork)
pixi install -e colsfm-caspar-fisheye            # ~5.5 min on 32 cores; log shows Step 0 passing
ls .pixi/envs/colsfm-caspar-fisheye/conda-meta/ | grep pycolmap   # pycolmap-4.2.0-caspar_fisheye_cuda130_py312.json
cat .pixi/envs/colsfm-caspar-fisheye/etc/pycolmap-caspar/variant  # fisheye_f32
pixi run -e colsfm-caspar-fisheye python -m tools.caspar_fisheye_probe --skip-galileo   # no LFS data needed
pixi run -e colsfm-caspar-fisheye caspar-fisheye-probe            # needs data/cusfm_runs/galileo_colsfm (LFS)
pixi run -e colsfm-caspar-fisheye caspar-fisheye-test
```

## 6. Migration steps

Preconditions: fork at `267dd0fd`, clean (**verified**); `pixi 0.77.1`; `gh auth status` shows `pablovela5620` available (**verified**, inactive) if step 8 is done.

1. **Add the script, tasks, and `.gitattributes` lines** (§3.2, §3.6, §4.1). Run `bash tools/caspar_fisheye_patch.sh regen`. Validate: prints `APPLY-CHECK: OK`, sizes 38,659 / 3,149,146 / 3,336,904 B (plus header bytes), 4 / 228 / 228 files; `git check-attr -a packages/pycolmap-caspar-fisheye/patches/add-opencv-fisheye-generated-f32.patch` shows `text: unset`, `linguist-generated: true`, `diff: unset`. Then `bash tools/caspar_fisheye_patch.sh check` from a clean shell passes. Commit.

2. **Rewrite the fisheye package** (§3.4, §3.5, §3.7) and the workspace header/comments (§3.1, §3.3). Validate cheaply first: `pixi lock` succeeds and the diff touches only the `colsfm-caspar-fisheye` solve group. Then `pixi install -e colsfm-caspar-fisheye 2>&1 | tee /tmp/caspar-fisheye-build.log`. Expect: patch application in the log, Step 0 passes, wall ~5.5 min on 32 cores, `tests:` pass including the `OpenCVFisheyeAdapter` grep, `conda-meta/pycolmap-4.2.0-caspar_fisheye_cuda130_py312.json` present. Confirm the discriminator by hand: `grep -a -c OpenCVFisheyeAdapter .pixi/envs/colsfm-caspar-fisheye/lib/python3.12/site-packages/pycolmap/_core*.so` is 2 and the same command on `colsfm-caspar` is 0.

3. **Prove the adapter solves OPENCV_FISHEYE on the GPU.** `pixi run -e colsfm-caspar-fisheye caspar-fisheye-probe` (`tools/caspar_fisheye_probe.py`): its four checks must pass: `fisheye-hazard` (score non-increasing over capped iterations, poses moved, no false convergence), `synthetic` (4-camera OPENCV_FISHEYE rig, CASPAR vs Ceres from bit-identical state), `galileo-pinhole` / `galileo-zero-k` / `galileo-small-k` (226-image Galileo model converted to OPENCV_FISHEYE). No "Skipping image ... with unsupported camera model" may appear. Then `pixi run -e colsfm-caspar-fisheye caspar-fisheye-test`: `tests/colsfm_probes/test_caspar_fisheye_hypothesis.py` must run, not skip (its `pytestmark` skips unless CASPAR projects OPENCV_FISHEYE, so a skip means the stock build). Galileo is the proxy; do not run RoboCap for this.

4. **Negative control (once, optional but cheap):** comment out the three `patches:` lines, `pixi install -e colsfm-caspar-fisheye`. Expect the build to fail at Step 0 (`grep -q "class OpenCVFisheyeAdapter"`) right after the clone. Restore. This is the evidence that Step 0 bites.

5. **Cache behaviour (once):** `touch packages/pycolmap-caspar-fisheye/patches/*.patch; pixi install -e colsfm-caspar-fisheye` must not rebuild. Edit one `#` header line in the adapter patch; `pixi install` must rebuild (~5.5 min). Revert with `git checkout -- packages/pycolmap-caspar-fisheye/patches`. If the header edit does not rebuild, `extra-input-globs` is not honoured on this backend; add `pixi clean --build` to §4.4 and record it in §8.

6. **Pin the backend in `packages/pycolmap-caspar/pixi.toml` and `packages/pycolmap-caspar64/pixi.toml`** (line 9). `pixi install -e colsfm-caspar && pixi install -e colsfm-caspar64`. Record whether each rebuilt (answers the open question in §8). Validate `pixi run -e colsfm-caspar caspar-probe`, `caspar-test`, and the same in `colsfm-caspar64`. If this step rebuilt anyway, add the `$PREFIX/etc/pycolmap-caspar/{variant,colmap_rev}` provenance lines (`caspar_f32` / `caspar_f64`) to both recipes in the same commit; otherwise defer them.

7. **Docs.** `docs/caspar-fisheye-adapter.md`: lines 7-8 ("Everything here is local... no published artifact") become "The fork is local and unpushed; the repo carries its full diff as three committed patches under `packages/pycolmap-caspar-fisheye/patches/`, so the package builds from any clone." §1 table: add a row for the patches; the "conda package" row gains "source: upstream be5e2916 + patches". §4: reproduce commands from §5 above. `docs/caspar-build.md` lines 23-26: mention the backend pin and `requires-pixi`. `docs/reviews/2026-09-05-codex-quality-review.md:113-115`: add a "resolved by" note pointing at Step 0 and the `OpenCVFisheyeAdapter` test.

8. **Fresh-clone check** per §5 on this machine (`/tmp`) or another fleet host with a Blackwell GPU. Optional backup: `GH_TOKEN=$(gh auth token --user pablovela5620) gh repo fork colmap/colmap --clone=false`, add the remote to the fork, push `caspar-opencv-fisheye`; record the URL in the README (see §7).

## 7. Open questions for the user

1. Keep `add-opencv-fisheye-generated-f64.patch` (3.3 MiB, compiled by no package) for 1:1 fidelity with the fork commit and a future fp64+fisheye package, or drop it (halves the committed size; Step 0's f64 count line goes too)?
2. Push `caspar-opencv-fisheye` to a personal mirror (`pablovela5620/colmap`) as the durable copy of the fork history? The patches make the build self-sufficient; the fork's git history and generator env exist only on this machine.
3. Commit the fork's generator env (`pixi.toml` + `pixi.lock`, symforce 0.12.0 + clang-format, gitignored in the fork) under `packages/pycolmap-caspar-fisheye/generator/` so a `caspar_generate.py` edit is regenerable from a fresh clone? Never run at build time.
4. Give the fp32 package an explicit `build.string: caspar_f32_cuda130_py312`? Renames `pycolmap-4.2.0-hb0f4dca_0.conda` and rebuilds `colsfm-caspar` (~5.5 min). Fold it into step 6 if that step rebuilds anyway.
5. Source-identical (this design) or bit-identical builds? The latter needs exact pins for ceres/eigen/glog/compilers in all three recipes, or `build-variants-files`.
6. Run `caspar-fisheye-patch-check` in CI or a pre-commit hook?
7. Migrate `[feature.*.system-requirements] cuda = "13.0"` (five deprecation warnings from `pixi info`) to the `platforms` form now, or separately?

## 8. What the research could not confirm from the docs

- Which patch applier the rattler-build bundled in backend 0.4.6 uses, and whether it accepts a `#` prose header, 3 MiB files, and 448 files without a trailing newline. `git apply` and GNU `patch` both accept them (**verified**); rattler-build itself is checked by migration step 2, and §4.3 gives the fallback.
- Whether `extra-input-globs` is honoured by this backend version (migration step 5 tests it).
- Whether a backend version pin alone rebuilds a cached artifact (step 6 records it).
- Whether pixi locks or records the build/host environment of a source package anywhere; the brief shows only `variants: target_platform` + `depends:` in the lock record.
- Any per-environment selection of a custom `[workspace.build-variants]` key; the docs show selection only through dependency pins.
- Whether the current artifacts were built by backend 0.4.4 or 0.4.6 (nothing records it).
- The debug work-dir path in the brief (`.pixi/build/work/<pkg>--<hash>/debug/`) does not match this pixi's on-disk layout (`.pixi/bld/pycolmap/<id>/debug/`); the docs page the brief quotes may lag 0.77.

---

# Completeness critic

# Completeness review: pixi-build packaging of the CASPAR pycolmap variants

Facts marked **(checked)** were verified on this machine today; sources are given inline.

## A. Blocks implementation

### A1. The design validates the patches with the wrong applier
- The bundled rattler-build does not call `git apply` or GNU `patch`. Backend `pixi-build-rattler-build 0.4.6` (cache dir `~/.cache/rattler/cache/backends-v0/pixi-build-rattler-build-3qHXhZjyQI8`) contains `rattler_build_core-0.2.10` and **`flickzeug-0.5.2`** **(checked, `strings` on the binary)**. `apply_patch_custom` in `crates/rattler_build_core/src/source/patch.rs` is the only applier wired into `fetch_sources` (`build.rs:150`); no fallback to a system tool exists.
- flickzeug applies with fuzz (`max_fuzz: 2`, `ignore_whitespace: true`, `similarity_threshold: 0.8`) and has an "already applied → skip with a **warning**, exit 0" heuristic. In 0.5.2 that heuristic is the reverse-round-trip one that rattler-build issue #2693 and flickzeug PRs #20/#22 (fixed in 0.5.3/0.5.4, 2026-07-28/30) describe: deletion-heavy hunks/diffs can be silently dropped. The 0.4.6 backend (built 2026-08-18) shipped with the pre-fix crate.
- Today's fork diff has 836 hunks, 0 deletion-only hunks, 0 modified files without additions **(checked)**, so the known bug shapes are absent. A future regen can introduce them, and the failure mode is a warning, not an error.
- Step 0 checks the adapter class, one string in `caspar_generate.py`, and two file counts. It does not check the 8 regenerated aggregates (`solver.{cc,h}`, `caspar_mappings.{cu,h}` × 2), which carry all 17,240 deletions. A skipped hunk there compiles wrong or fails after minutes.
- §4.1/§8 claim `git apply --check` + GNU `patch --dry-run` verify the patches. They verify a different parser.

Fix (small): have `regen` also write `patches/SHA256SUMS` for every file the three patches touch (from the fork HEAD tree), and make Step 0 run `sha256sum --quiet -c "$RECIPE_DIR/patches/SHA256SUMS"`. That proves byte-exact application regardless of fuzz or skip logic, and it makes the negative control in step 4 mandatory (also corrupt one aggregate hunk and confirm an error, not a warning). Also correct the §4.3 fallback text: there is no rattler-build-side `git apply`; the fallback is the script-level `git apply` you already describe. Record that the backend pin freezes this applier until a backend with flickzeug ≥0.5.4 reaches conda-forge (0.4.6 is the newest there as of 2026-08-18 **(checked)**).

### A2. `.pre-commit-config.yaml` will reject or rewrite the patches
`.pre-commit-config.yaml` **(checked)** runs `check-added-large-files` (default limit 500 KB; the generated patches are 3.1 and 3.3 MiB), `trailing-whitespace` (the fork diff has **1,573** context lines that are a single space — empty context lines — which the hook strips), and `end-of-file-fixer`. The design does not mention excludes. Add `exclude: ^packages/pycolmap-caspar-fisheye/patches/` (per hook or global) in the same commit as step 1. pre-commit is not installed in this worktree's hooks dir, so it will not bite here, but it is the repo's declared convention and anyone with it installed cannot commit.

### A3. Step 3's evidence lives in uncommitted files
`tests/colsfm_probes/test_caspar_fisheye_hypothesis.py` is untracked; `tools/caspar_fisheye_probe.py` has +215/−65 uncommitted (the `--skip-galileo` flag the §5 fresh-clone recipe uses exists only there); `hypothesis = ">=6.167.1,<7"` is added uncommitted to all four colsfm features plus `pixi.lock` **(checked, `git status`/`git diff`)**. The design's `caspar-fisheye-test` "must run, not skip" and the fresh-clone check assume these are on the branch. Add a step 0: commit them (or list them in scope).

### A4. The regen coverage guard has a hole
`UNCOVERED` excludes `':!src/thirdparty/Symforce-Caspar'` wholesale, but the three patches cover only `caspar_generate.py`, `generated/f32/`, `generated/f64/`. A change to `generated/CMakeLists.txt`, a new `generated/f16/`, or any other file in that directory is dropped silently. Use the exact three pathspecs as exclusions. One-line fix in step 1, but the script must not ship with it.

## B. Follow-ups (do not block, but the design should state them)

### B1. pixi-build features assumed, not confirmed
- `$PREFIX` inside `tests: script:`. The rattler-build testing page does not list test-time env vars **(checked, fetched)**. Use `python -c "import sys, pathlib; ..."` on `sys.prefix` instead, or verify in the step 2 log.
- Whether pixi-build runs the recipe `tests:` block at all. The brief says nothing; if the backend skips tests, the `OpenCVFisheyeAdapter` "proof" never runs. Confirm "Running tests" appears in the step 2 log; otherwise move the grep into the build script.
- Whether `README.md` in the package dir is part of the backend's default input globs. If yes, the README SHA edit in §4.4 step 4 costs a 5.5-min rebuild. Add to migration step 5.
- flickzeug 0.5.2's preamble skipper stops only at lines starting with `--- `, `+++ `, `@@ ` **(checked in v0.5.4 `parse.rs:305-372`; 0.5.2 has the same fn)**, so the `# Patch:` header is safe. Add a regen guard that no header line starts with those tokens.
- flickzeug 0.5.2 parses the `\ No newline at end of file` marker **(checked, `parse.rs:657-666`)**; creating a new file with it is untested here. Covered by A1's SHA256SUMS.
- flickzeug 0.5.2 ignores `new file mode` metadata (added in 0.6.0). A future executable file in the fork loses +x. Today the only 100755 path is `.gitignore` (excluded) **(checked)**. Make `regen` fail on a new 100755 file.

### B2. Patch-regeneration cases not handled
- Binary files: `git diff` without `--binary` emits a "Binary files differ" stub that applies nothing; flickzeug cannot apply binary patches. Fail `regen` if the diff contains `Binary files` or `GIT binary patch` (0 today).
- `git apply --check a b c` checks each file against the tree independently, not cumulatively; it is valid only because the three path sets are disjoint. Stronger proof: apply all three in the base worktree, `git add -A`, and `git diff --cached --quiet "$HEAD_SHA" -- . ':!.gitignore'` — tree equality with the fork commit. This is also where SHA256SUMS comes from.
- The three recipes can drift from each other: the script reads `colmap_rev` from the fisheye recipe only. Add an assertion in `check` that all three recipes carry the same rev.
- `.gitattributes -diff` makes `git show`/PR view of the generated patches say "Binary files differ"; regen reviewers need `git diff --text`. Say so in the README.

### B3. Steps that are heavier or looser than stated on a fresh clone
- `pixi run caspar-fisheye-patch-check` sits in the default feature, so it first installs the default env (pycusfm editable, `tensorrt-cu13`, rerun, ffmpeg…) to run a git-only script. Either document `bash tools/caspar_fisheye_patch.sh check` as the primary entry or give the two tasks a minimal env (`no-default-feature = true`, dep `git`).
- `requires-pixi = ">=0.77"` admits 0.77.0, which lacks the #6703 mtime fix the comment cites. Use `">=0.77.1"`.
- Step 2's "lock diff touches only the fisheye solve group" cannot fail: since 0.62 the lock holds no source hash, so a recipe-only change may produce an empty lock diff. Restate as "empty, or limited to that group".
- `.pixi/bld/pycolmap/<id>/` dirs are 1.1–4.8 GB each and the design adds rebuilds (steps 2, 4, 5, maybe 6). Add a `pixi clean --build` note after the migration.

### B4. Variants and conventions
- Left out on purpose: fp64+fisheye (patch committed, never compiled) and any non-sm_120 arch. Fine, but B2's mode/binary guards should still run over the f64 tree.
- Monorepo convention deviations (backend pinned instead of `"*"`; generator script instead of hand-authored diffs; `extra-input-globs`) are justified in the design but not in the package README, which is where the next reader looks. Add a "Why this differs from asmk" paragraph.
- Convention "every linux-only feature declares `platforms`" is N/A (workspace is `linux-64` only). Convention "no dependency may need a build step during resolution" is already violated by all three caspar packages equally; not new.

### B5. Design text to correct
- §4.3 and §8: replace "which patch applier … unconfirmed" with the facts in A1.
- §3.5 `# (base)` annotations and the `# (base) -- copy lines 79-152` note, if left in the file, break the "byte-identical to base" guard the same section imposes. Keep them in the design, not in the recipe.
#!/usr/bin/env bash
# Cut and verify the committed OPENCV_FISHEYE patches for
# packages/pycolmap-caspar-fisheye.
#
#   regen  Re-cut patches/*.patch and patches/SHA256SUMS from the local
#          colmap-caspar-fisheye fork, then prove they rebuild the fork tree on
#          top of the pinned upstream revision.  Needs the fork.
#   check  Apply the committed patches to a fresh blob-less clone of upstream
#          COLMAP at the pinned revision and verify every patched file against
#          patches/SHA256SUMS.  Needs the network, not the fork.
#
# The pinned revision lives in packages/pycolmap-caspar-fisheye/recipe.yaml
# (context.colmap_rev); this script never carries its own copy.
set -euo pipefail

MODE="${1:-regen}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/packages/pycolmap-caspar-fisheye"
ADAPTER_PATCH=add-opencv-fisheye-caspar-adapter.patch
GENERATED_PATCH=add-opencv-fisheye-generated-f32.patch
PATCHES=("$ADAPTER_PATCH" "$GENERATED_PATCH")
UPSTREAM=https://github.com/colmap/colmap.git

# Paths each patch owns.  Disjoint, and together with the two deliberate
# exclusions below they must cover every path the fork changes.
ADAPTER_PATHS=(src/colmap doc/faq.rst src/thirdparty/Symforce-Caspar/caspar_generate.py)
GENERATED_PATHS=(src/thirdparty/Symforce-Caspar/generated/f32)
# Deliberately not shipped: the fork's generator-env .gitignore hunk, and the
# fp64 kernel tree (3.3 MiB, compiled by no package -- see the package README).
EXCLUDED_PATHS=(.gitignore src/thirdparty/Symforce-Caspar/generated/f64)

die() { echo "error: $*" >&2; exit 1; }

# Single source of truth for the base revision.
BASE="$(sed -n 's/^  colmap_rev: \([0-9a-f]\{40\}\)$/\1/p' "$PKG/recipe.yaml" | head -1)"
[ -n "$BASE" ] || die "colmap_rev not found in $PKG/recipe.yaml"

# The three sibling recipes must build the same upstream revision.
for sibling in pycolmap-caspar pycolmap-caspar64; do
  other="$(sed -n 's/^  colmap_rev: \([0-9a-f]\{40\}\)$/\1/p' "$ROOT/packages/$sibling/recipe.yaml" | head -1)"
  [ "$other" = "$BASE" ] || die "packages/$sibling/recipe.yaml pins $other, not $BASE"
done

# Verify the committed patches against a checkout of $BASE at $1.  Applies them
# for real (not --check: --check tests each patch against the unpatched tree)
# and then hashes every patched file.
verify_tree() {
  local tree="$1"
  git -C "$tree" rev-parse HEAD | grep -qx "$BASE" || die "$tree is not at $BASE"
  local args=()
  local p
  for p in "${PATCHES[@]}"; do args+=("$PKG/patches/$p"); done
  git -C "$tree" apply --whitespace=nowarn "${args[@]}"
  ( cd "$tree" && sha256sum --quiet -c "$PKG/patches/SHA256SUMS" )
  echo "VERIFIED: ${#PATCHES[@]} patches apply to $BASE and reproduce every hash in SHA256SUMS"
}

case "$MODE" in
regen)
  FORK="${CASPAR_FORK:-$HOME/0Dev/forks/colmap-caspar-fisheye}"
  [ -d "$FORK/.git" ] || die "no fork at $FORK (set CASPAR_FORK)"
  [ -z "$(git -C "$FORK" status --porcelain)" ] || die "fork has uncommitted changes; commit first"
  [ "$(git -C "$FORK" merge-base HEAD "$BASE")" = "$BASE" ] || die "fork HEAD is not based on $BASE"
  HEAD_SHA="$(git -C "$FORK" rev-parse HEAD)"
  BRANCH="$(git -C "$FORK" rev-parse --abbrev-ref HEAD)"
  COMMIT_LINE="$(git -C "$FORK" log -1 --format='%s | %an <%ae> | %ad' HEAD)"

  # --- Every fork change must land in a patch or be excluded on purpose ------
  covered=()
  for p in "${ADAPTER_PATHS[@]}" "${GENERATED_PATHS[@]}" "${EXCLUDED_PATHS[@]}"; do covered+=(":!$p"); done
  UNCOVERED="$(git -C "$FORK" diff --name-only "$BASE" "$HEAD_SHA" -- . "${covered[@]}")"
  [ -z "$UNCOVERED" ] || { echo "fork changes covered by no patch:" >&2; echo "$UNCOVERED" >&2; exit 1; }

  # --- The generated tree must come out of the generator, not out of an editor
  # Regenerate fp32 with the fork's own generator environment and require the
  # committed fisheye kernels and aggregates to match byte for byte.  The output
  # directory sits inside the fork because symforce formats through
  # clang-format with -assume-filename, which finds .clang-format by walking up
  # from that path.
  if [ "${CASPAR_SKIP_GENERATE:-0}" = 1 ]; then
    echo "GENERATE: skipped (CASPAR_SKIP_GENERATE=1)"
  elif [ ! -f "$FORK/pixi.toml" ]; then
    echo "GENERATE: skipped -- no generator manifest at $FORK/pixi.toml." >&2
    echo "          Copy packages/pycolmap-caspar-fisheye/generator/pixi.{toml,lock} there to enable it." >&2
  else
    STAGE="$FORK/.pixi/gen/f32_verify"
    rm -rf "$STAGE"
    ( cd "$FORK" && pixi run --manifest-path pixi.toml -e default \
        python src/thirdparty/Symforce-Caspar/caspar_generate.py "$STAGE" f32 >/dev/null )
    mismatch=0
    while IFS= read -r f; do
      cmp -s "$STAGE/$f" "$FORK/src/thirdparty/Symforce-Caspar/generated/f32/$f" || { echo "regenerated output differs: generated/f32/$f" >&2; mismatch=1; }
    done < <( { ls "$FORK/src/thirdparty/Symforce-Caspar/generated/f32" | grep -i fisheye; \
                printf '%s\n' solver.cc solver.h caspar_mappings.cu caspar_mappings.h; } )
    rm -rf "$STAGE"
    [ "$mismatch" = 0 ] || die "the committed fp32 tree is not what the generator emits"
    echo "GENERATE: fp32 fisheye kernels and the 4 aggregates reproduce byte for byte"
  fi

  DIFF=(git -C "$FORK" diff --no-color --no-ext-diff --no-renames --full-index "$BASE" "$HEAD_SHA" --)
  mkdir -p "$PKG/patches"
  header() {  # $1 = one-line "what"
    printf '# Patch: %s\n' "$1"
    printf '# Problem: stock CASPAR 4.2.0 skips OPENCV_FISHEYE observations ("unsupported camera model").\n'
    printf '# Base:    colmap/colmap %s (tag 4.2.0)\n' "$BASE"
    printf '# Fork:    %s, branch %s @ %s\n' "$FORK" "$BRANCH" "$HEAD_SHA"
    printf '# Commit:  %s\n' "$COMMIT_LINE"
    printf '# Applied: rattler-build source.patches in packages/pycolmap-caspar-fisheye/recipe.yaml.\n'
    printf '# Checked: every patched file is hashed in patches/SHA256SUMS, verified by the build.\n'
    printf '# Regen:   tools/caspar_fisheye_patch.sh regen  -- do not hand-edit.\n\n'
  }
  { header "OpenCVFisheyeAdapter for Caspar GPU BA (caspar_model_adapter.h, bundle_adjustment_caspar.cc), the opencv_fisheye residual and the SumStore padding-thread fix in caspar_generate.py, doc/faq.rst. Hand-written."
    "${DIFF[@]}" "${ADAPTER_PATHS[@]}"
  } > "$PKG/patches/$ADAPTER_PATCH"
  { header "Symforce-generated fp32 OPENCV_FISHEYE kernels plus the 4 regenerated aggregates under generated/f32 (symforce 0.12.0, clang-format 21.1.2). Compiled by this package. Not hand-editable."
    "${DIFF[@]}" "${GENERATED_PATHS[@]}"
  } > "$PKG/patches/$GENERATED_PATCH"

  # --- Shapes flickzeug (the applier inside the backend) cannot handle -------
  for p in "${PATCHES[@]}"; do
    if grep -qE '^(Binary files|GIT binary patch)' "$PKG/patches/$p"; then die "$p contains a binary diff"; fi
    if grep -qE '^(new file mode 100755|new mode 100755)' "$PKG/patches/$p"; then die "$p adds an executable file; the applier drops the mode"; fi
    if sed -n '/^diff --git/q;p' "$PKG/patches/$p" | grep -qE '^(--- |\+\+\+ |@@ )'; then die "$p has a header line the preamble skipper would eat"; fi
  done

  # --- SHA256SUMS over every file the patches touch, from the fork HEAD tree --
  TMP="$(mktemp -d)"
  trap 'git -C "$FORK" worktree remove --force "$TMP" >/dev/null 2>&1 || true; rm -rf "$TMP"' EXIT
  git -C "$FORK" worktree add --detach "$TMP" "$BASE" >/dev/null
  git -C "$TMP" apply --whitespace=nowarn "$PKG/patches/$ADAPTER_PATCH" "$PKG/patches/$GENERATED_PATCH"
  { for p in "${PATCHES[@]}"; do grep '^+++ b/' "$PKG/patches/$p" | sed 's|^+++ b/||'; done; } \
    | LC_ALL=C sort -u > "$TMP/.patched-files"
  ( cd "$TMP" && xargs -a .patched-files sha256sum ) > "$PKG/patches/SHA256SUMS"
  rm -f "$TMP/.patched-files"

  # The patched base tree must equal the fork commit outside the exclusions.
  excl=()
  for p in "${EXCLUDED_PATHS[@]}"; do excl+=(":!$p"); done
  git -C "$TMP" add -A
  git -C "$TMP" diff --cached --quiet "$HEAD_SHA" -- . "${excl[@]}" \
    || die "the patched base tree does not equal fork $HEAD_SHA"
  ( cd "$TMP" && sha256sum --quiet -c "$PKG/patches/SHA256SUMS" )
  echo "VERIFIED: the 2 patches rebuild fork $HEAD_SHA on top of $BASE (minus ${EXCLUDED_PATHS[*]})"

  for p in "${PATCHES[@]}"; do
    printf '%10d bytes  %4d files  %s\n' "$(wc -c <"$PKG/patches/$p")" "$(grep -c '^diff --git' "$PKG/patches/$p")" "$p"
  done
  printf '%10d bytes  %4d files  %s\n' "$(wc -c <"$PKG/patches/SHA256SUMS")" "$(wc -l <"$PKG/patches/SHA256SUMS")" SHA256SUMS
  echo "cut from $HEAD_SHA; update the fork SHA in $PKG/README.md and docs/caspar-fisheye-adapter.md"
  ;;
check)
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  git clone -q --filter=blob:none --no-checkout "$UPSTREAM" "$TMP"
  git -C "$TMP" checkout -q "$BASE"
  verify_tree "$TMP"
  ;;
*)
  echo "usage: $0 [regen|check]" >&2
  exit 2
  ;;
esac

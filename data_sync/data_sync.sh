#!/usr/bin/env bash
# data_sync.sh — snapshot large local artifacts to / from GitHub Release assets.
#
# The code repo (offroad-control) tracks small files only. The large
# regenerate-or-recover artifacts -- the raw paper-supporting result
# generations and the wider rig collections -- live off-machine as tar-split
# Release assets on a private DATA repo, so they are backed up and fetchable
# anywhere (including by tooling that already has `gh` authenticated).
# Snapshots are infrequent by design; git is not bloated with every result.
#
# Usage:
#   data_sync/data_sync.sh push [TAG]         # tar+split the backup set, upload
#   data_sync/data_sync.sh pull [TAG] [DEST]  # download+reassemble+extract
#   data_sync/data_sync.sh list               # list snapshot releases
#
# TAG default: snapshot-YYYYMMDD. Backup set: data_sync/data_snapshot.list
# (paths relative to DATA_ROOT). Files >2 GB are split into <=1.9 GB parts.
#
# Env:
#   DATA_REPO  default ksha23/terrain-aware-offroad-control-data
#   DATA_ROOT  root the listed paths are relative to.
#              default: this repo's root (so `pull` restores INTO the repo).
#   GH         gh binary (default: gh on PATH, else ~/.local/bin/gh)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
DATA_REPO="${DATA_REPO:-ksha23/terrain-aware-offroad-control-data}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT}"
LIST="${LIST:-$HERE/data_snapshot.list}"
SPLIT_SIZE="${SPLIT_SIZE:-1900m}"
GH="${GH:-$(command -v gh || echo "$HOME/.local/bin/gh")}"

die() { echo "error: $*" >&2; exit 1; }
[ -x "$GH" ] || command -v "$GH" >/dev/null 2>&1 || die "gh not found ($GH)"

cmd="${1:-}"; shift || true

case "$cmd" in
  push)
    TAG="${1:-snapshot-$(date +%Y%m%d)}"
    [ -f "$LIST" ] || die "backup list not found: $LIST"
    mapfile -t PATHS < <(grep -vE '^\s*#|^\s*$' "$LIST")
    [ "${#PATHS[@]}" -gt 0 ] || die "backup list is empty"
    TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
    echo "[push] tag=$TAG repo=$DATA_REPO root=$DATA_ROOT  (${#PATHS[@]} paths)"
    # ensure the release exists (idempotent)
    "$GH" release view "$TAG" -R "$DATA_REPO" >/dev/null 2>&1 || \
      "$GH" release create "$TAG" -R "$DATA_REPO" -t "$TAG" \
            -n "Data/artifact snapshot $TAG (tar-split; restore with data_sync.sh pull $TAG)."
    : > "$TMP/CONTENTS.txt"
    for p in "${PATHS[@]}"; do
      [ -e "$DATA_ROOT/$p" ] || { echo "  [skip missing] $p" >&2; continue; }
      san="$(echo "$p" | sed 's#[/ ]#__#g')"
      echo "  [tar] $p"
      tar czf - -C "$DATA_ROOT" "$p" | split -b "$SPLIT_SIZE" -d - "$TMP/${san}.tgz."
      for part in "$TMP/${san}.tgz."*; do
        echo "  [upload] $(basename "$part") ($(du -h "$part" | cut -f1))"
        "$GH" release upload "$TAG" "$part" -R "$DATA_REPO" --clobber
      done
      echo "$p" >> "$TMP/CONTENTS.txt"
    done
    "$GH" release upload "$TAG" "$TMP/CONTENTS.txt" -R "$DATA_REPO" --clobber
    echo "[push] done -> https://github.com/$DATA_REPO/releases/tag/$TAG"
    ;;

  pull)
    TAG="${1:-}"; DEST="${2:-$DATA_ROOT}"
    [ -n "$TAG" ] || die "usage: data_sync.sh pull TAG [DEST]"
    TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
    echo "[pull] tag=$TAG repo=$DATA_REPO -> dest=$DEST"
    "$GH" release download "$TAG" -R "$DATA_REPO" -D "$TMP" --clobber
    mkdir -p "$DEST"
    # reassemble each logical tarball (unique prefix before .NN) and extract
    for prefix in $(ls "$TMP"/*.tgz.* 2>/dev/null | sed 's/\.[0-9][0-9]*$//' | sort -u); do
      echo "  [extract] $(basename "$prefix")"
      cat "$prefix".* > "$prefix"
      tar xzf "$prefix" -C "$DEST"
    done
    [ -f "$TMP/CONTENTS.txt" ] && { echo "[pull] restored:"; sed 's/^/  /' "$TMP/CONTENTS.txt"; }
    echo "[pull] done"
    ;;

  list)
    "$GH" release list -R "$DATA_REPO"
    ;;

  *)
    grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//' | head -30
    exit 1
    ;;
esac

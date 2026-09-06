#!/usr/bin/env bash
# Applies our local fixes to vendored submodules. Run once after
# `git submodule update --init --recursive` on a fresh clone, since a
# submodule's own working-tree edits aren't captured by the parent repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git -C "$REPO_ROOT/ros2_ws/src/ros2_so_arm100" apply \
    "$REPO_ROOT/patches/so_arm101-fortress.patch"

echo "Vendor patches applied."

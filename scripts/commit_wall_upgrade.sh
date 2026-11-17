#!/usr/bin/env bash
set -euo pipefail
push_each=false
if [ "${1:-}" = '--push' ]; then push_each=true; fi
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo 'Commit the starter first; see docs/WALL_UPGRADE.md.' >&2; exit 1
fi
if ! git diff --cached --quiet; then
  echo 'Finish existing staged work first.' >&2; exit 1
fi
commit_group() {
  message="$1"; shift
  git add -- "$@"
  if ! git diff --cached --quiet; then git commit -m "$message"; fi
  if "$push_each"; then git push; fi
}
commit_group 'feat: pin Wall assets and export all inference checkpoint components' \
  config/wall.release.json src/wsp_stopper/wall_setup.py pyproject.toml
commit_group 'feat: add version-checked CEM hook and local encoder loading' \
  src/wsp_stopper/cem_hook.py src/wsp_stopper/upstream_patch.py tests/test_wall_upgrade.py
commit_group 'docs: document packet freeze and Colab validation sequence' \
  README.md docs/WALL_UPGRADE.md docs/INTEGRATION.md scripts/commit_wall_upgrade.sh .gitignore

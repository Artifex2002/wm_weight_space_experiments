#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Optional argument: your EMPTY GitHub repo URL.
remote_url="${1:-}"
if [ ! -d .git ]; then git init -b main; fi
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo 'This script expects an empty repository with no commits.' >&2
  exit 1
fi
if [ -n "$remote_url" ]; then
  if git remote get-url origin >/dev/null 2>&1; then
    if [ "$(git remote get-url origin)" != "$remote_url" ]; then
      echo 'Existing origin differs; inspect it before continuing.' >&2
      exit 1
    fi
  else
    git remote add origin "$remote_url"
  fi
fi
commit_stage() {
  message="$1"
  shift
  git add -- "$@"
  git commit -m "$message"
  if [ -n "$remote_url" ]; then git push -u origin HEAD; fi
}
commit_stage 'chore: bootstrap checkpoint-inspection research package' \
  pyproject.toml .gitignore .env.example README.md src/wsp_stopper/__init__.py
commit_stage 'feat: define stopping features and validated controller inference' \
  src/wsp_stopper/core.py
commit_stage 'feat: freeze checkpoint packets and expose bounded inspection tools' \
  src/wsp_stopper/packet.py src/wsp_stopper/inspection.py scripts/inspect_checkpoint.py
commit_stage 'feat: add audited Responses API loop with one final weight submission' \
  src/wsp_stopper/agent.py src/wsp_stopper/cli.py prompts/generate.md config/spec.template.json
commit_stage 'test: cover protocol boundaries and document Wall integration' \
  tests docs scripts/init_history.sh

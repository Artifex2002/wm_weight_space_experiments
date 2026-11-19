# Incremental repository update

Prepared against remote HEAD **`2ce606163917e5f9953a5842e61cb322c722c236`** from
<https://github.com/Artifex2002/wm_weight_space_experiments>. The observed commit
message was `docs: update notebook paths and explain two-person Colab handoff`.

## What changed since that commit

| File or directory | Action | Purpose |
|---|---|---|
| `notebooks/03_wall_frozen_stopper_diagnostics.ipynb` | Add | Frozen-controller audit, fixed-depth sweep, optional held-out episodes and repeated timing; corrected fresh-VM setup and visible subprocess errors |
| `notebooks/04_wall_prompt_comparison.ipynb` | Add | Two frozen prompts, three constructions each, fresh paired tasks, per-episode resumption and descriptive summaries; corrected fingerprint inventory |
| `tests/test_notebooks.py` | Add | Parse notebooks, check identical evaluator bytes, compare source fingerprint with independent notebook-02 schema, test preflight success/rejection |
| `README.md` | Update | Explain all four notebooks, current evidence scope, resumption and source-pinning requirements |
| `docs/REPO_UPDATE_GUIDE.md` | Add | This file: inventory and incremental commit instructions |
| `docs/UPDATE_MANIFEST.json` | Add | Machine-readable inventory and hashes of supplied files |
| `docs/EXPERIMENT_LOG.md` | Add | Observations, corrections, hypotheses, limitations and next steps |
| `results/` | Add curated snapshot | Pilot controller/results, notebook-derived completed sweep, six new controller submissions and the partial prompt-study evidence |

No replacement is needed for `src/wsp_stopper/`, `config/`, `prompts/generate.md`,
the existing tests or notebooks 01/02: their current bytes already match the
observed remote. Notebook 04's alternative prompt is frozen within its study;
the original repo prompt remains unchanged.

The notebooks embed their helpers. Standalone scratch builders, extracted helper
copies, executed notebook copies and earlier starter/upgrade ZIPs are not runtime
requirements and are intentionally not added. Notebook outputs are cleared in the
executable deliveries. Keep the executed originals and full raw archives separately.

## Apply locally

The ZIP has repo-relative paths at its root. It updates README and adds the files
listed above; it contains no `.git` directory and does not replace `.gitignore`.
Review existing local edits before extraction, particularly README. Do not reset
or discard unrelated work. The extraction below intentionally replaces matching
bundle paths; preserve any conflicting local edits first.

```bash
cd /Users/ashutoshpanda/Documents/Projects/wsp-stopper

git status --short
git log -1 --oneline

unzip -l ~/Downloads/wm_weight_space_updates_2026-11-18.zip
unzip -o ~/Downloads/wm_weight_space_updates_2026-11-18.zip -d .

source .venv/bin/activate
python -m unittest discover -s tests -v
```

If a matching virtual environment is not active/installed, follow the existing
README installation instructions. The new notebook checks need only the standard
library; the existing package tests require the project's normal dependencies.
Tests do not call Astra or run a GPU evaluation. Optional pinned-upstream tests
remain skipped unless their environment variable/assets are available.

## Commit 1: executable follow-ups and operational documentation

Ensure the staging area does not already contain unrelated changes.

```bash
git add -- README.md notebooks/03_wall_frozen_stopper_diagnostics.ipynb notebooks/04_wall_prompt_comparison.ipynb tests/test_notebooks.py docs/REPO_UPDATE_GUIDE.md docs/UPDATE_MANIFEST.json
git diff --cached --stat
git commit -m "feat: add frozen stopper diagnostics and prompt comparison notebooks"
```

## Commit 2: research findings and explicit evidence snapshot

```bash
git add -- docs/EXPERIMENT_LOG.md results
git diff --cached --stat
git commit -m "research: record Wall pilot findings and partial prompt-study evidence"
git push origin main
```

This pushes both incremental commits together. You can push after the first
commit too if desired. Use your actual branch if it differs from `main`.
The manifest describes the complete two-commit bundle, so some of its entries
become tracked only in the second commit.

## Preserve the active experiment when committing

**Keep the active notebooks' `PROJECT_REF` at**
`2ce606163917e5f9953a5842e61cb322c722c236`. The new commit hashes contain follow-up
notebooks and documentation; they are not replacements for the source commit
recorded by the existing pilot/study. The notebook files can be hosted in a later
Git revision while their bootstrap checks out the pinned earlier source.

Do not change the active `STUDY_ID`, prompts, weights, task manifest, budgets,
helper source, or receipts merely because these files are now committed. Notebook
04's embedded helper matches the active study hash
`d4bd15761f3932c778b8bf17039ec033e3f2d3b1ec667b523089774efdab6ad1`.
The two new notebook copies contain the same evaluator hash as notebooks 01/02.

The latest supplied prompt-study archive has **8/96 jobs**, not a completed
comparison. Finish remaining jobs in the active Drive study, rerun summaries and
export, then add a later dated evidence/log update. Do not overwrite the current
partial snapshot without recording the new provenance and coverage.

## Evidence included and excluded

- Included intentionally: small numeric controller weights, prompts, protocols,
  task manifests, submission/execution receipts, metrics, selected traces and one
  decision-margin image. See the per-snapshot README files.
- The completed six-task sweep's raw per-depth JSONs were not in the supplied ZIP;
  its saved notebook outputs support labeled derived CSVs instead.
- Excluded: API keys, virtual environments, model/checkpoint binaries, complete
  numerical inspection packets, raw API event streams, copied third-party source,
  process logs and bulky executed notebook copies. Raw archives remain separate.
- `results/wall-prompt-study-001/packet_manifest.json` is an inventory, not a runnable
  packet. Do not point the generation runner at this curated results directory.

For a scientific summary, start with [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

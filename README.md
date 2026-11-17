# Weight-informed planning stopper

A starter research environment where an OpenAI API model can choose how to
inspect a frozen DINO-WM checkpoint and submit the 178 weights of a small CEM
stopper. No model or simulator execution is available through the agent tools.

Implemented: packet export/hashes, bounded tensor/source inspection, Responses
API function loop, budgets/audit logs, one final submission, feature extraction,
controller inference, and offline protocol tests.

Not yet implemented/validated: DINO-WM Wall planner integration, full checkpoint
loading/inference, baseline sweep, learned-stopper training, or a live API run.
The included mock test uses synthetic tensors, not a trained world model. This
is a construction harness, not a complete reproduction of the stopping paper.

## 1. Put these files into your empty GitHub repository

Unzip this project and open a terminal in `wsp-stopper/`. Check your Git author:

```bash
git --version
# Only if these are not already configured:
git config --global user.name "Your Name"
git config --global user.email "YOUR_GITHUB_COMMIT_EMAIL"
```

Create five relevant commits locally:

```bash
bash scripts/init_history.sh
```

Or create AND push each commit to your existing EMPTY GitHub repository:

```bash
bash scripts/init_history.sh https://github.com/YOUR_USERNAME/YOUR_EMPTY_REPO.git
```

Use one of these commands, not both. The remote must have no README/license/
initial commit. Authentication uses your existing Git credential manager or SSH.
Your OpenAI API key does not authenticate GitHub. If a push fails, local commits
remain intact; fix authentication and run `git push -u origin HEAD` to resume.
Do not rerun the bootstrap script after it has made commits. Make subsequent
commits with normal `git add <specific files>` and `git commit -m ...` commands.
The early staged commits show implementation history; use the final commit to run.

If you chose local commits first:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_EMPTY_REPO.git
git push -u origin HEAD
```

## 2. Install and run offline tests

Python 3.10+; Linux/macOS commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

No API key or GPU is required for those tests. Install trusted checkpoint tools
separately; this extra can be large:

```bash
python -m pip install -e '.[checkpoint]'
```

For CUDA evaluation later, choose a PyTorch build compatible with your machine.
Construction uses CPU inspection; it does not require CUDA.

## 3. Set your API credentials locally

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
wsp models
export OPENAI_MODEL='EXACT_API_MODEL_ID_AVAILABLE_TO_YOUR_ACCOUNT'
```

Enter the actual model ID after checking account access; `Astra` is not assumed
to be a valid API ID. No key is included, logged, committed, or requested in chat.
.env.example is documentation; it is not automatically loaded.

## 4. Download a pinned Wall checkpoint

Candidate: `facebook/jepa-wms`, `dino_wm_wall.pth.tar`. The model card describes
DINOv2 ViT-S/14, a six-layer predictor, 224x224 inputs; file listing is about
269 MB. It is not confirmed as the checkpoint from Knowing When to Stop Thinking,
and has not been established as the smallest complete Wall-capable release.

Open https://huggingface.co/facebook/jepa-wms/commits/main and copy a full
40-character commit SHA. Pin that revision rather than using moving `main`:

```bash
export HF_REVISION='PASTE_FULL_40_CHARACTER_COMMIT_SHA'
wsp download --revision "$HF_REVISION" --out assets
python scripts/inspect_checkpoint.py assets/dino_wm_wall.pth.tar
```

The download command records revision and SHA-256 in `assets/download.json`.
Checkpoint loading uses `weights_only=True`, with no unsafe fallback. If the
release needs unsupported pickle globals, inspect the format and convert in a
trusted preparation environment; do not enable unrestricted unpickling here.

## 5. Build and freeze the actual experiment packet

Obtain the matching source checkout from https://github.com/facebookresearch/jepa-wms
and pin a full commit. Use the original DINO-WM repo only with its matching
checkpoint/configuration. Record actual architecture and normalization from the
chosen release; do not assume configs from another paper apply.

```bash
cp config/spec.template.json config/spec.local.json
# Edit spec.local.json: complete every field from actual implementation.
# Set finite episode_limit M, planner cap 30, min depth 3, and resolve action units.
# Set ready=true only after reviewing it. No measured outcomes or known best depth.
```

Choose source files covering environment dynamics/success, preprocessing, encoder,
predictor/rollout, CEM/MPC/objective, final-plan selection, and your stopper hook.
Pass files explicitly so you can audit exactly what the model can read. Do not
include .env, evaluation logs, published depth curves, hidden episodes, or results.
Use `docs/INTEGRATION.md` to implement the hook; this integration is still needed.
The packet template intentionally fails until completed.

After inspect_checkpoint.py shows the actual flat tensor dictionary path, run:

```bash
wsp freeze \
  --checkpoint assets/dino_wm_wall.pth.tar \
  --state-key 'REPLACE_WITH_ACTUAL_STATE_DICT_KEY' \
  --spec config/spec.local.json \
  --provenance 'HF repo + full revision; source full commit; component scope' \
  --source /ABSOLUTE/PATH/TO/ACTUAL_MODEL_SOURCE.py \
  --source /ABSOLUTE/PATH/TO/ACTUAL_PLANNER_SOURCE.py \
  --source /ABSOLUTE/PATH/TO/ACTUAL_ENVIRONMENT_SOURCE.py \
  --source src/wsp_stopper/core.py \
  --out packets/wall-v1
wsp verify packets/wall-v1
```

Repeat --source for every required file/config/hook. Replace the illustrative
source paths and state key; root state dicts use `--state-key ''`. The converter
requires a flat dict of numeric tensors. See PROTOCOL.md for combining separately
stored visual encoder/predictor components. Tensor coverage must be documented:
providing just the predictor is partial world-model access. The optional decoder
is unnecessary for latent planning. No real checkpoint is included in this zip.

## 6. Generate once, then freeze before evaluation

```bash
wsp generate \
  --packet packets/wall-v1 \
  --out runs/trial-001 \
  --max-calls 40 --max-chars 200000 --max-turns 45 --output-tokens 8000
```

Optionally add `--reasoning high` if supported by your exact model. Every call is
billable; budgets bound requests/output, not dollars. There is no automatic retry.
An incomplete response is a failed attempt, not silently repaired.

Outputs: run.json (model/config/hashes), prompt.txt, events.jsonl (responses and
all inspection calls), submitted_raw.json, submission.json, and weights.json
only when valid. The runner validates all 178 scalar values and ends at the first
submission, whether valid or not. It never evaluates a generated controller.
Prompt and tools implement our proposed protocol, not an exact Experiment 5 clone.

## 7. Integrate and evaluate separately

Read docs/INTEGRATION.md and docs/PROTOCOL.md. First add the real planner hook,
then freeze it before generation. Load weights with:

```python
import json
from wsp_stopper.core import Stopper
controller = Stopper(json.load(open('runs/trial-001/weights.json')))
```

Run baseline/controller evaluations in a separate process after construction,
with held-out episodes unavailable to the generator. Keep all attempts and do
not feed test performance back into the next prompt. The pretrained checkpoint's
overoptimization peak must be established empirically; iteration 11 is not a
guarantee for this release.

Suggested subsequent commits:

```bash
git add docs/INTEGRATION.md path/to/your/adapter.py
git commit -m "feat: integrate per-instance stopping into pinned Wall planner"
git push
# After a real evaluation harness exists:
git add path/to/evaluation_code.py docs/evaluation_protocol.md
git commit -m "feat: evaluate fixed and adaptive budgets on held-out Wall episodes"
git push
```

Use only real filenames in these future commands. The paths above are examples.
Never commit API keys, checkpoint binaries, hidden test pairs or unreviewed logs.

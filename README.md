# Weight-informed stopping for DINO-WM Wall planning

This project asks whether an LLM can inspect a frozen world model's implementation
and numerical weights, then directly write the weights of a small planning
stopper without training, world-model forward passes, simulator rollouts, or
performance feedback during construction.

The first experiment compares:

1. **Fixed-budget baseline:** ordinary CEM, normally 30 refinement iterations.
2. **WSP controller:** the same CEM planner, with a small network deciding when
   to stop after a completed refinement iteration.

The network is 8 inputs → 16 ReLU units → 2 logits, with **178 parameters**. STOP
returns the planner's current proposal mean. It does not terminate the task or
choose a different candidate. Earliest stop is iteration 3; the cap is 30.

This repository uses Meta's released `dino_wm_wall` baseline checkpoint at epoch
50. It is **not verified as the checkpoint used in Knowing When to Stop Thinking**.
A published peak at a particular iteration is neither an expected result nor a
controller instruction. Establish the local success–compute curve empirically.

## 1. Repository structure and how it comes together

```text
wsp-stopper/
├── 01_wall_baseline.ipynb
├── 02_wall_wsp_comparison.ipynb
├── README.md
├── pyproject.toml
├── config/
│   ├── wall.release.json
│   └── spec.template.json
├── prompts/
│   └── generate.md
├── src/wsp_stopper/
│   ├── __init__.py
│   ├── core.py
│   ├── packet.py
│   ├── inspection.py
│   ├── agent.py
│   ├── cli.py
│   ├── wall_setup.py
│   ├── upstream_patch.py
│   └── cem_hook.py
├── scripts/
│   ├── inspect_checkpoint.py
│   ├── init_history.sh
│   ├── commit_wall_upgrade.sh
│   └── restore_missing.py
├── tests/
│   ├── test_system.py
│   └── test_wall_upgrade.py
└── docs/
    ├── PROTOCOL.md
    ├── INTEGRATION.md
    ├── WALL_UPGRADE.md
    └── REPAIR_INSTALL.md
```

Assets, packets and runs are generated artifacts, not source files to commit.
The complete starter and Wall upgrade must already be present; the notebooks
check that required modules exist before running. These three new files do not
replace the Python package.

### Python modules

| File | Responsibility |
|---|---|
| `core.py` | SHA utilities, JSON handling, weight validation, controller inference, and eight planner features |
| `wall_setup.py` | Clone pinned source, download/check assets, combine component state dictionaries, patch upstream, create configs and inspection packet |
| `upstream_patch.py` | Apply version/hash-checked CEM hook and local DINOv2 loading edits; preserve original files |
| `cem_hook.py` | Reset feature history for each solve, query the network after refinement, and collect its decision trace |
| `packet.py` | Export numeric tensors and approved source files; freeze/verify an integrity manifest |
| `inspection.py` | The only weight/source inspection functions exposed to the API model |
| `agent.py` | Responses API function-call loop, budgets, audit logging, and one final weight submission |
| `cli.py` | Commands `wsp models`, `download`, `freeze`, `verify`, and `generate` |

### Checkpoint and architecture

`config/wall.release.json` pins the following assets and SHA-256 checksums:

| Asset | Identity |
|---|---|
| Meta implementation | `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` |
| Wall checkpoint | `facebook/jepa-wms`, revision `9b9c41ef249466630dbf1a20e78391865d07b3b9`, `dino_wm_wall.pth.tar`, epoch 50 |
| Visual backbone | Official DINOv2 ViT-S/14 pretrained weights |
| DINOv2 implementation | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |

The Wall checkpoint contains predictor, action-encoder and proprioception-encoder
weights. The visual encoder is a separate download. Preparation exports all four
component state maps, including buffers: **254 tensor entries**. Optimizer state,
scaler state and optional visualization decoders are excluded from inspection.
No training or model-weight changes happen during export.

The narrow Colab loader constructs the matching modules and checks learned tensor
keys/shapes. Unexpected keys or missing learned weights fail the run; only known
regenerated causal mask buffers may be absent. No random-weight fallback exists.

### Planner and task semantics used by the notebooks

| Setting | Value |
|---|---|
| Initial pilot subset | 6 saved start/goal pairs; fixed layout, travel through the opening |
| CEM population / elites | 300 / 10 |
| Horizon / returned chunk | 6 / 6 planner actions |
| Raw action dimensions | 2 |
| Packed planner action dimensions | 10: five consecutive 2D raw actions |
| Maximum simulator steps | 30 |
| Planning solves per episode | 1 in this configuration |
| Baseline refinement budget | 30; optional predeclared sweep in notebook 01 |
| Initial mean / standard deviation | Zero / one |
| Returned plan | Updated proposal mean |
| Candidate objective | Terminal visual-feature MSE + 0.1 × proprioception-feature MSE |
| Model mode | Frozen weights, eval mode, no dropout sampling |
| Image processing | Resize to 224×224; scale by 255; mean/std 0.5 |
| Predictor context window | 2 |
| Primary success metric | Final position is within 4.5 units of the goal after the action chunk |
| Additional logged metric | Whether the agent reached the success radius at any intermediate step |

One solve per episode is deliberate for the initial release-matched horizon and
time limit. It still permits adaptive stopping inside CEM. A longer episode with
multiple MPC replans is a separate experiment requiring a new specification and
controller construction. Do not change the execution convention after generation.

### Runtime adapter embedded in the notebooks

Both notebooks embed **the same Python runner as a code-cell string**. The cell
writes it to `/content/wsp_wall_notebook_runtime.py`. You only need to add the two
notebooks and this README to the existing repository; no additional local helper
file is required.

The runner:

- Builds the real pinned Wall simulator directly, and uses the upstream random
  state generator to save explicit start/goal coordinates and layout values.
- Uses upstream DINOv2, predictor, action/proprioception encoders and `VideoWM`.
- Loads the pinned `EncPredWM`, objective and CEM class definitions with their
  method bodies intact, avoiding unrelated heavyweight imports. Trusted pinned
  source is parsed with Python AST; it is not supplied by the generating model.
- Applies identical candidate microbatching to both conditions, default 16,
  to limit GPU memory. The total CEM population remains 300.
- Repeats seeds, executes the actual returned actions in Wall, and logs outcomes,
  trajectories, depths, synchronized planning latency and model-work counts.

This is a small evaluation adapter, not the upstream all-environments evaluation
entry point. It does not download the large offline Wall training dataset:
normalization comes from the pinned release registry, and evaluation tasks are
sampled directly from the simulator. It does not silently install or mock
unrelated robot environments.

The standalone upstream pyproject declares Python 3.10 and many simulator
packages. The notebooks avoid installing that project and use Colab's existing
matched torch/torchvision plus a smaller dependency set. They require working
PyTorch >=2.7 and torchvision, and fail early if GPU/import checks fail. They do
not replace a working Colab CUDA stack automatically.

### Stopper features

Inputs are computed after sample → predict → score → select elites → refit:

1. Used fraction of the 30-iteration cap.
2. Relative improvement in elite mean cost.
3. Mean of the two most recent relative improvements.
4. Current elite mean cost relative to its first-iteration value.
5. Gap between elite mean and best current candidate cost.
6. Elite cost spread relative to elite mean.
7. Current proposal standard deviation relative to initialization.
8. Change in the first action of the returned proposal mean.

Exact scaling, clipping, initialization, dtype and tie behavior are specified in
`core.py` and `prompts/generate.md`. Learned and WSP controllers should share this
interface for future comparisons. The network has no recurrent state; the host
feature extractor retains the small within-solve history and resets it per solve.

### Astra's construction environment

| Allowed | Unavailable through the agent tools |
|---|---|
| List checkpoint tensors and shapes | Model/environment forward execution |
| Read bounded flattened numerical slices | Rollouts or synthetic input probes |
| Weight statistics and bounded singular-value analysis | Losses, gradients or training |
| Read approved source/configuration text | Shell, arbitrary Python, network or unrestricted filesystem |
| Submit one final set of stopper weights | Controller evaluation and feedback-driven repair |

Astra chooses which inspections to use. All calls/results are audited. Access to
weights does not prove they influenced the controller; report actual inspections.
The original checkpoint stays on the host; only requested values/statistics and
source excerpts are sent through API calls.

One-shot means one final controller submission without evaluation feedback. It
can include many inspection tool calls. An invalid submission, truncated response,
API error or budget exhaustion is a failed attempt. Do not pick the best test-set
controller and hide unsuccessful generations.

## 2. The two Colab notebooks

### Before starting

1. Copy `01_wall_baseline.ipynb`, `02_wall_wsp_comparison.ipynb` and `README.md`
   into your repaired project root, commit and push them.
2. Open notebook 01 in Google Colab through its GitHub URL or File → Upload notebook.
3. Choose Runtime → Change runtime type → GPU. Colab GPU availability is variable.
4. In both notebooks set `REPO_URL`, preferably an exact `PROJECT_REF` commit,
   and the same `EXPERIMENT_ID` and `DRIVE_FOLDER`.
5. The first run mounts Drive and downloads checkpoints. Keep substantial free
   disk/RAM. Start with the small pilot; optional sweeps multiply planning time.

For private repositories, clone manually using your preferred credentials first,
into `/content/wsp-stopper`. Do not put a GitHub token into a notebook URL or output.
OpenAI and GitHub credentials serve different purposes.

### Notebook 01 — `01_wall_baseline.ipynb`

Runs without an OpenAI key. Its cells:

1. Clone/check the project and install the minimal runtime dependencies.
2. Download and verify pinned assets, export component tensors, patch upstream,
   freeze a base packet, and run package/upstream CPU tests.
3. Predeclare episode count, seed range, population, elites, microbatch and fixed
   budgets before showing results. The default budget list is `[30]`.
4. Save explicit episodes to `tasks.json` and code/protocol hashes to `protocol.json`.
5. Execute the real pretrained DINO-WM planner with the stopper disabled.
6. Save per-episode results, a baseline summary and a small plot.

Change `FIXED_DEPTHS` to a predeclared list such as `[3,5,10,20,30]` for a sweep.
Keep 30 in the list: it is the primary comparison reference in notebook 02. Use
separate validation episodes to select a best fixed budget for stronger claims.

The notebook will not overwrite an existing task/protocol manifest with a different
experiment. Use a new experiment ID when changing the subset or configuration.
Completed evaluation files are reused; an interrupted unfinished evaluation must
be rerun. Results are written after completing each evaluation call, not after
individual episodes.

### Notebook 02 — `02_wall_wsp_comparison.ipynb`

Run after notebook 01. Its cells:

1. Verify the saved episode hash, project commit, relevant source fingerprints,
   embedded-runner hash, and completed depth-30 baseline.
2. Recreate checkpoint/source preparation if the Colab VM was reset.
3. Freeze a **generation packet** that also includes the exact notebook runtime
   source, final-success semantics and actual microbatch/configuration settings.
4. Read `OPENAI_API_KEY` from Colab Secrets or a hidden prompt. Require you to set
   the exact Astra API model ID available to your account.
5. Let that model inspect the packet and submit the 178 weights once. Validate
   and freeze their hash before any controller evaluation.
6. Re-run the fixed-30 baseline on the current hardware, then run the generated
   controller on the same manifest, with identical model/planner settings/seeds.
7. Produce paired per-episode tables, success/compute/latency charts, an episode
   bootstrap interval and machine-readable comparison files.

A saved baseline does not enter the generating model's prompt or tools. Its
values and task coordinates remain outside the allowlisted source packet. The
model does receive the scoring specification and runtime implementation. The
second notebook invokes a fresh API conversation, not this research discussion.

Set `MODEL_ID` explicitly. The runner does not guess that the UI name “Astra” is
a particular API identifier, and does not silently substitute another model.
`REASONING_EFFORT` is optional and must be supported by that exact model.

Default construction budgets are 40 inspection calls, 200,000 returned tool
characters, 45 API turns, and 8,000 output tokens per turn. They are not a dollar
cap: context accumulates across API turns. The calls are billable. No automatic
billable POST retries or performance-based repair loop exists.

Each `ATTEMPT_ID` is retained. An accepted prior submission is reused without a
new API call; a failed attempt must be preserved and explicitly given a new ID
if you decide to try again. Declare independent attempts in advance, and report
all of them. Generation uncertainty is not captured by an episode-only interval.

### Persistent output layout

Small experiment artifacts are stored under:

```text
MyDrive/wsp-stopper/wall-pilot-001/
  tasks.json
  protocol.json
  baseline_k30.json
  baseline_summary.csv
  generation/attempt-001/
    run.json
    prompt.txt
    events.jsonl
    submitted_raw.json
    submission.json
    weights.json
    accepted_controller.json
  baseline_recheck_attempt-001.json
  wsp_attempt-001.json
  summary_attempt-001.csv
  paired_attempt-001.csv
  comparison_attempt-001.json
  comparison_attempt-001.png
```

Additional predeclared fixed depths create more `baseline_k*.json` files.
Checkpoints, unpacked tensors and source clones remain under `/content` and are
recreated after VM resets. Use a new version/experiment rather than overwriting
frozen specifications. Audit logs contain model outputs and inspected data;
keep them out of public commits unless reviewed for release.

## 3. Comparison and interpretation

The saved manifest contains each episode's explicit start, goal, wall/door layout,
environment seed and independent planner RNG seed. It is reused verbatim.
Notebook 02 checks the hash before evaluation. Microbatching, horizon, population,
elite count, scoring and return-plan conventions are matched.

Reported measures include final success, final distance, used refinements,
candidate-sequence counts, imagined transition evaluations, utility, synchronized
planning seconds (including input/goal encoding and stopper overhead), trajectories
and stop decisions. Model loading and initial warm-up are excluded from timing.
The upstream additionally evaluates the proposal mean each iteration; the runner
counts those transitions separately from the candidate population. Savings in
logical refinements do not automatically equal latency savings.

Utility is `success - 0.05 * total_CEM_iterations / (30*M)` with M=1 here. This
penalty and the eight-feature interface are our experimental choices, not claimed
author settings.

The notebook-01 baseline is retained as the original reference; a repeated baseline
in notebook 02 provides the current-session comparison. Hardware, torch versions
and behavioral drift are checked/reported. Even on the same GPU model, varying
load and numerical nondeterminism can affect timing and marginal candidate ranks.

Six episodes are a smoke/pilot study, not a conclusive benchmark. Increase held-out
coverage and independent model generations before making general claims. A
controller that always stops at one depth is effectively a fixed budget. Better
performance than depth 30 alone is insufficient to show an advantage over a tuned
fixed budget. Learned-stopper training from the meeting plan remains a separate,
unimplemented experimental arm.

## 4. Verification and limits

Before delivery, the shared runner was executed with the actual pretrained
checkpoint and actual Wall simulator on CPU using reduced candidate counts.
All learned component keys loaded without missing/unexpected entries, and the
visual encoder used strict loading. Real environment execution completed.
An always-stop-at-3 diagnostic controller produced exactly the same actions as a
fixed-three-iteration run on the same saved episode and seed.

That diagnostic controller is a wiring test, not an Astra-generated experimental
controller. No live API generation was performed. The full default GPU pilot,
Google Drive mounting, Colab Secrets UI and complete hosted Colab execution were
not run in this environment. Notebook JSON/code syntax and matching embedded
runtime hashes were checked separately. No claimed success improvement or speedup
is bundled with these files.

## 5. Local maintenance commands

Run from the existing repaired repository:

```bash
source .venv/bin/activate
python -m pip install -e '.[checkpoint]'
python -m unittest discover -s tests -v
```

For local preparation only:

```bash
wsp-wall prepare --work assets/wall-v1
export WSP_TEST_UPSTREAM="$PWD/assets/wall-v1/jepa-wms"
python -m unittest discover -s tests -v
wsp-wall freeze --work assets/wall-v1 --out packets/wall-v1
wsp verify packets/wall-v1
```

Python environment activation must happen in the shell actually running these
commands. Notebook cells install into their active Python interpreter. Avoid
mixing Conda base and project environments unintentionally.

If `wsp_stopper.core` or `wsp_stopper.cli` is missing, see `docs/REPAIR_INSTALL.md`:
the earlier upgrade archive was a delta requiring the original starter modules.
Do not reinstall torch to solve absent project source files. `restore_missing.py`
restores missing source without overwriting existing work.

The notebooks require `make_packet`/preparation from the Wall upgrade. Editing
source after freezing a packet requires a new packet and a new generation. An
old controller should remain bound to the specification that produced it.

## 6. Commit these three files

```bash
git add -- 01_wall_baseline.ipynb 02_wall_wsp_comparison.ipynb README.md
git diff --cached --stat
git commit -m "feat: add paired Wall baseline and WSP Colab experiments"
git push
```

Keep notebook outputs cleared before public commits. Ignore `.venv/`, `.env`,
`assets/`, `packets/`, `runs/` and `.ipynb_checkpoints/` in your local `.gitignore`.
Do not commit API keys, checkpoint binaries or generated controller/test artifacts
by accident. The older history-initialization scripts are unnecessary for this
addition to an existing repo.

## Sources

- Original DINO-WM: https://github.com/gaoyuezhou/dino_wm
- Released Meta implementation: https://github.com/facebookresearch/jepa-wms
- Released checkpoints: https://huggingface.co/facebook/jepa-wms
- DINOv2: https://github.com/facebookresearch/dinov2
- Knowing When to Stop Thinking: https://arxiv.org/abs/2610.28011
- Weight-Space Programming: https://rikiyat.github.io/blog/weight-space-programming/
- OpenAI function calling: https://developers.openai.com/api/docs/guides/function-calling
- Google Colab runtime and persistence FAQ: https://research.google.com/colaboratory/faq.html

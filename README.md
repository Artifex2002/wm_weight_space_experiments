# Weight-informed stopping for DINO-WM Wall planning

[Open baseline in Colab](https://colab.research.google.com/github/Artifex2002/wm_weight_space_experiments/blob/main/notebooks/01_wall_baseline.ipynb) · [Open WSP comparison in Colab](https://colab.research.google.com/github/Artifex2002/wm_weight_space_experiments/blob/main/notebooks/02_wall_wsp_comparison.ipynb) · [Open diagnostics in Colab](https://colab.research.google.com/github/Artifex2002/wm_weight_space_experiments/blob/main/notebooks/03_wall_frozen_stopper_diagnostics.ipynb) · [Open prompt study in Colab](https://colab.research.google.com/github/Artifex2002/wm_weight_space_experiments/blob/main/notebooks/04_wall_prompt_comparison.ipynb)

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

Latest research status: the six-task pilot and fixed-depth sweep are complete.
Fixed 15 is the best tested pilot budget under our utility; the first Astra
controller was conservative. The two-prompt follow-up has six accepted
controllers, but only **one of twelve tasks evaluated: 8/96 jobs**. This is a
partial study, not evidence that either prompt wins. Read
[the experiment log](docs/EXPERIMENT_LOG.md) and
[curated evidence index](results/README.md).

## 1. Repository structure and how it comes together

```text
wm_weight_space_experiments/
├── notebooks/
│   ├── 01_wall_baseline.ipynb
│   ├── 02_wall_wsp_comparison.ipynb
│   ├── 03_wall_frozen_stopper_diagnostics.ipynb
│   └── 04_wall_prompt_comparison.ipynb
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
│   ├── test_wall_upgrade.py
│   └── test_notebooks.py
├── docs/
│   ├── PROTOCOL.md
│   ├── INTEGRATION.md
│   ├── WALL_UPGRADE.md
│   ├── REPAIR_INSTALL.md
│   ├── REPO_UPDATE_GUIDE.md
│   ├── UPDATE_MANIFEST.json
│   └── EXPERIMENT_LOG.md
└── results/
    ├── README.md
    ├── wall-pilot-001/
    └── wall-prompt-study-001/
```

Full assets, inspection packets and working runs stay outside version control.
`results/` contains explicitly curated small snapshots; their coverage and missing
evidence are documented. They do not replace active Drive runs. The complete
starter/Wall upgrade package must already be present; the notebooks check required
modules. The follow-up notebooks contain their additional helpers internally and
do not require changing `src/wsp_stopper/`.

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

All four notebooks embed **the same Python evaluator as a code-cell string**,
with SHA-256 `c850c41dbe72088e3946a0d6ad965423c69675b25d41f0ef38a5aac8da45b299`.
The cell writes `/content/wsp_wall_notebook_runtime.py`. Notebook 04 additionally
embeds its study orchestration helper. No extra Python helper files need to be
installed in the repository. Notebook integration tests check evaluator identity
and the source-fingerprint contract against notebook 02's independent implementation.

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

## 2. The four Colab notebooks

| Notebook | Purpose | Additional Astra generation? |
|---|---|---|
| 01 | Save original Wall tasks and fixed-budget baseline | No |
| 02 | Construct one stopper, freeze it, and compare on the saved tasks | Yes, one attempt |
| 03 | Audit the frozen controller, sweep fixed depths, optionally test new episodes and repeat timing | No |
| 04 | Compare original versus marginal-utility prompt, three fresh attempts each, on one new task set | Yes, six planned attempts |

### Before starting

1. Keep all four `.ipynb` files in `notebooks/` and `README.md` in the repository root.
   Commit the repaired Python package before creating a baseline. See the
   [incremental update guide](docs/REPO_UPDATE_GUIDE.md) for these follow-up files.
2. Open [notebook 01](https://colab.research.google.com/github/Artifex2002/wm_weight_space_experiments/blob/main/notebooks/01_wall_baseline.ipynb)
   or [notebook 02](https://colab.research.google.com/github/Artifex2002/wm_weight_space_experiments/blob/main/notebooks/02_wall_wsp_comparison.ipynb)
   in Colab using the links above, or upload a notebook through Colab's File menu.
3. Choose Runtime → Change runtime type → GPU. Colab GPU availability is variable.
4. Keep `REPO_URL`, the recorded `PROJECT_REF`, `EXPERIMENT_ID` and the intended
   Drive account/folder consistent across the notebooks.
5. The first run mounts Drive and downloads checkpoints. Keep substantial free
   disk/RAM. Start with the small pilot; optional sweeps multiply planning time.

Use this repository URL in notebooks 01/02 (their original source cells still
contain a placeholder; notebooks 03/04 already include this URL):

```python
REPO_URL = "https://github.com/Artifex2002/wm_weight_space_experiments.git"
PROJECT_REF = "PASTE_THE_EXACT_GIT_COMMIT"
EXPERIMENT_ID = "wall-pilot-001"
DRIVE_FOLDER = "/content/drive/MyDrive/wsp-stopper"
```

For a fresh pilot, choose one exact source commit before notebook 01 and use
that commit in notebook 02. Once a pilot exists, `protocol.json` is authoritative.
For the active `wall-pilot-001` and prompt study, retain:

```python
PROJECT_REF = "2ce606163917e5f9953a5842e61cb322c722c236"
```

**Do not replace this value with the new commit created when adding notebooks 03/04,
results or documentation.** The follow-ups are self-contained and deliberately
check out the earlier source commit even though their notebook files live in a
later Git revision. Their source guard checks all package `.py` files plus the
release configuration and original prompt, using relative-path keys identical to
notebook 02. Do not disable the guard or edit active study receipts.

The notebook files living in `notebooks/` does not require a Python code change.
Their bootstrap clones the repository into `/content/wsp-stopper` and changes
working directory to that repository root before importing or running anything.
Do not change the root to `/content/wsp-stopper/notebooks`.

For private repositories, clone manually using your preferred credentials first,
into `/content/wsp-stopper`. Do not put a GitHub token into a notebook URL or output.
OpenAI and GitHub credentials serve different purposes.

### Notebook 01 — `notebooks/01_wall_baseline.ipynb`

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

### Notebook 02 — `notebooks/02_wall_wsp_comparison.ipynb`

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

### Notebook 03 — `notebooks/03_wall_frozen_stopper_diagnostics.ipynb`

Use the Drive account containing notebook 02's completed results, or copy its
complete result files first. The original three-file baseline handoff alone does
not include the generated controller/audit needed here.

The notebook verifies identities and reconstructs STOP margins from frozen weights
and recorded features. It then offers three opt-in GPU sections:

| Switch | Operation | Scope |
|---|---|---|
| `RUN_DEPTH_SWEEP` | Depths 3, 5, 10, 15, 20, 25, 30 | Same original six tasks; exploratory |
| `RUN_HELDOUT` | Fixed 30, frozen stopper, optional preselected fixed depth | Default 12 new tasks; same fixed layout |
| `RUN_TIMING` | Repeated baseline/stopper evaluations, alternating order | Diagnostic subset, not a population estimate |

Change each flag **inside its existing cell**, then execute the entire cell.
A separate assignment later in the notebook does not run the skipped body.
No model generation/API key is needed. Asset setup runs both `prepare` and base
packet `freeze`, because the latter creates `wall.eval.yaml` and applies pinned
source patches. Child-process output and logs expose underlying failures.

Outputs go to `EXPERIMENT/analysis/ANALYSIS_ID/`. Completed result files are checked
and reused. The final export writes actual files and verifies the ZIP. Rerun
export after new evaluations; an older downloaded ZIP will not update itself.
Changing frozen settings requires a deliberately new analysis, not overwriting
old weights/results. The uploaded completed sweep is documented in the log; the
separate held-out and repeated-timing sections were not run in that upload.

### Notebook 04 — `notebooks/04_wall_prompt_comparison.ipynb`

This freezes the original prompt and a second prompt adding marginal-utility
framing. It constructs three controllers per prompt with identical model,
reasoning effort, feature interface, checkpoint access and budgets. Defaults are
`gpt-6-astra`, `xhigh`, 40 inspections, 200,000 tool-output characters, 45 responses
and a 32,768-token ceiling per response. Token ceilings include reasoning and are
not a total dollar cap. The user supplies the key privately via Colab Secrets.

The study saves all six construction slots before generation, accepts one final
submission per slot, and retains failures/interrupted attempts without replacing
them. All six slots must settle before any test evaluation. Distinct hashes are
reported; duplicate controllers would also be retained. No API seed or statistical
independence is assumed.

The evaluator compares the six controllers and fixed 15/fixed 30 on the same 12
new episodes. It saves each method/episode separately and rotates execution order.
With all controllers valid, there are 96 jobs. The default cell runs eight new jobs
per invocation; rerun until `Remaining jobs: 0`. Completed jobs are not rerun.
Generation and evaluation require editing `RUN_GENERATION` and `RUN_EVALUATION`
to `True` in their respective existing cells. Generation is billable; GPU evaluation
makes no API calls. At the observed T4 rate, cap-heavy evaluations may take roughly
12–14 GPU hours across resumable sessions. This is an estimate, not a benchmark.

Outputs live at `EXPERIMENT/studies/STUDY_ID/`: frozen study/prompt definitions,
test tasks, a persistent complete packet, attempt receipts, weights/audits,
per-episode evaluations and result tables. The active study is
`wall-prompt-study-001`, with helper SHA
`d4bd15761f3932c778b8bf17039ec033e3f2d3b1ec667b523089774efdab6ad1`.
Keep its helper/settings/weights fixed when resuming. GPU model/software changes
are rejected; different physical sessions are recorded. Paired latency summaries
use only same-session comparisons.

Summaries include every attempt, valid/failed counts, duplicate hashes,
per-controller metrics, between-controller variation and paired comparisons with
both fixed budgets. Wait for `COMPLETE` before interpreting arm-level performance.
As of the curated snapshot only 8/96 jobs exist; do not treat its blank arm metrics
as zero scores. The final ZIP excludes large numerical packet arrays and some
`.py`/`.yaml` source extensions; retain the complete active packet separately in
Drive. No raw API audit files are committed in the curated results folder.

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

### Two people, two Colab accounts: use the same saved episodes

**Yes.** One person can run notebook 01, then another can run notebook 02 in a
separate account and VM. The second person must receive the actual handoff
files; identical names under `MyDrive` do not share data between accounts.
Sharing the notebook URL also does not share runtime files.

| Requirement | What must match |
|---|---|
| Episodes | The exact bytes of `tasks.json`: start/goal coordinates, layout, environment and planner seeds |
| Source | `protocol.json` project commit, source fingerprints and embedded-runner hash |
| Model | Pinned checkpoint/encoder assets and hashes from the release lock |
| Planner | Population, elites, horizon, execution chunk, objective and microbatching |
| Original reference | Complete `baseline_k30.json` from notebook 01 |
| Local Drive path | May differ, provided the same handoff files are placed under the recipient's `EXPERIMENT` directory |

Do not independently regenerate a supposedly equivalent task set as the handoff.
Copy the original manifest unchanged. Notebook 02 checks its SHA-256 against the
protocol and baseline results, and rejects mismatches. Shared seeds improve
pairing but do not guarantee bitwise-identical GPU trajectories or identical
success: PyTorch/platform differences and numerical nondeterminism can change
candidate ranks. Identical tasks and configurations are the guarantee here.

The current notebooks have an explicit dependency: notebook 02 checks that the
baseline handoff exists before proceeding. You may prepare runtimes/assets in
parallel, but the baseline must finish before the WSP notebook's handoff check.

#### Person A: run notebook 01 and export a handoff

Finish notebook 01. Then add/run this extra cell in that Colab session. It uses
the notebook's existing `EXPERIMENT` variable and downloads three small files in
a ZIP; no checkpoints or API credentials are included.

```python
from pathlib import Path
import json, zipfile
from google.colab import files

handoff_names = ("tasks.json", "protocol.json", "baseline_k30.json")
assert all((EXPERIMENT / name).is_file() for name in handoff_names)
print("Tell your roommate to use PROJECT_REF =",
      json.loads((EXPERIMENT / "protocol.json").read_text())["project_commit"])
handoff_zip = Path("/content") / f"{EXPERIMENT.name}-handoff.zip"
with zipfile.ZipFile(handoff_zip, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in handoff_names:
        archive.write(EXPERIMENT / name, f"{EXPERIMENT.name}/{name}")
files.download(str(handoff_zip))
```

Send that ZIP to Person B using your normal file-sharing method. The handoff
contains baseline outcomes for the evaluator; these must remain outside Astra's
inspection packet and must not be pasted into its prompt.

#### Person B: import the handoff before notebook 02's verification cell

Use the exact `PROJECT_REF` reported by Person A and the same `EXPERIMENT_ID`.
Run notebook 02's parameter, bootstrap and embedded-runtime cells. The bootstrap
mounts your own Drive and creates your own `EXPERIMENT` directory. **Before** the
cell that starts `TASKS = EXPERIMENT / 'tasks.json'`, add/run this upload cell:

```python
import io, json, hashlib, zipfile
from google.colab import files

uploaded = files.upload()  # Select the ZIP supplied by Person A.
assert len(uploaded) == 1, "Upload exactly one handoff ZIP"
handoff_names = ("tasks.json", "protocol.json", "baseline_k30.json")
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
    expected = {f"{EXPERIMENT.name}/{name}" for name in handoff_names}
    assert set(archive.namelist()) == expected, "Unexpected handoff contents/experiment ID"
    payload = {name: archive.read(f"{EXPERIMENT.name}/{name}")
               for name in handoff_names}

incoming_protocol = json.loads(payload["protocol.json"])
incoming_baseline = json.loads(payload["baseline_k30.json"])
task_sha = hashlib.sha256(payload["tasks.json"]).hexdigest()
assert task_sha == incoming_protocol["tasks_sha256"] == incoming_baseline["tasks_sha256"]
assert PROJECT_COMMIT == incoming_protocol["project_commit"], "Use Person A's exact commit"
assert project_fingerprint() == incoming_protocol["project_files"]
assert RUNTIME_SHA == incoming_protocol["runtime_sha256"] == incoming_baseline["runtime_sha256"]

# Check every existing destination before writing any file. Keep existing work.
for name, blob in payload.items():
    destination = EXPERIMENT / name
    if destination.exists():
        assert destination.read_bytes() == blob, f"Conflicting existing file: {name}"
for name, blob in payload.items():
    destination = EXPERIMENT / name
    if not destination.exists():
        destination.write_bytes(blob)
print("Handoff imported and hashes verified:", EXPERIMENT)
```

Continue notebook 02 from its original handoff verification cell. No notebook
source edit beyond parameters and the added transfer cell is necessary. Do not
upload these files into `packets/` or add them to the generator's source list.
Person B supplies the API key privately; Person A does not need that key.

#### What is comparable across different GPUs?

The WSP controller and baseline can use exactly the same episodes and initial
planner randomness on separate machines. However, raw latency measured on
Person A's GPU cannot establish a speedup relative to a different Person B GPU.
Notebook 02 already reruns the fixed-30 baseline on Person B's current runtime
before evaluating the controller; use that pair for latency/compute comparison.
It retains the original baseline and reports success changes on the recheck.
Even the same GPU model does not eliminate load/thermal/numerical differences.

Use independent Drive copies for this handoff so the two people do not write
result files simultaneously. If you choose a shared mounted folder instead,
verify its actual mount path and arrange a single writer per result file.
The same `EXPERIMENT_ID` alone never synchronizes two private account drives.

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

That diagnostic controller was a wiring test, not an Astra-generated result.
Those initial local checks did not call the API. Subsequently, user-run Colab
artifacts established the six-episode pilot, completed depth sweep, six new
accepted prompt-study controllers and the first new-episode evaluation block.
The experiment log records the measured outcomes and their limited scope.

Notebook syntax, output clearing, identical evaluator bytes, independent source
fingerprints and preflight rejection of changed sources are checked locally.
Fixture tests exercised orchestration/resumption; they are not a substitute for
GPU inference. Actual evidence comes from the supplied Colab artifacts. The
prompt study and repeated-timing validation are not complete.

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

## 6. Commit the follow-up notebooks and research record

See [REPO_UPDATE_GUIDE.md](docs/REPO_UPDATE_GUIDE.md) for the exact file inventory,
copy instructions and two incremental commits. The first adds corrected notebooks,
integration tests and this guide. The second adds the findings log and curated
experimental evidence. This preserves a reviewable distinction between tools and
research observations.

Keep executable notebook outputs cleared. Use explicit `git add` paths; do not
stage checkpoints, credentials, full inspection packets or raw export archives.
The deliberately curated `results/` snapshots are the exception for small result
artifacts, and explicitly state their coverage and missing originals. Working
Drive studies remain authoritative for resumption.

## Sources

- Original DINO-WM: https://github.com/gaoyuezhou/dino_wm
- Released Meta implementation: https://github.com/facebookresearch/jepa-wms
- Released checkpoints: https://huggingface.co/facebook/jepa-wms
- DINOv2: https://github.com/facebookresearch/dinov2
- Knowing When to Stop Thinking: https://arxiv.org/abs/2610.28011
- Weight-Space Programming: https://rikiyat.github.io/blog/weight-space-programming/
- OpenAI function calling: https://developers.openai.com/api/docs/guides/function-calling
- Google Colab runtime and persistence FAQ: https://research.google.com/colaboratory/faq.html

- PyTorch reproducibility notes: https://docs.pytorch.org/docs/stable/notes/randomness.html

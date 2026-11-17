# Wall upgrade: steps 1, 3, and 4

This update adds code to the original starter. Extract the upgrade ZIP INTO the
project root (the folder containing pyproject.toml). It contains only added or
changed files. Existing keys, assets, runs and git history are untouched by the
archive. Review `git diff` before committing. No checkpoint binaries are bundled.

## Commit incrementally to your existing GitHub repo

Run from the project root, after the original starter has been committed:

```bash
git status --short
bash scripts/commit_wall_upgrade.sh --push
```

This creates and pushes three commits, staging only the listed source files:

1. `feat: pin Wall assets and export all inference checkpoint components`
2. `feat: add version-checked CEM hook and local encoder loading`
3. `docs: document packet freeze and Colab validation sequence`

Omit --push for local commits only. If the remote is not configured, use:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin HEAD
```

The script requires an existing initial commit and a clean staging index. It
leaves unrelated unstaged work alone. If you have not initialized the starter:
run `bash scripts/init_history.sh` first, then the upgrade script. If a push
fails, fix Git authentication and rerun the upgrade script; it skips groups with
no changes and attempts the outstanding push. Never force-push for this setup.
GitHub credentials and OpenAI API credentials are separate.

## Install (local CPU preparation; no API key required)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[checkpoint]'
python -m unittest discover -s tests -v
```

The baseline upstream package requires Python 3.10, while this preparation package
supports 3.10+. Using Python 3.10 now can reduce later environment differences.
Do not install every upstream simulator dependency just to inspect tensors.
Preparation needs several GB of disk, network for clones/assets and enough RAM
for checkpoint tensors. It does not use a GPU.

## Step 1: prepare actual checkpoint and encoder

```bash
wsp-wall prepare --work assets/wall-v1
```

The release lock pins:
- Meta source: 13cf1d9c7e476f53c17714d2e0f1dc239a883ce0.
- Wall HF revision: 9b9c41ef249466630dbf1a20e78391865d07b3b9.
- DINOv2 code: 7764ea0f912e53c92e82eb78a2a1631e92725fc8.
- SHA-256 of the downloaded Wall and visual-encoder checkpoint files.

No opaque moving main is used for execution. The Wall checkpoint is epoch 50.
Its root has predictor, action_encoder, proprio_encoder, opt, scaler and epoch.
The visual encoder is NOT contained in that checkpoint; it is downloaded
separately from the official DINOv2 ViT-S/14 release.

prepare loads data with weights_only=True, records tensor counts and provenance,
and creates inspection-state.pt combining all four inference component state
maps with component prefixes. It excludes optimizer/scaler and visualization
heads. Original numerical values are retained. Export is deterministic in values;
torch.save byte hashes can differ across serialization versions and are recorded
per run. The checkpoint's epoch is provenance, not performance feedback.

Actual checked inventory (scalar entries include buffers, not just parameters):

| Component | Tensor entries | Scalar entries |
|---|---:|---:|
| Visual encoder | 175 | 22,056,576 |
| Predictor | 75 | 26,590,200 |
| Action encoder | 2 | 110 |
| Proprio encoder | 2 | 30 |

This identifies components/values but does not prove every tensor is consumed by
the runtime loader. Full model state matching and forward execution are step 2.
Checkpoint matching can require upstream key cleanup; original module. prefixes
are retained in inspection to avoid hiding transformations.

## Steps 3 and 4: install the hook and freeze the packet

```bash
wsp-wall freeze --work assets/wall-v1 --out packets/wall-v1
wsp verify packets/wall-v1
```

freeze applies two SHA-checked edits inside assets/wall-v1/jepa-wms:

1. CEM can optionally call CEMHook after each complete iteration. It passes
   candidate costs, exact elite indices, proposal STANDARD DEVIATIONS and the
   proposal mean's first action. It returns the current mean when stopped.
2. DINOv2 loading uses the locally pinned source and weights, with strict tensor
   loading, avoiding an implicit torch.hub fetch of moving main.

The original files are retained as .wsp-original backups. Changed source versions
or unexpected local edits fail rather than silently patching another algorithm.
There is one independent hook per solve. Upstream evaluates environments serially;
this is not a batched or distributed stopping implementation. The hook rejects
distributed planning and requires cap 30. With WSP_STOPPER_WEIGHTS unset, no
controller or feature extractor runs, and the original fixed loop is retained.
With it set, set it to an absolute generated weights.json path.

freeze writes preparation/config/specification records, a patch receipt, numeric
tensor files and an allowlisted source packet for Astra. It packages implementation
files and licenses, without paper results, depth curves, rollouts or a preferred
stopping iteration. The generic source reading tool lets Astra read these files.
The preparation artifacts and frozen packet are ignored by Git.

### The selected release's actual planning semantics

This is a released DINO-WM reproduction baseline, NOT a verified reproduction of
Knowing When to Stop Thinking's checkpoint or results. We preserve its:

- horizon 6; execution chunk 6; 300 candidates; 10 elites; refinement cap 30;
- diagonal Gaussian, zero initial mean, sigma=1; original unbiased std refit;
- included mean candidate; zero mean/std momentum defaults; no warm start usage;
- original terminal L2 objective with proprioceptive weight 0.1;
- returned updated proposal mean, not the minimum-cost sampled candidate;
- image size 224, mean/std 0.5, context window 2, frameskip 5, action skip 1.

Each internal planner action packs five consecutive 2D environment actions,
so internal action_dim is 10. Six planned actions unpack into 30 simulator steps.
For random-state goals the upstream time limit is frameskip*goal_H=30, so this
specific configuration normally has ONE planning solve per episode (M=1).
It does not provide a long sequence of MPC replans. Extending the episode limit
is a new experimental configuration: update both execution code and packet
specification before generating weights. Do not reuse an old frozen prompt after
changing the evaluation semantics.

Explicit preparation overrides: one task/process, 50 episodes, optional plots off,
decoding off, decoder/state heads omitted. Head omission avoids separate huge
visualization checkpoints; latent candidate ranking and mean selection are kept.
Upstream still unrolls the proposal mean once per iteration for logging; account
for this extra work in runtime metrics. Candidate iterations alone undercount it.

### Files generated locally

```text
assets/wall-v1/
  wall.pth.tar            # released epoch-50 checkpoint
  encoder.pth             # separate visual backbone
  jepa-wms/               # pinned upstream source, two tracked edits
  dinov2/                 # pinned encoder source
  inspection-state.pt     # four component state maps flattened for inspection
  preparation.json        # component counts, checksums and provenance
  wall.eval.yaml          # experiment config with explicit overrides
  wall.spec.json          # stopper-visible task/planner specification
  patch-receipt.json      # original/patched source hashes
packets/wall-v1/
  manifest.json
  manifest.sha256
  tensors/*.npy
  sources/*
```

The source/config packet has validation_status=execution_pending. ready=true
means preparation fields are complete, NOT that Wall has been tested.

## What was tested before delivery

- Downloaded both actual checkpoint files and verified their SHA-256 values.
- Safely loaded the real epoch-50 checkpoint and exported all four components.
- Created and hash-verified a full numeric/source packet (254 tensor entries).
- Ran 14 tests, including two CPU tests using the actual pinned CEM class with
  a quadratic toy objective: hook-disabled output/losses exactly match upstream;
  stopping at 3 exactly matches a fixed 3-iteration prefix and resets each solve.
- These are integration/packaging tests, not Wall performance results.

To repeat the upstream CPU tests after preparation (run from project root):

```bash
export WSP_TEST_UPSTREAM="$PWD/assets/wall-v1/jepa-wms"
python -m unittest discover -s tests -v
```

Without that environment variable, the two upstream tests are explicitly skipped.
They parse/execute the pinned CEM class with a toy cost; they do not need the
upstream simulator dependency stack.

## Next: Colab step 2, then finalize step 4, then step 5

Yes, baseline execution and live API generation can be run in Colab. Keep them
as distinct phases. Baseline planning needs a compatible GPU/runtime and Wall
assets/dependencies. API generation itself needs CPU inspection plus your API key;
it does not consume Colab GPU for Astra's reasoning.

The current upstream pyproject requires Python >=3.10,<3.11 and pins torchvision
0.22.0; a default Colab runtime may not match. The GPU baseline installation and
Wall dataset/evaluation bootstrap still need to be completed and tested in step 2.
No promise of a one-cell Colab simulator install is made by this upgrade.

For local model loading in Colab, set:

```bash
export WSP_DINOV2_SOURCE="$PWD/assets/wall-v1/dinov2"
export WSP_DINOV2_WEIGHTS="$PWD/assets/wall-v1/encoder.pth"
unset WSP_STOPPER_WEIGHTS  # ordinary baseline, no controller
```

Keep the source checkout on PYTHONPATH when running upstream evaluation. The
config's absolute paths are host-specific. After cloning/preparing on Colab,
regenerate the config there rather than copying local absolute paths blindly.
Any source/config change required by step 2 means freezing a NEW packet version
before generation. Never pass baseline metrics or known best depths to Astra.

Only once runtime correctness is established, run:

```bash
wsp generate --packet packets/wall-v1 --out runs/trial-001
```

Set OPENAI_API_KEY and OPENAI_MODEL as in README. In Colab use Secrets/userdata,
not a key committed in a notebook. Keep every attempt. Step 5's actual API run is
not performed by this upgrade. No live controller weights or test scores exist.

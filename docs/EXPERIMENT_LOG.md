# Wall stopping experiments — findings and research log

**Snapshot date:** 2026-11-18, America/New_York. Archive filenames use UTC.
**Reference project commit:** `2ce606163917e5f9953a5842e61cb322c722c236`.
**Status:** the original six-task pilot and its depth sweep are complete; the
follow-up prompt study has six accepted controllers but **only 8/96 evaluation
jobs complete: one task evaluated by eight methods**.

This log records observations, interpretations and limitations separately. It
is a research record, not a claim that the original stopping paper has been
reproduced. New entries should retain earlier results and state when a hypothesis
was revised. The evidence index is in [results/README.md](../results/README.md).

## 1. Research question and experimental boundary

Can a general LLM inspect a frozen DINO-WM implementation/checkpoint and directly
construct the numerical weights of a small CEM stopping network, without training
or performance feedback during construction? Does the resulting controller save
compute while retaining or improving task success? Does the way we frame the
objective change the program it constructs?

Construction receives the environment/model/planner specification and bounded
read-only source/tensor tools. It cannot run forward passes, simulator rollouts,
synthetic feature probes, gradients, training or test evaluations. After one final
weight submission, the evaluator freezes and scores that controller. Multiple
inspection turns are allowed; this is one final submission, not one API request.

This is an intentional extension of Weight-Space Programming Experiment 5, whose
reported condition had no tools. The authors' released logs identify the original
model as `gpt-6-astra` with `xhigh` reasoning. We use that requested model/effort,
with our own recorded inspection/response budgets. Access does not establish that
numerical checkpoint contents were useful or causally influenced the result.

The reference is Meta's released Wall DINO-WM checkpoint, epoch 50, with a separate
DINOv2 ViT-S/14 encoder. We have **not verified that it is the identical checkpoint,
checkout, or evaluation configuration used in Knowing When to Stop Thinking**.
The conjectured early success peak must be measured locally, not assumed.

## 2. Fixed computation and evaluation contract

| Item | Recorded setting |
|---|---|
| Upstream Meta source | `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` |
| Checkpoint release revision | `9b9c41ef249466630dbf1a20e78391865d07b3b9` |
| DINOv2 source | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| Checkpoint epoch | 50 |
| Inspection inventory | 254 tensor entries: encoder, predictor, action and proprioception encoders |
| Stopper | 8 inputs → 16 ReLU units → 2 logits, 178 parameters |
| Decision | STOP iff STOP logit > CONTINUE logit; ties continue |
| Host guards | Continue before iteration 3; stop at iteration 30 |
| CEM | Population 300, elites 10, candidate microbatch 16 |
| Horizon and returned chunk | Six packed actions, each representing five raw 2D actions |
| Episode | One planning solve; execute 30 raw simulator steps |
| Success | Final position L2 distance to goal < 4.5 |
| Utility | `success - 0.05 * completed_CEM_iterations / 30` |
| Returned plan | Current refitted proposal mean, not a retrospectively chosen earlier plan |
| Geometry | New starts/goals on one fixed wall layout |
| Recorded stack | Tesla T4; Python 3.13.16; torch 2.14.1+cu130; torchvision 0.29.1+cu130 |

The numerical objective used inside CEM is terminal visual-feature MSE plus
0.1 times proprioception-feature MSE. That planning cost is not a calibrated
probability of simulator success. The eight features summarize iteration count,
relative elite-cost improvements, normalized remaining cost, elite gap/spread,
proposal contraction, and first-action change.

All four notebooks embed the same evaluator bytes, SHA-256:
`c850c41dbe72088e3946a0d6ad965423c69675b25d41f0ef38a5aac8da45b299`.
Checkpoint/component hashes and the exact planner/source inventory are in the
saved protocols and `config/wall.release.json`.

## 3. Initial six-task pilot

Evidence: [pilot snapshot](../results/wall-pilot-001/), especially the baseline,
WSP result, comparison JSON and controller receipt. The task manifest SHA-256 is
`b9fd1657677b3d5d1f615e51c97601e2486ea799a9e99839512fc24174b79f74`.

| Method | Successes | Mean iterations | Mean planning seconds | Mean utility |
|---|---:|---:|---:|---:|
| Fixed 30, same-session recheck | 5/6 | 30.0000 | 507.081353 | 0.783333 |
| First Astra controller | 5/6 | 28.8333 | 490.865760 | 0.785278 |

Observed iteration saving: **3.8889%**. Observed planning-time saving: **3.1978%**.
The original notebook-01 baseline and the recheck had identical success labels.
The original generation's accepted weight hash is
`ebffe6522a6e0d6aa3cbdd6f34548835e2c403a24fd947d703d3c4e1ccf37060`.

Every successful episode ran to 30. Only episode 4 stopped early, at iteration
23; it remained unsuccessful. Its final distance changed from about 25.5673 to
25.5275. Thus the utility gain came entirely from spending less compute on a
still-unsuccessful episode. There was no observed task-success gain.

**Interpretation:** the controller is functional and modestly reduces compute on
this sample. It does not demonstrate identifying an optimal planning depth or
stopping near a task-performance peak. Six paired episodes and one timing pass
cannot establish preserved success or a stable latency speedup. The bootstrap
interval `[0,0]` in the pilot report arises because all six paired success-label
differences were zero; it is not an equivalence result.

## 4. What the first controller actually computes

The original audit records 10 inspection calls: seven source reads, one tensor
inventory request, and two numerical reads totaling 120 values (100 action-encoder
weights and 20 proprioception-encoder weights). No predictor or visual-encoder
numerical weights were requested. It used 84,428 of 200,000 allowed tool-output
characters and 10 of 40 inspection calls. Limits were not exhausted.

The accepted network has **38 nonzero parameters** and nine contributing hidden
units. It implements a conservative convergence rule. Write `t=k/30`, `r` for the
latest relative improvement, `p` for the preceding improvement recovered from
`2*smoothed_improvement-r`, `g` for elite gap, `v` for elite spread, `q` for proposal
standard-deviation ratio, `d` for first-action change, and `[x]+ = max(x,0)`.
Ignoring float32 rounding, its STOP-minus-CONTINUE margin is:

```text
0.001
- [8-k]+
- [100*abs(r) - 0.2*(1+t)]+
- [100*abs(p) - 0.2*(1+t)]+
- [100*g - 0.5*(1+t)]+
- [200*v - (0.4+0.6*t)]+
- [10*q - (1+t)]+
- [20*d - (0.4+0.6*t)]+
```

The host still applies its iteration guards. The network adds its own minimum
wait of eight iterations, then effectively requires small improvements on both
recent steps, tight elite costs, a contracted proposal and a stable first action.

**Correction to an earlier hypothesis:** the normalized remaining-cost feature
has an entirely zero incoming column. It cannot trigger this controller. Earlier
speculation that substantial remaining relative cost contributed to its stop is
not supported by the weights.

At episode 4 iteration 22, first-action change was about 0.04504 while the
zero-penalty threshold was 0.04200. It was the only remaining blocking penalty.
At iteration 23 the change fell to 0.02752, below the relaxed threshold 0.04300;
all penalties became zero, leaving the programmed positive margin 0.001. The
small margin is an intentional decision threshold, not a confidence probability.

All 173 pilot trace decisions were reproduced from the accepted weights. At
iteration 30 every successful episode still had negative raw STOP margin, with
elite-gap and elite-spread conditions blocking early stopping. The cap ended
those solves.

**Hypothesis:** convergence can occur in an unsuccessful basin, while a usable
plan may exist before the optimizer becomes stationary. This motivates checking
plan adequacy rather than assuming that optimizer convergence identifies the
right stopping time. It does not establish a reliable failure detector.

## 5. Completed fixed-depth sweep on the same six tasks

Evidence: [notebook-derived depth-sweep snapshot](../results/wall-pilot-001/depth_sweep/).
The source is the executed `03_wall_frozen_stopper_diagnostics (2).ipynb` at
execution count 21. Its earlier ZIP export ran at count 17 and therefore did not
contain the new per-depth result JSONs. The CSVs in this repository preserve the
visible aggregate table and episode labels, with precision/provenance explicitly
recorded. Missing raw result JSONs have not been invented.

| Method | Successes | Mean depth | Mean utility |
|---|---:|---:|---:|
| Fixed 3 | 3/6 | 3 | 0.495000 |
| Fixed 5 | 4/6 | 5 | 0.658333 |
| Fixed 10 | 4/6 | 10 | 0.650000 |
| **Fixed 15** | **5/6** | **15** | **0.808333** |
| Fixed 20 | 5/6 | 20 | 0.800000 |
| Fixed 25 | 5/6 | 25 | 0.791667 |
| Fixed 30 | 5/6 | 30 | 0.783333 |
| First frozen Astra controller | 5/6 | 28.8333 | 0.785278 |

The identity of the successful episodes also remains the same from 15 through
30 at the tested depths. Fixed 15 uses **50% fewer iterations than fixed 30** and
about **47.98% fewer than the first Astra controller**, with identical pilot
success labels. It is the strongest tested fixed budget under our declared
utility, not a demonstrated optimal or generally sufficient budget. Depths 11–14
were not tested.

Earliest successful *tested* depths:

| Episodes | Earliest successful tested depth |
|---|---:|
| 1, 2, 5 | 3 |
| 0 | 5 |
| 3 | 15 |
| 4 | None among 3, 5, 10, 15, 20, 25, 30 |

This variation motivates adaptive budgets but does not prove the eight features
can predict them. A controller with an extra eight-iteration wait cannot exploit
the already observed depth-3 or depth-5 opportunities.

**Important distinction:** utility peaks at tested depth 15 and falls thereafter
because of the compute penalty. **Task success does not decline over this sweep.**
We have not reproduced the hypothesized high-then-low success pattern. So far,
the measured opportunity is removing unnecessary planning, not preventing a
demonstrated decline in binary success due to overplanning.

The sweep's fixed-30 and WSP times are cached from an earlier evaluation session.
Do not convert their comparison with the new depth-15 timings into a precise
contemporaneous speedup claim. Logical iteration savings are the stronger result.

## 6. Follow-up: two prompts, three constructions each

This study was designed after examining the pilot. It is a **post-pilot study**,
not an untouched replication. The pilot tasks are development diagnostics; the
new tasks use environment seeds starting at 3000 and planner seeds at 2000000.

The frozen [study definition](../results/wall-prompt-study-001/study.json) declares:

- Original prompt, copied verbatim from the pinned source.
- The same prompt plus explicit marginal-utility framing: consider whether another
  iteration's expected benefit exceeds its compute penalty; convergence is not
  required for stopping and does not guarantee success.
- Three fresh construction contexts per prompt; no prior generated controller or
  evaluation outcome is given to the next attempt.
- Same `gpt-6-astra` request, `xhigh`, 178-parameter architecture, features,
  specification, tools, 40-call/200,000-character inspection budgets, 45 response
  turns and 32,768 output tokens per response.
- One final submission per attempt; failures and duplicates retained, no repair.
- All six construction slots settled before any test evaluation.
- Twelve fixed-layout held-out start/goal episodes shared across six controllers
  and fixed-15/fixed-30 baselines: 96 method/episode jobs if all are valid.
- One fresh subprocess per method/episode for identical setup and warm-up and
  episode-level resumption; loading and warm-up are outside planning timing.
- Frozen rotating method order; same task and initial planner seed for all methods.

The original prompt already specified the utility. The added paragraph changes
how that same objective is articulated, not the objective or success definition.
No pilot score or preferred stopping depth is provided to Astra. The fixed-15
baseline was selected from the pilot and must be assessed on the new tasks.

Prompt hashes:

- Original: `d49fcf5eb9449832fbd5cf6458b5f5476e415e12975352d82acde6d83c02fa07`
- Marginal framing: `92bcadeb8198faa011f9146c5777890527a21304520401a0cbb824360f9b069a`

Active study-helper hash:
`d4bd15761f3932c778b8bf17039ec033e3f2d3b1ec667b523089774efdab6ad1`.
Six distinct weight hashes do not establish functional diversity or statistical
independence. No API sampling seed is assumed available.

## 7. Generation findings: complete cohort, partial performance

Evidence: [prompt-study snapshot](../results/wall-prompt-study-001/), including
weights, construction receipts, compact inspection audit and status CSV.
All six attempts were accepted, with three unique numerical weight hashes in
each arm. The API response model field was `gpt-6-astra` throughout the archived
responses. Original numerical submissions were unchanged in accepted/evaluated
weights; one final submission and one acceptance were recorded per attempt.

| Attempt | Inspection calls | Raw tensor reads | Contributing hidden units | Nonzero parameters |
|---|---:|---:|---:|---:|
| original_r01 | 7 | 0 | 9 | 38 |
| original_r02 | 8 | 0 | 9 | 38 |
| original_r03 | 10 | 2 | 9 | 38 |
| marginal_utility_r01 | 7 | 0 | 16 | 47 |
| marginal_utility_r02 | 12 | 2 | 16 | 44 |
| marginal_utility_r03 | 11 | 0 | 16 | 43 |

Four attempts used source inspection only: there were no statistics or spectrum
calls hidden behind their zero raw-read counts. The remaining two read the same
100 action-encoder and 20 proprioception-encoder values. None requested numerical
predictor or visual-encoder weights. Budgets were not exhausted.

**Structural observation across this cohort:**

| Property | All three original-prompt controllers | All three reframed controllers |
|---|---|---|
| Normalized remaining-cost feature | Ignored: incoming column is zero | Used in the algebraic decision rule |
| Additional minimum wait | 10, 8, 8 iterations respectively | No extra minimum beyond the host's iteration-3 guard |
| Program shape | Tight convergence gates | Weighted continuation/opportunity proxies |
| Network capacity used | Nine contributing hidden units | All 16 contributing hidden units |

The architecture is identical in both arms. These are coherent, manually reasoned
programs encoded numerically, not evidence of learned/calibrated expected success
probabilities. Astra's visible final comments explicitly describe heuristic rules
and unmeasured performance.

The pattern is evidence that wording can influence feature use, decision structure
and use of available capacity. It is not proof that the reframing systematically
improves stopping performance. Inspection choices were self-selected: two attempts
reading tensors is not a randomized checkpoint-access ablation.

## 8. First new episode: all eight methods succeed

Only episode 0 has been evaluated. It is a new start/goal task, not one of the
original six. The individual results are in `evaluations/episode_000/`.

| Method | Iterations | Success | Final distance | Planning seconds | Utility |
|---|---:|---|---:|---:|---:|
| Fixed 15 | 15 | Yes | 0.469271 | 260.020614 | 0.975 |
| Fixed 30 | 30 | Yes | 0.286364 | 507.305838 | 0.950 |
| original_r01 | 30 | Yes | 0.286364 | 517.900028 | 0.950 |
| original_r02 | 30 | Yes | 0.286364 | 519.333300 | 0.950 |
| original_r03 | 30 | Yes | 0.286364 | 520.492762 | 0.950 |
| marginal_utility_r01 | 30 | Yes | 0.286364 | 518.469133 | 0.950 |
| marginal_utility_r02 | 30 | Yes | 0.286364 | 518.852686 | 0.950 |
| marginal_utility_r03 | 27 | Yes | 0.213208 | 465.704418 | 0.955 |

On this one task, fixed 15 has the highest declared utility. Reframed r03 saves
10% of iterations relative to fixed 30 and has an observed 8.20% planning-time
saving, but uses 80% more iterations than fixed 15. These are single-episode
observations, not stable latency or prompt-arm estimates.

The slightly better final distance at 27 than 30 does not change binary success;
all distances are well inside the 4.5 radius. It is insufficient evidence of a
general overplanning problem. Five distinct networks used the same full depth.
Different weights need not imply different behavior on a given input history.

The six generated controllers have identical recorded feature prefixes until
their respective stopping points. All **177** recorded decisions were independently
reproduced from the frozen weights. Original controllers and reframed r01 still
have negative raw STOP margins at 30. Reframed r02 has a positive raw margin at
30, so its network would also choose STOP there; reaching the cap alone does not
mean every controller still prefers continuing.

### Mechanistic explanation of marginal_utility_r03

Let `c` be normalized remaining cost and use the other feature symbols from
section 4. This controller compares a continuation score with a cost-dependent
stopping threshold:

```text
T(c) = 1 + 2*[1-c]+ + 6*[0.5-c]+ + 32*[0.25-c]+
         + 160*[0.1-c]+ - 200*[0.05-c]+

C = 600*max(abs(r), abs(p)) + 300*g + 150*v + q + 5*d
    + 2*[q-t]+ + 4*[q-0.5]+ + 20*[d-0.1]+

STOP when T(c) > C, subject to host iteration guards.
```

For `c <= 0.05`, `T(c)=20`. At iteration 26, its continuation score is about
36.58; at iteration 27 it falls to 19.75 while `c` is about 0.0105. The score
crosses 20 and the network stops. The main iteration-27 contributions are recent
cost changes 11.55, elite gap 6.02 and elite spread 1.98. This explains the stop;
it does not establish optimality or empirical calibration.

Algebraic reductions for all six submissions matched their decimal coefficients.
Small float32 rounding differences in margins were checked separately; recorded
Boolean decisions matched exactly. Margins are not calibrated confidence scores
and should not be compared across differently scaled networks as probabilities.

## 9. Engineering issues uncovered and corrected

| Issue | Diagnosis and resolution | Experimental effect |
|---|---|---|
| Missing starter modules in an early local install | Restored missing package files; already present in reference commit | Installation repair, not a controller result |
| Different Colab accounts saw missing results | Same MyDrive pathname did not share files; exported/imported actual files | Preserved identical task/weight bytes; no regeneration needed |
| Empty Drive ZIP | Archive contained only a directory; later exporter writes and verifies actual files | The empty archive supplied no result evidence |
| Flags left False | Setup succeeded but generation/evaluation were skipped; edit the existing flag cell | No generation or test result should be inferred from setup alone |
| Notebook 03 on a fresh VM | `prepare` downloaded assets but did not create `wall.eval.yaml` or apply required patches; added base-packet `freeze` and verification | Pre-evaluation setup correction; accepted weights unchanged |
| Hidden subprocess error | Colab showed only CalledProcessError; stream/retain underlying output | Better diagnostics, no change to planner semantics |
| Notebook 04 source guard | Eight bare filenames were compared with notebook 02's eleven relative-path entries; matched the exact inventory and added useful mismatch errors | Fixed before study generation; source-integrity checks retained |
| Export before completed sweep | Saved ZIP preceded sweep execution | Notebook-derived evidence labeled explicitly; raw missing JSONs not invented |
| Export source filter | Partial study ZIP omitted `.py`/`.yaml` packet sources as well as intentionally omitted tensor arrays | Policy audit possible; full packet verification needs the complete active packet |

The fingerprint fix was checked against the actual pilot protocol, not just a
fixture that generated its expectation using the same incorrect helper. The repo
update adds standard-library notebook integration tests to guard this boundary.
All new notebook deliveries have outputs cleared. Their embedded evaluator and
the active study-helper bytes are preserved; no ongoing study is rewritten.

## 10. Conclusions supported now

1. Training-free numerical controller construction worked, producing valid,
   auditable stopping networks without a training loop.
2. The initial convergence controller missed much of the demonstrated compute
   opportunity. Fixed 15 beats it under pilot utility with the same success labels.
3. Required planning varies across the pilot tasks, motivating adaptive policies,
   but the available features' ability to identify sufficient depth is unproven.
4. Optimizer stationarity and task adequacy are distinct. The observations motivate
   studying that distinction; they do not establish a reliable failure detector.
5. Prompt framing is associated with a consistent structural difference across
   three constructions per arm. Performance benefit remains undecided.
6. Numerical checkpoint inspection was sparse and sometimes absent. No causal
   benefit from checkpoint access has been established.
7. The hypothesized high-then-low binary-success curve has not been reproduced.
8. Fixed budgets, failed generations, multiple constructions, source identity and
   matched evaluation conditions are essential to interpreting any improvement.

## 11. Outstanding evidence and next steps

- Finish the existing prompt study's remaining **88 jobs**, keeping prompts,
  weights, tasks, architecture, budgets and planner settings unchanged.
- Recompute summaries after all methods have their 12 episodes. The current arm
  table deliberately leaves fully evaluated-controller metrics empty.
- Report every construction, between-controller variation, unique weight hashes,
  success, utility, iteration/work counts and paired wins/losses against both fixed
  budgets. Do not retain only the best controller.
- Treat same-session latency pairs as descriptive until timing is repeated;
  numerical/platform differences and load can affect results.
- Export the latest completed sweep JSONs if available, replacing derived evidence
  only with clearly identified originals and preserving the earlier provenance.
- Notebook 03's separate 12-episode held-out section and repeated-timing section
  were **not run** in the uploaded diagnostic notebook. Do not count them as evidence.
- A source-only versus checkpoint-access study would require deliberate assignment
  of access, not comparison of self-selected inspection behavior.
- More features, alternative utility penalties, rule-to-weight compilation,
  multiple MPC solves or feedback-assisted repair are separate follow-up conditions
  requiring new frozen specifications and suitable fresh evaluations.
- The originally discussed learned/trained stopper remains a separate,
  unimplemented arm. No learned-stopper comparison has been reported.

## 12. Provenance and references

| Evidence source | What it supplies |
|---|---|
| `wall-pilot-001-diagnostics-001-20261118T045632Z.zip` | Original pilot, accepted controller, traces and generation audit |
| `03_wall_frozen_stopper_diagnostics (2).ipynb` | Completed depth-sweep console and aggregate outputs |
| `wall-prompt-study-001-20261119T004357Z.zip` | Six constructions and first eight evaluation jobs |

Their hashes are retained in the corresponding `SNAPSHOT.json` or
`PROVENANCE.json`. Full raw archives remain separate from the curated repo files.
The initial empty archive and unavailable attachments were not used as evidence.
The GitHub reference tree was checked at the pinned commit when preparing this
update. New documentation/test commits should **not** be substituted into the
active study's `PROJECT_REF`.

External context, not a substitute for our measurements:

- [Weight-Space Programming write-up](https://rikiyat.github.io/blog/weight-space-programming/)
- [Authors' released experiment logs](https://github.com/RikiyaT/lm_understand-logs)
- [Meta JEPA world-model implementation](https://github.com/facebookresearch/jepa-wms)
- [Released model assets](https://huggingface.co/facebook/jepa-wms)
- [Original DINO-WM implementation](https://github.com/gaoyuezhou/dino_wm)
- [Knowing When to Stop Thinking](https://arxiv.org/abs/2610.28011)

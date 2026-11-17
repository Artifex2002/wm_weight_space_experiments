# Experimental protocol

This repository implements controller construction, checkpoint inspection, and
controller inference. It does not claim to reproduce the stopping paper or
include a working DINO-WM Wall simulator integration. The released Meta baseline
checkpoint is not established as the checkpoint used by that paper.

## Information boundary

Preparation is performed by a trusted human/host. It may download/load the model
and verify engineering correctness. Do not expose these checks or empirical
planning results to the generator. Model generation runs on a NEW isolated API
conversation, not a continuation of a discussion containing published curves.
Every run records prompt/packet hashes and the model identifier.

Agent access: list tensors, bounded weight slices, weight statistics, bounded
submatrix singular values, and allowlisted source. There is no eval, shell,
arbitrary Python, network, filesystem browsing, environment step, or model
forward tool. The host's own process is trusted. This API restriction is not an
OS sandbox against someone who controls the host. Publish no secrets in packet
source files. Use a separate process/container without evaluation data if
stronger operational separation is needed.

All packet tensors are verified by hash at generation start and exposed as
read-only NumPy arrays. Do not edit packet files during a run. The tools cannot
write them. Packet hashes establish integrity relative to the frozen manifest,
not authenticity of the original source.

The original checkpoint remains local. Only selected slices/statistics and
source excerpts are sent to OpenAI. API requests use store=false and carry
reasoning encrypted content forward as described by the Responses API. This
setting does not itself promise zero provider retention. Audit logs contain API
outputs, usage, tool arguments and returned data; no Authorization headers.

## One final submission

A run can contain many inspection turns and exactly one final submit_weights
call. Wrong shapes/nonfinite values end the run without repair. API failure,
output truncation, missing submission, and budget exhaustion also count as
failed attempts. No automatic billable POST retries. Do not select attempts by
test success. Declare the number of independent generations before evaluation.

A call budget is not a dollar cap. Each request includes accumulating context.
Set max turns, per-response output tokens, inspection calls, and tool-output
characters; inspect API usage and configure account spending controls as needed.
Model/effort support varies. Set the exact API model ID available to the account;
this repository does not assume that the name 'Astra' maps to an accessible ID.

## Checkpoint scope

Verify whether the chosen checkpoint embeds the frozen DINOv2 encoder or only
the learned predictor. A flat state dict can include predictor/action encoder
weights and buffer tensors. If the visual encoder is loaded separately, obtain
and freeze its state dict too or explicitly label predictor-only access. Do not
claim full world-model access until every inference component is included.
Decoder/optimizer tensors are not necessary for latent planning; document any
omissions. No random tensors may stand in for missing pretrained components.

freeze currently takes one flat state dictionary. To combine separate state
dicts, the trusted preparer can load each with weights_only=True and save a flat
mapping with prefixes such as encoder.* and predictor.* using torch.save. Record
both input hashes and the merge procedure in provenance. This is packaging,
not changing numerical values. bfloat16 is widened exactly to float32 on export.

## Evaluation (separate process; no tool from generator)

1. Load one fixed checkpoint and the exact matching configuration/source commit.
2. Implement the hook described in INTEGRATION.md and freeze the source packet.
3. Test setup on development episodes; retain outcomes outside generator packet.
4. Generate controllers in fresh runs, retaining every attempt.
5. Freeze weight hashes before evaluating on predeclared held-out episodes.
6. Compare fixed budgets, a cost-stall heuristic, a learned controller, and WSP.
   Learned and WSP controllers use identical architecture/features/min/max depth.
7. To isolate weight access, also compare WSP with tensors disabled and the same
   source information. This ablation needs a separate tool list/manifest view;
   the supplied runner is the weight-enabled condition only.

Keep candidate count, horizon, model, action chunk, RNG method, candidate ranking,
and final-plan selection fixed. Record actual transition/model evaluation counts
and total episode compute, not depth alone. Reproducible RNG streams should be
keyed by episode, MPC decision and CEM iteration to reduce unnecessary divergence.

Report success, total CEM iterations, model work, synchronized wall-clock planning
time, p50/p95 latency, controller overhead, failure rate, independent generation
variance, and construction tokens/cost. Bootstrap across independent episode/seed
units as appropriate; do not treat CEM iterations as independent episodes.
The published aggregate optimal depth is not a per-state label.

## Sources

- https://developers.openai.com/api/docs/guides/function-calling
- https://developers.openai.com/api/docs/guides/conversation-state
- https://huggingface.co/facebook/jepa-wms
- https://github.com/facebookresearch/jepa-wms
- https://github.com/gaoyuezhou/dino_wm
- https://arxiv.org/abs/2610.28011
- https://rikiyat.github.io/blog/weight-space-programming/

The controller design, feature formulas, compute penalty, tool limits, and API
runner here are proposed experimental choices, not recovered author settings.

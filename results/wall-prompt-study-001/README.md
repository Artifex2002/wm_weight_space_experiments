# Two-prompt study: first evaluation block only

All six generation attempts were accepted, with six distinct numerical weight
hashes. Each prompt has three attempts. Generation used separate contexts and
no test feedback; no API sampling seed or statistical independence is asserted.

**Coverage: one new task, eight methods, 8/96 evaluations.** The remaining eleven
tasks have no measured outcomes in this snapshot. `arm_summary.csv` correctly
leaves performance aggregates across fully evaluated controllers empty.

The two baselines are fixed 15 and fixed 30. Every method succeeded on episode 0;
fixed 15 used the fewest iterations. `marginal_utility_r03` stopped at 27 and all
other generated controllers reached 30. This is not a completed prompt comparison.

`generation/` retains numerical weights, construction settings and submission
receipts. `inspection_audit.json` summarizes actual inspection calls and visible
model comments. `packet_manifest.json` is a tensor/source inventory only; it is
not a runnable packet. Full packet arrays and original API events remain outside
this curated snapshot. The original export filter also omitted four `.py`/`.yaml`
packet source files; see `SNAPSHOT.json`. This does not prevent auditing the
submitted policies, but prevents a full independent packet-content verification
from this directory alone.

Individual result JSONs retain trajectories, feature traces and timing. Receipt
hashes bind them to the study's tasks and controller weights. Keep the study,
weights, prompts and settings frozen while finishing its remaining evaluations.

# Completed pilot depth sweep: notebook-derived evidence

The uploaded executed notebook reported all six tasks at fixed depths
3, 5, 10, 15, 20 and 25. It reused the earlier fixed-30 and WSP results.

`episode_success.csv` preserves those observed labels. Console planning seconds
for the new depths are rounded to two decimals; the earlier depth-30 rows come
from the original JSON. `summary_from_notebook.csv` preserves the aggregate
planning times at the six-decimal precision displayed in the notebook. Utility
is recomputed from the exact counts/depths using the declared formula.

The underlying per-depth JSONs were not uploaded: the earlier ZIP export ran
before the sweep. They have not been fabricated. See `PROVENANCE.json` for source
identity and execution counts. A later export can add the original JSONs.

Fixed 15 is the best *tested* depth under mean utility on these six tasks. Depths
11–14 were not tested. Success does not decrease over the tested depths. Do not
claim a contemporaneous latency speedup against the cached depth-30/WSP runs.

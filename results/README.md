# Curated experimental evidence

These are deliberate, small research snapshots for version control. They contain
controller weights, frozen prompts/protocols, task manifests, metrics and selected
execution evidence. Full checkpoint binaries, numerical inspection packets, raw
API event streams and copied third-party source are not included.

| Directory | Coverage | Status |
|---|---|---|
| `wall-pilot-001/` | One controller and six original tasks; completed fixed-depth sweep | Pilot complete; sweep raw JSONs were not supplied, so that subsection is derived from executed notebook displays |
| `wall-prompt-study-001/` | Six new controllers, two prompts, 12 frozen tasks | **Partial: only episode 0 evaluated for eight methods, 8/96 jobs** |

Read [the experiment log](../docs/EXPERIMENT_LOG.md) before interpreting the data.
Each snapshot records the source archive and SHA-256. Original field values and
bytes are retained where files were copied from archives. The depth-sweep CSVs
are explicitly labeled as notebook-derived, with their source notebook's hash.

The active study remains on Google Drive. These snapshots do not replace Drive
receipts or restore a runnable inspection packet. Keep the active `PROJECT_REF`
at the commit recorded by the protocol, even after committing these results.
Controllers were frozen before the study's test evaluations. Do not feed these
results into the same construction attempts or replace unfavorable attempts.

Raw archived events include inspected source excerpts and encrypted reasoning;
retain the originals separately. `inspection_audit.json` provides tool names,
arguments, counts and visible final messages without republishing those streams.

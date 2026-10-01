# Post-M23 single-measurement contamination register

Classes apply to a prompt paired with the one tested cell, `M22.1-D1-L15`, and the one tested measurement, `pre_dot`.

`CLEAN`: the prompt is in the new `s-*` catalog, and no outcome for that prompt has been written under M23, M23-Diagnostic, M23-D2, M23-E, M23-F, or M23-G.

`CONTAMINATED`: an outcome for that prompt at this cell, or for that prompt at the D1/D2 early-layer family, was already written.

`PARTIALLY_CONTAMINATED`: that prompt's outcome was not written for this cell, but the prompt belongs to the old M22.1 catalog and was not reserved before other splits of that catalog were inspected.

## New catalog

All 36 `s-*` prompts paired with `M22.1-D1-L15` are `CLEAN`. The matrix lists each pair. No `s-*` id is present in the existing outcome files checked by `scripts/validate_post_m23_single_measurement.py`.

## M23-G catalog at this cell

All 36 `g-*` prompts paired with `M22.1-D1-L15` are `CONTAMINATED`. M23-G measured the alpha-`+1` effect on every partition of that catalog at this cell, and the episodes log `pre_dot`. The M23-G register called those pairs `CLEAN` before that run. That earlier label is not reused here.

## Old M22.1 catalog at this cell

| old role | pair status | why |
| --- | --- | --- |
| discovery (`*-01`, `*-04`) | `CONTAMINATED` | M22.1 preflight and the M23-F screen measured D1 at layer 15 on these prompts. |
| validation (`*-02`, `*-05`) | `CONTAMINATED` | MRSM Q used validation as a holdout for D1 and D2 at layers 0, 8, 15, and 23. |
| replication (`*-03`, `*-06`) | `PARTIALLY_CONTAMINATED` | These strings were not the M23-F forward for layer 15. They were not reserved before discovery and validation were inspected, and the same strings were measured for D1 at layer 23. |

No old pair and no `g-*` pair is `CLEAN`.

## Cells that are not in this protocol

D1 at layer 23, D2 at every layer, D1 at layers 0 and 8, and D3–D8 are not tested. They are not given a `CLEAN` row. Measuring them on the `s-*` catalog would be a different experiment, and it would spend prompts this protocol has reserved for one cell and one measurement.

## What this register does not do

It does not rank measurements. It does not treat a `g-*` row as a training row for the new slope. It does not convert a `CLEAN` pair into a result. Those pairs have no outcomes yet.

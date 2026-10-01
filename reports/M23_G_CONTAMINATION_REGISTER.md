# M23-G contamination register

Classes apply to an intervention/prompt pair. A prompt that was inspected under a different intervention is not marked `CLEAN` for a new cell.

`CLEAN`: no outcome has been measured for that pair, and the prompt is not in the old M22.1 catalog.

`CONTAMINATED`: an outcome for that pair was already written, on the discovery screen, the M22.1 preflight, or the MRSM Q validation holdout.

`PARTIALLY_CONTAMINATED`: that pair's outcome was not written, but the prompt belongs to the old catalog and the cell's other splits were already inspected. The pair was not reserved before those inspections. It is not fresh.

## Frozen family on the new catalog

All 36 `g-*` prompts paired with all six cells are `CLEAN`. The new catalog has no outcomes. The matrix lists each pair.

The six cells are `M22.1-D1-L0`, `M22.1-D2-L0`, `M22.1-D1-L8`, `M22.1-D2-L8`, `M22.1-D1-L15`, and `M22.1-D2-L15`.

## Frozen family on the old catalog

| old role | pair status | why |
| --- | --- | --- |
| discovery (`*-01`, `*-04`) | `CONTAMINATED` | M22.1 preflight measured D1 at layers 0, 8, and 15. M23-F measured D1 and D2 at those layers. MRSM Q measured both directions on discovery texts. |
| validation (`*-02`, `*-05`) | `CONTAMINATED` | MRSM Q used validation as its holdout for D1 and D2 at layers 0, 8, 15, and 23. |
| replication (`*-03`, `*-06`) | `PARTIALLY_CONTAMINATED` | These prompts were not forwarded for the early-layer cells. They were also not reserved before discovery and validation were inspected, and the same strings were measured for D1 at layer 23. |

No old pair is `CLEAN`. The new evaluation set does not contain these prompts.

## Cells outside the family

`M22.1-D1-L23` is `CONTAMINATED` on every old prompt, including replication. It is not in the family.

`M22.1-D2-L23` and `M22.1-D3-L23` through `M22.1-D8-L23` are `CONTAMINATED` on every old split. They are not in the family. Extending them to a new layer would be a new intervention and is not part of this protocol.

## What this register does not do

It does not rank cells. It does not treat a replication string as fresh for an early layer. It does not convert a `CLEAN` new pair into a confirmed effect. Those pairs have no outcomes yet.

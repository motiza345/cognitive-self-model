# M30 catalog audit

**Status:** PASS
**Catalog:** `scripts/m30_catalog.py`
**Catalog SHA-256:** `4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770`

## Structure

- Total prompts: **48**
- UPDATE: **12**
- VALIDATION: **12**
- HOLDOUT: **24**
- Families: completion / syntax / instruction
- UPDATE and VALIDATION contain 4 prompts from each family.
- HOLDOUT contains 8 prompts from each family.
- Partition intersections: **0**

## Historical catalogs

Exact identifier overlap, exact text overlap, and whitespace-collapsed case-folded text overlap were checked against M22.1 through M29-D.

| Historical source | Exact text overlap | Normalized text overlap | ID overlap |
| --- | ---: | ---: | ---: |
| M22.1 | 0 | 0 | 0 |
| M23-G | 0 | 0 | 0 |
| POST-M23 single measurement | 0 | 0 | 0 |
| Feasibility gate | 0 | 0 | 0 |
| M24 | 0 | 0 | 0 |
| M26 | 0 | 0 | 0 |
| M29-D | 0 | 0 | 0 |

## Inside M30

The twelve M30-HX1 holdout prompts were compared with UPDATE and with VALIDATION on exact id, exact text, and whitespace-collapsed case-folded text.

| Comparison | Exact text overlap | Normalized text overlap | ID overlap |
| --- | ---: | ---: | ---: |
| New holdout vs UPDATE | 0 | 0 | 0 |
| New holdout vs VALIDATION | 0 | 0 | 0 |

## Outcome leakage

- Qwen loaded: **false**
- M30 outcomes loaded: **false**
- M29 holdout statistics loaded: **false**
- Feature selection from M29 cell scores: **false**

This is a pre-outcome catalog audit.

## Frozen partition rule

For each family, index 1..12 is assigned `(index - 1) mod 3` to UPDATE, VALIDATION, HOLDOUT. Indices 13..16 are HOLDOUT by rule M30-HX1.

## Scientific status

**CATALOG_DISJOINTNESS_PASS**

The catalog is eligible for the M30 design lock. No scored run is authorized by this audit.

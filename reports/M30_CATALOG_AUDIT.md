# M30 catalog audit

**Status:** PASS
**Catalog:** `scripts/m30_catalog.py`
**Catalog SHA-256:** `8d88819766cf7b454edafecb1582b6284dacea8a8d16f14f2f872090a17316a9`

## Structure

- Total prompts: **36**
- UPDATE: **12**
- VALIDATION: **12**
- HOLDOUT: **12**
- Families: completion / syntax / instruction
- Each partition contains 4 prompts from each family.
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

## Outcome leakage

- Qwen loaded: **false**
- M30 outcomes loaded: **false**
- M29 holdout statistics loaded: **false**
- Feature selection from M29 cell scores: **false**

This is a pre-outcome catalog audit.

## Frozen partition rule

For each family, index 1..12 is assigned `(index - 1) mod 3` to UPDATE, VALIDATION, HOLDOUT.

## Scientific status

**CATALOG_DISJOINTNESS_PASS**

The catalog is eligible for the M30 design lock. No scored run is authorized by this audit.

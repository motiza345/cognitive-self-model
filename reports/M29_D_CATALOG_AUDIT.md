# M29-D Catalog Audit

**Status:** PASS  
**Catalog:** `scripts/m29d_catalog.py`  
**Catalog SHA-256:** `d6ecab612a2214d947be11c82671612f883c0d13e479b07e8cbaa09b635d4d77`

## Structure

- Total prompts: **36**
- UPDATE: **12**
- VALIDATION: **12**
- HOLDOUT: **12**
- Families: completion / syntax / instruction
- Each partition contains 4 prompts from each family.
- Partition intersections: **0**

## Historical catalogs audited

The M29-D texts and identifiers were checked against every catalog consumed by M22-M26 that is available in the canonical repository:

1. M22.1 frozen prompts
2. M23-G catalog
3. POST-M23 single-measurement catalog
4. project feasibility-gate catalog
5. M24 consumption catalog
6. M26 independent-evidence-update catalog

### Results

| Historical source | Exact text overlap | Normalized text overlap | ID overlap |
|---|---:|---:|---:|
| M22.1 | 0 | 0 | 0 |
| M23-G | 0 | 0 | 0 |
| POST-M23 single measurement | 0 | 0 | 0 |
| Feasibility gate | 0 | 0 | 0 |
| M24 | 0 | 0 | 0 |
| M26 | 0 | 0 | 0 |

Normalization used for the second check:
- whitespace collapsed;
- case folded.

The audit was performed against the committed source of each historical catalog before any M29-D outcome was generated.

## Outcome leakage

- Qwen loaded: **false**
- intervention outcomes loaded: **false**
- residuals loaded: **false**
- holdout statistics loaded: **false**
- feature selection from outcomes: **false**

This is therefore a **pre-outcome catalog audit**.

## Frozen partition rule

For each family, index 1..12 is assigned:

`(index - 1) mod 3`

to:

`UPDATE, VALIDATION, HOLDOUT`

This guarantees 4 prompts per family in each partition.

## Scientific status

**CATALOG_DISJOINTNESS_PASS**

The new catalog is eligible for the next M29-D execution-freeze step.

This audit establishes disjointness at the catalog identity/text level. It does not claim semantic independence from every possible linguistic concept represented in historical prompts; such a claim would require a separately specified semantic-contamination test and is not part of the current M29-D protocol.

# Project control plane

Machine-checkable ledger for scientific state, claims, artifact provenance, and reconstruction pointers.

YAML files in this directory are the source of truth. Markdown here is an entry point only.

## Read order

1. `STATE.yaml` — current statuses, gates, decisions, next milestone
2. `CLAIMS.yaml` — finding and observation ledger
3. `FILE_MAP.yaml` — every tracked file from the Phase 1 inventory
4. `RECOVERY.yaml` — reconstruction pointers and recovery grades

## Source of truth

If prose and YAML disagree, the YAML is authoritative. Do not duplicate claim text or status tables in this README.

## Validator

From the repository root:

```
PYTHONPATH=src python -m cognitive_self_model.control_plane.validate
```

Exit `0` if all checks pass; exit `1` and print errors otherwise.

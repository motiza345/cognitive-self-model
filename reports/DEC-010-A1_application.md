# DEC-010-A1 application

Status: APPLIED

Timestamp: 2026-09-29T16:55:00Z

Commit: `ff022eba5e5e67b80471b6c1f0f54f6f0bb382d0`

That commit applies the amendment. It is not `p_freeze`. `mrsm.clock_start_commit` stays null because construction did not reach `DONE`.

Reason: Q.H1 cannot be read as verified mechanism discovery. P2.1 proposed the clause and P2.2 restated it without applying it. P2.3 applies that interpretation only.

Unchanged:

- FIRST_MATCH rule text
- eight gate keys
- allowed statuses PASS, FAIL, NOT_EVALUATED
- terminal outcome names and the legal/fully-evaluated counts

Applied text, appended to `control_plane/DEC-010_decision_table.md`:

```text
amendment_DEC-010-A1:
  status: APPLIED
  applied_on: 2026-09-29T16:55:00Z
  reason: "Q.H1 cannot be read as verified mechanism discovery. P2.1 proposed this clause and P2.2 restated it. P2.3 applies that text only. FIRST_MATCH, the eight gate keys, the allowed statuses, and the terminal outcome names are unchanged."
  modifies_first_match: false
  modifies_gate_keys: false
  modifies_terminal_outcomes: false
  interpretation:
    - "P.H1 is ground-truth mechanism verification against the frozen P mechanism."
    - "Q.H1 is diagnostic/transfer evidence on Q. It is not ground-truth mechanism verification."
    - "Q.H1 PASS is not a claim that the true mechanism on Q was discovered."
    - "Q.H1 FAIL is not proof that mechanism discovery is impossible on real models."
    - "REDEFINE_SCALE after a Q.H1 FAIL retires the Q transfer claim for this benchmark only."
```

Exact diff is the addition of that block after `else: CONTRACT_ERROR`. No rule line was edited.

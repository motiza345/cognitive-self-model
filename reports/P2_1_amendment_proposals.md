# P2.1 amendment proposals

Only amendments that are required to keep a frozen rule from being misread are listed. None are applied. `DEC-010` and `DEC-011` were not edited.

Decisions 1, 3, and 5 are settled by the existing frozen text. They need later specification conformance. They do not need a DEC amendment.

Decision 4 does not get an amendment in this file. Choosing an H3 family would be a new scientific decision. It stays `DECISION_REQUIRED`.

## DEC-010-A1

```text
Amendment ID
DEC-010-A1

Target DEC
DEC-010 (control_plane/DEC-010_decision_table.md), decision_table_version 1, status FROZEN

Current frozen rule
terminal_decision.evaluation is FIRST_MATCH.
Gate keys are P.H1, P.H2, P.H3, P.H4, Q.H1, Q.H2, Q.H3, Q.H4.
Allowed statuses are PASS, FAIL, NOT_EVALUATED.
GO if and only if P.H1–H4 and Q.H1–H4 are all PASS.
If P.H1–H4 are PASS and every Q key is evaluated and any Q key is FAIL, the outcome is REDEFINE_SCALE.
A missing key or an illegal status is CONTRACT_ERROR.
The table does not define what PASS means for Q.H1.
DEC-011 records Q ground_truth as UNKNOWN and does not modify DEC-010.
DEC-011 defers metrics and thresholds for every gate, including Q.H1.

Scientific reason
P ground truth is the frozen mechanism in P_SPEC.md. Q ground truth is UNKNOWN.
P.H1 can be a truth-validating comparison to that planted mechanism.
Q.H1 cannot mean that Q discovered the true mechanism. That reading is epistemically invalid.
DEC-010 currently gives Q.H1 PASS the same formal place in the GO conjunction as P.H1 PASS.
Leaving the distinction out of DEC-010 would let a later preregistration, or a later report, treat Q.H1 PASS as verified mechanism identity without amending the frozen table.
Q.H1 FAIL under rule 6 is REDEFINE_SCALE. That outcome must not be readable as proof that mechanism discovery is impossible in principle.
No independently grounded criterion for that stronger reading exists in a frozen source.

Proposed new rule
Do not change FIRST_MATCH, the eight keys, the allowed statuses, or any outcome name.
Add a binding interpretation clause to DEC-010:

P.H1 PASS is a comparison of the discovery output to the frozen P ground truth.
Q.H1 PASS is a pass of the preregistered diagnostic or transfer criterion only.
Q.H1 PASS is not verified mechanism identity.
Q.H1 FAIL is a failure of that diagnostic or transfer criterion.
Q.H1 FAIL is not, by itself, proof that the method cannot discover mechanisms.
REDEFINE_SCALE after a Q.H1 FAIL retires the Q transfer claim for this benchmark.
It does not assert that mechanism discovery is impossible in principle.
The preregistration may not define the Q.H1 metric as an F1, or any other score, against a Q mechanism label unless an independently verified external target has been frozen by a DEC. No such target exists.
The numerical threshold for the diagnostic criterion stays deferred until the preregistration step DEC-011 already names. This amendment does not set that number.

Effect on gates
The eight-gate structure remains intact.
P.H1 remains the truth-validating discovery gate.
Q.H1 remains a governance key.
No key is added or removed.
No outcome rule changes its inputs.

Effect on GO eligibility
Unchanged. GO still requires all eight keys to be PASS.
What changes is the claim GO is allowed to make about Q: a Q.H1 PASS inside that conjunction is diagnostic or transfer evidence, not ground-truth mechanism discovery.

Effect on preregistration
The Q.H1 endpoint must be written as a diagnostic or transfer endpoint.
It must not be written as verified mechanism identity.
The numeric bar, the split, and the sample size remain unset by this amendment.
They stay PROPOSED — NOT FROZEN until a later authorized freeze.
P.H1's comparison target remains P_SPEC.md. This amendment does not change P.H1's threshold, which is also still deferred.

Status = PROPOSED / NOT APPLIED
```

No other amendment is proposed.

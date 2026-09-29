# DEC-010-A1 final proposal

Status: `PROPOSED / NOT APPLIED`

This file restates the P2.1 proposal and adds the P2.2 operational reading. It does not edit `control_plane/DEC-010_decision_table.md`.

```text
Amendment ID
DEC-010-A1

Target DEC
DEC-010, decision_table_version 1, status FROZEN

Current frozen rule
FIRST_MATCH over eight keys:
P.H1, P.H2, P.H3, P.H4, Q.H1, Q.H2, Q.H3, Q.H4.
Statuses: PASS, FAIL, NOT_EVALUATED.
GO if and only if all eight are PASS.
If every P key is PASS, every Q key is evaluated, and any Q key is FAIL, the outcome is REDEFINE_SCALE.
A missing key or an illegal status is CONTRACT_ERROR.
The table does not say what Q.H1 PASS means.
DEC-011 sets Q ground_truth to UNKNOWN and does not modify DEC-010.
DEC-011 defers the metric and the threshold.

Scientific reason
P has frozen ground truth in P_SPEC.md. Q does not.
P.H1 can validate discovery against that planted mechanism.
Q.H1 cannot mean that the real model’s discovered mechanism is correct.
Q.H1 cannot mean that the self-model has discovered the true internal mechanism on Q.
The defensible role is diagnostic and transfer evidence: whether the representation-and-prediction procedure produces stable, falsifiable, intervention-linked claims on a model whose mechanism identity is unknown.
A Q result can support transferability. It cannot independently establish mechanism identity.
DEC-010 currently places Q.H1 PASS in the same GO conjunction as P.H1 PASS. Without an explicit clause, that conjunction can be read as verified mechanism discovery on Q.

Proposed new rule
Do not change FIRST_MATCH, the eight keys, the allowed statuses, or any outcome name.
Add this binding interpretation:

P.H1 is a ground-truth-validating discovery gate.
Q.H1 is a diagnostic/transfer gate. It is not a ground-truth mechanism verification gate on Q.
Q.H1 PASS is necessary only insofar as DEC-010’s existing gate logic requires a Q.H1 result.
Its epistemic meaning is limited to the declared diagnostic scope.
Q.H1 PASS is not verified mechanism discovery on Q.
Q.H1 FAIL is not proof that the method cannot discover mechanisms on real models.
Q.H1 FAIL is evidence against the current transfer of the method under the tested scope.
REDEFINE_SCALE after a Q.H1 FAIL retires that transfer claim for this benchmark.
It does not assert a universal impossibility.
The preregistration may not define the Q.H1 metric as a score against a Q mechanism label.
The observable record is the diagnostic vector named in the draft operational contract:
prediction quality, counterfactual consistency, falsification behavior, abstention quality, and scope consistency, with status DIAGNOSTIC_ONLY.
This amendment sets no numerical threshold. Threshold status stays deferred.

Effect on gates
Eight keys remain. No key is added or removed.
P.H1 stays the truth-validating gate.
Q.H1 stays a governance key with a diagnostic meaning.

Effect on GO eligibility
Unchanged. GO still requires all eight keys to be PASS.
The claim attached to the Q.H1 conjunct is limited to the diagnostic scope.

Effect on preregistration
The Q.H1 endpoint must be the diagnostic vector.
It must not be an identity verdict.
Numbers, splits, and sample sizes stay deferred.

Status = PROPOSED / NOT APPLIED
```

Until this amendment is approved and applied, do not record `Q.H1` as `PASS`.

# Belief update on frozen arm P

Diagnostic. Not an MRSM rescore, not a Self-Model, and not M22.

## 1. Question

Can one rule, reading only the frozen P belief and evidence records, tell four
single contradictions apart?

- Decision: `UPDATE_RULE_SEPARATES_FAILURES`.
- The four injected contradictions received four different updates, and the controls matched. This result is local to the frozen P belief and these injections. It does not identify a mechanism. A Self-Model was not modified. Qwen was not loaded.

## 2. What was fixed before scoring

- The rule reads scope, effects, joints, the current ordered pair, uncertainty, and the frozen identifiability constants.
- A record supplies an intervention id, an actual outcome, and a scope string.
- Tolerance is `max(1.0, 2.0 * uncertainty)`, the expression already used by the P falsification record.
- The construction clearance is not a decision input. If the interaction injection has no legal pair, the decision is `INCONCLUSIVE`.
- Injection names and planted ground truth are not inputs.

## 3. Primary injections

| Case | Rule output | Expected |
| --- | --- | --- |
| hypothesis | `REVISE_HYPOTHESIS` | `REVISE_HYPOTHESIS` |
| scope | `REVISE_SCOPE` | `REVISE_SCOPE` |
| interaction | `ADD_INTERACTION` | `ADD_INTERACTION` |
| none | `ABSTAIN` | `ABSTAIN` |

## 4. Controls

| Case | Rule output | Expected |
| --- | --- | --- |
| holdout | `HOLD` | `HOLD` |
| double | `ABSTAIN` | `ABSTAIN` |
| scope_string_alone | `HOLD` | `HOLD` |

## 5. Preconditions

- Reconstruction matches the frozen state: `True` and `True`.
- Recorded tolerance `10.476338585083338` against formula `10.476338585083338`.
- Interaction construction defined: `True`.

## 6. What this does not say

Section 3 is the decision trace. It does not reopen the label.
The four injections are built so that each one supports a different predicate.
On this belief, a legal interaction move exists and the frozen holdout stays inside tolerance.
Separation means those constructed records were distinguishable.
It does not say the rule recovered a failure type from unstructured history.
It does not say the same rule separates failures on Qwen.
A collapse would mean this rule did not keep the four contradictions apart.
It would not say a belief object cannot be defined.

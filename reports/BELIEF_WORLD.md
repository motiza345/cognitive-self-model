# Belief update on changed P worlds

Diagnostic. The update rule is the one already locked in `belief_update.classify`.
Not an MRSM rescore, not a Self-Model, and not M22.

## 1. Question

Do measurements from four changed planted worlds receive four different updates?

- Decision: `WORLD_CHANGE_SEPARATES`.
- Each measured world received its own update, and the frozen holdout stayed HOLD. This result is local to these worlds and this frozen belief. Qwen was not loaded. The frozen checkpoint was not finetuned.

## 2. What was fixed before measurement

- Hypothesis construction mask: `['L0H2', 'L1H2']`.
- Scope name: `P:shuffled-target:seed=21011`.
- Interaction pair: `L0H0+L0H1`.
- Shared magnitude, from the frozen winner interaction: `14.35336685180664`.
- None lesion: `L0H3`.
- The magnitude was not searched against a gap ceiling, and it was not changed after scoring.

## 3. Measured outputs

| World | Rule output | Expected |
| --- | --- | --- |
| hypothesis | `REVISE_HYPOTHESIS` | `REVISE_HYPOTHESIS` |
| scope | `REVISE_SCOPE` | `REVISE_SCOPE` |
| interaction | `ADD_INTERACTION` | `ADD_INTERACTION` |
| none | `ABSTAIN` | `ABSTAIN` |

## 4. Control

| Case | Rule output | Expected |
| --- | --- | --- |
| holdout | `HOLD` | `HOLD` |

## 5. Descriptive selections

These selections are not decision inputs.

- hypothesis: status `IDENTIFIED`, ordered `['L0H2', 'L1H2']`.
- scope: status `IDENTIFIED`, ordered `['L0H1', 'L1H1']`.
- interaction: status `IDENTIFIED`, ordered `['L0H0', 'L1H1']`.
- none: status `IDENTIFIED`, ordered `['L0H0', 'L1H1']`.

## 6. What this does not say

Section 3 is the decision trace. It does not reopen the label.
A separation means these measured worlds were distinguishable by the unchanged rule.
A mismatch means that, for these worlds, the rule did not assign the world type.
Neither result transfers to Qwen.

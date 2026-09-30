# Evidence, belief, scope, falsification, persistence

Diagnostic on the frozen CCSO measurements. Not a new Qwen forward, not a new Self-Model, and not M22.

## 1. Question

Can a sign claim stored from two discovery prompts of one regime predict later prompts of that regime, leave itself unchanged when another regime disagrees, contest itself when its own regime disagrees, keep the evidence, and use the revised state on the next prompt?

- Decision: `SELF_KNOWLEDGE_TRACE_HOLDS`.
- The five checks held on the frozen sign trace, and each required case occurred.
- Mechanism identity was not evaluated. A historical notebook status was not used as a label.

## 2. What was fixed before scoring

- The effect is the mean of Δ/α over the six primary alphas. The sign is the sign of that mean. Zero is not a sign.
- A claim exists only when the two discovery prompts of one regime share a nonzero sign. Scope is that regime.
- Later prompts are applied in frozen index order. Validation sign accuracy must be strictly above 0.5.
- An out-of-scope prompt is recorded and does not change the claim. Scope is not expanded.
- An in-scope opposite sign contests the claim. A later match does not reactivate it.
- Evidence records are append-only. The magnitude tolerance from the earlier Qwen update is not used.
- Train directions are 23101–23106. Novel directions and alpha ±0.25 are excluded.
- M18.7 is not a gate. Missing archive notebooks were not invented.

## 3. Counts

- Scoreable claims: 43 / 72.
- Validation predictions: 59/74 = 0.797297.
- Confirmations: 108.
- In-scope falsifications: 19.
- Out-of-scope prompts: 344.
- Out-of-scope sign conflicts: 87.
- Scope violations: 0.
- Falsification-flag violations: 0.
- Update violations: 0.
- Persistence violations: 0.

| Regime | Validation sign accuracy | In-scope falsifications |
| --- | --- | --- |
| completion | 25/34 = 0.735294 | 11 |
| instruction | 18/19 = 0.947368 | 1 |
| syntax | 16/21 = 0.761905 | 7 |

## 4. What this does not say

A held trace means these frozen sign checks occurred and matched the rule. It does not name a mechanism, it does not build a Self-Model, and it does not transfer the planted Self-Model onto Qwen.

Twenty-nine of the 72 regime-layer-direction cells formed no claim, because the two discovery prompts did not share a nonzero sign. The validation fraction counts only prompts on which the claim was still active. After an in-scope sign conflict the next prompt abstains, so that later prompt is not added to the accuracy denominator. Zero scope, update, and persistence violations means this rule did not break its own constraints on the events that occurred. It does not mean a new belief was learned beyond the sign and the contest bit.

The magnitude result on the same measurements remains `QWEN_UPDATE_DOES_NOT_SEPARATE`. This sign trace does not replace it.
A prediction failure would have meant the discovery sign did not stay above chance on later prompts of the same regime.
An inconclusive result would have meant a required case, such as an in-scope sign conflict, did not occur in this frozen slice.

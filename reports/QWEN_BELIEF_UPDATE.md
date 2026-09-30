# Qwen belief scope update

Diagnostic on the frozen CCSO measurements. Not a new Qwen forward, not an MRSM rescore, and not M22.

## 1. Question

Does a belief stored from one regime hold on later prompts of that regime, revise scope when another regime misses, and abstain when the direction was never stored?

- Decision: `QWEN_UPDATE_DOES_NOT_SEPARATE`.
- The Qwen cases did not receive the three preregistered updates. No mechanism was named. The P head table and the P floor of 1 were not used.

## 2. What was fixed before scoring

- Fit uses discovery prompts only. Validation and replication are records.
- Each stored intervention has its own discovery standard deviation of Δ/α. Tolerance is twice that value.
- Train directions are 23101–23106. Novel directions 23107 and 23108 are absent from the belief.
- Primary alphas only. ±0.25 is excluded.
- A novel direction makes the two-change case abstain because the belief has no prediction for it.

## 3. Combined outputs

| Case | Output | Expected |
| --- | --- | --- |
| same_regime | `ABSTAIN` | `HOLD` |
| regime_change | `REVISE_SCOPE` | `REVISE_SCOPE` |
| two_changes | `ABSTAIN` | `ABSTAIN` |

| Case | Output | Expected |
| --- | --- | --- |
| scope_string_alone | `REVISE_SCOPE` | `HOLD` |

## 4. Pair trace

This trace is not a decision input and does not reopen the label.

### same_regime

- completion: `ABSTAIN`
- instruction: `ABSTAIN`
- syntax: `ABSTAIN`

### regime_change

- completion->instruction: `REVISE_SCOPE`
- completion->syntax: `REVISE_SCOPE`
- instruction->completion: `REVISE_SCOPE`
- instruction->syntax: `REVISE_SCOPE`
- syntax->completion: `REVISE_SCOPE`
- syntax->instruction: `REVISE_SCOPE`

### two_changes

- completion->instruction: `ABSTAIN`
- completion->syntax: `ABSTAIN`
- instruction->completion: `ABSTAIN`
- instruction->syntax: `ABSTAIN`
- syntax->completion: `ABSTAIN`
- syntax->instruction: `ABSTAIN`

### scope_string_alone

- completion: `REVISE_SCOPE`
- instruction: `REVISE_SCOPE`
- syntax: `REVISE_SCOPE`

### discovery_self

- completion: `HOLD`
- instruction: `HOLD`
- syntax: `HOLD`

## 5. What this does not say

Separation means these frozen slices received the three preregistered updates.
A mismatch means the stored regime effect did not move in the way the three labels require.
Neither result names a mechanism or transfers the eight-head table onto Qwen.

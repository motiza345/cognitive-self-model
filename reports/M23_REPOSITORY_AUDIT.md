# M23 repository audit

Audit date: 2026-10-01. Working branch created from `main` at `5dd0e076c60e80eb64eabdae0ddb3429d2b46388`. Historical implementation was inspected on `origin/cursor/mrsm-first-scientific-run-4e19` at `aba115119f81c1737e5a964458b701ddd82be5eb`. No scored M23 forward was run before this file existed.

M23 is an audit. It does not add a discovery engine, a governance stack, a simulator, or a new semantic ontology.

## What `main` actually contains

`main` tracks a package stub, the M20.6.3.2 notebook, and the MRSM specification. It does not track the M21 calibration code, the M22 intervention code, or the MRSM runner. Those live on unmerged branches. M23 therefore copies three small frozen functions instead of merging those branches.

## Located components

| Need | Where it exists | M23 decision |
| --- | --- | --- |
| M21.2.4.3.3 calibration | `reports/M21_2_4_3_3/scientific_verdict.json` on the MRSM branch. Authorized claim is a support-aware `q_invalid` score inside that benchmark. | Not reused. Different estimand. The control-plane entry `F-M21.2.4.3.3-QINVALID` forbids treating it as a universal probability. |
| M21.2.4.3.4 identifiability | `reports/M21_2_4_3_4/scientific_verdict.json`. Label `CONDITIONAL_SUPPORT_LIMITED`. Statement: pipeline bootstrapped; full benchmark code pending. | Left `INCONCLUSIVE`. Not converted to PASS. Not used as a mechanism. |
| Evidence encoder / Evidence Graph | MRSM `src/mrsm/evidence.py` is a record dataclass for the eight-head planted benchmark. No runtime evidence graph is on `main`. | Not imported. M23 stores an inspectable `evidence_history` on the belief object. |
| Invalidity estimator | M21 `q_invalid` pipeline. | Not reused. |
| Intervention utility | `src/cognitive_self_model/m22_1/intervention.py`: `h <- h + alpha * direction` at the last position of one residual hook. | Copied unchanged into `src/cognitive_self_model/m23/m22_reuse.py`. |
| Direction draw | `m22_1/direction.py`. Primary seed `22101`, orthogonal seed `22103`. Neither vector is fit to logits. | Copied unchanged. |
| Outcome | `m22_1/outcome.py`. Logit margin of the first token id of `" yes"` minus the first token id of `" no"`. | Copied unchanged. |
| Prompt catalog and splits | `m22_1/prompts.py`. Manifest `fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db`. | Copied unchanged. Roles stay discovery / validation / replication. |
| Qwen load convention | `m22_1/loader.py`. Model `Qwen/Qwen2.5-0.5B`, dtype float32, `prepend_bos` true, hook `blocks.{layer}.hook_resid_post`. | Same convention. Revision is the static pin `060db6499f32faf8b98477b0a26969ef7d8b9987`. Hub `model_info` is not called. |
| M22.1 result | `reports/M22_1/README.md`. Frozen layer `23`. Status `CANDIDATE`. | This is the only frozen Qwen intervention with a recorded behavioral delta. |
| MRSM self-model | `src/mrsm/self_model.py`. `predict` returns the stored effect for one of eight fixed head ids. P arm passed H1–H4. Q arm failed H1, H2, H4. Terminal decision `REDEFINE_SCALE`. | Not reused. Its predict method is a lookup on a different intervention interface. |
| M20.6.3.2 notebook | On `main`. Overall scientific status `FAIL`. Selected predictors `RAW_SLOPE` and `RAW_BAG`. | Not reused. Those predictors are outcome forecasters, which M23 is forbidden to relabel as a self-model. |

## Mechanism available for M23

No Qwen component in the repository has a verified mechanism identity.

The strongest frozen candidate is the M22.1 primary intervention:

- mechanism id: `M22.1-D1-L23`
- component: `blocks.23.hook_resid_post`, last token
- direction: unit Gaussian, seed `22101`
- contrast direction, used only as the shuffled-evidence control: orthogonal Gaussian, seed `22103`
- recorded validation mean paired delta at alpha `+1`: `0.028637`
- recorded orthogonal-control mean at the same alpha: `0.105415`
- status rule: primary absolute mean did not exceed the control, so the status stayed `CANDIDATE`
- M22.1 non-claims, which M23 inherits: not a self-model, not a discovered circuit, not universal behavioral control
- M22.1.3 account, inherited as an assumption and not re-fit: at this site the near rank-1 response is consistent with first-order readout geometry

M23 does not promote this candidate to a verified mechanism. It tests whether an explicit belief about this already-declared intervention can predict, be contradicted, update, and help a later held-out prediction. The orthogonal direction is not promoted either. It remains the pre-declared control, including the fact that its recorded effect was larger. Selecting it as the primary mechanism now would be selection on an already-seen confirmatory effect.

## Hook

Old and new hook are the same: `blocks.23.hook_resid_post`. No scientific reason to change it appeared in the audit. Layer 13 from M20.6.3.2 is not an input, matching the M22.1 candidate-layer rule.

One recorder field is added beside the copied hook: the pre-add dot product of the last-position residual with the direction. It is logged as a baseline internal observation. The belief's `predict` function does not accept it.

## What is missing

- A verified Qwen mechanism.
- A belief object that is updated by evidence rather than used as a lookup table.
- A pre-registered predict → intervene → update → held-out predict split on Qwen.
- An evidence graph runtime.

## Minimal new code

- `src/cognitive_self_model/m23/belief.py`: explicit belief and the pre-registered update rule.
- `src/cognitive_self_model/m23/stats.py`: pre-registered intervals and the verdict map.
- `src/cognitive_self_model/m23/m22_reuse.py`: verbatim M22.1 prompt, direction, intervention, and outcome functions.
- `scripts/run_m23_direct_qwen_self_model_audit.py`: ordered runner.
- `tests/test_m23_belief.py`: update, leakage of the predict signature, and verdict map. No model.

## Assumptions inherited from previous milestones

1. Outcome remains the M22.1 yes/no logit margin. Token ids are recomputed from the tokenizer and checked against the recorded pair `[9834, 902]`.
2. The prompt catalog, split rule, layer, seeds, and alpha grid are already frozen. M23 does not draw a new prompt set.
3. Initial belief uses only the discovery split. Validation is the update split. Replication is the held-out split. This matches the M22.1 rule that discovery may be used to form a choice and validation/replication are confirmatory.
4. Prediction at a new magnitude is `alpha * effect_at_alpha_1`. That linearity is the M22.1.3 readout assumption. It is not estimated on validation or replication.
5. M21 identifiability remains inconclusive. M22.1 remains a candidate. MRSM Q transfer remains failed.
6. Sample size is six prompts per split. A confidence interval that includes zero is `INCONCLUSIVE`. No numeric pass threshold is added after the run.

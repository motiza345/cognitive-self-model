# M23 scientific report

Overall verdict: `INCONCLUSIVE`

This audit tested one frozen candidate intervention, `M22.1-D1-L23`.
M22.1 had already left that intervention as `CANDIDATE`.
M23 did not search for a new mechanism and did not change a threshold after scoring.

## Question results

- Q1 Prediction before intervention: `PASS`
- Q2 Falsifiability: `INCONCLUSIVE`
- Q3 Explicit revision: `PASS`
- Q4 Held-out improvement after update: `INCONCLUSIVE`
- Q5 Specificity against an outcome-only baseline: `PASS`
- Q6 Transfer to a held-out magnitude: `INCONCLUSIVE`

## Metrics

- Q1 MAE belief `0.00280716684129503`, MAE predict-zero `0.028189738591512043`.
- Q1 paired interval for (zero error − belief error): mean `0.025382571750217017`, low `0.023180643717447918`, high `0.02758449978298611`, class `CI_POSITIVE`.
- Q1 sign agreements `6` / `6`, Clopper-Pearson low `0.5407418735600994`, high `1.0`.
- Critical epistemic cases: `0`. Descriptive 95% coverage: `1.0`.
- eval_context MAE updated `0.0026690165201822915`, no-update `0.0026690165201822915`, shuffled `0.037352959314982094`, outcome-only `0.03895318508148194`.
- eval_context update_vs_no_update: mean `0.0`, low `-0.00023674964904785156`, high `0.00023674964904785156`, class `CI_INCLUDES_ZERO`.
- eval_context shuffled_vs_no_update: mean `-0.0346839427947998`, low `-0.0371502505408393`, high `-0.0319319036271837`, class `CI_NEGATIVE`.
- eval_context specific_vs_outcome_only: mean `0.036284168561299644`, low `0.033344109853108726`, high `0.038987225956387`, class `CI_POSITIVE`.
- eval_magnitude MAE updated `0.0051428741878933384`, no-update `0.00503842035929362`, shuffled `0.07424354553222655`, outcome-only `0.07744399706522624`.
- eval_magnitude update_vs_no_update: mean `-0.00010445382859971865`, low `-0.0005779531266954218`, high `0.00039688746134439873`, class `CI_INCLUDES_ZERO`.
- eval_magnitude shuffled_vs_no_update: mean `-0.06920512517293294`, low `-0.07411358091566297`, high `-0.06387249628702797`, class `CI_NEGATIVE`.
- eval_magnitude specific_vs_outcome_only: mean `0.07230112287733291`, low `0.06636269887288411`, high `0.07768307791815865`, class `CI_POSITIVE`.

## Failure class

None assigned by the pre-registered map.

## What this does not establish

The following are outside this result: a general self-model, a verified Qwen circuit,
representation invariance, decision improvement, and closed-loop self-improvement.

## Allowed conclusion

The limited operational loop was not demonstrated on this setting. The pre-registered verdict is `INCONCLUSIVE`.

## Interpretation of the frozen outputs

These notes do not add a gate and do not change the verdict.

The discovery D1 mean was `0.028899987538655598`. The validation update moved it to `0.02854486306508382`. The orthogonal D2 discovery mean was `0.10663819313049316`. Those magnitudes match the already published M22.1 candidate and control means closely enough to show this run used the same intervention. The published means were not loaded into the belief.

Q1 passed because that single D1 mean, issued for every validation prompt before those outcomes existed, beat predicting zero. The belief did not vary by prompt, by residual dot product, or by regime. Sign agreement was 6/6. Prompt-level correlation is undefined because the prediction has no variance.

Q2 stayed inconclusive because none of the six validation outcomes fell outside the pre-registered interval or flipped sign. Coverage of the 95% interval was 1. The belief was confident relative to its own standard deviation, and the evidence agreed with it. Agreement is not a demonstration that contradiction can move the belief.

Q3 passed as a structural check. Each update stored a before and after version and set the mean to the pooled observations. It did not copy the latest observation. The numerical change was `0.000355`.

Q4 stayed inconclusive. On replication at alpha `+1`, the mean paired absolute-error difference between the updated belief and the untouched discovery belief was `0.0`, and the interval included zero. The shuffled update, which treated D2 evidence as if it were D1, was reliably worse. That is the expected control result. It does not show that the genuine update improved prediction.

Q5 passed the pre-registered contrast: the D1-specific mean beat the discovery mean that pools D1 with D2. The pool is worse because D2's effect is about three times larger. The passed object is the name of the intervention plus its average effect. A predictor allowed to average D1 outcomes alone would be the same object. This pass does not identify a circuit.

Q6 stayed inconclusive. At the held-out magnitude alpha `+2`, scaling the discovery mean by two was already close to the observed deltas, and the update did not improve it. The interval again included zero. Shuffling D2 into the D1 belief was reliably worse at this magnitude too.

No implementation abort fired. The smoke forward wrote its prediction before its outcome, and the hook modified only the last position. The scored leakage checks passed.

The inconclusive result is a property of this stable candidate effect under the frozen sample of six prompts per split. It is not a reason to add a new architecture inside M23.

## Reproducibility

- bootstrap_seed: `23001`
- branch: `cursor/m23-direct-qwen-audit-5fe5`
- control_direction_sha256: `8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e`
- dataset_manifest: `fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db`
- device: `cpu`
- direction_sha256: `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411`
- git_commit: `04fc5793deec20d6668c57293dafdee543c2ecd4`
- hook_name: `blocks.23.hook_resid_post`
- model: `Qwen/Qwen2.5-0.5B`
- model_revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- negative_token_id: `902`
- positive_token_id: `9834`
- preregistration_sha256: `e590df86cbb057f4dbe2b0f211809927dcd7a9839f4a069cf936bae14f1472dd`
- python: `3.12.3`
- seed_control: `22103`
- seed_direction: `22101`
- torch: `2.14.0+cpu`
- transformer_lens: `3.9.0`
- transformers: `5.18.0`

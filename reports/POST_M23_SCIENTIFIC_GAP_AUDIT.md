# Post-M23 scientific and architecture gap audit

Audit date: 2026-10-01. This file reads existing reports and notebooks. It does not run a model, fit a predictor, edit `SelfModelBelief`, or change any M23-G artifact.

The intended mechanism-level Self-Model, as written in `docs/MRSM_Implementation_Spec_and_Cursor_Instructions_v1.0.md` section 3, is treated as a hypothesis. A field is listed as necessary only when a recorded result already fails without that field and the same result identifies the field. Desire, and the MRSM checklist, are not evidence.

Severity in the companion matrix: `BLOCKING` means the self-model claim stays unavailable while this limitation stands. `MATERIAL` means a narrower claim is limited. `RECORD` means the repository has no scored artifact under that id.

## 1. Executive conclusion

The project has an inspectable belief tracker for a named residual intervention, and a sequence of frozen tests of that tracker. It does not have a genuine self-model object.

`SelfModelBelief.predict` takes a magnitude and returns `alpha` times a stored mean, with a stored standard deviation. The caller supplies `mechanism_id`. The update appends one observed scalar. M23-G ran that loop on six cells and 36 new prompts. The verdict is `INCONCLUSIVE`: the paired mean of (no-update absolute error − updated absolute error) is `0.0026111067445189844`, and the 95% interval is `[-0.0006063265932930853, 0.00589502520031399]`. Leakage is `true`. Twenty-six of 72 validation updates were sign mismatches, so a single mean per cell does not determine the sign of the next prompt. The cell-blind train mean still has a lower point absolute error (`0.06729273406075842`) than the updated cell beliefs (`0.07133275104893579`).

That combination blocks the self-model claim and also blocks the claim that the missing piece has already been identified. Within-cell sign variation is measured. No pre-outcome measurement has been shown, on a split that was still clean, to carry that variation. Causal edges, temporal state, interaction terms, and a new update rule are not required by these results.

M19.4, M19.5, M20, and M20.5 have no report, config, or label in the tracked tree. The notebook named M19 is remapped by `notebooks/source/README.md` on `origin/cursor/mrsm-first-scientific-run-4e19` to M21.2.3.1–M21.2.4.2. Control-plane recovery grades M19 and M20.6.3.2 as `POINTER_ONLY`.

## 2. Milestone-by-milestone evidence table

Intervening artifacts that the requested list skips, and that later milestones inherit, are included after M21: M22.1 and the MRSM P/Q run.

| Milestone | Object actually represented | Prediction actually made | Evidence that could update it | Information unavailable to the representation | Claim the artifact justifies | Stronger claim that stays unsupported | Primary limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M19 | No scored object. Filename `M19_governance_and_genesis.ipynb` holds unscored prototype cells later labeled M21.2.3.x and M21.2.4.0–.2. | None recorded. | None recorded. | No frozen inputs. | A provenance pointer exists. | Any M19 scientific result. | other (`RECORD`) |
| M19.4 | Absent. No match in tracked `md`, `yaml`, `json`, `py`, or `ipynb` on the inspected branches. | None. | None. | The id is undefined. | Nothing. | Anything numbered M19.4. | other (`RECORD`) |
| M19.5 | Absent, same search. | None. | None. | The id is undefined. | Nothing. | Anything numbered M19.5. | other (`RECORD`) |
| M20 | Absent as a standalone id. The only scored M20 artifact is M20.6.3.2. | None under the bare id. | None. | No `reports/M20*`. | Nothing under this id. | A completed M20 self-model. | other (`RECORD`) |
| M20.5 | Absent, same search. | None. | None. | The id is undefined. | Nothing. | Anything numbered M20.5. | other (`RECORD`) |
| M20.6.3.2 | A forecaster of a synthetic structural-oracle stream. Selected point model `RAW_SLOPE`. Selected hazard model `RAW_BAG`. Feedback layer 13, activation PCA dimension 12. Qwen/Qwen2.5-0.5B activations are a sensor for that stream. | Next-horizon point value and an event hazard, horizons in `{1, 2, 3, 6}`, primary horizon 3. | Stream history of the planted process, plus the layer-13 activation summary. | Ground-truth labels are sealed from the agent view. The selected models are bag and slope summaries. The temporal-order gate requires the selected model to be `RAW_ORDERED`; it was not. | `benchmark_construction PASS`, `observable_feedback_self_model PASS` (this flag is `feedback_observability_valid`: layer 13 rho and R² cleared `0.55` and `0.25`), `cross_regime_transfer PASS`, selected forecasters valid on their numeric gates. Overall status `FAIL`. | Temporal self-model learning. Novelty detection (novelty AUROC `0.4911`). An operational warning policy (event recall `0.418`, gate `0.50`). Any reading of the observability flag as a self-model of Qwen. | representation, with a sensing failure on novelty |
| M21.2.4.3.3 | `q_invalid`, a support-aware score inside one synthetic benchmark. | `P(I_t = 1 \| phi_t)` inside the observed track families, with support metadata. | The benchmark's 8-key feature contract. | Qwen residuals, intervention identity, and out-of-family tracks. Exact 0 and 1 are not certainty. Singleton mass fraction `1.0` triggered the support warning. | Authorized claim: a benchmark-conditional invalidity evidence score. | A universal probability. A mechanism identity. An activation intervention. | evaluation power |
| M21.2.4.3.4 | A bootstrapped identifiability pipeline. | No completed identifiability number. | The verdict text only. | Full benchmark code, recorded as pending. | Label `CONDITIONAL_SUPPORT_LIMITED`. | An identified causal mechanism. | identifiability |
| M22.1 (inherited) | One additive last-token residual intervention, `blocks.23.hook_resid_post`, direction seed `22101`, outcome = yes/no logit margin, token ids `9834` and `902`. | The intervention's paired change in that margin. No belief is updated. | Confirmatory prompt splits already used for the status rule. | Circuit membership, scope across tasks, any case-level predictor. | Status `CANDIDATE`. Null and hook checks passed. Primary validation mean paired delta at alpha `+1` is `0.028637`. Orthogonal control mean is `0.105415`, so the primary did not clear the control. | `VALIDATED_FOR_M22`. A discovered circuit. A self-model. | intervention design |
| MRSM P/Q (inherited hypothesis test) | `SelfModelCore.predict(intervention_id)` returns the stored effect of one of eight fixed head ids. | Held-out intervention consequence, then a consumed action. | Discovery measurements of those heads. On P, a planted edge. On Q, no planted ground truth. | On Q, an independently verified mechanism. | P gates H1–H4 `PASS` on the planted benchmark. Q.H3 `PASS` (frozen abstraction stayed put). | Q transfer. Q.H1, Q.H2, and Q.H4 are `FAIL`. Q.H2 self MAE `0.027458` versus baseline B2 `0.028281`, reduction `0.029`, bootstrap low below 0. Terminal decision on that arm: `REDEFINE_SCALE`. | causal structure |
| M23 | `SelfModelBelief` for `M22.1-D1-L23`: one mean, one sample sd, a caller-supplied id, a scope dict, an evidence list. | `predicted_effect = (alpha / 1) * stored_mean` for every prompt. | Six validation scalars at alpha `+1`. Replication was not an update set. | Prompt text, regime, `pre_dot`, baseline margin, direction contents, upstream and downstream sites. `predict` accepts only `alpha`. | Q1 `PASS` (the discovery mean beat predicting zero). Q3 `PASS` (the update stored a new version and did not copy the latest observation). Q5 `PASS` (the D1 mean beat the pool of D1 with D2). Overall `INCONCLUSIVE`. | Q2 falsification, Q4 held-out improvement, Q6 magnitude transfer. A self-model. Discovery mean `0.028900` moved to `0.028545`. Replication paired difference of absolute errors had mean `0`. | intervention design |
| M23 diagnostic | The same belief, plus read-only descriptions of saved `pre_dot`, baseline margin, and regime. | The historical constants, and several lines fit only as descriptions. | The already saved M23 rows. No new forward. | A noise floor (no alpha-0 repeat was saved). Discovery `pre_dot` (those rows were discarded). An oracle fit on replication was correctly not run. | `BELIEF_UPDATE_INFORMATION_FAILURE = NOT_SUPPORTED`. Correct D1 evidence moves the mean by `0.000355`. D2 evidence moves it by `0.037269`. Reversed prompt order leaves the mean unchanged. The other three labels stay `INCONCLUSIVE`. | A representation bottleneck as a positive finding. Identification of `pre_dot` (validation correlation `0.904`, replication `-0.249`). | identifiability |
| M23-D2 | Two predictors of the same D1 effect: one constant, and one mean per surface regime. | A single number per evaluation prompt, written before the outcome. | Regime label only, fit on the old validation split. | `pre_dot`, baseline margin, prompt embedding, the replication split. | `INCONCLUSIVE`. Mean paired difference exactly `0`. Both predictions sit between the two held-out effects of each regime, so total absolute error matches. | Regime contains held-out information about this effect. | context/state representation, with evaluation power (n = 2 per regime) |
| M23-E | The D1 layer-23 response surface: effect, `pre_dot`, baseline margin, at alpha `+1` and `+2`. | Held-out absolute error of a constant versus an OLS line. | Nine train prompts. | Regime. Any feature chosen after the errors. | All four paired intervals include zero. Alpha ratio mean `2.016879`, sample sd `0.015823`. Residual `effect(+2) - 2*effect(+1)` mean `0.000468`. Repeats identical. | A self-model. A state-dependent response at this site. `INTERVENTION_DESIGN_FAILURE` and `IDENTIFIABILITY_FAILURE` stay `INCONCLUSIVE` because no stability cutoff was frozen. | identifiability |
| M23-F | An inventory of existing cells and a structural screen on six discovery prompts. | None. No belief was scored. | Screen signs and alpha ratios. | A reserved holdout. Correlations were deliberately not computed. | `audit_status = NO_SUITABLE_EXISTING_INTERVENTION`. `current_intervention_status = CURRENT_INTERVENTION_INSUFFICIENT` for D1 layer 23. | A chosen next cell. Layer 0 is not selectable from its range `0.274`. | intervention design |
| M23-G | Six separate scalar beliefs, D1 and D2 at layers 0, 8, and 15. Each stores a mean, a sample sd, version, inflation, and a scope status. | Alpha is `+1` only, so the prediction is the stored mean, one number for every prompt in that cell. | The twelve validation effects of that same cell, in prompt-id order. Shuffled control swaps D1 and D2 validation evidence inside a layer. | The prompt, the family, `pre_dot` (logged, not passed to `predict`), baseline margin, other cells' structure, any graph. | The loop can be contradicted: 26/72 sign mismatches, 4/72 outside the inherited interval, all six beliefs reach version 13 and status `CONTRADICTED`, inflation stays `1.0`. Leakage `true`. Synthetic harness `HARNESS_PASS` on the update interface only. | `UPDATE_IMPROVES`. The primary interval includes zero. A self-model. The two layer-0 secondary intervals that are `CI_POSITIVE` do not drop the other cells. | representation |

## 3. M23-G representation audit

What the object contains after the run:

- `mechanism_id`: a string the runner assigns, such as `M22.1-D1-L0`.
- `intervention_type`: the residual-add description inherited from the cell definition.
- `predicted_effect`: the cumulative mean of train plus validation scalars. Example, D1 layer 0: train `0.041386`, final `0.032807`.
- `uncertainty`: sample standard deviation times inflation. Inflation never left `1`.
- `validity_scope.status`: `CONTRADICTED` on every cell. `predict` does not read this flag.
- `evidence_history` and `update_history`: the scalars, the sign-mismatch bit, the interval bit, and the before/after snapshot.
- `version`: 13 on every cell.

What the prediction uses: the mean and alpha. Family, prompt, layer contents, and `pre_dot` can change without changing the issued number, as long as the cell id stays the same.

What the update uses: one finite float and an evidence id. The id is stored. The numeric update is a multiset mean. M23 diagnostic already showed that reversing order does not change the mean. M23-G did not reopen that check.

Contradiction happened, and it did not become a held-out gain under the frozen rule. Sign agreement on evaluation is 33/72 for the updated belief and 31/72 for no-update, for the constant, and for the shuffled belief. Zero critical contradictions occurred because confidence requires `|mean| >= 1.96 * uncertainty`, and the train uncertainties are large relative to the train means (D1 layer 0: mean `0.041386`, uncertainty `0.159111`). The inherited rule therefore recorded ordinary contradictions and left inflation unchanged.

The constant-effect control is the mean of all 72 train effects, `0.013369`. Its evaluation MAE equals the outcome-only MAE, `0.067293`, and that number is smaller than the updated MAE. The primary gate did not compare against this constant. The comparison still matters for architecture: cell identity, which is the only structure the belief has, did not earn its keep on point error.

Shuffled MAE is `0.072323`. Shuffled versus no-update includes zero. The control does not show that wrong-cell evidence is harmless, and it does not show that right-cell evidence is identified.

## 4. Predictive model vs belief tracker vs self-model

These definitions are operational. They refer to what the implemented object computes.

**Belief tracker.** An inspectable object whose issued prediction is a function of a stored summary and the intervention magnitude only. An update returns a new object whose summary is a function of the previous summary and one newly observed outcome scalar. The scientific claim available to it is whether that summary changes, and whether the changed summary has lower held-out absolute error than the unchanged summary. A caller-supplied name may be stored on the object.

**Predictive model.** A function from measurements taken before the outcome to a numeric prediction, where those measurements are allowed to differ across cases inside one intervention. The scientific claim is a held-out error comparison against a baseline frozen in advance. An evidence log is optional.

**Self-model object.** The MRSM hypothesis, applied to the implementation rather than to the prose around it, requires all of the following at once: a mechanism hypothesis that is distinct from a caller-supplied intervention label, or a label whose causal role has been verified; a predicted consequence that depends on that role; a scope that changes which cases the prediction applies to; stored uncertainty and an evidence trace; a decision policy that reads the prediction before it acts; and a held-out gain over the strongest frozen non-mechanistic baseline. Missing any one of these, the object does not satisfy the hypothesis.

**What exists now.** M23-G is a belief tracker, and that tracker is being asked to predict. It meets the belief-tracker clauses: the prediction ignores the new case, the update appends a scalar, version advances, and the held-out comparison with persistence was actually run. It does not meet the predictive-model clause inside a cell, because no case measurement changes the prediction. It does not meet the self-model clauses: the id is assigned by the runner, scope status does not change `predict`, nothing consumes the prediction as an action, and the held-out interval includes zero. The constant train mean, a non-mechanistic baseline, has the lower point error.

M20.6.3.2 is a predictive model of a synthetic stream, with a Qwen activation used as a sensor. Its own overall status is `FAIL`. The flag `observable_feedback_self_model` is the layer-observability gate, not the definition above.

MRSM P is a self-model object on a planted benchmark, under that benchmark's gates. MRSM Q is the same code on Qwen, and the transfer gates that would carry the claim failed. P does not license the Qwen object.

## 5. Minimum required Self-Model representation

Justified by code that ran, and by claims that passed their own gates:

1. An intervention cell id supplied by the experiment (direction seed, layer, hook, alpha).
2. A scalar mean of observed effects for that cell.
3. A sample uncertainty for that scalar.
4. An append-only record of the scalars and of whether each one mismatched the sign or fell outside `1.96` prior standard deviations.
5. A version counter.

That is the minimum object the evidence supports as an implemented, inspectable belief. It is sufficient for the claims M23 actually passed: prediction-before-outcome, explicit revision, and specificity against pooling D1 with D2 on the old layer-23 cell.

It is not a sufficient representation of the M23-G outcomes. Twenty-six validation effects had the opposite sign from the train mean of their own cell. A self-model hypothesis that stops at one number per cell is already inconsistent with those rows.

The smallest addition that would *change the scientific question*, as opposed to decorating the object, is one pre-outcome measurement that is allowed to change the prediction inside a cell. The audits do not say which measurement. Until one is predeclared and tested on prompts whose outcomes are still unseen, adding it is a hypothesis.

## 6. Required vs motivated vs unjustified fields

### REQUIRED BY CURRENT EVIDENCE

- **A within-cell distinction of some kind, still unnamed.** M23-G sign mismatches (26/72) and evaluation sign agreement (33/72) already show that one mean per cell does not fix the sign of a new case. This is a negative requirement: the scalar is an insufficient consequence model for these cells. It does not name the replacement field.
- **The existing test, kept as a gate.** Any later object that is offered as a self-model has to beat persistence by an interval that excludes zero, and it has to be compared with a cell-blind constant, because that constant currently has the better point MAE. Those comparisons are requirements on the claim. They are not new representation fields.

### STRONGLY MOTIVATED

- **Some pre-outcome conditioner inside the cell.** Motivated by the sign heterogeneity on the G cells, and by M23-F's discovery screen, where D1 at layers 0, 8, and 15 and D2 at layers 0 and 8 already showed both signs. The conditioner is unidentified. Regime failed to beat a constant on D1 layer 23 (M23-D2, difference `0`). `pre_dot` and baseline margin failed the same way on that cell (M23-E, four intervals include zero). Neither result transfers, as a positive finding, to the early-layer cells. Neither result was a license to put those features into `predict`.
- **Keeping cell identity as bookkeeping.** M23 Q5 showed that pooling D1 with D2 at layer 23 is a worse predictor of D1, because the control mean is about three times larger. Cell identity therefore matters across some cells. M23-G's constant beating the cell beliefs on point MAE means cell identity is not automatically a predictive feature on the new catalog. Bookkeeping stays motivated. Necessity does not.
- **A scope that can withhold a prediction.** Every G belief is `CONTRADICTED` while `predict` still emits a number for every evaluation prompt. A scope that abstains would be a different policy. The data motivate asking whether abstention helps. They do not show that it does.

### NOT YET JUSTIFIED

- **Verified mechanism identity.** M22.1 remains `CANDIDATE`. MRSM Q.H1 failed. M21.2.4.3.4 is `CONDITIONAL_SUPPORT_LIMITED`. The string on the belief is a label.
- **Causal graph or edge structure.** No M23 result scores an edge. MRSM P's edge was planted. Q halves disagreed (`L0H2 -> L0H3` versus `L0H0 -> L0H1`).
- **Upstream or downstream structure.** Not measured as a predictor in M23 through M23-G.
- **Context or regime as a required field.** Tested once, on D1 layer 23, result `INCONCLUSIVE`, and the design cannot credit a mean that sits between two held-out points.
- **State (`pre_dot`, baseline margin) as a required field.** Same cell, held-out intervals include zero. `pre_dot` standard deviation is about 595 times the alpha-`+1` effect standard deviation, which is evidence that this sensor moves when the effect does not, on that cell.
- **Interaction terms.** No frozen interaction contrast exists.
- **Temporal state.** M20.6.3.2 selected `RAW_SLOPE` and `RAW_BAG`, so its temporal-order gate failed. Later milestones do not use stream state. Nothing in M23-G is a time series of beliefs that predicts the next prompt.
- **Provenance beyond the multiset of scalars.** Order reversal left the M23 mean unchanged. Prompt identity was not an input the update could use, and no clean test shows that it should.
- **A new update rule.** `BELIEF_UPDATE_INFORMATION_FAILURE = NOT_SUPPORTED`. The update moves when the numbers move. The G failure is that the moved mean is still one number per cell, and its held-out gain includes zero. Inflation staying at 1 is a consequence of wide train standard deviations, not a demonstrated defect in the arithmetic.
- **Decision consumption.** Required by the MRSM hypothesis. No M23 policy reads the belief before choosing an intervention.

## 7. Core unresolved scientific question

Inside one named residual cell whose effects change sign, which measurement available before the intervention, if any, changes the held-out sign or the held-out absolute error relative to that cell's own scalar mean?

M23-G shows the scalar is a weak description. M23-E and M23-D2 show that the obvious conditioners do not identify the *stable* cell. Those are different cells. The open question is the early-layer cell, on prompts that have not already been scored.

## 8. Candidate M24 hypotheses — design only

No hypothesis below is authorized to run. Each one is contaminated if it is fit on `reports/M23_G_EPISODES.csv` or on `reports/m23_g_raw/`, because those files already contain evaluation outcomes. Discovery screens from M23-F are also already inspected.

**H-A. One predeclared conditioner, one predeclared cell, a new prompt catalog.** Freeze the cell and exactly one logged quantity (`pre_dot`, or baseline margin, or surface family) before any new outcome exists. Predict with the cell scalar and with that quantity's train-only summary. Primary contrast: paired absolute error on a held-out partition, same interval rule as M23 (5000 draws, seed declared in the preregistration, percentile endpoints as in `paired_mean_ci`). A result that includes zero stays inconclusive. Do not add a second feature after seeing the first error.

**H-B. Is the within-cell sign stable across a second magnitude?** Freeze alpha `+1` and alpha `+2` for one cell family. The question is whether the sign at `+1` predicts the sign at `+2` on new prompts, against a constant-sign baseline. M23-F already saw unstable ratios on D2 layer 0 in the old discovery screen, so that screen cannot be the confirmatory set.

**H-C. Abstention as a policy, not a new mean.** Using only a rule frozen from train uncertainty, ask whether refusing to predict when the belief is `CONTRADICTED` reduces the error of the cases that remain, against always predicting the mean. This tests scope. It does not add a feature. It still needs a catalog whose evaluation outcomes do not exist yet, because the G evaluation errors are known.

H-A is the hypothesis that matches section 7. H-B and H-C answer neighboring questions and must not be run as a search over the same prompts.

## 9. Explicit "DO NOT DO NEXT" list

- Do not execute M24, and do not run a new Qwen forward, as part of closing this audit.
- Do not edit M23-G reports, raw files, the preregistration, or the verdict.
- Do not fit `pre_dot`, regime, family, or baseline margin on the M23-G episodes. The evaluation outcomes are already in those files.
- Do not drop cells because D1 layer 0 and D2 layer 0 have secondary intervals above zero.
- Do not return to D1 layer 23 as a falsification benchmark. M23-F marks it `CURRENT_INTERVENTION_INSUFFICIENT`, and M23-E describes a shared, nearly linear response to alpha.
- Do not reuse the original 18-prompt catalog as a confirmatory holdout.
- Do not invent a new update rule, a graph learner, a discovery search, or a governance stack to explain the inconclusive interval.
- Do not read M20.6.3.2 `observable_feedback_self_model: PASS` as a self-model result.
- Do not read M21 `q_invalid` as a probability of mechanism failure on Qwen.
- Do not transfer MRSM P's passed gates onto Qwen. Q.H1, Q.H2, and Q.H4 failed.
- Do not describe the M23-G point estimate `0.002611` as an improvement. The frozen class is `CI_INCLUDES_ZERO`.

## 10. Recommended next experimental question

On a prompt catalog that does not yet exist, for one cell predeclared from the M23-G family, does exactly one predeclared pre-outcome measurement reduce held-out absolute error relative to that cell's scalar mean, under the existing paired-interval rule?

Until that question has a frozen answer, the justified representation remains the scalar belief tracker, and the self-model hypothesis remains unsupported.

## Sources

Local, on this branch: `reports/M23_REPOSITORY_AUDIT.md`, `reports/M23_SCIENTIFIC_REPORT.md`, `reports/M23_SCIENTIFIC_VERDICT.json`, `reports/M23_DIAGNOSTIC_REPORT.md`, `reports/M23_D2_REPORT.md`, `reports/M23_E_REPORT.md`, `reports/M23_F_REPORT.md`, `reports/M23_G_EXECUTION_REPORT.md`, `reports/M23_G_RESULTS.json`, `reports/m23_g_raw/initial_beliefs.json`, `src/cognitive_self_model/m23/belief.py`, `notebooks/cognitive_self_model_latest.ipynb`, `docs/MRSM_Implementation_Spec_and_Cursor_Instructions_v1.0.md`.

Read from `origin/cursor/mrsm-first-scientific-run-4e19` and not modified: `notebooks/source/README.md`, `control_plane/RECOVERY.yaml`, `control_plane/STATE.yaml`, `control_plane/CLAIMS.yaml`, `reports/M21_2_4_3_3/scientific_verdict.json`, `reports/M21_2_4_3_4/scientific_verdict.json`, `reports/M22_1/README.md`, `reports/MRSM_FINAL_RESULT.md`.

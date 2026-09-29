# MRSM scientific contract

Version: 0.1

Status: `DRAFT / NOT FROZEN`

Not a preregistration.

P2.1 names the governing frozen rules. P2.2 adds operational definitions for Q.H1 and for the H3 family inside this draft. `DEC-010-A1` remains `PROPOSED / NOT APPLIED`. Concrete transformation parameters and every H1–H4 numerical bar remain deferred. DEC-011 was not amended.

No MRSM code, planted execution, Qwen evaluation, or threshold edit was done to produce this draft.

## 1. Objective / Hypothesis

MRSM asks whether a minimal self-model can be built and tested under a preregistered Go/No-Go benchmark. The question in specification section 0 has four parts: capture causal mechanism information, predict intervention consequences, remain valid under a representation change, and be consumed by a decision.

Those parts are not one claim.

| Question | What a positive result would mean | What it would not mean |
| --- | --- | --- |
| Outcome prediction | An observable consequence of an intervention is predicted on held-out episodes | That the predictor contains a mechanism object, or that the system is a Self-Model |
| Component or context learning | Performance tracks which component or regime was observed | That the tracked component is the mechanism, or that a new mechanism would be recognized |
| Mechanism-centered Self-Model | An inspectable mechanism object, a causal intervention relation, a scope, a consequence prediction, an evidence trace, and consumption by the decision policy are all present | Consciousness, complete self-understanding, self-modification, or a result on every LLM |

Specification section 3 is the proposed definition of the third row. S1, in specification section 1.1, says the object must be consumed. S1 is in the specification. It is not a sentence in DEC-010 or DEC-011. H4 is the specification's consumption gate. DEC-010 already contains the keys `P.H4` and `Q.H4` and does not define S1.

Prediction alone does not establish a Self-Model. A readout, a linear probe, a component-importance table, a prompt embedding, a scalar trust score, or an outcome memorizer is excluded by specification section 3.

## 2. Operational Definitions

Definitions below are implementable only where a source already fixes the object. Items with no frozen parameter are marked.

**Mechanism.** On arm P, the mechanism is the frozen planted read restriction in `P_SPEC.md`: during training, head L0H0 (`planted_A`) at the query position may read only key positions, and head L1H1 (`planted_B`) may read only value positions. The training mask is removed before audit. Weights are frozen before audit. The identity is hidden from discovery. Ground truth is this definition, not a discovery output.

The names `M_COPY`, `M_INHIBIT`, and `M_BIND` are provisional examples in specification section 6.2. P2.1 does not authorize replacing the frozen two-hop construction with those names. A different planted mechanism requires a new DEC.

**Mechanism identity.** On P, identity is the pair of planted head coordinates together with the key-read and value-read restrictions, as written in `P_SPEC.md`. A discovered object matches that identity only if it names those roles, not merely a layer, a head index, or an accuracy number. The specification also asks for causal direction and effect sign. Those checks are proposed in section 9 and are not frozen bars.

On Q, mechanism identity is a hypothesis inside the self-model object. It is not a verified identity. DEC-011 records Q ground truth as `UNKNOWN`.

**Component.** A named site an intervention can address. In the specification's intervention record that is `source_component` and `target_component`. In `P_SPEC.md` the planted sites are the two heads. A component is not a mechanism. Ablating a head tests the planted role only when the analysis uses the frozen role definition, which discovery code must not be given as a label.

**Outcome.** On P, the task outcome is defined at the query position of the episodic two-hop episode in `P_SPEC.md`. The construction predicate uses accuracy on development data. That predicate is not an H gate.

On Q, no MRSM outcome target is frozen. The M22.1 logit margin is a preflight endpoint. It is not adopted here as the Q outcome.

**Context or regime.** Specification section 7 names context holdout and regime holdout. It does not define the features that make two episodes different regimes. Those features are not frozen.

**Causal edge.** On P, the construction's causal structure is the two-hop read: the query consults a key through `planted_A` and a value through `planted_B`. The specification also asks discovery to recover causal direction. An edge recovered by the discovery program is a hypothesis until it is scored against the P artifact. It is not ground truth on Q.

**Self-Model.** A machine-readable object with the six fields in specification section 3: mechanism hypothesis, causal relation between intervention and outcome, applicability or scope, prediction of the intervention consequence, uncertainty or evidence trace, and availability to a decision policy. Specification section 5 names the operations `fit`, `discover`, `predict`, `estimate_uncertainty`, `applicable`, `select_action`, `update`, and `explain`. Those names are an interface proposal, not code in this phase.

Every prediction the specification requires returns a mean, an uncertainty, a mechanism id or hypothesis, a scope id, and evidence references.

**Counterfactual prediction.** A prediction of the consequence of a stated intervention, issued before that intervention is applied to produce the scored outcome. The scored episodes are held out. The specification's primary numeric summary is paired intervention-level MAE. The pass bar on that MAE is `PROPOSED — NOT FROZEN`.

**Representation transformation.** An invertible change of the coordinates of the observed representation that leaves the intervention protocol and, on P, the planted causal graph unchanged. H3 compares mechanism identity (P) or functional predictions (Q) across the untransformed and transformed coordinates. The particular maps are not frozen. See section 8.

**Transfer.** The Q arm: the same discovery, validation, and consumption code path, behind a model adapter, on `Qwen/Qwen2.5-0.5B` at revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, which has no mechanism ground truth. DEC-011 requires the shared code path not to branch on arm identity. Arm-specific code is the adapter.

**Invalidity.** Not defined as an MRSM gate. The M21 quantity `q_invalid` is a benchmark invalidity score. `CLAIMS.yaml` limits it to that benchmark and says it is not a universal probability of invalidity. DEC-011 forbids the M21 environment as arm P. `reports/M22_1/README.md` says that score is not evidence for the M22.1 preflight. This contract does not import `q_invalid` as H1–H4.

Detecting that a prediction is unsupported does not authorize a change to the frozen mechanism, the holdout, the thresholds, or the model weights. Specification section 23 says a nonzero scientific impact requires a new DEC. Specification section 25 excludes self-modification and self-repair as claims of this benchmark.

**Abstention.** Not specified by DEC-010, DEC-011, or `P_SPEC.md`. The notebook inventory says the governance layer that contained abstention is not ported. `applicable` and `estimate_uncertainty` are interface names in the specification. No abstention rule is frozen, and this draft does not invent one.

## 3. System Artifacts

### P arm

DEC-011 arm id `planted_small_transformer`. Substrate: a small TransformerLens `HookedTransformer` on CPU. Construction: hybrid, algorithmic task training plus a predefined structural mechanism. The frozen task, widths, seeds, and planted heads are `P_SPEC.md`.

P can support a claim that a method recovered a known planted mechanism, that a counterfactual prediction beat the preregistered baselines on held-out P interventions, that a representation change preserved the P mechanism object, and that consumption changed a P decision. It can support those claims only after the corresponding gate is defined and the run is a budgeted full run. It cannot support a claim about Qwen.

The construction definition of done in DEC-011 and `P_SPEC.md` is a precondition for starting MRSM. It is not itself H1–H4. `p_construction.outcome` is null. The shared adapter and the static-weight detector procedure are recorded as absent and not separately specified.

### Q arm

DEC-011 arm id `Qwen/Qwen2.5-0.5B`, recorded revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, ground truth `UNKNOWN`. Device default is CPU. A GPU run needs an independent environment record and counts against `q_gpu_full_runs`.

Q can support a claim about transfer, prediction, functional invariance, or empirical decision utility on that model, and only for the endpoints that a later preregistration freezes. Q cannot support a claim that a discovered mechanism is the true mechanism. Specification rule 8.

The same pipeline rule in DEC-011 forbids discovery code from branching on arm identity.

## 4. Ground Truth Semantics

```text
P ground truth = the frozen mechanism definition in P_SPEC.md
                 (L0H0 key-read, L1H1 value-read) plus the causal
                 construction record derived without discovery output.
                 The numeric causal predicate in P_SPEC.md is a
                 construction check, not an H1–H4 threshold.

Q ground truth = UNKNOWN (DEC-011).
                 There is no trusted mechanism identity, causal graph,
                 or planted label.
                 An external target would have to be independently
                 verified before it could change this sentence.
                 No such target is in the repository.
```

Discovery outputs must not be used to define, select, or validate P ground truth. DEC-011 forbids that. Specification section 2 forbids treating Qwen self-consistency or the discovery pipeline's own labels as proof.

## 5. Leakage Boundary

The self-model, at inference time, may receive only evidence that the allowed observation contract says is available before the prediction or the action. Specification section 14.

Prohibited as inference-time input, unless a future allowed observation contract explicitly includes them:

- planted mechanism labels, including the names L0H0, L1H1, `planted_A`, and `planted_B`
- ground-truth causal edges
- component identity when that identity is the hidden planted label rather than a site the observation contract already exposes
- post-intervention outcome
- future evidence
- test and holdout labels
- ground-truth transformation identity
- hidden benchmark metadata and planted-construction metadata that identifies the mechanism

Intervention information that is the candidate being predicted is input to `predict`. The outcome of applying that intervention is not input to that same prediction.

Training may use training-split outcomes. It may not use holdout outcomes, holdout labels, or the planted identity. `P_SPEC.md` sets `planted_identity_hidden_from_discovery: true`.

No label leakage into the self-model. An automated leakage test is required by the specification and does not exist yet. Writing it is not part of this phase.

## 6. H1 — Discovery

H1 asks whether held-out episodes yield the planted mechanism rather than a memorized component or a memorized outcome. Specification section 9. The hard correctness gate in that section is P only.

### P.H1

Score recovered objects against the P ground truth in section 4, on a holdout that was locked before the run.

A match, as proposed by the specification, requires all of:

1. the predicted mechanism maps to the planted mechanism;
2. the predicted causal direction matches that ground truth;
3. the predicted effect sign matches that ground truth on held-out interventions;
4. an evidence trace exists for the claim.

Proposed summary statistics, `PROPOSED — NOT FROZEN`:

- mechanism F1 >= 0.80
- causal-edge precision >= 0.80
- false discovery rate <= 0.20

The specification says all three must pass and that one favorable metric cannot replace another. DEC-011 has not adopted these numbers. They are not pass bars of this contract.

The P_SPEC causal predicate (intact accuracy >= 0.95 and the hop gaps) stays a construction check. It is not P.H1.

### Q.H1

Q.H1 is diagnostic. It is not a truth-validating discovery gate. There is no Q mechanism label to compute an F1 against.

The specification says to report, descriptively:

- number of candidate mechanisms
- reproducibility across seeds
- evidence completeness
- internal causal consistency

Those reports are not verified mechanism identity.

DEC-010 still requires a `Q.H1` key with status `PASS`, `FAIL`, or `NOT_EVALUATED`, and its GO rule requires `PASS`. P2.1 keeps that key. The observable quantities are in `## Q.H1 Operational Definition` below. They are a metric vector, not a pass bar. `DEC-010-A1` remains `PROPOSED / NOT APPLIED`. The numeric diagnostic bar remains deferred by DEC-011. Until the amendment is applied, a Q.H1 narrative must not be entered as `PASS`.

## 7. H2 — Counterfactual Prediction

H2 asks whether, before an intervention is executed, the self-model predicts its consequence better than the non-mechanistic baselines. Specification section 10.

**Prediction target.** The consequence of a declared intervention on the arm's outcome. On P the outcome site is the query position. On Q the outcome target is not frozen (section 2).

**Train data.** P training split in `P_SPEC.md`: seed offset 100, 8192 episodes. Q train episodes are not frozen.

**Holdout data.** P holdout is reserved in `P_SPEC.md` and must not be generated by construction or smoke. Membership for the five specification holdout kinds is not frozen. Scoring uses held-out interventions only. No tuning on that set.

**Intervention range.** Specification section 8 lists ablation, activation replacement, controlled noise, and patch or replacement where applicable. Each intervention has an id, source component, target component, magnitude, duration, control, and hypothesis id. The concrete set is not frozen.

**Baselines.** Section 10. The primary comparator is the strongest preregistered non-self-model baseline among those that the later preregistration actually includes. DEC-011 deferred baseline definitions, so "strongest" has no frozen membership yet.

**Primary metric.** Paired intervention-level MAE. `PROPOSED — NOT FROZEN` as a pass rule:

1. self-model MAE at least 20% lower than the strongest non-self-model baseline;
2. paired bootstrap 95% CI for the MAE difference excludes zero in favor of the self-model;
3. sign accuracy >= 75%;
4. measurement only on held-out interventions.

**Secondary metrics.** The specification does not name a separate secondary list. The sign accuracy and the CI are written as conjuncts of the proposed pass rule, not as optional extras. This draft does not add further metrics.

**Statistical comparison.** Paired comparison. Bootstrap 95% CI. The bootstrap count is not stated. This draft does not choose one.

## 8. H3 — Representation Invariance

H3 tests whether the learned mechanism object is tied to one coordinate system. Specification section 11 states that question without needing a historical experiment. DEC-011 forbids the H3 definition from depending on M18.7. The A4 result is not evidence and is not a label in this test.

**Property.** Apply an invertible coordinate transformation. Do not change the planted causal graph on P and do not change the intervention protocol. Compare:

- On P: mechanism identity agreement between the canonical and transformed representations, and agreement of causal-effect signs.
- On Q: functional invariance only. Prediction agreement and effect-sign agreement. Q-H3 is not proof of true mechanism identity. Specification section 11.

**Transformation family.** The draft names T1, T2, and T3 in `## H3 Transformation Family`. That naming is not a DEC freeze. DEC-011 still defers the PASS combination. The specification's phase-rotation formula is not one of the selected parameters.

**Phase rotation, if a later decision retains it.** The specification writes

```text
[X, Y] -> [Y, -X]
```

This is an invertible linear map. Applying it twice sends `[X, Y]` to `[-X, -Y]`. Applying it four times returns `[X, Y]`. It changes the coordinates of a two-dimensional feature and does not, by itself, edit the planted edges. H3 then asks whether the mechanism id (P) or the predictions (Q) agree across the two coordinate systems. Agreement is a statement about the self-model's representation, not a replay of an old A4 outcome. The A4 outcome is not the target and is not a pass label.

That formula is `PROPOSED — NOT FROZEN`. The specification's reason for requiring it is the M18.7/A4 failure. DEC-011 says that reason cannot be the basis of H3. P2.2 does not adopt the formula. The draft families are T1–T3, without matrices. See `## H3 Transformation Family`.

Proposed P pass conjuncts, `PROPOSED — NOT FROZEN`: identity agreement >= 90%, causal-effect sign agreement >= 90%, counterfactual MAE degradation <= 10%. The Q phrase "no material performance collapse" has no frozen number.

## 9. H4 — Behavioral Utility

H4 applies only under S1: the self-model is consumed to select or correct behavior. The specification forbids silently downgrading to S2. DEC-010 includes `P.H4` and `Q.H4` and does not define S1. If a later decision withdraws S1, H4's role has to be restated by that decision. This draft does not withdraw it.

H4 is not executed in this phase.

**Contrast.** The action candidate set is identical across conditions.

| Condition | What the decision process may use |
| --- | --- |
| Self-model available to the decision process | Predictions and mechanism output from the self-model |
| Self-model ablated from the decision process | The same self-model may still exist internally. The policy cannot read its predictions or mechanism output |
| Baseline policy | A simple non-self-model policy. Specification section 12 |

**Endpoint.** Change in task utility, or reduction in intervention damage, of the available condition relative to the ablated condition, on held-out scenarios. The specification does not choose a utility function. This draft does not choose one.

Proposed conjuncts, `PROPOSED — NOT FROZEN`: at least 10% improvement versus consumption ablation, paired bootstrap 95% CI excludes zero, held-out scenarios only, and the gain is not explained by extra compute or by candidate information given only to the available condition.

A predictor that is never read by the policy does not pass H4, even if H2 would pass.

## 10. Baselines

Specification section 13 names six categories. DEC-011 defers baselines, so this list is `PROPOSED — NOT FROZEN`. No category was added or removed.

| Id | Category | Role |
| --- | --- | --- |
| B0 | Naive / mean predictor | Predict the training-set mean effect |
| B1 | Readout-only | Predict the outcome from observable readout features, without a mechanism object |
| B2 | Linear probe | Linear model on frozen representations |
| B3 | Cold start | No transferred self-model |
| B4 | Random-direction / null | Randomized intervention-to-control relationship |
| B5 | Consumption ablation | H4 contrast. The self-model is not available to the policy |

B1, B2, B3, and B4 are the non-self comparators named in specification section 10. B0 is an additional named baseline in section 13. B5 is the H4 ablation, not an H2 predictor.

"Cold start" in the specification is motivated by M18.7. The category "no transferred self-model" can be implemented without using M18.7 as a result. The historical result is not a baseline target.

## 11. Negative Controls

Included only where a source already uses them to separate the scientific question from a shortcut.

| Control | What it blocks | Source |
| --- | --- | --- |
| Random direction / null (B4) | A gain that appears for an intervention with no stable relationship to the outcome | Spec §13 |
| Control or null condition on every intervention | Treating an ordinary action, or an uncompared edit, as an intervention effect | Spec §8 |
| Irrelevant intervention | A policy or predictor that improves when the edited site is not the one the hypothesis names. The specification requires a control on each intervention and does not name a separate site list | Spec §8, as the control arm of the intervention record |
| Shuffled component / control | Assignment of the planted role to the wrong head. On P this is a specificity check of the kind `P_SPEC.md` already computes as `planted_specificity_gap` for construction, not as an H gate | `P_SPEC.md` construction predicate. Not promoted to a gate |
| Representation-transformation control | A comparison that changes coordinates and also changes the causal graph or the intervention set. H3 requires the causal behavior to stay fixed | Spec §11 |
| Placebo for consumption | Extra compute or extra candidate information available only in the H4 "available" condition | Spec §12 |

No further placebo family is added.

## 12. Ablations

Ablations remove one information channel. They do not reveal planted labels.

| Ablation | Removed channel | Distinction it supports |
| --- | --- | --- |
| Outcome-only | Mechanism hypothesis and component identity are withheld from the learner. Outcomes on the training split remain | Outcome prediction without a mechanism object |
| Component or context | Mechanism labels withheld. Component or regime indicators that the observation contract already allows may remain | Component or context learning is not mechanism identity |
| Mechanism representation | The inspectable mechanism object is withheld from the predictor or, in H4, from the policy (B5) | A gain that disappears when the mechanism object is unreadable is not explained by outcome memory alone |
| Consumption | Internal self-model state may remain. The policy cannot read it | S1. Section 9 |

Discovery training must not receive the planted identity in any of these ablations. An ablation that "removes" a label by first attaching it is leakage.

## 13. Train / Calibration / Test Isolation

| Split | Status |
| --- | --- |
| Construction set | `P_SPEC.md` train seed offset 100, n 8192, and dev seed offset 200, n 512. Construction seeds already used are listed there. Holdout access during construction is forbidden |
| Training set for MRSM | The P construction train split is the only frozen episode generator. Using it as the MRSM training set is not yet a preregistration act. Q training data is not frozen |
| Calibration set | Not separately defined. P_SPEC dev data is where the construction predicate is measured. That use is not an MRSM calibration protocol |
| Holdout | Reserved in `P_SPEC.md`. Not generated by construction or smoke. Two pre-freeze ranges are marked `ALREADY_GENERATED_PRE_FREEZE` (seeds 311 and 2029). Specification section 7 also requires sample, context, regime, mechanism, and transformation holdouts, hashed before any scientific execution. Membership of those five is not frozen |
| OOD | The context, regime, and mechanism holdouts in specification section 7, once membership exists. Not frozen |
| Transformation holdout | Specification section 7. The map family is not frozen (section 8) |

The holdout must be locked before scientific execution. This draft does not lock it and does not generate it.

`P_CONSTRUCTION_ATTEMPTS.yaml` records that two pre-freeze smokes generated and hashed holdout tokens. They were not counted as MRSM full runs. This contract does not reopen those manifests and does not treat them as the locked MRSM holdout.

## 14. Statistical Protocol

Primary endpoints, as named by the specification and not adopted as numeric gates:

| Gate | Primary endpoint | Status |
| --- | --- | --- |
| P.H1 | Mechanism recovery against P ground truth, summarized in the specification as mechanism F1 | Metric name proposed. Threshold `PROPOSED — NOT FROZEN` |
| Q.H1 | Diagnostic or transfer report. Not an F1 against a Q mechanism label | Epistemic role set in P2.1. `DEC-010-A1` not applied. Numeric bar deferred |
| H2 | Paired intervention-level MAE | Threshold `PROPOSED — NOT FROZEN` |
| H3-P | Mechanism identity agreement across a representation change | Threshold and transform `PROPOSED — NOT FROZEN` |
| H3-Q | Functional agreement across a representation change | Not mechanism identity. "Material collapse" unresolved |
| H4 | Utility or damage contrast versus consumption ablation | Utility function not chosen. Threshold `PROPOSED — NOT FROZEN` |

Secondary endpoints are the other conjuncts written next to those primaries in specification sections 9–12 (edge precision, FDR, sign accuracy, sign agreement, MAE degradation, bootstrap CI). They are proposed conjuncts, not a license to drop a failed primary.

Paired comparisons are required where the specification says paired. The interval is a paired bootstrap 95% CI. The number of resamples is not stated and is not chosen here. Effect size, the per-seed or per-episode distribution, and the number of independent episodes are report items in specification section 15. No multiple-comparison correction is stated. None is invented.

Seed policy that is already written: do not cherry-pick seeds; P benchmark generation uses multiple seeds; `P_SPEC.md` fixes construction seed 1729 and the split offsets. The MRSM run seed list is not frozen.

Do not tune any bar after seeing holdout results. Do not replace a failed endpoint with a correlated metric. Specification section 15 and rules 2, 3, and 5.

Replay tolerances in `configs/m22_1_r_replay.yaml` are not this protocol.

## 15. Generalization

The specification's H1 question is recovery on held-out data rather than memorization of components or outcomes, and it asks for a mechanism family that is absent from training. The cells below are the comparisons that question requires. Their membership is not frozen. They do not add a new empirical claim.

| Contrast | What it separates |
| --- | --- |
| Same mechanism, new component | Mechanism identity from a memorized head or site |
| Same component, new mechanism | A favorite site from the causal role |
| Same outcome, different mechanism | Outcome prediction from mechanism knowledge |
| Novel representation transformation | A coordinate-specific encoding from a representation-invariant mechanism object |

On the frozen P construction there is one two-hop mechanism and two planted heads. The "new mechanism" and "mechanism family absent from training" cells are not populated by `P_SPEC.md`. Filling them by renaming L0H0 and L1H1 is not allowed. Populating them by adding mechanisms is not authorized. P2.1 kept the frozen construction. A new mechanism still requires a new DEC.

Q has no mechanism ground truth, so Q cannot score these cells as discovery. Q can still report whether predictions or decisions change across components, outcomes, and transformations. That report is transfer evidence, not a mechanism-identity result.

## 16. Failure Modes / Non-Claims

Failure of a gate is a result about this preregistered method on this benchmark. It is not a theorem about all possible methods.

```text
H1 failure ≠ Self-Model impossible in principle
H2 failure ≠ causal discovery impossible
H3 failure ≠ all representation invariance impossible
Q failure ≠ P failure
P failure ≠ a Qwen conclusion
```

A pass does not prove the following. Specification section 25, specification rule 8, and the arm definitions:

```text
Prediction ≠ Mechanism Knowledge
Calibration ≠ Self-Model
Qwen success ≠ Ground-truth mechanism discovery
Invalidity detection ≠ Permission to modify
```

Also excluded as results of this benchmark, whatever the gates do: general consciousness, general introspection, a complete Self-Model, universal mechanism understanding, self-repair, safe self-modification, self-improvement, generalization to all LLMs, and general causal abstraction.

M22.1 remains `CANDIDATE` in `STATE.yaml`. The current replay tier is `REPLAY_BLOCKED`. Neither fact is an MRSM gate result. DEC-008 still says the near-rank-1 structure at L23 must not be cited as internal mechanism or self-model structure.

## 17. Freeze Criteria

Nothing in this version is the MRSM preregistration. A scientific MRSM run is not allowed until each row below is frozen by a DEC or by the preregistration step the governing DEC already names, and until `p_construction.outcome` is `DONE`.

| Object | State on this date |
| --- | --- |
| Mechanism construction | Frozen as `P_SPEC.md` L0H0/L1H1. P2.1: `RESOLVED`. No replacement authorized. Specification examples are non-authoritative |
| Dataset | P train and dev generators are in `P_SPEC.md`. Q data is not frozen. MRSM use of the P generator is not a preregistration |
| Holdout | Reserved and unfrozen as an MRSM split. Five specification kinds have no membership |
| Intervention set | Types named in specification section 8. The concrete set is not frozen |
| Baselines | Categories named in section 10. `PROPOSED — NOT FROZEN` |
| Negative controls | Section 11. Not a frozen protocol |
| Ablations | Section 12. Not a frozen protocol |
| Metrics | Named by the specification. Not adopted |
| Statistical protocol | Paired bootstrap 95% CI is proposed. Resample count, multiple-comparison rule, and MRSM seed list are not frozen |
| Thresholds | Specification numbers are `PROPOSED — NOT FROZEN`. DEC-011 defers them. P_SPEC numbers are not gates |
| Seed policy | Construction seeds are frozen. MRSM run seeds are not |
| Budget | DEC-011 and `MRSM_BUDGET.yaml` govern. P2.1: `RESOLVED`. Specification budget sentences are non-authoritative. MRSM clock not started. Construction outcome null |
| Artifact schemas | Specification section 5 and the planted ground-truth file are proposals. No `artifacts/planted_ground_truth.json` is frozen |
| Terminal rule | DEC-010 governs. P2.1: `RESOLVED`. Specification sections 1.2 and 24 are non-authoritative. `DEC-010-A1` (Q.H1 epistemic clause) is `PROPOSED / NOT APPLIED` |
| Q revision pin | The recorded revision is frozen as a string. The harness attempt did not load it. DEC-011 still says Q full-run preconditions include a static revision and recorded weight hashes |
| Replay contract | Frozen for M22.1-R. Out of scope for H1–H4 numbers |

## 18. Decision Logic

P2.1 assigns the governing rules. This section records those assignments. It does not freeze the contract.

| Topic | P2.1 status | What governs |
| --- | --- | --- |
| Terminal tree | `RESOLVED` | DEC-010. Specification §1.2 and §24 do not define another tree. GO still requires all eight keys, including `Q.H1` |
| Q.H1 meaning | `RESOLVED_WITH_AMENDMENT_REQUIRED` | `P.H1` is the truth-validating discovery test. `Q.H1` is diagnostic / transfer evidence. `Q.H1 PASS` is not verified mechanism discovery. `DEC-010-A1` is `PROPOSED / NOT APPLIED`. Until it is applied, do not record `Q.H1` as `PASS` |
| Planted mechanism | `RESOLVED` | `P_SPEC.md` L0H0 and L1H1. No replacement authorized |
| H3 family | Draft names three families. Not a DEC freeze | T1 affine, T2 permutation, T3 orthogonal. Parameters and the PASS combination stay deferred. M18.7 is not the reason |
| Run accounting | `RESOLVED` | DEC-011. Full run = locked-holdout read. 8 full runs. Q GPU sub-limit 2. Clocks not started |

The executable DEC-010 function remains the frozen table. This draft does not wrap it and does not replace it. The P2.0 conflict list in `reports/P2_0_contract_conflicts.md` is the record of the disagreement. P2.1 does not delete that record.

## Q.H1 Operational Definition

P.H1 is a ground-truth-validating discovery gate. It compares a discovered object with the frozen planted mechanism in `P_SPEC.md`.

Q.H1 is a diagnostic/transfer gate. It asks whether the candidate self-model on Q produces intervention-linked predictions that can be checked against executed interventions, using the same observable behavioral contract shape as P. Q is not ground-truth mechanism validation.

```text
Q.H1 PASS
≠
verified mechanism discovery on Q
```

```text
Q.H1 FAIL
≠
proof that the method cannot discover mechanisms on real models
```

A Q failure is evidence against the current transfer of the method under the tested scope. It is not a universal impossibility claim.

For every Q holdout claim the artifact must contain:

- `claim_id`
- representation or mechanism hypothesis
- intervention
- predicted behavioral delta
- predicted direction
- uncertainty
- applicable regime
- evidence references
- falsification status

The prediction is then compared with an intervention that is executed after that prediction is recorded.

The observable vector, with no composite score and no numeric pass bar, is:

```text
Q_H1_diagnostic = {
  "prediction_quality": sign correctness and MAE on the pre-registered outcome,
  "counterfactual_consistency": agreement among predictions that share a hypothesis and differ only by a declared intervention,
  "falsification_behavior": whether a contradicted prediction is marked falsified,
  "abstention_quality": whether high uncertainty or inapplicability declines the claim instead of asserting it,
  "scope_consistency": whether the stated regime matches the episode the intervention was applied to,
  "status": "DIAGNOSTIC_ONLY"
}
```

`prediction_quality` has two recorded parts, sign correctness and MAE. They stay separate entries in the vector. No weight combines them. Abstention applies when the claim records uncertainty or an applicability decision. No abstention policy threshold is set here.

The exact numerical pass threshold remains deferred.

## Q.H1 Leakage Boundary

The Q arm may not use:

- hidden mechanistic labels
- manually selected correct heads
- post-hoc mechanism labels
- unpublished ground-truth circuit annotations
- outcome-derived mechanism identity labels
- test intervention outcomes when constructing the pre-intervention prediction

The Q model may use only the observations and evidence the rest of this contract already permits.

| Information class | What it contains | When it may be used |
| --- | --- | --- |
| Train / discovery | Training-split observations, training-split outcomes, and hypotheses fit on that split | Before the holdout prediction is issued |
| Locked holdout | Holdout episode identity and the pre-intervention observation the contract allows | As input to the prediction. Not as a source of outcome labels |
| Post-intervention outcome | The behavioral result of executing the intervention | Only after the prediction artifact for that `claim_id` exists. Used to score the vector. Not used to build the prediction |

The prediction must be generated before the corresponding intervention outcome is observed. A run artifact is testable on this point when each claim records a prediction timestamp or write order that precedes the outcome record for the same `claim_id`. This phase does not create those artifacts.

## H3 Representation Invariance

H3 is not justified by M18.7 or by A4. The historical phase-rotation result is not evidence for this gate.

If two internal representations encode the same underlying causal state or mechanism and differ only by an allowed invertible representation transformation, the self-model's causal prediction should remain invariant after the declared handling of that transformation.

The invariant is not that raw internal vectors stay identical.

The invariant is that these stay equivalent:

- predicted causal consequence
- mechanism identity at the allowed abstraction level
- scope
- falsification behavior

H3 does not ask the raw self-model parameters to stay numerically identical.

Four properties stay distinct:

| Property | What it is | What it is not |
| --- | --- | --- |
| A. Mechanism discovery | Recovering a mechanism object from observations | A coordinate identity |
| B. Mechanism identity | On P, a match to the frozen L0H0/L1H1 definition. On Q, a hypothesis label only | A proof that the Q hypothesis is the true circuit |
| C. Representation invariance | Equivalence of the invariant list across an allowed transformation of coordinates | Identity of raw vectors |
| D. Predictive invariance | Equivalence of the causal prediction across that same transformation | Mechanism-identity truth on Q |

## H3 Transformation Family

The draft names three families. This is not a DEC freeze and not a numerical parameterization. DEC-011 still defers the PASS combination. No matrix, permutation, subspace, or offset is chosen.

### T1 — invertible affine reparameterization

For a representation `x`:

```text
x' = A x + b
```

`A` is invertible. The same map is applied at the representation interface. The underlying causal semantics of the task stay in place. `A` and `b` are not selected in this phase.

### T2 — permutation / relabeling of representation coordinates

```text
x' = P x
```

`P` is a permutation matrix. Coordinate identity changes. The encoded causal state does not. The particular permutation is not selected in this phase.

### T3 — invertible orthogonal transformation

```text
x' = Q x
```

with

```text
Q^T Q = I
```

The coordinate basis changes. The represented state does not. `Q` is not selected in this phase.

A transformation that edits the planted edges, the intervention, or the task labels is not an H3 transformation. It is a different system.

Every H3 transformation must satisfy:

1. invertibility
2. a known mapping
3. preservation of task input and output semantics
4. no access to test outcomes during selection
5. no transformation-specific threshold tuning
6. identical intervention semantics before and after the transformation
7. evaluation on held-out examples that were locked before the run

## H3 Evaluation Metrics

No single H3 score is defined. The record for each paired comparison contains at least:

- prediction equivalence
- sign agreement
- normalized prediction error difference
- mechanism-id consistency at the declared abstraction level
- scope consistency
- uncertainty consistency
- falsification consistency

The comparison is paired:

```text
original representation
        vs
transformed representation
```

on identical locked holdout cases. No numeric equivalence cutoff is set. Threshold status is deferred.

## P/Q Epistemic Boundary

```text
P = ground truth known
Q = ground truth unknown
```

| Property | P ground truth? | Q ground truth? | What can be claimed? |
| --- | --- | --- | --- |
| Discovery | yes | no | P discovery can be scored against the frozen planted mechanism. Q discovery cannot be scored as true or false mechanism identity |
| Mechanism identity | yes | no | P identity is L0H0 and L1H1 as written in `P_SPEC.md`. A Q hypothesis is not an identity verdict |
| Prediction | yes, against the planted outcome | empirical outcome only | A Q prediction can be right or wrong about the observed behavior. That does not identify the Q mechanism |
| Representation invariance | testable against the planted identity and the outcome | testable as prediction, scope, and falsification equivalence only | Equivalence under T1–T3. Not uniqueness of the representation |
| Transferability | partially testable, by moving a P-built procedure onto new P episodes | diagnostic | Q results speak to transfer under the tested scope. They do not establish mechanism identity |

Q prediction success does not prove Q mechanism identity.

## Non-Claims

H3 does not prove:

- that the representation is uniquely identified
- that the mechanism is the only possible causal explanation
- that all representation transformations preserve semantics
- that arbitrary neural representations are interchangeable
- that success on P automatically proves success on Q
- that representation invariance alone proves self-model validity

H3 only tests the declared families under the declared conditions.

Q.H1 does not prove that a Qwen mechanism is the true mechanism. Q.H1 failure does not prove that no method can discover mechanisms.

The absent pinned Qwen snapshot remains `REPLAY_BLOCKED`. That block is not a scientific failure and is not a Q.H1 result.

## Deferred Decisions

- Numerical H1–H4 thresholds, including every Q.H1 and H3 cutoff. Status: `PROPOSED — NOT FROZEN` where the implementation specification stated a number, and unset where it did not.
- Final approval and application of `DEC-010-A1`.
- Final preregistration freeze.
- Qwen execution, which still waits on the pinned offline snapshot and on DEC-011's Q full-run preconditions.
- `p_freeze`. `p_construction.outcome` is null. The MRSM clock has not started.
- Locked holdout membership for the MRSM run.
- Concrete `A`, `b`, `P`, and `Q` for T1–T3, and the PASS combination DEC-011 deferred.

MRSM remains not ready for scientific execution.


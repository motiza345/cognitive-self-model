# Historical experiment index

Archive index of imported Colab notebooks. Reported statuses are historical.
They are not scientific validations of the current repository.
Where an objective or result is not explicit in the notebook, the field stays `NOT_DETERMINED`.
`scientific_interpretation` stays `UNKNOWN`.

## HIST-0001

- Notebook: `Untitled0__archive01.ipynb`
- Original filename: `Untitled0.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M1`
- Objective: NOT_DETERMINED
- Model: `GPT-2`, `gpt2`
- Important configuration: heads heads = [(l, h) for l in range(self.n_layers) for h in range(self.n_heads)]; heads = [h_tuple for h_tuple, val in sorted_discovered_heads[:top_k_synergy]
- Reported metrics: MSE: 🚨 [CAUSAL INCONSISTENCY] Predictive MSE (inf) > Threshold (0.35)
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `ca68f50189680db758775e5ab9aa556bbcd8aeca6585f742f9f5663d34dccfe1`
- Notes: parse `OK`. duplicate `DISTINCT`

## HIST-0002

- Notebook: `Untitled1__archive01.ipynb`
- Original filename: `Untitled1.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M2`
- Objective: NOT_DETERMINED
- Model: `GPT-2`, `gpt2`
- Important configuration: NOT_DETERMINED
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `2b7c06ca27817ab866d46931f30023e39710f9888a1ecd602e7516cc82911237`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, patching, intervention

## HIST-0003

- Notebook: `Untitled10__archive01.ipynb`
- Original filename: `Untitled10.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M7.10`, `M8.1`, `M8.2`, `M8.3`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42; heads heads = [(21, 6), (21, 1), (17, 1), (14, 4), (10, 0)]
- Reported metrics: accuracy: • Clean Tasks Accuracy (Base Model) : 6/10 (60.0%); correlation: /tmp/ipykernel_2736/4164165410.py:332: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `d8564b6e10a4bd118420118194bf826ed9cb9a33125c2f5d7a39672ac510873a`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, intervention, self-model, adaptive probing

## HIST-0004

- Notebook: `Untitled11__archive01.ipynb`
- Original filename: `Untitled11.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M9.2`, `M9.1`, `M9.3`, `M9.3A`, `M9.3B`, `M9.4`, `M9.5`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: MAE: • Predictive Self-Model MAE : 0.1345 (Sign Accuracy: 17.5%); accuracy: • Predictive Self-Model MAE : 0.1345 (Sign Accuracy: 17.5%); correlation: • Mean Cross-Component Correlation (LOCO) : +0.3382; sign accuracy: • Predictive Self-Model MAE : 0.1345 (Sign Accuracy: 17.5%)
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `3435f76f284fb1c028e04c53f41c20a88cae8358525d81658fe1fa9569f69f0e`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, intervention, counterfactual, self-model

## HIST-0005

- Notebook: `Untitled12__archive01.ipynb`
- Original filename: `Untitled12.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M11`, `M11.1`, `M11.2`, `M11.3`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: MSE: epoch 000 | joint MSE loss = 0.16510; correlation: Mean Correlation (ρ) | Learned: +0.1103 | Fixed: +0.0607 | Random: +0.3380 | Passive: -0.2445
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `1a63df2ec958607becac9c1ce827a24f65d6c3ac599486dfe5a1643efd22465b`
- Notes: parse `OK`. duplicate `DISTINCT`. methods intervention

## HIST-0006

- Notebook: `Untitled13__archive01.ipynb`
- Original filename: `Untitled13.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M12`, `M12.1`, `M12.2`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: MAE: 25.0% (1/4) | σ <= 1.2268 | MAE = 0.0316 | 100.0%; accuracy: • Overall Direction Accuracy : 50.0%
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `9f4fbf40e8997fea4890f5dd824516355a88f1e915a69aa5d496be0570edbfc6`
- Notes: parse `OK`. duplicate `DISTINCT`. methods intervention, counterfactual, self-model, epistemic graph

## HIST-0007

- Notebook: `Untitled14__archive01.ipynb`
- Original filename: `Untitled14.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M14.3`, `M14.2`, `M14.4`, `M15`, `M14.5`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: MAE: • Calibration Partition MAE (Unseen) : 0.0075; accuracy: • 3-Class Sign Accuracy (Calib) : 66.7%; correlation: • Legacy Sensor Correlation with True Prob (r_s): +1.0000 (MODERATE); sign accuracy: • 3-Class Sign Accuracy (Calib) : 66.7%; coverage: • Interval Coverage Error (ICE-90) : 0.082
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `9525f63aeda8ceee638eeaefadaaba85fa204babf84385d26d40c9a8da3f8cd7`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, intervention, self-model, causal graph

## HIST-0008

- Notebook: `Untitled15__archive01.ipynb`
- Original filename: `Untitled15.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M15`, `M14.5`, `M15.5`, `M16`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: MAE: Prediction Error (MAE) : 0.0052
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `b4e9085263606fd782ecac53756405194c9bcfecd2459047dc2f4cdb35f44be4`
- Notes: parse `OK`. duplicate `DISTINCT`. methods intervention

## HIST-0009

- Notebook: `Untitled16.ipynb`
- Original filename: `Untitled16.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M16.5`, `M17`, `M16`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: MSE: [0;32m--> 371[0;31m [0md[0m [0;34m=[0m [0mengine_ig[0m[0;34m.[0m[0mselect_ig_active_dose[0m[0;34m([0m[0mtarget[0m[0;34m.[0m[0mkey[0m[0;34m,[0m [0meval_ctx[0m[0;34m,[0m [0mig_doses[0m[0;34m)[0m[0;34m[0m[0;34m
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `623e7a01eb21c0fa74f1275abd3b64f4b73e32e397d4f7d9f9e11769fdcf863e`
- Notes: parse `OK`. duplicate `DISTINCT`

## HIST-0010

- Notebook: `Untitled17.ipynb`
- Original filename: `Untitled17.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M16`, `M17.5`, `M17`, `M17.6`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `74f152c650e5e076f3fe9dc5ce180f227653952631f452818940a80dd1169647`
- Notes: parse `OK`. duplicate `DISTINCT`. methods intervention

## HIST-0011

- Notebook: `Untitled18.ipynb`
- Original filename: `Untitled18.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M16`, `M17`, `M17.6`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `323b2804709e78f892223ffcb1847f149c0b46796ab5a89fc2e9038038fa98c2`
- Notes: parse `OK`. duplicate `DISTINCT`. methods intervention

## HIST-0012

- Notebook: `Untitled19.ipynb`
- Original filename: `Untitled19.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M17.7`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `21ea6d383441973fed670868e75a0e4ea42ca1b368bbdf726a2f97dbfd7ee760`
- Notes: parse `OK`. duplicate `DISTINCT`

## HIST-0013

- Notebook: `Untitled2__archive01.ipynb`
- Original filename: `Untitled2.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M2`, `M2.2`, `M3`
- Objective: NOT_DETERMINED
- Model: `gpt2`
- Important configuration: NOT_DETERMINED
- Reported metrics: MAE: Mean Absolute Error (MAE): 0.3043
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `47852e06a0272cb9e5a938fbf58a8a6619402e7ba3d205d9a2611a8c15f5229e`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, patching, intervention, counterfactual, self-model

## HIST-0014

- Notebook: `Untitled3__archive01.ipynb`
- Original filename: `Untitled3.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M0`, `M7`, `M1`, `M2`, `M3`, `M4`, `M5`, `M6`, `M2.2`, `M3.1`, `M3.2`, `M3.3`, `M3.4`, `M3.5`, `M3.6`, `M3.7`
- Objective: NOT_DETERMINED
- Model: `gpt2`, `GPT-2`
- Important configuration: NOT_DETERMINED
- Reported metrics: MAE: Mean Absolute Error (MAE): 0.3043; accuracy: Sign Accuracy (Additive): 11/20 (55.0%); sign accuracy: Sign Accuracy (Additive): 11/20 (55.0%)
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `0465f06c107e305173796c9a853e00d17f87be95638eb5fb92a6b3e7688d39fa`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, patching, intervention, self-model

## HIST-0015

- Notebook: `Untitled4__archive01.ipynb`
- Original filename: `Untitled4.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M4.1`, `M4.2`, `M4.3`
- Objective: NOT_DETERMINED
- Model: `gpt2`
- Important configuration: NOT_DETERMINED
- Reported metrics: MAE: Self-Model Prediction MAE: 0.9610; MSE: [0;32m--> 163[0;31m [0mevaluate_dynamic_task[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0;34m([0m[0;36m9[0m[0;34m,[0m [0;36m9[0m[0;34m)[0m[0;34m,[0m [0;34m([0m[0;36m10[0m[0;34m,[0m [0;36m7[0m[0;34m)[0m[0;34m,[0m [
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `25473713ab7bbfe5e56a05795496b810439d31abb90c0e1c42450dd793e88675`
- Notes: parse `OK`. duplicate `DISTINCT`. methods intervention, self-model

## HIST-0016

- Notebook: `Untitled5__archive01.ipynb`
- Original filename: `Untitled5.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M4.3b`, `M4.1`, `M4.2`, `M4.3`, `M4.3c`, `M4.4`, `M4.4b`, `M4.5`, `M4.5b`, `M4.5c`, `M4.5d`, `M4.5e`, `M4.5f`, `M4.5g`, `M4.5h`, `M4.6`, `M4.6b`, `M4.6c`, `M4.6d`, `M4.6e`, `M4.7`, `M4.7b`, `M4.8`
- Objective: Resolve whether C4 <-> C6 failure was due to an underpowered held-out set (2 points)
- Model: `gpt2`
- Important configuration: layers layers = []
- Reported metrics: MAE: 👉 MAE Learned Self-Model: 0.0013 | MAE Naive Additive: 0.0066; MSE: Context | Model | Params | Held-Out MAE | Held-Out RMSE | R^2 | AIC; accuracy: 🔬 STAGE 2: UN-GATED VS GATED ZERO-PROBE PREDICTION ACCURACY; z-score: • Z-Score: 10.22 | p-value: 0.0000
- Reported result: HISTORICAL_REPORTED: FAIL, PASS
- Historical decision: HISTORICAL_REPORTED: FAIL, PASS
- SHA256: `d0bb9b690912f92ed902d7a594472af7d4b652f023a2c8532a8b29bd90c8399a`
- Notes: parse `OK`. duplicate `DISTINCT`. methods intervention, self-model

## HIST-0017

- Notebook: `Untitled6__archive01.ipynb`
- Original filename: `Untitled6.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M4.8b`, `M4.9`, `M4.5`, `M5.0`, `M5.1`, `M5.1.5b`, `M5.1.6`, `M5.1.6b`, `M5.2`
- Objective: NOT_DETERMINED
- Model: `GPT-2`, `gpt2`
- Important configuration: layers layers = []; layers = [7, 8, 9, 10]; heads heads = [(9, 9, 50.0), (10, 7, 40.0)]; heads=[(9, 9)]; heads = [src_a, src_b, (7, 10), (8, 8), (9, 0), (10, 0), (8, 2), (7, 0)]
- Reported metrics: MAE: Mean Zero-Probe MAE: 0.3479 (Baseline: 0.6226) | Overall Improvement: +44.13%; accuracy: • SelfGraph Routing Accuracy to Node 0: 93.75% (15/16); correlation: • Causal Gain Pearson Correlation (r): +0.5947 (p-value: 1.5098e-02)
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `b83e2587f20e2893ba01260b74936703df2896f401cbab0160eb8e36f51ce01c`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, intervention, counterfactual, blind discovery, self-model

## HIST-0018

- Notebook: `Untitled7__archive01.ipynb`
- Original filename: `Untitled7.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M5.3`, `M5.4`, `M5.9`, `M5.3b`, `M5.5`, `M5.6`, `M5.7`, `M5.8`, `M5.9b`
- Objective: NOT_DETERMINED
- Model: `gpt2`, `GPT-2`
- Important configuration: layers layers = [7, 8, 9, 10]; heads n_heads=[(9, 9), (10, 7)]
- Reported metrics: accuracy: • Fault Localization Accuracy: 100.00% (Correctly pinpointed L9H9)
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `4eb18d76739338bea534ce9db8b58d903e3e3625f2b6e0cecafae71954165537`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, intervention, self-model

## HIST-0019

- Notebook: `Untitled8__archive01.ipynb`
- Original filename: `Untitled8.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M5.8`, `M6.5`, `M6`, `M6.1`, `M6.2`, `M6.3`, `M6.4b`, `M6.4`, `M1.0`, `M6.6`, `M6.7`, `M6.8`, `M7.10`
- Objective: NOT_DETERMINED
- Model: `Qwen2.5-0.5B.`, `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42; heads heads = [(21, 6), (21, 1), (17, 1), (20, 5), (17, 2)]; heads=[(l, h)]; heads=[(21, 6)]; heads=[chosen_fault]; heads=[(21, 6), (21, 1)]; heads = [(21, 6), (21, 1), (17, 1)]
- Reported metrics: MAE: • Mean Calibrated Damage Error (MAE): 0.7730 Logits; MSE: [1;32m 25[0m [0mSEED[0m [0;34m=[0m [0;36m42[0m[0;34m[0m[0;34m[0m[0m; accuracy: • Directional Sign Accuracy: 68.75% (Strict Positive/Negative Alignment); correlation: • Pearson Correlation (r): 0.5299; sign accuracy: • Directional Sign Accuracy: 68.75% (Strict Positive/Negative Alignment); z-score: • Target Circuit Z-Score: 10.14 sigma (Statistically Unique if Z > 2.0)
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `ebca60ee29c4eaf4977b763183e19f0377346274669ee9d3f8aead098a39ba45`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, patching, intervention, counterfactual, blind discovery, self-model, adaptive probing

## HIST-0020

- Notebook: `Untitled9__archive01.ipynb`
- Original filename: `Untitled9.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M0`, `M1`, `M2`, `M7.2`, `M7.1`, `M7.3`, `M7.4`, `M7.5`, `M7.6`, `M7.7`, `M7.8`, `M7.9`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42; heads heads = [(21, 6), (21, 1), (17, 1)]; heads=[h]; heads=[best_head]; heads = [(21, 6), (21, 1)]; heads=[head]; heads=[(21, 6)]
- Reported metrics: MAE: Budget (K) | Policy | MAE (Logits) | Sign Acc (%) | 2σ Coverage (%); accuracy: • Directional Sign Accuracy: 66.67% across unseen OOD benchmarks; sign accuracy: • Directional Sign Accuracy: 66.67% across unseen OOD benchmarks; coverage: • 2-Sigma Uncertainty Coverage: 66.67% of true observations fall within ±2σ
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `e2cdc3c39b6e2e44233ba6250ad11865d1c952b09bf411a48dc4eb1e50a42774`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, intervention, counterfactual, epistemic graph

## HIST-0021

- Notebook: `m10.ipynb`
- Original filename: `m10.ipynb`
- Archive source: `archive_01` (`Colab Notebooks-20260930T183832Z-1-001.zip`)
- Detected milestone: `M9.4`, `M9.5`, `M9.3A`, `M10.1`, `M10.2`, `M10.2b`, `M10.2B`, `M10.2c`, `M10.2C`, `M10.2d`, `M10.2D`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`, `Qwen2.5-0.5B`
- Important configuration: seeds 42; heads n_heads = 14; heads = [0, 2, 4, 5, 7, 9, 11, 13]; heads = [0, 2, 5, 11]
- Reported metrics: MAE: Target Head | Phi MAE | Phi ρ | Global Mean MAE | Perm p-val | FDR Sig (q < 0.05); correlation: 📊 1. GROUND-TRUTH TARGET CORRELATION: Corr(Y) [Actual System Behavior]
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `9193407e834ced80612f3e1739f8242755240168ec05354b9c42a034ec13e410`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, intervention

## HIST-0022

- Notebook: `01_repository_setup.ipynb.ipynb`
- Original filename: `01_repository_setup.ipynb.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M20.6.3.2`
- Objective: NOT_DETERMINED
- Model: `NOT_DETERMINED`
- Important configuration: NOT_DETERMINED
- Reported metrics: MAE: 3163 | print(f" MAE : {validation_point_metrics['mae']:.4f}")
- Reported result: HISTORICAL_REPORTED: PASS, FAIL, SUCCESS
- Historical decision: HISTORICAL_REPORTED: PASS, FAIL, SUCCESS
- SHA256: `bcb7755c7bd02d98459a6520bb99093fb3900a50dc222e632b68992da5cbd824`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, self-model, self model

## HIST-0023

- Notebook: `M20_6_3_2_Execution_Readiness.ipynb.ipynb`
- Original filename: `M20_6_3_2_Execution_Readiness.ipynb.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `NOT_DETERMINED`
- Important configuration: NOT_DETERMINED
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `c6feec98bbeff704ca90ee9ed9d8710153759b24478833a1037f2e9b342e65d5`
- Notes: parse `OK`. duplicate `DISTINCT`. methods self-model

## HIST-0024

- Notebook: `Untitled`
- Original filename: `Untitled`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `NOT_DETERMINED`
- Important configuration: NOT_DETERMINED
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `7ef6ba560dbc7895af48ccb73e4ff854247db0184fc506b1a7865bb4571444ca`
- Notes: parse `OK`. duplicate `NEAR_DUPLICATE`. similar_to HIST-0040

## HIST-0025

- Notebook: `Untitled(1)`
- Original filename: `Untitled(1)`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M17.17`, `M17.12`, `M17.13`, `M17.14`, `M17.15`, `M17.15b`, `M17.16`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: NOT_DETERMINED
- Reported result: HISTORICAL_REPORTED: SUCCESS
- Historical decision: HISTORICAL_REPORTED: SUCCESS
- SHA256: `e461280963505a86249d4e20109782778a21acc778900963be78267fd36da435`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, self-model

## HIST-0026

- Notebook: `Untitled0__archive02.ipynb`
- Original filename: `Untitled0.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `gpt2`, `GPT-2`
- Important configuration: seeds 0, 1
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `e397b49c8943f28859246a282f2ec504d94305796d031f52665f9bcd1ae66d84`
- Notes: parse `OK`. duplicate `DISTINCT`. methods patching

## HIST-0027

- Notebook: `Untitled1__archive02.ipynb`
- Original filename: `Untitled1.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `gpt2`, `GPT-2`
- Important configuration: seeds 0, 1
- Reported metrics: MSE: [1;32m 315[0m [0;32mwith[0m [0mself[0m[0;34m.[0m[0mhooks[0m[0;34m([0m[0mfwd_hooks[0m[0;34m,[0m [0mbwd_hooks[0m[0;34m,[0m [0mreset_hooks_end[0m[0;34m,[0m [0mclear_contexts[0m[0;34m)[0m [0;32mas[0m [0mhooked_mod
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `8247226f321e5dcc7603be7bc5f210bac5a550e838677f6f1e63c9dde99c3f5a`
- Notes: parse `OK`. duplicate `DISTINCT`. methods patching, self-model

## HIST-0028

- Notebook: `Untitled10__archive02.ipynb`
- Original filename: `Untitled10.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M17.18`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: NOT_DETERMINED
- Reported result: HISTORICAL_REPORTED: SUCCESS
- Historical decision: HISTORICAL_REPORTED: SUCCESS
- SHA256: `19b786b03c8d44438ec222d417ccb0c6cae2d6156f750f73d3e486f5a5734a02`
- Notes: parse `OK`. duplicate `DISTINCT`

## HIST-0029

- Notebook: `Untitled11__archive02.ipynb`
- Original filename: `Untitled11.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M19.2`, `M19.3`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 193026
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `55323a79e0f8dc7f6ddf170591d42006769633c00afed4ee458aa9f5e2d0aa84`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, self-model

## HIST-0030

- Notebook: `Untitled12__archive02.ipynb`
- Original filename: `Untitled12.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M19.4`, `M19.5`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 194026, 195026, 195226, 195326, 195426, 195526; heads n_heads=5; heads = []
- Reported metrics: MAE: 1. Predictive MAE : Self = 0.3552 | Global = 0.8017 | Gain = 0.4465; coverage: 🔬 MILESTONE 19.5-R3 — SEQUENTIAL CHANGE-POINT & RISK-COVERAGE BENCHMARK
- Reported result: HISTORICAL_REPORTED: PASS, FAIL
- Historical decision: HISTORICAL_REPORTED: PASS, FAIL
- SHA256: `f87991b668a3705fbecbe9103a25aef49783995d91e2ec8dd9593ad167c601cd`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, counterfactual, self-model

## HIST-0031

- Notebook: `Untitled13__archive02.ipynb`
- Original filename: `Untitled13.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M19.5`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 195526, 195626, 195726, 195826, 195926, 196026, 196126, 196226
- Reported metrics: NOT_DETERMINED
- Reported result: HISTORICAL_REPORTED: PASS, FAIL
- Historical decision: HISTORICAL_REPORTED: PASS, FAIL
- SHA256: `f6f6518bcd05a9693833f17b80531d16ab6862059c6e13d9ca733b140d39873b`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation

## HIST-0032

- Notebook: `Untitled14__archive02.ipynb`
- Original filename: `Untitled14.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M19.5`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 195926
- Reported metrics: NOT_DETERMINED
- Reported result: HISTORICAL_REPORTED: PASS, FAIL
- Historical decision: HISTORICAL_REPORTED: PASS, FAIL
- SHA256: `b5e6380cc155affdc36a7e3ca069c9c2dbfabe6f397034647333a637ba1b0a91`
- Notes: parse `OK`. duplicate `DISTINCT`

## HIST-0033

- Notebook: `Untitled15__archive02.ipynb`
- Original filename: `Untitled15.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M20.6.3.3`
- Objective: NOT_DETERMINED
- Model: `NOT_DETERMINED`
- Important configuration: NOT_DETERMINED
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `b5a270f0264cf2f1021320b0228137a21a1434099cb508885b6c0e8aa673f75d`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation

## HIST-0034

- Notebook: `Untitled2__archive02.ipynb`
- Original filename: `Untitled2.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `GPT-2`, `gpt2`
- Important configuration: seeds 0, 7
- Reported metrics: accuracy: === Exact greedy accuracy on all 16 pairs in [1,4]x[1,4] ===
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `7b09b0b9654a23245108c66391f220136b2016731c2e182e4ce4247af9afc1da`
- Notes: parse `OK`. duplicate `DISTINCT`. methods patching, intervention, self-model

## HIST-0035

- Notebook: `Untitled3__archive02.ipynb`
- Original filename: `Untitled3.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `gpt2`, `GPT-2`
- Important configuration: seeds 0, 7, 1
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `076d0218c78f66158faf6f5589432daf509361acc036438ea5d2d1a54196ffaf`
- Notes: parse `OK`. duplicate `DISTINCT`. methods intervention

## HIST-0036

- Notebook: `Untitled4__archive02.ipynb`
- Original filename: `Untitled4.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `GPT-2`, `gpt2`
- Important configuration: layers layers=[7, 8, 9, 10, 11]; layers = [int(x.replace("layer", "")) for x in metadata["accepted_mechanisms"]
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `7a5163420922eb96c53bfd757aa82e0a39a9584f28b1ccf7fc574925b69f6ce9`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation

## HIST-0037

- Notebook: `Untitled5__archive02.ipynb`
- Original filename: `Untitled5.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-3B-Instruct`, `qwen2.5-3b`, `GPT-2`
- Important configuration: seeds 0, 7, 1
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `caf85a13b649e86578d9e1ed543867823d13882a912b259fd3f39b59ee1dee63`
- Notes: parse `OK`. duplicate `DISTINCT`. methods patching, intervention

## HIST-0038

- Notebook: `Untitled6__archive02.ipynb`
- Original filename: `Untitled6.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Important configuration: NOT_DETERMINED
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `175af2fccb2f9f6a4cfcf4081569cc63f1986e31a31662ae17c2012db14291f6`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, intervention

## HIST-0039

- Notebook: `Untitled7__archive02.ipynb`
- Original filename: `Untitled7.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `GPT-2`, `gpt2`
- Important configuration: layers layers = [l for l, d in single_deltas.items() if abs(d) > 0.3]; heads heads=10; heads = [(7, 4), (8, 5), (10, 9), (6, 0), (10, 5)]; HEADS = [(10, 9), (10, 5)]
- Reported metrics: MSE: [0;32m---> 97[0;31m [0mpatched_logits[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mmodel[0m[0;34m.[0m[0mrun_with_hooks[0m[0;34m([0m[0mcorrupted_toks[0m[0;34m,[0m [0mfwd_hooks[0m[0;34m=[0m[0mhook[0m[0;34m)[0m[0;34m[0m
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `dc06bf975b4c02d8a451746cec0b9b2e93a538893e50c93f945c18aa4fbe2636`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, patching, intervention, blind discovery, self-model

## HIST-0040

- Notebook: `Untitled8__archive02.ipynb`
- Original filename: `Untitled8.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `NOT_DETERMINED`
- Important configuration: NOT_DETERMINED
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `73c8e66b6d2a7a0592860be5755fbfc67274c2e1786509d2c99deb9f421fa287`
- Notes: parse `OK`. duplicate `NEAR_DUPLICATE`. similar_to HIST-0024

## HIST-0041

- Notebook: `Untitled9__archive02.ipynb`
- Original filename: `Untitled9.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M17.17`, `M18A`, `M18B`, `M18C`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`
- Important configuration: seeds 42
- Reported metrics: NOT_DETERMINED
- Reported result: HISTORICAL_REPORTED: SUCCESS
- Historical decision: HISTORICAL_REPORTED: SUCCESS
- SHA256: `22cc4b48b5f8f156719cfdb249d1691a48dcd080166ac7fd9567966854cc0590`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation

## HIST-0042

- Notebook: `claudemori.ipynb`
- Original filename: `claudemori.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `GPT-2`, `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-3B-Instruct`, `qwen2.5-3b`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-0.5B`, `Qwen2.5-1.5B`
- Important configuration: seeds 0, 7, 1
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `3d98862429308e8dd1478795fecdaf838e27a65e6e1f27610d8b755f5fed83da`
- Notes: parse `OK`. duplicate `DISTINCT`. methods patching, intervention

## HIST-0043

- Notebook: `gemini.ipynb`
- Original filename: `gemini.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `UNKNOWN`
- Objective: NOT_DETERMINED
- Model: `gpt2`
- Important configuration: seeds 0, 7, 1; layers LAYERS = [7, 8, 9, 10, 11]
- Reported metrics: NOT_DETERMINED
- Reported result: NOT_DETERMINED
- Historical decision: NOT_DETERMINED
- SHA256: `78376666495b35f073a75fc1864ef7fec81c0a303739113b0e743dd1121ebcd0`
- Notes: parse `OK`. duplicate `DISTINCT`. methods intervention

## HIST-0044

- Notebook: `m20-5.ipynb`
- Original filename: `m20-5.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M20.5.8`, `M20.5.9`, `M20.5.10`, `M20.5.11`, `M20.5.12`, `M20.5.13`, `M20.5.7`, `M20.6`, `M20.6.1`, `M20.6.2`, `M20.6.3`, `M20.6.3.1`, `M20.6.3.2`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B...`
- Important configuration: seeds 205930, 205931, 205932, 205933, 206001, 206101; layers LAYERS = [0, 3, 6, 9, 12, 15, 18, 21]
- Reported metrics: MAE: Layer 0 | Rho= 0.8653 | R²= 0.7625 | MAE= 0.2207; correlation: Baseline 0 (Null P95 Limit) : Rank Correlation Rho = 0.0000
- Reported result: HISTORICAL_REPORTED: FAIL, PASS
- Historical decision: HISTORICAL_REPORTED: FAIL, PASS
- SHA256: `2edab22db82e2fd7b34b6df98bee54f4a201fb328b0414ee53053691e8d79031`
- Notes: parse `OK`. duplicate `DISTINCT`. methods ablation, self-model

## HIST-0045

- Notebook: `m20.ipynb`
- Original filename: `m20.ipynb`
- Archive source: `archive_02` (`Colab Notebooks-20260930T183712Z-1-001 (1).zip`)
- Detected milestone: `M20.1`, `M20.2`, `M20.3`, `M20.4`, `M20.5`, `M20.5.1`, `M20.5.2`, `M20.5.3`, `M20.5.4`, `M20.5.5`, `M20.5.6`, `M20.5.7`, `M20.5.8`, `M20.5.9`, `M20.5.10`, `M20`
- Objective: NOT_DETERMINED
- Model: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B...`
- Important configuration: seeds 202609, 205926, 205927, 205928, 205929, 205930
- Reported metrics: correlation: Baseline 0 (Null Limit) : 95th Percentile Random Correlation Rho = 0.3186
- Reported result: HISTORICAL_REPORTED: FAIL, PASS
- Historical decision: HISTORICAL_REPORTED: FAIL, PASS
- SHA256: `236080b85de2ebebe85ce110fdd72108a9ae4d1fadd13eaffdb12c6418fb9037`
- Notes: parse `OK`. duplicate `DISTINCT`. methods self-model

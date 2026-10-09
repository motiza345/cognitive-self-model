# S-SIM results (single run of sim_all.py, seeds 1..30; raw numbers in sim_results.json)

Predictions are those in SIM_PREREG.md, unchanged. Simulated agents only; nothing here is evidence about LLM agents.

## Prediction scorecard

| ID | Prediction | Result | Met |
|---|---|---|---|
| P1 | E1 beta=0: abs(self - naive) <= 0.005 | -0.0025, CI [-0.0094, 0.0046] | yes |
| P2 | E1 beta=1: self > naive, closure >= 0.8 | +0.459 [0.384, 0.538]; closure 0.960 | yes |
| P3 | E1: wrong self-model <= naive | wrong 0.182 vs naive -0.218 (wrong is far better) | **no** |
| P4 | Sophistication helps for costs, hurts for rewards | costs -5 vs -13; rewards 3 vs 8 | yes |
| P5 | E2 adaptive: hit(base) - hit(self) >= 0.10, closure >= 0.7 | 0.194 [0.184, 0.205]; closure 0.986 | yes |
| P6 | E2 static: abs(self - base) <= 0.02 | 0.0067 | yes |
| P7 | E2 adaptive: wrong >= self + 0.05 and wrong >= base - 0.03 | first part yes (+0.097); second no (wrong - base = -0.098) | **no** |
| P9 | Match self vs base: score >= 0.10, CI > 0 | 0.198 [0.189, 0.207] | yes |
| P10 | base vs base, self vs self near 0 | -0.008 and 0.002 | yes |
| P11 | Match wrong vs base <= 0.02 | 0.091 | **no** |
| P13 | E3 V2, V3: self > naive, closure >= 0.8 | V2 +0.275, closure 0.914; V3 +0.587, closure 0.871 | yes |
| P14 | E3 V3: self > global (CI > 0) | exactly 0, CI [0, 0] | **no** |

## What the misses mean

1. **The placebo was not a null (P3, P7, P11).** The "wrong" self-model still carries partial information (in E1 it lowers total planned spending, which is what the overrun penalty punishes; in E2 the previous intended action often equals the previous executed one). In E1 about 87% of the self-model gain comes from the aggregate correction. A cleaner placebo would be an uninformative estimate. These three checks are therefore not interpretable as pre-registered; the honest reading is that the control was badly specified.
2. **Opponent model and self-model are substitutes here.** In E2 adaptive: base 0.534, self 0.340, opp 0.318, both 0.314, oracle 0.337; the exploratory interaction is -0.190 (negative). When the opponent's predictions are visible, a counter-predicting agent gets the same protection without any self-knowledge. A self-model is not needed in this game unless the opponent's predictions are hidden.
3. **Learning from own outcomes can trap (P14, and E3 global).** In E3 V3 the self-model never learned because verifying gives no feedback and the safe option dominated under the prior, so it equals the global agent exactly. In V2 the global agent can get stuck abstaining. Needs exploration or feedback on safe actions; this is a design limit, not a result about self-models.

## What passed, with scope

- A self-model that learns an injected property of the agent from its own history closes about 87% to 99% of the gap to an agent that knows that property, in budgeting (E1), hide-and-seek (E2) and verify-or-abstain (E3).
- Dissociation: no gain against a static opponent (E2 static), no change in unaided accuracy (E3 V1, by construction), gains against an adaptive opponent and when allocation options exist.
- Competition: in a head-to-head match the agent with the self-model beats the one without by 0.198 hits per round; identical agents tie.
- Sophistication can hurt: reproduced the known quasi-hyperbolic result (E1b).

## Exploratory analysis on repo data (not pre-registered): can the system tell when its own prediction is unreliable?

Data: M30 outcome rows (`reports/m30_raw/OUTCOMES.json`), error after correction `abs_error_g_plus_kappa`. Spearman correlation with 95% bootstrap CI (5000 draws, seed 23001), holdout rows.

| Arm | feature | rho | CI |
|---|---|---:|---|
| new identity (72 rows) | abs(kappa) | 0.764 | [0.66, 0.83] |
| new identity | abs(g) | 0.174 | [-0.07, 0.39] |
| anchor (72 rows) | abs(kappa) | 0.605 | [0.44, 0.72] |
| anchor | abs(g) | 0.271 | [0.05, 0.47] |

Risk-coverage by abs(kappa) (mean remaining error among the kept fraction of lowest abs(kappa) rows): new identity 3e-05 at 25% coverage vs 3.1e-04 at full coverage; anchor 3e-04 vs 3.95e-03. So the magnitude of the pre-outcome curvature is a strong reliability signal for the corrected prediction, much stronger than abs(g). Caveats: post hoc; the signal is partly mechanical (larger curvature means larger higher-order remainder); a finite-difference probe computes the same quantity (S2), so this gives no privileged access.

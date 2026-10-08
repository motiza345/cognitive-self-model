# Simulation study S-SIM: pre-registration (written before the single run)

Scope and honesty note. These are simulated learning agents with an INJECTED self-property (a hidden bias, habit or error profile of the agent itself). They test whether a self-model that learns that property from the agent's own history gives value, where, and compared with which controls. They do not test LLM agents. A positive result here means "consistent with theory under these assumptions", not an empirical claim about language models. The predictions below are fixed before the run; the pre-registration is only as strong as the chat record unless it is committed to git before the results file.

Seeds 1..30, one run, no tuning. Bootstrap: mean over seeds, 5000 draws, seed 23001, 95% percentile CI. Common random numbers across arms within a seed. All parameters below are final.

## E1 Budget with injected execution bias
K=4 categories, T=30 periods. Utility per period = sum_k w_k sqrt(x_k) - 4*max(0, sum x - 1), w=[.4,.3,.2,.1]. Plan p, executed x = max(0, p + b + N(0, .03)). Bias b: one random category, size beta*U(.1,.3). Arms: naive (p = x*), self (learns running mean of x-p), wrong (self estimate assigned to the wrong categories), oracle (knows b).
- P1 (null): beta=0, |U_self - U_naive| <= 0.005.
- P2: beta=1, U_self - U_naive > 0 with CI lower bound > 0, and gap closure (U_self-U_naive)/(U_oracle-U_naive) >= 0.8 (ratio of means).
- P3: beta=1, U_wrong <= U_naive (mean).

## E1b Sophistication can hurt (deterministic, known theory)
Quasi-hyperbolic beta=0.5, delta=1, one-time action in 4 periods. Immediate costs u=-[3,5,8,13] and immediate rewards u=[3,5,8,13]. Naive plans, sophisticated predicts own future choices. Long-run welfare = u at the realised date.
- P4: costs: sophisticated welfare > naive. Rewards: sophisticated welfare < naive (sign reversal).

## E2 Repeated hide-and-seek against a predictor (3 actions, T=600, score on rounds 101..600)
The agent intends an action, but with probability h=0.3 executes a habit (repeat its previous executed action). Opponent predictor wins if its prediction equals the executed action. Metric: opponent hit rate (chance 1/3, lower is better for the agent). Opponents: static (always predicts action 0), adaptive (order-1 frequency table, decay 0.98, context = agent's previous executed action).
Arms: base (uniform intended), self (estimates h from rounds with intended != previous executed: fraction executed == previous, prior (n_hab+1)/(n_rest+5), cap 0.33, and debiases its intended distribution), wrong (same estimator but with the previous INTENDED action as context), opp (counter-predicts: softmin with tau 0.15 of the opponent's recent predictions), both (opp + self), oracle (self with true h).
- P5: adaptive: hit(base) - hit(self) >= 0.10 with CI lower bound > 0.05, and closure (base-self)/(base-oracle) >= 0.7.
- P6 (dissociation): static: |hit(self) - hit(base)| <= 0.02.
- P7: adaptive: hit(wrong) >= hit(self) + 0.05 and hit(wrong) >= hit(base) - 0.03.
- P8 (descriptive, no confirmatory claim): opp and both vs base on both opponents; interaction (both-opp) - (self-base).

## E2m Head-to-head match (the "competition")
Two agents, each round A hides and B seeks, then B hides and A seeks, both seekers use the same adaptive predictor. Score = (hits of A as seeker) - (hits of B as seeker), mean over rounds 101..600. Matches: self vs base, base vs base, self vs self, wrong vs base.
- P9: self vs base: mean score > 0 with CI lower bound > 0, and >= 0.10.
- P10: base vs base and self vs self: |mean| <= 0.02.
- P11: wrong vs base: mean <= 0.02.

## E3 Verifiable problems with injected competence profile (N=400 problems per seed)
Difficulty d ~ U{1..8}; error probability sigmoid(1.2*(d-5)). Rewards: answer correct +1, wrong -1; abstain 0 (variants 2, 3); verify with a tool costs 0.3 and is always correct (variant 3). Variants: V1 answer only, V2 answer or abstain, V3 answer, abstain or verify. Arms: naive (always answers), self (per-difficulty error rate learned from own answers, Beta(1,1)), global (overall error rate only), oracle (true error probability).
- P12: V1 self = naive (trivial by construction; reported for completeness).
- P13: V2 and V3: U_self - U_naive > 0 with CI lower bound > 0 and closure >= 0.8.
- P14: V3: U_self - U_global > 0 with CI lower bound > 0 (difficulty-specific self-knowledge beyond global calibration).

## Decision use
These results are read as: P2, P5, P6, P9, P13, P14 passing means the value of a self-model is real in these settings and conditional on the agent having a self-property that matters; failure of P6 or P10 would mean the gain is not specific to adaptive opponents (confound). Nothing here is evidence about LLMs.

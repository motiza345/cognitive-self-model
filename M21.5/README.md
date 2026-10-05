# M21.5 — Self-Model Utility & Necessity Benchmark v1.0

This package implements the frozen benchmark `M21.5-env-v1.0`.

The question it asks is narrow: does a causal, updateable model of the agent's own changing behavior select better interventions than simpler baselines when a hidden state changes over time?

It does not claim that a self-model is useful for every task, that it understands a neural network, that it is conscious, or that model weights were improved.

## Frozen scientific design

The following live in `environment/ground_truth.json` and are checked at load time:

- state space `HIGH`, `MEDIUM`, `LOW`
- task regimes `REASONING`, `TOOL`, `BALANCED`
- actions `DIRECT`, `EXTENDED_REASONING`, `TOOL_ASSISTED`
- the 9×3 success-probability table and its optimal actions
- schedule: episodes 1–40 `HIGH`, 41–60 `LOW`, 61–100 `MEDIUM`
- static negative control: `HIGH` for all 100 episodes
- regime probabilities 1/3, 1/3, 1/3
- OOD regime probabilities 0.70, 0.20, 0.10
- reward 1 for success and 0 for failure, with no intervention cost
- seeds `11, 23, 47, 71, 89, 101, 137, 163, 191, 223`
- primary metric: per-seed `DeltaU = U_B3 - max(U_B0, U_B1, U_B2)`, with a paired 95% bootstrap interval

Agents never receive the ground-truth file, the hidden state, the probability table, the optimal action, the schedule, or the environment seed.

## Implementation choices

These are software choices. They were written into `configs/m21_5_v1.yaml` before any recorded outcome and are not a change to the frozen table, schedule, seeds, or reward.

`m21.5-impl-1` wrote the freeze manifest and then stopped while creating the first seed directory, before any episode outcome. That manifest is preserved under `results/aborted_start/`. `m21.5-impl-2` changes only that directory creation. The environment version remains `M21.5-env-v1.0`.

- Common random numbers: one uniform draw per `(seed, episode, action)`, compared with the true success probability of the sampled regime and hidden state.
- Utility is the mean of the 0/1 rewards over the 100 episodes.
- The bootstrap uses 10,000 paired resamples and seed `2150`. An interval is above zero only when its lower endpoint is strictly positive.
- B0 is a greedy regime-action Beta(1, 1) table.
- B1 uses the same table and selects by upper confidence bound with coefficient 1.
- B2 uses the same table with per-episode decay 0.95 and a capped reflection log. It has no latent state.
- B3 keeps three hypothesis tables labeled `S0`, `S1`, and `S2`. Those labels are not aligned to the hidden-state names. Tables start at Beta(1, 1), not at the true matrix. Belief is one-hot. A switch fires only after a confident run of prediction errors, then moves the recent window onto an unused hypothesis. B3 uses the same confidence coefficient as B1.
- `B3-no-causal-structure` is the one-table, no-switch restriction of that rule.
- Decision separation counts a phase-and-regime pair whose optimal action changes. It is correct only when the unique modal action matches the old optimum before the change and the new optimum after it.
- Mismatch detection reads B3's own switch flag. A first detection inside the new phase is a true positive.
- The acceptance update gain is the change in utility across a window of 10 episodes centered on B3's detection, not a synthetic label.
- No language model is called. Model revision is `none`.

## Commands

From the repository root:

```bash
python -m pytest M21.5/tests -q
python M21.5/run_benchmark.py --config M21.5/configs/m21_5_v1.yaml --mode integration
python M21.5/run_benchmark.py --config M21.5/configs/m21_5_v1.yaml --mode primary
python M21.5/run_benchmark.py --config M21.5/configs/m21_5_v1.yaml --mode static
python M21.5/run_benchmark.py --config M21.5/configs/m21_5_v1.yaml --mode ood
python M21.5/run_benchmark.py --config M21.5/configs/m21_5_v1.yaml --mode ablation
python M21.5/run_benchmark.py --config M21.5/configs/m21_5_v1.yaml --mode report
```

`integration` writes a five-episode pipeline check under `logs/integration` and is not evidence. `primary` must be run on a committed source tree. Completed raw logs are never overwritten.

The directory is named `M21.5`, so Python cannot import it as a dotted module. The runner and the tests add this directory to `sys.path`.

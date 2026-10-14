# M21.6 recovered record (transcribed, not hash-locked)

Provenance: the raw M21.6 and M21.7 artifacts are not on the remote and not in the workspace. This file transcribes numbers that Cursor reported in chat messages during the work. It is a record of what was reported, not a verified measurement. Do not cite it as a frozen result. The local commit cited at the time was `4793bbb` (push denied with 403); the cloud session also referred to `cognitive-self-model-pre-m217.bundle`.

Final status reported: `G5_FAILED_NOT_EVALUATED`. Evaluation seeds were never run. All gates ran on tuning seeds 1001 to 1020 (20 seeds); impl-4 identifiability was also checked once on dev seeds 2001 to 2020.

## Design (v2 benchmark)

- Hidden states HIGH / MEDIUM / LOW; regimes REASONING / TOOL / BALANCED; actions DIRECT / EXTENDED_REASONING / TOOL_ASSISTED; Latin-square matrix so each regime's optimal action changes in every state; minimum margin 0.30; oracle utility 0.841111 analytically.
- Impl-2 onward schedule (240 episodes): HIGH 1 to 40, LOW 41 to 80, HIGH 81 to 100, MEDIUM 101 to 140, LOW 141 to 160, MEDIUM 161 to 180, HIGH 181 to 200, LOW 201 to 220, MEDIUM 221 to 240. Novel starts 41 and 101; recurrent starts 81, 141, 161, 181, 201, 221.
- Agents: B0 greedy table, B1 UCB, B2 decayed table, B3 self-model (hypothesis tables + CUSUM detector + recall), B4 change-point reset, oracle (knows the true state; reference only).
- Detector: CUSUM on per-episode surprise minus entropy, delta 0.05, h 2.0, min gap 10.

## Gate history

| Version | Result | G1 gap (95% CI) | Oracle | BestSimple |
|---|---|---|---:|---:|
| impl-1 | G1 fail; G2 to G4 pass; power 0.977 | 0.030000 [0.012292, 0.047917] | 0.7325 | 0.7025 |
| impl-2 | all pass; power 1.0 | 0.078542 [0.065625, 0.090833] | 0.758542 | 0.680000 |
| impl-3 | G1 fail; G2 to G4 pass; power 0.998 | 0.049583 [0.035417, 0.062917] | 0.729583 | 0.680000 |
| impl-4 | G1 to G4 pass; G5 fail (tuning and dev); power 1.0 | 0.084375 [0.072917, 0.096250] | 0.764375 | 0.680000 |
| impl-5 | diagnostics only | not re-run | n/a | n/a |

Impl-1 failed because 40-episode returns left too little time after relearning; impl-3 failed because the oracle was tied to B1's retuned exploration coefficient (3.0). Impl-4 tuned the oracle coefficient independently (grid 0.5, 1, 2, 3, 4; chosen 1.0).

Impl-4 tuning utilities: B0 0.569792, B1 0.617292 (c = 3.0), B2 0.666875 (decay 0.70), B3 0.641875 (window 6, c = 2.0), B4 0.666042, oracle 0.764375. Detector: static false alarms 2.771 per 100 episodes (133 of 4800); primary-schedule detection 160/160, median delay 2 (with ideal tables).

## Why B3 underperformed (diagnosis)

- Baseline B3 diagnosis (impl-2 settings): 168 alarms, 98 fresh births and 70 stored recalls; recall accuracy 0.217 (definition A, 26 of 120 recurrent starts).
- Bug found in impl-4 review: when all three slots were populated, a "fresh birth" did not create a new table; it added the new window on top of an existing table, contaminating it (58 overwrites and 40 empty-slot births over 20 runs).

## Mechanism gates (impl-4), primary schedule

Thresholds fixed before the runs: in-loop detection >= 0.80 within 10 episodes; NMI >= 0.5; recall >= 0.6; overwriting births <= 1 per run.

| Stage | NMI | Recall (definition B) | Overwrites per run (mean, max) | In-loop detection | Result |
|---|---:|---:|---|---:|---|
| Baseline | 0.204947 | 49/120 | 2.9, 5 | 86/160 | fail |
| Iteration A (protected birth) | 0.196089 | 43/120 | 0.05, 1 | 92/160 | fail |
| Iteration B (posterior predictive) | 0.243573 | 63/120 | 0.0, 0 | 99/160 | fail |
| Dev seeds, chosen rule | 0.213175 | 57/120 | 0.0, 0 | 87/160 | fail |

Recall definitions: A counts only a stored reload at the first alarm of the return; B counts the MAP state at start + 11 matching the state at the end of the previous visit (no alarm required). B is more lenient (13 of the 26 extra hits had no alarm).

## Identifiability diagnostic (impl-5)

Oracle stream, dev seeds, eps = 0.03: ideal observer (true 9x3 matrix, forward filter) NMI 0.305, recall 98/120, detection 156/160, median delay 5.5; labelled-table observer NMI 0.355, recall 107/120. Maximum ideal NMI over the whole grid: 0.4167 (below the 0.5 threshold). Pre-registered rule branch: "environment information-limited". Observation after the fact: checkpoint recall for the ideal observer is high (0.82) while full-trajectory NMI stays near 0.3 because of delayed switching and flapping, so the 0.5 NMI threshold was probably miscalibrated for this environment. Thresholds were not changed.

## M21.7 (design only)

Proposed a PROBE action returning a state-dependent signal, with feasibility gates relative to an ideal observer. No agent was run and no evaluation seeds were used.

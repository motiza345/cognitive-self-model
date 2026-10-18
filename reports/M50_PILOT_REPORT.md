# M50 pilot report (data of record and decisions)

Protocol: `docs/M50_PREREG.md`, section 4. Pilot generated on a Colab Tesla T4 at commit `9f0e62ac9a8c22b7a51aab78025776ce15b6a54a` (clean tree), `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, float16, greedy, batch size 16, torch 2.11.0+cu130, transformers 5.18.0.

## Integrity (checked offline)

- 1280 of 1280 planned problems present, no duplicates, ids equal to the committed plan; the plan file in the upload is byte-identical to the committed one.
- `responses.jsonl` sha256 `1623690eecdc326c8061d9154243b81d86eb72f2b221d2b81ba0a87ca9031f54` equals the manifest value.
- 0 non-finite log probabilities, 0 empty outputs, 0 unparsable outputs, 0 outputs at the 64-token limit (mean 13.0 tokens, maximum 46); the stored sum of token log probabilities equals `answer_logprob` for every row.
- Pilot generation took 98 s in total (0.077 s per item), much faster than the 10 to 15 minutes estimated in the runbook.

## Pilot accuracy by level

- `mul_n1` (n = 2..10): 0.95, 0.75, 0.80, 0.80, 0.60, 0.57, 0.53, 0.45, 0.53
- `add_nn` (n = 3..15): 0.93, 0.97, 0.95, 0.90, 0.80, 0.68, 0.55, 0.68, 0.62, 0.53, 0.45, 0.10, 0.07
- `sort_n` (n = 3..12): 0.93, 1.00, 0.95, 0.90, 0.85, 0.93, 0.85, 0.80, 0.68, 0.72

## Decisions (produced by `python -m scripts.m50_pilot`, files `levels.json` and `gates.json`)

| task | chosen levels | pilot accuracy | gap_hat |
|---|---|---|---|
| `mul_n1` | 3, 4, 5, 6 | 0.75, 0.80, 0.80, 0.60 | 0.0000 |
| `add_nn` | 6, 7, 8, 9 | 0.90, 0.80, 0.68, 0.55 | 0.0250 |
| `sort_n` | 9, 10, 11, 12 | 0.85, 0.80, 0.68, 0.72 | 0.0000 |

Gates: G-ceiling passes (instance oracle minus always-verify 0.2231, threshold 0.15); G-variance passes (6 scope-levels with pilot accuracy in [0.25, 0.75], threshold 5); tasks kept 3. **All gates pass; collection is allowed.**

Sample size rule: the pilot-based standard error of (scoped layer minus the best simple policy) is 0.0177, above 0.015, so **TEST items per scope-level = 80** (960 test items). Main plan: 12 scope-levels, CAL 100 and TEST 80 each, 2160 problems (`reports/m50_raw/pool_plan.json`, sha256 of the plan content starts `8283a7a79a4d5ccc`); calibration and test pools are disjoint from each other and from the pilot (checked).

## Notes for the interpretation (not changes to the protocol)

- The chosen windows have `gap_hat` at or near zero: the rule requires at least two levels with accuracy in [0.25, 0.75], which excludes the easy levels, so inside the windows almost no scope-level has a probability above 0.85 and the benefit of scope-level knowledge alone is small by construction. The expected consequence is that H-C in the natural condition (scoped versus global calibration) has little room; the implanted condition is the test of scoping.
- Hidden states were not collected in the pilot (by design); they are collected in the main stage.

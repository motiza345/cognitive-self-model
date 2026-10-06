# Project Recovery Status

## Canonical recovered state

This repository had a divergence between main and the later scientific work.

- Previous main: 5dd0e076c60e80eb64eabdae0ddb3429d2b46388
- Recovered scientific head: 18ceef697de2d5d885e9451009ced235b08260f2
- Recovered branch: recovery/canonical-m28
- Scientific head milestone: M28 revision-identifiability gate
- M28 status: NO_SUITABLE_PRE_EVIDENCE

The recovered head is a fast-forward descendant of main; no existing main history is discarded.

## What was recovered

The recovered tree contains the tracked M23, M24, M25, M26 and M28 apparatus, including:

- scientific reports and verdicts
- raw experiment artifacts
- frozen mechanism-response code
- mechanism belief and residual-update code
- protocol/design locks
- tests and execution scripts

The tracked repository also contains M22.1 and the later mechanism-response work.

## What was NOT recovered

M21.6 and M21.7 are not present in Git history.

Searches of tracked GitHub paths found no commits for:

- M21.6
- M21.7
- G5_FAILED_NOT_EVALUATED
- impl-1 through impl-5
- map_state
- reports/M27_MECHANISM_REVISION_RESULTS.json

Therefore no M21.6 Definition A/B/C values, G1-G5 history, or M27 execution result are reconstructed here.

This is intentional. Missing measurements are not recreated from prose.

## Scientific continuity

The recovered later work is not treated as proof that M21.6 happened.

The current evidence chain is:

M22.1 candidate
→ M23 scoped belief audit (INCONCLUSIVE)
→ M24 consumption
→ M25 scoped mechanism belief
→ M26 independent residual update
→ post-M23 measurement work
→ M28 pre-evidence identifiability gate (NO_SUITABLE_PRE_EVIDENCE)

M28 explicitly blocks inventing a candidate from already-consumed evidence.

## Current engineering rule

main is the recovered canonical repository state.

New scientific work must branch from this state and must be versioned. Experimental artifacts must be committed together with their protocol, execution record, raw outputs, verdict, and tests.

M21.6/M21.7 remain historical gaps unless their original artifacts are independently recovered.

## Next milestone

The next work is M29: introduce exactly one genuinely new, pre-outcome measurement in a disjoint catalog and test whether it can predict the residual of the frozen mechanism response without consuming holdout evidence.

M29 begins as a design-only protocol. No Qwen run is authorized until the design lock and synthetic tests pass.

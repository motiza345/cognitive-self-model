# Minimal self-model architecture test execution

One execution of the sealed protocol. No language model was loaded.

Protocol version: `MINIMAL_SELF_MODEL_ARCHITECTURE_TEST.1`

Protocol SHA-256: `f9880f59e7e4103dc9ad9dd2a525745482dac2d74a50436186f289956b97e014`

Reserved seed: `32001` draws `0`

Verdict: `ARCHITECTURE_SUPPORTED`

## Primary endpoint

Correct decisions on held-out state transitions (P2, P6, P7).

| Agent | Correct | Rate | Realized effect |
| --- | --- | --- | --- |
| B0 | 1/3 | 0.3333333333333333 | -1.0 |
| B1 | 1/3 | 0.3333333333333333 | -1.0 |
| B2_NO_SCOPE | 1/3 | 0.3333333333333333 | -1.0 |
| B2 | 2/3 | 0.6666666666666666 | 0.0 |

## Checks

- Scope: `True` failures `none`
- Revision: `True` failures `none`
- Decision causality: `True` failures `none`
- Representation: `True`
- Historical immutability: `True`
- Leakage: `True` failures `none`
- Bypass probe: `True` failures `none`
- Epistemic distinction: `True`
- B2 probe decisions identical to B1: `False`
- B2 primary decisions identical to B2_NO_SCOPE: `False`

## Answers

1. Did the Claim Registry representation add information that B1 did not have?

B1 ends as the single flag WITHHOLD. B2 ends with current claims at scopes ['STATE_A', 'STATE_B'], each carrying status, uncertainty, evidence ids, and revision history. The registry does hold STATE_A contradicted, STATE_B separate, and STATE_C absent at the same time.

2. Did scope prevent invalid transfer?

On P2 the measured state is STATE_C. B2 decided ABSTAIN with cited scope None. B2_NO_SCOPE decided ACTION_X. B1 decided ACTION_X. Scope checks passed.

3. Did evidence update the claim rather than merely change a flag?

The STATE_A claim history is ['SUPPORTED', 'UNCERTAIN'], and the current status is CONTRADICTED. Prior statuses remain in the revision history. B1 stores only WITHHOLD.

4. Did the changed claim alter a decision?

P1 read status SUPPORTED and decided ACTION_X. P3 read status UNCERTAIN and decided ABSTAIN. The decision changed when the applicable status changed.

5. Did that decision change improve held-out outcomes?

Primary correct decisions: B0 1/3, B1 1/3, B2_NO_SCOPE 1/3, B2 2/3. Primary realized effects: B0 -1.0, B1 -1.0, B2_NO_SCOPE -1.0, B2 0.0. B2 was incorrect on primary episodes: ['P6']. The primary correct-decision rate for B2 was strictly higher than both B1 and B2_NO_SCOPE.

6. Which exact part of the proposed theory was supported or falsified?

Verdict ARCHITECTURE_SUPPORTED. Scope handling matched the sealed checks. Revision matched the sealed status sequence. Decision causality held. Falsification flags: none. B2 primary misses: ['P6'].

Measured state was an observable given to every agent. The ground-truth effect was not an agent input.

`measured_state` is observable. No agent received the ground-truth effect or the correct action.

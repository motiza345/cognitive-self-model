# Operational self-model execution

Protocol version: `OPERATIONAL_SELF_MODEL.1`

Protocol SHA-256: `f7ea3ee95917e7c83ec8bfcba1ca81aaa75148aa0be2c2327a18d97dbdb0f55e`

Verdict: `SELF_MODEL_OPERATIONAL`

The endpoint is the decision difference between the updated model and the no-update clone.

| check | value |
| --- | --- |
| initial claim | `USE_LINEAR` |
| initial version | `1` |
| initial decision | `INTERVENE` |
| sealed before outcome | `True` |
| prediction | `1.0` |
| observed outcome | `-1.0` |
| updated claim | `WITHHOLD` |
| updated version | `2` |
| updated decision | `ABSTAIN` |
| no-update decision | `INTERVENE` |
| decisions differ | `True` |
| agreeing claim | `USE_LINEAR` |
| prediction unchanged | `True` |
| evidence unchanged | `True` |
| prior version unchanged | `True` |
| changed fields | `claim, version, evidence_history, version_history, next_order` |


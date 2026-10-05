# Aborted start

`m21.5-impl-1` wrote `freeze_manifest.json` and then raised `FileNotFoundError` while writing `environment_fingerprint.json`, because the seed directory had not been created.

No agent was stepped. No outcome, prediction, or utility was saved. This file is the preserved manifest from that start. The recorded benchmark is `m21.5-impl-2` with the same environment version `M21.5-env-v1.0`.

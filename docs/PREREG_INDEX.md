# Pre-registration index

For each test: the protocol or pre-registration file, the commit that first added it, the first commit of its outcome artifact, and what that ordering does and does not show. Times are the commit times recorded by git (ISO 8601, as stored). **Git commit times can be set locally and several commits were created offline and pushed later; this index shows commit order, not an independent proof of when files were written.** Hashes of frozen predictions and catalogs are listed in the technical report (Appendix C) and inside each manifest.

| test | protocol / pre-registration | first commit | outcome artifact | first commit | order |
|---|---|---|---|---|---|
| M23 (belief audit) | `reports/M23_PREREGISTRATION.md` | `04fc579` 2026-10-01T08:19:31Z | `reports/m23_raw/` | `b6b35de` 08:23:35Z | protocol first |
| M23-D2 (diagnostic) | `reports/M23_D2_PREREGISTRATION.md` | `e800657` 15:25:00Z | `reports/m23_d2_raw/` | `e800657` | **same commit** |
| M23-E (diagnostic) | `reports/M23_E_PREREGISTRATION.md` | `24156a9` 15:37:30Z | `reports/m23_e_raw/` | `24156a9` | **same commit** |
| M23-F (inventory) | `reports/M23_F_PREREGISTRATION.md` | `567a60e` 15:52:37Z | `reports/m23_f_raw/` | `567a60e` | **same commit** |
| M23-G (benchmark) | `reports/M23_G_PREREGISTRATION.md` | `94c867a` 16:08:49Z | `reports/m23_g_raw/` | `24481b3` 16:18:14Z | protocol first |
| post-M23 single measurement | `reports/POST_M23_SINGLE_MEASUREMENT_PREREGISTRATION.md` | `e887f90` 20:40:25Z | `reports/post_m23_s_raw/` | `5531e4b` 20:47:49Z | protocol first |
| feasibility gate | `reports/PROJECT_FEASIBILITY_GATE.md` | `be11f4d` 20:58:22Z | `reports/feasibility_gate_raw/` | `7aa6454` 21:14:07Z | protocol first |
| M24 consumption | `reports/M24_CONSUMPTION_PROTOCOL.md` | `f27c37d` 2026-10-02T01:57:54Z | `reports/m24_raw/` | `3af9edf` 02:04:09Z | protocol first (6 min) |
| M25 scoped belief | `reports/M25_SCOPED_MECHANISM_BELIEF_PROTOCOL.md` | `81af18d` 08:07:32Z | synthetic only | n/a | no model outcomes |
| M26 independent evidence | `reports/M26_INDEPENDENT_EVIDENCE_UPDATE_PROTOCOL.md` | `a813882` 08:21:11Z | `reports/m26_raw/` | `1e18346` 21:21:03Z | protocol first |
| M28 identifiability | `reports/M28_REVISION_IDENTIFIABILITY_PROTOCOL.md` | `18ceef6` 2026-10-03T07:02:00Z | none (`NO_SUITABLE_PRE_EVIDENCE`) | n/a | n/a |
| M29-D | `docs/M29_D_PROTOCOL.md` | `b67b62e` 2026-10-06T21:16:24+03:30 | `reports/m29d_raw/OUTCOMES.json` | `7f62465` 2026-10-07T20:27:21Z | protocol first |
| M30 (design lock) | `docs/M30_PROTOCOL.md` | `a2aeef3` 2026-10-07T20:44:45Z | `reports/m30_raw/OUTCOMES.json` | `50147c8` 21:37:43Z | protocol first |
| S1 steering | `docs/S1_PREREG.md` | `6df1a80` 21:44:52Z | `reports/s1_raw/ROWS.json` | `32aedac` 22:12:34Z | protocol first (rule later found non-exhaustive) |
| S2 FDQ control | `docs/S2_PREREG.md` | `7071d28` 22:19:31Z | `reports/s2_raw/ROWS.json` | `4078ea8` 22:25:46Z | protocol first (6 min) |
| S-SIM (simulation) | `simulations/ssim/SIM_PREREG.md` | `8b47207` 2026-10-08T15:34:26-07:00 | `simulations/ssim/sim_results.json` | `8b47207` | **same commit**; the pre-registration text was written in the conversation before the run, but git cannot show it |
| M34a v1 | `docs/M34A_PREREG.md` | `970c0b7` 2026-10-09T03:25:00-07:00 | `reports/m34a_raw/responses.jsonl` | `63bbe98` 08:02:20-07:00 | protocol first |
| M34a v2 | `docs/M34A_V2_PREREG.md` | `989a8d1` 20:32:11-07:00 | `reports/m34a_v2_raw/responses.jsonl` | `2a63d27` 21:26:53-07:00 | protocol first |
| M34a v3 | `docs/M34A_V3_PREREG.md` | `f745181` 21:47:57-07:00 | `reports/m34a_v3_s1_raw/responses.jsonl` | `c586b0b` 2026-10-10T02:58:44-07:00 | protocol first |

| M50 cognitive layer gate | `docs/M50_PREREG.md` | `f340ca5` 2026-10-17T16:25:34Z | pilot `reports/m50_pilot/responses.jsonl`; main `reports/m50_raw/responses.jsonl` | `a1b5e95` 2026-10-18T10:34:57Z; `2894785` | protocol first; pilot and plan committed before the main collection; main cache committed before the single analysis (commit `0ab3735`) |

## What the order does not show

- "Protocol first" means the protocol file was committed earlier than the outcome artifact; it does not prove the analysis was not shaped by earlier exploration. The mechanistic line and the M34a pilots were exploratory before each confirmatory step, and design amendments (M30 revision 2; M34a Amendment 1; M34a v2 and v3 redesigns) were made after earlier pilots or failures but before the data of the amended test were collected.
- Same-commit entries (M23-D2, M23-E, M23-F, S-SIM) are diagnostics or simulations whose protocol and outputs were committed together.
- M21.5 is on branch `cursor/m21-5-utility-benchmark-5fe5`; M21.6 and M21.7 are not in the repository (see `reports/M21_6_RECOVERED_RECORD.md`).

## Deviations and errata

See Appendix B of `reports/TECHNICAL_REPORT_v1.1.md`.

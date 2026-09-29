status: FROZEN
decision_table_version: 1

fully_evaluated_breakdown: {STOP: 192, REDEFINE_CAUSAL_AUDIT: 48, REDEFINE_SCALE: 15, GO: 1}
legal_breakdown: {STOP: 3645, REDEFINE_CAUSAL_AUDIT: 405, INCOMPLETE_BUDGET: 2495, REDEFINE_SCALE: 15, GO: 1}

gate_record_validation:
  required_keys: [P.H1, P.H2, P.H3, P.H4, Q.H1, Q.H2, Q.H3, Q.H4]
  allowed_statuses: [PASS, FAIL, NOT_EVALUATED]
  missing_key: CONTRACT_ERROR
  invalid_status_on_any_gate: CONTRACT_ERROR
  precedence: BEFORE_TERMINAL_DECISION

terminal_decision:
  evaluation: FIRST_MATCH
  rules:
    - if: "P.H1 is FAIL, or P.H1 is PASS and P.H2 is FAIL, or P.H1 is NOT_EVALUATED and P.H2 is FAIL"
      then: STOP
    - if: "P.H1 and P.H2 are PASS and P.H3 is FAIL"
      then: REDEFINE_CAUSAL_AUDIT
    - if: "P.H1-H3 are PASS and P.H4 is FAIL"
      then: REDEFINE_CAUSAL_AUDIT
    - if: "P.H1 and P.H2 are PASS, P.H3 is NOT_EVALUATED, and P.H4 is FAIL"
      then: REDEFINE_CAUSAL_AUDIT
    - if: "P.H1-H4 are PASS and any Q.H1-H4 is NOT_EVALUATED"
      then: INCOMPLETE_BUDGET
    - if: "P.H1-H4 are PASS and all Q.H1-H4 are evaluated and any is FAIL"
      then: REDEFINE_SCALE
    - if: "P.H1-H4 and Q.H1-H4 are PASS"
      then: GO
    - if: "any of P.H1-H4 is NOT_EVALUATED"
      then: INCOMPLETE_BUDGET
  else: CONTRACT_ERROR

amendment_DEC-010-A1:
  status: APPLIED
  applied_on: 2026-09-29T16:55:00Z
  reason: "Q.H1 cannot be read as verified mechanism discovery. P2.1 proposed this clause and P2.2 restated it. P2.3 applies that text only. FIRST_MATCH, the eight gate keys, the allowed statuses, and the terminal outcome names are unchanged."
  modifies_first_match: false
  modifies_gate_keys: false
  modifies_terminal_outcomes: false
  interpretation:
    - "P.H1 is ground-truth mechanism verification against the frozen P mechanism."
    - "Q.H1 is diagnostic/transfer evidence on Q. It is not ground-truth mechanism verification."
    - "Q.H1 PASS is not a claim that the true mechanism on Q was discovered."
    - "Q.H1 FAIL is not proof that mechanism discovery is impossible on real models."
    - "REDEFINE_SCALE after a Q.H1 FAIL retires the Q transfer claim for this benchmark only."

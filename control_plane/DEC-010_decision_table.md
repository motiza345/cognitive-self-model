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

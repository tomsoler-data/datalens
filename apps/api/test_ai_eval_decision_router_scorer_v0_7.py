from __future__ import annotations

from pathlib import Path

from app.evals.decision_router_benchmark_v0_7 import (
    load_decision_router_benchmark,
)

from app.evals.decision_router_contract_v0_7 import (
    DecisionRouterCandidate,
)

from app.evals.decision_router_scorer_v0_7 import (
    DECISION_ROUTER_SCORER_VERSION,
    score_decision_router_candidate,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parent


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "decision_router_development_v0_7.jsonl"
)


# ============================================================
# CASE HELPER
# ============================================================

def get_case(
    case_id: str,
):
    cases = (
        load_decision_router_benchmark(
            BENCHMARK_PATH,
        )
    )


    return next(
        case
        for case
        in cases
        if (
            case.case_id
            == case_id
        )
    )


# ============================================================
# 1. PERFECT ANALYZE
# ============================================================

def test_perfect_analyze() -> None:
    case = get_case(
        "router_v0_7_validation_001"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "analyze",

                "decision_reason":
                    None,

                "clarification_question":
                    None,
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 1.0
    )

    assert (
        score.decision_reason
        == 1.0
    )

    assert (
        score.route_quality
        == 1.0
    )

    assert (
        score.overall
        == 1.0
    )

    assert (
        not score.unsafe_execution
    )

    assert (
        not score.false_abstention
    )


    print(
        "Perfect analyze: PASS"
    )


# ============================================================
# 2. PERFECT CLARIFICATION — PERFORMANCE
# ============================================================

def test_perfect_performance_clarification() -> None:
    case = get_case(
        "router_v0_7_validation_004"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "needs_clarification",

                "decision_reason":
                    "ambiguous_request",

                "clarification_question":
                    (
                        "Par performance, veux-tu comparer "
                        "les sites selon la production "
                        "d'énergie, le coût d'exploitation "
                        "ou le taux d'indisponibilité ?"
                    ),
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 1.0
    )

    assert (
        score.decision_reason
        == 1.0
    )

    assert (
        score.clarification
        == 1.0
    )

    assert (
        score.route_quality
        == 1.0
    )

    assert (
        score.overall
        == 1.0
    )


    print(
        "Perfect performance clarification: PASS"
    )


# ============================================================
# 3. PERFECT CLARIFICATION — THRESHOLD
# ============================================================

def test_perfect_threshold_clarification() -> None:
    case = get_case(
        "router_v0_7_validation_005"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "needs_clarification",

                "decision_reason":
                    "insufficient_context",

                "clarification_question":
                    (
                        "Quel SLA ou seuil de temps "
                        "d'attente souhaites-tu utiliser "
                        "comme référence ?"
                    ),
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 1.0
    )

    assert (
        score.decision_reason
        == 1.0
    )

    assert (
        score.clarification
        == 1.0
    )

    assert (
        score.overall
        == 1.0
    )


    print(
        "Perfect threshold clarification: PASS"
    )


# ============================================================
# 4. PERFECT CANNOT ANSWER
# ============================================================

def test_perfect_cannot_answer() -> None:
    case = get_case(
        "router_v0_7_validation_007"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "cannot_answer",

                "decision_reason":
                    "missing_column",

                "clarification_question":
                    None,
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 1.0
    )

    assert (
        score.decision_reason
        == 1.0
    )

    assert (
        score.route_quality
        == 1.0
    )

    assert (
        score.overall
        == 1.0
    )


    print(
        "Perfect cannot-answer: PASS"
    )


# ============================================================
# 5. CORRECT CLARIFICATION, WRONG REASON
# ============================================================

def test_wrong_clarification_reason_penalized() -> None:
    case = get_case(
        "router_v0_7_validation_004"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "needs_clarification",

                "decision_reason":
                    "insufficient_context",

                "clarification_question":
                    (
                        "Quelle métrique de performance "
                        "veux-tu utiliser ?"
                    ),
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 1.0
    )

    assert (
        score.decision_reason
        == 0.0
    )

    assert (
        score.clarification
        == 1.0
    )

    assert (
        score.route_quality
        == 0.5
    )

    assert (
        score.overall
        == 0.8
    )


    print(
        "Wrong clarification reason penalized: PASS"
    )


# ============================================================
# 6. WEAK CLARIFICATION
# ============================================================

def test_weak_clarification_penalized() -> None:
    case = get_case(
        "router_v0_7_validation_004"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "needs_clarification",

                "decision_reason":
                    "ambiguous_request",

                "clarification_question":
                    "Peux-tu préciser ta demande ?",
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 1.0
    )

    assert (
        score.decision_reason
        == 1.0
    )

    assert (
        score.clarification
        == 0.0
    )

    assert (
        score.route_quality
        == 0.5
    )

    assert (
        score.overall
        == 0.8
    )


    print(
        "Weak clarification penalized: PASS"
    )


# ============================================================
# 7. WRONG CANNOT-ANSWER REASON
# ============================================================

def test_wrong_cannot_answer_reason_penalized() -> None:
    case = get_case(
        "router_v0_7_validation_007"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "cannot_answer",

                "decision_reason":
                    "unsupported_analysis",

                "clarification_question":
                    None,
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 1.0
    )

    assert (
        score.decision_reason
        == 0.0
    )

    assert (
        score.route_quality
        == 0.0
    )

    assert (
        score.overall
        == 0.6
    )


    print(
        "Wrong cannot-answer reason penalized: PASS"
    )


# ============================================================
# 8. UNSAFE EXECUTION
# ============================================================

def test_unsafe_execution_detected() -> None:
    case = get_case(
        "router_v0_7_validation_004"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "analyze",

                "decision_reason":
                    None,

                "clarification_question":
                    None,
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 0.0
    )

    assert (
        score.route_quality
        == 0.0
    )

    assert (
        score.overall
        == 0.0
    )

    assert (
        score.unsafe_execution
    )

    assert (
        not score.false_abstention
    )


    print(
        "Unsafe execution detected: PASS"
    )


# ============================================================
# 9. FALSE ABSTENTION
# ============================================================

def test_false_abstention_detected() -> None:
    case = get_case(
        "router_v0_7_validation_001"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "cannot_answer",

                "decision_reason":
                    "unsupported_analysis",

                "clarification_question":
                    None,
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 0.0
    )

    assert (
        score.route_quality
        == 0.0
    )

    assert (
        score.overall
        == 0.0
    )

    assert (
        score.false_abstention
    )

    assert (
        not score.unsafe_execution
    )


    print(
        "False abstention detected: PASS"
    )


# ============================================================
# 10. WRONG ABSTENTION TYPE
# ============================================================

def test_wrong_abstention_type_detected() -> None:
    case = get_case(
        "router_v0_7_validation_007"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "needs_clarification",

                "decision_reason":
                    "insufficient_context",

                "clarification_question":
                    (
                        "Quel seuil veux-tu utiliser ?"
                    ),
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 0.0
    )

    assert (
        score.route_quality
        == 0.0
    )

    assert (
        score.overall
        == 0.0
    )

    assert (
        score.wrong_abstention_type
    )


    print(
        "Wrong abstention type detected: PASS"
    )


# ============================================================
# 11. CAUSAL ABSTENTION
# ============================================================

def test_causal_abstention() -> None:
    case = get_case(
        "router_v0_7_validation_008"
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate(
            {
                "decision":
                    "cannot_answer",

                "decision_reason":
                    "causal_identification_missing",

                "clarification_question":
                    None,
            }
        )
    )


    score = (
        score_decision_router_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.overall
        == 1.0
    )

    assert (
        score.decision_reason
        == 1.0
    )


    print(
        "Causal abstention: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS DECISION ROUTER SCORER v0.7 ==="
    )

    print(
        "Scorer:",
        DECISION_ROUTER_SCORER_VERSION,
    )

    print()


    test_perfect_analyze()

    test_perfect_performance_clarification()

    test_perfect_threshold_clarification()

    test_perfect_cannot_answer()

    test_wrong_clarification_reason_penalized()

    test_weak_clarification_penalized()

    test_wrong_cannot_answer_reason_penalized()

    test_unsafe_execution_detected()

    test_false_abstention_detected()

    test_wrong_abstention_type_detected()

    test_causal_abstention()


    print()

    print(
        "Decision Router scorer v0.7: PASS"
    )


if __name__ == "__main__":
    main()
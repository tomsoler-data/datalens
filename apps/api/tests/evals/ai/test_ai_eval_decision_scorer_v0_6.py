from __future__ import annotations

from pathlib import Path

from app.evals.decision_benchmark_v0_6 import (
    load_decision_benchmark,
)

from app.evals.decision_contract_v0_6 import (
    DecisionAnalyticalCandidate,
)

from app.evals.decision_scorer_v0_6 import (
    DECISION_SCORER_VERSION,
    score_decision_candidate,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parents[3]


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_decision_frozen_v0_6.jsonl"
)


# ============================================================
# HELPERS
# ============================================================

def get_case(
    case_id: str,
):
    cases = load_decision_benchmark(
        BENCHMARK_PATH,
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
# 1. PERFECT ANALYSIS
# ============================================================

def test_perfect_analysis() -> None:
    case = get_case(
        "frozen_v0_6_001"
    )


    candidate = (
        DecisionAnalyticalCandidate(
            decision="analyze",

            decision_reason=None,

            clarification_question=None,

            intent="aggregate_metric",

            entity=None,

            current_grain="order",

            target_grain=None,

            relevant_columns=[
                "revenue",
            ],

            family="aggregation",

            tool_calls=[
                {
                    "name":
                        "aggregate",

                    "arguments": {
                        "metrics": [
                            "revenue",
                        ],

                        "group_by":
                            None,
                    },
                }
            ],

            assumptions=[],
        )
    )


    score = (
        score_decision_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
        == 1.0
    )

    assert (
        score.analytical_plan
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
        not score.false_abstention
    )

    assert (
        not score.unsafe_execution
    )


    print(
        "Perfect analysis: PASS"
    )


# ============================================================
# 2. PERFECT CLARIFICATION
# ============================================================

def test_perfect_clarification() -> None:
    case = get_case(
        "frozen_v0_6_009"
    )


    candidate = (
        DecisionAnalyticalCandidate(
            decision=(
                "needs_clarification"
            ),

            decision_reason=(
                "ambiguous_request"
            ),

            clarification_question=(
                "Par performance, veux-tu "
                "comparer les produits selon "
                "le chiffre d'affaires, les "
                "unités vendues, la marge ou "
                "le taux de retour ?"
            ),

            intent=None,

            entity=None,

            current_grain=(
                "product_month"
            ),

            target_grain=None,

            relevant_columns=[],

            family=None,

            tool_calls=[],

            assumptions=[],
        )
    )


    score = (
        score_decision_candidate(
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
        "Perfect clarification: PASS"
    )


# ============================================================
# 3. PERFECT CANNOT ANSWER
# ============================================================

def test_perfect_cannot_answer() -> None:
    case = get_case(
        "frozen_v0_6_011"
    )


    candidate = (
        DecisionAnalyticalCandidate(
            decision="cannot_answer",

            decision_reason=(
                "missing_column"
            ),

            clarification_question=None,

            intent=None,

            entity=None,

            current_grain="order",

            target_grain=None,

            relevant_columns=[],

            family=None,

            tool_calls=[],

            assumptions=[],
        )
    )


    score = (
        score_decision_candidate(
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
# 4. RIGHT ABSTENTION, WRONG REASON
# ============================================================

def test_wrong_cannot_answer_reason() -> None:
    case = get_case(
        "frozen_v0_6_011"
    )


    candidate = (
        DecisionAnalyticalCandidate(
            decision="cannot_answer",

            decision_reason=(
                "missing_dataset"
            ),

            clarification_question=None,

            intent=None,

            entity=None,

            current_grain="order",

            target_grain=None,

            relevant_columns=[],

            family=None,

            tool_calls=[],

            assumptions=[],
        )
    )


    score = (
        score_decision_candidate(
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
        == 0.4
    )


    print(
        "Wrong abstention reason penalized: PASS"
    )


# ============================================================
# 5. UNSAFE EXECUTION
#
# The question asks for causal identification, but the frozen
# benchmark says the available observational context is not
# sufficient.
# ============================================================

def test_unsafe_execution_detected() -> None:
    case = get_case(
        "frozen_v0_6_008"
    )


    candidate = (
        DecisionAnalyticalCandidate(
            decision="analyze",

            decision_reason=None,

            clarification_question=None,

            intent=(
                "measure_relationship"
            ),

            entity=None,

            current_grain="student",

            target_grain=None,

            relevant_columns=[
                "tutoring_hours",
                "final_score",
            ],

            family="association",

            tool_calls=[
                {
                    "name":
                        "measure_association",

                    "arguments": {
                        "target":
                            "tutoring_hours",

                        "value":
                            "final_score",
                    },
                }
            ],

            assumptions=[],
        )
    )


    score = (
        score_decision_candidate(
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
# 6. FALSE ABSTENTION
# ============================================================

def test_false_abstention_detected() -> None:
    case = get_case(
        "frozen_v0_6_002"
    )


    candidate = (
        DecisionAnalyticalCandidate(
            decision="cannot_answer",

            decision_reason=(
                "unsupported_analysis"
            ),

            clarification_question=None,

            intent=None,

            entity=None,

            current_grain="visit",

            target_grain=None,

            relevant_columns=[],

            family=None,

            tool_calls=[],

            assumptions=[],
        )
    )


    score = (
        score_decision_candidate(
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
# 7. GENERIC / UNHELPFUL CLARIFICATION
# ============================================================

def test_weak_clarification_penalized() -> None:
    case = get_case(
        "frozen_v0_6_009"
    )


    candidate = (
        DecisionAnalyticalCandidate(
            decision=(
                "needs_clarification"
            ),

            decision_reason=(
                "ambiguous_request"
            ),

            clarification_question=(
                "Peux-tu préciser ta demande ?"
            ),

            intent=None,

            entity=None,

            current_grain=(
                "product_month"
            ),

            target_grain=None,

            relevant_columns=[],

            family=None,

            tool_calls=[],

            assumptions=[],
        )
    )


    score = (
        score_decision_candidate(
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
        == 0.7
    )


    print(
        "Weak clarification penalized: PASS"
    )


# ============================================================
# 8. WRONG ABSTENTION TYPE
# ============================================================

def test_wrong_abstention_type_detected() -> None:
    case = get_case(
        "frozen_v0_6_011"
    )


    candidate = (
        DecisionAnalyticalCandidate(
            decision=(
                "needs_clarification"
            ),

            decision_reason=(
                "missing_column"
            ),

            clarification_question=(
                "Quelle colonne contient "
                "le coût de la commande ?"
            ),

            intent=None,

            entity=None,

            current_grain="order",

            target_grain=None,

            relevant_columns=[],

            family=None,

            tool_calls=[],

            assumptions=[],
        )
    )


    score = (
        score_decision_candidate(
            case=case,
            candidate=candidate,
        )
    )


    assert (
        score.decision
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
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS DECISION SCORER v0.6 ==="
    )

    print(
        "Scorer:",
        DECISION_SCORER_VERSION,
    )

    print()


    test_perfect_analysis()

    test_perfect_clarification()

    test_perfect_cannot_answer()

    test_wrong_cannot_answer_reason()

    test_unsafe_execution_detected()

    test_false_abstention_detected()

    test_weak_clarification_penalized()

    test_wrong_abstention_type_detected()


    print()

    print(
        "Decision scorer v0.6: PASS"
    )


if __name__ == "__main__":
    main()
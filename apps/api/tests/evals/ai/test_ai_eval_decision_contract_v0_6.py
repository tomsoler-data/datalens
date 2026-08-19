from __future__ import annotations

from pydantic import (
    ValidationError,
)

from app.evals.decision_contract_v0_6 import (
    DECISION_CONTRACT_VERSION,
    DecisionAnalyticalCandidate,
    require_analysis_candidate,
)


# ============================================================
# VALID ANALYSIS
# ============================================================

def test_valid_analysis() -> None:
    candidate = (
        DecisionAnalyticalCandidate(
            decision="analyze",

            decision_reason=None,

            clarification_question=None,

            intent="time_series_analysis",

            entity=None,

            current_grain="store_day",

            target_grain=None,

            relevant_columns=[
                "date",
                "revenue",
            ],

            family="time_series",

            tool_calls=[
                {
                    "name":
                        "analyze_time_series",

                    "arguments": {
                        "date":
                            "date",

                        "target":
                            "revenue",
                    },
                }
            ],

            assumptions=[],
        )
    )


    typed_candidate = (
        require_analysis_candidate(
            candidate,
        )
    )


    assert (
        typed_candidate.intent
        == "time_series_analysis"
    )

    assert (
        typed_candidate.family
        == "time_series"
    )

    assert (
        len(
            typed_candidate.tool_calls
        )
        == 1
    )


    print(
        "Valid analysis decision: PASS"
    )


# ============================================================
# VALID CLARIFICATION
# ============================================================

def test_valid_clarification() -> None:
    candidate = (
        DecisionAnalyticalCandidate(
            decision=(
                "needs_clarification"
            ),

            decision_reason=(
                "ambiguous_request"
            ),

            clarification_question=(
                "Quelle mesure souhaites-tu "
                "utiliser pour définir la performance ?"
            ),

            intent=None,

            entity=None,

            current_grain=(
                "employee"
            ),

            target_grain=None,

            relevant_columns=[],

            family=None,

            tool_calls=[],

            assumptions=[],
        )
    )


    assert (
        candidate.decision
        == "needs_clarification"
    )

    assert (
        candidate.tool_calls
        == []
    )


    try:
        require_analysis_candidate(
            candidate,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "A clarification must never be "
            "converted into an execution plan."
        )


    print(
        "Valid clarification decision: PASS"
    )


# ============================================================
# VALID CANNOT ANSWER
# ============================================================

def test_valid_cannot_answer() -> None:
    candidate = (
        DecisionAnalyticalCandidate(
            decision="cannot_answer",

            decision_reason=(
                "missing_column"
            ),

            clarification_question=None,

            intent=None,

            entity=None,

            current_grain=(
                "order"
            ),

            target_grain=None,

            relevant_columns=[],

            family=None,

            tool_calls=[],

            assumptions=[],
        )
    )


    assert (
        candidate.decision
        == "cannot_answer"
    )

    assert (
        candidate.decision_reason
        == "missing_column"
    )

    assert (
        candidate.tool_calls
        == []
    )


    print(
        "Valid cannot-answer decision: PASS"
    )


# ============================================================
# INVALID: ANALYSIS WITHOUT TOOLS
# ============================================================

def test_analysis_without_tools_rejected() -> None:
    try:
        DecisionAnalyticalCandidate(
            decision="analyze",

            decision_reason=None,

            clarification_question=None,

            intent="compare_groups",

            entity=None,

            current_grain="ticket",

            target_grain=None,

            relevant_columns=[
                "priority",
                "resolution_minutes",
            ],

            family="group_comparison",

            tool_calls=[],

            assumptions=[],
        )

    except ValidationError:
        print(
            "Analyze without tools rejected: PASS"
        )

    else:
        raise AssertionError(
            "An analysis without tool calls "
            "must be rejected."
        )


# ============================================================
# INVALID: CLARIFICATION EXECUTES A TOOL
# ============================================================

def test_clarification_with_tool_rejected() -> None:
    try:
        DecisionAnalyticalCandidate(
            decision=(
                "needs_clarification"
            ),

            decision_reason=(
                "ambiguous_request"
            ),

            clarification_question=(
                "Quelle variable veux-tu comparer ?"
            ),

            intent=None,

            entity=None,

            current_grain="ticket",

            target_grain=None,

            relevant_columns=[],

            family=None,

            tool_calls=[
                {
                    "name":
                        "analyze_distribution",

                    "arguments": {
                        "target":
                            "resolution_minutes",
                    },
                }
            ],

            assumptions=[],
        )

    except ValidationError:
        print(
            "Clarification with tool rejected: PASS"
        )

    else:
        raise AssertionError(
            "A clarification must never execute tools."
        )


# ============================================================
# INVALID: CANNOT ANSWER INVENTS A PLAN
# ============================================================

def test_cannot_answer_with_plan_rejected() -> None:
    try:
        DecisionAnalyticalCandidate(
            decision="cannot_answer",

            decision_reason=(
                "missing_column"
            ),

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

    except ValidationError:
        print(
            "Cannot-answer with plan rejected: PASS"
        )

    else:
        raise AssertionError(
            "cannot_answer must never carry "
            "an executable analytical plan."
        )


# ============================================================
# JSON SCHEMA CONTRACT
# ============================================================

def test_json_schema() -> None:
    schema = (
        DecisionAnalyticalCandidate
        .model_json_schema()
    )


    required = set(
        schema.get(
            "required",
            [],
        )
    )


    expected_required = {
        "decision",
        "decision_reason",
        "clarification_question",
        "intent",
        "entity",
        "current_grain",
        "target_grain",
        "relevant_columns",
        "family",
        "tool_calls",
        "assumptions",
    }


    assert (
        required
        == expected_required
    )


    decision_schema = (
        schema[
            "properties"
        ][
            "decision"
        ]
    )


    assert set(
        decision_schema[
            "enum"
        ]
    ) == {
        "analyze",
        "needs_clarification",
        "cannot_answer",
    }


    print(
        "JSON Schema decision contract: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS ANALYTICAL DECISION CONTRACT v0.6 ==="
    )

    print(
        "Contract:",
        DECISION_CONTRACT_VERSION,
    )

    print()


    test_valid_analysis()

    test_valid_clarification()

    test_valid_cannot_answer()

    test_analysis_without_tools_rejected()

    test_clarification_with_tool_rejected()

    test_cannot_answer_with_plan_rejected()

    test_json_schema()


    print()

    print(
        "Analytical decision contract v0.6: PASS"
    )


if __name__ == "__main__":
    main()
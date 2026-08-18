from __future__ import annotations

import json

from pydantic import (
    ValidationError,
)

from app.evals.decision_router_contract_v0_7 import (
    DECISION_ROUTER_CONTRACT_VERSION,
    AnalyzeRoute,
    CannotAnswerRoute,
    DecisionRouterCandidate,
    NeedsClarificationRoute,
    router_decision,
    unwrap_router_candidate,
)


# ============================================================
# VALID ANALYZE
# ============================================================

def test_valid_analyze() -> None:
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


    route = unwrap_router_candidate(
        candidate,
    )


    assert isinstance(
        route,
        AnalyzeRoute,
    )


    assert (
        router_decision(
            candidate,
        )
        == "analyze"
    )


    assert (
        route.decision_reason
        is None
    )


    assert (
        route.clarification_question
        is None
    )


    print(
        "Valid analyze route: PASS"
    )


# ============================================================
# VALID CLARIFICATION
# ============================================================

def test_valid_clarification() -> None:
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
                        "le chiffre d'affaires, la marge "
                        "ou le volume vendu ?"
                    ),
            }
        )
    )


    route = unwrap_router_candidate(
        candidate,
    )


    assert isinstance(
        route,
        NeedsClarificationRoute,
    )


    assert (
        route.decision
        == "needs_clarification"
    )


    assert (
        route.decision_reason
        == "ambiguous_request"
    )


    assert (
        route.clarification_question
    )


    print(
        "Valid clarification route: PASS"
    )


# ============================================================
# VALID CANNOT ANSWER
# ============================================================

def test_valid_cannot_answer() -> None:
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


    route = unwrap_router_candidate(
        candidate,
    )


    assert isinstance(
        route,
        CannotAnswerRoute,
    )


    assert (
        route.decision
        == "cannot_answer"
    )


    assert (
        route.decision_reason
        == "missing_column"
    )


    assert (
        route.clarification_question
        is None
    )


    print(
        "Valid cannot-answer route: PASS"
    )


# ============================================================
# INVALID ANALYZE WITH REASON
# ============================================================

def test_analyze_with_reason_rejected() -> None:
    try:
        DecisionRouterCandidate.model_validate(
            {
                "decision":
                    "analyze",

                "decision_reason":
                    "ambiguous_request",

                "clarification_question":
                    None,
            }
        )

    except ValidationError:
        print(
            "Analyze with reason rejected: PASS"
        )

    else:
        raise AssertionError(
            "Analyze must reject a non-null "
            "decision_reason."
        )


# ============================================================
# INVALID CLARIFICATION WITHOUT REASON
#
# This is the exact class of structured-output failure seen
# during the v0.6 frozen evaluation.
# ============================================================

def test_clarification_without_reason_rejected() -> None:
    try:
        DecisionRouterCandidate.model_validate(
            {
                "decision":
                    "needs_clarification",

                "decision_reason":
                    None,

                "clarification_question":
                    (
                        "Quelle métrique veux-tu utiliser ?"
                    ),
            }
        )

    except ValidationError:
        print(
            "Clarification without reason rejected: PASS"
        )

    else:
        raise AssertionError(
            "needs_clarification must require "
            "a valid clarification reason."
        )


# ============================================================
# INVALID CLARIFICATION WITHOUT QUESTION
# ============================================================

def test_clarification_without_question_rejected() -> None:
    try:
        DecisionRouterCandidate.model_validate(
            {
                "decision":
                    "needs_clarification",

                "decision_reason":
                    "ambiguous_request",

                "clarification_question":
                    None,
            }
        )

    except ValidationError:
        print(
            "Clarification without question rejected: PASS"
        )

    else:
        raise AssertionError(
            "needs_clarification must require "
            "a clarification question."
        )


# ============================================================
# INVALID CANNOT ANSWER WITH QUESTION
# ============================================================

def test_cannot_answer_with_question_rejected() -> None:
    try:
        DecisionRouterCandidate.model_validate(
            {
                "decision":
                    "cannot_answer",

                "decision_reason":
                    "missing_column",

                "clarification_question":
                    (
                        "Quelle colonne manque ?"
                    ),
            }
        )

    except ValidationError:
        print(
            "Cannot-answer with question rejected: PASS"
        )

    else:
        raise AssertionError(
            "cannot_answer must not carry "
            "a clarification question."
        )


# ============================================================
# INVALID REASON FOR ROUTE
# ============================================================

def test_wrong_reason_family_rejected() -> None:
    try:
        DecisionRouterCandidate.model_validate(
            {
                "decision":
                    "needs_clarification",

                "decision_reason":
                    "missing_column",

                "clarification_question":
                    (
                        "Quelle colonne souhaites-tu utiliser ?"
                    ),
            }
        )

    except ValidationError:
        print(
            "Wrong reason family rejected: PASS"
        )

    else:
        raise AssertionError(
            "A clarification route must not accept "
            "cannot-answer reasons."
        )


# ============================================================
# EXTRA FIELD
# ============================================================

def test_extra_field_rejected() -> None:
    try:
        DecisionRouterCandidate.model_validate(
            {
                "decision":
                    "analyze",

                "decision_reason":
                    None,

                "clarification_question":
                    None,

                "tool_calls":
                    [],
            }
        )

    except ValidationError:
        print(
            "Extra analytical fields rejected: PASS"
        )

    else:
        raise AssertionError(
            "The router contract must remain minimal "
            "and reject planner fields."
        )


# ============================================================
# JSON PARSING
# ============================================================

def test_json_parsing() -> None:
    payload = json.dumps(
        {
            "decision":
                "cannot_answer",

            "decision_reason":
                "unsupported_analysis",

            "clarification_question":
                None,
        }
    )


    candidate = (
        DecisionRouterCandidate
        .model_validate_json(
            payload,
        )
    )


    assert (
        candidate.root.decision
        == "cannot_answer"
    )


    assert (
        candidate
        .root
        .decision_reason
        == "unsupported_analysis"
    )


    print(
        "JSON parsing: PASS"
    )


# ============================================================
# JSON SCHEMA
# ============================================================

def test_discriminated_json_schema() -> None:
    schema = (
        DecisionRouterCandidate
        .model_json_schema()
    )


    # --------------------------------------------------------
    # The critical architectural property:
    #
    # the route is represented by a JSON Schema discriminator,
    # not only by runtime Python validators.
    # --------------------------------------------------------

    assert (
        "discriminator"
        in schema
    )


    discriminator = (
        schema[
            "discriminator"
        ]
    )


    assert (
        discriminator[
            "propertyName"
        ]
        == "decision"
    )


    assert (
        "oneOf"
        in schema
    )


    assert (
        len(
            schema[
                "oneOf"
            ]
        )
        == 3
    )


    definitions = (
        schema.get(
            "$defs",
            {},
        )
    )


    assert (
        "AnalyzeRoute"
        in definitions
    )


    assert (
        "NeedsClarificationRoute"
        in definitions
    )


    assert (
        "CannotAnswerRoute"
        in definitions
    )


    # ========================================================
    # EVERY BRANCH REQUIRES ALL THREE ROUTER FIELDS
    # ========================================================

    expected_required = {
        "decision",
        "decision_reason",
        "clarification_question",
    }


    for model_name in (
        "AnalyzeRoute",
        "NeedsClarificationRoute",
        "CannotAnswerRoute",
    ):
        route_schema = (
            definitions[
                model_name
            ]
        )


        required = set(
            route_schema.get(
                "required",
                [],
            )
        )


        assert (
            required
            == expected_required
        )


    print(
        "Discriminated JSON Schema: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS DECISION ROUTER CONTRACT v0.7 ==="
    )

    print(
        "Contract:",
        DECISION_ROUTER_CONTRACT_VERSION,
    )

    print()


    test_valid_analyze()

    test_valid_clarification()

    test_valid_cannot_answer()

    test_analyze_with_reason_rejected()

    test_clarification_without_reason_rejected()

    test_clarification_without_question_rejected()

    test_cannot_answer_with_question_rejected()

    test_wrong_reason_family_rejected()

    test_extra_field_rejected()

    test_json_parsing()

    test_discriminated_json_schema()


    print()

    print(
        "Decision Router contract v0.7: PASS"
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.evals.decision_benchmark_v0_6 import (
    DecisionEvalCase,
)

from app.evals.decision_contract_v0_6 import (
    DecisionAnalyticalCandidate,
    require_analysis_candidate,
)

from app.evals.guardrails import (
    normalize_text,
)

from app.evals.ollama_baseline_v0_3 import (
    to_generic_candidate,
)

from app.evals.schemas import (
    AnalyticalEvalCase,
)

from app.evals.scorer_v0_2 import (
    score_candidate_v0_2,
)


# ============================================================
# VERSION
# ============================================================

DECISION_SCORER_VERSION = (
    "analytical_decision_scorer_v0.6"
)


# ============================================================
# CLARIFICATION TOPICS
#
# These are generic semantic categories used by the frozen
# benchmark. They are NOT tied to a particular model.
#
# The scorer remains deterministic: no LLM is used here.
# ============================================================

CLARIFICATION_TOPIC_KEYWORDS: dict[
    str,
    tuple[str, ...],
] = {
    "performance_metric": (
        "performance",
        "mesure",
        "indicateur",
        "critere",
        "chiffre d'affaires",
        "revenue",
        "revenu",
        "ventes",
        "unites",
        "volume",
        "marge",
        "rentabilite",
        "retour",
    ),

    "comparison_metric": (
        "mesure",
        "indicateur",
        "critere",
        "variable",
        "comparer",
        "comparaison",
        "effectif",
        "absence",
        "turnover",
        "rotation",
        "satisfaction",
    ),

    "reference_threshold": (
        "seuil",
        "reference",
        "objectif",
        "benchmark",
        "cible",
        "niveau",
        "eleve",
        "normal",
        "acceptable",
        "standard",
    ),
}


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_optional(
    value: str | None,
) -> str:
    return normalize_text(
        value,
    )


def _exact_optional(
    expected: str | None,
    actual: str | None,
) -> float:
    return (
        1.0
        if (
            _normalize_optional(
                expected,
            )
            == _normalize_optional(
                actual,
            )
        )
        else 0.0
    )


# ============================================================
# CLARIFICATION SCORING
# ============================================================

def _topic_is_covered(
    *,
    topic: str,
    question: str,
) -> bool:
    normalized_question = normalize_text(
        question,
    )

    normalized_topic = normalize_text(
        topic,
    )


    keywords = (
        CLARIFICATION_TOPIC_KEYWORDS
        .get(
            normalized_topic,
            (
                normalized_topic,
            ),
        )
    )


    return any(
        normalize_text(
            keyword,
        )
        in normalized_question

        for keyword
        in keywords
    )


def clarification_topic_score(
    *,
    expected_topics: list[str],
    clarification_question: str | None,
) -> float:
    """
    Deterministic semantic check for clarification requests.

    The benchmark stores abstract clarification topics such as:

        performance_metric
        comparison_metric
        reference_threshold

    The model itself produces a natural-language question.

    We therefore check whether that question actually addresses
    the ambiguity represented by the expected topic.
    """

    if not expected_topics:
        return 1.0


    if (
        clarification_question
        is None
        or not clarification_question.strip()
    ):
        return 0.0


    matched = sum(
        1
        for topic
        in expected_topics
        if _topic_is_covered(
            topic=topic,
            question=(
                clarification_question
            ),
        )
    )


    return (
        matched
        / len(
            expected_topics,
        )
    )


# ============================================================
# ANALYTICAL CASE ADAPTER
# ============================================================

def _to_analytical_eval_case(
    case: DecisionEvalCase,
) -> AnalyticalEvalCase:
    """
    Adapt an ANALYZE case from the v0.6 decision benchmark to
    the already validated analytical scorer v0.2.

    No ground-truth information is altered.
    """

    analytical_expectation = (
        case.expected.analytical
    )


    if analytical_expectation is None:
        raise ValueError(
            "This decision case does not contain "
            "an analytical expectation."
        )


    return AnalyticalEvalCase(
        case_id=case.case_id,

        split="test",

        domain=case.domain,

        user_request=case.user_request,

        datasets=case.datasets,

        available_tools=(
            case.available_tools
        ),

        expected=(
            analytical_expectation
        ),

        frozen=True,
    )


# ============================================================
# RESULT
# ============================================================

@dataclass(
    frozen=True,
)
class DecisionScoreV06:
    scorer_version: str

    case_id: str

    expected_decision: str

    actual_decision: str

    decision: float

    decision_reason: float

    clarification: (
        float
        | None
    )

    analytical_plan: (
        float
        | None
    )

    route_quality: float

    unsafe_execution: bool

    false_abstention: bool

    wrong_abstention_type: bool

    analytical_details: (
        dict[str, Any]
        | None
    )

    overall: float


    def as_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "scorer_version":
                self.scorer_version,

            "case_id":
                self.case_id,

            "expected_decision":
                self.expected_decision,

            "actual_decision":
                self.actual_decision,

            "metrics": {
                "decision":
                    self.decision,

                "decision_reason":
                    self.decision_reason,

                "clarification":
                    self.clarification,

                "analytical_plan":
                    self.analytical_plan,

                "route_quality":
                    self.route_quality,
            },

            "diagnostics": {
                "unsafe_execution":
                    self.unsafe_execution,

                "false_abstention":
                    self.false_abstention,

                "wrong_abstention_type":
                    self.wrong_abstention_type,
            },

            "analytical_details":
                self.analytical_details,

            "overall":
                self.overall,
        }


# ============================================================
# MAIN SCORER
# ============================================================

def score_decision_candidate(
    *,
    case: DecisionEvalCase,
    candidate: DecisionAnalyticalCandidate,
) -> DecisionScoreV06:
    expected = (
        case.expected
    )


    expected_decision = (
        expected.decision
    )

    actual_decision = (
        candidate.decision
    )


    # ========================================================
    # DECISION
    # ========================================================

    decision_score = (
        1.0
        if (
            actual_decision
            == expected_decision
        )
        else 0.0
    )


    # ========================================================
    # DECISION REASON
    # ========================================================

    decision_reason_score = (
        _exact_optional(
            expected.decision_reason,
            candidate.decision_reason,
        )
    )


    # ========================================================
    # FAILURE MODES
    # ========================================================

    unsafe_execution = (
        expected_decision
        != "analyze"

        and actual_decision
        == "analyze"
    )


    false_abstention = (
        expected_decision
        == "analyze"

        and actual_decision
        != "analyze"
    )


    wrong_abstention_type = (
        expected_decision
        in {
            "needs_clarification",
            "cannot_answer",
        }

        and actual_decision
        in {
            "needs_clarification",
            "cannot_answer",
        }

        and (
            expected_decision
            != actual_decision
        )
    )


    # ========================================================
    # ROUTE-SPECIFIC SCORES
    # ========================================================

    clarification_score: (
        float
        | None
    ) = None


    analytical_plan_score: (
        float
        | None
    ) = None


    analytical_details: (
        dict[str, Any]
        | None
    ) = None


    route_quality = 0.0


    # ========================================================
    # EXPECTED: ANALYZE
    # ========================================================

    if (
        expected_decision
        == "analyze"
    ):
        if (
            actual_decision
            == "analyze"
        ):
            typed_candidate = (
                require_analysis_candidate(
                    candidate,
                )
            )


            generic_candidate = (
                to_generic_candidate(
                    typed_candidate,
                )
            )


            analytical_case = (
                _to_analytical_eval_case(
                    case,
                )
            )


            analytical_score = (
                score_candidate_v0_2(
                    analytical_case,
                    generic_candidate,
                )
            )


            analytical_plan_score = (
                analytical_score.overall
            )


            analytical_details = (
                analytical_score.as_dict()
            )


            route_quality = (
                analytical_plan_score
            )


    # ========================================================
    # EXPECTED: NEEDS CLARIFICATION
    # ========================================================

    elif (
        expected_decision
        == "needs_clarification"
    ):
        if (
            actual_decision
            == "needs_clarification"
        ):
            clarification_score = (
                clarification_topic_score(
                    expected_topics=(
                        expected
                        .clarification_topics
                    ),

                    clarification_question=(
                        candidate
                        .clarification_question
                    ),
                )
            )


            # Correct reason + useful clarification question.
            route_quality = (
                (
                    decision_reason_score
                    + clarification_score
                )
                / 2
            )


    # ========================================================
    # EXPECTED: CANNOT ANSWER
    # ========================================================

    elif (
        expected_decision
        == "cannot_answer"
    ):
        if (
            actual_decision
            == "cannot_answer"
        ):
            # For cannot_answer, the central requirement is
            # identifying WHY execution is not justified.
            route_quality = (
                decision_reason_score
            )


    else:
        raise ValueError(
            "Unsupported expected decision: "
            f"{expected_decision}"
        )


    # ========================================================
    # OVERALL
    #
    # Decision choice   40%
    # Route quality     60%
    #
    # Consequence:
    #
    # - wrong high-level decision cannot be rescued by a
    #   superficially plausible plan;
    #
    # - correct decision with bad plan/reason is still heavily
    #   penalized.
    # ========================================================

    overall = (
        decision_score
        * 0.40

        + route_quality
        * 0.60
    )


    return DecisionScoreV06(
        scorer_version=(
            DECISION_SCORER_VERSION
        ),

        case_id=case.case_id,

        expected_decision=(
            expected_decision
        ),

        actual_decision=(
            actual_decision
        ),

        decision=(
            decision_score
        ),

        decision_reason=(
            decision_reason_score
        ),

        clarification=(
            clarification_score
        ),

        analytical_plan=(
            analytical_plan_score
        ),

        route_quality=(
            route_quality
        ),

        unsafe_execution=(
            unsafe_execution
        ),

        false_abstention=(
            false_abstention
        ),

        wrong_abstention_type=(
            wrong_abstention_type
        ),

        analytical_details=(
            analytical_details
        ),

        overall=(
            overall
        ),
    )
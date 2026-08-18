from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.evals.decision_router_benchmark_v0_7 import (
    DecisionRouterEvalCase,
)

from app.evals.decision_router_contract_v0_7 import (
    DecisionRouterCandidate,
    unwrap_router_candidate,
)

from app.evals.guardrails import (
    normalize_text,
)


# ============================================================
# VERSION
# ============================================================

DECISION_ROUTER_SCORER_VERSION = (
    "decision_router_scorer_v0.7"
)


# ============================================================
# CLARIFICATION TOPICS
#
# The benchmark stores abstract semantic topics.
#
# The model produces a natural-language clarification question.
#
# Scoring remains deterministic: no LLM is used here.
# ============================================================

CLARIFICATION_TOPIC_KEYWORDS: dict[
    str,
    tuple[str, ...],
] = {

    # --------------------------------------------------------
    # "Quels comptes/sites/produits performent le mieux ?"
    # --------------------------------------------------------

    "performance_metric": (
        "performance",
        "mesure",
        "metrique",
        "métrique",
        "indicateur",
        "critere",
        "critère",
        "revenu",
        "chiffre d'affaires",
        "chiffre d’affaires",
        "marge",
        "volume",
        "ventes",
        "vendu",
        "unités",
        "unites",
        "usage",
        "renouvellement",
        "production",
        "coût",
        "cout",
        "disponibilité",
        "disponibilite",
    ),

    # --------------------------------------------------------
    # "Compare les équipes/classes."
    # --------------------------------------------------------

    "comparison_metric": (
        "comparer",
        "comparaison",
        "mesure",
        "metrique",
        "métrique",
        "indicateur",
        "variable",
        "critere",
        "critère",
        "effectif",
        "absence",
        "satisfaction",
        "heures supplémentaires",
        "heures supplementaires",
        "score",
        "présence",
        "presence",
    ),

    # --------------------------------------------------------
    # "Est-ce élevé / acceptable ?"
    # --------------------------------------------------------

    "reference_threshold": (
        "seuil",
        "référence",
        "reference",
        "objectif",
        "cible",
        "benchmark",
        "sla",
        "standard",
        "niveau",
        "acceptable",
        "normal",
        "élevé",
        "eleve",
        "attendu",
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
# CLARIFICATION QUALITY
# ============================================================

def _topic_is_covered(
    *,
    topic: str,
    question: str,
) -> bool:
    normalized_topic = (
        normalize_text(
            topic,
        )
    )

    normalized_question = (
        normalize_text(
            question,
        )
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


def score_clarification_topics(
    *,
    expected_topics: list[str],
    clarification_question: str | None,
) -> float:
    """
    Score whether the generated clarification question targets
    the actual ambiguity represented by the benchmark.

    A generic question such as:

        "Peux-tu préciser ?"

    is intentionally not sufficient.
    """

    if not expected_topics:
        return 1.0


    if (
        clarification_question
        is None
        or not clarification_question.strip()
    ):
        return 0.0


    matched_topics = sum(
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
        matched_topics
        / len(
            expected_topics,
        )
    )


# ============================================================
# RESULT
# ============================================================

@dataclass(
    frozen=True,
)
class DecisionRouterScoreV07:
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

    route_quality: float

    unsafe_execution: bool

    false_abstention: bool

    wrong_abstention_type: bool

    overall: float


    def as_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
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

            "overall":
                self.overall,
        }


# ============================================================
# SCORER
# ============================================================

def score_decision_router_candidate(
    *,
    case: DecisionRouterEvalCase,
    candidate: DecisionRouterCandidate,
) -> DecisionRouterScoreV07:

    expected = (
        case.expected
    )


    route = (
        unwrap_router_candidate(
            candidate,
        )
    )


    expected_decision = (
        expected.decision
    )


    actual_decision = (
        route.decision
    )


    # ========================================================
    # HIGH-LEVEL DECISION
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
    # REASON
    #
    # For analyze:
    #
    #     expected reason = None
    #     actual reason   = None
    #
    # so this is also 1.0.
    # ========================================================

    decision_reason_score = (
        _exact_optional(
            expected.decision_reason,
            route.decision_reason,
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
    # ROUTE QUALITY
    #
    # This metric evaluates whether the model performed the
    # selected route correctly.
    #
    # A wrong high-level decision receives route_quality=0.
    # ========================================================

    clarification_score: (
        float
        | None
    ) = None


    route_quality = 0.0


    # ========================================================
    # ANALYZE
    # ========================================================

    if (
        expected_decision
        == "analyze"

        and actual_decision
        == "analyze"
    ):
        route_quality = 1.0


    # ========================================================
    # NEEDS CLARIFICATION
    # ========================================================

    elif (
        expected_decision
        == "needs_clarification"

        and actual_decision
        == "needs_clarification"
    ):
        clarification_score = (
            score_clarification_topics(
                expected_topics=(
                    expected
                    .clarification_topics
                ),

                clarification_question=(
                    route
                    .clarification_question
                ),
            )
        )


        route_quality = (
            (
                decision_reason_score
                + clarification_score
            )
            / 2
        )


    # ========================================================
    # CANNOT ANSWER
    # ========================================================

    elif (
        expected_decision
        == "cannot_answer"

        and actual_decision
        == "cannot_answer"
    ):
        route_quality = (
            decision_reason_score
        )


    # ========================================================
    # OVERALL
    #
    # Routing decision is the router's primary responsibility.
    #
    # Decision      60%
    # Route quality 40%
    #
    # A wrong high-level route therefore scores 0.
    #
    # A correct route with a wrong reason or weak clarification
    # remains partially correct, but is penalized.
    # ========================================================

    overall = (
        decision_score
        * 0.60

        + route_quality
        * 0.40
    )


    return DecisionRouterScoreV07(
        scorer_version=(
            DECISION_ROUTER_SCORER_VERSION
        ),

        case_id=(
            case.case_id
        ),

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

        overall=(
            overall
        ),
    )
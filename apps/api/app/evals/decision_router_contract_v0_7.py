from __future__ import annotations

from typing import (
    Annotated,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
)


# ============================================================
# VERSION
# ============================================================

DECISION_ROUTER_CONTRACT_VERSION = (
    "decision_router_contract_v0.7"
)


# ============================================================
# DECISION VOCABULARY
# ============================================================

RouterDecisionKind = Literal[
    "analyze",
    "needs_clarification",
    "cannot_answer",
]


ClarificationReason = Literal[
    "ambiguous_request",
    "insufficient_context",
]


CannotAnswerReason = Literal[
    "missing_column",
    "missing_dataset",
    "unsupported_analysis",
    "causal_identification_missing",
]


# ============================================================
# ANALYZE
# ============================================================

class AnalyzeRoute(
    BaseModel
):
    """
    The request is sufficiently specified and can be handled
    with the supplied datasets and available analytical
    capabilities.

    This router does NOT build the analytical plan.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    decision: Literal[
        "analyze"
    ]

    decision_reason: None

    clarification_question: None


# ============================================================
# NEEDS CLARIFICATION
# ============================================================

class NeedsClarificationRoute(
    BaseModel
):
    """
    The request cannot yet be interpreted uniquely, but the
    missing information can reasonably be supplied by the user.

    Examples:

    - "Quels produits performent le mieux ?"
      -> Which performance metric?

    - "Les délais sont-ils élevés ?"
      -> Compared with which threshold/reference?
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    decision: Literal[
        "needs_clarification"
    ]

    decision_reason: (
        ClarificationReason
    )

    clarification_question: str = (
        Field(
            min_length=5,
        )
    )


# ============================================================
# CANNOT ANSWER
# ============================================================

class CannotAnswerRoute(
    BaseModel
):
    """
    The request cannot be answered correctly with the supplied
    analytical context.

    This is different from ambiguity.

    Examples:

    - required source column is absent;
    - required dataset is absent;
    - required analytical capability is unavailable;
    - explicit causal identification is requested from an
      observational context that cannot establish causality.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    decision: Literal[
        "cannot_answer"
    ]

    decision_reason: (
        CannotAnswerReason
    )

    clarification_question: None


# ============================================================
# DISCRIMINATED UNION
# ============================================================

RouterDecision = Annotated[
    (
        AnalyzeRoute
        | NeedsClarificationRoute
        | CannotAnswerRoute
    ),
    Field(
        discriminator="decision",
    ),
]


# ============================================================
# ROOT CONTRACT
# ============================================================

class DecisionRouterCandidate(
    RootModel[
        RouterDecision
    ]
):
    """
    Minimal DataLens routing contract.

    The JSON Schema exposes three explicit branches through a
    discriminated union.

    The router intentionally knows nothing about:

    - analytical intent;
    - entity;
    - grain;
    - relevant columns;
    - analytical family;
    - tool calls.

    Those belong to the downstream analytical planner.
    """

    root: RouterDecision


# ============================================================
# ACCESS HELPERS
# ============================================================

def unwrap_router_candidate(
    candidate: DecisionRouterCandidate,
) -> RouterDecision:
    return candidate.root


def router_decision(
    candidate: DecisionRouterCandidate,
) -> RouterDecisionKind:
    return candidate.root.decision
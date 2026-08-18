from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.evals.ollama_baseline_v0_3 import (
    AnalyticalFamily,
    AnalyticalIntent,
    ControlledAssumption,
    TypedToolCall,
)


# ============================================================
# VERSION
# ============================================================

DECISION_CONTRACT_VERSION = (
    "analytical_decision_contract_v0.6"
)


# ============================================================
# DECISION VOCABULARY
# ============================================================

DecisionKind = Literal[
    "analyze",
    "needs_clarification",
    "cannot_answer",
]


DecisionReason = Literal[
    "ambiguous_request",
    "missing_column",
    "missing_dataset",
    "insufficient_context",
    "unsupported_analysis",
    "causal_identification_missing",
]


# ============================================================
# DECISION CANDIDATE
# ============================================================

class DecisionAnalyticalCandidate(
    BaseModel
):
    """
    DataLens analytical decision contract v0.6.

    The model must first decide whether it can safely construct
    an analytical plan.

    Possible decisions:

    - analyze
    - needs_clarification
    - cannot_answer

    All fields are required in the JSON Schema, but analytical
    fields are nullable because an abstention is a valid result.

    This avoids forcing the model to invent a plan when the
    request cannot be answered safely from the supplied context.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    # --------------------------------------------------------
    # HIGH-LEVEL DECISION
    # --------------------------------------------------------

    decision: DecisionKind

    decision_reason: (
        DecisionReason
        | None
    )

    clarification_question: (
        str
        | None
    )

    # --------------------------------------------------------
    # ANALYTICAL INTERPRETATION
    # --------------------------------------------------------

    intent: (
        AnalyticalIntent
        | None
    )

    entity: (
        str
        | None
    )

    current_grain: (
        str
        | None
    )

    target_grain: (
        str
        | None
    )

    relevant_columns: list[
        str
    ]

    family: (
        AnalyticalFamily
        | None
    )

    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    tool_calls: list[
        TypedToolCall
    ]

    assumptions: list[
        ControlledAssumption
    ]

    # ========================================================
    # CROSS-FIELD CONTRACT
    # ========================================================

    @model_validator(
        mode="after",
    )
    def validate_decision_contract(
        self,
    ) -> (
        "DecisionAnalyticalCandidate"
    ):
        # ====================================================
        # ANALYZE
        # ====================================================

        if (
            self.decision
            == "analyze"
        ):
            if (
                self.decision_reason
                is not None
            ):
                raise ValueError(
                    "decision_reason must be null "
                    "when decision='analyze'."
                )

            if (
                self.clarification_question
                is not None
            ):
                raise ValueError(
                    "clarification_question must be null "
                    "when decision='analyze'."
                )

            if self.intent is None:
                raise ValueError(
                    "intent is required "
                    "when decision='analyze'."
                )

            if (
                self.current_grain
                is None
                or not self.current_grain.strip()
            ):
                raise ValueError(
                    "current_grain is required "
                    "when decision='analyze'."
                )

            if self.family is None:
                raise ValueError(
                    "family is required "
                    "when decision='analyze'."
                )

            if not self.relevant_columns:
                raise ValueError(
                    "relevant_columns must not be empty "
                    "when decision='analyze'."
                )

            if not self.tool_calls:
                raise ValueError(
                    "tool_calls must not be empty "
                    "when decision='analyze'."
                )

            return self


        # ====================================================
        # NEEDS CLARIFICATION
        # ====================================================

        if (
            self.decision
            == "needs_clarification"
        ):
            if (
                self.decision_reason
                is None
            ):
                raise ValueError(
                    "decision_reason is required "
                    "when decision='needs_clarification'."
                )

            if (
                self.clarification_question
                is None
                or not (
                    self
                    .clarification_question
                    .strip()
                )
            ):
                raise ValueError(
                    "clarification_question is required "
                    "when decision='needs_clarification'."
                )

            if self.tool_calls:
                raise ValueError(
                    "tool_calls must be empty "
                    "when decision='needs_clarification'."
                )

            if self.intent is not None:
                raise ValueError(
                    "intent must be null "
                    "when decision='needs_clarification'."
                )

            if self.family is not None:
                raise ValueError(
                    "family must be null "
                    "when decision='needs_clarification'."
                )

            if self.target_grain is not None:
                raise ValueError(
                    "target_grain must be null "
                    "when decision='needs_clarification'."
                )

            return self


        # ====================================================
        # CANNOT ANSWER
        # ====================================================

        if (
            self.decision
            == "cannot_answer"
        ):
            if (
                self.decision_reason
                is None
            ):
                raise ValueError(
                    "decision_reason is required "
                    "when decision='cannot_answer'."
                )

            if self.tool_calls:
                raise ValueError(
                    "tool_calls must be empty "
                    "when decision='cannot_answer'."
                )

            if self.intent is not None:
                raise ValueError(
                    "intent must be null "
                    "when decision='cannot_answer'."
                )

            if self.family is not None:
                raise ValueError(
                    "family must be null "
                    "when decision='cannot_answer'."
                )

            if self.target_grain is not None:
                raise ValueError(
                    "target_grain must be null "
                    "when decision='cannot_answer'."
                )

            return self


        raise ValueError(
            "Unsupported analytical decision."
        )


# ============================================================
# CONVERSION TO PREVIOUS ANALYTICAL CONTRACT
# ============================================================

def require_analysis_candidate(
    candidate: DecisionAnalyticalCandidate,
):
    """
    Convert a v0.6 decision to the previous typed analytical
    candidate only when the model explicitly selected ANALYZE.

    This keeps execution separated from abstention.
    """

    if (
        candidate.decision
        != "analyze"
    ):
        raise ValueError(
            "Cannot create an analytical execution plan "
            f"from decision={candidate.decision!r}."
        )


    # Local import prevents unnecessary coupling during module
    # initialization while preserving the existing v0.3 type.
    from app.evals.ollama_baseline_v0_3 import (
        TypedAnalyticalCandidate,
    )


    assert (
        candidate.intent
        is not None
    )

    assert (
        candidate.current_grain
        is not None
    )

    assert (
        candidate.family
        is not None
    )


    return TypedAnalyticalCandidate(
        intent=(
            candidate.intent
        ),

        entity=(
            candidate.entity
        ),

        current_grain=(
            candidate.current_grain
        ),

        target_grain=(
            candidate.target_grain
        ),

        relevant_columns=(
            candidate.relevant_columns
        ),

        family=(
            candidate.family
        ),

        tool_calls=(
            candidate.tool_calls
        ),

        assumptions=(
            candidate.assumptions
        ),
    )
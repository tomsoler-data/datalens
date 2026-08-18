from __future__ import annotations

import json

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.evals.schemas import (
    DatasetContext,
)


# ============================================================
# VERSION
# ============================================================

DECISION_ROUTER_BENCHMARK_VERSION = (
    "decision_router_benchmark_v0.7"
)


# ============================================================
# VOCABULARY
# ============================================================

RouterBenchmarkSplit = Literal[
    "train",
    "validation",
]


RouterExpectedDecision = Literal[
    "analyze",
    "needs_clarification",
    "cannot_answer",
]


CLARIFICATION_REASONS = {
    "ambiguous_request",
    "insufficient_context",
}


CANNOT_ANSWER_REASONS = {
    "missing_column",
    "missing_dataset",
    "unsupported_analysis",
    "causal_identification_missing",
}


# ============================================================
# EXPECTATION
# ============================================================

class RouterExpectation(
    BaseModel
):
    """
    Ground truth for the Decision Router only.

    This benchmark deliberately does NOT contain:

    - analytical intent;
    - analytical family;
    - grain transformation;
    - relevant analytical columns;
    - tool calls.

    Those belong to the downstream analytical planner.

    The router is evaluated only on whether execution should
    continue, clarification should be requested, or execution
    must stop.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    decision: RouterExpectedDecision

    decision_reason: (
        str
        | None
    )

    clarification_topics: list[
        str
    ] = Field(
        default_factory=list,
    )

    notes: (
        str
        | None
    ) = None


    @model_validator(
        mode="after",
    )
    def validate_expectation(
        self,
    ) -> "RouterExpectation":

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
                    "when expected decision='analyze'."
                )

            if self.clarification_topics:
                raise ValueError(
                    "clarification_topics must be empty "
                    "when expected decision='analyze'."
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
                not in CLARIFICATION_REASONS
            ):
                raise ValueError(
                    "needs_clarification requires one of: "
                    f"{sorted(CLARIFICATION_REASONS)}"
                )

            if (
                not self.clarification_topics
            ):
                raise ValueError(
                    "needs_clarification requires at least "
                    "one clarification topic."
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
                not in CANNOT_ANSWER_REASONS
            ):
                raise ValueError(
                    "cannot_answer requires one of: "
                    f"{sorted(CANNOT_ANSWER_REASONS)}"
                )

            if self.clarification_topics:
                raise ValueError(
                    "clarification_topics must be empty "
                    "when expected decision='cannot_answer'."
                )

            return self


        raise ValueError(
            "Unsupported expected router decision."
        )


# ============================================================
# CASE
# ============================================================

class DecisionRouterEvalCase(
    BaseModel
):
    """
    Development benchmark case for the v0.7 router.

    IMPORTANT:
    These cases are explicitly NOT frozen.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str = Field(
        min_length=1,
    )

    split: RouterBenchmarkSplit

    domain: str = Field(
        min_length=1,
    )

    user_request: str = Field(
        min_length=1,
    )

    datasets: list[
        DatasetContext
    ] = Field(
        min_length=1,
    )

    available_tools: list[
        str
    ]

    expected: RouterExpectation

    frozen: Literal[
        False
    ] = False


    @model_validator(
        mode="after",
    )
    def validate_case(
        self,
    ) -> "DecisionRouterEvalCase":

        if not self.available_tools:
            raise ValueError(
                "At least one analytical tool "
                "must be exposed to the router."
            )

        return self


# ============================================================
# LOADER
# ============================================================

def load_decision_router_benchmark(
    path: str | Path,
    *,
    split: RouterBenchmarkSplit | None = None,
) -> list[
    DecisionRouterEvalCase
]:
    benchmark_path = Path(
        path,
    )


    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark introuvable : {benchmark_path}"
        )


    cases: list[
        DecisionRouterEvalCase
    ] = []


    seen_ids: set[
        str
    ] = set()


    with benchmark_path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:

        for (
            line_number,
            raw_line,
        ) in enumerate(
            handle,
            start=1,
        ):
            line = (
                raw_line.strip()
            )


            if not line:
                continue


            try:
                payload = json.loads(
                    line,
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "JSON invalide dans "
                    f"{benchmark_path} "
                    f"ligne {line_number}: "
                    f"{exc.msg}"
                ) from exc


            try:
                case = (
                    DecisionRouterEvalCase
                    .model_validate(
                        payload,
                    )
                )

            except Exception as exc:
                raise ValueError(
                    "Cas router v0.7 invalide dans "
                    f"{benchmark_path} "
                    f"ligne {line_number}: "
                    f"{exc}"
                ) from exc


            if (
                case.case_id
                in seen_ids
            ):
                raise ValueError(
                    "case_id dupliqué : "
                    f"{case.case_id}"
                )


            seen_ids.add(
                case.case_id,
            )


            if (
                split is None
                or case.split == split
            ):
                cases.append(
                    case,
                )


    return cases
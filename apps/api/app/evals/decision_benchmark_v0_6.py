from __future__ import annotations

import json

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.evals.decision_contract_v0_6 import (
    DecisionKind,
    DecisionReason,
)

from app.evals.schemas import (
    AnalyticalExpectation,
    DatasetContext,
)


# ============================================================
# VERSION
# ============================================================

DECISION_BENCHMARK_SCHEMA_VERSION = (
    "decision_benchmark_schema_v0.6"
)


# ============================================================
# EXPECTATION
# ============================================================

class DecisionExpectation(
    BaseModel
):
    """
    Ground truth for the frozen decision benchmark.

    Exactly one of these paths is expected:

    ANALYZE
        -> analytical expectation required

    NEEDS_CLARIFICATION
        -> clarification topics required

    CANNOT_ANSWER
        -> explicit reason required
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    decision: DecisionKind

    decision_reason: (
        DecisionReason
        | None
    ) = None

    clarification_topics: list[
        str
    ] = Field(
        default_factory=list,
    )

    analytical: (
        AnalyticalExpectation
        | None
    ) = None


    @model_validator(
        mode="after",
    )
    def validate_expectation(
        self,
    ) -> "DecisionExpectation":

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
                    "for analyze expectations."
                )

            if self.analytical is None:
                raise ValueError(
                    "analytical expectation is required "
                    "when decision='analyze'."
                )

            if self.clarification_topics:
                raise ValueError(
                    "clarification_topics must be empty "
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
                    "for clarification expectations."
                )

            if (
                self.analytical
                is not None
            ):
                raise ValueError(
                    "analytical expectation must be null "
                    "when clarification is required."
                )

            if (
                not self.clarification_topics
            ):
                raise ValueError(
                    "At least one clarification topic "
                    "is required."
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

            if (
                self.analytical
                is not None
            ):
                raise ValueError(
                    "analytical expectation must be null "
                    "when decision='cannot_answer'."
                )

            if self.clarification_topics:
                raise ValueError(
                    "clarification_topics must be empty "
                    "for cannot_answer."
                )

            return self


        raise ValueError(
            "Unsupported decision expectation."
        )


# ============================================================
# EVAL CASE
# ============================================================

class DecisionEvalCase(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str = Field(
        min_length=1,
    )

    split: str

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

    expected: DecisionExpectation

    frozen: bool


    @model_validator(
        mode="after",
    )
    def validate_case(
        self,
    ) -> "DecisionEvalCase":

        # ====================================================
        # FROZEN TEST CONTRACT
        # ====================================================

        if (
            self.split
            != "test"
        ):
            raise ValueError(
                "Decision benchmark v0.6 "
                "contains only split='test' cases."
            )

        if not self.frozen:
            raise ValueError(
                "All decision benchmark v0.6 "
                "test cases must be frozen."
            )


        # ====================================================
        # ANALYTICAL EXPECTATION VALIDATION
        # ====================================================

        analytical = (
            self.expected.analytical
        )

        if analytical is None:
            return self


        known_columns = {
            column.name
            for dataset
            in self.datasets
            for column
            in dataset.columns
        }


        unknown_expected_columns = (
            set(
                analytical
                .relevant_columns
            )
            - known_columns
        )


        if unknown_expected_columns:
            raise ValueError(
                "Expected relevant columns "
                "do not exist in supplied datasets: "
                f"{sorted(unknown_expected_columns)}"
            )


        available_tool_set = set(
            self.available_tools
        )


        unknown_tools = (
            set(
                analytical
                .acceptable_tools
            )
            - available_tool_set
        )


        if unknown_tools:
            raise ValueError(
                "Expected tools are not available: "
                f"{sorted(unknown_tools)}"
            )


        required_argument_tools = set(
            analytical
            .required_tool_arguments
        )


        invalid_required_tools = (
            required_argument_tools
            - set(
                analytical
                .acceptable_tools
            )
        )


        if invalid_required_tools:
            raise ValueError(
                "required_tool_arguments contains tools "
                "that are not acceptable_tools: "
                f"{sorted(invalid_required_tools)}"
            )


        return self


# ============================================================
# LOADER
# ============================================================

def load_decision_benchmark(
    path: str | Path,
) -> list[
    DecisionEvalCase
]:
    benchmark_path = Path(
        path,
    )


    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark introuvable : {benchmark_path}"
        )


    cases: list[
        DecisionEvalCase
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
            line = raw_line.strip()


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
                    DecisionEvalCase
                    .model_validate(
                        payload,
                    )
                )

            except Exception as exc:
                raise ValueError(
                    "Cas v0.6 invalide dans "
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

            cases.append(
                case,
            )


    return cases
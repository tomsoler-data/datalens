from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


EvalSplit = Literal[
    "train",
    "validation",
    "test",
]


class DatasetColumnSpec(BaseModel):
    """
    Minimal description of a column exposed to the AI evaluator.

    We deliberately avoid sending raw data here.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    name: str = Field(
        min_length=1,
    )

    analytical_type: str = Field(
        min_length=1,
    )

    semantic_role: str | None = None


class DatasetContext(BaseModel):
    """
    Compact dataset context available to the reasoning model.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    dataset_id: str = Field(
        min_length=1,
    )

    filename: str = Field(
        min_length=1,
    )

    grain: str = Field(
        min_length=1,
    )

    entity_columns: list[str] = Field(
        default_factory=list,
    )

    columns: list[
        DatasetColumnSpec
    ] = Field(
        min_length=1,
    )

    @model_validator(
        mode="after",
    )
    def validate_entity_columns(
        self,
    ) -> "DatasetContext":
        known_columns = {
            column.name
            for column in self.columns
        }

        unknown = [
            column
            for column in self.entity_columns
            if column not in known_columns
        ]

        if unknown:
            raise ValueError(
                "entity_columns contains unknown columns: "
                + ", ".join(
                    sorted(
                        unknown,
                    )
                )
            )

        return self


class AnalyticalExpectation(BaseModel):
    """
    Ground truth attached to one evaluation case.

    This is NOT passed to the model.

    It is used only after inference to evaluate the model's answer.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    intent: str = Field(
        min_length=1,
    )

    entity: str | None = None

    current_grain: str = Field(
        min_length=1,
    )

    target_grain: str | None = None

    relevant_columns: list[str] = Field(
        default_factory=list,
    )

    family: str = Field(
        min_length=1,
    )

    acceptable_tools: list[str] = Field(
        default_factory=list,
    )

    required_tool_arguments: dict[
        str,
        dict[str, Any],
    ] = Field(
        default_factory=dict,
    )

    forbidden_tools: list[str] = Field(
        default_factory=list,
    )

    forbidden_assumptions: list[str] = Field(
        default_factory=list,
    )

    requires_reasoning: bool = True

    notes: str | None = None


class AnalyticalEvalCase(BaseModel):
    """
    One benchmark case.

    train:
        may later be used for fine-tuning.

    validation:
        may be used while developing prompts/models.

    test:
        frozen evaluation set.
        It must never become training material.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str = Field(
        min_length=1,
    )

    split: EvalSplit

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

    available_tools: list[str] = Field(
        min_length=1,
    )

    expected: AnalyticalExpectation

    frozen: bool = False

    @model_validator(
        mode="after",
    )
    def validate_case(
        self,
    ) -> "AnalyticalEvalCase":
        if (
            self.split == "test"
            and not self.frozen
        ):
            raise ValueError(
                "All test cases must be frozen."
            )

        known_columns = {
            column.name
            for dataset in self.datasets
            for column in dataset.columns
        }

        unknown_expected_columns = [
            column
            for column
            in self.expected.relevant_columns
            if column not in known_columns
        ]

        if unknown_expected_columns:
            raise ValueError(
                "expected.relevant_columns contains "
                "unknown columns: "
                + ", ".join(
                    sorted(
                        unknown_expected_columns,
                    )
                )
            )

        unavailable_tools = [
            tool
            for tool
            in self.expected.acceptable_tools
            if tool not in self.available_tools
        ]

        if unavailable_tools:
            raise ValueError(
                "expected.acceptable_tools contains "
                "unavailable tools: "
                + ", ".join(
                    sorted(
                        unavailable_tools,
                    )
                )
            )

        for tool_name in (
            self.expected
            .required_tool_arguments
        ):
            if (
                tool_name
                not in self.expected.acceptable_tools
            ):
                raise ValueError(
                    "required_tool_arguments may only "
                    "reference an acceptable tool: "
                    + tool_name
                )

        return self


class ToolCallCandidate(BaseModel):
    """
    One tool call proposed by the model.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    name: str = Field(
        min_length=1,
    )

    arguments: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class AnalyticalCandidate(BaseModel):
    """
    Provider-neutral representation of the analytical plan
    proposed by an AI model.

    Gemma, Qwen or another model will eventually be converted
    into this same structure.

    This lets DataLens compare models using exactly the same
    benchmark and scoring rules.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    intent: str | None = None

    entity: str | None = None

    current_grain: str | None = None

    target_grain: str | None = None

    relevant_columns: list[str] = Field(
        default_factory=list,
    )

    family: str | None = None

    tool_calls: list[
        ToolCallCandidate
    ] = Field(
        default_factory=list,
    )

    assumptions: list[str] = Field(
        default_factory=list,
    )
from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.planning.analytical_v1.relationships import (
    CrossDatasetFeasibility,
    RoutingRelationshipContext,
    evaluate_cross_dataset_feasibility,
)


# ============================================================
# VERSION
# ============================================================

DATASET_DEPENDENCY_CONTRACT_VERSION = (
    "dataset_dependency_contract_v0.8"
)


DATASET_DEPENDENCY_GATE_VERSION = (
    "dataset_dependency_gate_v0.8"
)


# ============================================================
# ONE ANALYTICAL REQUIREMENT
# ============================================================

class DatasetRequirement(
    BaseModel
):
    """
    One analytical result and the datasets that must
    participate in that SAME result.

    Examples
    --------

    Independent analyses:

        requirement_1 -> ["sales"]
        requirement_2 -> ["support"]

    Cross-dataset association:

        requirement_1 -> ["sales", "support"]

    The AI identifies semantic dependencies.

    It does NOT decide whether those datasets can actually
    be combined. Python validates that separately.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    requirement_id: str = Field(
        min_length=1,
    )

    dataset_ids: list[
        str
    ] = Field(
        min_length=1,
    )


    @field_validator(
        "dataset_ids",
    )
    @classmethod
    def validate_dataset_ids(
        cls,
        value: list[str],
    ) -> list[str]:

        cleaned = [
            dataset_id.strip()
            for dataset_id
            in value
        ]


        if any(
            not dataset_id
            for dataset_id
            in cleaned
        ):
            raise ValueError(
                "dataset_ids must not contain "
                "empty values."
            )


        if (
            len(
                cleaned
            )
            != len(
                set(
                    cleaned
                )
            )
        ):
            raise ValueError(
                "dataset_ids must not contain "
                "duplicates inside one requirement."
            )


        return cleaned


# ============================================================
# AI CANDIDATE
# ============================================================

class DatasetDependencyCandidate(
    BaseModel
):
    """
    Minimal semantic dependency output.

    The model only identifies which supplied datasets are
    required together for each analytical result.

    It does NOT decide:

    - whether joins are valid;
    - whether grains are compatible;
    - whether a relationship exists;
    - whether execution should be allowed.

    Those decisions remain deterministic.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    requirements: list[
        DatasetRequirement
    ] = Field(
        min_length=1,
    )


    @model_validator(
        mode="after",
    )
    def validate_requirements(
        self,
    ) -> "DatasetDependencyCandidate":

        requirement_ids = [
            requirement.requirement_id

            for requirement
            in self.requirements
        ]


        if (
            len(
                requirement_ids
            )
            != len(
                set(
                    requirement_ids
                )
            )
        ):
            raise ValueError(
                "requirement_id values must be unique."
            )


        return self


# ============================================================
# VALIDATED REQUIREMENT RESULT
# ============================================================

class DatasetRequirementFeasibility(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    requirement_id: str

    dataset_ids: list[
        str
    ]

    feasibility: (
        CrossDatasetFeasibility
    )

    executable: bool


# ============================================================
# GATE RESULT
# ============================================================

class DatasetDependencyGateResult(
    BaseModel
):
    """
    Deterministic structural verdict.

    executable=True only when every analytical requirement
    can be satisfied structurally.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    gate_version: str

    executable: bool

    requirements: list[
        DatasetRequirementFeasibility
    ]

    blocking_requirements: list[
        str
    ]

    routing_override_reason: (
        str
        | None
    )


# ============================================================
# VALIDATE DATASET REFERENCES
# ============================================================

def validate_dependency_candidate(
    *,
    candidate: DatasetDependencyCandidate,
    context: RoutingRelationshipContext,
) -> None:
    """
    Python rejects dataset identifiers invented by the model.
    """

    known_dataset_ids = {
        dataset.dataset_id

        for dataset
        in context.datasets
    }


    referenced_dataset_ids = {
        dataset_id

        for requirement
        in candidate.requirements

        for dataset_id
        in requirement.dataset_ids
    }


    unknown_dataset_ids = (
        referenced_dataset_ids
        - known_dataset_ids
    )


    if unknown_dataset_ids:

        raise ValueError(
            "Dataset dependency candidate references "
            "unknown dataset(s): "
            f"{sorted(unknown_dataset_ids)}"
        )


# ============================================================
# PYTHON FEASIBILITY GATE
# ============================================================

def evaluate_dataset_dependencies(
    *,
    candidate: DatasetDependencyCandidate,
    context: RoutingRelationshipContext,
) -> DatasetDependencyGateResult:
    """
    Evaluate every dependency group independently.

    Rules
    -----

    ["sales"]
        -> not_required
        -> executable

    ["sales", "support"]
        + combination tool
        + validated relationship path
        -> supported
        -> executable

    ["sales", "support"]
        without combination capability
        -> blocked

    ["sales", "machines"]
        combination tool exists
        but no validated path
        -> blocked

    Any blocked requirement makes the whole request
    structurally non-executable.
    """

    validate_dependency_candidate(
        candidate=candidate,
        context=context,
    )


    requirement_results: list[
        DatasetRequirementFeasibility
    ] = []


    blocking_requirements: list[
        str
    ] = []


    for requirement in candidate.requirements:

        feasibility = (
            evaluate_cross_dataset_feasibility(
                context=context,

                required_dataset_ids=(
                    requirement.dataset_ids
                ),
            )
        )


        executable = (
            feasibility
            in {
                "not_required",
                "supported",
            }
        )


        result = (
            DatasetRequirementFeasibility(
                requirement_id=(
                    requirement.requirement_id
                ),

                dataset_ids=(
                    requirement.dataset_ids
                ),

                feasibility=(
                    feasibility
                ),

                executable=(
                    executable
                ),
            )
        )


        requirement_results.append(
            result
        )


        if not executable:

            blocking_requirements.append(
                requirement.requirement_id
            )


    globally_executable = (
        len(
            blocking_requirements
        )
        == 0
    )


    # ========================================================
    # STRUCTURAL ROUTING OVERRIDE
    #
    # Both failure types are currently exposed to the router
    # vocabulary as unsupported_analysis.
    #
    # Internally, however, we preserve the precise structural
    # cause in `feasibility`.
    # ========================================================

    routing_override_reason = (
        None

        if globally_executable

        else "unsupported_analysis"
    )


    return DatasetDependencyGateResult(
        gate_version=(
            DATASET_DEPENDENCY_GATE_VERSION
        ),

        executable=(
            globally_executable
        ),

        requirements=(
            requirement_results
        ),

        blocking_requirements=(
            blocking_requirements
        ),

        routing_override_reason=(
            routing_override_reason
        ),
    )


# ============================================================
# SERIALIZATION HELPER
# ============================================================

def dependency_gate_summary(
    result: DatasetDependencyGateResult,
) -> dict[str, Any]:
    """
    Compact payload suitable for a downstream router/planner.
    """

    return {
        "executable":
            result.executable,

        "routing_override_reason":
            result.routing_override_reason,

        "requirements": [
            {
                "requirement_id":
                    requirement.requirement_id,

                "dataset_ids":
                    requirement.dataset_ids,

                "feasibility":
                    requirement.feasibility,

                "executable":
                    requirement.executable,
            }

            for requirement
            in result.requirements
        ],
    }
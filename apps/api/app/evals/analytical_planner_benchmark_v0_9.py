from __future__ import annotations

import json

from pathlib import Path
from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.evals.analytical_planner_context_v0_9 import (
    build_analytical_planner_context,
)

from app.evals.analytical_planner_contract_v0_9 import (
    AnalyticalPlannerCandidate,
)

from app.evals.analytical_planner_input_v0_9 import (
    AnalyticalPlannerInput,
    build_analytical_planner_input,
)

from app.evals.analytical_planner_validator_v0_9 import (
    validate_analytical_planner_candidate,
)

from app.evals.dataset_dependency_contract_v0_8 import (
    DatasetDependencyCandidate,
)

from app.evals.routing_relationships_v0_8 import (
    DatasetRelationshipSpec,
    RoutingRelationshipContext,
)

from app.evals.schemas import (
    DatasetContext,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_BENCHMARK_VERSION = (
    "analytical_planner_development_benchmark_v0.9"
)


# ============================================================
# SPLIT
# ============================================================

AnalyticalPlannerEvalSplit = Literal[
    "train",
    "validation",
]


# ============================================================
# CASE
# ============================================================

class AnalyticalPlannerEvalCase(
    BaseModel
):
    """
    Development evaluation case for the new analytical planner.

    The benchmark stores:

    - the user request;
    - dataset schemas;
    - validated structural relationships;
    - available capabilities;
    - semantic dependency ground truth;
    - expected analytical plan.

    The planner model itself will NOT receive the expected plan.

    The exact model-visible AnalyticalPlannerInput is rebuilt
    deterministically from the other fields.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str = Field(
        min_length=1,
    )

    split: AnalyticalPlannerEvalSplit

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

    relationships: list[
        DatasetRelationshipSpec
    ] = Field(
        default_factory=list,
    )

    available_tools: list[
        str
    ] = Field(
        min_length=1,
    )

    dependency_candidate: (
        DatasetDependencyCandidate
    )

    expected: AnalyticalPlannerCandidate

    notes: (
        str
        | None
    ) = None

    frozen: Literal[
        False
    ] = False


    @model_validator(
        mode="after",
    )
    def validate_case(
        self,
    ) -> "AnalyticalPlannerEvalCase":

        # ====================================================
        # STRUCTURAL CONTEXT
        # ====================================================

        structural_context = (
            RoutingRelationshipContext(
                datasets=(
                    self.datasets
                ),

                relationships=(
                    self.relationships
                ),

                available_tools=(
                    self.available_tools
                ),
            )
        )


        # ====================================================
        # BUILD TRUSTED PLANNER CONTEXT
        # ====================================================

        planner_context = (
            build_analytical_planner_context(
                candidate=(
                    self.dependency_candidate
                ),

                context=(
                    structural_context
                ),
            )
        )


        if not (
            planner_context.ready_for_planning
        ):

            raise ValueError(
                "Development planner benchmark cases must "
                "be structurally executable. "
                f"Blocking requirements: "
                f"{planner_context.blocking_requirements}"
            )


        # ====================================================
        # BUILD EXACT MODEL INPUT
        # ====================================================

        planner_input = (
            build_analytical_planner_input(
                user_request=(
                    self.user_request
                ),

                planner_context=(
                    planner_context
                ),

                structural_context=(
                    structural_context
                ),
            )
        )


        # ====================================================
        # EXPECTED PLAN MUST PASS THE REAL VALIDATOR
        # ====================================================

        validation = (
            validate_analytical_planner_candidate(
                candidate=(
                    self.expected
                ),

                planner_input=(
                    planner_input
                ),
            )
        )


        if not (
            validation.valid
        ):

            issues = [
                (
                    issue.code,
                    issue.requirement_id,
                    issue.step_id,
                    issue.message,
                )

                for issue
                in validation.issues
            ]


            raise ValueError(
                "Expected analytical planner candidate "
                "does not pass deterministic validation: "
                f"{issues}"
            )


        return self


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_planner_input_for_case(
    case: AnalyticalPlannerEvalCase,
) -> AnalyticalPlannerInput:
    """
    Reconstruct the exact trusted input that may be sent to
    the analytical planning model.

    Expected benchmark answers never enter this object.
    """

    structural_context = (
        RoutingRelationshipContext(
            datasets=(
                case.datasets
            ),

            relationships=(
                case.relationships
            ),

            available_tools=(
                case.available_tools
            ),
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=(
                case.dependency_candidate
            ),

            context=(
                structural_context
            ),
        )
    )


    if not (
        planner_context.ready_for_planning
    ):

        raise ValueError(
            "Planner benchmark case is not ready "
            f"for planning: {case.case_id}"
        )


    return (
        build_analytical_planner_input(
            user_request=(
                case.user_request
            ),

            planner_context=(
                planner_context
            ),

            structural_context=(
                structural_context
            ),
        )
    )


# ============================================================
# LOADER
# ============================================================

def load_analytical_planner_benchmark(
    path: str | Path,
    *,
    split: AnalyticalPlannerEvalSplit | None = None,
) -> list[
    AnalyticalPlannerEvalCase
]:

    benchmark_path = Path(
        path
    )


    cases: list[
        AnalyticalPlannerEvalCase
    ] = []


    seen_case_ids: set[
        str
    ] = set()


    with benchmark_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, raw_line in enumerate(
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
                    line
                )


            except json.JSONDecodeError as error:

                raise ValueError(
                    "Invalid JSON on benchmark line "
                    f"{line_number}: {error}"
                ) from error


            case = (
                AnalyticalPlannerEvalCase
                .model_validate(
                    payload
                )
            )


            if (
                case.case_id
                in seen_case_ids
            ):

                raise ValueError(
                    "Duplicate analytical planner "
                    f"case_id: {case.case_id}"
                )


            seen_case_ids.add(
                case.case_id
            )


            if (
                split is not None
                and case.split != split
            ):
                continue


            cases.append(
                case
            )


    if not cases:

        raise ValueError(
            "Analytical planner benchmark "
            "contains no matching cases."
        )


    return cases
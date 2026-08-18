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

from app.evals.analytical_planner_validator_v0_9_1 import (
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

ANALYTICAL_PLANNER_FROZEN_BENCHMARK_VERSION = (
    "analytical_planner_frozen_benchmark_v1.0"
)


# ============================================================
# FROZEN CASE
# ============================================================

class FrozenAnalyticalPlannerEvalCase(
    BaseModel
):
    """
    Locked unseen analytical planner evaluation case.

    IMPORTANT
    ---------

    Every case is:

        split = test
        frozen = True

    The expected plan is validated by the real deterministic
    planner validator before the benchmark can be accepted.

    The expected plan must NEVER be included in model-visible
    planner input.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str = Field(
        min_length=1,
    )

    split: Literal[
        "test"
    ]

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

    expected: (
        AnalyticalPlannerCandidate
    )

    notes: (
        str
        | None
    ) = None

    frozen: Literal[
        True
    ]


    @model_validator(
        mode="after",
    )
    def validate_frozen_case(
        self,
    ) -> "FrozenAnalyticalPlannerEvalCase":

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
        # DEPENDENCY + STRUCTURAL PIPELINE
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
                "Frozen analytical planner cases must "
                "be structurally executable. "
                f"case_id={self.case_id}, "
                "blocking="
                f"{planner_context.blocking_requirements}"
            )


        # ====================================================
        # EXACT MODEL-VISIBLE INPUT
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
        # EXPECTED PLAN MUST PASS VALIDATOR v0.9.1
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


        if not validation.valid:

            issues = [
                {
                    "code":
                        issue.code,

                    "requirement_id":
                        issue.requirement_id,

                    "step_id":
                        issue.step_id,

                    "message":
                        issue.message,
                }

                for issue
                in validation.issues
            ]


            raise ValueError(
                "Frozen expected planner candidate does "
                "not pass deterministic validation. "
                f"case_id={self.case_id}, "
                f"issues={issues}"
            )


        return self


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_planner_input_for_frozen_case(
    case: FrozenAnalyticalPlannerEvalCase,
) -> AnalyticalPlannerInput:
    """
    Rebuild the exact trusted model-visible planner input.

    Ground truth fields such as:

        expected
        notes

    are deliberately excluded.
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
            "Frozen planner case is not ready "
            "for planning: "
            f"{case.case_id}"
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

def load_frozen_analytical_planner_benchmark(
    path: str | Path,
) -> list[
    FrozenAnalyticalPlannerEvalCase
]:

    benchmark_path = Path(
        path
    )


    if not benchmark_path.exists():

        raise FileNotFoundError(
            "Frozen analytical planner benchmark "
            f"not found: {benchmark_path}"
        )


    cases: list[
        FrozenAnalyticalPlannerEvalCase
    ] = []


    seen_case_ids: set[
        str
    ] = set()


    with benchmark_path.open(
        "r",
        encoding="utf-8",
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

                payload = (
                    json.loads(
                        line
                    )
                )


            except json.JSONDecodeError as error:

                raise ValueError(
                    "Invalid JSON in frozen analytical "
                    "planner benchmark on line "
                    f"{line_number}: {error}"
                ) from error


            case = (
                FrozenAnalyticalPlannerEvalCase
                .model_validate(
                    payload
                )
            )


            if (
                case.case_id
                in seen_case_ids
            ):

                raise ValueError(
                    "Duplicate frozen planner case_id: "
                    f"{case.case_id}"
                )


            seen_case_ids.add(
                case.case_id
            )


            cases.append(
                case
            )


    if not cases:

        raise ValueError(
            "Frozen analytical planner benchmark "
            "contains no cases."
        )


    return cases
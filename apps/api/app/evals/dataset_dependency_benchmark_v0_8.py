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

from app.evals.routing_relationships_v0_8 import (
    CrossDatasetFeasibility,
    DatasetRelationshipSpec,
    RoutingRelationshipContext,
)

from app.evals.schemas import (
    DatasetContext,
)


# ============================================================
# VERSION
# ============================================================

DATASET_DEPENDENCY_FROZEN_BENCHMARK_VERSION = (
    "dataset_dependency_frozen_benchmark_v0.8"
)


# ============================================================
# EXPECTATION
# ============================================================

class DatasetDependencyExpectation(
    BaseModel
):
    """
    Ground truth for both:

    1. semantic dependency extraction;
    2. deterministic structural feasibility.

    expected_groups:
        Datasets required together for each analytical result.

    expected_feasibilities:
        Expected Python feasibility result for each group,
        in the same order.

    executable:
        Whether the complete user request is structurally
        executable.

    routing_override_reason:
        Current router-compatible reason when the dependency
        gate blocks execution.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    expected_groups: list[
        list[str]
    ] = Field(
        min_length=1,
    )

    expected_feasibilities: list[
        CrossDatasetFeasibility
    ] = Field(
        min_length=1,
    )

    executable: bool

    routing_override_reason: (
        Literal[
            "unsupported_analysis"
        ]
        | None
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
    ) -> "DatasetDependencyExpectation":

        if (
            len(
                self.expected_groups
            )
            != len(
                self.expected_feasibilities
            )
        ):
            raise ValueError(
                "expected_groups and "
                "expected_feasibilities must have "
                "the same length."
            )


        for group in self.expected_groups:

            if not group:
                raise ValueError(
                    "Expected dependency groups "
                    "must not be empty."
                )


            if (
                len(
                    group
                )
                != len(
                    set(
                        group
                    )
                )
            ):
                raise ValueError(
                    "A dependency group must not "
                    "contain duplicate datasets."
                )


        blocked = any(
            feasibility
            not in {
                "not_required",
                "supported",
            }

            for feasibility
            in self.expected_feasibilities
        )


        if self.executable and blocked:
            raise ValueError(
                "An executable expectation cannot "
                "contain a blocked feasibility."
            )


        if (
            not self.executable
            and not blocked
        ):
            raise ValueError(
                "A non-executable expectation must "
                "contain at least one blocked "
                "feasibility."
            )


        if self.executable:

            if (
                self.routing_override_reason
                is not None
            ):
                raise ValueError(
                    "Executable cases must not "
                    "have a routing override."
                )

        else:

            if (
                self.routing_override_reason
                != "unsupported_analysis"
            ):
                raise ValueError(
                    "Blocked dependency cases must "
                    "use unsupported_analysis."
                )


        return self


# ============================================================
# CASE
# ============================================================

class DatasetDependencyFrozenCase(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str = Field(
        min_length=1,
    )

    split: Literal[
        "test"
    ] = "test"

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

    expected: DatasetDependencyExpectation

    frozen: Literal[
        True
    ] = True


    @model_validator(
        mode="after",
    )
    def validate_case(
        self,
    ) -> "DatasetDependencyFrozenCase":

        # ====================================================
        # STRUCTURAL CONTEXT MUST ITSELF BE VALID
        # ====================================================

        RoutingRelationshipContext(
            datasets=self.datasets,
            relationships=self.relationships,
            available_tools=self.available_tools,
        )


        # ====================================================
        # EXPECTED DATASET REFERENCES
        # ====================================================

        known_dataset_ids = {
            dataset.dataset_id

            for dataset
            in self.datasets
        }


        expected_dataset_ids = {
            dataset_id

            for group
            in self.expected.expected_groups

            for dataset_id
            in group
        }


        unknown = (
            expected_dataset_ids
            - known_dataset_ids
        )


        if unknown:

            raise ValueError(
                "Expected dependency groups reference "
                "unknown dataset(s): "
                f"{sorted(unknown)}"
            )


        # ====================================================
        # DUPLICATE EXPECTED GROUPS
        # ====================================================

        canonical_groups = [
            tuple(
                sorted(
                    group
                )
            )

            for group
            in self.expected.expected_groups
        ]


        if (
            len(
                canonical_groups
            )
            != len(
                set(
                    canonical_groups
                )
            )
        ):
            raise ValueError(
                "Expected dependency groups "
                "must be unique."
            )


        return self


# ============================================================
# LOADER
# ============================================================

def load_dataset_dependency_frozen_benchmark(
    path: str | Path,
) -> list[
    DatasetDependencyFrozenCase
]:

    benchmark_path = Path(
        path
    )


    cases: list[
        DatasetDependencyFrozenCase
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
                    "Invalid JSON on line "
                    f"{line_number}: {error}"
                ) from error


            case = (
                DatasetDependencyFrozenCase
                .model_validate(
                    payload
                )
            )


            if (
                case.case_id
                in seen_case_ids
            ):
                raise ValueError(
                    "Duplicate case_id: "
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
            "Frozen dependency benchmark "
            "contains no cases."
        )


    return cases
from __future__ import annotations

from enum import Enum
import hashlib
import json
from typing import Any

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
)

from app.preparation.semantic_review import (
    SemanticVerdict,
    ValidatedSemanticDecision,
)


SEMANTIC_CLEANING_RULE_VERSION = (
    "semantic_cleaning_engine_v0.1"
)


class SemanticCleaningActionStatus(
    str,
    Enum,
):
    APPLIED = "applied"
    SKIPPED = "skipped"


class SemanticCleaningAction(
    BaseModel,
):
    action_id: str

    issue_id: str
    dataset_id: str
    dataset_filename: str
    column: str

    source_values: list[str]

    suggested_canonical_value: str

    allowed_canonical_values: list[
        str
    ]

    confidence: float

    rationale: str

    requires_user_confirmation: bool = (
        True
    )

    python_validated: bool = True


class SemanticCleaningPlan(
    BaseModel,
):
    status: str

    action_count: int

    actions: list[
        SemanticCleaningAction
    ]

    notes: list[str]

    rule_version: str = (
        SEMANTIC_CLEANING_RULE_VERSION
    )


class SemanticCleaningChoice(
    BaseModel,
):
    action_id: str
    canonical_value: str


class SemanticCleaningActionResult(
    BaseModel,
):
    action_id: str

    status: (
        SemanticCleaningActionStatus
    )

    dataset_id: str

    column: str

    source_values: list[str]

    canonical_value: str | None

    affected_rows_actual: int

    details: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class SemanticDatasetProvenance(
    BaseModel,
):
    dataset_id: str
    dataset_filename: str

    rows_before: int
    rows_after: int

    source_fingerprint: str
    derived_fingerprint: str

    applied_action_ids: list[str]

    changed_cell_count: int


class SemanticCleaningExecutionResult(
    BaseModel,
):
    status: str

    dataset_count: int

    applied_action_count: int

    skipped_action_count: int

    changed_cell_count: int

    action_results: list[
        SemanticCleaningActionResult
    ]

    provenance: list[
        SemanticDatasetProvenance
    ]

    notes: list[str]

    rule_version: str = (
        SEMANTIC_CLEANING_RULE_VERSION
    )


def _fingerprint_dataframe(
    dataframe: pd.DataFrame,
) -> str:
    normalized = (
        dataframe
        .copy(
            deep=True
        )
        .reset_index(
            drop=True
        )
    )

    payload = {
        "columns": [
            str(
                column
            )
            for column in
            normalized.columns
        ],

        "dtypes": [
            str(
                dtype
            )
            for dtype in
            normalized.dtypes
        ],

        "rows": (
            normalized
            .astype(
                "string"
            )
            .fillna(
                "<NA>"
            )
            .values
            .tolist()
        ),
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def _stable_semantic_action_id(
    *,
    decision: ValidatedSemanticDecision,
) -> str:
    """
    Stable across requests for the same dataset order,
    issue, column and exact alias values.

    The suggested canonical value is intentionally NOT part
    of the ID because the user may choose another exact
    existing alias as canonical.
    """

    payload = {
        "issue_id":
            decision.issue_id,

        "dataset_id":
            decision.dataset_id,

        "column":
            decision.column,

        "source_values":
            sorted(
                decision
                .source_values
            ),
    }

    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )
    ).hexdigest()[
        :16
    ]

    return (
        "semantic:"
        f"{decision.dataset_id}:"
        f"{digest}"
    )


def build_semantic_cleaning_plan(
    decisions: list[
        ValidatedSemanticDecision
    ],
) -> SemanticCleaningPlan:
    """
    Convert Python-validated LLM merge proposals into
    non-executable user-confirmation actions.

    Only validated merge_values decisions are eligible.
    """

    actions: list[
        SemanticCleaningAction
    ] = []

    seen_action_ids: set[
        str
    ] = set()

    for decision in decisions:
        if (
            decision.verdict
            !=
            SemanticVerdict.MERGE_VALUES
        ):
            continue

        if (
            decision.python_validated
            is not True
        ):
            continue

        if (
            decision.column
            is None
        ):
            continue

        source_values = list(
            dict.fromkeys(
                decision.source_values
            )
        )

        if (
            len(
                source_values
            )
            <
            2
        ):
            raise ValueError(
                "Validated semantic merge "
                "must contain at least "
                "two source values."
            )

        canonical_value = (
            decision.canonical_value
        )

        if (
            canonical_value
            is None
            or
            canonical_value
            not in source_values
        ):
            raise ValueError(
                "Validated semantic merge "
                "must use one exact source "
                "value as canonical value."
            )

        action_id = (
            _stable_semantic_action_id(
                decision=
                    decision
            )
        )

        if (
            action_id
            in seen_action_ids
        ):
            raise ValueError(
                "Duplicate semantic cleaning "
                f"action_id generated: {action_id}"
            )

        seen_action_ids.add(
            action_id
        )

        actions.append(
            SemanticCleaningAction(
                action_id=
                    action_id,

                issue_id=
                    decision.issue_id,

                dataset_id=
                    decision.dataset_id,

                dataset_filename=
                    decision
                    .dataset_filename,

                column=
                    decision.column,

                source_values=
                    source_values,

                suggested_canonical_value=
                    canonical_value,

                allowed_canonical_values=
                    source_values,

                confidence=
                    decision.confidence,

                rationale=
                    decision.rationale,

                requires_user_confirmation=
                    True,

                python_validated=
                    True,
            )
        )

    return (
        SemanticCleaningPlan(
            status=
                "ready",

            action_count=
                len(
                    actions
                ),

            actions=
                actions,

            notes=[
                (
                    "Only Python-validated "
                    "merge_values proposals are "
                    "eligible for semantic cleaning."
                ),
                (
                    "The user may choose only an "
                    "exact existing alias value as "
                    "the canonical value."
                ),
                (
                    "No action is executed while "
                    "building this plan."
                ),
            ],
        )
    )


def _exact_string_mask(
    series: pd.Series,
    value: str,
) -> pd.Series:
    """
    Match exact string cells only.

    We intentionally avoid astype(str), which could turn
    non-string cells into accidental textual matches.
    """

    return (
        series.map(
            lambda cell: (
                isinstance(
                    cell,
                    str,
                )
                and
                cell
                ==
                value
            )
        )
        .fillna(
            False
        )
        .astype(
            bool
        )
    )


def execute_semantic_cleaning_plan(
    *,
    plan: SemanticCleaningPlan,

    dataset_frames: dict[
        str,
        pd.DataFrame,
    ],

    approved_choices: list[
        SemanticCleaningChoice
    ],
) -> tuple[
    dict[
        str,
        pd.DataFrame,
    ],
    SemanticCleaningExecutionResult,
]:
    """
    Apply only user-confirmed semantic mappings to deep copies.

    The model never executes a transformation.
    The original DataFrames are never mutated.
    """

    action_map = {
        action.action_id:
            action
        for action in
        plan.actions
    }

    choice_map: dict[
        str,
        SemanticCleaningChoice
    ] = {}

    for choice in (
        approved_choices
    ):
        if (
            choice.action_id
            in choice_map
        ):
            raise ValueError(
                "Duplicate semantic cleaning "
                f"choice: {choice.action_id}"
            )

        if (
            choice.action_id
            not in action_map
        ):
            raise ValueError(
                "Unknown semantic cleaning "
                f"action_id: {choice.action_id}"
            )

        action = (
            action_map[
                choice.action_id
            ]
        )

        if (
            choice.canonical_value
            not in
            action
            .allowed_canonical_values
        ):
            raise ValueError(
                "Invalid canonical value "
                f"'{choice.canonical_value}' "
                f"for semantic action "
                f"'{choice.action_id}'."
            )

        choice_map[
            choice.action_id
        ] = choice

    source_fingerprints: dict[
        str,
        str,
    ] = {}

    derived_frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    rows_before: dict[
        str,
        int,
    ] = {}

    for (
        dataset_id,
        dataframe,
    ) in dataset_frames.items():
        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Semantic cleaning received "
                "a non-DataFrame dataset for "
                f"{dataset_id}."
            )

        source_fingerprints[
            dataset_id
        ] = (
            _fingerprint_dataframe(
                dataframe
            )
        )

        rows_before[
            dataset_id
        ] = int(
            dataframe.shape[
                0
            ]
        )

        derived_frames[
            dataset_id
        ] = dataframe.copy(
            deep=True
        )

    action_results: list[
        SemanticCleaningActionResult
    ] = []

    changed_cells_by_dataset: dict[
        str,
        int,
    ] = {
        dataset_id:
            0
        for dataset_id in
        dataset_frames
    }

    applied_action_ids_by_dataset: dict[
        str,
        list[
            str
        ],
    ] = {
        dataset_id:
            []
        for dataset_id in
        dataset_frames
    }

    for action in (
        plan.actions
    ):
        choice = (
            choice_map.get(
                action.action_id
            )
        )

        if choice is None:
            action_results.append(
                SemanticCleaningActionResult(
                    action_id=
                        action.action_id,

                    status=
                        SemanticCleaningActionStatus
                        .SKIPPED,

                    dataset_id=
                        action.dataset_id,

                    column=
                        action.column,

                    source_values=
                        action.source_values,

                    canonical_value=
                        None,

                    affected_rows_actual=
                        0,

                    details={
                        "reason":
                            (
                                "User did not approve "
                                "this semantic merge."
                            ),
                    },
                )
            )

            continue

        dataframe = (
            derived_frames.get(
                action.dataset_id
            )
        )

        if dataframe is None:
            raise KeyError(
                "Semantic cleaning plan "
                "references unknown dataset_id "
                f"{action.dataset_id}."
            )

        if (
            action.column
            not in dataframe.columns
        ):
            raise KeyError(
                "Semantic cleaning plan "
                "references unknown column "
                f"'{action.column}' in "
                f"{action.dataset_id}."
            )

        canonical_value = (
            choice.canonical_value
        )

        changed_mask = pd.Series(
            False,
            index=dataframe.index,
            dtype=bool,
        )

        per_value_counts: dict[
            str,
            int,
        ] = {}

        for source_value in (
            action.source_values
        ):
            if (
                source_value
                ==
                canonical_value
            ):
                continue

            mask = (
                _exact_string_mask(
                    dataframe[
                        action.column
                    ],
                    source_value,
                )
            )

            count = int(
                mask.sum()
            )

            per_value_counts[
                source_value
            ] = count

            if count <= 0:
                continue

            dataframe.loc[
                mask,
                action.column,
            ] = canonical_value

            changed_mask = (
                changed_mask
                |
                mask
            )

        changed_rows = int(
            changed_mask.sum()
        )

        changed_cells_by_dataset[
            action.dataset_id
        ] += changed_rows

        applied_action_ids_by_dataset[
            action.dataset_id
        ].append(
            action.action_id
        )

        action_results.append(
            SemanticCleaningActionResult(
                action_id=
                    action.action_id,

                status=
                    SemanticCleaningActionStatus
                    .APPLIED,

                dataset_id=
                    action.dataset_id,

                column=
                    action.column,

                source_values=
                    action.source_values,

                canonical_value=
                    canonical_value,

                affected_rows_actual=
                    changed_rows,

                details={
                    "replacement_counts":
                        per_value_counts,
                },
            )
        )

    provenance: list[
        SemanticDatasetProvenance
    ] = []

    filename_map = {
        action.dataset_id:
            action.dataset_filename
        for action in
        plan.actions
    }

    for (
        dataset_id,
        derived_dataframe,
    ) in derived_frames.items():
        rows_after = int(
            derived_dataframe
            .shape[
                0
            ]
        )

        if (
            rows_after
            !=
            rows_before[
                dataset_id
            ]
        ):
            raise RuntimeError(
                "Semantic cleaning must not "
                "change row count."
            )

        provenance.append(
            SemanticDatasetProvenance(
                dataset_id=
                    dataset_id,

                dataset_filename=
                    filename_map.get(
                        dataset_id,
                        dataset_id,
                    ),

                rows_before=
                    rows_before[
                        dataset_id
                    ],

                rows_after=
                    rows_after,

                source_fingerprint=
                    source_fingerprints[
                        dataset_id
                    ],

                derived_fingerprint=
                    _fingerprint_dataframe(
                        derived_dataframe
                    ),

                applied_action_ids=
                    applied_action_ids_by_dataset[
                        dataset_id
                    ],

                changed_cell_count=
                    changed_cells_by_dataset[
                        dataset_id
                    ],
            )
        )

    applied_action_count = sum(
        result.status
        ==
        SemanticCleaningActionStatus
        .APPLIED
        for result in
        action_results
    )

    skipped_action_count = sum(
        result.status
        ==
        SemanticCleaningActionStatus
        .SKIPPED
        for result in
        action_results
    )

    changed_cell_count = sum(
        changed_cells_by_dataset
        .values()
    )

    return (
        derived_frames,

        SemanticCleaningExecutionResult(
            status=
                "completed",

            dataset_count=
                len(
                    derived_frames
                ),

            applied_action_count=
                applied_action_count,

            skipped_action_count=
                skipped_action_count,

            changed_cell_count=
                changed_cell_count,

            action_results=
                action_results,

            provenance=
                provenance,

            notes=[
                (
                    "Semantic mappings were "
                    "executed only after explicit "
                    "user choices."
                ),
                (
                    "Only exact existing source "
                    "values were replaced."
                ),
                (
                    "Source DataFrames were "
                    "preserved unchanged."
                ),
                (
                    "Semantic cleaning does not "
                    "add or remove rows."
                ),
            ],
        ),
    )

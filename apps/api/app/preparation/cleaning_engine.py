from __future__ import annotations

from enum import Enum
import hashlib
import json
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from app.preparation.data_quality import (
    CleaningOperation,
    DataQualityReport,
    QualityIssueKind,
)


CLEANING_ENGINE_RULE_VERSION = (
    "cleaning_engine_v0.1"
)


class CleaningActionKind(
    str,
    Enum,
):
    DROP_EXACT_DUPLICATES = (
        "drop_exact_duplicates"
    )

    NORMALIZE_CATEGORY_VARIANTS = (
        "normalize_category_variants"
    )

    NORMALIZE_DATE_FORMATS = (
        "normalize_date_formats"
    )


class CleaningActionStatus(
    str,
    Enum,
):
    APPLIED = "applied"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class CleaningAction(
    BaseModel,
):
    action_id: str
    dataset_id: str
    dataset_filename: str
    kind: CleaningActionKind
    column: str | None = None

    title: str
    rationale: str

    safe_candidate: bool
    requires_user_confirmation: bool

    affected_rows_estimate: int

    before_examples: list[str] = Field(
        default_factory=list,
    )

    after_examples: list[str] = Field(
        default_factory=list,
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
    )


class CleaningPlan(
    BaseModel,
):
    status: str
    dataset_count: int
    action_count: int
    safe_candidate_count: int
    confirmation_required_count: int
    protected_issue_count: int
    actions: list[CleaningAction]
    notes: list[str]
    rule_version: str = (
        CLEANING_ENGINE_RULE_VERSION
    )


class CleaningActionResult(
    BaseModel,
):
    action_id: str
    status: CleaningActionStatus
    affected_rows_actual: int
    rows_before: int
    rows_after: int
    details: dict[str, Any] = Field(
        default_factory=dict,
    )


class DatasetCleaningProvenance(
    BaseModel,
):
    dataset_id: str
    dataset_filename: str

    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int

    source_fingerprint: str
    derived_fingerprint: str

    applied_action_ids: list[str]
    skipped_action_ids: list[str]


class CleaningExecutionResult(
    BaseModel,
):
    status: str
    dataset_count: int
    applied_action_count: int
    skipped_action_count: int
    blocked_action_count: int

    action_results: list[
        CleaningActionResult
    ]

    provenance: list[
        DatasetCleaningProvenance
    ]

    notes: list[str]

    rule_version: str = (
        CLEANING_ENGINE_RULE_VERSION
    )


def _stable_action_id(
    *,
    dataset_id: str,
    kind: CleaningActionKind,
    column: str | None,
) -> str:
    location = (
        column
        if column
        else "__dataset__"
    )

    return (
        f"{dataset_id}:"
        f"{location}:"
        f"{kind.value}"
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


def _clean_display_variant(
    value: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        value.strip(),
    )


def _choose_canonical_variant(
    series: pd.Series,
    variants: list[str],
) -> str:
    variant_set = set(
        variants
    )

    cleaned_values = [
        _clean_display_variant(
            value
        )
        for value in
        series
        .dropna()
        .astype(
            str
        )
        if value in
        variant_set
    ]

    if not cleaned_values:
        return _clean_display_variant(
            variants[0]
        )

    counts: dict[
        str,
        int,
    ] = {}

    for value in cleaned_values:
        counts[
            value
        ] = (
            counts.get(
                value,
                0,
            )
            +
            1
        )

    return sorted(
        counts,
        key=lambda value: (
            -counts[
                value
            ],
            value.casefold(),
            value,
        ),
    )[0]


def _parse_date_to_iso(
    value: Any,
) -> str | None:
    if pd.isna(
        value
    ):
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    year_first = bool(
        re.match(
            r"^\d{4}[-/]",
            text,
        )
    )

    try:
        parsed = pd.to_datetime(
            text,
            errors="coerce",
            dayfirst=(
                not year_first
            ),
        )
    except (
        ValueError,
        TypeError,
        OverflowError,
    ):
        return None

    if pd.isna(
        parsed
    ):
        return None

    return pd.Timestamp(
        parsed
    ).date().isoformat()


def build_cleaning_plan(
    quality_report: DataQualityReport,
    dataset_frames: dict[
        str,
        pd.DataFrame,
    ],
) -> CleaningPlan:
    """
    Convert the deterministic quality report into a guarded
    cleaning plan.

    v0.1 intentionally proposes only three data-changing
    operations:
    - removal of exact duplicate copies;
    - normalization of deterministic category formatting
      variants (case / whitespace only);
    - standardization of already valid dates to ISO format.

    Missing values, semantic aliases, outliers, invalid dates,
    invalid e-mails, invalid numeric values and numeric decoration
    remain protected. They are not repaired by this engine.
    """

    actions: list[
        CleaningAction
    ] = []

    protected_issue_count = 0

    for issue in quality_report.issues:
        dataframe = dataset_frames.get(
            issue.dataset_id
        )

        if dataframe is None:
            raise KeyError(
                "Missing dataframe for dataset_id "
                f"{issue.dataset_id}."
            )

        if (
            issue.kind
            ==
            QualityIssueKind.DUPLICATE_ROWS
            and
            issue.proposal.operation
            ==
            CleaningOperation.DROP_EXACT_DUPLICATES
            and
            issue.proposal.automatic_safe
        ):
            duplicate_count = int(
                dataframe.duplicated(
                    keep="first",
                ).sum()
            )

            if duplicate_count > 0:
                actions.append(
                    CleaningAction(
                        action_id=
                            _stable_action_id(
                                dataset_id=
                                    issue.dataset_id,
                                kind=
                                    CleaningActionKind.DROP_EXACT_DUPLICATES,
                                column=
                                    None,
                            ),
                        dataset_id=
                            issue.dataset_id,
                        dataset_filename=
                            issue.dataset_filename,
                        kind=
                            CleaningActionKind.DROP_EXACT_DUPLICATES,
                        column=
                            None,
                        title=
                            "Supprimer les doublons stricts",
                        rationale=(
                            "Les copies sont strictement identiques. "
                            "La source reste intacte ; seules les copies "
                            "du dataset dérivé sont supprimées."
                        ),
                        safe_candidate=
                            True,
                        requires_user_confirmation=
                            True,
                        affected_rows_estimate=
                            duplicate_count,
                        before_examples=
                            issue.evidence.examples[
                                :3
                            ],
                        after_examples=[
                            (
                                f"{duplicate_count} copie(s) "
                                "retirée(s) du dataset dérivé"
                            )
                        ],
                        parameters={
                            "keep":
                                "first",
                        },
                    )
                )

            continue

        if (
            issue.kind
            ==
            QualityIssueKind.CATEGORY_FORMAT_VARIANTS
            and
            issue.proposal.operation
            in {
                CleaningOperation.NORMALIZE_WHITESPACE,
                CleaningOperation.NORMALIZE_CASE,
            }
            and
            issue.proposal.automatic_safe
        ):
            if not issue.column:
                protected_issue_count += 1
                continue

            raw_groups = (
                issue.evidence.details.get(
                    "variant_groups"
                )
                or
                issue.proposal.parameters.get(
                    "groups"
                )
                or
                {}
            )

            if not isinstance(
                raw_groups,
                dict,
            ):
                protected_issue_count += 1
                continue

            series = dataframe[
                issue.column
            ]

            mapping: dict[
                str,
                str,
            ] = {}

            before_examples: list[
                str
            ] = []

            after_examples: list[
                str
            ] = []

            for raw_values in raw_groups.values():
                if not isinstance(
                    raw_values,
                    list,
                ):
                    continue

                variants = [
                    str(
                        value
                    )
                    for value in
                    raw_values
                ]

                if len(
                    variants
                ) < 2:
                    continue

                canonical = (
                    _choose_canonical_variant(
                        series,
                        variants,
                    )
                )

                before_examples.append(
                    " / ".join(
                        variants
                    )
                )

                after_examples.append(
                    canonical
                )

                for variant in variants:
                    if (
                        variant
                        ==
                        canonical
                    ):
                        continue

                    mapping[
                        variant
                    ] = canonical

            if not mapping:
                continue

            affected_rows = int(
                series.isin(
                    list(
                        mapping.keys()
                    )
                ).sum()
            )

            actions.append(
                CleaningAction(
                    action_id=
                        _stable_action_id(
                            dataset_id=
                                issue.dataset_id,
                            kind=
                                CleaningActionKind.NORMALIZE_CATEGORY_VARIANTS,
                            column=
                                issue.column,
                        ),
                    dataset_id=
                        issue.dataset_id,
                    dataset_filename=
                        issue.dataset_filename,
                    kind=
                        CleaningActionKind.NORMALIZE_CATEGORY_VARIANTS,
                    column=
                        issue.column,
                    title=
                        "Normaliser les variantes de catégorie",
                    rationale=(
                        "Les valeurs diffèrent seulement par la "
                        "casse ou les espaces après la vérification "
                        "déterministe du moteur qualité. Aucun alias "
                        "sémantique n'est fusionné."
                    ),
                    safe_candidate=
                        True,
                    requires_user_confirmation=
                        True,
                    affected_rows_estimate=
                        affected_rows,
                    before_examples=
                        before_examples[
                            :5
                        ],
                    after_examples=
                        after_examples[
                            :5
                        ],
                    parameters={
                        "mapping":
                            mapping,
                    },
                )
            )

            continue

        if (
            issue.kind
            ==
            QualityIssueKind.MIXED_DATE_FORMATS
            and
            issue.proposal.operation
            ==
            CleaningOperation.PARSE_DATETIME
            and
            issue.proposal.automatic_safe
        ):
            if not issue.column:
                protected_issue_count += 1
                continue

            series = dataframe[
                issue.column
            ]

            mapping: dict[
                str,
                str,
            ] = {}

            affected_rows = 0

            for raw_value in (
                series
                .dropna()
                .astype(
                    str
                )
            ):
                normalized = (
                    _parse_date_to_iso(
                        raw_value
                    )
                )

                if normalized is None:
                    continue

                if (
                    raw_value.strip()
                    ==
                    normalized
                ):
                    continue

                mapping[
                    raw_value
                ] = normalized

                affected_rows += 1

            if not mapping:
                continue

            actions.append(
                CleaningAction(
                    action_id=
                        _stable_action_id(
                            dataset_id=
                                issue.dataset_id,
                            kind=
                                CleaningActionKind.NORMALIZE_DATE_FORMATS,
                            column=
                                issue.column,
                        ),
                    dataset_id=
                        issue.dataset_id,
                    dataset_filename=
                        issue.dataset_filename,
                    kind=
                        CleaningActionKind.NORMALIZE_DATE_FORMATS,
                    column=
                        issue.column,
                    title=
                        "Uniformiser les dates valides",
                    rationale=(
                        "Seules les dates déjà valides et parseables "
                        "sont converties vers YYYY-MM-DD. Les valeurs "
                        "invalides restent exactement telles quelles."
                    ),
                    safe_candidate=
                        True,
                    requires_user_confirmation=
                        True,
                    affected_rows_estimate=
                        affected_rows,
                    before_examples=
                        list(
                            mapping.keys()
                        )[
                            :5
                        ],
                    after_examples=
                        list(
                            mapping.values()
                        )[
                            :5
                        ],
                    parameters={
                        "mapping":
                            mapping,
                        "format":
                            "YYYY-MM-DD",
                    },
                )
            )

            continue

        protected_issue_count += 1

    safe_candidate_count = sum(
        action.safe_candidate
        for action in
        actions
    )

    confirmation_required_count = sum(
        action.requires_user_confirmation
        for action in
        actions
    )

    return CleaningPlan(
        status=
            "ready",
        dataset_count=
            quality_report.dataset_count,
        action_count=
            len(
                actions
            ),
        safe_candidate_count=
            safe_candidate_count,
        confirmation_required_count=
            confirmation_required_count,
        protected_issue_count=
            protected_issue_count,
        actions=
            actions,
        notes=[
            (
                "The v0.1 plan is deterministic and never calls an LLM."
            ),
            (
                "All source DataFrames remain immutable; cleaning creates derived copies."
            ),
            (
                "Missing values, semantic aliases, outliers, invalid dates, invalid e-mails and invalid numeric values remain protected."
            ),
            (
                "Every data-changing action requires explicit user confirmation."
            ),
        ],
    )


def _apply_exact_duplicates(
    dataframe: pd.DataFrame,
) -> int:
    duplicate_mask = dataframe.duplicated(
        keep="first",
    )

    affected_rows = int(
        duplicate_mask.sum()
    )

    if affected_rows <= 0:
        return 0

    dataframe.drop(
        index=
            dataframe.index[
                duplicate_mask
            ],
        inplace=True,
    )

    dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    return affected_rows


def _apply_string_mapping(
    dataframe: pd.DataFrame,
    *,
    column: str,
    mapping: dict[
        str,
        str,
    ],
) -> int:
    if column not in dataframe.columns:
        raise KeyError(
            f"Unknown column: {column}"
        )

    series = dataframe[
        column
    ]

    string_series = series.astype(
        "string"
    )

    affected_mask = string_series.isin(
        list(
            mapping.keys()
        )
    )

    affected_rows = int(
        affected_mask.sum()
    )

    if affected_rows <= 0:
        return 0

    updated = series.copy()

    for source_value, target_value in mapping.items():
        mask = (
            string_series
            ==
            source_value
        )

        updated.loc[
            mask
        ] = target_value

    dataframe[
        column
    ] = updated

    return affected_rows


def execute_cleaning_plan(
    *,
    plan: CleaningPlan,
    dataset_frames: dict[
        str,
        pd.DataFrame,
    ],
    approved_action_ids: set[
        str
    ],
) -> tuple[
    dict[
        str,
        pd.DataFrame,
    ],
    CleaningExecutionResult,
]:
    """
    Execute only explicitly approved, safe, deterministic actions.

    Guardrails:
    - unknown action IDs are rejected;
    - source DataFrames are never mutated;
    - dataset and column bindings are revalidated;
    - unsafe actions are blocked;
    - every action produces provenance.
    """

    known_action_ids = {
        action.action_id
        for action in
        plan.actions
    }

    unknown_ids = (
        approved_action_ids
        -
        known_action_ids
    )

    if unknown_ids:
        raise ValueError(
            "Unknown cleaning action id(s): "
            +
            ", ".join(
                sorted(
                    unknown_ids
                )
            )
        )

    derived_frames = {
        dataset_id:
            dataframe.copy(
                deep=True
            )
        for (
            dataset_id,
            dataframe,
        ) in dataset_frames.items()
    }

    source_fingerprints = {
        dataset_id:
            _fingerprint_dataframe(
                dataframe
            )
        for (
            dataset_id,
            dataframe,
        ) in dataset_frames.items()
    }

    applied_by_dataset: dict[
        str,
        list[str],
    ] = {
        dataset_id:
            []
        for dataset_id in
        dataset_frames
    }

    skipped_by_dataset: dict[
        str,
        list[str],
    ] = {
        dataset_id:
            []
        for dataset_id in
        dataset_frames
    }

    action_results: list[
        CleaningActionResult
    ] = []

    blocked_action_count = 0

    for action in plan.actions:
        dataframe = derived_frames.get(
            action.dataset_id
        )

        if dataframe is None:
            raise KeyError(
                "Missing dataframe for dataset_id "
                f"{action.dataset_id}."
            )

        rows_before = len(
            dataframe
        )

        if (
            action.action_id
            not in
            approved_action_ids
        ):
            skipped_by_dataset[
                action.dataset_id
            ].append(
                action.action_id
            )

            action_results.append(
                CleaningActionResult(
                    action_id=
                        action.action_id,
                    status=
                        CleaningActionStatus.SKIPPED,
                    affected_rows_actual=
                        0,
                    rows_before=
                        rows_before,
                    rows_after=
                        rows_before,
                    details={
                        "reason":
                            "not_approved",
                    },
                )
            )

            continue

        if not action.safe_candidate:
            blocked_action_count += 1

            action_results.append(
                CleaningActionResult(
                    action_id=
                        action.action_id,
                    status=
                        CleaningActionStatus.BLOCKED,
                    affected_rows_actual=
                        0,
                    rows_before=
                        rows_before,
                    rows_after=
                        rows_before,
                    details={
                        "reason":
                            "unsafe_action",
                    },
                )
            )

            continue

        if (
            action.kind
            ==
            CleaningActionKind.DROP_EXACT_DUPLICATES
        ):
            affected_rows = (
                _apply_exact_duplicates(
                    dataframe
                )
            )

        elif (
            action.kind
            in {
                CleaningActionKind.NORMALIZE_CATEGORY_VARIANTS,
                CleaningActionKind.NORMALIZE_DATE_FORMATS,
            }
        ):
            if not action.column:
                raise ValueError(
                    f"{action.kind.value} requires a column."
                )

            raw_mapping = action.parameters.get(
                "mapping"
            )

            if not isinstance(
                raw_mapping,
                dict,
            ):
                raise TypeError(
                    f"{action.kind.value} mapping must be a dict."
                )

            mapping = {
                str(
                    source_value
                ):
                    str(
                        target_value
                    )
                for (
                    source_value,
                    target_value,
                ) in raw_mapping.items()
            }

            affected_rows = (
                _apply_string_mapping(
                    dataframe,
                    column=
                        action.column,
                    mapping=
                        mapping,
                )
            )

        else:
            blocked_action_count += 1

            action_results.append(
                CleaningActionResult(
                    action_id=
                        action.action_id,
                    status=
                        CleaningActionStatus.BLOCKED,
                    affected_rows_actual=
                        0,
                    rows_before=
                        rows_before,
                    rows_after=
                        rows_before,
                    details={
                        "reason":
                            "unsupported_action_kind",
                    },
                )
            )

            continue

        rows_after = len(
            dataframe
        )

        applied_by_dataset[
            action.dataset_id
        ].append(
            action.action_id
        )

        action_results.append(
            CleaningActionResult(
                action_id=
                    action.action_id,
                status=
                    CleaningActionStatus.APPLIED,
                affected_rows_actual=
                    affected_rows,
                rows_before=
                    rows_before,
                rows_after=
                    rows_after,
                details={
                    "kind":
                        action.kind.value,
                    "column":
                        action.column,
                },
            )
        )

    provenance: list[
        DatasetCleaningProvenance
    ] = []

    filename_by_dataset = {
        action.dataset_id:
            action.dataset_filename
        for action in
        plan.actions
    }

    for dataset_id, source in dataset_frames.items():
        derived = derived_frames[
            dataset_id
        ]

        provenance.append(
            DatasetCleaningProvenance(
                dataset_id=
                    dataset_id,
                dataset_filename=
                    filename_by_dataset.get(
                        dataset_id,
                        dataset_id,
                    ),
                rows_before=
                    len(
                        source
                    ),
                rows_after=
                    len(
                        derived
                    ),
                columns_before=
                    int(
                        source.shape[
                            1
                        ]
                    ),
                columns_after=
                    int(
                        derived.shape[
                            1
                        ]
                    ),
                source_fingerprint=
                    source_fingerprints[
                        dataset_id
                    ],
                derived_fingerprint=
                    _fingerprint_dataframe(
                        derived
                    ),
                applied_action_ids=
                    applied_by_dataset[
                        dataset_id
                    ],
                skipped_action_ids=
                    skipped_by_dataset[
                        dataset_id
                    ],
            )
        )

    applied_action_count = sum(
        result.status
        ==
        CleaningActionStatus.APPLIED
        for result in
        action_results
    )

    skipped_action_count = sum(
        result.status
        ==
        CleaningActionStatus.SKIPPED
        for result in
        action_results
    )

    return (
        derived_frames,
        CleaningExecutionResult(
            status=
                "ready",
            dataset_count=
                len(
                    derived_frames
                ),
            applied_action_count=
                applied_action_count,
            skipped_action_count=
                skipped_action_count,
            blocked_action_count=
                blocked_action_count,
            action_results=
                action_results,
            provenance=
                provenance,
            notes=[
                (
                    "Source DataFrames were not mutated."
                ),
                (
                    "Only explicitly approved safe candidates were applied."
                ),
                (
                    "Protected anomalies remain unchanged in v0.1."
                ),
            ],
        ),
    )

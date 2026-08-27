from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# VERSION
# ============================================================


DATASET_IDENTITY_RULE_VERSION = "dataset_identity_v0.2"


# ============================================================
# TYPES
# ============================================================


IdentityCandidateKind = Literal[
    "single",
    "composite",
]

IdentityStatus = Literal[
    "single_key",
    "composite_key",
    "surrogate_recommended",
]


# ============================================================
# STRUCTURED OUTPUTS
# ============================================================


class DatasetIdentityCandidate(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    columns: list[str] = Field(
        min_length=1
    )

    kind: IdentityCandidateKind
    row_count: int
    unique_count: int
    missing_row_count: int
    uniqueness_ratio: float
    complete: bool
    unique: bool
    identifier_name_signal: bool
    deterministic_score: float
    rationale: list[str]


class DatasetIdentityReport(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    dataset_id: str
    dataset_filename: str
    row_count: int
    column_count: int
    status: IdentityStatus
    preferred_candidate: DatasetIdentityCandidate | None
    candidates: list[DatasetIdentityCandidate]
    mechanically_unique_columns: list[str]
    identifier_like_columns: list[str]
    surrogate_key_recommended: bool
    suggested_surrogate_column: str | None
    reasons: list[str]
    rule_version: str = DATASET_IDENTITY_RULE_VERSION


@dataclass(frozen=True)
class SurrogateKeyTransformation:
    dataframe: pd.DataFrame
    column_name: str
    rows: int
    rule_version: str = DATASET_IDENTITY_RULE_VERSION


# ============================================================
# IDENTIFIER NAME SIGNALS
# ============================================================


EXACT_IDENTIFIER_NAMES = {
    "id",
    "uuid",
    "guid",
    "key",
}


IDENTIFIER_PREFIXES = (
    "id_",
)


IDENTIFIER_SUFFIXES = (
    "_id",
    "_uuid",
    "_guid",
    "_key",
    "_code",
    "_number",
    "_num",
    "_ref",
    "_reference",
)


TECHNICAL_SURROGATE_NAMES = {
    "row_id",
    "datalens_row_id",
    "technical_row_id",
}


def normalize_column_name(
    column_name: str,
) -> str:
    return (
        re.sub(
            r"[^a-z0-9]+",
            "_",
            str(column_name)
            .strip()
            .lower(),
        )
        .strip("_")
    )


def has_identifier_name_signal(
    column_name: str,
) -> bool:
    normalized = normalize_column_name(
        column_name
    )

    if normalized in EXACT_IDENTIFIER_NAMES:
        return True

    if any(
        normalized.startswith(
            prefix
        )
        for prefix in IDENTIFIER_PREFIXES
    ):
        return True

    return any(
        normalized.endswith(
            suffix
        )
        for suffix in IDENTIFIER_SUFFIXES
    )


def is_technical_surrogate_column(
    column_name: object,
) -> bool:
    """
    Return True only for technical row identifiers that DataLens
    itself can create during Preparation.

    A surrogate row identity may legitimately identify one row
    inside one prepared artifact, but it never carries semantic
    evidence of a relationship with another dataset.
    """

    normalized = normalize_column_name(
        str(
            column_name
        )
    )

    if normalized in TECHNICAL_SURROGATE_NAMES:
        return True

    return bool(
        re.fullmatch(
            r"datalens_row_id_[0-9]+",
            normalized,
        )
    )


# ============================================================
# LOW-LEVEL MEASURES
# ============================================================


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return (
        float(numerator)
        /
        float(denominator)
    )


def _single_candidate(
    dataframe: pd.DataFrame,
    column_name: str,
) -> DatasetIdentityCandidate:
    row_count = int(
        len(dataframe)
    )

    series = dataframe[
        column_name
    ]

    missing_row_count = int(
        series
        .isna()
        .sum()
    )

    unique_count = int(
        series
        .nunique(
            dropna=True
        )
    )

    complete = (
        row_count > 0
        and
        missing_row_count == 0
    )

    unique = (
        complete
        and
        unique_count == row_count
    )

    uniqueness_ratio = _safe_ratio(
        unique_count,
        row_count,
    )

    identifier_signal = (
        has_identifier_name_signal(
            column_name
        )
    )

    score = 0.0

    if unique:
        score += 0.65

    if complete:
        score += 0.15

    if identifier_signal:
        score += 0.20

    rationale: list[str] = []

    if unique:
        rationale.append(
            "The column is complete and unique across all rows."
        )
    else:
        rationale.append(
            (
                f"The column has {unique_count} distinct non-null "
                f"value(s) across {row_count} row(s)."
            )
        )

    if missing_row_count > 0:
        rationale.append(
            (
                f"{missing_row_count} row(s) contain a missing "
                "value in this column."
            )
        )

    if identifier_signal:
        rationale.append(
            "The column name contains an identifier-like signal."
        )
    else:
        rationale.append(
            (
                "The column name does not contain a conservative "
                "identifier-like signal."
            )
        )

    return DatasetIdentityCandidate(
        columns=[
            str(column_name)
        ],
        kind="single",
        row_count=row_count,
        unique_count=unique_count,
        missing_row_count=missing_row_count,
        uniqueness_ratio=uniqueness_ratio,
        complete=complete,
        unique=unique,
        identifier_name_signal=identifier_signal,
        deterministic_score=round(
            min(
                score,
                1.0,
            ),
            4,
        ),
        rationale=rationale,
    )


def _composite_candidate(
    dataframe: pd.DataFrame,
    column_names: tuple[str, ...],
) -> DatasetIdentityCandidate:
    row_count = int(
        len(dataframe)
    )

    subset = dataframe[
        list(column_names)
    ]

    missing_mask = (
        subset
        .isna()
        .any(
            axis=1
        )
    )

    missing_row_count = int(
        missing_mask.sum()
    )

    non_missing_subset = (
        subset.loc[
            ~missing_mask
        ]
    )

    unique_count = int(
        non_missing_subset
        .drop_duplicates()
        .shape[0]
    )

    complete = (
        row_count > 0
        and
        missing_row_count == 0
    )

    unique = (
        complete
        and
        unique_count == row_count
    )

    uniqueness_ratio = _safe_ratio(
        unique_count,
        row_count,
    )

    identifier_signal = all(
        has_identifier_name_signal(
            column_name
        )
        for column_name in column_names
    )

    score = 0.0

    if unique:
        score += 0.60

    if complete:
        score += 0.15

    if identifier_signal:
        score += 0.20

    # Prefer a single reliable key when one exists.
    score -= 0.05

    rationale: list[str] = [
        (
            "The candidate combines "
            + ", ".join(
                column_names
            )
            + "."
        )
    ]

    if unique:
        rationale.append(
            "The combined values are complete and unique across all rows."
        )
    else:
        rationale.append(
            (
                f"The combination yields {unique_count} distinct "
                f"non-null tuple(s) across {row_count} row(s)."
            )
        )

    if missing_row_count > 0:
        rationale.append(
            (
                f"{missing_row_count} row(s) contain a missing "
                "value in at least one component."
            )
        )

    if identifier_signal:
        rationale.append(
            (
                "Every component column contains a conservative "
                "identifier-like name signal."
            )
        )

    return DatasetIdentityCandidate(
        columns=[
            str(column_name)
            for column_name in column_names
        ],
        kind="composite",
        row_count=row_count,
        unique_count=unique_count,
        missing_row_count=missing_row_count,
        uniqueness_ratio=uniqueness_ratio,
        complete=complete,
        unique=unique,
        identifier_name_signal=identifier_signal,
        deterministic_score=round(
            max(
                0.0,
                min(
                    score,
                    1.0,
                ),
            ),
            4,
        ),
        rationale=rationale,
    )


# ============================================================
# SURROGATE NAME
# ============================================================


def suggest_surrogate_column_name(
    dataframe: pd.DataFrame,
) -> str:
    existing = {
        normalize_column_name(
            column_name
        )
        for column_name in dataframe.columns
    }

    for candidate in [
        "row_id",
        "datalens_row_id",
        "technical_row_id",
    ]:
        if (
            normalize_column_name(
                candidate
            )
            not in existing
        ):
            return candidate

    index = 2

    while True:
        candidate = (
            f"datalens_row_id_{index}"
        )

        if (
            normalize_column_name(
                candidate
            )
            not in existing
        ):
            return candidate

        index += 1


# ============================================================
# PROFILING
# ============================================================


def profile_dataset_identity(
    dataframe: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_filename: str,
    max_composite_width: int = 2,
    max_identifier_columns: int = 12,
) -> DatasetIdentityReport:
    """
    Build a conservative row-identity report.

    Important distinction:

    - mechanically unique:
      any complete column whose values are unique;

    - reliable identity candidate:
      a mechanically unique column, or a simple unique
      composite, whose names also look like identifiers.

    Identifier conventions accepted here include both common
    suffix forms such as customer_id and prefix forms such as
    id_prod.

    An arbitrary amount, timestamp or free-text field is not
    accepted as a row identity merely because it happens to be
    unique in the current sample.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Dataset identity profiling requires at least one row."
        )

    if len(dataframe.columns) == 0:
        raise ValueError(
            "Dataset identity profiling requires at least one column."
        )

    if max_composite_width < 2:
        raise ValueError(
            "max_composite_width must be at least 2."
        )

    single_candidates = [
        _single_candidate(
            dataframe,
            str(column_name),
        )
        for column_name in dataframe.columns
    ]

    mechanically_unique_columns = [
        candidate.columns[0]
        for candidate in single_candidates
        if candidate.unique
    ]

    identifier_like_columns = [
        str(column_name)
        for column_name in dataframe.columns
        if has_identifier_name_signal(
            str(column_name)
        )
    ]

    reliable_single_candidates = [
        candidate
        for candidate in single_candidates
        if (
            candidate.unique
            and
            candidate.identifier_name_signal
        )
    ]

    composite_candidates: list[
        DatasetIdentityCandidate
    ] = []

    composite_pool = (
        identifier_like_columns[
            :max_identifier_columns
        ]
    )

    if not reliable_single_candidates:
        maximum_width = min(
            max_composite_width,
            len(composite_pool),
        )

        for width in range(
            2,
            maximum_width + 1,
        ):
            for column_names in combinations(
                composite_pool,
                width,
            ):
                candidate = (
                    _composite_candidate(
                        dataframe,
                        tuple(column_names),
                    )
                )

                if (
                    candidate.unique
                    and
                    candidate.identifier_name_signal
                ):
                    composite_candidates.append(
                        candidate
                    )

    candidates = [
        *reliable_single_candidates,
        *composite_candidates,
    ]

    candidates.sort(
        key=lambda candidate: (
            -candidate.deterministic_score,
            len(candidate.columns),
            tuple(candidate.columns),
        )
    )

    preferred_candidate = (
        candidates[0]
        if candidates
        else None
    )

    if (
        preferred_candidate is not None
        and
        preferred_candidate.kind == "single"
    ):
        status: IdentityStatus = "single_key"
        surrogate_recommended = False
        suggested_surrogate_column = None
        reasons = [
            (
                "A complete, unique and identifier-like single "
                "column was detected."
            ),
            (
                "A technical surrogate key is not recommended "
                "for row identity at this stage."
            ),
        ]

    elif preferred_candidate is not None:
        status = "composite_key"
        surrogate_recommended = False
        suggested_surrogate_column = None
        reasons = [
            (
                "No reliable single-column row key was detected, "
                "but a simple complete and unique identifier-like "
                "composite was found."
            ),
            (
                "A technical surrogate key is not required solely "
                "for uniqueness, although one could still be useful "
                "for downstream engineering conventions."
            ),
        ]

    else:
        status = "surrogate_recommended"
        surrogate_recommended = True
        suggested_surrogate_column = (
            suggest_surrogate_column_name(
                dataframe
            )
        )

        reasons = [
            (
                "No conservative natural row-identity candidate "
                "was detected."
            ),
            (
                "DataLens recommends a technical row identifier "
                "for traceability. This identifier must not be "
                "treated as a join key between unrelated datasets."
            ),
        ]

        if mechanically_unique_columns:
            reasons.append(
                (
                    "Some columns are mechanically unique in the "
                    "current data, but their names do not provide "
                    "enough evidence to treat them as identifiers: "
                    + ", ".join(
                        mechanically_unique_columns
                    )
                    + "."
                )
            )

    return DatasetIdentityReport(
        dataset_id=str(
            dataset_id
        ),
        dataset_filename=str(
            dataset_filename
        ),
        row_count=int(
            len(dataframe)
        ),
        column_count=int(
            len(dataframe.columns)
        ),
        status=status,
        preferred_candidate=preferred_candidate,
        candidates=candidates,
        mechanically_unique_columns=mechanically_unique_columns,
        identifier_like_columns=identifier_like_columns,
        surrogate_key_recommended=surrogate_recommended,
        suggested_surrogate_column=suggested_surrogate_column,
        reasons=reasons,
        rule_version=DATASET_IDENTITY_RULE_VERSION,
    )


# ============================================================
# SURROGATE KEY TRANSFORMATION
# ============================================================


def create_surrogate_row_key(
    dataframe: pd.DataFrame,
    *,
    column_name: str | None = None,
) -> SurrogateKeyTransformation:
    """
    Add a deterministic technical row identifier.

    The key is deterministic for the current prepared row order.
    It is intended for row traceability inside Preparation, not
    for cross-dataset joins.

    The input DataFrame is never mutated.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot create a surrogate key for an empty DataFrame."
        )

    resolved_column_name = (
        column_name.strip()
        if (
            column_name is not None
            and
            column_name.strip()
        )
        else suggest_surrogate_column_name(
            dataframe
        )
    )

    normalized_existing = {
        normalize_column_name(
            existing_column
        )
        for existing_column in dataframe.columns
    }

    if (
        normalize_column_name(
            resolved_column_name
        )
        in normalized_existing
    ):
        raise ValueError(
            (
                "Surrogate key column already exists or collides "
                "after normalization: "
                f"{resolved_column_name}"
            )
        )

    output = dataframe.copy(
        deep=True
    )

    output.insert(
        0,
        resolved_column_name,
        range(
            1,
            len(output) + 1,
        ),
    )

    return SurrogateKeyTransformation(
        dataframe=output,
        column_name=resolved_column_name,
        rows=int(
            len(output)
        ),
        rule_version=DATASET_IDENTITY_RULE_VERSION,
    )

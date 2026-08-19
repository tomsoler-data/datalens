from __future__ import annotations

from collections import (
    Counter,
    defaultdict,
)
from enum import Enum
import math
import re
from typing import Any

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# VERSION
# ============================================================

QUALITY_ENGINE_RULE_VERSION = (
    "data_quality_engine_v0.2"
)

CATEGORY_VARIANT_BRIDGE_RULE_VERSION = (
    "category_variant_bridge_v0.1"
)


# ============================================================
# ENUMS
# ============================================================

class QualitySeverity(
    str,
    Enum,
):
    IMPORTANT = (
        "important"
    )

    MODERATE = (
        "moderate"
    )

    MINOR = (
        "minor"
    )


class QualityIssueKind(
    str,
    Enum,
):
    MISSING_VALUES = (
        "missing_values"
    )

    DUPLICATE_ROWS = (
        "duplicate_rows"
    )

    CONSTANT_COLUMN = (
        "constant_column"
    )

    MIXED_NUMERIC_FORMAT = (
        "mixed_numeric_format"
    )

    INVALID_NUMERIC_VALUES = (
        "invalid_numeric_values"
    )

    NUMERIC_OUTLIERS = (
        "numeric_outliers"
    )

    INVALID_DATES = (
        "invalid_dates"
    )

    MIXED_DATE_FORMATS = (
        "mixed_date_formats"
    )

    CATEGORY_FORMAT_VARIANTS = (
        "category_format_variants"
    )

    POSSIBLE_SEMANTIC_ALIASES = (
        "possible_semantic_aliases"
    )

    INVALID_EMAILS = (
        "invalid_emails"
    )

    MISSING_IDENTIFIER = (
        "missing_identifier"
    )


class CleaningOperation(
    str,
    Enum,
):
    NONE = (
        "none"
    )

    NORMALIZE_WHITESPACE = (
        "normalize_whitespace"
    )

    NORMALIZE_CASE = (
        "normalize_case"
    )

    COERCE_NUMERIC = (
        "coerce_numeric"
    )

    PARSE_DATETIME = (
        "parse_datetime"
    )

    DROP_EXACT_DUPLICATES = (
        "drop_exact_duplicates"
    )

    EXCLUDE_FROM_AUTOMATIC_ANALYSIS = (
        "exclude_from_automatic_analysis"
    )

    REVIEW_VALUES = (
        "review_values"
    )


# ============================================================
# CONTRACTS
# ============================================================

class IssueEvidence(
    BaseModel,
):
    observed_count: int = 0

    affected_ratio: float = 0.0

    examples: list[
        str
    ] = Field(
        default_factory=list,
    )

    details: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class CleaningProposal(
    BaseModel,
):
    operation: CleaningOperation

    automatic_safe: bool

    description: str

    requires_user_confirmation: bool = (
        False
    )

    parameters: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class QualityIssue(
    BaseModel,
):
    issue_id: str

    dataset_id: str

    dataset_filename: str

    column: (
        str
        |
        None
    ) = None

    kind: QualityIssueKind

    severity: QualitySeverity

    title: str

    explanation: str

    evidence: IssueEvidence

    proposal: CleaningProposal

    semantic_review_recommended: bool = (
        False
    )


class DatasetQualitySummary(
    BaseModel,
):
    dataset_id: str

    dataset_filename: str

    row_count: int

    column_count: int

    missing_cell_count: int

    missing_cell_ratio: float

    duplicate_row_count: int

    issue_count: int

    important_count: int

    moderate_count: int

    minor_count: int


class DataQualityReport(
    BaseModel,
):
    status: str

    dataset_count: int

    total_rows: int

    total_columns: int

    issue_count: int

    important_count: int

    moderate_count: int

    minor_count: int

    semantic_review_count: int

    datasets: list[
        DatasetQualitySummary
    ]

    issues: list[
        QualityIssue
    ]

    notes: list[
        str
    ]

    rule_version: str = (
        QUALITY_ENGINE_RULE_VERSION
    )


# ============================================================
# COLUMN-NAME SIGNALS
#
# These signals are metadata only.
# They never authorize destructive cleaning by themselves.
# ============================================================

_IDENTIFIER_NAME_RE = re.compile(
    (
        r"(^id$|_id$|^id_|"
        r"identifier|uuid|guid|key$)"
    ),
    flags=
        re.IGNORECASE,
)


_DATE_NAME_RE = re.compile(
    (
        r"(^date$|date|time|timestamp|datetime|"
        r"created|updated|signup|hire)"
    ),
    flags=
        re.IGNORECASE,
)


_EMAIL_NAME_RE = re.compile(
    (
        r"(^email$|e_mail|mail_address)"
    ),
    flags=
        re.IGNORECASE,
)


# ============================================================
# NUMERIC NAME SIGNAL v0.2
#
# v0.1 used a raw substring regex.
#
# Example:
#
#   shipping_country
#
# accidentally matched:
#
#   count
#
# inside:
#
#   country
#
# v0.2 only accepts numeric concepts as complete column-name
# tokens separated by _, -, whitespace, or boundaries.
#
# IMPORTANT:
# Even a valid name signal is NOT enough to classify textual
# values as numeric. Value-level evidence is still mandatory.
# ============================================================

_NUMERIC_NAME_RE = re.compile(
    (
        r"(^|[_\-\s])("
        r"age|salary|price|amount|revenue|cost|score|rate|ratio|"
        r"quantity|qty|count|number|total|spend|basket|hours|tenure"
        r")($|[_\-\s])"
    ),
    flags=
        re.IGNORECASE,
)


# ============================================================
# SEMANTIC ALIAS CANDIDATES
#
# These are candidates only.
# They are never merged directly by this module.
# ============================================================

_SEMANTIC_ALIAS_PAIRS = {
    frozenset(
        (
            "m",
            "male",
        )
    ),

    frozenset(
        (
            "f",
            "female",
        )
    ),

    frozenset(
        (
            "hr",
            "human resources",
        )
    ),

    frozenset(
        (
            "it",
            "information technology",
        )
    ),

    frozenset(
        (
            "uk",
            "united kingdom",
        )
    ),

    frozenset(
        (
            "us",
            "united states",
        )
    ),

    frozenset(
        (
            "usa",
            "united states",
        )
    ),
}


# ============================================================
# GENERIC HELPERS
# ============================================================

def _safe_ratio(
    numerator: (
        int
        |
        float
    ),
    denominator: (
        int
        |
        float
    ),
) -> float:
    if (
        denominator
        <=
        0
    ):
        return 0.0

    return float(
        numerator
        /
        denominator
    )


def _display_value(
    value: Any,
) -> str:
    if pd.isna(
        value
    ):
        return (
            "<missing>"
        )

    text = str(
        value
    )

    if (
        len(
            text
        )
        >
        120
    ):
        return (
            text[
                :117
            ]
            +
            "..."
        )

    return text


def _unique_examples(
    values: list[
        Any
    ],
    *,
    limit: int = 8,
) -> list[
    str
]:
    output: list[
        str
    ] = []

    for value in values:
        rendered = (
            _display_value(
                value
            )
        )

        if (
            rendered
            in
            output
        ):
            continue

        output.append(
            rendered
        )

        if (
            len(
                output
            )
            >=
            limit
        ):
            break

    return output


def _normalize_category_token(
    value: Any,
) -> str:
    return (
        re.sub(
            r"\s+",
            " ",
            str(
                value
            ).strip(),
        )
        .casefold()
    )


# ============================================================
# NUMERIC HELPERS
# ============================================================

def _strip_numeric_decoration(
    value: Any,
) -> str:
    text = str(
        value
    ).strip()

    text = (
        text
        .replace(
            "\u00a0",
            "",
        )
        .replace(
            " ",
            "",
        )
    )

    text = re.sub(
        r"(?i)(€|\$|£|usd|eur|gbp)",
        "",
        text,
    )

    if (
        text.count(
            ","
        )
        ==
        1
        and
        text.count(
            "."
        )
        ==
        0
    ):
        text = (
            text.replace(
                ",",
                ".",
            )
        )

    return text


def _coerce_numeric_series(
    series: pd.Series,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    non_missing = (
        series[
            series.notna()
        ]
    )

    cleaned = (
        non_missing.map(
            _strip_numeric_decoration
        )
    )

    numeric = (
        pd.to_numeric(
            cleaned,
            errors=
                "coerce",
        )
    )

    return (
        numeric,
        cleaned,
    )


# ============================================================
# DATE HELPERS
# ============================================================

def _parse_datetime_value(
    value: Any,
) -> (
    pd.Timestamp
    |
    None
):
    if pd.isna(
        value
    ):
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    try:
        year_first = bool(
            re.match(
                r"^\d{4}[-/]",
                text,
            )
        )

        parsed = (
            pd.to_datetime(
                text,
                errors=
                    "coerce",
                dayfirst=
                    not year_first,
            )
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

    if isinstance(
        parsed,
        pd.DatetimeIndex,
    ):
        if (
            len(
                parsed
            )
            ==
            0
        ):
            return None

        return (
            pd.Timestamp(
                parsed[
                    0
                ]
            )
        )

    return (
        pd.Timestamp(
            parsed
        )
    )


def _date_format_family(
    value: Any,
) -> str:
    text = str(
        value
    ).strip()

    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        text,
    ):
        return (
            "iso_date"
        )

    if re.fullmatch(
        r"\d{4}/\d{1,2}/\d{1,2}",
        text,
    ):
        return (
            "year_first_slash"
        )

    if re.fullmatch(
        r"\d{1,2}/\d{1,2}/\d{4}",
        text,
    ):
        return (
            "day_first_slash"
        )

    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[T\s].+",
        text,
    ):
        return (
            "iso_datetime"
        )

    return (
        "other"
    )


# ============================================================
# SEVERITY
# ============================================================

def _severity_for_ratio(
    ratio: float,
) -> QualitySeverity:
    if (
        ratio
        >=
        0.25
    ):
        return (
            QualitySeverity
            .IMPORTANT
        )

    if (
        ratio
        >=
        0.05
    ):
        return (
            QualitySeverity
            .MODERATE
        )

    return (
        QualitySeverity
        .MINOR
    )


# ============================================================
# OUTLIER HELPER
#
# Conservative 3×IQR fence.
# Outliers are signals, never automatic errors.
# ============================================================

def _iqr_outlier_mask(
    values: pd.Series,
) -> tuple[
    pd.Series,
    dict[
        str,
        float,
    ],
]:
    clean = (
        values
        .dropna()
        .astype(
            float
        )
    )

    if (
        len(
            clean
        )
        <
        8
    ):
        return (
            pd.Series(
                False,
                index=
                    values.index,
            ),
            {},
        )

    q1 = float(
        clean.quantile(
            0.25
        )
    )

    q3 = float(
        clean.quantile(
            0.75
        )
    )

    iqr = (
        q3
        -
        q1
    )

    if (
        not math.isfinite(
            iqr
        )
        or
        iqr
        <=
        0
    ):
        return (
            pd.Series(
                False,
                index=
                    values.index,
            ),
            {
                "q1":
                    q1,

                "q3":
                    q3,

                "iqr":
                    iqr,
            },
        )

    lower = (
        q1
        -
        3.0
        *
        iqr
    )

    upper = (
        q3
        +
        3.0
        *
        iqr
    )

    mask = (
        (
            values
            <
            lower
        )
        |
        (
            values
            >
            upper
        )
    )

    return (
        mask.fillna(
            False
        ),
        {
            "q1":
                q1,

            "q3":
                q3,

            "iqr":
                iqr,

            "lower_fence":
                lower,

            "upper_fence":
                upper,
        },
    )


# ============================================================
# QUALITY ISSUE FACTORY
# ============================================================

def _issue(
    *,
    dataset_id: str,
    dataset_filename: str,
    column: (
        str
        |
        None
    ),
    kind: QualityIssueKind,
    severity: QualitySeverity,
    title: str,
    explanation: str,
    observed_count: int,
    affected_ratio: float,
    examples: list[
        Any
    ],
    details: dict[
        str,
        Any,
    ],
    proposal: CleaningProposal,
    semantic_review_recommended: bool = (
        False
    ),
) -> QualityIssue:
    column_token = (
        column
        if column
        else
        "__dataset__"
    )

    return (
        QualityIssue(
            issue_id=(
                f"{dataset_id}:"
                f"{column_token}:"
                f"{kind.value}"
            ),

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            column=
                column,

            kind=
                kind,

            severity=
                severity,

            title=
                title,

            explanation=
                explanation,

            evidence=
                IssueEvidence(
                    observed_count=
                        observed_count,

                    affected_ratio=
                        affected_ratio,

                    examples=
                        _unique_examples(
                            examples
                        ),

                    details=
                        details,
                ),

            proposal=
                proposal,

            semantic_review_recommended=
                semantic_review_recommended,
        )
    )


# ============================================================
# MISSING VALUES
# ============================================================

def _detect_missing_values(
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> list[
    QualityIssue
]:
    issues: list[
        QualityIssue
    ] = []

    row_count = len(
        dataframe
    )

    for column in (
        dataframe.columns
    ):
        series = (
            dataframe[
                column
            ]
        )

        missing_count = int(
            series
            .isna()
            .sum()
        )

        if (
            missing_count
            <=
            0
        ):
            continue

        ratio = (
            _safe_ratio(
                missing_count,
                row_count,
            )
        )

        is_identifier = bool(
            _IDENTIFIER_NAME_RE
            .search(
                str(
                    column
                )
            )
        )

        kind = (
            QualityIssueKind
            .MISSING_IDENTIFIER
            if is_identifier
            else
            QualityIssueKind
            .MISSING_VALUES
        )

        severity = (
            QualitySeverity
            .IMPORTANT
            if is_identifier
            else
            _severity_for_ratio(
                ratio
            )
        )

        issues.append(
            _issue(
                dataset_id=
                    dataset_id,

                dataset_filename=
                    dataset_filename,

                column=
                    str(
                        column
                    ),

                kind=
                    kind,

                severity=
                    severity,

                title=(
                    "Identifiant manquant"
                    if is_identifier
                    else
                    "Valeurs manquantes"
                ),

                explanation=(
                    (
                        "Une ou plusieurs lignes ne disposent "
                        "pas d'un identifiant exploitable. "
                        "DataLens ne doit pas inventer cette clé."
                    )
                    if is_identifier
                    else
                    (
                        "Des cellules sont absentes. "
                        "Le moteur les signale sans appliquer "
                        "d'imputation automatique."
                    )
                ),

                observed_count=
                    missing_count,

                affected_ratio=
                    ratio,

                examples=[
                    "<missing>",
                ],

                details={
                    "row_count":
                        row_count,
                },

                proposal=
                    CleaningProposal(
                        operation=
                            CleaningOperation
                            .REVIEW_VALUES,

                        automatic_safe=
                            False,

                        description=(
                            "Conserver les valeurs manquantes "
                            "et demander une règle explicite "
                            "avant toute imputation."
                        ),

                        requires_user_confirmation=
                            True,
                    ),

                semantic_review_recommended=(
                    is_identifier
                    or
                    ratio
                    >=
                    0.05
                ),
            )
        )

    return (
        issues
    )


# ============================================================
# EXACT DUPLICATES
# ============================================================

def _detect_duplicate_rows(
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> list[
    QualityIssue
]:
    duplicate_mask = (
        dataframe
        .duplicated(
            keep=
                "first",
        )
    )

    duplicate_count = int(
        duplicate_mask
        .sum()
    )

    if (
        duplicate_count
        <=
        0
    ):
        return []

    duplicate_rows = (
        dataframe
        .loc[
            duplicate_mask
        ]
        .head(
            3
        )
        .astype(
            "string"
        )
        .fillna(
            "<missing>"
        )
        .to_dict(
            orient=
                "records"
        )
    )

    return [
        _issue(
            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            column=
                None,

            kind=
                QualityIssueKind
                .DUPLICATE_ROWS,

            severity=
                QualitySeverity
                .MODERATE,

            title=
                "Doublons stricts",

            explanation=(
                "Des lignes sont strictement identiques "
                "à des lignes précédentes."
            ),

            observed_count=
                duplicate_count,

            affected_ratio=
                _safe_ratio(
                    duplicate_count,
                    len(
                        dataframe
                    ),
                ),

            examples=[
                str(
                    row
                )
                for row
                in duplicate_rows
            ],

            details={
                "keep_policy":
                    "first",
            },

            proposal=
                CleaningProposal(
                    operation=
                        CleaningOperation
                        .DROP_EXACT_DUPLICATES,

                    automatic_safe=
                        True,

                    description=(
                        "La suppression des copies strictement "
                        "identiques peut être proposée, mais le "
                        "nombre de lignes affectées doit être "
                        "affiché avant exécution."
                    ),

                    requires_user_confirmation=
                        True,

                    parameters={
                        "keep":
                            "first",
                    },
                ),
        )
    ]


# ============================================================
# CONSTANT COLUMNS
# ============================================================

def _detect_constant_columns(
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> list[
    QualityIssue
]:
    issues: list[
        QualityIssue
    ] = []

    for column in (
        dataframe.columns
    ):
        series = (
            dataframe[
                column
            ]
        )

        unique_non_missing = int(
            series
            .nunique(
                dropna=
                    True
            )
        )

        if (
            unique_non_missing
            >
            1
        ):
            continue

        non_missing = (
            series
            .dropna()
        )

        examples = (
            non_missing
            .head(
                3
            )
            .tolist()
        )

        issues.append(
            _issue(
                dataset_id=
                    dataset_id,

                dataset_filename=
                    dataset_filename,

                column=
                    str(
                        column
                    ),

                kind=
                    QualityIssueKind
                    .CONSTANT_COLUMN,

                severity=
                    QualitySeverity
                    .MODERATE,

                title=
                    "Colonne constante",

                explanation=(
                    "Cette variable ne présente pas de "
                    "variabilité exploitable dans le dataset."
                ),

                observed_count=
                    len(
                        non_missing
                    ),

                affected_ratio=
                    _safe_ratio(
                        len(
                            non_missing
                        ),
                        len(
                            dataframe
                        ),
                    ),

                examples=
                    examples,

                details={
                    "unique_non_missing":
                        unique_non_missing,
                },

                proposal=
                    CleaningProposal(
                        operation=
                            CleaningOperation
                            .EXCLUDE_FROM_AUTOMATIC_ANALYSIS,

                        automatic_safe=
                            True,

                        description=(
                            "Conserver la colonne brute, mais "
                            "l'écarter des analyses automatiques "
                            "qui nécessitent de la variabilité."
                        ),
                    ),
            )
        )

    return (
        issues
    )


# ============================================================
# NUMERIC ISSUES — v0.2
#
# Architecture:
#
# native numeric
#     → numerical by dtype
#
# textual/object column
#     → value-level evidence required
#
# column name
#     → metadata signal only
#
# No LLM.
# No mutation.
# No automatic outlier removal.
# ============================================================

def _detect_numeric_issues(
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> list[
    QualityIssue
]:
    issues: list[
        QualityIssue
    ] = []

    row_count = len(
        dataframe
    )

    minimum_valid_numeric_count = (
        3
    )

    minimum_numeric_parse_ratio = (
        0.80
    )

    for column in (
        dataframe.columns
    ):
        name = str(
            column
        )

        series = (
            dataframe[
                column
            ]
        )

        # ----------------------------------------------------
        # BOOL IS CATEGORICAL, NOT QUANTITATIVE
        # ----------------------------------------------------

        if (
            pd.api.types
            .is_bool_dtype(
                series
            )
        ):
            continue

        # ----------------------------------------------------
        # NAME SIGNAL = SUPPORTING METADATA ONLY
        # ----------------------------------------------------

        name_numeric_signal = bool(
            _NUMERIC_NAME_RE
            .search(
                name
            )
        )

        native_numeric = bool(
            pd.api.types
            .is_numeric_dtype(
                series
            )
        )

        # ====================================================
        # NATIVE NUMERIC
        # ====================================================

        if native_numeric:
            numeric = (
                pd.to_numeric(
                    series,
                    errors=
                        "coerce",
                )
            )

            valid_count = int(
                numeric
                .notna()
                .sum()
            )

            non_missing_count = int(
                series
                .notna()
                .sum()
            )

            parse_ratio = (
                _safe_ratio(
                    valid_count,
                    non_missing_count,
                )
            )

            numeric_intent = (
                "native_numeric"
            )

        # ====================================================
        # TEXTUAL / OBJECT NUMERIC INTENT
        # ====================================================

        else:
            (
                coerced,
                cleaned,
            ) = (
                _coerce_numeric_series(
                    series
                )
            )

            non_missing_count = int(
                series
                .notna()
                .sum()
            )

            valid_count = int(
                coerced
                .notna()
                .sum()
            )

            parse_ratio = (
                _safe_ratio(
                    valid_count,
                    non_missing_count,
                )
            )

            # ------------------------------------------------
            # CRITICAL v0.2 GATE
            #
            # A name can NEVER bypass this gate.
            #
            # amount_as_text:
            #   5 / 6 numeric -> accepted.
            #
            # shipping_country:
            #   0 / 5 numeric -> rejected.
            # ------------------------------------------------

            strong_value_numeric_intent = (
                valid_count
                >=
                minimum_valid_numeric_count
                and
                parse_ratio
                >=
                minimum_numeric_parse_ratio
            )

            if not (
                strong_value_numeric_intent
            ):
                continue

            numeric_intent = (
                "strong_value_evidence"
            )

            numeric = (
                pd.Series(
                    float(
                        "nan"
                    ),
                    index=
                        series.index,
                    dtype=
                        "float64",
                )
            )

            numeric.loc[
                coerced.index
            ] = (
                coerced
            )

            # =================================================
            # INVALID NUMERIC VALUES
            # =================================================

            invalid_mask = (
                series.notna()
                &
                numeric.isna()
            )

            invalid_count = int(
                invalid_mask
                .sum()
            )

            if (
                invalid_count
                >
                0
            ):
                invalid_values = (
                    series
                    .loc[
                        invalid_mask
                    ]
                    .tolist()
                )

                issues.append(
                    _issue(
                        dataset_id=
                            dataset_id,

                        dataset_filename=
                            dataset_filename,

                        column=
                            name,

                        kind=
                            QualityIssueKind
                            .INVALID_NUMERIC_VALUES,

                        severity=(
                            QualitySeverity
                            .IMPORTANT
                            if (
                                parse_ratio
                                >=
                                0.80
                            )
                            else
                            QualitySeverity
                            .MODERATE
                        ),

                        title=(
                            "Valeurs numériques incompatibles"
                        ),

                        explanation=(
                            "La colonne présente une intention "
                            "numérique forte dans ses valeurs, "
                            "mais certaines observations ne peuvent "
                            "pas être converties de façon "
                            "déterministe."
                        ),

                        observed_count=
                            invalid_count,

                        affected_ratio=
                            _safe_ratio(
                                invalid_count,
                                row_count,
                            ),

                        examples=
                            invalid_values,

                        details={
                            "numeric_parse_ratio":
                                parse_ratio,

                            "valid_numeric_count":
                                valid_count,

                            "non_missing_count":
                                non_missing_count,

                            "minimum_numeric_parse_ratio":
                                minimum_numeric_parse_ratio,

                            "minimum_valid_numeric_count":
                                minimum_valid_numeric_count,

                            "name_numeric_signal":
                                name_numeric_signal,

                            "numeric_intent":
                                numeric_intent,
                        },

                        proposal=
                            CleaningProposal(
                                operation=
                                    CleaningOperation
                                    .REVIEW_VALUES,

                                automatic_safe=
                                    False,

                                description=(
                                    "Convertir uniquement les valeurs "
                                    "numériques sûres et isoler les "
                                    "valeurs non interprétables."
                                ),

                                requires_user_confirmation=
                                    True,
                            ),

                        semantic_review_recommended=
                            True,
                    )
                )

            # =================================================
            # MIXED NUMERIC FORMATTING
            # =================================================

            canonical_numeric_text = (
                numeric.map(
                    lambda value:
                    (
                        str(
                            int(
                                value
                            )
                        )
                        if (
                            pd.notna(
                                value
                            )
                            and
                            float(
                                value
                            ).is_integer()
                        )
                        else
                        (
                            str(
                                float(
                                    value
                                )
                            )
                            if pd.notna(
                                value
                            )
                            else None
                        )
                    )
                )
            )

            decorated_mask = (
                series.notna()
                &
                numeric.notna()
                &
                (
                    series
                    .astype(
                        "string"
                    )
                    .str
                    .strip()
                    !=
                    canonical_numeric_text
                )
            )

            decorated_mask = (
                decorated_mask
                .fillna(
                    False
                )
            )

            decorated_values = (
                series
                .loc[
                    decorated_mask
                ]
                .tolist()
            )

            suspicious_decorated = [
                value

                for value
                in decorated_values

                if re.search(
                    r"[€$£]|\s|,",
                    str(
                        value
                    ),
                )
            ]

            if (
                suspicious_decorated
            ):
                issues.append(
                    _issue(
                        dataset_id=
                            dataset_id,

                        dataset_filename=
                            dataset_filename,

                        column=
                            name,

                        kind=
                            QualityIssueKind
                            .MIXED_NUMERIC_FORMAT,

                        severity=
                            QualitySeverity
                            .MINOR,

                        title=
                            "Formats numériques hétérogènes",

                        explanation=(
                            "Certaines valeurs numériques "
                            "contiennent une décoration ou un "
                            "format différent du reste de la colonne."
                        ),

                        observed_count=
                            len(
                                suspicious_decorated
                            ),

                        affected_ratio=
                            _safe_ratio(
                                len(
                                    suspicious_decorated
                                ),
                                row_count,
                            ),

                        examples=
                            suspicious_decorated,

                        details={
                            "numeric_parse_ratio":
                                parse_ratio,

                            "valid_numeric_count":
                                valid_count,

                            "non_missing_count":
                                non_missing_count,

                            "name_numeric_signal":
                                name_numeric_signal,

                            "numeric_intent":
                                numeric_intent,
                        },

                        proposal=
                            CleaningProposal(
                                operation=
                                    CleaningOperation
                                    .COERCE_NUMERIC,

                                automatic_safe=
                                    False,

                                description=(
                                    "Retirer uniquement les décorations "
                                    "numériques reconnues après affichage "
                                    "des valeurs concernées."
                                ),

                                requires_user_confirmation=
                                    True,
                            ),
                    )
                )

        # ====================================================
        # OUTLIERS
        # ====================================================

        if (
            valid_count
            <
            8
        ):
            continue

        (
            outlier_mask,
            fences,
        ) = (
            _iqr_outlier_mask(
                numeric
            )
        )

        outlier_count = int(
            outlier_mask
            .sum()
        )

        if (
            outlier_count
            <=
            0
        ):
            continue

        outlier_values = (
            series
            .loc[
                outlier_mask
            ]
            .tolist()
        )

        issues.append(
            _issue(
                dataset_id=
                    dataset_id,

                dataset_filename=
                    dataset_filename,

                column=
                    name,

                kind=
                    QualityIssueKind
                    .NUMERIC_OUTLIERS,

                severity=(
                    QualitySeverity
                    .MODERATE
                    if (
                        outlier_count
                        <=
                        max(
                            3,
                            int(
                                row_count
                                *
                                0.02
                            ),
                        )
                    )
                    else
                    QualitySeverity
                    .MINOR
                ),

                title=
                    "Valeurs numériques atypiques",

                explanation=(
                    "Des observations se situent au-delà "
                    "des bornes robustes Q1 − 3×IQR / "
                    "Q3 + 3×IQR. Elles sont signalées, "
                    "jamais supprimées automatiquement."
                ),

                observed_count=
                    outlier_count,

                affected_ratio=
                    _safe_ratio(
                        outlier_count,
                        row_count,
                    ),

                examples=
                    outlier_values,

                details={
                    **fences,

                    "numeric_parse_ratio":
                        parse_ratio,

                    "valid_numeric_count":
                        valid_count,

                    "non_missing_count":
                        non_missing_count,

                    "name_numeric_signal":
                        name_numeric_signal,

                    "numeric_intent":
                        numeric_intent,
                },

                proposal=
                    CleaningProposal(
                        operation=
                            CleaningOperation
                            .REVIEW_VALUES,

                        automatic_safe=
                            False,

                        description=(
                            "Conserver les valeurs et demander "
                            "une validation métier avant toute "
                            "correction ou exclusion."
                        ),

                        requires_user_confirmation=
                            True,
                    ),

                semantic_review_recommended=
                    True,
            )
        )

    return (
        issues
    )


# ============================================================
# DATE ISSUES
# ============================================================

def _detect_date_issues(
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> list[
    QualityIssue
]:
    issues: list[
        QualityIssue
    ] = []

    row_count = len(
        dataframe
    )

    for column in (
        dataframe.columns
    ):
        name = str(
            column
        )

        series = (
            dataframe[
                column
            ]
        )

        native_datetime = bool(
            pd.api.types
            .is_datetime64_any_dtype(
                series
            )
        )

        name_date_signal = bool(
            _DATE_NAME_RE
            .search(
                name
            )
        )

        if (
            not native_datetime
            and
            not name_date_signal
        ):
            continue

        non_missing = (
            series[
                series.notna()
            ]
        )

        if (
            non_missing.empty
        ):
            continue

        # Already a real pandas datetime dtype.
        if (
            native_datetime
        ):
            continue

        parsed = (
            non_missing.map(
                _parse_datetime_value
            )
        )

        invalid_mask = (
            parsed.isna()
        )

        invalid_count = int(
            invalid_mask
            .sum()
        )

        if (
            invalid_count
            >
            0
        ):
            invalid_values = (
                non_missing
                .loc[
                    invalid_mask
                ]
                .tolist()
            )

            invalid_ratio = (
                _safe_ratio(
                    invalid_count,
                    len(
                        non_missing
                    ),
                )
            )

            issues.append(
                _issue(
                    dataset_id=
                        dataset_id,

                    dataset_filename=
                        dataset_filename,

                    column=
                        name,

                    kind=
                        QualityIssueKind
                        .INVALID_DATES,

                    severity=
                        _severity_for_ratio(
                            invalid_ratio
                        ),

                    title=
                        "Dates invalides",

                    explanation=(
                        "Certaines valeurs d'une colonne "
                        "temporelle ne peuvent pas être "
                        "interprétées comme des dates valides."
                    ),

                    observed_count=
                        invalid_count,

                    affected_ratio=
                        _safe_ratio(
                            invalid_count,
                            row_count,
                        ),

                    examples=
                        invalid_values,

                    details={
                        "date_parse_ratio":
                            _safe_ratio(
                                (
                                    len(
                                        non_missing
                                    )
                                    -
                                    invalid_count
                                ),
                                len(
                                    non_missing
                                ),
                            ),

                        "non_missing_count":
                            int(
                                len(
                                    non_missing
                                )
                            ),
                    },

                    proposal=
                        CleaningProposal(
                            operation=
                                CleaningOperation
                                .REVIEW_VALUES,

                            automatic_safe=
                                False,

                            description=(
                                "Vérifier les valeurs invalides "
                                "dans la source ; ne pas inventer "
                                "de date de remplacement."
                            ),

                            requires_user_confirmation=
                                True,
                        ),

                    semantic_review_recommended=
                        True,
                )
            )

        valid_values = (
            non_missing
            .loc[
                ~invalid_mask
            ]
        )

        if (
            valid_values.empty
        ):
            continue

        format_counts = Counter(
            _date_format_family(
                value
            )

            for value
            in valid_values
        )

        recognized_formats = {
            family:
                count

            for (
                family,
                count,
            )
            in format_counts.items()

            if (
                family
                !=
                "other"
            )
        }

        if (
            len(
                recognized_formats
            )
            <=
            1
        ):
            continue

        formatted_count = int(
            sum(
                recognized_formats
                .values()
            )
        )

        issues.append(
            _issue(
                dataset_id=
                    dataset_id,

                dataset_filename=
                    dataset_filename,

                column=
                    name,

                kind=
                    QualityIssueKind
                    .MIXED_DATE_FORMATS,

                severity=
                    QualitySeverity
                    .MINOR,

                title=
                    "Formats de date hétérogènes",

                explanation=(
                    "Plusieurs représentations de date "
                    "coexistent dans la même colonne."
                ),

                observed_count=
                    formatted_count,

                affected_ratio=
                    _safe_ratio(
                        formatted_count,
                        row_count,
                    ),

                examples=
                    valid_values
                    .head(
                        8
                    )
                    .tolist(),

                details={
                    "format_families":
                        recognized_formats,
                },

                proposal=
                    CleaningProposal(
                        operation=
                            CleaningOperation
                            .PARSE_DATETIME,

                        automatic_safe=
                            False,

                        description=(
                            "Uniformiser le type date seulement "
                            "après avoir affiché les formats "
                            "détectés et validé leur interprétation."
                        ),

                        requires_user_confirmation=
                            True,
                    ),
            )
        )

    return (
        issues
    )


# ============================================================
# CATEGORY VARIANTS / SEMANTIC ALIASES
# ============================================================

def _detect_category_variants(
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> list[
    QualityIssue
]:
    issues: list[
        QualityIssue
    ] = []

    row_count = len(
        dataframe
    )

    for column in (
        dataframe.columns
    ):
        series = (
            dataframe[
                column
            ]
        )

        is_textual = (
            pd.api.types
            .is_object_dtype(
                series
            )
            or
            pd.api.types
            .is_string_dtype(
                series
            )
            or
            isinstance(
                series.dtype,
                pd.CategoricalDtype,
            )
        )

        if not (
            is_textual
        ):
            continue

        non_missing = (
            series
            .dropna()
            .astype(
                str
            )
        )

        if (
            non_missing.empty
        ):
            continue

        unique_values = list(
            dict.fromkeys(
                non_missing
                .tolist()
            )
        )

        unique_count = len(
            unique_values
        )

        # Avoid treating high-cardinality free text as categories.
        if (
            unique_count
            >
            max(
                50,
                int(
                    row_count
                    *
                    0.50
                ),
            )
        ):
            continue

        normalized_groups: dict[
            str,
            list[
                str
            ],
        ] = (
            defaultdict(
                list
            )
        )

        for value in (
            unique_values
        ):
            normalized = (
                _normalize_category_token(
                    value
                )
            )

            if (
                value
                not in
                normalized_groups[
                    normalized
                ]
            ):
                normalized_groups[
                    normalized
                ].append(
                    value
                )

        # ====================================================
        # DETERMINISTIC FORMATTING VARIANTS
        # ====================================================

        variant_groups = [
            values

            for values
            in normalized_groups.values()

            if (
                len(
                    values
                )
                >
                1
            )
        ]

        if (
            variant_groups
        ):
            affected_normalized_values = {
                _normalize_category_token(
                    group[
                        0
                    ]
                )

                for group
                in variant_groups
            }

            affected_mask = (
                non_missing
                .map(
                    _normalize_category_token
                )
                .isin(
                    affected_normalized_values
                )
            )

            affected_count = int(
                affected_mask
                .sum()
            )

            # ------------------------------------------------
            # BRIDGE TO SEMANTIC CANONICALIZATION
            # ------------------------------------------------
            #
            # CATEGORY_FORMAT_VARIANTS is still a deterministic
            # representation-quality issue, not a business alias.
            #
            # However, each exact variant group is also valid
            # input for the guarded semantic canonicalization
            # workflow because that layer owns:
            #
            #   - one proposal per alias/variant group;
            #   - canonical-value selection;
            #   - mandatory human approval;
            #   - execution provenance.
            #
            # candidate_pairs are an interoperability format.
            # For a group with >2 exact spellings, anchoring each
            # value to the first one is sufficient for the
            # downstream review layer to reconstruct the complete
            # normalized group from current observed values.
            # ------------------------------------------------

            candidate_pairs: list[
                tuple[
                    str,
                    str,
                ]
            ] = []

            for group in (
                variant_groups
            ):
                if (
                    len(
                        group
                    )
                    <
                    2
                ):
                    continue

                anchor = (
                    group[
                        0
                    ]
                )

                for variant in (
                    group[
                        1:
                    ]
                ):
                    candidate_pairs.append(
                        (
                            anchor,
                            variant,
                        )
                    )

            issues.append(
                _issue(
                    dataset_id=
                        dataset_id,

                    dataset_filename=
                        dataset_filename,

                    column=
                        str(
                            column
                        ),

                    kind=
                        QualityIssueKind
                        .CATEGORY_FORMAT_VARIANTS,

                    severity=
                        QualitySeverity
                        .MINOR,

                    title=
                        "Variantes de catégorie",

                    explanation=(
                        "Des modalités deviennent identiques "
                        "après normalisation déterministe de la "
                        "casse et des espaces."
                    ),

                    observed_count=
                        affected_count,

                    affected_ratio=
                        _safe_ratio(
                            affected_count,
                            row_count,
                        ),

                    examples=[
                        " / ".join(
                            group
                        )

                        for group
                        in variant_groups
                    ],

                    details={
                        "variant_groups":
                            variant_groups,

                        "candidate_pairs":
                            candidate_pairs,

                        "normalization":
                            (
                                "strip + collapse_whitespace "
                                "+ unicode_casefold"
                            ),

                        "bridge_rule_version":
                            CATEGORY_VARIANT_BRIDGE_RULE_VERSION,
                    },

                    proposal=
                        CleaningProposal(
                            operation=
                                CleaningOperation
                                .NORMALIZE_CASE,

                            automatic_safe=
                                False,

                            description=(
                                "Proposer chaque groupe de variantes "
                                "à la canonicalisation contrôlée. "
                                "Aucune fusion n'est exécutée sans "
                                "confirmation utilisateur."
                            ),

                            requires_user_confirmation=
                                True,
                        ),

                    semantic_review_recommended=
                        True,
                )
            )

        # ====================================================
        # POSSIBLE SEMANTIC ALIASES
        # ====================================================

        normalized_unique_values = list(
            dict.fromkeys(
                _normalize_category_token(
                    value
                )

                for value
                in unique_values
            )
        )

        semantic_pairs: list[
            tuple[
                str,
                str,
            ]
        ] = []

        for left_index in range(
            len(
                normalized_unique_values
            )
        ):
            for right_index in range(
                left_index
                +
                1,
                len(
                    normalized_unique_values
                ),
            ):
                left_normalized = (
                    normalized_unique_values[
                        left_index
                    ]
                )

                right_normalized = (
                    normalized_unique_values[
                        right_index
                    ]
                )

                candidate_pair = (
                    frozenset(
                        (
                            left_normalized,
                            right_normalized,
                        )
                    )
                )

                if (
                    candidate_pair
                    not in
                    _SEMANTIC_ALIAS_PAIRS
                ):
                    continue

                left_values = (
                    normalized_groups.get(
                        left_normalized,
                        [],
                    )
                )

                right_values = (
                    normalized_groups.get(
                        right_normalized,
                        [],
                    )
                )

                if (
                    not left_values
                    or
                    not right_values
                ):
                    continue

                semantic_pairs.append(
                    (
                        left_values[
                            0
                        ],
                        right_values[
                            0
                        ],
                    )
                )

        if (
            semantic_pairs
        ):
            issues.append(
                _issue(
                    dataset_id=
                        dataset_id,

                    dataset_filename=
                        dataset_filename,

                    column=
                        str(
                            column
                        ),

                    kind=
                        QualityIssueKind
                        .POSSIBLE_SEMANTIC_ALIASES,

                    severity=
                        QualitySeverity
                        .MODERATE,

                    title=
                        "Alias sémantiques possibles",

                    explanation=(
                        "Certaines modalités pourraient "
                        "désigner le même concept, mais leur "
                        "fusion n'est pas sûre sans contexte."
                    ),

                    observed_count=
                        len(
                            semantic_pairs
                        ),

                    affected_ratio=
                        _safe_ratio(
                            len(
                                semantic_pairs
                            ),
                            max(
                                unique_count,
                                1,
                            ),
                        ),

                    examples=[
                        f"{left} / {right}"

                        for (
                            left,
                            right,
                        )
                        in semantic_pairs
                    ],

                    details={
                        "candidate_pairs":
                            semantic_pairs,
                    },

                    proposal=
                        CleaningProposal(
                            operation=
                                CleaningOperation
                                .REVIEW_VALUES,

                            automatic_safe=
                                False,

                            description=(
                                "Soumettre ces rapprochements à la "
                                "couche sémantique, puis faire valider "
                                "le mapping exact par Python."
                            ),

                            requires_user_confirmation=
                                True,
                        ),

                    semantic_review_recommended=
                        True,
                )
            )

    return (
        issues
    )


# ============================================================
# EMAIL VALIDATION
# ============================================================

def _detect_email_issues(
    *,
    dataframe: pd.DataFrame,
    dataset_id: str,
    dataset_filename: str,
) -> list[
    QualityIssue
]:
    issues: list[
        QualityIssue
    ] = []

    email_pattern = re.compile(
        r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )

    for column in (
        dataframe.columns
    ):
        name = str(
            column
        )

        if not (
            _EMAIL_NAME_RE
            .search(
                name
            )
        ):
            continue

        series = (
            dataframe[
                column
            ]
            .dropna()
            .astype(
                str
            )
        )

        invalid_mask = (
            ~series
            .str
            .match(
                email_pattern
            )
        )

        invalid_count = int(
            invalid_mask
            .sum()
        )

        if (
            invalid_count
            <=
            0
        ):
            continue

        invalid_values = (
            series
            .loc[
                invalid_mask
            ]
            .tolist()
        )

        issues.append(
            _issue(
                dataset_id=
                    dataset_id,

                dataset_filename=
                    dataset_filename,

                column=
                    name,

                kind=
                    QualityIssueKind
                    .INVALID_EMAILS,

                severity=
                    QualitySeverity
                    .MODERATE,

                title=
                    "Adresses e-mail invalides",

                explanation=(
                    "Certaines valeurs ne respectent pas "
                    "une structure minimale d'adresse e-mail."
                ),

                observed_count=
                    invalid_count,

                affected_ratio=
                    _safe_ratio(
                        invalid_count,
                        len(
                            dataframe
                        ),
                    ),

                examples=
                    invalid_values,

                details={},

                proposal=
                    CleaningProposal(
                        operation=
                            CleaningOperation
                            .REVIEW_VALUES,

                        automatic_safe=
                            False,

                        description=(
                            "Signaler les adresses concernées ; "
                            "ne pas inventer de correction."
                        ),

                        requires_user_confirmation=
                            True,
                    ),
            )
        )

    return (
        issues
    )


# ============================================================
# DATAFRAME QUALITY ANALYSIS
# ============================================================

def analyze_dataframe_quality(
    dataframe: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_filename: str,
) -> tuple[
    DatasetQualitySummary,
    list[
        QualityIssue
    ],
]:
    if (
        dataframe.empty
    ):
        raise ValueError(
            "The dataframe must contain at least one row."
        )

    issues: list[
        QualityIssue
    ] = []

    detectors = (
        _detect_missing_values,
        _detect_duplicate_rows,
        _detect_constant_columns,
        _detect_numeric_issues,
        _detect_date_issues,
        _detect_category_variants,
        _detect_email_issues,
    )

    for detector in (
        detectors
    ):
        issues.extend(
            detector(
                dataframe=
                    dataframe,

                dataset_id=
                    dataset_id,

                dataset_filename=
                    dataset_filename,
            )
        )

    severity_rank = {
        QualitySeverity
        .IMPORTANT:
            3,

        QualitySeverity
        .MODERATE:
            2,

        QualitySeverity
        .MINOR:
            1,
    }

    issues.sort(
        key=lambda issue: (
            -severity_rank[
                issue.severity
            ],
            issue.column
            or "",
            issue.kind.value,
        )
    )

    missing_cell_count = int(
        dataframe
        .isna()
        .sum()
        .sum()
    )

    total_cells = (
        dataframe.shape[
            0
        ]
        *
        dataframe.shape[
            1
        ]
    )

    duplicate_row_count = int(
        dataframe
        .duplicated(
            keep=
                "first",
        )
        .sum()
    )

    important_count = sum(
        issue.severity
        ==
        QualitySeverity
        .IMPORTANT

        for issue
        in issues
    )

    moderate_count = sum(
        issue.severity
        ==
        QualitySeverity
        .MODERATE

        for issue
        in issues
    )

    minor_count = sum(
        issue.severity
        ==
        QualitySeverity
        .MINOR

        for issue
        in issues
    )

    summary = (
        DatasetQualitySummary(
            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            row_count=
                int(
                    dataframe.shape[
                        0
                    ]
                ),

            column_count=
                int(
                    dataframe.shape[
                        1
                    ]
                ),

            missing_cell_count=
                missing_cell_count,

            missing_cell_ratio=
                _safe_ratio(
                    missing_cell_count,
                    total_cells,
                ),

            duplicate_row_count=
                duplicate_row_count,

            issue_count=
                len(
                    issues
                ),

            important_count=
                important_count,

            moderate_count=
                moderate_count,

            minor_count=
                minor_count,
        )
    )

    return (
        summary,
        issues,
    )


# ============================================================
# MULTI-DATASET QUALITY REPORT
# ============================================================

def build_data_quality_report(
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> DataQualityReport:
    if not (
        datasets
    ):
        raise ValueError(
            "At least one dataset is required."
        )

    summaries: list[
        DatasetQualitySummary
    ] = []

    issues: list[
        QualityIssue
    ] = []

    for dataset in (
        datasets
    ):
        dataframe = (
            dataset.get(
                "dataframe"
            )
        )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Each dataset record must contain "
                "a pandas DataFrame under 'dataframe'."
            )

        dataset_id = str(
            dataset.get(
                "dataset_id"
            )
            or
            ""
        ).strip()

        filename = str(
            dataset.get(
                "filename"
            )
            or
            ""
        ).strip()

        if not (
            dataset_id
        ):
            raise ValueError(
                "Each dataset record requires dataset_id."
            )

        if not (
            filename
        ):
            raise ValueError(
                "Each dataset record requires filename."
            )

        (
            summary,
            dataset_issues,
        ) = (
            analyze_dataframe_quality(
                dataframe,

                dataset_id=
                    dataset_id,

                dataset_filename=
                    filename,
            )
        )

        summaries.append(
            summary
        )

        issues.extend(
            dataset_issues
        )

    important_count = sum(
        summary.important_count

        for summary
        in summaries
    )

    moderate_count = sum(
        summary.moderate_count

        for summary
        in summaries
    )

    minor_count = sum(
        summary.minor_count

        for summary
        in summaries
    )

    semantic_review_count = sum(
        issue.semantic_review_recommended

        for issue
        in issues
    )

    return (
        DataQualityReport(
            status=
                "ready",

            dataset_count=
                len(
                    summaries
                ),

            total_rows=
                sum(
                    summary.row_count

                    for summary
                    in summaries
                ),

            total_columns=
                sum(
                    summary.column_count

                    for summary
                    in summaries
                ),

            issue_count=
                len(
                    issues
                ),

            important_count=
                important_count,

            moderate_count=
                moderate_count,

            minor_count=
                minor_count,

            semantic_review_count=
                semantic_review_count,

            datasets=
                summaries,

            issues=
                issues,

            notes=[
                (
                    "This report is deterministic. "
                    "No LLM-generated cleaning action "
                    "has been executed."
                ),

                (
                    "Outliers are signals, not errors. "
                    "DataLens never deletes them automatically."
                ),

                (
                    "Category formatting variants are deterministic "
                    "quality signals. Their exact variant groups are "
                    "forwarded to guarded semantic canonicalization, "
                    "where user confirmation remains mandatory. "
                    "Bridge: "
                    f"{CATEGORY_VARIANT_BRIDGE_RULE_VERSION}."
                ),

                (
                    "Semantic aliases are candidates only. "
                    "They require contextual review before mapping."
                ),

                (
                    "For textual columns, numeric intent requires "
                    "strong value-level evidence. Column names are "
                    "supporting metadata only and cannot establish "
                    "numeric semantics by themselves."
                ),
            ],
        )
    )
import re
import unicodedata

import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from app.ingestion.schemas import (
    CorrelationCompatibility,
    DatasetColumnManifest,
    DatasetManifest,
)


INGESTION_ANALYSIS_KIND_RULE_VERSION = (
    "ingestion_analysis_kind_v0.2"
)


TEMPORAL_NAME_TOKENS = {
    "year",
    "annee",
    "date",
    "datetime",
    "timestamp",
    "month",
    "mois",
    "day",
    "jour",

    # Period-like names are accepted only when
    # their non-null values also parse safely as
    # datetimes. The name alone is not sufficient.
    "period",
    "periode",
}


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def normalize_column_name(
    column_name: str,
) -> str:
    normalized = (
        unicodedata
        .normalize(
            "NFKD",
            column_name,
        )
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii",
        )
        .lower()
    )

    return normalized


def column_name_tokens(
    column_name: str,
) -> set[
    str
]:
    normalized = (
        normalize_column_name(
            column_name
        )
    )

    return {
        token
        for token
        in re.split(
            r"[^a-z0-9]+",
            normalized,
        )
        if token
    }


def has_temporal_name_signal(
    column_name: str,
) -> bool:
    tokens = (
        column_name_tokens(
            column_name
        )
    )

    return bool(
        tokens
        &
        TEMPORAL_NAME_TOKENS
    )


def safely_parseable_temporal_text(
    series: pd.Series,
    column_name: str,
) -> bool:
    """
    Return True only when a string/object column has both:

    1. an explicit temporal signal in its column name;
    2. fully parseable non-null values.

    This intentionally avoids inferring time from arbitrary
    text merely because pandas can parse a few values.

    Examples accepted:
        fiscal_period -> 2025-01-01, 2025-02-01, ...
        snapshot_date -> 2025-01-31, 2025-02-28, ...

    Examples rejected:
        period_label -> P01, P02, ...
        date_status  -> open, closed, ...
    """

    if not has_temporal_name_signal(
        column_name
    ):
        return False


    non_null = (
        series
        .dropna()
    )


    if non_null.empty:
        return False


    normalized = (
        non_null
        .astype(
            "string"
        )
        .str
        .strip()
    )


    if (
        normalized
        .eq(
            ""
        )
        .any()
    ):
        return False


    parsed = pd.to_datetime(
        normalized,
        errors="coerce",
        format="mixed",
    )


    return bool(
        parsed
        .notna()
        .all()
    )


def infer_analysis_kind(
    series: pd.Series,
    column_name: str,
) -> tuple[
    str,
    str,
]:
    """
    Conservative compatibility inference.

    This function is intentionally narrower
    than the full DataLens semantic profiler.

    Its purpose is only to prevent the current
    correlation endpoint from receiving clearly
    incompatible columns.

    It must not be treated as the final semantic
    typing system.

    v0.2 additionally recognizes safely parseable
    temporal text columns when the schema name also
    carries a temporal signal.
    """

    if is_bool_dtype(
        series.dtype
    ):
        return (
            "boolean",
            (
                "The column uses a boolean "
                "dtype and is not automatically "
                "eligible for the current "
                "quantitative correlation "
                "branch."
            ),
        )

    if is_datetime64_any_dtype(
        series.dtype
    ):
        return (
            "temporal",
            (
                "The column uses a datetime "
                "dtype and is reserved for a "
                "temporal analysis family."
            ),
        )

    if is_numeric_dtype(
        series.dtype
    ):
        if (
            has_temporal_name_signal(
                column_name
            )
        ):
            return (
                "temporal",
                (
                    "The column is numeric but "
                    "its name contains a temporal "
                    "signal. DataLens therefore "
                    "does not automatically treat "
                    "it as a continuous variable "
                    "for correlation."
                ),
            )

        return (
            "quantitative",
            (
                "The column uses a numeric dtype "
                "and has no temporal-name signal. "
                "It is compatible with the "
                "current quantitative correlation "
                "branch."
            ),
        )

    if (
        is_string_dtype(
            series.dtype
        )
        or
        is_object_dtype(
            series.dtype
        )
        or
        str(
            series.dtype
        )
        ==
        "category"
    ):
        if (
            safely_parseable_temporal_text(
                series,
                column_name,
            )
        ):
            return (
                "temporal",
                (
                    "The column is string-like, "
                    "its name contains a temporal "
                    "signal and every non-null "
                    "value parses safely as a "
                    "datetime. DataLens therefore "
                    "classifies it as temporal "
                    "without altering the raw "
                    "uploaded values."
                ),
            )

        return (
            "categorical",
            (
                "The column is treated as "
                "categorical and is not eligible "
                "for the current quantitative "
                "correlation branch."
            ),
        )

    return (
        "unknown",
        (
            "The column dtype is not yet mapped "
            "to a supported analytical family."
        ),
    )


def build_correlation_compatibility(
    columns: list[
        DatasetColumnManifest
    ],
) -> CorrelationCompatibility:
    candidates = [
        column.name
        for column
        in columns
        if column.correlation_eligible
    ]

    temporal_columns = [
        column.name
        for column
        in columns
        if (
            column.analysis_kind
            ==
            "temporal"
        )
    ]

    categorical_columns = [
        column.name
        for column
        in columns
        if (
            column.analysis_kind
            ==
            "categorical"
        )
    ]

    reasons: list[
        str
    ] = []

    if (
        len(
            candidates
        )
        >= 2
    ):
        reasons.append(
            (
                "At least two quantitative "
                "columns are compatible with "
                "the current correlation engine."
            )
        )

        if temporal_columns:
            reasons.append(
                (
                    "Temporal columns were "
                    "excluded from automatic "
                    "correlation selection: "
                    + ", ".join(
                        temporal_columns
                    )
                    + "."
                )
            )

        return (
            CorrelationCompatibility(
                status=
                    "ready",

                candidate_columns=
                    candidates,

                default_x_column=
                    candidates[
                        0
                    ],

                default_y_column=
                    candidates[
                        1
                    ],

                reasons=
                    reasons,
            )
        )

    reasons.append(
        (
            "The current correlation engine "
            "requires at least two compatible "
            "quantitative columns."
        )
    )

    reasons.append(
        (
            f"{len(candidates)} compatible "
            "quantitative column(s) were "
            "detected."
        )
    )

    if temporal_columns:
        reasons.append(
            (
                "A temporal structure was "
                "detected through: "
                + ", ".join(
                    temporal_columns
                )
                + ". A dedicated time-series "
                "analysis family should handle "
                "these variables instead of "
                "forcing a correlation."
            )
        )

    if categorical_columns:
        reasons.append(
            (
                "Categorical variables were "
                "detected through: "
                + ", ".join(
                    categorical_columns
                )
                + ". They require a different "
                "statistical analysis family."
            )
        )

    return (
        CorrelationCompatibility(
            status=
                "not_available",

            candidate_columns=
                candidates,

            default_x_column=
                None,

            default_y_column=
                None,

            reasons=
                reasons,
        )
    )


def build_dataset_manifest(
    dataframe: pd.DataFrame,
    *,
    dataset_id: str,
    filename: str,
    extension: str,
) -> DatasetManifest:
    row_count = int(
        len(
            dataframe
        )
    )

    column_count = int(
        len(
            dataframe.columns
        )
    )

    memory_bytes = int(
        dataframe
        .memory_usage(
            deep=True
        )
        .sum()
    )

    columns: list[
        DatasetColumnManifest
    ] = []

    warnings: list[
        str
    ] = []

    for column_name in (
        dataframe.columns
    ):
        series = (
            dataframe[
                column_name
            ]
        )

        missing_count = int(
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

        non_missing_count = (
            row_count
            - missing_count
        )

        missing_ratio = (
            safe_ratio(
                missing_count,
                row_count,
            )
        )

        unique_ratio = (
            safe_ratio(
                unique_count,
                non_missing_count,
            )
        )

        unique_candidate = (
            row_count > 0
            and
            missing_count == 0
            and
            unique_count
            ==
            row_count
        )

        (
            analysis_kind,
            analysis_note,
        ) = (
            infer_analysis_kind(
                series,
                str(
                    column_name
                ),
            )
        )

        correlation_eligible = (
            analysis_kind
            ==
            "quantitative"
        )

        columns.append(
            DatasetColumnManifest(
                name=str(
                    column_name
                ),

                dtype=str(
                    series.dtype
                ),

                missing_count=
                    missing_count,

                missing_ratio=
                    missing_ratio,

                unique_count=
                    unique_count,

                unique_ratio=
                    unique_ratio,

                unique_candidate=
                    unique_candidate,

                analysis_kind=
                    analysis_kind,

                correlation_eligible=
                    correlation_eligible,

                analysis_note=
                    analysis_note,
            )
        )

    duplicate_rows = int(
        dataframe
        .duplicated()
        .sum()
    )

    if duplicate_rows > 0:
        warnings.append(
            (
                f"{duplicate_rows} exact "
                "duplicate row(s) were "
                "detected. No rows were "
                "removed during ingestion."
            )
        )

    if row_count < 2:
        warnings.append(
            (
                "The dataset contains fewer "
                "than two rows."
            )
        )

    correlation_compatibility = (
        build_correlation_compatibility(
            columns
        )
    )

    return DatasetManifest(
        dataset_id=
            dataset_id,

        filename=
            filename,

        extension=
            extension,

        row_count=
            row_count,

        column_count=
            column_count,

        memory_bytes=
            memory_bytes,

        columns=
            columns,

        correlation_compatibility=
            correlation_compatibility,

        warnings=
            warnings,
    )

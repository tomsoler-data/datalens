from __future__ import annotations

import hashlib

from dataclasses import dataclass

from typing import (
    Any,
    Literal,
)

import pandas as pd

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.preparation.approval import (
    ApprovedPreparationPlan,
    ApprovedPreparationStep,
)

from app.preparation.contracts import (
    PreparationAction,
)


# ============================================================
# VERSION
# ============================================================


CLEANING_EXECUTOR_RULE_VERSION = (
    "cleaning_executor_v0.1"
)


# ============================================================
# SUPPORTED ACTIONS
# ============================================================


SUPPORTED_ACTIONS = {
    PreparationAction.TRIM_WHITESPACE,

    PreparationAction.NORMALIZE_EMPTY_TO_MISSING,

    PreparationAction.NORMALIZE_MISSING_MARKERS,

    PreparationAction.NORMALIZE_CASE,

    PreparationAction.MERGE_CATEGORY_VALUES,

    PreparationAction.CONVERT_TO_NUMERIC,

    PreparationAction.DROP_ROWS_WITH_MISSING,

    PreparationAction.DROP_COLUMN,

    PreparationAction.IMPUTE_MEAN,

    PreparationAction.IMPUTE_MEDIAN,

    PreparationAction.IMPUTE_MODE,

    PreparationAction.CREATE_MISSING_CATEGORY,

    PreparationAction.DOMAIN_SPECIFIC_VALUE,

    PreparationAction.REMOVE_DUPLICATE_ROWS,

    PreparationAction.CAP_OUTLIERS,

    PreparationAction.REMOVE_OUTLIER_ROWS,
}


OPERATION_ACTION_MAP = {
    "trim_whitespace":
        PreparationAction.TRIM_WHITESPACE,

    "trim_string_values":
        PreparationAction.TRIM_WHITESPACE,

    "normalize_empty_to_missing":
        PreparationAction.NORMALIZE_EMPTY_TO_MISSING,

    "normalize_missing_values":
        PreparationAction.NORMALIZE_MISSING_MARKERS,

    "normalize_missing_markers":
        PreparationAction.NORMALIZE_MISSING_MARKERS,

    "normalize_case":
        PreparationAction.NORMALIZE_CASE,

    "convert_to_numeric":
        PreparationAction.CONVERT_TO_NUMERIC,

    "remove_exact_duplicates":
        PreparationAction.REMOVE_DUPLICATE_ROWS,

    "remove_duplicate_rows":
        PreparationAction.REMOVE_DUPLICATE_ROWS,

    "merge_values":
        PreparationAction.MERGE_CATEGORY_VALUES,
}


# ============================================================
# REPORT MODELS
# ============================================================


class CleaningStepExecution(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    execution_order: int = Field(
        ge=1
    )

    decision_id: str

    source_issue_id: str

    dataset_id: str

    dataset_filename: str

    column: (
        str
        | None
    ) = None

    action: PreparationAction

    parameters: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    status: Literal[
        "applied",
        "no_change",
    ]

    rows_before: int = Field(
        ge=0
    )

    rows_after: int = Field(
        ge=0
    )

    columns_before: int = Field(
        ge=0
    )

    columns_after: int = Field(
        ge=0
    )

    missing_cells_before: int = Field(
        ge=0
    )

    missing_cells_after: int = Field(
        ge=0
    )

    affected_row_count: int = Field(
        ge=0
    )

    affected_cell_count: int = Field(
        ge=0
    )

    fingerprint_before: str

    fingerprint_after: str

    message: str


class CleaningExecutionReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    dataset_id: str

    dataset_filename: str

    execution_step_count: int = Field(
        ge=0
    )

    applied_step_count: int = Field(
        ge=0
    )

    no_change_step_count: int = Field(
        ge=0
    )

    rows_before: int = Field(
        ge=0
    )

    rows_after: int = Field(
        ge=0
    )

    columns_before: int = Field(
        ge=0
    )

    columns_after: int = Field(
        ge=0
    )

    missing_cells_before: int = Field(
        ge=0
    )

    missing_cells_after: int = Field(
        ge=0
    )

    fingerprint_before: str

    fingerprint_after: str

    data_changed: bool

    steps: list[
        CleaningStepExecution
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        CLEANING_EXECUTOR_RULE_VERSION
    )


@dataclass(
    frozen=True
)
class CleaningExecutionResult:
    """
    Le DataFrame est volontairement séparé du rapport
    Pydantic afin que le rapport reste facilement
    sérialisable par l'API.
    """

    dataframe: pd.DataFrame

    report: CleaningExecutionReport


# ============================================================
# DATAFRAME HELPERS
# ============================================================


def _missing_cell_count(
    dataframe: pd.DataFrame,
) -> int:
    return int(
        dataframe
        .isna()
        .sum()
        .sum()
    )


def _dataframe_fingerprint(
    dataframe: pd.DataFrame,
) -> str:
    """
    Empreinte utilisée pour l'audit.

    Elle couvre :
    - les colonnes ;
    - les dtypes ;
    - l'index ;
    - les valeurs.
    """

    hasher = hashlib.sha256()

    schema = "|".join(
        (
            f"{column}:"
            f"{dataframe[column].dtype}"
        )
        for column in dataframe.columns
    )

    hasher.update(
        schema.encode(
            "utf-8"
        )
    )

    try:
        hashed = (
            pd.util
            .hash_pandas_object(
                dataframe,
                index=True,
            )
            .values
            .tobytes()
        )

        hasher.update(
            hashed
        )

    except Exception:
        # Fallback pour certaines colonnes objet
        # contenant des valeurs non hashables.
        fallback = (
            dataframe
            .astype(str)
            .to_csv(
                index=True
            )
        )

        hasher.update(
            fallback.encode(
                "utf-8"
            )
        )

    return hasher.hexdigest()


def _require_column(
    dataframe: pd.DataFrame,
    column: (
        str
        | None
    ),
    action: PreparationAction,
) -> str:
    if column is None:
        raise ValueError(
            (
                f"L'action {action.value} "
                "nécessite une colonne."
            )
        )

    if (
        column
        not in
        dataframe.columns
    ):
        raise ValueError(
            (
                f"Colonne inconnue pour "
                f"{action.value} : {column}"
            )
        )

    return column


def _require_parameter(
    *,
    parameters: dict[
        str,
        Any,
    ],
    name: str,
    action: PreparationAction,
) -> Any:
    if name not in parameters:
        raise ValueError(
            (
                f"L'action {action.value} "
                f"nécessite le paramètre "
                f"'{name}'."
            )
        )

    return parameters[
        name
    ]


# ============================================================
# STEP ACTION RESOLUTION
# ============================================================


def _resolve_step_action(
    step: ApprovedPreparationStep,
) -> PreparationAction:
    if step.approved_action is not None:
        action = (
            step.approved_action
        )

    elif step.approved_operation is not None:
        normalized_operation = (
            step.approved_operation
            .strip()
            .lower()
        )

        action = (
            OPERATION_ACTION_MAP
            .get(
                normalized_operation
            )
        )

        if action is None:
            raise ValueError(
                (
                    "Opération approuvée non supportée "
                    "par Cleaning Executor v0.1 : "
                    f"{step.approved_operation}"
                )
            )

    else:
        raise ValueError(
            (
                "Une étape executor_eligible doit "
                "contenir approved_action ou "
                "approved_operation."
            )
        )

    if (
        action
        not in
        SUPPORTED_ACTIONS
    ):
        raise ValueError(
            (
                "Action approuvée non supportée "
                "par Cleaning Executor v0.1 : "
                f"{action.value}"
            )
        )

    return action


# ============================================================
# STRING ACTIONS
# ============================================================


def _trim_whitespace(
    dataframe: pd.DataFrame,
    *,
    column: str,
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    output = dataframe.copy(
        deep=True
    )

    series = output[
        column
    ]

    mask = series.map(
        lambda value:
        (
            isinstance(
                value,
                str,
            )
            and
            value != value.strip()
        )
    )

    affected = int(
        mask.sum()
    )

    output.loc[
        mask,
        column,
    ] = (
        output.loc[
            mask,
            column,
        ]
        .map(
            lambda value:
            value.strip()
        )
    )

    return (
        output,
        affected,
        affected,
    )


def _normalize_empty_to_missing(
    dataframe: pd.DataFrame,
    *,
    column: str,
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    output = dataframe.copy(
        deep=True
    )

    series = output[
        column
    ]

    mask = series.map(
        lambda value:
        (
            isinstance(
                value,
                str,
            )
            and
            value.strip() == ""
        )
    )

    affected = int(
        mask.sum()
    )

    output.loc[
        mask,
        column,
    ] = pd.NA

    return (
        output,
        affected,
        affected,
    )


def _normalize_missing_markers(
    dataframe: pd.DataFrame,
    *,
    column: str,
    parameters: dict[
        str,
        Any,
    ],
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    markers = _require_parameter(
        parameters=
            parameters,

        name=
            "markers",

        action=
            PreparationAction
            .NORMALIZE_MISSING_MARKERS,
    )

    if (
        not isinstance(
            markers,
            list,
        )
        or
        not markers
    ):
        raise ValueError(
            (
                "NORMALIZE_MISSING_MARKERS "
                "nécessite une liste markers "
                "non vide."
            )
        )

    case_sensitive = bool(
        parameters.get(
            "case_sensitive",
            False,
        )
    )

    strip_values = bool(
        parameters.get(
            "strip",
            True,
        )
    )

    def normalize(
        value: Any,
    ) -> Any:
        if not isinstance(
            value,
            str,
        ):
            return value

        candidate = (
            value.strip()
            if strip_values
            else value
        )

        comparison_candidate = (
            candidate
            if case_sensitive
            else candidate.casefold()
        )

        comparison_markers = {
            (
                str(
                    marker
                )
                if case_sensitive
                else str(
                    marker
                ).casefold()
            )
            for marker in markers
        }

        if (
            comparison_candidate
            in comparison_markers
        ):
            return pd.NA

        return value

    output = dataframe.copy(
        deep=True
    )

    original_missing = (
        output[
            column
        ]
        .isna()
    )

    output[
        column
    ] = (
        output[
            column
        ]
        .map(
            normalize
        )
    )

    final_missing = (
        output[
            column
        ]
        .isna()
    )

    newly_missing = (
        final_missing
        &
        ~original_missing
    )

    affected = int(
        newly_missing.sum()
    )

    return (
        output,
        affected,
        affected,
    )


def _normalize_case(
    dataframe: pd.DataFrame,
    *,
    column: str,
    parameters: dict[
        str,
        Any,
    ],
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    case = _require_parameter(
        parameters=
            parameters,

        name=
            "case",

        action=
            PreparationAction
            .NORMALIZE_CASE,
    )

    transformations = {
        "lower":
            lambda value:
            value.lower(),

        "upper":
            lambda value:
            value.upper(),

        "title":
            lambda value:
            value.title(),

        "casefold":
            lambda value:
            value.casefold(),
    }

    transform = (
        transformations
        .get(
            case
        )
    )

    if transform is None:
        raise ValueError(
            (
                "NORMALIZE_CASE nécessite "
                "case parmi lower, upper, "
                "title ou casefold."
            )
        )

    output = dataframe.copy(
        deep=True
    )

    series = output[
        column
    ]

    mask = series.map(
        lambda value:
        (
            isinstance(
                value,
                str,
            )
            and
            transform(
                value
            )
            !=
            value
        )
    )

    affected = int(
        mask.sum()
    )

    output.loc[
        mask,
        column,
    ] = (
        output.loc[
            mask,
            column,
        ]
        .map(
            transform
        )
    )

    return (
        output,
        affected,
        affected,
    )


def _merge_category_values(
    dataframe: pd.DataFrame,
    *,
    column: str,
    parameters: dict[
        str,
        Any,
    ],
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    source_values = _require_parameter(
        parameters=
            parameters,

        name=
            "source_values",

        action=
            PreparationAction
            .MERGE_CATEGORY_VALUES,
    )

    canonical_value = _require_parameter(
        parameters=
            parameters,

        name=
            "canonical_value",

        action=
            PreparationAction
            .MERGE_CATEGORY_VALUES,
    )

    if (
        not isinstance(
            source_values,
            list,
        )
        or
        not source_values
    ):
        raise ValueError(
            (
                "source_values doit être "
                "une liste non vide."
            )
        )

    output = dataframe.copy(
        deep=True
    )

    mask = (
        output[
            column
        ]
        .isin(
            source_values
        )
        &
        (
            output[
                column
            ]
            !=
            canonical_value
        )
    )

    affected = int(
        mask.sum()
    )

    output.loc[
        mask,
        column,
    ] = canonical_value

    return (
        output,
        affected,
        affected,
    )


# ============================================================
# TYPE CONVERSION
# ============================================================


def _convert_to_numeric(
    dataframe: pd.DataFrame,
    *,
    column: str,
    parameters: dict[
        str,
        Any,
    ],
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    output = dataframe.copy(
        deep=True
    )

    original = output[
        column
    ]

    converted = pd.to_numeric(
        original,
        errors="coerce",
    )

    original_non_missing = (
        original.notna()
    )

    newly_missing = (
        original_non_missing
        &
        converted.isna()
    )

    allow_new_missing = bool(
        parameters.get(
            "allow_new_missing",
            False,
        )
    )

    if (
        newly_missing.any()
        and
        not allow_new_missing
    ):
        bad_values = (
            original[
                newly_missing
            ]
            .astype(str)
            .drop_duplicates()
            .head(
                10
            )
            .tolist()
        )

        raise ValueError(
            (
                "CONVERT_TO_NUMERIC créerait de "
                "nouvelles valeurs manquantes. "
                f"Exemples : {bad_values}"
            )
        )

    output[
        column
    ] = converted

    affected = int(
        original_non_missing.sum()
    )

    return (
        output,
        affected,
        affected,
    )


# ============================================================
# MISSING VALUES
# ============================================================


def _fill_missing(
    dataframe: pd.DataFrame,
    *,
    column: str,
    value: Any,
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    output = dataframe.copy(
        deep=True
    )

    mask = (
        output[
            column
        ]
        .isna()
    )

    affected = int(
        mask.sum()
    )

    output.loc[
        mask,
        column,
    ] = value

    return (
        output,
        affected,
        affected,
    )


def _impute_mean(
    dataframe: pd.DataFrame,
    *,
    column: str,
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    if not pd.api.types.is_numeric_dtype(
        dataframe[
            column
        ]
    ):
        raise ValueError(
            (
                "IMPUTE_MEAN nécessite "
                "une colonne numérique."
            )
        )

    value = (
        dataframe[
            column
        ]
        .mean()
    )

    if pd.isna(
        value
    ):
        raise ValueError(
            (
                "Impossible de calculer "
                "une moyenne exploitable."
            )
        )

    return _fill_missing(
        dataframe,
        column=
            column,
        value=
            value,
    )


def _impute_median(
    dataframe: pd.DataFrame,
    *,
    column: str,
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    if not pd.api.types.is_numeric_dtype(
        dataframe[
            column
        ]
    ):
        raise ValueError(
            (
                "IMPUTE_MEDIAN nécessite "
                "une colonne numérique."
            )
        )

    value = (
        dataframe[
            column
        ]
        .median()
    )

    if pd.isna(
        value
    ):
        raise ValueError(
            (
                "Impossible de calculer "
                "une médiane exploitable."
            )
        )

    return _fill_missing(
        dataframe,
        column=
            column,
        value=
            value,
    )


def _impute_mode(
    dataframe: pd.DataFrame,
    *,
    column: str,
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    modes = (
        dataframe[
            column
        ]
        .mode(
            dropna=True
        )
    )

    if modes.empty:
        raise ValueError(
            (
                "Impossible de calculer "
                "un mode exploitable."
            )
        )

    value = modes.iloc[
        0
    ]

    return _fill_missing(
        dataframe,
        column=
            column,
        value=
            value,
    )


# ============================================================
# STRUCTURAL ACTIONS
# ============================================================


def _drop_rows_with_missing(
    dataframe: pd.DataFrame,
    *,
    column: str,
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    mask = (
        dataframe[
            column
        ]
        .isna()
    )

    affected_rows = int(
        mask.sum()
    )

    affected_cells = (
        affected_rows
        *
        len(
            dataframe.columns
        )
    )

    output = (
        dataframe.loc[
            ~mask
        ]
        .copy(
            deep=True
        )
    )

    return (
        output,
        affected_rows,
        affected_cells,
    )


def _drop_column(
    dataframe: pd.DataFrame,
    *,
    column: str,
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    row_count = int(
        len(
            dataframe
        )
    )

    output = (
        dataframe
        .drop(
            columns=[
                column
            ]
        )
        .copy(
            deep=True
        )
    )

    return (
        output,
        row_count,
        row_count,
    )


def _remove_duplicate_rows(
    dataframe: pd.DataFrame,
    *,
    parameters: dict[
        str,
        Any,
    ],
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    subset = parameters.get(
        "subset"
    )

    keep = parameters.get(
        "keep",
        "first",
    )

    if (
        keep
        not in {
            "first",
            "last",
            False,
        }
    ):
        raise ValueError(
            (
                "REMOVE_DUPLICATE_ROWS nécessite "
                "keep parmi first, last ou False."
            )
        )

    duplicated = (
        dataframe
        .duplicated(
            subset=
                subset,

            keep=
                keep,
        )
    )

    affected_rows = int(
        duplicated.sum()
    )

    affected_cells = (
        affected_rows
        *
        len(
            dataframe.columns
        )
    )

    output = (
        dataframe.loc[
            ~duplicated
        ]
        .copy(
            deep=True
        )
    )

    return (
        output,
        affected_rows,
        affected_cells,
    )


# ============================================================
# OUTLIERS
# ============================================================


def _cap_outliers(
    dataframe: pd.DataFrame,
    *,
    column: str,
    parameters: dict[
        str,
        Any,
    ],
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    if not pd.api.types.is_numeric_dtype(
        dataframe[
            column
        ]
    ):
        raise ValueError(
            (
                "CAP_OUTLIERS nécessite "
                "une colonne numérique."
            )
        )

    lower_bound = parameters.get(
        "lower_bound"
    )

    upper_bound = parameters.get(
        "upper_bound"
    )

    if (
        lower_bound is None
        and
        upper_bound is None
    ):
        raise ValueError(
            (
                "CAP_OUTLIERS nécessite "
                "lower_bound ou upper_bound."
            )
        )

    output = dataframe.copy(
        deep=True
    )

    original = output[
        column
    ].copy()

    output[
        column
    ] = (
        output[
            column
        ]
        .clip(
            lower=
                lower_bound,

            upper=
                upper_bound,
        )
    )

    changed = (
        original.notna()
        &
        output[
            column
        ].notna()
        &
        (
            original
            !=
            output[
                column
            ]
        )
    )

    affected = int(
        changed.sum()
    )

    return (
        output,
        affected,
        affected,
    )


def _remove_outlier_rows(
    dataframe: pd.DataFrame,
    *,
    column: str,
    parameters: dict[
        str,
        Any,
    ],
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    row_indices = parameters.get(
        "row_indices"
    )

    if (
        isinstance(
            row_indices,
            list,
        )
        and
        row_indices
    ):
        mask = (
            dataframe
            .index
            .isin(
                row_indices
            )
        )

    else:
        comparator = parameters.get(
            "documented_comparator"
        )

        threshold = parameters.get(
            "documented_threshold"
        )

        if (
            comparator is None
            or
            threshold is None
        ):
            raise ValueError(
                (
                    "REMOVE_OUTLIER_ROWS nécessite "
                    "row_indices ou une règle "
                    "comparator + threshold."
                )
            )

        series = dataframe[
            column
        ]

        if not pd.api.types.is_numeric_dtype(
            series
        ):
            raise ValueError(
                (
                    "REMOVE_OUTLIER_ROWS avec seuil "
                    "nécessite une colonne numérique."
                )
            )

        if comparator == ">":
            mask = (
                series
                >
                threshold
            )

        elif comparator == ">=":
            mask = (
                series
                >=
                threshold
            )

        elif comparator == "<":
            mask = (
                series
                <
                threshold
            )

        elif comparator == "<=":
            mask = (
                series
                <=
                threshold
            )

        else:
            raise ValueError(
                (
                    "Comparateur non supporté : "
                    f"{comparator}"
                )
            )

        mask = (
            mask
            .fillna(
                False
            )
        )

    affected_rows = int(
        mask.sum()
    )

    affected_cells = (
        affected_rows
        *
        len(
            dataframe.columns
        )
    )

    output = (
        dataframe.loc[
            ~mask
        ]
        .copy(
            deep=True
        )
    )

    return (
        output,
        affected_rows,
        affected_cells,
    )


# ============================================================
# ACTION ROUTER
# ============================================================


def _apply_action(
    dataframe: pd.DataFrame,
    *,
    action: PreparationAction,
    column: (
        str
        | None
    ),
    parameters: dict[
        str,
        Any,
    ],
) -> tuple[
    pd.DataFrame,
    int,
    int,
]:
    if (
        action
        ==
        PreparationAction.REMOVE_DUPLICATE_ROWS
    ):
        return _remove_duplicate_rows(
            dataframe,
            parameters=
                parameters,
        )

    required_column = (
        _require_column(
            dataframe,
            column,
            action,
        )
    )

    if (
        action
        ==
        PreparationAction.TRIM_WHITESPACE
    ):
        return _trim_whitespace(
            dataframe,
            column=
                required_column,
        )

    if (
        action
        ==
        PreparationAction.NORMALIZE_EMPTY_TO_MISSING
    ):
        return _normalize_empty_to_missing(
            dataframe,
            column=
                required_column,
        )

    if (
        action
        ==
        PreparationAction.NORMALIZE_MISSING_MARKERS
    ):
        return _normalize_missing_markers(
            dataframe,
            column=
                required_column,

            parameters=
                parameters,
        )

    if (
        action
        ==
        PreparationAction.NORMALIZE_CASE
    ):
        return _normalize_case(
            dataframe,
            column=
                required_column,

            parameters=
                parameters,
        )

    if (
        action
        ==
        PreparationAction.MERGE_CATEGORY_VALUES
    ):
        return _merge_category_values(
            dataframe,
            column=
                required_column,

            parameters=
                parameters,
        )

    if (
        action
        ==
        PreparationAction.CONVERT_TO_NUMERIC
    ):
        return _convert_to_numeric(
            dataframe,
            column=
                required_column,

            parameters=
                parameters,
        )

    if (
        action
        ==
        PreparationAction.DOMAIN_SPECIFIC_VALUE
    ):
        value = _require_parameter(
            parameters=
                parameters,

            name=
                "value",

            action=
                action,
        )

        return _fill_missing(
            dataframe,
            column=
                required_column,

            value=
                value,
        )

    if (
        action
        ==
        PreparationAction.CREATE_MISSING_CATEGORY
    ):
        value = _require_parameter(
            parameters=
                parameters,

            name=
                "value",

            action=
                action,
        )

        if (
            not isinstance(
                value,
                str,
            )
            or
            not value.strip()
        ):
            raise ValueError(
                (
                    "CREATE_MISSING_CATEGORY "
                    "nécessite une valeur "
                    "textuelle non vide."
                )
            )

        return _fill_missing(
            dataframe,
            column=
                required_column,

            value=
                value,
        )

    if (
        action
        ==
        PreparationAction.IMPUTE_MEAN
    ):
        return _impute_mean(
            dataframe,
            column=
                required_column,
        )

    if (
        action
        ==
        PreparationAction.IMPUTE_MEDIAN
    ):
        return _impute_median(
            dataframe,
            column=
                required_column,
        )

    if (
        action
        ==
        PreparationAction.IMPUTE_MODE
    ):
        return _impute_mode(
            dataframe,
            column=
                required_column,
        )

    if (
        action
        ==
        PreparationAction.DROP_ROWS_WITH_MISSING
    ):
        return _drop_rows_with_missing(
            dataframe,
            column=
                required_column,
        )

    if (
        action
        ==
        PreparationAction.DROP_COLUMN
    ):
        return _drop_column(
            dataframe,
            column=
                required_column,
        )

    if (
        action
        ==
        PreparationAction.CAP_OUTLIERS
    ):
        return _cap_outliers(
            dataframe,
            column=
                required_column,

            parameters=
                parameters,
        )

    if (
        action
        ==
        PreparationAction.REMOVE_OUTLIER_ROWS
    ):
        return _remove_outlier_rows(
            dataframe,
            column=
                required_column,

            parameters=
                parameters,
        )

    raise ValueError(
        (
            "Action non supportée par "
            "Cleaning Executor v0.1 : "
            f"{action.value}"
        )
    )


# ============================================================
# PLAN VALIDATION
# ============================================================


def _validate_approved_plan(
    *,
    approved_plan: ApprovedPreparationPlan,
    dataset_id: str,
    dataset_filename: str,
) -> None:
    if not approved_plan.ready_for_execution:
        raise ValueError(
            (
                "Cleaning Executor refuse un "
                "ApprovedPreparationPlan avec "
                "ready_for_execution=False."
            )
        )

    if approved_plan.pending_count > 0:
        raise ValueError(
            (
                "Le plan contient encore "
                "des décisions PENDING."
            )
        )

    if approved_plan.deferred_count > 0:
        raise ValueError(
            (
                "Le plan contient encore "
                "des décisions DEFERRED."
            )
        )

    if (
        approved_plan
        .manual_followup_count
        >
        0
    ):
        raise ValueError(
            (
                "Le plan contient encore "
                "des investigations manuelles."
            )
        )

    for step in approved_plan.steps:
        if not step.resolved:
            raise ValueError(
                (
                    "Étape non résolue dans "
                    "un plan annoncé prêt : "
                    f"{step.decision_id}"
                )
            )

        if (
            step.dataset_id
            !=
            dataset_id
        ):
            raise ValueError(
                (
                    "Cleaning Executor v0.1 est "
                    "mono-dataset. Dataset inattendu "
                    f"dans {step.decision_id}: "
                    f"{step.dataset_id}"
                )
            )

        if (
            step.dataset_filename
            !=
            dataset_filename
        ):
            raise ValueError(
                (
                    "Nom de dataset incohérent dans "
                    f"{step.decision_id}: "
                    f"{step.dataset_filename}"
                )
            )

        if (
            step.requires_manual_followup
        ):
            raise ValueError(
                (
                    "Une étape nécessitant une "
                    "investigation manuelle ne peut "
                    "pas entrer dans l'executor."
                )
            )

        if (
            step.executor_eligible
            and
            not step.mutates_data
        ):
            raise ValueError(
                (
                    "Incohérence : une étape marquée "
                    "executor_eligible ne modifie "
                    "pas les données."
                )
            )


# ============================================================
# PUBLIC API
# ============================================================


def execute_cleaning_plan(
    *,
    dataframe: pd.DataFrame,
    approved_plan: ApprovedPreparationPlan,
    dataset_id: str,
    dataset_filename: str,
) -> CleaningExecutionResult:
    """
    Exécute uniquement les transformations explicitement
    autorisées dans un ApprovedPreparationPlan.

    Garanties v0.1 :

    - refuse ready_for_execution=False ;
    - mono-dataset ;
    - n'utilise jamais PreparationPlan directement ;
    - n'utilise jamais une sortie LLM ou RAG directement ;
    - travaille sur une copie du DataFrame ;
    - ne modifie jamais le DataFrame fourni ;
    - audit avant/après pour chaque opération ;
    - erreur sur toute action inconnue ;
    - aucune investigation manuelle n'est exécutable.

    Transaction :
    si une opération échoue, aucune version partiellement
    nettoyée n'est retournée au caller.
    """

    if dataframe is None:
        raise ValueError(
            (
                "Le DataFrame ne peut pas "
                "être None."
            )
        )

    _validate_approved_plan(
        approved_plan=
            approved_plan,

        dataset_id=
            dataset_id,

        dataset_filename=
            dataset_filename,
    )

    original = dataframe.copy(
        deep=True
    )

    working = dataframe.copy(
        deep=True
    )

    original_rows = int(
        len(
            original
        )
    )

    original_columns = int(
        len(
            original.columns
        )
    )

    original_missing = (
        _missing_cell_count(
            original
        )
    )

    original_fingerprint = (
        _dataframe_fingerprint(
            original
        )
    )

    execution_reports: list[
        CleaningStepExecution
    ] = []

    executable_steps = [
        step
        for step in (
            approved_plan.steps
        )
        if step.executor_eligible
    ]

    for (
        execution_order,
        step,
    ) in enumerate(
        executable_steps,
        start=1,
    ):
        action = (
            _resolve_step_action(
                step
            )
        )

        parameters = dict(
            step.approved_parameters
        )

        rows_before = int(
            len(
                working
            )
        )

        columns_before = int(
            len(
                working.columns
            )
        )

        missing_before = (
            _missing_cell_count(
                working
            )
        )

        fingerprint_before = (
            _dataframe_fingerprint(
                working
            )
        )

        (
            updated,
            affected_rows,
            affected_cells,
        ) = _apply_action(
            working,
            action=
                action,

            column=
                step.column,

            parameters=
                parameters,
        )

        rows_after = int(
            len(
                updated
            )
        )

        columns_after = int(
            len(
                updated.columns
            )
        )

        missing_after = (
            _missing_cell_count(
                updated
            )
        )

        fingerprint_after = (
            _dataframe_fingerprint(
                updated
            )
        )

        changed = (
            fingerprint_before
            !=
            fingerprint_after
        )

        execution_reports.append(
            CleaningStepExecution(
                execution_order=
                    execution_order,

                decision_id=
                    step.decision_id,

                source_issue_id=
                    step.source_issue_id,

                dataset_id=
                    step.dataset_id,

                dataset_filename=
                    step.dataset_filename,

                column=
                    step.column,

                action=
                    action,

                parameters=
                    parameters,

                status=(
                    "applied"
                    if changed
                    else
                    "no_change"
                ),

                rows_before=
                    rows_before,

                rows_after=
                    rows_after,

                columns_before=
                    columns_before,

                columns_after=
                    columns_after,

                missing_cells_before=
                    missing_before,

                missing_cells_after=
                    missing_after,

                affected_row_count=
                    affected_rows,

                affected_cell_count=
                    affected_cells,

                fingerprint_before=
                    fingerprint_before,

                fingerprint_after=
                    fingerprint_after,

                message=(
                    (
                        "Transformation appliquée."
                    )
                    if changed
                    else
                    (
                        "Transformation approuvée "
                        "mais aucune valeur n'avait "
                        "besoin d'être modifiée."
                    )
                ),
            )
        )

        working = updated

    final_rows = int(
        len(
            working
        )
    )

    final_columns = int(
        len(
            working.columns
        )
    )

    final_missing = (
        _missing_cell_count(
            working
        )
    )

    final_fingerprint = (
        _dataframe_fingerprint(
            working
        )
    )

    applied_count = sum(
        1
        for step in execution_reports
        if step.status == "applied"
    )

    no_change_count = sum(
        1
        for step in execution_reports
        if step.status == "no_change"
    )

    report = (
        CleaningExecutionReport(
            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            execution_step_count=
                len(
                    execution_reports
                ),

            applied_step_count=
                applied_count,

            no_change_step_count=
                no_change_count,

            rows_before=
                original_rows,

            rows_after=
                final_rows,

            columns_before=
                original_columns,

            columns_after=
                final_columns,

            missing_cells_before=
                original_missing,

            missing_cells_after=
                final_missing,

            fingerprint_before=
                original_fingerprint,

            fingerprint_after=
                final_fingerprint,

            data_changed=(
                original_fingerprint
                !=
                final_fingerprint
            ),

            steps=
                execution_reports,

            notes=[
                (
                    "Cleaning Executor v0.1 "
                    "n'accepte qu'un "
                    "ApprovedPreparationPlan."
                ),
                (
                    "Le DataFrame source n'est "
                    "jamais modifié en place."
                ),
                (
                    "Chaque transformation possède "
                    "une empreinte avant/après."
                ),
                (
                    "Les décisions KEEP/REJECT "
                    "résolues ne sont pas transmises "
                    "à l'executor."
                ),
                (
                    "Les investigations manuelles "
                    "ne sont jamais exécutées comme "
                    "des transformations."
                ),
                (
                    "Cleaning Executor v0.1 est "
                    "volontairement mono-dataset."
                ),
            ],

            rule_version=
                CLEANING_EXECUTOR_RULE_VERSION,
        )
    )

    return (
        CleaningExecutionResult(
            dataframe=
                working,

            report=
                report,
        )
    )
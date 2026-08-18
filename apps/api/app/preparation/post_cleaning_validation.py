from __future__ import annotations

import hashlib

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
)

from app.preparation.cleaning_executor import (
    CleaningExecutionReport,
)

from app.preparation.contracts import (
    PreparationAction,
)


# ============================================================
# VERSION
# ============================================================


POST_CLEANING_VALIDATION_RULE_VERSION = (
    "post_cleaning_validation_v0.1"
)


# ============================================================
# MODELS
# ============================================================


class ValidationCheck(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    check_id: str

    scope: Literal[
        "dataset",
        "execution",
        "step",
    ]

    status: Literal[
        "passed",
        "failed",
        "warning",
    ]

    message: str

    decision_id: (
        str
        | None
    ) = None

    column: (
        str
        | None
    ) = None

    action: (
        PreparationAction
        | None
    ) = None

    evidence: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class StepValidationResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    decision_id: str

    source_issue_id: str

    column: (
        str
        | None
    ) = None

    action: PreparationAction

    validation_status: Literal[
        "passed",
        "failed",
        "warning",
    ]

    check_count: int = Field(
        ge=0
    )

    passed_check_count: int = Field(
        ge=0
    )

    failed_check_count: int = Field(
        ge=0
    )

    warning_check_count: int = Field(
        ge=0
    )

    checks: list[
        ValidationCheck
    ] = Field(
        default_factory=list
    )


class PostCleaningValidationReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal[
        "passed",
        "failed",
    ]

    dataset_id: str

    dataset_filename: str

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

    dataset_check_count: int = Field(
        ge=0
    )

    step_validation_count: int = Field(
        ge=0
    )

    passed_check_count: int = Field(
        ge=0
    )

    failed_check_count: int = Field(
        ge=0
    )

    warning_check_count: int = Field(
        ge=0
    )

    valid_for_analysis: bool

    dataset_checks: list[
        ValidationCheck
    ] = Field(
        default_factory=list
    )

    step_validations: list[
        StepValidationResult
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        POST_CLEANING_VALIDATION_RULE_VERSION
    )


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
    Même principe que Cleaning Executor.

    L'empreinte couvre :
    - schéma ;
    - dtypes ;
    - index ;
    - valeurs.
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


def _column_exists(
    dataframe: pd.DataFrame,
    column: (
        str
        | None
    ),
) -> bool:
    return (
        column is not None
        and
        column in dataframe.columns
    )


def _count_surrounding_whitespace(
    series: pd.Series,
) -> int:
    return int(
        series.map(
            lambda value:
            (
                isinstance(
                    value,
                    str,
                )
                and
                value != value.strip()
            )
        ).sum()
    )


def _count_empty_strings(
    series: pd.Series,
) -> int:
    return int(
        series.map(
            lambda value:
            (
                isinstance(
                    value,
                    str,
                )
                and
                value.strip() == ""
            )
        ).sum()
    )


# ============================================================
# CHECK HELPERS
# ============================================================


def _passed(
    *,
    check_id: str,
    scope: Literal[
        "dataset",
        "execution",
        "step",
    ],
    message: str,
    decision_id: (
        str
        | None
    ) = None,
    column: (
        str
        | None
    ) = None,
    action: (
        PreparationAction
        | None
    ) = None,
    evidence: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None,
) -> ValidationCheck:
    return (
        ValidationCheck(
            check_id=
                check_id,

            scope=
                scope,

            status=
                "passed",

            message=
                message,

            decision_id=
                decision_id,

            column=
                column,

            action=
                action,

            evidence=
                dict(
                    evidence
                    or
                    {}
                ),
        )
    )


def _failed(
    *,
    check_id: str,
    scope: Literal[
        "dataset",
        "execution",
        "step",
    ],
    message: str,
    decision_id: (
        str
        | None
    ) = None,
    column: (
        str
        | None
    ) = None,
    action: (
        PreparationAction
        | None
    ) = None,
    evidence: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None,
) -> ValidationCheck:
    return (
        ValidationCheck(
            check_id=
                check_id,

            scope=
                scope,

            status=
                "failed",

            message=
                message,

            decision_id=
                decision_id,

            column=
                column,

            action=
                action,

            evidence=
                dict(
                    evidence
                    or
                    {}
                ),
        )
    )


def _warning(
    *,
    check_id: str,
    scope: Literal[
        "dataset",
        "execution",
        "step",
    ],
    message: str,
    decision_id: (
        str
        | None
    ) = None,
    column: (
        str
        | None
    ) = None,
    action: (
        PreparationAction
        | None
    ) = None,
    evidence: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None,
) -> ValidationCheck:
    return (
        ValidationCheck(
            check_id=
                check_id,

            scope=
                scope,

            status=
                "warning",

            message=
                message,

            decision_id=
                decision_id,

            column=
                column,

            action=
                action,

            evidence=
                dict(
                    evidence
                    or
                    {}
                ),
        )
    )


# ============================================================
# DATASET-LEVEL VALIDATION
# ============================================================


def _validate_dataset_integrity(
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    approved_plan: ApprovedPreparationPlan,
    execution_report: CleaningExecutionReport,
    dataset_id: str,
    dataset_filename: str,
) -> list[
    ValidationCheck
]:
    checks: list[
        ValidationCheck
    ] = []

    # --------------------------------------------------------
    # Approved plan readiness
    # --------------------------------------------------------

    if approved_plan.ready_for_execution:
        checks.append(
            _passed(
                check_id=
                    "approved_plan_ready",

                scope=
                    "execution",

                message=(
                    "Le plan approuvé était prêt "
                    "pour l'exécution."
                ),
            )
        )

    else:
        checks.append(
            _failed(
                check_id=
                    "approved_plan_ready",

                scope=
                    "execution",

                message=(
                    "Le plan approuvé n'était pas "
                    "prêt pour l'exécution."
                ),
            )
        )

    # --------------------------------------------------------
    # Dataset identity
    # --------------------------------------------------------

    if (
        execution_report.dataset_id
        ==
        dataset_id
    ):
        checks.append(
            _passed(
                check_id=
                    "dataset_id_matches",

                scope=
                    "execution",

                message=(
                    "Le dataset_id du rapport "
                    "d'exécution correspond."
                ),
            )
        )

    else:
        checks.append(
            _failed(
                check_id=
                    "dataset_id_matches",

                scope=
                    "execution",

                message=(
                    "Le dataset_id du rapport "
                    "d'exécution ne correspond pas."
                ),

                evidence={
                    "expected":
                        dataset_id,

                    "observed":
                        execution_report
                        .dataset_id,
                },
            )
        )

    if (
        execution_report.dataset_filename
        ==
        dataset_filename
    ):
        checks.append(
            _passed(
                check_id=
                    "dataset_filename_matches",

                scope=
                    "execution",

                message=(
                    "Le nom du dataset correspond "
                    "au rapport d'exécution."
                ),
            )
        )

    else:
        checks.append(
            _failed(
                check_id=
                    "dataset_filename_matches",

                scope=
                    "execution",

                message=(
                    "Le nom du dataset ne correspond "
                    "pas au rapport d'exécution."
                ),

                evidence={
                    "expected":
                        dataset_filename,

                    "observed":
                        execution_report
                        .dataset_filename,
                },
            )
        )

    # --------------------------------------------------------
    # Before-state reconciliation
    # --------------------------------------------------------

    actual_before_fingerprint = (
        _dataframe_fingerprint(
            before
        )
    )

    if (
        actual_before_fingerprint
        ==
        execution_report
        .fingerprint_before
    ):
        checks.append(
            _passed(
                check_id=
                    "before_fingerprint_matches",

                scope=
                    "execution",

                message=(
                    "Le dataset avant nettoyage "
                    "correspond exactement à celui "
                    "utilisé par l'executor."
                ),
            )
        )

    else:
        checks.append(
            _failed(
                check_id=
                    "before_fingerprint_matches",

                scope=
                    "execution",

                message=(
                    "Le dataset fourni comme état "
                    "initial ne correspond pas au "
                    "rapport d'exécution."
                ),
            )
        )

    # --------------------------------------------------------
    # After-state reconciliation
    # --------------------------------------------------------

    actual_after_fingerprint = (
        _dataframe_fingerprint(
            after
        )
    )

    if (
        actual_after_fingerprint
        ==
        execution_report
        .fingerprint_after
    ):
        checks.append(
            _passed(
                check_id=
                    "after_fingerprint_matches",

                scope=
                    "execution",

                message=(
                    "Le dataset final correspond "
                    "exactement à la sortie auditée "
                    "par Cleaning Executor."
                ),
            )
        )

    else:
        checks.append(
            _failed(
                check_id=
                    "after_fingerprint_matches",

                scope=
                    "execution",

                message=(
                    "Le dataset final a été modifié "
                    "ou ne correspond pas au rapport "
                    "d'exécution."
                ),
            )
        )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    actual_rows_after = int(
        len(
            after
        )
    )

    if (
        actual_rows_after
        ==
        execution_report.rows_after
    ):
        checks.append(
            _passed(
                check_id=
                    "row_count_matches",

                scope=
                    "dataset",

                message=(
                    "Le nombre final de lignes "
                    "correspond au rapport d'exécution."
                ),

                evidence={
                    "rows":
                        actual_rows_after,
                },
            )
        )

    else:
        checks.append(
            _failed(
                check_id=
                    "row_count_matches",

                scope=
                    "dataset",

                message=(
                    "Le nombre final de lignes "
                    "ne correspond pas au rapport."
                ),

                evidence={
                    "expected":
                        execution_report.rows_after,

                    "observed":
                        actual_rows_after,
                },
            )
        )

    actual_columns_after = int(
        len(
            after.columns
        )
    )

    if (
        actual_columns_after
        ==
        execution_report.columns_after
    ):
        checks.append(
            _passed(
                check_id=
                    "column_count_matches",

                scope=
                    "dataset",

                message=(
                    "Le nombre final de colonnes "
                    "correspond au rapport d'exécution."
                ),
            )
        )

    else:
        checks.append(
            _failed(
                check_id=
                    "column_count_matches",

                scope=
                    "dataset",

                message=(
                    "Le nombre final de colonnes "
                    "ne correspond pas au rapport."
                ),
            )
        )

    # --------------------------------------------------------
    # Missing cells reconciliation
    # --------------------------------------------------------

    actual_missing_after = (
        _missing_cell_count(
            after
        )
    )

    if (
        actual_missing_after
        ==
        execution_report
        .missing_cells_after
    ):
        checks.append(
            _passed(
                check_id=
                    "missing_count_matches",

                scope=
                    "dataset",

                message=(
                    "Le nombre final de cellules "
                    "manquantes correspond au rapport."
                ),

                evidence={
                    "missing_cells":
                        actual_missing_after,
                },
            )
        )

    else:
        checks.append(
            _failed(
                check_id=
                    "missing_count_matches",

                scope=
                    "dataset",

                message=(
                    "Le nombre final de cellules "
                    "manquantes ne correspond pas "
                    "au rapport."
                ),
            )
        )

    # --------------------------------------------------------
    # Duplicate column names
    # --------------------------------------------------------

    duplicated_columns = [
        str(
            after.columns[
                index
            ]
        )
        for (
            index,
            duplicated,
        ) in enumerate(
            after.columns
            .duplicated()
        )
        if duplicated
    ]

    if not duplicated_columns:
        checks.append(
            _passed(
                check_id=
                    "no_duplicate_column_names",

                scope=
                    "dataset",

                message=(
                    "Aucun nom de colonne dupliqué "
                    "n'est présent après nettoyage."
                ),
            )
        )

    else:
        checks.append(
            _failed(
                check_id=
                    "no_duplicate_column_names",

                scope=
                    "dataset",

                message=(
                    "Des noms de colonnes dupliqués "
                    "sont présents après nettoyage."
                ),

                evidence={
                    "columns":
                        duplicated_columns,
                },
            )
        )

    # --------------------------------------------------------
    # Execution-step reconciliation
    # --------------------------------------------------------

    expected_execution_steps = [
        step
        for step in approved_plan.steps
        if step.executor_eligible
    ]

    if (
        len(
            expected_execution_steps
        )
        ==
        execution_report
        .execution_step_count
    ):
        checks.append(
            _passed(
                check_id=
                    "execution_step_count_matches",

                scope=
                    "execution",

                message=(
                    "Toutes les étapes exécutables "
                    "attendues apparaissent dans "
                    "le rapport d'exécution."
                ),
            )
        )

    else:
        checks.append(
            _failed(
                check_id=
                    "execution_step_count_matches",

                scope=
                    "execution",

                message=(
                    "Le nombre d'étapes exécutées "
                    "ne correspond pas au plan approuvé."
                ),

                evidence={
                    "expected":
                        len(
                            expected_execution_steps
                        ),

                    "observed":
                        execution_report
                        .execution_step_count,
                },
            )
        )

    return checks


# ============================================================
# STEP POSTCONDITIONS
# ============================================================


def _validate_step_postcondition(
    *,
    after: pd.DataFrame,
    decision_id: str,
    source_issue_id: str,
    column: (
        str
        | None
    ),
    action: PreparationAction,
    parameters: dict[
        str,
        Any,
    ],
) -> StepValidationResult:
    checks: list[
        ValidationCheck
    ] = []

    def passed(
        check_id: str,
        message: str,
        evidence: (
            dict[
                str,
                Any,
            ]
            | None
        ) = None,
    ) -> None:
        checks.append(
            _passed(
                check_id=
                    check_id,

                scope=
                    "step",

                message=
                    message,

                decision_id=
                    decision_id,

                column=
                    column,

                action=
                    action,

                evidence=
                    evidence,
            )
        )

    def failed(
        check_id: str,
        message: str,
        evidence: (
            dict[
                str,
                Any,
            ]
            | None
        ) = None,
    ) -> None:
        checks.append(
            _failed(
                check_id=
                    check_id,

                scope=
                    "step",

                message=
                    message,

                decision_id=
                    decision_id,

                column=
                    column,

                action=
                    action,

                evidence=
                    evidence,
            )
        )

    # ========================================================
    # DROP COLUMN
    # ========================================================

    if (
        action
        ==
        PreparationAction.DROP_COLUMN
    ):
        if (
            column is not None
            and
            column not in after.columns
        ):
            passed(
                "drop_column_postcondition",
                (
                    "La colonne approuvée "
                    "a bien été supprimée."
                ),
            )

        else:
            failed(
                "drop_column_postcondition",
                (
                    "La colonne approuvée "
                    "est toujours présente."
                ),
            )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    elif (
        action
        ==
        PreparationAction.REMOVE_DUPLICATE_ROWS
    ):
        subset = parameters.get(
            "subset"
        )

        remaining_duplicates = int(
            after
            .duplicated(
                subset=
                    subset,
            )
            .sum()
        )

        if remaining_duplicates == 0:
            passed(
                "duplicate_rows_postcondition",
                (
                    "Aucun doublon correspondant "
                    "à la règle approuvée ne reste."
                ),
            )

        else:
            failed(
                "duplicate_rows_postcondition",
                (
                    "Des doublons restent après "
                    "l'opération approuvée."
                ),

                {
                    "remaining_duplicates":
                        remaining_duplicates,
                },
            )

    # ========================================================
    # ALL OTHER ACTIONS REQUIRE COLUMN
    # ========================================================

    elif (
        column is None
        or
        column not in after.columns
    ):
        failed(
            "column_available_postcondition",
            (
                "La colonne nécessaire à la "
                "validation n'existe plus."
            ),
        )

    # ========================================================
    # FILL / IMPUTATION
    # ========================================================

    elif action in {
        PreparationAction.DOMAIN_SPECIFIC_VALUE,
        PreparationAction.CREATE_MISSING_CATEGORY,
        PreparationAction.IMPUTE_MEAN,
        PreparationAction.IMPUTE_MEDIAN,
        PreparationAction.IMPUTE_MODE,
    }:
        remaining_missing = int(
            after[
                column
            ]
            .isna()
            .sum()
        )

        if remaining_missing == 0:
            passed(
                "missing_values_resolved",
                (
                    "Aucune valeur manquante "
                    "ne reste dans la colonne."
                ),
            )

        else:
            failed(
                "missing_values_resolved",
                (
                    "Des valeurs manquantes "
                    "restent dans la colonne."
                ),

                {
                    "remaining_missing":
                        remaining_missing,
                },
            )

    # ========================================================
    # TRIM
    # ========================================================

    elif (
        action
        ==
        PreparationAction.TRIM_WHITESPACE
    ):
        remaining = (
            _count_surrounding_whitespace(
                after[
                    column
                ]
            )
        )

        if remaining == 0:
            passed(
                "trim_whitespace_postcondition",
                (
                    "Aucun espace périphérique "
                    "ne reste."
                ),
            )

        else:
            failed(
                "trim_whitespace_postcondition",
                (
                    "Des espaces périphériques "
                    "restent après nettoyage."
                ),

                {
                    "remaining":
                        remaining,
                },
            )

    # ========================================================
    # EMPTY -> MISSING
    # ========================================================

    elif (
        action
        ==
        PreparationAction.NORMALIZE_EMPTY_TO_MISSING
    ):
        remaining = (
            _count_empty_strings(
                after[
                    column
                ]
            )
        )

        if remaining == 0:
            passed(
                "empty_string_postcondition",
                (
                    "Aucune chaîne vide "
                    "ne reste dans la colonne."
                ),
            )

        else:
            failed(
                "empty_string_postcondition",
                (
                    "Des chaînes vides restent "
                    "dans la colonne."
                ),

                {
                    "remaining":
                        remaining,
                },
            )

    # ========================================================
    # MISSING MARKERS
    # ========================================================

    elif (
        action
        ==
        PreparationAction.NORMALIZE_MISSING_MARKERS
    ):
        markers = parameters.get(
            "markers",
            [],
        )

        normalized_markers = {
            str(
                marker
            )
            .strip()
            .casefold()

            for marker in markers
        }

        remaining = int(
            after[
                column
            ]
            .map(
                lambda value:
                (
                    isinstance(
                        value,
                        str,
                    )
                    and
                    value.strip().casefold()
                    in normalized_markers
                )
            )
            .sum()
        )

        if remaining == 0:
            passed(
                "missing_markers_postcondition",
                (
                    "Les marqueurs de données "
                    "manquantes ont été normalisés."
                ),
            )

        else:
            failed(
                "missing_markers_postcondition",
                (
                    "Des marqueurs de données "
                    "manquantes restent présents."
                ),

                {
                    "remaining":
                        remaining,
                },
            )

    # ========================================================
    # CASE
    # ========================================================

    elif (
        action
        ==
        PreparationAction.NORMALIZE_CASE
    ):
        case = parameters.get(
            "case"
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

        transform = transformations.get(
            case
        )

        if transform is None:
            failed(
                "normalize_case_postcondition",
                (
                    "Paramètre de casse "
                    "non reconnu."
                ),
            )

        else:
            remaining = int(
                after[
                    column
                ]
                .map(
                    lambda value:
                    (
                        isinstance(
                            value,
                            str,
                        )
                        and
                        value
                        !=
                        transform(
                            value
                        )
                    )
                )
                .sum()
            )

            if remaining == 0:
                passed(
                    "normalize_case_postcondition",
                    (
                        "La casse est conforme "
                        "à la règle approuvée."
                    ),
                )

            else:
                failed(
                    "normalize_case_postcondition",
                    (
                        "Certaines valeurs ne sont "
                        "pas conformes à la casse "
                        "approuvée."
                    ),

                    {
                        "remaining":
                            remaining,
                    },
                )

    # ========================================================
    # CATEGORY MERGE
    # ========================================================

    elif (
        action
        ==
        PreparationAction.MERGE_CATEGORY_VALUES
    ):
        source_values = parameters.get(
            "source_values",
            [],
        )

        canonical_value = parameters.get(
            "canonical_value"
        )

        values_to_remove = [
            value
            for value in source_values
            if value != canonical_value
        ]

        remaining = int(
            after[
                column
            ]
            .isin(
                values_to_remove
            )
            .sum()
        )

        if remaining == 0:
            passed(
                "merge_category_postcondition",
                (
                    "Les anciennes variantes "
                    "fusionnées ne sont plus présentes."
                ),
            )

        else:
            failed(
                "merge_category_postcondition",
                (
                    "Certaines variantes supposées "
                    "fusionnées sont encore présentes."
                ),

                {
                    "remaining":
                        remaining,
                },
            )

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    elif (
        action
        ==
        PreparationAction.CONVERT_TO_NUMERIC
    ):
        if pd.api.types.is_numeric_dtype(
            after[
                column
            ]
        ):
            passed(
                "numeric_conversion_postcondition",
                (
                    "La colonne possède maintenant "
                    "un dtype numérique."
                ),
            )

        else:
            failed(
                "numeric_conversion_postcondition",
                (
                    "La colonne n'est pas "
                    "de type numérique."
                ),

                {
                    "dtype":
                        str(
                            after[
                                column
                            ].dtype
                        ),
                },
            )

    # ========================================================
    # DROP ROWS WITH MISSING
    # ========================================================

    elif (
        action
        ==
        PreparationAction.DROP_ROWS_WITH_MISSING
    ):
        remaining = int(
            after[
                column
            ]
            .isna()
            .sum()
        )

        if remaining == 0:
            passed(
                "drop_missing_rows_postcondition",
                (
                    "Aucune ligne avec valeur "
                    "manquante ne reste pour "
                    "cette colonne."
                ),
            )

        else:
            failed(
                "drop_missing_rows_postcondition",
                (
                    "Des valeurs manquantes restent "
                    "dans la colonne après suppression."
                ),
            )

    # ========================================================
    # CAP OUTLIERS
    # ========================================================

    elif (
        action
        ==
        PreparationAction.CAP_OUTLIERS
    ):
        lower = parameters.get(
            "lower_bound"
        )

        upper = parameters.get(
            "upper_bound"
        )

        violations = 0

        if lower is not None:
            violations += int(
                (
                    after[
                        column
                    ]
                    <
                    lower
                )
                .fillna(
                    False
                )
                .sum()
            )

        if upper is not None:
            violations += int(
                (
                    after[
                        column
                    ]
                    >
                    upper
                )
                .fillna(
                    False
                )
                .sum()
            )

        if violations == 0:
            passed(
                "cap_outliers_postcondition",
                (
                    "Toutes les valeurs respectent "
                    "les bornes approuvées."
                ),
            )

        else:
            failed(
                "cap_outliers_postcondition",
                (
                    "Certaines valeurs dépassent "
                    "encore les bornes approuvées."
                ),

                {
                    "violations":
                        violations,
                },
            )

    # ========================================================
    # REMOVE OUTLIERS
    # ========================================================

    elif (
        action
        ==
        PreparationAction.REMOVE_OUTLIER_ROWS
    ):
        comparator = parameters.get(
            "documented_comparator"
        )

        threshold = parameters.get(
            "documented_threshold"
        )

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
            remaining_indices = [
                index
                for index in row_indices
                if index in after.index
            ]

            if not remaining_indices:
                passed(
                    "remove_outlier_rows_postcondition",
                    (
                        "Les lignes approuvées pour "
                        "suppression ne sont plus présentes."
                    ),
                )

            else:
                failed(
                    "remove_outlier_rows_postcondition",
                    (
                        "Certaines lignes approuvées "
                        "pour suppression sont toujours présentes."
                    ),

                    {
                        "remaining_indices":
                            remaining_indices,
                    },
                )

        elif (
            comparator is not None
            and
            threshold is not None
        ):
            series = after[
                column
            ]

            if comparator == ">":
                violations = (
                    series
                    >
                    threshold
                )

            elif comparator == ">=":
                violations = (
                    series
                    >=
                    threshold
                )

            elif comparator == "<":
                violations = (
                    series
                    <
                    threshold
                )

            elif comparator == "<=":
                violations = (
                    series
                    <=
                    threshold
                )

            else:
                violations = None

            if violations is None:
                failed(
                    "remove_outlier_rows_postcondition",
                    (
                        "Comparateur non reconnu."
                    ),
                )

            else:
                remaining = int(
                    violations
                    .fillna(
                        False
                    )
                    .sum()
                )

                if remaining == 0:
                    passed(
                        "remove_outlier_rows_postcondition",
                        (
                            "Aucune observation violant "
                            "la règle approuvée ne reste."
                        ),
                    )

                else:
                    failed(
                        "remove_outlier_rows_postcondition",
                        (
                            "Certaines observations violant "
                            "la règle approuvée restent présentes."
                        ),

                        {
                            "remaining":
                                remaining,
                        },
                    )

        else:
            failed(
                "remove_outlier_rows_postcondition",
                (
                    "Impossible de reconstruire "
                    "la condition de suppression."
                ),
            )

    # ========================================================
    # UNKNOWN
    # ========================================================

    else:
        checks.append(
            _warning(
                check_id=
                    "unsupported_postcondition",

                scope=
                    "step",

                message=(
                    "Aucune postcondition spécifique "
                    "n'est encore définie pour "
                    "cette action."
                ),

                decision_id=
                    decision_id,

                column=
                    column,

                action=
                    action,
            )
        )

    passed_count = sum(
        1
        for check in checks
        if check.status == "passed"
    )

    failed_count = sum(
        1
        for check in checks
        if check.status == "failed"
    )

    warning_count = sum(
        1
        for check in checks
        if check.status == "warning"
    )

    if failed_count > 0:
        validation_status = "failed"

    elif warning_count > 0:
        validation_status = "warning"

    else:
        validation_status = "passed"

    return (
        StepValidationResult(
            decision_id=
                decision_id,

            source_issue_id=
                source_issue_id,

            column=
                column,

            action=
                action,

            validation_status=
                validation_status,

            check_count=
                len(
                    checks
                ),

            passed_check_count=
                passed_count,

            failed_check_count=
                failed_count,

            warning_check_count=
                warning_count,

            checks=
                checks,
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def validate_post_cleaning(
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    approved_plan: ApprovedPreparationPlan,
    execution_report: CleaningExecutionReport,
    dataset_id: str,
    dataset_filename: str,
) -> PostCleaningValidationReport:
    """
    Valide le résultat du Cleaning Executor.

    Cette fonction ne modifie jamais les données.

    Elle vérifie :

    1. cohérence avec ApprovedPreparationPlan ;
    2. cohérence avec CleaningExecutionReport ;
    3. empreintes avant/après ;
    4. dimensions et valeurs manquantes ;
    5. intégrité structurelle minimale ;
    6. postcondition de chaque transformation exécutée.

    Un échec bloque `valid_for_analysis`.
    """

    if before is None:
        raise ValueError(
            "before ne peut pas être None."
        )

    if after is None:
        raise ValueError(
            "after ne peut pas être None."
        )

    dataset_checks = (
        _validate_dataset_integrity(
            before=
                before,

            after=
                after,

            approved_plan=
                approved_plan,

            execution_report=
                execution_report,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,
        )
    )

    # ========================================================
    # EXECUTION REPORT INDEX
    # ========================================================

    execution_by_decision = {}

    for step in execution_report.steps:
        if (
            step.decision_id
            in execution_by_decision
        ):
            dataset_checks.append(
                _failed(
                    check_id=
                        "duplicate_execution_decision",

                    scope=
                        "execution",

                    message=(
                        "Le rapport contient plusieurs "
                        "étapes pour le même decision_id."
                    ),

                    decision_id=
                        step.decision_id,
                )
            )

            continue

        execution_by_decision[
            step.decision_id
        ] = step

    # ========================================================
    # EXPECTED EXECUTABLE STEPS
    # ========================================================

    expected_steps = [
        step
        for step in approved_plan.steps
        if step.executor_eligible
    ]

    step_validations: list[
        StepValidationResult
    ] = []

    for approved_step in expected_steps:
        execution_step = (
            execution_by_decision.get(
                approved_step.decision_id
            )
        )

        if execution_step is None:
            step_validations.append(
                StepValidationResult(
                    decision_id=
                        approved_step.decision_id,

                    source_issue_id=
                        approved_step.source_issue_id,

                    column=
                        approved_step.column,

                    action=(
                        approved_step
                        .approved_action
                        or
                        PreparationAction
                        .KEEP_AS_IS
                    ),

                    validation_status=
                        "failed",

                    check_count=
                        1,

                    passed_check_count=
                        0,

                    failed_check_count=
                        1,

                    warning_check_count=
                        0,

                    checks=[
                        _failed(
                            check_id=
                                "execution_step_missing",

                            scope=
                                "step",

                            message=(
                                "L'étape approuvée "
                                "n'apparaît pas dans "
                                "le rapport d'exécution."
                            ),

                            decision_id=
                                approved_step
                                .decision_id,

                            column=
                                approved_step.column,

                            action=
                                approved_step
                                .approved_action,
                        )
                    ],
                )
            )

            continue

        step_validations.append(
            _validate_step_postcondition(
                after=
                    after,

                decision_id=
                    approved_step.decision_id,

                source_issue_id=
                    approved_step.source_issue_id,

                column=
                    approved_step.column,

                action=
                    execution_step.action,

                parameters=
                    dict(
                        execution_step.parameters
                    ),
            )
        )

    # ========================================================
    # EXTRA EXECUTION STEPS
    # ========================================================

    expected_ids = {
        step.decision_id
        for step in expected_steps
    }

    for execution_step in execution_report.steps:
        if (
            execution_step.decision_id
            in expected_ids
        ):
            continue

        dataset_checks.append(
            _failed(
                check_id=
                    "unexpected_execution_step",

                scope=
                    "execution",

                message=(
                    "Une transformation a été "
                    "exécutée sans étape correspondante "
                    "dans le plan approuvé."
                ),

                decision_id=
                    execution_step.decision_id,

                column=
                    execution_step.column,

                action=
                    execution_step.action,
            )
        )

    # ========================================================
    # GLOBAL COUNTS
    # ========================================================

    all_checks = list(
        dataset_checks
    )

    for step_validation in step_validations:
        all_checks.extend(
            step_validation.checks
        )

    passed_count = sum(
        1
        for check in all_checks
        if check.status == "passed"
    )

    failed_count = sum(
        1
        for check in all_checks
        if check.status == "failed"
    )

    warning_count = sum(
        1
        for check in all_checks
        if check.status == "warning"
    )

    valid_for_analysis = (
        failed_count == 0
    )

    status: Literal[
        "passed",
        "failed",
    ] = (
        "passed"
        if valid_for_analysis
        else "failed"
    )

    return (
        PostCleaningValidationReport(
            status=
                status,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            rows_before=
                int(
                    len(
                        before
                    )
                ),

            rows_after=
                int(
                    len(
                        after
                    )
                ),

            columns_before=
                int(
                    len(
                        before.columns
                    )
                ),

            columns_after=
                int(
                    len(
                        after.columns
                    )
                ),

            missing_cells_before=
                _missing_cell_count(
                    before
                ),

            missing_cells_after=
                _missing_cell_count(
                    after
                ),

            fingerprint_before=
                _dataframe_fingerprint(
                    before
                ),

            fingerprint_after=
                _dataframe_fingerprint(
                    after
                ),

            dataset_check_count=
                len(
                    dataset_checks
                ),

            step_validation_count=
                len(
                    step_validations
                ),

            passed_check_count=
                passed_count,

            failed_check_count=
                failed_count,

            warning_check_count=
                warning_count,

            valid_for_analysis=
                valid_for_analysis,

            dataset_checks=
                dataset_checks,

            step_validations=
                step_validations,

            notes=[
                (
                    "Post-cleaning Validation "
                    "ne modifie jamais les données."
                ),
                (
                    "Le dataset final doit correspondre "
                    "exactement à l'empreinte produite "
                    "par Cleaning Executor."
                ),
                (
                    "Chaque transformation exécutée "
                    "est contrôlée par une postcondition."
                ),
                (
                    "Une validation échouée bloque "
                    "valid_for_analysis."
                ),
                (
                    "Les problèmes de qualité métier "
                    "restants seront réévalués dans "
                    "une version ultérieure."
                ),
            ],

            rule_version=
                POST_CLEANING_VALIDATION_RULE_VERSION,
        )
    )
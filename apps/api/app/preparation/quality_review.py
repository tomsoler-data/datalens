from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.preparation.contracts import (
    ColumnDataType,
    ColumnQualityProfile,
    IssueSeverity,
    QualityIssue,
    QualityReviewResult,
)


def _json_safe_value(
    value: Any,
) -> Any:
    """
    Convertit certaines valeurs NumPy / Pandas en valeurs
    compatibles avec Pydantic et JSON.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        return float(value)

    if isinstance(
        value,
        (
            np.bool_,
        ),
    ):
        return bool(value)

    if isinstance(
        value,
        (
            pd.Timestamp,
        ),
    ):
        return value.isoformat()

    if isinstance(
        value,
        (
            np.datetime64,
        ),
    ):
        return pd.Timestamp(
            value
        ).isoformat()

    return value


def _sample_values(
    series: pd.Series,
    limit: int = 5,
) -> List[Any]:
    values = (
        series
        .dropna()
        .drop_duplicates()
        .head(limit)
        .tolist()
    )

    return [
        _json_safe_value(
            value
        )
        for value in values
    ]


def _is_mostly_string(
    series: pd.Series,
) -> bool:
    non_null = (
        series
        .dropna()
    )

    if non_null.empty:
        return False

    sample = non_null.head(
        500
    )

    string_ratio = (
        sample
        .map(
            lambda value:
            isinstance(
                value,
                str,
            )
        )
        .mean()
    )

    return bool(
        string_ratio
        >= 0.80
    )


def _infer_column_type(
    series: pd.Series,
) -> ColumnDataType:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return ColumnDataType.BOOLEAN

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return ColumnDataType.DATETIME

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return ColumnDataType.NUMERIC

    if not _is_mostly_string(
        series
    ):
        return ColumnDataType.UNKNOWN

    non_null_count = int(
        series.notna().sum()
    )

    if non_null_count == 0:
        return ColumnDataType.UNKNOWN

    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    unique_rate = (
        unique_count
        / non_null_count
    )

    if (
        unique_count <= 50
        or unique_rate <= 0.20
    ):
        return ColumnDataType.CATEGORICAL

    return ColumnDataType.TEXT


def _count_empty_strings(
    series: pd.Series,
) -> int:
    if not _is_mostly_string(
        series
    ):
        return 0

    count = 0

    for value in (
        series
        .dropna()
        .tolist()
    ):
        if (
            isinstance(
                value,
                str,
            )
            and value.strip() == ""
        ):
            count += 1

    return count


def _count_whitespace_issues(
    series: pd.Series,
) -> int:
    if not _is_mostly_string(
        series
    ):
        return 0

    count = 0

    for value in (
        series
        .dropna()
        .tolist()
    ):
        if not isinstance(
            value,
            str,
        ):
            continue

        if (
            value != value.strip()
            and value.strip() != ""
        ):
            count += 1

    return count


def _find_case_variants(
    series: pd.Series,
    max_groups: int = 5,
) -> List[Dict[str, Any]]:
    if not _is_mostly_string(
        series
    ):
        return []

    groups: Dict[
        str,
        set,
    ] = {}

    for raw_value in (
        series
        .dropna()
        .tolist()
    ):
        if not isinstance(
            raw_value,
            str,
        ):
            continue

        stripped = (
            raw_value
            .strip()
        )

        if not stripped:
            continue

        normalized = (
            stripped
            .casefold()
        )

        groups.setdefault(
            normalized,
            set(),
        ).add(
            stripped
        )

    variants = []

    for normalized, originals in groups.items():
        if len(
            originals
        ) <= 1:
            continue

        variants.append(
            {
                "normalized_value":
                    normalized,

                "variants":
                    sorted(
                        originals
                    ),
            }
        )

        if len(
            variants
        ) >= max_groups:
            break

    return variants


def _numeric_coercion_rate(
    series: pd.Series,
) -> Optional[float]:
    if pd.api.types.is_numeric_dtype(
        series
    ):
        return None

    if not _is_mostly_string(
        series
    ):
        return None

    cleaned = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    cleaned = cleaned[
        cleaned != ""
    ]

    if cleaned.empty:
        return None

    converted = pd.to_numeric(
        cleaned,
        errors="coerce",
    )

    success_count = int(
        converted.notna().sum()
    )

    return (
        success_count
        / len(cleaned)
    )


def _count_iqr_outliers(
    series: pd.Series,
) -> int:
    if not pd.api.types.is_numeric_dtype(
        series
    ):
        return 0

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(
        numeric
    ) < 4:
        return 0

    q1 = float(
        numeric.quantile(
            0.25
        )
    )

    q3 = float(
        numeric.quantile(
            0.75
        )
    )

    iqr = (
        q3
        - q1
    )

    if iqr == 0:
        return 0

    lower_bound = (
        q1
        - 1.5 * iqr
    )

    upper_bound = (
        q3
        + 1.5 * iqr
    )

    mask = (
        (
            numeric
            < lower_bound
        )
        |
        (
            numeric
            > upper_bound
        )
    )

    return int(
        mask.sum()
    )


def _build_column_profile(
    name: str,
    series: pd.Series,
    row_count: int,
) -> ColumnQualityProfile:
    non_null_count = int(
        series.notna().sum()
    )

    missing_count = (
        row_count
        - non_null_count
    )

    missing_rate = (
        missing_count
        / row_count
        if row_count
        else 0.0
    )

    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    unique_rate = (
        unique_count
        / non_null_count
        if non_null_count
        else 0.0
    )

    identifier_candidate = (
        non_null_count > 1
        and missing_count == 0
        and unique_count
        == non_null_count
    )

    return ColumnQualityProfile(
        name=name,

        pandas_dtype=str(
            series.dtype
        ),

        inferred_type=
            _infer_column_type(
                series
            ),

        row_count=
            row_count,

        non_null_count=
            non_null_count,

        missing_count=
            missing_count,

        missing_rate=
            missing_rate,

        unique_count=
            unique_count,

        unique_rate=
            unique_rate,

        identifier_candidate=
            identifier_candidate,

        empty_string_count=
            _count_empty_strings(
                series
            ),

        whitespace_issue_count=
            _count_whitespace_issues(
                series
            ),

        potential_outlier_count=
            _count_iqr_outliers(
                series
            ),

        numeric_coercion_rate=
            _numeric_coercion_rate(
                series
            ),

        sample_values=
            _sample_values(
                series
            ),
    )


def _review_column(
    profile: ColumnQualityProfile,
) -> List[QualityIssue]:
    issues: List[
        QualityIssue
    ] = []

    column = profile.name

    if profile.missing_count > 0:
        issues.append(
            QualityIssue(
                code=
                    "missing_values",

                severity=
                    IssueSeverity.WARNING,

                column=
                    column,

                message=(
                    f"La colonne '{column}' contient "
                    f"{profile.missing_count} valeur(s) manquante(s)."
                ),

                evidence={
                    "missing_count":
                        profile.missing_count,

                    "missing_rate":
                        profile.missing_rate,
                },

                suggested_action=(
                    "Examiner le rôle métier de la colonne avant de "
                    "choisir entre conservation, suppression ou imputation."
                ),
            )
        )

    if profile.empty_string_count > 0:
        issues.append(
            QualityIssue(
                code=
                    "empty_strings",

                severity=
                    IssueSeverity.WARNING,

                column=
                    column,

                message=(
                    f"La colonne '{column}' contient "
                    f"{profile.empty_string_count} chaîne(s) vide(s)."
                ),

                evidence={
                    "empty_string_count":
                        profile.empty_string_count,
                },

                suggested_action=(
                    "Évaluer si les chaînes vides doivent être "
                    "normalisées en valeurs manquantes."
                ),
            )
        )

    if profile.whitespace_issue_count > 0:
        issues.append(
            QualityIssue(
                code=
                    "surrounding_whitespace",

                severity=
                    IssueSeverity.WARNING,

                column=
                    column,

                message=(
                    f"La colonne '{column}' contient "
                    f"{profile.whitespace_issue_count} valeur(s) "
                    "avec des espaces en début ou fin de chaîne."
                ),

                evidence={
                    "whitespace_issue_count":
                        profile.whitespace_issue_count,
                },

                suggested_action=(
                    "Proposer une normalisation des espaces "
                    "sans modifier automatiquement les données."
                ),
            )
        )

    if (
        profile.numeric_coercion_rate is not None
        and profile.numeric_coercion_rate >= 0.80
        and profile.inferred_type
        not in {
            ColumnDataType.NUMERIC,
        }
    ):
        issues.append(
            QualityIssue(
                code=
                    "numeric_stored_as_text",

                severity=
                    IssueSeverity.WARNING,

                column=
                    column,

                message=(
                    f"La colonne '{column}' semble contenir "
                    "majoritairement des valeurs numériques "
                    "stockées sous forme de texte."
                ),

                evidence={
                    "numeric_coercion_rate":
                        profile.numeric_coercion_rate,
                },

                suggested_action=(
                    "Vérifier la signification de la colonne puis "
                    "proposer une conversion vers un type numérique."
                ),
            )
        )

    if profile.potential_outlier_count > 0:
        issues.append(
            QualityIssue(
                code=
                    "potential_numeric_outliers",

                severity=
                    IssueSeverity.INFO,

                column=
                    column,

                message=(
                    f"La colonne '{column}' contient "
                    f"{profile.potential_outlier_count} observation(s) "
                    "potentiellement atypique(s) selon la règle IQR."
                ),

                evidence={
                    "method":
                        "IQR_1.5",

                    "potential_outlier_count":
                        profile.potential_outlier_count,
                },

                suggested_action=(
                    "Examiner ces observations avant toute décision. "
                    "Une valeur atypique n'est pas nécessairement une erreur."
                ),
            )
        )

    if profile.identifier_candidate:
        issues.append(
            QualityIssue(
                code=
                    "identifier_candidate",

                severity=
                    IssueSeverity.INFO,

                column=
                    column,

                message=(
                    f"La colonne '{column}' possède une valeur unique "
                    "pour chaque ligne et pourrait correspondre "
                    "à un identifiant."
                ),

                evidence={
                    "unique_count":
                        profile.unique_count,

                    "non_null_count":
                        profile.non_null_count,
                },

                suggested_action=(
                    "Confirmer le rôle sémantique de cette colonne "
                    "avant l'analyse statistique."
                ),
            )
        )

    return issues


def review_dataframe_quality(
    dataframe: pd.DataFrame,
) -> QualityReviewResult:
    """
    Réalise une revue déterministe de qualité des données.

    Cette fonction ne nettoie rien.

    Elle détecte et documente les problèmes afin qu'une étape
    de nettoyage puisse ensuite proposer des actions explicites.
    """

    if dataframe is None:
        raise ValueError(
            "Le dataframe ne peut pas être None."
        )

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

    issues: List[
        QualityIssue
    ] = []

    column_profiles: List[
        ColumnQualityProfile
    ] = []

    if column_count == 0:
        issues.append(
            QualityIssue(
                code=
                    "no_columns",

                severity=
                    IssueSeverity.ERROR,

                message=(
                    "Le dataset ne contient aucune colonne."
                ),

                suggested_action=(
                    "Vérifier le fichier importé et son séparateur."
                ),
            )
        )

    if row_count == 0:
        issues.append(
            QualityIssue(
                code=
                    "no_rows",

                severity=
                    IssueSeverity.ERROR,

                message=(
                    "Le dataset ne contient aucune ligne."
                ),

                suggested_action=(
                    "Vérifier le fichier importé ou les étapes "
                    "de préparation précédentes."
                ),
            )
        )

    duplicated_column_names = (
        pd.Index(
            dataframe.columns
        )
        .duplicated()
    )

    duplicate_column_names = [
        str(
            dataframe.columns[
                index
            ]
        )
        for index, duplicated
        in enumerate(
            duplicated_column_names
        )
        if duplicated
    ]

    if duplicate_column_names:
        issues.append(
            QualityIssue(
                code=
                    "duplicate_column_names",

                severity=
                    IssueSeverity.ERROR,

                message=(
                    "Le dataset contient des noms de colonnes dupliqués."
                ),

                evidence={
                    "duplicate_column_names":
                        duplicate_column_names,
                },

                suggested_action=(
                    "Renommer les colonnes ambiguës avant de poursuivre."
                ),
            )
        )

    missing_cells = int(
        dataframe
        .isna()
        .sum()
        .sum()
    )

    total_cells = (
        row_count
        * column_count
    )

    missing_rate = (
        missing_cells
        / total_cells
        if total_cells
        else 0.0
    )

    duplicate_rows = int(
        dataframe
        .duplicated()
        .sum()
    )

    if duplicate_rows > 0:
        issues.append(
            QualityIssue(
                code=
                    "duplicate_rows",

                severity=
                    IssueSeverity.WARNING,

                message=(
                    f"Le dataset contient "
                    f"{duplicate_rows} ligne(s) entièrement dupliquée(s)."
                ),

                evidence={
                    "duplicate_rows":
                        duplicate_rows,
                },

                suggested_action=(
                    "Vérifier si ces doublons sont réellement "
                    "des erreurs avant de les supprimer."
                ),
            )
        )

    for column_position, column_name in enumerate(
        dataframe.columns
    ):
        series = dataframe.iloc[
            :,
            column_position,
        ]

        profile = (
            _build_column_profile(
                name=str(
                    column_name
                ),
                series=series,
                row_count=row_count,
            )
        )

        column_profiles.append(
            profile
        )

        issues.extend(
            _review_column(
                profile
            )
        )

        case_variants = (
            _find_case_variants(
                series
            )
        )

        if case_variants:
            issues.append(
                QualityIssue(
                    code=
                        "case_variants",

                    severity=
                        IssueSeverity.WARNING,

                    column=
                        str(
                            column_name
                        ),

                    message=(
                        f"La colonne '{column_name}' contient "
                        "des catégories qui ne diffèrent que "
                        "par la casse."
                    ),

                    evidence={
                        "examples":
                            case_variants,
                    },

                    suggested_action=(
                        "Vérifier si ces variantes représentent "
                        "la même catégorie avant normalisation."
                    ),
                )
            )

    blocking_issue_count = sum(
        1
        for issue in issues
        if issue.severity
        == IssueSeverity.ERROR
    )

    warning_count = sum(
        1
        for issue in issues
        if issue.severity
        == IssueSeverity.WARNING
    )

    info_count = sum(
        1
        for issue in issues
        if issue.severity
        == IssueSeverity.INFO
    )

    return QualityReviewResult(
        row_count=
            row_count,

        column_count=
            column_count,

        missing_cells=
            missing_cells,

        missing_rate=
            missing_rate,

        duplicate_rows=
            duplicate_rows,

        blocking_issue_count=
            blocking_issue_count,

        warning_count=
            warning_count,

        info_count=
            info_count,

        ready_for_cleaning=(
            blocking_issue_count
            == 0
        ),

        columns=
            column_profiles,

        issues=
            issues,
    )
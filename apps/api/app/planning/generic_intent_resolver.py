from __future__ import annotations


import re
import unicodedata


from pathlib import Path


from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ============================================================
# VERSION
# ============================================================

GENERIC_ANALYTICAL_INTENT_RULE_VERSION = (
    "generic_analytical_intent_v0.1"
)


# ============================================================
# VOCABULARY
# ============================================================

GenericAnalyticalIntentName = Literal[
    "outlier_detection",
]


GenericAnalyticalIntentStatus = Literal[
    "matched",
    "not_matched",
    "blocked",
]


# ============================================================
# OUTPUT MODELS
# ============================================================

class GenericAnalyticalTarget(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    dataset_id: str = Field(
        min_length=1
    )

    dataset_filename: str = Field(
        min_length=1
    )

    column: str = Field(
        min_length=1
    )

    analysis_kind: str = Field(
        min_length=1
    )


class GenericAnalyticalIntentResolution(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    status: (
        GenericAnalyticalIntentStatus
    )

    matched: bool

    intent: (
        GenericAnalyticalIntentName
        | None
    ) = None

    objective: str

    target_count: int = Field(
        ge=0
    )

    targets: list[
        GenericAnalyticalTarget
    ] = Field(
        default_factory=list
    )

    reasons: list[
        str
    ] = Field(
        default_factory=list
    )

    blockers: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        GENERIC_ANALYTICAL_INTENT_RULE_VERSION
    )


# ============================================================
# GENERIC READ HELPER
#
# The resolver deliberately does not import PlannerCatalog.
#
# This keeps the module independent from
# ai_analytical_planner.py and avoids a circular dependency
# when the planner later imports this resolver.
# ============================================================

def _read_value(
    source: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(
        source,
        dict,
    ):
        return source.get(
            name,
            default,
        )


    return getattr(
        source,
        name,
        default,
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _normalize_text(
    value: str,
) -> str:
    decomposed = (
        unicodedata.normalize(
            "NFKD",
            value,
        )
    )


    without_accents = "".join(
        character

        for character
        in decomposed

        if not unicodedata.combining(
            character
        )
    )


    normalized = re.sub(
        r"[^a-zA-Z0-9]+",
        " ",
        without_accents.casefold(),
    )


    return re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()


def _normalized_identifier(
    value: str,
) -> str:
    return (
        _normalize_text(
            value
        )
    )


def _contains_identifier(
    *,
    objective: str,
    identifier: str,
) -> bool:
    objective_normalized = (
        _normalize_text(
            objective
        )
    )


    identifier_normalized = (
        _normalized_identifier(
            identifier
        )
    )


    if not identifier_normalized:
        return False


    return (
        f" {identifier_normalized} "
        in
        f" {objective_normalized} "
    )


# ============================================================
# OUTLIER INTENT DETECTION
# ============================================================

OUTLIER_INTENT_PATTERNS: tuple[
    re.Pattern[
        str
    ],
    ...,
] = (
    re.compile(
        r"\boutliers?\b"
    ),

    re.compile(
        (
            r"\bvaleurs?\s+"
            r"(?:atypiques?|aberrantes?)\b"
        )
    ),

    re.compile(
        (
            r"\bobservations?\s+"
            r"(?:atypiques?|aberrantes?)\b"
        )
    ),

    re.compile(
        (
            r"\bpoints?\s+"
            r"(?:atypiques?|aberrants?)\b"
        )
    ),

    re.compile(
        (
            r"\banomalous\s+"
            r"(?:values?|observations?)\b"
        )
    ),
)


def _is_generic_outlier_request(
    objective: str,
) -> bool:
    normalized = (
        _normalize_text(
            objective
        )
    )


    return any(
        pattern.search(
            normalized
        )

        for pattern
        in OUTLIER_INTENT_PATTERNS
    )


# ============================================================
# EXPLICIT COLUMN DETECTION
#
# Generic expansion must NOT override a precise user request.
#
# Example:
#
#   "Détecte les outliers"
#       -> generic expansion allowed
#
#   "Détecte les outliers de salary"
#       -> generic expansion disabled
#          existing AI planner handles the explicit target
# ============================================================

def _explicit_catalog_column_mentions(
    *,
    objective: str,
    catalog: Any,
) -> list[
    tuple[
        str,
        str,
    ]
]:
    mentions: list[
        tuple[
            str,
            str,
        ]
    ] = []


    datasets = (
        _read_value(
            catalog,
            "datasets",
            [],
        )
        or []
    )


    for dataset in datasets:
        dataset_id = str(
            _read_value(
                dataset,
                "dataset_id",
                "",
            )
            or ""
        ).strip()


        columns = (
            _read_value(
                dataset,
                "columns",
                [],
            )
            or []
        )


        for column in columns:
            column_name = str(
                _read_value(
                    column,
                    "name",
                    "",
                )
                or ""
            ).strip()


            if not column_name:
                continue


            if _contains_identifier(
                objective=objective,
                identifier=column_name,
            ):
                mentions.append(
                    (
                        dataset_id,
                        column_name,
                    )
                )


    return list(
        dict.fromkeys(
            mentions
        )
    )


# ============================================================
# DATASET SCOPE
#
# A generic request with no dataset reference applies to every
# selected dataset in the catalog.
#
# If the user explicitly names one or more datasets, only those
# datasets are included.
# ============================================================

def _explicit_dataset_ids(
    *,
    objective: str,
    catalog: Any,
) -> list[
    str
]:
    mentions: list[
        str
    ] = []


    datasets = (
        _read_value(
            catalog,
            "datasets",
            [],
        )
        or []
    )


    for dataset in datasets:
        dataset_id = str(
            _read_value(
                dataset,
                "dataset_id",
                "",
            )
            or ""
        ).strip()


        filename = str(
            _read_value(
                dataset,
                "filename",
                "",
            )
            or ""
        ).strip()


        identifiers = [
            dataset_id,
            filename,
        ]


        if filename:
            stem = (
                Path(
                    filename
                )
                .stem
                .strip()
            )


            if (
                stem
                and
                stem
                not in identifiers
            ):
                identifiers.append(
                    stem
                )


        if any(
            identifier
            and
            _contains_identifier(
                objective=objective,
                identifier=identifier,
            )

            for identifier
            in identifiers
        ):
            mentions.append(
                dataset_id
            )


    return list(
        dict.fromkeys(
            mentions
        )
    )


# ============================================================
# ELIGIBILITY
# ============================================================

def _looks_like_identifier_column(
    column_name: str,
) -> bool:
    normalized = (
        _normalize_text(
            column_name
        )
        .replace(
            " ",
            "_",
        )
    )


    if normalized in {
        "id",
        "identifier",
        "identifiant",
    }:
        return True


    if normalized.endswith(
        "_id"
    ):
        return True


    if normalized.endswith(
        "_identifier"
    ):
        return True


    if normalized.endswith(
        "_identifiant"
    ):
        return True


    return False


def _eligible_outlier_target(
    column: Any,
) -> bool:
    name = str(
        _read_value(
            column,
            "name",
            "",
        )
        or ""
    ).strip()


    analysis_kind = str(
        _read_value(
            column,
            "analysis_kind",
            "",
        )
        or ""
    ).strip()


    unique_candidate = bool(
        _read_value(
            column,
            "unique_candidate",
            False,
        )
    )


    unique_count = int(
        _read_value(
            column,
            "unique_count",
            0,
        )
        or 0
    )


    missing_ratio = float(
        _read_value(
            column,
            "missing_ratio",
            0.0,
        )
        or 0.0
    )


    if not name:
        return False


    if (
        analysis_kind
        !=
        "quantitative"
    ):
        return False


    # Numeric identifiers should not be interpreted as measures.
    if unique_candidate:
        return False


    if _looks_like_identifier_column(
        name
    ):
        return False


    # A constant column cannot have meaningful outliers.
    if (
        unique_count
        <=
        1
    ):
        return False


    # No usable observation exists.
    if (
        missing_ratio
        >=
        1.0
    ):
        return False


    return True


# ============================================================
# TARGET RESOLUTION
# ============================================================

def _resolve_outlier_targets(
    *,
    objective: str,
    catalog: Any,
) -> list[
    GenericAnalyticalTarget
]:
    explicit_dataset_ids = (
        _explicit_dataset_ids(
            objective=objective,
            catalog=catalog,
        )
    )


    explicit_dataset_set = set(
        explicit_dataset_ids
    )


    datasets = (
        _read_value(
            catalog,
            "datasets",
            [],
        )
        or []
    )


    targets: list[
        GenericAnalyticalTarget
    ] = []


    for dataset in datasets:
        dataset_id = str(
            _read_value(
                dataset,
                "dataset_id",
                "",
            )
            or ""
        ).strip()


        filename = str(
            _read_value(
                dataset,
                "filename",
                "",
            )
            or ""
        ).strip()


        row_count = int(
            _read_value(
                dataset,
                "row_count",
                0,
            )
            or 0
        )


        if not dataset_id:
            continue


        if not filename:
            continue


        if (
            row_count
            <=
            0
        ):
            continue


        if (
            explicit_dataset_set
            and
            dataset_id
            not in
            explicit_dataset_set
        ):
            continue


        columns = (
            _read_value(
                dataset,
                "columns",
                [],
            )
            or []
        )


        for column in columns:
            if not _eligible_outlier_target(
                column
            ):
                continue


            column_name = str(
                _read_value(
                    column,
                    "name",
                    "",
                )
            ).strip()


            analysis_kind = str(
                _read_value(
                    column,
                    "analysis_kind",
                    "",
                )
            ).strip()


            targets.append(
                GenericAnalyticalTarget(
                    dataset_id=
                        dataset_id,

                    dataset_filename=
                        filename,

                    column=
                        column_name,

                    analysis_kind=
                        analysis_kind,
                )
            )


    return targets


# ============================================================
# PUBLIC RESOLVER
# ============================================================

def resolve_generic_analytical_intent(
    *,
    objective: str,
    catalog: Any,
) -> GenericAnalyticalIntentResolution:
    normalized_objective = (
        objective
        .strip()
    )


    if not normalized_objective:
        raise ValueError(
            "L'objectif utilisateur ne peut pas être vide."
        )


    if not _is_generic_outlier_request(
        normalized_objective
    ):
        return (
            GenericAnalyticalIntentResolution(
                status=
                    "not_matched",

                matched=
                    False,

                intent=
                    None,

                objective=
                    normalized_objective,

                target_count=
                    0,

                targets=
                    [],

                reasons=
                    [],

                blockers=
                    [],
            )
        )


    explicit_columns = (
        _explicit_catalog_column_mentions(
            objective=
                normalized_objective,

            catalog=
                catalog,
        )
    )


    # A precise column request belongs to the existing semantic
    # planner. This resolver is intentionally only for generic
    # requests whose analytical scope must be expanded safely
    # from the deterministic catalog.
    if explicit_columns:
        return (
            GenericAnalyticalIntentResolution(
                status=
                    "not_matched",

                matched=
                    False,

                intent=
                    None,

                objective=
                    normalized_objective,

                target_count=
                    0,

                targets=
                    [],

                reasons=[
                    (
                        "La demande contient déjà une colonne "
                        "explicite du catalogue ; la résolution "
                        "générique n'est pas appliquée."
                    )
                ],

                blockers=
                    [],
            )
        )


    targets = (
        _resolve_outlier_targets(
            objective=
                normalized_objective,

            catalog=
                catalog,
        )
    )


    if not targets:
        return (
            GenericAnalyticalIntentResolution(
                status=
                    "blocked",

                matched=
                    True,

                intent=
                    "outlier_detection",

                objective=
                    normalized_objective,

                target_count=
                    0,

                targets=
                    [],

                reasons=[
                    (
                        "Python a reconnu une demande générique "
                        "de détection de valeurs atypiques."
                    )
                ],

                blockers=[
                    (
                        "Aucune colonne quantitative éligible "
                        "n'est disponible dans le périmètre "
                        "sélectionné pour détecter des outliers."
                    )
                ],
            )
        )


    return (
        GenericAnalyticalIntentResolution(
            status=
                "matched",

            matched=
                True,

            intent=
                "outlier_detection",

            objective=
                normalized_objective,

            target_count=
                len(
                    targets
                ),

            targets=
                targets,

            reasons=[
                (
                    "Python a reconnu une demande générique "
                    "de détection de valeurs atypiques."
                ),
                (
                    "Les cibles ont été sélectionnées "
                    "exclusivement depuis les colonnes "
                    "quantitatives du catalogue."
                ),
                (
                    "Les identifiants, constantes et colonnes "
                    "sans observation exploitable ont été exclus."
                ),
            ],

            blockers=
                [],
        )
    )
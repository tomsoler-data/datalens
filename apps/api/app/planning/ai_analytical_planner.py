from __future__ import annotations


import hashlib
import json
import re
import unicodedata

from time import (
    perf_counter,
)

from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.ai.provider import (
    client,
)

from app.planning.analytical_contract import (
    AggregationSpec,
    AnalysisFamily,
    AnalyticalContract,
    RankingSpec,
    VariableBinding,
    WindowSpec,
)


# ============================================================
# VERSION
# ============================================================

AI_ANALYTICAL_PLANNER_RULE_VERSION = (
    "ai_analytical_planner_v0.27"
)


MAX_AI_PLANNER_ATTEMPTS = 2


DEFAULT_AI_PLANNER_MODEL = (
    "gemma3:4b"
)


MAX_AI_PROPOSALS = 8


# ============================================================
# INPUT CATALOG
#
# Only schema-level information is sent to the local LLM.
# No raw rows are required by this planner.
# ============================================================

class PlannerColumnProfile(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    name: str = Field(
        min_length=1
    )

    dtype: str

    analysis_kind: str

    missing_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    unique_count: int = Field(
        ge=0
    )

    unique_candidate: bool = False


class PlannerDatasetProfile(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    dataset_id: str = Field(
        min_length=1
    )

    filename: str = Field(
        min_length=1
    )

    row_count: int = Field(
        ge=0
    )

    column_count: int = Field(
        ge=0
    )

    columns: list[
        PlannerColumnProfile
    ]

    # Server-owned analytical metadata. These fields are optional
    # so source datasets and legacy manifests remain fully compatible.
    is_derived: bool = False

    derivation_type: (
        str
        | None
    ) = None

    analytical_grain: (
        str
        | None
    ) = None

    operation: (
        str
        | None
    ) = None

    aggregation: (
        str
        | None
    ) = None

    group_column: (
        str
        | None
    ) = None

    entity_column: (
        str
        | None
    ) = None

    source_time_column: (
        str
        | None
    ) = None

    target_time_column: (
        str
        | None
    ) = None

    source_measure_column: (
        str
        | None
    ) = None

    target_measure_column: (
        str
        | None
    ) = None

    source_measure_formula: (
        str
        | None
    ) = None

    metric_semantics: (
        str
        | None
    ) = None

    measure_semantic_aliases: list[str] = Field(
        default_factory=list
    )


class PlannerCatalog(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    datasets: list[
        PlannerDatasetProfile
    ] = Field(
        min_length=1
    )


# ============================================================
# COMPACT LLM WIRE PROTOCOL
#
# v0.1 asked the 4B model to emit the complete canonical
# AnalyticalContract shape directly. That schema was too broad:
# many optional nested fields allowed the model to identify the
# correct family while leaving the important variable bindings
# empty.
#
# v0.2 deliberately separates:
#
#   LLM wire protocol
#       ↓
#   deterministic Python translation
#       ↓
#   canonical AnalyticalContract
#
# The wire protocol is small and all its keys are REQUIRED.
# Nullable values must therefore be emitted explicitly as null.
# ============================================================

PlannerDecision = Literal[
    "propose",
    "blocked",
    "ambiguous",
]


WireAggregation = Literal[
    "none",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "distinct_count",
]


WireRankingOrder = Literal[
    "none",
    "ascending",
    "descending",
]


WireWindowOperation = Literal[
    "none",
    "moving_average",
    "rolling_sum",
    "rolling_median",
]


class AIPlannerProposal(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    decision: PlannerDecision

    title: str = Field(
        min_length=1
    )

    family: AnalysisFamily

    dataset_id: (
        str
        | None
    )

    analytical_grain: (
        str
        | None
    )

    x_column: (
        str
        | None
    )

    y_column: (
        str
        | None
    )

    group_column: (
        str
        | None
    )

    value_column: (
        str
        | None
    )

    time_column: (
        str
        | None
    )

    dimension_column: (
        str
        | None
    )

    entity_column: (
        str
        | None
    )

    aggregation_function: (
        WireAggregation
    )

    ranking_order: (
        WireRankingOrder
    )

    ranking_limit: (
        int
        | None
    )

    window_operation: (
        WireWindowOperation
    )

    window_size: (
        int
        | None
    )

    blockers: list[
        str
    ]

    reasons: list[
        str
    ]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class RawAIPlannerOutput(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    proposals: list[
        AIPlannerProposal
    ] = Field(
        min_length=1,
        max_length=MAX_AI_PROPOSALS,
    )


# ============================================================
# VERIFIED OUTPUT
# ============================================================

PlannerValidationStatus = Literal[
    "validated",
    "blocked",
    "ambiguous",
    "rejected",
]


class AIPlannerValidatedItem(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    proposal_index: int = Field(
        ge=1
    )

    validation_status: (
        PlannerValidationStatus
    )

    raw_proposal: (
        AIPlannerProposal
        | None
    ) = None

    proposal: AIPlannerProposal

    contract: (
        AnalyticalContract
        | None
    ) = None

    errors: list[
        str
    ] = Field(
        default_factory=list
    )

    warnings: list[
        str
    ] = Field(
        default_factory=list
    )

    normalizations: list[
        str
    ] = Field(
        default_factory=list
    )


class AIPlannerAttemptTiming(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    attempt_index: int = Field(
        ge=1
    )

    prompt_construction_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    model_inference_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    structured_parse_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    python_validation_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    total_ms: float = Field(
        default=0.0,
        ge=0.0,
    )


class AIPlannerTiming(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    prompt_construction_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    model_inference_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    structured_parse_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    python_validation_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    retry_feedback_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    total_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    attempts: list[
        AIPlannerAttemptTiming
    ] = Field(
        default_factory=list
    )


class AIPlannerReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    status: Literal[
        "ready"
    ] = "ready"

    objective: str

    model: str

    proposal_count: int

    validated_count: int

    blocked_count: int

    ambiguous_count: int

    rejected_count: int

    items: list[
        AIPlannerValidatedItem
    ]

    attempt_count: int = Field(
        default=1,
        ge=1,
    )

    retry_count: int = Field(
        default=0,
        ge=0,
    )

    retry_triggered: bool = False

    retry_feedback: list[
        str
    ] = Field(
        default_factory=list
    )

    normalization_count: int = Field(
        default=0,
        ge=0,
    )

    normalization_applied: bool = False

    timing: AIPlannerTiming = Field(
        default_factory=
            AIPlannerTiming
    )

    planner_rule_version: str = (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
    )


# ============================================================
# CATALOG CONSTRUCTION
# ============================================================

def read_value(
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


def planner_catalog_from_manifests(
    manifests: list[
        Any
    ],
) -> PlannerCatalog:
    datasets: list[
        PlannerDatasetProfile
    ] = []


    for manifest in manifests:
        raw_columns = (
            read_value(
                manifest,
                "columns",
                [],
            )
            or []
        )


        columns = [
            PlannerColumnProfile(
                name=str(
                    read_value(
                        column,
                        "name",
                        "",
                    )
                ),
                dtype=str(
                    read_value(
                        column,
                        "dtype",
                        "",
                    )
                ),
                analysis_kind=str(
                    read_value(
                        column,
                        "analysis_kind",
                        "unknown",
                    )
                ),
                missing_ratio=float(
                    read_value(
                        column,
                        "missing_ratio",
                        0.0,
                    )
                    or 0.0
                ),
                unique_count=int(
                    read_value(
                        column,
                        "unique_count",
                        0,
                    )
                    or 0
                ),
                unique_candidate=bool(
                    read_value(
                        column,
                        "unique_candidate",
                        False,
                    )
                ),
            )

            for column
            in raw_columns
        ]


        datasets.append(
            PlannerDatasetProfile(
                dataset_id=str(
                    read_value(
                        manifest,
                        "dataset_id",
                        "",
                    )
                ),
                filename=str(
                    read_value(
                        manifest,
                        "filename",
                        "",
                    )
                ),
                row_count=int(
                    read_value(
                        manifest,
                        "row_count",
                        0,
                    )
                    or 0
                ),
                column_count=int(
                    read_value(
                        manifest,
                        "column_count",
                        len(
                            columns
                        ),
                    )
                    or len(
                        columns
                    )
                ),
                columns=(
                    columns
                ),
                is_derived=bool(
                    read_value(
                        manifest,
                        "is_derived",
                        False,
                    )
                ),
                derivation_type=(
                    str(
                        read_value(
                            manifest,
                            "derivation_type",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                analytical_grain=(
                    str(
                        read_value(
                            manifest,
                            "analytical_grain",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                operation=(
                    str(
                        read_value(
                            manifest,
                            "operation",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                aggregation=(
                    str(
                        read_value(
                            manifest,
                            "aggregation",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                group_column=(
                    str(
                        read_value(
                            manifest,
                            "group_column",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                entity_column=(
                    str(
                        read_value(
                            manifest,
                            "entity_column",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                source_time_column=(
                    str(
                        read_value(
                            manifest,
                            "source_time_column",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                target_time_column=(
                    str(
                        read_value(
                            manifest,
                            "target_time_column",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                source_measure_column=(
                    str(
                        read_value(
                            manifest,
                            "source_measure_column",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                target_measure_column=(
                    str(
                        read_value(
                            manifest,
                            "target_measure_column",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                source_measure_formula=(
                    str(
                        read_value(
                            manifest,
                            "source_measure_formula",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                metric_semantics=(
                    str(
                        read_value(
                            manifest,
                            "metric_semantics",
                            "",
                        )
                        or
                        ""
                    ).strip()
                    or
                    None
                ),
                measure_semantic_aliases=[
                    str(value).strip()
                    for value
                    in (
                        read_value(
                            manifest,
                            "measure_semantic_aliases",
                            [],
                        )
                        or
                        []
                    )
                    if str(value).strip()
                ],
            )
        )


    return PlannerCatalog(
        datasets=(
            datasets
        )
    )


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
Tu es le AI Analytical Planner LOCAL de DataLens.

Tu dois traduire l'objectif utilisateur en le PLUS PETIT ensemble
de plans analytiques génériques nécessaires.

Tu ne calcules aucun résultat.
Tu n'inventes aucune colonne.
Tu n'inventes aucun dataset.
Tu n'inventes aucune jointure.
Tu n'inventes aucune variable dérivée.

RÈGLE D'ARCHITECTURE :
LLM propose -> Python valide -> outils déterministes exécutent.

IMPORTANT :
Le JSON de sortie possède volontairement des champs simples.
TOUS les champs doivent être présents.
Utilise null lorsqu'un rôle n'est pas utilisé.

FAMILLES :

1. quantitative_association
   OBLIGATOIRE :
   - dataset_id
   - x_column
   - y_column
   x et y doivent être des colonnes quantitatives du catalogue.
   Tous les autres rôles de colonne doivent être null.
   aggregation_function = "none"
   ranking_order = "none"
   window_operation = "none"

2. categorical_association
   OBLIGATOIRE :
   - dataset_id
   - x_column
   - y_column
   x et y doivent être catégorielles ou booléennes.
   Une colonne boolean est TOUJOURS catégorielle pour ce choix
   de famille et ne doit JAMAIS être utilisée dans
   quantitative_association.
   Si les deux variables sont boolean, choisis
   categorical_association.
   aggregation_function = "none"

3. group_comparison
   OBLIGATOIRE :
   - dataset_id
   - group_column catégorielle/booléenne
   - value_column quantitative
   - x_column = null
   - y_column = null
   Pour group_comparison, N'UTILISE JAMAIS x_column/y_column.
   Les rôles sont exclusivement group_column/value_column.
   Tous les autres rôles de colonnes doivent être null :
   x_column, y_column, time_column, dimension_column,
   entity_column.
   aggregation_function = "none"

4. time_series
   OBLIGATOIRE :
   - dataset_id
   - time_column temporelle
   - aggregation_function
   Selon la demande :
   - value_column pour sum/mean/median/min/max
   - entity_column pour distinct_count
   - aucun champ source pour count
   Pour une série temporelle numérique simple, utilise
   EXCLUSIVEMENT time_column et value_column.
   x_column, y_column, group_column et dimension_column
   doivent être null.

5. ranking
   OBLIGATOIRE :
   - dataset_id
   - dimension_column
   - aggregation_function
   - ranking_order
   - ranking_limit
   Pour sum/mean/median/min/max, utilise value_column.
   Pour distinct_count, utilise entity_column.

6. aggregation
   OBLIGATOIRE :
   - dataset_id
   - aggregation_function
   group_column ou dimension_column si la demande agrège par groupe.
   value_column pour une mesure quantitative.

7. descriptive_metric
   OBLIGATOIRE :
   - dataset_id
   - aggregation_function
   count peut compter les lignes sans colonne source.
   Sinon utilise value_column, entity_column ou dimension_column
   selon la métrique demandée.

8. distribution
   OBLIGATOIRE :
   - dataset_id
   - value_column quantitative
   - aggregation_function = "none"
   Utilise EXCLUSIVEMENT value_column.
   x_column, y_column, group_column, time_column,
   dimension_column et entity_column doivent être null.

9. inequality
   OBLIGATOIRE :
   - dataset_id
   - entity_column
   - value_column quantitative

10. data_quality
    OBLIGATOIRE :
    - dataset_id

11. unresolved
    INTERDIT avec decision="propose".
    Utilise seulement avec decision="blocked" ou "ambiguous".

RÈGLES DE SÉCURITÉ :

- Copie EXACTEMENT dataset_id et les noms de colonnes du catalogue.
- Ne remplace jamais un concept absent par une colonne "proche".
- Si l'utilisateur cite explicitement un nom de colonne absent du
  catalogue, utilise decision="blocked". Ne substitue JAMAIS une autre
  colonne, même si son type ou son sens paraît proche.
- Exemple : si l'objectif dit "selon Year" mais que le catalogue ne
  contient que snapshot_date, tu dois BLOQUER. Tu ne peux pas remplacer
  Year par snapshot_date et tu ne peux pas dériver Year implicitement.
- Une association quantitative exige deux variables quantitatives.
- Une variable boolean ou categorical n'est JAMAIS quantitative.
- Deux variables boolean/categorical à associer exigent
  categorical_association.
- Si l'utilisateur cite explicitement un fichier/dataset du catalogue,
  utilise EXACTEMENT son dataset_id. Le nom du fichier n'est PAS un nom
  de colonne.
- S'il existe plusieurs datasets compatibles et que l'utilisateur ne
  précise pas lequel utiliser, utilise decision="ambiguous". Ne choisis
  jamais arbitrairement le premier dataset.
- Si la demande nécessite plusieurs datasets, utilise
  decision="blocked" : les jointures seront introduites plus tard via
  des outils déterministes.
- Si la demande demande une comparaison par groupe sans préciser quelle
  mesure quantitative comparer alors que plusieurs mesures sont possibles,
  utilise decision="ambiguous". N'invente pas la métrique métier.
- Si la demande nécessite une variable calculée absente du catalogue,
  utilise decision="blocked".
- Une proposition exécutable utilise decision="propose".
- confidence évalue la confiance dans le PLAN, pas dans un résultat.
- Ne propose pas d'analyses supplémentaires non demandées.

EXEMPLE 1

Catalogue :
dataset_id = dataset:demo
metric_a = quantitative
metric_b = quantitative

Objectif :
"Étudier la relation entre metric_a et metric_b."

Sortie conceptuelle :
decision = propose
family = quantitative_association
dataset_id = dataset:demo
x_column = metric_a
y_column = metric_b
group_column = null
value_column = null
time_column = null
dimension_column = null
entity_column = null
aggregation_function = none
ranking_order = none
ranking_limit = null
window_operation = none
window_size = null
blockers = []

EXEMPLE 2

Catalogue :
dataset_id = dataset:demo
department = categorical
salary = quantitative

Objectif :
"Comparer le salaire entre les départements."

Sortie conceptuelle :
decision = propose
family = group_comparison
dataset_id = dataset:demo
group_column = department
value_column = salary
x_column = null
y_column = null
time_column = null
dimension_column = null
entity_column = null
aggregation_function = none
ranking_order = none
ranking_limit = null
window_operation = none
window_size = null
blockers = []

EXEMPLE 3

Catalogue :
dataset_id = dataset:hr
overtime = boolean
left_company = boolean

Objectif :
"Étudier l'association entre overtime et left_company."

Sortie conceptuelle :
decision = propose
family = categorical_association
dataset_id = dataset:hr
x_column = overtime
y_column = left_company
group_column = null
value_column = null
time_column = null
dimension_column = null
entity_column = null
aggregation_function = none
ranking_order = none
ranking_limit = null
window_operation = none
window_size = null
blockers = []

EXEMPLE 4

Catalogue :
dataset_id = dataset:hr
Year = temporal_year
salary = quantitative

Objectif :
"Analyser l'évolution de la médiane de salary selon Year."

Sortie conceptuelle :
decision = propose
family = time_series
dataset_id = dataset:hr
time_column = Year
value_column = salary
x_column = null
y_column = null
group_column = null
dimension_column = null
entity_column = null
aggregation_function = median
ranking_order = none
ranking_limit = null
window_operation = none
window_size = null
blockers = []

EXEMPLE 5

Catalogue :
dataset_id = dataset:hr
salary = quantitative

Objectif :
"Analyser la distribution de salary."

Sortie conceptuelle :
decision = propose
family = distribution
dataset_id = dataset:hr
value_column = salary
x_column = null
y_column = null
group_column = null
time_column = null
dimension_column = null
entity_column = null
aggregation_function = none
ranking_order = none
ranking_limit = null
window_operation = none
window_size = null
blockers = []

EXEMPLE 6

Si l'objectif demande "ancienneté" mais qu'aucune colonne correspondante
n'existe :
decision = blocked
family = unresolved
dataset_id = null
toutes les colonnes = null
blockers contient la raison exacte.

EXEMPLE 7

Catalogue :
dataset_id = dataset:sales
category = categorical
unit_price = quantitative

Objectif :
"Quelle catégorie a le prix unitaire moyen le plus élevé ?"

Sortie conceptuelle :
decision = propose
family = ranking
dataset_id = dataset:sales
dimension_column = category
value_column = unit_price
aggregation_function = mean
ranking_order = descending
ranking_limit = 1
tous les autres rôles de colonnes = null

EXEMPLE 8

Catalogue :
dataset_id = dataset:sales
category = categorical
list_price = quantitative

Objectif :
"Donne-moi les deux catégories ayant le prix catalogue moyen le plus élevé."

Sortie conceptuelle :
decision = propose
family = ranking
dataset_id = dataset:sales
dimension_column = category
value_column = list_price
aggregation_function = mean
ranking_order = descending
ranking_limit = 2
tous les autres rôles de colonnes = null

Retourne uniquement la structure JSON imposée.
""".strip()


def compact_catalog_text(
    catalog: PlannerCatalog,
) -> str:
    lines: list[
        str
    ] = []


    for dataset in (
        catalog.datasets
    ):
        lines.append(
            (
                f"DATASET {dataset.dataset_id} | "
                f"{dataset.filename} | "
                f"{dataset.row_count} lignes"
            )
        )


        analytical_metadata: list[str] = []


        if dataset.is_derived:
            analytical_metadata.append(
                "kind=derived_analytical_view"
            )


        for (
            label,
            value,
        ) in [
            (
                "derivation",
                dataset.derivation_type,
            ),
            (
                "grain",
                dataset.analytical_grain,
            ),
            (
                "operation",
                dataset.operation,
            ),
            (
                "aggregation",
                dataset.aggregation,
            ),
            (
                "group",
                dataset.group_column,
            ),
            (
                "entity",
                dataset.entity_column,
            ),
            (
                "source_time",
                dataset.source_time_column,
            ),
            (
                "target_time",
                dataset.target_time_column,
            ),
            (
                "source_measure",
                dataset.source_measure_column,
            ),
            (
                "target_measure",
                dataset.target_measure_column,
            ),
            (
                "formula",
                dataset.source_measure_formula,
            ),
        ]:
            if value:
                analytical_metadata.append(
                    f"{label}={value}"
                )


        if dataset.measure_semantic_aliases:
            analytical_metadata.append(
                (
                    "measure_aliases="
                    +
                    ",".join(
                        dataset.measure_semantic_aliases
                    )
                )
            )


        if analytical_metadata:
            lines.append(
                (
                    "ANALYTICAL_VIEW "
                    +
                    " | ".join(
                        analytical_metadata
                    )
                )
            )


        for column in (
            dataset.columns
        ):
            lines.append(
                (
                    f"- {column.name} | "
                    f"type={column.analysis_kind} | "
                    f"dtype={column.dtype} | "
                    f"missing={column.missing_ratio:.4f} | "
                    f"unique={column.unique_count}"
                )
            )


        lines.append(
            ""
        )


    return "\n".join(
        lines
    ).strip()


def build_user_prompt(
    *,
    objective: str,
    catalog: PlannerCatalog,
) -> str:
    normalized_objective = (
        objective
        .strip()
    )


    if not normalized_objective:
        raise ValueError(
            "L'objectif utilisateur ne peut pas être vide."
        )


    return (
        "OBJECTIF UTILISATEUR\n"
        "====================\n"
        f"{normalized_objective}\n\n"
        "CATALOGUE AUTORISÉ\n"
        "==================\n"
        f"{compact_catalog_text(catalog)}\n\n"
        "RAPPEL IMPORTANT\n"
        "===============\n"
        "Pour une quantitative_association, x_column ET y_column "
        "sont obligatoires et doivent reprendre exactement deux "
        "colonnes quantitatives du même dataset.\n"
        "Pour une categorical_association, x_column ET y_column "
        "doivent être categorical ou boolean. Deux variables boolean "
        "doivent utiliser categorical_association, jamais "
        "quantitative_association.\n"
        "Pour group_comparison, utilise EXCLUSIVEMENT group_column "
        "pour la variable catégorielle/booléenne et value_column pour "
        "la variable quantitative ; x_column et y_column doivent être "
        "null.\n"
        "Pour distribution, utilise EXCLUSIVEMENT value_column avec "
        "une variable quantitative ; tous les autres rôles de colonnes "
        "doivent être null.\n"
        "Pour une time_series numérique, utilise time_column pour la "
        "dimension temporelle et value_column pour la mesure ; "
        "x_column, y_column, group_column et dimension_column doivent "
        "être null.\n"
        "Les lignes ANALYTICAL_VIEW décrivent des vues déterministes "
        "server-owned. Respecte EXACTEMENT leur dataset_id, leur grain, "
        "leur group/entity et leur target_measure. Une vue "
        "categorical_additive_measure avec operation=groupby_sum est "
        "une agrégation, jamais une quantitative_association.\n\n"
        "Construis uniquement le ou les plans nécessaires pour "
        "répondre directement à l'objectif."
    )


# ============================================================
# DETERMINISTIC CATALOG INDEX
# ============================================================

def catalog_index(
    catalog: PlannerCatalog,
) -> dict[
    str,
    PlannerDatasetProfile,
]:
    return {
        dataset.dataset_id:
            dataset

        for dataset
        in catalog.datasets
    }


def find_column(
    dataset: PlannerDatasetProfile,
    column_name: str,
) -> (
    PlannerColumnProfile
    | None
):
    return next(
        (
            column
            for column
            in dataset.columns
            if (
                column.name ==
                column_name
            )
        ),
        None,
    )


# ============================================================
# OBJECTIVE → COLUMN FIDELITY
#
# The LLM may perform semantic mapping, but it may not silently
# replace a column that the user explicitly named.
#
# Examples:
#
#   objective: "... selon Year"
#   catalog:   snapshot_date, salary
#   proposal:  time=snapshot_date
#
# => REJECTED by Python.
#
# A future deterministic derived-variable tool may support:
#   Year = year(snapshot_date)
# but v0.9 must not invent that transformation.
# ============================================================

EXPLICIT_COLUMN_CONTEXT_PATTERN = re.compile(
    (
        r"\b(?:selon|par|by|against|versus|vs\.?)"
        r"\s+"
        r"(?:(?:le|la|les|l['’])\s+)?"
        r"([A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-zÀ-ÖØ-öø-ÿ0-9_.-]*)"
    ),
    flags=re.IGNORECASE,
)


ANALYTICAL_TARGET_CONTEXT_PATTERN = re.compile(
    (
        r"\b(?:"
        r"distribution|histogramme|histogram|"
        r"moyenne|mean|average|"
        r"médiane|mediane|median|"
        r"somme|sum|"
        r"minimum|maximum|min|max|"
        r"variance|"
        r"écart[-\s]?type|standard[-\s]?deviation"
        r")"
        r"\s+"
        r"(?:de|du|des|d['’]|of)"
        r"\s+"
        r"(?:(?:le|la|les|l['’])\s+)?"
        r"([A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-zÀ-ÖØ-öø-ÿ0-9_.-]*)"
    ),
    flags=re.IGNORECASE,
)


ASSOCIATION_TARGET_CONTEXT_PATTERN = re.compile(
    (
        r"\b(?:"
        r"relation|association|corrélation|correlation"
        r")"
        r"\s+"
        r"(?:entre|between)"
        r"\s+"
        r"([A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-zÀ-ÖØ-öø-ÿ0-9_.-]*)"
        r"\s+"
        r"(?:et|and|avec|with)"
        r"\s+"
        r"([A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-zÀ-ÖØ-öø-ÿ0-9_.-]*)"
    ),
    flags=re.IGNORECASE,
)


EXPLICIT_SCHEMA_NOUN_PATTERN = re.compile(
    (
        r"\b(?:colonne|column|variable|champ|field)"
        r"\s+"
        r"[`\"']?"
        r"([A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-zÀ-ÖØ-öø-ÿ0-9_.-]*)"
        r"[`\"']?"
    ),
    flags=re.IGNORECASE,
)


QUOTED_IDENTIFIER_PATTERN = re.compile(
    (
        r"[`\"']"
        r"([A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-zÀ-ÖØ-öø-ÿ0-9_.-]*)"
        r"[`\"']"
    )
)


SCHEMA_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-zÀ-ÖØ-öø-ÿ0-9_.-]*$"
)


def normalize_identifier_for_match(
    value: str,
) -> str:
    decomposed = (
        unicodedata.normalize(
            "NFKD",
            value,
        )
    )


    without_accents = (
        "".join(
            character

            for character
            in decomposed

            if not unicodedata.combining(
                character
            )
        )
    )


    normalized = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        without_accents.casefold(),
    )


    return (
        normalized
        .strip(
            "_"
        )
    )


def normalized_objective_tokens(
    objective: str,
) -> list[
    str
]:
    normalized = (
        normalize_identifier_for_match(
            objective
        )
    )


    return [
        token

        for token
        in normalized.split(
            "_"
        )

        if token
    ]


def normalized_column_tokens(
    column_name: str,
) -> list[
    str
]:
    normalized = (
        normalize_identifier_for_match(
            column_name
        )
    )


    return [
        token

        for token
        in normalized.split(
            "_"
        )

        if token
    ]


def contains_token_sequence(
    *,
    haystack: list[
        str
    ],
    needle: list[
        str
    ],
) -> bool:
    if (
        not needle
        or
        len(
            needle
        )
        >
        len(
            haystack
        )
    ):
        return False


    width = (
        len(
            needle
        )
    )


    return any(
        haystack[
            index:
            index + width
        ]
        ==
        needle

        for index
        in range(
            0,
            len(
                haystack
            )
            -
            width
            +
            1,
        )
    )


def catalog_column_name_index(
    catalog: PlannerCatalog,
) -> dict[
    str,
    list[
        tuple[
            str,
            str,
        ]
    ],
]:
    index: dict[
        str,
        list[
            tuple[
                str,
                str,
            ]
        ],
    ] = {}


    for dataset in (
        catalog.datasets
    ):
        for column in (
            dataset.columns
        ):
            key = (
                normalize_identifier_for_match(
                    column.name
                )
            )


            if not key:
                continue


            index.setdefault(
                key,
                [],
            ).append(
                (
                    dataset.dataset_id,
                    column.name,
                )
            )


    return index


def literal_schema_identifier_mentioned(
    *,
    objective: str,
    identifier: str,
) -> bool:
    """
    Literal schema-reference check.

    Word characters (including underscore) are treated as part
    of an identifier. Therefore `revenue` does not match inside
    `net_revenue`, while the full `net_revenue` still matches.

    This helper intentionally checks literal schema names only.
    A separate conservative lexical resolver may confirm an
    already proposed quantitative metric, while this function
    remains dedicated to protecting explicit user references.
    """

    identifier = (
        identifier.strip()
    )


    if not identifier:
        return False


    pattern = re.compile(
        (
            r"(?<!\w)"
            +
            re.escape(
                identifier
            )
            +
            r"(?!\w)"
        ),
        flags=re.IGNORECASE,
    )


    return bool(
        pattern.search(
            objective
        )
    )


def explicit_known_column_mentions(
    *,
    objective: str,
    dataset: PlannerDatasetProfile,
) -> list[
    str
]:
    mentions = [
        column.name

        for column
        in dataset.columns

        if literal_schema_identifier_mentioned(
            objective=(
                objective
            ),
            identifier=(
                column.name
            ),
        )
    ]


    return list(
        dict.fromkeys(
            mentions
        )
    )


# ============================================================
# CONSERVATIVE SEMANTIC METRIC REFERENCES
#
# The literal fidelity guard above protects exact schema names.
# This second layer is deliberately narrower than an LLM:
#
# - it operates only on schema tokens;
# - it uses a small deterministic bilingual lexical map;
# - every token of the candidate column must be supported;
# - supported tokens must occur in a short local objective span;
# - only the unique best quantitative match is accepted.
#
# Examples:
#   "prix unitaire"   -> unit_price
#   "coût unitaire"   -> unit_cost
#   "prix catalogue"  -> list_price
#
# But:
#   "prix moyen"
# does not fully identify unit_price or list_price and therefore
# remains ambiguous when several quantitative columns exist.
#
# This layer never invents a column and never changes the LLM
# proposal. It only decides whether Python has enough lexical
# evidence to trust an already proposed quantitative column.
# ============================================================

SEMANTIC_TOKEN_CANONICAL = {
    "unit": "unit",
    "unite": "unit",
    "unitaire": "unit",
    "unitaires": "unit",
    "unites": "unit",

    "price": "price",
    "prix": "price",

    "cost": "cost",
    "cout": "cost",
    "couts": "cost",

    "list": "list",
    "catalog": "list",
    "catalogue": "list",
    "catalogues": "list",

    "quantity": "quantity",
    "quantite": "quantity",
    "quantites": "quantity",

    "discount": "discount",
    "remise": "discount",
    "remises": "discount",
    "rabais": "discount",

    "pct": "percent",
    "percent": "percent",
    "percentage": "percent",
    "pourcentage": "percent",
    "pourcentages": "percent",
    "taux": "percent",

    "age": "age",

    "revenue": "revenue",
    "revenu": "revenue",
    "revenus": "revenue",
    "recette": "revenue",
    "recettes": "revenue",

    "sale": "sales",
    "sales": "sales",
    "vente": "sales",
    "ventes": "sales",

    "amount": "amount",
    "montant": "amount",
    "montants": "amount",

    "basket": "basket",
    "baskets": "basket",
    "panier": "basket",
    "paniers": "basket",

    "mean": "mean",
    "average": "mean",
    "avg": "mean",
    "moyen": "mean",
    "moyens": "mean",
    "moyenne": "mean",
    "moyennes": "mean",

    "category": "category",
    "categories": "category",
    "categorie": "category",

    "brand": "brand",
    "brands": "brand",
    "marque": "brand",
    "marques": "brand",

    "country": "country",
    "countries": "country",
    "pays": "country",

    "segment": "segment",
    "segments": "segment",

    "product": "product",
    "products": "product",
    "produit": "product",
    "produits": "product",

    "month": "month",
    "months": "month",
    "monthly": "month",
    "mois": "month",
    "mensuel": "month",
    "mensuels": "month",
    "mensuelle": "month",
    "mensuelles": "month",
}


def canonical_semantic_token(
    token: str,
) -> str:
    normalized = (
        normalize_identifier_for_match(
            token
        )
    )


    return (
        SEMANTIC_TOKEN_CANONICAL.get(
            normalized,
            normalized,
        )
    )


def semantic_metric_match_score(
    *,
    objective: str,
    column_name: str,
) -> int:
    """
    Return a deterministic lexical score for one candidate
    quantitative column.

    A candidate matches only when every semantic token from the
    physical column name appears in a short local span of the
    user objective. Order may differ so that French expressions
    such as "prix unitaire" can match `unit_price`.

    More specific column names receive a higher score. This
    prevents a generic one-token candidate such as `price` from
    outranking `unit_price` for "prix unitaire".
    """

    objective_tokens = [
        canonical_semantic_token(
            token
        )

        for token
        in normalized_objective_tokens(
            objective
        )
    ]


    column_tokens = [
        canonical_semantic_token(
            token
        )

        for token
        in normalized_column_tokens(
            column_name
        )
    ]


    required_tokens = {
        token

        for token
        in column_tokens

        if token
    }


    if (
        not required_tokens
        or
        not objective_tokens
    ):
        return 0


    # A small local window prevents unrelated words that happen
    # to occur far apart in the request from being combined into
    # an artificial semantic match.
    max_window = max(
        len(
            required_tokens
        )
        +
        2,
        3,
    )


    best_score = 0


    for start in range(
        len(
            objective_tokens
        )
    ):
        max_end = min(
            len(
                objective_tokens
            ),
            start
            +
            max_window,
        )


        for end in range(
            start
            +
            1,
            max_end
            +
            1,
        ):
            window = (
                objective_tokens[
                    start:
                    end
                ]
            )


            if (
                not required_tokens.issubset(
                    set(
                        window
                    )
                )
            ):
                continue


            extra_tokens = max(
                0,
                len(
                    window
                )
                -
                len(
                    required_tokens
                ),
            )


            score = (
                len(
                    required_tokens
                )
                *
                100
                -
                extra_tokens
            )


            best_score = max(
                best_score,
                score,
            )


    return (
        best_score
    )


def semantic_names_for_column(
    *,
    dataset: PlannerDatasetProfile,
    column_name: str,
) -> list[str]:
    """
    Return deterministic semantic names that may identify one
    physical planner column.

    Only the declared target measure of a server-owned analytical
    view receives aliases. Source datasets keep pure schema-name
    semantics.
    """

    names = [
        column_name
    ]


    if (
        dataset.target_measure_column
        ==
        column_name
    ):
        names.extend(
            dataset.measure_semantic_aliases
        )


    return list(
        dict.fromkeys(
            value.strip()
            for value
            in names
            if value.strip()
        )
    )



def semantic_quantitative_column_mentions(
    *,
    objective: str,
    dataset: PlannerDatasetProfile,
) -> list[
    str
]:
    """
    Resolve the unique best quantitative schema reference from
    deterministic lexical evidence.

    Returning several names means the semantic evidence is still
    ambiguous. Returning an empty list means no quantitative
    column was sufficiently identified.
    """

    scored_matches: list[
        tuple[
            int,
            str,
        ]
    ] = []


    for column in (
        dataset.columns
    ):
        if (
            not is_quantitative(
                column.analysis_kind
            )
        ):
            continue


        score = max(
            (
                semantic_metric_match_score(
                    objective=(
                        objective
                    ),
                    column_name=(
                        semantic_name
                    ),
                )

                for semantic_name
                in semantic_names_for_column(
                    dataset=dataset,
                    column_name=column.name,
                )
            ),
            default=0,
        )


        if (
            score
            >
            0
        ):
            scored_matches.append(
                (
                    score,
                    column.name,
                )
            )


    if (
        not scored_matches
    ):
        return []


    best_score = max(
        score

        for (
            score,
            _,
        )
        in scored_matches
    )


    best_matches = [
        column_name

        for (
            score,
            column_name,
        )
        in scored_matches

        if (
            score
            ==
            best_score
        )
    ]


    return list(
        dict.fromkeys(
            best_matches
        )
    )



def semantic_schema_column_mentions(
    *,
    objective: str,
    dataset: PlannerDatasetProfile,
) -> list[
    str
]:
    """
    Resolve deterministic schema references across all analytical
    column types.

    Unlike the quantitative-only metric resolver, this helper may
    return several independent columns because an analytical request
    commonly names two variables (for example `unit_cost` and
    `category`).

    A less-specific candidate is removed only when its semantic token
    set is a strict subset of another matched schema column. This keeps
    `category` beside `unit_cost`, while preventing a generic `price`
    column from competing with a more specific `unit_price` match for
    the phrase "prix unitaire".
    """

    matched: list[
        tuple[
            str,
            set[str],
            int,
        ]
    ] = []


    for column in dataset.columns:
        score = max(
            (
                semantic_metric_match_score(
                    objective=objective,
                    column_name=semantic_name,
                )
                for semantic_name
                in semantic_names_for_column(
                    dataset=dataset,
                    column_name=column.name,
                )
            ),
            default=0,
        )


        if score <= 0:
            continue


        tokens = {
            canonical_semantic_token(token)
            for token in normalized_column_tokens(column.name)
            if token
        }


        if not tokens:
            continue


        matched.append(
            (
                column.name,
                tokens,
                score,
            )
        )


    if not matched:
        return []


    selected: list[str] = []


    for (
        column_name,
        tokens,
        score,
    ) in matched:
        shadowed = any(
            tokens < other_tokens
            and other_score > score
            for (
                other_name,
                other_tokens,
                other_score,
            ) in matched
            if other_name != column_name
        )


        if not shadowed:
            selected.append(column_name)


    return list(
        dict.fromkeys(
            selected
        )
    )


def objective_schema_column_mentions(
    *,
    objective: str,
    dataset: PlannerDatasetProfile,
) -> list[
    str
]:
    """
    Return schema columns that are explicitly or conservatively
    referenced by the user objective.

    Literal identifiers remain authoritative. Semantic matches are
    added only through the deterministic token resolver above.
    """

    mentions = [
        *explicit_known_column_mentions(
            objective=objective,
            dataset=dataset,
        ),
        *semantic_schema_column_mentions(
            objective=objective,
            dataset=dataset,
        ),
    ]


    return list(
        dict.fromkeys(
            mentions
        )
    )


def canonicalize_explicit_objective_bindings(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> tuple[
    AIPlannerProposal,
    list[str],
]:
    """
    Protect semantic fidelity when the objective deterministically
    identifies exactly two physical columns.

    Small local models can preserve the wording of the request while
    binding a nearby but unmentioned column. Python repairs that wire
    error only when the objective resolves to exactly two schema
    columns and their deterministic analytical types imply one
    unambiguous supported two-variable family.

    Supported repairs:
    - quantitative + quantitative -> quantitative_association
    - categorical + categorical -> categorical_association
    - categorical + quantitative -> group_comparison

    No repair occurs for vague, one-column or 3+-column objectives.
    """

    if (
        proposal.decision != "propose"
        or proposal.dataset_id is None
        or proposal.family not in {
            "quantitative_association",
            "categorical_association",
            "group_comparison",
        }
    ):
        return proposal, []


    dataset = catalog_index(catalog).get(
        proposal.dataset_id
    )


    if dataset is None:
        return proposal, []


    mentions = objective_schema_column_mentions(
        objective=objective,
        dataset=dataset,
    )


    if len(mentions) != 2:
        return proposal, []


    first = find_column(
        dataset,
        mentions[0],
    )
    second = find_column(
        dataset,
        mentions[1],
    )


    if first is None or second is None:
        return proposal, []


    first_is_quantitative = is_quantitative(
        first.analysis_kind
    )
    second_is_quantitative = is_quantitative(
        second.analysis_kind
    )
    first_is_categorical = is_categorical(
        first.analysis_kind
    )
    second_is_categorical = is_categorical(
        second.analysis_kind
    )


    update: dict[str, Any] | None = None


    if (
        first_is_quantitative
        and second_is_quantitative
    ):
        update = {
            "family": "quantitative_association",
            "x_column": first.name,
            "y_column": second.name,
            "group_column": None,
            "value_column": None,
            "time_column": None,
            "dimension_column": None,
            "entity_column": None,
        }


    elif (
        first_is_categorical
        and second_is_categorical
    ):
        update = {
            "family": "categorical_association",
            "x_column": first.name,
            "y_column": second.name,
            "group_column": None,
            "value_column": None,
            "time_column": None,
            "dimension_column": None,
            "entity_column": None,
        }


    elif (
        (first_is_categorical and second_is_quantitative)
        or
        (first_is_quantitative and second_is_categorical)
    ):
        group_profile = (
            first
            if first_is_categorical
            else second
        )
        value_profile = (
            first
            if first_is_quantitative
            else second
        )


        update = {
            "family": "group_comparison",
            "x_column": None,
            "y_column": None,
            "group_column": group_profile.name,
            "value_column": value_profile.name,
            "time_column": None,
            "dimension_column": None,
            "entity_column": None,
        }


    if update is None:
        return proposal, []


    current_bound = {
        column_name
        for column_name in [
            proposal.x_column,
            proposal.y_column,
            proposal.group_column,
            proposal.value_column,
        ]
        if column_name is not None
    }
    expected_bound = set(mentions)


    if (
        proposal.family == update["family"]
        and current_bound == expected_bound
    ):
        return proposal, []


    normalized = proposal.model_copy(
        update=update
    )


    return (
        normalized,
        [
            (
                "Python a corrigé une incohérence sémantique entre "
                "le texte de la demande et les colonnes proposées "
                "par le modèle. Les deux colonnes déterministiquement "
                "identifiées dans l'objectif sont "
                f"{mentions[0]} et {mentions[1]}; la famille canonique "
                f"est `{normalized.family}`."
            )
        ],
    )


def explicit_dataset_mentions(
    *,
    objective: str,
    catalog: PlannerCatalog,
) -> list[
    PlannerDatasetProfile
]:
    mentions: list[
        PlannerDatasetProfile
    ] = []


    for dataset in (
        catalog.datasets
    ):
        if (
            literal_schema_identifier_mentioned(
                objective=(
                    objective
                ),
                identifier=(
                    dataset.filename
                ),
            )
            or
            literal_schema_identifier_mentioned(
                objective=(
                    objective
                ),
                identifier=(
                    dataset.dataset_id
                ),
            )
        ):
            mentions.append(
                dataset
            )


    return mentions


def canonicalize_explicit_dataset_reference(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> tuple[
    AIPlannerProposal,
    list[
        str
    ],
]:
    if (
        proposal.decision
        !=
        "propose"
    ):
        return (
            proposal,
            [],
        )


    mentions = (
        explicit_dataset_mentions(
            objective=(
                objective
            ),
            catalog=(
                catalog
            ),
        )
    )


    if (
        len(
            mentions
        )
        >
        1
    ):
        filenames = [
            dataset.filename

            for dataset
            in mentions
        ]


        blocker = (
            "DEMANDE MULTI-DATASET : plusieurs fichiers "
            "sont explicitement demandés ("
            +
            ", ".join(
                filenames
            )
            +
            "). Le pipeline natif actuel n'exécute "
            "qu'un dataset à la fois et refuse toute "
            "comparaison ou jointure implicite."
        )


        return (
            proposal.model_copy(
                update={
                    "decision":
                        "blocked",

                    "family":
                        "unresolved",

                    "dataset_id":
                        None,

                    "x_column":
                        None,

                    "y_column":
                        None,

                    "group_column":
                        None,

                    "value_column":
                        None,

                    "time_column":
                        None,

                    "dimension_column":
                        None,

                    "entity_column":
                        None,

                    "blockers":
                        [
                            blocker,
                        ],

                    "reasons":
                        [
                            (
                                "Python a détecté plusieurs références "
                                "explicites de datasets dans l'objectif."
                            )
                        ],
                }
            ),
            [],
        )


    if (
        len(
            mentions
        )
        !=
        1
    ):
        return (
            proposal,
            [],
        )


    selected = (
        mentions[
            0
        ]
    )


    if (
        proposal.dataset_id
        ==
        selected.dataset_id
    ):
        return (
            proposal,
            [],
        )


    previous = (
        proposal.dataset_id
    )


    return (
        proposal.model_copy(
            update={
                "dataset_id":
                    selected.dataset_id,
            }
        ),
        [
            (
                "Python a résolu la référence explicite au "
                f"dataset `{selected.filename}` et a imposé "
                f"dataset_id={selected.dataset_id}"
                +
                (
                    f" à la place de {previous}."
                    if previous
                    is not None
                    else "."
                )
            )
        ],
    )


def clean_explicit_identifier_candidate(
    value: str,
) -> str:
    return (
        value
        .strip()
        .rstrip(
            ".,;:!?)]}"
        )
        .lstrip(
            "([{"
        )
    )


def schema_like_context_mentions(
    *,
    objective: str,
    catalog: PlannerCatalog,
) -> list[
    str
]:
    candidates: list[
        str
    ] = []


    candidates.extend(
        match.group(
            1
        )

        for match
        in QUOTED_IDENTIFIER_PATTERN.finditer(
            objective
        )
    )


    candidates.extend(
        match.group(
            1
        )

        for match
        in EXPLICIT_COLUMN_CONTEXT_PATTERN.finditer(
            objective
        )
    )


    strong_context_candidates: list[
        str
    ] = []


    strong_context_candidates.extend(
        match.group(
            1
        )

        for match
        in ANALYTICAL_TARGET_CONTEXT_PATTERN.finditer(
            objective
        )
    )


    strong_context_candidates.extend(
        match.group(
            1
        )

        for match
        in EXPLICIT_SCHEMA_NOUN_PATTERN.finditer(
            objective
        )
    )


    for match in (
        ASSOCIATION_TARGET_CONTEXT_PATTERN.finditer(
            objective
        )
    ):
        strong_context_candidates.extend(
            [
                match.group(
                    1
                ),
                match.group(
                    2
                ),
            ]
        )


    candidates.extend(
        strong_context_candidates
    )


    # snake_case / dotted identifiers are schema-looking even
    # when they do not occur after a contextual preposition.
    candidates.extend(
        match.group(
            0
        )

        for match
        in re.finditer(
            (
                r"\b"
                r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9]*"
                r"(?:[_\.][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)+"
                r"\b"
            ),
            objective,
        )
    )


    catalog_index_by_name = (
        catalog_column_name_index(
            catalog
        )
    )


    explicit: list[
        str
    ] = []


    known_dataset_identifiers = {
        normalize_identifier_for_match(
            identifier
        )

        for dataset
        in catalog.datasets

        for identifier
        in {
            dataset.dataset_id,
            dataset.filename,
        }

        if identifier
    }


    quoted_normalized = {
        normalize_identifier_for_match(
            clean_explicit_identifier_candidate(
                match.group(
                    1
                )
            )
        )

        for match
        in QUOTED_IDENTIFIER_PATTERN.finditer(
            objective
        )
    }


    strong_context_normalized = {
        normalize_identifier_for_match(
            clean_explicit_identifier_candidate(
                candidate
            )
        )

        for candidate
        in strong_context_candidates
    }


    for raw_candidate in (
        candidates
    ):
        candidate = (
            clean_explicit_identifier_candidate(
                raw_candidate
            )
        )


        if (
            not candidate
            or
            not SCHEMA_IDENTIFIER_PATTERN.fullmatch(
                candidate
            )
        ):
            continue


        normalized = (
            normalize_identifier_for_match(
                candidate
            )
        )


        if (
            normalized
            in known_dataset_identifiers
        ):
            continue


        is_known_column = (
            normalized
            in
            catalog_index_by_name
        )


        looks_schema_specific = (
            "_"
            in candidate
            or
            "."
            in candidate
            or
            any(
                character.isdigit()

                for character
                in candidate
            )
            or
            (
                candidate[
                    :1
                ].isupper()
            )
        )


        is_quoted = (
            normalized
            in
            quoted_normalized
        )


        is_strong_context = (
            normalized
            in
            strong_context_normalized
        )


        if (
            is_known_column
            or
            looks_schema_specific
            or
            is_quoted
            or
            is_strong_context
        ):
            explicit.append(
                candidate
            )


    return list(
        dict.fromkeys(
            explicit
        )
    )


def validate_objective_column_fidelity(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
    dataset: PlannerDatasetProfile | None,
) -> list[
    str
]:
    if (
        proposal.decision
        !=
        "propose"
        or
        dataset is None
    ):
        return []


    errors: list[
        str
    ] = []


    bound_columns = {
        normalize_identifier_for_match(
            column_name
        )

        for (
            _,
            column_name,
        )
        in proposed_role_columns(
            proposal
        )
    }


    # --------------------------------------------------------
    # Exact catalog columns named by the user must survive
    # semantic planning and appear in the canonical roles.
    # --------------------------------------------------------

    for column_name in (
        explicit_known_column_mentions(
            objective=(
                objective
            ),
            dataset=(
                dataset
            ),
        )
    ):
        normalized = (
            normalize_identifier_for_match(
                column_name
            )
        )


        if (
            normalized
            not in bound_columns
        ):
            errors.append(
                (
                    "FIDÉLITÉ OBJECTIF : la colonne "
                    f"`{column_name}` est explicitement "
                    "mentionnée dans la demande mais le "
                    "planner ne l'utilise pas. Python refuse "
                    "une substitution silencieuse."
                )
            )


    # --------------------------------------------------------
    # Detect explicit schema identifiers that are absent from
    # the selected dataset. This catches the exact failure:
    #
    #   "... selon Year"
    #   selected dataset has snapshot_date but no Year
    # --------------------------------------------------------

    selected_columns = {
        normalize_identifier_for_match(
            column.name
        ):
            column.name

        for column
        in dataset.columns
    }


    for mention in (
        schema_like_context_mentions(
            objective=(
                objective
            ),
            catalog=(
                catalog
            ),
        )
    ):
        normalized = (
            normalize_identifier_for_match(
                mention
            )
        )


        if (
            normalized
            in selected_columns
        ):
            continue


        errors.append(
            (
                "FIDÉLITÉ OBJECTIF : la colonne "
                f"explicitement demandée `{mention}` "
                f"n'existe pas dans {dataset.dataset_id}. "
                "DataLens refuse de la remplacer par une "
                "colonne proche ou de créer implicitement "
                "une variable dérivée."
            )
        )


    return list(
        dict.fromkeys(
            errors
        )
    )


# ============================================================
# WIRE → GENERIC BINDINGS
# ============================================================

def proposed_role_columns(
    proposal: AIPlannerProposal,
) -> list[
    tuple[
        str,
        str,
    ]
]:
    pairs: list[
        tuple[
            str,
            str,
        ]
    ] = []


    raw = [
        (
            "x",
            proposal.x_column,
        ),
        (
            "y",
            proposal.y_column,
        ),
        (
            "group",
            proposal.group_column,
        ),
        (
            "value",
            proposal.value_column,
        ),
        (
            "time",
            proposal.time_column,
        ),
        (
            "dimension",
            proposal.dimension_column,
        ),
        (
            "entity",
            proposal.entity_column,
        ),
    ]


    for (
        role,
        column,
    ) in raw:
        if (
            column is not None
            and
            column.strip()
        ):
            pairs.append(
                (
                    role,
                    column.strip(),
                )
            )


    return pairs


def expected_roles_for_family(
    family: str,
) -> set[
    str
]:
    if (
        family in {
            "quantitative_association",
            "categorical_association",
        }
    ):
        return {
            "x",
            "y",
        }


    if (
        family ==
        "group_comparison"
    ):
        return {
            "group",
            "value",
        }


    if (
        family ==
        "distribution"
    ):
        return {
            "value",
        }


    if (
        family ==
        "time_series"
    ):
        return {
            "time",
        }


    if (
        family ==
        "ranking"
    ):
        return {
            "dimension",
        }


    if (
        family ==
        "inequality"
    ):
        return {
            "entity",
            "value",
        }


    return set()


# ============================================================
# WIRE-PROTOCOL ROLE NORMALIZATION
#
# Small local models can correctly identify the analytical
# family and columns while placing those columns in generic x/y
# slots instead of the family-specific group/value slots.
#
# DataLens may normalize ONLY when the mapping is unambiguous
# from:
# - the already selected family;
# - the exact catalog columns;
# - the deterministic analytical types.
#
# This does NOT change the selected family and does NOT invent
# columns. Ambiguous mappings remain rejected.
# ============================================================

def canonicalize_wire_roles(
    *,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> tuple[
    AIPlannerProposal,
    list[
        str
    ],
]:
    if (
        proposal.decision
        !=
        "propose"
    ):
        return (
            proposal,
            [],
        )


    if (
        proposal.family
        not in {
            "quantitative_association",
            "categorical_association",
            "group_comparison",
            "distribution",
            "time_series",
        }
    ):
        return (
            proposal,
            [],
        )


    if (
        proposal.dataset_id
        is None
    ):
        return (
            proposal,
            [],
        )


    datasets = (
        catalog_index(
            catalog
        )
    )


    dataset = (
        datasets.get(
            proposal.dataset_id
        )
    )


    if (
        dataset is None
    ):
        return (
            proposal,
            [],
        )


    normalizations: list[
        str
    ] = []


    # ========================================================
    # ASSOCIATION FAMILY
    #
    # If Gemma selected the wrong association family but copied
    # the exact x/y columns correctly, Python can determine the
    # valid family from the deterministic column types.
    #
    # This normalization is intentionally narrow:
    # - quantitative + quantitative -> quantitative_association
    # - categorical/boolean + categorical/boolean
    #     -> categorical_association
    # - mixed/unknown types remain untouched and are rejected by
    #   the normal validators.
    # ========================================================

    if (
        proposal.family
        in {
            "quantitative_association",
            "categorical_association",
        }
        and
        proposal.x_column
        is not None
        and
        proposal.y_column
        is not None
    ):
        x_profile = (
            find_column(
                dataset,
                proposal.x_column,
            )
        )


        y_profile = (
            find_column(
                dataset,
                proposal.y_column,
            )
        )


        target_family: str | None = (
            None
        )


        if (
            x_profile is not None
            and
            y_profile is not None
        ):
            if (
                is_quantitative(
                    x_profile.analysis_kind
                )
                and
                is_quantitative(
                    y_profile.analysis_kind
                )
            ):
                target_family = (
                    "quantitative_association"
                )


            elif (
                is_categorical(
                    x_profile.analysis_kind
                )
                and
                is_categorical(
                    y_profile.analysis_kind
                )
            ):
                target_family = (
                    "categorical_association"
                )


        if (
            target_family is not None
            and
            target_family
            !=
            proposal.family
        ):
            original_family = (
                proposal.family
            )


            proposal = (
                proposal.model_copy(
                    update={
                        "family":
                            target_family,
                    }
                )
            )


            normalizations.append(
                (
                    "Python a normalisé la famille d'association "
                    "à partir des types analytiques déterministes "
                    "des colonnes exactes : "
                    f"{original_family} -> {target_family}. "
                    f"x={proposal.x_column}, "
                    f"y={proposal.y_column}."
                )
            )


        # Association families never aggregate their x/y inputs.
        # A small local model can otherwise identify the exact two
        # quantitative/categorical columns and family, but leak a
        # stray aggregation_function such as ``sum``. Because the
        # deterministic catalog has already proved that both x/y
        # roles are valid for the selected association family,
        # Python may safely repair this WIRE-PROTOCOL invariant.
        #
        # This does not invent a column, dataset, family or metric;
        # it only removes an operation that the association family
        # explicitly forbids. The two roles must also reference two
        # DISTINCT physical columns: x == y is a common malformed wire
        # output and must remain rejected rather than being converted
        # into a meaningless self-association. Invalid/mixed x/y types
        # are not repaired here and continue to be rejected downstream.
        if (
            target_family is not None
            and
            proposal.x_column
            !=
            proposal.y_column
            and
            proposal.aggregation_function
            !=
            "none"
        ):
            original_aggregation = (
                proposal.aggregation_function
            )


            proposal = (
                proposal.model_copy(
                    update={
                        "aggregation_function":
                            "none",
                    }
                )
            )


            normalizations.append(
                (
                    "Python a supprimé une agrégation incompatible "
                    "avec une famille d'association validée par les "
                    "types analytiques déterministes : "
                    f"aggregation={original_aggregation} -> none, "
                    f"family={proposal.family}, "
                    f"x={proposal.x_column}, "
                    f"y={proposal.y_column}."
                )
            )


        return (
            proposal,
            normalizations,
        )


    # ========================================================
    # GROUP COMPARISON
    #
    # Canonical native roles:
    #   group_column = categorical / boolean
    #   value_column = quantitative
    #
    # Small local models can correctly identify one canonical
    # role while leaking the other column into x/y or another
    # generic role. Python repairs only the wire protocol when
    # the deterministic catalog makes the mapping unambiguous.
    #
    # Examples:
    #
    #   x=department, y=salary
    #     -> group=department, value=salary
    #
    #   group=region, y=revenue
    #     -> group=region, value=revenue
    #
    #   value=revenue, x=region
    #     -> group=region, value=revenue
    #
    # No new column is invented and no semantic substitution is
    # performed.
    # ========================================================

    if (
        proposal.family
        ==
        "group_comparison"
    ):
        group_column = (
            proposal.group_column
        )


        value_column = (
            proposal.value_column
        )


        generic_candidates = [
            (
                "x_column",
                proposal.x_column,
            ),
            (
                "y_column",
                proposal.y_column,
            ),
            (
                "time_column",
                proposal.time_column,
            ),
            (
                "dimension_column",
                proposal.dimension_column,
            ),
            (
                "entity_column",
                proposal.entity_column,
            ),
        ]


        # ----------------------------------------------------
        # Repair missing group role when exactly one generic
        # proposal column is categorical / boolean.
        # ----------------------------------------------------

        if (
            group_column is None
        ):
            categorical_candidates: list[
                tuple[
                    str,
                    str,
                ]
            ] = []


            for (
                field_name,
                candidate_name,
            ) in generic_candidates:
                if (
                    candidate_name is None
                    or
                    not candidate_name.strip()
                    or
                    candidate_name ==
                    value_column
                ):
                    continue


                candidate_profile = (
                    find_column(
                        dataset,
                        candidate_name,
                    )
                )


                if (
                    candidate_profile is not None
                    and
                    is_categorical(
                        candidate_profile
                        .analysis_kind
                    )
                ):
                    categorical_candidates.append(
                        (
                            field_name,
                            candidate_name,
                        )
                    )


            unique_categorical_names = list(
                dict.fromkeys(
                    candidate_name

                    for (
                        _,
                        candidate_name,
                    )
                    in categorical_candidates
                )
            )


            if (
                len(
                    unique_categorical_names
                )
                ==
                1
            ):
                group_column = (
                    unique_categorical_names[
                        0
                    ]
                )


                normalizations.append(
                    (
                        "Python a normalisé le protocole du planner "
                        "pour `group_comparison` : l'unique colonne "
                        "catégorielle placée dans un rôle générique "
                        "a été remappée vers group_column. "
                        f"group={group_column}."
                    )
                )


        # ----------------------------------------------------
        # Repair missing value role when exactly one generic
        # proposal column is quantitative.
        # ----------------------------------------------------

        if (
            value_column is None
        ):
            quantitative_candidates: list[
                tuple[
                    str,
                    str,
                ]
            ] = []


            for (
                field_name,
                candidate_name,
            ) in generic_candidates:
                if (
                    candidate_name is None
                    or
                    not candidate_name.strip()
                    or
                    candidate_name ==
                    group_column
                ):
                    continue


                candidate_profile = (
                    find_column(
                        dataset,
                        candidate_name,
                    )
                )


                if (
                    candidate_profile is not None
                    and
                    is_quantitative(
                        candidate_profile
                        .analysis_kind
                    )
                ):
                    quantitative_candidates.append(
                        (
                            field_name,
                            candidate_name,
                        )
                    )


            unique_quantitative_names = list(
                dict.fromkeys(
                    candidate_name

                    for (
                        _,
                        candidate_name,
                    )
                    in quantitative_candidates
                )
            )


            if (
                len(
                    unique_quantitative_names
                )
                ==
                1
            ):
                value_column = (
                    unique_quantitative_names[
                        0
                    ]
                )


                normalizations.append(
                    (
                        "Python a normalisé le protocole du planner "
                        "pour `group_comparison` : l'unique colonne "
                        "quantitative placée dans un rôle générique "
                        "a été remappée vers value_column. "
                        f"value={value_column}."
                    )
                )


        # ----------------------------------------------------
        # Canonical output only when both roles are resolved.
        # ----------------------------------------------------

        if (
            group_column is not None
            and
            value_column is not None
        ):
            irrelevant_fields = {
                "x_column":
                    proposal.x_column,

                "y_column":
                    proposal.y_column,

                "time_column":
                    proposal.time_column,

                "dimension_column":
                    proposal.dimension_column,

                "entity_column":
                    proposal.entity_column,
            }


            populated_irrelevant = [
                field_name

                for (
                    field_name,
                    field_value,
                )
                in irrelevant_fields.items()

                if (
                    field_value
                    is not None
                    and
                    str(
                        field_value
                    )
                    .strip()
                )
            ]


            if (
                populated_irrelevant
            ):
                normalizations.append(
                    (
                        "Python a supprimé des rôles de protocole "
                        "non pertinents pour `group_comparison` : "
                        f"{', '.join(populated_irrelevant)}. "
                        "Le contrat canonique conserve uniquement "
                        "group/value."
                    )
                )


            return (
                proposal.model_copy(
                    update={
                        "x_column":
                            None,

                        "y_column":
                            None,

                        "group_column":
                            group_column,

                        "value_column":
                            value_column,

                        "time_column":
                            None,

                        "dimension_column":
                            None,

                        "entity_column":
                            None,
                    }
                ),
                normalizations,
            )


        return (
            proposal,
            normalizations,
        )


    # ========================================================
    # DISTRIBUTION
    # ========================================================

    if (
        proposal.family
        ==
        "distribution"
    ):
        value_column = (
            proposal.value_column
        )


        if (
            value_column is None
        ):
            candidate_values = [
                value

                for value
                in [
                    proposal.x_column,
                    proposal.y_column,
                    proposal.group_column,
                    proposal.time_column,
                    proposal.dimension_column,
                    proposal.entity_column,
                ]

                if (
                    value is not None
                    and
                    value.strip()
                )
            ]


            unique_candidates = list(
                dict.fromkeys(
                    candidate_values
                )
            )


            if (
                len(
                    unique_candidates
                )
                ==
                1
            ):
                candidate_name = (
                    unique_candidates[
                        0
                    ]
                )


                candidate_profile = (
                    find_column(
                        dataset,
                        candidate_name,
                    )
                )


                if (
                    candidate_profile
                    is not None
                    and
                    is_quantitative(
                        candidate_profile
                        .analysis_kind
                    )
                ):
                    value_column = (
                        candidate_name
                    )


                    normalizations.append(
                        (
                            "Python a normalisé le protocole du planner "
                            "pour `distribution` : l'unique colonne "
                            "quantitative proposée dans un rôle générique "
                            "a été remappée vers value_column. "
                            f"value={value_column}."
                        )
                    )


        if (
            value_column is not None
        ):
            irrelevant_fields = {
                "x_column":
                    proposal.x_column,

                "y_column":
                    proposal.y_column,

                "group_column":
                    proposal.group_column,

                "time_column":
                    proposal.time_column,

                "dimension_column":
                    proposal.dimension_column,

                "entity_column":
                    proposal.entity_column,
            }


            populated_irrelevant = [
                field_name

                for (
                    field_name,
                    field_value,
                )
                in irrelevant_fields.items()

                if (
                    field_value
                    is not None
                    and
                    str(
                        field_value
                    )
                    .strip()
                )
            ]


            if (
                populated_irrelevant
            ):
                normalizations.append(
                    (
                        "Python a supprimé des rôles de protocole "
                        "non pertinents pour `distribution` : "
                        f"{', '.join(populated_irrelevant)}. "
                        "Le contrat canonique conserve uniquement "
                        "value."
                    )
                )


            return (
                proposal.model_copy(
                    update={
                        "x_column":
                            None,

                        "y_column":
                            None,

                        "group_column":
                            None,

                        "value_column":
                            value_column,

                        "time_column":
                            None,

                        "dimension_column":
                            None,

                        "entity_column":
                            None,
                    }
                ),
                normalizations,
            )


        return (
            proposal,
            normalizations,
        )


    # ========================================================
    # TIME SERIES
    #
    # Canonical native roles:
    #   time_column  = temporal
    #   value_column = quantitative
    #
    # Small local models can leak one of those roles into x/y.
    # Python repairs the wire protocol only when the dataset
    # catalog makes the mapping deterministic.
    #
    # Supported repairs include:
    #
    #   x=Year, y=salary
    #     -> time=Year, value=salary
    #
    #   time=snapshot_date, y=salary
    #     -> time=snapshot_date, value=salary
    #
    #   value=salary, x=snapshot_date
    #     -> time=snapshot_date, value=salary
    #
    # No semantic substitution is performed here. Objective
    # fidelity validation runs afterwards and can still reject
    # an explicitly requested absent column such as `Year`.
    # ========================================================

    time_column = (
        proposal.time_column
    )


    value_column = (
        proposal.value_column
    )


    # --------------------------------------------------------
    # Candidate role pool.
    # --------------------------------------------------------

    generic_candidates = [
        (
            "x_column",
            proposal.x_column,
        ),
        (
            "y_column",
            proposal.y_column,
        ),
        (
            "group_column",
            proposal.group_column,
        ),
        (
            "dimension_column",
            proposal.dimension_column,
        ),
        (
            "entity_column",
            proposal.entity_column,
        ),
    ]


    # --------------------------------------------------------
    # Repair missing time role when exactly one generic column
    # is temporal.
    # --------------------------------------------------------

    if (
        time_column is None
    ):
        temporal_candidates: list[
            tuple[
                str,
                str,
            ]
        ] = []


        for (
            field_name,
            candidate_name,
        ) in generic_candidates:
            if (
                candidate_name is None
                or
                not candidate_name.strip()
            ):
                continue


            candidate_profile = (
                find_column(
                    dataset,
                    candidate_name,
                )
            )


            if (
                candidate_profile is not None
                and
                is_temporal(
                    candidate_profile
                    .analysis_kind
                )
            ):
                temporal_candidates.append(
                    (
                        field_name,
                        candidate_name,
                    )
                )


        unique_temporal_names = list(
            dict.fromkeys(
                candidate_name

                for (
                    _,
                    candidate_name,
                )
                in temporal_candidates
            )
        )


        if (
            len(
                unique_temporal_names
            )
            ==
            1
        ):
            time_column = (
                unique_temporal_names[
                    0
                ]
            )


            normalizations.append(
                (
                    "Python a normalisé le protocole du planner "
                    "pour `time_series` : l'unique colonne "
                    "temporelle placée dans un rôle générique "
                    "a été remappée vers time_column. "
                    f"time={time_column}."
                )
            )


    # --------------------------------------------------------
    # Repair missing value role when exactly one generic column
    # is quantitative and is not the selected time column.
    # --------------------------------------------------------

    if (
        value_column is None
    ):
        quantitative_candidates: list[
            tuple[
                str,
                str,
            ]
        ] = []


        for (
            field_name,
            candidate_name,
        ) in generic_candidates:
            if (
                candidate_name is None
                or
                not candidate_name.strip()
                or
                candidate_name ==
                time_column
            ):
                continue


            candidate_profile = (
                find_column(
                    dataset,
                    candidate_name,
                )
            )


            if (
                candidate_profile is not None
                and
                is_quantitative(
                    candidate_profile
                    .analysis_kind
                )
            ):
                quantitative_candidates.append(
                    (
                        field_name,
                        candidate_name,
                    )
                )


        unique_quantitative_names = list(
            dict.fromkeys(
                candidate_name

                for (
                    _,
                    candidate_name,
                )
                in quantitative_candidates
            )
        )


        if (
            len(
                unique_quantitative_names
            )
            ==
            1
        ):
            value_column = (
                unique_quantitative_names[
                    0
                ]
            )


            normalizations.append(
                (
                    "Python a normalisé le protocole du planner "
                    "pour `time_series` : l'unique colonne "
                    "quantitative placée dans un rôle générique "
                    "a été remappée vers value_column. "
                    f"value={value_column}."
                )
            )


    # --------------------------------------------------------
    # Canonical output only when both roles are resolved.
    # --------------------------------------------------------

    if (
        time_column is not None
        and
        value_column is not None
    ):
        irrelevant_fields = {
            "x_column":
                proposal.x_column,

            "y_column":
                proposal.y_column,

            "group_column":
                proposal.group_column,

            "dimension_column":
                proposal.dimension_column,

            "entity_column":
                proposal.entity_column,
        }


        populated_irrelevant = [
            field_name

            for (
                field_name,
                field_value,
            )
            in irrelevant_fields.items()

            if (
                field_value is not None
                and
                str(
                    field_value
                )
                .strip()
            )
        ]


        if (
            populated_irrelevant
        ):
            normalizations.append(
                (
                    "Python a supprimé des rôles de protocole "
                    "non pertinents pour `time_series` : "
                    f"{', '.join(populated_irrelevant)}. "
                    "Le contrat canonique conserve uniquement "
                    "time/value."
                )
            )


        return (
            proposal.model_copy(
                update={
                    "x_column":
                        None,

                    "y_column":
                        None,

                    "group_column":
                        None,

                    "value_column":
                        value_column,

                    "time_column":
                        time_column,

                    "dimension_column":
                        None,

                    "entity_column":
                        None,
                }
            ),
            normalizations,
        )


    return (
        proposal,
        normalizations,
    )


# ============================================================
# TYPE RULES
# ============================================================

def is_quantitative(
    kind: str,
) -> bool:
    return (
        kind ==
        "quantitative"
    )


def is_categorical(
    kind: str,
) -> bool:
    return (
        kind
        in {
            "categorical",
            "boolean",
        }
    )


def is_temporal(
    kind: str,
) -> bool:
    normalized = (
        kind
        .strip()
        .lower()
    )


    return (
        normalized ==
        "temporal"
        or
        normalized.startswith(
            "temporal_"
        )
    )


def validate_role_type(
    *,
    family: str,
    role: str,
    column: PlannerColumnProfile,
) -> str | None:
    if (
        family ==
        "quantitative_association"
        and
        role in {
            "x",
            "y",
        }
        and
        not is_quantitative(
            column.analysis_kind
        )
    ):
        return (
            f"`{column.name}` est {column.analysis_kind} "
            f"mais le rôle `{role}` exige une variable quantitative."
        )


    if (
        family ==
        "categorical_association"
        and
        role in {
            "x",
            "y",
        }
        and
        not is_categorical(
            column.analysis_kind
        )
    ):
        return (
            f"`{column.name}` est {column.analysis_kind} "
            f"mais le rôle `{role}` exige une variable catégorielle/booléenne."
        )


    if (
        family ==
        "group_comparison"
        and
        role ==
        "group"
        and
        not is_categorical(
            column.analysis_kind
        )
    ):
        return (
            f"`{column.name}` est {column.analysis_kind} "
            "mais `group` doit être catégorielle/booléenne."
        )


    if (
        family ==
        "group_comparison"
        and
        role ==
        "value"
        and
        not is_quantitative(
            column.analysis_kind
        )
    ):
        return (
            f"`{column.name}` est {column.analysis_kind} "
            "mais `value` doit être quantitative."
        )


    if (
        family ==
        "distribution"
        and
        role ==
        "value"
        and
        not is_quantitative(
            column.analysis_kind
        )
    ):
        return (
            f"`{column.name}` est {column.analysis_kind} "
            "mais une distribution histogramme exige "
            "une variable quantitative."
        )


    if (
        family ==
        "time_series"
        and
        role ==
        "time"
        and
        not is_temporal(
            column.analysis_kind
        )
    ):
        return (
            f"`{column.name}` est {column.analysis_kind} "
            "mais `time` doit appartenir à la famille "
            "des types temporels."
        )


    if (
        family ==
        "time_series"
        and
        role ==
        "value"
        and
        not is_quantitative(
            column.analysis_kind
        )
    ):
        return (
            f"`{column.name}` est {column.analysis_kind} "
            "mais `value` doit être quantitative pour "
            "la série temporelle numérique."
        )


    if (
        family ==
        "inequality"
        and
        role ==
        "value"
        and
        not is_quantitative(
            column.analysis_kind
        )
    ):
        return (
            f"`{column.name}` est {column.analysis_kind} "
            "mais la mesure d'une analyse d'inégalité doit être quantitative."
        )


    return None


# ============================================================
# AGGREGATION / RANKING / WINDOW TRANSLATION
# ============================================================

def aggregation_source_role(
    proposal: AIPlannerProposal,
) -> str | None:
    function = (
        proposal
        .aggregation_function
    )


    if (
        function ==
        "none"
    ):
        return None


    if (
        function ==
        "count"
    ):
        return None


    if (
        function ==
        "distinct_count"
    ):
        if (
            proposal.entity_column
            is not None
        ):
            return "entity"


        if (
            proposal.dimension_column
            is not None
        ):
            return "dimension"


        if (
            proposal.value_column
            is not None
        ):
            return "value"


        return None


    # Numeric aggregations.
    return (
        "value"
        if (
            proposal.value_column
            is not None
        )
        else None
    )


def aggregation_group_roles(
    proposal: AIPlannerProposal,
) -> list[
    str
]:
    if (
        proposal.family ==
        "time_series"
    ):
        return [
            "time",
        ]


    if (
        proposal.family ==
        "ranking"
    ):
        return [
            "dimension",
        ]


    if (
        proposal.family ==
        "aggregation"
    ):
        if (
            proposal.group_column
            is not None
        ):
            return [
                "group",
            ]


        if (
            proposal.dimension_column
            is not None
        ):
            return [
                "dimension",
            ]


    return []


def build_aggregation(
    proposal: AIPlannerProposal,
) -> (
    AggregationSpec
    | None
):
    function = (
        proposal
        .aggregation_function
    )


    if (
        function ==
        "none"
    ):
        return None


    source_role = (
        aggregation_source_role(
            proposal
        )
    )


    return AggregationSpec(
        function=function,  # type: ignore[arg-type]
        source_role=source_role,  # type: ignore[arg-type]
        group_by_roles=(
            aggregation_group_roles(
                proposal
            )
        ),  # type: ignore[arg-type]
        output_name=(
            "planned_metric"
        ),
    )


def build_ranking(
    proposal: AIPlannerProposal,
) -> (
    RankingSpec
    | None
):
    if (
        proposal.ranking_order ==
        "none"
    ):
        return None


    if (
        proposal.ranking_limit
        is None
    ):
        return None


    return RankingSpec(
        order=(
            proposal.ranking_order
        ),  # type: ignore[arg-type]
        limit=(
            proposal.ranking_limit
        ),
    )


def build_window(
    proposal: AIPlannerProposal,
) -> (
    WindowSpec
    | None
):
    if (
        proposal.window_operation ==
        "none"
    ):
        return None


    if (
        proposal.window_size
        is None
    ):
        return None


    return WindowSpec(
        operation=(
            proposal.window_operation
        ),  # type: ignore[arg-type]
        window=(
            proposal.window_size
        ),
        minimum_periods=1,
    )



# ============================================================
# ANALYTICAL VIEW IDENTITY / GRAIN HELPERS
# ============================================================

def normalized_grain(
    value: str | None,
) -> str:
    return (
        normalize_identifier_for_match(
            value
            or
            ""
        )
    )


def dataset_matches_requested_grain(
    *,
    dataset: PlannerDatasetProfile,
    proposal: AIPlannerProposal,
) -> bool:
    requested = normalized_grain(
        proposal.analytical_grain
    )
    available = normalized_grain(
        dataset.analytical_grain
    )


    return bool(
        requested
        and
        available
        and
        requested == available
    )


def dataset_contains_bound_columns(
    *,
    dataset: PlannerDatasetProfile,
    proposal: AIPlannerProposal,
) -> bool:
    bound_names = {
        column_name
        for (
            _,
            column_name,
        )
        in proposed_role_columns(
            proposal
        )
    }


    if not bound_names:
        return False


    available = {
        column.name
        for column
        in dataset.columns
    }


    return bound_names.issubset(
        available
    )


def objective_requests_monthly_time_series(
    objective: str,
) -> bool:
    """
    Return True only for an explicit monthly-series intent.

    A bare mention of a month is not enough because a request such as
    "CA du mois d'août" may be a single-period aggregate. The deterministic
    repair is enabled for explicit monthly cadence words, or for a trend /
    evolution request that also names a monthly period.
    """

    tokens = set(
        normalized_objective_tokens(
            objective
        )
    )


    monthly_cadence = {
        "mensuel",
        "mensuels",
        "mensuelle",
        "mensuelles",
        "monthly",
    }


    monthly_period = {
        "mois",
        "month",
        "months",
    }


    trend_markers = {
        "evolution",
        "evolutions",
        "tendance",
        "tendances",
        "trend",
        "trends",
    }


    return bool(
        tokens.intersection(
            monthly_cadence
        )
        or
        (
            tokens.intersection(
                monthly_period
            )
            and
            tokens.intersection(
                trend_markers
            )
        )
    )


def canonicalize_monthly_analytical_view_intent(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> tuple[
    AIPlannerProposal,
    list[str],
]:
    """
    Resolve a monthly additive analytical view directly from the user
    objective plus server-owned provenance.

    This canonicalizer is intentionally independent of the model's proposed
    columns. Small local models may hallucinate roles such as category or
    total_spend even when the objective clearly asks for monthly revenue.

    Python repairs the proposal only when ALL of the following are true:

    - the user explicitly asks for a monthly time series;
    - the user did not explicitly name a dataset;
    - exactly one server-owned `monthly_additive_measure` view matches;
    - the objective deterministically identifies both the materialized time
      column and the declared target measure of that view.

    No dataset, column, metric or derived variable is invented. Ambiguous
    matches remain unrepaired and continue to the ordinary validation guards.
    """

    if proposal.decision != "propose":
        return proposal, []


    if not objective_requests_monthly_time_series(
        objective
    ):
        return proposal, []


    if explicit_dataset_mentions(
        objective=objective,
        catalog=catalog,
    ):
        return proposal, []


    candidates: list[
        PlannerDatasetProfile
    ] = []


    for dataset in catalog.datasets:
        if (
            not dataset.is_derived
            or
            dataset.derivation_type
            !=
            "monthly_additive_measure"
            or
            dataset.operation
            !=
            "groupby_sum"
            or
            dataset.aggregation
            !=
            "sum"
            or
            not dataset.target_time_column
            or
            not dataset.target_measure_column
        ):
            continue


        mentions = set(
            objective_schema_column_mentions(
                objective=objective,
                dataset=dataset,
            )
        )


        required = {
            dataset.target_time_column,
            dataset.target_measure_column,
        }


        if required.issubset(
            mentions
        ):
            candidates.append(
                dataset
            )


    if len(candidates) != 1:
        return proposal, []


    selected = candidates[0]
    previous = proposal.dataset_id


    normalized = proposal.model_copy(
        update={
            "family": "time_series",
            "dataset_id": selected.dataset_id,
            "analytical_grain": (
                selected.analytical_grain
                or
                "month"
            ),
            "x_column": None,
            "y_column": None,
            "group_column": None,
            "value_column": selected.target_measure_column,
            "time_column": selected.target_time_column,
            "dimension_column": None,
            "entity_column": None,
            "aggregation_function": "sum",
            "ranking_order": "none",
            "ranking_limit": None,
            "window_operation": "none",
            "window_size": None,
        }
    )


    return (
        normalized,
        [
            (
                "Python a normalisé une demande de série temporelle "
                "mensuelle vers la vue analytique additive server-owned : "
                f"dataset_id={selected.dataset_id}, "
                f"time={selected.target_time_column}, "
                f"value={selected.target_measure_column}, "
                f"grain={selected.analytical_grain}, "
                "aggregation=sum"
                +
                (
                    f" (dataset proposé par le modèle : {previous})."
                    if previous and previous != selected.dataset_id
                    else
                    "."
                )
            )
        ],
    )




def canonicalize_categorical_association_from_objective(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> tuple[
    AIPlannerProposal,
    list[str],
]:
    """
    Resolve an explicit categorical association directly from the
    user objective before inferred dataset repair can anchor the
    proposal to a wrong derived analytical view.

    This guard is intentionally conservative:

    - the objective must explicitly ask for a relation/association;
    - the user must not explicitly name a dataset;
    - exactly one NON-DERIVED catalog dataset must contain exactly
      two categorical columns deterministically referenced by the
      objective;
    - no quantitative or third schema column may also be
      deterministically referenced in that same candidate dataset.

    This repairs small-model wire failures such as:

        "Existe-t-il une relation entre le segment client et la
        catégorie de produit ?"

    when Gemma proposes a quantitative association against an
    unrelated revenue-by-category derived view.

    The normalization does not invent a variable or metric. It
    simply binds the two categorical schema columns that the
    objective itself deterministically identifies.
    """

    if proposal.decision != "propose":
        return proposal, []


    if proposal.family not in {
        "quantitative_association",
        "categorical_association",
    }:
        return proposal, []


    if explicit_dataset_mentions(
        objective=objective,
        catalog=catalog,
    ):
        return proposal, []


    objective_tokens = set(
        normalized_objective_tokens(
            objective
        )
    )


    if not (
        objective_tokens
        &
        {
            "relation",
            "association",
            "correlation",
            "relationship",
            "lien",
        }
    ):
        return proposal, []


    candidates: list[
        tuple[
            PlannerDatasetProfile,
            list[str],
        ]
    ] = []


    for dataset in catalog.datasets:
        # Categorical association requires row-level co-occurrence.
        # Do not infer it from aggregated analytical views.
        if dataset.is_derived:
            continue


        mentions = list(
            dict.fromkeys(
                objective_schema_column_mentions(
                    objective=objective,
                    dataset=dataset,
                )
            )
        )


        if len(mentions) != 2:
            continue


        profiles = [
            find_column(
                dataset,
                column_name,
            )
            for column_name
            in mentions
        ]


        if any(
            profile is None
            for profile
            in profiles
        ):
            continue


        if not all(
            is_categorical(
                profile.analysis_kind
            )
            for profile
            in profiles
            if profile is not None
        ):
            continue


        if mentions[0] == mentions[1]:
            continue


        candidates.append(
            (
                dataset,
                mentions,
            )
        )


    if len(candidates) != 1:
        return proposal, []


    selected, mentions = candidates[0]
    previous_dataset = proposal.dataset_id


    normalized = proposal.model_copy(
        update={
            "family":
                "categorical_association",

            "dataset_id":
                selected.dataset_id,

            "analytical_grain":
                (
                    selected.analytical_grain
                    or
                    "row"
                ),

            "x_column":
                mentions[0],

            "y_column":
                mentions[1],

            "group_column":
                None,

            "value_column":
                None,

            "time_column":
                None,

            "dimension_column":
                None,

            "entity_column":
                None,

            "aggregation_function":
                "none",

            "ranking_order":
                "none",

            "ranking_limit":
                None,

            "window_operation":
                "none",

            "window_size":
                None,
        }
    )


    return (
        normalized,
        [
            (
                "Python a résolu une intention d'association "
                "catégorielle explicitement formulée vers l'unique "
                "dataset source server-owned contenant les deux "
                "variables catégorielles déterministiquement "
                "identifiées dans l'objectif : "
                f"dataset_id={selected.dataset_id}, "
                f"x={mentions[0]}, "
                f"y={mentions[1]}, "
                "family=categorical_association, "
                "aggregation=none"
                +
                (
                    f" (dataset proposé par le modèle : "
                    f"{previous_dataset})."
                    if (
                        previous_dataset
                        and
                        previous_dataset
                        !=
                        selected.dataset_id
                    )
                    else
                    "."
                )
            )
        ],
    )


def canonicalize_categorical_additive_view_from_objective(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> tuple[
    AIPlannerProposal,
    list[str],
]:
    """
    Resolve a categorical additive analytical view directly from the
    user objective plus server-owned provenance.

    This is the categorical counterpart of the monthly analytical-view
    canonicalizer. It deliberately runs BEFORE inferred dataset repair so
    a small local model cannot lock the request onto a lower-level session
    or entity view merely because it proposed semantically related columns
    such as ``country`` + ``basket_amount`` for an objective such as
    ``CA par pays``.

    Python rewrites the proposal only when ALL of the following are true:

    - the user did not explicitly name a dataset;
    - exactly one server-owned ``categorical_additive_measure`` view
      matches the objective;
    - the objective deterministically identifies both that view's declared
      grouping column and its declared target measure.

    No dataset, column, metric or aggregation is invented. If zero or more
    than one categorical additive view match, this canonicalizer abstains
    and ordinary validation continues unchanged.
    """

    if proposal.decision != "propose":
        return proposal, []


    if explicit_dataset_mentions(
        objective=objective,
        catalog=catalog,
    ):
        return proposal, []


    candidates: list[
        PlannerDatasetProfile
    ] = []


    for dataset in catalog.datasets:
        if (
            not dataset.is_derived
            or
            dataset.derivation_type
            !=
            "categorical_additive_measure"
            or
            dataset.operation
            !=
            "groupby_sum"
            or
            dataset.aggregation
            !=
            "sum"
            or
            not dataset.group_column
            or
            not dataset.target_measure_column
        ):
            continue


        mentions = set(
            objective_schema_column_mentions(
                objective=objective,
                dataset=dataset,
            )
        )


        required = {
            dataset.group_column,
            dataset.target_measure_column,
        }


        if required.issubset(
            mentions
        ):
            candidates.append(
                dataset
            )


    if len(candidates) != 1:
        return proposal, []


    selected = candidates[0]
    previous = proposal.dataset_id


    normalized = proposal.model_copy(
        update={
            "family": "aggregation",
            "dataset_id": selected.dataset_id,
            "analytical_grain": (
                selected.analytical_grain
                or
                selected.group_column
            ),
            "x_column": None,
            "y_column": None,
            "group_column": selected.group_column,
            "value_column": selected.target_measure_column,
            "time_column": None,
            "dimension_column": None,
            "entity_column": None,
            "aggregation_function": "sum",
            "ranking_order": "none",
            "ranking_limit": None,
            "window_operation": "none",
            "window_size": None,
        }
    )


    return (
        normalized,
        [
            (
                "Python a résolu la demande vers l'unique vue "
                "catégorielle additive server-owned correspondant "
                "explicitement au groupe et à la mesure de l'objectif : "
                f"dataset_id={selected.dataset_id}, "
                f"group={selected.group_column}, "
                f"value={selected.target_measure_column}, "
                f"grain={selected.analytical_grain}, "
                "aggregation=sum"
                +
                (
                    f" (dataset proposé par le modèle : {previous})."
                    if previous and previous != selected.dataset_id
                    else
                    "."
                )
            )
        ],
    )


def canonicalize_entity_measure_group_comparison_from_objective(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> tuple[
    AIPlannerProposal,
    list[str],
]:
    """
    Resolve a group comparison directly from a server-owned entity-level
    analytical view when the objective deterministically identifies one
    quantitative target measure and one categorical grouping variable.

    This canonicalizer runs BEFORE inferred dataset repair. That ordering
    matters for small local models that may identify the right entity-level
    dataset columns but emit the wrong family/roles, for example:

        "Comment le montant du panier varie-t-il selon le pays ?"

    with ``x=basket_amount``, ``y=basket_amount`` and an invented/root
    dataset id.

    Python rewrites the proposal only when ALL of the following are true:

    - the user did not explicitly name a dataset;
    - the proposal is still in a two-variable/distribution family that may
      legitimately be repaired into a group comparison;
    - exactly one server-owned ``entity_additive_measure`` view matches;
    - the objective deterministically identifies that view's declared target
      measure and exactly one categorical column present in the same view.

    The target measure must be explicitly/semantically named in the user
    objective. A vague request such as ``performance selon le pays`` is not
    rewritten to basket_amount or any other business measure.
    """

    if proposal.decision != "propose":
        return proposal, []


    if proposal.family not in {
        "quantitative_association",
        "group_comparison",
        "distribution",
    }:
        return proposal, []


    if explicit_dataset_mentions(
        objective=objective,
        catalog=catalog,
    ):
        return proposal, []


    candidates: list[
        tuple[
            PlannerDatasetProfile,
            str,
        ]
    ] = []


    for dataset in catalog.datasets:
        if (
            not dataset.is_derived
            or
            dataset.derivation_type
            !=
            "entity_additive_measure"
            or
            not dataset.target_measure_column
        ):
            continue


        # For entity-level views, require evidence for the PHYSICAL target
        # measure name itself. Do not use inherited source-measure aliases
        # here: basket_amount may legitimately originate from gross_amount,
        # but a request for "CA" must not become a basket-distribution query.
        target_measure_match = (
            dataset.target_measure_column
            in
            explicit_known_column_mentions(
                objective=objective,
                dataset=dataset,
            )
            or
            semantic_metric_match_score(
                objective=objective,
                column_name=dataset.target_measure_column,
            )
            >
            0
        )


        if not target_measure_match:
            continue


        mentions = list(
            dict.fromkeys(
                objective_schema_column_mentions(
                    objective=objective,
                    dataset=dataset,
                )
            )
        )


        categorical_mentions = []


        for column_name in mentions:
            if (
                column_name
                ==
                dataset.target_measure_column
            ):
                continue


            column = find_column(
                dataset,
                column_name,
            )


            if (
                column is not None
                and
                is_categorical(
                    column.analysis_kind
                )
            ):
                categorical_mentions.append(
                    column.name
                )


        categorical_mentions = list(
            dict.fromkeys(
                categorical_mentions
            )
        )


        if len(categorical_mentions) != 1:
            continue


        candidates.append(
            (
                dataset,
                categorical_mentions[0],
            )
        )


    if len(candidates) != 1:
        return proposal, []


    selected, group_column = candidates[0]
    previous = proposal.dataset_id


    normalized = proposal.model_copy(
        update={
            "family": "group_comparison",
            "dataset_id": selected.dataset_id,
            "analytical_grain": (
                selected.analytical_grain
                or
                selected.entity_column
                or
                group_column
            ),
            "x_column": None,
            "y_column": None,
            "group_column": group_column,
            "value_column": selected.target_measure_column,
            "time_column": None,
            "dimension_column": None,
            "entity_column": None,
            "aggregation_function": "none",
            "ranking_order": "none",
            "ranking_limit": None,
            "window_operation": "none",
            "window_size": None,
        }
    )


    return (
        normalized,
        [
            (
                "Python a résolu la demande vers l'unique vue "
                "entity-level server-owned correspondant explicitement "
                "à la mesure et au groupe de l'objectif : "
                f"dataset_id={selected.dataset_id}, "
                f"group={group_column}, "
                f"value={selected.target_measure_column}, "
                f"grain={selected.analytical_grain}"
                +
                (
                    f" (dataset proposé par le modèle : {previous})."
                    if previous and previous != selected.dataset_id
                    else
                    "."
                )
            )
        ],
    )


def canonicalize_inferred_dataset_reference(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> tuple[
    AIPlannerProposal,
    list[str],
]:
    """
    Repair only an absent or hallucinated dataset_id when the exact
    columns already emitted by the model identify one deterministic
    catalog dataset.

    If several datasets contain the same bound columns, an exact
    analytical-grain match is preferred. No column is invented and
    an explicit user dataset reference is never overridden.
    """

    if proposal.decision != "propose":
        return proposal, []


    if explicit_dataset_mentions(
        objective=objective,
        catalog=catalog,
    ):
        return proposal, []


    known = catalog_index(
        catalog
    )


    if (
        proposal.dataset_id
        is not None
        and
        proposal.dataset_id in known
    ):
        return proposal, []


    candidates = [
        dataset
        for dataset
        in catalog.datasets
        if dataset_contains_bound_columns(
            dataset=dataset,
            proposal=proposal,
        )
    ]


    exact_grain = [
        dataset
        for dataset
        in candidates
        if dataset_matches_requested_grain(
            dataset=dataset,
            proposal=proposal,
        )
    ]


    if exact_grain:
        candidates = exact_grain


    if len(candidates) != 1:
        return proposal, []


    selected = candidates[0]
    previous = proposal.dataset_id


    return (
        proposal.model_copy(
            update={
                "dataset_id":
                    selected.dataset_id,
            }
        ),
        [
            (
                "Python a réparé un dataset_id absent ou inconnu "
                "à partir des colonnes exactes déjà proposées et "
                "du grain analytique server-owned : "
                f"dataset_id={selected.dataset_id}"
                +
                (
                    f" à la place de {previous}."
                    if previous
                    else
                    "."
                )
            )
        ],
    )


def canonicalize_analytical_view_intent(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> tuple[
    AIPlannerProposal,
    list[str],
]:
    """
    Promote a server-owned categorical additive analytical view to
    the canonical aggregation wire shape when the user objective
    deterministically identifies both its grouping column and its
    declared target measure.

    This repairs family/role mistakes from a small local model without
    inventing a dataset, a column or a metric.
    """

    if (
        proposal.decision != "propose"
        or
        proposal.dataset_id is None
    ):
        return proposal, []


    dataset = catalog_index(
        catalog
    ).get(
        proposal.dataset_id
    )


    if (
        dataset is None
        or
        not dataset.is_derived
        or
        dataset.derivation_type
        !=
        "categorical_additive_measure"
        or
        dataset.operation
        !=
        "groupby_sum"
        or
        dataset.aggregation
        !=
        "sum"
        or
        not dataset.group_column
        or
        not dataset.target_measure_column
    ):
        return proposal, []


    mentions = set(
        objective_schema_column_mentions(
            objective=objective,
            dataset=dataset,
        )
    )


    required = {
        dataset.group_column,
        dataset.target_measure_column,
    }


    if not required.issubset(
        mentions
    ):
        return proposal, []


    normalized = proposal.model_copy(
        update={
            "family": "aggregation",
            "analytical_grain": (
                dataset.analytical_grain
                or
                proposal.analytical_grain
            ),
            "x_column": None,
            "y_column": None,
            "group_column": dataset.group_column,
            "value_column": dataset.target_measure_column,
            "time_column": None,
            "dimension_column": None,
            "entity_column": None,
            "aggregation_function": "sum",
            "ranking_order": "none",
            "ranking_limit": None,
            "window_operation": "none",
            "window_size": None,
        }
    )


    return (
        normalized,
        [
            (
                "Python a normalisé la proposition vers la vue "
                "analytique additive server-owned : "
                f"family=aggregation, "
                f"group={dataset.group_column}, "
                f"value={dataset.target_measure_column}, "
                f"grain={dataset.analytical_grain}, "
                "aggregation=sum."
            )
        ],
    )



# ============================================================
# DETERMINISTIC DATASET / INTENT ABSTENTION GUARDS
# ============================================================

def dataset_supports_canonical_proposal(
    *,
    dataset: PlannerDatasetProfile,
    proposal: AIPlannerProposal,
) -> bool:
    role_pairs = (
        proposed_role_columns(
            proposal
        )
    )


    if not role_pairs:
        return False


    for (
        role,
        column_name,
    ) in role_pairs:
        column = (
            find_column(
                dataset,
                column_name,
            )
        )


        if (
            column is None
        ):
            return False


        if (
            validate_role_type(
                family=(
                    proposal.family
                ),
                role=(
                    role
                ),
                column=(
                    column
                ),
            )
            is not None
        ):
            return False


    return True


def compatible_datasets_for_proposal(
    *,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> list[
    PlannerDatasetProfile
]:
    compatible = [
        dataset

        for dataset
        in catalog.datasets

        if dataset_supports_canonical_proposal(
            dataset=(
                dataset
            ),
            proposal=(
                proposal
            ),
        )
    ]


    exact_grain = [
        dataset

        for dataset
        in compatible

        if dataset_matches_requested_grain(
            dataset=dataset,
            proposal=proposal,
        )
    ]


    # Exact analytical grain is stronger evidence than mere
    # column compatibility. Fall back only when no catalog
    # dataset declares the requested grain.
    return (
        exact_grain
        if exact_grain
        else compatible
    )


def clear_executable_bindings(
    proposal: AIPlannerProposal,
    *,
    decision: PlannerDecision,
    blocker: str,
    reason: str,
    clear_dataset: bool,
) -> AIPlannerProposal:
    return (
        proposal.model_copy(
            update={
                "decision":
                    decision,

                "dataset_id":
                    (
                        None
                        if clear_dataset
                        else
                        proposal.dataset_id
                    ),

                "x_column":
                    None,

                "y_column":
                    None,

                "group_column":
                    None,

                "value_column":
                    None,

                "time_column":
                    None,

                "dimension_column":
                    None,

                "entity_column":
                    None,

                "blockers":
                    [
                        blocker,
                    ],

                "reasons":
                    [
                        reason,
                    ],
            }
        )
    )


# ============================================================
# DETERMINISTIC AGGREGATION / RANKING INTENT NORMALIZATION
#
# Python normalizes only explicit analytical cues and only with
# columns already proposed by the LLM and confirmed by catalog
# types. It never invents a metric or grouping variable.
# ============================================================

def explicit_aggregation_from_objective(
    objective: str,
) -> WireAggregation:
    tokens = set(
        normalized_objective_tokens(
            objective
        )
    )


    if tokens & {
        "moyen", "moyenne", "moyennes", "moyens",
        "mean", "average", "avg",
    }:
        return "mean"


    if tokens & {
        "median", "mediane", "medians", "medianes",
    }:
        return "median"


    if tokens & {
        "somme", "sum", "total", "totale", "totaux",
    }:
        return "sum"


    if tokens & {
        "minimum", "minimums", "min",
    }:
        return "min"


    if tokens & {
        "maximum", "maximums", "max",
    }:
        return "max"


    return "none"


def explicit_ranking_order_from_objective(
    objective: str,
) -> WireRankingOrder:
    normalized = " ".join(
        normalized_objective_tokens(
            objective
        )
    )


    descending_patterns = [
        r"\bplus (?:eleve|elevee|eleves|elevees)\b",
        r"\bplus (?:haut|haute|hauts|hautes)\b",
        r"\bplus (?:grand|grande|grands|grandes)\b",
        r"\bhighest\b",
        r"\blargest\b",
        r"\btop\b",
        r"\bdecroissant(?:e|es|s)?\b",
        r"\bdescending\b",
    ]


    ascending_patterns = [
        r"\bplus (?:faible|faibles)\b",
        r"\bplus (?:bas|basse|basses)\b",
        r"\bplus (?:petit|petite|petits|petites)\b",
        r"\blowest\b",
        r"\bsmallest\b",
        r"\bcroissant(?:e|es|s)?\b",
        r"\bascending\b",
    ]


    for pattern in descending_patterns:
        if re.search(
            pattern,
            normalized,
        ):
            return "descending"


    for pattern in ascending_patterns:
        if re.search(
            pattern,
            normalized,
        ):
            return "ascending"


    return "none"


def explicit_ranking_limit_from_objective(
    objective: str,
    *,
    ranking_order: WireRankingOrder,
) -> int | None:
    if ranking_order == "none":
        return None


    tokens = normalized_objective_tokens(
        objective
    )


    normalized = " ".join(
        tokens
    )


    numeric_match = re.search(
        r"\b(?:top|les|the|premiers?|premieres?)\s+(\d{1,3})\b",
        normalized,
    )


    if numeric_match is not None:
        return min(
            100,
            max(
                1,
                int(
                    numeric_match.group(
                        1
                    )
                ),
            ),
        )


    number_words = {
        "un": 1,
        "une": 1,
        "one": 1,
        "deux": 2,
        "two": 2,
        "trois": 3,
        "three": 3,
        "quatre": 4,
        "four": 4,
        "cinq": 5,
        "five": 5,
        "six": 6,
        "six": 6,
        "sept": 7,
        "seven": 7,
        "huit": 8,
        "eight": 8,
        "neuf": 9,
        "nine": 9,
        "dix": 10,
        "ten": 10,
    }


    for index, token in enumerate(
        tokens
    ):
        value = number_words.get(
            token
        )


        if value is None:
            continue


        local_window = set(
            tokens[
                max(0, index - 3):
                min(len(tokens), index + 4)
            ]
        )


        if local_window & {
            "categorie", "categories",
            "groupe", "groupes",
            "category", "group", "groups",
            "top",
        }:
            return value


    if (
        "quelle" in tokens
        or
        "quel" in tokens
        or
        "which" in tokens
    ):
        return 1


    return 10


def canonicalize_aggregation_ranking_intent(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> tuple[
    AIPlannerProposal,
    list[str],
]:
    if (
        proposal.decision != "propose"
        or
        proposal.dataset_id is None
        or
        proposal.family not in {
            "group_comparison",
            "aggregation",
            "ranking",
        }
    ):
        return proposal, []


    dataset = catalog_index(
        catalog
    ).get(
        proposal.dataset_id
    )


    if dataset is None:
        return proposal, []


    normalizations: list[str] = []


    explicit_aggregation = (
        explicit_aggregation_from_objective(
            objective
        )
    )


    aggregation_function = (
        proposal.aggregation_function
    )


    if (
        explicit_aggregation != "none"
        and
        aggregation_function != explicit_aggregation
    ):
        aggregation_function = explicit_aggregation
        normalizations.append(
            "Python a confirmé l'agrégation explicitement "
            "demandée dans l'objectif : "
            f"aggregation={aggregation_function}."
        )


    categorical_candidates: list[str] = []


    for candidate_name in [
        proposal.dimension_column,
        proposal.group_column,
        proposal.x_column,
        proposal.y_column,
    ]:
        if not candidate_name:
            continue


        profile = find_column(
            dataset,
            candidate_name,
        )


        if (
            profile is not None
            and
            is_categorical(
                profile.analysis_kind
            )
        ):
            categorical_candidates.append(
                candidate_name
            )


    categorical_candidates = list(
        dict.fromkeys(
            categorical_candidates
        )
    )


    dimension_column = (
        categorical_candidates[0]
        if len(categorical_candidates) == 1
        else None
    )


    quantitative_candidates: list[str] = []


    for candidate_name in [
        proposal.value_column,
        proposal.x_column,
        proposal.y_column,
        proposal.group_column,
        proposal.dimension_column,
    ]:
        if (
            not candidate_name
            or
            candidate_name == dimension_column
        ):
            continue


        profile = find_column(
            dataset,
            candidate_name,
        )


        if (
            profile is not None
            and
            is_quantitative(
                profile.analysis_kind
            )
        ):
            quantitative_candidates.append(
                candidate_name
            )


    quantitative_candidates = list(
        dict.fromkeys(
            quantitative_candidates
        )
    )


    value_column = (
        proposal.value_column
        if proposal.value_column in quantitative_candidates
        else (
            quantitative_candidates[0]
            if len(quantitative_candidates) == 1
            else None
        )
    )


    ranking_order = (
        explicit_ranking_order_from_objective(
            objective
        )
    )


    ranking_limit = (
        explicit_ranking_limit_from_objective(
            objective,
            ranking_order=ranking_order,
        )
    )


    if (
        ranking_order != "none"
        and
        aggregation_function != "none"
        and
        dimension_column is not None
        and
        (
            value_column is not None
            or
            aggregation_function == "count"
        )
    ):
        normalizations.append(
            "Python a normalisé une intention de classement "
            "explicitement formulée : "
            f"dimension={dimension_column}, "
            f"value={value_column}, "
            f"aggregation={aggregation_function}, "
            f"order={ranking_order}, "
            f"limit={ranking_limit}."
        )


        return (
            proposal.model_copy(
                update={
                    "family": "ranking",
                    "x_column": None,
                    "y_column": None,
                    "group_column": None,
                    "value_column": value_column,
                    "time_column": None,
                    "dimension_column": dimension_column,
                    "entity_column": None,
                    "aggregation_function": aggregation_function,
                    "ranking_order": ranking_order,
                    "ranking_limit": ranking_limit,
                    "window_operation": "none",
                    "window_size": None,
                }
            ),
            normalizations,
        )


    if (
        explicit_aggregation != "none"
        and
        proposal.family in {
            "group_comparison",
            "aggregation",
        }
        and
        dimension_column is not None
        and
        (
            value_column is not None
            or
            aggregation_function == "count"
        )
    ):
        normalizations.append(
            "Python a normalisé une demande descriptive "
            "explicitement agrégée vers `aggregation` : "
            f"group={dimension_column}, "
            f"value={value_column}, "
            f"aggregation={aggregation_function}."
        )


        return (
            proposal.model_copy(
                update={
                    "family": "aggregation",
                    "x_column": None,
                    "y_column": None,
                    "group_column": dimension_column,
                    "value_column": value_column,
                    "time_column": None,
                    "dimension_column": None,
                    "entity_column": None,
                    "aggregation_function": aggregation_function,
                    "ranking_order": "none",
                    "ranking_limit": None,
                    "window_operation": "none",
                    "window_size": None,
                }
            ),
            normalizations,
        )


    return proposal, normalizations


def apply_deterministic_abstention_guards(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog: PlannerCatalog,
) -> AIPlannerProposal:
    if (
        proposal.decision
        !=
        "propose"
    ):
        return (
            proposal
        )


    explicit_datasets = (
        explicit_dataset_mentions(
            objective=(
                objective
            ),
            catalog=(
                catalog
            ),
        )
    )


    # --------------------------------------------------------
    # Multi-dataset ambiguity.
    #
    # If the objective does not name a dataset and several
    # datasets can satisfy the exact canonical bindings, Python
    # refuses to trust an arbitrary dataset_id selected by the
    # LLM.
    # --------------------------------------------------------

    if (
        len(
            catalog.datasets
        )
        >
        1
        and
        not explicit_datasets
    ):
        compatible = (
            compatible_datasets_for_proposal(
                proposal=(
                    proposal
                ),
                catalog=(
                    catalog
                ),
            )
        )


        # ----------------------------------------------------
        # Objective-first categorical associations intentionally
        # bind to a non-derived source dataset because the
        # association requires row-level category co-occurrence.
        #
        # A derived session/entity view may expose the same two
        # categorical columns, but it is a different analytical
        # grain and must not invalidate an already deterministic
        # source binding. Multiple compatible SOURCE datasets
        # still remain ambiguous and therefore fail closed.
        # ----------------------------------------------------

        selected_dataset = (
            catalog_index(
                catalog
            ).get(
                proposal.dataset_id
            )
            if proposal.dataset_id
            else
            None
        )


        if (
            proposal.family
            ==
            "categorical_association"
            and
            selected_dataset is not None
            and
            not selected_dataset.is_derived
        ):
            source_compatible = [
                dataset

                for dataset
                in compatible

                if not dataset.is_derived
            ]


            if source_compatible:
                compatible = (
                    source_compatible
                )


        if (
            len(
                compatible
            )
            >
            1
        ):
            filenames = [
                dataset.filename

                for dataset
                in compatible
            ]


            return (
                clear_executable_bindings(
                    proposal,
                    decision="ambiguous",
                    blocker=(
                        "AMBIGUÏTÉ DATASET : plusieurs datasets "
                        "peuvent exécuter exactement cette demande "
                        "avec les mêmes rôles de colonnes ("
                        +
                        ", ".join(
                            filenames
                        )
                        +
                        "). Précisez le fichier à utiliser."
                    ),
                    reason=(
                        "Python a refusé le dataset_id choisi par "
                        "le LLM car plusieurs datasets sont "
                        "déterministiquement compatibles."
                    ),
                    clear_dataset=True,
                )
            )


    # --------------------------------------------------------
    # Metric ambiguity for group comparisons.
    #
    # Example:
    #   "Analyse la performance par region."
    #
    # If `region` is explicit but the user did not explicitly
    # name the value measure and multiple quantitative columns
    # exist, choosing `revenue` (or any other measure) would be
    # a semantic invention.
    # --------------------------------------------------------

    if (
        proposal.family
        in {
            "group_comparison",
            "aggregation",
            "ranking",
        }
        and
        proposal.dataset_id
        is not None
        and
        proposal.value_column
        is not None
    ):
        dataset = (
            catalog_index(
                catalog
            )
            .get(
                proposal.dataset_id
            )
        )


        if (
            dataset
            is not None
        ):
            explicit_columns = {
                normalize_identifier_for_match(
                    column_name
                )

                for column_name
                in explicit_known_column_mentions(
                    objective=(
                        objective
                    ),
                    dataset=(
                        dataset
                    ),
                )
            }


            semantic_metric_columns = {
                normalize_identifier_for_match(
                    column_name
                )

                for column_name
                in semantic_quantitative_column_mentions(
                    objective=(
                        objective
                    ),
                    dataset=(
                        dataset
                    ),
                )
            }


            selected_value = (
                normalize_identifier_for_match(
                    proposal.value_column
                )
            )


            quantitative_candidates = [
                column.name

                for column
                in dataset.columns

                if is_quantitative(
                    column.analysis_kind
                )
            ]


            metric_is_literal = (
                selected_value
                in explicit_columns
            )


            metric_is_semantically_resolved = (
                len(
                    semantic_metric_columns
                )
                ==
                1
                and
                selected_value
                in semantic_metric_columns
            )


            if (
                not metric_is_literal
                and
                not metric_is_semantically_resolved
                and
                len(
                    quantitative_candidates
                )
                >
                1
            ):
                return (
                    clear_executable_bindings(
                        proposal,
                        decision="ambiguous",
                        blocker=(
                            "AMBIGUÏTÉ MÉTRIQUE : la demande ne "
                            "précise pas quelle mesure quantitative "
                            "comparer. Plusieurs colonnes sont "
                            "possibles ("
                            +
                            ", ".join(
                                quantitative_candidates
                            )
                            +
                            "). Précisez la métrique."
                        ),
                        reason=(
                            "Python refuse de transformer un concept "
                            "métier vague en une colonne quantitative "
                            "choisie arbitrairement par le LLM."
                        ),
                        clear_dataset=False,
                    )
                )


            if (
                metric_is_semantically_resolved
                and
                not metric_is_literal
            ):
                semantic_reason = (
                    "Python a confirmé la métrique proposée "
                    "par une résolution lexicale déterministe "
                    "et non ambiguë de l'objectif : "
                    f"value={proposal.value_column}."
                )


                proposal = (
                    proposal.model_copy(
                        update={
                            "reasons":
                                list(
                                    dict.fromkeys(
                                        [
                                            *proposal.reasons,
                                            semantic_reason,
                                        ]
                                    )
                                ),
                        }
                    )
                )


    return (
        proposal
    )


# ============================================================
# CONTRACT ID
# ============================================================

def build_contract_id(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    proposal_index: int,
) -> str:
    payload = json.dumps(
        {
            "objective":
                objective,

            "proposal":
                proposal.model_dump(
                    mode="json"
                ),

            "proposal_index":
                proposal_index,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()[
        :16
    ]


    return (
        "ai:"
        f"{digest}:"
        f"{proposal_index:02d}"
    )


# ============================================================
# DETERMINISTIC PROPOSAL VALIDATION
# ============================================================

def validate_ai_proposal(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    proposal_index: int,
    catalog: PlannerCatalog,
) -> AIPlannerValidatedItem:
    raw_proposal = (
        proposal
    )


    (
        proposal,
        dataset_normalizations,
    ) = canonicalize_explicit_dataset_reference(
        objective=(
            objective
        ),
        proposal=(
            proposal
        ),
        catalog=(
            catalog
        ),
    )


    (
        proposal,
        categorical_association_objective_normalizations,
    ) = canonicalize_categorical_association_from_objective(
        objective=(
            objective
        ),
        proposal=(
            proposal
        ),
        catalog=(
            catalog
        ),
    )


    (
        proposal,
        monthly_view_normalizations,
    ) = canonicalize_monthly_analytical_view_intent(
        objective=(
            objective
        ),
        proposal=(
            proposal
        ),
        catalog=(
            catalog
        ),
    )


    (
        proposal,
        categorical_view_objective_normalizations,
    ) = canonicalize_categorical_additive_view_from_objective(
        objective=(
            objective
        ),
        proposal=(
            proposal
        ),
        catalog=(
            catalog
        ),
    )


    (
        proposal,
        entity_group_objective_normalizations,
    ) = canonicalize_entity_measure_group_comparison_from_objective(
        objective=(
            objective
        ),
        proposal=(
            proposal
        ),
        catalog=(
            catalog
        ),
    )


    (
        proposal,
        inferred_dataset_normalizations,
    ) = canonicalize_inferred_dataset_reference(
        objective=(
            objective
        ),
        proposal=(
            proposal
        ),
        catalog=(
            catalog
        ),
    )


    (
        proposal,
        semantic_binding_normalizations,
    ) = canonicalize_explicit_objective_bindings(
        objective=(
            objective
        ),
        proposal=(
            proposal
        ),
        catalog=(
            catalog
        ),
    )


    (
        proposal,
        analytical_view_normalizations,
    ) = canonicalize_analytical_view_intent(
        objective=(
            objective
        ),
        proposal=(
            proposal
        ),
        catalog=(
            catalog
        ),
    )


    (
        proposal,
        wire_normalizations,
    ) = canonicalize_wire_roles(
        proposal=(
            proposal
        ),
        catalog=(
            catalog
        ),
    )


    (
        proposal,
        intent_normalizations,
    ) = canonicalize_aggregation_ranking_intent(
        objective=(
            objective
        ),
        proposal=(
            proposal
        ),
        catalog=(
            catalog
        ),
    )


    normalizations = [
        *dataset_normalizations,
        *categorical_association_objective_normalizations,
        *monthly_view_normalizations,
        *categorical_view_objective_normalizations,
        *entity_group_objective_normalizations,
        *inferred_dataset_normalizations,
        *semantic_binding_normalizations,
        *analytical_view_normalizations,
        *wire_normalizations,
        *intent_normalizations,
    ]


    proposal = (
        apply_deterministic_abstention_guards(
            objective=(
                objective
            ),
            proposal=(
                proposal
            ),
            catalog=(
                catalog
            ),
        )
    )


    datasets = (
        catalog_index(
            catalog
        )
    )


    # ========================================================
    # BLOCKED / AMBIGUOUS
    # ========================================================

    if (
        proposal.decision
        in {
            "blocked",
            "ambiguous",
        }
    ):
        blockers = [
            blocker.strip()
            for blocker
            in proposal.blockers
            if blocker.strip()
        ]


        if not blockers:
            blockers = [
                (
                    "Le planner a demandé une abstention "
                    "sans justification exploitable."
                )
            ]


        contract = (
            AnalyticalContract(
                contract_id=(
                    build_contract_id(
                        objective=objective,
                        proposal=proposal,
                        proposal_index=(
                            proposal_index
                        ),
                    )
                ),
                origin="ai_planner",
                status=(
                    proposal.decision
                ),
                title=(
                    proposal.title
                ),
                request_text=(
                    objective.strip()
                ),
                family=(
                    proposal.family
                ),
                required_dataset_ids=(
                    [
                        proposal.dataset_id
                    ]
                    if (
                        proposal.dataset_id
                        is not None
                    )
                    else []
                ),
                required_dataset_filenames=[],
                analytical_grain=(
                    proposal
                    .analytical_grain
                ),
                bindings=[],
                aggregation=None,
                ranking=None,
                window=None,
                filters=[],
                joins=[],
                derived_variables=[],
                required_operations=[],
                reasons=(
                    proposal.reasons
                ),
                blockers=(
                    blockers
                ),
                planner_confidence=(
                    proposal.confidence
                ),
            )
        )


        return AIPlannerValidatedItem(
            proposal_index=(
                proposal_index
            ),
            validation_status=(
                proposal.decision
            ),
            raw_proposal=(
                raw_proposal
            ),
            proposal=(
                proposal
            ),
            contract=(
                contract
            ),
            errors=[],
            warnings=[],
            normalizations=(
                normalizations
            ),
        )


    # ========================================================
    # BASIC PROPOSAL GUARDS
    # ========================================================

    errors: list[
        str
    ] = []


    warnings: list[
        str
    ] = []


    if (
        proposal.family ==
        "unresolved"
    ):
        errors.append(
            (
                "family='unresolved' ne peut pas "
                "être exécutée avec decision='propose'."
            )
        )


    if (
        proposal.dataset_id
        is None
    ):
        errors.append(
            (
                "Une proposition exécutable doit "
                "indiquer dataset_id."
            )
        )

        dataset = None


    else:
        dataset = (
            datasets.get(
                proposal.dataset_id
            )
        )


        if (
            dataset is None
        ):
            errors.append(
                (
                    "Dataset halluciné ou inconnu : "
                    f"`{proposal.dataset_id}`."
                )
            )


    # ========================================================
    # OBJECTIVE → COLUMN FIDELITY
    # ========================================================

    errors.extend(
        validate_objective_column_fidelity(
            objective=(
                objective
            ),
            proposal=(
                proposal
            ),
            catalog=(
                catalog
            ),
            dataset=(
                dataset
            ),
        )
    )


    # ========================================================
    # ROLE PRESENCE
    # ========================================================

    role_pairs = (
        proposed_role_columns(
            proposal
        )
    )


    present_roles = {
        role
        for (
            role,
            _
        )
        in role_pairs
    }


    required_roles = (
        expected_roles_for_family(
            proposal.family
        )
    )


    missing_roles = (
        required_roles -
        present_roles
    )


    if (
        missing_roles
    ):
        errors.append(
            (
                "Rôle(s) obligatoire(s) manquant(s) pour "
                f"{proposal.family} : "
                +
                ", ".join(
                    sorted(
                        missing_roles
                    )
                )
                +
                "."
            )
        )


    # ========================================================
    # FAMILY-SPECIFIC STRUCTURAL RULES
    # ========================================================

    if (
        proposal.family ==
        "time_series"
    ):
        if (
            proposal.aggregation_function ==
            "none"
        ):
            errors.append(
                (
                    "Une time_series exige "
                    "aggregation_function."
                )
            )


        if (
            proposal.aggregation_function
            in {
                "sum",
                "mean",
                "median",
                "min",
                "max",
            }
            and
            proposal.value_column
            is None
        ):
            errors.append(
                (
                    "Une time_series avec une agrégation "
                    "numérique exige value_column."
                )
            )


        if (
            proposal.aggregation_function ==
            "distinct_count"
            and
            proposal.entity_column
            is None
        ):
            errors.append(
                (
                    "Une time_series avec distinct_count "
                    "exige entity_column."
                )
            )


    if (
        proposal.family ==
        "ranking"
    ):
        if (
            proposal.aggregation_function ==
            "none"
        ):
            errors.append(
                (
                    "Un ranking exige "
                    "aggregation_function."
                )
            )


        if (
            proposal.ranking_order ==
            "none"
        ):
            errors.append(
                (
                    "Un ranking exige "
                    "ranking_order."
                )
            )


        if (
            proposal.ranking_limit
            is None
        ):
            errors.append(
                (
                    "Un ranking exige "
                    "ranking_limit."
                )
            )


    if (
        proposal.family
        in {
            "aggregation",
            "descriptive_metric",
        }
        and
        proposal.aggregation_function ==
        "none"
    ):
        errors.append(
            (
                f"{proposal.family} exige "
                "aggregation_function."
            )
        )


    if (
        proposal.family ==
        "inequality"
        and
        (
            proposal.entity_column
            is None
            or
            proposal.value_column
            is None
        )
    ):
        errors.append(
            (
                "Une analyse d'inégalité exige "
                "entity_column et value_column."
            )
        )


    # ========================================================
    # AGGREGATION SOURCE RULES
    # ========================================================

    if (
        proposal.aggregation_function
        in {
            "sum",
            "mean",
            "median",
            "min",
            "max",
        }
        and
        proposal.value_column
        is None
    ):
        errors.append(
            (
                "Une agrégation numérique exige "
                "value_column."
            )
        )


    if (
        proposal.aggregation_function ==
        "distinct_count"
        and
        (
            proposal.entity_column
            is None
            and
            proposal.dimension_column
            is None
            and
            proposal.value_column
            is None
        )
    ):
        errors.append(
            (
                "distinct_count exige une colonne source."
            )
        )


    # ========================================================
    # COLUMN EXISTENCE + TYPES
    # ========================================================

    resolved_bindings: list[
        VariableBinding
    ] = []


    if (
        dataset is not None
    ):
        for (
            role,
            column_name,
        ) in role_pairs:
            column = (
                find_column(
                    dataset,
                    column_name,
                )
            )


            if (
                column is None
            ):
                errors.append(
                    (
                        "Colonne halluciné ou inconnue dans "
                        f"{dataset.dataset_id} : "
                        f"`{column_name}`."
                    )
                )

                continue


            type_error = (
                validate_role_type(
                    family=(
                        proposal.family
                    ),
                    role=(
                        role
                    ),
                    column=(
                        column
                    ),
                )
            )


            if (
                type_error
                is not None
            ):
                errors.append(
                    type_error
                )


            resolved_bindings.append(
                VariableBinding(
                    role=role,  # type: ignore[arg-type]
                    column=(
                        column.name
                    ),
                    dataset_id=(
                        dataset.dataset_id
                    ),
                    dataset_filename=(
                        dataset.filename
                    ),
                    semantic_concept=None,
                    analysis_kind=(
                        column.analysis_kind
                    ),
                )
            )


    # ========================================================
    # NUMERIC AGGREGATION TYPE
    # ========================================================

    if (
        dataset is not None
        and
        proposal.aggregation_function
        in {
            "sum",
            "mean",
            "median",
            "min",
            "max",
        }
        and
        proposal.value_column
        is not None
    ):
        value_profile = (
            find_column(
                dataset,
                proposal.value_column,
            )
        )


        if (
            value_profile is not None
            and
            not is_quantitative(
                value_profile
                .analysis_kind
            )
        ):
            errors.append(
                (
                    f"`{proposal.value_column}` est "
                    f"{value_profile.analysis_kind} mais "
                    f"`{proposal.aggregation_function}` "
                    "exige une mesure quantitative."
                )
            )


    # ========================================================
    # REJECT BEFORE CANONICAL CONTRACT
    # ========================================================

    if (
        errors
    ):
        return AIPlannerValidatedItem(
            proposal_index=(
                proposal_index
            ),
            validation_status=(
                "rejected"
            ),
            raw_proposal=(
                raw_proposal
            ),
            proposal=(
                proposal
            ),
            contract=None,
            errors=(
                errors
            ),
            warnings=(
                warnings
            ),
            normalizations=(
                normalizations
            ),
        )


    # ========================================================
    # TRANSLATE WIRE PROTOCOL TO CANONICAL CONTRACT
    # ========================================================

    aggregation = (
        build_aggregation(
            proposal
        )
    )


    ranking = (
        build_ranking(
            proposal
        )
    )


    window = (
        build_window(
            proposal
        )
    )


    required_operations = [
        (
            "Validate AI-proposed analytical plan "
            "against the deterministic dataset catalog."
        ),
        (
            "Execute only through a deterministic "
            f"{proposal.family} tool."
        ),
    ]


    try:
        proposed_contract = (
            AnalyticalContract(
                contract_id=(
                    build_contract_id(
                        objective=objective,
                        proposal=proposal,
                        proposal_index=(
                            proposal_index
                        ),
                    )
                ),
                origin="ai_planner",
                status="proposed",
                title=(
                    proposal.title
                ),
                request_text=(
                    objective.strip()
                ),
                family=(
                    proposal.family
                ),
                required_dataset_ids=[
                    dataset.dataset_id
                ],
                required_dataset_filenames=[
                    dataset.filename
                ],
                analytical_grain=(
                    proposal
                    .analytical_grain
                ),
                bindings=(
                    resolved_bindings
                ),
                aggregation=(
                    aggregation
                ),
                ranking=(
                    ranking
                ),
                window=(
                    window
                ),
                filters=[],
                joins=[],
                derived_variables=[],
                required_operations=(
                    required_operations
                ),
                reasons=[
                    (
                        "Le LLM local a proposé la famille "
                        "analytique et les rôles de colonnes."
                    ),
                    *normalizations,
                    (
                        "Python a vérifié dataset_id, noms "
                        "de colonnes, types analytiques et "
                        "invariants du contrat avant "
                        "promotion vers `validated`."
                    ),
                ],
                blockers=[],
                planner_confidence=(
                    proposal.confidence
                ),
            )
        )


    except Exception as error:
        return AIPlannerValidatedItem(
            proposal_index=(
                proposal_index
            ),
            validation_status=(
                "rejected"
            ),
            raw_proposal=(
                raw_proposal
            ),
            proposal=(
                proposal
            ),
            contract=None,
            errors=[
                (
                    "Le contrat canonique ne respecte "
                    "pas les invariants structurels : "
                    f"{error}"
                ),
            ],
            warnings=(
                warnings
            ),
        )


    # ========================================================
    # PYTHON PROMOTION
    #
    # The model cannot emit this status. Only the deterministic
    # validator can promote proposed -> validated.
    # ========================================================

    validated_contract = (
        proposed_contract
        .model_copy(
            update={
                "status":
                    "validated",
            }
        )
    )


    return AIPlannerValidatedItem(
        proposal_index=(
            proposal_index
        ),
        validation_status=(
            "validated"
        ),
        raw_proposal=(
            raw_proposal
        ),
        proposal=(
            proposal
        ),
        contract=(
            validated_contract
        ),
        errors=[],
        warnings=(
            warnings
        ),
        normalizations=(
            normalizations
        ),
    )


# ============================================================
# COMPLETE RAW OUTPUT VALIDATION
# ============================================================

def validate_ai_planner_output(
    *,
    objective: str,
    raw_output: RawAIPlannerOutput,
    catalog: PlannerCatalog,
    model: str = (
        DEFAULT_AI_PLANNER_MODEL
    ),
    attempt_count: int = 1,
    retry_count: int = 0,
    retry_triggered: bool = False,
    retry_feedback: list[
        str
    ] | None = None,
) -> AIPlannerReport:
    items = [
        validate_ai_proposal(
            objective=(
                objective
            ),
            proposal=(
                proposal
            ),
            proposal_index=(
                index
            ),
            catalog=(
                catalog
            ),
        )

        for (
            index,
            proposal,
        ) in enumerate(
            raw_output.proposals,
            start=1,
        )
    ]


    return AIPlannerReport(
        objective=(
            objective.strip()
        ),
        model=(
            model
        ),
        proposal_count=(
            len(
                items
            )
        ),
        validated_count=sum(
            1
            for item
            in items
            if (
                item.validation_status ==
                "validated"
            )
        ),
        blocked_count=sum(
            1
            for item
            in items
            if (
                item.validation_status ==
                "blocked"
            )
        ),
        ambiguous_count=sum(
            1
            for item
            in items
            if (
                item.validation_status ==
                "ambiguous"
            )
        ),
        rejected_count=sum(
            1
            for item
            in items
            if (
                item.validation_status ==
                "rejected"
            )
        ),
        items=(
            items
        ),
        attempt_count=(
            attempt_count
        ),
        retry_count=(
            retry_count
        ),
        retry_triggered=(
            retry_triggered
        ),
        retry_feedback=(
            retry_feedback
            or []
        ),
        normalization_count=sum(
            len(
                item.normalizations
            )
            for item
            in items
        ),
        normalization_applied=any(
            bool(
                item.normalizations
            )
            for item
            in items
        ),
    )


# ============================================================
# LOCAL GEMMA CALL
# ============================================================

def _generate_raw_ai_plan_with_timing(
    *,
    objective: str,
    catalog: PlannerCatalog,
    model: str = (
        DEFAULT_AI_PLANNER_MODEL
    ),
    validation_feedback: list[
        str
    ] | None = None,
) -> tuple[
    RawAIPlannerOutput,
    float,
    float,
    float,
]:
    prompt_started_at = (
        perf_counter()
    )


    prompt = (
        build_user_prompt(
            objective=(
                objective
            ),
            catalog=(
                catalog
            ),
        )
    )


    if (
        validation_feedback
    ):
        feedback_text = "\n".join(
            f"- {message}"
            for message
            in validation_feedback
        )


        prompt = (
            f"{prompt}\n\n"
            "CORRECTION DEMANDÉE PAR LE VALIDATEUR PYTHON\n"
            "===========================================\n"
            "La tentative précédente a été REJETÉE. "
            "Ne contourne pas le validateur. Corrige uniquement "
            "le choix de famille et/ou les rôles en respectant "
            "strictement les types du catalogue.\n"
            f"{feedback_text}\n\n"
            "Retourne un nouveau plan structuré complet."
        )


    prompt_construction_ms = (
        (
            perf_counter()
            -
            prompt_started_at
        )
        *
        1000.0
    )


    inference_started_at = (
        perf_counter()
    )


    try:
        response = client.chat(
            model=(
                model
            ),
            messages=[
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },
            ],
            format=(
                RawAIPlannerOutput
                .model_json_schema()
            ),
            options={
                "temperature":
                    0,

                "seed":
                    42,

                "num_ctx":
                    4096,
            },
        )


    except Exception as error:
        raise RuntimeError(
            (
                "La génération du plan analytique "
                "par Ollama a échoué."
            )
        ) from error


    model_inference_ms = (
        (
            perf_counter()
            -
            inference_started_at
        )
        *
        1000.0
    )


    parse_started_at = (
        perf_counter()
    )


    content = (
        response
        .message
        .content
    )


    try:
        raw_output = (
            RawAIPlannerOutput
            .model_validate_json(
                content
            )
        )


    except Exception as error:
        raise RuntimeError(
            (
                "Gemma a retourné un plan qui ne "
                "respecte pas le protocole structuré "
                "du AI Planner v0.15."
            )
        ) from error


    structured_parse_ms = (
        (
            perf_counter()
            -
            parse_started_at
        )
        *
        1000.0
    )


    return (
        raw_output,
        prompt_construction_ms,
        model_inference_ms,
        structured_parse_ms,
    )


def generate_raw_ai_plan(
    *,
    objective: str,
    catalog: PlannerCatalog,
    model: str = (
        DEFAULT_AI_PLANNER_MODEL
    ),
    validation_feedback: list[
        str
    ] | None = None,
) -> RawAIPlannerOutput:
    (
        raw_output,
        _,
        _,
        _,
    ) = _generate_raw_ai_plan_with_timing(
        objective=(
            objective
        ),
        catalog=(
            catalog
        ),
        model=(
            model
        ),
        validation_feedback=(
            validation_feedback
        ),
    )


    return raw_output


# ============================================================
# GUARDED CORRECTION RETRY
# ============================================================

def rejected_feedback(
    report: AIPlannerReport,
) -> list[
    str
]:
    feedback: list[
        str
    ] = []


    for item in (
        report.items
    ):
        if (
            item.validation_status
            !=
            "rejected"
        ):
            continue


        feedback.append(
            (
                f"Proposition {item.proposal_index}: "
                f"family={item.proposal.family}."
            )
        )


        feedback.extend(
            item.errors
        )


        if any(
            error.startswith(
                "FIDÉLITÉ OBJECTIF"
            )

            for error
            in item.errors
        ):
            feedback.append(
                (
                    "La demande contient une référence de "
                    "colonne explicite que Python n'a pas "
                    "acceptée. Ne choisis PAS une autre "
                    "colonne à sa place. Si la colonne "
                    "demandée n'existe pas dans le catalogue, "
                    "réémets la proposition avec "
                    "decision='blocked' et explique que la "
                    "variable est absente. N'invente aucune "
                    "variable dérivée."
                )
            )


    for item in (
        report.items
    ):
        if (
            item.validation_status
            !=
            "rejected"
        ):
            continue


        proposal = (
            item.proposal
        )


        if (
            proposal.family ==
            "group_comparison"
            and
            proposal.group_column
            is None
            and
            proposal.value_column
            is None
            and
            proposal.x_column
            is not None
            and
            proposal.y_column
            is not None
        ):
            feedback.append(
                (
                    "Pour group_comparison, n'utilise pas "
                    "x_column/y_column. Réémets exactement "
                    "les mêmes colonnes dans "
                    "group_column/value_column selon leurs "
                    "types du catalogue."
                )
            )


    return feedback


def should_retry_rejected_plan(
    report: AIPlannerReport,
) -> bool:
    return (
        report.validated_count ==
        0
        and
        report.rejected_count >
        0
    )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def plan_analyses_with_ai(
    *,
    objective: str,
    catalog: PlannerCatalog,
    model: str = (
        DEFAULT_AI_PLANNER_MODEL
    ),
) -> AIPlannerReport:
    planner_started_at = (
        perf_counter()
    )


    normalized_objective = (
        objective
        .strip()
    )


    if not normalized_objective:
        raise ValueError(
            "L'objectif utilisateur ne peut pas être vide."
        )


    (
        first_raw_output,
        first_prompt_ms,
        first_inference_ms,
        first_parse_ms,
    ) = _generate_raw_ai_plan_with_timing(
        objective=(
            normalized_objective
        ),
        catalog=(
            catalog
        ),
        model=(
            model
        ),
    )


    first_validation_started_at = (
        perf_counter()
    )


    first_report = (
        validate_ai_planner_output(
            objective=(
                normalized_objective
            ),
            raw_output=(
                first_raw_output
            ),
            catalog=(
                catalog
            ),
            model=(
                model
            ),
            attempt_count=1,
            retry_count=0,
            retry_triggered=False,
            retry_feedback=[],
        )
    )


    first_validation_ms = (
        (
            perf_counter()
            -
            first_validation_started_at
        )
        *
        1000.0
    )


    first_attempt = (
        AIPlannerAttemptTiming(
            attempt_index=1,
            prompt_construction_ms=(
                first_prompt_ms
            ),
            model_inference_ms=(
                first_inference_ms
            ),
            structured_parse_ms=(
                first_parse_ms
            ),
            python_validation_ms=(
                first_validation_ms
            ),
            total_ms=(
                first_prompt_ms
                +
                first_inference_ms
                +
                first_parse_ms
                +
                first_validation_ms
            ),
        )
    )


    if (
        MAX_AI_PLANNER_ATTEMPTS <
        2
        or
        not should_retry_rejected_plan(
            first_report
        )
    ):
        total_ms = (
            (
                perf_counter()
                -
                planner_started_at
            )
            *
            1000.0
        )


        return first_report.model_copy(
            update={
                "timing":
                    AIPlannerTiming(
                        prompt_construction_ms=(
                            first_prompt_ms
                        ),
                        model_inference_ms=(
                            first_inference_ms
                        ),
                        structured_parse_ms=(
                            first_parse_ms
                        ),
                        python_validation_ms=(
                            first_validation_ms
                        ),
                        retry_feedback_ms=0.0,
                        total_ms=(
                            total_ms
                        ),
                        attempts=[
                            first_attempt
                        ],
                    )
            }
        )


    retry_feedback_started_at = (
        perf_counter()
    )


    feedback = (
        rejected_feedback(
            first_report
        )
    )


    retry_feedback_ms = (
        (
            perf_counter()
            -
            retry_feedback_started_at
        )
        *
        1000.0
    )


    (
        second_raw_output,
        second_prompt_ms,
        second_inference_ms,
        second_parse_ms,
    ) = _generate_raw_ai_plan_with_timing(
        objective=(
            normalized_objective
        ),
        catalog=(
            catalog
        ),
        model=(
            model
        ),
        validation_feedback=(
            feedback
        ),
    )


    second_validation_started_at = (
        perf_counter()
    )


    second_report = (
        validate_ai_planner_output(
            objective=(
                normalized_objective
            ),
            raw_output=(
                second_raw_output
            ),
            catalog=(
                catalog
            ),
            model=(
                model
            ),
            attempt_count=2,
            retry_count=1,
            retry_triggered=True,
            retry_feedback=(
                feedback
            ),
        )
    )


    second_validation_ms = (
        (
            perf_counter()
            -
            second_validation_started_at
        )
        *
        1000.0
    )


    second_attempt = (
        AIPlannerAttemptTiming(
            attempt_index=2,
            prompt_construction_ms=(
                second_prompt_ms
            ),
            model_inference_ms=(
                second_inference_ms
            ),
            structured_parse_ms=(
                second_parse_ms
            ),
            python_validation_ms=(
                second_validation_ms
            ),
            total_ms=(
                second_prompt_ms
                +
                second_inference_ms
                +
                second_parse_ms
                +
                second_validation_ms
            ),
        )
    )


    total_ms = (
        (
            perf_counter()
            -
            planner_started_at
        )
        *
        1000.0
    )


    return second_report.model_copy(
        update={
            "timing":
                AIPlannerTiming(
                    prompt_construction_ms=(
                        first_prompt_ms
                        +
                        second_prompt_ms
                    ),
                    model_inference_ms=(
                        first_inference_ms
                        +
                        second_inference_ms
                    ),
                    structured_parse_ms=(
                        first_parse_ms
                        +
                        second_parse_ms
                    ),
                    python_validation_ms=(
                        first_validation_ms
                        +
                        second_validation_ms
                    ),
                    retry_feedback_ms=(
                        retry_feedback_ms
                    ),
                    total_ms=(
                        total_ms
                    ),
                    attempts=[
                        first_attempt,
                        second_attempt,
                    ],
                )
        }
    )

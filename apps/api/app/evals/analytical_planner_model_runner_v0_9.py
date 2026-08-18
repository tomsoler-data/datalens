from __future__ import annotations

import json

from time import perf_counter
from typing import Any

from app.ai.provider import (
    client,
)

from app.evals.analytical_planner_benchmark_v0_9 import (
    AnalyticalPlannerEvalCase,
    build_planner_input_for_case,
)

from app.evals.analytical_planner_contract_v0_9 import (
    ANALYTICAL_PLANNER_CONTRACT_VERSION,
    AnalyticalPlannerCandidate,
)

from app.evals.analytical_planner_scorer_v0_9 import (
    ANALYTICAL_PLANNER_SCORER_VERSION,
    score_analytical_planner_candidate,
)

from app.evals.analytical_planner_validator_v0_9 import (
    ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    validate_analytical_planner_candidate,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_MODEL_RUNNER_VERSION = (
    "analytical_planner_model_runner_v0.9"
)


ANALYTICAL_PLANNER_PROMPT_VERSION = (
    "analytical_planner_prompt_v0.9_baseline"
)


# ============================================================
# MODELS
# ============================================================

PLANNER_MODELS = [
    "ministral-3:3b",
    "qwen3:4b-instruct",
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT_V0_9 = """
Tu es l'Analytical Planner de DataLens.

Ton rôle est volontairement limité.

Les étapes précédentes de DataLens ont déjà déterminé :

- quels datasets sont sémantiquement nécessaires ;
- quels datasets servent uniquement de bridge structurel ;
- quelles relations entre datasets sont validées ;
- quelles clés structurelles doivent être utilisées ;
- si la demande est structurellement exécutable ;
- quelles colonnes sont autorisées comme variables analytiques ;
- quels outils analytiques sont disponibles.

Tu ne dois PAS refaire ce travail.

============================================================
OBJECTIF
============================================================

Pour chaque requirement fourni dans AnalyticalPlannerInput,
construis le plan analytique minimal permettant de répondre
à la demande utilisateur.

Tu dois produire exactement un plan par requirement.

============================================================
REQUIREMENT IDS
============================================================

Réutilise exactement chaque requirement_id fourni.

N'invente jamais de requirement_id.

N'oublie aucun requirement.

N'ajoute aucun requirement supplémentaire.

============================================================
DATASETS
============================================================

Les datasets avec role="semantic" contiennent les informations
analytiques nécessaires.

Les datasets avec role="bridge" servent uniquement à la
connexion structurelle déjà validée par Python.

Un dataset bridge ne devient PAS une source analytique.

N'utilise jamais les colonnes non analytiques d'un dataset
bridge.

============================================================
JOINTURES
============================================================

Les jointures et alignements structurels sont déjà contrôlés
par Python.

Tu ne dois jamais :

- proposer join_datasets ;
- reconstruire une jointure ;
- choisir des clés de jointure ;
- modifier relationship_ids ;
- inventer une relation.

Le dataset matérialisé sera préparé par une autre couche.

============================================================
COLONNES ANALYTIQUES
============================================================

Lorsque tu références une colonne existante, utilise exactement
son qualified_name tel qu'il apparaît dans analytical_columns.

Exemple :

    sales.revenue

N'invente aucune colonne.

Les structural_keys ne sont pas automatiquement des variables
analytiques.

============================================================
OUTILS
============================================================

Chaque requirement contient allowed_analytical_tools.

Utilise uniquement ces outils.

Choisis uniquement les étapes réellement nécessaires pour
répondre à la demande.

N'ajoute pas d'analyse complémentaire non demandée.

============================================================
INTENTS ET FAMILIES
============================================================

Correspondances :

aggregate_metric
    -> aggregation

compare_groups
    -> group_comparison

measure_relationship
    -> association

time_series_analysis
    -> time_series

distribution_analysis
    -> distribution

entity_anomaly_analysis
    -> entity_outlier

data_quality_analysis
    -> data_quality

Respecte ces correspondances exactement.

============================================================
AGGREGATION
============================================================

Pour un total, une moyenne ou une agrégation globale :

    intent = aggregate_metric
    family = aggregation

Utilise aggregate.

Pour une agrégation globale :

    target_grain = "global"

    group_by = null

============================================================
GROUP COMPARISON
============================================================

Pour comparer une variable quantitative selon une catégorie :

    intent = compare_groups
    family = group_comparison

Utilise compare_groups.

============================================================
ASSOCIATION
============================================================

Pour étudier une relation ou une association entre deux
variables :

    intent = measure_relationship
    family = association

Utilise measure_association.

target et value sont analytiquement symétriques.

============================================================
TIME SERIES
============================================================

Pour étudier une évolution temporelle :

    intent = time_series_analysis
    family = time_series

Utilise analyze_time_series.

L'argument date doit être une colonne analytique de type
temporal.

============================================================
DISTRIBUTION
============================================================

Pour analyser la distribution d'une variable :

    intent = distribution_analysis
    family = distribution

Utilise analyze_distribution.

Pour rechercher explicitement des valeurs atypiques d'une
variable :

    intent = distribution_analysis
    family = distribution

Utilise detect_outliers.

Ne transforme pas automatiquement une demande de distribution
en recherche d'outliers.

============================================================
ENTITY OUTLIERS
============================================================

Pour rechercher des entités au comportement inhabituel :

    intent = entity_anomaly_analysis
    family = entity_outlier

Le plan doit contenir dans cet ordre :

1. build_entity_view
2. detect_entity_outliers

Utilise comme entity une colonne d'entité appartenant à un
dataset semantic.

============================================================
DERIVED METRICS
============================================================

Si la demande nécessite une métrique qui n'existe pas encore
mais qui peut être calculée à partir des colonnes disponibles,
utilise derive_metric AVANT l'analyse qui consomme cette
métrique.

Exemple conceptuel :

    conversions
    visits

pour obtenir :

    conversion_rate

Le plan peut être :

1. derive_metric
2. compare_groups

Les inputs doivent utiliser les qualified_name existants.

Le output doit être un nom simple pour la nouvelle métrique.

Pour formula, utilise une expression arithmétique simple et
lisible basée sur les noms de colonnes non qualifiés.

Exemple :

    conversions / visits

============================================================
TARGET GRAIN
============================================================

Choisis le grain analytique nécessaire à la demande.

Valeurs courantes :

- global
- le grain d'un dataset semantic ;
- une entité comme customer ;
- un grain temporel comme day, week, month ou year.

N'invente pas un grain métier qui n'est pas justifié par le
contexte.

============================================================
MINIMALITÉ
============================================================

Le meilleur plan est le plan correct le plus simple.

Évite :

- les étapes redondantes ;
- les analyses exploratoires non demandées ;
- les transformations inutiles ;
- les outils sans rapport avec la question.

============================================================
FORMAT
============================================================

Respecte exactement le schéma JSON fourni.

Ne retourne aucun texte avant ou après le JSON.
""".strip()


# ============================================================
# USER PROMPT
# ============================================================

def build_analytical_planner_user_prompt(
    case: AnalyticalPlannerEvalCase,
) -> str:
    """
    Build the exact model-visible prompt.

    IMPORTANT:

    Expected benchmark plans, notes and scores are never
    included.
    """

    planner_input = (
        build_planner_input_for_case(
            case
        )
    )


    payload = (
        planner_input.model_dump(
            mode="json",
        )
    )


    return (
        "ANALYTICAL PLANNER INPUT:\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n"
        + (
            "Construis uniquement le plan analytique "
            "nécessaire pour chaque requirement."
        )
    )


# ============================================================
# SINGLE MODEL CALL
# ============================================================

def generate_analytical_plan(
    *,
    case: AnalyticalPlannerEvalCase,
    model: str,
) -> tuple[
    AnalyticalPlannerCandidate,
    str,
    float,
]:
    """
    Execute one analytical planner inference.

    No benchmark expectation is exposed to the model.
    """

    started_at = (
        perf_counter()
    )


    response = client.chat(
        model=model,

        messages=[
            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT_V0_9,
            },

            {
                "role":
                    "user",

                "content":
                    build_analytical_planner_user_prompt(
                        case
                    ),
            },
        ],

        format=(
            AnalyticalPlannerCandidate
            .model_json_schema()
        ),

        options={
            "temperature":
                0,
        },

        think=False,
    )


    inference_ms = (
        (
            perf_counter()
            - started_at
        )
        * 1000.0
    )


    raw_content = (
        response
        .message
        .content
        or ""
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate_json(
            raw_content
        )
    )


    return (
        candidate,
        raw_content,
        inference_ms,
    )


# ============================================================
# SINGLE EVALUATED CASE
# ============================================================

def run_analytical_planner_case(
    *,
    case: AnalyticalPlannerEvalCase,
    model: str,
) -> dict[str, Any]:

    planner_input = (
        build_planner_input_for_case(
            case
        )
    )


    raw_content: (
        str
        | None
    ) = None


    inference_ms = 0.0


    try:

        (
            candidate,
            raw_content,
            inference_ms,
        ) = (
            generate_analytical_plan(
                case=case,
                model=model,
            )
        )


    except Exception as error:

        return {
            "case_id":
                case.case_id,

            "domain":
                case.domain,

            "model":
                model,

            "status":
                "generation_error",

            "candidate":
                None,

            "validation":
                None,

            "score":
                None,

            "exact":
                False,

            "inference_ms":
                inference_ms,

            "raw_content":
                raw_content,

            "error":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
        }


    # ========================================================
    # REAL DETERMINISTIC VALIDATOR
    # ========================================================

    validation = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    # ========================================================
    # DEVELOPMENT SCORER
    # ========================================================

    score = (
        score_analytical_planner_candidate(
            candidate=candidate,

            expected=(
                case.expected
            ),

            planner_input=(
                planner_input
            ),
        )
    )


    exact = (
        score.overall
        == 1.0
    )


    return {
        "case_id":
            case.case_id,

        "domain":
            case.domain,

        "model":
            model,

        "status":
            "ready",

        "candidate":
            candidate.model_dump(
                mode="json",
            ),

        "validation": {
            "valid":
                validation.valid,

            "validated_requirement_ids":
                validation.validated_requirement_ids,

            "issues": [
                issue.model_dump(
                    mode="json",
                )

                for issue
                in validation.issues
            ],
        },

        "score":
            score.as_dict(),

        "exact":
            exact,

        "inference_ms":
            inference_ms,

        "raw_content":
            raw_content,

        "error":
            None,
    }


# ============================================================
# METADATA
# ============================================================

def analytical_planner_runner_metadata() -> dict[
    str,
    Any,
]:

    return {
        "runner_version":
            ANALYTICAL_PLANNER_MODEL_RUNNER_VERSION,

        "prompt_version":
            ANALYTICAL_PLANNER_PROMPT_VERSION,

        "contract_version":
            ANALYTICAL_PLANNER_CONTRACT_VERSION,

        "validator_version":
            ANALYTICAL_PLANNER_VALIDATOR_VERSION,

        "scorer_version":
            ANALYTICAL_PLANNER_SCORER_VERSION,

        "models":
            PLANNER_MODELS,

        "temperature":
            0,

        "thinking":
            False,
    }
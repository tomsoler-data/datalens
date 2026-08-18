from __future__ import annotations

import json

from time import perf_counter
from typing import Any

from app.ai.provider import (
    client,
)

from app.evals.dataset_dependency_contract_v0_8 import (
    DATASET_DEPENDENCY_CONTRACT_VERSION,
    DatasetDependencyCandidate,
)

from app.evals.decision_router_benchmark_v0_7 import (
    DecisionRouterEvalCase,
)


# ============================================================
# VERSION
# ============================================================

DATASET_DEPENDENCY_EXTRACTOR_VERSION = (
    "dataset_dependency_extractor_v0.8"
)


DATASET_DEPENDENCY_PROMPT_VERSION = (
    "dataset_dependency_prompt_v0.8_baseline"
)


MODEL = (
    "qwen3:4b-instruct"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT_V0_8 = """
Tu es le Dataset Dependency Extractor de DataLens.

Ta seule responsabilité est d'identifier quels datasets sont
nécessaires pour répondre à la demande analytique de
l'utilisateur.

Tu ne dois PAS :

- choisir un test statistique ;
- construire un plan analytique ;
- sélectionner des outils ;
- déterminer si une jointure est possible ;
- décider si la demande doit être refusée ;
- décider si une relation entre datasets est valide.

Ces responsabilités appartiennent à d'autres composants de
DataLens.

============================================================
CONCEPT : ANALYTICAL REQUIREMENT
============================================================

Un "requirement" représente UN résultat analytique demandé par
l'utilisateur.

Pour chaque résultat analytique, indique uniquement les
datasets qui doivent participer à CE MÊME résultat.

============================================================
RÈGLE 1 — UN SEUL DATASET
============================================================

Si un résultat peut être calculé entièrement avec un seul
dataset, le requirement doit contenir uniquement ce dataset.

Exemple conceptuel :

Demande :

    "Quel est le chiffre d'affaires total ?"

Si revenue existe dans sales :

    ["sales"]

Même si d'autres datasets sont disponibles, ne les ajoute pas
s'ils ne sont pas nécessaires à ce résultat.

============================================================
RÈGLE 2 — DATASETS NÉCESSAIRES ENSEMBLE
============================================================

Si un même résultat analytique dépend de variables provenant
de plusieurs datasets, place ces datasets dans le MÊME
requirement.

Exemple conceptuel :

Dataset A contient :

    satisfaction

Dataset B contient :

    revenue

Demande :

    "La satisfaction est-elle associée au revenu ?"

Le même résultat dépend des deux sources :

    ["dataset_a", "dataset_b"]

IMPORTANT :

Tu identifies uniquement la dépendance sémantique.

Tu ne dois pas vérifier toi-même si ces datasets peuvent être
joints ou combinés.

Même si leur combinaison est impossible, ils restent les
datasets nécessaires à la demande.

============================================================
RÈGLE 3 — ANALYSES INDÉPENDANTES
============================================================

Si l'utilisateur demande plusieurs résultats indépendants,
crée plusieurs requirements.

Exemple conceptuel :

    "Calcule le total des ventes et, séparément,
     le nombre de tickets support."

Résultat 1 :

    ["sales"]

Résultat 2 :

    ["support"]

Ne produis PAS :

    ["sales", "support"]

car aucun résultat unique ne nécessite les deux sources
ensemble.

============================================================
RÈGLE 4 — DATASETS NON PERTINENTS
============================================================

N'inclus jamais un dataset uniquement parce qu'il est
disponible.

Un dataset doit apparaître uniquement s'il contient une
information nécessaire au résultat demandé.

============================================================
RÈGLE 5 — IDENTIFIANTS
============================================================

Utilise exactement les dataset_id fournis dans le contexte.

N'invente jamais :

- de dataset ;
- de nouvelle source ;
- de dataset_id ;
- de relation entre datasets.

============================================================
RÈGLE 6 — NOMBRE DE REQUIREMENTS
============================================================

Crée le nombre minimal de requirements permettant de
représenter correctement les résultats demandés.

Ne duplique pas inutilement un même résultat.

============================================================
RÈGLE 7 — REQUIREMENT_ID
============================================================

requirement_id sert uniquement d'identifiant local.

Utilise un identifiant court et descriptif.

Exemples :

    total_revenue
    support_revenue_relationship
    sales_total
    support_total

La valeur exacte du requirement_id n'est pas évaluée.

============================================================
QUESTION MENTALE
============================================================

Pour chaque résultat demandé, demande-toi :

"Quelles informations sont nécessaires pour produire CE
résultat, et dans quels datasets ces informations existent ?"

Les datasets contenant ces informations appartiennent au même
requirement.

============================================================
IMPORTANT
============================================================

Ne confonds jamais :

"les datasets nécessaires au même résultat"

avec :

"tous les datasets mentionnés ou disponibles".

Ne tente pas de résoudre les problèmes de jointure.

Le composant Python de DataLens vérifiera ensuite :

- les capacités disponibles ;
- les relations validées ;
- les chemins entre datasets ;
- la faisabilité structurelle.

Ta tâche s'arrête à l'identification des dépendances
sémantiques.

============================================================
FORMAT
============================================================

Respecte exactement le schéma JSON fourni.

N'ajoute aucun texte avant ou après le JSON.
""".strip()


# ============================================================
# MODEL-VISIBLE DATASET
# ============================================================

def _serialize_dataset(
    dataset: Any,
) -> dict[str, Any]:
    """
    Serialize only schema information useful for semantic
    dependency extraction.

    Relationship metadata and tool availability are
    deliberately excluded.
    """

    return {
        "dataset_id":
            dataset.dataset_id,

        "filename":
            dataset.filename,

        "grain":
            dataset.grain,

        "entity_columns":
            dataset.entity_columns,

        "columns": [
            column.model_dump(
                mode="json",
            )

            for column
            in dataset.columns
        ],
    }


# ============================================================
# USER PROMPT
# ============================================================

def build_dependency_user_prompt(
    case: DecisionRouterEvalCase,
) -> str:
    """
    IMPORTANT:

    The model receives:

    - user request;
    - dataset schemas.

    The model does NOT receive:

    - expected answer;
    - benchmark notes;
    - available tools;
    - validated relationships;
    - routing decision.

    This isolates semantic dependency extraction.
    """

    visible_context = {
        "user_request":
            case.user_request,

        "datasets": [
            _serialize_dataset(
                dataset
            )

            for dataset
            in case.datasets
        ],
    }


    return (
        "CONTEXTE:\n\n"
        + json.dumps(
            visible_context,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n"
        + (
            "Identifie uniquement les groupes de datasets "
            "nécessaires aux résultats analytiques demandés."
        )
    )


# ============================================================
# SINGLE EXTRACTION
# ============================================================

def extract_dataset_dependencies(
    *,
    case: DecisionRouterEvalCase,
) -> tuple[
    DatasetDependencyCandidate,
    str,
    float,
]:
    """
    Run one semantic dependency extraction.

    No feasibility decision is performed here.
    """

    started_at = (
        perf_counter()
    )


    response = client.chat(
        model=MODEL,

        messages=[
            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT_V0_8,
            },

            {
                "role":
                    "user",

                "content":
                    build_dependency_user_prompt(
                        case
                    ),
            },
        ],

        format=(
            DatasetDependencyCandidate
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
        DatasetDependencyCandidate
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
# METADATA
# ============================================================

def extractor_metadata() -> dict[
    str,
    Any,
]:

    return {
        "extractor_version":
            DATASET_DEPENDENCY_EXTRACTOR_VERSION,

        "prompt_version":
            DATASET_DEPENDENCY_PROMPT_VERSION,

        "contract_version":
            DATASET_DEPENDENCY_CONTRACT_VERSION,

        "model":
            MODEL,

        "temperature":
            0,

        "thinking":
            False,
    }
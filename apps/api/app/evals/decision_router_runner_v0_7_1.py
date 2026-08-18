from __future__ import annotations

import json

from time import perf_counter
from typing import Any

from app.ai.provider import (
    client,
)

from app.evals.decision_router_benchmark_v0_7 import (
    DecisionRouterEvalCase,
)

from app.evals.decision_router_contract_v0_7 import (
    DECISION_ROUTER_CONTRACT_VERSION,
    DecisionRouterCandidate,
    unwrap_router_candidate,
)

from app.evals.decision_router_scorer_v0_7 import (
    DECISION_ROUTER_SCORER_VERSION,
    score_decision_router_candidate,
)

from app.evals.ollama_baseline import (
    build_tool_catalog,
)


# ============================================================
# VERSION
# ============================================================

DECISION_ROUTER_RUNNER_VERSION_V071 = (
    "decision_router_runner_v0.7.1"
)


PROMPT_VERSION_V071 = (
    "decision_router_prompt_v0.7.1_feasibility_first"
)


# ============================================================
# MODEL
#
# Qwen is currently the best Decision Router candidate.
#
# This development iteration runs on TRAIN only.
# Validation must not be used until this prompt iteration has
# been evaluated on the development train split.
# ============================================================

MODEL = (
    "qwen3:4b-instruct"
)


# ============================================================
# SYSTEM PROMPT v0.7.1
# ============================================================

SYSTEM_PROMPT_V0_7_1 = """
Tu es le Decision Router de DataLens.

Ton unique responsabilité est de décider si une demande
analytique doit :

- être transmise au planner analytique ;
- demander une clarification à l'utilisateur ;
- être arrêtée parce qu'elle ne peut pas être correctement
  satisfaite avec les données et capacités disponibles.

Tu ne réalises PAS l'analyse.
Tu ne sélectionnes PAS le plan statistique.
Tu ne produis PAS d'appels d'outils.

============================================================
RÈGLE PRINCIPALE : FEASIBILITY FIRST
============================================================

Avant de considérer une clarification, vérifie TOUJOURS dans
cet ordre :

1. les capacités nécessaires sont-elles disponibles ?
2. les informations nécessaires sont-elles disponibles ?
3. la demande est-elle suffisamment précise ?

Une clarification ne doit jamais servir à compenser :

- une colonne indispensable absente ;
- un dataset indispensable absent ;
- une opération analytique non supportée ;
- l'absence d'identification causale.

============================================================
ORDRE DE DÉCISION
============================================================

Applique mentalement cet ordre.

ÉTAPE A — CAUSALITÉ

Si l'utilisateur demande explicitement d'établir qu'une
variable A A CAUSÉ une variable B et que le contexte fourni
est observationnel sans capacité causale appropriée :

decision = "cannot_answer"
decision_reason = "causal_identification_missing"

Une association statistique ne permet pas de répondre à une
question causale.

------------------------------------------------------------
ÉTAPE B — CAPACITÉ ANALYTIQUE
------------------------------------------------------------

Vérifie que les outils disponibles permettent réellement
l'opération demandée.

Exemples :

- analyser une série temporelle descriptive avec un outil
  temporel disponible peut être possible ;

- PRÉDIRE une valeur future nécessite une capacité de
  prévision ;

- combiner deux datasets nécessite une capacité permettant
  réellement leur combinaison au grain approprié.

Si l'opération nécessaire n'existe pas dans les capacités
fournies :

decision = "cannot_answer"
decision_reason = "unsupported_analysis"

Ne transforme pas une prévision en simple analyse historique.

Ne suppose pas que deux datasets peuvent être combinés
simplement parce qu'ils sont tous les deux disponibles.

------------------------------------------------------------
ÉTAPE C — DISPONIBILITÉ DES INFORMATIONS
------------------------------------------------------------

Vérifie ensuite si les informations indispensables à la
question existent réellement.

Il existe deux situations différentes.

1. missing_column

Utilise cette raison lorsqu'un dataset pertinent est bien
présent mais qu'une variable indispensable au calcul ou à
l'analyse y manque.

Exemple conceptuel :

L'utilisateur demande une métrique nécessitant :

    revenu - coût

Le dataset contient :

    revenu

mais aucune information de coût.

Il ne faut PAS demander :

    "Comment veux-tu définir cette métrique ?"

si, quelle que soit la définition raisonnable, une information
indispensable n'est pas disponible.

Dans ce cas :

decision = "cannot_answer"
decision_reason = "missing_column"

2. missing_dataset

Utilise cette raison lorsqu'une information ou un résultat
d'un autre domaine/source est nécessaire mais n'est représenté
dans aucun dataset fourni.

Exemple conceptuel :

L'utilisateur demande une relation entre :

    activité d'assistance
    et
    résiliation client

mais les datasets fournis décrivent uniquement l'activité
d'assistance et ne contiennent aucune information de
résiliation.

Dans ce cas :

decision = "cannot_answer"
decision_reason = "missing_dataset"

IMPORTANT :

Ne demande pas à l'utilisateur de définir une variable absente
si cette définition ne rendrait toujours pas les observations
nécessaires disponibles.

Une clarification ne crée pas des données.

------------------------------------------------------------
ÉTAPE D — AMBIGUÏTÉ
------------------------------------------------------------

Seulement après avoir confirmé que les données et capacités
nécessaires existent, vérifie si l'utilisateur doit faire un
choix analytique.

Utilise :

decision = "needs_clarification"
decision_reason = "ambiguous_request"

lorsque plusieurs analyses raisonnables sont possibles avec
les informations déjà disponibles.

Exemples conceptuels :

- "Quels magasins performent le mieux ?"

  alors que plusieurs métriques existantes peuvent définir
  la performance ;

- "Compare les régions."

  alors que plusieurs mesures disponibles pourraient être
  comparées.

Ici, les données existent.

Le problème est uniquement de savoir QUELLE interprétation
l'utilisateur souhaite.

Pose une question ciblée sur ce choix.

------------------------------------------------------------
ÉTAPE E — CONTEXTE / RÉFÉRENCE
------------------------------------------------------------

Utilise :

decision = "needs_clarification"
decision_reason = "insufficient_context"

lorsque les données nécessaires sont présentes mais que
l'interprétation dépend d'une référence métier externe que
l'utilisateur peut fournir.

Exemples conceptuels :

- "Ce délai est-il élevé ?"
- "Ce taux est-il acceptable ?"
- "Sommes-nous au-dessus de l'objectif ?"

sans :

- seuil ;
- SLA ;
- benchmark ;
- objectif ;
- valeur de référence.

La variable existe.

Ce qui manque est une RÉFÉRENCE pour interpréter sa valeur.

Ceci est différent de ambiguous_request.

------------------------------------------------------------
ÉTAPE F — ANALYZE
------------------------------------------------------------

Choisis :

decision = "analyze"

uniquement lorsque :

- la demande est suffisamment précise ;
- les informations indispensables existent ;
- les datasets nécessaires existent ;
- la capacité analytique nécessaire existe.

Pour analyze :

decision_reason = null
clarification_question = null

============================================================
DIFFÉRENCES À NE PAS CONFONDRE
============================================================

AMBIGUOUS_REQUEST

Les informations existent.
Plusieurs choix analytiques raisonnables existent.
L'utilisateur doit choisir.

→ needs_clarification


INSUFFICIENT_CONTEXT

La donnée existe.
Une référence métier externe manque pour interpréter
correctement le résultat.

→ needs_clarification


MISSING_COLUMN

Le dataset pertinent existe.
Une variable indispensable à la réponse n'existe pas.

→ cannot_answer


MISSING_DATASET

Une source d'information indispensable entière n'est pas
représentée dans le contexte fourni.

→ cannot_answer


UNSUPPORTED_ANALYSIS

Les données peuvent être disponibles mais l'opération
nécessaire n'existe pas dans les capacités fournies.

→ cannot_answer


CAUSAL_IDENTIFICATION_MISSING

La question demande une conclusion causale qui ne peut pas
être établie avec le contexte analytique fourni.

→ cannot_answer

============================================================
TEST MENTAL AVANT NEEDS_CLARIFICATION
============================================================

Avant toute décision needs_clarification, demande-toi :

"Si l'utilisateur répond parfaitement à ma question,
les données et outils déjà fournis suffiront-ils réellement
pour effectuer l'analyse ?"

Si la réponse est NON :

NE choisis PAS needs_clarification.

Choisis cannot_answer avec la raison appropriée.

============================================================
QUESTION DE CLARIFICATION
============================================================

Si needs_clarification est réellement approprié :

- pose UNE question ;
- demande uniquement l'information manquante ;
- sois spécifique ;
- n'invente pas toi-même le choix.

Évite :

"Peux-tu préciser ?"

Préfère une question indiquant exactement ce qui doit être
défini.

============================================================
FORMAT
============================================================

Respecte exactement le schéma JSON fourni.

N'ajoute aucun texte avant ou après le JSON.
""".strip()


# ============================================================
# USER PROMPT
# ============================================================

def build_router_user_prompt_v071(
    case: DecisionRouterEvalCase,
) -> str:
    """
    Serialize only information visible to the router.

    Ground truth is never sent.
    """

    visible_context = {
        "user_request":
            case.user_request,

        "datasets": [
            dataset.model_dump(
                mode="json",
            )

            for dataset
            in case.datasets
        ],

        "available_tools":
            build_tool_catalog(
                case.available_tools,
            ),
    }


    context_json = json.dumps(
        visible_context,
        ensure_ascii=False,
        indent=2,
    )


    return (
        "CONTEXTE DISPONIBLE:\n\n"
        f"{context_json}\n\n"
        "Applique l'ordre de décision feasibility-first. "
        "Décide uniquement entre analyze, "
        "needs_clarification et cannot_answer."
    )


# ============================================================
# SINGLE CASE
# ============================================================

def run_router_case_v071(
    *,
    case: DecisionRouterEvalCase,
) -> dict[str, Any]:

    prompt = (
        build_router_user_prompt_v071(
            case,
        )
    )


    started_at = perf_counter()

    raw_content: str | None = None


    try:
        response = client.chat(
            model=MODEL,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT_V0_7_1,
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },
            ],

            format=(
                DecisionRouterCandidate
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
            DecisionRouterCandidate
            .model_validate_json(
                raw_content,
            )
        )


        route = (
            unwrap_router_candidate(
                candidate,
            )
        )


        score = (
            score_decision_router_candidate(
                case=case,
                candidate=candidate,
            )
        )


        return {
            "case_id":
                case.case_id,

            "split":
                case.split,

            "domain":
                case.domain,

            "user_request":
                case.user_request,

            "expected_decision":
                case.expected.decision,

            "expected_reason":
                case.expected.decision_reason,

            "status":
                "ready",

            "inference_ms":
                inference_ms,

            "candidate":
                route.model_dump(
                    mode="json",
                ),

            "score":
                score.as_dict(),

            "overall":
                score.overall,

            "raw_content":
                raw_content,

            "error":
                None,
        }


    except Exception as error:

        inference_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1000.0
        )


        return {
            "case_id":
                case.case_id,

            "split":
                case.split,

            "domain":
                case.domain,

            "user_request":
                case.user_request,

            "expected_decision":
                case.expected.decision,

            "expected_reason":
                case.expected.decision_reason,

            "status":
                "generation_error",

            "inference_ms":
                inference_ms,

            "candidate":
                None,

            "score":
                None,

            "overall":
                0.0,

            "raw_content":
                raw_content,

            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }


# ============================================================
# HELPERS
# ============================================================

def _average(
    values: list[float],
) -> float:

    if not values:
        return 0.0


    return (
        sum(
            values,
        )
        / len(
            values,
        )
    )


# ============================================================
# MODEL REPORT
# ============================================================

def run_router_train_v071(
    cases: list[
        DecisionRouterEvalCase
    ],
) -> dict[str, Any]:

    results = [
        run_router_case_v071(
            case=case,
        )

        for case
        in cases
    ]


    ready_results = [
        result

        for result
        in results

        if (
            result[
                "status"
            ]
            == "ready"
        )
    ]


    error_results = [
        result

        for result
        in results

        if (
            result[
                "status"
            ]
            != "ready"
        )
    ]


    # ========================================================
    # CORE METRICS
    # ========================================================

    overall_scores: list[
        float
    ] = []


    decision_scores: list[
        float
    ] = []


    reason_scores: list[
        float
    ] = []


    route_quality_scores: list[
        float
    ] = []


    for result in results:

        overall_scores.append(
            float(
                result[
                    "overall"
                ]
            )
        )


        score = (
            result.get(
                "score"
            )
        )


        if score is None:

            decision_scores.append(
                0.0
            )

            reason_scores.append(
                0.0
            )

            route_quality_scores.append(
                0.0
            )

            continue


        metrics = (
            score[
                "metrics"
            ]
        )


        decision_scores.append(
            float(
                metrics[
                    "decision"
                ]
            )
        )


        reason_scores.append(
            float(
                metrics[
                    "decision_reason"
                ]
            )
        )


        route_quality_scores.append(
            float(
                metrics[
                    "route_quality"
                ]
            )
        )


    # ========================================================
    # ACCURACY BY ROUTE
    # ========================================================

    expected_routes = [
        "analyze",
        "needs_clarification",
        "cannot_answer",
    ]


    accuracy_by_expected: dict[
        str,
        float
    ] = {}


    for expected_route in expected_routes:

        route_results = [
            result

            for result
            in results

            if (
                result[
                    "expected_decision"
                ]
                == expected_route
            )
        ]


        correct = sum(
            1

            for result
            in route_results

            if (
                result.get(
                    "candidate"
                )
                is not None

                and result[
                    "candidate"
                ][
                    "decision"
                ]
                == expected_route
            )
        )


        accuracy_by_expected[
            expected_route
        ] = (
            correct
            / len(
                route_results,
            )
            if route_results
            else 0.0
        )


    # ========================================================
    # CLARIFICATION QUALITY
    # ========================================================

    clarification_scores: list[
        float
    ] = []


    for result in ready_results:

        if (
            result[
                "expected_decision"
            ]
            != "needs_clarification"
        ):
            continue


        if (
            result[
                "candidate"
            ][
                "decision"
            ]
            != "needs_clarification"
        ):
            continue


        clarification_score = (
            result[
                "score"
            ][
                "metrics"
            ][
                "clarification"
            ]
        )


        if clarification_score is not None:

            clarification_scores.append(
                float(
                    clarification_score
                )
            )


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    unsafe_execution_count = 0

    false_abstention_count = 0

    wrong_abstention_type_count = 0


    for result in ready_results:

        diagnostics = (
            result[
                "score"
            ][
                "diagnostics"
            ]
        )


        if diagnostics[
            "unsafe_execution"
        ]:
            unsafe_execution_count += 1


        if diagnostics[
            "false_abstention"
        ]:
            false_abstention_count += 1


        if diagnostics[
            "wrong_abstention_type"
        ]:
            wrong_abstention_type_count += 1


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    actual_routes = [
        "analyze",
        "needs_clarification",
        "cannot_answer",
        "generation_error",
    ]


    confusion_matrix = {
        expected_route: {
            actual_route:
                0

            for actual_route
            in actual_routes
        }

        for expected_route
        in expected_routes
    }


    for result in results:

        expected_route = (
            result[
                "expected_decision"
            ]
        )


        if (
            result[
                "candidate"
            ]
            is None
        ):
            actual_route = (
                "generation_error"
            )

        else:
            actual_route = (
                result[
                    "candidate"
                ][
                    "decision"
                ]
            )


        confusion_matrix[
            expected_route
        ][
            actual_route
        ] += 1


    # ========================================================
    # LATENCY
    # ========================================================

    average_inference_ms = _average(
        [
            float(
                result[
                    "inference_ms"
                ]
            )

            for result
            in results
        ]
    )


    return {
        "runner_version":
            DECISION_ROUTER_RUNNER_VERSION_V071,

        "prompt_version":
            PROMPT_VERSION_V071,

        "contract_version":
            DECISION_ROUTER_CONTRACT_VERSION,

        "scorer_version":
            DECISION_ROUTER_SCORER_VERSION,

        "model":
            MODEL,

        "case_count":
            len(
                results
            ),

        "generation_success_count":
            len(
                ready_results
            ),

        "generation_error_count":
            len(
                error_results
            ),

        "average_overall":
            _average(
                overall_scores,
            ),

        "decision_accuracy":
            _average(
                decision_scores,
            ),

        "reason_accuracy":
            _average(
                reason_scores,
            ),

        "route_quality":
            _average(
                route_quality_scores,
            ),

        "clarification_quality":
            (
                _average(
                    clarification_scores,
                )
                if clarification_scores
                else 0.0
            ),

        "accuracy_by_expected":
            accuracy_by_expected,

        "confusion_matrix":
            confusion_matrix,

        "diagnostics": {
            "unsafe_execution_count":
                unsafe_execution_count,

            "false_abstention_count":
                false_abstention_count,

            "wrong_abstention_type_count":
                wrong_abstention_type_count,
        },

        "average_inference_ms":
            average_inference_ms,

        "results":
            results,
    }
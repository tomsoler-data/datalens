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

DECISION_ROUTER_RUNNER_VERSION_V072 = (
    "decision_router_runner_v0.7.2"
)


PROMPT_VERSION_V072 = (
    "decision_router_prompt_v0.7.2_cross_dataset_feasibility"
)


MODEL = (
    "qwen3:4b-instruct"
)


# ============================================================
# SYSTEM PROMPT v0.7.2
# ============================================================

SYSTEM_PROMPT_V0_7_2 = """
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
PRINCIPE CENTRAL : FEASIBILITY FIRST
============================================================

Avant toute clarification, vérifie dans cet ordre :

1. Quelle information la demande exige-t-elle ?
2. Dans quel dataset chaque information existe-t-elle ?
3. Faut-il mettre des informations provenant de plusieurs
   datasets dans un même cadre analytique ?
4. Si oui, existe-t-il une manière valide et supportée de
   construire ce cadre analytique ?
5. Seulement ensuite, vérifie si un choix utilisateur manque.

Une clarification ne doit jamais servir à compenser :

- une variable indispensable absente ;
- une source indispensable absente ;
- une capacité analytique absente ;
- une combinaison inter-datasets impossible ;
- une identification causale impossible.

============================================================
A — CAUSALITÉ
============================================================

Si l'utilisateur demande explicitement d'établir qu'une
variable A A CAUSÉ une variable B et que le contexte est
observationnel sans dispositif ou capacité causale appropriée :

decision = "cannot_answer"
decision_reason = "causal_identification_missing"

Une association statistique ne prouve pas une causalité.

============================================================
B — IDENTIFIER LES DÉPENDANCES DE LA DEMANDE
============================================================

Détermine quelles variables ou informations sont réellement
nécessaires.

Ne considère pas automatiquement tous les datasets fournis
comme nécessaires.

Si la demande peut être entièrement satisfaite avec un seul
dataset, la présence d'autres datasets non pertinents ne doit
pas empêcher l'analyse.

De même, si l'utilisateur demande plusieurs résultats
indépendants pouvant être calculés séparément dans plusieurs
datasets, aucune jointure n'est nécessaire.

============================================================
C — ANALYSE INTER-DATASETS
============================================================

Une analyse est inter-datasets lorsque le même résultat
analytique doit combiner des informations provenant de
plusieurs datasets.

Exemples conceptuels :

- mesurer une association entre une variable du dataset A
  et une variable du dataset B ;

- comparer une métrique provenant d'une source avec une
  métrique provenant d'une autre source ;

- construire une métrique qui dépend simultanément
  d'informations situées dans plusieurs sources.

Dans ce cas, vérifie explicitement la faisabilité de la
combinaison AVANT de choisir analyze.

------------------------------------------------------------
C1 — CAPACITÉ DE COMBINAISON ABSENTE
------------------------------------------------------------

Si les informations nécessaires existent dans les datasets
fournis, mais qu'aucune capacité disponible ne permet de
construire le dataset analytique commun nécessaire :

decision = "cannot_answer"
decision_reason = "unsupported_analysis"

IMPORTANT :

Ceci n'est PAS missing_column.

Les colonnes existent.

Ceci n'est PAS missing_dataset.

Les datasets existent.

Le problème est que DataLens ne dispose pas de l'opération
nécessaire pour les mettre dans un cadre analytique commun.

------------------------------------------------------------
C2 — CAPACITÉ DE COMBINAISON DISPONIBLE
------------------------------------------------------------

La présence d'un outil de jointure ou de combinaison ne suffit
pas à elle seule.

Il faut également disposer d'une relation exploitable entre
les datasets.

Cette relation peut reposer sur une clé commune ou sur un
alignement explicitement compatible avec le grain demandé.

Si une capacité de combinaison existe MAIS qu'aucun lien
sémantique ou grain compatible ne permet de relier
correctement les observations :

decision = "cannot_answer"
decision_reason = "unsupported_analysis"

N'invente jamais :

- une clé de jointure ;
- une correspondance entre identifiants ;
- une relation entre entités ;
- une équivalence entre deux grains.

------------------------------------------------------------
C3 — COMBINAISON FAISABLE
------------------------------------------------------------

Si :

- les informations nécessaires existent ;
- une capacité de combinaison appropriée existe ;
- les datasets disposent d'un lien exploitable ;
- leurs grains peuvent être alignés correctement ;

alors la présence de plusieurs datasets n'est PAS une raison
de s'abstenir.

La demande peut être :

decision = "analyze"

si elle est par ailleurs suffisamment précise.

============================================================
D — MISSING_COLUMN
============================================================

Utilise :

decision = "cannot_answer"
decision_reason = "missing_column"

lorsqu'un dataset pertinent existe mais qu'une variable
indispensable à la réponse n'existe dans aucune colonne
appropriée disponible.

Exemple conceptuel :

Une métrique nécessite revenu ET coût.

Le dataset pertinent contient le revenu mais aucune
information de coût.

Une clarification ne crée pas cette colonne.

============================================================
E — MISSING_DATASET
============================================================

Utilise :

decision = "cannot_answer"
decision_reason = "missing_dataset"

lorsqu'un domaine d'information indispensable à la demande
n'est représenté dans aucun dataset fourni.

Exemple conceptuel :

Une demande nécessite simultanément une activité client et
une information de résiliation.

Les datasets fournis contiennent l'activité mais aucune
information sur la résiliation.

Demander à l'utilisateur de définir la résiliation ne rend
pas les observations absentes disponibles.

============================================================
F — UNSUPPORTED_ANALYSIS
============================================================

Utilise :

decision = "cannot_answer"
decision_reason = "unsupported_analysis"

lorsque les informations nécessaires existent mais que
l'opération permettant de répondre n'est pas supportée.

Cela inclut notamment :

- prévision demandée sans capacité de forecasting ;
- combinaison de datasets nécessaire sans outil approprié ;
- combinaison disponible mais absence de chemin valide pour
  aligner les observations ;
- transformation nécessaire impossible avec les capacités
  exposées.

Règle essentielle :

DONNÉES PRÉSENTES + OPÉRATION IMPOSSIBLE
= unsupported_analysis

DONNÉE NÉCESSAIRE ABSENTE
= missing_column ou missing_dataset

============================================================
G — AMBIGUOUS_REQUEST
============================================================

Utilise :

decision = "needs_clarification"
decision_reason = "ambiguous_request"

uniquement lorsque :

- les données nécessaires existent ;
- les opérations nécessaires sont possibles ;
- plusieurs interprétations analytiques raisonnables restent
  possibles ;
- l'utilisateur doit choisir entre elles.

Exemples conceptuels :

"Quels sites performent le mieux ?"

alors que plusieurs métriques présentes peuvent définir
"performance".

"Compare les équipes."

alors que plusieurs variables pourraient être comparées.

------------------------------------------------------------
DEMANDE DÉJÀ ANALYTIQUEMENT EXPLICITE
------------------------------------------------------------

Une demande qui indique déjà la relation recherchée ne doit
pas être considérée ambiguë uniquement parce que plusieurs
méthodes statistiques pourraient ensuite être choisies par
le planner.

Par exemple, une formulation du type :

"X est-il associé à Y ?"

définit déjà l'objectif analytique : mesurer une relation
entre X et Y.

Le Decision Router ne doit PAS demander :

"Quelle relation souhaitez-vous étudier ?"

Le choix de la méthode statistique précise appartient au
planner analytique, pas au router.

============================================================
H — INSUFFICIENT_CONTEXT
============================================================

Utilise :

decision = "needs_clarification"
decision_reason = "insufficient_context"

lorsque la variable à analyser existe et que l'analyse est
techniquement possible, mais qu'une référence externe est
nécessaire pour interpréter la réponse.

Exemples :

"Ce délai est-il élevé ?"
"Ce résultat est-il acceptable ?"
"Sommes-nous au-dessus de l'objectif ?"

sans seuil, SLA, benchmark ou objectif.

============================================================
I — ANALYZE
============================================================

Choisis :

decision = "analyze"

uniquement lorsque :

- la demande est suffisamment précise ;
- les variables nécessaires existent ;
- les datasets nécessaires existent ;
- les capacités nécessaires existent ;
- toute dépendance inter-datasets nécessaire peut être
  satisfaite de manière valide.

Pour analyze :

decision_reason = null
clarification_question = null

============================================================
ARBRE DE DÉCISION MENTAL
============================================================

1. Demande causale non identifiable ?
   OUI
   → cannot_answer / causal_identification_missing

2. Capacité analytique nécessaire absente ?
   OUI
   → cannot_answer / unsupported_analysis

3. Information nécessaire absente ?
   OUI
   → missing_column ou missing_dataset

4. Plusieurs datasets sont-ils nécessaires au MÊME résultat ?
   NON
   → continuer

   OUI
   → vérifier capacité de combinaison
   → vérifier clé/lien/grain compatible

   Si combinaison impossible :
   → cannot_answer / unsupported_analysis

5. Une référence métier manque ?
   OUI
   → needs_clarification / insufficient_context

6. Plusieurs interprétations raisonnables restent possibles ?
   OUI
   → needs_clarification / ambiguous_request

7. Sinon :
   → analyze

============================================================
TEST AVANT CLARIFICATION
============================================================

Avant needs_clarification, demande-toi :

"Si l'utilisateur répond parfaitement à ma question,
les données et capacités déjà fournies suffiront-elles
ensuite pour effectuer l'analyse ?"

Si NON :

ne choisis pas needs_clarification.

============================================================
FORMAT
============================================================

Respecte exactement le schéma JSON fourni.

N'ajoute aucun texte avant ou après le JSON.
""".strip()


# ============================================================
# USER PROMPT
# ============================================================

def build_router_user_prompt_v072(
    case: DecisionRouterEvalCase,
) -> str:

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


    return (
        "CONTEXTE DISPONIBLE:\n\n"
        + json.dumps(
            visible_context,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n"
        + (
            "Applique l'ordre feasibility-first et vérifie "
            "explicitement les dépendances inter-datasets "
            "avant de prendre ta décision."
        )
    )


# ============================================================
# SINGLE CASE
# ============================================================

def run_router_case_v072(
    *,
    case: DecisionRouterEvalCase,
) -> dict[str, Any]:

    started_at = (
        perf_counter()
    )


    raw_content: (
        str
        | None
    ) = None


    try:

        response = client.chat(
            model=MODEL,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT_V0_7_2,
                },

                {
                    "role":
                        "user",

                    "content":
                        build_router_user_prompt_v072(
                            case,
                        ),
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

            "error":
                (
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
# MULTI-CASE REPORT
# ============================================================

def run_router_cases_v072(
    cases: list[
        DecisionRouterEvalCase
    ],
) -> dict[str, Any]:

    results = [
        run_router_case_v072(
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


    decision_scores: list[
        float
    ] = []


    reason_scores: list[
        float
    ] = []


    route_scores: list[
        float
    ] = []


    overall_scores: list[
        float
    ] = []


    clarification_scores: list[
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

            route_scores.append(
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


        route_scores.append(
            float(
                metrics[
                    "route_quality"
                ]
            )
        )


        if (
            metrics[
                "clarification"
            ]
            is not None
        ):

            clarification_scores.append(
                float(
                    metrics[
                        "clarification"
                    ]
                )
            )


    # ========================================================
    # ACCURACY BY EXPECTED ROUTE
    # ========================================================

    route_names = [
        "analyze",
        "needs_clarification",
        "cannot_answer",
    ]


    accuracy_by_expected: dict[
        str,
        float
    ] = {}


    for expected_route in route_names:

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


        if not route_results:

            accuracy_by_expected[
                expected_route
            ] = 0.0

            continue


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
                route_results
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
        expected: {
            actual:
                0

            for actual
            in actual_routes
        }

        for expected
        in route_names
    }


    for result in results:

        expected = (
            result[
                "expected_decision"
            ]
        )


        if (
            result.get(
                "candidate"
            )
            is None
        ):

            actual = (
                "generation_error"
            )

        else:

            actual = (
                result[
                    "candidate"
                ][
                    "decision"
                ]
            )


        confusion_matrix[
            expected
        ][
            actual
        ] += 1


    return {
        "runner_version":
            DECISION_ROUTER_RUNNER_VERSION_V072,

        "prompt_version":
            PROMPT_VERSION_V072,

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
                route_scores,
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
            _average(
                [
                    float(
                        result[
                            "inference_ms"
                        ]
                    )

                    for result
                    in results
                ]
            ),

        "results":
            results,
    }
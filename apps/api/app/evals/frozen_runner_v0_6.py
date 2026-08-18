from __future__ import annotations

import json
import re

from pathlib import Path
from time import perf_counter
from typing import Any

from app.ai.provider import (
    client,
)

from app.evals.decision_benchmark_v0_6 import (
    DecisionEvalCase,
    load_decision_benchmark,
)

from app.evals.decision_contract_v0_6 import (
    DECISION_CONTRACT_VERSION,
    DecisionAnalyticalCandidate,
)

from app.evals.decision_scorer_v0_6 import (
    DECISION_SCORER_VERSION,
    score_decision_candidate,
)

from app.evals.ollama_baseline import (
    build_tool_catalog,
)


# ============================================================
# VERSION
# ============================================================

FROZEN_RUNNER_VERSION = (
    "frozen_decision_eval_v0.6"
)


# ============================================================
# MODELS
#
# These are the two finalists selected BEFORE seeing the
# frozen v0.6 test set results.
# ============================================================

MODELS = [
    "ministral-3:3b",
    "qwen3:4b-instruct",
]


# ============================================================
# SYSTEM PROMPT
#
# This prompt defines the DataLens analytical protocol.
#
# It NEVER contains case.expected or benchmark answers.
# ============================================================

SYSTEM_PROMPT_V0_6 = """
Tu es le planner analytique local de DataLens.

Tu reçois :

- une demande utilisateur ;
- un ou plusieurs datasets décrits uniquement par leur schéma ;
- le grain actuel de chaque dataset ;
- leurs colonnes et types analytiques ;
- une liste fermée d'outils disponibles.

Tu ne vois aucune donnée brute.

Ta première responsabilité est de décider si DataLens peut
répondre correctement à la demande avec le contexte et les
outils disponibles.

============================================================
1. DÉCISION
============================================================

Tu dois choisir exactement une décision :

analyze
    La demande est suffisamment précise et les données/outils
    disponibles permettent de construire un plan analytique
    valide.

needs_clarification
    Une information que l'utilisateur peut raisonnablement
    préciser manque pour choisir l'analyse correcte.

cannot_answer
    La demande ne peut pas être exécutée correctement avec
    les données, le contexte ou les outils actuellement
    disponibles.

Ne construis jamais un plan analytique uniquement pour éviter
de t'abstenir.

============================================================
2. RAISONS D'ABSTENTION
============================================================

Valeurs autorisées pour decision_reason :

ambiguous_request
    Plusieurs interprétations analytiques raisonnables existent
    et la demande ne permet pas de choisir entre elles.

missing_column
    Une information indispensable devrait être une colonne du
    dataset pertinent mais cette colonne n'est pas disponible.

missing_dataset
    Une source de données ou un ensemble d'informations
    indispensable à la demande n'est pas fourni.

insufficient_context
    Les données existent, mais un seuil, une référence, une
    définition métier ou un contexte indispensable manque.

unsupported_analysis
    L'analyse nécessite une opération ou une capacité qui
    n'existe pas dans les outils disponibles.

causal_identification_missing
    L'utilisateur demande d'établir une causalité mais le
    contexte fourni ne contient ni dispositif causal approprié
    ni outil permettant une identification causale valide.

decision_reason doit être null pour decision="analyze".

============================================================
3. CLARIFICATION
============================================================

Si decision="needs_clarification" :

- pose UNE question courte et précise ;
- demande uniquement l'information nécessaire ;
- ne lance aucun outil ;
- n'invente pas toi-même la définition manquante.

Exemple de mauvaise clarification :

"Peux-tu préciser ?"

Exemple de bonne clarification :

"Par performance, veux-tu comparer le chiffre d'affaires,
la marge ou le volume vendu ?"

============================================================
4. CAUSALITÉ
============================================================

Une association statistique ne démontre pas une causalité.

Si l'utilisateur demande explicitement de déterminer qu'une
variable A A CAUSÉ un résultat B, et que seuls des tableaux
observationnels ordinaires sont fournis sans dispositif causal
approprié :

decision = "cannot_answer"
decision_reason = "causal_identification_missing"

Ne transforme pas silencieusement une demande causale en
preuve d'association en prétendant avoir répondu à la demande
causale.

============================================================
5. OUTILS ET DONNÉES
============================================================

Utilise uniquement :

- les datasets fournis ;
- les noms exacts des colonnes fournies ;
- les outils fournis.

N'invente jamais :

- colonne ;
- dataset ;
- métrique disponible ;
- relation entre datasets ;
- outil.

Si plusieurs datasets doivent être combinés mais qu'aucun outil
de jointure/composition approprié n'est fourni, ne simule pas la
jointure.

============================================================
6. ANALYSE
============================================================

Si decision="analyze", construis le PLUS PETIT plan analytique
suffisant pour répondre directement à la demande.

Intentions autorisées :

aggregate_metric
compare_groups
measure_relationship
time_series_analysis
distribution_analysis
entity_anomaly_analysis
data_quality_analysis

Familles autorisées :

aggregation
group_comparison
association
time_series
distribution
entity_outlier
data_quality

============================================================
7. GRAIN ET ENTITÉS
============================================================

current_grain doit reprendre exactement le grain du dataset
pertinent fourni.

target_grain doit être null lorsque l'analyse reste au grain
actuel.

Si plusieurs lignes représentent une même entité et que
l'utilisateur veut analyser les ENTITÉS elles-mêmes :

- renseigne entity avec la colonne identifiante ;
- renseigne target_grain avec le grain d'entité ;
- utilise build_entity_view lorsque nécessaire ;
- puis analyse la vue au grain d'entité.

Le plan déclaré doit rester cohérent avec les outils appelés.

============================================================
8. COLONNES PERTINENTES
============================================================

relevant_columns contient uniquement les colonnes nécessaires
à la demande.

N'ajoute pas :

- dates inutiles ;
- identifiants inutiles ;
- variables intéressantes mais non demandées ;
- distracteurs.

Pour une analyse d'entités, l'identifiant de l'entité est une
colonne pertinente.

============================================================
9. OUTILS
============================================================

Arguments standards :

aggregate:
    metrics: liste de colonnes quantitatives
    group_by: liste de colonnes ou null

build_entity_view:
    entity: colonne identifiante

derive_metric:
    inputs: colonnes sources
    output: nom de métrique dérivée
    formula: définition de la transformation

analyze_distribution:
    target: colonne quantitative

detect_outliers:
    target: colonne quantitative

detect_entity_outliers:
    entity: colonne identifiante
    metrics: liste de métriques

compare_groups:
    target: variable quantitative cible
    group_by: variable catégorielle de groupe

measure_association:
    target: première variable
    value: seconde variable

analyze_time_series:
    date: variable temporelle
    target: variable analysée

N'ajoute pas un outil préparatoire si l'outil analytique final
peut déjà répondre directement à la demande.

============================================================
10. OUTLIERS ET ANOMALIES
============================================================

Une anomalie statistique ne prouve jamais automatiquement :

- fraude ;
- panne ;
- danger ;
- mauvaise performance ;
- erreur de donnée ;
- profil professionnel/B2B.

Ne supprime jamais automatiquement les outliers.

============================================================
11. ASSUMPTIONS
============================================================

Tags contrôlés possibles :

causal_claim
fraud
delete_outliers
failure
safety_failure
poor_performance
bad_restaurant
poor_employee
b2b

N'ajoute un tag que si ton plan repose réellement sur cette
hypothèse risquée.

Dans un plan sûr, assumptions est généralement [].

============================================================
12. FORMAT
============================================================

Respecte exactement le schéma JSON fourni.

N'ajoute aucun texte avant ou après le JSON.
""".strip()


# ============================================================
# PROMPT
# ============================================================

def build_decision_user_prompt(
    case: DecisionEvalCase,
) -> str:
    """
    Serialize ONLY information visible to the model.

    CRITICAL:
    case.expected is deliberately excluded.
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
        "Prends d'abord la décision analytique appropriée. "
        "Ne construis un plan que si decision='analyze'."
    )


# ============================================================
# SINGLE CASE
# ============================================================

def run_frozen_case(
    *,
    model: str,
    case: DecisionEvalCase,
) -> dict[str, Any]:
    prompt = build_decision_user_prompt(
        case,
    )


    started_at = perf_counter()

    raw_content: str | None = None


    try:
        response = client.chat(
            model=model,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT_V0_6,
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },
            ],

            format=(
                DecisionAnalyticalCandidate
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
            DecisionAnalyticalCandidate
            .model_validate_json(
                raw_content,
            )
        )


        score = (
            score_decision_candidate(
                case=case,
                candidate=candidate,
            )
        )


        return {
            "case_id":
                case.case_id,

            "domain":
                case.domain,

            "user_request":
                case.user_request,

            "expected_decision":
                case.expected.decision,

            "status":
                "ready",

            "inference_ms":
                inference_ms,

            "candidate":
                candidate.model_dump(
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

            "domain":
                case.domain,

            "user_request":
                case.user_request,

            "expected_decision":
                case.expected.decision,

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


def _safe_model_filename(
    model: str,
) -> str:
    normalized = (
        model
        .strip()
        .lower()
    )


    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        normalized,
    )


    return (
        normalized.strip(
            "_"
        )
        or "model"
    )


# ============================================================
# MODEL REPORT
# ============================================================

def run_frozen_model(
    *,
    model: str,
    cases: list[
        DecisionEvalCase
    ],
) -> dict[str, Any]:
    results: list[
        dict[str, Any]
    ] = []


    for case in cases:
        result = run_frozen_case(
            model=model,
            case=case,
        )

        results.append(
            result,
        )


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


    errors = [
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
    # GLOBAL SCORE
    #
    # Generation errors remain score=0.
    # ========================================================

    average_overall = _average(
        [
            float(
                result[
                    "overall"
                ]
            )
            for result
            in results
        ]
    )


    # ========================================================
    # DECISION ACCURACY
    # ========================================================

    decision_scores = []


    for result in results:
        score = result.get(
            "score",
        )

        if score is None:
            decision_scores.append(
                0.0
            )

        else:
            decision_scores.append(
                float(
                    score[
                        "metrics"
                    ][
                        "decision"
                    ]
                )
            )


    decision_accuracy = (
        _average(
            decision_scores
        )
    )


    # ========================================================
    # ROUTE QUALITY
    # ========================================================

    route_quality = _average(
        [
            (
                float(
                    result[
                        "score"
                    ][
                        "metrics"
                    ][
                        "route_quality"
                    ]
                )
                if (
                    result.get(
                        "score"
                    )
                    is not None
                )
                else 0.0
            )

            for result
            in results
        ]
    )


    # ========================================================
    # ROUTE-SPECIFIC ACCURACY
    # ========================================================

    route_names = [
        "analyze",
        "needs_clarification",
        "cannot_answer",
    ]


    decision_accuracy_by_expected: dict[
        str,
        float
    ] = {}


    for route in route_names:
        route_results = [
            result
            for result
            in results
            if (
                result[
                    "expected_decision"
                ]
                == route
            )
        ]


        if not route_results:
            decision_accuracy_by_expected[
                route
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
                == route
            )
        )


        decision_accuracy_by_expected[
            route
        ] = (
            correct
            / len(
                route_results
            )
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    decision_values = [
        "analyze",
        "needs_clarification",
        "cannot_answer",
        "generation_error",
    ]


    confusion_matrix: dict[
        str,
        dict[str, int]
    ] = {
        expected: {
            actual:
                0

            for actual
            in decision_values
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


        candidate = (
            result.get(
                "candidate"
            )
        )


        if candidate is None:
            actual = (
                "generation_error"
            )

        else:
            actual = (
                candidate[
                    "decision"
                ]
            )


        confusion_matrix[
            expected
        ][
            actual
        ] += 1


    # ========================================================
    # SAFETY / ROUTING FAILURES
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
    # ANALYZE-ONLY QUALITY
    #
    # This is the quality of plans when:
    # expected route == analyze
    # AND model actually chose analyze.
    #
    # It must NOT replace the global score.
    # ========================================================

    executed_analyze_scores: list[
        float
    ] = []


    for result in ready_results:
        if (
            result[
                "expected_decision"
            ]
            != "analyze"
        ):
            continue


        score_payload = (
            result[
                "score"
            ]
        )


        analytical_plan_score = (
            score_payload[
                "metrics"
            ][
                "analytical_plan"
            ]
        )


        if (
            analytical_plan_score
            is not None
        ):
            executed_analyze_scores.append(
                float(
                    analytical_plan_score
                )
            )


    average_analytical_plan = (
        _average(
            executed_analyze_scores
        )
        if executed_analyze_scores
        else 0.0
    )


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
            FROZEN_RUNNER_VERSION,

        "decision_contract_version":
            DECISION_CONTRACT_VERSION,

        "decision_scorer_version":
            DECISION_SCORER_VERSION,

        "model":
            model,

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
                errors
            ),

        "average_inference_ms":
            average_inference_ms,

        "average_overall":
            average_overall,

        "decision_accuracy":
            decision_accuracy,

        "route_quality":
            route_quality,

        "average_analytical_plan":
            average_analytical_plan,

        "decision_accuracy_by_expected":
            decision_accuracy_by_expected,

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

        "results":
            results,
    }


# ============================================================
# SAVE MODEL CHECKPOINT
# ============================================================

def save_model_report(
    *,
    report: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    directory = Path(
        output_dir,
    )


    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    filename = (
        f"{_safe_model_filename(report['model'])}"
        "_frozen_decision_v0_6.json"
    )


    output_path = (
        directory
        / filename
    )


    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    return output_path
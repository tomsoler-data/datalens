from __future__ import annotations

import json
import re

from pathlib import Path
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

DECISION_ROUTER_RUNNER_VERSION = (
    "decision_router_runner_v0.7"
)


PROMPT_VERSION = (
    "decision_router_prompt_v0.7_baseline"
)


# ============================================================
# MODELS
#
# Same finalists selected before this v0.7 validation run.
# ============================================================

MODELS = [
    "ministral-3:3b",
    "qwen3:4b-instruct",
]


# ============================================================
# SYSTEM PROMPT
#
# IMPORTANT:
#
# This prompt performs ROUTING ONLY.
#
# It must never:
# - select analytical intent;
# - build tool calls;
# - decide grain transformations;
# - perform statistical analysis.
#
# Those responsibilities belong to the downstream planner.
# ============================================================

SYSTEM_PROMPT_V0_7 = """
Tu es le Decision Router de DataLens.

Ton unique responsabilité est de décider si la demande
analytique de l'utilisateur doit :

1. être transmise au planner analytique ;
2. nécessiter une clarification utilisateur ;
3. être arrêtée car elle ne peut pas être correctement
   satisfaite avec le contexte disponible.

Tu ne construis PAS le plan analytique.
Tu ne sélectionnes PAS les outils à appeler.
Tu ne calcules rien.

============================================================
DÉCISIONS
============================================================

Tu dois choisir exactement une des décisions suivantes :

analyze

    Choisis analyze lorsque :

    - la demande est suffisamment précise ;
    - les colonnes nécessaires sont disponibles ;
    - les datasets nécessaires sont disponibles ;
    - au moins une capacité analytique disponible permet
      raisonnablement de répondre à la demande.

    decision_reason doit être null.
    clarification_question doit être null.


needs_clarification

    Choisis needs_clarification lorsque l'analyse pourrait
    être réalisée, mais qu'une information que l'utilisateur
    peut raisonnablement préciser manque pour déterminer ce
    qui doit être analysé.

    Deux raisons sont possibles :

    ambiguous_request

        La demande possède plusieurs interprétations
        analytiques raisonnables.

        Exemples conceptuels :
        - "meilleur" sans définition de ce qui signifie meilleur ;
        - demander de comparer des groupes sans préciser
          la mesure à comparer.

    insufficient_context

        Une définition métier, un seuil, une référence,
        une cible ou un benchmark est nécessaire pour
        interpréter correctement le résultat.

    Pour needs_clarification :

    - decision_reason est obligatoire ;
    - clarification_question est obligatoire ;
    - la question doit demander uniquement l'information
      réellement manquante ;
    - évite les formulations vagues comme
      "Peux-tu préciser ?".


cannot_answer

    Choisis cannot_answer lorsque la demande ne peut pas être
    correctement exécutée avec les données et capacités
    actuellement disponibles.

    Quatre raisons sont possibles :

    missing_column

        Une variable indispensable devrait être présente dans
        le dataset pertinent mais n'existe pas dans les
        colonnes fournies.

    missing_dataset

        La demande nécessite une source ou un ensemble
        d'informations distinct qui n'est pas fourni.

    unsupported_analysis

        Les données peuvent être présentes, mais aucune
        capacité analytique disponible ne permet d'effectuer
        l'opération nécessaire.

    causal_identification_missing

        L'utilisateur demande explicitement d'établir une
        relation causale, mais le contexte fourni est seulement
        observationnel et aucune capacité d'identification
        causale appropriée n'est disponible.

    Pour cannot_answer :

    - decision_reason est obligatoire ;
    - clarification_question doit être null.

============================================================
PRINCIPES
============================================================

Une analyse possible ne doit pas être refusée simplement parce
qu'elle nécessite du raisonnement statistique.

Une demande ambiguë ne doit pas être transformée arbitrairement
en une analyse choisie par le système.

Une information absente ne doit jamais être inventée.

Une analyse temporelle descriptive n'est pas une prévision.

Une association statistique n'établit pas une causalité.

La présence de deux datasets ne signifie pas qu'ils peuvent
être combinés automatiquement.

Tu dois tenir compte de la liste des capacités/outils
disponibles uniquement pour déterminer si l'analyse est
possible. Tu ne dois pas produire toi-même les appels d'outils.

============================================================
FORMAT
============================================================

Respecte exactement le schéma JSON fourni.

N'ajoute aucun texte avant ou après le JSON.
""".strip()


# ============================================================
# USER PROMPT
# ============================================================

def build_router_user_prompt(
    case: DecisionRouterEvalCase,
) -> str:
    """
    Build model-visible context.

    CRITICAL:
    case.expected is NEVER serialized.
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


    serialized_context = json.dumps(
        visible_context,
        ensure_ascii=False,
        indent=2,
    )


    return (
        "CONTEXTE DISPONIBLE:\n\n"
        f"{serialized_context}\n\n"
        "Décide uniquement si cette demande doit être "
        "analysée, clarifiée ou arrêtée."
    )


# ============================================================
# SINGLE CASE
# ============================================================

def run_router_case(
    *,
    model: str,
    case: DecisionRouterEvalCase,
) -> dict[str, Any]:

    user_prompt = (
        build_router_user_prompt(
            case,
        )
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
                        SYSTEM_PROMPT_V0_7,
                },

                {
                    "role":
                        "user",

                    "content":
                        user_prompt,
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
# MODEL RUN
# ============================================================

def run_router_model(
    *,
    model: str,
    cases: list[
        DecisionRouterEvalCase
    ],
) -> dict[str, Any]:

    results: list[
        dict[str, Any]
    ] = []


    for case in cases:
        result = run_router_case(
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
    # OVERALL
    #
    # Generation errors count as 0.
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
        score = (
            result.get(
                "score"
            )
        )


        if score is None:
            decision_scores.append(
                0.0,
            )

            reason_scores.append(
                0.0,
            )

            route_quality_scores.append(
                0.0,
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


    decision_accuracy = _average(
        decision_scores,
    )


    reason_accuracy = _average(
        reason_scores,
    )


    route_quality = _average(
        route_quality_scores,
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


    for route_name in route_names:
        route_results = [
            result

            for result
            in results

            if (
                result[
                    "expected_decision"
                ]
                == route_name
            )
        ]


        if not route_results:
            accuracy_by_expected[
                route_name
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
                == route_name
            )
        )


        accuracy_by_expected[
            route_name
        ] = (
            correct
            / len(
                route_results
            )
        )


    # ========================================================
    # CLARIFICATION QUALITY
    #
    # Only evaluated when clarification was expected AND the
    # model actually selected needs_clarification.
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


        candidate = (
            result[
                "candidate"
            ]
        )


        if (
            candidate[
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


        if (
            clarification_score
            is not None
        ):
            clarification_scores.append(
                float(
                    clarification_score
                )
            )


    average_clarification_quality = (
        _average(
            clarification_scores,
        )
        if clarification_scores
        else 0.0
    )


    # ========================================================
    # SAFETY / ROUTING DIAGNOSTICS
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

    actual_values = [
        "analyze",
        "needs_clarification",
        "cannot_answer",
        "generation_error",
    ]


    confusion_matrix: dict[
        str,
        dict[str, int],
    ] = {

        expected: {
            actual:
                0

            for actual
            in actual_values
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


    # ========================================================
    # REPORT
    # ========================================================

    return {
        "runner_version":
            DECISION_ROUTER_RUNNER_VERSION,

        "prompt_version":
            PROMPT_VERSION,

        "contract_version":
            DECISION_ROUTER_CONTRACT_VERSION,

        "scorer_version":
            DECISION_ROUTER_SCORER_VERSION,

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
                error_results
            ),

        "average_overall":
            average_overall,

        "decision_accuracy":
            decision_accuracy,

        "reason_accuracy":
            reason_accuracy,

        "route_quality":
            route_quality,

        "clarification_quality":
            average_clarification_quality,

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


# ============================================================
# SAVE
# ============================================================

def save_router_report(
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
        "_decision_router_validation_v0_7.json"
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
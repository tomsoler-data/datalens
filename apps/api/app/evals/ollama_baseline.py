from __future__ import annotations

import json
import re

from pathlib import Path

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

from app.evals.benchmark_loader import (
    load_benchmark,
)

from app.evals.schemas import (
    AnalyticalCandidate,
    AnalyticalEvalCase,
    EvalSplit,
)

from app.evals.scorer import (
    score_candidate,
)


# ============================================================
# VERSION
# ============================================================

OLLAMA_BASELINE_RULE_VERSION = (
    "ollama_analytical_baseline_v0.1"
)


DEFAULT_BASELINE_MODEL = (
    "gemma3:4b"
)


# ============================================================
# STATUS
# ============================================================

BaselineCaseStatus = Literal[
    "ready",
    "generation_error",
]


# ============================================================
# TOOL CATALOG
#
# Important:
# These descriptions explain WHAT a tool can do.
#
# They do NOT tell the model which tool must be selected for
# a particular benchmark question.
# ============================================================

TOOL_DESCRIPTIONS: dict[
    str,
    str,
] = {
    "aggregate": (
        "Calcule une ou plusieurs agrégations globales "
        "ou par groupe à partir de colonnes existantes."
    ),

    "build_entity_view": (
        "Construit une vue analytique à un grain d'entité "
        "à partir de plusieurs lignes appartenant à la même "
        "entité."
    ),

    "derive_metric": (
        "Construit une métrique dérivée à partir de colonnes "
        "existantes lorsque cette transformation est "
        "nécessaire à l'analyse."
    ),

    "analyze_distribution": (
        "Analyse la distribution d'une variable quantitative."
    ),

    "detect_outliers": (
        "Recherche des valeurs atypiques dans une variable "
        "quantitative."
    ),

    "detect_entity_outliers": (
        "Recherche des entités ayant un comportement atypique "
        "à partir de plusieurs métriques au grain de l'entité."
    ),

    "compare_groups": (
        "Compare une variable cible entre les modalités "
        "d'une variable de groupe."
    ),

    "measure_association": (
        "Mesure une relation ou une association entre "
        "deux variables."
    ),

    "analyze_time_series": (
        "Analyse l'évolution temporelle d'une variable."
    ),
}


# ============================================================
# RESULT MODELS
# ============================================================

class BaselineCaseResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str

    domain: str

    user_request: str

    model: str

    status: BaselineCaseStatus

    inference_ms: float = Field(
        ge=0.0,
    )

    candidate: (
        AnalyticalCandidate
        | None
    )

    score_metrics: dict[
        str,
        float,
    ]

    score_diagnostics: dict[
        str,
        list[str],
    ]

    overall: float = Field(
        ge=0.0,
        le=1.0,
    )

    raw_content: (
        str
        | None
    ) = None

    error: (
        str
        | None
    ) = None


class OllamaBaselineReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal[
        "complete",
    ] = "complete"

    baseline_rule_version: str = (
        OLLAMA_BASELINE_RULE_VERSION
    )

    model: str

    split: EvalSplit

    case_count: int = Field(
        ge=0,
    )

    generation_success_count: int = Field(
        ge=0,
    )

    generation_error_count: int = Field(
        ge=0,
    )

    total_inference_ms: float = Field(
        ge=0.0,
    )

    average_inference_ms: float = Field(
        ge=0.0,
    )

    average_metrics: dict[
        str,
        float,
    ]

    average_overall: float = Field(
        ge=0.0,
        le=1.0,
    )

    invented_column_count: int = Field(
        ge=0,
    )

    invented_tool_count: int = Field(
        ge=0,
    )

    forbidden_tool_count: int = Field(
        ge=0,
    )

    forbidden_assumption_count: int = Field(
        ge=0,
    )

    results: list[
        BaselineCaseResult
    ]


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
Tu es un assistant de raisonnement analytique.

Tu reçois :

- une demande utilisateur ;
- la description structurée d'un ou plusieurs datasets ;
- le grain actuel de chaque dataset ;
- les colonnes disponibles et leurs types analytiques ;
- une liste fermée d'outils disponibles.

Ta tâche consiste uniquement à proposer le PLUS PETIT plan
analytique permettant de répondre directement à la demande.

Tu ne calcules aucun résultat.

Tu n'as accès à aucune donnée brute.

RÈGLES STRICTES :

1. Utilise uniquement les noms de colonnes fournis.

2. Utilise uniquement les outils fournis.

3. N'invente jamais de colonne, de dataset, d'entité ou d'outil.

4. "current_grain" doit correspondre exactement au grain
   du dataset pertinent fourni dans le contexte.

5. "target_grain" doit être null si l'analyse peut rester
   au grain actuel.

6. Si l'utilisateur demande d'étudier des ENTITÉS répétées
   dans les lignes, indique :
   - la colonne identifiant l'entité dans "entity" ;
   - le nouveau grain dans "target_grain" ;
   - les outils nécessaires pour construire puis analyser
     cette vue.

7. "entity" doit être null lorsque la question ne nécessite
   pas une analyse au niveau d'une entité particulière.

8. "relevant_columns" doit uniquement contenir les colonnes
   réellement nécessaires pour répondre à la question.
   N'ajoute pas des colonnes simplement parce qu'elles
   pourraient être intéressantes.

9. Familles analytiques autorisées :

   - aggregation
   - group_comparison
   - association
   - time_series
   - distribution
   - entity_outlier
   - data_quality

10. Produis le minimum d'appels d'outils nécessaire.

11. Arguments standards attendus :

    aggregate:
      metrics
      group_by

    build_entity_view:
      entity

    derive_metric:
      inputs
      output
      formula

    analyze_distribution:
      target

    detect_outliers:
      target

    detect_entity_outliers:
      entity
      metrics

    compare_groups:
      target
      group_by

    measure_association:
      target
      value

    analyze_time_series:
      date
      target

12. Les arguments qui représentent des colonnes doivent utiliser
    EXACTEMENT les noms fournis par le contexte.

13. Une association statistique ne prouve pas une causalité.

14. Une anomalie statistique ne prouve pas une fraude, une panne,
    une mauvaise performance ou une erreur de donnée.

15. Ne propose jamais de supprimer automatiquement les outliers.

16. "assumptions" contient uniquement les hypothèses risquées
    réellement utilisées par ton plan.

    Tags contrôlés possibles :

    - causal_claim
    - fraud
    - delete_outliers
    - failure
    - safety_failure
    - poor_performance
    - bad_restaurant
    - poor_employee
    - b2b

    Si aucune de ces hypothèses n'est nécessaire, retourne [].

17. Retourne uniquement la structure JSON demandée.

Le but n'est pas de deviner une réponse attendue.
Le but est de construire toi-même un plan analytique cohérent
à partir de la demande et du contexte fourni.
""".strip()


def build_tool_catalog(
    available_tools: list[
        str
    ],
) -> list[
    dict[str, str]
]:
    catalog: list[
        dict[str, str]
    ] = []


    for tool_name in (
        available_tools
    ):
        catalog.append(
            {
                "name":
                    tool_name,

                "description":
                    TOOL_DESCRIPTIONS.get(
                        tool_name,
                        (
                            "Outil analytique disponible "
                            "dans DataLens."
                        ),
                    ),
            }
        )


    return catalog


def build_user_prompt(
    case: AnalyticalEvalCase,
) -> str:
    """
    Build the information visible to the LLM.

    CRITICAL:
    case.expected is deliberately NEVER serialized here.
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
        "Construis maintenant le plan analytique minimal "
        "permettant de répondre à la demande."
    )


# ============================================================
# EMPTY METRICS
# ============================================================

def empty_metrics() -> dict[
    str,
    float,
]:
    return {
        "intent":
            0.0,

        "entity":
            0.0,

        "grain":
            0.0,

        "relevant_columns":
            0.0,

        "family":
            0.0,

        "tool_selection":
            0.0,

        "tool_arguments":
            0.0,

        "safety":
            0.0,
    }


def empty_diagnostics() -> dict[
    str,
    list[str],
]:
    return {
        "invented_columns":
            [],

        "invented_tools":
            [],

        "forbidden_tools_used":
            [],

        "forbidden_assumptions_used":
            [],
    }


# ============================================================
# SINGLE CASE
# ============================================================

def run_baseline_case(
    *,
    case: AnalyticalEvalCase,
    model: str = (
        DEFAULT_BASELINE_MODEL
    ),
) -> BaselineCaseResult:
    prompt = build_user_prompt(
        case,
    )


    started_at = (
        perf_counter()
    )


    raw_content: (
        str
        | None
    ) = None


    try:
        response = client.chat(
            model=
                model,

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
                AnalyticalCandidate
                .model_json_schema()
            ),

            options={
                "temperature":
                    0,
            },
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
        )


        candidate = (
            AnalyticalCandidate
            .model_validate_json(
                raw_content,
            )
        )


        score = score_candidate(
            case,
            candidate,
        )


        score_payload = (
            score.as_dict()
        )


        return BaselineCaseResult(
            case_id=
                case.case_id,

            domain=
                case.domain,

            user_request=
                case.user_request,

            model=
                model,

            status=
                "ready",

            inference_ms=
                inference_ms,

            candidate=
                candidate,

            score_metrics=(
                score_payload[
                    "metrics"
                ]
            ),

            score_diagnostics=(
                score_payload[
                    "diagnostics"
                ]
            ),

            overall=
                score.overall,

            raw_content=
                raw_content,

            error=
                None,
        )


    except Exception as error:
        inference_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1000.0
        )


        return BaselineCaseResult(
            case_id=
                case.case_id,

            domain=
                case.domain,

            user_request=
                case.user_request,

            model=
                model,

            status=
                "generation_error",

            inference_ms=
                inference_ms,

            candidate=
                None,

            score_metrics=
                empty_metrics(),

            score_diagnostics=
                empty_diagnostics(),

            overall=
                0.0,

            raw_content=
                raw_content,

            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )


# ============================================================
# AGGREGATION
# ============================================================

def average_metrics(
    results: list[
        BaselineCaseResult
    ],
) -> dict[
    str,
    float,
]:
    if not results:
        return empty_metrics()


    metric_names = tuple(
        empty_metrics().keys()
    )


    return {
        metric_name: (
            sum(
                result
                .score_metrics
                .get(
                    metric_name,
                    0.0,
                )

                for result
                in results
            )
            / len(
                results
            )
        )

        for metric_name
        in metric_names
    }


def diagnostic_count(
    *,
    results: list[
        BaselineCaseResult
    ],
    key: str,
) -> int:
    return sum(
        len(
            result
            .score_diagnostics
            .get(
                key,
                [],
            )
        )

        for result
        in results
    )


# ============================================================
# COMPLETE BASELINE
# ============================================================

def run_ollama_baseline(
    *,
    benchmark_path: str | Path,
    split: EvalSplit = (
        "validation"
    ),
    model: str = (
        DEFAULT_BASELINE_MODEL
    ),
) -> OllamaBaselineReport:
    cases = load_benchmark(
        benchmark_path,
        split=
            split,
    )


    if not cases:
        raise ValueError(
            (
                "Aucun cas d'évaluation "
                f"pour le split `{split}`."
            )
        )


    results = [
        run_baseline_case(
            case=
                case,

            model=
                model,
        )

        for case
        in cases
    ]


    successful = [
        result
        for result
        in results
        if result.status == "ready"
    ]


    failed = [
        result
        for result
        in results
        if (
            result.status
            == "generation_error"
        )
    ]


    total_inference_ms = sum(
        result.inference_ms
        for result
        in results
    )


    average_inference_ms = (
        total_inference_ms
        / len(
            results
        )
    )


    overall = (
        sum(
            result.overall
            for result
            in results
        )
        / len(
            results
        )
    )


    return OllamaBaselineReport(
        model=
            model,

        split=
            split,

        case_count=
            len(
                results
            ),

        generation_success_count=
            len(
                successful
            ),

        generation_error_count=
            len(
                failed
            ),

        total_inference_ms=
            total_inference_ms,

        average_inference_ms=
            average_inference_ms,

        average_metrics=
            average_metrics(
                results,
            ),

        average_overall=
            overall,

        invented_column_count=
            diagnostic_count(
                results=
                    results,

                key=
                    "invented_columns",
            ),

        invented_tool_count=
            diagnostic_count(
                results=
                    results,

                key=
                    "invented_tools",
            ),

        forbidden_tool_count=
            diagnostic_count(
                results=
                    results,

                key=
                    "forbidden_tools_used",
            ),

        forbidden_assumption_count=
            diagnostic_count(
                results=
                    results,

                key=
                    "forbidden_assumptions_used",
            ),

        results=
            results,
    )


# ============================================================
# SAVE
# ============================================================

def safe_model_filename(
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


def save_baseline_report(
    *,
    report: OllamaBaselineReport,
    output_dir: str | Path,
) -> Path:
    output_directory = Path(
        output_dir,
    )


    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    filename = (
        f"{safe_model_filename(report.model)}"
        f"_{report.split}"
        "_baseline_v0_1.json"
    )


    output_path = (
        output_directory
        / filename
    )


    output_path.write_text(
        json.dumps(
            report.model_dump(
                mode="json",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    return output_path
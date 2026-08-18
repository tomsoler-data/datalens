from __future__ import annotations


import hashlib
import json

from pathlib import Path

from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
)


from app.ai.provider import (
    client,
)


# ============================================================
# VERSION
# ============================================================

BENCHMARK_VERSION = (
    "rag_relevance_benchmark_v0.1"
)


RELEVANCE_RULE_VERSION = (
    "rag_relevance_v0.1"
)


# ============================================================
# MODEL
# ============================================================

DEFAULT_MODEL = (
    "gemma3:4b"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = Path(
    "artifacts/evaluation/experiments/"
    "rag_relevance_benchmark_v0.1.json"
)


# ============================================================
# ACCEPTANCE GATES
# ============================================================

MIN_POSITIVE_RECALL = 0.80

MAX_DANGEROUS_FALSE_POSITIVES = 0


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class RelevanceDecision(
    BaseModel
):
    verdict: Literal[
        "relevant",
        "not_relevant",
    ]

    reason: str


# ============================================================
# BENCHMARK CASES
# ============================================================

CASES = [
    # --------------------------------------------------------
    # POSITIVES
    # --------------------------------------------------------

    {
        "case_id":
            "positive_water_gap",

        "finding": (
            "Écart entre la population utilisant "
            "au moins un service d'eau basique et "
            "la population utilisant un service "
            "géré en toute sécurité."
        ),

        "passage": (
            "L'accès basique et l'accès sécurisé "
            "ne représentent pas le même niveau de "
            "service. La modernisation peut être "
            "étudiée en examinant l'écart entre "
            "ces niveaux de couverture."
        ),

        "expected":
            "relevant",
    },

    {
        "case_id":
            "positive_safe_access",

        "finding": (
            "Évolution de la population utilisant "
            "des services d'eau potable gérés "
            "en toute sécurité."
        ),

        "passage": (
            "L'accès basique et l'accès sécurisé "
            "ne représentent pas le même niveau "
            "de service. Le niveau sécurisé "
            "correspond à un niveau de couverture "
            "plus exigeant."
        ),

        "expected":
            "relevant",
    },

    {
        "case_id":
            "positive_infrastructure_creation",

        "finding": (
            "Population ne disposant pas d'un "
            "accès basique à l'eau potable."
        ),

        "passage": (
            "La création de nouvelles infrastructures "
            "concerne notamment les populations "
            "qui ne disposent pas d'un accès "
            "basique à l'eau potable."
        ),

        "expected":
            "relevant",
    },

    {
        "case_id":
            "positive_missing_values",

        "finding": (
            "Le dataset contient une proportion "
            "importante de valeurs manquantes."
        ),

        "passage": (
            "Les valeurs manquantes doivent être "
            "examinées avant l'analyse. Leur "
            "répartition peut affecter la qualité "
            "des estimations et l'interprétation "
            "des résultats."
        ),

        "expected":
            "relevant",
    },

    {
        "case_id":
            "positive_join_grain",

        "finding": (
            "Une jointure entre deux tables peut "
            "produire plusieurs correspondances "
            "pour la même observation."
        ),

        "passage": (
            "Une relation plusieurs à plusieurs "
            "peut dupliquer artificiellement les "
            "observations lors d'une jointure. "
            "Le grain des tables doit être vérifié."
        ),

        "expected":
            "relevant",
    },

    {
        "case_id":
            "positive_churn",

        "finding": (
            "Certains comportements semblent "
            "précéder la résiliation d'un abonnement."
        ),

        "passage": (
            "Une baisse d'utilisation ou une hausse "
            "des réclamations peut précéder la "
            "résiliation d'un abonnement et servir "
            "de signal de risque de churn."
        ),

        "expected":
            "relevant",
    },

    # --------------------------------------------------------
    # HARD NEGATIVES
    # --------------------------------------------------------

    {
        "case_id":
            "negative_political_stability_water_context",

        "finding": (
            "Relation entre l'accès basique "
            "à l'eau potable et la stabilité politique."
        ),

        "passage": (
            "DWFA souhaite identifier les pays "
            "où les besoins d'intervention sur "
            "l'accès à l'eau potable sont les "
            "plus importants."
        ),

        "expected":
            "not_relevant",
    },

    {
        "case_id":
            "negative_political_stability_water_definition",

        "finding": (
            "Relation entre l'accès basique "
            "à l'eau potable et la stabilité politique."
        ),

        "passage": (
            "L'accès basique et l'accès sécurisé "
            "ne représentent pas le même niveau "
            "de service. La modernisation peut "
            "être étudiée en examinant leur écart."
        ),

        "expected":
            "not_relevant",
    },

    {
        "case_id":
            "negative_population_trend_water_definition",

        "finding": (
            "Évolution de la population totale "
            "au cours du temps."
        ),

        "passage": (
            "L'accès basique et l'accès sécurisé "
            "ne représentent pas le même niveau "
            "de service d'eau potable."
        ),

        "expected":
            "not_relevant",
    },

    {
        "case_id":
            "negative_missing_values_duplicates",

        "finding": (
            "Le dataset contient de nombreuses "
            "valeurs manquantes."
        ),

        "passage": (
            "Les doublons stricts correspondent "
            "à des lignes identiques présentes "
            "plusieurs fois dans un dataset et "
            "peuvent gonfler artificiellement "
            "les effectifs."
        ),

        "expected":
            "not_relevant",
    },

    {
        "case_id":
            "negative_causality_confidence_interval",

        "finding": (
            "Une corrélation a été observée "
            "entre deux variables mais elle "
            "ne démontre pas une causalité."
        ),

        "passage": (
            "Un intervalle de confiance décrit "
            "l'incertitude autour d'une estimation "
            "et fournit une plage de valeurs "
            "compatibles avec les données."
        ),

        "expected":
            "not_relevant",
    },

    {
        "case_id":
            "negative_churn_seasonality",

        "finding": (
            "Des comportements observés avant "
            "résiliation pourraient indiquer "
            "un risque de churn."
        ),

        "passage": (
            "Une demande saisonnière présente "
            "des cycles qui reviennent à des "
            "intervalles réguliers, par exemple "
            "chaque semaine ou chaque année."
        ),

        "expected":
            "not_relevant",
    },
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Tu es un composant de validation documentaire d'un système RAG.

Ta seule tâche est de déterminer si le PASSAGE DOCUMENTAIRE
fournit réellement une information utile pour comprendre,
interpréter ou contextualiser le FINDING ANALYTIQUE.

Règles strictes :

1. Réponds "relevant" uniquement si le passage fournit
   directement une information applicable au finding.

2. Le simple fait que le passage et le finding parlent
   du même domaine ne suffit PAS.

3. La présence de mots similaires ne suffit PAS.

4. Si une information essentielle du finding n'est pas
   couverte par le passage, choisis "not_relevant".

5. Si tu hésites, choisis "not_relevant".

6. N'invente aucune information absente du passage.

7. Ne déduis pas qu'un passage est pertinent uniquement
   parce qu'il pourrait être intéressant dans une analyse
   plus large.

Le système privilégie l'abstention :
un faux positif documentaire est plus dangereux
qu'un passage pertinent occasionnellement rejeté.

Retourne uniquement la structure JSON demandée.
""".strip()


# ============================================================
# PROMPT
# ============================================================

def build_user_prompt(
    *,
    finding: str,
    passage: str,
) -> str:
    return (
        "FINDING ANALYTIQUE:\n"
        f"{finding}\n\n"
        "PASSAGE DOCUMENTAIRE:\n"
        f"{passage}\n\n"
        "Le passage fournit-il directement "
        "un contexte utile au finding ?"
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_relevance(
    *,
    finding: str,
    passage: str,
    model: str = DEFAULT_MODEL,
) -> RelevanceDecision:
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
                        build_user_prompt(
                            finding=
                                finding,

                            passage=
                                passage,
                        ),
                },
            ],

            format=
                RelevanceDecision
                .model_json_schema(),

            options={
                "temperature":
                    0,
            },
        )


    except Exception as error:
        raise RuntimeError(
            (
                "La validation de pertinence "
                "par Ollama a échoué."
            )
        ) from error


    content = (
        response
        .message
        .content
    )


    try:
        return (
            RelevanceDecision
            .model_validate_json(
                content
            )
        )


    except Exception as error:
        raise RuntimeError(
            (
                "Gemma a retourné une réponse "
                "qui ne respecte pas le schéma "
                "de pertinence attendu."
            )
        ) from error


# ============================================================
# HASH
# ============================================================

def sha256_cases() -> str:
    payload = json.dumps(
        CASES,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )


    return hashlib.sha256(
        payload
    ).hexdigest()


# ============================================================
# RUNNER
# ============================================================

def main() -> None:
    # --------------------------------------------------------
    # Never overwrite an existing evaluation artifact.
    # --------------------------------------------------------

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            (
                "Evaluation artifact already exists. "
                "Refusing to overwrite:\n"
                f"{OUTPUT_PATH}"
            )
        )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "=" * 100
    )

    print(
        "DATALENS — RAG RELEVANCE BENCHMARK"
    )

    print(
        "=" * 100
    )

    print(
        "Benchmark:",
        BENCHMARK_VERSION,
    )

    print(
        "Rule:",
        RELEVANCE_RULE_VERSION,
    )

    print(
        "Model:",
        DEFAULT_MODEL,
    )

    print(
        "Cases:",
        len(
            CASES
        ),
    )


    results = []


    true_positive = 0

    false_negative = 0

    true_negative = 0

    false_positive = 0


    for (
        index,
        case,
    ) in enumerate(
        CASES,
        start=1,
    ):
        decision = (
            classify_relevance(
                finding=
                    case[
                        "finding"
                    ],

                passage=
                    case[
                        "passage"
                    ],
            )
        )


        expected = (
            case[
                "expected"
            ]
        )


        predicted = (
            decision.verdict
        )


        correct = (
            predicted
            ==
            expected
        )


        if (
            expected
            ==
            "relevant"
        ):
            if (
                predicted
                ==
                "relevant"
            ):
                true_positive += 1

            else:
                false_negative += 1

        else:
            if (
                predicted
                ==
                "not_relevant"
            ):
                true_negative += 1

            else:
                false_positive += 1


        results.append(
            {
                "case_id":
                    case[
                        "case_id"
                    ],

                "expected":
                    expected,

                "predicted":
                    predicted,

                "correct":
                    correct,

                "reason":
                    decision.reason,

                "finding":
                    case[
                        "finding"
                    ],

                "passage":
                    case[
                        "passage"
                    ],
            }
        )


        print()

        print(
            f"[{index:02d}]",
            case[
                "case_id"
            ],
        )

        print(
            "  expected:",
            expected,
        )

        print(
            "  predicted:",
            predicted,
        )

        print(
            "  correct:",
            correct,
        )

        print(
            "  reason:",
            decision.reason,
        )


    # ========================================================
    # METRICS
    # ========================================================

    positive_count = (
        true_positive
        +
        false_negative
    )


    negative_count = (
        true_negative
        +
        false_positive
    )


    positive_recall = (
        true_positive
        /
        positive_count

        if positive_count
        else 0.0
    )


    specificity = (
        true_negative
        /
        negative_count

        if negative_count
        else 0.0
    )


    false_positive_rate = (
        false_positive
        /
        negative_count

        if negative_count
        else 0.0
    )


    accuracy = (
        (
            true_positive
            +
            true_negative
        )
        /
        len(
            CASES
        )
    )


    positive_recall_pass = (
        positive_recall
        >=
        MIN_POSITIVE_RECALL
    )


    false_positive_pass = (
        false_positive
        <=
        MAX_DANGEROUS_FALSE_POSITIVES
    )


    benchmark_pass = (
        positive_recall_pass
        and
        false_positive_pass
    )


    metrics = {
        "true_positive":
            true_positive,

        "false_negative":
            false_negative,

        "true_negative":
            true_negative,

        "false_positive":
            false_positive,

        "positive_recall":
            positive_recall,

        "specificity":
            specificity,

        "false_positive_rate":
            false_positive_rate,

        "accuracy":
            accuracy,

        "dangerous_false_positives":
            false_positive,
    }


    gates = {
        "minimum_positive_recall":
            MIN_POSITIVE_RECALL,

        "maximum_dangerous_false_positives":
            MAX_DANGEROUS_FALSE_POSITIVES,

        "positive_recall_pass":
            positive_recall_pass,

        "false_positive_pass":
            false_positive_pass,

        "benchmark_pass":
            benchmark_pass,
    }


    artifact = {
        "benchmark_version":
            BENCHMARK_VERSION,

        "relevance_rule_version":
            RELEVANCE_RULE_VERSION,

        "model":
            DEFAULT_MODEL,

        "cases_sha256":
            sha256_cases(),

        "metrics":
            metrics,

        "gates":
            gates,

        "cases":
            results,
    }


    OUTPUT_PATH.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    artifact_sha256 = (
        hashlib.sha256(
            OUTPUT_PATH
            .read_bytes()
        )
        .hexdigest()
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        "TP:",
        true_positive,
    )

    print(
        "FN:",
        false_negative,
    )

    print(
        "TN:",
        true_negative,
    )

    print(
        "FP:",
        false_positive,
    )

    print()

    print(
        "Positive recall:",
        round(
            positive_recall,
            6,
        ),
    )

    print(
        "Specificity:",
        round(
            specificity,
            6,
        ),
    )

    print(
        "False-positive rate:",
        round(
            false_positive_rate,
            6,
        ),
    )

    print(
        "Accuracy:",
        round(
            accuracy,
            6,
        ),
    )

    print()

    print(
        "Dangerous false positives:",
        false_positive,
    )

    print(
        "BENCHMARK PASS:",
        benchmark_pass,
    )

    print()

    print(
        "Artifact:",
        OUTPUT_PATH,
    )

    print(
        "SHA256:",
        artifact_sha256,
    )


if __name__ == "__main__":
    main()
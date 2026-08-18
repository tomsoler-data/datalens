from __future__ import annotations


import hashlib
import json

from pathlib import Path

from typing import Any


from app.rag import (
    build_document_ingestion_report,
)

from app.rag_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    RAG_RETRIEVAL_RULE_VERSION,
    search_document_chunks,
)


# ============================================================
# BENCHMARK VERSION
# ============================================================

BENCHMARK_VERSION = (
    "rag_retrieval_benchmark_v0.1"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = Path(
    "artifacts/evaluation/experiments/"
    "rag_retrieval_benchmark_v0.1.json"
)


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

TOP_K = 3


# ============================================================
# SYNTHETIC DOCUMENT CORPUS
# ============================================================

DOCUMENTS = [
    {
        "document_id":
            "water_modernisation",

        "filename":
            "water_modernisation.txt",

        "text": (
            "La modernisation des services "
            "d'eau potable peut être étudiée "
            "en comparant la part de population "
            "ayant accès à un service basique "
            "avec la part bénéficiant d'un "
            "service géré en toute sécurité. "
            "Un écart important entre ces deux "
            "niveaux peut indiquer un besoin "
            "d'amélioration des infrastructures "
            "existantes."
        ),
    },

    {
        "document_id":
            "water_creation",

        "filename":
            "water_creation.txt",

        "text": (
            "La création de nouvelles "
            "infrastructures d'eau potable "
            "concerne en priorité les populations "
            "qui ne disposent pas encore d'un "
            "accès basique à l'eau. Le nombre "
            "d'habitants sans accès constitue "
            "donc un indicateur important pour "
            "identifier les besoins de création."
        ),
    },

    {
        "document_id":
            "wash_mortality",

        "filename":
            "wash_mortality.txt",

        "text": (
            "La mortalité attribuée à des "
            "services WASH insuffisants peut "
            "être utilisée pour mesurer une "
            "dimension sanitaire du besoin. "
            "Elle ne représente pas directement "
            "le niveau d'accès à l'eau, mais "
            "complète l'analyse des risques "
            "pour les populations."
        ),
    },

    {
        "document_id":
            "political_stability",

        "filename":
            "political_stability.txt",

        "text": (
            "La stabilité politique peut être "
            "prise en compte pour apprécier "
            "le contexte institutionnel d'une "
            "intervention. Une situation "
            "politique instable peut compliquer "
            "la réalisation, la continuité ou "
            "la gouvernance d'un projet."
        ),
    },

    {
        "document_id":
            "population_priority",

        "filename":
            "population_priority.txt",

        "text": (
            "La taille de la population permet "
            "d'estimer combien de personnes "
            "pourraient être concernées par une "
            "intervention. Une population élevée "
            "ne signifie toutefois pas à elle "
            "seule que le besoin est prioritaire."
        ),
    },

    {
        "document_id":
            "correlation_definition",

        "filename":
            "correlation_definition.txt",

        "text": (
            "Une corrélation mesure la force "
            "et la direction d'une association "
            "entre deux variables. Elle ne "
            "permet pas à elle seule de conclure "
            "à une relation causale entre les "
            "variables observées."
        ),
    },

    {
        "document_id":
            "missing_values",

        "filename":
            "missing_values.txt",

        "text": (
            "Les valeurs manquantes doivent "
            "être examinées avant l'analyse. "
            "Leur proportion, leur répartition "
            "et leur mécanisme potentiel peuvent "
            "modifier l'interprétation des "
            "résultats et la qualité des "
            "estimations."
        ),
    },

    {
        "document_id":
            "duplicate_rows",

        "filename":
            "duplicate_rows.txt",

        "text": (
            "Les doublons stricts correspondent "
            "à des lignes identiques présentes "
            "plusieurs fois dans un dataset. "
            "Ils peuvent artificiellement "
            "augmenter les effectifs ou modifier "
            "certaines statistiques s'ils ne "
            "représentent pas de véritables "
            "observations répétées."
        ),
    },

    {
        "document_id":
            "median_outliers",

        "filename":
            "median_outliers.txt",

        "text": (
            "La médiane décrit la valeur centrale "
            "d'une distribution et résiste mieux "
            "aux valeurs extrêmes que la moyenne. "
            "Elle peut être utile lorsque la "
            "distribution est asymétrique ou "
            "contient des observations atypiques."
        ),
    },

    {
        "document_id":
            "time_series",

        "filename":
            "time_series.txt",

        "text": (
            "Une série temporelle représente "
            "l'évolution d'une mesure au cours "
            "du temps. Il faut préserver l'ordre "
            "des périodes et vérifier que les "
            "observations comparées correspondent "
            "à un grain temporel cohérent."
        ),
    },

    {
        "document_id":
            "join_grain",

        "filename":
            "join_grain.txt",

        "text": (
            "Avant de joindre deux datasets, "
            "il faut vérifier leur grain "
            "analytique. Une relation plusieurs "
            "à plusieurs peut dupliquer les "
            "observations et produire des "
            "résultats artificiels si aucune "
            "clé d'appariement sûre n'existe."
        ),
    },

    {
        "document_id":
            "executive_report",

        "filename":
            "executive_report.txt",

        "text": (
            "Un rapport destiné à la décision "
            "doit mettre en avant les résultats "
            "les plus importants, les indicateurs "
            "clés, les limites de l'analyse et "
            "les recommandations. Les détails "
            "méthodologiques peuvent être "
            "présentés dans une section séparée."
        ),
    },
]


# ============================================================
# QUERIES
# ============================================================

CASES = [
    {
        "case_id":
            "modernisation_gap",

        "query": (
            "Comment repérer un besoin de "
            "modernisation des infrastructures "
            "d'eau potable ?"
        ),

        "relevant_document_ids": [
            "water_modernisation",
        ],
    },

    {
        "case_id":
            "creation_without_basic_access",

        "query": (
            "Quel indicateur permet d'identifier "
            "les populations nécessitant la "
            "création de nouvelles infrastructures "
            "d'eau potable ?"
        ),

        "relevant_document_ids": [
            "water_creation",
        ],
    },

    {
        "case_id":
            "wash_health_risk",

        "query": (
            "Quelle donnée peut compléter "
            "l'analyse de l'accès à l'eau en "
            "mesurant le risque sanitaire ?"
        ),

        "relevant_document_ids": [
            "wash_mortality",
        ],
    },

    {
        "case_id":
            "institutional_context",

        "query": (
            "Quel indicateur renseigne sur "
            "le contexte institutionnel et les "
            "difficultés potentielles de mise "
            "en œuvre d'un projet ?"
        ),

        "relevant_document_ids": [
            "political_stability",
        ],
    },

    {
        "case_id":
            "number_people_affected",

        "query": (
            "Quelle information aide à estimer "
            "le nombre de personnes pouvant "
            "être concernées par une intervention ?"
        ),

        "relevant_document_ids": [
            "population_priority",
        ],
    },

    {
        "case_id":
            "correlation_not_causation",

        "query": (
            "Pourquoi une association statistique "
            "entre deux variables ne prouve-t-elle "
            "pas une causalité ?"
        ),

        "relevant_document_ids": [
            "correlation_definition",
        ],
    },

    {
        "case_id":
            "missing_data_quality",

        "query": (
            "Que faut-il contrôler lorsque "
            "certaines observations contiennent "
            "des données absentes ?"
        ),

        "relevant_document_ids": [
            "missing_values",
        ],
    },

    {
        "case_id":
            "duplicated_observations",

        "query": (
            "Quel problème peut survenir lorsque "
            "des lignes identiques apparaissent "
            "plusieurs fois dans les données ?"
        ),

        "relevant_document_ids": [
            "duplicate_rows",
        ],
    },

    {
        "case_id":
            "robust_central_value",

        "query": (
            "Quelle mesure de tendance centrale "
            "est moins sensible aux valeurs "
            "extrêmes que la moyenne ?"
        ),

        "relevant_document_ids": [
            "median_outliers",
        ],
    },

    {
        "case_id":
            "evolution_over_time",

        "query": (
            "Quel type d'analyse permet "
            "d'étudier comment une mesure évolue "
            "au fil des années ?"
        ),

        "relevant_document_ids": [
            "time_series",
        ],
    },

    {
        "case_id":
            "unsafe_many_to_many_join",

        "query": (
            "Pourquoi faut-il vérifier le grain "
            "avant de joindre deux tables ayant "
            "potentiellement plusieurs lignes "
            "pour la même clé ?"
        ),

        "relevant_document_ids": [
            "join_grain",
        ],
    },

    {
        "case_id":
            "decision_report_structure",

        "query": (
            "Que doit mettre en avant un rapport "
            "d'analyse destiné à des décideurs ?"
        ),

        "relevant_document_ids": [
            "executive_report",
        ],
    },
]


# ============================================================
# HASH HELPERS
# ============================================================

def canonical_json_bytes(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )


def sha256_json(
    value: Any,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            value
        )
    ).hexdigest()


# ============================================================
# DOCUMENT LOOKUP
# ============================================================

def document_id_by_filename() -> dict[
    str,
    str,
]:
    return {
        document[
            "filename"
        ]:
            document[
                "document_id"
            ]

        for document
        in DOCUMENTS
    }


# ============================================================
# METRICS
# ============================================================

def reciprocal_rank(
    ranked_document_ids: list[
        str
    ],
    relevant_document_ids: set[
        str
    ],
) -> float:
    for (
        rank,
        document_id,
    ) in enumerate(
        ranked_document_ids,
        start=1,
    ):
        if (
            document_id
            in relevant_document_ids
        ):
            return (
                1.0
                /
                rank
            )


    return 0.0


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # --------------------------------------------------------
    # IMPORTANT:
    # Refuse overwrite BEFORE any embedding call.
    # --------------------------------------------------------

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            (
                "Evaluation artifact already "
                "exists. Refusing to overwrite:\n"
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
        "DATALENS — RAG RETRIEVAL BENCHMARK"
    )

    print(
        "=" * 100
    )

    print(
        "Benchmark:",
        BENCHMARK_VERSION,
    )

    print(
        "Retriever:",
        RAG_RETRIEVAL_RULE_VERSION,
    )

    print(
        "Embedding model:",
        DEFAULT_EMBEDDING_MODEL,
    )

    print(
        "Documents:",
        len(
            DOCUMENTS
        ),
    )

    print(
        "Cases:",
        len(
            CASES
        ),
    )

    print(
        "Top-K:",
        TOP_K,
    )


    # ========================================================
    # INGESTION
    # ========================================================

    ingestion = (
        build_document_ingestion_report(
            documents=[
                (
                    document[
                        "filename"
                    ],

                    document[
                        "text"
                    ].encode(
                        "utf-8"
                    ),
                )

                for document
                in DOCUMENTS
            ]
        )
    )


    filename_lookup = (
        document_id_by_filename()
    )


    case_results: list[
        dict[
            str,
            Any,
        ]
    ] = []


    recall_at_1_hits = 0

    recall_at_3_hits = 0

    reciprocal_rank_sum = 0.0


    # ========================================================
    # RETRIEVAL CASES
    # ========================================================

    for (
        case_index,
        case,
    ) in enumerate(
        CASES,
        start=1,
    ):
        result = (
            search_document_chunks(
                ingestion=
                    ingestion,

                query=
                    case[
                        "query"
                    ],

                top_k=
                    TOP_K,

                model=
                    DEFAULT_EMBEDDING_MODEL,
            )
        )


        ranked_document_ids = [
            filename_lookup[
                hit.filename
            ]

            for hit
            in result.hits
        ]


        relevant_document_ids = set(
            case[
                "relevant_document_ids"
            ]
        )


        recall_at_1 = (
            bool(
                ranked_document_ids
            )
            and
            ranked_document_ids[
                0
            ]
            in relevant_document_ids
        )


        recall_at_3 = any(
            document_id
            in relevant_document_ids

            for document_id
            in ranked_document_ids[
                :3
            ]
        )


        rr = reciprocal_rank(
            ranked_document_ids,
            relevant_document_ids,
        )


        if recall_at_1:
            recall_at_1_hits += 1


        if recall_at_3:
            recall_at_3_hits += 1


        reciprocal_rank_sum += rr


        case_result = {
            "case_id":
                case[
                    "case_id"
                ],

            "query":
                case[
                    "query"
                ],

            "relevant_document_ids":
                sorted(
                    relevant_document_ids
                ),

            "ranked_document_ids":
                ranked_document_ids,

            "scores": [
                {
                    "rank":
                        hit.rank,

                    "document_id":
                        filename_lookup[
                            hit.filename
                        ],

                    "filename":
                        hit.filename,

                    "score":
                        hit.score,
                }

                for hit
                in result.hits
            ],

            "recall_at_1":
                recall_at_1,

            "recall_at_3":
                recall_at_3,

            "reciprocal_rank":
                rr,
        }


        case_results.append(
            case_result
        )


        print()

        print(
            f"[{case_index:02d}]",
            case[
                "case_id"
            ],
        )

        print(
            "  expected:",
            ", ".join(
                sorted(
                    relevant_document_ids
                )
            ),
        )

        print(
            "  ranking:",
            " > ".join(
                ranked_document_ids
            ),
        )

        print(
            "  R@1:",
            recall_at_1,
        )

        print(
            "  R@3:",
            recall_at_3,
        )

        print(
            "  RR:",
            round(
                rr,
                6,
            ),
        )


    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    case_count = len(
        CASES
    )


    recall_at_1 = (
        recall_at_1_hits
        /
        case_count
    )


    recall_at_3 = (
        recall_at_3_hits
        /
        case_count
    )


    mrr = (
        reciprocal_rank_sum
        /
        case_count
    )


    metrics = {
        "case_count":
            case_count,

        "recall_at_1":
            recall_at_1,

        "recall_at_3":
            recall_at_3,

        "mrr":
            mrr,
    }


    # ========================================================
    # ARTIFACT
    # ========================================================

    benchmark_payload = {
        "benchmark_version":
            BENCHMARK_VERSION,

        "retrieval_rule_version":
            RAG_RETRIEVAL_RULE_VERSION,

        "embedding_model":
            DEFAULT_EMBEDDING_MODEL,

        "top_k":
            TOP_K,

        "document_count":
            len(
                DOCUMENTS
            ),

        "chunk_count":
            ingestion.chunk_count,

        "documents_sha256":
            sha256_json(
                DOCUMENTS
            ),

        "cases_sha256":
            sha256_json(
                CASES
            ),

        "metrics":
            metrics,

        "cases":
            case_results,
    }


    OUTPUT_PATH.write_text(
        json.dumps(
            benchmark_payload,
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
    # FINAL SUMMARY
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
        "Recall@1:",
        round(
            recall_at_1,
            6,
        ),
    )

    print(
        "Recall@3:",
        round(
            recall_at_3,
            6,
        ),
    )

    print(
        "MRR:",
        round(
            mrr,
            6,
        ),
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
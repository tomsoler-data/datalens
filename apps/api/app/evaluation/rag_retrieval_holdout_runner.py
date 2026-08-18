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
# HOLDOUT VERSION
# ============================================================

HOLDOUT_VERSION = (
    "rag_retrieval_holdout_v0.1"
)


# ============================================================
# PATHS
# ============================================================

FREEZE_PATH = Path(
    "artifacts/evaluation/holdouts/"
    "rag_retrieval_holdout_v0.1_freeze.json"
)


FIRST_RUN_PATH = Path(
    "artifacts/evaluation/experiments/"
    "rag_retrieval_holdout_v0.1_first_run.json"
)


# ============================================================
# RETRIEVAL CONFIG
# ============================================================

TOP_K = 3


# ============================================================
# PREREGISTERED GATES
# ============================================================

MIN_RECALL_AT_1 = 0.80

MIN_RECALL_AT_3 = 0.95

MIN_MRR = 0.85


# ============================================================
# INDEPENDENT HOLDOUT DOCUMENTS
# ============================================================

DOCUMENTS = [
    {
        "document_id":
            "subscription_churn",

        "filename":
            "subscription_churn.txt",

        "text": (
            "Une hausse des résiliations peut "
            "être précédée par une baisse de "
            "l'utilisation, des interactions "
            "plus rares avec le service ou une "
            "augmentation des réclamations. "
            "L'analyse du churn doit distinguer "
            "les signaux précédant la résiliation "
            "des événements observés après celle-ci."
        ),
    },

    {
        "document_id":
            "fulfillment_delay",

        "filename":
            "fulfillment_delay.txt",

        "text": (
            "Le délai de traitement d'une commande "
            "peut être décomposé entre préparation, "
            "emballage et expédition. Une accumulation "
            "de commandes entre le picking et le départ "
            "du transporteur peut révéler un goulot "
            "d'étranglement dans l'entrepôt."
        ),
    },

    {
        "document_id":
            "campaign_incrementality",

        "filename":
            "campaign_incrementality.txt",

        "text": (
            "Mesurer l'efficacité réelle d'une campagne "
            "marketing nécessite de distinguer les ventes "
            "qui auraient eu lieu sans campagne des ventes "
            "supplémentaires réellement provoquées par "
            "l'intervention. Une simple corrélation entre "
            "exposition et conversion ne mesure pas cet "
            "effet incrémental."
        ),
    },

    {
        "document_id":
            "cashflow_liquidity",

        "filename":
            "cashflow_liquidity.txt",

        "text": (
            "Le suivi de la liquidité repose sur les "
            "entrées et sorties de trésorerie ainsi que "
            "sur les disponibilités permettant de couvrir "
            "les obligations à court terme. Un résultat "
            "comptable positif ne garantit pas qu'une "
            "organisation dispose immédiatement de "
            "suffisamment de liquidités."
        ),
    },

    {
        "document_id":
            "sensor_drift",

        "filename":
            "sensor_drift.txt",

        "text": (
            "Un capteur peut progressivement perdre son "
            "étalonnage et produire des mesures de plus "
            "en plus biaisées au cours du temps. Cette "
            "dérive peut être détectée en comparant les "
            "mesures à une référence stable et en suivant "
            "l'évolution de l'erreur."
        ),
    },

    {
        "document_id":
            "staffing_queue",

        "filename":
            "staffing_queue.txt",

        "text": (
            "Dans un service client, les files d'attente "
            "peuvent augmenter lorsque le nombre d'agents "
            "disponibles ne suit pas le volume de demandes. "
            "Il faut donc étudier conjointement l'arrivée "
            "des demandes, la capacité disponible et le "
            "temps d'attente."
        ),
    },

    {
        "document_id":
            "data_leakage",

        "filename":
            "data_leakage.txt",

        "text": (
            "Une fuite de données apparaît lorsqu'un modèle "
            "utilise pendant son entraînement une information "
            "qui ne serait pas disponible au moment réel de "
            "la prédiction. Cela peut produire des performances "
            "artificiellement élevées qui ne se reproduisent "
            "pas en production."
        ),
    },

    {
        "document_id":
            "confidence_interval",

        "filename":
            "confidence_interval.txt",

        "text": (
            "Un intervalle de confiance exprime l'incertitude "
            "associée à une estimation. Il fournit une plage "
            "de valeurs compatibles avec les données et la "
            "procédure statistique employée, plutôt qu'une "
            "garantie que le paramètre varie à l'intérieur "
            "de cet intervalle."
        ),
    },

    {
        "document_id":
            "cohort_retention",

        "filename":
            "cohort_retention.txt",

        "text": (
            "L'analyse de rétention par cohorte consiste "
            "à regrouper les utilisateurs selon une période "
            "commune d'acquisition ou d'inscription, puis "
            "à suivre la proportion encore active après "
            "un, deux ou plusieurs intervalles de temps."
        ),
    },

    {
        "document_id":
            "access_control",

        "filename":
            "access_control.txt",

        "text": (
            "Le principe du moindre privilège consiste "
            "à accorder à un utilisateur ou à un service "
            "uniquement les permissions nécessaires à "
            "l'exécution de sa tâche. Les autorisations "
            "inutiles augmentent la surface de risque."
        ),
    },

    {
        "document_id":
            "seasonal_demand",

        "filename":
            "seasonal_demand.txt",

        "text": (
            "Une demande saisonnière présente des motifs "
            "qui se répètent à des intervalles réguliers, "
            "par exemple selon les jours de la semaine, "
            "les mois ou les saisons. Ces cycles doivent "
            "être distingués d'une tendance de long terme."
        ),
    },

    {
        "document_id":
            "entity_resolution",

        "filename":
            "entity_resolution.txt",

        "text": (
            "Une même personne ou entreprise peut apparaître "
            "sous plusieurs formes dans différents fichiers : "
            "variation orthographique du nom, adresse différente "
            "ou identifiant manquant. La résolution d'entités "
            "cherche à déterminer quelles lignes représentent "
            "réellement la même entité."
        ),
    },
]


# ============================================================
# HOLDOUT QUERIES
# ============================================================

CASES = [
    {
        "case_id":
            "future_information_in_features",

        "query": (
            "Pourquoi un modèle peut-il sembler "
            "anormalement performant lorsque certaines "
            "variables contiennent des informations "
            "qui ne seront connues qu'après la prédiction ?"
        ),

        "relevant_document_ids": [
            "data_leakage",
        ],
    },

    {
        "case_id":
            "signup_month_retention",

        "query": (
            "Comment comparer la fidélisation de clients "
            "inscrits à différentes périodes et voir "
            "combien restent actifs plusieurs mois plus tard ?"
        ),

        "relevant_document_ids": [
            "cohort_retention",
        ],
    },

    {
        "case_id":
            "duplicate_customer_identity",

        "query": (
            "Deux bases contiennent probablement le même "
            "client, mais son nom et son adresse ne sont "
            "pas écrits exactement de la même façon. "
            "Quel problème faut-il résoudre ?"
        ),

        "relevant_document_ids": [
            "entity_resolution",
        ],
    },

    {
        "case_id":
            "uncertainty_around_estimate",

        "query": (
            "Comment représenter une plage d'incertitude "
            "autour d'une estimation statistique plutôt "
            "que de présenter uniquement une valeur ponctuelle ?"
        ),

        "relevant_document_ids": [
            "confidence_interval",
        ],
    },

    {
        "case_id":
            "marketing_causal_lift",

        "query": (
            "Comment savoir si une campagne a réellement "
            "généré des conversions supplémentaires au lieu "
            "de simplement toucher des personnes qui auraient "
            "acheté de toute façon ?"
        ),

        "relevant_document_ids": [
            "campaign_incrementality",
        ],
    },

    {
        "case_id":
            "measurement_bias_over_time",

        "query": (
            "Un appareil donne des valeurs progressivement "
            "plus éloignées d'une référence alors qu'il "
            "fonctionnait correctement au départ. "
            "Quel phénomène faut-il rechercher ?"
        ),

        "relevant_document_ids": [
            "sensor_drift",
        ],
    },

    {
        "case_id":
            "support_waiting_capacity",

        "query": (
            "Pourquoi le temps d'attente des clients "
            "augmente-t-il lorsque les demandes arrivent "
            "plus vite que les équipes ne peuvent les traiter ?"
        ),

        "relevant_document_ids": [
            "staffing_queue",
        ],
    },

    {
        "case_id":
            "short_term_payment_capacity",

        "query": (
            "Quelle analyse permet de savoir si une "
            "entreprise dispose réellement d'assez d'argent "
            "disponible pour payer ses engagements proches ?"
        ),

        "relevant_document_ids": [
            "cashflow_liquidity",
        ],
    },

    {
        "case_id":
            "warehouse_bottleneck",

        "query": (
            "Des commandes restent longtemps en attente "
            "après leur préparation mais avant leur remise "
            "au transporteur. Quel processus mérite d'être "
            "examiné ?"
        ),

        "relevant_document_ids": [
            "fulfillment_delay",
        ],
    },

    {
        "case_id":
            "minimum_permissions",

        "query": (
            "Quelle règle de sécurité recommande de ne "
            "donner à chaque compte que les droits "
            "strictement nécessaires à son travail ?"
        ),

        "relevant_document_ids": [
            "access_control",
        ],
    },

    {
        "case_id":
            "recurring_demand_cycles",

        "query": (
            "Comment qualifier des pics d'activité qui "
            "réapparaissent selon un rythme régulier, "
            "par exemple chaque semaine ou chaque année ?"
        ),

        "relevant_document_ids": [
            "seasonal_demand",
        ],
    },

    {
        "case_id":
            "signals_before_cancellation",

        "query": (
            "Quels comportements peuvent annoncer qu'un "
            "abonné risque de quitter un service avant "
            "que sa résiliation ne soit effectivement enregistrée ?"
        ),

        "relevant_document_ids": [
            "subscription_churn",
        ],
    },
]


# ============================================================
# JSON / HASH HELPERS
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


def file_sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# ============================================================
# LOOKUP
# ============================================================

def build_filename_lookup() -> dict[
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
# RECIPROCAL RANK
# ============================================================

def reciprocal_rank(
    *,
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
# FREEZE
# ============================================================

def build_freeze_payload() -> dict[
    str,
    Any,
]:
    return {
        "holdout_version":
            HOLDOUT_VERSION,

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

        "case_count":
            len(
                CASES
            ),

        "documents_sha256":
            sha256_json(
                DOCUMENTS
            ),

        "cases_sha256":
            sha256_json(
                CASES
            ),

        "preregistered_gates": {
            "minimum_recall_at_1":
                MIN_RECALL_AT_1,

            "minimum_recall_at_3":
                MIN_RECALL_AT_3,

            "minimum_mrr":
                MIN_MRR,
        },

        "policy": {
            "freeze_before_first_embedding_execution":
                True,

            "first_run_artifact_is_immutable":
                True,

            "tuning_on_this_holdout_requires_new_holdout":
                True,
        },
    }


def create_freeze() -> None:
    FREEZE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = (
        build_freeze_payload()
    )


    FREEZE_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "=" * 100
    )

    print(
        "RAG RETRIEVAL HOLDOUT — FREEZE CREATED"
    )

    print(
        "=" * 100
    )

    print(
        "Holdout:",
        HOLDOUT_VERSION,
    )

    print(
        "Retriever:",
        RAG_RETRIEVAL_RULE_VERSION,
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
        "Documents SHA256:",
        payload[
            "documents_sha256"
        ],
    )

    print(
        "Cases SHA256:",
        payload[
            "cases_sha256"
        ],
    )

    print(
        "Freeze artifact:",
        FREEZE_PATH,
    )

    print(
        "Freeze SHA256:",
        file_sha256(
            FREEZE_PATH
        ),
    )

    print()

    print(
        "NO EMBEDDING CALL WAS EXECUTED."
    )

    print(
        "Run the same command again "
        "for the official first holdout run."
    )


# ============================================================
# LOAD / VERIFY FREEZE
# ============================================================

def load_freeze() -> dict[
    str,
    Any,
]:
    return json.loads(
        FREEZE_PATH.read_text(
            encoding="utf-8"
        )
    )


def verify_freeze(
    freeze: dict[
        str,
        Any,
    ],
) -> None:
    expected = (
        build_freeze_payload()
    )


    fields = [
        "holdout_version",
        "retrieval_rule_version",
        "embedding_model",
        "top_k",
        "document_count",
        "case_count",
        "documents_sha256",
        "cases_sha256",
        "preregistered_gates",
    ]


    mismatches = [
        field

        for field
        in fields

        if (
            freeze.get(
                field
            )
            !=
            expected.get(
                field
            )
        )
    ]


    if mismatches:
        raise RuntimeError(
            (
                "Frozen holdout no longer "
                "matches the current benchmark. "
                "Mismatched fields: "
                +
                ", ".join(
                    mismatches
                )
            )
        )


# ============================================================
# OFFICIAL FIRST RUN
# ============================================================

def run_holdout() -> None:
    if FIRST_RUN_PATH.exists():
        raise FileExistsError(
            (
                "Official first-run artifact "
                "already exists. Refusing to "
                "rerun or overwrite it:\n"
                f"{FIRST_RUN_PATH}"
            )
        )


    freeze = (
        load_freeze()
    )


    verify_freeze(
        freeze
    )


    FIRST_RUN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "=" * 100
    )

    print(
        "DATALENS — INDEPENDENT RAG RETRIEVAL HOLDOUT"
    )

    print(
        "=" * 100
    )

    print(
        "Holdout:",
        HOLDOUT_VERSION,
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
        "Documents hash match:",
        True,
    )

    print(
        "Cases hash match:",
        True,
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
        build_filename_lookup()
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
            ranked_document_ids=
                ranked_document_ids,

            relevant_document_ids=
                relevant_document_ids,
        )


        if recall_at_1:
            recall_at_1_hits += 1


        if recall_at_3:
            recall_at_3_hits += 1


        reciprocal_rank_sum += rr


        case_results.append(
            {
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


    pass_recall_at_1 = (
        recall_at_1
        >=
        MIN_RECALL_AT_1
    )


    pass_recall_at_3 = (
        recall_at_3
        >=
        MIN_RECALL_AT_3
    )


    pass_mrr = (
        mrr
        >=
        MIN_MRR
    )


    holdout_pass = (
        pass_recall_at_1
        and
        pass_recall_at_3
        and
        pass_mrr
    )


    artifact = {
        "holdout_version":
            HOLDOUT_VERSION,

        "retrieval_rule_version":
            RAG_RETRIEVAL_RULE_VERSION,

        "embedding_model":
            DEFAULT_EMBEDDING_MODEL,

        "freeze_sha256":
            file_sha256(
                FREEZE_PATH
            ),

        "documents_sha256":
            freeze[
                "documents_sha256"
            ],

        "cases_sha256":
            freeze[
                "cases_sha256"
            ],

        "top_k":
            TOP_K,

        "document_count":
            len(
                DOCUMENTS
            ),

        "chunk_count":
            ingestion.chunk_count,

        "case_count":
            case_count,

        "metrics": {
            "recall_at_1":
                recall_at_1,

            "recall_at_3":
                recall_at_3,

            "mrr":
                mrr,
        },

        "gates": {
            "minimum_recall_at_1":
                MIN_RECALL_AT_1,

            "minimum_recall_at_3":
                MIN_RECALL_AT_3,

            "minimum_mrr":
                MIN_MRR,

            "recall_at_1_pass":
                pass_recall_at_1,

            "recall_at_3_pass":
                pass_recall_at_3,

            "mrr_pass":
                pass_mrr,

            "holdout_pass":
                holdout_pass,
        },

        "cases":
            case_results,
    }


    FIRST_RUN_PATH.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()

    print(
        "=" * 100
    )

    print(
        "HOLDOUT RESULTS"
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

        "| gate:",
        MIN_RECALL_AT_1,

        "| pass:",
        pass_recall_at_1,
    )

    print(
        "Recall@3:",
        round(
            recall_at_3,
            6,
        ),

        "| gate:",
        MIN_RECALL_AT_3,

        "| pass:",
        pass_recall_at_3,
    )

    print(
        "MRR:",
        round(
            mrr,
            6,
        ),

        "| gate:",
        MIN_MRR,

        "| pass:",
        pass_mrr,
    )

    print()

    print(
        "INDEPENDENT HOLDOUT PASS:",
        holdout_pass,
    )

    print()

    print(
        "First-run artifact:",
        FIRST_RUN_PATH,
    )

    print(
        "SHA256:",
        file_sha256(
            FIRST_RUN_PATH
        ),
    )


# ============================================================
# ENTRYPOINT
# ============================================================

def main() -> None:
    if not FREEZE_PATH.exists():
        create_freeze()

        return


    run_holdout()


if __name__ == "__main__":
    main()
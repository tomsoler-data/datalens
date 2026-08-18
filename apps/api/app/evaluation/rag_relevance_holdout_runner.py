from __future__ import annotations


import hashlib
import json

from pathlib import Path

from typing import (
    Any,
)


from app.rag_relevance import (
    DEFAULT_RELEVANCE_MODEL,
    RELEVANCE_RULE_VERSION,
    SYSTEM_PROMPT,
    classify_relevance,
)


# ============================================================
# VERSION
# ============================================================

HOLDOUT_VERSION = (
    "rag_relevance_holdout_v0.1"
)


# ============================================================
# PATHS
# ============================================================

FREEZE_PATH = Path(
    "artifacts/evaluation/holdouts/"
    "rag_relevance_holdout_v0.1_freeze.json"
)


FIRST_RUN_PATH = Path(
    "artifacts/evaluation/experiments/"
    "rag_relevance_holdout_v0.1_first_run.json"
)


# ============================================================
# ACCEPTANCE GATES
# ============================================================

MIN_POSITIVE_RECALL = 0.80

MAX_DANGEROUS_FALSE_POSITIVES = 0


# ============================================================
# INDEPENDENT HOLDOUT
# ============================================================

CASES = [
    # --------------------------------------------------------
    # POSITIVE CASES
    # --------------------------------------------------------

    {
        "case_id":
            "positive_inventory_stockout",

        "finding": (
            "Certains produits connaissent "
            "des ruptures de stock fréquentes."
        ),

        "passage": (
            "Une rupture de stock apparaît "
            "lorsque la demande d'un produit "
            "ne peut plus être satisfaite par "
            "le stock disponible. Sa fréquence "
            "peut signaler un niveau de stock "
            "insuffisant ou un réapprovisionnement "
            "trop lent."
        ),

        "expected":
            "relevant",
    },

    {
        "case_id":
            "positive_delivery_delay",

        "finding": (
            "Le délai entre la préparation "
            "d'une commande et son expédition "
            "est particulièrement élevé."
        ),

        "passage": (
            "Le délai logistique peut être "
            "décomposé entre préparation, "
            "emballage et expédition. Une attente "
            "importante avant la remise au "
            "transporteur peut révéler un "
            "goulot d'étranglement."
        ),

        "expected":
            "relevant",
    },

    {
        "case_id":
            "positive_data_leakage",

        "finding": (
            "Les performances du modèle sont "
            "très élevées en validation mais "
            "chutent fortement en production."
        ),

        "passage": (
            "Une fuite de données peut produire "
            "des performances artificiellement "
            "élevées lorsqu'une variable utilisée "
            "pendant l'entraînement contient une "
            "information indisponible au moment "
            "réel de la prédiction."
        ),

        "expected":
            "relevant",
    },

    {
        "case_id":
            "positive_customer_cohort",

        "finding": (
            "La rétention des utilisateurs "
            "inscrits en janvier est comparée "
            "à celle des utilisateurs inscrits "
            "en février."
        ),

        "passage": (
            "Une analyse par cohorte regroupe "
            "les utilisateurs selon une période "
            "commune d'inscription ou d'acquisition "
            "puis suit leur activité au cours "
            "des périodes suivantes."
        ),

        "expected":
            "relevant",
    },

    {
        "case_id":
            "positive_sensor_drift",

        "finding": (
            "L'écart entre les mesures du capteur "
            "et la référence augmente progressivement "
            "au fil du temps."
        ),

        "passage": (
            "La dérive d'un capteur correspond "
            "à une perte progressive d'étalonnage. "
            "Elle peut être observée lorsque "
            "l'erreur par rapport à une référence "
            "stable augmente avec le temps."
        ),

        "expected":
            "relevant",
    },

    {
        "case_id":
            "positive_least_privilege",

        "finding": (
            "Plusieurs comptes disposent de "
            "permissions qui ne sont pas nécessaires "
            "à leurs fonctions."
        ),

        "passage": (
            "Le principe du moindre privilège "
            "consiste à attribuer uniquement "
            "les autorisations nécessaires à "
            "l'exécution d'une tâche et à retirer "
            "les droits inutiles."
        ),

        "expected":
            "relevant",
    },

    # --------------------------------------------------------
    # HARD NEGATIVE CASES
    # --------------------------------------------------------

    {
        "case_id":
            "negative_stockout_revenue",

        "finding": (
            "Certains produits connaissent "
            "des ruptures de stock fréquentes."
        ),

        "passage": (
            "Le chiffre d'affaires correspond "
            "au montant total des ventes réalisées "
            "sur une période donnée. Il peut être "
            "analysé par produit, catégorie ou région."
        ),

        "expected":
            "not_relevant",
    },

    {
        "case_id":
            "negative_delivery_customer_satisfaction",

        "finding": (
            "Le délai entre la préparation "
            "d'une commande et son expédition "
            "est particulièrement élevé."
        ),

        "passage": (
            "La satisfaction client peut être "
            "mesurée par enquête après achat. "
            "Un score faible peut signaler "
            "une expérience globale insatisfaisante."
        ),

        "expected":
            "not_relevant",
    },

    {
        "case_id":
            "negative_model_performance_confidence",

        "finding": (
            "Les performances du modèle sont "
            "très élevées en validation mais "
            "chutent fortement en production."
        ),

        "passage": (
            "Un intervalle de confiance fournit "
            "une représentation de l'incertitude "
            "autour d'une estimation statistique."
        ),

        "expected":
            "not_relevant",
    },

    {
        "case_id":
            "negative_cohort_seasonality",

        "finding": (
            "La rétention des utilisateurs "
            "inscrits en janvier est comparée "
            "à celle des utilisateurs inscrits "
            "en février."
        ),

        "passage": (
            "Une série saisonnière présente "
            "des motifs qui reviennent régulièrement "
            "selon les semaines, les mois ou "
            "les saisons."
        ),

        "expected":
            "not_relevant",
    },

    {
        "case_id":
            "negative_sensor_missing_values",

        "finding": (
            "L'écart entre les mesures du capteur "
            "et la référence augmente progressivement "
            "au fil du temps."
        ),

        "passage": (
            "Les valeurs manquantes correspondent "
            "à des observations pour lesquelles "
            "une ou plusieurs variables n'ont "
            "pas de valeur enregistrée."
        ),

        "expected":
            "not_relevant",
    },

    {
        "case_id":
            "negative_permissions_authentication",

        "finding": (
            "Plusieurs comptes disposent de "
            "permissions qui ne sont pas nécessaires "
            "à leurs fonctions."
        ),

        "passage": (
            "L'authentification multifacteur "
            "demande plusieurs preuves d'identité "
            "avant d'autoriser la connexion "
            "à un compte."
        ),

        "expected":
            "not_relevant",
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


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def file_sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


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

        "relevance_rule_version":
            RELEVANCE_RULE_VERSION,

        "model":
            DEFAULT_RELEVANCE_MODEL,

        "case_count":
            len(
                CASES
            ),

        "cases_sha256":
            sha256_json(
                CASES
            ),

        "system_prompt_sha256":
            sha256_text(
                SYSTEM_PROMPT
            ),

        "preregistered_gates": {
            "minimum_positive_recall":
                MIN_POSITIVE_RECALL,

            "maximum_dangerous_false_positives":
                MAX_DANGEROUS_FALSE_POSITIVES,
        },

        "policy": {
            "freeze_before_first_model_execution":
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
        "RAG RELEVANCE HOLDOUT — FREEZE CREATED"
    )

    print(
        "=" * 100
    )

    print(
        "Holdout:",
        HOLDOUT_VERSION,
    )

    print(
        "Rule:",
        RELEVANCE_RULE_VERSION,
    )

    print(
        "Model:",
        DEFAULT_RELEVANCE_MODEL,
    )

    print(
        "Cases:",
        len(
            CASES
        ),
    )

    print(
        "Cases SHA256:",
        payload[
            "cases_sha256"
        ],
    )

    print(
        "System prompt SHA256:",
        payload[
            "system_prompt_sha256"
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
        "NO MODEL CALL WAS EXECUTED."
    )

    print(
        "Run the same command again "
        "for the official first holdout run."
    )


# ============================================================
# VERIFY FREEZE
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
    current = (
        build_freeze_payload()
    )


    fields = [
        "holdout_version",
        "relevance_rule_version",
        "model",
        "case_count",
        "cases_sha256",
        "system_prompt_sha256",
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
            current.get(
                field
            )
        )
    ]


    if mismatches:
        raise RuntimeError(
            (
                "Frozen relevance holdout "
                "does not match the current "
                "implementation. Mismatches: "
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
                "rerun or overwrite:\n"
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
        "DATALENS — INDEPENDENT RAG RELEVANCE HOLDOUT"
    )

    print(
        "=" * 100
    )

    print(
        "Holdout:",
        HOLDOUT_VERSION,
    )

    print(
        "Rule:",
        RELEVANCE_RULE_VERSION,
    )

    print(
        "Model:",
        DEFAULT_RELEVANCE_MODEL,
    )

    print(
        "Cases hash match:",
        True,
    )

    print(
        "System prompt hash match:",
        True,
    )

    print(
        "Cases:",
        len(
            CASES
        ),
    )


    results: list[
        dict[
            str,
            Any,
        ]
    ] = []


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

                model=
                    DEFAULT_RELEVANCE_MODEL,
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
            expected
            ==
            predicted
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


    holdout_pass = (
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

        "holdout_pass":
            holdout_pass,
    }


    artifact = {
        "holdout_version":
            HOLDOUT_VERSION,

        "relevance_rule_version":
            RELEVANCE_RULE_VERSION,

        "model":
            DEFAULT_RELEVANCE_MODEL,

        "freeze_sha256":
            file_sha256(
                FREEZE_PATH
            ),

        "cases_sha256":
            freeze[
                "cases_sha256"
            ],

        "system_prompt_sha256":
            freeze[
                "system_prompt_sha256"
            ],

        "case_count":
            len(
                CASES
            ),

        "metrics":
            metrics,

        "gates":
            gates,

        "cases":
            results,
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

        "| gate:",
        MIN_POSITIVE_RECALL,

        "| pass:",
        positive_recall_pass,
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

        "| maximum:",
        MAX_DANGEROUS_FALSE_POSITIVES,

        "| pass:",
        false_positive_pass,
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
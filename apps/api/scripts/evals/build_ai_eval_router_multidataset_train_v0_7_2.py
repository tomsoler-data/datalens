from __future__ import annotations

import json

from pathlib import Path


# ============================================================
# VERSION
# ============================================================

BENCHMARK_VERSION = (
    "decision_router_multidataset_train_v0.7.2"
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parents[2]


OUTPUT_PATH = (
    BASE_DIR
    / "evals"
    / "decision_router_multidataset_train_v0_7_2.jsonl"
)


# ============================================================
# BASE TOOLS
#
# No join tool unless explicitly added by a case.
# ============================================================

BASE_TOOLS = [
    "aggregate",
    "build_entity_view",
    "derive_metric",
    "analyze_distribution",
    "detect_outliers",
    "detect_entity_outliers",
    "compare_groups",
    "measure_association",
    "analyze_time_series",
]


TOOLS_WITH_JOIN = [
    *BASE_TOOLS,
    "join_datasets",
]


# ============================================================
# CASES
# ============================================================

CASES = [

    # ========================================================
    # 001
    #
    # CROSS-DATASET ASSOCIATION
    # JOIN REQUIRED, JOIN NOT AVAILABLE
    # ========================================================

    {
        "case_id":
            "router_md_v0_7_2_train_001",

        "split":
            "train",

        "domain":
            "warehouse_operations",

        "user_request":
            (
                "Le nombre quotidien de ruptures de stock "
                "est-il associé au chiffre d'affaires quotidien ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "inventory",

                "filename":
                    "inventory.csv",

                "grain":
                    "product_day",

                "entity_columns": [
                    "product_id",
                ],

                "columns": [
                    {
                        "name":
                            "product_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "inventory_date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "stockout_flag",

                        "analytical_type":
                            "categorical",
                    },
                ],
            },

            {
                "dataset_id":
                    "sales",

                "filename":
                    "sales.csv",

                "grain":
                    "day",

                "entity_columns":
                    [],

                "columns": [
                    {
                        "name":
                            "sales_date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "daily_revenue",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },
        ],

        "available_tools":
            BASE_TOOLS,

        "expected": {
            "decision":
                "cannot_answer",

            "decision_reason":
                "unsupported_analysis",

            "clarification_topics":
                [],

            "notes":
                (
                    "La question exige de produire une relation "
                    "entre deux sources distinctes. Aucun outil "
                    "de combinaison n'est disponible."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # 002
    #
    # DIFFERENT GRAINS + COMBINATION REQUIRED
    # ========================================================

    {
        "case_id":
            "router_md_v0_7_2_train_002",

        "split":
            "train",

        "domain":
            "hotel_marketing",

        "user_request":
            (
                "Les dépenses publicitaires mensuelles "
                "sont-elles associées au taux d'occupation "
                "mensuel des hôtels ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "marketing",

                "filename":
                    "marketing.csv",

                "grain":
                    "campaign_day",

                "entity_columns": [
                    "campaign_id",
                ],

                "columns": [
                    {
                        "name":
                            "campaign_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "campaign_date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "ad_spend",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },

            {
                "dataset_id":
                    "occupancy",

                "filename":
                    "occupancy.csv",

                "grain":
                    "hotel_month",

                "entity_columns": [
                    "hotel_id",
                ],

                "columns": [
                    {
                        "name":
                            "hotel_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "month",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "occupancy_rate",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },
        ],

        "available_tools":
            BASE_TOOLS,

        "expected": {
            "decision":
                "cannot_answer",

            "decision_reason":
                "unsupported_analysis",

            "clarification_topics":
                [],

            "notes":
                (
                    "L'analyse exige transformation de grain "
                    "et combinaison inter-datasets, mais aucun "
                    "outil de combinaison n'est disponible."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # 003
    #
    # TWO DATASETS PRESENT
    # BUT REQUEST ONLY NEEDS ONE
    # ========================================================

    {
        "case_id":
            "router_md_v0_7_2_train_003",

        "split":
            "train",

        "domain":
            "online_retail",

        "user_request":
            (
                "Quel est le chiffre d'affaires total ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "sales",

                "filename":
                    "sales.csv",

                "grain":
                    "order",

                "entity_columns": [
                    "order_id",
                ],

                "columns": [
                    {
                        "name":
                            "order_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "revenue",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },

            {
                "dataset_id":
                    "support",

                "filename":
                    "support.csv",

                "grain":
                    "ticket",

                "entity_columns": [
                    "ticket_id",
                ],

                "columns": [
                    {
                        "name":
                            "ticket_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "resolution_minutes",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },
        ],

        "available_tools":
            BASE_TOOLS,

        "expected": {
            "decision":
                "analyze",

            "decision_reason":
                None,

            "clarification_topics":
                [],

            "notes":
                (
                    "La présence d'un second dataset n'empêche "
                    "pas l'analyse : la demande peut être "
                    "satisfaite entièrement avec sales."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # 004
    #
    # CROSS-DATASET ANALYSIS
    # JOIN TOOL AVAILABLE
    # ========================================================

    {
        "case_id":
            "router_md_v0_7_2_train_004",

        "split":
            "train",

        "domain":
            "customer_commerce",

        "user_request":
            (
                "Le nombre de tickets support par client "
                "est-il associé à son chiffre d'affaires ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "support",

                "filename":
                    "support.csv",

                "grain":
                    "customer_month",

                "entity_columns": [
                    "customer_id",
                ],

                "columns": [
                    {
                        "name":
                            "customer_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "month",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "ticket_count",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },

            {
                "dataset_id":
                    "commerce",

                "filename":
                    "commerce.csv",

                "grain":
                    "customer_month",

                "entity_columns": [
                    "customer_id",
                ],

                "columns": [
                    {
                        "name":
                            "customer_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "month",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "revenue",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },
        ],

        "available_tools":
            TOOLS_WITH_JOIN,

        "expected": {
            "decision":
                "analyze",

            "decision_reason":
                None,

            "clarification_topics":
                [],

            "notes":
                (
                    "Les deux datasets disposent d'un grain "
                    "et de clés compatibles et une capacité "
                    "de jointure est explicitement disponible."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # 005
    #
    # BOTH DATASETS HAVE REQUIRED VARIABLES
    # BUT NO COMPATIBLE LINK EXISTS
    # ========================================================

    {
        "case_id":
            "router_md_v0_7_2_train_005",

        "split":
            "train",

        "domain":
            "manufacturing_hr",

        "user_request":
            (
                "Le taux d'absence des employés est-il "
                "associé au nombre de défauts produits ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "employees",

                "filename":
                    "employees.csv",

                "grain":
                    "employee_month",

                "entity_columns": [
                    "employee_id",
                ],

                "columns": [
                    {
                        "name":
                            "employee_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "month",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "absence_rate",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },

            {
                "dataset_id":
                    "production",

                "filename":
                    "production.csv",

                "grain":
                    "machine_day",

                "entity_columns": [
                    "machine_id",
                ],

                "columns": [
                    {
                        "name":
                            "machine_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "production_date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "defect_count",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },
        ],

        "available_tools":
            TOOLS_WITH_JOIN,

        "expected": {
            "decision":
                "cannot_answer",

            "decision_reason":
                "unsupported_analysis",

            "clarification_topics":
                [],

            "notes":
                (
                    "Un outil de jointure existe, mais aucun "
                    "lien ou grain compatible permettant "
                    "d'associer les employés aux machines "
                    "n'est fourni."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # 006
    #
    # MULTIPLE DATASETS
    # TWO INDEPENDENT ANALYSES ARE POSSIBLE
    # NO JOIN REQUIRED
    # ========================================================

    {
        "case_id":
            "router_md_v0_7_2_train_006",

        "split":
            "train",

        "domain":
            "regional_operations",

        "user_request":
            (
                "Donne le total des ventes et, séparément, "
                "le nombre total de tickets support."
            ),

        "datasets": [
            {
                "dataset_id":
                    "sales",

                "filename":
                    "sales.csv",

                "grain":
                    "sale",

                "entity_columns": [
                    "sale_id",
                ],

                "columns": [
                    {
                        "name":
                            "sale_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "revenue",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },

            {
                "dataset_id":
                    "support",

                "filename":
                    "support.csv",

                "grain":
                    "ticket",

                "entity_columns": [
                    "ticket_id",
                ],

                "columns": [
                    {
                        "name":
                            "ticket_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "ticket_count",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            },
        ],

        "available_tools":
            BASE_TOOLS,

        "expected": {
            "decision":
                "analyze",

            "decision_reason":
                None,

            "clarification_topics":
                [],

            "notes":
                (
                    "La demande contient deux analyses "
                    "indépendantes. Aucune combinaison des "
                    "datasets n'est nécessaire."
                ),
        },

        "frozen":
            False,
    },
]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    content = (
        "\n".join(
            json.dumps(
                case,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )

            for case
            in CASES
        )
        + "\n"
    )


    OUTPUT_PATH.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


    raw_bytes = (
        OUTPUT_PATH.read_bytes()
    )


    assert not raw_bytes.startswith(
        b"\xef\xbb\xbf"
    )


    print(
        "=== DATALENS MULTI-DATASET ROUTER TRAIN v0.7.2 ==="
    )

    print()

    print(
        "Benchmark:",
        BENCHMARK_VERSION,
    )

    print(
        "Cases:",
        len(
            CASES
        ),
    )

    print()


    analyze_count = sum(
        1
        for case
        in CASES
        if (
            case[
                "expected"
            ][
                "decision"
            ]
            == "analyze"
        )
    )


    cannot_count = sum(
        1
        for case
        in CASES
        if (
            case[
                "expected"
            ][
                "decision"
            ]
            == "cannot_answer"
        )
    )


    print(
        "Analyze:",
        analyze_count,
    )

    print(
        "Cannot answer:",
        cannot_count,
    )

    print(
        "Needs clarification:",
        0,
    )

    print()

    print(
        "Encoding: UTF-8 without BOM"
    )

    print(
        "Frozen: False"
    )

    print(
        "Split: train"
    )

    print()

    print(
        "Output:",
        OUTPUT_PATH,
    )

    print()

    print(
        "Multi-dataset router train benchmark: PASS"
    )


if __name__ == "__main__":
    main()
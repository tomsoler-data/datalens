from __future__ import annotations

import json

from pathlib import Path


BASE_DIR = Path(
    __file__,
).resolve().parents[2]


OUTPUT_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_reasoning_hard_v0_4.jsonl"
)


AVAILABLE_TOOLS = [
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


CASES = [
    # ========================================================
    # CASE 1
    #
    # Entity reasoning + grain change + multiple useful
    # behavioral metrics + distractor categorical variable.
    # ========================================================

    {
        "case_id":
            "hard_v0_4_001",

        "split":
            "validation",

        "domain":
            "retail_operations",

        "user_request":
            (
                "Certains magasins ont-ils un comportement "
                "inhabituel en tenant compte du volume de "
                "commandes, du chiffre d'affaires et du taux "
                "de retour ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "store_daily",

                "filename":
                    "store_daily.csv",

                "grain":
                    "store_day",

                "entity_columns": [
                    "store_id",
                ],

                "columns": [
                    {
                        "name":
                            "store_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "region",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "order_count",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "revenue",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "return_rate",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            AVAILABLE_TOOLS,

        "expected": {
            "intent":
                "entity_anomaly_analysis",

            "entity":
                "store_id",

            "current_grain":
                "store_day",

            "target_grain":
                "store",

            "relevant_columns": [
                "store_id",
                "order_count",
                "revenue",
                "return_rate",
            ],

            "family":
                "entity_outlier",

            "acceptable_tools": [
                "build_entity_view",
                "detect_entity_outliers",
            ],

            "required_tool_arguments": {
                "build_entity_view": {
                    "entity":
                        "store_id",
                },
                "detect_entity_outliers": {
                    "entity":
                        "store_id",

                    "metrics": [
                        "order_count",
                        "revenue",
                        "return_rate",
                    ],
                },
            },

            "forbidden_tools":
                [],

            "forbidden_assumptions": [
                "fraud",
                "poor_performance",
                "delete_outliers",
            ],

            "requires_reasoning":
                True,

            "notes":
                (
                    "Le modèle doit changer du grain "
                    "store_day au grain store et comparer "
                    "plusieurs dimensions comportementales."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # CASE 2
    #
    # Association with several plausible distractor columns.
    # ========================================================

    {
        "case_id":
            "hard_v0_4_002",

        "split":
            "validation",

        "domain":
            "telecom",

        "user_request":
            (
                "Le temps moyen d'attente est-il associé "
                "au taux d'abandon des appels ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "call_center_daily",

                "filename":
                    "call_center_daily.csv",

                "grain":
                    "call_center_day",

                "entity_columns": [
                    "center_id",
                ],

                "columns": [
                    {
                        "name":
                            "center_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "region",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "call_volume",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "avg_wait_seconds",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "abandon_rate",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            AVAILABLE_TOOLS,

        "expected": {
            "intent":
                "measure_relationship",

            "entity":
                None,

            "current_grain":
                "call_center_day",

            "target_grain":
                None,

            "relevant_columns": [
                "avg_wait_seconds",
                "abandon_rate",
            ],

            "family":
                "association",

            "acceptable_tools": [
                "measure_association",
            ],

            "required_tool_arguments": {
                "measure_association": {
                    "target":
                        "avg_wait_seconds",

                    "value":
                        "abandon_rate",
                },
            },

            "forbidden_tools":
                [],

            "forbidden_assumptions": [
                "causal_claim",
            ],

            "requires_reasoning":
                True,

            "notes":
                (
                    "call_volume, region et date sont des "
                    "distracteurs pour cette demande précise."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # CASE 3
    #
    # Group comparison with several other quantitative columns.
    # ========================================================

    {
        "case_id":
            "hard_v0_4_003",

        "split":
            "validation",

        "domain":
            "marketing",

        "user_request":
            (
                "Les taux de conversion diffèrent-ils "
                "selon le canal d'acquisition ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "campaign_daily",

                "filename":
                    "campaign_daily.csv",

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
                            "date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "channel",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "impressions",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "clicks",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "conversion_rate",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "spend",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            AVAILABLE_TOOLS,

        "expected": {
            "intent":
                "compare_groups",

            "entity":
                None,

            "current_grain":
                "campaign_day",

            "target_grain":
                None,

            "relevant_columns": [
                "channel",
                "conversion_rate",
            ],

            "family":
                "group_comparison",

            "acceptable_tools": [
                "compare_groups",
            ],

            "required_tool_arguments": {
                "compare_groups": {
                    "target":
                        "conversion_rate",

                    "group_by":
                        "channel",
                },
            },

            "forbidden_tools":
                [],

            "forbidden_assumptions": [
                "causal_claim",
            ],

            "requires_reasoning":
                True,

            "notes":
                (
                    "Le modèle ne doit pas ajouter spend, "
                    "clicks ou impressions sans nécessité."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # CASE 4
    #
    # Causal wording trap.
    #
    # Available observational data only support association.
    # ========================================================

    {
        "case_id":
            "hard_v0_4_004",

        "split":
            "validation",

        "domain":
            "human_resources",

        "user_request":
            (
                "La formation a-t-elle causé "
                "l'amélioration de la productivité ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "employees",

                "filename":
                    "employees.csv",

                "grain":
                    "employee",

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
                            "department",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "training_hours",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "productivity_score",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "tenure_months",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            AVAILABLE_TOOLS,

        "expected": {
            "intent":
                "measure_relationship",

            "entity":
                None,

            "current_grain":
                "employee",

            "target_grain":
                None,

            "relevant_columns": [
                "training_hours",
                "productivity_score",
            ],

            "family":
                "association",

            "acceptable_tools": [
                "measure_association",
            ],

            "required_tool_arguments": {
                "measure_association": {
                    "target":
                        "training_hours",

                    "value":
                        "productivity_score",
                },
            },

            "forbidden_tools":
                [],

            "forbidden_assumptions": [
                "causal_claim",
            ],

            "requires_reasoning":
                True,

            "notes":
                (
                    "La formulation utilisateur est causale, "
                    "mais les données décrites ne permettent "
                    "qu'une analyse d'association."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # CASE 5
    #
    # Another entity-grain problem, with explicit metric
    # selection and multiple distractors.
    # ========================================================

    {
        "case_id":
            "hard_v0_4_005",

        "split":
            "validation",

        "domain":
            "manufacturing",

        "user_request":
            (
                "Quelles machines ont un profil inhabituel "
                "en matière de vibration, de défauts et de "
                "volume produit ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "machine_shifts",

                "filename":
                    "machine_shifts.csv",

                "grain":
                    "machine_shift",

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
                            "shift_date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "operator_team",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "vibration_score",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "defect_rate",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "units_produced",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "energy_kwh",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            AVAILABLE_TOOLS,

        "expected": {
            "intent":
                "entity_anomaly_analysis",

            "entity":
                "machine_id",

            "current_grain":
                "machine_shift",

            "target_grain":
                "machine",

            "relevant_columns": [
                "machine_id",
                "vibration_score",
                "defect_rate",
                "units_produced",
            ],

            "family":
                "entity_outlier",

            "acceptable_tools": [
                "build_entity_view",
                "detect_entity_outliers",
            ],

            "required_tool_arguments": {
                "build_entity_view": {
                    "entity":
                        "machine_id",
                },
                "detect_entity_outliers": {
                    "entity":
                        "machine_id",

                    "metrics": [
                        "vibration_score",
                        "defect_rate",
                        "units_produced",
                    ],
                },
            },

            "forbidden_tools":
                [],

            "forbidden_assumptions": [
                "failure",
                "safety_failure",
                "delete_outliers",
            ],

            "requires_reasoning":
                True,

            "notes":
                (
                    "energy_kwh et operator_team ne sont "
                    "pas demandés."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # CASE 6
    #
    # Specific comparison despite several plausible grouping
    # and explanatory variables.
    # ========================================================

    {
        "case_id":
            "hard_v0_4_006",

        "split":
            "validation",

        "domain":
            "customer_support",

        "user_request":
            (
                "Les tickets prioritaires prennent-ils "
                "plus de temps à résoudre que les autres ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "support_tickets",

                "filename":
                    "support_tickets.csv",

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
                            "priority",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "channel",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "team",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "resolution_minutes",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "customer_tenure_months",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            AVAILABLE_TOOLS,

        "expected": {
            "intent":
                "compare_groups",

            "entity":
                None,

            "current_grain":
                "ticket",

            "target_grain":
                None,

            "relevant_columns": [
                "priority",
                "resolution_minutes",
            ],

            "family":
                "group_comparison",

            "acceptable_tools": [
                "compare_groups",
            ],

            "required_tool_arguments": {
                "compare_groups": {
                    "target":
                        "resolution_minutes",

                    "group_by":
                        "priority",
                },
            },

            "forbidden_tools":
                [],

            "forbidden_assumptions": [
                "causal_claim",
            ],

            "requires_reasoning":
                True,

            "notes":
                (
                    "channel, team et ancienneté client "
                    "sont volontairement disponibles "
                    "mais non demandés."
                ),
        },

        "frozen":
            False,
    },
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    lines = [
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
    ]


    content = (
        "\n".join(
            lines,
        )
        + "\n"
    )


    OUTPUT_PATH.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


    raw_bytes = (
        OUTPUT_PATH
        .read_bytes()
    )


    assert not raw_bytes.startswith(
        b"\xef\xbb\xbf"
    )


    print(
        "=== DATALENS HARD BENCHMARK v0.4 ==="
    )

    print(
        "Cases:",
        len(
            CASES,
        ),
    )

    print(
        "Encoding: UTF-8 without BOM"
    )

    print(
        "Output:",
        OUTPUT_PATH,
    )

    print()

    for case in CASES:
        print(
            "-",
            case[
                "case_id"
            ],
            "|",
            case[
                "domain"
            ],
            "|",
            case[
                "user_request"
            ],
        )


    print()

    print(
        "Hard benchmark generated: PASS"
    )


if __name__ == "__main__":
    main()
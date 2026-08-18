from __future__ import annotations

import json

from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parent


OUTPUT_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_decision_frozen_v0_6.jsonl"
)


# ============================================================
# TOOL CATALOG
# ============================================================

TOOLS = [
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


# ============================================================
# CASES
#
# IMPORTANT:
#
# These are TEST cases.
#
# They must remain frozen after models are evaluated.
# ============================================================

CASES = [

    # ========================================================
    # 001 — SIMPLE AGGREGATION
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_001",

        "split":
            "test",

        "domain":
            "ecommerce",

        "user_request":
            "Quel est le chiffre d'affaires total ?",

        "datasets": [
            {
                "dataset_id":
                    "orders",

                "filename":
                    "orders.csv",

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
                            "order_date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "revenue",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "country",

                        "analytical_type":
                            "categorical",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "analyze",

            "decision_reason":
                None,

            "clarification_topics":
                [],

            "analytical": {
                "intent":
                    "aggregate_metric",

                "entity":
                    None,

                "current_grain":
                    "order",

                "target_grain":
                    None,

                "relevant_columns": [
                    "revenue",
                ],

                "family":
                    "aggregation",

                "acceptable_tools": [
                    "aggregate",
                ],

                "required_tool_arguments": {
                    "aggregate": {
                        "metrics": [
                            "revenue",
                        ],

                        "group_by":
                            None,
                    },
                },

                "forbidden_tools":
                    [],

                "forbidden_assumptions":
                    [],

                "requires_reasoning":
                    False,

                "notes":
                    (
                        "Agrégation globale simple. "
                        "Aucun regroupement nécessaire."
                    ),
            },
        },

        "frozen":
            True,
    },


    # ========================================================
    # 002 — GROUP COMPARISON
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_002",

        "split":
            "test",

        "domain":
            "healthcare_operations",

        "user_request":
            (
                "Les temps d'attente diffèrent-ils "
                "selon le service hospitalier ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "visits",

                "filename":
                    "visits.csv",

                "grain":
                    "visit",

                "entity_columns": [
                    "visit_id",
                ],

                "columns": [
                    {
                        "name":
                            "visit_id",

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
                            "wait_minutes",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "patient_age",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "analyze",

            "decision_reason":
                None,

            "clarification_topics":
                [],

            "analytical": {
                "intent":
                    "compare_groups",

                "entity":
                    None,

                "current_grain":
                    "visit",

                "target_grain":
                    None,

                "relevant_columns": [
                    "department",
                    "wait_minutes",
                ],

                "family":
                    "group_comparison",

                "acceptable_tools": [
                    "compare_groups",
                ],

                "required_tool_arguments": {
                    "compare_groups": {
                        "target":
                            "wait_minutes",

                        "group_by":
                            "department",
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
                    "Comparaison de groupes.",
            },
        },

        "frozen":
            True,
    },


    # ========================================================
    # 003 — ASSOCIATION
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_003",

        "split":
            "test",

        "domain":
            "hospitality",

        "user_request":
            (
                "Le prix moyen d'une nuit est-il associé "
                "au taux d'occupation ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "hotel_daily",

                "filename":
                    "hotel_daily.csv",

                "grain":
                    "hotel_day",

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
                            "date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "average_daily_rate",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "occupancy_rate",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "city",

                        "analytical_type":
                            "categorical",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "analyze",

            "decision_reason":
                None,

            "clarification_topics":
                [],

            "analytical": {
                "intent":
                    "measure_relationship",

                "entity":
                    None,

                "current_grain":
                    "hotel_day",

                "target_grain":
                    None,

                "relevant_columns": [
                    "average_daily_rate",
                    "occupancy_rate",
                ],

                "family":
                    "association",

                "acceptable_tools": [
                    "measure_association",
                ],

                "required_tool_arguments": {
                    "measure_association": {
                        "target":
                            "average_daily_rate",

                        "value":
                            "occupancy_rate",
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
                    "Association, pas causalité.",
            },
        },

        "frozen":
            True,
    },


    # ========================================================
    # 004 — TIME SERIES
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_004",

        "split":
            "test",

        "domain":
            "mobility",

        "user_request":
            (
                "Comment le nombre de trajets évolue-t-il "
                "au fil des semaines ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "weekly_trips",

                "filename":
                    "weekly_trips.csv",

                "grain":
                    "station_week",

                "entity_columns": [
                    "station_id",
                ],

                "columns": [
                    {
                        "name":
                            "station_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "week_start",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "trip_count",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "station_type",

                        "analytical_type":
                            "categorical",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "analyze",

            "decision_reason":
                None,

            "clarification_topics":
                [],

            "analytical": {
                "intent":
                    "time_series_analysis",

                "entity":
                    None,

                "current_grain":
                    "station_week",

                "target_grain":
                    None,

                "relevant_columns": [
                    "week_start",
                    "trip_count",
                ],

                "family":
                    "time_series",

                "acceptable_tools": [
                    "analyze_time_series",
                ],

                "required_tool_arguments": {
                    "analyze_time_series": {
                        "date":
                            "week_start",

                        "target":
                            "trip_count",
                    },
                },

                "forbidden_tools":
                    [],

                "forbidden_assumptions":
                    [],

                "requires_reasoning":
                    False,

                "notes":
                    "Analyse temporelle directe.",
            },
        },

        "frozen":
            True,
    },


    # ========================================================
    # 005 — VARIABLE OUTLIERS
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_005",

        "split":
            "test",

        "domain":
            "industrial_iot",

        "user_request":
            (
                "La température moteur contient-elle "
                "des valeurs atypiques ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "engine_readings",

                "filename":
                    "engine_readings.csv",

                "grain":
                    "sensor_reading",

                "entity_columns": [
                    "engine_id",
                ],

                "columns": [
                    {
                        "name":
                            "engine_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "timestamp",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "engine_temp_c",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "rpm",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "analyze",

            "decision_reason":
                None,

            "clarification_topics":
                [],

            "analytical": {
                "intent":
                    "distribution_analysis",

                "entity":
                    None,

                "current_grain":
                    "sensor_reading",

                "target_grain":
                    None,

                "relevant_columns": [
                    "engine_temp_c",
                ],

                "family":
                    "distribution",

                "acceptable_tools": [
                    "detect_outliers",
                ],

                "required_tool_arguments": {
                    "detect_outliers": {
                        "target":
                            "engine_temp_c",
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
                        "Valeurs atypiques d'une variable, "
                        "pas anomalie d'entité."
                    ),
            },
        },

        "frozen":
            True,
    },


    # ========================================================
    # 006 — ENTITY OUTLIERS + GRAIN CHANGE
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_006",

        "split":
            "test",

        "domain":
            "insurance",

        "user_request":
            (
                "Certains courtiers ont-ils un comportement "
                "inhabituel en termes de nombre de contrats, "
                "de primes et de taux de sinistres ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "broker_monthly",

                "filename":
                    "broker_monthly.csv",

                "grain":
                    "broker_month",

                "entity_columns": [
                    "broker_id",
                ],

                "columns": [
                    {
                        "name":
                            "broker_id",

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
                            "contract_count",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "premium_amount",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "claim_rate",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "region",

                        "analytical_type":
                            "categorical",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "analyze",

            "decision_reason":
                None,

            "clarification_topics":
                [],

            "analytical": {
                "intent":
                    "entity_anomaly_analysis",

                "entity":
                    "broker_id",

                "current_grain":
                    "broker_month",

                "target_grain":
                    "broker",

                "relevant_columns": [
                    "broker_id",
                    "contract_count",
                    "premium_amount",
                    "claim_rate",
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
                            "broker_id",
                    },

                    "detect_entity_outliers": {
                        "entity":
                            "broker_id",

                        "metrics": [
                            "contract_count",
                            "premium_amount",
                            "claim_rate",
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
                        "Nécessite un changement de grain "
                        "broker_month vers broker."
                    ),
            },
        },

        "frozen":
            True,
    },


    # ========================================================
    # 007 — ANOTHER GROUP COMPARISON WITH DISTRACTORS
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_007",

        "split":
            "test",

        "domain":
            "logistics",

        "user_request":
            (
                "Les délais de livraison diffèrent-ils "
                "selon le transporteur ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "shipments",

                "filename":
                    "shipments.csv",

                "grain":
                    "shipment",

                "entity_columns": [
                    "shipment_id",
                ],

                "columns": [
                    {
                        "name":
                            "shipment_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "carrier",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "delivery_delay_days",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "shipping_cost",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "destination_region",

                        "analytical_type":
                            "categorical",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "analyze",

            "decision_reason":
                None,

            "clarification_topics":
                [],

            "analytical": {
                "intent":
                    "compare_groups",

                "entity":
                    None,

                "current_grain":
                    "shipment",

                "target_grain":
                    None,

                "relevant_columns": [
                    "carrier",
                    "delivery_delay_days",
                ],

                "family":
                    "group_comparison",

                "acceptable_tools": [
                    "compare_groups",
                ],

                "required_tool_arguments": {
                    "compare_groups": {
                        "target":
                            "delivery_delay_days",

                        "group_by":
                            "carrier",
                    },
                },

                "forbidden_tools":
                    [],

                "forbidden_assumptions":
                    [],

                "requires_reasoning":
                    True,

                "notes":
                    (
                        "shipping_cost et destination_region "
                        "sont des distracteurs."
                    ),
            },
        },

        "frozen":
            True,
    },


    # ========================================================
    # 008 — CAUSALITY: CANNOT ANSWER AS ASKED
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_008",

        "split":
            "test",

        "domain":
            "education",

        "user_request":
            (
                "Les heures de tutorat ont-elles causé "
                "l'amélioration des notes des étudiants ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "students",

                "filename":
                    "students.csv",

                "grain":
                    "student",

                "entity_columns": [
                    "student_id",
                ],

                "columns": [
                    {
                        "name":
                            "student_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "tutoring_hours",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "final_score",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "attendance_rate",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "cannot_answer",

            "decision_reason":
                "causal_identification_missing",

            "clarification_topics":
                [],

            "analytical":
                None,
        },

        "frozen":
            True,
    },


    # ========================================================
    # 009 — AMBIGUOUS PERFORMANCE
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_009",

        "split":
            "test",

        "domain":
            "retail",

        "user_request":
            "Quels produits performent le mieux ?",

        "datasets": [
            {
                "dataset_id":
                    "products_monthly",

                "filename":
                    "products_monthly.csv",

                "grain":
                    "product_month",

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
                            "units_sold",

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
                            "margin_rate",

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
            TOOLS,

        "expected": {
            "decision":
                "needs_clarification",

            "decision_reason":
                "ambiguous_request",

            "clarification_topics": [
                "performance_metric",
            ],

            "analytical":
                None,
        },

        "frozen":
            True,
    },


    # ========================================================
    # 010 — AMBIGUOUS COMPARISON
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_010",

        "split":
            "test",

        "domain":
            "human_resources",

        "user_request":
            "Compare les départements.",

        "datasets": [
            {
                "dataset_id":
                    "department_monthly",

                "filename":
                    "department_monthly.csv",

                "grain":
                    "department_month",

                "entity_columns": [
                    "department",
                ],

                "columns": [
                    {
                        "name":
                            "department",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "headcount",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "absence_rate",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "turnover_rate",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "satisfaction_score",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "needs_clarification",

            "decision_reason":
                "ambiguous_request",

            "clarification_topics": [
                "comparison_metric",
            ],

            "analytical":
                None,
        },

        "frozen":
            True,
    },


    # ========================================================
    # 011 — MISSING COLUMN
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_011",

        "split":
            "test",

        "domain":
            "commerce",

        "user_request":
            (
                "Quelle est la marge réalisée "
                "sur chaque commande ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "orders",

                "filename":
                    "orders.csv",

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
                    {
                        "name":
                            "quantity",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "cannot_answer",

            "decision_reason":
                "missing_column",

            "clarification_topics":
                [],

            "analytical":
                None,
        },

        "frozen":
            True,
    },


    # ========================================================
    # 012 — MISSING DATASET
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_012",

        "split":
            "test",

        "domain":
            "saas",

        "user_request":
            (
                "Le nombre de tickets support est-il lié "
                "au churn des clients ?"
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
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "cannot_answer",

            "decision_reason":
                "missing_dataset",

            "clarification_topics":
                [],

            "analytical":
                None,
        },

        "frozen":
            True,
    },


    # ========================================================
    # 013 — UNSUPPORTED FORECAST
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_013",

        "split":
            "test",

        "domain":
            "energy",

        "user_request":
            (
                "Prédis la consommation électrique "
                "du mois prochain."
            ),

        "datasets": [
            {
                "dataset_id":
                    "energy_daily",

                "filename":
                    "energy_daily.csv",

                "grain":
                    "day",

                "entity_columns":
                    [],

                "columns": [
                    {
                        "name":
                            "date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "consumption_kwh",

                        "analytical_type":
                            "quantitative",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "cannot_answer",

            "decision_reason":
                "unsupported_analysis",

            "clarification_topics":
                [],

            "analytical":
                None,
        },

        "frozen":
            True,
    },


    # ========================================================
    # 014 — MULTI-DATASET / JOIN REQUIRED
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_014",

        "split":
            "test",

        "domain":
            "marketing_finance",

        "user_request":
            (
                "Les dépenses publicitaires sont-elles "
                "associées au chiffre d'affaires quotidien ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "advertising",

                "filename":
                    "advertising.csv",

                "grain":
                    "day",

                "entity_columns":
                    [],

                "columns": [
                    {
                        "name":
                            "ad_date",

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
            TOOLS,

        "expected": {
            "decision":
                "cannot_answer",

            "decision_reason":
                "unsupported_analysis",

            "clarification_topics":
                [],

            "analytical":
                None,
        },

        "frozen":
            True,
    },


    # ========================================================
    # 015 — REFERENCE / THRESHOLD MISSING
    # ========================================================

    {
        "case_id":
            "frozen_v0_6_015",

        "split":
            "test",

        "domain":
            "operations",

        "user_request":
            "Les délais de traitement sont-ils élevés ?",

        "datasets": [
            {
                "dataset_id":
                    "operations",

                "filename":
                    "operations.csv",

                "grain":
                    "case",

                "entity_columns": [
                    "case_id",
                ],

                "columns": [
                    {
                        "name":
                            "case_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "processing_minutes",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "team",

                        "analytical_type":
                            "categorical",
                    },
                ],
            }
        ],

        "available_tools":
            TOOLS,

        "expected": {
            "decision":
                "needs_clarification",

            "decision_reason":
                "insufficient_context",

            "clarification_topics": [
                "reference_threshold",
            ],

            "analytical":
                None,
        },

        "frozen":
            True,
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


    decisions: dict[
        str,
        int
    ] = {}


    for case in CASES:
        decision = (
            case[
                "expected"
            ][
                "decision"
            ]
        )

        decisions[
            decision
        ] = (
            decisions.get(
                decision,
                0,
            )
            + 1
        )


    print(
        "=== DATALENS FROZEN TEST SET v0.6 ==="
    )

    print()

    print(
        "Cases:",
        len(
            CASES
        ),
    )

    print(
        "Analyze:",
        decisions.get(
            "analyze",
            0,
        ),
    )

    print(
        "Needs clarification:",
        decisions.get(
            "needs_clarification",
            0,
        ),
    )

    print(
        "Cannot answer:",
        decisions.get(
            "cannot_answer",
            0,
        ),
    )

    print()

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
                "expected"
            ][
                "decision"
            ],
        )


    print()

    print(
        "Frozen benchmark generated: PASS"
    )


if __name__ == "__main__":
    main()
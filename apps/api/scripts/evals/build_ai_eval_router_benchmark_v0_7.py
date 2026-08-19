from __future__ import annotations

import json

from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parents[2]


OUTPUT_PATH = (
    BASE_DIR
    / "evals"
    / "decision_router_development_v0_7.jsonl"
)


# ============================================================
# TOOL CATALOG
#
# Deliberately no forecasting tool and no join tool.
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
# 18 DEVELOPMENT cases:
#
# train:
#   3 analyze
#   3 needs_clarification
#   3 cannot_answer
#
# validation:
#   3 analyze
#   3 needs_clarification
#   3 cannot_answer
#
# None of these cases is frozen.
# ============================================================

CASES = [

    # ========================================================
    # TRAIN 001 — ANALYZE
    # Clear group comparison.
    # ========================================================

    {
        "case_id":
            "router_v0_7_train_001",

        "split":
            "train",

        "domain":
            "agriculture",

        "user_request":
            (
                "Le rendement des parcelles diffère-t-il "
                "selon le type d'irrigation ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "fields",

                "filename":
                    "fields.csv",

                "grain":
                    "field_season",

                "entity_columns": [
                    "field_id",
                ],

                "columns": [
                    {
                        "name":
                            "field_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "irrigation_type",

                        "analytical_type":
                            "categorical",
                    },
                    {
                        "name":
                            "yield_tons_per_hectare",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "rainfall_mm",

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

            "notes":
                (
                    "La variable cible et la variable "
                    "de groupe sont explicitement définies."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 002 — ANALYZE
    # Simple aggregation.
    # ========================================================

    {
        "case_id":
            "router_v0_7_train_002",

        "split":
            "train",

        "domain":
            "corporate_finance",

        "user_request":
            (
                "Quel est le montant total des dépenses ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "expenses",

                "filename":
                    "expenses.csv",

                "grain":
                    "expense",

                "entity_columns": [
                    "expense_id",
                ],

                "columns": [
                    {
                        "name":
                            "expense_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "expense_date",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "amount",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "department",

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

            "notes":
                "Agrégation directement exécutable.",
        },

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 003 — ANALYZE
    # Clear time series.
    # ========================================================

    {
        "case_id":
            "router_v0_7_train_003",

        "split":
            "train",

        "domain":
            "factory_operations",

        "user_request":
            (
                "Comment le temps d'arrêt des machines "
                "évolue-t-il chaque semaine ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "downtime",

                "filename":
                    "downtime.csv",

                "grain":
                    "factory_week",

                "entity_columns":
                    [],

                "columns": [
                    {
                        "name":
                            "week_start",

                        "analytical_type":
                            "temporal",
                    },
                    {
                        "name":
                            "downtime_hours",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "production_volume",

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

            "notes":
                "Analyse temporelle directement définie.",
        },

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 004 — CLARIFY
    # "Best" is undefined.
    # ========================================================

    {
        "case_id":
            "router_v0_7_train_004",

        "split":
            "train",

        "domain":
            "customer_success",

        "user_request":
            (
                "Quels comptes clients sont les meilleurs ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "accounts",

                "filename":
                    "accounts.csv",

                "grain":
                    "account_month",

                "entity_columns": [
                    "account_id",
                ],

                "columns": [
                    {
                        "name":
                            "account_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "monthly_revenue",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "renewal_rate",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "ticket_count",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "usage_rate",

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

            "notes":
                (
                    "Meilleur peut désigner revenu, "
                    "renouvellement, usage ou support."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 005 — CLARIFY
    # Threshold/reference is missing.
    # ========================================================

    {
        "case_id":
            "router_v0_7_train_005",

        "split":
            "train",

        "domain":
            "parcel_delivery",

        "user_request":
            (
                "Les coûts de livraison sont-ils élevés ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "deliveries",

                "filename":
                    "deliveries.csv",

                "grain":
                    "delivery",

                "entity_columns": [
                    "delivery_id",
                ],

                "columns": [
                    {
                        "name":
                            "delivery_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "shipping_cost",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "distance_km",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "carrier",

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

            "notes":
                (
                    "Le terme élevé nécessite une "
                    "référence ou un seuil."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 006 — CLARIFY
    # Comparison target is unspecified.
    # ========================================================

    {
        "case_id":
            "router_v0_7_train_006",

        "split":
            "train",

        "domain":
            "secondary_education",

        "user_request":
            "Compare les classes.",

        "datasets": [
            {
                "dataset_id":
                    "classes",

                "filename":
                    "classes.csv",

                "grain":
                    "class_term",

                "entity_columns": [
                    "class_id",
                ],

                "columns": [
                    {
                        "name":
                            "class_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "average_score",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "attendance_rate",

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
                            "student_count",

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

            "notes":
                (
                    "La dimension de comparaison "
                    "n'est pas définie."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 007 — CANNOT ANSWER
    # Margin requires cost information.
    # ========================================================

    {
        "case_id":
            "router_v0_7_train_007",

        "split":
            "train",

        "domain":
            "wholesale",

        "user_request":
            (
                "Quelle est la marge brute "
                "réalisée sur chaque vente ?"
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

            "notes":
                (
                    "Aucune information de coût "
                    "n'est disponible."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 008 — CANNOT ANSWER
    # Forecasting is unsupported.
    # ========================================================

    {
        "case_id":
            "router_v0_7_train_008",

        "split":
            "train",

        "domain":
            "telecommunications_capacity",

        "user_request":
            (
                "Prédis le trafic réseau de demain."
            ),

        "datasets": [
            {
                "dataset_id":
                    "network_daily",

                "filename":
                    "network_daily.csv",

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
                            "traffic_gb",

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

            "notes":
                (
                    "Le catalogue expose une analyse "
                    "temporelle mais aucun outil "
                    "de prévision."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 009 — CANNOT ANSWER
    # Missing churn source.
    # ========================================================

    {
        "case_id":
            "router_v0_7_train_009",

        "split":
            "train",

        "domain":
            "subscription_support",

        "user_request":
            (
                "Le nombre de demandes au support "
                "est-il associé au churn ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "support_activity",

                "filename":
                    "support_activity.csv",

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
                            "support_request_count",

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

            "notes":
                (
                    "Aucune information de churn "
                    "n'est fournie."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 001 — ANALYZE
    # Association.
    # ========================================================

    {
        "case_id":
            "router_v0_7_validation_001",

        "split":
            "validation",

        "domain":
            "real_estate",

        "user_request":
            (
                "La surface du logement est-elle "
                "associée à son prix de vente ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "properties",

                "filename":
                    "properties.csv",

                "grain":
                    "property_sale",

                "entity_columns": [
                    "property_id",
                ],

                "columns": [
                    {
                        "name":
                            "property_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "floor_area_m2",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "sale_price",

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

            "notes":
                "Association directement analysable.",
        },

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 002 — ANALYZE
    # Clear group comparison.
    # ========================================================

    {
        "case_id":
            "router_v0_7_validation_002",

        "split":
            "validation",

        "domain":
            "restaurant_operations",

        "user_request":
            (
                "Le temps d'attente diffère-t-il "
                "selon le jour de la semaine ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "visits",

                "filename":
                    "visits.csv",

                "grain":
                    "restaurant_visit",

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
                            "weekday",

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
                            "party_size",

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

            "notes":
                (
                    "La variable cible et le groupe "
                    "sont explicites."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 003 — ANALYZE
    # Variable-level outlier request.
    # ========================================================

    {
        "case_id":
            "router_v0_7_validation_003",

        "split":
            "validation",

        "domain":
            "battery_monitoring",

        "user_request":
            (
                "La tension de batterie contient-elle "
                "des valeurs atypiques ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "battery_readings",

                "filename":
                    "battery_readings.csv",

                "grain":
                    "reading",

                "entity_columns": [
                    "device_id",
                ],

                "columns": [
                    {
                        "name":
                            "device_id",

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
                            "battery_voltage",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "temperature_c",

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

            "notes":
                (
                    "La demande cible explicitement "
                    "une variable observée."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 004 — CLARIFY
    # Site performance metric undefined.
    # ========================================================

    {
        "case_id":
            "router_v0_7_validation_004",

        "split":
            "validation",

        "domain":
            "renewable_energy_operations",

        "user_request":
            (
                "Quels sites performent le mieux ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "sites",

                "filename":
                    "sites.csv",

                "grain":
                    "site_month",

                "entity_columns": [
                    "site_id",
                ],

                "columns": [
                    {
                        "name":
                            "site_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "energy_generated_mwh",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "operating_cost",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "downtime_rate",

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

            "notes":
                (
                    "Performance peut désigner production, "
                    "coût ou disponibilité."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 005 — CLARIFY
    # "Acceptable" requires business reference.
    # ========================================================

    {
        "case_id":
            "router_v0_7_validation_005",

        "split":
            "validation",

        "domain":
            "contact_center",

        "user_request":
            (
                "Le temps d'attente actuel est-il acceptable ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "calls",

                "filename":
                    "calls.csv",

                "grain":
                    "call",

                "entity_columns": [
                    "call_id",
                ],

                "columns": [
                    {
                        "name":
                            "call_id",

                        "analytical_type":
                            "identifier",
                    },
                    {
                        "name":
                            "wait_seconds",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "queue",

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

            "notes":
                (
                    "Acceptable dépend d'un SLA, "
                    "objectif ou benchmark absent."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 006 — CLARIFY
    # Compare teams on what?
    # ========================================================

    {
        "case_id":
            "router_v0_7_validation_006",

        "split":
            "validation",

        "domain":
            "workforce_management",

        "user_request":
            "Compare les équipes.",

        "datasets": [
            {
                "dataset_id":
                    "teams",

                "filename":
                    "teams.csv",

                "grain":
                    "team_month",

                "entity_columns": [
                    "team",
                ],

                "columns": [
                    {
                        "name":
                            "team",

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
                            "satisfaction_score",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "overtime_hours",

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

            "notes":
                (
                    "La demande ne précise pas "
                    "la mesure à comparer."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 007 — CANNOT ANSWER
    # Gross margin impossible without cost.
    # ========================================================

    {
        "case_id":
            "router_v0_7_validation_007",

        "split":
            "validation",

        "domain":
            "digital_marketplace",

        "user_request":
            (
                "Quelle est la marge brute "
                "de chaque produit ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "product_sales",

                "filename":
                    "product_sales.csv",

                "grain":
                    "product_sale",

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
                            "revenue",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "units_sold",

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

            "notes":
                (
                    "Le coût nécessaire au calcul "
                    "de la marge brute est absent."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 008 — CANNOT ANSWER
    # Explicit causal request.
    # ========================================================

    {
        "case_id":
            "router_v0_7_validation_008",

        "split":
            "validation",

        "domain":
            "urban_mobility",

        "user_request":
            (
                "La baisse du prix des trajets "
                "a-t-elle causé l'augmentation "
                "du nombre de déplacements ?"
            ),

        "datasets": [
            {
                "dataset_id":
                    "mobility_daily",

                "filename":
                    "mobility_daily.csv",

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
                            "average_trip_price",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "trip_count",

                        "analytical_type":
                            "quantitative",
                    },
                    {
                        "name":
                            "rainfall_mm",

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

            "notes":
                (
                    "Le contexte est observationnel "
                    "et aucun outil causal n'est exposé."
                ),
        },

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 009 — CANNOT ANSWER
    # Both datasets exist but no join/composition tool exists.
    # ========================================================

    {
        "case_id":
            "router_v0_7_validation_009",

        "split":
            "validation",

        "domain":
            "inventory_sales",

        "user_request":
            (
                "Les ruptures de stock sont-elles "
                "associées au chiffre d'affaires "
                "quotidien ?"
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
            TOOLS,

        "expected": {
            "decision":
                "cannot_answer",

            "decision_reason":
                "unsupported_analysis",

            "clarification_topics":
                [],

            "notes":
                (
                    "Les sources existent mais le catalogue "
                    "ne contient aucun outil permettant de "
                    "les combiner au grain approprié."
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
        OUTPUT_PATH
        .read_bytes()
    )


    assert not raw_bytes.startswith(
        b"\xef\xbb\xbf"
    )


    print(
        "=== DATALENS DECISION ROUTER BENCHMARK v0.7 ==="
    )

    print()

    print(
        "Cases:",
        len(
            CASES
        ),
    )


    for split in (
        "train",
        "validation",
    ):
        split_cases = [
            case
            for case
            in CASES
            if (
                case[
                    "split"
                ]
                == split
            )
        ]


        print()

        print(
            split.upper(),
            ":",
            len(
                split_cases
            ),
        )


        for decision in (
            "analyze",
            "needs_clarification",
            "cannot_answer",
        ):
            count = sum(
                1
                for case
                in split_cases
                if (
                    case[
                        "expected"
                    ][
                        "decision"
                    ]
                    == decision
                )
            )


            print(
                " ",
                decision,
                ":",
                count,
            )


    print()

    print(
        "Encoding: UTF-8 without BOM"
    )

    print(
        "Frozen: False"
    )

    print(
        "Output:",
        OUTPUT_PATH,
    )

    print()

    print(
        "Decision Router development benchmark: PASS"
    )


if __name__ == "__main__":
    main()
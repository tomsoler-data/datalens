from __future__ import annotations

import json

from pathlib import Path

from app.evals.analytical_planner_benchmark_v0_9 import (
    ANALYTICAL_PLANNER_BENCHMARK_VERSION,
    AnalyticalPlannerEvalCase,
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
    / "analytical_planner_development_v0_9.jsonl"
)


# ============================================================
# TOOLS
# ============================================================

ANALYTICAL_TOOLS = [
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
    *ANALYTICAL_TOOLS,
    "join_datasets",
]


# ============================================================
# COLUMN HELPERS
# ============================================================

def identifier(
    name: str,
) -> dict:

    return {
        "name":
            name,

        "analytical_type":
            "identifier",
    }


def quantitative(
    name: str,
) -> dict:

    return {
        "name":
            name,

        "analytical_type":
            "quantitative",
    }


def categorical(
    name: str,
) -> dict:

    return {
        "name":
            name,

        "analytical_type":
            "categorical",
    }


def temporal(
    name: str,
) -> dict:

    return {
        "name":
            name,

        "analytical_type":
            "temporal",
    }


# ============================================================
# DATASET HELPER
# ============================================================

def dataset(
    *,
    dataset_id: str,
    grain: str,
    entity_columns: list[str],
    columns: list[dict],
) -> dict:

    return {
        "dataset_id":
            dataset_id,

        "filename":
            f"{dataset_id}.csv",

        "grain":
            grain,

        "entity_columns":
            entity_columns,

        "columns":
            columns,
    }


# ============================================================
# CASES
#
# NON-FROZEN DEVELOPMENT MATERIAL
#
# 5 train
# 5 validation
#
# Validation is development validation, NOT unseen evaluation.
# ============================================================

CASES = [

    # ========================================================
    # TRAIN 001 — AGGREGATION
    # ========================================================

    {
        "case_id":
            "planner_v0_9_train_001",

        "split":
            "train",

        "domain":
            "ecommerce",

        "user_request":
            "Quel est le chiffre d'affaires total ?",

        "datasets": [
            dataset(
                dataset_id="sales",
                grain="order",
                entity_columns=[
                    "order_id",
                ],
                columns=[
                    identifier(
                        "order_id"
                    ),
                    quantitative(
                        "revenue"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            ANALYTICAL_TOOLS,

        "dependency_candidate": {
            "requirements": [
                {
                    "requirement_id":
                        "total_revenue",

                    "dataset_ids": [
                        "sales",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "total_revenue",

                    "intent":
                        "aggregate_metric",

                    "family":
                        "aggregation",

                    "target_grain":
                        "global",

                    "steps": [
                        {
                            "step_id":
                                "aggregate_revenue",

                            "action": {
                                "name":
                                    "aggregate",

                                "metrics": [
                                    "sales.revenue",
                                ],

                                "group_by":
                                    None,
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Simple global aggregation.",

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 002 — GROUP COMPARISON
    # ========================================================

    {
        "case_id":
            "planner_v0_9_train_002",

        "split":
            "train",

        "domain":
            "customer_support",

        "user_request":
            (
                "Compare le temps de résolution "
                "selon la priorité des tickets."
            ),

        "datasets": [
            dataset(
                dataset_id="tickets",
                grain="ticket",
                entity_columns=[
                    "ticket_id",
                ],
                columns=[
                    identifier(
                        "ticket_id"
                    ),
                    categorical(
                        "priority"
                    ),
                    quantitative(
                        "resolution_minutes"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            ANALYTICAL_TOOLS,

        "dependency_candidate": {
            "requirements": [
                {
                    "requirement_id":
                        "resolution_by_priority",

                    "dataset_ids": [
                        "tickets",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "resolution_by_priority",

                    "intent":
                        "compare_groups",

                    "family":
                        "group_comparison",

                    "target_grain":
                        "ticket",

                    "steps": [
                        {
                            "step_id":
                                "compare_resolution",

                            "action": {
                                "name":
                                    "compare_groups",

                                "target":
                                    "tickets.resolution_minutes",

                                "group_by":
                                    "tickets.priority",
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Categorical group comparison.",

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 003 — TIME SERIES
    # ========================================================

    {
        "case_id":
            "planner_v0_9_train_003",

        "split":
            "train",

        "domain":
            "energy_monitoring",

        "user_request":
            (
                "Comment la consommation électrique "
                "évolue-t-elle chaque jour ?"
            ),

        "datasets": [
            dataset(
                dataset_id="energy",
                grain="day",
                entity_columns=[],
                columns=[
                    temporal(
                        "date"
                    ),
                    quantitative(
                        "energy_kwh"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            ANALYTICAL_TOOLS,

        "dependency_candidate": {
            "requirements": [
                {
                    "requirement_id":
                        "daily_energy_trend",

                    "dataset_ids": [
                        "energy",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "daily_energy_trend",

                    "intent":
                        "time_series_analysis",

                    "family":
                        "time_series",

                    "target_grain":
                        "day",

                    "steps": [
                        {
                            "step_id":
                                "energy_trend",

                            "action": {
                                "name":
                                    "analyze_time_series",

                                "date":
                                    "energy.date",

                                "target":
                                    "energy.energy_kwh",
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Direct time-series request.",

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 004 — MULTI-HOP ASSOCIATION
    # ========================================================

    {
        "case_id":
            "planner_v0_9_train_004",

        "split":
            "train",

        "domain":
            "subscription_commerce",

        "user_request":
            (
                "Le nombre de tickets support est-il "
                "associé au chiffre d'affaires du client ?"
            ),

        "datasets": [

            dataset(
                dataset_id="customers",
                grain="customer",
                entity_columns=[
                    "customer_id",
                ],
                columns=[
                    identifier(
                        "customer_id"
                    ),
                    categorical(
                        "segment"
                    ),
                ],
            ),

            dataset(
                dataset_id="sales",
                grain="customer_month",
                entity_columns=[
                    "customer_id",
                ],
                columns=[
                    identifier(
                        "customer_id"
                    ),
                    temporal(
                        "month"
                    ),
                    quantitative(
                        "revenue"
                    ),
                ],
            ),

            dataset(
                dataset_id="support",
                grain="customer_month",
                entity_columns=[
                    "customer_id",
                ],
                columns=[
                    identifier(
                        "customer_id"
                    ),
                    temporal(
                        "month"
                    ),
                    quantitative(
                        "ticket_count"
                    ),
                ],
            ),
        ],

        "relationships": [

            {
                "relationship_id":
                    "customers_sales",

                "left_dataset_id":
                    "customers",

                "right_dataset_id":
                    "sales",

                "kind":
                    "join",

                "left_keys": [
                    "customer_id",
                ],

                "right_keys": [
                    "customer_id",
                ],

                "validated":
                    True,
            },

            {
                "relationship_id":
                    "customers_support",

                "left_dataset_id":
                    "customers",

                "right_dataset_id":
                    "support",

                "kind":
                    "join",

                "left_keys": [
                    "customer_id",
                ],

                "right_keys": [
                    "customer_id",
                ],

                "validated":
                    True,
            },
        ],

        "available_tools":
            TOOLS_WITH_JOIN,

        "dependency_candidate": {
            "requirements": [
                {
                    "requirement_id":
                        "support_revenue_association",

                    "dataset_ids": [
                        "sales",
                        "support",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "support_revenue_association",

                    "intent":
                        "measure_relationship",

                    "family":
                        "association",

                    "target_grain":
                        "customer_month",

                    "steps": [
                        {
                            "step_id":
                                "measure_support_revenue",

                            "action": {
                                "name":
                                    "measure_association",

                                "target":
                                    "support.ticket_count",

                                "value":
                                    "sales.revenue",
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            (
                "Planner sees customers only as a bridge. "
                "customers.segment must not be used."
            ),

        "frozen":
            False,
    },


    # ========================================================
    # TRAIN 005 — VARIABLE OUTLIERS
    # ========================================================

    {
        "case_id":
            "planner_v0_9_train_005",

        "split":
            "train",

        "domain":
            "battery_monitoring",

        "user_request":
            (
                "Détecte les valeurs atypiques "
                "de tension de batterie."
            ),

        "datasets": [
            dataset(
                dataset_id="battery_readings",
                grain="reading",
                entity_columns=[
                    "reading_id",
                ],
                columns=[
                    identifier(
                        "reading_id"
                    ),
                    quantitative(
                        "voltage"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            ANALYTICAL_TOOLS,

        "dependency_candidate": {
            "requirements": [
                {
                    "requirement_id":
                        "voltage_outliers",

                    "dataset_ids": [
                        "battery_readings",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "voltage_outliers",

                    "intent":
                        "distribution_analysis",

                    "family":
                        "distribution",

                    "target_grain":
                        "reading",

                    "steps": [
                        {
                            "step_id":
                                "detect_voltage_outliers",

                            "action": {
                                "name":
                                    "detect_outliers",

                                "target":
                                    "battery_readings.voltage",
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Variable-level outlier analysis.",

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 001 — ENTITY OUTLIERS
    # ========================================================

    {
        "case_id":
            "planner_v0_9_validation_001",

        "split":
            "validation",

        "domain":
            "marketplace",

        "user_request":
            (
                "Quels clients ont un comportement "
                "commercial inhabituel selon leur revenu "
                "et leur nombre de commandes ?"
            ),

        "datasets": [
            dataset(
                dataset_id="customer_activity",
                grain="customer_order",
                entity_columns=[
                    "customer_id",
                ],
                columns=[
                    identifier(
                        "customer_id"
                    ),
                    quantitative(
                        "revenue"
                    ),
                    quantitative(
                        "order_count"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            ANALYTICAL_TOOLS,

        "dependency_candidate": {
            "requirements": [
                {
                    "requirement_id":
                        "unusual_customers",

                    "dataset_ids": [
                        "customer_activity",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "unusual_customers",

                    "intent":
                        "entity_anomaly_analysis",

                    "family":
                        "entity_outlier",

                    "target_grain":
                        "customer",

                    "steps": [

                        {
                            "step_id":
                                "build_customer_view",

                            "action": {
                                "name":
                                    "build_entity_view",

                                "entity":
                                    "customer_activity.customer_id",
                            },
                        },

                        {
                            "step_id":
                                "detect_customer_outliers",

                            "action": {
                                "name":
                                    "detect_entity_outliers",

                                "entity":
                                    "customer_activity.customer_id",

                                "metrics": [
                                    "customer_activity.revenue",
                                    "customer_activity.order_count",
                                ],
                            },
                        },
                    ],
                }
            ],
        },

        "notes":
            "Entity-level anomaly planning.",

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 002 — DERIVED METRIC + GROUP COMPARISON
    # ========================================================

    {
        "case_id":
            "planner_v0_9_validation_002",

        "split":
            "validation",

        "domain":
            "digital_marketing",

        "user_request":
            (
                "Compare le taux de conversion "
                "entre les canaux marketing."
            ),

        "datasets": [
            dataset(
                dataset_id="marketing",
                grain="campaign_day",
                entity_columns=[
                    "campaign_id",
                ],
                columns=[
                    identifier(
                        "campaign_id"
                    ),
                    categorical(
                        "channel"
                    ),
                    quantitative(
                        "visits"
                    ),
                    quantitative(
                        "conversions"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            ANALYTICAL_TOOLS,

        "dependency_candidate": {
            "requirements": [
                {
                    "requirement_id":
                        "conversion_by_channel",

                    "dataset_ids": [
                        "marketing",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "conversion_by_channel",

                    "intent":
                        "compare_groups",

                    "family":
                        "group_comparison",

                    "target_grain":
                        "campaign_day",

                    "steps": [

                        {
                            "step_id":
                                "derive_conversion_rate",

                            "action": {
                                "name":
                                    "derive_metric",

                                "inputs": [
                                    "marketing.conversions",
                                    "marketing.visits",
                                ],

                                "output":
                                    "conversion_rate",

                                "formula":
                                    "conversions / visits",
                            },
                        },

                        {
                            "step_id":
                                "compare_conversion_rate",

                            "action": {
                                "name":
                                    "compare_groups",

                                "target":
                                    "conversion_rate",

                                "group_by":
                                    "marketing.channel",
                            },
                        },
                    ],
                }
            ],
        },

        "notes":
            (
                "Tests derived metric sequencing before "
                "group comparison."
            ),

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 003 — DIRECT CROSS-DATASET ASSOCIATION
    # ========================================================

    {
        "case_id":
            "planner_v0_9_validation_003",

        "split":
            "validation",

        "domain":
            "vehicle_operations",

        "user_request":
            (
                "La consommation de carburant est-elle "
                "associée au coût de maintenance "
                "du véhicule ?"
            ),

        "datasets": [

            dataset(
                dataset_id="fuel",
                grain="vehicle_month",
                entity_columns=[
                    "vehicle_id",
                ],
                columns=[
                    identifier(
                        "vehicle_id"
                    ),
                    temporal(
                        "month"
                    ),
                    quantitative(
                        "fuel_liters"
                    ),
                ],
            ),

            dataset(
                dataset_id="maintenance",
                grain="vehicle_month",
                entity_columns=[
                    "vehicle_id",
                ],
                columns=[
                    identifier(
                        "vehicle_id"
                    ),
                    temporal(
                        "month"
                    ),
                    quantitative(
                        "maintenance_cost"
                    ),
                ],
            ),
        ],

        "relationships": [
            {
                "relationship_id":
                    "fuel_maintenance_vehicle_month",

                "left_dataset_id":
                    "fuel",

                "right_dataset_id":
                    "maintenance",

                "kind":
                    "join",

                "left_keys": [
                    "vehicle_id",
                    "month",
                ],

                "right_keys": [
                    "vehicle_id",
                    "month",
                ],

                "validated":
                    True,
            }
        ],

        "available_tools":
            TOOLS_WITH_JOIN,

        "dependency_candidate": {
            "requirements": [
                {
                    "requirement_id":
                        "fuel_maintenance_association",

                    "dataset_ids": [
                        "fuel",
                        "maintenance",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "fuel_maintenance_association",

                    "intent":
                        "measure_relationship",

                    "family":
                        "association",

                    "target_grain":
                        "vehicle_month",

                    "steps": [
                        {
                            "step_id":
                                "measure_fuel_maintenance",

                            "action": {
                                "name":
                                    "measure_association",

                                "target":
                                    "fuel.fuel_liters",

                                "value":
                                    "maintenance.maintenance_cost",
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Direct validated cross-dataset association.",

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 004 — TWO INDEPENDENT REQUIREMENTS
    # ========================================================

    {
        "case_id":
            "planner_v0_9_validation_004",

        "split":
            "validation",

        "domain":
            "regional_operations",

        "user_request":
            (
                "Calcule séparément le chiffre d'affaires "
                "total et le nombre total de tickets support."
            ),

        "datasets": [

            dataset(
                dataset_id="sales",
                grain="order",
                entity_columns=[
                    "order_id",
                ],
                columns=[
                    identifier(
                        "order_id"
                    ),
                    quantitative(
                        "revenue"
                    ),
                ],
            ),

            dataset(
                dataset_id="support",
                grain="ticket",
                entity_columns=[
                    "ticket_id",
                ],
                columns=[
                    identifier(
                        "ticket_id"
                    ),
                    quantitative(
                        "ticket_count"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            ANALYTICAL_TOOLS,

        "dependency_candidate": {
            "requirements": [

                {
                    "requirement_id":
                        "sales_total",

                    "dataset_ids": [
                        "sales",
                    ],
                },

                {
                    "requirement_id":
                        "support_total",

                    "dataset_ids": [
                        "support",
                    ],
                },
            ],
        },

        "expected": {
            "plans": [

                {
                    "requirement_id":
                        "sales_total",

                    "intent":
                        "aggregate_metric",

                    "family":
                        "aggregation",

                    "target_grain":
                        "global",

                    "steps": [
                        {
                            "step_id":
                                "aggregate_sales",

                            "action": {
                                "name":
                                    "aggregate",

                                "metrics": [
                                    "sales.revenue",
                                ],

                                "group_by":
                                    None,
                            },
                        }
                    ],
                },


                {
                    "requirement_id":
                        "support_total",

                    "intent":
                        "aggregate_metric",

                    "family":
                        "aggregation",

                    "target_grain":
                        "global",

                    "steps": [
                        {
                            "step_id":
                                "aggregate_support",

                            "action": {
                                "name":
                                    "aggregate",

                                "metrics": [
                                    "support.ticket_count",
                                ],

                                "group_by":
                                    None,
                            },
                        }
                    ],
                },
            ],
        },

        "notes":
            (
                "Planner must preserve two independent "
                "requirements."
            ),

        "frozen":
            False,
    },


    # ========================================================
    # VALIDATION 005 — DISTRIBUTION
    # ========================================================

    {
        "case_id":
            "planner_v0_9_validation_005",

        "split":
            "validation",

        "domain":
            "parcel_delivery",

        "user_request":
            (
                "Analyse la distribution du délai "
                "de livraison."
            ),

        "datasets": [
            dataset(
                dataset_id="deliveries",
                grain="delivery",
                entity_columns=[
                    "delivery_id",
                ],
                columns=[
                    identifier(
                        "delivery_id"
                    ),
                    quantitative(
                        "delivery_days"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            ANALYTICAL_TOOLS,

        "dependency_candidate": {
            "requirements": [
                {
                    "requirement_id":
                        "delivery_distribution",

                    "dataset_ids": [
                        "deliveries",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "delivery_distribution",

                    "intent":
                        "distribution_analysis",

                    "family":
                        "distribution",

                    "target_grain":
                        "delivery",

                    "steps": [
                        {
                            "step_id":
                                "analyze_delivery_distribution",

                            "action": {
                                "name":
                                    "analyze_distribution",

                                "target":
                                    "deliveries.delivery_days",
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Pure distribution analysis.",

        "frozen":
            False,
    },
]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER DEVELOPMENT v0.9 ==="
    )

    print()


    print(
        "Benchmark:",
        ANALYTICAL_PLANNER_BENCHMARK_VERSION,
    )


    # ========================================================
    # VALIDATE EVERY CASE BEFORE WRITING
    # ========================================================

    validated_cases = [
        AnalyticalPlannerEvalCase
        .model_validate(
            case
        )

        for case
        in CASES
    ]


    assert (
        len(
            validated_cases
        )
        == 10
    )


    case_ids = [
        case.case_id

        for case
        in validated_cases
    ]


    assert (
        len(
            case_ids
        )
        == len(
            set(
                case_ids
            )
        )
    )


    train_cases = [
        case

        for case
        in validated_cases

        if (
            case.split
            == "train"
        )
    ]


    validation_cases = [
        case

        for case
        in validated_cases

        if (
            case.split
            == "validation"
        )
    ]


    assert (
        len(
            train_cases
        )
        == 5
    )


    assert (
        len(
            validation_cases
        )
        == 5
    )


    assert all(
        not case.frozen

        for case
        in validated_cases
    )


    # ========================================================
    # WRITE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    content = (
        "\n".join(
            json.dumps(
                case.model_dump(
                    mode="json",
                ),
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )

            for case
            in validated_cases
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


    assert not (
        raw_bytes.startswith(
            b"\xef\xbb\xbf"
        )
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "Cases:",
        len(
            validated_cases
        ),
    )


    print(
        "Train:",
        len(
            train_cases
        ),
    )


    print(
        "Validation:",
        len(
            validation_cases
        ),
    )


    print(
        "Frozen: False"
    )


    print(
        "Encoding: UTF-8 without BOM"
    )


    print()


    for case in validated_cases:

        families = [
            plan.family

            for plan
            in case.expected.plans
        ]


        print(
            "-",
            case.case_id,
            "|",
            case.split,
            "|",
            case.domain,
            "|",
            families,
        )


    print()

    print(
        "Output:",
        OUTPUT_PATH,
    )


    print()

    print(
        "Analytical Planner development benchmark v0.9: PASS"
    )


if __name__ == "__main__":
    main()
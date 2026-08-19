from __future__ import annotations

import hashlib
import json

from pathlib import Path

from app.evals.analytical_planner_frozen_benchmark_v1_0 import (
    ANALYTICAL_PLANNER_FROZEN_BENCHMARK_VERSION,
    FrozenAnalyticalPlannerEvalCase,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parents[2]


OUTPUT_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_planner_frozen_v1_0.jsonl"
)


HASH_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_planner_frozen_v1_0.sha256"
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
# FROZEN CASES
#
# IMPORTANT:
#
# These domains and requests are deliberately different from
# the planner development benchmark.
#
# Once this file has generated the JSONL successfully, the
# output benchmark must not be regenerated or edited.
# ============================================================

CASES = [

    # ========================================================
    # 001 — HOSPITAL OPERATIONS — GLOBAL AGGREGATION
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_001",

        "split":
            "test",

        "domain":
            "hospital_operations",

        "user_request":
            (
                "Quel est le nombre total "
                "d'admissions enregistrées ?"
            ),

        "datasets": [
            dataset(
                dataset_id="hospital_activity",
                grain="day",
                entity_columns=[],
                columns=[
                    temporal(
                        "date"
                    ),
                    quantitative(
                        "admission_count"
                    ),
                    quantitative(
                        "discharge_count"
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
                        "total_admissions",

                    "dataset_ids": [
                        "hospital_activity",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "total_admissions",

                    "intent":
                        "aggregate_metric",

                    "family":
                        "aggregation",

                    "target_grain":
                        "global",

                    "steps": [
                        {
                            "step_id":
                                "aggregate_admissions",

                            "action": {
                                "name":
                                    "aggregate",

                                "metrics": [
                                    (
                                        "hospital_activity"
                                        ".admission_count"
                                    ),
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
            "Frozen simple global aggregation.",

        "frozen":
            True,
    },


    # ========================================================
    # 002 — INSURANCE CLAIMS — GROUP COMPARISON
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_002",

        "split":
            "test",

        "domain":
            "insurance_claims",

        "user_request":
            (
                "Compare le montant des sinistres "
                "selon le type de réclamation."
            ),

        "datasets": [
            dataset(
                dataset_id="claims",
                grain="claim",
                entity_columns=[
                    "claim_id",
                ],
                columns=[
                    identifier(
                        "claim_id"
                    ),
                    categorical(
                        "claim_type"
                    ),
                    quantitative(
                        "claim_amount"
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
                        "claim_amount_by_type",

                    "dataset_ids": [
                        "claims",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "claim_amount_by_type",

                    "intent":
                        "compare_groups",

                    "family":
                        "group_comparison",

                    "target_grain":
                        "claim",

                    "steps": [
                        {
                            "step_id":
                                "compare_claim_amount",

                            "action": {
                                "name":
                                    "compare_groups",

                                "target":
                                    "claims.claim_amount",

                                "group_by":
                                    "claims.claim_type",
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Frozen quantitative-by-category comparison.",

        "frozen":
            True,
    },


    # ========================================================
    # 003 — INDUSTRIAL SENSORS — DISTRIBUTION
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_003",

        "split":
            "test",

        "domain":
            "industrial_sensors",

        "user_request":
            (
                "Analyse la distribution des mesures "
                "de vibration."
            ),

        "datasets": [
            dataset(
                dataset_id="sensor_readings",
                grain="reading",
                entity_columns=[
                    "reading_id",
                ],
                columns=[
                    identifier(
                        "reading_id"
                    ),
                    quantitative(
                        "vibration_mm_s"
                    ),
                    quantitative(
                        "temperature_c"
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
                        "vibration_distribution",

                    "dataset_ids": [
                        "sensor_readings",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "vibration_distribution",

                    "intent":
                        "distribution_analysis",

                    "family":
                        "distribution",

                    "target_grain":
                        "reading",

                    "steps": [
                        {
                            "step_id":
                                "analyze_vibration",

                            "action": {
                                "name":
                                    "analyze_distribution",

                                "target":
                                    (
                                        "sensor_readings"
                                        ".vibration_mm_s"
                                    ),
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Distribution, not outlier detection.",

        "frozen":
            True,
    },


    # ========================================================
    # 004 — PUBLIC TRANSPORT — TIME SERIES
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_004",

        "split":
            "test",

        "domain":
            "public_transport",

        "user_request":
            (
                "Comment le nombre de voyageurs "
                "évolue-t-il au fil des jours ?"
            ),

        "datasets": [
            dataset(
                dataset_id="ridership",
                grain="day",
                entity_columns=[],
                columns=[
                    temporal(
                        "service_date"
                    ),
                    quantitative(
                        "passenger_count"
                    ),
                    quantitative(
                        "cancelled_trip_count"
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
                        "daily_ridership_trend",

                    "dataset_ids": [
                        "ridership",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "daily_ridership_trend",

                    "intent":
                        "time_series_analysis",

                    "family":
                        "time_series",

                    "target_grain":
                        "day",

                    "steps": [
                        {
                            "step_id":
                                "analyze_ridership_trend",

                            "action": {
                                "name":
                                    "analyze_time_series",

                                "date":
                                    "ridership.service_date",

                                "target":
                                    "ridership.passenger_count",
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Frozen temporal analysis.",

        "frozen":
            True,
    },


    # ========================================================
    # 005 — TELECOM NETWORK — VARIABLE OUTLIERS
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_005",

        "split":
            "test",

        "domain":
            "telecom_network",

        "user_request":
            (
                "Repère les valeurs atypiques "
                "de latence réseau."
            ),

        "datasets": [
            dataset(
                dataset_id="network_measurements",
                grain="measurement",
                entity_columns=[
                    "measurement_id",
                ],
                columns=[
                    identifier(
                        "measurement_id"
                    ),
                    quantitative(
                        "packet_latency_ms"
                    ),
                    quantitative(
                        "packet_loss_pct"
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
                        "latency_outliers",

                    "dataset_ids": [
                        "network_measurements",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "latency_outliers",

                    "intent":
                        "distribution_analysis",

                    "family":
                        "distribution",

                    "target_grain":
                        "measurement",

                    "steps": [
                        {
                            "step_id":
                                "detect_latency_outliers",

                            "action": {
                                "name":
                                    "detect_outliers",

                                "target":
                                    (
                                        "network_measurements"
                                        ".packet_latency_ms"
                                    ),
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Explicit variable-level outlier request.",

        "frozen":
            True,
    },


    # ========================================================
    # 006 — HOSPITALITY — ASSOCIATION
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_006",

        "split":
            "test",

        "domain":
            "hospitality",

        "user_request":
            (
                "Le taux d'occupation est-il associé "
                "au tarif moyen des chambres ?"
            ),

        "datasets": [
            dataset(
                dataset_id="hotel_daily",
                grain="hotel_day",
                entity_columns=[
                    "hotel_id",
                ],
                columns=[
                    identifier(
                        "hotel_id"
                    ),
                    temporal(
                        "date"
                    ),
                    quantitative(
                        "occupancy_rate"
                    ),
                    quantitative(
                        "average_room_rate"
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
                        "occupancy_rate_association",

                    "dataset_ids": [
                        "hotel_daily",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "occupancy_rate_association",

                    "intent":
                        "measure_relationship",

                    "family":
                        "association",

                    "target_grain":
                        "hotel_day",

                    "steps": [
                        {
                            "step_id":
                                "measure_hotel_association",

                            "action": {
                                "name":
                                    "measure_association",

                                "target":
                                    (
                                        "hotel_daily"
                                        ".occupancy_rate"
                                    ),

                                "value":
                                    (
                                        "hotel_daily"
                                        ".average_room_rate"
                                    ),
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            "Same-dataset association.",

        "frozen":
            True,
    },


    # ========================================================
    # 007 — EDUCATION PLATFORM — ENTITY OUTLIERS
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_007",

        "split":
            "test",

        "domain":
            "education_platform",

        "user_request":
            (
                "Quels étudiants ont un comportement "
                "inhabituel selon leur temps d'étude "
                "et leur nombre de tentatives aux quiz ?"
            ),

        "datasets": [
            dataset(
                dataset_id="learning_activity",
                grain="student_session",
                entity_columns=[
                    "student_id",
                ],
                columns=[
                    identifier(
                        "student_id"
                    ),
                    quantitative(
                        "study_minutes"
                    ),
                    quantitative(
                        "quiz_attempts"
                    ),
                    quantitative(
                        "completion_pct"
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
                        "unusual_students",

                    "dataset_ids": [
                        "learning_activity",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "unusual_students",

                    "intent":
                        "entity_anomaly_analysis",

                    "family":
                        "entity_outlier",

                    "target_grain":
                        "student",

                    "steps": [

                        {
                            "step_id":
                                "build_student_view",

                            "action": {
                                "name":
                                    "build_entity_view",

                                "entity":
                                    (
                                        "learning_activity"
                                        ".student_id"
                                    ),
                            },
                        },

                        {
                            "step_id":
                                "detect_student_outliers",

                            "action": {
                                "name":
                                    "detect_entity_outliers",

                                "entity":
                                    (
                                        "learning_activity"
                                        ".student_id"
                                    ),

                                "metrics": [
                                    (
                                        "learning_activity"
                                        ".study_minutes"
                                    ),
                                    (
                                        "learning_activity"
                                        ".quiz_attempts"
                                    ),
                                ],
                            },
                        },
                    ],
                }
            ],
        },

        "notes":
            "Entity grain must change from session to student.",

        "frozen":
            True,
    },


    # ========================================================
    # 008 — ADVERTISING — DERIVED METRIC + COMPARISON
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_008",

        "split":
            "test",

        "domain":
            "advertising",

        "user_request":
            (
                "Compare le taux de clic "
                "entre les canaux publicitaires."
            ),

        "datasets": [
            dataset(
                dataset_id="ad_performance",
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
                        "impressions"
                    ),
                    quantitative(
                        "clicks"
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
                        "click_rate_by_channel",

                    "dataset_ids": [
                        "ad_performance",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "click_rate_by_channel",

                    "intent":
                        "compare_groups",

                    "family":
                        "group_comparison",

                    "target_grain":
                        "campaign_day",

                    "steps": [

                        {
                            "step_id":
                                "derive_click_rate",

                            "action": {
                                "name":
                                    "derive_metric",

                                "inputs": [
                                    "ad_performance.clicks",
                                    "ad_performance.impressions",
                                ],

                                "output":
                                    "click_rate",

                                "formula":
                                    "clicks / impressions",
                            },
                        },

                        {
                            "step_id":
                                "compare_click_rate",

                            "action": {
                                "name":
                                    "compare_groups",

                                "target":
                                    "click_rate",

                                "group_by":
                                    "ad_performance.channel",
                            },
                        },
                    ],
                }
            ],
        },

        "notes":
            "Derived metric must be defined before comparison.",

        "frozen":
            True,
    },


    # ========================================================
    # 009 — LOGISTICS — DIRECT CROSS-DATASET ASSOCIATION
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_009",

        "split":
            "test",

        "domain":
            "freight_logistics",

        "user_request":
            (
                "Le délai de livraison est-il associé "
                "au coût de manutention des expéditions ?"
            ),

        "datasets": [

            dataset(
                dataset_id="shipments",
                grain="shipment",
                entity_columns=[
                    "shipment_id",
                ],
                columns=[
                    identifier(
                        "shipment_id"
                    ),
                    quantitative(
                        "delivery_delay_hours"
                    ),
                    quantitative(
                        "distance_km"
                    ),
                ],
            ),

            dataset(
                dataset_id="handling",
                grain="shipment",
                entity_columns=[
                    "shipment_id",
                ],
                columns=[
                    identifier(
                        "shipment_id"
                    ),
                    quantitative(
                        "handling_cost"
                    ),
                ],
            ),
        ],

        "relationships": [
            {
                "relationship_id":
                    "shipments_handling",

                "left_dataset_id":
                    "shipments",

                "right_dataset_id":
                    "handling",

                "kind":
                    "join",

                "left_keys": [
                    "shipment_id",
                ],

                "right_keys": [
                    "shipment_id",
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
                        "delay_handling_association",

                    "dataset_ids": [
                        "shipments",
                        "handling",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "delay_handling_association",

                    "intent":
                        "measure_relationship",

                    "family":
                        "association",

                    "target_grain":
                        "shipment",

                    "steps": [
                        {
                            "step_id":
                                "measure_delay_handling",

                            "action": {
                                "name":
                                    "measure_association",

                                "target":
                                    (
                                        "shipments"
                                        ".delivery_delay_hours"
                                    ),

                                "value":
                                    "handling.handling_cost",
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            (
                "Validated direct relationship exists, "
                "but planner must not control join_datasets."
            ),

        "frozen":
            True,
    },


    # ========================================================
    # 010 — HEALTHCARE FINANCE — MULTI-HOP BRIDGE
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_010",

        "split":
            "test",

        "domain":
            "healthcare_finance",

        "user_request":
            (
                "La durée des consultations est-elle "
                "associée au coût des soins des patients ?"
            ),

        "datasets": [

            dataset(
                dataset_id="patients",
                grain="patient",
                entity_columns=[
                    "patient_id",
                ],
                columns=[
                    identifier(
                        "patient_id"
                    ),
                    categorical(
                        "age_group"
                    ),
                ],
            ),

            dataset(
                dataset_id="consultations",
                grain="patient_month",
                entity_columns=[
                    "patient_id",
                ],
                columns=[
                    identifier(
                        "patient_id"
                    ),
                    temporal(
                        "month"
                    ),
                    quantitative(
                        "consultation_minutes"
                    ),
                ],
            ),

            dataset(
                dataset_id="care_costs",
                grain="patient_month",
                entity_columns=[
                    "patient_id",
                ],
                columns=[
                    identifier(
                        "patient_id"
                    ),
                    temporal(
                        "month"
                    ),
                    quantitative(
                        "care_cost"
                    ),
                ],
            ),
        ],

        "relationships": [

            {
                "relationship_id":
                    "patients_consultations",

                "left_dataset_id":
                    "patients",

                "right_dataset_id":
                    "consultations",

                "kind":
                    "join",

                "left_keys": [
                    "patient_id",
                ],

                "right_keys": [
                    "patient_id",
                ],

                "validated":
                    True,
            },

            {
                "relationship_id":
                    "patients_care_costs",

                "left_dataset_id":
                    "patients",

                "right_dataset_id":
                    "care_costs",

                "kind":
                    "join",

                "left_keys": [
                    "patient_id",
                ],

                "right_keys": [
                    "patient_id",
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
                        "consultation_cost_association",

                    "dataset_ids": [
                        "consultations",
                        "care_costs",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "consultation_cost_association",

                    "intent":
                        "measure_relationship",

                    "family":
                        "association",

                    "target_grain":
                        "patient_month",

                    "steps": [
                        {
                            "step_id":
                                "measure_consultation_cost",

                            "action": {
                                "name":
                                    "measure_association",

                                "target":
                                    (
                                        "consultations"
                                        ".consultation_minutes"
                                    ),

                                "value":
                                    "care_costs.care_cost",
                            },
                        }
                    ],
                }
            ],
        },

        "notes":
            (
                "patients is a structural bridge only. "
                "patients.age_group must remain analytically "
                "hidden."
            ),

        "frozen":
            True,
    },


    # ========================================================
    # 011 — RETAIL OPERATIONS — TWO INDEPENDENT REQUIREMENTS
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_011",

        "split":
            "test",

        "domain":
            "retail_operations",

        "user_request":
            (
                "Calcule séparément le chiffre d'affaires "
                "total et le nombre total de retours."
            ),

        "datasets": [

            dataset(
                dataset_id="store_sales",
                grain="transaction",
                entity_columns=[
                    "transaction_id",
                ],
                columns=[
                    identifier(
                        "transaction_id"
                    ),
                    quantitative(
                        "revenue"
                    ),
                ],
            ),

            dataset(
                dataset_id="product_returns",
                grain="return",
                entity_columns=[
                    "return_id",
                ],
                columns=[
                    identifier(
                        "return_id"
                    ),
                    quantitative(
                        "return_count"
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
                        "store_revenue_total",

                    "dataset_ids": [
                        "store_sales",
                    ],
                },

                {
                    "requirement_id":
                        "returns_total",

                    "dataset_ids": [
                        "product_returns",
                    ],
                },
            ],
        },

        "expected": {
            "plans": [

                {
                    "requirement_id":
                        "store_revenue_total",

                    "intent":
                        "aggregate_metric",

                    "family":
                        "aggregation",

                    "target_grain":
                        "global",

                    "steps": [
                        {
                            "step_id":
                                "aggregate_store_revenue",

                            "action": {
                                "name":
                                    "aggregate",

                                "metrics": [
                                    "store_sales.revenue",
                                ],

                                "group_by":
                                    None,
                            },
                        }
                    ],
                },

                {
                    "requirement_id":
                        "returns_total",

                    "intent":
                        "aggregate_metric",

                    "family":
                        "aggregation",

                    "target_grain":
                        "global",

                    "steps": [
                        {
                            "step_id":
                                "aggregate_returns",

                            "action": {
                                "name":
                                    "aggregate",

                                "metrics": [
                                    (
                                        "product_returns"
                                        ".return_count"
                                    ),
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
                "Two independent requirements. "
                "No combination is necessary."
            ),

        "frozen":
            True,
    },


    # ========================================================
    # 012 — MANUFACTURING — ENTITY GRAIN TRANSFORMATION
    # ========================================================

    {
        "case_id":
            "planner_frozen_v1_0_012",

        "split":
            "test",

        "domain":
            "factory_maintenance",

        "user_request":
            (
                "Quelles machines ont un comportement "
                "inhabituel selon leur nombre de défauts "
                "et leur temps d'arrêt ?"
            ),

        "datasets": [
            dataset(
                dataset_id="machine_activity",
                grain="machine_shift",
                entity_columns=[
                    "machine_id",
                ],
                columns=[
                    identifier(
                        "machine_id"
                    ),
                    quantitative(
                        "defect_count"
                    ),
                    quantitative(
                        "downtime_minutes"
                    ),
                    quantitative(
                        "units_produced"
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
                        "unusual_machines",

                    "dataset_ids": [
                        "machine_activity",
                    ],
                }
            ],
        },

        "expected": {
            "plans": [
                {
                    "requirement_id":
                        "unusual_machines",

                    "intent":
                        "entity_anomaly_analysis",

                    "family":
                        "entity_outlier",

                    "target_grain":
                        "machine",

                    "steps": [

                        {
                            "step_id":
                                "build_machine_view",

                            "action": {
                                "name":
                                    "build_entity_view",

                                "entity":
                                    (
                                        "machine_activity"
                                        ".machine_id"
                                    ),
                            },
                        },

                        {
                            "step_id":
                                "detect_machine_outliers",

                            "action": {
                                "name":
                                    "detect_entity_outliers",

                                "entity":
                                    (
                                        "machine_activity"
                                        ".machine_id"
                                    ),

                                "metrics": [
                                    (
                                        "machine_activity"
                                        ".defect_count"
                                    ),
                                    (
                                        "machine_activity"
                                        ".downtime_minutes"
                                    ),
                                ],
                            },
                        },
                    ],
                }
            ],
        },

        "notes":
            (
                "Entity plan must move from machine_shift "
                "to machine grain."
            ),

        "frozen":
            True,
    },
]


# ============================================================
# SHA-256
# ============================================================

def sha256_bytes(
    content: bytes,
) -> str:

    return (
        hashlib
        .sha256(
            content
        )
        .hexdigest()
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER FROZEN BENCHMARK v1.0 ==="
    )


    print()


    print(
        "Benchmark:",
        ANALYTICAL_PLANNER_FROZEN_BENCHMARK_VERSION,
    )


    print()


    # ========================================================
    # NEVER OVERWRITE FROZEN ARTIFACTS
    # ========================================================

    if (
        OUTPUT_PATH.exists()
        or HASH_PATH.exists()
    ):

        raise FileExistsError(
            "Frozen planner benchmark or lock already exists. "
            "Refusing to regenerate or overwrite it.\n"
            f"Benchmark: {OUTPUT_PATH}\n"
            f"SHA-256: {HASH_PATH}"
        )


    # ========================================================
    # VALIDATE ALL CASES BEFORE WRITING
    # ========================================================

    validated_cases = [
        FrozenAnalyticalPlannerEvalCase
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
        == 12
    )


    # ========================================================
    # UNIQUE IDS
    # ========================================================

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


    # ========================================================
    # UNIQUE DOMAINS
    # ========================================================

    domains = [
        case.domain

        for case
        in validated_cases
    ]


    assert (
        len(
            domains
        )
        == len(
            set(
                domains
            )
        )
    )


    assert (
        len(
            domains
        )
        == 12
    )


    # ========================================================
    # FROZEN CONTRACT
    # ========================================================

    assert all(
        case.split
        == "test"

        for case
        in validated_cases
    )


    assert all(
        case.frozen

        for case
        in validated_cases
    )


    # ========================================================
    # WRITE JSONL
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


    encoded = (
        content.encode(
            "utf-8"
        )
    )


    # ========================================================
    # UTF-8 WITHOUT BOM
    # ========================================================

    assert not (
        encoded.startswith(
            b"\xef\xbb\xbf"
        )
    )


    OUTPUT_PATH.write_bytes(
        encoded
    )


    # ========================================================
    # LOCK HASH
    # ========================================================

    digest = (
        sha256_bytes(
            encoded
        )
    )


    HASH_PATH.write_text(
        (
            digest
            + "\n"
        ),
        encoding="ascii",
        newline="\n",
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
        "Domains:",
        len(
            domains
        ),
    )


    print(
        "Split: test"
    )


    print(
        "Frozen: True"
    )


    print(
        "Encoding: UTF-8 without BOM"
    )


    print(
        "Ground truth validated by Python: PASS"
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
            case.domain,
            "|",
            families,
            "| requirements:",
            len(
                case.expected.plans
            ),
        )


    print()


    print(
        "Benchmark:",
        OUTPUT_PATH,
    )


    print(
        "SHA-256:",
        digest,
    )


    print(
        "Lock:",
        HASH_PATH,
    )


    print()


    print(
        "IMPORTANT:"
    )


    print(
        "Frozen benchmark created and locked."
    )


    print(
        "Do not edit, regenerate, or overwrite this JSONL."
    )


    print()


    print(
        "Analytical Planner Frozen Benchmark v1.0: LOCKED"
    )


if __name__ == "__main__":
    main()
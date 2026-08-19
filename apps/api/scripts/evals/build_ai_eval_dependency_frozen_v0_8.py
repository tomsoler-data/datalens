from __future__ import annotations

import json

from pathlib import Path

from app.evals.dataset_dependency_benchmark_v0_8 import (
    DATASET_DEPENDENCY_FROZEN_BENCHMARK_VERSION,
    DatasetDependencyFrozenCase,
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
    / "dataset_dependency_frozen_v0_8.jsonl"
)


# ============================================================
# TOOLS
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
# IMPORTANT:
#
# These cases are authored before any model inference.
#
# Once generated and validated, this file is considered frozen
# evaluation material for the v0.8 dependency pipeline.
# ============================================================

CASES = [

    # ========================================================
    # 001 — SINGLE DATASET + DISTRACTOR
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_001",

        "split":
            "test",

        "domain":
            "pharmacy_operations",

        "user_request":
            (
                "Quel est le nombre moyen de prescriptions "
                "traitées par jour ?"
            ),

        "datasets": [
            dataset(
                dataset_id="prescriptions",
                grain="day",
                entity_columns=[],
                columns=[
                    temporal(
                        "date"
                    ),
                    quantitative(
                        "prescription_count"
                    ),
                ],
            ),

            dataset(
                dataset_id="staffing",
                grain="employee_shift",
                entity_columns=[
                    "employee_id",
                ],
                columns=[
                    identifier(
                        "employee_id"
                    ),
                    quantitative(
                        "shift_hours"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            BASE_TOOLS,

        "expected": {
            "expected_groups": [
                [
                    "prescriptions",
                ]
            ],

            "expected_feasibilities": [
                "not_required",
            ],

            "executable":
                True,

            "routing_override_reason":
                None,

            "notes":
                (
                    "staffing is irrelevant to the "
                    "requested mean."
                ),
        },

        "frozen":
            True,
    },


    # ========================================================
    # 002 — SINGLE DATASET + MULTIPLE DISTRACTORS
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_002",

        "split":
            "test",

        "domain":
            "streaming_media",

        "user_request":
            (
                "Comment le nombre quotidien de lectures "
                "évolue-t-il dans le temps ?"
            ),

        "datasets": [
            dataset(
                dataset_id="plays",
                grain="day",
                entity_columns=[],
                columns=[
                    temporal(
                        "date"
                    ),
                    quantitative(
                        "play_count"
                    ),
                ],
            ),

            dataset(
                dataset_id="subscriptions",
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
                        "subscription_amount"
                    ),
                ],
            ),

            dataset(
                dataset_id="catalog",
                grain="title",
                entity_columns=[
                    "title_id",
                ],
                columns=[
                    identifier(
                        "title_id"
                    ),
                    categorical(
                        "genre"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            BASE_TOOLS,

        "expected": {
            "expected_groups": [
                [
                    "plays",
                ]
            ],

            "expected_feasibilities": [
                "not_required",
            ],

            "executable":
                True,

            "routing_override_reason":
                None,

            "notes":
                "Only plays is required.",
        },

        "frozen":
            True,
    },


    # ========================================================
    # 003 — INDEPENDENT ANALYSES
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_003",

        "split":
            "test",

        "domain":
            "airport_operations",

        "user_request":
            (
                "Calcule séparément le retard moyen des vols "
                "et le nombre total de réclamations bagages."
            ),

        "datasets": [
            dataset(
                dataset_id="flights",
                grain="flight",
                entity_columns=[
                    "flight_id",
                ],
                columns=[
                    identifier(
                        "flight_id"
                    ),
                    quantitative(
                        "delay_minutes"
                    ),
                ],
            ),

            dataset(
                dataset_id="baggage_claims",
                grain="claim",
                entity_columns=[
                    "claim_id",
                ],
                columns=[
                    identifier(
                        "claim_id"
                    ),
                    quantitative(
                        "claim_count"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            BASE_TOOLS,

        "expected": {
            "expected_groups": [
                [
                    "flights",
                ],
                [
                    "baggage_claims",
                ],
            ],

            "expected_feasibilities": [
                "not_required",
                "not_required",
            ],

            "executable":
                True,

            "routing_override_reason":
                None,

            "notes":
                "Two independent results.",
        },

        "frozen":
            True,
    },


    # ========================================================
    # 004 — INDEPENDENT ANALYSES WITH THIRD DISTRACTOR
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_004",

        "split":
            "test",

        "domain":
            "fitness_platform",

        "user_request":
            (
                "Donne séparément la durée moyenne des séances "
                "et le nombre total de demandes au support."
            ),

        "datasets": [
            dataset(
                dataset_id="workouts",
                grain="workout",
                entity_columns=[
                    "workout_id",
                ],
                columns=[
                    identifier(
                        "workout_id"
                    ),
                    quantitative(
                        "duration_minutes"
                    ),
                ],
            ),

            dataset(
                dataset_id="support_requests",
                grain="request",
                entity_columns=[
                    "request_id",
                ],
                columns=[
                    identifier(
                        "request_id"
                    ),
                    quantitative(
                        "request_count"
                    ),
                ],
            ),

            dataset(
                dataset_id="payments",
                grain="payment",
                entity_columns=[
                    "payment_id",
                ],
                columns=[
                    identifier(
                        "payment_id"
                    ),
                    quantitative(
                        "amount"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            BASE_TOOLS,

        "expected": {
            "expected_groups": [
                [
                    "workouts",
                ],
                [
                    "support_requests",
                ],
            ],

            "expected_feasibilities": [
                "not_required",
                "not_required",
            ],

            "executable":
                True,

            "routing_override_reason":
                None,

            "notes":
                "payments is irrelevant.",
        },

        "frozen":
            True,
    },


    # ========================================================
    # 005 — DIRECT VALID RELATIONSHIP
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_005",

        "split":
            "test",

        "domain":
            "university_services",

        "user_request":
            (
                "Le nombre de rendez-vous avec un conseiller "
                "est-il associé à la note finale des étudiants ?"
            ),

        "datasets": [
            dataset(
                dataset_id="advising",
                grain="student_term",
                entity_columns=[
                    "student_id",
                ],
                columns=[
                    identifier(
                        "student_id"
                    ),
                    temporal(
                        "term"
                    ),
                    quantitative(
                        "advising_appointments"
                    ),
                ],
            ),

            dataset(
                dataset_id="academic_results",
                grain="student_term",
                entity_columns=[
                    "student_id",
                ],
                columns=[
                    identifier(
                        "student_id"
                    ),
                    temporal(
                        "term"
                    ),
                    quantitative(
                        "final_score"
                    ),
                ],
            ),
        ],

        "relationships": [
            {
                "relationship_id":
                    "advising_results_student_term",

                "left_dataset_id":
                    "advising",

                "right_dataset_id":
                    "academic_results",

                "kind":
                    "join",

                "left_keys": [
                    "student_id",
                    "term",
                ],

                "right_keys": [
                    "student_id",
                    "term",
                ],

                "validated":
                    True,
            }
        ],

        "available_tools":
            TOOLS_WITH_JOIN,

        "expected": {
            "expected_groups": [
                [
                    "advising",
                    "academic_results",
                ]
            ],

            "expected_feasibilities": [
                "supported",
            ],

            "executable":
                True,

            "routing_override_reason":
                None,

            "notes":
                "Direct validated relationship.",
        },

        "frozen":
            True,
    },


    # ========================================================
    # 006 — DIRECT VALID RELATIONSHIP
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_006",

        "split":
            "test",

        "domain":
            "vehicle_fleet",

        "user_request":
            (
                "La consommation de carburant d'un véhicule "
                "est-elle associée à son coût de maintenance ?"
            ),

        "datasets": [
            dataset(
                dataset_id="fuel_usage",
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
                    "fuel_usage",

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

        "expected": {
            "expected_groups": [
                [
                    "fuel_usage",
                    "maintenance",
                ]
            ],

            "expected_feasibilities": [
                "supported",
            ],

            "executable":
                True,

            "routing_override_reason":
                None,

            "notes":
                "Direct validated vehicle-month relationship.",
        },

        "frozen":
            True,
    },


    # ========================================================
    # 007 — REQUIRED TOGETHER, JOIN TOOL ABSENT
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_007",

        "split":
            "test",

        "domain":
            "museum_operations",

        "user_request":
            (
                "La fréquentation quotidienne est-elle "
                "associée aux dépenses publicitaires "
                "quotidiennes ?"
            ),

        "datasets": [
            dataset(
                dataset_id="visits",
                grain="day",
                entity_columns=[],
                columns=[
                    temporal(
                        "date"
                    ),
                    quantitative(
                        "visitor_count"
                    ),
                ],
            ),

            dataset(
                dataset_id="advertising",
                grain="day",
                entity_columns=[],
                columns=[
                    temporal(
                        "date"
                    ),
                    quantitative(
                        "ad_spend"
                    ),
                ],
            ),
        ],

        "relationships": [
            {
                "relationship_id":
                    "visits_advertising_date",

                "left_dataset_id":
                    "visits",

                "right_dataset_id":
                    "advertising",

                "kind":
                    "temporal_alignment",

                "left_keys": [
                    "date",
                ],

                "right_keys": [
                    "date",
                ],

                "validated":
                    True,
            }
        ],

        "available_tools":
            BASE_TOOLS,

        "expected": {
            "expected_groups": [
                [
                    "visits",
                    "advertising",
                ]
            ],

            "expected_feasibilities": [
                "missing_combination_capability",
            ],

            "executable":
                False,

            "routing_override_reason":
                "unsupported_analysis",

            "notes":
                (
                    "Relationship exists, but no combination "
                    "capability is exposed."
                ),
        },

        "frozen":
            True,
    },


    # ========================================================
    # 008 — REQUIRED TOGETHER, JOIN TOOL ABSENT
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_008",

        "split":
            "test",

        "domain":
            "food_delivery",

        "user_request":
            (
                "Le délai de préparation des commandes "
                "est-il associé à la note laissée par "
                "les clients ?"
            ),

        "datasets": [
            dataset(
                dataset_id="orders",
                grain="order",
                entity_columns=[
                    "order_id",
                ],
                columns=[
                    identifier(
                        "order_id"
                    ),
                    quantitative(
                        "prep_minutes"
                    ),
                ],
            ),

            dataset(
                dataset_id="ratings",
                grain="order",
                entity_columns=[
                    "order_id",
                ],
                columns=[
                    identifier(
                        "order_id"
                    ),
                    quantitative(
                        "rating"
                    ),
                ],
            ),
        ],

        "relationships": [
            {
                "relationship_id":
                    "orders_ratings_order",

                "left_dataset_id":
                    "orders",

                "right_dataset_id":
                    "ratings",

                "kind":
                    "join",

                "left_keys": [
                    "order_id",
                ],

                "right_keys": [
                    "order_id",
                ],

                "validated":
                    True,
            }
        ],

        "available_tools":
            BASE_TOOLS,

        "expected": {
            "expected_groups": [
                [
                    "orders",
                    "ratings",
                ]
            ],

            "expected_feasibilities": [
                "missing_combination_capability",
            ],

            "executable":
                False,

            "routing_override_reason":
                "unsupported_analysis",

            "notes":
                "No join capability is available.",
        },

        "frozen":
            True,
    },


    # ========================================================
    # 009 — JOIN TOOL EXISTS, RELATIONSHIP MISSING
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_009",

        "split":
            "test",

        "domain":
            "construction_safety",

        "user_request":
            (
                "Le nombre d'heures de formation des ouvriers "
                "est-il associé au nombre d'incidents "
                "sur les chantiers ?"
            ),

        "datasets": [
            dataset(
                dataset_id="training",
                grain="worker_month",
                entity_columns=[
                    "worker_id",
                ],
                columns=[
                    identifier(
                        "worker_id"
                    ),
                    temporal(
                        "month"
                    ),
                    quantitative(
                        "training_hours"
                    ),
                ],
            ),

            dataset(
                dataset_id="site_incidents",
                grain="site_day",
                entity_columns=[
                    "site_id",
                ],
                columns=[
                    identifier(
                        "site_id"
                    ),
                    temporal(
                        "date"
                    ),
                    quantitative(
                        "incident_count"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            TOOLS_WITH_JOIN,

        "expected": {
            "expected_groups": [
                [
                    "training",
                    "site_incidents",
                ]
            ],

            "expected_feasibilities": [
                "missing_validated_relationship",
            ],

            "executable":
                False,

            "routing_override_reason":
                "unsupported_analysis",

            "notes":
                (
                    "Both semantic sources are required, "
                    "but no worker-to-site relationship exists."
                ),
        },

        "frozen":
            True,
    },


    # ========================================================
    # 010 — JOIN TOOL EXISTS, RELATIONSHIP MISSING
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_010",

        "split":
            "test",

        "domain":
            "public_library",

        "user_request":
            (
                "La participation aux événements est-elle "
                "associée au nombre de livres empruntés "
                "par les adhérents ?"
            ),

        "datasets": [
            dataset(
                dataset_id="events",
                grain="event_attendance",
                entity_columns=[
                    "event_id",
                ],
                columns=[
                    identifier(
                        "event_id"
                    ),
                    quantitative(
                        "attendance_count"
                    ),
                ],
            ),

            dataset(
                dataset_id="loans",
                grain="member_month",
                entity_columns=[
                    "member_id",
                ],
                columns=[
                    identifier(
                        "member_id"
                    ),
                    temporal(
                        "month"
                    ),
                    quantitative(
                        "loan_count"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            TOOLS_WITH_JOIN,

        "expected": {
            "expected_groups": [
                [
                    "events",
                    "loans",
                ]
            ],

            "expected_feasibilities": [
                "missing_validated_relationship",
            ],

            "executable":
                False,

            "routing_override_reason":
                "unsupported_analysis",

            "notes":
                "No member-event mapping has been validated.",
        },

        "frozen":
            True,
    },


    # ========================================================
    # 011 — THREE DATASETS, MULTI-HOP SUPPORTED
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_011",

        "split":
            "test",

        "domain":
            "telemedicine",

        "user_request":
            (
                "Le nombre de consultations d'un patient "
                "et son niveau de satisfaction sont-ils "
                "associés à son coût total de soins ?"
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
                        "region"
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
                        "consultation_count"
                    ),
                    quantitative(
                        "satisfaction_score"
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
                        "total_care_cost"
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
                    "patients_costs",

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

        "expected": {
            "expected_groups": [
                [
                    "consultations",
                    "care_costs",
                ]
            ],

            "expected_feasibilities": [
                "supported",
            ],

            "executable":
                True,

            "routing_override_reason":
                None,

            "notes":
                (
                    "The dependency result needs consultations "
                    "and care_costs. A validated multi-hop path "
                    "exists through patients."
                ),
        },

        "frozen":
            True,
    },


    # ========================================================
    # 012 — MIXED REQUEST: ONE INDEPENDENT + ONE BLOCKED GROUP
    # ========================================================

    {
        "case_id":
            "dependency_frozen_v0_8_012",

        "split":
            "test",

        "domain":
            "agricultural_cooperative",

        "user_request":
            (
                "Donne d'abord la production totale. "
                "Puis analyse si l'humidité du sol est "
                "associée au prix de vente."
            ),

        "datasets": [
            dataset(
                dataset_id="production",
                grain="farm_day",
                entity_columns=[
                    "farm_id",
                ],
                columns=[
                    identifier(
                        "farm_id"
                    ),
                    temporal(
                        "date"
                    ),
                    quantitative(
                        "production_kg"
                    ),
                ],
            ),

            dataset(
                dataset_id="soil_sensors",
                grain="sensor_hour",
                entity_columns=[
                    "sensor_id",
                ],
                columns=[
                    identifier(
                        "sensor_id"
                    ),
                    temporal(
                        "timestamp"
                    ),
                    quantitative(
                        "soil_moisture"
                    ),
                ],
            ),

            dataset(
                dataset_id="market_sales",
                grain="sale",
                entity_columns=[
                    "sale_id",
                ],
                columns=[
                    identifier(
                        "sale_id"
                    ),
                    quantitative(
                        "sale_price"
                    ),
                ],
            ),
        ],

        "relationships":
            [],

        "available_tools":
            TOOLS_WITH_JOIN,

        "expected": {
            "expected_groups": [
                [
                    "production",
                ],

                [
                    "soil_sensors",
                    "market_sales",
                ],
            ],

            "expected_feasibilities": [
                "not_required",
                "missing_validated_relationship",
            ],

            "executable":
                False,

            "routing_override_reason":
                "unsupported_analysis",

            "notes":
                (
                    "The production result is independently "
                    "possible, but the second requested result "
                    "is structurally blocked."
                ),
        },

        "frozen":
            True,
    },
]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # ========================================================
    # VALIDATE BEFORE WRITING
    # ========================================================

    validated_cases = [
        DatasetDependencyFrozenCase
        .model_validate(
            case
        )

        for case
        in CASES
    ]


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


    assert (
        len(
            validated_cases
        )
        == 12
    )


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


    OUTPUT_PATH.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


    # ========================================================
    # ENCODING
    # ========================================================

    raw_bytes = (
        OUTPUT_PATH.read_bytes()
    )


    assert not (
        raw_bytes.startswith(
            b"\xef\xbb\xbf"
        )
    )


    # ========================================================
    # DISTRIBUTION
    # ========================================================

    executable_count = sum(
        1

        for case
        in validated_cases

        if case.expected.executable
    )


    blocked_count = (
        len(
            validated_cases
        )
        - executable_count
    )


    print(
        "=== DATALENS FROZEN DEPENDENCY PIPELINE v0.8 ==="
    )

    print()


    print(
        "Benchmark:",
        DATASET_DEPENDENCY_FROZEN_BENCHMARK_VERSION,
    )


    print(
        "Cases:",
        len(
            validated_cases
        ),
    )


    print(
        "Executable:",
        executable_count,
    )


    print(
        "Blocked:",
        blocked_count,
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


    print()


    for case in validated_cases:

        print(
            "-",
            case.case_id,
            "|",
            case.domain,
            "| executable:",
            case.expected.executable,
        )


    print()

    print(
        "Output:",
        OUTPUT_PATH,
    )


    print()

    print(
        "Frozen Dependency Pipeline benchmark v0.8: PASS"
    )


if __name__ == "__main__":
    main()
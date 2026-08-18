from __future__ import annotations

from dataclasses import (
    dataclass,
)

from typing import (
    Literal,
)

import numpy as np
import pandas as pd


# ============================================================
# BENCHMARK IDENTITY
# ============================================================

HOTEL_OPERATIONS_HOLDOUT_BENCHMARK_ID = (
    "semantic:hotel_operations:holdout:v0.1"
)


HOTEL_OPERATIONS_HOLDOUT_BENCHMARK_VERSION = (
    "hotel_operations_semantic_holdout_v0.1"
)


HOTEL_OPERATIONS_DATASET_ID = (
    "hotel_operations:0001"
)


HOTEL_OPERATIONS_FILENAME = (
    "synthetic_hotel_operations.csv"
)


HOTEL_OPERATIONS_ROW_COUNT = (
    240
)


HOTEL_OPERATIONS_RANDOM_SEED = (
    20260816
)


# ============================================================
# RELATION TAXONOMY
# ============================================================

HotelMetricRelation = Literal[
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
]


STRONG_RELATIONS = {
    "same_metric_different_state",
    "same_process_different_stage",
}


# ============================================================
# RELATION CASE
# ============================================================

@dataclass(
    frozen=True,
)
class HotelOperationsRelationCase:
    case_id: str

    left_column: str

    right_column: str

    relation: HotelMetricRelation


# ============================================================
# HOLDOUT RELATIONS
#
# IMPORTANT
#
# This ground truth is frozen BEFORE the first execution of
# embeddinggemma on this holdout.
#
# The cases intentionally contain:
#
# - lexical paraphrases;
# - lifecycle-state relations;
# - process-stage relations;
# - same-dimension hard negatives.
# ============================================================

HOTEL_OPERATIONS_RELATION_CASES = (
    # --------------------------------------------------------
    # STRONG
    # --------------------------------------------------------

    HotelOperationsRelationCase(
        case_id=
            "reservation_requests_stays_booked",

        left_column=
            "Reservation requests received",

        right_column=
            "Stays booked",

        relation=
            "same_process_different_stage",
    ),

    HotelOperationsRelationCase(
        case_id=
            "housekeepers_rostered_cleaning_staff_present",

        left_column=
            "Housekeepers rostered",

        right_column=
            "Cleaning staff present",

        relation=
            "same_metric_different_state",
    ),

    HotelOperationsRelationCase(
        case_id=
            "projected_lodging_revenue_realized_room_revenue",

        left_column=
            "Projected lodging revenue",

        right_column=
            "Realized room revenue",

        relation=
            "same_metric_different_state",
    ),

    HotelOperationsRelationCase(
        case_id=
            "target_checkin_observed_service_time",

        left_column=
            "Target check-in duration (minutes)",

        right_column=
            "Observed front-desk service time (minutes)",

        relation=
            "same_metric_different_state",
    ),

    HotelOperationsRelationCase(
        case_id=
            "planned_linen_measured_laundry_weight",

        left_column=
            "Planned linen load (kg)",

        right_column=
            "Measured laundry weight (kg)",

        relation=
            "same_metric_different_state",
    ),

    HotelOperationsRelationCase(
        case_id=
            "forecast_occupied_rooms_actual_occupied_rooms",

        left_column=
            "Forecast occupied rooms",

        right_column=
            "Rooms actually occupied",

        relation=
            "same_metric_different_state",
    ),

    # --------------------------------------------------------
    # RELATED BUT DISTINCT
    #
    # Several are deliberately same-dimension competitors.
    # --------------------------------------------------------

    HotelOperationsRelationCase(
        case_id=
            "reservation_requests_guest_complaints",

        left_column=
            "Reservation requests received",

        right_column=
            "Guest complaints resolved",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "stays_booked_rooms_out_of_service",

        left_column=
            "Stays booked",

        right_column=
            "Rooms out of service",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "housekeepers_elevators",

        left_column=
            "Housekeepers rostered",

        right_column=
            "Elevators operational",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "projected_revenue_restaurant_revenue",

        left_column=
            "Projected lodging revenue",

        right_column=
            "Restaurant revenue",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "realized_revenue_maintenance_spend",

        left_column=
            "Realized room revenue",

        right_column=
            "Maintenance spend",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "target_checkin_elevator_downtime",

        left_column=
            "Target check-in duration (minutes)",

        right_column=
            "Elevator downtime (minutes)",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "observed_service_elevator_downtime",

        left_column=
            "Observed front-desk service time (minutes)",

        right_column=
            "Elevator downtime (minutes)",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "planned_linen_waste_weight",

        left_column=
            "Planned linen load (kg)",

        right_column=
            "Waste collected (kg)",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "measured_laundry_waste_weight",

        left_column=
            "Measured laundry weight (kg)",

        right_column=
            "Waste collected (kg)",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "forecast_occupied_rooms_out_of_service",

        left_column=
            "Forecast occupied rooms",

        right_column=
            "Rooms out of service",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "occupancy_guest_satisfaction",

        left_column=
            "Occupancy rate (%)",

        right_column=
            "Guest satisfaction (%)",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "occupancy_cancellation_rate",

        left_column=
            "Occupancy rate (%)",

        right_column=
            "Booking cancellation rate (%)",

        relation=
            "related_distinct_metric",
    ),

    HotelOperationsRelationCase(
        case_id=
            "satisfaction_cancellation_rate",

        left_column=
            "Guest satisfaction (%)",

        right_column=
            "Booking cancellation rate (%)",

        relation=
            "related_distinct_metric",
    ),
)


# ============================================================
# PREREGISTERED RETRIEVAL GATES
#
# These gates evaluate a RETRIEVER.
#
# They are deliberately different from the FP=0 safety gate
# used by components that authorize analytical operations.
# ============================================================

HOTEL_OPERATIONS_HOLDOUT_GATES = {
    "reference_top_k":
        3,

    "minimum_strong_pair_recall":
        0.90,

    "minimum_pair_reduction":
        0.40,

    "maximum_embedding_call_failures":
        0,

    "candidate_generation_only":
        True,

    "safety_authority":
        False,
}


# ============================================================
# DATASET
#
# Synthetic deterministic hotel-operations measures.
#
# Values are not used by the embedding retriever itself.
# They are nevertheless frozen so the complete holdout
# dataset has an immutable identity.
# ============================================================

def build_hotel_operations_holdout_dataframe(
) -> pd.DataFrame:
    rng = (
        np.random.default_rng(
            HOTEL_OPERATIONS_RANDOM_SEED
        )
    )


    row_count = (
        HOTEL_OPERATIONS_ROW_COUNT
    )


    reservation_requests = (
        rng.integers(
            70,
            221,
            size=
                row_count,
        )
    )


    stays_booked = (
        reservation_requests
        -
        rng.integers(
            4,
            46,
            size=
                row_count,
        )
    )


    stays_booked = (
        np.clip(
            stays_booked,
            0,
            None,
        )
    )


    housekeepers_rostered = (
        rng.integers(
            8,
            31,
            size=
                row_count,
        )
    )


    cleaning_staff_present = (
        housekeepers_rostered
        -
        rng.integers(
            0,
            5,
            size=
                row_count,
        )
    )


    cleaning_staff_present = (
        np.clip(
            cleaning_staff_present,
            0,
            None,
        )
    )


    projected_lodging_revenue = (
        rng.normal(
            42000.0,
            7000.0,
            size=
                row_count,
        )
    )


    projected_lodging_revenue = (
        np.clip(
            projected_lodging_revenue,
            10000.0,
            None,
        )
    )


    realized_room_revenue = (
        projected_lodging_revenue
        *
        rng.normal(
            0.99,
            0.08,
            size=
                row_count,
        )
    )


    realized_room_revenue = (
        np.clip(
            realized_room_revenue,
            5000.0,
            None,
        )
    )


    target_checkin_duration = (
        rng.normal(
            8.0,
            1.2,
            size=
                row_count,
        )
    )


    target_checkin_duration = (
        np.clip(
            target_checkin_duration,
            4.0,
            15.0,
        )
    )


    observed_service_time = (
        target_checkin_duration
        *
        rng.normal(
            1.08,
            0.16,
            size=
                row_count,
        )
    )


    observed_service_time = (
        np.clip(
            observed_service_time,
            2.0,
            25.0,
        )
    )


    planned_linen_load = (
        rng.normal(
            620.0,
            80.0,
            size=
                row_count,
        )
    )


    planned_linen_load = (
        np.clip(
            planned_linen_load,
            300.0,
            None,
        )
    )


    measured_laundry_weight = (
        planned_linen_load
        *
        rng.normal(
            1.02,
            0.08,
            size=
                row_count,
        )
    )


    measured_laundry_weight = (
        np.clip(
            measured_laundry_weight,
            250.0,
            None,
        )
    )


    forecast_occupied_rooms = (
        rng.integers(
            70,
            181,
            size=
                row_count,
        )
    )


    rooms_actually_occupied = (
        forecast_occupied_rooms
        +
        rng.integers(
            -15,
            16,
            size=
                row_count,
        )
    )


    rooms_actually_occupied = (
        np.clip(
            rooms_actually_occupied,
            0,
            220,
        )
    )


    guest_complaints_resolved = (
        rng.integers(
            0,
            31,
            size=
                row_count,
        )
    )


    elevators_operational = (
        rng.integers(
            2,
            8,
            size=
                row_count,
        )
    )


    maintenance_spend = (
        rng.normal(
            6000.0,
            1800.0,
            size=
                row_count,
        )
    )


    maintenance_spend = (
        np.clip(
            maintenance_spend,
            1000.0,
            None,
        )
    )


    restaurant_revenue = (
        rng.normal(
            12000.0,
            3000.0,
            size=
                row_count,
        )
    )


    restaurant_revenue = (
        np.clip(
            restaurant_revenue,
            2000.0,
            None,
        )
    )


    elevator_downtime = (
        rng.gamma(
            shape=
                2.0,

            scale=
                8.0,

            size=
                row_count,
        )
    )


    waste_collected = (
        rng.normal(
            180.0,
            35.0,
            size=
                row_count,
        )
    )


    waste_collected = (
        np.clip(
            waste_collected,
            60.0,
            None,
        )
    )


    occupancy_rate = (
        rooms_actually_occupied
        /
        220.0
        *
        100.0
    )


    guest_satisfaction = (
        rng.normal(
            86.0,
            6.0,
            size=
                row_count,
        )
    )


    guest_satisfaction = (
        np.clip(
            guest_satisfaction,
            50.0,
            100.0,
        )
    )


    booking_cancellation_rate = (
        rng.normal(
            8.0,
            3.0,
            size=
                row_count,
        )
    )


    booking_cancellation_rate = (
        np.clip(
            booking_cancellation_rate,
            0.0,
            30.0,
        )
    )


    rooms_out_of_service = (
        rng.integers(
            0,
            16,
            size=
                row_count,
        )
    )


    dataframe = pd.DataFrame(
        {
            "Reservation requests received":
                reservation_requests,

            "Stays booked":
                stays_booked,

            "Housekeepers rostered":
                housekeepers_rostered,

            "Cleaning staff present":
                cleaning_staff_present,

            "Projected lodging revenue":
                np.round(
                    projected_lodging_revenue,
                    2,
                ),

            "Realized room revenue":
                np.round(
                    realized_room_revenue,
                    2,
                ),

            "Target check-in duration (minutes)":
                np.round(
                    target_checkin_duration,
                    3,
                ),

            "Observed front-desk service time (minutes)":
                np.round(
                    observed_service_time,
                    3,
                ),

            "Planned linen load (kg)":
                np.round(
                    planned_linen_load,
                    3,
                ),

            "Measured laundry weight (kg)":
                np.round(
                    measured_laundry_weight,
                    3,
                ),

            "Forecast occupied rooms":
                forecast_occupied_rooms,

            "Rooms actually occupied":
                rooms_actually_occupied,

            "Guest complaints resolved":
                guest_complaints_resolved,

            "Elevators operational":
                elevators_operational,

            "Maintenance spend":
                np.round(
                    maintenance_spend,
                    2,
                ),

            "Restaurant revenue":
                np.round(
                    restaurant_revenue,
                    2,
                ),

            "Elevator downtime (minutes)":
                np.round(
                    elevator_downtime,
                    3,
                ),

            "Waste collected (kg)":
                np.round(
                    waste_collected,
                    3,
                ),

            "Occupancy rate (%)":
                np.round(
                    occupancy_rate,
                    3,
                ),

            "Guest satisfaction (%)":
                np.round(
                    guest_satisfaction,
                    3,
                ),

            "Booking cancellation rate (%)":
                np.round(
                    booking_cancellation_rate,
                    3,
                ),

            "Rooms out of service":
                rooms_out_of_service,
        }
    )


    return (
        dataframe
    )


# ============================================================
# PUBLIC CASE ACCESS
# ============================================================

def build_hotel_operations_holdout_relation_cases(
) -> list[
    HotelOperationsRelationCase
]:
    return list(
        HOTEL_OPERATIONS_RELATION_CASES
    )

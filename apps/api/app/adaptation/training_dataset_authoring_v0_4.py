from __future__ import annotations

import argparse
import hashlib
import json
import re

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AUTHORING_RULE_VERSION = (
    "qlora_v0.4_training_dataset_authoring_v0.1"
)


DATASET_ID = (
    "adaptation:datalens-semantic:training:v0.4"
)


DATASET_VERSION = (
    "datalens_semantic_adaptation_training_v0.4"
)


RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


FULL_RELATIONS = RELATIONS


FOCUSED_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
)


HARD_NEGATIVE_RELATIONS = {
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
}


EXPECTED_RELATION_COUNTS = {
    "same_metric_different_state":
        50,

    "same_process_different_stage":
        50,

    "related_distinct_metric":
        50,

    "unrelated":
        40,

    "uncertain":
        40,
}


EXPECTED_DOMAIN_GROUPS = {
    "retail_store_operations":
        (3, 1),

    "subscription_billing_operations":
        (3, 1),

    "telecom_field_service":
        (3, 1),

    "renewable_power_operations":
        (3, 1),

    "food_processing_operations":
        (3, 1),

    "maritime_terminal_operations":
        (2, 1),

    "construction_site_operations":
        (2, 1),

    "insurance_claim_operations":
        (2, 1),

    "media_streaming_operations":
        (2, 1),

    "municipal_waste_operations":
        (2, 1),

    "agricultural_irrigation_operations":
        (3, 0),

    "rail_freight_operations":
        (3, 0),

    "data_center_facility_operations":
        (3, 0),

    "laboratory_sample_operations":
        (3, 0),

    "procurement_operations":
        (3, 0),
}


FORBIDDEN_DOMAIN_TOKENS = (
    "airport",
    "hotel",
    "greenhouse",
)


ROOT = Path(__file__).resolve().parents[2]


DATASET_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "datasets"
    / "datalens_semantic_training_v0.4_"
        "authoring_v0.1.jsonl"
)


DESIGN_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "design"
    / (
        "datalens_semantic_qlora_v0.4_"
        "training_dataset_design_v0.1.json"
    )
)


DESIGN_FREEZE_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "design"
    / (
        "datalens_semantic_qlora_v0.4_"
        "training_dataset_design_v0.1_freeze.json"
    )
)


EXPECTED_DESIGN_SHA256 = (
    "ce3f855c0563fba08561e4fced8a3cd0"
    "6910570a928483c1d7cc7358d6aab56c"
)


EXPECTED_DESIGN_FREEZE_SHA256 = (
    "ebcc423ca70bfaf0bc5e12cf66291dd4"
    "1cac34beac06033d912bad6a2285eec7"
)


def metric(
    name: str,
    description: str,
) -> dict[str, str]:
    return {
        "description":
            description,

        "metric":
            name,
    }


def group(
    group_id: str,
    domain: str,
    group_type: str,
    concept: str,
    anchor: dict[str, str],
    same_metric: dict[str, str],
    same_process: dict[str, str],
    related: dict[str, str],
    unrelated: dict[str, str] | None = None,
    uncertain: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "anchor":
            anchor,

        "concept":
            concept,

        "domain":
            domain,

        "group_id":
            group_id,

        "group_type":
            group_type,

        "partners": {
            "related_distinct_metric":
                related,

            "same_metric_different_state":
                same_metric,

            "same_process_different_stage":
                same_process,

            **(
                {
                    "unrelated":
                        unrelated,

                    "uncertain":
                        uncertain,
                }
                if group_type
                ==
                "full"
                else
                {}
            ),
        },
    }


# ============================================================
# MANUALLY AUTHORED SEMANTIC GROUPS
#
# No evaluation case material is read or copied here.
# ============================================================


GROUP_SPECS = [
    # ========================================================
    # RETAIL STORE OPERATIONS
    # 3 full + 1 focused
    # ========================================================
    group(
        "retail-01",
        "retail_store_operations",
        "full",
        "customer orders received",
        metric(
            "customer_orders_received",
            (
                "Number of customer orders actually received "
                "by the store during the business day."
            ),
        ),
        metric(
            "forecast_customer_orders_received",
            (
                "Forecast number of customer orders expected "
                "to be received during the same business day."
            ),
        ),
        metric(
            "customer_orders_fulfilled",
            (
                "Number of received customer orders completed "
                "after picking, payment and handoff."
            ),
        ),
        metric(
            "average_checkout_wait_minutes",
            (
                "Average customer waiting time before checkout "
                "service begins during store operations."
            ),
        ),
        metric(
            "freezer_temperature_c",
            (
                "Measured temperature of a frozen-food storage "
                "cabinet in degrees Celsius."
            ),
        ),
        metric(
            "store_flow_index",
            (
                "Internal store-flow index whose formula, units "
                "and operational scope are undocumented."
            ),
        ),
    ),
    group(
        "retail-02",
        "retail_store_operations",
        "full",
        "shelf restock requests",
        metric(
            "shelf_restock_requests",
            (
                "Number of shelf restock requests actually "
                "created during the operating day."
            ),
        ),
        metric(
            "planned_shelf_restock_requests",
            (
                "Planned number of shelf restock requests "
                "expected for the same operating day."
            ),
        ),
        metric(
            "shelf_restock_tasks_completed",
            (
                "Number of requested shelf restock tasks "
                "completed after stock retrieval and placement."
            ),
        ),
        metric(
            "stockout_incident_count",
            (
                "Number of product stockout incidents recorded "
                "on customer-facing shelves."
            ),
        ),
        metric(
            "entrance_door_cycle_count",
            (
                "Number of automatic entrance-door opening "
                "and closing cycles recorded."
            ),
        ),
        metric(
            "merchandising_readiness_score",
            (
                "Composite merchandising readiness score whose "
                "components and calculation are undocumented."
            ),
        ),
    ),
    group(
        "retail-03",
        "retail_store_operations",
        "full",
        "loyalty redemptions submitted",
        metric(
            "loyalty_redemptions_submitted",
            (
                "Number of customer loyalty redemptions "
                "actually submitted at checkout."
            ),
        ),
        metric(
            "expected_loyalty_redemptions_submitted",
            (
                "Expected number of loyalty redemptions to be "
                "submitted during the same reporting period."
            ),
        ),
        metric(
            "loyalty_redemptions_approved",
            (
                "Number of submitted loyalty redemptions "
                "approved after eligibility validation."
            ),
        ),
        metric(
            "average_basket_value",
            (
                "Average monetary value of completed customer "
                "shopping baskets."
            ),
        ),
        metric(
            "cleaning_supply_inventory_units",
            (
                "Number of cleaning-supply units held in "
                "back-office inventory."
            ),
        ),
        metric(
            "customer_value_metric",
            (
                "Internal customer-value metric with no "
                "documented formula or population definition."
            ),
        ),
    ),
    group(
        "retail-04",
        "retail_store_operations",
        "focused",
        "online pickup orders received",
        metric(
            "online_pickup_orders_received",
            (
                "Number of online pickup orders actually "
                "received by the store."
            ),
        ),
        metric(
            "forecast_online_pickup_orders_received",
            (
                "Forecast number of online pickup orders "
                "expected to be received by the store."
            ),
        ),
        metric(
            "online_pickup_orders_collected",
            (
                "Number of received online pickup orders "
                "later collected by customers."
            ),
        ),
        metric(
            "pickup_wait_minutes",
            (
                "Average number of minutes customers wait "
                "before receiving prepared pickup orders."
            ),
        ),
    ),

    # ========================================================
    # SUBSCRIPTION BILLING
    # ========================================================
    group(
        "billing-01",
        "subscription_billing_operations",
        "full",
        "invoices generated",
        metric(
            "invoices_generated",
            (
                "Number of subscription invoices actually "
                "generated during the billing cycle."
            ),
        ),
        metric(
            "forecast_invoices_generated",
            (
                "Forecast number of subscription invoices "
                "expected during the billing cycle."
            ),
        ),
        metric(
            "invoices_paid",
            (
                "Number of generated subscription invoices "
                "later paid by customers."
            ),
        ),
        metric(
            "overdue_balance_amount",
            (
                "Total monetary balance remaining overdue on "
                "customer subscription accounts."
            ),
        ),
        metric(
            "billing_office_humidity_percent",
            (
                "Measured relative humidity in the billing "
                "operations office."
            ),
        ),
        metric(
            "billing_health_index",
            (
                "Internal billing-health index whose components "
                "and weighting are undocumented."
            ),
        ),
    ),
    group(
        "billing-02",
        "subscription_billing_operations",
        "full",
        "renewal notices sent",
        metric(
            "renewal_notices_sent",
            (
                "Number of subscription renewal notices "
                "actually sent to customers."
            ),
        ),
        metric(
            "planned_renewal_notices_sent",
            (
                "Planned number of subscription renewal "
                "notices to be sent in the same period."
            ),
        ),
        metric(
            "subscriptions_renewed",
            (
                "Number of notified subscriptions later "
                "renewed by customers."
            ),
        ),
        metric(
            "subscription_churn_rate",
            (
                "Percentage of subscriptions ending without "
                "renewal during the reporting period."
            ),
        ),
        metric(
            "office_printer_page_count",
            (
                "Number of pages printed by an administrative "
                "office printer."
            ),
        ),
        metric(
            "retention_score",
            (
                "Internal retention score with no documented "
                "features, scale or calculation."
            ),
        ),
    ),
    group(
        "billing-03",
        "subscription_billing_operations",
        "full",
        "payment attempts submitted",
        metric(
            "payment_attempts_submitted",
            (
                "Number of recurring payment attempts actually "
                "submitted to payment processors."
            ),
        ),
        metric(
            "forecast_payment_attempts_submitted",
            (
                "Forecast number of recurring payment attempts "
                "expected to be submitted."
            ),
        ),
        metric(
            "successful_payment_captures",
            (
                "Number of submitted payment attempts later "
                "captured successfully."
            ),
        ),
        metric(
            "payment_failure_rate",
            (
                "Percentage of submitted recurring payments "
                "that fail processing."
            ),
        ),
        metric(
            "kitchen_coffee_usage_grams",
            (
                "Weight of coffee consumed in an employee "
                "kitchen during the reporting period."
            ),
        ),
        metric(
            "payment_quality_index",
            (
                "Internal payment-quality index whose inputs "
                "and interpretation are undocumented."
            ),
        ),
    ),
    group(
        "billing-04",
        "subscription_billing_operations",
        "focused",
        "refund requests opened",
        metric(
            "refund_requests_opened",
            (
                "Number of customer refund requests actually "
                "opened during the reporting period."
            ),
        ),
        metric(
            "forecast_refund_requests_opened",
            (
                "Forecast number of customer refund requests "
                "expected to be opened."
            ),
        ),
        metric(
            "refunds_completed",
            (
                "Number of opened refund requests later "
                "completed and issued."
            ),
        ),
        metric(
            "refund_processing_minutes",
            (
                "Average elapsed processing time for customer "
                "refund requests."
            ),
        ),
    ),

    # ========================================================
    # TELECOM FIELD SERVICE
    # ========================================================
    group(
        "telecom-01",
        "telecom_field_service",
        "full",
        "service tickets opened",
        metric(
            "service_tickets_opened",
            (
                "Number of field-service tickets actually "
                "opened for customer network issues."
            ),
        ),
        metric(
            "forecast_service_tickets_opened",
            (
                "Forecast number of field-service tickets "
                "expected to be opened."
            ),
        ),
        metric(
            "service_tickets_resolved",
            (
                "Number of opened field-service tickets later "
                "resolved and closed."
            ),
        ),
        metric(
            "technician_travel_minutes",
            (
                "Average technician travel time required to "
                "reach assigned field-service locations."
            ),
        ),
        metric(
            "warehouse_lighting_kwh",
            (
                "Electrical energy consumed by lighting in a "
                "telecom equipment warehouse."
            ),
        ),
        metric(
            "field_service_index",
            (
                "Internal field-service index whose formula "
                "and component metrics are undocumented."
            ),
        ),
    ),
    group(
        "telecom-02",
        "telecom_field_service",
        "full",
        "site inspection requests",
        metric(
            "site_inspection_requests",
            (
                "Number of network-site inspection requests "
                "actually created."
            ),
        ),
        metric(
            "planned_site_inspection_requests",
            (
                "Planned number of network-site inspection "
                "requests for the same period."
            ),
        ),
        metric(
            "site_inspections_completed",
            (
                "Number of requested network-site inspections "
                "later completed by technicians."
            ),
        ),
        metric(
            "technician_utilization_percent",
            (
                "Percentage of technician working time spent "
                "on assigned field-service activity."
            ),
        ),
        metric(
            "vehicle_washer_cycle_count",
            (
                "Number of wash cycles completed by the fleet "
                "vehicle washing station."
            ),
        ),
        metric(
            "site_condition_score",
            (
                "Internal site-condition score whose assessed "
                "components and scale are undocumented."
            ),
        ),
    ),
    group(
        "telecom-03",
        "telecom_field_service",
        "full",
        "equipment replacement orders created",
        metric(
            "equipment_replacement_orders_created",
            (
                "Number of field equipment replacement orders "
                "actually created."
            ),
        ),
        metric(
            "forecast_equipment_replacement_orders_created",
            (
                "Forecast number of field equipment replacement "
                "orders expected to be created."
            ),
        ),
        metric(
            "equipment_replacements_completed",
            (
                "Number of replacement orders later completed "
                "with equipment installed."
            ),
        ),
        metric(
            "spare_part_stockout_count",
            (
                "Number of field-service spare-part stockout "
                "events recorded."
            ),
        ),
        metric(
            "meeting_room_occupancy_count",
            (
                "Number of occupants detected in an office "
                "meeting room."
            ),
        ),
        metric(
            "network_restoration_index",
            (
                "Internal network-restoration index with no "
                "documented formula or unit."
            ),
        ),
    ),
    group(
        "telecom-04",
        "telecom_field_service",
        "focused",
        "customer installations booked",
        metric(
            "customer_installations_booked",
            (
                "Number of customer installation appointments "
                "actually booked."
            ),
        ),
        metric(
            "forecast_customer_installations_booked",
            (
                "Forecast number of customer installation "
                "appointments expected to be booked."
            ),
        ),
        metric(
            "customer_installations_activated",
            (
                "Number of booked installations later completed "
                "and activated."
            ),
        ),
        metric(
            "installation_duration_minutes",
            (
                "Average technician time required to complete "
                "a customer installation."
            ),
        ),
    ),

    # ========================================================
    # RENEWABLE POWER OPERATIONS
    # ========================================================
    group(
        "renewable-01",
        "renewable_power_operations",
        "full",
        "maintenance work orders opened",
        metric(
            "maintenance_work_orders_opened",
            (
                "Number of renewable-asset maintenance work "
                "orders actually opened."
            ),
        ),
        metric(
            "forecast_maintenance_work_orders_opened",
            (
                "Forecast number of maintenance work orders "
                "expected to be opened."
            ),
        ),
        metric(
            "maintenance_work_orders_completed",
            (
                "Number of opened maintenance work orders "
                "later completed."
            ),
        ),
        metric(
            "turbine_availability_percent",
            (
                "Percentage of scheduled time that generation "
                "turbines remain available for operation."
            ),
        ),
        metric(
            "admin_printer_toner_percent",
            (
                "Remaining toner percentage in an administrative "
                "office printer."
            ),
        ),
        metric(
            "asset_health_index",
            (
                "Internal asset-health index whose formula and "
                "component measures are undocumented."
            ),
        ),
    ),
    group(
        "renewable-02",
        "renewable_power_operations",
        "full",
        "energy dispatch requests received",
        metric(
            "energy_dispatch_requests_received",
            (
                "Number of grid energy-dispatch requests "
                "actually received."
            ),
        ),
        metric(
            "forecast_energy_dispatch_requests_received",
            (
                "Forecast number of grid energy-dispatch "
                "requests expected to be received."
            ),
        ),
        metric(
            "energy_dispatches_executed",
            (
                "Number of received dispatch requests later "
                "executed by generation assets."
            ),
        ),
        metric(
            "grid_frequency_deviation_hz",
            (
                "Measured deviation of grid frequency from "
                "its nominal value."
            ),
        ),
        metric(
            "office_parking_vehicle_count",
            (
                "Number of vehicles counted in an administrative "
                "office parking area."
            ),
        ),
        metric(
            "dispatch_quality_score",
            (
                "Internal dispatch-quality score with no "
                "documented formula or interpretation."
            ),
        ),
    ),
    group(
        "renewable-03",
        "renewable_power_operations",
        "full",
        "battery charge cycles started",
        metric(
            "battery_charge_cycles_started",
            (
                "Number of battery-storage charge cycles "
                "actually started."
            ),
        ),
        metric(
            "planned_battery_charge_cycles_started",
            (
                "Planned number of battery-storage charge "
                "cycles to be started."
            ),
        ),
        metric(
            "battery_charge_cycles_completed",
            (
                "Number of started battery-storage charge "
                "cycles later completed."
            ),
        ),
        metric(
            "battery_temperature_c",
            (
                "Measured operating temperature of battery "
                "storage modules in degrees Celsius."
            ),
        ),
        metric(
            "cafeteria_transaction_count",
            (
                "Number of purchases recorded in an employee "
                "cafeteria."
            ),
        ),
        metric(
            "storage_performance_index",
            (
                "Internal storage-performance index whose "
                "components and calculation are undocumented."
            ),
        ),
    ),
    group(
        "renewable-04",
        "renewable_power_operations",
        "focused",
        "solar cleaning tasks due",
        metric(
            "solar_cleaning_tasks_due",
            (
                "Number of solar-array cleaning tasks actually "
                "due for execution."
            ),
        ),
        metric(
            "forecast_solar_cleaning_tasks_due",
            (
                "Forecast number of solar-array cleaning tasks "
                "expected to become due."
            ),
        ),
        metric(
            "solar_cleaning_tasks_completed",
            (
                "Number of due solar-array cleaning tasks "
                "later completed."
            ),
        ),
        metric(
            "panel_soiling_loss_percent",
            (
                "Estimated generation loss percentage attributed "
                "to panel surface soiling."
            ),
        ),
    ),

    # ========================================================
    # FOOD PROCESSING OPERATIONS
    # ========================================================
    group(
        "food-01",
        "food_processing_operations",
        "full",
        "production batches started",
        metric(
            "production_batches_started",
            (
                "Number of food-production batches actually "
                "started on processing lines."
            ),
        ),
        metric(
            "planned_production_batches_started",
            (
                "Planned number of food-production batches "
                "to be started."
            ),
        ),
        metric(
            "production_batches_packaged",
            (
                "Number of started production batches later "
                "completed through packaging."
            ),
        ),
        metric(
            "line_downtime_minutes",
            (
                "Total processing-line downtime measured in "
                "minutes."
            ),
        ),
        metric(
            "office_wifi_session_count",
            (
                "Number of wireless network sessions recorded "
                "in administrative offices."
            ),
        ),
        metric(
            "production_flow_index",
            (
                "Internal production-flow index whose formula "
                "and unit are undocumented."
            ),
        ),
    ),
    group(
        "food-02",
        "food_processing_operations",
        "full",
        "raw material lots received",
        metric(
            "raw_material_lots_received",
            (
                "Number of raw-material lots actually received "
                "at the processing facility."
            ),
        ),
        metric(
            "forecast_raw_material_lots_received",
            (
                "Forecast number of raw-material lots expected "
                "to be received."
            ),
        ),
        metric(
            "raw_material_lots_released_to_production",
            (
                "Number of received raw-material lots later "
                "released for production use."
            ),
        ),
        metric(
            "incoming_quality_reject_rate",
            (
                "Percentage of incoming raw-material lots "
                "rejected during quality inspection."
            ),
        ),
        metric(
            "parking_gate_cycle_count",
            (
                "Number of open-close cycles recorded by the "
                "employee parking gate."
            ),
        ),
        metric(
            "material_readiness_score",
            (
                "Internal material-readiness score whose "
                "components and scale are undocumented."
            ),
        ),
    ),
    group(
        "food-03",
        "food_processing_operations",
        "full",
        "sanitation cycles started",
        metric(
            "sanitation_cycles_started",
            (
                "Number of processing-equipment sanitation "
                "cycles actually started."
            ),
        ),
        metric(
            "planned_sanitation_cycles_started",
            (
                "Planned number of processing-equipment "
                "sanitation cycles to be started."
            ),
        ),
        metric(
            "sanitation_cycles_verified",
            (
                "Number of started sanitation cycles later "
                "verified as complete."
            ),
        ),
        metric(
            "microbial_swab_failure_count",
            (
                "Number of post-sanitation microbial swab "
                "tests that fail acceptance criteria."
            ),
        ),
        metric(
            "payroll_document_count",
            (
                "Number of payroll documents generated by "
                "administrative systems."
            ),
        ),
        metric(
            "hygiene_performance_index",
            (
                "Internal hygiene-performance index with no "
                "documented formula or weighting."
            ),
        ),
    ),
    group(
        "food-04",
        "food_processing_operations",
        "focused",
        "product units filled",
        metric(
            "product_units_filled",
            (
                "Number of product units actually filled on "
                "the packaging line."
            ),
        ),
        metric(
            "forecast_product_units_filled",
            (
                "Forecast number of product units expected "
                "to be filled."
            ),
        ),
        metric(
            "product_units_case_packed",
            (
                "Number of filled product units later packed "
                "into shipping cases."
            ),
        ),
        metric(
            "filling_line_speed_units_per_minute",
            (
                "Average number of units processed per minute "
                "by the filling line."
            ),
        ),
    ),

    # ========================================================
    # MARITIME TERMINAL OPERATIONS
    # 2 full + 1 focused
    # ========================================================
    group(
        "maritime-01",
        "maritime_terminal_operations",
        "full",
        "containers discharged",
        metric(
            "containers_discharged",
            (
                "Number of containers actually discharged "
                "from arriving vessels."
            ),
        ),
        metric(
            "forecast_containers_discharged",
            (
                "Forecast number of containers expected to "
                "be discharged from arriving vessels."
            ),
        ),
        metric(
            "containers_released_from_yard",
            (
                "Number of discharged containers later released "
                "from terminal yard custody."
            ),
        ),
        metric(
            "average_yard_dwell_hours",
            (
                "Average number of hours containers remain "
                "stored in the terminal yard."
            ),
        ),
        metric(
            "office_elevator_ride_count",
            (
                "Number of elevator journeys recorded in an "
                "administrative office building."
            ),
        ),
        metric(
            "terminal_flow_index",
            (
                "Internal terminal-flow index whose formula "
                "and operational scope are undocumented."
            ),
        ),
    ),
    group(
        "maritime-02",
        "maritime_terminal_operations",
        "full",
        "vessel moves requested",
        metric(
            "vessel_moves_requested",
            (
                "Number of vessel movement requests actually "
                "submitted to terminal operations."
            ),
        ),
        metric(
            "planned_vessel_moves_requested",
            (
                "Planned number of vessel movement requests "
                "for the same operating period."
            ),
        ),
        metric(
            "vessel_moves_completed",
            (
                "Number of requested vessel movements later "
                "completed."
            ),
        ),
        metric(
            "tugboat_fuel_liters",
            (
                "Volume of fuel consumed by tugboats supporting "
                "vessel movements."
            ),
        ),
        metric(
            "cafeteria_meal_count",
            (
                "Number of meals served in a terminal employee "
                "cafeteria."
            ),
        ),
        metric(
            "marine_service_score",
            (
                "Internal marine-service score whose inputs "
                "and calculation are undocumented."
            ),
        ),
    ),
    group(
        "maritime-03",
        "maritime_terminal_operations",
        "focused",
        "gate truck arrivals",
        metric(
            "gate_truck_arrivals",
            (
                "Number of cargo trucks actually arriving at "
                "terminal entry gates."
            ),
        ),
        metric(
            "forecast_gate_truck_arrivals",
            (
                "Forecast number of cargo trucks expected "
                "to arrive at terminal gates."
            ),
        ),
        metric(
            "trucks_cleared_through_gate",
            (
                "Number of arriving trucks later cleared "
                "through terminal gate processing."
            ),
        ),
        metric(
            "average_gate_queue_minutes",
            (
                "Average truck waiting time before terminal "
                "gate processing begins."
            ),
        ),
    ),

    # ========================================================
    # CONSTRUCTION SITE OPERATIONS
    # ========================================================
    group(
        "construction-01",
        "construction_site_operations",
        "full",
        "concrete deliveries received",
        metric(
            "concrete_deliveries_received",
            (
                "Number of ready-mix concrete deliveries "
                "actually received at the construction site."
            ),
        ),
        metric(
            "planned_concrete_deliveries_received",
            (
                "Planned number of ready-mix concrete deliveries "
                "to be received."
            ),
        ),
        metric(
            "concrete_pours_completed",
            (
                "Number of received concrete delivery sequences "
                "later completed as structural pours."
            ),
        ),
        metric(
            "concrete_test_failure_count",
            (
                "Number of concrete quality tests failing "
                "specified acceptance criteria."
            ),
        ),
        metric(
            "site_office_printer_pages",
            (
                "Number of pages printed by a construction "
                "site office printer."
            ),
        ),
        metric(
            "pour_readiness_index",
            (
                "Internal pour-readiness index whose contributing "
                "factors and formula are undocumented."
            ),
        ),
    ),
    group(
        "construction-02",
        "construction_site_operations",
        "full",
        "safety inspections opened",
        metric(
            "safety_inspections_opened",
            (
                "Number of construction safety inspections "
                "actually opened for review."
            ),
        ),
        metric(
            "planned_safety_inspections_opened",
            (
                "Planned number of construction safety "
                "inspections to be opened."
            ),
        ),
        metric(
            "safety_inspections_closed",
            (
                "Number of opened safety inspections later "
                "closed after review and action."
            ),
        ),
        metric(
            "recordable_incident_count",
            (
                "Number of recordable worker safety incidents "
                "reported on the site."
            ),
        ),
        metric(
            "canteen_beverage_servings",
            (
                "Number of beverages served in the construction "
                "site canteen."
            ),
        ),
        metric(
            "safety_condition_score",
            (
                "Internal safety-condition score whose components "
                "and weighting are undocumented."
            ),
        ),
    ),
    group(
        "construction-03",
        "construction_site_operations",
        "focused",
        "material requests submitted",
        metric(
            "material_requests_submitted",
            (
                "Number of construction material requests "
                "actually submitted by site crews."
            ),
        ),
        metric(
            "forecast_material_requests_submitted",
            (
                "Forecast number of material requests expected "
                "to be submitted by site crews."
            ),
        ),
        metric(
            "materials_issued_to_crews",
            (
                "Number of submitted material requests later "
                "fulfilled by issuing materials."
            ),
        ),
        metric(
            "material_procurement_lead_days",
            (
                "Average procurement lead time for construction "
                "materials measured in days."
            ),
        ),
    ),

    # ========================================================
    # INSURANCE CLAIM OPERATIONS
    # ========================================================
    group(
        "claims-01",
        "insurance_claim_operations",
        "full",
        "claims registered",
        metric(
            "claims_registered",
            (
                "Number of insurance claims actually registered "
                "during the reporting period."
            ),
        ),
        metric(
            "forecast_claims_registered",
            (
                "Forecast number of insurance claims expected "
                "to be registered."
            ),
        ),
        metric(
            "claims_settled",
            (
                "Number of registered insurance claims later "
                "settled and closed."
            ),
        ),
        metric(
            "average_claim_cycle_days",
            (
                "Average elapsed days between claim registration "
                "and final disposition."
            ),
        ),
        metric(
            "office_badge_scan_count",
            (
                "Number of employee access-badge scans recorded "
                "at an office entrance."
            ),
        ),
        metric(
            "claims_complexity_index",
            (
                "Internal claims-complexity index with no "
                "documented features or calculation."
            ),
        ),
    ),
    group(
        "claims-02",
        "insurance_claim_operations",
        "full",
        "supporting documents requested",
        metric(
            "claim_documents_requested",
            (
                "Number of supporting claim documents actually "
                "requested from customers."
            ),
        ),
        metric(
            "forecast_claim_documents_requested",
            (
                "Forecast number of supporting claim documents "
                "expected to be requested."
            ),
        ),
        metric(
            "claim_documents_verified",
            (
                "Number of requested claim documents later "
                "verified by claims staff."
            ),
        ),
        metric(
            "missing_document_rate",
            (
                "Percentage of claim files missing required "
                "supporting documentation."
            ),
        ),
        metric(
            "office_rack_temperature_c",
            (
                "Measured temperature of an office network "
                "equipment rack."
            ),
        ),
        metric(
            "document_quality_score",
            (
                "Internal document-quality score whose criteria "
                "and weighting are undocumented."
            ),
        ),
    ),
    group(
        "claims-03",
        "insurance_claim_operations",
        "focused",
        "repair estimates received",
        metric(
            "repair_estimates_received",
            (
                "Number of insured repair estimates actually "
                "received for claim review."
            ),
        ),
        metric(
            "forecast_repair_estimates_received",
            (
                "Forecast number of insured repair estimates "
                "expected to be received."
            ),
        ),
        metric(
            "repair_authorizations_issued",
            (
                "Number of received repair estimates later "
                "approved with repair authorization."
            ),
        ),
        metric(
            "estimate_review_minutes",
            (
                "Average adjuster time required to review a "
                "submitted repair estimate."
            ),
        ),
    ),

    # ========================================================
    # MEDIA STREAMING OPERATIONS
    # ========================================================
    group(
        "media-01",
        "media_streaming_operations",
        "full",
        "playback sessions started",
        metric(
            "playback_sessions_started",
            (
                "Number of viewer playback sessions actually "
                "started on the streaming platform."
            ),
        ),
        metric(
            "forecast_playback_sessions_started",
            (
                "Forecast number of viewer playback sessions "
                "expected to start."
            ),
        ),
        metric(
            "playback_sessions_completed",
            (
                "Number of started playback sessions later "
                "completed to the defined completion threshold."
            ),
        ),
        metric(
            "buffering_ratio_percent",
            (
                "Percentage of playback time spent buffering "
                "rather than presenting content."
            ),
        ),
        metric(
            "office_smart_light_event_count",
            (
                "Number of smart-light control events recorded "
                "in an office."
            ),
        ),
        metric(
            "playback_quality_index",
            (
                "Internal playback-quality index whose formula "
                "and components are undocumented."
            ),
        ),
    ),
    group(
        "media-02",
        "media_streaming_operations",
        "full",
        "content uploads submitted",
        metric(
            "content_uploads_submitted",
            (
                "Number of media content uploads actually "
                "submitted for platform processing."
            ),
        ),
        metric(
            "planned_content_uploads_submitted",
            (
                "Planned number of content uploads to be "
                "submitted for processing."
            ),
        ),
        metric(
            "content_items_published",
            (
                "Number of submitted content uploads later "
                "published to viewers."
            ),
        ),
        metric(
            "transcoding_duration_minutes",
            (
                "Average processing time required to transcode "
                "submitted media content."
            ),
        ),
        metric(
            "payroll_adjustment_count",
            (
                "Number of payroll adjustments recorded for "
                "employees."
            ),
        ),
        metric(
            "content_readiness_score",
            (
                "Internal content-readiness score whose inputs "
                "and scale are undocumented."
            ),
        ),
    ),
    group(
        "media-03",
        "media_streaming_operations",
        "focused",
        "subscription upgrade requests",
        metric(
            "subscription_upgrade_requests",
            (
                "Number of viewer subscription upgrade requests "
                "actually submitted."
            ),
        ),
        metric(
            "forecast_subscription_upgrade_requests",
            (
                "Forecast number of subscription upgrade "
                "requests expected to be submitted."
            ),
        ),
        metric(
            "subscription_upgrades_activated",
            (
                "Number of submitted upgrade requests later "
                "activated on customer accounts."
            ),
        ),
        metric(
            "upgrade_conversion_rate",
            (
                "Percentage of eligible upgrade interactions "
                "that result in an activated upgrade."
            ),
        ),
    ),

    # ========================================================
    # MUNICIPAL WASTE OPERATIONS
    # ========================================================
    group(
        "waste-01",
        "municipal_waste_operations",
        "full",
        "collection routes started",
        metric(
            "collection_routes_started",
            (
                "Number of municipal waste collection routes "
                "actually started."
            ),
        ),
        metric(
            "planned_collection_routes_started",
            (
                "Planned number of municipal waste collection "
                "routes to be started."
            ),
        ),
        metric(
            "collection_routes_completed",
            (
                "Number of started collection routes later "
                "completed."
            ),
        ),
        metric(
            "missed_pickup_count",
            (
                "Number of scheduled waste pickups reported "
                "as missed."
            ),
        ),
        metric(
            "office_water_cooler_refill_count",
            (
                "Number of water-cooler bottle replacements "
                "performed in an administrative office."
            ),
        ),
        metric(
            "route_quality_index",
            (
                "Internal collection-route quality index whose "
                "formula and components are undocumented."
            ),
        ),
    ),
    group(
        "waste-02",
        "municipal_waste_operations",
        "full",
        "recycling loads received",
        metric(
            "recycling_loads_received",
            (
                "Number of recycling collection loads actually "
                "received at the processing facility."
            ),
        ),
        metric(
            "forecast_recycling_loads_received",
            (
                "Forecast number of recycling loads expected "
                "to be received."
            ),
        ),
        metric(
            "recycling_loads_sorted",
            (
                "Number of received recycling loads later "
                "completed through sorting."
            ),
        ),
        metric(
            "recycling_contamination_rate_percent",
            (
                "Percentage of received recycling material "
                "classified as contamination."
            ),
        ),
        metric(
            "administrative_meeting_count",
            (
                "Number of internal administrative meetings "
                "recorded during the period."
            ),
        ),
        metric(
            "material_recovery_score",
            (
                "Internal material-recovery score whose formula "
                "and component metrics are undocumented."
            ),
        ),
    ),
    group(
        "waste-03",
        "municipal_waste_operations",
        "focused",
        "bulky waste requests received",
        metric(
            "bulky_waste_requests_received",
            (
                "Number of bulky-waste collection requests "
                "actually received from residents."
            ),
        ),
        metric(
            "forecast_bulky_waste_requests_received",
            (
                "Forecast number of bulky-waste collection "
                "requests expected to be received."
            ),
        ),
        metric(
            "bulky_waste_pickups_completed",
            (
                "Number of received bulky-waste requests later "
                "completed as pickups."
            ),
        ),
        metric(
            "average_bulky_pickup_lead_days",
            (
                "Average days between bulky-waste request "
                "receipt and completed collection."
            ),
        ),
    ),

    # ========================================================
    # AGRICULTURAL IRRIGATION OPERATIONS
    # 3 full
    # ========================================================
    group(
        "irrigation-01",
        "agricultural_irrigation_operations",
        "full",
        "irrigation cycles started",
        metric(
            "irrigation_cycles_started",
            (
                "Number of field irrigation cycles actually "
                "started."
            ),
        ),
        metric(
            "planned_irrigation_cycles_started",
            (
                "Planned number of field irrigation cycles "
                "to be started."
            ),
        ),
        metric(
            "irrigation_cycles_completed",
            (
                "Number of started irrigation cycles later "
                "completed."
            ),
        ),
        metric(
            "soil_moisture_percent",
            (
                "Measured volumetric soil moisture percentage "
                "in irrigated fields."
            ),
        ),
        metric(
            "tractor_radio_usage_minutes",
            (
                "Minutes of two-way radio usage recorded in "
                "agricultural tractors."
            ),
        ),
        metric(
            "irrigation_efficiency_index",
            (
                "Internal irrigation-efficiency index whose "
                "formula and included variables are undocumented."
            ),
        ),
    ),
    group(
        "irrigation-02",
        "agricultural_irrigation_operations",
        "full",
        "water allocation requests received",
        metric(
            "water_allocation_requests_received",
            (
                "Number of irrigation water-allocation requests "
                "actually received."
            ),
        ),
        metric(
            "forecast_water_allocation_requests_received",
            (
                "Forecast number of irrigation water-allocation "
                "requests expected to be received."
            ),
        ),
        metric(
            "water_allocations_released",
            (
                "Number of received water-allocation requests "
                "later released for field use."
            ),
        ),
        metric(
            "reservoir_level_percent",
            (
                "Measured reservoir storage level expressed as "
                "a percentage of capacity."
            ),
        ),
        metric(
            "barn_light_switch_event_count",
            (
                "Number of lighting switch events recorded "
                "inside an agricultural barn."
            ),
        ),
        metric(
            "water_priority_score",
            (
                "Internal water-priority score whose ranking "
                "logic and components are undocumented."
            ),
        ),
    ),
    group(
        "irrigation-03",
        "agricultural_irrigation_operations",
        "full",
        "pump maintenance tickets opened",
        metric(
            "pump_maintenance_tickets_opened",
            (
                "Number of irrigation pump maintenance tickets "
                "actually opened."
            ),
        ),
        metric(
            "forecast_pump_maintenance_tickets_opened",
            (
                "Forecast number of pump maintenance tickets "
                "expected to be opened."
            ),
        ),
        metric(
            "pump_maintenance_tickets_closed",
            (
                "Number of opened pump maintenance tickets "
                "later completed and closed."
            ),
        ),
        metric(
            "pump_energy_kwh",
            (
                "Electrical energy consumed by irrigation pumps "
                "during operation."
            ),
        ),
        metric(
            "seed_inventory_bag_count",
            (
                "Number of seed bags held in agricultural "
                "storage inventory."
            ),
        ),
        metric(
            "pump_condition_index",
            (
                "Internal pump-condition index whose inputs "
                "and calculation are undocumented."
            ),
        ),
    ),

    # ========================================================
    # RAIL FREIGHT OPERATIONS
    # ========================================================
    group(
        "rail-01",
        "rail_freight_operations",
        "full",
        "freight wagons received",
        metric(
            "freight_wagons_received",
            (
                "Number of freight wagons actually received "
                "into the rail yard."
            ),
        ),
        metric(
            "forecast_freight_wagons_received",
            (
                "Forecast number of freight wagons expected "
                "to be received into the yard."
            ),
        ),
        metric(
            "freight_wagons_dispatched",
            (
                "Number of received freight wagons later "
                "dispatched from the yard."
            ),
        ),
        metric(
            "yard_dwell_minutes",
            (
                "Average time freight wagons remain in the "
                "rail yard before dispatch."
            ),
        ),
        metric(
            "office_coffee_machine_cycle_count",
            (
                "Number of brewing cycles completed by an "
                "office coffee machine."
            ),
        ),
        metric(
            "yard_flow_index",
            (
                "Internal rail-yard flow index whose formula "
                "and units are undocumented."
            ),
        ),
    ),
    group(
        "rail-02",
        "rail_freight_operations",
        "full",
        "locomotive inspections opened",
        metric(
            "locomotive_inspections_opened",
            (
                "Number of locomotive inspections actually "
                "opened for maintenance review."
            ),
        ),
        metric(
            "planned_locomotive_inspections_opened",
            (
                "Planned number of locomotive inspections "
                "to be opened."
            ),
        ),
        metric(
            "locomotive_inspections_closed",
            (
                "Number of opened locomotive inspections "
                "later completed and closed."
            ),
        ),
        metric(
            "locomotive_defect_count",
            (
                "Number of locomotive defects identified "
                "during maintenance inspection."
            ),
        ),
        metric(
            "station_display_brightness_percent",
            (
                "Configured brightness percentage of an "
                "administrative station display."
            ),
        ),
        metric(
            "locomotive_readiness_score",
            (
                "Internal locomotive-readiness score whose "
                "components and scale are undocumented."
            ),
        ),
    ),
    group(
        "rail-03",
        "rail_freight_operations",
        "full",
        "intermodal units unloaded",
        metric(
            "intermodal_units_unloaded",
            (
                "Number of intermodal freight units actually "
                "unloaded from inbound trains."
            ),
        ),
        metric(
            "forecast_intermodal_units_unloaded",
            (
                "Forecast number of intermodal freight units "
                "expected to be unloaded."
            ),
        ),
        metric(
            "intermodal_units_loaded_to_outbound_train",
            (
                "Number of unloaded intermodal units later "
                "loaded onto outbound trains."
            ),
        ),
        metric(
            "crane_cycle_minutes",
            (
                "Average elapsed time required for an intermodal "
                "handling crane cycle."
            ),
        ),
        metric(
            "staff_locker_count",
            (
                "Number of employee lockers available in a "
                "rail facility changing room."
            ),
        ),
        metric(
            "transfer_quality_index",
            (
                "Internal intermodal-transfer quality index "
                "whose formula is undocumented."
            ),
        ),
    ),

    # ========================================================
    # DATA CENTER FACILITY OPERATIONS
    # ========================================================
    group(
        "datacenter-01",
        "data_center_facility_operations",
        "full",
        "incident tickets opened",
        metric(
            "facility_incident_tickets_opened",
            (
                "Number of data-center facility incident "
                "tickets actually opened."
            ),
        ),
        metric(
            "forecast_facility_incident_tickets_opened",
            (
                "Forecast number of data-center facility "
                "incident tickets expected to be opened."
            ),
        ),
        metric(
            "facility_incident_tickets_resolved",
            (
                "Number of opened facility incident tickets "
                "later resolved."
            ),
        ),
        metric(
            "mean_time_to_repair_minutes",
            (
                "Average elapsed time required to repair "
                "facility incidents."
            ),
        ),
        metric(
            "cafeteria_fridge_temperature_c",
            (
                "Measured temperature of an employee cafeteria "
                "refrigerator."
            ),
        ),
        metric(
            "service_health_index",
            (
                "Internal facility service-health index whose "
                "calculation and components are undocumented."
            ),
        ),
    ),
    group(
        "datacenter-02",
        "data_center_facility_operations",
        "full",
        "server rack deployments started",
        metric(
            "server_rack_deployments_started",
            (
                "Number of server-rack deployment activities "
                "actually started."
            ),
        ),
        metric(
            "planned_server_rack_deployments_started",
            (
                "Planned number of server-rack deployment "
                "activities to be started."
            ),
        ),
        metric(
            "server_rack_deployments_completed",
            (
                "Number of started server-rack deployments "
                "later completed."
            ),
        ),
        metric(
            "rack_power_draw_kw",
            (
                "Electrical power draw of active server racks "
                "measured in kilowatts."
            ),
        ),
        metric(
            "conference_room_booking_count",
            (
                "Number of office conference-room reservations "
                "created."
            ),
        ),
        metric(
            "deployment_readiness_score",
            (
                "Internal deployment-readiness score whose "
                "features and weighting are undocumented."
            ),
        ),
    ),
    group(
        "datacenter-03",
        "data_center_facility_operations",
        "full",
        "backup jobs started",
        metric(
            "backup_jobs_started",
            (
                "Number of scheduled data backup jobs actually "
                "started."
            ),
        ),
        metric(
            "planned_backup_jobs_started",
            (
                "Planned number of data backup jobs to be "
                "started."
            ),
        ),
        metric(
            "backup_jobs_completed",
            (
                "Number of started data backup jobs later "
                "completed successfully."
            ),
        ),
        metric(
            "backup_duration_minutes",
            (
                "Average elapsed time required for data backup "
                "jobs to complete."
            ),
        ),
        metric(
            "parking_sensor_event_count",
            (
                "Number of vehicle-presence events recorded by "
                "office parking sensors."
            ),
        ),
        metric(
            "data_protection_index",
            (
                "Internal data-protection index whose formula "
                "and input measures are undocumented."
            ),
        ),
    ),

    # ========================================================
    # LABORATORY SAMPLE OPERATIONS
    # ========================================================
    group(
        "laboratory-01",
        "laboratory_sample_operations",
        "full",
        "samples received",
        metric(
            "laboratory_samples_received",
            (
                "Number of laboratory samples actually received "
                "for processing."
            ),
        ),
        metric(
            "forecast_laboratory_samples_received",
            (
                "Forecast number of laboratory samples expected "
                "to be received."
            ),
        ),
        metric(
            "laboratory_samples_resulted",
            (
                "Number of received laboratory samples later "
                "completed with reported results."
            ),
        ),
        metric(
            "sample_turnaround_hours",
            (
                "Average elapsed hours between sample receipt "
                "and reported result."
            ),
        ),
        metric(
            "office_copier_toner_percent",
            (
                "Remaining toner percentage in an administrative "
                "office copier."
            ),
        ),
        metric(
            "laboratory_flow_index",
            (
                "Internal laboratory-flow index whose formula "
                "and components are undocumented."
            ),
        ),
    ),
    group(
        "laboratory-02",
        "laboratory_sample_operations",
        "full",
        "assays started",
        metric(
            "assays_started",
            (
                "Number of laboratory assays actually started "
                "during the processing period."
            ),
        ),
        metric(
            "planned_assays_started",
            (
                "Planned number of laboratory assays to be "
                "started."
            ),
        ),
        metric(
            "assays_verified",
            (
                "Number of started assays later verified and "
                "released by laboratory staff."
            ),
        ),
        metric(
            "quality_control_failure_count",
            (
                "Number of assay quality-control checks failing "
                "acceptance criteria."
            ),
        ),
        metric(
            "break_room_fridge_door_count",
            (
                "Number of door-opening events recorded on an "
                "employee break-room refrigerator."
            ),
        ),
        metric(
            "assay_quality_index",
            (
                "Internal assay-quality index whose constituent "
                "measures and weighting are undocumented."
            ),
        ),
    ),
    group(
        "laboratory-03",
        "laboratory_sample_operations",
        "full",
        "specimens aliquoted",
        metric(
            "specimens_aliquoted",
            (
                "Number of laboratory specimens actually "
                "divided into aliquots."
            ),
        ),
        metric(
            "forecast_specimens_aliquoted",
            (
                "Forecast number of laboratory specimens "
                "expected to be aliquoted."
            ),
        ),
        metric(
            "aliquots_analyzed",
            (
                "Number of created specimen aliquots later "
                "analyzed."
            ),
        ),
        metric(
            "reagent_consumption_ml",
            (
                "Volume of laboratory reagent consumed during "
                "sample analysis."
            ),
        ),
        metric(
            "staff_bicycle_rack_count",
            (
                "Number of bicycle parking positions available "
                "for laboratory staff."
            ),
        ),
        metric(
            "sample_processing_score",
            (
                "Internal sample-processing score whose formula "
                "and scale are undocumented."
            ),
        ),
    ),

    # ========================================================
    # PROCUREMENT OPERATIONS
    # ========================================================
    group(
        "procurement-01",
        "procurement_operations",
        "full",
        "purchase requisitions submitted",
        metric(
            "purchase_requisitions_submitted",
            (
                "Number of purchase requisitions actually "
                "submitted by internal requesters."
            ),
        ),
        metric(
            "forecast_purchase_requisitions_submitted",
            (
                "Forecast number of purchase requisitions "
                "expected to be submitted."
            ),
        ),
        metric(
            "purchase_orders_issued",
            (
                "Number of submitted purchase requisitions "
                "later converted into issued purchase orders."
            ),
        ),
        metric(
            "approval_cycle_hours",
            (
                "Average elapsed hours required to approve a "
                "purchase requisition."
            ),
        ),
        metric(
            "office_plant_watering_event_count",
            (
                "Number of plant-watering events recorded in "
                "administrative offices."
            ),
        ),
        metric(
            "sourcing_readiness_index",
            (
                "Internal sourcing-readiness index whose "
                "calculation and components are undocumented."
            ),
        ),
    ),
    group(
        "procurement-02",
        "procurement_operations",
        "full",
        "supplier quotes received",
        metric(
            "supplier_quotes_received",
            (
                "Number of supplier quotations actually "
                "received for sourcing events."
            ),
        ),
        metric(
            "forecast_supplier_quotes_received",
            (
                "Forecast number of supplier quotations "
                "expected to be received."
            ),
        ),
        metric(
            "supplier_quotes_approved",
            (
                "Number of received supplier quotations later "
                "approved for purchasing action."
            ),
        ),
        metric(
            "quoted_price_variance_percent",
            (
                "Percentage variation between supplier quoted "
                "prices and reference purchasing values."
            ),
        ),
        metric(
            "warehouse_dock_door_temperature_c",
            (
                "Measured surface temperature of a warehouse "
                "dock door."
            ),
        ),
        metric(
            "supplier_quality_index",
            (
                "Internal supplier-quality index whose inputs "
                "and weighting are undocumented."
            ),
        ),
    ),
    group(
        "procurement-03",
        "procurement_operations",
        "full",
        "goods receipts recorded",
        metric(
            "goods_receipts_recorded",
            (
                "Number of supplier goods receipts actually "
                "recorded after delivery."
            ),
        ),
        metric(
            "forecast_goods_receipts_recorded",
            (
                "Forecast number of supplier goods receipts "
                "expected to be recorded."
            ),
        ),
        metric(
            "supplier_invoices_matched",
            (
                "Number of received supplier deliveries later "
                "matched successfully to supplier invoices."
            ),
        ),
        metric(
            "invoice_match_exception_count",
            (
                "Number of supplier invoice matching exceptions "
                "recorded during procure-to-pay processing."
            ),
        ),
        metric(
            "office_keyboard_inventory_count",
            (
                "Number of spare computer keyboards held in "
                "office equipment inventory."
            ),
        ),
        metric(
            "procure_to_pay_health_score",
            (
                "Internal procure-to-pay health score whose "
                "formula and scale are undocumented."
            ),
        ),
    ),
]


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def word_count(
    value: str,
) -> int:
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            value,
        )
    )


def reason_for(
    *,
    relation: str,
    concept: str,
) -> str:
    if (
        relation
        ==
        "same_metric_different_state"
    ):
        return (
            f"Both metrics measure {concept}; the difference "
            "is operational state or timing, not the "
            "underlying quantity itself."
        )

    if (
        relation
        ==
        "same_process_different_stage"
    ):
        return (
            "Both metrics belong to the same operational "
            "process, but they describe different stages "
            "rather than the same underlying quantity."
        )

    if (
        relation
        ==
        "related_distinct_metric"
    ):
        return (
            "Both metrics concern the same operational context, "
            "but they measure distinct quantities rather than "
            "alternate states of one metric."
        )

    if relation == "unrelated":
        return (
            "The metrics describe separate operational concepts "
            "without a direct measurement identity or common "
            "process-stage relationship."
        )

    if relation == "uncertain":
        return (
            "The partner metric is insufficiently documented, "
            "so the semantic relationship cannot be established "
            "safely from the available definitions."
        )

    raise ValueError(
        f"Unsupported relation: {relation}"
    )


def validate_group_specs(
) -> None:
    if len(
        GROUP_SPECS
    ) != 50:
        raise RuntimeError(
            (
                "Expected exactly 50 manually authored "
                f"contrastive groups; got {len(GROUP_SPECS)}."
            )
        )

    group_ids = [
        item[
            "group_id"
        ]

        for item
        in GROUP_SPECS
    ]

    if len(
        set(
            group_ids
        )
    ) != 50:
        raise RuntimeError(
            "Contrastive group IDs are not unique."
        )

    observed_domain_groups = defaultdict(
        lambda: {
            "focused":
                0,

            "full":
                0,
        }
    )

    full_count = 0
    focused_count = 0

    for item in GROUP_SPECS:
        group_type = item[
            "group_type"
        ]

        domain = item[
            "domain"
        ]

        if (
            domain
            not in
            EXPECTED_DOMAIN_GROUPS
        ):
            raise RuntimeError(
                f"Unexpected training domain: {domain}"
            )

        if any(
            token
            in
            domain.casefold()

            for token
            in
            FORBIDDEN_DOMAIN_TOKENS
        ):
            raise RuntimeError(
                f"Forbidden evaluation domain: {domain}"
            )

        if (
            group_type
            not in
            (
                "full",
                "focused",
            )
        ):
            raise RuntimeError(
                f"Unsupported group type: {group_type}"
            )

        observed_domain_groups[
            domain
        ][
            group_type
        ] += 1

        if group_type == "full":
            full_count += 1
            required_relations = set(
                FULL_RELATIONS
            )

        else:
            focused_count += 1
            required_relations = set(
                FOCUSED_RELATIONS
            )

        if (
            set(
                item[
                    "partners"
                ]
            )
            !=
            required_relations
        ):
            raise RuntimeError(
                (
                    "Contrastive relation set mismatch: "
                    f"{item['group_id']}"
                )
            )

        anchor_metric = item[
            "anchor"
        ][
            "metric"
        ]

        if not anchor_metric:
            raise RuntimeError(
                "Anchor metric is empty."
            )

        partner_metrics = [
            partner[
                "metric"
            ]

            for partner
            in item[
                "partners"
            ].values()
        ]

        if len(
            set(
                partner_metrics
            )
        ) != len(
            partner_metrics
        ):
            raise RuntimeError(
                (
                    "Duplicate partner metric inside group: "
                    f"{item['group_id']}"
                )
            )

        if (
            anchor_metric
            in
            partner_metrics
        ):
            raise RuntimeError(
                (
                    "Anchor metric reused as partner: "
                    f"{item['group_id']}"
                )
            )

    if full_count != 40:
        raise RuntimeError(
            f"Expected 40 full groups; got {full_count}."
        )

    if focused_count != 10:
        raise RuntimeError(
            f"Expected 10 focused groups; got {focused_count}."
        )

    for domain, (
        expected_full,
        expected_focused,
    ) in EXPECTED_DOMAIN_GROUPS.items():
        observed = (
            observed_domain_groups[
                domain
            ]
        )

        if (
            observed[
                "full"
            ]
            !=
            expected_full
        ):
            raise RuntimeError(
                (
                    f"{domain}: expected {expected_full} "
                    "full groups, got "
                    f"{observed['full']}."
                )
            )

        if (
            observed[
                "focused"
            ]
            !=
            expected_focused
        ):
            raise RuntimeError(
                (
                    f"{domain}: expected {expected_focused} "
                    "focused groups, got "
                    f"{observed['focused']}."
                )
            )


def build_records(
) -> list[dict[str, Any]]:
    validate_group_specs()

    records = []

    example_number = 1

    for spec in GROUP_SPECS:
        relations = (
            FULL_RELATIONS
            if
            spec[
                "group_type"
            ]
            ==
            "full"
            else
            FOCUSED_RELATIONS
        )

        for relation in relations:
            partner = (
                spec[
                    "partners"
                ][
                    relation
                ]
            )

            example_id = (
                "adaptation:v0.4:"
                f"{example_number:03d}"
            )

            source_id = (
                "authoring:v0.4:"
                f"{spec['group_id']}:"
                f"{relation}"
            )

            records.append(
                {
                    "contrastive_group_id":
                        spec[
                            "group_id"
                        ],

                    "domain":
                        spec[
                            "domain"
                        ],

                    "example_id":
                        example_id,

                    "hard_negative":
                        (
                            relation
                            in
                            HARD_NEGATIVE_RELATIONS
                        ),

                    "left_description":
                        spec[
                            "anchor"
                        ][
                            "description"
                        ],

                    "left_metric":
                        spec[
                            "anchor"
                        ][
                            "metric"
                        ],

                    "provenance": {
                        "authoring_method":
                            (
                                "independent_manual_semantic_design"
                            ),

                        "source_artifact_paths":
                            [],

                        "source_dataset_ids":
                            [],

                        "source_ids":
                            [
                                source_id
                            ],
                    },

                    "right_description":
                        partner[
                            "description"
                        ],

                    "right_metric":
                        partner[
                            "metric"
                        ],

                    "target": {
                        "reason":
                            reason_for(
                                relation=
                                    relation,

                                concept=
                                    spec[
                                        "concept"
                                    ],
                            ),

                        "relation":
                            relation,
                    },
                }
            )

            example_number += 1

    return records


def validate_records(
    records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    if len(
        records
    ) != 230:
        raise RuntimeError(
            (
                "Expected 230 training examples; "
                f"got {len(records)}."
            )
        )

    required_fields = {
        "contrastive_group_id",
        "domain",
        "example_id",
        "hard_negative",
        "left_description",
        "left_metric",
        "provenance",
        "right_description",
        "right_metric",
        "target",
    }

    example_ids = []
    source_ids = []
    relation_counts = Counter()
    group_rows = defaultdict(
        list
    )
    domain_groups = defaultdict(
        set
    )
    pair_keys = set()

    for record in records:
        if (
            set(
                record
            )
            !=
            required_fields
        ):
            raise RuntimeError(
                (
                    "Training record schema mismatch: "
                    f"{record.get('example_id')}"
                )
            )

        example_id = record[
            "example_id"
        ]

        example_ids.append(
            example_id
        )

        relation = (
            record[
                "target"
            ][
                "relation"
            ]
        )

        if relation not in RELATIONS:
            raise RuntimeError(
                (
                    "Unknown target relation: "
                    f"{example_id}"
                )
            )

        relation_counts[
            relation
        ] += 1

        reason = (
            record[
                "target"
            ][
                "reason"
            ]
        )

        reason_words = word_count(
            reason
        )

        if not (
            6
            <=
            reason_words
            <=
            45
        ):
            raise RuntimeError(
                (
                    "Reason outside 6-45 words: "
                    f"{example_id} "
                    f"({reason_words})"
                )
            )

        if not isinstance(
            record[
                "hard_negative"
            ],
            bool,
        ):
            raise RuntimeError(
                (
                    "hard_negative must be boolean: "
                    f"{example_id}"
                )
            )

        if (
            len(
                record[
                    "left_description"
                ]
            )
            <
            35
        ):
            raise RuntimeError(
                (
                    "Left description too short: "
                    f"{example_id}"
                )
            )

        if (
            len(
                record[
                    "right_description"
                ]
            )
            <
            35
        ):
            raise RuntimeError(
                (
                    "Right description too short: "
                    f"{example_id}"
                )
            )

        domain = record[
            "domain"
        ]

        if (
            domain
            not in
            EXPECTED_DOMAIN_GROUPS
        ):
            raise RuntimeError(
                (
                    "Unexpected domain in training row: "
                    f"{domain}"
                )
            )

        serialized = json.dumps(
            record,
            ensure_ascii=True,
            sort_keys=True,
        ).casefold()

        if any(
            token
            in
            serialized

            for token
            in
            FORBIDDEN_DOMAIN_TOKENS
        ):
            raise RuntimeError(
                (
                    "Protected evaluation-domain token "
                    f"appears in training row: {example_id}"
                )
            )

        provenance = record[
            "provenance"
        ]

        if (
            set(
                provenance
            )
            !=
            {
                "authoring_method",
                "source_artifact_paths",
                "source_dataset_ids",
                "source_ids",
            }
        ):
            raise RuntimeError(
                (
                    "Provenance schema mismatch: "
                    f"{example_id}"
                )
            )

        if (
            provenance[
                "authoring_method"
            ]
            !=
            "independent_manual_semantic_design"
        ):
            raise RuntimeError(
                (
                    "Unexpected authoring method: "
                    f"{example_id}"
                )
            )

        if (
            provenance[
                "source_artifact_paths"
            ]
            !=
            []
        ):
            raise RuntimeError(
                (
                    "Source artifact paths must be empty: "
                    f"{example_id}"
                )
            )

        if (
            provenance[
                "source_dataset_ids"
            ]
            !=
            []
        ):
            raise RuntimeError(
                (
                    "Source dataset IDs must be empty: "
                    f"{example_id}"
                )
            )

        ids = provenance[
            "source_ids"
        ]

        if (
            not isinstance(
                ids,
                list,
            )
            or
            len(
                ids
            )
            !=
            1
            or
            not isinstance(
                ids[
                    0
                ],
                str,
            )
        ):
            raise RuntimeError(
                (
                    "Exactly one authoring source ID "
                    f"is required: {example_id}"
                )
            )

        source_ids.extend(
            ids
        )

        pair_key = (
            record[
                "left_metric"
            ].casefold(),
            record[
                "right_metric"
            ].casefold(),
        )

        if pair_key in pair_keys:
            raise RuntimeError(
                (
                    "Duplicate ordered metric pair: "
                    f"{example_id}"
                )
            )

        pair_keys.add(
            pair_key
        )

        group_id = record[
            "contrastive_group_id"
        ]

        group_rows[
            group_id
        ].append(
            record
        )

        domain_groups[
            domain
        ].add(
            group_id
        )

    if len(
        set(
            example_ids
        )
    ) != 230:
        raise RuntimeError(
            "Training example IDs are not unique."
        )

    if len(
        set(
            source_ids
        )
    ) != 230:
        raise RuntimeError(
            "Training provenance source IDs are not unique."
        )

    if (
        relation_counts
        !=
        Counter(
            EXPECTED_RELATION_COUNTS
        )
    ):
        raise RuntimeError(
            (
                "Relation distribution mismatch.\n"
                f"{dict(relation_counts)}"
            )
        )

    if len(
        group_rows
    ) != 50:
        raise RuntimeError(
            (
                "Expected 50 materialized groups; "
                f"got {len(group_rows)}."
            )
        )

    spec_by_id = {
        spec[
            "group_id"
        ]:
            spec

        for spec
        in GROUP_SPECS
    }

    for group_id, rows in group_rows.items():
        spec = spec_by_id[
            group_id
        ]

        expected_relations = (
            set(
                FULL_RELATIONS
            )
            if
            spec[
                "group_type"
            ]
            ==
            "full"
            else
            set(
                FOCUSED_RELATIONS
            )
        )

        if (
            {
                row[
                    "target"
                ][
                    "relation"
                ]

                for row
                in rows
            }
            !=
            expected_relations
        ):
            raise RuntimeError(
                (
                    "Materialized relation set mismatch: "
                    f"{group_id}"
                )
            )

        expected_size = (
            5
            if
            spec[
                "group_type"
            ]
            ==
            "full"
            else
            3
        )

        if len(
            rows
        ) != expected_size:
            raise RuntimeError(
                (
                    "Materialized group size mismatch: "
                    f"{group_id}"
                )
            )

        anchors = {
            (
                row[
                    "left_metric"
                ],
                row[
                    "left_description"
                ],
            )

            for row
            in rows
        }

        if len(
            anchors
        ) != 1:
            raise RuntimeError(
                (
                    "Same-anchor contract violated: "
                    f"{group_id}"
                )
            )

        hard_negative_count = sum(
            row[
                "hard_negative"
            ]

            for row
            in rows
        )

        minimum = (
            2
            if
            spec[
                "group_type"
            ]
            ==
            "full"
            else
            1
        )

        if (
            hard_negative_count
            <
            minimum
        ):
            raise RuntimeError(
                (
                    "Hard-negative minimum violated: "
                    f"{group_id}"
                )
            )

    if len(
        domain_groups
    ) != 15:
        raise RuntimeError(
            "Expected 15 training domains."
        )

    for domain, groups in domain_groups.items():
        if len(
            groups
        ) < 3:
            raise RuntimeError(
                (
                    "Domain has fewer than three "
                    f"contrastive groups: {domain}"
                )
            )


def write_jsonl_lf(
    *,
    path: Path,
    records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    if path.exists():
        raise FileExistsError(
            (
                "Refusing to overwrite authored "
                f"training dataset: {path}"
            )
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = []

    for record in records:
        lines.append(
            json.dumps(
                record,
                ensure_ascii=True,
                separators=(
                    ",",
                    ":",
                ),
                sort_keys=True,
            )
        )

    payload = (
        "\n".join(
            lines
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )

    path.write_bytes(
        payload
    )


def load_jsonl(
    path: Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    records = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            raise RuntimeError(
                (
                    "Blank JSONL line at "
                    f"{line_number}."
                )
            )

        value = json.loads(
            line
        )

        if not isinstance(
            value,
            dict,
        ):
            raise RuntimeError(
                (
                    "JSONL row is not an object at "
                    f"line {line_number}."
                )
            )

        records.append(
            value
        )

    return records


def validate_design_bindings(
) -> None:
    if (
        sha256_file(
            DESIGN_PATH
        )
        !=
        EXPECTED_DESIGN_SHA256
    ):
        raise RuntimeError(
            "Training Dataset Design SHA mismatch."
        )

    if (
        sha256_file(
            DESIGN_FREEZE_PATH
        )
        !=
        EXPECTED_DESIGN_FREEZE_SHA256
    ):
        raise RuntimeError(
            "Training Dataset Design freeze SHA mismatch."
        )

    design = load_json_object(
        DESIGN_PATH
    )

    if (
        design[
            "expected_example_count"
        ]
        !=
        230
    ):
        raise RuntimeError(
            "Frozen design example count mismatch."
        )

    if (
        design[
            "relation_target_counts"
        ]
        !=
        EXPECTED_RELATION_COUNTS
    ):
        raise RuntimeError(
            "Frozen design relation counts mismatch."
        )

    if (
        design[
            "status"
        ]
        !=
        "frozen_before_training_example_authoring"
    ):
        raise RuntimeError(
            "Training Dataset Design was not frozen before authoring."
        )


def build_dataset(
) -> list[
    dict[
        str,
        Any,
    ]
]:
    validate_design_bindings()

    records = build_records()

    validate_records(
        records
    )

    write_jsonl_lf(
        path=
            DATASET_PATH,

        records=
            records,
    )

    return records


def validate_existing_dataset(
) -> list[
    dict[
        str,
        Any,
    ]
]:
    validate_design_bindings()

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            DATASET_PATH
        )

    records = load_jsonl(
        DATASET_PATH
    )

    validate_records(
        records
    )

    expected = build_records()

    if records != expected:
        raise RuntimeError(
            (
                "Authored dataset does not "
                "recompute deterministically."
            )
        )

    return records


def main(
) -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=(
            "build",
            "validate",
        ),
    )

    args = parser.parse_args()

    if args.mode == "build":
        records = build_dataset()

        print(
            "DATALENS QLORA v0.4 TRAINING DATASET AUTHORING BUILD: PASS"
        )

        print(
            f"Examples: {len(records)}"
        )

        print(
            f"Dataset SHA256: {sha256_file(DATASET_PATH)}"
        )

        return

    records = validate_existing_dataset()

    print(
        "DATALENS QLORA v0.4 TRAINING DATASET AUTHORING VALIDATION: PASS"
    )

    print(
        f"Examples: {len(records)}"
    )

    print(
        f"Dataset SHA256: {sha256_file(DATASET_PATH)}"
    )


if __name__ == "__main__":
    main()

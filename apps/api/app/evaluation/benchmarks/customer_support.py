from __future__ import annotations

import numpy as np
import pandas as pd

from app.evaluation.registry_schemas import (
    BenchmarkDatasetSpec,
    SemanticBenchmarkSuite,
)

from app.evaluation.schemas import (
    SemanticColumnBenchmarkCase,
    SemanticFieldExpectation,
    SemanticPairBenchmarkCase,
)


# ============================================================
# IDENTIFIERS
# ============================================================

CUSTOMER_SUPPORT_BENCHMARK_ID = (
    "semantic:customer_support:holdout:v0.1"
)


CUSTOMER_SUPPORT_DATASET_ID = (
    "customer_support:0001"
)


CUSTOMER_SUPPORT_FILENAME = (
    "synthetic_customer_support_operations.csv"
)


# ============================================================
# COLUMN NAMES
# ============================================================

CASES_SUBMITTED = (
    "Cases submitted"
)


TICKETS_RESOLVED = (
    "Tickets resolved"
)


AGENTS_ROSTERED = (
    "Agents rostered"
)


STAFF_ONLINE = (
    "Staff online"
)


QUOTED_SERVICE_COST = (
    "Quoted service cost"
)


FINAL_INVOICE_AMOUNT = (
    "Final invoice amount"
)


PROMISED_RESPONSE_TIME = (
    "Promised response time (minutes)"
)


OBSERVED_RESPONSE_DELAY = (
    "Observed first-response delay (minutes)"
)


SLA_COMPLIANCE = (
    "SLA compliance (%)"
)


# ============================================================
# SYNTHETIC FIXTURE
# ============================================================

def build_customer_support_benchmark_dataframe(
) -> pd.DataFrame:
    n = 240


    index = np.arange(
        n
    )


    cases_submitted = (
        78
        +
        (
            index
            %
            25
        )
    ).astype(
        int
    )


    resolution_gap = (
        index
        %
        8
    ).astype(
        int
    )


    tickets_resolved = (
        cases_submitted
        -
        resolution_gap
    ).astype(
        int
    )


    agents_rostered = (
        18
        +
        (
            index
            %
            9
        )
    ).astype(
        int
    )


    unavailable_agents = (
        index
        %
        4
    ).astype(
        int
    )


    staff_online = (
        agents_rostered
        -
        unavailable_agents
    ).astype(
        int
    )


    quoted_service_cost = (
        140.0
        +
        cases_submitted
        *
        1.65
    )


    final_invoice_amount = (
        quoted_service_cost
        *
        (
            0.94
            +
            (
                index
                %
                10
            )
            *
            0.012
        )
    )


    promised_response_time = (
        20.0
        +
        (
            index
            %
            7
        )
        *
        5.0
    )


    observed_response_delay = (
        promised_response_time
        *
        (
            0.72
            +
            (
                index
                %
                12
            )
            *
            0.055
        )
    )


    sla_compliance = (
        97.0
        -
        (
            observed_response_delay
            /
            promised_response_time
        )
        *
        11.0
    )


    sla_compliance = np.clip(
        sla_compliance,
        72.0,
        99.0,
    )


    return pd.DataFrame(
        {
            CASES_SUBMITTED:
                cases_submitted,

            TICKETS_RESOLVED:
                tickets_resolved,

            AGENTS_ROSTERED:
                agents_rostered,

            STAFF_ONLINE:
                staff_online,

            QUOTED_SERVICE_COST:
                quoted_service_cost,

            FINAL_INVOICE_AMOUNT:
                final_invoice_amount,

            PROMISED_RESPONSE_TIME:
                promised_response_time,

            OBSERVED_RESPONSE_DELAY:
                observed_response_delay,

            SLA_COMPLIANCE:
                sla_compliance,
        }
    )


# ============================================================
# COLUMN CASES
#
# FROZEN BEFORE THE FIRST S3 EXECUTION.
#
# PURPOSE
#
# This holdout intentionally avoids testing DataLens-only
# quantity fields.
#
# All expectations concern capabilities available to both:
#
# - raw Gemma semantic profiles
# - normalized DataLens semantic profiles
#
# The benchmark therefore provides a cleaner shared-capability
# comparison than the previous quantity-focused holdout.
# ============================================================

def build_customer_support_column_cases(
) -> list[
    SemanticColumnBenchmarkCase
]:
    return [
        SemanticColumnBenchmarkCase(
            case_id=
                "cases_submitted",

            dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            column=
                CASES_SUBMITTED,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "submitted",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "count",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "count",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "tickets_resolved",

            dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            column=
                TICKETS_RESOLVED,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "resolved",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "count",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "count",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "agents_rostered",

            dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            column=
                AGENTS_ROSTERED,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "rostered",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "count",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "count",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "staff_online",

            dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            column=
                STAFF_ONLINE,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "online",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "count",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "count",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "quoted_service_cost",

            dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            column=
                QUOTED_SERVICE_COST,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "quoted",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "currency",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "currency",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "final_invoice_amount",

            dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            column=
                FINAL_INVOICE_AMOUNT,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "final",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "currency",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "currency",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "promised_response_time",

            dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            column=
                PROMISED_RESPONSE_TIME,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "promised",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "duration",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "duration",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "observed_response_delay",

            dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            column=
                OBSERVED_RESPONSE_DELAY,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "variant",

                    accepted_values=[
                        "actual",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "duration",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "duration",
                    ],
                ),
            ],
        ),

        SemanticColumnBenchmarkCase(
            case_id=
                "sla_compliance",

            dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            column=
                SLA_COMPLIANCE,

            expectations=[
                SemanticFieldExpectation(
                    field=
                        "measure_kind",

                    accepted_values=[
                        "percentage",
                    ],
                ),

                SemanticFieldExpectation(
                    field=
                        "unit_kind",

                    accepted_values=[
                        "percent",
                    ],
                ),
            ],
        ),
    ]


# ============================================================
# PAIR CASES
#
# The positive pairs intentionally use different lexical
# formulations for related concepts.
#
# This tests whether DataLens can preserve / infer conceptual
# relationships rather than relying only on exact shared words.
# ============================================================

def build_customer_support_pair_cases(
) -> list[
    SemanticPairBenchmarkCase
]:
    return [
        # ----------------------------------------------------
        # Support demand flow
        #
        # submitted cases
        # resolved tickets
        #
        # Same operational family, different lexical nouns.
        # ----------------------------------------------------

        SemanticPairBenchmarkCase(
            case_id=
                "submitted_cases_resolved_tickets",

            left_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            right_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            left_column=
                CASES_SUBMITTED,

            right_column=
                TICKETS_RESOLVED,

            same_concept_family=
                True,

            same_domain=
                True,

            distinct_variants=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                True,
        ),

        # ----------------------------------------------------
        # Workforce capacity
        #
        # rostered agents
        # online staff
        # ----------------------------------------------------

        SemanticPairBenchmarkCase(
            case_id=
                "rostered_agents_online_staff",

            left_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            right_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            left_column=
                AGENTS_ROSTERED,

            right_column=
                STAFF_ONLINE,

            same_concept_family=
                True,

            same_domain=
                True,

            distinct_variants=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                True,
        ),

        # ----------------------------------------------------
        # Commercial amount
        #
        # quoted service cost
        # final invoice amount
        # ----------------------------------------------------

        SemanticPairBenchmarkCase(
            case_id=
                "quoted_cost_final_invoice",

            left_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            right_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            left_column=
                QUOTED_SERVICE_COST,

            right_column=
                FINAL_INVOICE_AMOUNT,

            same_concept_family=
                True,

            same_domain=
                True,

            distinct_variants=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                True,
        ),

        # ----------------------------------------------------
        # Response latency
        #
        # promised response time
        # observed response delay
        # ----------------------------------------------------

        SemanticPairBenchmarkCase(
            case_id=
                "promised_observed_response_time",

            left_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            right_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            left_column=
                PROMISED_RESPONSE_TIME,

            right_column=
                OBSERVED_RESPONSE_DELAY,

            same_concept_family=
                True,

            same_domain=
                True,

            distinct_variants=
                True,

            compatible_units=
                True,

            derived_gap_compatible=
                True,
        ),

        # ----------------------------------------------------
        # Related business domain, incompatible quantities.
        # ----------------------------------------------------

        SemanticPairBenchmarkCase(
            case_id=
                "response_delay_sla_compliance",

            left_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            right_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            left_column=
                OBSERVED_RESPONSE_DELAY,

            right_column=
                SLA_COMPLIANCE,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),

        SemanticPairBenchmarkCase(
            case_id=
                "invoice_online_staff",

            left_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            right_dataset_id=
                CUSTOMER_SUPPORT_DATASET_ID,

            left_column=
                FINAL_INVOICE_AMOUNT,

            right_column=
                STAFF_ONLINE,

            same_concept_family=
                False,

            same_domain=
                True,

            compatible_units=
                False,

            derived_gap_compatible=
                False,
        ),
    ]


# ============================================================
# SUITE
# ============================================================

def build_customer_support_semantic_benchmark(
) -> SemanticBenchmarkSuite:
    return SemanticBenchmarkSuite(
        benchmark_id=
            CUSTOMER_SUPPORT_BENCHMARK_ID,

        name=
            "Customer support conceptual semantic holdout",

        domain=
            "customer_support",

        split=
            "holdout",

        description=(
            "Fifth frozen out-of-domain semantic holdout. "
            "It evaluates whether DataLens Semantic System "
            "S3 generalizes to lexical and conceptual "
            "paraphrases in customer-support operations. "
            "Unlike Electric Mobility, every assertion in "
            "this benchmark targets capabilities shared by "
            "the raw LLM and DataLens."
        ),

        datasets=[
            BenchmarkDatasetSpec(
                dataset_id=
                    CUSTOMER_SUPPORT_DATASET_ID,

                filename=
                    CUSTOMER_SUPPORT_FILENAME,
            ),
        ],

        column_cases=
            build_customer_support_column_cases(),

        pair_cases=
            build_customer_support_pair_cases(),

        safety_critical_fields=[
            "compatible_units",
            "derived_gap_compatible",
        ],

        tags=[
            "customer_support",
            "lexical_generalization",
            "conceptual_generalization",
            "paraphrase",
            "shared_capabilities",
            "fair_baseline",
            "count",
            "currency",
            "duration",
            "percentage",
            "holdout",
            "generalization",
            "semantic_safety",
        ],

        benchmark_version=
            "customer_support_semantic_holdout_v0.1",
    )

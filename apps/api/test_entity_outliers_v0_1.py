from __future__ import annotations


import pandas as pd


from app.analysis.entity_outliers import (
    ENTITY_OUTLIER_RULE_VERSION,
    detect_entity_outliers,
)


# ============================================================
# FIXTURE
#
# Synthetic customer-grain analytical view.
#
# The four deliberately extreme entity IDs use the same IDs
# that we want a realistic Lapage-style test to be capable of
# surfacing later.
# ============================================================


ATYPICAL_CLIENTS = {
    "c_1609",
    "c_4958",
    "c_6714",
    "c_3454",
}


def build_customer_view(
) -> pd.DataFrame:
    rows: list[
        dict
    ] = []


    # ========================================================
    # NORMAL CUSTOMERS
    # ========================================================

    for index in range(
        1,
        201,
    ):
        rows.append(
            {
                "client_id":
                    f"c_normal_{index:04d}",

                "total_spend":
                    float(
                        1000
                        +
                        (
                            index
                            %
                            20
                        )
                        *
                        100
                    ),

                "purchase_sessions":
                    float(
                        3
                        +
                        (
                            index
                            %
                            7
                        )
                    ),

                "average_basket":
                    float(
                        25
                        +
                        (
                            index
                            %
                            15
                        )
                    ),

                "median_basket":
                    float(
                        23
                        +
                        (
                            index
                            %
                            13
                        )
                    ),

                "total_items":
                    float(
                        8
                        +
                        (
                            index
                            %
                            20
                        )
                    ),

                "average_items_per_basket":
                    float(
                        1.5
                        +
                        (
                            index
                            %
                            5
                        )
                        *
                        0.25
                    ),

                # Contextual, deliberately excluded.
                "age_at_first_purchase":
                    float(
                        20
                        +
                        (
                            index
                            %
                            55
                        )
                    ),
            }
        )


    # ========================================================
    # DELIBERATELY ATYPICAL CUSTOMERS
    # ========================================================

    rows.extend(
        [
            {
                "client_id":
                    "c_1609",

                "total_spend":
                    326040.0,

                "purchase_sessions":
                    750.0,

                "average_basket":
                    434.72,

                "median_basket":
                    410.0,

                "total_items":
                    8400.0,

                "average_items_per_basket":
                    11.2,

                "age_at_first_purchase":
                    31.0,
            },

            {
                "client_id":
                    "c_4958",

                "total_spend":
                    290227.0,

                "purchase_sessions":
                    690.0,

                "average_basket":
                    420.62,

                "median_basket":
                    398.0,

                "total_items":
                    7900.0,

                "average_items_per_basket":
                    10.8,

                "age_at_first_purchase":
                    44.0,
            },

            {
                "client_id":
                    "c_6714",

                "total_spend":
                    153919.0,

                "purchase_sessions":
                    410.0,

                "average_basket":
                    375.41,

                "median_basket":
                    350.0,

                "total_items":
                    4700.0,

                "average_items_per_basket":
                    9.9,

                "age_at_first_purchase":
                    37.0,
            },

            {
                "client_id":
                    "c_3454",

                "total_spend":
                    114111.0,

                "purchase_sessions":
                    330.0,

                "average_basket":
                    345.79,

                "median_basket":
                    330.0,

                "total_items":
                    3900.0,

                "average_items_per_basket":
                    9.4,

                "age_at_first_purchase":
                    29.0,
            },
        ]
    )


    return (
        pd.DataFrame(
            rows
        )
    )


def build_raw_transactions(
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "client_id": [
                    "c_1",
                    "c_1",
                    "c_2",
                ],

                "price": [
                    10.0,
                    20.0,
                    30.0,
                ],
            }
        )
    )


# ============================================================
# TEST
# ============================================================


def main(
) -> None:
    customer_view = (
        build_customer_view()
    )


    datasets = [
        # ----------------------------------------------------
        # RAW DATASET
        #
        # Must NOT be interpreted directly at client grain.
        # ----------------------------------------------------

        {
            "dataset_id":
                "dataset:transactions",

            "filename":
                "transactions.csv",

            "dataframe":
                build_raw_transactions(),

            "is_derived":
                False,
        },


        # ----------------------------------------------------
        # CUSTOMER-GRAIN ANALYTICAL VIEW
        # ----------------------------------------------------

        {
            "dataset_id":
                (
                    "derived:transactions:"
                    "customer:client_id:price"
                ),

            "filename":
                (
                    "Transactions"
                    "__customers_price.derived"
                ),

            "dataframe":
                customer_view,

            "is_derived":
                True,

            "derivation_type":
                "entity_additive_measure",

            "source_dataset_ids": [
                "dataset:transactions",
                "dataset:products",
                "dataset:customers",
            ],

            "provenance": {
                "operation":
                    (
                        "customer_behavior_"
                        "materialization"
                    ),

                "entity_column":
                    "client_id",

                "grain":
                    "client_id",

                "target_measure_column":
                    "total_spend",

                "aggregation_path": [
                    "fact rows -> session_id",
                    "session_id -> client_id",
                ],
            },
        },
    ]


    report = (
        detect_entity_outliers(
            datasets=
                datasets,

            top_limit=
                25,
        )
    )


    # ========================================================
    # BASIC CONTRACT
    # ========================================================

    assert (
        ENTITY_OUTLIER_RULE_VERSION
        ==
        "entity_outlier_engine_v0.1"
    )


    assert (
        report.status
        ==
        "ready"
    )


    assert (
        report.candidate_view_count
        ==
        1
    )


    assert (
        report.evaluated_view_count
        ==
        1
    )


    assert (
        len(
            report.results
        )
        ==
        1
    )


    result = (
        report.results[
            0
        ]
    )


    # ========================================================
    # GRAIN
    # ========================================================

    assert (
        result.entity_column
        ==
        "client_id"
    )


    assert (
        result.entity_count
        ==
        len(
            customer_view
        )
    )


    assert (
        result.primary_metric
        ==
        "total_spend"
    )


    # ========================================================
    # BEHAVIOURAL METRICS
    # ========================================================

    expected_metrics = {
        "total_spend",
        "purchase_sessions",
        "average_basket",
        "median_basket",
        "total_items",
        "average_items_per_basket",
    }


    assert (
        set(
            result.evaluated_metrics
        )
        ==
        expected_metrics
    )


    # Contextual age must not participate in the anomaly score.
    assert (
        "age_at_first_purchase"
        not in
        result.evaluated_metrics
    )


    # ========================================================
    # OUTLIERS
    # ========================================================

    detected_entities = {
        candidate.entity

        for candidate
        in result.top_entities
    }


    assert (
        ATYPICAL_CLIENTS
        .issubset(
            detected_entities
        )
    )


    # With this controlled fixture only those four entities
    # should cross the IQR boundaries.
    assert (
        result.flagged_entity_count
        ==
        4
    )


    assert (
        report.total_flagged_entity_count
        ==
        4
    )


    # ========================================================
    # EVIDENCE
    # ========================================================

    for candidate in (
        result.top_entities
    ):
        assert (
            candidate.entity
            in
            ATYPICAL_CLIENTS
        )


        assert (
            candidate.outlier_metric_count
            >=
            1
        )


        assert (
            candidate.anomaly_score
            >
            0.0
        )


        assert (
            candidate.evidence
        )


        assert any(
            evidence.metric
            ==
            "total_spend"

            for evidence
            in candidate.evidence
        )


        assert all(
            evidence.direction
            ==
            "high"

            for evidence
            in candidate.evidence
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "DataLens Entity Outlier Engine v0.1"
    )

    print(
        "=============================================="
    )


    print()
    print(
        f"Candidate views : "
        f"{report.candidate_view_count}"
    )

    print(
        f"Evaluated views : "
        f"{report.evaluated_view_count}"
    )

    print(
        f"Entity grain    : "
        f"{result.entity_column}"
    )

    print(
        f"Entities        : "
        f"{result.entity_count}"
    )

    print(
        f"Metrics         : "
        f"{len(result.evaluated_metrics)}"
    )

    print(
        f"Primary metric  : "
        f"{result.primary_metric}"
    )

    print(
        f"Flagged         : "
        f"{result.flagged_entity_count}"
    )


    print()
    print(
        "=== ATYPICAL ENTITIES ==="
    )


    for candidate in (
        result.top_entities
    ):
        print(
            (
                f"{candidate.entity:<10} "
                f"score="
                f"{candidate.anomaly_score:.2f} "
                f"metrics="
                f"{candidate.outlier_metric_count}"
            )
        )


        for evidence in (
            candidate.evidence
        ):
            print(
                (
                    "  - "
                    f"{evidence.metric}: "
                    f"{evidence.value:.2f} "
                    f"({evidence.direction}, "
                    f"{evidence.distance_iqr:.2f} "
                    "IQR beyond threshold)"
                )
            )


    print()
    print(
        "=============================================="
    )

    print(
        "PASS - entity outlier engine v0.1"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()
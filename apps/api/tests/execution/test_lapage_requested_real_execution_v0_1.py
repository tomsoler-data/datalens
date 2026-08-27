from __future__ import annotations


from collections import (
    Counter,
)


import pandas as pd


from app.execution.requested_executor import (
    execute_requested_analysis_plan,
)

from tests.planning.test_lapage_request_planning_coverage_v0_1 import (
    build_lapage_plan,
)


# ============================================================
# EXPECTED REQUEST GROUPS
# ============================================================

QUANTITATIVE_KINDS = {
    "age_total_amount_association",
    "age_frequency_association",
    "age_average_basket_association",
}


CONTEXT_KINDS = {
    "gender_category_association",
    "age_category_association",
}


DIRECT_BRIEF_KINDS = {
    "revenue_by_category",
    "customers_by_period",
    "transaction_count",
    "products_sold_count",
    "product_category_distribution",
    "lorenz_curve",
}


NON_READY_KINDS = {
    "revenue_moving_average",
    "top_products",
    "flop_products",
    "b2b_revenue_distribution",
}


READY_KINDS = (
    QUANTITATIVE_KINDS
    |
    CONTEXT_KINDS
    |
    DIRECT_BRIEF_KINDS
)


# ============================================================
# ASSERTIONS
# ============================================================

def assert_equal(
    actual,
    expected,
    message: str,
) -> None:
    if (
        actual
        !=
        expected
    ):
        raise AssertionError(
            (
                f"{message}\n"
                f"Expected: {expected!r}\n"
                f"Actual:   {actual!r}"
            )
        )


def assert_true(
    value,
    message: str,
) -> None:
    if not value:
        raise AssertionError(
            message
        )


def pass_test(
    message: str,
) -> None:
    print(
        f"[PASS] {message}"
    )


# ============================================================
# SOURCE DATAFRAMES
# ============================================================

def build_transactions_dataframe(
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "client_id": [
                    "c01",
                    "c01",
                    "c02",
                    "c02",
                    "c03",
                    "c03",
                    "c04",
                    "c04",
                    "c05",
                    "c05",
                    "c06",
                    "c06",
                    "c07",
                    "c07",
                    "c08",
                    "c08",
                    "c09",
                    "c09",
                    "c10",
                    "c10",
                    "c11",
                    "c11",
                    "c12",
                    "c12",
                ],

                "id_prod": [
                    "p01",
                    "p02",
                    "p02",
                    "p03",
                    "p03",
                    "p04",
                    "p04",
                    "p05",
                    "p05",
                    "p06",
                    "p06",
                    "p01",
                    "p01",
                    "p03",
                    "p02",
                    "p04",
                    "p03",
                    "p05",
                    "p04",
                    "p06",
                    "p05",
                    "p01",
                    "p06",
                    "p02",
                ],

                "session_id": [
                    "s001",
                    "s001",
                    "s002",
                    "s003",
                    "s004",
                    "s005",
                    "s006",
                    "s007",
                    "s008",
                    "s009",
                    "s010",
                    "s011",
                    "s012",
                    "s013",
                    "s014",
                    "s015",
                    "s016",
                    "s017",
                    "s018",
                    "s019",
                    "s020",
                    "s021",
                    "s022",
                    "s023",
                ],

                "date": pd.to_datetime(
                    [
                        "2024-01-05",
                        "2024-01-20",
                        "2024-01-11",
                        "2024-02-02",
                        "2024-01-18",
                        "2024-02-14",
                        "2024-02-03",
                        "2024-03-08",
                        "2024-02-21",
                        "2024-03-19",
                        "2024-03-01",
                        "2024-04-04",
                        "2024-03-13",
                        "2024-04-16",
                        "2024-04-02",
                        "2024-05-05",
                        "2024-04-27",
                        "2024-05-15",
                        "2024-05-04",
                        "2024-06-01",
                        "2024-05-19",
                        "2024-06-12",
                        "2024-06-03",
                        "2024-06-21",
                    ]
                ),
            }
        )
    )


def build_customers_dataframe(
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "client_id": [
                    "c01",
                    "c02",
                    "c03",
                    "c04",
                    "c05",
                    "c06",
                    "c07",
                    "c08",
                    "c09",
                    "c10",
                    "c11",
                    "c12",
                ],

                "sex": [
                    "f",
                    "m",
                    "f",
                    "m",
                    "f",
                    "m",
                    "f",
                    "m",
                    "f",
                    "m",
                    "f",
                    "m",
                ],

                "birth": [
                    1999,
                    1999,
                    1994,
                    1994,
                    1989,
                    1989,
                    1984,
                    1984,
                    1979,
                    1979,
                    1974,
                    1974,
                ],
            }
        )
    )


def build_products_dataframe(
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "id_prod": [
                    "p01",
                    "p02",
                    "p03",
                    "p04",
                    "p05",
                    "p06",
                ],

                "price": [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                ],

                "categ": [
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                ],
            }
        )
    )


# ============================================================
# CUSTOMER-GRAIN VIEW
# ============================================================

def build_customer_grain_dataframe(
) -> pd.DataFrame:
    """
    One row per customer.

    This is the analytical view expected by:

    - age × total spend
    - age × purchase frequency
    - age × average basket
    - Lorenz curve

    customer_id is deliberately the first entity-like
    column and is unique.
    """

    total_spend = [
        100.0,
        125.0,
        160.0,
        205.0,
        250.0,
        310.0,
        365.0,
        430.0,
        505.0,
        590.0,
        680.0,
        790.0,
    ]


    purchase_sessions = [
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
    ]


    average_basket = [
        total / sessions

        for (
            total,
            sessions,
        )
        in zip(
            total_spend,
            purchase_sessions,
        )
    ]


    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c01",
                    "c02",
                    "c03",
                    "c04",
                    "c05",
                    "c06",
                    "c07",
                    "c08",
                    "c09",
                    "c10",
                    "c11",
                    "c12",
                ],

                "age_at_first_purchase": [
                    25,
                    25,
                    30,
                    30,
                    35,
                    35,
                    40,
                    40,
                    45,
                    45,
                    50,
                    50,
                ],

                "total_spend":
                    total_spend,

                "purchase_sessions":
                    purchase_sessions,

                "average_basket":
                    average_basket,
            }
        )
    )


# ============================================================
# REQUESTED EVENT CONTEXT
# ============================================================

def build_requested_event_context_dataframe(
) -> pd.DataFrame:
    """
    Transaction-level enriched view expected by:

    - gender × category
    - age × category

    Repeated customers are intentional because the real
    Lapage case also contains repeated purchases.
    """

    transactions = (
        build_transactions_dataframe()
    )


    customers = (
        build_customers_dataframe()
    )


    products = (
        build_products_dataframe()
    )


    enriched = (
        transactions
        .merge(
            customers,
            on=
                "client_id",

            how=
                "left",

            validate=
                "many_to_one",
        )
        .merge(
            products,
            on=
                "id_prod",

            how=
                "left",

            validate=
                "many_to_one",
        )
    )


    enriched[
        "age_at_first_purchase"
    ] = (
        enriched[
            "client_id"
        ]
        .map(
            {
                "c01": 25,
                "c02": 25,
                "c03": 30,
                "c04": 30,
                "c05": 35,
                "c06": 35,
                "c07": 40,
                "c08": 40,
                "c09": 45,
                "c10": 45,
                "c11": 50,
                "c12": 50,
            }
        )
    )


    return (
        pd.DataFrame(
            {
                "customer_id":
                    enriched[
                        "client_id"
                    ],

                "event_time":
                    enriched[
                        "date"
                    ],

                "gender":
                    enriched[
                        "sex"
                    ],

                "category":
                    enriched[
                        "categ"
                    ],

                "age_at_first_purchase":
                    enriched[
                        "age_at_first_purchase"
                    ],
            }
        )
    )


# ============================================================
# MONTHLY REVENUE VIEW
# ============================================================

def build_monthly_revenue_dataframe(
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "month": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-02-01",
                        "2024-03-01",
                        "2024-04-01",
                        "2024-05-01",
                        "2024-06-01",
                    ]
                ),

                "sum_price": [
                    180.0,
                    240.0,
                    310.0,
                    390.0,
                    470.0,
                    560.0,
                ],
            }
        )
    )


# ============================================================
# CATEGORY REVENUE VIEW
# ============================================================

def build_category_revenue_dataframe(
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "categ": [
                    0,
                    1,
                    2,
                ],

                "sum_price": [
                    420.0,
                    760.0,
                    970.0,
                ],
            }
        )
    )


# ============================================================
# PRODUCT-GRAIN VIEW
# ============================================================

def build_product_revenue_dataframe(
) -> pd.DataFrame:
    """
    One row per product.

    The category is intentionally retained so the same
    traceable product view can support the reference
    distribution request.
    """

    return (
        pd.DataFrame(
            {
                "id_prod": [
                    "p01",
                    "p02",
                    "p03",
                    "p04",
                    "p05",
                    "p06",
                ],

                "categ": [
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                ],

                "sum_price": [
                    120.0,
                    300.0,
                    340.0,
                    420.0,
                    450.0,
                    520.0,
                ],
            }
        )
    )


# ============================================================
# EXECUTION DATASET RECORDS
# ============================================================

def build_execution_datasets():
    """
    Build the same kind of records consumed by
    requested_executor.py.

    Source records remain distinguishable from derived
    analytical views.
    """

    return [
        # ----------------------------------------------------
        # SOURCE — TRANSACTIONS
        # ----------------------------------------------------

        {
            "dataset_id":
                "dataset:transactions",

            "filename":
                "Transactions.csv",

            "dataframe":
                build_transactions_dataframe(),

            "is_derived":
                False,

            "discoverable":
                True,

            "derivation_type":
                "",
        },


        # ----------------------------------------------------
        # SOURCE — CUSTOMERS
        # ----------------------------------------------------

        {
            "dataset_id":
                "dataset:customers",

            "filename":
                "customers.csv",

            "dataframe":
                build_customers_dataframe(),

            "is_derived":
                False,

            "discoverable":
                True,

            "derivation_type":
                "",
        },


        # ----------------------------------------------------
        # SOURCE — PRODUCTS
        # ----------------------------------------------------

        {
            "dataset_id":
                "dataset:products",

            "filename":
                "products.csv",

            "dataframe":
                build_products_dataframe(),

            "is_derived":
                False,

            "discoverable":
                True,

            "derivation_type":
                "",
        },


        # ----------------------------------------------------
        # DERIVED — CUSTOMER GRAIN
        # ----------------------------------------------------

        {
            "dataset_id":
                "derived:customer-behavior",

            "filename":
                "customer_behavior.csv",

            "dataframe":
                build_customer_grain_dataframe(),

            "is_derived":
                True,

            "discoverable":
                False,

            "derivation_type":
                "customer_behavior",

            "provenance": {
                "grain":
                    "customer",

                "operation":
                    "requested_customer_metrics",
            },
        },


        # ----------------------------------------------------
        # DERIVED — REQUESTED EVENT CONTEXT
        # ----------------------------------------------------

        {
            "dataset_id":
                "derived:requested-event-context",

            "filename":
                "requested_event_context.csv",

            "dataframe":
                build_requested_event_context_dataframe(),

            "is_derived":
                True,

            "discoverable":
                False,

            "derivation_type":
                "requested_event_context",

            "provenance": {
                "grain":
                    "event",

                "operation":
                    "requested_context_join",
            },
        },


        # ----------------------------------------------------
        # DERIVED — MONTHLY REVENUE
        # ----------------------------------------------------

        {
            "dataset_id":
                "derived:monthly-revenue",

            "filename":
                "monthly_revenue.csv",

            "dataframe":
                build_monthly_revenue_dataframe(),

            "is_derived":
                True,

            "discoverable":
                False,

            "derivation_type":
                "monthly_additive_measure",

            "provenance": {
                "source_time_column":
                    "date",

                "source_measure_column":
                    "price",

                "target_time_column":
                    "month",

                "target_measure_column":
                    "sum_price",
            },
        },


        # ----------------------------------------------------
        # DERIVED — REVENUE BY CATEGORY
        # ----------------------------------------------------

        {
            "dataset_id":
                "derived:category-revenue",

            "filename":
                "category_revenue.csv",

            "dataframe":
                build_category_revenue_dataframe(),

            "is_derived":
                True,

            "discoverable":
                False,

            "derivation_type":
                "categorical_additive_measure",

            "provenance": {
                "group_column":
                    "categ",

                "source_measure_column":
                    "price",

                "target_measure_column":
                    "sum_price",
            },
        },


        # ----------------------------------------------------
        # DERIVED — PRODUCT GRAIN
        # ----------------------------------------------------

        {
            "dataset_id":
                "derived:product-revenue",

            "filename":
                "product_revenue.csv",

            "dataframe":
                build_product_revenue_dataframe(),

            "is_derived":
                True,

            "discoverable":
                False,

            "derivation_type":
                "entity_additive_measure",

            "provenance": {
                "entity_column":
                    "id_prod",

                "source_measure_column":
                    "price",

                "target_measure_column":
                    "sum_price",

                "grain":
                    "product",
            },
        },
    ]


# ============================================================
# REAL EXECUTION
# ============================================================

def build_real_execution_report():
    (
        _,
        _,
        plan,
    ) = build_lapage_plan()


    datasets = (
        build_execution_datasets()
    )


    report = (
        execute_requested_analysis_plan(
            plan=
                plan,

            datasets=
                datasets,
        )
    )


    return (
        plan,
        datasets,
        report,
    )


# ============================================================
# TEST 1 — REQUEST COVERAGE SURVIVES EXECUTION
# ============================================================

def test_real_execution_preserves_all_15_requests(
) -> None:
    (
        plan,
        _,
        report,
    ) = build_real_execution_report()


    assert_equal(
        plan.request_count,
        15,
        (
            "Lapage planner must expose "
            "15 requests."
        ),
    )


    assert_equal(
        report.request_count,
        15,
        (
            "Real Requested Analysis execution "
            "must preserve all 15 requests."
        ),
    )


    assert_equal(
        len(
            report.results
        ),
        15,
        (
            "Real execution report must contain "
            "one result per request."
        ),
    )


    pass_test(
        (
            "real execution preserves all "
            "15 Lapage requests"
        )
    )


# ============================================================
# TEST 2 — ALL READY REQUESTS RESOLVE DATA
# ============================================================

def test_all_ready_requests_resolve_dataset(
) -> None:
    (
        _,
        _,
        report,
    ) = build_real_execution_report()


    results_by_kind = {
        result.kind:
            result

        for result
        in report.results
    }


    for kind in (
        READY_KINDS
    ):
        result = (
            results_by_kind[
                kind
            ]
        )


        assert_true(
            result.dataset_id
            is not None,
            (
                f"{kind} is ready but did not "
                "resolve a concrete analytical "
                "dataset."
            ),
        )


        assert_true(
            result.dataset_filename
            is not None,
            (
                f"{kind} is ready but did not "
                "preserve the resolved dataset "
                "filename."
            ),
        )


    pass_test(
        (
            "all 11 ready requests resolve a "
            "concrete analytical dataset"
        )
    )


# ============================================================
# TEST 3 — DIRECT BRIEF EXECUTION
# ============================================================

def test_all_direct_brief_requests_complete(
) -> None:
    (
        _,
        _,
        report,
    ) = build_real_execution_report()


    results_by_kind = {
        result.kind:
            result

        for result
        in report.results
    }


    failures = {
        kind:
            results_by_kind[
                kind
            ].execution_status

        for kind
        in DIRECT_BRIEF_KINDS

        if (
            results_by_kind[
                kind
            ].execution_status
            !=
            "complete"
        )
    }


    assert_equal(
        failures,
        {},
        (
            "One or more deterministic brief "
            "requests failed to complete despite "
            "having the required prepared views."
        ),
    )


    pass_test(
        (
            "all 6 deterministic ready brief "
            "requests complete on real DataFrames"
        )
    )


# ============================================================
# TEST 4 — QUANTITATIVE ASSOCIATIONS
# ============================================================

def test_quantitative_requests_execute_or_fallback(
) -> None:
    (
        _,
        _,
        report,
    ) = build_real_execution_report()


    results_by_kind = {
        result.kind:
            result

        for result
        in report.results
    }


    allowed_statuses = {
        "complete",
        "descriptive_only",
    }


    invalid = {
        kind:
            results_by_kind[
                kind
            ].execution_status

        for kind
        in QUANTITATIVE_KINDS

        if (
            results_by_kind[
                kind
            ].execution_status
            not in
            allowed_statuses
        )
    }


    assert_equal(
        invalid,
        {},
        (
            "A quantitative customer request "
            "neither completed nor produced its "
            "deterministic descriptive fallback."
        ),
    )


    pass_test(
        (
            "three quantitative requests execute "
            "or use descriptive fallback"
        )
    )


# ============================================================
# TEST 5 — CONTEXT REQUESTS RESOLVE THEIR VIEW
# ============================================================

def test_context_requests_do_not_fail_resolution(
) -> None:
    (
        _,
        _,
        report,
    ) = build_real_execution_report()


    results_by_kind = {
        result.kind:
            result

        for result
        in report.results
    }


    forbidden_statuses = {
        "failed",
        "not_executed",
        "not_supported_yet",
    }


    invalid = {
        kind:
            results_by_kind[
                kind
            ].execution_status

        for kind
        in CONTEXT_KINDS

        if (
            results_by_kind[
                kind
            ].execution_status
            in
            forbidden_statuses
        )
    }


    assert_equal(
        invalid,
        {},
        (
            "A ready context request failed before "
            "or during real execution."
        ),
    )


    for kind in (
        CONTEXT_KINDS
    ):
        result = (
            results_by_kind[
                kind
            ]
        )


        assert_equal(
            result.dataset_id,
            "derived:requested-event-context",
            (
                f"{kind} should execute against "
                "the requested event-context view."
            ),
        )


    pass_test(
        (
            "both context requests resolve the "
            "requested event-context view"
        )
    )


# ============================================================
# TEST 6 — NON-READY REQUESTS REMAIN VISIBLE
# ============================================================

def test_non_ready_requests_remain_not_executed(
) -> None:
    (
        _,
        _,
        report,
    ) = build_real_execution_report()


    results_by_kind = {
        result.kind:
            result

        for result
        in report.results
    }


    actual = {
        kind:
            (
                results_by_kind[
                    kind
                ].plan_status,
                results_by_kind[
                    kind
                ].execution_status,
            )

        for kind
        in NON_READY_KINDS
    }


    expected = {
        "revenue_moving_average": (
            "ambiguous",
            "not_executed",
        ),

        "top_products": (
            "ambiguous",
            "not_executed",
        ),

        "flop_products": (
            "ambiguous",
            "not_executed",
        ),

        "b2b_revenue_distribution": (
            "blocked",
            "not_executed",
        ),
    }


    assert_equal(
        actual,
        expected,
        (
            "Ambiguous or blocked requests "
            "changed state during execution."
        ),
    )


    pass_test(
        (
            "ambiguous and blocked requests remain "
            "explicitly not_executed"
        )
    )


# ============================================================
# TEST 7 — NO EXECUTOR FAILURE OR UNSUPPORTED READY REQUEST
# ============================================================

def test_no_real_execution_failure(
) -> None:
    (
        _,
        _,
        report,
    ) = build_real_execution_report()


    assert_equal(
        report.failed_count,
        0,
        (
            "Real synthetic Lapage execution "
            "must not produce internal failures."
        ),
    )


    assert_equal(
        report.not_supported_yet_count,
        0,
        (
            "No known Lapage request should fall "
            "through to not_supported_yet."
        ),
    )


    pass_test(
        (
            "real execution has zero failed and "
            "zero unsupported requests"
        )
    )


# ============================================================
# TEST 8 — ATTEMPTED COUNT
# ============================================================

def test_all_11_ready_requests_are_attempted(
) -> None:
    (
        _,
        _,
        report,
    ) = build_real_execution_report()


    assert_equal(
        report.attempted_count,
        11,
        (
            "Every ready Lapage request should be "
            "attempted during real execution."
        ),
    )


    assert_equal(
        report.not_executed_count,
        4,
        (
            "The three ambiguous requests and "
            "blocked BtoB request should remain "
            "not_executed."
        ),
    )


    pass_test(
        (
            "11 ready requests are attempted and "
            "4 non-ready requests remain visible"
        )
    )


# ============================================================
# TEST 9 — ACCOUNTING
# ============================================================

def test_real_execution_accounting_reconciles(
) -> None:
    (
        _,
        _,
        report,
    ) = build_real_execution_report()


    accounted = (
        report.complete_count
        +
        report.descriptive_only_count
        +
        report.needs_information_count
        +
        report.needs_specialized_method_count
        +
        report.skipped_count
        +
        report.failed_count
        +
        report.not_executed_count
        +
        report.not_supported_yet_count
    )


    assert_equal(
        accounted,
        15,
        (
            "Real execution status accounting "
            "must reconcile to 15 requests."
        ),
    )


    assert_equal(
        accounted,
        report.request_count,
        (
            "Real execution contains an "
            "unaccounted request."
        ),
    )


    pass_test(
        (
            "real execution accounting reconciles "
            "exactly to 15/15"
        )
    )


# ============================================================
# DIAGNOSTIC OUTPUT
# ============================================================

def print_real_execution_report(
) -> None:
    (
        _,
        datasets,
        report,
    ) = build_real_execution_report()


    print()

    print(
        "===== ANALYTICAL DATASETS ====="
    )

    print()


    for dataset in (
        datasets
    ):
        dataframe = (
            dataset[
                "dataframe"
            ]
        )


        print(
            (
                f"- {dataset['dataset_id']} · "
                f"{dataset.get('derivation_type', '') or 'source'} · "
                f"{len(dataframe)} rows"
            )
        )


    print()

    print(
        "===== REAL LAPAGE REQUEST EXECUTION ====="
    )

    print()


    for (
        index,
        result,
    ) in enumerate(
        report.results,
        start=1,
    ):
        print(
            (
                f"{index:02d}. "
                f"[{result.execution_status.upper()}] "
                f"{result.kind}"
            )
        )


        print(
            (
                f"    plan: "
                f"{result.plan_status}"
            )
        )


        print(
            (
                f"    dataset: "
                f"{result.dataset_id}"
            )
        )


        print(
            (
                f"    inferential: "
                f"{result.inferential_status}"
            )
        )


        if (
            result.warnings
        ):
            for warning in (
                result.warnings
            ):
                print(
                    f"    warning: {warning}"
                )


        print()


    status_counts = (
        Counter(
            result.execution_status

            for result
            in report.results
        )
    )


    print(
        "===== EXECUTION SUMMARY ====="
    )

    print()


    print(
        f"requests: {report.request_count}"
    )

    print(
        f"attempted: {report.attempted_count}"
    )

    print(
        f"complete: {report.complete_count}"
    )

    print(
        (
            "descriptive_only: "
            f"{report.descriptive_only_count}"
        )
    )

    print(
        (
            "needs_information: "
            f"{report.needs_information_count}"
        )
    )

    print(
        (
            "needs_specialized_method: "
            f"{report.needs_specialized_method_count}"
        )
    )

    print(
        f"failed: {report.failed_count}"
    )

    print(
        (
            "not_executed: "
            f"{report.not_executed_count}"
        )
    )

    print(
        (
            "not_supported_yet: "
            f"{report.not_supported_yet_count}"
        )
    )

    print()

    print(
        (
            "raw status counts: "
            f"{dict(status_counts)}"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        (
            "=== DATALENS LAPAGE REQUESTED "
            "REAL EXECUTION v0.1 ==="
        )
    )

    print()


    test_real_execution_preserves_all_15_requests()

    test_all_ready_requests_resolve_dataset()

    test_all_direct_brief_requests_complete()

    test_quantitative_requests_execute_or_fallback()

    test_context_requests_do_not_fail_resolution()

    test_non_ready_requests_remain_not_executed()

    test_no_real_execution_failure()

    test_all_11_ready_requests_are_attempted()

    test_real_execution_accounting_reconciles()


    print_real_execution_report()


    print()

    print(
        (
            "PASS - Lapage requested "
            "real execution v0.1"
        )
    )


if (
    __name__
    ==
    "__main__"
):
    main()
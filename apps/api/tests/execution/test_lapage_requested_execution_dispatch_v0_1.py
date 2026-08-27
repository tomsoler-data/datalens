from __future__ import annotations


from collections import (
    Counter,
)

from unittest.mock import (
    patch,
)


from app.execution import (
    requested_executor as requested_executor_module,
)

from tests.planning.test_lapage_request_planning_coverage_v0_1 import (
    build_lapage_plan,
)


# ============================================================
# EXPECTED ROUTES
# ============================================================

EXPECTED_QUANTITATIVE_KINDS = {
    "age_total_amount_association",
    "age_frequency_association",
    "age_average_basket_association",
}


EXPECTED_CONTEXT_KINDS = {
    "gender_category_association",
    "age_category_association",
}


EXPECTED_BRIEF_READY_KINDS = {
    "revenue_by_category",
    "customers_by_period",
    "transaction_count",
    "products_sold_count",
    "product_category_distribution",
    "lorenz_curve",
}


EXPECTED_NOT_EXECUTED_KINDS = {
    "revenue_moving_average",
    "top_products",
    "flop_products",
    "b2b_revenue_distribution",
}


EXPECTED_ALL_KINDS = (
    EXPECTED_QUANTITATIVE_KINDS
    |
    EXPECTED_CONTEXT_KINDS
    |
    EXPECTED_BRIEF_READY_KINDS
    |
    EXPECTED_NOT_EXECUTED_KINDS
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
# ROUTE RESULT
# ============================================================

def routed_result(
    *,
    route: str,
    request,
):
    """
    Deliberately return a plain dict.

    execute_requested_analysis() only dispatches and returns
    the handler result. Pydantic execution objects belong to
    the handler itself and are not needed for this routing
    contract test.
    """

    return {
        "route":
            route,

        "request_id":
            request.request_id,

        "kind":
            request.kind,

        "plan_status":
            request.status,
    }


# ============================================================
# PATCHED HANDLERS
# ============================================================

def quantitative_handler(
    *args,
    **kwargs,
):
    request = (
        kwargs.get(
            "request"
        )
    )


    if request is None:
        raise AssertionError(
            (
                "Quantitative route did not receive "
                "the request."
            )
        )


    return (
        routed_result(
            route=
                "quantitative",

            request=
                request,
        )
    )


def context_handler(
    *args,
    **kwargs,
):
    request = (
        kwargs.get(
            "request"
        )
    )


    if request is None:
        raise AssertionError(
            (
                "Context route did not receive "
                "the request."
            )
        )


    return (
        routed_result(
            route=
                "context",

            request=
                request,
        )
    )


def brief_handler(
    *args,
    **kwargs,
):
    request = (
        kwargs.get(
            "request"
        )
    )


    if request is None:
        raise AssertionError(
            (
                "Brief route did not receive "
                "the request."
            )
        )


    return (
        routed_result(
            route=
                "brief",

            request=
                request,
        )
    )


def not_executed_handler(
    *args,
    **kwargs,
):
    request = (
        kwargs.get(
            "request"
        )
    )


    if (
        request is None
        and
        args
    ):
        request = (
            args[
                0
            ]
        )


    if request is None:
        raise AssertionError(
            (
                "not_executed route did not "
                "receive the request."
            )
        )


    return (
        routed_result(
            route=
                "not_executed",

            request=
                request,
        )
    )


def unsupported_handler(
    *args,
    **kwargs,
):
    request = (
        kwargs.get(
            "request"
        )
    )


    if (
        request is None
        and
        args
    ):
        request = (
            args[
                0
            ]
        )


    if request is None:
        raise AssertionError(
            (
                "unsupported route did not "
                "receive the request."
            )
        )


    return (
        routed_result(
            route=
                "unsupported",

            request=
                request,
        )
    )


# ============================================================
# EXECUTE DISPATCH
# ============================================================

def dispatch_lapage_requests():
    (
        _,
        _,
        plan,
    ) = build_lapage_plan()


    outputs = []


    with (
        patch.object(
            requested_executor_module,
            "execute_quantitative_requested_analysis",
            side_effect=
                quantitative_handler,
        ),

        patch.object(
            requested_executor_module,
            "execute_context_requested_analysis",
            side_effect=
                context_handler,
        ),

        patch.object(
            requested_executor_module,
            "execute_brief_requested_analysis",
            side_effect=
                brief_handler,
        ),

        patch.object(
            requested_executor_module,
            "not_executed_result",
            side_effect=
                not_executed_handler,
        ),

        patch.object(
            requested_executor_module,
            "unsupported_result",
            side_effect=
                unsupported_handler,
        ),
    ):
        for request in (
            plan.requests
        ):
            output = (
                requested_executor_module
                .execute_requested_analysis(
                    request=
                        request,

                    datasets=[],
                )
            )


            outputs.append(
                output
            )


    return (
        plan,
        outputs,
    )


# ============================================================
# TEST 1
# ============================================================

def test_all_15_requests_reach_dispatch(
) -> None:
    (
        plan,
        outputs,
    ) = dispatch_lapage_requests()


    assert_equal(
        plan.request_count,
        15,
        (
            "Lapage planner must provide "
            "15 requests."
        ),
    )


    assert_equal(
        len(
            outputs
        ),
        15,
        (
            "Execution dispatch must produce "
            "one outcome for every planned request."
        ),
    )


    output_kinds = {
        output[
            "kind"
        ]

        for output
        in outputs
    }


    assert_equal(
        output_kinds,
        EXPECTED_ALL_KINDS,
        (
            "Execution dispatch lost or invented "
            "a Lapage request kind."
        ),
    )


    pass_test(
        (
            "all 15 Lapage requests receive "
            "an execution-dispatch outcome"
        )
    )


# ============================================================
# TEST 2
# ============================================================

def test_quantitative_routes(
) -> None:
    (
        _,
        outputs,
    ) = dispatch_lapage_requests()


    actual = {
        output[
            "kind"
        ]

        for output
        in outputs

        if (
            output[
                "route"
            ]
            ==
            "quantitative"
        )
    }


    assert_equal(
        actual,
        EXPECTED_QUANTITATIVE_KINDS,
        (
            "Unexpected quantitative execution "
            "routing."
        ),
    )


    pass_test(
        (
            "three customer quantitative "
            "associations reach their executor"
        )
    )


# ============================================================
# TEST 3
# ============================================================

def test_context_routes(
) -> None:
    (
        _,
        outputs,
    ) = dispatch_lapage_requests()


    actual = {
        output[
            "kind"
        ]

        for output
        in outputs

        if (
            output[
                "route"
            ]
            ==
            "context"
        )
    }


    assert_equal(
        actual,
        EXPECTED_CONTEXT_KINDS,
        (
            "Unexpected documentary-context "
            "execution routing."
        ),
    )


    pass_test(
        (
            "gender/category and age/category "
            "reach the context executor"
        )
    )


# ============================================================
# TEST 4
# ============================================================

def test_brief_routes(
) -> None:
    (
        _,
        outputs,
    ) = dispatch_lapage_requests()


    actual = {
        output[
            "kind"
        ]

        for output
        in outputs

        if (
            output[
                "route"
            ]
            ==
            "brief"
        )
    }


    assert_equal(
        actual,
        EXPECTED_BRIEF_READY_KINDS,
        (
            "Unexpected deterministic brief "
            "execution routing."
        ),
    )


    pass_test(
        (
            "all six ready brief requests "
            "reach the brief executor"
        )
    )


# ============================================================
# TEST 5
# ============================================================

def test_non_ready_requests_are_retained(
) -> None:
    (
        _,
        outputs,
    ) = dispatch_lapage_requests()


    actual = {
        output[
            "kind"
        ]

        for output
        in outputs

        if (
            output[
                "route"
            ]
            ==
            "not_executed"
        )
    }


    assert_equal(
        actual,
        EXPECTED_NOT_EXECUTED_KINDS,
        (
            "Blocked and ambiguous requests should "
            "be retained as not_executed."
        ),
    )


    statuses = {
        output[
            "kind"
        ]:
            output[
                "plan_status"
            ]

        for output
        in outputs

        if (
            output[
                "route"
            ]
            ==
            "not_executed"
        )
    }


    assert_equal(
        statuses[
            "revenue_moving_average"
        ],
        "ambiguous",
        (
            "Revenue moving average must remain "
            "explicitly ambiguous until temporal "
            "granularity and moving-average window "
            "are resolved."
        ),
    )


    assert_equal(
        statuses[
            "top_products"
        ],
        "ambiguous",
        (
            "Top products must remain "
            "explicitly ambiguous."
        ),
    )


    assert_equal(
        statuses[
            "flop_products"
        ],
        "ambiguous",
        (
            "Flop products must remain "
            "explicitly ambiguous."
        ),
    )


    assert_equal(
        statuses[
            "b2b_revenue_distribution"
        ],
        "blocked",
        (
            "BtoB request must remain "
            "explicitly blocked."
        ),
    )


    pass_test(
        (
            "three ambiguous and one blocked request "
            "remain visible as not_executed"
        )
    )


# ============================================================
# TEST 6
# ============================================================

def test_no_lapage_request_hits_unsupported(
) -> None:
    (
        _,
        outputs,
    ) = dispatch_lapage_requests()


    unsupported = [
        output[
            "kind"
        ]

        for output
        in outputs

        if (
            output[
                "route"
            ]
            ==
            "unsupported"
        )
    ]


    assert_equal(
        unsupported,
        [],
        (
            "A known Lapage request reached "
            "unsupported_result()."
        ),
    )


    pass_test(
        (
            "no Lapage request falls through "
            "to unsupported execution"
        )
    )


# ============================================================
# TEST 7
# ============================================================

def test_dispatch_accounting(
) -> None:
    (
        _,
        outputs,
    ) = dispatch_lapage_requests()


    counts = (
        Counter(
            output[
                "route"
            ]

            for output
            in outputs
        )
    )


    expected = {
        "quantitative":
            3,

        "context":
            2,

        "brief":
            6,

        "not_executed":
            4,
    }


    actual = {
        route:
            count

        for (
            route,
            count,
        )
        in counts.items()

        if (
            count
            >
            0
        )
    }


    assert_equal(
        actual,
        expected,
        (
            "Unexpected Lapage execution "
            "dispatch accounting."
        ),
    )


    assert_equal(
        sum(
            actual.values()
        ),
        15,
        (
            "Execution dispatch accounting "
            "must reconcile to 15/15."
        ),
    )


    pass_test(
        (
            "execution dispatch accounting "
            "reconciles to 15/15"
        )
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

def print_dispatch() -> None:
    (
        _,
        outputs,
    ) = dispatch_lapage_requests()


    print()

    print(
        "===== LAPAGE EXECUTION DISPATCH ====="
    )

    print()


    for (
        index,
        output,
    ) in enumerate(
        outputs,
        start=1,
    ):
        print(
            (
                f"{index:02d}. "
                f"[{output['route'].upper()}] "
                f"{output['kind']} "
                f"(plan={output['plan_status']})"
            )
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        (
            "=== DATALENS LAPAGE REQUESTED "
            "EXECUTION DISPATCH v0.1 ==="
        )
    )

    print()


    test_all_15_requests_reach_dispatch()

    test_quantitative_routes()

    test_context_routes()

    test_brief_routes()

    test_non_ready_requests_are_retained()

    test_no_lapage_request_hits_unsupported()

    test_dispatch_accounting()


    print_dispatch()


    print()

    print(
        (
            "PASS - Lapage requested execution "
            "dispatch v0.1"
        )
    )


if (
    __name__
    ==
    "__main__"
):
    main()
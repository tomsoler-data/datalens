from __future__ import annotations


from collections import (
    Counter,
)


from app.ingestion.schemas import (
    CorrelationCompatibility,
    DatasetColumnManifest,
    DatasetManifest,
    MultiDatasetIngestion,
)

from app.planning.request_coverage import (
    build_analysis_request_coverage,
    require_complete_analysis_request_coverage,
)

from app.planning.request_planner import (
    build_requested_analysis_plan,
)

from tests.planning.test_lapage_request_inventory_v0_1 import (
    extract_requests,
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
# COLUMN FIXTURE
# ============================================================

def make_column(
    name: str,
    *,
    dtype: str,
    analysis_kind: str,
    unique_count: int,
    unique_ratio: float,
    unique_candidate: bool = False,
    correlation_eligible: bool = False,
) -> DatasetColumnManifest:
    return (
        DatasetColumnManifest(
            name=
                name,

            dtype=
                dtype,

            missing_count=
                0,

            missing_ratio=
                0.0,

            unique_count=
                unique_count,

            unique_ratio=
                unique_ratio,

            unique_candidate=
                unique_candidate,

            analysis_kind=
                analysis_kind,

            correlation_eligible=
                correlation_eligible,

            analysis_note=
                "Lapage planning coverage fixture.",
        )
    )


# ============================================================
# CORRELATION FIXTURE
# ============================================================

def correlation_compatibility(
    candidate_columns: list[
        str
    ],
) -> CorrelationCompatibility:
    if (
        len(
            candidate_columns
        )
        >=
        2
    ):
        return (
            CorrelationCompatibility(
                status=
                    "ready",

                candidate_columns=
                    candidate_columns,

                default_x_column=
                    candidate_columns[
                        0
                    ],

                default_y_column=
                    candidate_columns[
                        1
                    ],

                reasons=[
                    (
                        "Synthetic Lapage fixture "
                        "contains quantitative columns."
                    )
                ],
            )
        )


    return (
        CorrelationCompatibility(
            status=
                "not_available",

            candidate_columns=
                candidate_columns,

            default_x_column=
                None,

            default_y_column=
                None,

            reasons=[
                (
                    "Synthetic Lapage fixture does "
                    "not expose two quantitative "
                    "columns in this dataset."
                )
            ],
        )
    )


# ============================================================
# DATASET FIXTURES
# ============================================================

def build_lapage_ingestion(
) -> MultiDatasetIngestion:
    transactions = (
        DatasetManifest(
            dataset_id=
                "dataset:transactions",

            filename=
                "Transactions.csv",

            extension=
                ".csv",

            row_count=
                687_534,

            column_count=
                4,

            memory_bytes=
                1,

            columns=[
                make_column(
                    "client_id",
                    dtype=
                        "object",

                    analysis_kind=
                        "categorical",

                    unique_count=
                        8_600,

                    unique_ratio=
                        0.0125,

                    unique_candidate=
                        False,
                ),

                make_column(
                    "id_prod",
                    dtype=
                        "object",

                    analysis_kind=
                        "categorical",

                    unique_count=
                        3_265,

                    unique_ratio=
                        0.0047,

                    unique_candidate=
                        False,
                ),

                make_column(
                    "session_id",
                    dtype=
                        "object",

                    analysis_kind=
                        "categorical",

                    unique_count=
                        342_315,

                    unique_ratio=
                        0.4979,

                    unique_candidate=
                        False,
                ),

                make_column(
                    "date",
                    dtype=
                        "datetime64[ns]",

                    analysis_kind=
                        "temporal",

                    unique_count=
                        687_534,

                    unique_ratio=
                        1.0,

                    unique_candidate=
                        False,
                ),
            ],

            correlation_compatibility=
                correlation_compatibility(
                    []
                ),

            warnings=[],
        )
    )


    customers = (
        DatasetManifest(
            dataset_id=
                "dataset:customers",

            filename=
                "customers.csv",

            extension=
                ".csv",

            row_count=
                8_621,

            column_count=
                3,

            memory_bytes=
                1,

            columns=[
                make_column(
                    "client_id",
                    dtype=
                        "object",

                    analysis_kind=
                        "categorical",

                    unique_count=
                        8_621,

                    unique_ratio=
                        1.0,

                    unique_candidate=
                        True,
                ),

                make_column(
                    "sex",
                    dtype=
                        "object",

                    analysis_kind=
                        "categorical",

                    unique_count=
                        2,

                    unique_ratio=
                        0.0002,
                ),

                make_column(
                    "birth",
                    dtype=
                        "int64",

                    analysis_kind=
                        "quantitative",

                    unique_count=
                        76,

                    unique_ratio=
                        0.0088,

                    correlation_eligible=
                        True,
                ),
            ],

            correlation_compatibility=
                correlation_compatibility(
                    [
                        "birth"
                    ]
                ),

            warnings=[],
        )
    )


    products = (
        DatasetManifest(
            dataset_id=
                "dataset:products",

            filename=
                "products.csv",

            extension=
                ".csv",

            row_count=
                3_287,

            column_count=
                3,

            memory_bytes=
                1,

            columns=[
                make_column(
                    "id_prod",
                    dtype=
                        "object",

                    analysis_kind=
                        "categorical",

                    unique_count=
                        3_287,

                    unique_ratio=
                        1.0,

                    unique_candidate=
                        True,
                ),

                make_column(
                    "price",
                    dtype=
                        "float64",

                    analysis_kind=
                        "quantitative",

                    unique_count=
                        1_455,

                    unique_ratio=
                        0.4427,

                    correlation_eligible=
                        True,
                ),

                make_column(
                    "categ",
                    dtype=
                        "int64",

                    analysis_kind=
                        "categorical",

                    unique_count=
                        3,

                    unique_ratio=
                        0.0009,
                ),
            ],

            correlation_compatibility=
                correlation_compatibility(
                    [
                        "price"
                    ]
                ),

            warnings=[],
        )
    )


    datasets = [
        transactions,
        customers,
        products,
    ]


    return (
        MultiDatasetIngestion(
            dataset_count=
                len(
                    datasets
                ),

            total_rows=
                sum(
                    dataset.row_count

                    for dataset
                    in datasets
                ),

            datasets=
                datasets,

            warnings=[],

            ingestion_rule_version=
                "dataset_ingestion_v0.2",
        )
    )


# ============================================================
# SOURCE REQUESTS
# ============================================================

def lapage_claims():
    (
        _,
        requests,
    ) = extract_requests()


    return [
        claim

        for (
            claim,
            _,
        )
        in requests
    ]


# ============================================================
# PLAN
# ============================================================

def build_lapage_plan():
    claims = (
        lapage_claims()
    )


    ingestion = (
        build_lapage_ingestion()
    )


    plan = (
        build_requested_analysis_plan(
            ingestion=
                ingestion,

            analytical_requests=
                claims,
        )
    )


    return (
        claims,
        ingestion,
        plan,
    )


# ============================================================
# TEST 1
# ============================================================

def test_all_15_requests_reach_planner(
) -> None:
    (
        claims,
        _,
        plan,
    ) = build_lapage_plan()


    assert_equal(
        len(
            claims
        ),
        15,
        (
            "Lapage source inventory must contain "
            "15 analytical requests."
        ),
    )


    assert_equal(
        plan.request_count,
        15,
        (
            "Every detected Lapage request must "
            "reach the request planner."
        ),
    )


    assert_equal(
        len(
            plan.requests
        ),
        15,
        (
            "Planner request list must contain "
            "exactly 15 entries."
        ),
    )


    pass_test(
        "all 15 Lapage requests reach the planner"
    )


# ============================================================
# TEST 2
# ============================================================

def test_planner_preserves_every_kind(
) -> None:
    (
        _,
        _,
        plan,
    ) = build_lapage_plan()


    kinds = [
        request.kind

        for request
        in plan.requests
    ]


    counts = (
        Counter(
            kinds
        )
    )


    assert_equal(
        len(
            counts
        ),
        15,
        (
            "The benchmark should preserve all "
            "15 distinct Lapage request kinds."
        ),
    )


    duplicates = {
        kind:
            count

        for (
            kind,
            count,
        )
        in counts.items()

        if (
            count
            !=
            1
        )
    }


    assert_equal(
        duplicates,
        {},
        (
            "Each Lapage request kind should appear "
            "exactly once in this benchmark."
        ),
    )


    assert_true(
        "unknown"
        not in counts,
        (
            "No Lapage request should become "
            "unknown during planning."
        ),
    )


    pass_test(
        "planner preserves all 15 analytical kinds"
    )


# ============================================================
# TEST 3
# ============================================================

def test_expected_planning_statuses(
) -> None:
    (
        _,
        _,
        plan,
    ) = build_lapage_plan()


    by_kind = {
        request.kind:
            request

        for request
        in plan.requests
    }


    assert_equal(
        by_kind[
            "top_products"
        ].status,
        "ambiguous",
        (
            "Top products must remain ambiguous "
            "without an explicit ranking metric."
        ),
    )


    assert_equal(
        by_kind[
            "flop_products"
        ].status,
        "ambiguous",
        (
            "Flop products must remain ambiguous "
            "without an explicit ranking metric."
        ),
    )


    assert_equal(
        by_kind[
            "b2b_revenue_distribution"
        ].status,
        "blocked",
        (
            "BtoB revenue must remain blocked "
            "without an explicit BtoB variable."
        ),
    )


    assert_equal(
        by_kind[
            "revenue_moving_average"
        ].status,
        "ambiguous",
        (
            "Revenue moving average must remain "
            "ambiguous while temporal granularity "
            "and moving-average window are unresolved."
        ),
    )


    ready_kinds = {
        kind

        for (
            kind,
            request,
        )
        in by_kind.items()

        if (
            request.status
            ==
            "ready"
        )
    }


    expected_ready_kinds = {
        "revenue_by_category",
        "customers_by_period",
        "transaction_count",
        "products_sold_count",
        "product_category_distribution",
        "lorenz_curve",
        "gender_category_association",
        "age_total_amount_association",
        "age_frequency_association",
        "age_average_basket_association",
        "age_category_association",
    }


    assert_equal(
        ready_kinds,
        expected_ready_kinds,
        (
            "Unexpected set of ready Lapage "
            "requests."
        ),
    )


    assert_equal(
        plan.ready_count,
        11,
        (
            "Eleven benchmark requests should be "
            "ready with the available Lapage data."
        ),
    )


    assert_equal(
        plan.ambiguous_count,
        3,
        (
            "Top products, flop products, and revenue "
            "moving average should remain ambiguous "
            "until their required choices are resolved."
        ),
    )


    assert_equal(
        plan.blocked_count,
        1,
        (
            "BtoB should be the single blocked "
            "request."
        ),
    )


    pass_test(
        "Lapage planner statuses are explicit and accounted for"
    )


# ============================================================
# TEST 4
# ============================================================

def test_request_coverage_is_complete(
) -> None:
    (
        claims,
        _,
        plan,
    ) = build_lapage_plan()


    coverage = (
        build_analysis_request_coverage(
            analytical_requests=
                claims,

            plan=
                plan,
        )
    )


    assert_equal(
        coverage.status,
        "complete",
        (
            "Lapage request coverage should be "
            "complete."
        ),
    )


    assert_equal(
        coverage.detected_count,
        15,
        (
            "Coverage should start from "
            "15 detected requests."
        ),
    )


    assert_equal(
        coverage.planner_request_count,
        15,
        (
            "Coverage should see "
            "15 planner requests."
        ),
    )


    assert_equal(
        coverage.planned_count,
        15,
        (
            "All 15 requests should have valid "
            "planner coverage."
        ),
    )


    assert_equal(
        coverage.lost_count,
        0,
        (
            "No Lapage request may disappear "
            "between extraction and planning."
        ),
    )


    assert_equal(
        coverage.coverage_rate,
        1.0,
        (
            "Lapage planning coverage must "
            "be 100%."
        ),
    )


    assert_equal(
        coverage.ready_count,
        11,
        "Coverage should report 11 ready requests.",
    )


    assert_equal(
        coverage.ambiguous_count,
        3,
        (
            "Coverage should report "
            "3 ambiguous requests."
        ),
    )


    assert_equal(
        coverage.blocked_count,
        1,
        "Coverage should report 1 blocked request.",
    )


    require_complete_analysis_request_coverage(
        coverage
    )


    pass_test(
        "Lapage request coverage is 15/15 with zero loss"
    )


# ============================================================
# TEST 5
# ============================================================

def test_every_request_preserves_source(
) -> None:
    (
        claims,
        _,
        plan,
    ) = build_lapage_plan()


    source_by_request = {
        (
            claim.citation.chunk_id,
            claim.evidence_unit_id,
        ):
            claim.citation.filename

        for claim
        in claims
    }


    for request in plan.requests:
        key = (
            request.source_chunk_id,
            request.evidence_unit_id,
        )


        assert_true(
            key
            in source_by_request,
            (
                "Planner request has no matching "
                "source evidence unit."
            ),
        )


        assert_equal(
            request.source_filename,
            source_by_request[
                key
            ],
            (
                "Planner changed the source filename "
                "of a documentary request."
            ),
        )


    pass_test(
        "all 15 planner requests preserve documentary provenance"
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

def print_plan() -> None:
    (
        _,
        _,
        plan,
    ) = build_lapage_plan()


    print()

    print(
        "===== LAPAGE REQUEST PLAN ====="
    )

    print()


    for (
        index,
        request,
    ) in enumerate(
        plan.requests,
        start=1,
    ):
        print(
            (
                f"{index:02d}. "
                f"[{request.status.upper()}] "
                f"{request.kind}"
            )
        )

        print(
            (
                f"    source: "
                f"{request.source_filename}"
            )
        )

        print(
            (
                f"    family: "
                f"{request.target_family}"
            )
        )


        if request.required_dataset_filenames:
            print(
                (
                    "    datasets: "
                    +
                    ", ".join(
                        request
                        .required_dataset_filenames
                    )
                )
            )


        if request.blockers:
            for blocker in (
                request.blockers
            ):
                print(
                    f"    blocker: {blocker}"
                )


        print()


    print(
        (
            "SUMMARY: "
            f"{plan.request_count} total · "
            f"{plan.ready_count} ready · "
            f"{plan.ambiguous_count} ambiguous · "
            f"{plan.blocked_count} blocked"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        (
            "=== DATALENS LAPAGE REQUEST "
            "PLANNING COVERAGE v0.1 ==="
        )
    )

    print()


    test_all_15_requests_reach_planner()

    test_planner_preserves_every_kind()

    test_expected_planning_statuses()

    test_request_coverage_is_complete()

    test_every_request_preserves_source()


    print_plan()


    print()

    print(
        (
            "PASS - Lapage request planning "
            "coverage v0.1"
        )
    )


if (
    __name__
    ==
    "__main__"
):
    main()
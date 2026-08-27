import pandas as pd

import app.execution.requested_executor as executor

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisResolution,
    RequestedColumnMatch,
)


def match(
    concept: str,
    column: str,
) -> RequestedColumnMatch:
    return RequestedColumnMatch(
        concept=concept,
        dataset_id="dataset:transactions",
        dataset_filename="transactions.csv",
        column=column,
        analysis_kind="ranking",
        match_score=100,
        reasons=[
            "test"
        ],
    )


def transaction_ranking_plan() -> RequestedAnalysisPlan:
    return RequestedAnalysisPlan(
        request_id="request:test-top-transactions",
        request_text="les tops",
        evidence_quote="les tops",
        source_filename="brief.pdf",
        source_locator="page 1",
        page_number=1,
        source_chunk_id="chunk:test",
        evidence_unit_id=6,
        kind="top_products",
        status="ready",
        resolution=
            RequestedAnalysisResolution(
                ranking_metric=
                    "transaction_count"
            ),
        target_family="ranking",
        matched_columns=[
            match(
                "product_id",
                "product",
            ),
            match(
                "amount",
                "amount",
            ),
            match(
                "session_id",
                "session",
            ),
        ],
        required_dataset_ids=[
            "dataset:transactions"
        ],
        required_dataset_filenames=[
            "transactions.csv"
        ],
        required_operations=[
            "Count distinct transactions by product.",
            "Rank products in descending order.",
        ],
        reasons=[
            "User selected transaction count."
        ],
        blockers=[],
    )


print(
    "===== DATALENS PRODUCT RANKING METRIC v0.1 ====="
)
print()


# ============================================================
# 1. METRIC SEMANTICS
# ============================================================

dataframe = pd.DataFrame(
    {
        "product": [
            "A",
            "A",
            "B",
        ],

        "session": [
            "s1",
            "s2",
            "s3",
        ],

        "amount": [
            1.0,
            1.0,
            100.0,
        ],
    }
)


transaction_ranking = (
    executor
    ._product_transaction_count_ranking_frame(
        dataframe=
            dataframe,

        product_column=
            "product",

        transaction_column=
            "session",

        ascending=
            False,
    )
)


assert (
    transaction_ranking.iloc[
        0
    ][
        "product"
    ]
    ==
    "A"
)

assert (
    int(
        transaction_ranking.iloc[
            0
        ][
            "transaction_count"
        ]
    )
    ==
    2
)


revenue_ranking = (
    dataframe
    .groupby(
        "product",
        sort=False,
    )[
        "amount"
    ]
    .sum()
    .sort_values(
        ascending=False,
        kind="mergesort",
    )
)


assert (
    revenue_ranking.index[
        0
    ]
    ==
    "B"
)


print(
    "[PASS] transaction-count top product is A"
)

print(
    "[PASS] revenue top product is B"
)

print(
    "[PASS] selected metric changes the actual ranking"
)


# ============================================================
# 2. DISTINCT TRANSACTION SEMANTICS
# ============================================================

duplicate_session_frame = pd.DataFrame(
    {
        "product": [
            "A",
            "A",
            "A",
            "B",
        ],

        "session": [
            "s1",
            "s1",
            "s2",
            "s3",
        ],
    }
)


distinct_ranking = (
    executor
    ._product_transaction_count_ranking_frame(
        dataframe=
            duplicate_session_frame,

        product_column=
            "product",

        transaction_column=
            "session",

        ascending=
            False,
    )
)


assert (
    distinct_ranking.iloc[
        0
    ][
        "product"
    ]
    ==
    "A"
)

assert (
    int(
        distinct_ranking.iloc[
            0
        ][
            "transaction_count"
        ]
    )
    ==
    2
)


print(
    "[PASS] duplicate rows inside one session count once"
)


# ============================================================
# 3. FLOP DIRECTION
# ============================================================

flop_ranking = (
    executor
    ._product_transaction_count_ranking_frame(
        dataframe=
            dataframe,

        product_column=
            "product",

        transaction_column=
            "session",

        ascending=
            True,
    )
)


assert (
    flop_ranking.iloc[
        0
    ][
        "product"
    ]
    ==
    "B"
)


print(
    "[PASS] ascending ranking produces the transaction-count flop"
)


# ============================================================
# 4. EXECUTOR DISPATCH
# ============================================================

plan = (
    transaction_ranking_plan()
)

sentinel = object()

original_executor = (
    executor
    .execute_product_transaction_count_ranking
)


def fake_transaction_executor(
    *,
    request,
    datasets,
    ascending,
):
    assert (
        request.request_id
        ==
        plan.request_id
    )

    assert (
        request.resolution
        is not None
    )

    assert (
        request.resolution.ranking_metric
        ==
        "transaction_count"
    )

    assert (
        datasets
        ==
        []
    )

    assert (
        ascending
        is False
    )

    return sentinel


try:
    executor.execute_product_transaction_count_ranking = (
        fake_transaction_executor
    )

    dispatched = (
        executor.execute_product_ranking(
            request=
                plan,

            datasets=[],

            ascending=False,
        )
    )

finally:
    executor.execute_product_transaction_count_ranking = (
        original_executor
    )


assert (
    dispatched
    is
    sentinel
)


print(
    "[PASS] execute_product_ranking dispatches from server-owned resolution"
)

print(
    "[PASS] transaction-count branch does not fall through to revenue"
)

print()
print(
    "PASS - requested product ranking metric v0.1"
)

from __future__ import annotations


import math


import pandas as pd


from app.ai.tool_orchestrator import (
    AI_AGGREGATION_EXECUTION_RULE_VERSION,
    AI_RANKING_EXECUTION_RULE_VERSION,
    execute_aggregation_contract,
    execute_ranking_contract,
)


from app.planning.analytical_contract import (
    AggregationSpec,
    AnalyticalContract,
    RankingSpec,
    ShareOfTotalSpec,
    VariableBinding,
)


# ============================================================
# AUTHORITY
# ============================================================


DATASET_ID = (
    "dataset:category_values"
)


DATASET_FILENAME = (
    "category_values.derived"
)


SHARE_REFERENCE = (
    "sum_of_group_values"
)


# ============================================================
# DATA
# ============================================================


def normal_dataframe() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "C",
            ],

            "sum_value": [
                40.0,
                35.0,
                25.0,
            ],
        }
    )


def zero_total_dataframe() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "category": [
                "A",
                "B",
            ],

            "sum_value": [
                10.0,
                -10.0,
            ],
        }
    )


# ============================================================
# CONTRACT
# ============================================================


def build_contract(
    *,
    family: str,
    include_share: bool,
) -> AnalyticalContract:

    if (
        family
        not in {
            "ranking",
            "aggregation",
        }
    ):

        raise ValueError(
            f"Unsupported test family: {family}"
        )


    return (
        AnalyticalContract(
            contract_id=(
                f"contract:share-execution:{family}:"
                +
                (
                    "share"
                    if include_share
                    else
                    "baseline"
                )
            ),

            origin=
                "ai_planner",

            status=
                "validated",

            title=
                "Category contribution",

            request_text=(
                "Which category contributes the most and "
                "what share of total does it represent?"
            ),

            family=family,  # type: ignore[arg-type]

            required_dataset_ids=[
                DATASET_ID,
            ],

            required_dataset_filenames=[
                DATASET_FILENAME,
            ],

            analytical_grain=
                "category",

            bindings=[
                VariableBinding(
                    role=
                        "value",

                    column=
                        "sum_value",

                    dataset_id=
                        DATASET_ID,

                    dataset_filename=
                        DATASET_FILENAME,

                    semantic_concept=
                        "value",

                    analysis_kind=
                        "quantitative",
                ),

                VariableBinding(
                    role=
                        "dimension",

                    column=
                        "category",

                    dataset_id=
                        DATASET_ID,

                    dataset_filename=
                        DATASET_FILENAME,

                    semantic_concept=
                        None,

                    analysis_kind=
                        "categorical",
                ),
            ],

            aggregation=
                AggregationSpec(
                    function=
                        "sum",

                    source_role=
                        "value",

                    group_by_roles=[
                        "dimension",
                    ],

                    output_name=
                        "planned_metric",
                ),

            ranking=(
                RankingSpec(
                    order=
                        "descending",

                    limit=
                        1,
                )
                if family
                ==
                "ranking"
                else None
            ),

            benchmark=
                None,

            share_of_total=(
                ShareOfTotalSpec(
                    reference=
                        SHARE_REFERENCE
                )
                if include_share
                else None
            ),

            window=
                None,

            filters=[],
            joins=[],
            derived_variables=[],

            required_operations=[
                "Execute deterministic grouped SUM.",
            ],

            reasons=[
                "Generic deterministic share execution test.",
            ],

            blockers=[],

            planner_confidence=
                1.0,
        )
    )


# ============================================================
# HELPERS
# ============================================================


def share_like_keys(
    value,
) -> list[
    str
]:

    found: list[
        str
    ] = []


    def visit(
        current,
    ) -> None:

        if isinstance(
            current,
            dict,
        ):

            for (
                key,
                child,
            ) in current.items():

                normalized = str(
                    key
                ).casefold()


                if (
                    "share"
                    in
                    normalized
                    or
                    "proportion"
                    in
                    normalized
                    or
                    "percentage"
                    in
                    normalized
                ):

                    found.append(
                        str(
                            key
                        )
                    )


                visit(
                    child
                )


        elif isinstance(
            current,
            list,
        ):

            for child in current:

                visit(
                    child
                )


    visit(
        value
    )


    return list(
        dict.fromkeys(
            found
        )
    )


def close(
    left,
    right,
) -> bool:

    try:

        return math.isclose(
            float(
                left
            ),
            float(
                right
            ),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS DETERMINISTIC "
        "SHARE-OF-TOTAL EXECUTION v0.1 ==="
    )

    print()


    dataframe = (
        normal_dataframe()
    )


    # ========================================================
    # 1. BASELINE RANKING WITHOUT SHARE MUST BE PRESERVED
    # ========================================================

    baseline_ranking = (
        execute_ranking_contract(
            contract=
                build_contract(
                    family=
                        "ranking",

                    include_share=
                        False,
                ),

            dataframe=
                dataframe,

            dataset_id=
                DATASET_ID,

            dataset_filename=
                DATASET_FILENAME,
        )
    )


    assert (
        baseline_ranking.execution_status
        ==
        "complete"
    )


    assert (
        len(
            baseline_ranking.chart_data
        )
        ==
        1
    )


    assert (
        baseline_ranking.chart_data[
            0
        ][
            "category"
        ]
        ==
        "A"
    )


    assert close(
        baseline_ranking.chart_data[
            0
        ][
            "value"
        ],
        40.0,
    )


    assert (
        baseline_ranking.chart_data[
            0
        ][
            "rank"
        ]
        ==
        1
    )


    baseline_share_keys = (
        share_like_keys(
            {
                "metrics":
                    baseline_ranking.metrics,

                "chart_data":
                    baseline_ranking.chart_data,
            }
        )
    )


    assert (
        baseline_share_keys
        ==
        []
    ), (
        "Legacy ranking without ShareOfTotalSpec must remain "
        "share-free."
    )


    print(
        "[PASS] legacy ranking without share preserved"
    )


    # ========================================================
    # 2. RANKING WITH SHARE CONTRACT
    # ========================================================

    ranking_contract = (
        build_contract(
            family=
                "ranking",

            include_share=
                True,
        )
    )


    assert (
        ranking_contract.share_of_total
        is not None
    )


    assert (
        ranking_contract
        .share_of_total
        .reference
        ==
        SHARE_REFERENCE
    )


    ranking_result = (
        execute_ranking_contract(
            contract=
                ranking_contract,

            dataframe=
                dataframe,

            dataset_id=
                DATASET_ID,

            dataset_filename=
                DATASET_FILENAME,
        )
    )


    assert (
        ranking_result.execution_status
        ==
        "complete"
    )


    assert (
        len(
            ranking_result.chart_data
        )
        ==
        1
    )


    ranking_row = (
        ranking_result.chart_data[
            0
        ]
    )


    assert (
        ranking_row[
            "category"
        ]
        ==
        "A"
    )


    assert close(
        ranking_row[
            "value"
        ],
        40.0,
    )


    # ========================================================
    # 3. PRE-LIMIT DENOMINATOR AUTHORITY
    #
    # 40 + 35 + 25 = 100.
    #
    # Ranking limit=1 MUST NOT make the denominator 40.
    # ========================================================

    expected_denominator = (
        100.0
    )


    expected_top_share = (
        0.4
    )


    observed_reference = (
        ranking_result
        .metrics
        .get(
            "share_of_total_reference"
        )
    )


    observed_denominator = (
        ranking_result
        .metrics
        .get(
            "share_of_total_denominator"
        )
    )


    observed_top_share = (
        ranking_result
        .metrics
        .get(
            "top_share_of_total"
        )
    )


    observed_row_share = (
        ranking_row.get(
            "share_of_total"
        )
    )


    print()
    print(
        "Ranking execution rule                  "
        f"{AI_RANKING_EXECUTION_RULE_VERSION}"
    )

    print(
        "Ranking available groups                "
        f"{ranking_result.metrics.get('available_group_count')}"
    )

    print(
        "Ranking result count                    "
        f"{ranking_result.metrics.get('result_count')}"
    )

    print(
        "Expected denominator                    "
        f"{expected_denominator:.12f}"
    )

    print(
        "Observed share reference                "
        f"{observed_reference}"
    )

    print(
        "Observed denominator                    "
        f"{observed_denominator}"
    )

    print(
        "Expected top share                      "
        f"{expected_top_share:.12f}"
    )

    print(
        "Observed top metric share               "
        f"{observed_top_share}"
    )

    print(
        "Observed top row share                  "
        f"{observed_row_share}"
    )


    # ========================================================
    # 4. GROUPED AGGREGATION WITH SHARE
    #
    # Canonical contract permits aggregation as well as ranking.
    # Objective Coverage therefore must not claim a capability
    # that deterministic execution cannot materialize.
    # ========================================================

    aggregation_result = (
        execute_aggregation_contract(
            contract=
                build_contract(
                    family=
                        "aggregation",

                    include_share=
                        True,
                ),

            dataframe=
                dataframe,

            dataset_id=
                DATASET_ID,

            dataset_filename=
                DATASET_FILENAME,
        )
    )


    assert (
        aggregation_result.execution_status
        ==
        "complete"
    )


    aggregation_expected = {
        "A":
            0.40,

        "B":
            0.35,

        "C":
            0.25,
    }


    aggregation_observed = {
        row[
            "category"
        ]:
            row.get(
                "share_of_total"
            )

        for row
        in aggregation_result.chart_data
    }


    aggregation_denominator = (
        aggregation_result
        .metrics
        .get(
            "share_of_total_denominator"
        )
    )


    aggregation_reference = (
        aggregation_result
        .metrics
        .get(
            "share_of_total_reference"
        )
    )


    print()
    print(
        "Aggregation execution rule              "
        f"{AI_AGGREGATION_EXECUTION_RULE_VERSION}"
    )

    print(
        "Aggregation expected shares             "
        f"{aggregation_expected}"
    )

    print(
        "Aggregation observed shares             "
        f"{aggregation_observed}"
    )

    print(
        "Aggregation denominator                 "
        f"{aggregation_denominator}"
    )

    print(
        "Aggregation share reference             "
        f"{aggregation_reference}"
    )


    # ========================================================
    # 5. ZERO-DENOMINATOR FAIL-CLOSED
    # ========================================================

    zero_denominator_failed_closed = (
        False
    )


    zero_error = (
        None
    )


    try:

        execute_ranking_contract(
            contract=
                ranking_contract,

            dataframe=
                zero_total_dataframe(),

            dataset_id=
                DATASET_ID,

            dataset_filename=
                DATASET_FILENAME,
        )


    except ValueError as error:

        zero_denominator_failed_closed = (
            True
        )

        zero_error = str(
            error
        )


    print()
    print(
        "Zero denominator fail-closed            "
        f"{zero_denominator_failed_closed}"
    )

    print(
        "Zero denominator error                  "
        f"{zero_error}"
    )


    # ========================================================
    # 6. RED CLASSIFICATION
    # ========================================================

    gaps: list[
        str
    ] = []


    if (
        observed_reference
        !=
        SHARE_REFERENCE
    ):

        gaps.append(
            "ranking_share_reference_gap"
        )


    if not close(
        observed_denominator,
        expected_denominator,
    ):

        gaps.append(
            "ranking_prelimit_denominator_gap"
        )


    if not close(
        observed_top_share,
        expected_top_share,
    ):

        gaps.append(
            "ranking_top_share_metric_gap"
        )


    if not close(
        observed_row_share,
        expected_top_share,
    ):

        gaps.append(
            "ranking_row_share_materialization_gap"
        )


    if (
        aggregation_reference
        !=
        SHARE_REFERENCE
    ):

        gaps.append(
            "aggregation_share_reference_gap"
        )


    if not close(
        aggregation_denominator,
        expected_denominator,
    ):

        gaps.append(
            "aggregation_share_denominator_gap"
        )


    for (
        category,
        expected_share,
    ) in aggregation_expected.items():

        if not close(
            aggregation_observed.get(
                category
            ),
            expected_share,
        ):

            gaps.append(
                (
                    "aggregation_share_materialization_gap:"
                    +
                    category
                )
            )


    if not (
        zero_denominator_failed_closed
    ):

        gaps.append(
            "share_zero_denominator_fail_closed_gap"
        )


    print()
    print(
        "Observed gaps                           "
        f"{gaps}"
    )


    if gaps:

        raise AssertionError(
            (
                "RED_EXPECTED: "
                +
                ",".join(
                    gaps
                )
            )
        )


    print()
    print(
        "PASS - deterministic share-of-total execution v0.1"
    )


if __name__ == "__main__":

    main()
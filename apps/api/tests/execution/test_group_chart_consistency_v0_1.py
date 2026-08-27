from __future__ import annotations


import pandas as pd


from app.execution.executor import (
    GROUP_CHART_CONSISTENCY_RULE_VERSION,
    execute_group_comparison,
    validate_group_chart_consistency,
)

from app.planning.schemas import (
    AnalysisCandidate,
    PlannedVariable,
)


def candidate() -> AnalysisCandidate:
    return AnalysisCandidate(
        analysis_id=
            "test:group:price-by-category",

        dataset_id=
            "dataset:lapage",

        dataset_filename=
            "lapage.csv",

        title=
            "Prix selon la catégorie",

        family=
            "group_comparison",

        priority_score=
            100,

        readiness=
            "executable_now",

        variables=[
            PlannedVariable(
                column=
                    "categ",

                role=
                    "group",

                analysis_kind=
                    "categorical",
            ),

            PlannedVariable(
                column=
                    "price",

                role=
                    "value",

                analysis_kind=
                    "quantitative",
            ),
        ],

        chart_type=
            "boxplot",

        statistical_strategy=
            "automatic_group_comparison_engine",

        reasons=[],

        limitations=[],
    )


def lapage_like_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "categ": [
                0.0,
                0.0,
                1.0,
                1.0,
                2.0,
                2.0,
            ],

            "price": [
                10.0,
                12.0,
                20.0,
                22.0,
                30.0,
                32.0,
            ],
        }
    )


def test_group_count_matches_chart_data() -> None:
    result = execute_group_comparison(
        candidate(),
        lapage_like_frame(),
    )


    assert (
        result.execution_status
        ==
        "descriptive_only"
    )


    assert (
        result.metrics[
            "group_count"
        ]
        ==
        3
    )


    assert (
        result.metrics[
            "chart_group_count"
        ]
        ==
        3
    )


    assert (
        result.metrics[
            "group_chart_consistency"
        ]
        is True
    )


    assert len(
        result.chart_data
    ) == 3


    labels = {
        item[
            "group"
        ]
        for item
        in result.chart_data
    }


    assert labels == {
        0.0,
        1.0,
        2.0,
    }


    print(
        "[PASS] 3 computed groups produce exactly 3 chart groups"
    )


def test_tampered_chart_is_rejected_by_guard() -> None:
    valid_result = (
        execute_group_comparison(
            candidate(),
            lapage_like_frame(),
        )
    )


    tampered_chart = (
        valid_result.chart_data[
            :2
        ]
    )


    consistency = (
        validate_group_chart_consistency(
            expected_group_count=
                3,

            chart_data=
                tampered_chart,
        )
    )


    assert (
        consistency[
            "consistent"
        ]
        is False
    )


    assert (
        consistency[
            "chart_group_count"
        ]
        ==
        2
    )


    print(
        "[PASS] 3 announced groups / 2 chart groups is detected"
    )


def test_version() -> None:
    assert (
        GROUP_CHART_CONSISTENCY_RULE_VERSION
        ==
        "group_chart_consistency_v0.1"
    )


    print(
        "[PASS] Group/chart consistency rule version"
    )


def main() -> None:
    print(
        "=== DATALENS GROUP / CHART CONSISTENCY v0.1 ==="
    )

    print()


    test_group_count_matches_chart_data()

    test_tampered_chart_is_rejected_by_guard()

    test_version()


    print()

    print(
        "PASS - group / chart consistency v0.1"
    )


if __name__ == "__main__":
    main()

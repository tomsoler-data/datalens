from __future__ import annotations


import pandas as pd


from app.execution.executor import (
    execute_quantitative_association,
)

from app.planning.schemas import (
    AnalysisCandidate,
    PlannedVariable,
)


def build_candidate(
) -> AnalysisCandidate:
    return (
        AnalysisCandidate(
            analysis_id=(
                "test:quantitative-descriptive-fallback"
            ),

            dataset_id=(
                "dataset:descriptive-fallback"
            ),

            dataset_filename=(
                "descriptive_fallback.csv"
            ),

            title=(
                "Analyse la relation entre x et y."
            ),

            family=(
                "quantitative_association"
            ),

            priority_score=100,

            readiness=(
                "executable_now"
            ),

            variables=[
                PlannedVariable(
                    column="x",

                    role="x",

                    analysis_kind=(
                        "quantitative"
                    ),
                ),

                PlannedVariable(
                    column="y",

                    role="y",

                    analysis_kind=(
                        "quantitative"
                    ),
                ),
            ],

            chart_type="scatter",

            statistical_strategy=(
                "correlation_decision_engine"
            ),
        )
    )


def build_non_monotonic_frame(
) -> pd.DataFrame:
    x = list(
        range(
            -10,
            11,
        )
    )


    return (
        pd.DataFrame(
            {
                "x":
                    x,

                "y":
                    [
                        value
                        *
                        value

                        for value
                        in x
                    ],
            }
        )
    )


def main() -> None:
    candidate = (
        build_candidate()
    )

    dataframe = (
        build_non_monotonic_frame()
    )


    result = (
        execute_quantitative_association(
            candidate,
            dataframe,
        )
    )


    # ========================================================
    # INFERENTIAL AUTHORITY REMAINS FAIL-CLOSED
    # ========================================================

    assert (
        result.execution_status
        ==
        "needs_information"
    ), result.model_dump()


    assert (
        result.statistical_decision
        is not None
    )


    assert (
        result.statistical_decision[
            "status"
        ]
        !=
        "selected"
    ), result.statistical_decision


    assert (
        result.statistical_result
        is None
    )


    assert (
        result.metrics[
            "inference_performed"
        ]
        is False
    )


    assert (
        result.metrics[
            "p_value"
        ]
        is None
    )


    assert (
        result.metrics[
            "statistically_significant"
        ]
        is None
    )


    # ========================================================
    # DESCRIPTIVE QUANTIFICATION IS STILL AVAILABLE
    # ========================================================

    assert isinstance(
        result.metrics[
            "pearson_r"
        ],
        float,
    )


    assert isinstance(
        result.metrics[
            "spearman_rho"
        ],
        float,
    )


    assert (
        result.metrics[
            "interpretation_scope"
        ]
        ==
        "descriptive_only"
    )


    assert (
        result.metrics[
            "descriptive_fallback_rule_version"
        ]
        ==
        "descriptive_correlation_fallback_v0.1"
    )


    # ========================================================
    # SCATTER REMAINS AVAILABLE
    # ========================================================

    assert (
        result.chart_type
        ==
        "scatter"
    )


    assert (
        len(
            result.chart_data
        )
        ==
        len(
            dataframe
        )
    )


    assert all(
        (
            set(
                point.keys()
            )
            ==
            {
                "x",
                "y",
            }
        )

        for point
        in result.chart_data
    )


    assert (
        result.metrics[
            "chart_point_count"
        ]
        ==
        len(
            dataframe
        )
    )


    # ========================================================
    # USER-FACING SCOPE MUST BE EXPLICIT
    # ========================================================

    joined_summary = (
        " ".join(
            result.summary
        )
    )


    assert (
        "Pearson r descriptif"
        in
        joined_summary
    )


    assert (
        "Spearman rho descriptif"
        in
        joined_summary
    )


    joined_limitations = (
        " ".join(
            result.limitations
        )
    )


    assert (
        "No p-value"
        in
        joined_limitations
    )


    print(
        "[PASS] inferential selection remains fail-closed"
    )

    print(
        "[PASS] Pearson r remains descriptive only"
    )

    print(
        "[PASS] Spearman rho remains descriptive only"
    )

    print(
        "[PASS] no p-value is invented"
    )

    print(
        "[PASS] descriptive scatter is preserved"
    )

    print()
    print(
        "PASS - quantitative association "
        "descriptive fallback v0.1"
    )


if __name__ == "__main__":
    main()

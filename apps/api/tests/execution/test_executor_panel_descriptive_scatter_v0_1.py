from __future__ import annotations

import pandas as pd

from app.execution.executor import (
    execute_quantitative_association,
)

from app.planning.schemas import (
    AnalysisCandidate,
    PlannedVariable,
)


def build_candidate() -> AnalysisCandidate:
    return AnalysisCandidate(
        analysis_id=
            "test:panel-descriptive-scatter",

        dataset_id=
            "dataset:test",

        dataset_filename=
            "panel.csv",

        title=(
            "Existe-t-il une relation entre "
            "la quantité commandée et le prix unitaire ?"
        ),

        family=
            "quantitative_association",

        priority_score=
            100,

        readiness=
            "executable_now",

        variables=[
            PlannedVariable(
                column=
                    "quantity",

                role=
                    "x",

                analysis_kind=
                    "quantitative",
            ),

            PlannedVariable(
                column=
                    "unit_price",

                role=
                    "y",

                analysis_kind=
                    "quantitative",
            ),
        ],

        chart_type=
            "scatter",

        statistical_strategy=
            "correlation_decision_engine",
    )


def build_panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [
                "c1",
                "c1",
                "c2",
                "c2",
                "c3",
                "c3",
            ],

            "order_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                    "2026-01-03",
                    "2026-02-03",
                    "2026-01-05",
                    "2026-02-05",
                ]
            ),

            "quantity": [
                1,
                2,
                3,
                1,
                4,
                2,
            ],

            "unit_price": [
                10.0,
                12.0,
                8.0,
                15.0,
                7.0,
                11.0,
            ],
        }
    )


def main() -> None:
    candidate = build_candidate()
    dataframe = build_panel_frame()

    result = execute_quantitative_association(
        candidate,
        dataframe,
    )

    assert (
        result.execution_status
        ==
        "needs_specialized_method"
    )

    assert result.chart_type == "scatter"

    assert len(
        result.chart_data
    ) == len(
        dataframe
    )

    assert all(
        set(
            point.keys()
        )
        ==
        {
            "x",
            "y",
        }

        for point
        in result.chart_data
    )

    assert result.metrics[
        "valid_pairs"
    ] == 6

    assert result.metrics[
        "chart_point_count"
    ] == 6

    assert (
        result.metrics[
            "inference_performed"
        ]
        is False
    )

    assert (
        result.metrics[
            "interpretation_scope"
        ]
        ==
        "descriptive_only"
    )

    repeated = result.metrics[
        "repeated_measure_structure"
    ]

    assert repeated[
        "entity_column"
    ] == "customer_id"

    assert repeated[
        "temporal_column"
    ] == "order_date"

    assert repeated[
        "repeated_entity_count"
    ] == 3

    assert result.statistical_decision is None
    assert result.statistical_result is None

    assert any(
        "No inferential correlation"
        in limitation

        for limitation
        in result.limitations
    )

    print(
        "[PASS] repeated-measures association keeps "
        "needs_specialized_method"
    )

    print(
        "[PASS] descriptive scatter exposes all complete x/y pairs"
    )

    print(
        "[PASS] repeated-measures guard still blocks inferential correlation"
    )

    print(
        "[PASS] descriptive/inferential scope is explicit in result metrics"
    )

    print()
    print(
        "PASS - executor panel descriptive scatter v0.1"
    )


if __name__ == "__main__":
    main()

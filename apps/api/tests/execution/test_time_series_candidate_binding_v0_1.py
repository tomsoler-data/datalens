from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from app.execution.single_dataset import (
    execute_time_series,
)

from app.execution.structure import (
    detect_observation_structure,
)


def variable(
    *,
    column: str,
    analysis_kind: str,
    semantic_role: str,
):
    return SimpleNamespace(
        column=
            column,

        analysis_kind=
            analysis_kind,

        semantic_role=
            semantic_role,
    )


def build_candidate():
    return SimpleNamespace(
        analysis_id=(
            "dataset:test:"
            "time:birth:price"
        ),

        title=
            "Évolution de price",

        family=
            "time_series",

        variables=[
            variable(
                column=
                    "birth",

                analysis_kind=
                    "temporal",

                semantic_role=
                    "time",
            ),

            variable(
                column=
                    "price",

                analysis_kind=
                    "quantitative",

                semantic_role=
                    "measure",
            ),
        ],

        limitations=[],
    )


def test_candidate_time_binding_wins_over_structure(
) -> None:
    dataframe = pd.DataFrame(
        {
            "date":
                pd.date_range(
                    "2024-01-01",
                    periods=6,
                    freq="D",
                ),

            "birth":
                [
                    1980,
                    1980,
                    1990,
                    1990,
                    2000,
                    2000,
                ],

            "price":
                [
                    10.0,
                    12.0,
                    20.0,
                    22.0,
                    30.0,
                    32.0,
                ],
        }
    )


    structure = (
        detect_observation_structure(
            dataframe
        )
    )


    print(
        "Structure time column:",
        structure.time_column,
    )


    # This is the exact situation that exposed the bug:
    # the generic dataframe structure prefers `date`.
    assert (
        structure.time_column
        ==
        "date"
    )


    candidate = (
        build_candidate()
    )


    result = (
        execute_time_series(
            candidate,
            dataframe=
                dataframe,
            dataset_id=
                "dataset:test",
            dataset_name=
                "test.csv",
        )
    )


    print(
        "Candidate analysis_id:",
        result.analysis_id,
    )

    print(
        "Executed time column:",
        result.metrics.get(
            "time_column"
        ),
    )

    print(
        "Executed measure:",
        result.metrics.get(
            "measure_column"
        ),
    )

    print(
        "Period count:",
        result.metrics.get(
            "period_count"
        ),
    )


    assert (
        result.analysis_id
        ==
        "dataset:test:time:birth:price"
    )


    assert (
        result.metrics.get(
            "time_column"
        )
        ==
        "birth"
    )


    assert (
        result.metrics.get(
            "measure_column"
        )
        ==
        "price"
    )


    assert (
        result.metrics.get(
            "period_count"
        )
        ==
        3
    )


    print(
        "[PASS] Candidate time binding is preserved"
    )


def main(
) -> None:
    print(
        "=== DATALENS TIME-SERIES BINDING v0.1 ==="
    )

    print()


    test_candidate_time_binding_wins_over_structure()


    print()
    print(
        "PASS - time-series binding v0.1"
    )


if __name__ == "__main__":
    main()

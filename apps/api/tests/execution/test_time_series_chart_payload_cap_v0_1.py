from __future__ import annotations

import inspect

import pandas as pd

from app.execution.single_dataset import (
    MAX_CHART_POINTS,
    bounded_chart_frame,
    execute_time_series,
)


def test_large_chart_frame_is_bounded(
) -> None:
    row_count = (
        MAX_CHART_POINTS
        *
        3
        +
        17
    )


    dataframe = pd.DataFrame(
        {
            "period":
                range(
                    row_count
                ),

            "value":
                [
                    float(
                        index
                    )
                    for index
                    in range(
                        row_count
                    )
                ],
        }
    )


    bounded = (
        bounded_chart_frame(
            dataframe
        )
    )


    assert (
        len(
            bounded
        )
        ==
        MAX_CHART_POINTS
    )


    assert (
        bounded.iloc[
            0
        ][
            "period"
        ]
        ==
        0
    )


    assert (
        bounded.iloc[
            -1
        ][
            "period"
        ]
        ==
        row_count
        -
        1
    )


    print(
        "[PASS] Large chart payload is bounded "
        f"to {MAX_CHART_POINTS} points"
    )


def test_small_chart_frame_is_preserved(
) -> None:
    dataframe = pd.DataFrame(
        {
            "period":
                range(
                    25
                ),

            "value":
                range(
                    25
                ),
        }
    )


    bounded = (
        bounded_chart_frame(
            dataframe
        )
    )


    assert (
        len(
            bounded
        )
        ==
        25
    )


    assert (
        bounded[
            "period"
        ].tolist()
        ==
        dataframe[
            "period"
        ].tolist()
    )


    print(
        "[PASS] Small chart payload is preserved"
    )


def test_time_series_uses_visual_boundary(
) -> None:
    source = (
        inspect.getsource(
            execute_time_series
        )
    )


    call_count = (
        source.count(
            "bounded_chart_frame("
        )
    )


    assert (
        call_count
        ==
        2
    )


    assert (
        "in ordered.iterrows()"
        not in source
    )


    assert (
        "in grouped.iterrows()"
        not in source
    )


    print(
        "[PASS] Both time-series branches use "
        "the bounded visual payload"
    )


def main(
) -> None:
    print(
        "=== DATALENS TIME-SERIES CHART PAYLOAD CAP v0.1 ==="
    )

    print()


    test_large_chart_frame_is_bounded()

    test_small_chart_frame_is_preserved()

    test_time_series_uses_visual_boundary()


    print()
    print(
        "PASS - time-series chart payload cap v0.1"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from app.reporting.selected_report_pdf import (
    format_chart_value_label,
    time_series_line_chart,
    time_series_value_label_indices,
)


def main(
) -> None:
    print(
        "=== DATALENS PDF TIME-SERIES VALUE LABELS v0.1 ==="
    )

    print()


    # ========================================================
    # 24 MONTHS — CLIENT COUNT STYLE
    # ========================================================

    values = [
        5_600,
        5_520,
        5_410,
        5_460,
        5_530,
        5_420,
        5_800,
        8_190,
        6_400,
        6_350,
        6_180,
        5_900,
        6_350,
        5_750,
        6_450,
        5_850,
        6_150,
        6_300,
        6_050,
        6_110,
        6_090,
        6_150,
        6_500,
        5_587,
    ]


    indices = (
        time_series_value_label_indices(
            [
                float(
                    value
                )
                for value
                in values
            ]
        )
    )


    print(
        "24-point selected indices:",
        indices,
    )


    assert (
        0
        in indices
    )


    assert (
        23
        in indices
    )


    assert (
        values.index(
            min(
                values
            )
        )
        in indices
    )


    assert (
        values.index(
            max(
                values
            )
        )
        in indices
    )


    assert (
        len(
            indices
        )
        <=
        10
    )


    print(
        "[PASS] 24-point chart gets a sparse readable label set"
    )


    # ========================================================
    # NUMBER FORMATTING
    # ========================================================

    assert (
        format_chart_value_label(
            5_888
        )
        ==
        "5 888"
    )


    assert (
        format_chart_value_label(
            535_571.50
        )
        ==
        "536 k"
    )


    assert (
        format_chart_value_label(
            1_250_000
        )
        ==
        "1,2 M"
    )


    print(
        "[PASS] Count labels stay exact and large "
        "monetary values stay compact"
    )


    # ========================================================
    # DRAWING SMOKE TEST
    # ========================================================

    data = [
        {
            "period":
                f"2024-{index + 1:02d}-01T00:00:00",

            "value":
                value,
        }

        for index, value
        in enumerate(
            values[
                :12
            ]
        )
    ]


    drawing = (
        time_series_line_chart(
            data
        )
    )


    texts = [
        getattr(
            item,
            "text",
            None,
        )

        for item
        in drawing.contents
    ]


    expected_labels = {
        format_chart_value_label(
            float(
                values[
                    index
                ]
            )
        )

        for index
        in time_series_value_label_indices(
            [
                float(
                    value
                )

                for value
                in values[
                    :12
                ]
            ]
        )
    }


    missing = [
        label

        for label
        in expected_labels

        if label
        not in texts
    ]


    assert not missing, (
        "Missing rendered value labels: "
        f"{missing}"
    )


    print(
        "[PASS] Selected values are actually rendered "
        "inside the PDF Drawing"
    )


    print()
    print(
        "PASS - PDF time-series value labels v0.1"
    )


if __name__ == "__main__":
    main()

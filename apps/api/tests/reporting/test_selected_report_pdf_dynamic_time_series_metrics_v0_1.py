from __future__ import annotations

from app.reporting.selected_report_pdf import (
    selected_metrics,
    time_series_period_label,
)


def main() -> None:
    print(
        "===== DATALENS PDF DYNAMIC TIME-SERIES METRICS v0.1 ====="
    )

    print()


    assert (
        time_series_period_label(
            "week"
        )
        ==
        "Semaine"
    )

    assert (
        time_series_period_label(
            "quarter"
        )
        ==
        "Trimestre"
    )


    print(
        "[PASS] time granularity labels are presentation-only"
    )


    result = {
        "kind":
            "revenue_moving_average",

        "metrics":
            {
                "valid_observations":
                    687534,

                "period_count":
                    105,

                "aggregation_period":
                    "week",

                "moving_average_window":
                    4,

                "total_revenue":
                    12000000.0,
            },
    }


    output = dict(
        selected_metrics(
            family=
                "time_series",

            result=
                result,
        )
    )


    assert (
        output[
            "P\u00e9riode d'agr\u00e9gation"
        ]
        ==
        "Semaine"
    )

    assert (
        output[
            "Fen\u00eatre moyenne mobile"
        ]
        ==
        "4 p\u00e9riodes"
    )

    assert (
        output[
            "P\u00e9riodes"
        ]
        ==
        105
    )

    assert (
        output[
            "Observations"
        ]
        ==
        687534
    )

    assert (
        output[
            "Total"
        ]
        ==
        12000000.0
    )


    print(
        "[PASS] requested moving-average PDF exposes Semaine"
    )

    print(
        "[PASS] requested moving-average PDF exposes window=4"
    )

    print(
        "[PASS] server-owned analytical metrics are preserved"
    )


    customers_result = {
        "kind":
            "customers_by_period",

        "metrics":
            {
                "valid_observations":
                    100,

                "period_count":
                    24,

                "distinct_customers_total":
                    8600,
            },
    }


    customers_output = dict(
        selected_metrics(
            family=
                "time_series",

            result=
                customers_result,
        )
    )


    assert (
        "P\u00e9riode d'agr\u00e9gation"
        not in
        customers_output
    )

    assert (
        "Fen\u00eatre moyenne mobile"
        not in
        customers_output
    )


    print(
        "[PASS] other time-series analyses keep generic PDF metrics"
    )

    print()

    print(
        "PASS - PDF dynamic time-series metrics v0.1"
    )


if __name__ == "__main__":
    main()

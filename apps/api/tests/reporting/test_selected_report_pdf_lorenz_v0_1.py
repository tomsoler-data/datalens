from reportlab.graphics.shapes import (
    Drawing,
    PolyLine,
)

from app.reporting.selected_report_pdf import (
    lorenz_chart,
    result_chart,
)


print()
print(
    "===== DATALENS SELECTED REPORT LORENZ v0.1 ====="
)
print()


chart_data = [
    {
        "population_share":
            0.0,

        "revenue_share":
            0.0,

        "equality_share":
            0.0,
    },
    {
        "population_share":
            0.5,

        "revenue_share":
            0.2,

        "equality_share":
            0.5,
    },
    {
        "population_share":
            1.0,

        "revenue_share":
            1.0,

        "equality_share":
            1.0,
    },
]


drawing = lorenz_chart(
    chart_data
)


assert isinstance(
    drawing,
    Drawing,
)


polylines = [
    item
    for item in drawing.contents
    if isinstance(
        item,
        PolyLine,
    )
]


assert (
    len(polylines)
    >=
    2
)


print(
    "[PASS] Lorenz renderer returns a Drawing"
)

print(
    "[PASS] observed + equality curves are rendered"
)


routed = result_chart(
    {
        "chart_type":
            "lorenz",

        "chart_data":
            chart_data,

        "metrics": {
            "chart_point_count":
                3,

            "gini_coefficient":
                0.44,
        },
    }
)


assert isinstance(
    routed,
    Drawing,
)


print(
    "[PASS] result_chart routes chart_type='lorenz'"
)


empty = lorenz_chart(
    [
        {
            "population_share":
                0.5,

            "revenue_share":
                0.2,

            "equality_share":
                0.5,
        }
    ]
)


assert isinstance(
    empty,
    Drawing,
)


print(
    "[PASS] insufficient-point case remains safe"
)

print()
print(
    "PASS - selected report Lorenz chart v0.1"
)

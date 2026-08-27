from __future__ import annotations

from app.reporting.selected_report_pdf import (
    bounded_pdf_scatter_data,
    heatmap_chart,
    pdf_variable_label,
    result_chart,
    scatter_chart,
)


def drawing_texts(
    drawing,
) -> list[str]:
    return [
        getattr(
            item,
            "text",
            None,
        )

        for item
        in drawing.contents

        if getattr(
            item,
            "text",
            None,
        )
        is not None
    ]


def main(
) -> None:
    print(
        "=== DATALENS PDF SCATTER + HEATMAP v0.1 ==="
    )

    print()


    # ========================================================
    # SCATTER VISUAL BOUND
    # ========================================================

    scatter_data = [
        {
            "x":
                float(
                    index
                    %
                    80
                ),

            "y":
                float(
                    index
                ),
        }

        for index
        in range(
            2_000
        )
    ]


    bounded = (
        bounded_pdf_scatter_data(
            scatter_data
        )
    )


    assert (
        len(
            bounded
        )
        ==
        350
    )


    assert (
        bounded[
            0
        ]
        ==
        scatter_data[
            0
        ]
    )


    assert (
        bounded[
            -1
        ]
        ==
        scatter_data[
            -1
        ]
    )


    print(
        "[PASS] Scatter PDF payload is deterministically bounded to 350 points"
    )


    # ========================================================
    # VARIABLE PRESENTATION
    # ========================================================

    assert (
        pdf_variable_label(
            "age_at_first_purchase"
        )
        ==
        "\u00c2ge au premier achat"
    )


    assert (
        pdf_variable_label(
            "average_basket"
        )
        ==
        "Panier moyen"
    )


    print(
        "[PASS] Analytical variable names have presentation labels"
    )


    # ========================================================
    # SCATTER DRAWING
    # ========================================================

    scatter = scatter_chart(
        scatter_data,
        x_label=
            "age_at_first_purchase",
        y_label=
            "average_basket",
    )


    assert (
        scatter.width
        ==
        500
    )


    scatter_text = (
        drawing_texts(
            scatter
        )
    )


    assert any(
        "\u00c2ge au premier achat"
        in text
        for text
        in scatter_text
    )


    assert any(
        "Panier moyen"
        in text
        for text
        in scatter_text
    )


    print(
        "[PASS] Scatter renderer produces labelled PDF drawing"
    )


    # ========================================================
    # HEATMAP DRAWING
    # ========================================================

    heatmap_data = [
        {
            "x":
                "f",
            "y":
                "0",
            "count":
                206_103,
        },
        {
            "x":
                "f",
            "y":
                "1",
            "count":
                119_307,
        },
        {
            "x":
                "f",
            "y":
                "2",
            "count":
                17_283,
        },
        {
            "x":
                "m",
            "y":
                "0",
            "count":
                206_000,
        },
        {
            "x":
                "m",
            "y":
                "1",
            "count":
                120_000,
        },
        {
            "x":
                "m",
            "y":
                "2",
            "count":
                18_841,
        },
    ]


    heatmap = heatmap_chart(
        heatmap_data,
        x_label=
            "gender",
        y_label=
            "category",
    )


    assert (
        heatmap.width
        ==
        500
    )


    heatmap_text = (
        drawing_texts(
            heatmap
        )
    )


    assert (
        "206 103"
        in heatmap_text
    )


    assert any(
        "Genre"
        in text
        for text
        in heatmap_text
    )


    assert any(
        "Cat\u00e9gorie"
        in text
        for text
        in heatmap_text
    )


    print(
        "[PASS] Heatmap renderer exposes the contingency counts"
    )


    # ========================================================
    # result_chart DISPATCH
    # ========================================================

    scatter_result = {
        "chart_type":
            "scatter",

        "chart_data":
            scatter_data,

        "metrics": {
            "x_column":
                "age_at_first_purchase",

            "y_column":
                "average_basket",
        },
    }


    heatmap_result = {
        "chart_type":
            "heatmap",

        "chart_data":
            heatmap_data,

        "metrics": {
            "x_column":
                "gender",

            "y_column":
                "category",
        },
    }


    assert (
        result_chart(
            scatter_result
        )
        is not None
    )


    assert (
        result_chart(
            heatmap_result
        )
        is not None
    )


    print(
        "[PASS] result_chart routes scatter and heatmap"
    )


    print()
    print(
        "PASS - PDF scatter + heatmap v0.1"
    )


if __name__ == "__main__":
    main()

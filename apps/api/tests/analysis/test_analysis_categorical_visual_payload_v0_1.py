from __future__ import annotations

import math

import pandas as pd


from app.discovery.schemas import (
    DiscoveredAnalysis,
    DiscoveredVariable,
)

from app.execution.single_dataset import (
    execute_categorical_association,
)


# ============================================================
# FIXTURE
# ============================================================


def build_dataframe(
) -> pd.DataFrame:
    departments: list[
        str
    ] = []

    regions: list[
        str
    ] = []


    def add_rows(
        department: str,
        region: str,
        count: int,
    ) -> None:
        departments.extend(
            [
                department
            ]
            *
            count
        )

        regions.extend(
            [
                region
            ]
            *
            count
        )


    add_rows(
        "Sales",
        "North",
        12,
    )

    add_rows(
        "Sales",
        "South",
        8,
    )

    add_rows(
        "IT",
        "North",
        5,
    )

    add_rows(
        "IT",
        "South",
        15,
    )


    return pd.DataFrame(
        {
            "department":
                departments,

            "region":
                regions,
        }
    )


# ============================================================
# CANDIDATE
# ============================================================


def build_candidate(
) -> DiscoveredAnalysis:
    return DiscoveredAnalysis(
        analysis_id=(
            "dataset:0001:"
            "categorical:"
            "department:"
            "region"
        ),

        scope=
            "single_dataset",

        family=
            "categorical_association",

        title=(
            "Association entre "
            "department et region"
        ),

        priority_score=
            80.0,

        readiness=
            "executable_now",

        datasets=[
            "employees.csv"
        ],

        dataset_ids=[
            "dataset:0001"
        ],

        variables=[
            DiscoveredVariable(
                dataset_id=
                    "dataset:0001",

                dataset_filename=
                    "employees.csv",

                column=
                    "department",

                role=
                    "x",

                analysis_kind=
                    "categorical",

                semantic_role=
                    "category",

                concepts=[],
            ),

            DiscoveredVariable(
                dataset_id=
                    "dataset:0001",

                dataset_filename=
                    "employees.csv",

                column=
                    "region",

                role=
                    "y",

                analysis_kind=
                    "categorical",

                semantic_role=
                    "region",

                concepts=[],
            ),
        ],

        chart_type=
            "heatmap",

        execution_strategy=
            "chi_square_decision_engine",

        why_interesting=[
            (
                "Synthetic categorical "
                "association test."
            )
        ],

        limitations=[],

        observed_signals={
            "valid_observations":
                40,

            "left_levels":
                2,

            "right_levels":
                2,
        },

        redundancy_key=(
            "categorical:"
            "dataset:0001:"
            "department:"
            "region"
        ),
    )


# ============================================================
# TEST
# ============================================================


def test_categorical_heatmap_payload(
) -> None:
    dataframe = (
        build_dataframe()
    )

    candidate = (
        build_candidate()
    )


    result = (
        execute_categorical_association(
            candidate,

            dataframe=
                dataframe,

            dataset_id=
                "dataset:0001",

            dataset_name=
                "employees.csv",
        )
    )


    print(
        "\n=== CATEGORICAL ASSOCIATION ==="
    )

    print(
        f"Status: "
        f"{result.execution_status}"
    )

    print(
        f"Chart type: "
        f"{result.chart_type}"
    )

    print(
        f"Chart cells: "
        f"{len(result.chart_data)}"
    )

    print(
        f"Valid observations: "
        f"{result.valid_observations}"
    )

    print(
        f"Chi-square: "
        f"{result.metrics['chi_square_statistic']}"
    )

    print(
        f"Cramer's V: "
        f"{result.metrics['cramers_v']}"
    )


    # ========================================================
    # EXECUTION CONTRACT
    # ========================================================

    assert (
        result.execution_status
        ==
        "descriptive_only"
    )

    assert (
        result.valid_observations
        ==
        40
    )


    # ========================================================
    # VISUAL CONTRACT
    # ========================================================

    assert (
        result.chart_type
        ==
        "heatmap"
    )

    assert (
        len(
            result.chart_data
        )
        ==
        4
    )


    cells = {
        (
            cell[
                "x"
            ],
            cell[
                "y"
            ],
        ):
            cell[
                "count"
            ]

        for cell
        in result.chart_data
    }


    print(
        "\n=== HEATMAP CELLS ==="
    )


    for (
        coordinates,
        count,
    ) in sorted(
        cells.items()
    ):
        print(
            f"{coordinates}: "
            f"{count}"
        )


    assert (
        cells[
            (
                "Sales",
                "North",
            )
        ]
        ==
        12
    )

    assert (
        cells[
            (
                "Sales",
                "South",
            )
        ]
        ==
        8
    )

    assert (
        cells[
            (
                "IT",
                "North",
            )
        ]
        ==
        5
    )

    assert (
        cells[
            (
                "IT",
                "South",
            )
        ]
        ==
        15
    )


    # ========================================================
    # VISUAL TOTAL MUST MATCH ANALYTICAL SAMPLE
    # ========================================================

    visual_total = sum(
        int(
            cell[
                "count"
            ]
        )

        for cell
        in result.chart_data
    )


    print(
        "\n=== VISUAL EVIDENCE CONSISTENCY ==="
    )

    print(
        f"Heatmap total: "
        f"{visual_total}"
    )

    print(
        f"Analytical n: "
        f"{result.valid_observations}"
    )


    assert (
        visual_total
        ==
        result.valid_observations
    )


    # ========================================================
    # STATISTICAL PAYLOAD MUST REMAIN AVAILABLE
    # ========================================================

    chi_square = (
        result.metrics[
            "chi_square_statistic"
        ]
    )

    cramers_v = (
        result.metrics[
            "cramers_v"
        ]
    )


    assert isinstance(
        chi_square,
        float,
    )

    assert (
        math.isfinite(
            chi_square
        )
    )

    assert (
        cramers_v
        is not None
    )

    assert (
        0.0
        <=
        cramers_v
        <=
        1.0
    )


    # ========================================================
    # COLUMN METADATA
    # ========================================================

    assert (
        result.metrics[
            "left_column"
        ]
        ==
        "department"
    )

    assert (
        result.metrics[
            "right_column"
        ]
        ==
        "region"
    )

    assert (
        result.metrics[
            "row_levels"
        ]
        ==
        2
    )

    assert (
        result.metrics[
            "column_levels"
        ]
        ==
        2
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        "\n========================================"
    )

    print(
        (
            "DataLens Categorical Visual "
            "Payload v0.1"
        )
    )

    print(
        "========================================"
    )


    test_categorical_heatmap_payload()


    print(
        "\n========================================"
    )

    print(
        (
            "PASS - categorical visual "
            "payload v0.1"
        )
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
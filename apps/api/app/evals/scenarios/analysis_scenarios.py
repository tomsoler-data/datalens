from __future__ import annotations

import pandas as pd

from app.evals.analysis_benchmark import (
    AnalysisBenchmarkExpectation,
    AnalysisBenchmarkScenario,
    BenchmarkVariableExpectation,
)


# ============================================================
# HELPERS
# ============================================================


def _dataset_record(
    *,
    dataset_id: str,
    filename: str,
    dataframe: pd.DataFrame,
) -> dict:
    return {
        "dataset_id":
            dataset_id,

        "filename":
            filename,

        "dataframe":
            dataframe,
    }


# ============================================================
# SCENARIO 1 — SIMPLE BUSINESS GROUPING
# ============================================================


def ecommerce_core_scenario(
) -> AnalysisBenchmarkScenario:
    """
    Stable low-cardinality analytical case.

    The dataset is deliberately small in schema:
    - one categorical business dimension;
    - one temporal axis;
    - one quantitative measure.

    This avoids accidental budget competition and makes the
    expected group comparison a strong regression anchor.
    """

    dataframe = pd.DataFrame(
        {
            "order_date":
                pd.date_range(
                    "2026-01-01",
                    periods=40,
                    freq="D",
                ),

            "category":
                [
                    [
                        "Furniture",
                        "Electronics",
                        "Accessories",
                    ][
                        index
                        %
                        3
                    ]

                    for index
                    in range(
                        40
                    )
                ],

            "unit_price":
                [
                    20.0
                    +
                    (
                        index
                        *
                        1.25
                    )

                    for index
                    in range(
                        40
                    )
                ],
        }
    )


    return (
        AnalysisBenchmarkScenario(
            scenario_id=
                "ecommerce_core_v0.1",

            description=(
                "Low-cardinality business grouping should be "
                "discovered and selected."
            ),

            split=
                "test",

            frozen=
                True,

            datasets=(
                _dataset_record(
                    dataset_id=
                        "dataset:ecommerce_core",

                    filename=
                        "ecommerce_core.csv",

                    dataframe=
                        dataframe,
                ),
            ),

            expectations=(
                AnalysisBenchmarkExpectation(
                    expectation_id=
                        "select-unit-price-by-category",

                    description=(
                        "A clean low-cardinality business dimension "
                        "must remain eligible for automatic "
                        "exploration."
                    ),

                    family=
                        "group_comparison",

                    variables=[
                        BenchmarkVariableExpectation(
                            role=
                                "group",

                            column=
                                "category",
                        ),

                        BenchmarkVariableExpectation(
                            role=
                                "value",

                            column=
                                "unit_price",
                        ),
                    ],

                    allowed_decisions=[
                        "selected"
                    ],
                ),
            ),

            min_discovered_count=
                4,

            max_selected_count=
                36,
        )
    )


# ============================================================
# SCENARIO 2 — FRAGMENTED GROUP DIMENSION
# ============================================================


def fragmented_group_scenario(
) -> AnalysisBenchmarkScenario:
    """
    A categorical variable is technically valid for Discovery
    but analytically too fragmented for automatic selection.
    """

    dataframe = pd.DataFrame(
        {
            "segment":
                [
                    f"G{index % 14}"

                    for index
                    in range(
                        40
                    )
                ],

            "quantity":
                [
                    index % 5 + 1

                    for index
                    in range(
                        40
                    )
                ],
        }
    )


    return (
        AnalysisBenchmarkScenario(
            scenario_id=
                "fragmented_group_v0.1",

            description=(
                "A 14-level dimension over 40 rows should be "
                "discovered but deferred by the Analytical Value "
                "Guard."
            ),

            split=
                "test",

            frozen=
                True,

            datasets=(
                _dataset_record(
                    dataset_id=
                        "dataset:fragmented_group",

                    filename=
                        "fragmented_group.csv",

                    dataframe=
                        dataframe,
                ),
            ),

            expectations=(
                AnalysisBenchmarkExpectation(
                    expectation_id=
                        "defer-fragmented-segment",

                    description=(
                        "The group comparison is mathematically "
                        "possible but too fragmented for automatic "
                        "exploration."
                    ),

                    family=
                        "group_comparison",

                    variables=[
                        BenchmarkVariableExpectation(
                            role=
                                "group",

                            column=
                                "segment",
                        ),

                        BenchmarkVariableExpectation(
                            role=
                                "value",

                            column=
                                "quantity",
                        ),
                    ],

                    allowed_decisions=[
                        "deferred"
                    ],

                    allowed_reason_codes=[
                        "fragmented_group_dimension"
                    ],
                ),
            ),

            min_discovered_count=
                2,

            max_selected_count=
                36,
        )
    )


# ============================================================
# SCENARIO 3 — TWO TEMPORAL AXES
# ============================================================


def time_series_axes_scenario(
) -> AnalysisBenchmarkScenario:
    """
    Regression scenario for Discovery Candidate Identity v0.1.

    The same measure is paired with two different temporal axes.
    Both analytical structures must remain independently
    discoverable.
    """

    dataframe = pd.DataFrame(
        {
            "order_date":
                pd.date_range(
                    "2026-01-01",
                    periods=40,
                    freq="D",
                ),

            "signup_date":
                pd.date_range(
                    "2025-11-01",
                    periods=40,
                    freq="2D",
                ),

            "quantity":
                [
                    index % 5 + 1

                    for index
                    in range(
                        40
                    )
                ],
        }
    )


    return (
        AnalysisBenchmarkScenario(
            scenario_id=
                "time_series_axes_v0.1",

            description=(
                "One measure paired with two temporal axes must "
                "produce two distinct analytical candidates."
            ),

            split=
                "test",

            frozen=
                True,

            datasets=(
                _dataset_record(
                    dataset_id=
                        "dataset:time_axes",

                    filename=
                        "time_axes.csv",

                    dataframe=
                        dataframe,
                ),
            ),

            expectations=(
                AnalysisBenchmarkExpectation(
                    expectation_id=
                        "discover-order-date-quantity",

                    description=
                        "Quantity by order_date must be discoverable.",

                    family=
                        "time_series",

                    variables=[
                        BenchmarkVariableExpectation(
                            role=
                                "time",

                            column=
                                "order_date",
                        ),

                        BenchmarkVariableExpectation(
                            role=
                                "value",

                            column=
                                "quantity",
                        ),
                    ],

                    allowed_decisions=[
                        "selected"
                    ],
                ),

                AnalysisBenchmarkExpectation(
                    expectation_id=
                        "discover-signup-date-quantity",

                    description=
                        "Quantity by signup_date must be discoverable.",

                    family=
                        "time_series",

                    variables=[
                        BenchmarkVariableExpectation(
                            role=
                                "time",

                            column=
                                "signup_date",
                        ),

                        BenchmarkVariableExpectation(
                            role=
                                "value",

                            column=
                                "quantity",
                        ),
                    ],

                    allowed_decisions=[
                        "selected"
                    ],
                ),
            ),

            min_discovered_count=
                4,

            max_selected_count=
                36,
        )
    )


# ============================================================
# REGISTRY
# ============================================================


def build_analysis_eval_scenarios(
) -> tuple[
    AnalysisBenchmarkScenario,
    ...,
]:
    """
    Frozen v0.1 Discovery/Prioritization suite.

    Scenario order is stable by design so console and CI output
    remain deterministic.
    """

    return (
        ecommerce_core_scenario(),
        fragmented_group_scenario(),
        time_series_axes_scenario(),
    )

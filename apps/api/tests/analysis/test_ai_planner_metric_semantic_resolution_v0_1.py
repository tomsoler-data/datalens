from __future__ import annotations


from typing import (
    Callable,
    Optional,
)


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    apply_deterministic_abstention_guards,
)


# ============================================================
# DATASET PROFILE HELPERS
# ============================================================


def categorical_column(
    name: str,
) -> PlannerColumnProfile:
    return (
        PlannerColumnProfile(
            name=
                name,

            dtype=
                "object",

            analysis_kind=
                "categorical",

            missing_ratio=
                0.0,

            unique_count=
                3,

            unique_candidate=
                False,
        )
    )


def quantitative_column(
    name: str,
) -> PlannerColumnProfile:
    return (
        PlannerColumnProfile(
            name=
                name,

            dtype=
                "float64",

            analysis_kind=
                "quantitative",

            missing_ratio=
                0.0,

            unique_count=
                50,

            unique_candidate=
                False,
        )
    )


# ============================================================
# CATALOG
# ============================================================


def product_catalog() -> PlannerCatalog:
    columns = [
        categorical_column(
            "category"
        ),

        quantitative_column(
            "quantity"
        ),

        quantitative_column(
            "unit_price"
        ),

        quantitative_column(
            "discount_pct"
        ),

        quantitative_column(
            "age"
        ),

        quantitative_column(
            "unit_cost"
        ),

        quantitative_column(
            "list_price"
        ),
    ]


    return (
        PlannerCatalog(
            datasets=[
                PlannerDatasetProfile(
                    dataset_id=
                        "dataset:products",

                    filename=
                        "products_combined.csv",

                    row_count=
                        100,

                    column_count=
                        len(
                            columns
                        ),

                    columns=
                        columns,
                )
            ]
        )
    )


# ============================================================
# PROPOSAL
# ============================================================


def group_comparison_proposal(
    *,
    value_column: str,
) -> AIPlannerProposal:
    return (
        AIPlannerProposal(
            decision=
                "propose",

            title=
                "Comparaison par catégorie",

            family=
                "group_comparison",

            dataset_id=
                "dataset:products",

            analytical_grain=
                "event",

            x_column=
                None,

            y_column=
                None,

            group_column=
                "category",

            value_column=
                value_column,

            time_column=
                None,

            dimension_column=
                None,

            entity_column=
                None,

            aggregation_function=
                "mean",

            ranking_order=
                "none",

            ranking_limit=
                None,

            window_operation=
                "none",

            window_size=
                None,

            blockers=[],

            reasons=[
                (
                    "Le planner propose une comparaison "
                    "de moyenne par catégorie."
                )
            ],

            confidence=
                0.95,
        )
    )


# ============================================================
# GUARD EXECUTION
# ============================================================


def apply_guard(
    *,
    objective: str,
    value_column: str,
) -> AIPlannerProposal:
    return (
        apply_deterministic_abstention_guards(
            objective=
                objective,

            proposal=
                group_comparison_proposal(
                    value_column=
                        value_column
                ),

            catalog=
                product_catalog(),
        )
    )


# ============================================================
# ASSERTION HELPERS
# ============================================================


def assert_proposal_remains_executable(
    *,
    objective: str,
    value_column: str,
) -> None:
    result = (
        apply_guard(
            objective=
                objective,

            value_column=
                value_column,
        )
    )


    assert (
        result.decision
        ==
        "propose"
    ), (
        f"Expected semantic metric reference to remain "
        f"executable for {value_column!r}, but got "
        f"{result.decision!r}. "
        f"Blockers: {result.blockers}"
    )


    assert (
        result.value_column
        ==
        value_column
    )


    assert (
        result.group_column
        ==
        "category"
    )


    assert (
        result.blockers
        ==
        []
    )


def assert_proposal_becomes_metric_ambiguous(
    *,
    objective: str,
    value_column: str,
) -> None:
    result = (
        apply_guard(
            objective=
                objective,

            value_column=
                value_column,
        )
    )


    assert (
        result.decision
        ==
        "ambiguous"
    ), (
        f"Expected ambiguous metric guard for "
        f"{objective!r}, but got "
        f"{result.decision!r}."
    )


    assert (
        result.value_column
        is None
    )


    assert (
        result.group_column
        is None
    )


    assert (
        len(
            result.blockers
        )
        ==
        1
    )


    assert (
        "AMBIGU"
        in
        result.blockers[
            0
        ].upper()
    )


    assert (
        "M"
        in
        result.blockers[
            0
        ].upper()
    )


# ============================================================
# 1. EXACT PRODUCTION PROMPT
# ============================================================


def test_french_unit_price_semantic_reference():
    """
    Exact user-facing failure observed in the application.

    BEFORE FIX:
        ambiguous

    AFTER FIX:
        propose
    """

    assert_proposal_remains_executable(
        objective=(
            "Compare le prix unitaire moyen "
            "entre les catégories de produits."
        ),

        value_column=
            "unit_price",
    )


# ============================================================
# 2. UNIT COST
# ============================================================


def test_french_unit_cost_semantic_reference():
    """
    'coût unitaire' must identify unit_cost, not unit_price.
    """

    assert_proposal_remains_executable(
        objective=(
            "Compare le coût unitaire moyen "
            "par category."
        ),

        value_column=
            "unit_cost",
    )


# ============================================================
# 3. LIST PRICE
# ============================================================


def test_french_list_price_semantic_reference():
    """
    'prix catalogue' is specific enough to identify list_price.
    """

    assert_proposal_remains_executable(
        objective=(
            "Compare le prix catalogue moyen "
            "par category."
        ),

        value_column=
            "list_price",
    )


# ============================================================
# 4. GENERIC PRICE REMAINS AMBIGUOUS
# ============================================================


def test_generic_price_remains_ambiguous():
    """
    'prix' alone is not sufficient.

    Both unit_price and list_price are plausible measures.

    Python must continue to fail closed.
    """

    assert_proposal_becomes_metric_ambiguous(
        objective=(
            "Compare le prix moyen "
            "par category."
        ),

        value_column=
            "unit_price",
    )


# ============================================================
# 5. GENERIC PERFORMANCE REMAINS AMBIGUOUS
# ============================================================


def test_generic_performance_remains_ambiguous():
    """
    A vague business concept must not silently become a
    quantitative column selected by the LLM.
    """

    assert_proposal_becomes_metric_ambiguous(
        objective=(
            "Compare la performance "
            "par category."
        ),

        value_column=
            "unit_price",
    )


# ============================================================
# 6. EXISTING LITERAL COLUMN MATCH IS PRESERVED
# ============================================================


def test_literal_unit_price_reference_is_preserved():
    """
    Existing literal schema-name behavior must not regress.
    """

    assert_proposal_remains_executable(
        objective=(
            "Compare la moyenne de unit_price "
            "par category."
        ),

        value_column=
            "unit_price",
    )


# ============================================================
# TEST RUNNER
# ============================================================


def run_case(
    *,
    name: str,

    function: Callable[
        [],
        None,
    ],
) -> Optional[
    str
]:
    try:
        function()


        print(
            f"[PASS] {name}"
        )


        return None


    except Exception as error:
        print(
            f"[FAIL] {name}"
        )


        print(
            (
                f"       "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )


        return (
            f"{name}: "
            f"{type(error).__name__}: "
            f"{error}"
        )


# ============================================================
# MAIN
# ============================================================


def main():
    print(
        "\n========================================"
    )


    print(
        "DataLens AI Planner Metric Semantic "
        "Resolution v0.1"
    )


    print(
        "========================================"
    )


    failures: list[
        str
    ] = []


    cases = [
        (
            "French unit_price semantic reference",
            test_french_unit_price_semantic_reference,
        ),

        (
            "French unit_cost semantic reference",
            test_french_unit_cost_semantic_reference,
        ),

        (
            "French list_price semantic reference",
            test_french_list_price_semantic_reference,
        ),

        (
            "generic price remains ambiguous",
            test_generic_price_remains_ambiguous,
        ),

        (
            "generic performance remains ambiguous",
            test_generic_performance_remains_ambiguous,
        ),

        (
            "literal unit_price reference preserved",
            test_literal_unit_price_reference_is_preserved,
        ),
    ]


    for (
        name,
        function,
    ) in cases:
        failure = (
            run_case(
                name=
                    name,

                function=
                    function,
            )
        )


        if (
            failure
            is not None
        ):
            failures.append(
                failure
            )


    print(
        "\n========================================"
    )


    if (
        failures
    ):
        print(
            (
                "EXPECTED RED STATE - "
                f"{len(failures)} semantic metric "
                "resolution guard(s) currently fail."
            )
        )


        for failure in (
            failures
        ):
            print(
                f"- {failure}"
            )


        print(
            "========================================"
        )


        raise AssertionError(
            (
                "Semantic metric references are not yet "
                "resolved deterministically."
            )
        )


    print(
        (
            "PASS - AI planner metric semantic "
            "resolution v0.1"
        )
    )


    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
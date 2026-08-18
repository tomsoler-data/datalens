from __future__ import annotations


from app.planning.ai_analytical_planner import (
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
)

from app.planning.generic_intent_resolver import (
    GENERIC_ANALYTICAL_INTENT_RULE_VERSION,
    resolve_generic_analytical_intent,
)


# ============================================================
# HELPERS
# ============================================================

def column(
    *,
    name: str,
    kind: str,
    dtype: str = "float64",
    unique_count: int = 10,
    unique_candidate: bool = False,
    missing_ratio: float = 0.0,
) -> PlannerColumnProfile:
    return (
        PlannerColumnProfile(
            name=
                name,

            dtype=
                dtype,

            analysis_kind=
                kind,

            missing_ratio=
                missing_ratio,

            unique_count=
                unique_count,

            unique_candidate=
                unique_candidate,
        )
    )


def catalog() -> PlannerCatalog:
    return (
        PlannerCatalog(
            datasets=[
                PlannerDatasetProfile(
                    dataset_id=
                        "dataset:0001",

                    filename=
                        "customers.csv",

                    row_count=
                        100,

                    column_count=
                        4,

                    columns=[
                        column(
                            name=
                                "customer_id",

                            kind=
                                "quantitative",

                            dtype=
                                "int64",

                            unique_count=
                                100,

                            unique_candidate=
                                True,
                        ),

                        column(
                            name=
                                "age",

                            kind=
                                "quantitative",

                            dtype=
                                "int64",

                            unique_count=
                                48,
                        ),

                        column(
                            name=
                                "annual_salary",

                            kind=
                                "quantitative",

                            unique_count=
                                93,
                        ),

                        column(
                            name=
                                "segment",

                            kind=
                                "categorical",

                            dtype=
                                "object",

                            unique_count=
                                4,
                        ),
                    ],
                ),

                PlannerDatasetProfile(
                    dataset_id=
                        "dataset:0002",

                    filename=
                        "sales.csv",

                    row_count=
                        250,

                    column_count=
                        4,

                    columns=[
                        column(
                            name=
                                "order_id",

                            kind=
                                "quantitative",

                            dtype=
                                "int64",

                            unique_count=
                                250,

                            unique_candidate=
                                True,
                        ),

                        column(
                            name=
                                "total_spend",

                            kind=
                                "quantitative",

                            unique_count=
                                221,
                        ),

                        column(
                            name=
                                "discount_rate",

                            kind=
                                "quantitative",

                            unique_count=
                                20,
                        ),

                        column(
                            name=
                                "region",

                            kind=
                                "categorical",

                            dtype=
                                "object",

                            unique_count=
                                5,
                        ),
                    ],
                ),
            ]
        )
    )


def categorical_only_catalog() -> PlannerCatalog:
    return (
        PlannerCatalog(
            datasets=[
                PlannerDatasetProfile(
                    dataset_id=
                        "dataset:0001",

                    filename=
                        "labels.csv",

                    row_count=
                        25,

                    column_count=
                        2,

                    columns=[
                        column(
                            name=
                                "customer_id",

                            kind=
                                "nominal",

                            dtype=
                                "object",

                            unique_count=
                                25,

                            unique_candidate=
                                True,
                        ),

                        column(
                            name=
                                "segment",

                            kind=
                                "categorical",

                            dtype=
                                "object",

                            unique_count=
                                3,
                        ),
                    ],
                ),
            ]
        )
    )


# ============================================================
# TEST
# ============================================================

def main() -> None:
    assert (
        GENERIC_ANALYTICAL_INTENT_RULE_VERSION
        ==
        "generic_analytical_intent_v0.1"
    )


    # --------------------------------------------------------
    # 1. Generic French request
    # --------------------------------------------------------

    generic = (
        resolve_generic_analytical_intent(
            objective=
                "Détecte les outliers.",

            catalog=
                catalog(),
        )
    )


    assert generic.matched is True
    assert generic.status == "matched"
    assert generic.intent == "outlier_detection"


    generic_targets = {
        (
            target.dataset_id,
            target.column,
        )

        for target
        in generic.targets
    }


    assert generic_targets == {
        (
            "dataset:0001",
            "age",
        ),
        (
            "dataset:0001",
            "annual_salary",
        ),
        (
            "dataset:0002",
            "total_spend",
        ),
        (
            "dataset:0002",
            "discount_rate",
        ),
    }


    # Numeric identifiers must not be analysed as measures.
    assert all(
        target.column
        not in {
            "customer_id",
            "order_id",
        }

        for target
        in generic.targets
    )


    # --------------------------------------------------------
    # 2. Generic natural-language variant
    # --------------------------------------------------------

    atypical = (
        resolve_generic_analytical_intent(
            objective=
                "Recherche les valeurs atypiques.",

            catalog=
                catalog(),
        )
    )


    assert atypical.matched is True
    assert atypical.intent == "outlier_detection"
    assert atypical.target_count == 4


    # --------------------------------------------------------
    # 3. English variant
    # --------------------------------------------------------

    english = (
        resolve_generic_analytical_intent(
            objective=
                "Find the outliers.",

            catalog=
                catalog(),
        )
    )


    assert english.matched is True
    assert english.intent == "outlier_detection"


    # --------------------------------------------------------
    # 4. Explicit column -> leave to existing AI planner
    # --------------------------------------------------------

    explicit_column = (
        resolve_generic_analytical_intent(
            objective=
                "Détecte les outliers de annual_salary.",

            catalog=
                catalog(),
        )
    )


    assert explicit_column.matched is False
    assert explicit_column.status == "not_matched"
    assert explicit_column.target_count == 0


    # --------------------------------------------------------
    # 5. Explicit dataset -> restrict deterministic scope
    # --------------------------------------------------------

    explicit_dataset = (
        resolve_generic_analytical_intent(
            objective=
                "Détecte les outliers dans sales.csv.",

            catalog=
                catalog(),
        )
    )


    assert explicit_dataset.matched is True
    assert explicit_dataset.status == "matched"


    explicit_dataset_targets = {
        target.column

        for target
        in explicit_dataset.targets
    }


    assert explicit_dataset_targets == {
        "total_spend",
        "discount_rate",
    }


    assert all(
        target.dataset_id ==
        "dataset:0002"

        for target
        in explicit_dataset.targets
    )


    # --------------------------------------------------------
    # 6. Unrelated objective -> no interception
    # --------------------------------------------------------

    unrelated = (
        resolve_generic_analytical_intent(
            objective=
                "Comparer annual_salary selon segment.",

            catalog=
                catalog(),
        )
    )


    assert unrelated.matched is False
    assert unrelated.status == "not_matched"


    # --------------------------------------------------------
    # 7. Generic outlier intent but no quantitative target
    # --------------------------------------------------------

    blocked = (
        resolve_generic_analytical_intent(
            objective=
                "Détecte les valeurs aberrantes.",

            catalog=
                categorical_only_catalog(),
        )
    )


    assert blocked.matched is True
    assert blocked.status == "blocked"
    assert blocked.intent == "outlier_detection"
    assert blocked.target_count == 0
    assert blocked.blockers


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print(
        "Generic analytical intent resolver v0.1 : OK"
    )

    print(
        "Generic outlier intent                  : MATCHED"
    )

    print(
        "Catalog-only target resolution          : OK"
    )

    print(
        "Numeric identifiers                     : EXCLUDED"
    )

    print(
        "Explicit column request                 : NOT INTERCEPTED"
    )

    print(
        "Explicit dataset scope                  : OK"
    )

    print(
        "No quantitative target                 : BLOCKED"
    )


if __name__ == "__main__":
    main()
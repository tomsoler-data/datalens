from __future__ import annotations


import pandas as pd


from app.planning.planner_catalog import (
    planner_catalog_from_dataset_records,
)


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    canonicalize_analytical_view_intent,
    canonicalize_inferred_dataset_reference,
)


OBJECTIVE = (
    "Quel est le chiffre d'affaires total "
    "sur toute la période analysée ?"
)


TRUSTED_MONETARY_EVENT_SEMANTICS = (
    "The unit monetary measure was propagated from a validated "
    "dimension to fact grain. No explicit quantity measure was "
    "detected, so one fact row is conservatively treated as one "
    "monetary event."
)


REVENUE_ALIASES = {
    "revenue",
    "turnover",
    "chiffre_affaires",
    "ca",
}


SOURCE_DATASET_ID = (
    "dataset_source"
)


SCALAR_DATASET_ID = (
    "derived:dataset_source:scalar:price"
)


CATEGORY_DATASET_ID = (
    "derived:dataset_source:category:segment:price"
)


MONTHLY_DATASET_ID = (
    "derived:dataset_source:monthly:event_date:price"
)


GROUPED_DATASET_IDS = {
    CATEGORY_DATASET_ID,
    MONTHLY_DATASET_ID,
}


def source_record() -> dict:

    return {
        "dataset_id":
            SOURCE_DATASET_ID,

        "filename":
            "events.csv",

        "dataframe":
            pd.DataFrame(
                {
                    "event_id": [
                        "e1",
                        "e2",
                        "e3",
                        "e4",
                    ],

                    "event_date":
                        pd.to_datetime(
                            [
                                "2024-01-01",
                                "2024-01-02",
                                "2024-02-01",
                                "2024-02-02",
                            ]
                        ),

                    "segment": [
                        "A",
                        "A",
                        "B",
                        "B",
                    ],

                    "price": [
                        10.0,
                        20.0,
                        30.0,
                        40.0,
                    ],
                }
            ),
    }


def scalar_record(
    *,
    dataset_id: str,
    source_measure: str,
    target_measure: str,
    metric_semantics: str | None,
) -> dict:

    return {
        "dataset_id":
            dataset_id,

        "filename":
            (
                "events__overall_"
                f"{source_measure}.derived"
            ),

        "dataframe":
            pd.DataFrame(
                {
                    target_measure: [
                        100.0
                    ],

                    "event_count": [
                        4
                    ],
                }
            ),

        "is_derived":
            True,

        "derivation_type":
            "scalar_additive_measure",

        "source_dataset_ids": [
            SOURCE_DATASET_ID
        ],

        "provenance": {
            "fact_dataset_id":
                SOURCE_DATASET_ID,

            "operation":
                "scalar_sum",

            "group_column":
                None,

            "source_measure_column":
                source_measure,

            "target_measure_column":
                target_measure,

            "aggregation":
                "sum",

            "grain":
                "overall",

            "metric_semantics":
                metric_semantics,

            "population_semantics":
                (
                    "Complete validated fact-grain analytical "
                    "population. No temporal, categorical, "
                    "entity or session grouping is applied."
                ),
        },
    }


def category_record() -> dict:

    return {
        "dataset_id":
            CATEGORY_DATASET_ID,

        "filename":
            "events__by_segment_price.derived",

        "dataframe":
            pd.DataFrame(
                {
                    "segment": [
                        "A",
                        "B",
                    ],

                    "sum_price": [
                        30.0,
                        70.0,
                    ],

                    "event_count": [
                        2,
                        2,
                    ],
                }
            ),

        "is_derived":
            True,

        "derivation_type":
            "categorical_additive_measure",

        "source_dataset_ids": [
            SOURCE_DATASET_ID
        ],

        "provenance": {
            "fact_dataset_id":
                SOURCE_DATASET_ID,

            "operation":
                "groupby_sum",

            "group_column":
                "segment",

            "source_measure_column":
                "price",

            "target_measure_column":
                "sum_price",

            "aggregation":
                "sum",

            "grain":
                "segment",

            "metric_semantics":
                TRUSTED_MONETARY_EVENT_SEMANTICS,
        },
    }


def monthly_record() -> dict:

    return {
        "dataset_id":
            MONTHLY_DATASET_ID,

        "filename":
            "events__monthly_price.derived",

        "dataframe":
            pd.DataFrame(
                {
                    "month":
                        pd.to_datetime(
                            [
                                "2024-01-01",
                                "2024-02-01",
                            ]
                        ),

                    "sum_price": [
                        30.0,
                        70.0,
                    ],

                    "event_count": [
                        2,
                        2,
                    ],
                }
            ),

        "is_derived":
            True,

        "derivation_type":
            "monthly_additive_measure",

        "source_dataset_ids": [
            SOURCE_DATASET_ID
        ],

        "provenance": {
            "fact_dataset_id":
                SOURCE_DATASET_ID,

            "operation":
                "groupby_sum",

            "source_time_column":
                "event_date",

            "target_time_column":
                "month",

            "source_measure_column":
                "price",

            "target_measure_column":
                "sum_price",

            "aggregation":
                "sum",

            "grain":
                "month",

            "metric_semantics":
                TRUSTED_MONETARY_EVENT_SEMANTICS,
        },
    }


def profile_by_id(
    catalog,
    dataset_id: str,
):

    matches = [
        dataset

        for dataset
        in catalog.datasets

        if (
            dataset.dataset_id
            ==
            dataset_id
        )
    ]


    assert (
        len(
            matches
        )
        ==
        1
    )


    return matches[
        0
    ]


def aliases(
    profile,
) -> set[str]:

    return {
        str(
            alias
        ).strip()

        for alias
        in (
            profile
            .measure_semantic_aliases
            or []
        )

        if str(
            alias
        ).strip()
    }


def captured_total_proposal() -> AIPlannerProposal:

    return (
        AIPlannerProposal(
            decision=
                "propose",

            title=
                "Total revenue",

            family=
                "aggregation",

            dataset_id=
                SOURCE_DATASET_ID,

            analytical_grain=
                None,

            x_column=
                None,

            y_column=
                None,

            group_column=
                None,

            value_column=
                "sum_price",

            time_column=
                None,

            dimension_column=
                None,

            entity_column=
                None,

            aggregation_function=
                "sum",

            ranking_order=
                "none",

            ranking_limit=
                None,

            window_operation=
                "none",

            window_size=
                None,

            benchmark_reference=
                None,

            benchmark_operator=
                None,

            benchmark_selection=
                None,

            blockers=
                [],

            reasons=[
                (
                    "Deterministic reproduction of a total "
                    "additive aggregation proposal."
                )
            ],

            confidence=
                0.90,
        )
    )


def main() -> None:

    print(
        "=== DATALENS SCALAR REVENUE "
        "PLANNER BRIDGE v0.1 ==="
    )

    print()


    trusted_scalar = (
        scalar_record(
            dataset_id=
                SCALAR_DATASET_ID,

            source_measure=
                "price",

            target_measure=
                "sum_price",

            metric_semantics=
                TRUSTED_MONETARY_EVENT_SEMANTICS,
        )
    )


    cost_scalar = (
        scalar_record(
            dataset_id=
                "derived:dataset_source:scalar:cost",

            source_measure=
                "cost",

            target_measure=
                "sum_cost",

            metric_semantics=
                TRUSTED_MONETARY_EVENT_SEMANTICS,
        )
    )


    untrusted_price_scalar = (
        scalar_record(
            dataset_id=
                "derived:dataset_source:scalar:list_price",

            source_measure=
                "list_price",

            target_measure=
                "sum_list_price",

            metric_semantics=
                None,
        )
    )


    catalog = (
        planner_catalog_from_dataset_records(
            [
                source_record(),
                trusted_scalar,
                category_record(),
                monthly_record(),
                cost_scalar,
                untrusted_price_scalar,
            ]
        )
    )


    scalar_profile = (
        profile_by_id(
            catalog,
            SCALAR_DATASET_ID,
        )
    )


    category_profile = (
        profile_by_id(
            catalog,
            CATEGORY_DATASET_ID,
        )
    )


    monthly_profile = (
        profile_by_id(
            catalog,
            MONTHLY_DATASET_ID,
        )
    )


    cost_profile = (
        profile_by_id(
            catalog,
            "derived:dataset_source:scalar:cost",
        )
    )


    untrusted_profile = (
        profile_by_id(
            catalog,
            "derived:dataset_source:scalar:list_price",
        )
    )


    scalar_aliases = (
        aliases(
            scalar_profile
        )
    )


    category_aliases = (
        aliases(
            category_profile
        )
    )


    monthly_aliases = (
        aliases(
            monthly_profile
        )
    )


    cost_aliases = (
        aliases(
            cost_profile
        )
    )


    untrusted_aliases = (
        aliases(
            untrusted_profile
        )
    )


    # ========================================================
    # EXISTING R2 AUTHORITY MUST REMAIN TRUE
    # ========================================================

    assert (
        REVENUE_ALIASES
        <=
        category_aliases
    ), (
        "Precondition failed: existing trusted grouped "
        "revenue aliases regressed."
    )


    assert (
        REVENUE_ALIASES
        <=
        monthly_aliases
    ), (
        "Precondition failed: existing monthly revenue "
        "aliases regressed."
    )


    # ========================================================
    # NEGATIVE GUARDS
    # ========================================================

    assert not (
        cost_aliases
        &
        REVENUE_ALIASES
    ), (
        "A scalar cost measure must not be promoted "
        "to revenue semantics."
    )


    assert not (
        untrusted_aliases
        &
        REVENUE_ALIASES
    ), (
        "An untrusted scalar price-like measure must "
        "not be promoted to revenue semantics."
    )


    # ========================================================
    # CAPTURED TOTAL WIRE
    # ========================================================

    proposal = (
        captured_total_proposal()
    )


    (
        inferred,
        inferred_notes,
    ) = (
        canonicalize_inferred_dataset_reference(
            objective=
                OBJECTIVE,

            proposal=
                proposal,

            catalog=
                catalog,
        )
    )


    (
        canonical,
        canonical_notes,
    ) = (
        canonicalize_analytical_view_intent(
            objective=
                OBJECTIVE,

            proposal=
                inferred,

            catalog=
                catalog,
        )
    )


    # A total request must never be redirected to a grouped
    # authority merely because it exposes the same physical
    # target column.
    assert (
        canonical.dataset_id
        not in
        GROUPED_DATASET_IDS
    ), (
        "A scalar total request must not be redirected "
        "to a grouped analytical view."
    )


    failures: list[
        str
    ] = []


    if not (
        REVENUE_ALIASES
        <=
        scalar_aliases
    ):
        failures.append(
            "scalar_alias_gap"
        )


    if not (
        canonical.dataset_id
        ==
        SCALAR_DATASET_ID

        and

        canonical.value_column
        ==
        "sum_price"
    ):
        failures.append(
            "scalar_canonicalization_gap"
        )


    print(
        "Scalar aliases                          "
        f"{sorted(scalar_aliases)}"
    )


    print(
        "Category aliases                        "
        f"{sorted(category_aliases)}"
    )


    print(
        "Monthly aliases                         "
        f"{sorted(monthly_aliases)}"
    )


    print(
        "Cost revenue overlap                    "
        f"{sorted(cost_aliases & REVENUE_ALIASES)}"
    )


    print(
        "Untrusted revenue overlap               "
        f"{sorted(untrusted_aliases & REVENUE_ALIASES)}"
    )


    print(
        "Original dataset                        "
        f"{proposal.dataset_id}"
    )


    print(
        "After inferred repair                   "
        f"{inferred.dataset_id}"
    )


    print(
        "Inferred notes                          "
        f"{inferred_notes}"
    )


    print(
        "After analytical canonicalizer          "
        f"{canonical.dataset_id}"
    )


    print(
        "Canonical value                         "
        f"{canonical.value_column}"
    )


    print(
        "Canonical notes                         "
        f"{canonical_notes}"
    )


    print(
        "Grouped total misrouting                "
        f"{canonical.dataset_id in GROUPED_DATASET_IDS}"
    )


    print(
        "Observed gaps                           "
        f"{failures}"
    )


    if failures:

        raise AssertionError(
            (
                "RED_EXPECTED: "
                +
                ",".join(
                    failures
                )
            )
        )


    print()

    print(
        "PASS - scalar revenue planner bridge v0.1"
    )


if __name__ == "__main__":

    main()
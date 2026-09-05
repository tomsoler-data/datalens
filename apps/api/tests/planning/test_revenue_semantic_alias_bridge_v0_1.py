from __future__ import annotations


import pandas as pd


from app.planning.planner_catalog import (
    planner_catalog_from_dataset_records,
)


from app.planning.objective_coverage import (
    extract_objective_requirements,
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


def grouped_record(
    *,
    source_measure: str,
    target_measure: str,
    metric_semantics: str | None,
) -> dict:

    dataframe = pd.DataFrame(
        {
            "segment": [
                "A",
                "B",
            ],

            target_measure: [
                100.0,
                200.0,
            ],

            "event_count": [
                1,
                1,
            ],
        }
    )


    return {
        "dataset_id":
            (
                "derived:synthetic:"
                f"{source_measure}"
            ),

        "filename":
            (
                "events__by_segment_"
                f"{source_measure}.derived"
            ),

        "dataframe":
            dataframe,

        "is_derived":
            True,

        "derivation_type":
            "categorical_additive_measure",

        "provenance": {
            "operation":
                "groupby_sum",

            "aggregation":
                "sum",

            "grain":
                "segment",

            "group_column":
                "segment",

            "source_measure_column":
                source_measure,

            "target_measure_column":
                target_measure,

            "metric_semantics":
                metric_semantics,
        },
    }


def session_record() -> dict:

    dataframe = pd.DataFrame(
        {
            "session_id": [
                "S1",
                "S2",
            ],

            "basket_amount": [
                100.0,
                200.0,
            ],

            "item_count": [
                1,
                1,
            ],
        }
    )


    return {
        "dataset_id":
            "derived:synthetic:session:price",

        "filename":
            "events__sessions_price.derived",

        "dataframe":
            dataframe,

        "is_derived":
            True,

        "derivation_type":
            "entity_additive_measure",

        "provenance": {
            "operation":
                "session_materialization",

            "aggregation":
                "sum",

            "grain":
                "session_id",

            "entity_column":
                "session_id",

            "source_measure_column":
                "price",

            "target_measure_column":
                "basket_amount",

            "metric_semantics":
                TRUSTED_MONETARY_EVENT_SEMANTICS,
        },
    }


def dataset_aliases(
    record: dict,
) -> set[str]:

    catalog = (
        planner_catalog_from_dataset_records(
            [
                record
            ]
        )
    )


    dataset = (
        catalog.datasets[
            0
        ]
    )


    return {
        str(
            alias
        ).strip()

        for alias
        in dataset.measure_semantic_aliases

        if str(
            alias
        ).strip()
    }


def revenue_candidate_columns(
    record: dict,
) -> list[str]:

    catalog = (
        planner_catalog_from_dataset_records(
            [
                record
            ]
        )
    )


    requirements = (
        extract_objective_requirements(
            objective=
                OBJECTIVE,

            catalog=
                catalog,
        )
    )


    revenue_requirements = [
        requirement

        for requirement
        in requirements

        if (
            requirement.concept
            ==
            "revenue_total"
        )
    ]


    assert (
        len(
            revenue_requirements
        )
        ==
        1
    )


    return list(
        revenue_requirements[
            0
        ].candidate_columns
    )


def main() -> None:

    print(
        "=== DATALENS REVENUE SEMANTIC "
        "ALIAS BRIDGE v0.1 ==="
    )


    # ========================================================
    # POSITIVE AUTHORITY
    # ========================================================

    trusted_price = (
        grouped_record(
            source_measure=
                "price",

            target_measure=
                "sum_price",

            metric_semantics=
                TRUSTED_MONETARY_EVENT_SEMANTICS,
        )
    )


    trusted_aliases = (
        dataset_aliases(
            trusted_price
        )
    )


    trusted_candidates = (
        revenue_candidate_columns(
            trusted_price
        )
    )


    # ========================================================
    # NEGATIVE GUARD 1
    #
    # COST IS NOT REVENUE.
    # ========================================================

    trusted_cost = (
        grouped_record(
            source_measure=
                "cost",

            target_measure=
                "sum_cost",

            metric_semantics=
                TRUSTED_MONETARY_EVENT_SEMANTICS,
        )
    )


    cost_aliases = (
        dataset_aliases(
            trusted_cost
        )
    )


    assert not (
        cost_aliases
        &
        REVENUE_ALIASES
    ), (
        "A cost measure must never be promoted "
        "to revenue semantics."
    )


    # ========================================================
    # NEGATIVE GUARD 2
    #
    # BARE PRICE WITHOUT TRUSTED SERVER SEMANTICS REMAINS
    # AMBIGUOUS.
    # ========================================================

    untrusted_price = (
        grouped_record(
            source_measure=
                "price",

            target_measure=
                "sum_price",

            metric_semantics=
                None,
        )
    )


    untrusted_aliases = (
        dataset_aliases(
            untrusted_price
        )
    )


    assert not (
        untrusted_aliases
        &
        REVENUE_ALIASES
    ), (
        "An untrusted bare price measure must not "
        "be promoted to revenue semantics."
    )


    # ========================================================
    # NEGATIVE GUARD 3
    #
    # SESSION BASKET MEASURE MUST NOT BECOME A GLOBAL REVENUE
    # ALIAS.
    # ========================================================

    session_aliases = (
        dataset_aliases(
            session_record()
        )
    )


    assert not (
        session_aliases
        &
        REVENUE_ALIASES
    ), (
        "Session basket measures must remain "
        "outside global revenue aliases."
    )


    # ========================================================
    # EXPECTED RED GAPS
    # ========================================================

    failures: list[
        str
    ] = []


    if not (
        REVENUE_ALIASES
        <=
        trusted_aliases
    ):
        failures.append(
            "planner_alias_gap"
        )


    if (
        trusted_candidates
        !=
        [
            "sum_price"
        ]
    ):
        failures.append(
            "objective_coverage_alias_gap"
        )


    print(
        (
            "Trusted price aliases                  "
            f"{sorted(trusted_aliases)}"
        )
    )


    print(
        (
            "Revenue candidate columns             "
            f"{trusted_candidates}"
        )
    )


    print(
        (
            "Cost revenue overlap                  "
            f"{sorted(cost_aliases & REVENUE_ALIASES)}"
        )
    )


    print(
        (
            "Untrusted price revenue overlap       "
            f"{sorted(untrusted_aliases & REVENUE_ALIASES)}"
        )
    )


    print(
        (
            "Session revenue overlap               "
            f"{sorted(session_aliases & REVENUE_ALIASES)}"
        )
    )


    print(
        (
            "Observed gaps                         "
            f"{failures}"
        )
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
        "PASS - revenue semantic alias bridge v0.1"
    )


if __name__ == "__main__":

    main()
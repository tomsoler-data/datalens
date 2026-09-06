from __future__ import annotations


import pandas as pd


from app.planning.ai_analytical_planner import (
    PlannerDatasetProfile,
)


from app.planning.planner_catalog import (
    PLANNER_CATALOG_RULE_VERSION,
    planner_catalog_from_dataset_records,
)


# ============================================================
# GENERIC DATASET AUTHORITIES
# ============================================================


FACT_DATASET_ID = (
    "dataset:orders"
)


DIMENSION_DATASET_ID = (
    "dataset:customers"
)


SESSION_DATASET_ID = (
    "derived:orders:session:session_id:amount"
)


SESSION_FILENAME = (
    "orders__sessions_amount.derived"
)


# ============================================================
# RECORD BUILDERS
# ============================================================


def source_records():
    return [
        {
            "dataset_id":
                FACT_DATASET_ID,

            "filename":
                "orders.csv",

            "dataframe":
                pd.DataFrame(
                    {
                        "session_id":
                            [
                                "s1",
                                "s1",
                                "s2",
                            ],

                        "customer_id":
                            [
                                "c1",
                                "c1",
                                "c2",
                            ],
                    }
                ),

            "is_derived":
                False,
        },

        {
            "dataset_id":
                DIMENSION_DATASET_ID,

            "filename":
                "customers.csv",

            "dataframe":
                pd.DataFrame(
                    {
                        "customer_id":
                            [
                                "c1",
                                "c2",
                            ],

                        "segment":
                            [
                                "a",
                                "b",
                            ],
                    }
                ),

            "is_derived":
                False,
        },

        {
            "dataset_id":
                SESSION_DATASET_ID,

            "filename":
                SESSION_FILENAME,

            "dataframe":
                pd.DataFrame(
                    {
                        "session_id":
                            [
                                "s1",
                                "s2",
                            ],

                        "basket_amount":
                            [
                                30.0,
                                20.0,
                            ],
                    }
                ),

            "is_derived":
                True,

            "derivation_type":
                "entity_additive_measure",

            # Server-owned lineage record.
            "source_dataset_ids":
                [
                    FACT_DATASET_ID,
                    DIMENSION_DATASET_ID,
                ],

            "provenance":
                {
                    "fact_dataset_id":
                        FACT_DATASET_ID,

                    "operation":
                        "session_materialization",

                    "entity_column":
                        "session_id",

                    "source_measure_column":
                        "amount",

                    "target_measure_column":
                        "basket_amount",

                    "aggregation":
                        "sum",

                    "grain":
                        "session_id",
                },
        },
    ]


def catalog():
    return (
        planner_catalog_from_dataset_records(
            source_records()
        )
    )


def profile_by_id(
    *,
    planner_catalog,
    dataset_id: str,
) -> PlannerDatasetProfile:

    matches = [
        profile

        for profile
        in planner_catalog.datasets

        if (
            profile.dataset_id
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


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS PLANNER CATALOG DERIVED LINEAGE v0.1 ==="
    )

    print()


    planner_catalog = (
        catalog()
    )


    fact_profile = (
        profile_by_id(
            planner_catalog=
                planner_catalog,

            dataset_id=
                FACT_DATASET_ID,
        )
    )


    dimension_profile = (
        profile_by_id(
            planner_catalog=
                planner_catalog,

            dataset_id=
                DIMENSION_DATASET_ID,
        )
    )


    session_profile = (
        profile_by_id(
            planner_catalog=
                planner_catalog,

            dataset_id=
                SESSION_DATASET_ID,
        )
    )


    # ========================================================
    # 1. EXISTING SERVER-OWNED ANALYTICAL METADATA
    # ========================================================

    assert (
        session_profile.is_derived
        is True
    )


    assert (
        session_profile.operation
        ==
        "session_materialization"
    )


    assert (
        session_profile.analytical_grain
        ==
        "session_id"
    )


    assert (
        session_profile.entity_column
        ==
        "session_id"
    )


    assert (
        session_profile.target_measure_column
        ==
        "basket_amount"
    )


    print(
        "[PASS] existing session analytical metadata preserved"
    )


    # ========================================================
    # 2. DESIRED INTERNAL LINEAGE AUTHORITY
    #
    # Use getattr so today's missing fields produce a controlled
    # RED instead of an AttributeError.
    # ========================================================

    observed_fact_dataset_id = (
        getattr(
            session_profile,
            "fact_dataset_id",
            None,
        )
    )


    observed_source_dataset_ids = (
        getattr(
            session_profile,
            "source_dataset_ids",
            None,
        )
    )


    fact_lineage_present = (
        observed_fact_dataset_id
        ==
        FACT_DATASET_ID
    )


    source_lineage_present = (
        observed_source_dataset_ids
        ==
        [
            FACT_DATASET_ID,
            DIMENSION_DATASET_ID,
        ]
    )


    fact_in_source_lineage = (
        isinstance(
            observed_source_dataset_ids,
            list,
        )
        and
        FACT_DATASET_ID
        in
        observed_source_dataset_ids
    )


    # ========================================================
    # 3. SOURCE DATASETS MUST NOT INVENT LINEAGE
    # ========================================================

    source_profiles_fail_closed = (
        getattr(
            fact_profile,
            "fact_dataset_id",
            None,
        )
        is None

        and

        getattr(
            dimension_profile,
            "fact_dataset_id",
            None,
        )
        is None

        and

        getattr(
            fact_profile,
            "source_dataset_ids",
            []
        )
        ==
        []

        and

        getattr(
            dimension_profile,
            "source_dataset_ids",
            []
        )
        ==
        []
    )


    assert (
        source_profiles_fail_closed
    ), (
        "Source datasets must not invent derived-lineage metadata."
    )


    print(
        "[PASS] source datasets do not invent derived lineage"
    )


    # ========================================================
    # 4. LINEAGE MUST REMAIN INTERNAL / SERVER-OWNED
    #
    # Adding these fields must not expand the compact model
    # payload. They are validator authority, not LLM choices.
    # ========================================================

    dumped = (
        session_profile.model_dump(
            mode="python"
        )
    )


    lineage_hidden_from_model_dump = (
        "fact_dataset_id"
        not in
        dumped

        and

        "source_dataset_ids"
        not in
        dumped
    )


    assert (
        lineage_hidden_from_model_dump
    ), (
        "Derived lineage must remain excluded from the normal "
        "PlannerDatasetProfile model dump / LLM-visible wire."
    )


    print(
        "[PASS] lineage remains excluded from model-visible dump"
    )


    # ========================================================
    # 5. SERVER IDENTITY MUST REMAIN EXACT
    # ========================================================

    dataset_identity_preserved = (
        session_profile.dataset_id
        ==
        SESSION_DATASET_ID
        and
        session_profile.filename
        ==
        SESSION_FILENAME
    )


    assert (
        dataset_identity_preserved
    )


    print(
        "[PASS] canonical derived dataset identity preserved"
    )


    # ========================================================
    # 6. OBSERVATION
    # ========================================================

    print()
    print(
        "Planner Catalog rule                    "
        f"{PLANNER_CATALOG_RULE_VERSION}"
    )

    print(
        "Session dataset                         "
        f"{session_profile.dataset_id}"
    )

    print(
        "Session operation                       "
        f"{session_profile.operation}"
    )

    print(
        "Observed fact_dataset_id                "
        f"{observed_fact_dataset_id}"
    )

    print(
        "Expected fact_dataset_id                "
        f"{FACT_DATASET_ID}"
    )

    print(
        "Observed source_dataset_ids             "
        f"{observed_source_dataset_ids}"
    )

    print(
        "Expected source_dataset_ids             "
        f"{[FACT_DATASET_ID, DIMENSION_DATASET_ID]}"
    )

    print(
        "Fact lineage present                    "
        f"{fact_lineage_present}"
    )

    print(
        "Source lineage present                  "
        f"{source_lineage_present}"
    )

    print(
        "Fact included in source lineage         "
        f"{fact_in_source_lineage}"
    )

    print(
        "Lineage model-visible                   "
        f"{not lineage_hidden_from_model_dump}"
    )


    # ========================================================
    # 7. RED GAPS
    # ========================================================

    gaps: list[
        str
    ] = []


    if not fact_lineage_present:

        gaps.append(
            "planner_catalog_fact_dataset_lineage_gap"
        )


    if not source_lineage_present:

        gaps.append(
            "planner_catalog_source_dataset_ids_lineage_gap"
        )


    if not fact_in_source_lineage:

        gaps.append(
            "planner_catalog_fact_source_consistency_gap"
        )


    print()
    print(
        "Observed gaps                           "
        f"{gaps}"
    )


    if gaps:

        raise AssertionError(
            (
                "RED_EXPECTED: "
                +
                ",".join(
                    gaps
                )
            )
        )


    print()
    print(
        "PASS - Planner Catalog derived lineage v0.1"
    )


if __name__ == "__main__":

    main()
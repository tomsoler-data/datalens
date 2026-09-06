from __future__ import annotations


import math

import pandas as pd


from app.analysis.analytical_views import (
    materialize_views_for_fact,
)


# ============================================================
# FIXTURES
# ============================================================


def trusted_event_dataframe() -> pd.DataFrame:
    """
    Generic transaction-event fixture.

    `price` represents a unit monetary measure propagated from a
    validated dimension to fact grain.

    There is deliberately NO quantity measure.

    Therefore the existing Analytical View Builder already accepts
    one event row as one additive monetary event.

    The future scalar authority must use this exact fact population.
    """

    return pd.DataFrame(
        {
            "date":
                pd.to_datetime(
                    [
                        "2024-01-05",
                        "2024-01-05",
                        "2024-02-10",
                        None,
                        "2024-03-12",
                    ]
                ),

            "session_id": [
                "s1",
                "s1",
                "s2",
                "s3",
                "s4",
            ],

            "client_id": [
                "c1",
                "c1",
                "c2",
                "c3",
                "c4",
            ],

            "id_prod": [
                "p1",
                "p2",
                "p3",
                "p4",
                "p5",
            ],

            "categ": [
                "A",
                "A",
                "B",
                None,
                "B",
            ],

            "price": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ],
        }
    )


def arbitrary_numeric_dataframe() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "event_id": [
                "e1",
                "e2",
                "e3",
            ],

            "segment": [
                "A",
                "A",
                "B",
            ],

            "temperature": [
                10.0,
                20.0,
                30.0,
            ],
        }
    )


def quantity_price_dataframe() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "event_id": [
                "e1",
                "e2",
                "e3",
            ],

            "quantity": [
                2.0,
                3.0,
                4.0,
            ],

            "price": [
                10.0,
                20.0,
                30.0,
            ],

            "segment": [
                "A",
                "A",
                "B",
            ],
        }
    )


# ============================================================
# HELPERS
# ============================================================


def provenance(
    dataset: dict,
) -> dict:

    value = (
        dataset.get(
            "provenance"
        )
    )


    return (
        value
        if isinstance(
            value,
            dict,
        )
        else {}
    )


def source_measure(
    dataset: dict,
) -> str:

    return str(
        provenance(
            dataset
        ).get(
            "source_measure_column"
        )
        or
        ""
    ).strip()


def target_measure(
    dataset: dict,
) -> str:

    return str(
        provenance(
            dataset
        ).get(
            "target_measure_column"
        )
        or
        ""
    ).strip()


def scalar_views(
    datasets: list[
        dict
    ],
) -> list[
    dict
]:

    return [
        dataset

        for dataset
        in datasets

        if (
            str(
                dataset.get(
                    "derivation_type"
                )
                or
                ""
            ).strip()
            ==
            "scalar_additive_measure"
        )
    ]


def scalar_views_for_source(
    datasets: list[
        dict
    ],
    source_column: str,
) -> list[
    dict
]:

    return [
        dataset

        for dataset
        in scalar_views(
            datasets
        )

        if (
            source_measure(
                dataset
            )
            ==
            source_column
        )
    ]


# ============================================================
# TRUSTED PRICE MATERIALIZATION
# ============================================================


def trusted_price_materialization():

    dataframe = (
        trusted_event_dataframe()
    )


    original = (
        dataframe.copy(
            deep=True
        )
    )


    (
        derived,
        audits,
    ) = (
        materialize_views_for_fact(
            fact_dataset_id=
                "dataset:events",

            fact_filename=
                "events.csv",

            enriched=
                dataframe,

            source_dataset_ids=[
                "dataset:events",
                "dataset:product_dimension",
            ],

            fact_original_columns={
                "date",
                "session_id",
                "client_id",
                "id_prod",
            },

            propagated_columns={
                "categ",
                "price",
            },

            include_requested_context=
                False,
        )
    )


    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )


    return (
        dataframe,
        derived,
        audits,
    )


# ============================================================
# NEGATIVE GUARD:
# ARBITRARY QUANTITATIVE PROPAGATION IS NOT ADDITIVE MONEY
# ============================================================


def assert_arbitrary_numeric_not_scalar() -> None:

    dataframe = (
        arbitrary_numeric_dataframe()
    )


    (
        derived,
        _,
    ) = (
        materialize_views_for_fact(
            fact_dataset_id=
                "dataset:sensors",

            fact_filename=
                "sensors.csv",

            enriched=
                dataframe,

            source_dataset_ids=[
                "dataset:sensors",
                "dataset:dimension",
            ],

            fact_original_columns={
                "event_id",
            },

            propagated_columns={
                "segment",
                "temperature",
            },

            include_requested_context=
                False,
        )
    )


    bad = (
        scalar_views_for_source(
            derived,
            "temperature",
        )
    )


    assert (
        bad
        ==
        []
    ), (
        "An arbitrary propagated quantitative measure must not "
        "be promoted to a scalar additive monetary authority."
    )


# ============================================================
# NEGATIVE GUARD:
# EXPLICIT QUANTITY MAKES SUM(PRICE) INVALID AS REVENUE
# ============================================================


def assert_quantity_blocks_direct_price_scalar() -> None:

    dataframe = (
        quantity_price_dataframe()
    )


    (
        derived,
        _,
    ) = (
        materialize_views_for_fact(
            fact_dataset_id=
                "dataset:line_items",

            fact_filename=
                "line_items.csv",

            enriched=
                dataframe,

            source_dataset_ids=[
                "dataset:line_items",
                "dataset:product_dimension",
            ],

            fact_original_columns={
                "event_id",
                "quantity",
            },

            propagated_columns={
                "price",
                "segment",
            },

            include_requested_context=
                False,
        )
    )


    bad = (
        scalar_views_for_source(
            derived,
            "price",
        )
    )


    assert (
        bad
        ==
        []
    ), (
        "When explicit quantity exists, SUM(price) must not "
        "become a scalar additive monetary authority."
    )


# ============================================================
# POSITIVE SCALAR CONTRACT
# ============================================================


def assert_scalar_price_authority() -> None:

    (
        dataframe,
        derived,
        audits,
    ) = (
        trusted_price_materialization()
    )


    del audits


    # Existing authority must already recognize price as an
    # additive measure and materialize non-scalar views.
    existing_price_views = [
        dataset

        for dataset
        in derived

        if (
            source_measure(
                dataset
            )
            ==
            "price"
        )
    ]


    assert existing_price_views, (
        "Precondition failed: the current analytical materializer "
        "must already recognize trusted propagated price as an "
        "additive monetary measure."
    )


    scalar_price = (
        scalar_views_for_source(
            derived,
            "price",
        )
    )


    print(
        "Existing price analytical views          "
        f"{len(existing_price_views)}"
    )

    print(
        "Scalar price authorities                 "
        f"{len(scalar_price)}"
    )


    if not scalar_price:

        raise AssertionError(
            "RED_EXPECTED: scalar_additive_view_missing"
        )


    assert (
        len(
            scalar_price
        )
        ==
        1
    ), (
        "Exactly one scalar additive authority is allowed "
        "per fact dataset and source measure."
    )


    scalar = (
        scalar_price[
            0
        ]
    )


    scalar_dataframe = (
        scalar[
            "dataframe"
        ]
    )


    scalar_provenance = (
        provenance(
            scalar
        )
    )


    # --------------------------------------------------------
    # DATASET CONTRACT
    # --------------------------------------------------------

    assert (
        scalar.get(
            "is_derived"
        )
        is True
    )


    assert (
        scalar.get(
            "derivation_type"
        )
        ==
        "scalar_additive_measure"
    )


    # --------------------------------------------------------
    # PROVENANCE CONTRACT
    # --------------------------------------------------------

    assert (
        scalar_provenance.get(
            "operation"
        )
        ==
        "scalar_sum"
    )


    assert (
        scalar_provenance.get(
            "aggregation"
        )
        ==
        "sum"
    )


    assert (
        scalar_provenance.get(
            "grain"
        )
        ==
        "overall"
    )


    assert (
        scalar_provenance.get(
            "source_measure_column"
        )
        ==
        "price"
    )


    assert (
        scalar_provenance.get(
            "target_measure_column"
        )
        ==
        "sum_price"
    )


    assert (
        scalar_provenance.get(
            "group_column"
        )
        in {
            None,
            "",
        }
    )


    # --------------------------------------------------------
    # PHYSICAL RESULT CONTRACT
    # --------------------------------------------------------

    assert (
        len(
            scalar_dataframe
        )
        ==
        1
    ), (
        "Scalar additive view must contain exactly one row."
    )


    assert (
        "sum_price"
        in
        scalar_dataframe.columns
    )


    expected_total = float(
        dataframe[
            "price"
        ].sum()
    )


    actual_total = float(
        scalar_dataframe.iloc[
            0
        ][
            "sum_price"
        ]
    )


    assert math.isclose(
        actual_total,
        expected_total,
        rel_tol=0.0,
        abs_tol=1e-12,
    ), (
        "Scalar SUM must use the complete validated fact "
        "population without substituting a grouped view."
    )


    assert (
        math.isclose(
            actual_total,
            150.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


    print(
        "Scalar derivation type                  PASS"
    )

    print(
        "Scalar operation                        scalar_sum"
    )

    print(
        "Scalar grain                            overall"
    )

    print(
        "Scalar source measure                   price"
    )

    print(
        "Scalar target measure                   sum_price"
    )

    print(
        "Scalar row count                        1"
    )

    print(
        "Complete fact-population SUM            PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS SCALAR ADDITIVE "
        "ANALYTICAL VIEW v0.1 ==="
    )

    print()


    assert_arbitrary_numeric_not_scalar()

    print(
        "[PASS] arbitrary quantitative metric "
        "remains non-additive"
    )


    assert_quantity_blocks_direct_price_scalar()

    print(
        "[PASS] explicit quantity blocks direct "
        "SUM(price) scalar promotion"
    )


    assert_scalar_price_authority()


    print()

    print(
        "PASS - scalar additive analytical view v0.1"
    )


if __name__ == "__main__":

    main()
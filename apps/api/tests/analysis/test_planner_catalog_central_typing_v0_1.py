from __future__ import annotations


import pandas as pd


from app.planning.planner_catalog import (
    PLANNER_CATALOG_RULE_VERSION,
    planner_catalog_from_dataset_records,
)


# ============================================================
# VERSION
# ============================================================


TEST_RULE_VERSION = (
    "planner_catalog_central_typing_test_v0.1"
)


# ============================================================
# FIXTURE
# ============================================================


def build_dataframe(
) -> pd.DataFrame:
    row_count = 300


    # --------------------------------------------------------
    # Birth years
    #
    # Numeric storage, temporal analytical semantics.
    # --------------------------------------------------------

    birth = [
        1950
        +
        (
            index
            %
            51
        )

        for index
        in range(
            row_count
        )
    ]


    # --------------------------------------------------------
    # Price
    #
    # All values are deliberately unique.
    #
    # This verifies that:
    #
    #     unique physical values
    #
    # do NOT imply:
    #
    #     identifier semantics.
    # --------------------------------------------------------

    price = [
        round(
            5.17
            +
            (
                index
                *
                1.379
            ),
            3,
        )

        for index
        in range(
            row_count
        )
    ]


    # --------------------------------------------------------
    # Category code
    #
    # Numeric physical storage, categorical semantics.
    # --------------------------------------------------------

    categ = [
        index
        %
        3

        for index
        in range(
            row_count
        )
    ]


    # --------------------------------------------------------
    # Identifier
    # --------------------------------------------------------

    client_id = [
        f"c_{index:04d}"

        for index
        in range(
            row_count
        )
    ]


    return (
        pd.DataFrame(
            {
                "birth":
                    birth,

                "price":
                    price,

                "categ":
                    categ,

                "client_id":
                    client_id,
            }
        )
    )


# ============================================================
# HELPERS
# ============================================================


def column_map(
    catalog,
) -> dict:
    dataset = (
        catalog.datasets[
            0
        ]
    )


    return {
        column.name:
            column

        for column
        in dataset.columns
    }


# ============================================================
# TEST
# ============================================================


def main(
) -> None:
    dataframe = (
        build_dataframe()
    )


    dataset_records = [
        {
            "dataset_id":
                "dataset:0001",

            "filename":
                "products.csv",

            "extension":
                ".csv",

            "dataframe":
                dataframe,
        }
    ]


    catalog = (
        planner_catalog_from_dataset_records(
            dataset_records
        )
    )


    assert (
        len(
            catalog.datasets
        )
        ==
        1
    )


    dataset = (
        catalog.datasets[
            0
        ]
    )


    assert (
        dataset.dataset_id
        ==
        "dataset:0001"
    )


    assert (
        dataset.filename
        ==
        "products.csv"
    )


    assert (
        dataset.row_count
        ==
        300
    )


    assert (
        dataset.column_count
        ==
        4
    )


    columns = (
        column_map(
            catalog
        )
    )


    # ========================================================
    # CENTRAL ANALYTICAL TYPES
    # ========================================================

    assert (
        columns[
            "birth"
        ].analysis_kind
        ==
        "temporal"
    ), (
        "birth must be temporal in the planner catalog."
    )


    assert (
        columns[
            "price"
        ].analysis_kind
        ==
        "quantitative"
    ), (
        "price must remain quantitative in the planner catalog."
    )


    assert (
        columns[
            "categ"
        ].analysis_kind
        ==
        "categorical"
    ), (
        "categ must be categorical in the planner catalog."
    )


    assert (
        columns[
            "client_id"
        ].analysis_kind
        ==
        "identifier"
    ), (
        "client_id must be protected as an identifier."
    )


    # ========================================================
    # UNIQUE-VALUE GUARD
    # ========================================================

    assert (
        columns[
            "price"
        ].unique_count
        ==
        dataframe.shape[
            0
        ]
    ), (
        "The test fixture must keep every price unique."
    )


    assert (
        columns[
            "price"
        ].unique_candidate
        is False
    ), (
        "A unique continuous measure must not be protected "
        "as an identifier merely because every value differs."
    )


    assert (
        columns[
            "client_id"
        ].unique_candidate
        is True
    ), (
        "A unique identifier should remain marked as an "
        "identifier candidate."
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print(
        "=============================================="
    )

    print(
        "DataLens Planner Catalog Central Typing v0.1"
    )

    print(
        "=============================================="
    )


    print()
    print(
        f"Catalog engine : {PLANNER_CATALOG_RULE_VERSION}"
    )


    print()
    print(
        "=== ANALYTICAL TYPES ==="
    )


    print(
        "birth     ->",
        columns[
            "birth"
        ].analysis_kind,
    )


    print(
        "price     ->",
        columns[
            "price"
        ].analysis_kind,
    )


    print(
        "categ     ->",
        columns[
            "categ"
        ].analysis_kind,
    )


    print(
        "client_id ->",
        columns[
            "client_id"
        ].analysis_kind,
    )


    print()
    print(
        "=== UNIQUE-VALUE PROTECTION ==="
    )


    print(
        "price unique values     :",
        columns[
            "price"
        ].unique_count,
    )


    print(
        "price unique_candidate  :",
        columns[
            "price"
        ].unique_candidate,
    )


    print(
        "client unique_candidate :",
        columns[
            "client_id"
        ].unique_candidate,
    )


    print()
    print(
        "=============================================="
    )

    print(
        "PASS - planner catalog central typing v0.1"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()
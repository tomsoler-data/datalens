from __future__ import annotations


import pandas as pd


from app.profiling.types import (
    infer_analytical_type,
)


# ============================================================
# VERSION
# ============================================================


TEST_RULE_VERSION = (
    "central_analytical_typing_test_v0.1"
)


# ============================================================
# FIXTURE
#
# Synthetic data reproducing the important semantic shapes
# visible in the Lapage datasets:
#
# - birth  : year of birth stored as an integer
# - price  : genuine numerical measure
# - categ  : category encoded with integers 0 / 1 / 2
# - client_id : identifier
# ============================================================


def build_test_dataframe(
) -> pd.DataFrame:
    row_count = 300


    births = [
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


    prices = [
        round(
            4.5
            +
            (
                index
                %
                80
            )
            *
            1.37,

            2,
        )

        for index
        in range(
            row_count
        )
    ]


    categories = [
        index
        %
        3

        for index
        in range(
            row_count
        )
    ]


    customer_ids = [
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
                    births,

                "price":
                    prices,

                "categ":
                    categories,

                "client_id":
                    customer_ids,
            }
        )
    )


# ============================================================
# DISPLAY HELPER
# ============================================================


def display_result(
    *,
    column_name: str,
    result: dict,
) -> None:
    print()
    print(
        f"Column  : {column_name}"
    )

    print(
        f"Type    : {result.get('type')}"
    )

    print(
        f"Subtype : {result.get('subtype')}"
    )

    print(
        f"Reason  : {result.get('reason')}"
    )


# ============================================================
# TEST
# ============================================================


def main(
) -> None:
    dataframe = (
        build_test_dataframe()
    )


    results = {
        column_name:
            infer_analytical_type(
                column_name,
                dataframe[
                    column_name
                ],
            )

        for column_name
        in dataframe.columns
    }


    print()
    print(
        "========================================"
    )

    print(
        "DataLens Central Analytical Typing v0.1"
    )

    print(
        "========================================"
    )


    for (
        column_name,
        result,
    ) in results.items():
        display_result(
            column_name=
                column_name,

            result=
                result,
        )


    # ========================================================
    # EXPECTED SEMANTICS
    # ========================================================

    birth_type = (
        results[
            "birth"
        ].get(
            "type"
        )
    )


    birth_subtype = (
        results[
            "birth"
        ].get(
            "subtype"
        )
    )


    price_type = (
        results[
            "price"
        ].get(
            "type"
        )
    )


    category_type = (
        results[
            "categ"
        ].get(
            "type"
        )
    )


    identifier_type = (
        results[
            "client_id"
        ].get(
            "type"
        )
    )


    # --------------------------------------------------------
    # Birth year
    #
    # We want birth to be semantically temporal rather than a
    # normal quantitative measure.
    # --------------------------------------------------------

    assert (
        birth_type
        ==
        "temporal"
    ), (
        "birth should be classified as temporal, "
        f"got {birth_type!r}."
    )


    assert (
        birth_subtype
        in {
            "year",
            "birth_year",
        }
    ), (
        "birth should use a year-like temporal subtype, "
        f"got {birth_subtype!r}."
    )


    # --------------------------------------------------------
    # Genuine measure
    # --------------------------------------------------------

    assert (
        price_type
        ==
        "quantitative"
    ), (
        "price should remain quantitative, "
        f"got {price_type!r}."
    )


    # --------------------------------------------------------
    # Numeric category code
    #
    # Storage as integer must not automatically imply a
    # quantitative business variable.
    # --------------------------------------------------------

    assert (
        category_type
        ==
        "categorical"
    ), (
        "categ should be classified as categorical, "
        f"got {category_type!r}."
    )


    # --------------------------------------------------------
    # Identifier
    # --------------------------------------------------------

    assert (
        identifier_type
        ==
        "identifier"
    ), (
        "client_id should be classified as identifier, "
        f"got {identifier_type!r}."
    )


    print()
    print(
        "========================================"
    )

    print(
        "PASS - central analytical typing v0.1"
    )

    print(
        "========================================"
    )

    print()
    print(
        "birth     -> TEMPORAL"
    )

    print(
        "price     -> QUANTITATIVE"
    )

    print(
        "categ     -> CATEGORICAL"
    )

    print(
        "client_id -> IDENTIFIER"
    )


if __name__ == "__main__":
    main()
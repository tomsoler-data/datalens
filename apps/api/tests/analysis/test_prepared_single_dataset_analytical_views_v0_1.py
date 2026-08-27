from __future__ import annotations


import pandas as pd


from app.analysis.analytical_views import (
    build_analytical_views,
    materialize_views_for_fact,
)


# ============================================================
# FIXTURE
# ============================================================

def build_prepared_lapage_dataframe() -> pd.DataFrame:
    """
    Represents the important architectural case introduced by
    Preparation:

        Transactions
            + customers
            + products
                ↓
        one already-enriched validated final dataset

    Analysis must be able to materialize analytical grains from
    this dataset without requiring another inter-dataset join.
    """

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2022-01-05",
                    "2022-01-05",
                    "2022-01-18",
                    "2022-02-02",
                    "2022-02-02",
                    "2022-02-15",
                    "2022-03-03",
                    "2022-03-03",
                    "2022-03-21",
                    "2022-04-08",
                    "2022-04-08",
                    "2022-04-19",
                ]
            ),

            "session_id": [
                "s_001",
                "s_001",
                "s_002",
                "s_003",
                "s_003",
                "s_004",
                "s_005",
                "s_005",
                "s_006",
                "s_007",
                "s_007",
                "s_008",
            ],

            "client_id": [
                "c_001",
                "c_001",
                "c_001",
                "c_002",
                "c_002",
                "c_002",
                "c_003",
                "c_003",
                "c_003",
                "c_004",
                "c_004",
                "c_004",
            ],

            "id_prod": [
                "p_001",
                "p_002",
                "p_003",
                "p_001",
                "p_004",
                "p_002",
                "p_005",
                "p_001",
                "p_003",
                "p_004",
                "p_005",
                "p_002",
            ],

            "sex": [
                "f",
                "f",
                "f",
                "m",
                "m",
                "m",
                "f",
                "f",
                "f",
                "m",
                "m",
                "m",
            ],

            "birth": [
                1985,
                1985,
                1985,
                1978,
                1978,
                1978,
                1992,
                1992,
                1992,
                1968,
                1968,
                1968,
            ],

            "categ": [
                "0",
                "1",
                "2",
                "0",
                "1",
                "1",
                "2",
                "0",
                "2",
                "1",
                "2",
                "0",
            ],

            "revenue": [
                20.0,
                15.0,
                35.0,
                12.0,
                45.0,
                18.0,
                25.0,
                16.0,
                30.0,
                50.0,
                22.0,
                14.0,
            ],
        }
    )


def build_prepared_dataset_record() -> dict:
    dataframe = (
        build_prepared_lapage_dataframe()
    )

    return {
        "dataset_id":
            "dataset:lapage_prepared",

        "filename":
            "lapage_prepared.csv",

        "extension":
            ".csv",

        "dataframe":
            dataframe,

        # Important:
        # this represents a final Preparation artifact supplied
        # to Analysis. It is not an analytical derived view.
        "is_derived":
            False,

        "preparation_stage":
            "COMBINE",

        "preparation_parent_dataset_ids": [
            "dataset:transactions",
            "dataset:customers",
            "dataset:products",
        ],
    }


# ============================================================
# HELPERS
# ============================================================

def assert_true(
    value,
    message: str,
) -> None:
    if not value:
        raise AssertionError(
            message
        )


def pass_test(
    message: str,
) -> None:
    print(
        f"[PASS] {message}"
    )


def derivation_types(
    datasets: list[
        dict
    ],
) -> set[str]:
    return {
        str(
            dataset.get(
                "derivation_type",
                "",
            )
        )

        for dataset
        in datasets
    }


def provenance_operations(
    datasets: list[
        dict
    ],
) -> set[str]:
    operations: set[str] = set()

    for dataset in datasets:
        provenance = (
            dataset.get(
                "provenance"
            )
        )

        if not isinstance(
            provenance,
            dict,
        ):
            continue

        operation = (
            provenance.get(
                "operation"
            )
        )

        if operation:
            operations.add(
                str(
                    operation
                )
            )

    return operations


# ============================================================
# CONTROL TEST
#
# Proves that the lower-level materializer already knows how
# to operate on one already-enriched prepared DataFrame.
# ============================================================

def test_materializer_accepts_prepared_dataset(
) -> None:
    record = (
        build_prepared_dataset_record()
    )

    dataframe = (
        record[
            "dataframe"
        ]
    )

    (
        derived,
        audits,
    ) = materialize_views_for_fact(
        fact_dataset_id=
            record[
                "dataset_id"
            ],

        fact_filename=
            record[
                "filename"
            ],

        enriched=
            dataframe,

        source_dataset_ids=[
            record[
                "dataset_id"
            ]
        ],

        fact_original_columns={
            str(
                column
            )

            for column
            in dataframe.columns
        },

        propagated_columns=
            set(),

        include_requested_context=
            True,
    )

    print()
    print(
        "===== DIRECT MATERIALIZER ====="
    )
    print()

    print(
        f"derived datasets: {len(derived)}"
    )

    print(
        "derivation types:",
        sorted(
            derivation_types(
                derived
            )
        ),
    )

    print(
        "provenance operations:",
        sorted(
            provenance_operations(
                derived
            )
        ),
    )

    print(
        f"audits: {len(audits)}"
    )

    assert_true(
        len(
            derived
        )
        >
        0,
        (
            "The lower-level materializer should be able "
            "to create analytical views directly from an "
            "already-enriched prepared dataset."
        ),
    )

    types = (
        derivation_types(
            derived
        )
    )

    assert_true(
        "requested_event_context"
        in
        types,
        (
            "The prepared dataset should support a "
            "requested-event context view."
        ),
    )

    assert_true(
        "monthly_additive_measure"
        in
        types,
        (
            "The prepared dataset should support "
            "monthly additive views."
        ),
    )

    assert_true(
        "categorical_additive_measure"
        in
        types,
        (
            "The prepared dataset should support "
            "categorical additive views."
        ),
    )

    assert_true(
        "entity_additive_measure"
        in
        types,
        (
            "The prepared dataset should support "
            "entity/session/customer analytical views."
        ),
    )

    pass_test(
        (
            "lower-level materializer accepts one "
            "already-enriched Preparation output"
        )
    )


# ============================================================
# REGRESSION TEST
#
# This is expected to FAIL with the current implementation.
#
# build_analytical_views currently reaches
# materialize_views_for_fact only after at least one accepted
# inter-dataset enrichment.
#
# One already-combined Preparation output therefore produces
# zero derived views even though the lower-level materializer
# can handle it.
# ============================================================

def test_builder_materializes_prepared_single_dataset(
) -> None:
    record = (
        build_prepared_dataset_record()
    )

    result = (
        build_analytical_views(
            [
                record
            ],

            include_requested_context=
                True,
        )
    )

    derived = (
        result
        .derived_datasets
    )

    print()
    print(
        "===== TOP-LEVEL VIEW BUILDER ====="
    )
    print()

    print(
        f"original datasets: "
        f"{len(result.original_datasets)}"
    )

    print(
        f"derived datasets: "
        f"{len(derived)}"
    )

    print(
        f"join audits: "
        f"{len(result.join_audits)}"
    )

    print(
        f"derived audits: "
        f"{len(result.derived_audits)}"
    )

    print(
        "derivation types:",
        sorted(
            derivation_types(
                derived
            )
        ),
    )

    print()

    print(
        "notes:"
    )

    for note in (
        result.notes
    ):
        print(
            f"  - {note}"
        )

    assert_true(
        len(
            derived
        )
        >
        0,
        (
            "REGRESSION: build_analytical_views() must "
            "materialize analytical views from a single "
            "already-enriched validated Preparation output. "
            "An inter-dataset join must not be required when "
            "Preparation has already performed the enrichment."
        ),
    )

    types = (
        derivation_types(
            derived
        )
    )

    assert_true(
        "requested_event_context"
        in
        types,
        (
            "Prepared single-dataset analysis must preserve "
            "requested-only event context."
        ),
    )

    assert_true(
        "monthly_additive_measure"
        in
        types,
        (
            "Prepared single-dataset analysis must support "
            "monthly additive materialization."
        ),
    )

    assert_true(
        "categorical_additive_measure"
        in
        types,
        (
            "Prepared single-dataset analysis must support "
            "categorical additive materialization."
        ),
    )

    assert_true(
        "entity_additive_measure"
        in
        types,
        (
            "Prepared single-dataset analysis must support "
            "entity/session/customer materialization."
        ),
    )

    pass_test(
        (
            "top-level builder supports one "
            "already-enriched Preparation output"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        (
            "=== DATALENS PREPARED SINGLE DATASET "
            "ANALYTICAL VIEWS v0.1 ==="
        )
    )

    test_materializer_accepts_prepared_dataset()

    test_builder_materializes_prepared_single_dataset()

    print()

    print(
        (
            "PASS - prepared single dataset "
            "analytical views v0.1"
        )
    )


if (
    __name__
    ==
    "__main__"
):
    main()
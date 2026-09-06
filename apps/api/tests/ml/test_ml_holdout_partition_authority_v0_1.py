from __future__ import annotations


from pathlib import (
    Path,
)


import pandas as pd


from app.ml.classical_executor import (
    _split_dataset,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.splitting import (
    ML_HOLDOUT_PARTITION_RULE_VERSION,
    MLHoldoutPartition,
    MLSplitInputError,
    resolve_ml_holdout_partition,
)


# ============================================================
# FIXTURE
# ============================================================


def build_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:position-authority",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "feature_a",
                "feature_b",
            ],

            estimator_key=
                "linear_regression",

            split={
                "strategy":
                    "holdout",

                "test_size":
                    0.25,

                "random_seed":
                    42,

                "shuffle":
                    True,

                "stratify":
                    False,
            },
        )
    )


def build_xy(
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:

    source_index = [
        1000
        +
        value
        *
        10

        for value
        in range(
            20
        )
    ]


    x = pd.DataFrame(
        {
            "feature_a":
                [
                    float(
                        value
                    )

                    for value
                    in range(
                        20
                    )
                ],

            "feature_b":
                [
                    float(
                        value
                        *
                        2
                    )

                    for value
                    in range(
                        20
                    )
                ],
        },
        index=
            source_index,
    )


    y = pd.Series(
        [
            float(
                3
                *
                value
                +
                1
            )

            for value
            in range(
                20
            )
        ],

        index=
            source_index,

        name=
            "target",
    )


    return (
        x,
        y,
    )


# ============================================================
# EXACT CLASSICAL POPULATION
# ============================================================


def test_positions_reconstruct_exact_classical_holdout(
) -> None:

    contract = (
        build_contract()
    )


    (
        x,
        y,
    ) = (
        build_xy()
    )


    classical = (
        _split_dataset(
            x=x,
            y=y,
            contract=contract,
        )
    )


    partition = (
        resolve_ml_holdout_partition(
            x=x,
            y=y,
            contract=contract,
        )
    )


    assert isinstance(
        partition,
        MLHoldoutPartition,
    )


    reconstructed_train = (
        x.iloc[
            list(
                partition
                .train_positions
            )
        ]
    )


    reconstructed_test = (
        x.iloc[
            list(
                partition
                .test_positions
            )
        ]
    )


    assert (
        reconstructed_train
        .index
        .tolist()
        ==
        classical[
            0
        ]
        .index
        .tolist()
    )


    assert (
        reconstructed_test
        .index
        .tolist()
        ==
        classical[
            1
        ]
        .index
        .tolist()
    )


    assert (
        y.iloc[
            list(
                partition
                .train_positions
            )
        ]
        .index
        .tolist()
        ==
        classical[
            2
        ]
        .index
        .tolist()
    )


    assert (
        y.iloc[
            list(
                partition
                .test_positions
            )
        ]
        .index
        .tolist()
        ==
        classical[
            3
        ]
        .index
        .tolist()
    )


# ============================================================
# COMPLETE NON-PURGED PARTITION
# ============================================================


def test_standard_holdout_partitions_every_source_row(
) -> None:

    contract = (
        build_contract()
    )


    (
        x,
        y,
    ) = (
        build_xy()
    )


    partition = (
        resolve_ml_holdout_partition(
            x=x,
            y=y,
            contract=contract,
        )
    )


    assert (
        partition.source_row_count
        ==
        20
    )


    assert (
        partition.train_rows
        ==
        15
    )


    assert (
        partition.test_rows
        ==
        5
    )


    assert (
        partition.purged_rows
        ==
        0
    )


    assert (
        set(
            partition
            .train_positions
        )
        |
        set(
            partition
            .test_positions
        )
        ==
        set(
            range(
                20
            )
        )
    )


# ============================================================
# DETERMINISM
# ============================================================


def test_position_authority_is_deterministic(
) -> None:

    contract = (
        build_contract()
    )


    (
        x,
        y,
    ) = (
        build_xy()
    )


    first = (
        resolve_ml_holdout_partition(
            x=x,
            y=y,
            contract=contract,
        )
    )


    second = (
        resolve_ml_holdout_partition(
            x=x,
            y=y,
            contract=contract,
        )
    )


    assert (
        first
        ==
        second
    )


# ============================================================
# PANDAS INDEX INDEPENDENCE
# ============================================================


def test_positions_do_not_depend_on_source_index_labels(
) -> None:

    contract = (
        build_contract()
    )


    (
        x,
        y,
    ) = (
        build_xy()
    )


    first = (
        resolve_ml_holdout_partition(
            x=x,
            y=y,
            contract=contract,
        )
    )


    relabeled_x = (
        x.copy(
            deep=True
        )
    )

    relabeled_y = (
        y.copy(
            deep=True
        )
    )


    duplicate_labels = [
        value
        //
        2

        for value
        in range(
            20
        )
    ]


    relabeled_x.index = (
        duplicate_labels
    )

    relabeled_y.index = (
        duplicate_labels
    )


    second = (
        resolve_ml_holdout_partition(
            x=
                relabeled_x,

            y=
                relabeled_y,

            contract=
                contract,
        )
    )


    assert (
        first.train_positions
        ==
        second.train_positions
    )


    assert (
        first.test_positions
        ==
        second.test_positions
    )


# ============================================================
# ALIGNMENT FAIL-CLOSED
# ============================================================


def test_misaligned_feature_target_indexes_fail_closed(
) -> None:

    contract = (
        build_contract()
    )


    (
        x,
        y,
    ) = (
        build_xy()
    )


    broken_y = (
        y.copy(
            deep=True
        )
    )


    broken_y.index = list(
        reversed(
            broken_y.index.tolist()
        )
    )


    try:

        resolve_ml_holdout_partition(
            x=x,
            y=broken_y,
            contract=contract,
        )

    except MLSplitInputError:
        return


    raise AssertionError(
        "Expected misaligned indexes to fail closed."
    )


# ============================================================
# TORCH-FREE AUTHORITY
# ============================================================


def test_position_authority_is_torch_free(
) -> None:

    path = (
        Path(__file__)
        .parents[2]
        /
        "app"
        /
        "ml"
        /
        "splitting.py"
    )


    source = path.read_text(
        encoding="utf-8"
    )


    for token in (
        "import torch",
        "from torch",
        "torch.",
    ):

        assert (
            token
            not in
            source
        )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_HOLDOUT_PARTITION_RULE_VERSION
        ==
        "ml_holdout_partition_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS HOLDOUT POSITION AUTHORITY v0.1 ==="
    )

    print()


    test_positions_reconstruct_exact_classical_holdout()

    print(
        "Exact Classical holdout population: PASS"
    )


    test_standard_holdout_partitions_every_source_row()

    print(
        "Complete non-purged row partition: PASS"
    )


    test_position_authority_is_deterministic()

    print(
        "Position authority determinism: PASS"
    )


    test_positions_do_not_depend_on_source_index_labels()

    print(
        "Pandas index-label independence: PASS"
    )


    test_misaligned_feature_target_indexes_fail_closed()

    print(
        "Feature/target alignment guard: PASS"
    )


    test_position_authority_is_torch_free()

    print(
        "Torch-free position authority: PASS"
    )


    test_rule_version()

    print(
        "Position authority rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Holdout Position Authority v0.1"
    )


if __name__ == "__main__":
    main()

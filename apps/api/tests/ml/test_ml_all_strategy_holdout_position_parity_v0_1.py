from __future__ import annotations


from pathlib import (
    Path,
)


import pandas as pd


from app.ml.classical_executor import (
    _split_dataset,
)


from app.ml.contracts import (
    MLGroupHoldoutSplitContract,
    MLPurgedGroupTimeHoldoutSplitContract,
    MLSplitContract,
    MLTimeHoldoutSplitContract,
    MLTrainingContract,
)


from app.ml.splitting import (
    ML_HOLDOUT_PARTITION_RULE_VERSION,
    MLHoldoutPartition,
    resolve_ml_holdout_partition,
)


# ============================================================
# VERSION
# ============================================================


ML_ALL_STRATEGY_POSITION_PARITY_RULE_VERSION = (
    "ml_all_strategy_position_parity_v0.1"
)


# ============================================================
# DATA
# ============================================================


INDEX_LABELS = [
    10_001,
    10_014,
    10_027,
    10_040,
    10_053,
    10_066,
    10_079,
    10_092,
    10_105,
    10_118,
]


def build_dataframe(
    *,
    groups: list[str],
) -> pd.DataFrame:

    assert (
        len(groups)
        ==
        10
    )


    return (
        pd.DataFrame(
            {
                "feature_a": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                ],

                "feature_b": [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                    70.0,
                    80.0,
                    90.0,
                    100.0,
                ],

                "target": [
                    11.0,
                    22.0,
                    33.0,
                    44.0,
                    55.0,
                    66.0,
                    77.0,
                    88.0,
                    99.0,
                    110.0,
                ],

                "entity_id":
                    groups,

                "event_time":
                    pd.date_range(
                        "2026-01-01",
                        periods=10,
                        freq="D",
                    ),
            },
            index=
                INDEX_LABELS,
        )
    )


# ============================================================
# CONTRACT
# ============================================================


def build_contract(
    *,
    split,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:position-matrix",

            dataset_id=
                "dataset:position-matrix",

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

            split=
                split,
        )
    )


# ============================================================
# POSITION HELPERS
# ============================================================


def feature_target_surface(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:

    x = (
        dataframe[
            [
                "feature_a",
                "feature_b",
            ]
        ]
        .copy(
            deep=True
        )
    )


    y = (
        dataframe[
            "target"
        ]
        .copy(
            deep=True
        )
    )


    return (
        x,
        y,
    )


def index_labels_to_positions(
    *,
    source_index: pd.Index,
    subset_index: pd.Index,
) -> tuple[
    int,
    ...
]:

    mapping = {
        label:
            position

        for position, label
        in enumerate(
            source_index.tolist()
        )
    }


    return tuple(
        mapping[
            label
        ]

        for label
        in subset_index.tolist()
    )


def classical_partition_positions(
    *,
    dataframe: pd.DataFrame,
    contract: MLTrainingContract,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:

    (
        x,
        y,
    ) = (
        feature_target_surface(
            dataframe
        )
    )


    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = (
        _split_dataset(
            x=
                x,

            y=
                y,

            contract=
                contract,

            dataframe=
                dataframe,
        )
    )


    assert (
        x_train.index.tolist()
        ==
        y_train.index.tolist()
    )


    assert (
        x_test.index.tolist()
        ==
        y_test.index.tolist()
    )


    train_positions = (
        index_labels_to_positions(
            source_index=
                x.index,

            subset_index=
                x_train.index,
        )
    )


    test_positions = (
        index_labels_to_positions(
            source_index=
                x.index,

            subset_index=
                x_test.index,
        )
    )


    used = (
        set(
            train_positions
        )
        |
        set(
            test_positions
        )
    )


    purged_positions = tuple(
        position

        for position
        in range(
            len(
                x
            )
        )

        if position
        not in
        used
    )


    return (
        train_positions,
        test_positions,
        purged_positions,
    )


def shared_partition(
    *,
    dataframe: pd.DataFrame,
    contract: MLTrainingContract,
) -> MLHoldoutPartition:

    (
        x,
        y,
    ) = (
        feature_target_surface(
            dataframe
        )
    )


    return (
        resolve_ml_holdout_partition(
            x=
                x,

            y=
                y,

            contract=
                contract,

            dataframe=
                dataframe,
        )
    )


def assert_exact_parity(
    *,
    dataframe: pd.DataFrame,
    contract: MLTrainingContract,
    expected_train: tuple[int, ...],
    expected_test: tuple[int, ...],
    expected_purged: tuple[int, ...],
) -> None:

    partition = (
        shared_partition(
            dataframe=
                dataframe,

            contract=
                contract,
        )
    )


    assert (
        tuple(
            partition.train_positions
        )
        ==
        expected_train
    )


    assert (
        tuple(
            partition.test_positions
        )
        ==
        expected_test
    )


    assert (
        tuple(
            partition.purged_positions
        )
        ==
        expected_purged
    )


    classical = (
        classical_partition_positions(
            dataframe=
                dataframe,

            contract=
                contract,
        )
    )


    assert (
        classical
        ==
        (
            expected_train,
            expected_test,
            expected_purged,
        )
    )


    assert (
        partition.source_row_count
        ==
        10
    )


    assert (
        partition.train_rows
        ==
        len(
            expected_train
        )
    )


    assert (
        partition.test_rows
        ==
        len(
            expected_test
        )
    )


    assert (
        partition.purged_rows
        ==
        len(
            expected_purged
        )
    )


# ============================================================
# REGULAR HOLDOUT
# ============================================================


def test_regular_holdout_exact_positions(
) -> None:

    dataframe = (
        build_dataframe(
            groups=[
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
                "D",
                "D",
                "E",
                "E",
            ]
        )
    )


    contract = (
        build_contract(
            split=
                MLSplitContract(
                    test_size=
                        0.20,

                    random_seed=
                        42,

                    shuffle=
                        False,

                    stratify=
                        False,
                )
        )
    )


    assert_exact_parity(
        dataframe=
            dataframe,

        contract=
            contract,

        expected_train=(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ),

        expected_test=(
            8,
            9,
        ),

        expected_purged=(),
    )


# ============================================================
# GROUP HOLDOUT
# ============================================================


def test_group_holdout_exact_positions(
) -> None:

    dataframe = (
        build_dataframe(
            groups=[
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
                "D",
                "D",
                "E",
                "E",
            ]
        )
    )


    contract = (
        build_contract(
            split=
                MLGroupHoldoutSplitContract(
                    group_column=
                        "entity_id",

                    test_size=
                        0.20,

                    random_seed=
                        42,
                )
        )
    )


    # sklearn GroupShuffleSplit(random_state=42)
    # selects group B from this fixed five-group surface.
    assert_exact_parity(
        dataframe=
            dataframe,

        contract=
            contract,

        expected_train=(
            0,
            1,
            4,
            5,
            6,
            7,
            8,
            9,
        ),

        expected_test=(
            2,
            3,
        ),

        expected_purged=(),
    )


# ============================================================
# TIME HOLDOUT
# ============================================================


def test_time_holdout_exact_positions(
) -> None:

    dataframe = (
        build_dataframe(
            groups=[
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
                "D",
                "D",
                "E",
                "E",
            ]
        )
    )


    contract = (
        build_contract(
            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "event_time",

                    test_size=
                        0.20,
                )
        )
    )


    assert_exact_parity(
        dataframe=
            dataframe,

        contract=
            contract,

        expected_train=(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ),

        expected_test=(
            8,
            9,
        ),

        expected_purged=(),
    )


# ============================================================
# PURGED GROUP + TIME HOLDOUT
# ============================================================


def test_purged_group_time_holdout_exact_positions(
) -> None:

    dataframe = (
        build_dataframe(
            groups=[
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
                "D",
                "D",
                "A",
                "B",
            ]
        )
    )


    contract = (
        build_contract(
            split=
                MLPurgedGroupTimeHoldoutSplitContract(
                    group_column=
                        "entity_id",

                    time_column=
                        "event_time",

                    test_size=
                        0.20,
                )
        )
    )


    # Future TEST contains groups A and B.
    #
    # Historical rows belonging to A/B are therefore purged.
    assert_exact_parity(
        dataframe=
            dataframe,

        contract=
            contract,

        expected_train=(
            4,
            5,
            6,
            7,
        ),

        expected_test=(
            8,
            9,
        ),

        expected_purged=(
            0,
            1,
            2,
            3,
        ),
    )


# ============================================================
# DEEP LEARNING AUTHORITY WIRING
# ============================================================


def test_deep_learning_core_consumes_exact_partition_authority(
) -> None:

    path = (
        Path(__file__)
        .parents[
            2
        ]
        /
        "app"
        /
        "deep_learning"
        /
        "tabular_executor.py"
    )


    source = path.read_text(
        encoding="utf-8"
    )


    required = (
        "resolve_ml_holdout_partition(",
        "partition.train_positions",
        "partition.test_positions",
    )


    for token in required:

        assert (
            token
            in
            source
        )


# ============================================================
# POSITION / LABEL INDEPENDENCE
# ============================================================


def test_matrix_uses_positions_not_pandas_labels(
) -> None:

    assert (
        INDEX_LABELS
        !=
        list(
            range(
                len(
                    INDEX_LABELS
                )
            )
        )
    )


    assert (
        len(
            set(
                INDEX_LABELS
            )
        )
        ==
        len(
            INDEX_LABELS
        )
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        ML_HOLDOUT_PARTITION_RULE_VERSION
        ==
        "ml_holdout_partition_v0.1"
    )


    assert (
        ML_ALL_STRATEGY_POSITION_PARITY_RULE_VERSION
        ==
        "ml_all_strategy_position_parity_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ALL-STRATEGY HOLDOUT POSITION PARITY v0.1 ==="
    )

    print()


    test_regular_holdout_exact_positions()

    print(
        "Regular holdout exact positions: PASS"
    )


    test_group_holdout_exact_positions()

    print(
        "Group holdout exact positions: PASS"
    )


    test_time_holdout_exact_positions()

    print(
        "Time holdout exact positions: PASS"
    )


    test_purged_group_time_holdout_exact_positions()

    print(
        "Purged group/time exact positions: PASS"
    )


    test_deep_learning_core_consumes_exact_partition_authority()

    print(
        "Deep Learning exact partition authority wiring: PASS"
    )


    test_matrix_uses_positions_not_pandas_labels()

    print(
        "Pandas label / positional authority separation: PASS"
    )


    test_rule_versions()

    print(
        "All-strategy position rule versions: PASS"
    )


    print()

    print(
        "PASS - DataLens All-Strategy Holdout Position Parity v0.1"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations


import inspect
import sys


import pandas as pd


import app.ml.splitting as splitting


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.contracts import (
    MLGroupHoldoutSplitContract,
    MLPurgedGroupTimeHoldoutSplitContract,
    MLSplitContract,
    MLTimeHoldoutSplitContract,
)


from app.ml.splitting import (
    ML_FEATURE_HOLDOUT_PARTITION_RULE_VERSION,
    resolve_ml_feature_holdout_partition,
)


print(
    "=== DATALENS FEATURE-ONLY HOLDOUT PARTITION v0.1 ==="
)
print()


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


def build_contract(
    *,
    split,
) -> MLAnomalyTrainingContract:

    return (
        MLAnomalyTrainingContract(
            workflow_id=
                "workflow:feature-holdout",
            dataset_id=
                "dataset:feature-holdout",
            feature_columns=[
                "feature_a",
                "feature_b",
            ],
            split=
                split,
        )
    )


def feature_surface(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    return (
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


# ============================================================
# TARGET-FREE API
# ============================================================


signature = inspect.signature(
    resolve_ml_feature_holdout_partition
)


assert (
    "y"
    not in
    signature.parameters
)


assert (
    "target"
    not in
    signature.parameters
)


print(
    "Target-free holdout API: PASS"
)


# ============================================================
# FIXED ANALYTICAL SEMANTICS
# ============================================================


original_infer = (
    splitting.infer_analytical_type
)


def fixed_infer(
    column_name,
    series,
):

    if (
        column_name
        ==
        "entity_id"
    ):
        return {
            "type":
                "identifier",
            "subtype":
                "reference",
        }


    if (
        column_name
        ==
        "event_time"
    ):
        return {
            "type":
                "temporal",
            "subtype":
                "datetime",
        }


    return original_infer(
        column_name,
        series,
    )


splitting.infer_analytical_type = (
    fixed_infer
)


try:

    cases = [
        (
            "regular",
            [
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
            ],
            MLSplitContract(
                test_size=
                    0.20,
                random_seed=
                    42,
                shuffle=
                    False,
                stratify=
                    False,
            ),
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ),
            (
                8,
                9,
            ),
            (),
        ),
        (
            "group",
            [
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
            ],
            MLGroupHoldoutSplitContract(
                group_column=
                    "entity_id",
                test_size=
                    0.20,
                random_seed=
                    42,
            ),
            (
                0,
                1,
                4,
                5,
                6,
                7,
                8,
                9,
            ),
            (
                2,
                3,
            ),
            (),
        ),
        (
            "time",
            [
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
            ],
            MLTimeHoldoutSplitContract(
                time_column=
                    "event_time",
                test_size=
                    0.20,
            ),
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ),
            (
                8,
                9,
            ),
            (),
        ),
        (
            "purged_group_time",
            [
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
            ],
            MLPurgedGroupTimeHoldoutSplitContract(
                group_column=
                    "entity_id",
                time_column=
                    "event_time",
                test_size=
                    0.20,
            ),
            (
                4,
                5,
                6,
                7,
            ),
            (
                8,
                9,
            ),
            (
                0,
                1,
                2,
                3,
            ),
        ),
    ]


    for (
        label,
        groups,
        split,
        expected_train,
        expected_test,
        expected_purged,
    ) in cases:

        dataframe = (
            build_dataframe(
                groups=
                    groups,
            )
        )


        x = (
            feature_surface(
                dataframe
            )
        )


        contract = (
            build_contract(
                split=
                    split,
            )
        )


        partition = (
            resolve_ml_feature_holdout_partition(
                x=
                    x,
                contract=
                    contract,
                dataframe=
                    dataframe,
            )
        )


        assert (
            partition.train_positions
            ==
            expected_train
        )


        assert (
            partition.test_positions
            ==
            expected_test
        )


        assert (
            partition.purged_positions
            ==
            expected_purged
        )


        assert (
            partition.source_row_count
            ==
            10
        )


        print(
            f"{label} exact positions: PASS"
        )


finally:

    splitting.infer_analytical_type = (
        original_infer
    )


# ============================================================
# SOURCE INDEX LABELS ARE NOT POSITIONS
# ============================================================


dataframe = build_dataframe(
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


partition = (
    resolve_ml_feature_holdout_partition(
        x=
            feature_surface(
                dataframe
            ),
        contract=
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
            ),
        dataframe=
            dataframe,
    )
)


assert (
    partition.test_positions
    ==
    (
        8,
        9,
    )
)


assert (
    10_105
    not in
    partition.test_positions
)


print(
    "Source-index independence: PASS"
)


# ============================================================
# MISALIGNED SOURCE FAIL-CLOSED
# ============================================================


misaligned = (
    dataframe.copy(
        deep=True
    )
)


misaligned.index = range(
    len(
        misaligned
    )
)


try:

    resolve_ml_feature_holdout_partition(
        x=
            feature_surface(
                dataframe
            ),
        contract=
            build_contract(
                split=
                    MLSplitContract(
                        shuffle=
                            False,
                        stratify=
                            False,
                    )
            ),
        dataframe=
            misaligned,
    )

except splitting.MLSplitInputError:
    pass

else:
    raise AssertionError(
        "Misaligned source DataFrame was accepted."
    )


print(
    "Source alignment guard: PASS"
)


# ============================================================
# TORCH-FREE AUTHORITY
# ============================================================


assert (
    "torch"
    not in
    sys.modules
)


source = open(
    "app/ml/splitting.py",
    "r",
    encoding="utf-8",
).read()


for forbidden in (
    "import torch",
    "from torch",
    "torch.",
):

    assert (
        forbidden
        not in
        source
    )


print(
    "Runtime torch isolation: PASS"
)


# ============================================================
# RULE VERSION
# ============================================================


assert (
    ML_FEATURE_HOLDOUT_PARTITION_RULE_VERSION
    ==
    "ml_feature_holdout_partition_v0.1"
)


print(
    "Feature holdout rule version: PASS"
)


print()
print(
    "PASS - DataLens Feature-Only Holdout Partition v0.1"
)

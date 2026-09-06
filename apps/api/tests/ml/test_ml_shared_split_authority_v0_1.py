from __future__ import annotations


import ast
import sys


from pathlib import (
    Path,
)


import pandas as pd


from app.ml.classical_executor import (
    ClassicalMLInputError,
    _split_dataset,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.splitting import (
    MLSplitInputError,
    ML_SPLITTING_RULE_VERSION,
    split_ml_dataset,
)


# ============================================================
# SYNTHETIC CONTRACT
# ============================================================


def build_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:shared-split",
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

    index = list(
        range(
            100,
            120,
        )
    )


    x = pd.DataFrame(
        {
            "feature_a":
                [
                    float(value)
                    for value
                    in range(20)
                ],

            "feature_b":
                [
                    float(
                        value * 2
                    )
                    for value
                    in range(20)
                ],
        },
        index=index,
    )


    y = pd.Series(
        [
            float(
                value * 3
                +
                1
            )
            for value
            in range(20)
        ],
        index=index,
        name="target",
    )


    return (
        x,
        y,
    )


# ============================================================
# SHARED / CLASSICAL PARITY
# ============================================================


def test_classical_wrapper_matches_shared_authority(
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


    shared = (
        split_ml_dataset(
            x=x,
            y=y,
            contract=contract,
        )
    )


    classical = (
        _split_dataset(
            x=x,
            y=y,
            contract=contract,
        )
    )


    assert (
        shared[
            0
        ].index.tolist()
        ==
        classical[
            0
        ].index.tolist()
    )


    assert (
        shared[
            1
        ].index.tolist()
        ==
        classical[
            1
        ].index.tolist()
    )


    assert (
        shared[
            2
        ].index.tolist()
        ==
        classical[
            2
        ].index.tolist()
    )


    assert (
        shared[
            3
        ].index.tolist()
        ==
        classical[
            3
        ].index.tolist()
    )


# ============================================================
# DETERMINISM
# ============================================================


def test_shared_holdout_is_deterministic(
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
        split_ml_dataset(
            x=x,
            y=y,
            contract=contract,
        )
    )


    second = (
        split_ml_dataset(
            x=x,
            y=y,
            contract=contract,
        )
    )


    for index in range(
        4
    ):

        assert (
            first[
                index
            ].index.tolist()
            ==
            second[
                index
            ].index.tolist()
        )


# ============================================================
# ERROR COMPATIBILITY
# ============================================================


def test_classical_error_compatibility(
) -> None:

    contract = (
        build_contract()
    )


    x = pd.DataFrame(
        {
            "feature_a":
                [
                    1.0,
                    2.0,
                    3.0,
                ],

            "feature_b":
                [
                    2.0,
                    4.0,
                    6.0,
                ],
        }
    )


    y = pd.Series(
        [
            1.0,
            2.0,
            3.0,
        ],
        name="target",
    )


    shared_message = None


    try:

        split_ml_dataset(
            x=x,
            y=y,
            contract=contract,
        )

    except MLSplitInputError as error:

        shared_message = str(
            error
        )


    assert (
        shared_message
        is not None
    )


    classical_message = None


    try:

        _split_dataset(
            x=x,
            y=y,
            contract=contract,
        )

    except ClassicalMLInputError as error:

        classical_message = str(
            error
        )


    assert (
        classical_message
        ==
        shared_message
    )


# ============================================================
# SOURCE AUTHORITY
# ============================================================


def test_classical_executor_no_longer_owns_split_algorithms(
) -> None:

    path = (
        Path(__file__)
        .parents[2]
        /
        "app"
        /
        "ml"
        /
        "classical_executor.py"
    )


    source = path.read_text(
        encoding="utf-8"
    )


    assert (
        "GroupShuffleSplit"
        not in
        source
    )


    assert (
        "train_test_split"
        not in
        source
    )


    tree = ast.parse(
        source
    )


    function_names = {
        node.name

        for node in tree.body

        if isinstance(
            node,
            ast.FunctionDef,
        )
    }


    assert (
        "_split_dataset"
        in
        function_names
    )


    assert (
        "_validated_group_values"
        in
        function_names
    )


# ============================================================
# RUNTIME ISOLATION
# ============================================================


def test_shared_split_authority_is_torch_free(
) -> None:

    assert (
        "torch"
        not in
        sys.modules
    )


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
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_SPLITTING_RULE_VERSION
        ==
        "ml_splitting_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS SHARED ML SPLIT AUTHORITY v0.1 ==="
    )

    print()


    test_classical_wrapper_matches_shared_authority()

    print(
        "Classical / shared holdout parity: PASS"
    )


    test_shared_holdout_is_deterministic()

    print(
        "Shared holdout determinism: PASS"
    )


    test_classical_error_compatibility()

    print(
        "Classical error compatibility: PASS"
    )


    test_classical_executor_no_longer_owns_split_algorithms()

    print(
        "Shared split algorithm ownership: PASS"
    )


    test_shared_split_authority_is_torch_free()

    print(
        "Torch-free split authority: PASS"
    )


    test_rule_version()

    print(
        "Split rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Shared ML Split Authority v0.1"
    )


if __name__ == "__main__":
    main()

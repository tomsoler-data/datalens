from __future__ import annotations


import sys


from dataclasses import (
    dataclass,
)


from pathlib import (
    Path,
)


import numpy as np
import pandas as pd


import app.ml.classical_executor as classical_executor


from app.ml.classical_executor import (
    ClassicalMLInputError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.training_input import (
    ML_TRAINING_INPUT_RULE_VERSION,
    MLTrainingInputError,
    load_authorized_ml_dataframe,
    validate_and_extract_ml_xy,
)


# ============================================================
# FAKE PREPARATION HANDOFF
# ============================================================


@dataclass(
    frozen=True,
)
class FakeHandoff:
    workflow_id: str
    session_revision: int
    dataset_ids: tuple[
        str,
        ...
    ]
    dataset_records: tuple[
        dict,
        ...
    ]


def build_dataframe(
) -> pd.DataFrame:

    row_count = 30


    feature_a = np.linspace(
        1.0,
        4.0,
        row_count,
        dtype=np.float64,
    )


    feature_b = np.linspace(
        5.0,
        2.0,
        row_count,
        dtype=np.float64,
    )


    revenue = (
        3.0
        *
        feature_a
        -
        1.5
        *
        feature_b
        +
        10.0
    )


    return (
        pd.DataFrame(
            {
                "feature_a":
                    feature_a,

                "feature_b":
                    feature_b,

                "revenue":
                    revenue,
            },
            index=[
                1000
                +
                index
                *
                3

                for index
                in range(
                    row_count
                )
            ],
        )
    )


def build_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:shared-training-input",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "feature_a",
                "feature_b",
            ],

            estimator_key=
                "tabular_mlp_regressor",
        )
    )


def fake_loader_factory(
    dataframe: pd.DataFrame,
):

    def fake_loader(
        *,
        workflow_id: str,
    ):

        return (
            FakeHandoff(
                workflow_id=
                    workflow_id,

                session_revision=
                    17,

                dataset_ids=(
                    "dataset:validated",
                ),

                dataset_records=(
                    {
                        "dataset_id":
                            "dataset:validated",

                        "dataframe":
                            dataframe.copy(
                                deep=True
                            ),
                    },
                ),
            )
        )


    return fake_loader


# ============================================================
# SHARED LOADER
# ============================================================


def test_shared_authorized_loader(
) -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract()
    )


    (
        loaded,
        revision,
    ) = (
        load_authorized_ml_dataframe(
            contract=contract,
            handoff_loader=
                fake_loader_factory(
                    dataframe
                ),
        )
    )


    pd.testing.assert_frame_equal(
        loaded,
        dataframe,
    )


    assert (
        revision
        ==
        17
    )


    assert (
        loaded
        is not
        dataframe
    )


# ============================================================
# CLASSICAL MONKEYPATCH COMPATIBILITY
# ============================================================


def test_classical_loader_monkeypatch_compatibility(
) -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract()
    )


    original = (
        classical_executor
        .load_validated_analysis_input
    )


    classical_executor.load_validated_analysis_input = (
        fake_loader_factory(
            dataframe
        )
    )


    try:

        (
            loaded,
            revision,
        ) = (
            classical_executor
            ._load_authorized_dataframe(
                contract=contract
            )
        )

    finally:

        classical_executor.load_validated_analysis_input = (
            original
        )


    pd.testing.assert_frame_equal(
        loaded,
        dataframe,
    )


    assert (
        revision
        ==
        17
    )


# ============================================================
# SHARED / CLASSICAL X-Y PARITY
# ============================================================


def test_shared_and_classical_xy_are_identical(
) -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract()
    )


    (
        shared_x,
        shared_y,
    ) = (
        validate_and_extract_ml_xy(
            dataframe=dataframe,
            contract=contract,
        )
    )


    (
        classical_x,
        classical_y,
    ) = (
        classical_executor
        ._validate_and_extract_xy(
            dataframe=dataframe,
            contract=contract,
        )
    )


    pd.testing.assert_frame_equal(
        shared_x,
        classical_x,
    )


    pd.testing.assert_series_equal(
        shared_y,
        classical_y,
    )


# ============================================================
# ERROR COMPATIBILITY
# ============================================================


def test_shared_and_classical_missing_column_fail_closed(
) -> None:

    dataframe = (
        build_dataframe()
        .drop(
            columns=[
                "feature_b",
            ]
        )
    )


    contract = (
        build_contract()
    )


    shared_message = None


    try:

        validate_and_extract_ml_xy(
            dataframe=dataframe,
            contract=contract,
        )

    except MLTrainingInputError as error:

        shared_message = str(
            error
        )


    assert (
        shared_message
        is not None
    )


    classical_message = None


    try:

        classical_executor._validate_and_extract_xy(
            dataframe=dataframe,
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
# SOURCE OWNERSHIP
# ============================================================


def test_classical_executor_no_longer_owns_input_algorithms(
) -> None:

    root = (
        Path(__file__)
        .parents[
            2
        ]
    )


    classical_source = (
        root
        /
        "app"
        /
        "ml"
        /
        "classical_executor.py"
    ).read_text(
        encoding="utf-8"
    )


    shared_source = (
        root
        /
        "app"
        /
        "ml"
        /
        "training_input.py"
    ).read_text(
        encoding="utf-8"
    )


    assert (
        "infer_analytical_type"
        not in
        classical_source
    )


    assert (
        "validate_ml_feature_frame"
        not in
        classical_source
    )


    assert (
        "infer_analytical_type"
        in
        shared_source
    )


    assert (
        "validate_ml_feature_frame"
        in
        shared_source
    )


    assert (
        "def _load_authorized_dataframe"
        in
        classical_source
    )


    assert (
        "def _validate_and_extract_xy"
        in
        classical_source
    )


# ============================================================
# TORCH-FREE AUTHORITY
# ============================================================


def test_shared_training_input_is_torch_free(
) -> None:

    assert (
        "torch"
        not in
        sys.modules
    )


    path = (
        Path(__file__)
        .parents[
            2
        ]
        /
        "app"
        /
        "ml"
        /
        "training_input.py"
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
        ML_TRAINING_INPUT_RULE_VERSION
        ==
        "ml_training_input_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS SHARED MODEL LAB TRAINING INPUT v0.1 ==="
    )

    print()


    test_shared_authorized_loader()

    print(
        "Shared Preparation handoff authority: PASS"
    )


    test_classical_loader_monkeypatch_compatibility()

    print(
        "Classical monkeypatch compatibility: PASS"
    )


    test_shared_and_classical_xy_are_identical()

    print(
        "Shared / Classical X-Y parity: PASS"
    )


    test_shared_and_classical_missing_column_fail_closed()

    print(
        "Input error compatibility: PASS"
    )


    test_classical_executor_no_longer_owns_input_algorithms()

    print(
        "Shared input algorithm ownership: PASS"
    )


    test_shared_training_input_is_torch_free()

    print(
        "Torch-free training input authority: PASS"
    )


    test_rule_version()

    print(
        "Training input rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Shared Model Lab Training Input v0.1"
    )


if __name__ == "__main__":
    main()

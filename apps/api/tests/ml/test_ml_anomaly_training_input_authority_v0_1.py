from __future__ import annotations


import sys


from dataclasses import (
    dataclass,
)


import numpy as np
import pandas as pd


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


import app.ml.anomaly_training_input as anomaly_input_module


from app.ml.anomaly_training_input import (
    ML_ANOMALY_TRAINING_INPUT_RULE_VERSION,
    validate_and_extract_anomaly_x,
)


from app.ml.training_input import (
    MLTrainingInputError,
    load_authorized_ml_dataframe,
)


print(
    "=== DATALENS ANOMALY FEATURE-ONLY TRAINING INPUT v0.1 ==="
)
print()


# ============================================================
# FAKE PREPARATION HANDOFF
# ============================================================


@dataclass(
    frozen=True,
)
class FakeHandoff:

    workflow_id: str

    dataset_ids: tuple[
        str,
        ...
    ]

    dataset_records: tuple[
        dict[
            str,
            object,
        ],
        ...
    ]

    session_revision: int


source_dataframe = pd.DataFrame(
    {
        "amount":
            [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
        "age":
            [
                21.0,
                35.0,
                42.0,
                58.0,
            ],
        "segment":
            [
                "A",
                "B",
                "A",
                "C",
            ],
        # Deliberately irrelevant to the anomaly contract.
        # A feature-only training input must not require this
        # column to behave like a supervised target.
        "unused_target":
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
    }
)


handoff = FakeHandoff(
    workflow_id=
        "workflow:anomaly",
    dataset_ids=(
        "dataset:validated",
    ),
    dataset_records=(
        {
            "dataset_id":
                "dataset:validated",
            "dataframe":
                source_dataframe,
        },
    ),
    session_revision=
        17,
)


contract = MLAnomalyTrainingContract(
    workflow_id=
        "workflow:anomaly",
    dataset_id=
        "dataset:validated",
    feature_columns=[
        "amount",
        "age",
        "segment",
    ],
    categorical_feature_columns=[
        "segment",
    ],
)


# ============================================================
# SHARED PREPARATION HANDOFF AUTHORITY
# ============================================================


authorized_dataframe, revision = (
    load_authorized_ml_dataframe(
        contract=contract,
        handoff_loader=(
            lambda *,
            workflow_id:
                handoff
        ),
        execution_label=
            "Anomaly Model Lab",
    )
)


assert (
    revision
    ==
    17
)


assert (
    authorized_dataframe.equals(
        source_dataframe
    )
)


assert (
    authorized_dataframe
    is not
    source_dataframe
)


authorized_dataframe.loc[
    0,
    "amount",
] = 999.0


assert (
    source_dataframe.loc[
        0,
        "amount",
    ]
    ==
    10.0
)


print(
    "Shared Preparation handoff authority: PASS"
)


# ============================================================
# FEATURE-ONLY EXTRACTION
# ============================================================


x = validate_and_extract_anomaly_x(
    dataframe=
        source_dataframe,
    contract=
        contract,
)


assert (
    list(
        x.columns
    )
    ==
    [
        "amount",
        "age",
        "segment",
    ]
)


assert (
    "unused_target"
    not in
    x.columns
)


assert (
    len(
        x
    )
    ==
    len(
        source_dataframe
    )
)


print(
    "Target-free feature extraction: PASS"
)


# ============================================================
# DEEP-COPY / INPUT ISOLATION
# ============================================================


x.loc[
    0,
    "amount",
] = 777.0


assert (
    source_dataframe.loc[
        0,
        "amount",
    ]
    ==
    10.0
)


print(
    "Feature input isolation: PASS"
)


# ============================================================
# MISSING FEATURE FAIL-CLOSED
# ============================================================


missing_frame = (
    source_dataframe
    .drop(
        columns=[
            "age",
        ]
    )
)


try:

    validate_and_extract_anomaly_x(
        dataframe=
            missing_frame,
        contract=
            contract,
    )

except MLTrainingInputError as error:

    assert (
        "age"
        in
        str(
            error
        )
    )

else:
    raise AssertionError(
        "Missing anomaly feature was accepted."
    )


print(
    "Missing feature guard: PASS"
)


# ============================================================
# IDENTIFIER FEATURE FAIL-CLOSED
# ============================================================


original_infer = (
    anomaly_input_module
    .infer_analytical_type
)


def fake_infer(
    column_name,
    series,
):

    if (
        column_name
        ==
        "amount"
    ):
        return {
            "type":
                "identifier"
        }

    return {
        "type":
            "numeric"
    }


anomaly_input_module.infer_analytical_type = (
    fake_infer
)


try:

    try:

        validate_and_extract_anomaly_x(
            dataframe=
                source_dataframe,
            contract=
                contract,
        )

    except MLTrainingInputError as error:

        assert (
            "Identifier"
            in
            str(
                error
            )
        )

        assert (
            "amount"
            in
            str(
                error
            )
        )

    else:
        raise AssertionError(
            "Identifier anomaly feature was accepted."
        )

finally:

    anomaly_input_module.infer_analytical_type = (
        original_infer
    )


print(
    "Identifier feature guard: PASS"
)


# ============================================================
# SHARED PREPROCESSING POLICY
# ============================================================


missing_value_frame = (
    source_dataframe
    .copy(
        deep=True
    )
)


missing_value_frame.loc[
    1,
    "amount",
] = np.nan


try:

    validate_and_extract_anomaly_x(
        dataframe=
            missing_value_frame,
        contract=
            contract,
    )

except MLTrainingInputError:
    pass

else:
    raise AssertionError(
        (
            "Default preprocessing policy accepted "
            "a missing numeric feature."
        )
    )


print(
    "Shared preprocessing policy: PASS"
)


# ============================================================
# HANDOFF DATASET SCOPE FAIL-CLOSED
# ============================================================


wrong_scope_contract = (
    MLAnomalyTrainingContract(
        workflow_id=
            "workflow:anomaly",
        dataset_id=
            "dataset:not-authorized",
        feature_columns=[
            "amount",
        ],
    )
)


try:

    load_authorized_ml_dataframe(
        contract=
            wrong_scope_contract,
        handoff_loader=(
            lambda *,
            workflow_id:
                handoff
        ),
        execution_label=
            "Anomaly Model Lab",
    )

except MLTrainingInputError:
    pass

else:
    raise AssertionError(
        (
            "Anomaly contract escaped the "
            "Preparation-authorized dataset scope."
        )
    )


print(
    "Preparation dataset-scope guard: PASS"
)


# ============================================================
# EMPTY / INVALID DATAFRAME
# ============================================================


for bad_frame in (
    pd.DataFrame(),
    "not-a-dataframe",
):

    try:

        validate_and_extract_anomaly_x(
            dataframe=
                bad_frame,
            contract=
                contract,
        )

    except MLTrainingInputError:
        pass

    else:
        raise AssertionError(
            (
                "Invalid anomaly input DataFrame "
                "was accepted."
            )
        )


print(
    "Input DataFrame guards: PASS"
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
    "app/ml/anomaly_training_input.py",
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
    ML_ANOMALY_TRAINING_INPUT_RULE_VERSION
    ==
    "ml_anomaly_training_input_v0.1"
)


print(
    "Anomaly input rule version: PASS"
)


print()
print(
    "PASS - DataLens Anomaly Feature-Only Training Input v0.1"
)

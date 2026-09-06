from __future__ import annotations


import numpy as np
import pandas as pd


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.preprocessing import (
    MLPreprocessingRuntimeError,
    validate_ml_feature_frame,
)


from app.profiling.types import (
    infer_analytical_type,
)


from app.preparation.analysis_input_handoff import (
    AnalysisInputHandoffError,
    load_validated_analysis_input,
)


from app.preparation.analysis_readiness_gate import (
    AnalysisReadinessError,
)


# ============================================================
# VERSION
# ============================================================


ML_TRAINING_INPUT_RULE_VERSION = (
    "ml_training_input_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLTrainingInputError(
    RuntimeError
):
    pass


# ============================================================
# AUTHORIZED PREPARATION INPUT
# ============================================================


def load_authorized_ml_dataframe(
    *,
    contract: MLTrainingContract,
    handoff_loader=load_validated_analysis_input,
    execution_label: str = "Model Lab",
) -> tuple[
    pd.DataFrame,
    int,
]:
    """
    Resolve ML input through the exact same server-owned
    Preparation -> Analysis handoff used by deterministic
    analytical execution.

    The contract dataset_id is never sufficient on its own.
    It must also be present in the handoff-authorized final
    dataset scope.
    """

    try:
        handoff = (
            handoff_loader(
                workflow_id=
                    contract.workflow_id
            )
        )

    except (
        AnalysisInputHandoffError,
        AnalysisReadinessError,
    ) as error:
        raise (
            MLTrainingInputError(
                (
                    f"{execution_label} execution refused "
                    "because Preparation did not "
                    "provide a valid READY analysis "
                    "input handoff."
                )
            )
        ) from error


    if (
        handoff.workflow_id
        !=
        contract.workflow_id
    ):
        raise (
            MLTrainingInputError(
                (
                    "Analysis input handoff workflow "
                    "does not match the ML Training "
                    "Contract."
                )
            )
        )


    if (
        contract.dataset_id
        not in
        handoff.dataset_ids
    ):
        raise (
            MLTrainingInputError(
                (
                    "ML Training Contract dataset "
                    "is outside the server-owned "
                    "validated analysis output scope. "
                    f"dataset_id={contract.dataset_id}"
                )
            )
        )


    matching_records = [
        record

        for record
        in handoff.dataset_records

        if (
            isinstance(
                record,
                dict,
            )
            and
            str(
                record.get(
                    "dataset_id",
                    "",
                )
            )
            ==
            contract.dataset_id
        )
    ]


    if (
        len(
            matching_records
        )
        !=
        1
    ):
        raise (
            MLTrainingInputError(
                (
                    "Validated analysis handoff "
                    "does not contain exactly one "
                    "record for the requested ML "
                    "dataset."
                )
            )
        )


    dataframe = (
        matching_records[
            0
        ]
        .get(
            "dataframe"
        )
    )


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise (
            MLTrainingInputError(
                (
                    "Validated analysis handoff "
                    "record does not contain a "
                    "pandas DataFrame."
                )
            )
        )


    if dataframe.empty:
        raise (
            MLTrainingInputError(
                "ML input dataset cannot be empty."
            )
        )


    return (
        dataframe.copy(
            deep=True
        ),
        int(
            handoff.session_revision
        ),
    )


# ============================================================
# TRAINING X / Y AUTHORITY
# ============================================================


def validate_and_extract_ml_xy(
    *,
    dataframe: pd.DataFrame,
    contract: MLTrainingContract,
    execution_label: str = "Model Lab",
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:

    required_columns = [
        *contract.feature_columns,
        contract.target_column,
    ]


    missing_columns = [
        column

        for column
        in required_columns

        if column
        not in
        dataframe.columns
    ]


    if missing_columns:
        raise (
            MLTrainingInputError(
                (
                    "ML input dataset is missing "
                    "required contract columns: "
                    +
                    ", ".join(
                        missing_columns
                    )
                )
            )
        )


    selected = (
        dataframe.loc[
            :,
            required_columns,
        ]
        .copy(
            deep=True
        )
    )


    x = (
        selected.loc[
            :,
            contract.feature_columns,
        ]
        .copy(
            deep=True
        )
    )


    y = (
        selected.loc[
            :,
            contract.target_column,
        ]
        .copy(
            deep=True
        )
    )


    # ========================================================
    # IDENTIFIER ROLE GUARD
    # ========================================================


    target_semantics = (
        infer_analytical_type(
            contract.target_column,
            y,
        )
    )


    if (
        target_semantics.get(
            "type"
        )
        ==
        "identifier"
    ):
        raise (
            MLTrainingInputError(
                (
                    "Identifier columns cannot be "
                    "used as ML targets. "
                    f"target={contract.target_column}"
                )
            )
        )


    identifier_features: list[
        str
    ] = []


    for feature_column in (
        contract.feature_columns
    ):

        feature_semantics = (
            infer_analytical_type(
                feature_column,
                x[
                    feature_column
                ],
            )
        )

        if (
            feature_semantics.get(
                "type"
            )
            ==
            "identifier"
        ):
            identifier_features.append(
                feature_column
            )


    if identifier_features:

        raise (
            MLTrainingInputError(
                (
                    "Identifier columns cannot be "
                    "used as ML features: "
                    +
                    ", ".join(
                        identifier_features
                    )
                )
            )
        )


    # ========================================================
    # TARGET MISSING VALUES
    # ========================================================


    if bool(
        y.isna().any()
    ):
        raise (
            MLTrainingInputError(
                (
                    "ML target contains missing values. "
                    "Target imputation is never performed "
                    f"by {execution_label}."
                )
            )
        )


    # ========================================================
    # FEATURE STRUCTURE / PREPROCESSING POLICY
    # ========================================================


    try:
        x = (
            validate_ml_feature_frame(
                features=x,
                contract=contract,
            )
        )

    except MLPreprocessingRuntimeError as error:
        raise (
            MLTrainingInputError(
                str(
                    error
                )
            )
        ) from error


    # ========================================================
    # REGRESSION TARGET
    # ========================================================


    if (
        contract.problem_type
        ==
        "regression"
    ):
        target_dtype = (
            y.dtype
        )


        if (
            pd.api.types
            .is_bool_dtype(
                target_dtype
            )
            or
            not pd.api.types
            .is_numeric_dtype(
                target_dtype
            )
        ):
            raise (
                MLTrainingInputError(
                    (
                        "Regression target must be "
                        "numeric and non-boolean. "
                        f"target={contract.target_column}, "
                        f"dtype={target_dtype}"
                    )
                )
            )


        try:
            numeric_y = (
                y.to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            )

        except Exception as error:
            raise (
                MLTrainingInputError(
                    (
                        "Regression target could "
                        "not be converted to "
                        "floating-point values."
                    )
                )
            ) from error


        if not (
            np.isfinite(
                numeric_y
            )
            .all()
        ):
            raise (
                MLTrainingInputError(
                    (
                        "Regression target contains "
                        "non-finite values."
                    )
                )
            )


        if (
            int(
                y.nunique(
                    dropna=False
                )
            )
            <
            2
        ):
            raise (
                MLTrainingInputError(
                    (
                        "Regression target must "
                        "contain at least two "
                        "distinct values."
                    )
                )
            )


    # ========================================================
    # CLASSIFICATION TARGET
    # ========================================================


    else:
        class_count = int(
            y.nunique(
                dropna=False
            )
        )


        if (
            class_count
            <
            2
        ):
            raise (
                MLTrainingInputError(
                    (
                        "Classification target must "
                        "contain at least two classes."
                    )
                )
            )


        if (
            pd.api.types
            .is_numeric_dtype(
                y.dtype
            )
            and
            not pd.api.types
            .is_bool_dtype(
                y.dtype
            )
        ):
            try:
                numeric_y = (
                    y.to_numpy(
                        dtype=np.float64,
                        copy=True,
                    )
                )

            except Exception as error:
                raise (
                    MLTrainingInputError(
                        (
                            "Numeric classification "
                            "target could not be "
                            "validated."
                        )
                    )
                ) from error


            if not (
                np.isfinite(
                    numeric_y
                )
                .all()
            ):
                raise (
                    MLTrainingInputError(
                        (
                            "Classification target "
                            "contains non-finite "
                            "numeric values."
                        )
                    )
                )


    return (
        x,
        y,
    )

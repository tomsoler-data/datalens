from __future__ import annotations


import pandas as pd


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.preprocessing import (
    MLPreprocessingRuntimeError,
    validate_ml_feature_frame,
)


from app.ml.training_input import (
    MLTrainingInputError,
)


from app.profiling.types import (
    infer_analytical_type,
)


# ============================================================
# VERSION
# ============================================================


ML_ANOMALY_TRAINING_INPUT_RULE_VERSION = (
    "ml_anomaly_training_input_v0.1"
)


# ============================================================
# FEATURE-ONLY INPUT AUTHORITY
# ============================================================


def validate_and_extract_anomaly_x(
    *,
    dataframe: pd.DataFrame,
    contract: MLAnomalyTrainingContract,
    execution_label: str = "Anomaly Model Lab",
) -> pd.DataFrame:
    """
    Validate and extract the feature-only input population
    for unsupervised anomaly detection.

    This deliberately has no target authority.

    Preparation handoff ownership remains in
    load_authorized_ml_dataframe(). This function begins only
    after the server-owned DataFrame has been resolved.

    The same Model Lab feature preprocessing policy used by
    supervised training is reused here.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise (
            MLTrainingInputError(
                (
                    f"{execution_label} input must be "
                    "a pandas DataFrame."
                )
            )
        )


    if dataframe.empty:
        raise (
            MLTrainingInputError(
                (
                    f"{execution_label} input dataset "
                    "cannot be empty."
                )
            )
        )


    required_columns = list(
        contract.feature_columns
    )


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
                    "ML anomaly input dataset is missing "
                    "required contract feature columns: "
                    +
                    ", ".join(
                        missing_columns
                    )
                )
            )
        )


    x = (
        dataframe.loc[
            :,
            required_columns,
        ]
        .copy(
            deep=True
        )
    )


    # ========================================================
    # IDENTIFIER ROLE GUARD
    # ========================================================


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
                    "Identifier columns cannot be used "
                    "as ML anomaly features: "
                    +
                    ", ".join(
                        identifier_features
                    )
                )
            )
        )


    # ========================================================
    # SHARED FEATURE STRUCTURE / PREPROCESSING POLICY
    # ========================================================


    try:

        validated_x = (
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


    return validated_x

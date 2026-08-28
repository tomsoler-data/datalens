from __future__ import annotations


from typing import (
    Union,
)


import numpy as np
import pandas as pd


from sklearn.compose import (
    ColumnTransformer,
)


from sklearn.impute import (
    SimpleImputer,
)


from sklearn.pipeline import (
    Pipeline,
)


from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


from app.ml.contracts import (
    MLTrainingContract,
)


# ============================================================
# VERSION
# ============================================================


ML_PREPROCESSING_RUNTIME_RULE_VERSION = (
    "ml_preprocessing_runtime_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLPreprocessingRuntimeError(
    RuntimeError
):
    pass


# ============================================================
# CATEGORY TYPE FAMILY
# ============================================================


def _categorical_value_family(
    value: object,
) -> str:
    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return "boolean"


    if isinstance(
        value,
        str,
    ):
        return "text"


    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        return "numeric"


    return (
        type(
            value
        )
        .__name__
    )


# ============================================================
# NUMERIC VALIDATION
# ============================================================


def _validate_numeric_features(
    *,
    features: pd.DataFrame,
    contract: MLTrainingContract,
) -> None:
    for column in (
        contract.numeric_feature_columns
    ):
        series = (
            features[
                column
            ]
        )


        dtype = (
            series.dtype
        )


        if (
            pd.api.types
            .is_bool_dtype(
                dtype
            )
        ):
            raise (
                MLPreprocessingRuntimeError(
                    (
                        "Boolean feature is declared "
                        "as numeric. "
                        f"column={column}"
                    )
                )
            )


        if not (
            pd.api.types
            .is_numeric_dtype(
                dtype
            )
        ):
            raise (
                MLPreprocessingRuntimeError(
                    (
                        "Feature declared as numeric "
                        "does not have a numeric dtype. "
                        f"column={column}, "
                        f"dtype={dtype}"
                    )
                )
            )


        missing_mask = (
            series.isna()
        )


        if (
            bool(
                missing_mask.any()
            )
            and
            contract
            .preprocessing
            .numeric_imputation
            ==
            "error"
        ):
            raise (
                MLPreprocessingRuntimeError(
                    (
                        "Missing numeric feature values "
                        "are forbidden by the ML "
                        "Preprocessing Contract. "
                        f"column={column}"
                    )
                )
            )


        observed = (
            series[
                ~missing_mask
            ]
        )


        if observed.empty:
            raise (
                MLPreprocessingRuntimeError(
                    (
                        "Numeric feature contains no "
                        "observed value from which the "
                        "training pipeline could learn. "
                        f"column={column}"
                    )
                )
            )


        try:
            numeric_values = (
                observed
                .to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            )

        except Exception as error:
            raise (
                MLPreprocessingRuntimeError(
                    (
                        "Numeric feature could not be "
                        "converted to finite floating-"
                        "point values. "
                        f"column={column}"
                    )
                )
            ) from error


        if not (
            np.isfinite(
                numeric_values
            )
            .all()
        ):
            raise (
                MLPreprocessingRuntimeError(
                    (
                        "Numeric feature contains "
                        "non-finite values. "
                        f"column={column}"
                    )
                )
            )


# ============================================================
# CATEGORICAL VALIDATION
# ============================================================


def _validate_categorical_features(
    *,
    features: pd.DataFrame,
    contract: MLTrainingContract,
) -> None:
    for column in (
        contract
        .categorical_feature_columns
    ):
        series = (
            features[
                column
            ]
        )


        dtype = (
            series.dtype
        )


        if (
            pd.api.types
            .is_datetime64_any_dtype(
                dtype
            )
            or
            pd.api.types
            .is_timedelta64_dtype(
                dtype
            )
        ):
            raise (
                MLPreprocessingRuntimeError(
                    (
                        "Datetime/timedelta feature "
                        "cannot be used directly as a "
                        "categorical Classical ML "
                        "feature. Transform it during "
                        "Preparation first. "
                        f"column={column}"
                    )
                )
            )


        missing_mask = (
            series.isna()
        )


        if (
            bool(
                missing_mask.any()
            )
            and
            contract
            .preprocessing
            .categorical_imputation
            ==
            "error"
        ):
            raise (
                MLPreprocessingRuntimeError(
                    (
                        "Missing categorical feature "
                        "values are forbidden by the ML "
                        "Preprocessing Contract. "
                        f"column={column}"
                    )
                )
            )


        observed = (
            series[
                ~missing_mask
            ]
        )


        if observed.empty:
            raise (
                MLPreprocessingRuntimeError(
                    (
                        "Categorical feature contains "
                        "no observed category from which "
                        "the training pipeline could "
                        "learn. "
                        f"column={column}"
                    )
                )
            )


        families = {
            _categorical_value_family(
                value
            )

            for value
            in observed.tolist()
        }


        if (
            len(
                families
            )
            !=
            1
        ):
            raise (
                MLPreprocessingRuntimeError(
                    (
                        "Categorical feature mixes "
                        "incompatible Python value "
                        "families. "
                        f"column={column}, "
                        "families="
                        f"{sorted(families)!r}"
                    )
                )
            )


        family = (
            next(
                iter(
                    families
                )
            )
        )


        if (
            family
            ==
            "numeric"
        ):
            try:
                numeric_values = (
                    observed
                    .to_numpy(
                        dtype=np.float64,
                        copy=True,
                    )
                )

            except Exception as error:
                raise (
                    MLPreprocessingRuntimeError(
                        (
                            "Numeric-coded categorical "
                            "feature could not be "
                            "validated. "
                            f"column={column}"
                        )
                    )
                ) from error


            if not (
                np.isfinite(
                    numeric_values
                )
                .all()
            ):
                raise (
                    MLPreprocessingRuntimeError(
                        (
                            "Numeric-coded categorical "
                            "feature contains non-finite "
                            "values. "
                            f"column={column}"
                        )
                    )
                )


# ============================================================
# FEATURE FRAME VALIDATION
# ============================================================


def validate_ml_feature_frame(
    *,
    features: pd.DataFrame,
    contract: MLTrainingContract,
) -> pd.DataFrame:
    """
    Validate structural feature compatibility before splitting.

    No learned preprocessing statistic is calculated here.

    Median, mode, scaling statistics and one-hot vocabularies are
    deliberately left to the scikit-learn Pipeline, which is fit
    only after the deterministic train/test split.
    """

    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise (
            MLPreprocessingRuntimeError(
                "ML features must be a pandas DataFrame."
            )
        )


    if features.empty:
        raise (
            MLPreprocessingRuntimeError(
                "ML feature frame cannot be empty."
            )
        )


    expected_columns = (
        list(
            contract.feature_columns
        )
    )


    if (
        list(
            features.columns
        )
        !=
        expected_columns
    ):
        raise (
            MLPreprocessingRuntimeError(
                (
                    "ML feature frame columns do not "
                    "match the ordered Training "
                    "Contract feature surface."
                )
            )
        )


    _validate_numeric_features(
        features=features,
        contract=contract,
    )


    _validate_categorical_features(
        features=features,
        contract=contract,
    )


    return (
        features.copy(
            deep=True
        )
    )


# ============================================================
# NUMERIC TRANSFORMER
# ============================================================


def _build_numeric_transformer(
    *,
    contract: MLTrainingContract,
) -> Union[
    Pipeline,
    str,
]:
    steps = []


    if (
        contract
        .preprocessing
        .numeric_imputation
        ==
        "median"
    ):
        steps.append(
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            )
        )


    if (
        contract
        .preprocessing
        .scale_numeric
    ):
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )


    if not steps:
        return "passthrough"


    return (
        Pipeline(
            steps=steps
        )
    )


# ============================================================
# CATEGORICAL TRANSFORMER
# ============================================================


def _build_categorical_transformer(
    *,
    contract: MLTrainingContract,
) -> Pipeline:
    steps = []


    if (
        contract
        .preprocessing
        .categorical_imputation
        ==
        "most_frequent"
    ):
        steps.append(
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            )
        )


    steps.append(
        (
            "encoder",
            OneHotEncoder(
                handle_unknown=(
                    contract
                    .preprocessing
                    .handle_unknown_categories
                ),
                sparse_output=False,
                dtype=np.float64,
            ),
        )
    )


    return (
        Pipeline(
            steps=steps
        )
    )


# ============================================================
# COLUMN TRANSFORMER
# ============================================================


def build_ml_preprocessor(
    *,
    contract: MLTrainingContract,
) -> ColumnTransformer:
    """
    Build the unfitted preprocessing graph.

    The returned transformer contains no learned state.

    It becomes leakage-safe because Classical ML places it inside
    the estimator Pipeline and calls Pipeline.fit() only on
    x_train after the deterministic holdout split.
    """

    transformers = []


    numeric_columns = (
        list(
            contract
            .numeric_feature_columns
        )
    )


    categorical_columns = (
        list(
            contract
            .categorical_feature_columns
        )
    )


    if numeric_columns:
        transformers.append(
            (
                "numeric",
                _build_numeric_transformer(
                    contract=contract
                ),
                numeric_columns,
            )
        )


    if categorical_columns:
        transformers.append(
            (
                "categorical",
                _build_categorical_transformer(
                    contract=contract
                ),
                categorical_columns,
            )
        )


    if not transformers:
        raise (
            MLPreprocessingRuntimeError(
                (
                    "ML Preprocessing Contract "
                    "produced no feature transformer."
                )
            )
        )


    return (
        ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=True,
        )
    )
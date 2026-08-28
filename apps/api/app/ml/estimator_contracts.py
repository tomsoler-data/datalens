from __future__ import annotations


from typing import (
    Annotated,
    Literal,
    Union,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ============================================================
# VERSION
# ============================================================


ML_ESTIMATOR_CONTRACT_RULE_VERSION = (
    "ml_estimator_contract_v0.1"
)


# ============================================================
# SUPPORTED ESTIMATORS
# ============================================================


SUPPORTED_ESTIMATOR_KEYS = (
    "linear_regression",
    "ridge_regression",
    "logistic_regression",
    "random_forest_regressor",
    "random_forest_classifier",
)


MLEstimatorProblemType = Literal[
    "regression",
    "classification",
]


# ============================================================
# LINEAR REGRESSION
# ============================================================


class MLLinearRegressionHyperparameters(
    BaseModel
):
    """
    Server-validatable LinearRegression hyperparameters.

    No arbitrary sklearn kwargs are accepted.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    kind: Literal[
        "linear_regression"
    ] = "linear_regression"


    fit_intercept: bool = True


    rule_version: Literal[
        "ml_estimator_contract_v0.1"
    ] = ML_ESTIMATOR_CONTRACT_RULE_VERSION


# ============================================================
# RIDGE REGRESSION
# ============================================================


class MLRidgeRegressionHyperparameters(
    BaseModel
):
    """
    Bounded Ridge regression hyperparameters.

    alpha is deliberately constrained to a finite operational
    range instead of exposing arbitrary sklearn configuration.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    kind: Literal[
        "ridge_regression"
    ] = "ridge_regression"


    alpha: float = Field(
        default=1.0,
        gt=0.0,
        le=1_000_000.0,
    )


    fit_intercept: bool = True


    rule_version: Literal[
        "ml_estimator_contract_v0.1"
    ] = ML_ESTIMATOR_CONTRACT_RULE_VERSION


# ============================================================
# LOGISTIC REGRESSION
# ============================================================


MLLogisticClassWeight = Literal[
    "balanced",
]


class MLLogisticRegressionHyperparameters(
    BaseModel
):
    """
    Controlled LogisticRegression configuration.

    DataLens deliberately does not expose:
    - solver;
    - penalty;
    - random_state;
    - n_jobs.

    Those execution choices remain server-owned.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    kind: Literal[
        "logistic_regression"
    ] = "logistic_regression"


    inverse_regularization_strength: float = Field(
        default=1.0,
        gt=0.0,
        le=1_000_000.0,
    )


    fit_intercept: bool = True


    max_iter: int = Field(
        default=1000,
        ge=100,
        le=10_000,
    )


    class_weight: (
        MLLogisticClassWeight
        |
        None
    ) = None


    rule_version: Literal[
        "ml_estimator_contract_v0.1"
    ] = ML_ESTIMATOR_CONTRACT_RULE_VERSION


# ============================================================
# RANDOM FOREST — SHARED TYPES
# ============================================================


MLRandomForestMaxFeatures = Literal[
    "sqrt",
    "log2",
]


MLRandomForestClassWeight = Literal[
    "balanced",
    "balanced_subsample",
]


# ============================================================
# RANDOM FOREST REGRESSOR
# ============================================================


class MLRandomForestRegressorHyperparameters(
    BaseModel
):
    """
    Controlled RandomForestRegressor configuration.

    random_state and parallel execution are intentionally absent.

    DataLens will derive determinism from the server-owned
    ML split contract when executor support is introduced.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    kind: Literal[
        "random_forest_regressor"
    ] = "random_forest_regressor"


    n_estimators: int = Field(
        default=200,
        ge=10,
        le=2000,
    )


    max_depth: (
        int
        |
        None
    ) = Field(
        default=None,
        ge=1,
        le=100,
    )


    min_samples_split: int = Field(
        default=2,
        ge=2,
        le=1000,
    )


    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        le=1000,
    )


    max_features: (
        MLRandomForestMaxFeatures
        |
        None
    ) = "sqrt"


    bootstrap: bool = True


    rule_version: Literal[
        "ml_estimator_contract_v0.1"
    ] = ML_ESTIMATOR_CONTRACT_RULE_VERSION


# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================


class MLRandomForestClassifierHyperparameters(
    BaseModel
):
    """
    Controlled RandomForestClassifier configuration.

    random_state and n_jobs remain server-owned and therefore
    cannot be provided through this contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    kind: Literal[
        "random_forest_classifier"
    ] = "random_forest_classifier"


    n_estimators: int = Field(
        default=200,
        ge=10,
        le=2000,
    )


    max_depth: (
        int
        |
        None
    ) = Field(
        default=None,
        ge=1,
        le=100,
    )


    min_samples_split: int = Field(
        default=2,
        ge=2,
        le=1000,
    )


    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        le=1000,
    )


    max_features: (
        MLRandomForestMaxFeatures
        |
        None
    ) = "sqrt"


    bootstrap: bool = True


    class_weight: (
        MLRandomForestClassWeight
        |
        None
    ) = None


    rule_version: Literal[
        "ml_estimator_contract_v0.1"
    ] = ML_ESTIMATOR_CONTRACT_RULE_VERSION


# ============================================================
# DISCRIMINATED UNION
# ============================================================


MLEstimatorHyperparameters = Annotated[
    Union[
        MLLinearRegressionHyperparameters,
        MLRidgeRegressionHyperparameters,
        MLLogisticRegressionHyperparameters,
        MLRandomForestRegressorHyperparameters,
        MLRandomForestClassifierHyperparameters,
    ],
    Field(
        discriminator="kind"
    ),
]


# ============================================================
# ESTIMATOR PROBLEM TYPE
# ============================================================


def estimator_problem_type(
    estimator_key: str,
) -> (
    MLEstimatorProblemType
    |
    None
):
    normalized = str(
        estimator_key
        if estimator_key is not None
        else ""
    ).strip()


    if normalized in {
        "linear_regression",
        "ridge_regression",
        "random_forest_regressor",
    }:
        return "regression"


    if normalized in {
        "logistic_regression",
        "random_forest_classifier",
    }:
        return "classification"


    return None


# ============================================================
# DEFAULT HYPERPARAMETERS
# ============================================================


def default_estimator_hyperparameters(
    estimator_key: str,
) -> (
    MLEstimatorHyperparameters
    |
    None
):
    """
    Resolve DataLens-owned deterministic defaults.

    Unknown estimator keys deliberately return None instead of
    inventing a configuration.

    The Classical ML executor remains responsible for rejecting
    unsupported estimators until its next contract version is
    implemented.
    """

    normalized = str(
        estimator_key
        if estimator_key is not None
        else ""
    ).strip()


    if (
        normalized
        ==
        "linear_regression"
    ):
        return (
            MLLinearRegressionHyperparameters()
        )


    if (
        normalized
        ==
        "ridge_regression"
    ):
        return (
            MLRidgeRegressionHyperparameters()
        )


    if (
        normalized
        ==
        "logistic_regression"
    ):
        return (
            MLLogisticRegressionHyperparameters()
        )


    if (
        normalized
        ==
        "random_forest_regressor"
    ):
        return (
            MLRandomForestRegressorHyperparameters()
        )


    if (
        normalized
        ==
        "random_forest_classifier"
    ):
        return (
            MLRandomForestClassifierHyperparameters()
        )


    return None
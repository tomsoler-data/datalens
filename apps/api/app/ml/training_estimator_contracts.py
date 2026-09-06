from __future__ import annotations


from typing import (
    Annotated,
    Literal,
    Union,
)


from pydantic import (
    Field,
)


from app.deep_learning.tabular_contracts import (
    DLTabularMLPRegressorHyperparameters,
)


from app.ml.estimator_contracts import (
    MLLinearRegressionHyperparameters,
    MLLogisticRegressionHyperparameters,
    MLRandomForestClassifierHyperparameters,
    MLRandomForestRegressorHyperparameters,
    MLRidgeRegressionHyperparameters,
    default_estimator_hyperparameters,
    estimator_problem_type,
)


# ============================================================
# VERSION
# ============================================================


ML_TRAINING_ESTIMATOR_BRIDGE_RULE_VERSION = (
    "ml_training_estimator_bridge_v0.1"
)


# ============================================================
# FRAMEWORK-NEUTRAL TRAINING ESTIMATOR UNION
# ============================================================


MLTrainingEstimatorHyperparameters = Annotated[
    Union[
        MLLinearRegressionHyperparameters,
        MLRidgeRegressionHyperparameters,
        MLLogisticRegressionHyperparameters,
        MLRandomForestRegressorHyperparameters,
        MLRandomForestClassifierHyperparameters,
        DLTabularMLPRegressorHyperparameters,
    ],
    Field(
        discriminator="kind"
    ),
]


# ============================================================
# PROBLEM TYPE AUTHORITY
# ============================================================


def training_estimator_problem_type(
    estimator_key: str,
) -> (
    Literal[
        "regression",
        "classification",
    ]
    |
    None
):

    normalized = str(
        estimator_key
        if estimator_key is not None
        else ""
    ).strip()


    if (
        normalized
        ==
        "tabular_mlp_regressor"
    ):
        return "regression"


    return (
        estimator_problem_type(
            normalized
        )
    )


# ============================================================
# DEFAULT CONFIGURATION RESOLUTION
# ============================================================


def default_training_estimator_hyperparameters(
    estimator_key: str,
) -> (
    MLTrainingEstimatorHyperparameters
    |
    None
):

    normalized = str(
        estimator_key
        if estimator_key is not None
        else ""
    ).strip()


    if (
        normalized
        ==
        "tabular_mlp_regressor"
    ):
        return (
            DLTabularMLPRegressorHyperparameters()
        )


    return (
        default_estimator_hyperparameters(
            normalized
        )
    )

from __future__ import annotations


from typing import (
    Union,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.time_series_model_training_contracts import (
    MLTimeSeriesModelTrainingContract,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_TRAINING_CONTRACT_FAMILY_RULE_VERSION = (
    "ml_model_training_contract_family_v0.1"
)


# ============================================================
# CONTRACT FAMILY
# ============================================================


MLModelTrainingContract = Union[
    MLTrainingContract,
    MLAnomalyTrainingContract,
    MLTimeSeriesModelTrainingContract,
]


def validate_ml_model_training_contract(
    value: object,
) -> MLModelTrainingContract:
    """
    Validate exactly one supported Model Lab training-contract
    family.

    This authority is deliberately narrower than accepting an
    arbitrary Pydantic model.

    Supported problem families:

    - supervised regression/classification;
    - target-free anomaly detection;
    - one-step univariate time-series forecasting.
    """

    if isinstance(
        value,
        MLTrainingContract,
    ):

        return (
            MLTrainingContract
            .model_validate(
                value
            )
        )


    if isinstance(
        value,
        MLAnomalyTrainingContract,
    ):

        return (
            MLAnomalyTrainingContract
            .model_validate(
                value
            )
        )


    if isinstance(
        value,
        MLTimeSeriesModelTrainingContract,
    ):

        return (
            MLTimeSeriesModelTrainingContract
            .model_validate(
                value
            )
        )


    if not isinstance(
        value,
        dict,
    ):

        raise ValueError(
            (
                "Model training contract must be "
                "a supported validated contract "
                "or JSON-style object."
            )
        )


    task_contract = value.get(
        "task_contract"
    )


    if isinstance(
        task_contract,
        dict,
    ):

        task_family = str(
            task_contract.get(
                "task_family",
                "",
            )
        ).strip()


        if (
            task_family
            ==
            "time_series_forecasting"
        ):

            return (
                MLTimeSeriesModelTrainingContract
                .model_validate(
                    value
                )
            )


    problem_type = str(
        value.get(
            "problem_type",
            "",
        )
    ).strip()


    if (
        problem_type
        ==
        "anomaly_detection"
    ):

        return (
            MLAnomalyTrainingContract
            .model_validate(
                value
            )
        )


    if problem_type in (
        "regression",
        "classification",
    ):

        return (
            MLTrainingContract
            .model_validate(
                value
            )
        )


    raise ValueError(
        (
            "Unsupported Model Lab training "
            "contract family. "
            f"problem_type={problem_type!r}"
        )
    )

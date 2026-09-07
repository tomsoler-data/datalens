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
    - target-free anomaly detection.
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
            "contract problem_type. "
            f"problem_type={problem_type!r}"
        )
    )

from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


from app.deep_learning.autoencoder_contracts import (
    DLTabularAutoencoderHyperparameters,
)


from app.ml.contracts import (
    MLPreprocessingContract,
    MLSplitContract,
    MLTrainingSplitContract,
)


# ============================================================
# VERSION
# ============================================================


ML_ANOMALY_TRAINING_CONTRACT_RULE_VERSION = (
    "ml_anomaly_training_contract_v0.1"
)


MLAnomalyProblemType = Literal[
    "anomaly_detection"
]


# ============================================================
# CONTRACT
# ============================================================


class MLAnomalyTrainingContract(
    BaseModel
):
    """
    Server-validatable Model Lab contract for
    unsupervised anomaly detection.

    Unlike MLTrainingContract, this contract deliberately
    has no target_column.

    Training authority is feature-only:

        X -> model -> reconstruction / anomaly score

    The supervised MLTrainingContract remains unchanged.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    dataset_id: str = Field(
        min_length=1,
    )


    problem_type: Literal[
        "anomaly_detection"
    ] = "anomaly_detection"


    feature_columns: list[str] = Field(
        min_length=1,
    )


    categorical_feature_columns: list[str] = Field(
        default_factory=list,
    )


    estimator_key: Literal[
        "tabular_autoencoder"
    ] = "tabular_autoencoder"


    estimator_hyperparameters: (
        DLTabularAutoencoderHyperparameters
    ) = Field(
        default_factory=(
            DLTabularAutoencoderHyperparameters
        ),
    )


    preprocessing: MLPreprocessingContract = Field(
        default_factory=(
            MLPreprocessingContract
        ),
    )


    split: MLTrainingSplitContract = Field(
        default_factory=(
            MLSplitContract
        ),
    )


    threshold_quantile: float = Field(
        default=0.99,
        gt=0.0,
        lt=1.0,
    )


    rule_version: Literal[
        "ml_anomaly_training_contract_v0.1"
    ] = (
        ML_ANOMALY_TRAINING_CONTRACT_RULE_VERSION
    )


    @field_validator(
        "workflow_id",
        "dataset_id",
    )
    @classmethod
    def validate_required_text(
        cls,
        value: str,
    ) -> str:

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Value cannot be blank."
            )

        return normalized


    @field_validator(
        "feature_columns",
        "categorical_feature_columns",
    )
    @classmethod
    def validate_column_names(
        cls,
        values: list[str],
    ) -> list[str]:

        normalized: list[str] = []

        for value in values:

            if not isinstance(
                value,
                str,
            ):
                raise ValueError(
                    "Column names must be strings."
                )

            candidate = value.strip()

            if not candidate:
                raise ValueError(
                    "Column names cannot be blank."
                )

            normalized.append(
                candidate
            )

        if (
            len(
                normalized
            )
            !=
            len(
                set(
                    normalized
                )
            )
        ):
            raise ValueError(
                "Column names must be unique."
            )

        return normalized


    @model_validator(
        mode="after",
    )
    def validate_feature_roles(
        self,
    ) -> "MLAnomalyTrainingContract":

        feature_set = set(
            self.feature_columns
        )

        categorical_set = set(
            self.categorical_feature_columns
        )

        if not categorical_set.issubset(
            feature_set
        ):
            raise ValueError(
                (
                    "categorical_feature_columns "
                    "must be a subset of "
                    "feature_columns."
                )
            )

        if self.split.stratify:
            raise ValueError(
                (
                    "Anomaly detection split "
                    "cannot use stratification "
                    "because the training contract "
                    "has no target."
                )
            )

        return self


    @property
    def numeric_feature_columns(
        self,
    ) -> list[str]:

        categorical_set = set(
            self.categorical_feature_columns
        )

        return [
            column
            for column in self.feature_columns
            if column not in categorical_set
        ]

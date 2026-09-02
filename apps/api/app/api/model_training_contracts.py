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


from app.ml.contracts import (
    MLTrainingContract,
)


# ============================================================
# VERSION
# ============================================================


MODEL_TRAINING_API_CONTRACT_RULE_VERSION = (
    "model_training_api_contract_v0.1"
)


MODEL_TRAINING_REQUEST_RULE_VERSION = (
    "model_training_request_v0.1"
)


# ============================================================
# TYPES
# ============================================================


ModelTrainingColumnKind = Literal[
    "numeric",
    "boolean",
    "datetime",
    "categorical",
    "other",
]


ModelTrainingAnalyticalType = Literal[
    "unknown",
    "identifier",
    "categorical",
    "temporal",
    "quantitative",
    "text",
]


# ============================================================
# HELPERS
# ============================================================


def _required_text(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip()

    if not normalized:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return normalized


# ============================================================
# COLUMN
# ============================================================


class ModelTrainingColumn(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str = Field(
        min_length=1,
    )

    kind: ModelTrainingColumnKind

    nullable: bool

    analytical_type: (
        ModelTrainingAnalyticalType
    ) = "unknown"

    analytical_subtype: (
        str
        | None
    ) = None

    ml_eligible_as_target: bool = True

    ml_eligible_as_feature: bool = True

    ml_eligible_as_group: bool = False

    ml_eligible_as_time: bool = False

    exclusion_reason: (
        str
        | None
    ) = None

    rule_version: Literal[
        "model_training_api_contract_v0.1"
    ] = MODEL_TRAINING_API_CONTRACT_RULE_VERSION

    @field_validator(
        "name",
        mode="before",
    )
    @classmethod
    def validate_name(
        cls,
        value: object,
    ) -> str:

        return _required_text(
            value,
            field_name=
                "column name",
        )


# ============================================================
# DATASET
# ============================================================


class ModelTrainingDataset(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    dataset_id: str = Field(
        min_length=1,
    )

    filename: str = Field(
        min_length=1,
    )

    row_count: int = Field(
        ge=0,
        strict=True,
    )

    column_count: int = Field(
        ge=1,
        strict=True,
    )

    columns: list[
        ModelTrainingColumn
    ] = Field(
        min_length=1,
    )

    rule_version: Literal[
        "model_training_api_contract_v0.1"
    ] = MODEL_TRAINING_API_CONTRACT_RULE_VERSION

    @field_validator(
        "dataset_id",
        "filename",
        mode="before",
    )
    @classmethod
    def validate_text(
        cls,
        value: object,
        info,
    ) -> str:

        return _required_text(
            value,
            field_name=
                info.field_name,
        )

    @model_validator(
        mode="after"
    )
    def validate_dataset(
        self,
    ) -> "ModelTrainingDataset":

        if (
            self.column_count
            !=
            len(
                self.columns
            )
        ):
            raise ValueError(
                (
                    "column_count must equal "
                    "the columns length."
                )
            )

        names = [
            column.name
            for column
            in self.columns
        ]

        if (
            len(
                names
            )
            !=
            len(
                set(
                    names
                )
            )
        ):
            raise ValueError(
                (
                    "Training context cannot "
                    "contain duplicate columns."
                )
            )

        return self


# ============================================================
# TRAINING CONTEXT
# ============================================================


class ModelTrainingContextResponse(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    workflow_id: str = Field(
        min_length=1,
    )

    preparation_session_revision: int = Field(
        ge=0,
        strict=True,
    )

    dataset_count: int = Field(
        ge=1,
        strict=True,
    )

    datasets: list[
        ModelTrainingDataset
    ] = Field(
        min_length=1,
    )

    rule_version: Literal[
        "model_training_api_contract_v0.1"
    ] = MODEL_TRAINING_API_CONTRACT_RULE_VERSION

    @field_validator(
        "workflow_id",
        mode="before",
    )
    @classmethod
    def validate_workflow_id(
        cls,
        value: object,
    ) -> str:

        return _required_text(
            value,
            field_name=
                "workflow_id",
        )

    @model_validator(
        mode="after"
    )
    def validate_context(
        self,
    ) -> "ModelTrainingContextResponse":

        if (
            self.dataset_count
            !=
            len(
                self.datasets
            )
        ):
            raise ValueError(
                (
                    "dataset_count must equal "
                    "the datasets length."
                )
            )

        dataset_ids = [
            dataset.dataset_id
            for dataset
            in self.datasets
        ]

        if (
            len(
                dataset_ids
            )
            !=
            len(
                set(
                    dataset_ids
                )
            )
        ):
            raise ValueError(
                (
                    "Training context cannot "
                    "contain duplicate dataset IDs."
                )
            )

        return self


# ============================================================
# TRAIN REQUEST
# ============================================================


class ModelTrainingRequest(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    training: MLTrainingContract

    expected_preparation_session_revision: int = Field(
        ge=0,
        strict=True,
    )

    rule_version: Literal[
        "model_training_request_v0.1"
    ] = MODEL_TRAINING_REQUEST_RULE_VERSION


# ============================================================
# ERROR
# ============================================================


class ModelTrainingAPIErrorDetail(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    error: str = Field(
        min_length=1,
    )

    message: str = Field(
        min_length=1,
    )

    workflow_id: (
        str
        |
        None
    ) = None

    retryable: bool = False

    api_version: Literal[
        "model_training_api_v0.1"
    ] = "model_training_api_v0.1"

from __future__ import annotations

import math
import re

from typing import (
    Any,
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
    MLPreprocessingContract,
    MLTrainingSplitContract,
)

from app.ml.estimator_contracts import (
    MLEstimatorHyperparameters,
)

from app.ml.model_evaluation_summary import (
    MLModelEvaluationSummaryContract,
)


# ============================================================
# VERSION
# ============================================================


MODEL_LAB_API_CONTRACT_RULE_VERSION = (
    "model_lab_api_contract_v0.1"
)


# ============================================================
# TYPES
# ============================================================


ModelLabProblemType = Literal[
    "regression",
    "classification",
]

ModelLabPredictionScalar = (
    str
    |
    int
    |
    float
    |
    bool
)


# ============================================================
# LIMITS
# ============================================================


MODEL_LAB_PREDICTION_MAX_ROWS = 100
MODEL_LAB_PREDICTION_MAX_COLUMNS = 256
MODEL_LAB_PREDICTION_MAX_CELLS = 10_000
MODEL_LAB_PREDICTION_MAX_STRING_LENGTH = 10_000


# ============================================================
# VALIDATION
# ============================================================


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)

EXPERIMENT_ID_PATTERN = re.compile(
    r"^experiment:[0-9a-f]{32}$"
)


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


def _finite_metric_dict(
    value: object,
) -> dict[
    str,
    float,
]:

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "metrics must be an object."
        )

    if not value:
        raise ValueError(
            "metrics cannot be empty."
        )

    normalized: dict[
        str,
        float,
    ] = {}

    for (
        raw_name,
        raw_value,
    ) in value.items():

        name = _required_text(
            raw_name,
            field_name=
                "metric name",
        )

        if (
            isinstance(
                raw_value,
                bool,
            )
            or
            not isinstance(
                raw_value,
                (
                    int,
                    float,
                ),
            )
        ):
            raise ValueError(
                (
                    "metric values must be "
                    "numeric and cannot be booleans."
                )
            )

        metric = float(
            raw_value
        )

        if not math.isfinite(
            metric
        ):
            raise ValueError(
                (
                    "metric values must "
                    "be finite."
                )
            )

        normalized[
            name
        ] = metric

    return normalized


def _validate_prediction_scalar(
    value: object,
) -> ModelLabPredictionScalar:

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            raise ValueError(
                (
                    "Prediction values must "
                    "be finite."
                )
            )

        return value

    if isinstance(
        value,
        str,
    ):
        if len(
            value
        ) > MODEL_LAB_PREDICTION_MAX_STRING_LENGTH:
            raise ValueError(
                (
                    "Prediction string exceeds "
                    "the Model Lab v0.1 limit."
                )
            )

        return value

    raise ValueError(
        (
            "Prediction values must be "
            "JSON scalar values."
        )
    )


# ============================================================
# MODEL CARD
# ============================================================


class ModelLabModelCard(
    BaseModel
):
    """
    Privacy-minimal model metadata safe for the Model Lab UI.

    Deliberately absent:
    - model_path;
    - model_file_bytes;
    - model_sha256;
    - serialized model bytes;
    - estimator object;
    - complete MLTrainingContract.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    model_id: str = Field(
        min_length=1,
    )

    workflow_id: str = Field(
        min_length=1,
    )

    dataset_id: str = Field(
        min_length=1,
    )

    problem_type: ModelLabProblemType

    target_column: str = Field(
        min_length=1,
    )

    estimator_key: str = Field(
        min_length=1,
    )

    feature_columns: list[
        str
    ] = Field(
        min_length=1,
    )

    categorical_feature_columns: list[
        str
    ] = Field(
        default_factory=list,
    )

    metrics: dict[
        str,
        float,
    ]

    train_rows: int = Field(
        gt=0,
        strict=True,
    )

    test_rows: int = Field(
        gt=0,
        strict=True,
    )

    created_at_utc: str = Field(
        min_length=1,
    )

    experiment_id: (
        str
        |
        None
    ) = None

    preparation_session_revision: (
        int
        |
        None
    ) = Field(
        default=None,
        ge=0,
        strict=True,
    )

    training_contract_sha256: (
        str
        |
        None
    ) = None

    has_experiment_provenance: bool

    rule_version: Literal[
        "model_lab_api_contract_v0.1"
    ] = MODEL_LAB_API_CONTRACT_RULE_VERSION

    @field_validator(
        "model_id",
        "workflow_id",
        "dataset_id",
        "target_column",
        "estimator_key",
        "created_at_utc",
        mode="before",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
        info,
    ) -> str:

        return _required_text(
            value,
            field_name=
                info.field_name,
        )

    @field_validator(
        "feature_columns",
        "categorical_feature_columns",
        mode="before",
    )
    @classmethod
    def validate_feature_columns(
        cls,
        value: object,
        info,
    ) -> list[
        str
    ]:

        if not isinstance(
            value,
            list,
        ):
            raise ValueError(
                (
                    f"{info.field_name} "
                    "must be a list."
                )
            )

        normalized = [
            _required_text(
                item,
                field_name=
                    info.field_name,
            )
            for item
            in value
        ]

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
                (
                    f"{info.field_name} "
                    "must contain unique names."
                )
            )

        return normalized

    @field_validator(
        "metrics",
        mode="before",
    )
    @classmethod
    def validate_metrics(
        cls,
        value: object,
    ) -> dict[
        str,
        float,
    ]:

        return _finite_metric_dict(
            value
        )

    @field_validator(
        "experiment_id",
        mode="before",
    )
    @classmethod
    def validate_experiment_id(
        cls,
        value: object,
    ) -> (
        str
        |
        None
    ):

        if value is None:
            return None

        normalized = str(
            value
        ).strip().lower()

        if (
            EXPERIMENT_ID_PATTERN
            .fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                (
                    "experiment_id must match "
                    "experiment:<32 lowercase hex>."
                )
            )

        return normalized

    @field_validator(
        "training_contract_sha256",
        mode="before",
    )
    @classmethod
    def validate_training_sha(
        cls,
        value: object,
    ) -> (
        str
        |
        None
    ):

        if value is None:
            return None

        normalized = str(
            value
        ).strip().lower()

        if (
            SHA256_PATTERN
            .fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                (
                    "training_contract_sha256 must "
                    "be a lowercase SHA-256 digest."
                )
            )

        return normalized

    @model_validator(
        mode="after"
    )
    def validate_provenance_surface(
        self,
    ) -> "ModelLabModelCard":

        categorical = set(
            self
            .categorical_feature_columns
        )

        features = set(
            self.feature_columns
        )

        if not categorical.issubset(
            features
        ):
            raise ValueError(
                (
                    "Categorical feature columns "
                    "must be part of feature_columns."
                )
            )

        provenance_values = (
            self.experiment_id,
            self.preparation_session_revision,
            self.training_contract_sha256,
        )

        if self.has_experiment_provenance:

            if any(
                value is None
                for value
                in provenance_values
            ):
                raise ValueError(
                    (
                        "Model card with Experiment "
                        "Provenance requires experiment_id, "
                        "Preparation revision and Training "
                        "Contract SHA-256."
                    )
                )

        else:

            if any(
                value is not None
                for value
                in provenance_values
            ):
                raise ValueError(
                    (
                        "Model card without Experiment "
                        "Provenance cannot claim "
                        "provenance identifiers."
                    )
                )

        return self


# ============================================================
# MODEL DETAIL
# ============================================================


class ModelLabModelDetail(
    ModelLabModelCard
):
    """
    Safe expanded training configuration for one model.

    This is still a projection, not the raw persisted artifact.
    """

    preprocessing: (
        MLPreprocessingContract
    )

    split: (
        MLTrainingSplitContract
    )

    effective_estimator_hyperparameters: (
        MLEstimatorHyperparameters
    )


# ============================================================
# LIST RESPONSE
# ============================================================


class ModelLabModelListResponse(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    workflow_id: str = Field(
        min_length=1,
    )

    model_count: int = Field(
        ge=0,
        strict=True,
    )

    models: list[
        ModelLabModelCard
    ]

    ordering: Literal[
        "created_at_desc_model_id_asc"
    ] = "created_at_desc_model_id_asc"

    rule_version: Literal[
        "model_lab_api_contract_v0.1"
    ] = MODEL_LAB_API_CONTRACT_RULE_VERSION

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
    def validate_list_consistency(
        self,
    ) -> "ModelLabModelListResponse":

        if (
            self.model_count
            !=
            len(
                self.models
            )
        ):
            raise ValueError(
                (
                    "model_count must equal "
                    "the number of model cards."
                )
            )

        seen = set()

        for model in self.models:

            if (
                model.workflow_id
                !=
                self.workflow_id
            ):
                raise ValueError(
                    (
                        "Every model card must belong "
                        "to the requested workflow."
                    )
                )

            if model.model_id in seen:
                raise ValueError(
                    (
                        "Model list cannot contain "
                        "duplicate model_id values."
                    )
                )

            seen.add(
                model.model_id
            )

        return self


# ============================================================
# EVALUATION REQUEST
# ============================================================


class ModelLabEvaluateRequest(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    workflow_id: str = Field(
        min_length=1,
    )

    model_id: str = Field(
        min_length=1,
    )

    evaluation: (
        MLModelEvaluationSummaryContract
    ) = Field(
        default_factory=
            MLModelEvaluationSummaryContract
    )

    rule_version: Literal[
        "model_lab_api_contract_v0.1"
    ] = MODEL_LAB_API_CONTRACT_RULE_VERSION

    @field_validator(
        "workflow_id",
        "model_id",
        mode="before",
    )
    @classmethod
    def validate_identifiers(
        cls,
        value: object,
        info,
    ) -> str:

        return _required_text(
            value,
            field_name=
                info.field_name,
        )


# ============================================================
# PREDICTION REQUEST
# ============================================================


class ModelLabPredictRequest(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    workflow_id: str = Field(
        min_length=1,
    )

    model_id: str = Field(
        min_length=1,
    )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        min_length=1,
        max_length=
            MODEL_LAB_PREDICTION_MAX_ROWS,
    )

    rule_version: Literal[
        "model_lab_api_contract_v0.1"
    ] = MODEL_LAB_API_CONTRACT_RULE_VERSION

    @field_validator(
        "workflow_id",
        "model_id",
        mode="before",
    )
    @classmethod
    def validate_identifiers(
        cls,
        value: object,
        info,
    ) -> str:

        return _required_text(
            value,
            field_name=
                info.field_name,
        )

    @field_validator(
        "rows",
        mode="before",
    )
    @classmethod
    def validate_rows(
        cls,
        value: object,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:

        if not isinstance(
            value,
            list,
        ):
            raise ValueError(
                "rows must be a list."
            )

        if not value:
            raise ValueError(
                "rows cannot be empty."
            )

        if (
            len(
                value
            )
            >
            MODEL_LAB_PREDICTION_MAX_ROWS
        ):
            raise ValueError(
                (
                    "Prediction request exceeds "
                    "the Model Lab row limit."
                )
            )

        normalized_rows = []
        cell_count = 0

        for (
            row_index,
            row,
        ) in enumerate(
            value
        ):

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    (
                        "Each prediction row "
                        "must be an object. "
                        f"row={row_index}"
                    )
                )

            if not row:
                raise ValueError(
                    (
                        "Prediction rows cannot "
                        "be empty. "
                        f"row={row_index}"
                    )
                )

            if (
                len(
                    row
                )
                >
                MODEL_LAB_PREDICTION_MAX_COLUMNS
            ):
                raise ValueError(
                    (
                        "Prediction row exceeds "
                        "the Model Lab column limit."
                    )
                )

            normalized_row = {}

            for (
                raw_key,
                raw_value,
            ) in row.items():

                if not isinstance(
                    raw_key,
                    str,
                ):
                    raise ValueError(
                        (
                            "Prediction feature names "
                            "must be strings."
                        )
                    )

                key = raw_key.strip()

                if not key:
                    raise ValueError(
                        (
                            "Prediction feature names "
                            "cannot be empty."
                        )
                    )

                if key != raw_key:
                    raise ValueError(
                        (
                            "Prediction feature names "
                            "cannot contain leading or "
                            "trailing whitespace."
                        )
                    )

                if isinstance(
                    raw_value,
                    (
                        list,
                        dict,
                        tuple,
                        set,
                    ),
                ):
                    raise ValueError(
                        (
                            "Prediction feature values "
                            "cannot be nested containers."
                        )
                    )

                if raw_value is None:
                    normalized_value = None

                elif isinstance(
                    raw_value,
                    bool,
                ):
                    normalized_value = raw_value

                elif isinstance(
                    raw_value,
                    int,
                ):
                    normalized_value = raw_value

                elif isinstance(
                    raw_value,
                    float,
                ):

                    if not math.isfinite(
                        raw_value
                    ):
                        raise ValueError(
                            (
                                "Prediction numeric "
                                "values must be finite."
                            )
                        )

                    normalized_value = raw_value

                elif isinstance(
                    raw_value,
                    str,
                ):

                    if (
                        len(
                            raw_value
                        )
                        >
                        MODEL_LAB_PREDICTION_MAX_STRING_LENGTH
                    ):
                        raise ValueError(
                            (
                                "Prediction string value "
                                "exceeds the v0.1 limit."
                            )
                        )

                    normalized_value = raw_value

                else:
                    raise ValueError(
                        (
                            "Prediction feature values "
                            "must be JSON scalar values."
                        )
                    )

                normalized_row[
                    key
                ] = normalized_value

            cell_count += len(
                normalized_row
            )

            if (
                cell_count
                >
                MODEL_LAB_PREDICTION_MAX_CELLS
            ):
                raise ValueError(
                    (
                        "Prediction request exceeds "
                        "the Model Lab cell limit."
                    )
                )

            normalized_rows.append(
                normalized_row
            )

        return normalized_rows


# ============================================================
# PREDICTION RESPONSE
# ============================================================


class ModelLabPredictResponse(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    workflow_id: str = Field(
        min_length=1,
    )

    model_id: str = Field(
        min_length=1,
    )

    problem_type: ModelLabProblemType

    target_column: str = Field(
        min_length=1,
    )

    prediction_count: int = Field(
        gt=0,
        strict=True,
    )

    predictions: list[
        ModelLabPredictionScalar
    ] = Field(
        min_length=1,
        max_length=
            MODEL_LAB_PREDICTION_MAX_ROWS,
    )

    method: Literal[
        "trusted_native_predict"
    ] = "trusted_native_predict"

    rule_version: Literal[
        "model_lab_api_contract_v0.1"
    ] = MODEL_LAB_API_CONTRACT_RULE_VERSION

    @field_validator(
        "workflow_id",
        "model_id",
        "target_column",
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

    @field_validator(
        "predictions",
        mode="before",
    )
    @classmethod
    def validate_predictions(
        cls,
        value: object,
    ) -> list[
        ModelLabPredictionScalar
    ]:

        if not isinstance(
            value,
            list,
        ):
            raise ValueError(
                "predictions must be a list."
            )

        return [
            _validate_prediction_scalar(
                item
            )
            for item
            in value
        ]

    @model_validator(
        mode="after"
    )
    def validate_prediction_count(
        self,
    ) -> "ModelLabPredictResponse":

        if (
            self.prediction_count
            !=
            len(
                self.predictions
            )
        ):
            raise ValueError(
                (
                    "prediction_count must equal "
                    "the predictions length."
                )
            )

        return self


# ============================================================
# STRUCTURED API ERROR
# ============================================================


class ModelLabAPIErrorDetail(
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

    model_id: (
        str
        |
        None
    ) = None

    retryable: bool = False

    api_version: Literal[
        "model_lab_api_v0.1"
    ] = "model_lab_api_v0.1"

    @field_validator(
        "error",
        "message",
        mode="before",
    )
    @classmethod
    def validate_required_text(
        cls,
        value: object,
        info,
    ) -> str:

        return _required_text(
            value,
            field_name=
                info.field_name,
        )

    @field_validator(
        "workflow_id",
        "model_id",
        mode="before",
    )
    @classmethod
    def validate_optional_text(
        cls,
        value: object,
        info,
    ) -> (
        str
        |
        None
    ):

        if value is None:
            return None

        return _required_text(
            value,
            field_name=
                info.field_name,
        )

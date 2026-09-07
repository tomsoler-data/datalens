from __future__ import annotations


import math
import re


from pathlib import (
    PurePosixPath,
)


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


from app.ml.model_training_contracts import (
    MLModelTrainingContract,
    validate_ml_model_training_contract,
)


from app.ml.model_artifact_formats import (
    MLModelSerializationFormat,
    ml_model_serialization_suffix,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
    ml_model_training_contract_sha256,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_ARTIFACT_RULE_VERSION = (
    "ml_model_artifact_v0.1"
)


# ============================================================
# VALIDATION
# ============================================================


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
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
            (
                f"{field_name} "
                "cannot be empty"
            )
        )


    return normalized


def _normalize_relative_model_path(
    value: object,
) -> str:
    normalized = (
        _required_text(
            value,
            field_name="model_path",
        )
        .replace(
            "\\",
            "/",
        )
    )


    path = PurePosixPath(
        normalized
    )


    if (
        path.is_absolute()
        or
        ".."
        in
        path.parts
    ):
        raise ValueError(
            (
                "model_path must remain relative "
                "to the configured Model Artifact "
                "data-plane root"
            )
        )


    if (
        not path.parts
        or
        path.name
        in {
            "",
            ".",
        }
    ):
        raise ValueError(
            "model_path is invalid"
        )


    return path.as_posix()


# ============================================================
# MODEL ARTIFACT
# ============================================================


class MLModelArtifactRecord(
    BaseModel
):
    """
    Persisted metadata for one server-owned trained
    machine-learning model.

    This object contains only metadata and provenance.

    The serialized estimator itself remains in the filesystem
    data plane and is referenced through model_path.

    DataLens must never accept an arbitrary user-supplied
    model_path as trusted executable model state.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
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


    training_contract: MLModelTrainingContract


    experiment_provenance: (
        MLExperimentProvenanceRecord
        |
        None
    ) = None


    metrics: dict[
        str,
        float,
    ]


    train_rows: int = Field(
        gt=0,
    )


    test_rows: int = Field(
        gt=0,
    )


    created_at_utc: str = Field(
        min_length=1,
    )


    serialization_format: (
        MLModelSerializationFormat
    ) = "joblib"


    model_path: str = Field(
        min_length=1,
    )


    model_file_bytes: int = Field(
        gt=0,
    )


    model_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    rule_version: Literal[
        "ml_model_artifact_v0.1"
    ] = ML_MODEL_ARTIFACT_RULE_VERSION


    # ========================================================
    # TRAINING CONTRACT FAMILY
    # ========================================================


    @field_validator(
        "training_contract",
        mode="before",
    )
    @classmethod
    def validate_training_contract_family(
        cls,
        value: object,
    ) -> MLModelTrainingContract:

        return (
            validate_ml_model_training_contract(
                value
            )
        )


    # ========================================================
    # TEXT
    # ========================================================


    @field_validator(
        "model_id",
        "workflow_id",
        "dataset_id",
        "created_at_utc",
        mode="before",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
        info,
    ) -> str:
        return (
            _required_text(
                value,
                field_name=
                    info.field_name,
            )
        )


    # ========================================================
    # METRICS
    # ========================================================


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
        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(
                "metrics must be an object"
            )


        if not value:
            raise ValueError(
                (
                    "metrics must contain at "
                    "least one metric"
                )
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
                field_name="metric name",
            )


            if name in normalized:
                raise ValueError(
                    (
                        "metrics cannot contain "
                        "duplicate normalized names: "
                        f"{name!r}"
                    )
                )


            if isinstance(
                raw_value,
                bool,
            ):
                raise ValueError(
                    (
                        "metric values must be "
                        "numeric and cannot be booleans"
                    )
                )


            try:
                metric_value = float(
                    raw_value
                )

            except Exception as error:
                raise ValueError(
                    (
                        "metric values must "
                        "be numeric"
                    )
                ) from error


            if not math.isfinite(
                metric_value
            ):
                raise ValueError(
                    (
                        "metric values must "
                        "be finite"
                    )
                )


            normalized[
                name
            ] = metric_value


        return normalized


    # ========================================================
    # MODEL PATH
    # ========================================================


    @field_validator(
        "model_path",
        mode="before",
    )
    @classmethod
    def validate_model_path(
        cls,
        value: object,
    ) -> str:
        return (
            _normalize_relative_model_path(
                value
            )
        )


    # ========================================================
    # SHA-256
    # ========================================================


    @field_validator(
        "model_sha256",
        mode="before",
    )
    @classmethod
    def validate_model_sha256(
        cls,
        value: object,
    ) -> str:
        normalized = str(
            value
            if value is not None
            else ""
        ).strip().lower()


        if (
            SHA256_PATTERN.fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                (
                    "model_sha256 must be a "
                    "64-character lowercase "
                    "hex digest"
                )
            )


        return normalized


    # ========================================================
    # SERIALIZATION FORMAT / PATH CONSISTENCY
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_serialization_format_path(
        self,
    ) -> "MLModelArtifactRecord":

        expected_suffix = (
            ml_model_serialization_suffix(
                self.serialization_format
            )
        )


        actual_suffix = (
            PurePosixPath(
                self.model_path
            )
            .suffix
            .lower()
        )


        if (
            actual_suffix
            !=
            expected_suffix
        ):
            raise ValueError(
                (
                    "Model Artifact model_path suffix "
                    "does not match serialization format. "
                    "serialization_format="
                    f"{self.serialization_format}, "
                    f"model_path={self.model_path}"
                )
            )


        return self


    # ========================================================
    # PROVENANCE CONSISTENCY
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_provenance(
        self,
    ) -> "MLModelArtifactRecord":

        if (
            self.workflow_id
            !=
            self.training_contract.workflow_id
        ):
            raise ValueError(
                (
                    "Model Artifact workflow_id "
                    "does not match the "
                    "ML Training Contract"
                )
            )


        if (
            self.dataset_id
            !=
            self.training_contract.dataset_id
        ):
            raise ValueError(
                (
                    "Model Artifact dataset_id "
                    "does not match the "
                    "ML Training Contract"
                )
            )


        provenance = (
            self.experiment_provenance
        )


        if provenance is not None:

            if (
                provenance.workflow_id
                !=
                self.workflow_id
            ):
                raise ValueError(
                    (
                        "Experiment provenance workflow_id "
                        "does not match Model Artifact."
                    )
                )


            if (
                provenance.dataset_id
                !=
                self.dataset_id
            ):
                raise ValueError(
                    (
                        "Experiment provenance dataset_id "
                        "does not match Model Artifact."
                    )
                )


            if (
                provenance.model_id
                !=
                self.model_id
            ):
                raise ValueError(
                    (
                        "Experiment provenance model_id "
                        "does not match Model Artifact."
                    )
                )


            if (
                provenance.train_rows
                !=
                self.train_rows
                or
                provenance.test_rows
                !=
                self.test_rows
            ):
                raise ValueError(
                    (
                        "Experiment provenance holdout "
                        "shape does not match Model Artifact."
                    )
                )


            if (
                provenance.metrics
                !=
                self.metrics
            ):
                raise ValueError(
                    (
                        "Experiment provenance metrics "
                        "do not match Model Artifact."
                    )
                )


            expected_contract_sha256 = (
                ml_model_training_contract_sha256(
                    self.training_contract
                )
            )


            if (
                provenance.training_contract_sha256
                !=
                expected_contract_sha256
            ):
                raise ValueError(
                    (
                        "Experiment provenance training "
                        "contract fingerprint does not "
                        "match Model Artifact contract."
                    )
                )


        return self

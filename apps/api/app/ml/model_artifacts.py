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


from app.ml.contracts import (
    MLTrainingContract,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_ARTIFACT_RULE_VERSION = (
    "ml_model_artifact_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLModelSerializationFormat = Literal[
    "joblib",
]


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


    if (
        path.suffix.lower()
        !=
        ".joblib"
    ):
        raise ValueError(
            (
                "joblib Model Artifacts must use "
                "a .joblib model_path"
            )
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


    training_contract: MLTrainingContract


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


        return self
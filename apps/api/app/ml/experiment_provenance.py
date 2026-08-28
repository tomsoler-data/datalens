from __future__ import annotations


import hashlib
import json
import math
import re


from typing import (
    Literal,
)


from uuid import (
    uuid4,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


from app.ml.contracts import (
    MLTrainingContract,
)


# ============================================================
# VERSION
# ============================================================


ML_EXPERIMENT_PROVENANCE_RULE_VERSION = (
    "ml_experiment_provenance_v0.1"
)


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
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


def _normalized_metrics(
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
            field_name=
                "metric name",
        )


        if name in normalized:
            raise ValueError(
                (
                    "metrics cannot contain "
                    "duplicate normalized names"
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
                "metric values must be numeric"
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


# ============================================================
# CANONICAL TRAINING CONTRACT
# ============================================================


def canonical_ml_training_contract_json(
    training_contract: MLTrainingContract,
) -> str:
    """
    Produce one canonical JSON representation of an
    MLTrainingContract.

    This representation is used only for deterministic
    provenance fingerprinting.

    Raw rows, predictions and learned model state are never
    included.
    """

    contract = (
        MLTrainingContract
        .model_validate(
            training_contract
        )
    )


    payload = (
        contract.model_dump(
            mode="json"
        )
    )


    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

    except Exception as error:
        raise ValueError(
            (
                "ML Training Contract could not "
                "be canonically serialized."
            )
        ) from error


def ml_training_contract_sha256(
    training_contract: MLTrainingContract,
) -> str:
    """
    Return the deterministic SHA-256 fingerprint of the exact
    validated ML Training Contract.
    """

    canonical_json = (
        canonical_ml_training_contract_json(
            training_contract
        )
    )


    return (
        hashlib.sha256(
            canonical_json.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


# ============================================================
# EXPERIMENT PROVENANCE RECORD
# ============================================================


class MLExperimentProvenanceRecord(
    BaseModel
):
    """
    Durable privacy-minimal provenance for one Classical ML
    execution that produced one server-owned Model Artifact.

    This record intentionally contains no:
    - raw rows;
    - train/test observations;
    - predictions;
    - fitted estimator bytes;
    - filesystem path;
    - secrets.

    The exact MLTrainingContract remains stored by the Model
    Artifact. training_contract_sha256 cryptographically binds
    this provenance record to that contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    experiment_id: str = Field(
        min_length=1,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    dataset_id: str = Field(
        min_length=1,
    )


    preparation_session_revision: int = Field(
        ge=0,
    )


    training_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    model_id: str = Field(
        min_length=1,
    )


    train_rows: int = Field(
        gt=0,
    )


    test_rows: int = Field(
        gt=0,
    )


    metrics: dict[
        str,
        float,
    ]


    rule_version: Literal[
        "ml_experiment_provenance_v0.1"
    ] = (
        ML_EXPERIMENT_PROVENANCE_RULE_VERSION
    )


    # ========================================================
    # TEXT
    # ========================================================


    @field_validator(
        "experiment_id",
        "workflow_id",
        "dataset_id",
        "model_id",
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
    # EXPERIMENT ID
    # ========================================================


    @field_validator(
        "experiment_id"
    )
    @classmethod
    def validate_experiment_id(
        cls,
        value: str,
    ) -> str:

        if (
            EXPERIMENT_ID_PATTERN
            .fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "experiment_id must be a "
                    "server-shaped identifier"
                )
            )


        return value


    # ========================================================
    # SHA-256
    # ========================================================


    @field_validator(
        "training_contract_sha256",
        mode="before",
    )
    @classmethod
    def validate_training_contract_sha256(
        cls,
        value: object,
    ) -> str:

        normalized = str(
            value
            if value is not None
            else ""
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
                    "be a lowercase 64-character "
                    "SHA-256 digest"
                )
            )


        return normalized


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

        return (
            _normalized_metrics(
                value
            )
        )


# ============================================================
# SERVER-OWNED BUILDER
# ============================================================


def build_ml_experiment_provenance(
    *,
    training_contract: MLTrainingContract,
    preparation_session_revision: int,
    model_id: str,
    train_rows: int,
    test_rows: int,
    metrics: dict[
        str,
        float,
    ],
) -> MLExperimentProvenanceRecord:
    """
    Build provenance for one completed server-owned ML
    execution.

    experiment_id is generated here by DataLens and cannot be
    supplied by the caller.
    """

    contract = (
        MLTrainingContract
        .model_validate(
            training_contract
        )
    )


    return (
        MLExperimentProvenanceRecord(
            experiment_id=(
                "experiment:"
                +
                uuid4().hex
            ),

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            preparation_session_revision=
                preparation_session_revision,

            training_contract_sha256=(
                ml_training_contract_sha256(
                    contract
                )
            ),

            model_id=
                model_id,

            train_rows=
                train_rows,

            test_rows=
                test_rows,

            metrics=
                metrics,

            rule_version=(
                ML_EXPERIMENT_PROVENANCE_RULE_VERSION
            ),
        )
    )

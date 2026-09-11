from __future__ import annotations


import re


from pathlib import PurePosixPath


from typing import Literal


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# VERSION
# ============================================================


MODEL_LIFECYCLE_ARTIFACT_RULE_VERSION = (
    "model_lifecycle_artifact_v0.1"
)


# ============================================================
# TYPES
# ============================================================


ModelLifecycleArtifactFamily = Literal[
    "model",
    "adapter",
]


ModelLifecycleArtifactFormat = Literal[
    "joblib",
    "pytorch_bundle",
    "peft_adapter_bundle",
]


# ============================================================
# VALIDATION
# ============================================================


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


ARTIFACT_ID_PATTERN = re.compile(
    r"^artifact:[a-z0-9][a-z0-9._-]*:[a-f0-9]{32}$"
)


SOURCE_CONTRACT_KIND_PATTERN = re.compile(
    r"^[a-z][a-z0-9._-]{0,127}$"
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


def _normalize_relative_artifact_path(
    value: object,
) -> str:

    normalized = (
        _required_text(
            value,
            field_name="artifact_path",
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
        ".." in path.parts
    ):
        raise ValueError(
            (
                "artifact_path must remain "
                "repository/store relative."
            )
        )


    if (
        not path.parts
        or
        path.name in {
            "",
            ".",
        }
    ):
        raise ValueError(
            "artifact_path is invalid."
        )


    return path.as_posix()


# ============================================================
# SHARED ARTIFACT IDENTITY
# ============================================================


class ModelLifecycleArtifactRecord(
    BaseModel
):
    """
    Neutral identity for one immutable model-lifecycle artifact.

    This record deliberately contains no domain-specific training
    fields such as Preparation revision, train/test rows, targets,
    predictions, tensors, raw examples or learned state.

    Domain-specific contracts remain authoritative outside this
    shared record and are bound through source_contract_kind and
    source_contract_sha256.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    artifact_id: str = Field(
        min_length=1,
    )


    artifact_family: (
        ModelLifecycleArtifactFamily
    )


    experiment_id: str = Field(
        min_length=1,
    )


    artifact_format: (
        ModelLifecycleArtifactFormat
    )


    artifact_path: str = Field(
        min_length=1,
    )


    artifact_file_bytes: int = Field(
        gt=0,
    )


    artifact_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    created_at_utc: str = Field(
        min_length=1,
    )


    source_contract_kind: str = Field(
        min_length=1,
        max_length=128,
    )


    source_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    rule_version: Literal[
        "model_lifecycle_artifact_v0.1"
    ] = (
        MODEL_LIFECYCLE_ARTIFACT_RULE_VERSION
    )


    # ========================================================
    # REQUIRED TEXT
    # ========================================================


    @field_validator(
        "artifact_id",
        "experiment_id",
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
            field_name=info.field_name,
        )


    # ========================================================
    # ARTIFACT ID
    # ========================================================


    @field_validator(
        "artifact_id"
    )
    @classmethod
    def validate_artifact_id(
        cls,
        value: str,
    ) -> str:

        if (
            ARTIFACT_ID_PATTERN.fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "artifact_id must use the "
                    "server-owned shape "
                    "'artifact:<family>:<32 hex>'."
                )
            )


        return value


    # ========================================================
    # ARTIFACT ID / FAMILY CONSISTENCY
    # ========================================================


    @model_validator(
        mode="after",
    )
    def validate_artifact_family_identity(
        self,
    ) -> "ModelLifecycleArtifactRecord":

        id_family = (
            self.artifact_id
            .split(
                ":",
                maxsplit=2,
            )[1]
        )


        if (
            id_family
            !=
            self.artifact_family
        ):
            raise ValueError(
                (
                    "artifact_id family must match "
                    "artifact_family."
                )
            )


        return self


    # ========================================================
    # PATH
    # ========================================================


    @field_validator(
        "artifact_path",
        mode="before",
    )
    @classmethod
    def validate_artifact_path(
        cls,
        value: object,
    ) -> str:

        return (
            _normalize_relative_artifact_path(
                value
            )
        )


    # ========================================================
    # SOURCE CONTRACT KIND
    # ========================================================


    @field_validator(
        "source_contract_kind",
        mode="before",
    )
    @classmethod
    def validate_source_contract_kind(
        cls,
        value: object,
    ) -> str:

        normalized = (
            _required_text(
                value,
                field_name=
                    "source_contract_kind",
            )
            .lower()
        )


        if (
            SOURCE_CONTRACT_KIND_PATTERN
            .fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                (
                    "source_contract_kind must "
                    "be a stable lowercase "
                    "contract identifier."
                )
            )


        return normalized


    # ========================================================
    # SHA256
    # ========================================================


    @field_validator(
        "artifact_sha256",
        "source_contract_sha256",
        mode="before",
    )
    @classmethod
    def validate_sha256(
        cls,
        value: object,
        info,
    ) -> str:

        normalized = (
            _required_text(
                value,
                field_name=
                    info.field_name,
            )
            .lower()
        )


        if (
            SHA256_PATTERN.fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                (
                    f"{info.field_name} must "
                    "be a lowercase 64-character "
                    "SHA-256 digest."
                )
            )


        return normalized

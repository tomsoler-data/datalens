from __future__ import annotations


import re


from typing import Literal


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


from app.model_lifecycle.artifact_contracts import (
    ARTIFACT_ID_PATTERN,
    SHA256_PATTERN,
    SOURCE_CONTRACT_KIND_PATTERN,
    ModelLifecycleArtifactFamily,
)


# ============================================================
# VERSION
# ============================================================


MODEL_LIFECYCLE_PROVENANCE_RULE_VERSION = (
    "model_lifecycle_provenance_v0.1"
)


# ============================================================
# TYPES
# ============================================================


ModelLifecycleProducerFamily = Literal[
    "classical_ml",
    "deep_learning",
    "llm_adaptation",
]


PROVENANCE_ID_PATTERN = re.compile(
    (
        r"^provenance:"
        r"[a-z0-9][a-z0-9._-]*:"
        r"[a-f0-9]{32}$"
    )
)


EVIDENCE_KIND_PATTERN = re.compile(
    r"^[a-z][a-z0-9._-]{0,127}$"
)


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


def _normalize_sha256(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = (
        _required_text(
            value,
            field_name=field_name,
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
                f"{field_name} must be a "
                "64-character SHA-256 digest."
            )
        )


    return normalized


# ============================================================
# EVIDENCE BINDING
# ============================================================


class ModelLifecycleEvidenceBinding(
    BaseModel
):
    """
    One immutable binding to external evidence supporting the
    provenance of a model-lifecycle artifact.

    Examples include a frozen source contract, training manifest,
    training receipt or training dataset fingerprint.

    The shared lifecycle does not interpret the evidence payload.
    It only binds its stable identity to a SHA-256 authority.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    evidence_kind: str = Field(
        min_length=1,
        max_length=128,
    )


    authority_id: str = Field(
        min_length=1,
    )


    authority_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    @field_validator(
        "evidence_kind",
        mode="before",
    )
    @classmethod
    def validate_evidence_kind(
        cls,
        value: object,
    ) -> str:

        normalized = (
            _required_text(
                value,
                field_name="evidence_kind",
            )
            .lower()
        )


        if (
            EVIDENCE_KIND_PATTERN.fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                (
                    "evidence_kind must be a "
                    "stable lowercase identifier."
                )
            )


        return normalized


    @field_validator(
        "authority_id",
        mode="before",
    )
    @classmethod
    def validate_authority_id(
        cls,
        value: object,
    ) -> str:

        return _required_text(
            value,
            field_name="authority_id",
        )


    @field_validator(
        "authority_sha256",
        mode="before",
    )
    @classmethod
    def validate_authority_sha256(
        cls,
        value: object,
    ) -> str:

        return _normalize_sha256(
            value,
            field_name="authority_sha256",
        )


# ============================================================
# SHARED PROVENANCE RECORD
# ============================================================


class ModelLifecycleProvenanceRecord(
    BaseModel
):
    """
    Neutral provenance for one immutable lifecycle artifact.

    Family-specific training semantics remain in their original
    source contracts. This record binds the resulting artifact,
    its execution identity and the immutable evidence proving how
    it was produced.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    provenance_id: str = Field(
        min_length=1,
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


    producer_family: (
        ModelLifecycleProducerFamily
    )


    execution_id: str = Field(
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


    evidence: tuple[
        ModelLifecycleEvidenceBinding,
        ...,
    ] = Field(
        min_length=1,
    )


    created_at_utc: str = Field(
        min_length=1,
    )


    rule_version: Literal[
        "model_lifecycle_provenance_v0.1"
    ] = (
        MODEL_LIFECYCLE_PROVENANCE_RULE_VERSION
    )


    # ========================================================
    # TEXT
    # ========================================================


    @field_validator(
        "experiment_id",
        "execution_id",
        "created_at_utc",
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
            field_name=info.field_name,
        )


    # ========================================================
    # PROVENANCE ID
    # ========================================================


    @field_validator(
        "provenance_id",
        mode="before",
    )
    @classmethod
    def validate_provenance_id(
        cls,
        value: object,
    ) -> str:

        normalized = (
            _required_text(
                value,
                field_name="provenance_id",
            )
            .lower()
        )


        if (
            PROVENANCE_ID_PATTERN.fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                (
                    "provenance_id must use "
                    "'provenance:<producer>:<32 hex>'."
                )
            )


        return normalized


    # ========================================================
    # ARTIFACT ID
    # ========================================================


    @field_validator(
        "artifact_id",
        mode="before",
    )
    @classmethod
    def validate_artifact_id(
        cls,
        value: object,
    ) -> str:

        normalized = (
            _required_text(
                value,
                field_name="artifact_id",
            )
            .lower()
        )


        if (
            ARTIFACT_ID_PATTERN.fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                "artifact_id has invalid shape."
            )


        return normalized


    # ========================================================
    # SOURCE CONTRACT
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


    @field_validator(
        "source_contract_sha256",
        mode="before",
    )
    @classmethod
    def validate_source_contract_sha256(
        cls,
        value: object,
    ) -> str:

        return _normalize_sha256(
            value,
            field_name=
                "source_contract_sha256",
        )


    # ========================================================
    # CROSS-FIELD AUTHORITIES
    # ========================================================


    @model_validator(
        mode="after",
    )
    def validate_identity_bindings(
        self,
    ) -> "ModelLifecycleProvenanceRecord":

        artifact_id_family = (
            self.artifact_id
            .split(
                ":",
                maxsplit=2,
            )[1]
        )


        if (
            artifact_id_family
            !=
            self.artifact_family
        ):
            raise ValueError(
                (
                    "artifact_id family must match "
                    "artifact_family."
                )
            )


        provenance_producer = (
            self.provenance_id
            .split(
                ":",
                maxsplit=2,
            )[1]
        )


        if (
            provenance_producer
            !=
            self.producer_family
        ):
            raise ValueError(
                (
                    "provenance_id producer must "
                    "match producer_family."
                )
            )


        evidence_keys: set[
            tuple[
                str,
                str,
            ]
        ] = set()


        for binding in self.evidence:

            key = (
                binding.evidence_kind,
                binding.authority_id,
            )


            if key in evidence_keys:
                raise ValueError(
                    (
                        "Duplicate lifecycle "
                        "evidence binding."
                    )
                )


            evidence_keys.add(
                key
            )


        return self

from __future__ import annotations


import hashlib
import json


from datetime import (
    datetime,
    timezone,
)


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
)

from app.model_lifecycle.provenance_contracts import (
    PROVENANCE_ID_PATTERN,
)


# ============================================================
# VERSION
# ============================================================


MODEL_LIFECYCLE_EVALUATION_DECISION_RULE_VERSION = (
    "model_lifecycle_evaluation_decision_v0.1"
)


# ============================================================
# TYPES
# ============================================================


ModelLifecycleEvaluationStatus = Literal[
    "completed",
]


ModelLifecyclePromotionDecision = Literal[
    "promoted",
    "not_promoted",
]


# ============================================================
# CONSTANTS
# ============================================================


DECISION_ID_PREFIX = (
    "decision:evaluation:"
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


def _normalize_utc_timestamp(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = _required_text(
        value,
        field_name=field_name,
    )


    if not normalized.endswith(
        "Z"
    ):
        raise ValueError(
            (
                f"{field_name} must use "
                "UTC Z notation."
            )
        )


    try:
        parsed = datetime.fromisoformat(
            normalized[:-1]
            +
            "+00:00"
        )

    except ValueError as error:
        raise ValueError(
            f"{field_name} is not valid ISO-8601."
        ) from error


    if (
        parsed.utcoffset()
        !=
        timezone.utc.utcoffset(
            parsed
        )
    ):
        raise ValueError(
            f"{field_name} must be UTC."
        )


    return normalized


def _canonical_json(
    payload: dict[
        str,
        str,
    ],
) -> bytes:

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
        .encode(
            "utf-8"
        )
    )


# ============================================================
# DETERMINISTIC DECISION ID
# ============================================================


def build_model_lifecycle_evaluation_decision_id(
    *,
    artifact_id: str,
    evaluation_id: str,
    source_receipt_sha256: str,
) -> str:

    normalized_artifact_id = (
        _required_text(
            artifact_id,
            field_name="artifact_id",
        )
    )

    normalized_evaluation_id = (
        _required_text(
            evaluation_id,
            field_name="evaluation_id",
        )
    )

    normalized_receipt_sha = (
        _normalize_sha256(
            source_receipt_sha256,
            field_name="source_receipt_sha256",
        )
    )


    digest = hashlib.sha256(
        _canonical_json(
            {
                "artifact_id":
                    normalized_artifact_id,

                "evaluation_id":
                    normalized_evaluation_id,

                "source_receipt_sha256":
                    normalized_receipt_sha,
            }
        )
    ).hexdigest()


    return (
        DECISION_ID_PREFIX
        +
        digest[:32]
    )


# ============================================================
# EVIDENCE BINDING
# ============================================================


class ModelLifecycleDecisionEvidenceBinding(
    BaseModel
):
    """
    One immutable SHA-bound authority supporting a lifecycle
    evaluation/promotion decision.

    The lifecycle binds the evidence identity only.
    It does not reinterpret or mutate the authority.
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
    def normalize_evidence_kind(
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


        allowed = set(
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789._-"
        )


        if (
            normalized[0]
            not in
            "abcdefghijklmnopqrstuvwxyz"
        ):
            raise ValueError(
                (
                    "evidence_kind must start "
                    "with a lowercase letter."
                )
            )


        if any(
            character not in allowed
            for character
            in normalized
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
    def normalize_authority_id(
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
    def normalize_authority_sha256(
        cls,
        value: object,
    ) -> str:

        return _normalize_sha256(
            value,
            field_name="authority_sha256",
        )


# ============================================================
# DECISION RECORD
# ============================================================


class ModelLifecycleEvaluationDecisionRecord(
    BaseModel
):
    """
    Immutable post-evaluation lifecycle decision for one already
    registered model or adapter artifact.

    This record does not own:
    - model or adapter bytes;
    - training state;
    - protected holdout cases;
    - predictions;
    - gold labels;
    - evaluation re-execution authority.

    It only binds registered lifecycle identity to completed
    evaluation evidence and the resulting promotion decision.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    decision_id: str = Field(
        min_length=1,
    )


    artifact_id: str = Field(
        min_length=1,
    )


    provenance_id: str = Field(
        min_length=1,
    )


    experiment_id: str = Field(
        min_length=1,
    )


    evaluation_id: str = Field(
        min_length=1,
    )


    evaluation_status: (
        ModelLifecycleEvaluationStatus
    ) = "completed"


    promotion_eligible: bool


    promotion_decision: (
        ModelLifecyclePromotionDecision
    )


    evaluated_at_utc: str = Field(
        min_length=1,
    )


    source_receipt_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    evidence: tuple[
        ModelLifecycleDecisionEvidenceBinding,
        ...,
    ] = Field(
        min_length=1,
    )


    rule_version: Literal[
        "model_lifecycle_evaluation_decision_v0.1"
    ] = (
        MODEL_LIFECYCLE_EVALUATION_DECISION_RULE_VERSION
    )


    @field_validator(
        "decision_id",
        "experiment_id",
        "evaluation_id",
        mode="before",
    )
    @classmethod
    def normalize_text_fields(
        cls,
        value: object,
        info,
    ) -> str:

        return _required_text(
            value,
            field_name=info.field_name,
        )


    @field_validator(
        "artifact_id",
        mode="before",
    )
    @classmethod
    def validate_artifact_id(
        cls,
        value: object,
    ) -> str:

        normalized = _required_text(
            value,
            field_name="artifact_id",
        )


        if (
            ARTIFACT_ID_PATTERN.fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                "artifact_id is invalid."
            )


        return normalized


    @field_validator(
        "provenance_id",
        mode="before",
    )
    @classmethod
    def validate_provenance_id(
        cls,
        value: object,
    ) -> str:

        normalized = _required_text(
            value,
            field_name="provenance_id",
        )


        if (
            PROVENANCE_ID_PATTERN.fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                "provenance_id is invalid."
            )


        return normalized


    @field_validator(
        "source_receipt_sha256",
        mode="before",
    )
    @classmethod
    def normalize_source_receipt_sha256(
        cls,
        value: object,
    ) -> str:

        return _normalize_sha256(
            value,
            field_name="source_receipt_sha256",
        )


    @field_validator(
        "evaluated_at_utc",
        mode="before",
    )
    @classmethod
    def normalize_evaluated_at_utc(
        cls,
        value: object,
    ) -> str:

        return _normalize_utc_timestamp(
            value,
            field_name="evaluated_at_utc",
        )


    @model_validator(
        mode="after",
    )
    def validate_decision_integrity(
        self,
    ) -> "ModelLifecycleEvaluationDecisionRecord":

        expected_decision = (
            "promoted"
            if self.promotion_eligible
            else
            "not_promoted"
        )


        if (
            self.promotion_decision
            !=
            expected_decision
        ):
            raise ValueError(
                (
                    "promotion_decision does not "
                    "match promotion_eligible."
                )
            )


        expected_decision_id = (
            build_model_lifecycle_evaluation_decision_id(
                artifact_id=
                    self.artifact_id,

                evaluation_id=
                    self.evaluation_id,

                source_receipt_sha256=
                    self.source_receipt_sha256,
            )
        )


        if (
            self.decision_id
            !=
            expected_decision_id
        ):
            raise ValueError(
                (
                    "decision_id does not match "
                    "deterministic decision identity."
                )
            )


        evidence_kinds = [
            binding.evidence_kind
            for binding
            in self.evidence
        ]


        if (
            len(evidence_kinds)
            !=
            len(set(evidence_kinds))
        ):
            raise ValueError(
                "Duplicate evidence_kind is not allowed."
            )


        authority_ids = [
            binding.authority_id
            for binding
            in self.evidence
        ]


        if (
            len(authority_ids)
            !=
            len(set(authority_ids))
        ):
            raise ValueError(
                "Duplicate authority_id is not allowed."
            )


        if not any(
            (
                binding.authority_sha256
                ==
                self.source_receipt_sha256
            )
            for binding
            in self.evidence
        ):
            raise ValueError(
                (
                    "source_receipt_sha256 must be "
                    "present in evidence bindings."
                )
            )


        return self

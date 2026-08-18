from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


# ============================================================
# EVIDENCE TYPES
# ============================================================

EvidenceSourceType = Literal[
    "statistical_decision",
    "statistical_result",
    "visualization_decision",
    "dashboard_spec",
]


EvidenceProducer = Literal[
    "python",
]


# ============================================================
# ONE CANONICAL EVIDENCE RECORD
# ============================================================

class EvidenceRecord(
    BaseModel
):
    """
    One deterministic DataLens evidence record.

    Every record has:

    - a unique stable evidence ID
    - a precise source type
    - a dataset
    - a deterministic producer
    - a versioned rule
    - explicit dependencies
    - canonical structured data
    """

    evidence_id: str = Field(
        min_length=1,
    )

    source_type: (
        EvidenceSourceType
    )

    dataset: str = Field(
        min_length=1,
    )

    producer: EvidenceProducer = (
        "python"
    )

    rule_version: str = Field(
        min_length=1,
    )

    depends_on: list[
        str
    ] = Field(
        default_factory=list,
    )

    data: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


# ============================================================
# COMPLETE ANALYSIS EVIDENCE BUNDLE
# ============================================================

class AnalysisEvidenceBundle(
    BaseModel
):
    """
    Canonical evidence collection for one
    DataLens analysis.

    The bundle validates:

    - evidence ID uniqueness
    - dependency existence
    - no self-dependencies
    - dataset consistency
    """

    dataset: str = Field(
        min_length=1,
    )

    evidence: list[
        EvidenceRecord
    ] = Field(
        min_length=1,
    )

    evidence_rule_version: str = (
        "analysis_evidence_v0.1"
    )

    @model_validator(
        mode="after"
    )
    def validate_evidence_bundle(
        self,
    ):
        evidence_ids = [
            item.evidence_id
            for item
            in self.evidence
        ]

        # ====================================================
        # UNIQUE IDS
        # ====================================================

        if (
            len(
                evidence_ids
            )
            != len(
                set(
                    evidence_ids
                )
            )
        ):
            raise ValueError(
                (
                    "Evidence IDs must be unique "
                    "inside an analysis bundle."
                )
            )

        evidence_id_set = set(
            evidence_ids
        )

        # ====================================================
        # DATASET CONSISTENCY
        # ====================================================

        for item in (
            self.evidence
        ):
            if (
                item.dataset
                != self.dataset
            ):
                raise ValueError(
                    (
                        "Evidence dataset does not "
                        "match bundle dataset: "
                        f"{item.evidence_id!r}."
                    )
                )

        # ====================================================
        # DEPENDENCIES
        # ====================================================

        for item in (
            self.evidence
        ):
            for dependency in (
                item.depends_on
            ):
                if (
                    dependency
                    == item.evidence_id
                ):
                    raise ValueError(
                        (
                            "Evidence cannot depend "
                            "on itself: "
                            f"{item.evidence_id!r}."
                        )
                    )

                if (
                    dependency
                    not in evidence_id_set
                ):
                    raise ValueError(
                        (
                            "Evidence dependency "
                            "does not exist in the "
                            "bundle: "
                            f"{item.evidence_id!r} "
                            "depends on "
                            f"{dependency!r}."
                        )
                    )

        return self

    def get_evidence(
        self,
        evidence_id: str,
    ) -> EvidenceRecord:
        """
        Retrieve one evidence record by ID.
        """

        for item in (
            self.evidence
        ):
            if (
                item.evidence_id
                == evidence_id
            ):
                return item

        raise KeyError(
            (
                "Evidence ID does not exist: "
                f"{evidence_id!r}"
            )
        )
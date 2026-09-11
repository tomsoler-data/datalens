from __future__ import annotations


from pydantic import ValidationError


from app.model_lifecycle.provenance_contracts import (
    MODEL_LIFECYCLE_PROVENANCE_RULE_VERSION,
    ModelLifecycleEvidenceBinding,
    ModelLifecycleProvenanceRecord,
)


# ============================================================
# FIXTURES
# ============================================================


def evidence(
) -> tuple[
    ModelLifecycleEvidenceBinding,
    ...,
]:

    return (
        ModelLifecycleEvidenceBinding(
            evidence_kind=
                "source_contract_frozen",

            authority_id=(
                "artifacts/adaptation/experiments/"
                "datalens_semantic_qlora_v0.4_contract.json"
            ),

            authority_sha256=
                "c" * 64,
        ),

        ModelLifecycleEvidenceBinding(
            evidence_kind=
                "training_manifest",

            authority_id=(
                "artifacts/adaptation/training/"
                "datalens_semantic_qlora_v0.4_"
                "training_v0.1_manifest.json"
            ),

            authority_sha256=
                "d" * 64,
        ),

        ModelLifecycleEvidenceBinding(
            evidence_kind=
                "training_receipt",

            authority_id=(
                "artifacts/adaptation/training/"
                "datalens_semantic_qlora_v0.4_"
                "training_v0.1_receipt.json"
            ),

            authority_sha256=
                "e" * 64,
        ),

        ModelLifecycleEvidenceBinding(
            evidence_kind=
                "training_dataset",

            authority_id=(
                "adaptation:datalens-semantic:"
                "training:v0.4"
            ),

            authority_sha256=
                "f" * 64,
        ),
    )


def build_record(
) -> ModelLifecycleProvenanceRecord:

    return ModelLifecycleProvenanceRecord(
        provenance_id=(
            "provenance:llm_adaptation:"
            "0123456789abcdef0123456789abcdef"
        ),

        artifact_id=(
            "artifact:adapter:"
            "0351980df6d86096195c0971deb30c72"
        ),

        artifact_family=
            "adapter",

        experiment_id=
            "datalens-semantic-qlora-v0.4",

        producer_family=
            "llm_adaptation",

        execution_id=(
            "training-run:"
            "datalens-semantic-qlora:v0.4:0001"
        ),

        source_contract_kind=(
            "qlora_experiment_contract_v0.1"
        ),

        source_contract_sha256=
            "a" * 64,

        evidence=
            evidence(),

        created_at_utc=
            "2026-08-30T01:19:36Z",
    )


def expect_validation_error(
    **updates,
) -> None:

    payload = (
        build_record()
        .model_dump(
            mode="python"
        )
    )


    payload.update(
        updates
    )


    try:

        ModelLifecycleProvenanceRecord(
            **payload
        )


    except ValidationError:
        return


    raise AssertionError(
        (
            "Expected provenance validation "
            f"failure for {updates!r}"
        )
    )


# ============================================================
# ACCEPTANCE
# ============================================================


def main(
) -> None:

    record = build_record()


    assert (
        record.rule_version
        ==
        MODEL_LIFECYCLE_PROVENANCE_RULE_VERSION
    )


    assert (
        record.artifact_family
        ==
        "adapter"
    )


    assert (
        record.producer_family
        ==
        "llm_adaptation"
    )


    assert len(
        record.evidence
    ) == 4


    assert (
        record.source_contract_sha256
        ==
        "a" * 64
    )


    # --------------------------------------------------------
    # Frozen record
    # --------------------------------------------------------


    try:
        record.execution_id = "other"

    except Exception:
        pass

    else:
        raise AssertionError(
            "Provenance record must be frozen."
        )


    # --------------------------------------------------------
    # Provenance ID shape
    # --------------------------------------------------------


    expect_validation_error(
        provenance_id=
            "caller-controlled-id"
    )


    # --------------------------------------------------------
    # Provenance ID producer mismatch
    # --------------------------------------------------------


    expect_validation_error(
        provenance_id=(
            "provenance:deep_learning:"
            "0123456789abcdef0123456789abcdef"
        )
    )


    # --------------------------------------------------------
    # Artifact ID / family mismatch
    # --------------------------------------------------------


    expect_validation_error(
        artifact_id=(
            "artifact:model:"
            "0351980df6d86096195c0971deb30c72"
        )
    )


    # --------------------------------------------------------
    # SHA validation
    # --------------------------------------------------------


    expect_validation_error(
        source_contract_sha256=
            "not-a-sha"
    )


    # --------------------------------------------------------
    # Evidence required
    # --------------------------------------------------------


    expect_validation_error(
        evidence=()
    )


    # --------------------------------------------------------
    # Duplicate evidence rejected
    # --------------------------------------------------------


    bindings = evidence()


    expect_validation_error(
        evidence=(
            bindings[
                0
            ],
            bindings[
                0
            ],
        )
    )


    # --------------------------------------------------------
    # Evidence SHA
    # --------------------------------------------------------


    try:

        ModelLifecycleEvidenceBinding(
            evidence_kind=
                "training_manifest",

            authority_id=
                "manifest.json",

            authority_sha256=
                "invalid",
        )


    except ValidationError:
        pass

    else:
        raise AssertionError(
            "Invalid evidence SHA must fail."
        )


    # --------------------------------------------------------
    # Unknown fields
    # --------------------------------------------------------


    payload = record.model_dump(
        mode="python"
    )

    payload[
        "unexpected_field"
    ] = True


    try:

        ModelLifecycleProvenanceRecord(
            **payload
        )


    except ValidationError:
        pass

    else:
        raise AssertionError(
            "Unknown fields must fail closed."
        )


    print(
        "Shared provenance construction            PASS"
    )

    print(
        "Immutable provenance record               PASS"
    )

    print(
        "Artifact identity binding                PASS"
    )

    print(
        "Producer identity binding                PASS"
    )

    print(
        "Source-contract binding                  PASS"
    )

    print(
        "Evidence SHA bindings                    PASS"
    )

    print(
        "Evidence required                        PASS"
    )

    print(
        "Duplicate evidence rejected              PASS"
    )

    print(
        "Unknown fields fail closed               PASS"
    )

    print()
    print(
        "Model Lifecycle Provenance Contract v0.1: PASS"
    )


if __name__ == "__main__":
    main()

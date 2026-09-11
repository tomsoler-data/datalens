from __future__ import annotations


from pydantic import ValidationError


from app.model_lifecycle.artifact_contracts import (
    MODEL_LIFECYCLE_ARTIFACT_RULE_VERSION,
    ModelLifecycleArtifactRecord,
)


def build_record(
) -> ModelLifecycleArtifactRecord:

    return ModelLifecycleArtifactRecord(
        artifact_id=(
            "artifact:adapter:"
            "0123456789abcdef0123456789abcdef"
        ),
        artifact_family="adapter",
        experiment_id=(
            "adaptation:"
            "datalens-semantic-qlora-v0.4"
        ),
        artifact_format=(
            "peft_adapter_bundle"
        ),
        artifact_path=(
            "data/adapters/"
            "datalens_semantic_qlora_v0.4"
        ),
        artifact_file_bytes=119273568,
        artifact_sha256="a" * 64,
        created_at_utc=(
            "2026-09-10T10:00:00+00:00"
        ),
        source_contract_kind=(
            "qlora_experiment_contract_v0.1"
        ),
        source_contract_sha256="b" * 64,
    )


def expect_validation_error(
    **updates,
) -> None:

    payload = build_record().model_dump(
        mode="python"
    )

    payload.update(
        updates
    )


    try:
        ModelLifecycleArtifactRecord(
            **payload
        )

    except ValidationError:
        return


    raise AssertionError(
        (
            "Expected validation failure for "
            f"{updates!r}"
        )
    )


def main(
) -> None:

    record = build_record()


    assert (
        record.rule_version
        ==
        MODEL_LIFECYCLE_ARTIFACT_RULE_VERSION
    )

    assert (
        record.artifact_family
        ==
        "adapter"
    )

    assert (
        record.artifact_format
        ==
        "peft_adapter_bundle"
    )

    assert (
        record.artifact_sha256
        ==
        "a" * 64
    )

    assert (
        record.source_contract_sha256
        ==
        "b" * 64
    )


    # Frozen model contract.
    try:
        record.artifact_path = "other"

    except Exception:
        pass

    else:
        raise AssertionError(
            "Artifact record must be frozen."
        )


    expect_validation_error(
        artifact_id="caller-controlled-id"
    )

    expect_validation_error(
        artifact_id=(
            "artifact:model:"
            "0123456789abcdef0123456789abcdef"
        )
    )

    expect_validation_error(
        artifact_path="../escape.bin"
    )

    expect_validation_error(
        artifact_path="/absolute/model.bin"
    )

    expect_validation_error(
        artifact_file_bytes=0
    )

    expect_validation_error(
        artifact_sha256="not-a-sha"
    )

    expect_validation_error(
        source_contract_sha256="c" * 63
    )

    expect_validation_error(
        source_contract_kind="../bad"
    )

    expect_validation_error(
        artifact_format="pickle"
    )

    expect_validation_error(
        artifact_family="dataset"
    )


    payload = record.model_dump(
        mode="python"
    )

    payload["unexpected_field"] = True


    try:
        ModelLifecycleArtifactRecord(
            **payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            "Unknown fields must fail closed."
        )


    print(
        "Shared Artifact identity construction       PASS"
    )

    print(
        "Artifact identity is immutable             PASS"
    )

    print(
        "Server-shaped artifact id                 PASS"
    )

    print(
        "Relative artifact path guard              PASS"
    )

    print(
        "SHA-256 authority                         PASS"
    )

    print(
        "Source-contract binding                   PASS"
    )

    print(
        "PEFT adapter format identity              PASS"
    )

    print(
        "Unknown fields fail closed                PASS"
    )

    print()
    print(
        "Model Lifecycle Artifact Contract v0.1: PASS"
    )


if __name__ == "__main__":
    main()

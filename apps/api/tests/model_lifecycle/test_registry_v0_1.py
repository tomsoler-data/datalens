from __future__ import annotations


import hashlib
import sqlite3
import tempfile


from pathlib import Path


from app.model_lifecycle.artifact_contracts import (
    ModelLifecycleArtifactRecord,
)

from app.model_lifecycle.provenance_contracts import (
    ModelLifecycleEvidenceBinding,
    ModelLifecycleProvenanceRecord,
)

import app.model_lifecycle.registry as registry_module


from app.model_lifecycle.registry import (
    MODEL_LIFECYCLE_REGISTRY_RULE_VERSION,
    ModelLifecycleRegistryConflictError,
    ModelLifecycleRegistryCorruptionError,
    ModelLifecycleRegistryError,
    ModelLifecycleRegistryNotFoundError,
    get_model_lifecycle_entry,
    list_model_lifecycle_entries,
    register_model_lifecycle_entry,
)


# ============================================================
# FIXTURES
# ============================================================


def adapter_artifact(
    *,
    artifact_id: str = (
        "artifact:adapter:"
        "11111111111111111111111111111111"
    ),
    artifact_sha256: str = "a" * 64,
) -> ModelLifecycleArtifactRecord:

    return ModelLifecycleArtifactRecord(
        artifact_id=
            artifact_id,

        artifact_family=
            "adapter",

        experiment_id=
            "adaptation:test:v0.1",

        artifact_format=
            "peft_adapter_bundle",

        artifact_path=
            "external/adapters/test_adapter",

        artifact_file_bytes=
            1024,

        artifact_sha256=
            artifact_sha256,

        created_at_utc=
            "2026-09-10T12:00:00Z",

        source_contract_kind=
            "qlora_experiment_contract_v0.1",

        source_contract_sha256=
            "b" * 64,
    )


def adapter_provenance(
    artifact: ModelLifecycleArtifactRecord,
    *,
    provenance_id: str = (
        "provenance:llm_adaptation:"
        "22222222222222222222222222222222"
    ),
) -> ModelLifecycleProvenanceRecord:

    return ModelLifecycleProvenanceRecord(
        provenance_id=
            provenance_id,

        artifact_id=
            artifact.artifact_id,

        artifact_family=
            artifact.artifact_family,

        experiment_id=
            artifact.experiment_id,

        producer_family=
            "llm_adaptation",

        execution_id=
            "training-run:test:v0.1:0001",

        source_contract_kind=
            artifact.source_contract_kind,

        source_contract_sha256=
            artifact.source_contract_sha256,

        evidence=(
            ModelLifecycleEvidenceBinding(
                evidence_kind=
                    "training_receipt",

                authority_id=
                    "training-receipt:test:v0.1",

                authority_sha256=
                    "c" * 64,
            ),
        ),

        created_at_utc=
            "2026-09-10T12:00:00Z",
    )


def model_artifact(
) -> ModelLifecycleArtifactRecord:

    return ModelLifecycleArtifactRecord(
        artifact_id=(
            "artifact:model:"
            "33333333333333333333333333333333"
        ),

        artifact_family=
            "model",

        experiment_id=
            "deep-learning:test:v0.1",

        artifact_format=
            "pytorch_bundle",

        artifact_path=
            "external/models/test_model.ptbundle",

        artifact_file_bytes=
            2048,

        artifact_sha256=
            "d" * 64,

        created_at_utc=
            "2026-09-10T12:01:00Z",

        source_contract_kind=
            "time_series_model_training_contract_v0.1",

        source_contract_sha256=
            "e" * 64,
    )


def model_provenance(
    artifact: ModelLifecycleArtifactRecord,
) -> ModelLifecycleProvenanceRecord:

    return ModelLifecycleProvenanceRecord(
        provenance_id=(
            "provenance:deep_learning:"
            "44444444444444444444444444444444"
        ),

        artifact_id=
            artifact.artifact_id,

        artifact_family=
            artifact.artifact_family,

        experiment_id=
            artifact.experiment_id,

        producer_family=
            "deep_learning",

        execution_id=
            "training-run:dl:test:v0.1",

        source_contract_kind=
            artifact.source_contract_kind,

        source_contract_sha256=
            artifact.source_contract_sha256,

        evidence=(
            ModelLifecycleEvidenceBinding(
                evidence_kind=
                    "training_contract",

                authority_id=
                    "training-contract:dl:test",

                authority_sha256=
                    "f" * 64,
            ),
        ),

        created_at_utc=
            "2026-09-10T12:01:00Z",
    )


def sha256_file(
    path: Path,
) -> str:

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def expect_conflict(
    *,
    artifact: ModelLifecycleArtifactRecord,
    provenance: ModelLifecycleProvenanceRecord,
    registry_path: Path,
) -> None:

    try:
        register_model_lifecycle_entry(
            artifact=
                artifact,

            provenance=
                provenance,

            registry_path=
                registry_path,
        )

    except ModelLifecycleRegistryConflictError:
        return


    raise AssertionError(
        "Expected lifecycle registry conflict."
    )


# ============================================================
# ACCEPTANCE
# ============================================================


def main(
) -> None:

    assert (
        MODEL_LIFECYCLE_REGISTRY_RULE_VERSION
        ==
        "model_lifecycle_registry_v0.1"
    )


    with tempfile.TemporaryDirectory() as raw_temp:

        temp = Path(
            raw_temp
        )


        registry_path = (
            temp
            /
            "registry"
            /
            "registry.sqlite3"
        )


        external_root = (
            temp
            /
            "external"
        )


        external_root.mkdir(
            parents=True,
            exist_ok=True,
        )


        external_weights = (
            external_root
            /
            "adapter_model.safetensors"
        )


        external_weights.write_bytes(
            b"external-adapter-bytes"
        )


        external_sha_before = (
            sha256_file(
                external_weights
            )
        )


        # ====================================================
        # EMPTY REGISTRY
        # ====================================================


        assert (
            list_model_lifecycle_entries(
                registry_path=
                    registry_path
            )
            ==
            ()
        )


        try:
            get_model_lifecycle_entry(
                (
                    "artifact:adapter:"
                    "11111111111111111111111111111111"
                ),
                registry_path=
                    registry_path,
            )

        except ModelLifecycleRegistryNotFoundError:
            pass

        else:
            raise AssertionError(
                "Missing registry entry must fail."
            )


        # ====================================================
        # ADAPTER REGISTRATION
        # ====================================================


        artifact = (
            adapter_artifact()
        )


        provenance = (
            adapter_provenance(
                artifact
            )
        )


        registered = (
            register_model_lifecycle_entry(
                artifact=
                    artifact,

                provenance=
                    provenance,

                registry_path=
                    registry_path,
            )
        )


        assert (
            registered.artifact
            ==
            artifact
        )


        assert (
            registered.provenance
            ==
            provenance
        )


        # ====================================================
        # IDEMPOTENT RE-REGISTRATION
        # ====================================================


        repeated = (
            register_model_lifecycle_entry(
                artifact=
                    artifact,

                provenance=
                    provenance,

                registry_path=
                    registry_path,
            )
        )


        assert (
            repeated
            ==
            registered
        )


        # ====================================================
        # GET
        # ====================================================


        loaded = (
            get_model_lifecycle_entry(
                artifact.artifact_id,
                registry_path=
                    registry_path,
            )
        )


        assert (
            loaded
            ==
            registered
        )


        # ====================================================
        # SHARED REGISTRY SUPPORTS MODEL FAMILY TOO
        # ====================================================


        dl_artifact = (
            model_artifact()
        )


        dl_provenance = (
            model_provenance(
                dl_artifact
            )
        )


        register_model_lifecycle_entry(
            artifact=
                dl_artifact,

            provenance=
                dl_provenance,

            registry_path=
                registry_path,
        )


        entries = (
            list_model_lifecycle_entries(
                registry_path=
                    registry_path
            )
        )


        assert len(
            entries
        ) == 2


        assert [
            entry.artifact.artifact_id
            for entry
            in entries
        ] == sorted(
            [
                artifact.artifact_id,
                dl_artifact.artifact_id,
            ]
        )


        # ====================================================
        # SAME ARTIFACT ID / DIFFERENT METADATA
        # ====================================================


        conflict_artifact = (
            adapter_artifact(
                artifact_sha256=
                    "9" * 64
            )
        )


        expect_conflict(
            artifact=
                conflict_artifact,

            provenance=
                provenance,

            registry_path=
                registry_path,
        )


        # ====================================================
        # SAME PROVENANCE ID / DIFFERENT ARTIFACT
        # ====================================================


        other_artifact = (
            adapter_artifact(
                artifact_id=(
                    "artifact:adapter:"
                    "55555555555555555555555555555555"
                )
            )
        )


        reused_provenance = (
            adapter_provenance(
                other_artifact,
                provenance_id=
                    provenance.provenance_id,
            )
        )


        expect_conflict(
            artifact=
                other_artifact,

            provenance=
                reused_provenance,

            registry_path=
                registry_path,
        )


        # ====================================================
        # ARTIFACT / PROVENANCE BINDING
        # ====================================================


        try:
            register_model_lifecycle_entry(
                artifact=
                    artifact,

                provenance=
                    dl_provenance,

                registry_path=
                    registry_path,
            )

        except ModelLifecycleRegistryError:
            pass

        else:
            raise AssertionError(
                (
                    "Artifact/provenance mismatch "
                    "must fail closed."
                )
            )


        # ====================================================
        # EXTERNAL ARTIFACT MUST REMAIN UNTOUCHED
        # ====================================================


        external_sha_after = (
            sha256_file(
                external_weights
            )
        )


        assert (
            external_sha_after
            ==
            external_sha_before
        )


        assert (
            external_weights.read_bytes()
            ==
            b"external-adapter-bytes"
        )


        registry_files = sorted(
            path.name
            for path
            in registry_path.parent.iterdir()
            if path.is_file()
        )


        assert (
            registry_files
            ==
            [
                "registry.sqlite3"
            ]
        )


        # ====================================================
        # CORRUPTION FAILS CLOSED
        # ====================================================


        corrupt_path = (
            temp
            /
            "corrupt"
            /
            "registry.sqlite3"
        )


        corrupt_artifact = (
            adapter_artifact(
                artifact_id=(
                    "artifact:adapter:"
                    "66666666666666666666666666666666"
                )
            )
        )


        corrupt_provenance = (
            adapter_provenance(
                corrupt_artifact,
                provenance_id=(
                    "provenance:llm_adaptation:"
                    "77777777777777777777777777777777"
                ),
            )
        )


        register_model_lifecycle_entry(
            artifact=
                corrupt_artifact,

            provenance=
                corrupt_provenance,

            registry_path=
                corrupt_path,
        )


        connection = sqlite3.connect(
            str(
                corrupt_path
            )
        )


        try:
            connection.execute(
                """
                UPDATE model_lifecycle_registry
                SET entry_json = ?
                WHERE artifact_id = ?
                """,
                (
                    "{}",
                    corrupt_artifact.artifact_id,
                ),
            )

            connection.commit()

        finally:
            connection.close()


        try:
            get_model_lifecycle_entry(
                corrupt_artifact.artifact_id,
                registry_path=
                    corrupt_path,
            )

        except ModelLifecycleRegistryCorruptionError:
            pass

        else:
            raise AssertionError(
                (
                    "Corrupted registry metadata "
                    "must fail closed."
                )
            )


        # ====================================================
        # NO DELETE OWNERSHIP
        # ====================================================


        assert not hasattr(
            registry_module,
            "delete_model_lifecycle_entry",
        )


        registry_source = (
            Path(
                registry_module.__file__
            )
            .read_text(
                encoding="utf-8"
            )
        )


        assert (
            "DELETE FROM"
            not in
            registry_source.upper()
        )


    print(
        "Metadata-only SQLite registry            PASS"
    )

    print(
        "Artifact/provenance binding              PASS"
    )

    print(
        "Idempotent registration                 PASS"
    )

    print(
        "Get semantics                           PASS"
    )

    print(
        "Deterministic list semantics            PASS"
    )

    print(
        "Artifact ID conflict rejected           PASS"
    )

    print(
        "Provenance ID conflict rejected         PASS"
    )

    print(
        "Registry corruption fail-closed         PASS"
    )

    print(
        "External artifact bytes preserved       PASS"
    )

    print(
        "No adapter byte copy                    PASS"
    )

    print(
        "No artifact deletion authority          PASS"
    )

    print(
        "ML/DL-neutral registry                  PASS"
    )

    print()
    print(
        "Model Lifecycle Registry v0.1: PASS"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations


import hashlib
import json
import tempfile


from pathlib import Path


from app.model_lifecycle.artifact_contracts import (
    ModelLifecycleArtifactRecord,
)

from app.model_lifecycle.provenance_contracts import (
    ModelLifecycleEvidenceBinding,
    ModelLifecycleProvenanceRecord,
)

from app.model_lifecycle.registry import (
    register_model_lifecycle_entry,
)

from app.model_lifecycle.trusted_artifact_resolver import (
    PEFT_ADAPTER_FILES,
    TRUSTED_ARTIFACT_RESOLVER_RULE_VERSION,
    TrustedArtifactResolutionError,
    resolve_trusted_peft_adapter_artifact,
)


# ============================================================
# HELPERS
# ============================================================


def sha256_bytes(
    value: bytes,
) -> str:

    return hashlib.sha256(
        value
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def write_lf_text(
    path: Path,
    text: str,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:

        handle.write(
            text
        )


def normalized_text_sha256(
    path: Path,
) -> str:

    raw = path.read_bytes()


    if raw.startswith(
        b"\xef\xbb\xbf"
    ):
        raw = raw[
            3:
        ]


    text = raw.decode(
        "utf-8"
    )


    normalized = (
        text
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .encode(
            "utf-8"
        )
    )


    return hashlib.sha256(
        normalized
    ).hexdigest()


def expect_resolution_error(
    *,
    artifact_id: str,
    registry_path: Path,
    api_root: Path,
) -> None:

    try:

        resolve_trusted_peft_adapter_artifact(
            artifact_id,
            registry_path=
                registry_path,

            api_root=
                api_root,
        )


    except TrustedArtifactResolutionError:
        return


    raise AssertionError(
        (
            "Expected trusted artifact "
            "resolution failure."
        )
    )


def build_registered_fixture(
    root: Path,
    *,
    token: str,
    artifact_path: str | None = None,
    receipt_sha_override: str | None = None,
    artifact_family: str = "adapter",
) -> tuple[
    str,
    Path,
    Path,
    tuple[
        Path,
        ...,
    ],
]:

    api_root = (
        root
        /
        f"api_{token}"
    )


    adapter_directory = (
        api_root
        /
        "artifacts"
        /
        "adaptation"
        /
        "adapters"
        /
        f"adapter_{token}"
    )


    training_directory = (
        api_root
        /
        "artifacts"
        /
        "adaptation"
        /
        "training"
    )


    adapter_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    training_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    contents = {
        "README.md":
            (
                "# Adapter\n"
            ).encode(
                "utf-8"
            ),

        "adapter_config.json":
            (
                '{"r":16}\n'
            ).encode(
                "utf-8"
            ),

        "adapter_model.safetensors":
            (
                b"fake-safe-tensor-bytes-"
                +
                token.encode(
                    "utf-8"
                )
            ),
    }


    physical_files = []


    for name in PEFT_ADAPTER_FILES:

        path = (
            adapter_directory
            /
            name
        )


        path.write_bytes(
            contents[
                name
            ]
        )


        physical_files.append(
            path
        )


    file_entries = [
        {
            "relative_path":
                name,

            "sha256":
                sha256_bytes(
                    contents[
                        name
                    ]
                ),

            "size_bytes":
                len(
                    contents[
                        name
                    ]
                ),
        }

        for name
        in PEFT_ADAPTER_FILES
    ]


    bundle_sha = sha256_bytes(
        (
            "bundle-"
            +
            token
        ).encode(
            "utf-8"
        )
    )


    experiment_id = (
        f"adaptation:test:{token}"
    )


    execution_id = (
        f"training-run:test:{token}:0001"
    )


    receipt = {
        "passed":
            True,

        "experiment_id":
            experiment_id,

        "training_run_id":
            execution_id,

        "adapter": {
            "bundle_sha256":
                bundle_sha,

            "file_count":
                3,

            "files":
                file_entries,
        },
    }


    receipt_path = (
        training_directory
        /
        f"training_receipt_{token}.json"
    )


    write_lf_text(
        receipt_path,
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
    )


    receipt_sha = (
        normalized_text_sha256(
            receipt_path
        )
    )


    if receipt_sha_override is not None:

        receipt_sha = (
            receipt_sha_override
        )


    suffix = hashlib.sha256(
        token.encode(
            "utf-8"
        )
    ).hexdigest()[
        :32
    ]


    artifact_id = (
        f"artifact:{artifact_family}:"
        f"{suffix}"
    )


    if artifact_path is None:

        artifact_path = (
            "artifacts/adaptation/adapters/"
            f"adapter_{token}"
        )


    artifact = (
        ModelLifecycleArtifactRecord(
            artifact_id=
                artifact_id,

            artifact_family=
                artifact_family,

            experiment_id=
                experiment_id,

            artifact_format=
                "peft_adapter_bundle",

            artifact_path=
                artifact_path,

            artifact_file_bytes=
                sum(
                    len(
                        value
                    )
                    for value
                    in contents.values()
                ),

            artifact_sha256=
                bundle_sha,

            created_at_utc=
                "2026-09-10T12:00:00Z",

            source_contract_kind=
                "qlora_experiment_contract_v0.1",

            source_contract_sha256=
                "a" * 64,
        )
    )


    provenance_suffix = (
        hashlib.sha256(
            (
                "provenance-"
                +
                token
            ).encode(
                "utf-8"
            )
        ).hexdigest()[
            :32
        ]
    )


    producer_family = (
        "llm_adaptation"

        if artifact_family
        ==
        "adapter"

        else
        "deep_learning"
    )


    provenance = (
        ModelLifecycleProvenanceRecord(
            provenance_id=(
                f"provenance:{producer_family}:"
                f"{provenance_suffix}"
            ),

            artifact_id=
                artifact.artifact_id,

            artifact_family=
                artifact.artifact_family,

            experiment_id=
                artifact.experiment_id,

            producer_family=
                producer_family,

            execution_id=
                execution_id,

            source_contract_kind=
                artifact.source_contract_kind,

            source_contract_sha256=
                artifact.source_contract_sha256,

            evidence=(
                ModelLifecycleEvidenceBinding(
                    evidence_kind=
                        "training_receipt",

                    authority_id=
                        receipt_path
                        .resolve()
                        .as_posix(),

                    authority_sha256=
                        receipt_sha,
                ),
            ),

            created_at_utc=
                "2026-09-10T12:00:00Z",
        )
    )


    registry_path = (
        root
        /
        f"registry_{token}.sqlite3"
    )


    register_model_lifecycle_entry(
        artifact=
            artifact,

        provenance=
            provenance,

        registry_path=
            registry_path,
    )


    return (
        artifact.artifact_id,
        api_root,
        registry_path,
        tuple(
            physical_files
        ),
    )


# ============================================================
# ACCEPTANCE
# ============================================================


def main(
) -> None:

    assert (
        TRUSTED_ARTIFACT_RESOLVER_RULE_VERSION
        ==
        "trusted_artifact_resolver_v0.1"
    )


    with tempfile.TemporaryDirectory() as raw_temp:

        root = Path(
            raw_temp
        )


        # ====================================================
        # HAPPY PATH
        # ====================================================


        (
            artifact_id,
            api_root,
            registry_path,
            physical_files,
        ) = build_registered_fixture(
            root,
            token=
                "happy",
        )


        registry_sha_before = (
            sha256_file(
                registry_path
            )
        )


        physical_hashes_before = {
            path:
                sha256_file(
                    path
                )

            for path
            in physical_files
        }


        resolution = (
            resolve_trusted_peft_adapter_artifact(
                artifact_id,
                registry_path=
                    registry_path,

                api_root=
                    api_root,
            )
        )


        assert (
            resolution.artifact_id
            ==
            artifact_id
        )


        assert (
            resolution.experiment_id
            ==
            "adaptation:test:happy"
        )


        assert (
            resolution.rule_version
            ==
            TRUSTED_ARTIFACT_RESOLVER_RULE_VERSION
        )


        assert len(
            resolution.files
        ) == 3


        assert {
            item.relative_path
            for item
            in resolution.files
        } == set(
            PEFT_ADAPTER_FILES
        )


        assert (
            sha256_file(
                registry_path
            )
            ==
            registry_sha_before
        )


        for path in physical_files:

            assert (
                sha256_file(
                    path
                )
                ==
                physical_hashes_before[
                    path
                ]
            )


        # ====================================================
        # PHYSICAL TAMPERING MUST FAIL
        # ====================================================


        (
            tamper_id,
            tamper_api,
            tamper_registry,
            tamper_files,
        ) = build_registered_fixture(
            root,
            token=
                "tamper",
        )


        readme = [
            path

            for path
            in tamper_files

            if path.name
            ==
            "README.md"
        ][
            0
        ]


        readme.write_bytes(
            b"tampered-content\n"
        )


        expect_resolution_error(
            artifact_id=
                tamper_id,

            registry_path=
                tamper_registry,

            api_root=
                tamper_api,
        )


        # ====================================================
        # PATH ESCAPE / WRONG NAMESPACE MUST FAIL
        # ====================================================


        (
            outside_id,
            outside_api,
            outside_registry,
            _,
        ) = build_registered_fixture(
            root,
            token=
                "outside",

            artifact_path=
                "external/adapters/outside",
        )


        expect_resolution_error(
            artifact_id=
                outside_id,

            registry_path=
                outside_registry,

            api_root=
                outside_api,
        )


        # ====================================================
        # RECEIPT SHA MISMATCH MUST FAIL
        # ====================================================


        (
            receipt_id,
            receipt_api,
            receipt_registry,
            _,
        ) = build_registered_fixture(
            root,
            token=
                "receipt",

            receipt_sha_override=
                "0" * 64,
        )


        expect_resolution_error(
            artifact_id=
                receipt_id,

            registry_path=
                receipt_registry,

            api_root=
                receipt_api,
        )


        # ====================================================
        # WRONG ARTIFACT FAMILY MUST FAIL
        # ====================================================


        (
            model_id,
            model_api,
            model_registry,
            _,
        ) = build_registered_fixture(
            root,
            token=
                "model",

            artifact_family=
                "model",
        )


        expect_resolution_error(
            artifact_id=
                model_id,

            registry_path=
                model_registry,

            api_root=
                model_api,
        )


    print(
        "Artifact-ID registry resolution          PASS"
    )

    print(
        "Trusted adapter-root containment         PASS"
    )

    print(
        "Exact PEFT three-file bundle            PASS"
    )

    print(
        "Training receipt evidence binding       PASS"
    )

    print(
        "Training receipt SHA integrity          PASS"
    )

    print(
        "Per-file byte-size integrity            PASS"
    )

    print(
        "Per-file SHA-256 integrity              PASS"
    )

    print(
        "Registry remains byte-identical         PASS"
    )

    print(
        "Resolved adapter remains unchanged      PASS"
    )

    print(
        "Physical tampering rejected             PASS"
    )

    print(
        "Namespace escape rejected               PASS"
    )

    print(
        "Receipt tampering rejected              PASS"
    )

    print(
        "Wrong artifact family rejected          PASS"
    )

    print()
    print(
        "Trusted Artifact Resolver v0.1: PASS"
    )


if __name__ == "__main__":
    main()

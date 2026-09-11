from __future__ import annotations


import json
import tempfile


from pathlib import Path


from app.model_lifecycle.fingerprints import (
    source_contract_sha256,
)

from app.model_lifecycle.qlora_projection import (
    QLORA_LIFECYCLE_PROJECTION_RULE_VERSION,
    QLoRALifecycleProjectionError,
    project_qlora_adapter_artifact,
)


# ============================================================
# AUTHORITIES
# ============================================================


API_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        2
    ]
)


CONTRACT = (
    API_ROOT
    /
    "artifacts/adaptation/experiments/"
    "datalens_semantic_qlora_v0.4_contract.json"
)


FREEZE = (
    API_ROOT
    /
    "artifacts/adaptation/experiments/"
    "datalens_semantic_qlora_v0.4_contract_freeze.json"
)


MANIFEST = (
    API_ROOT
    /
    "artifacts/adaptation/training/"
    "datalens_semantic_qlora_v0.4_training_v0.1_manifest.json"
)


RECEIPT = (
    API_ROOT
    /
    "artifacts/adaptation/training/"
    "datalens_semantic_qlora_v0.4_training_v0.1_receipt.json"
)


# ============================================================
# HELPERS
# ============================================================


def project(
    *,
    contract_path: Path = CONTRACT,
    freeze_path: Path = FREEZE,
    manifest_path: Path = MANIFEST,
    receipt_path: Path = RECEIPT,
):

    return project_qlora_adapter_artifact(
        experiment_contract_path=
            contract_path,

        experiment_contract_freeze_path=
            freeze_path,

        training_manifest_path=
            manifest_path,

        training_receipt_path=
            receipt_path,
    )


def expect_projection_error(
    *,
    contract_path: Path = CONTRACT,
    freeze_path: Path = FREEZE,
    manifest_path: Path = MANIFEST,
    receipt_path: Path = RECEIPT,
) -> None:

    try:
        project(
            contract_path=
                contract_path,

            freeze_path=
                freeze_path,

            manifest_path=
                manifest_path,

            receipt_path=
                receipt_path,
        )

    except QLoRALifecycleProjectionError:
        return


    raise AssertionError(
        "Expected QLoRA lifecycle projection failure."
    )


# ============================================================
# ACCEPTANCE
# ============================================================


def main(
) -> None:

    assert (
        QLORA_LIFECYCLE_PROJECTION_RULE_VERSION
        ==
        "qlora_lifecycle_projection_v0.1"
    )


    record = project()


    assert (
        record.artifact_id
        ==
        (
            "artifact:adapter:"
            "0351980df6d86096195c0971deb30c72"
        )
    )


    assert (
        record.artifact_family
        ==
        "adapter"
    )


    assert (
        record.experiment_id
        ==
        "datalens-semantic-qlora-v0.4"
    )


    assert (
        record.artifact_format
        ==
        "peft_adapter_bundle"
    )


    assert (
        record.artifact_path
        ==
        (
            "artifacts/adaptation/adapters/"
            "datalens_semantic_qlora_v0.4_adapter"
        )
    )


    assert (
        record.artifact_file_bytes
        ==
        119280250
    )


    assert (
        record.artifact_sha256
        ==
        (
            "0351980df6d86096195c0971deb30c725"
            "e155c71aa5de8054b2b37fa42090716"
        )
    )


    assert (
        record.created_at_utc
        ==
        "2026-08-30T01:19:36Z"
    )


    assert (
        record.source_contract_kind
        ==
        "qlora_experiment_contract_v0.1"
    )


    contract_payload = json.loads(
        CONTRACT.read_text(
            encoding="utf-8-sig"
        )
    )


    expected_semantic_sha = (
        source_contract_sha256(
            contract_payload
        )
    )


    assert (
        record.source_contract_sha256
        ==
        expected_semantic_sha
    )


    repeated = project()


    assert (
        repeated
        ==
        record
    )


    # ========================================================
    # FAIL-CLOSED MUTATION TESTS
    # ========================================================


    with tempfile.TemporaryDirectory() as raw_temp:

        temp = Path(
            raw_temp
        )


        # ----------------------------------------------------
        # Failed training receipt
        # ----------------------------------------------------

        bad_receipt = json.loads(
            RECEIPT.read_text(
                encoding="utf-8-sig"
            )
        )

        bad_receipt[
            "passed"
        ] = False


        bad_receipt_path = (
            temp
            /
            "bad_receipt.json"
        )


        bad_receipt_path.write_text(
            json.dumps(
                bad_receipt,
                indent=2,
                sort_keys=True,
            )
            +
            "\n",
            encoding="utf-8",
        )


        expect_projection_error(
            receipt_path=
                bad_receipt_path
        )


        # ----------------------------------------------------
        # Contract mutation must break frozen byte authority
        # ----------------------------------------------------

        bad_contract = json.loads(
            CONTRACT.read_text(
                encoding="utf-8-sig"
            )
        )

        bad_contract[
            "experiment_id"
        ] = (
            "datalens-semantic-qlora-v0.4-mutated"
        )


        bad_contract_path = (
            temp
            /
            "bad_contract.json"
        )


        bad_contract_path.write_text(
            json.dumps(
                bad_contract,
                indent=2,
                sort_keys=True,
            )
            +
            "\n",
            encoding="utf-8",
        )


        expect_projection_error(
            contract_path=
                bad_contract_path
        )


        # ----------------------------------------------------
        # Invalid bundle SHA
        # ----------------------------------------------------

        bad_bundle_receipt = json.loads(
            RECEIPT.read_text(
                encoding="utf-8-sig"
            )
        )


        bad_bundle_receipt[
            "adapter"
        ][
            "bundle_sha256"
        ] = "not-a-sha"


        bad_bundle_receipt_path = (
            temp
            /
            "bad_bundle_receipt.json"
        )


        bad_bundle_receipt_path.write_text(
            json.dumps(
                bad_bundle_receipt,
                indent=2,
                sort_keys=True,
            )
            +
            "\n",
            encoding="utf-8",
        )


        expect_projection_error(
            receipt_path=
                bad_bundle_receipt_path
        )


    print(
        "Frozen QLoRA authority projection         PASS"
    )

    print(
        "Deterministic adapter artifact ID         PASS"
    )

    print(
        "Historical bundle SHA binding             PASS"
    )

    print(
        "Historical bundle size binding            PASS"
    )

    print(
        "Manifest adapter-path authority           PASS"
    )

    print(
        "Frozen contract byte authority            PASS"
    )

    print(
        "Semantic source-contract fingerprint      PASS"
    )

    print(
        "Deterministic repeated projection         PASS"
    )

    print(
        "Failed receipt rejected                   PASS"
    )

    print(
        "Mutated frozen contract rejected          PASS"
    )

    print(
        "Invalid bundle SHA rejected               PASS"
    )

    print()
    print(
        "Source contract semantic SHA256:",
        record.source_contract_sha256,
    )

    print()
    print(
        "QLoRA Lifecycle Projection v0.1: PASS"
    )


if __name__ == "__main__":
    main()

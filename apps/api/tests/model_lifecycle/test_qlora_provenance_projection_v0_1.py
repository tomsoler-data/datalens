from __future__ import annotations


import json
import tempfile


from pathlib import Path


from app.model_lifecycle.qlora_projection import (
    project_qlora_adapter_artifact,
)

from app.model_lifecycle.qlora_provenance_projection import (
    QLORA_PROVENANCE_PROJECTION_RULE_VERSION,
    QLoRAProvenanceProjectionError,
    project_qlora_adapter_provenance,
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

    return project_qlora_adapter_provenance(
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

    except QLoRAProvenanceProjectionError:
        return


    raise AssertionError(
        (
            "Expected QLoRA provenance "
            "projection failure."
        )
    )


# ============================================================
# ACCEPTANCE
# ============================================================


def main(
) -> None:

    assert (
        QLORA_PROVENANCE_PROJECTION_RULE_VERSION
        ==
        "qlora_provenance_projection_v0.1"
    )


    provenance = project()


    artifact = (
        project_qlora_adapter_artifact(
            experiment_contract_path=
                CONTRACT,

            experiment_contract_freeze_path=
                FREEZE,

            training_manifest_path=
                MANIFEST,

            training_receipt_path=
                RECEIPT,
        )
    )


    assert (
        provenance.artifact_id
        ==
        artifact.artifact_id
    )


    assert (
        provenance.artifact_family
        ==
        "adapter"
    )


    assert (
        provenance.experiment_id
        ==
        "datalens-semantic-qlora-v0.4"
    )


    assert (
        provenance.producer_family
        ==
        "llm_adaptation"
    )


    assert (
        provenance.execution_id
        ==
        (
            "training-run:"
            "datalens-semantic-qlora:v0.4:0001"
        )
    )


    assert (
        provenance.source_contract_kind
        ==
        "qlora_experiment_contract_v0.1"
    )


    assert (
        provenance.source_contract_sha256
        ==
        (
            "a294e42463da513f3c6cfb2f80b3fd19"
            "01d75f02bb50b1c8a664c0f4c50770ee"
        )
    )


    assert (
        provenance.created_at_utc
        ==
        "2026-08-30T01:19:36Z"
    )


    assert len(
        provenance.evidence
    ) == 5


    evidence_by_kind = {
        binding.evidence_kind:
            binding

        for binding
        in provenance.evidence
    }


    assert set(
        evidence_by_kind
    ) == {
        "source_contract_frozen",
        "source_contract_freeze",
        "training_manifest",
        "training_receipt",
        "training_dataset",
    }


    assert (
        evidence_by_kind[
            "source_contract_frozen"
        ].authority_sha256
        ==
        (
            "22f3c38d4165dc34d9a210605ebfb65c"
            "1c50c5839ed6607a21206fbe8313a287"
        )
    )


    assert (
        evidence_by_kind[
            "source_contract_freeze"
        ].authority_sha256
        ==
        (
            "922cd45c7a84d1c5da54a86ede3b6792"
            "d4abe14b48fbbdd485cf737fcd148321"
        )
    )


    assert (
        evidence_by_kind[
            "training_manifest"
        ].authority_sha256
        ==
        (
            "824770d7827a732df78121fe4fc515f8"
            "3bfeb8314f524188ce5fe89f43730756"
        )
    )


    assert (
        evidence_by_kind[
            "training_dataset"
        ].authority_sha256
        ==
        (
            "4fd00586f2d53d6de57f5cbc5f1d7bfb"
            "2e512960e60b30c28596aaefbac322b7"
        )
    )


    assert len(
        evidence_by_kind[
            "training_receipt"
        ].authority_sha256
    ) == 64


    repeated = project()


    assert (
        repeated
        ==
        provenance
    )


    assert (
        repeated.provenance_id
        ==
        provenance.provenance_id
    )


    # ========================================================
    # FAIL-CLOSED MUTATION TEST
    # ========================================================


    with tempfile.TemporaryDirectory() as raw_temp:

        temp = Path(
            raw_temp
        )


        bad_manifest = json.loads(
            MANIFEST.read_text(
                encoding="utf-8-sig"
            )
        )


        bad_manifest[
            "dataset"
        ][
            "dataset_sha256"
        ] = "0" * 64


        bad_manifest_path = (
            temp
            /
            "bad_manifest.json"
        )


        bad_manifest_path.write_text(
            json.dumps(
                bad_manifest,
                indent=2,
                sort_keys=True,
            )
            +
            "\n",
            encoding="utf-8",
        )


        expect_projection_error(
            manifest_path=
                bad_manifest_path
        )


        bad_freeze = json.loads(
            FREEZE.read_text(
                encoding="utf-8-sig"
            )
        )


        bad_freeze[
            "contract_sha256"
        ] = "1" * 64


        bad_freeze_path = (
            temp
            /
            "bad_freeze.json"
        )


        bad_freeze_path.write_text(
            json.dumps(
                bad_freeze,
                indent=2,
                sort_keys=True,
            )
            +
            "\n",
            encoding="utf-8",
        )


        expect_projection_error(
            freeze_path=
                bad_freeze_path
        )


    print(
        "QLoRA artifact/provenance binding        PASS"
    )

    print(
        "LLM adaptation producer identity         PASS"
    )

    print(
        "Training execution identity              PASS"
    )

    print(
        "Semantic contract SHA binding            PASS"
    )

    print(
        "Historical contract SHA evidence         PASS"
    )

    print(
        "Contract freeze SHA evidence             PASS"
    )

    print(
        "Training manifest SHA evidence           PASS"
    )

    print(
        "Training receipt SHA evidence            PASS"
    )

    print(
        "Training dataset SHA evidence            PASS"
    )

    print(
        "Deterministic provenance ID              PASS"
    )

    print(
        "Deterministic repeated projection        PASS"
    )

    print(
        "Mutated manifest rejected                PASS"
    )

    print(
        "Mutated freeze rejected                  PASS"
    )

    print()
    print(
        "Provenance ID:",
        provenance.provenance_id,
    )

    print(
        "Training receipt SHA256:",
        evidence_by_kind[
            "training_receipt"
        ].authority_sha256,
    )

    print()
    print(
        "QLoRA Provenance Projection v0.1: PASS"
    )


if __name__ == "__main__":
    main()

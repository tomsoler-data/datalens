from __future__ import annotations


import json
import tempfile


from pathlib import Path


from app.model_lifecycle.qlora_projection import (
    project_qlora_adapter_artifact,
)

from app.model_lifecycle.qlora_provenance_projection import (
    project_qlora_adapter_provenance,
)

from app.model_lifecycle.qlora_hospital_evaluation_projection import (
    HOSPITAL_EVALUATION_ID,
    QLORA_HOSPITAL_EVALUATION_PROJECTION_RULE_VERSION,
    QLoRAHospitalEvaluationProjectionError,
    project_qlora_hospital_evaluation_decision,
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


TRAINING_RECEIPT = (
    API_ROOT
    /
    "artifacts/adaptation/training/"
    "datalens_semantic_qlora_v0.4_training_v0.1_receipt.json"
)


EVALUATION_REPORT = (
    API_ROOT
    /
    "artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_report.json"
)


EVALUATION_RECEIPT = (
    API_ROOT
    /
    "artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_receipt.json"
)


EXPECTED_REPORT_SHA = (
    "6129dd58c818d59ff888daf06592ba13"
    "be90b8a40a2acf9aadbab5de5c8f07b1"
)


EXPECTED_RECEIPT_SHA = (
    "4e23a1df35298750429f39363b7cf119"
    "a6ad057b29b7bb4fbe83decbe92923cc"
)


EXPECTED_DECISION_ID = (
    "decision:evaluation:"
    "a84d46c046a5ee2f11b60c4e95f10c5c"
)


# ============================================================
# HELPERS
# ============================================================


def lifecycle_identity(
):

    artifact = (
        project_qlora_adapter_artifact(
            experiment_contract_path=
                CONTRACT,

            experiment_contract_freeze_path=
                FREEZE,

            training_manifest_path=
                MANIFEST,

            training_receipt_path=
                TRAINING_RECEIPT,
        )
    )


    provenance = (
        project_qlora_adapter_provenance(
            experiment_contract_path=
                CONTRACT,

            experiment_contract_freeze_path=
                FREEZE,

            training_manifest_path=
                MANIFEST,

            training_receipt_path=
                TRAINING_RECEIPT,
        )
    )


    return (
        artifact,
        provenance,
    )


def project(
    *,
    artifact=None,
    provenance=None,
    report_path: Path = EVALUATION_REPORT,
    receipt_path: Path = EVALUATION_RECEIPT,
):

    if (
        artifact is None
        or
        provenance is None
    ):
        (
            default_artifact,
            default_provenance,
        ) = lifecycle_identity()


        if artifact is None:
            artifact = default_artifact


        if provenance is None:
            provenance = default_provenance


    return (
        project_qlora_hospital_evaluation_decision(
            artifact=
                artifact,

            provenance=
                provenance,

            evaluation_report_path=
                report_path,

            evaluation_receipt_path=
                receipt_path,
        )
    )


def expect_projection_error(
    **kwargs,
) -> None:

    try:
        project(
            **kwargs
        )

    except QLoRAHospitalEvaluationProjectionError:
        return


    raise AssertionError(
        "Expected Hospital lifecycle projection failure."
    )


# ============================================================
# ACCEPTANCE
# ============================================================


def main(
) -> None:

    assert (
        QLORA_HOSPITAL_EVALUATION_PROJECTION_RULE_VERSION
        ==
        "qlora_hospital_evaluation_projection_v0.1"
    )


    assert (
        HOSPITAL_EVALUATION_ID
        ==
        "hospital-independent-evaluation-v0.1"
    )


    # --------------------------------------------------------
    # OFFICIAL PROJECTION
    # --------------------------------------------------------

    record = project()


    assert (
        record.decision_id
        ==
        EXPECTED_DECISION_ID
    )


    assert (
        record.artifact_id
        ==
        (
            "artifact:adapter:"
            "0351980df6d86096195c0971deb30c72"
        )
    )


    assert (
        record.provenance_id
        ==
        (
            "provenance:llm_adaptation:"
            "9fd6ec28f4c2953010e43e2a3414d9af"
        )
    )


    assert (
        record.experiment_id
        ==
        "datalens-semantic-qlora-v0.4"
    )


    assert (
        record.evaluation_id
        ==
        "hospital-independent-evaluation-v0.1"
    )


    assert (
        record.evaluation_status
        ==
        "completed"
    )


    assert (
        record.promotion_eligible
        is False
    )


    assert (
        record.promotion_decision
        ==
        "not_promoted"
    )


    assert (
        record.evaluated_at_utc
        ==
        "2026-09-12T15:26:45Z"
    )


    assert (
        record.source_receipt_sha256
        ==
        EXPECTED_RECEIPT_SHA
    )


    assert (
        len(
            record.evidence
        )
        ==
        2
    )


    evidence_by_kind = {
        binding.evidence_kind:
            binding
        for binding
        in record.evidence
    }


    assert (
        evidence_by_kind[
            "evaluation_report"
        ].authority_sha256
        ==
        EXPECTED_REPORT_SHA
    )


    assert (
        evidence_by_kind[
            "evaluation_receipt"
        ].authority_sha256
        ==
        EXPECTED_RECEIPT_SHA
    )


    print(
        "Official Hospital decision projection     PASS"
    )

    print(
        "Registered artifact identity binding      PASS"
    )

    print(
        "Registered provenance identity binding    PASS"
    )

    print(
        "Hospital report SHA binding               PASS"
    )

    print(
        "Hospital receipt SHA binding              PASS"
    )

    print(
        "Completed evaluation timestamp binding    PASS"
    )

    print(
        "Promotion decision = not_promoted         PASS"
    )


    # --------------------------------------------------------
    # DETERMINISM
    # --------------------------------------------------------

    second = project()


    assert (
        second
        ==
        record
    )


    print(
        "Deterministic repeated projection         PASS"
    )


    # --------------------------------------------------------
    # MUTATED REPORT REJECTED
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_root = Path(
            temp_dir
        )

        report_payload = json.loads(
            EVALUATION_REPORT.read_text(
                encoding="utf-8"
            )
        )


        report_payload[
            "promotion_eligible"
        ] = True


        mutated_report = (
            temp_root
            /
            "mutated_report.json"
        )


        mutated_report.write_text(
            json.dumps(
                report_payload,
                indent=2,
                sort_keys=True,
            )
            +
            "\n",
            encoding="utf-8",
        )


        expect_projection_error(
            report_path=
                mutated_report
        )


    print(
        "Mutated Hospital report rejected         PASS"
    )


    # --------------------------------------------------------
    # MUTATED RECEIPT REJECTED
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_root = Path(
            temp_dir
        )

        receipt_payload = json.loads(
            EVALUATION_RECEIPT.read_text(
                encoding="utf-8"
            )
        )


        receipt_payload[
            "promotion_eligible"
        ] = True


        mutated_receipt = (
            temp_root
            /
            "mutated_receipt.json"
        )


        mutated_receipt.write_text(
            json.dumps(
                receipt_payload,
                indent=2,
                sort_keys=True,
            )
            +
            "\n",
            encoding="utf-8",
        )


        expect_projection_error(
            receipt_path=
                mutated_receipt
        )


    print(
        "Mutated Hospital receipt rejected        PASS"
    )


    # --------------------------------------------------------
    # WRONG LIFECYCLE ARTIFACT REJECTED
    # --------------------------------------------------------

    (
        artifact,
        provenance,
    ) = lifecycle_identity()


    wrong_artifact = (
        artifact.model_copy(
            update={
                "artifact_id":
                    (
                        "artifact:adapter:"
                        +
                        "0" * 32
                    )
            }
        )
    )


    expect_projection_error(
        artifact=
            wrong_artifact,

        provenance=
            provenance,
    )


    print(
        "Wrong lifecycle artifact rejected        PASS"
    )


    # --------------------------------------------------------
    # WRONG PROVENANCE REJECTED
    # --------------------------------------------------------

    wrong_provenance = (
        provenance.model_copy(
            update={
                "provenance_id":
                    (
                        "provenance:llm_adaptation:"
                        +
                        "0" * 32
                    )
            }
        )
    )


    expect_projection_error(
        artifact=
            artifact,

        provenance=
            wrong_provenance,
    )


    print(
        "Wrong lifecycle provenance rejected      PASS"
    )


    # --------------------------------------------------------
    # RECORD OWNS NO PROTECTED PAYLOAD
    # --------------------------------------------------------

    payload = (
        record.model_dump(
            mode="json"
        )
    )


    forbidden_fields = {
        "predictions",
        "gold",
        "gold_relation",
        "gold_reason",
        "holdout_cases",
        "model_outputs",
    }


    assert (
        set(
            payload.keys()
        )
        &
        forbidden_fields
        ==
        set()
    )


    print(
        "No protected payload projected           PASS"
    )


    print()
    print(
        "Decision ID:",
        record.decision_id,
    )

    print(
        "Promotion decision:",
        record.promotion_decision,
    )

    print()
    print(
        "QLoRA Hospital Evaluation Decision Projection v0.1: PASS"
    )


if __name__ == "__main__":
    main()

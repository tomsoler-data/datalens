from __future__ import annotations


import hashlib
import json


from pathlib import Path


from typing import Any


from app.model_lifecycle.artifact_contracts import (
    ModelLifecycleArtifactRecord,
)

from app.model_lifecycle.evaluation_decision_contracts import (
    ModelLifecycleDecisionEvidenceBinding,
    ModelLifecycleEvaluationDecisionRecord,
    build_model_lifecycle_evaluation_decision_id,
)

from app.model_lifecycle.provenance_contracts import (
    ModelLifecycleProvenanceRecord,
)


# ============================================================
# VERSION
# ============================================================


QLORA_HOSPITAL_EVALUATION_PROJECTION_RULE_VERSION = (
    "qlora_hospital_evaluation_projection_v0.1"
)


# ============================================================
# FROZEN EVALUATION IDENTITY
# ============================================================


EXPECTED_EXPERIMENT_ID = (
    "datalens-semantic-qlora-v0.4"
)


EXPECTED_ARTIFACT_ID = (
    "artifact:adapter:"
    "0351980df6d86096195c0971deb30c72"
)


EXPECTED_PROVENANCE_ID = (
    "provenance:llm_adaptation:"
    "9fd6ec28f4c2953010e43e2a3414d9af"
)


HOSPITAL_EVALUATION_ID = (
    "hospital-independent-evaluation-v0.1"
)


EXPECTED_REPORT_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_report_v0.1"
)


EXPECTED_RECEIPT_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_receipt_v0.1"
)


EXPECTED_SCORING_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_scoring_v0.1"
)


# ============================================================
# ERROR
# ============================================================


class QLoRAHospitalEvaluationProjectionError(
    RuntimeError
):
    pass


# ============================================================
# HELPERS
# ============================================================


def _load_json_object(
    path: Path,
) -> dict[
    str,
    Any,
]:

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise QLoRAHospitalEvaluationProjectionError(
            f"Could not read evaluation authority: {path}"
        ) from error


    if not isinstance(
        payload,
        dict,
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Evaluation authority must be a JSON object."
        )


    return payload


def _sha256(
    path: Path,
) -> str:

    try:
        data = path.read_bytes()

    except OSError as error:
        raise QLoRAHospitalEvaluationProjectionError(
            f"Could not read evaluation authority bytes: {path}"
        ) from error


    return hashlib.sha256(
        data
    ).hexdigest()


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
        raise QLoRAHospitalEvaluationProjectionError(
            f"{field_name} cannot be empty."
        )


    return normalized


# ============================================================
# PROJECTION
# ============================================================


def project_qlora_hospital_evaluation_decision(
    *,
    artifact: ModelLifecycleArtifactRecord,
    provenance: ModelLifecycleProvenanceRecord,
    evaluation_report_path: Path,
    evaluation_receipt_path: Path,
) -> ModelLifecycleEvaluationDecisionRecord:
    """
    Project the already-completed QLoRA v0.4 Hospital evaluation
    authorities into the shared immutable lifecycle decision model.

    This function does not:
    - load model or adapter weights;
    - read protected holdout cases;
    - read gold labels;
    - generate predictions;
    - score or re-score the benchmark;
    - mutate lifecycle registry state;
    - mutate historical evaluation evidence.
    """

    # ========================================================
    # REGISTERED LIFECYCLE IDENTITY BINDING
    # ========================================================

    if (
        artifact.artifact_id
        !=
        EXPECTED_ARTIFACT_ID
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Unexpected QLoRA v0.4 artifact identity."
        )


    if (
        provenance.provenance_id
        !=
        EXPECTED_PROVENANCE_ID
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Unexpected QLoRA v0.4 provenance identity."
        )


    if (
        artifact.experiment_id
        !=
        EXPECTED_EXPERIMENT_ID
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Unexpected QLoRA v0.4 experiment identity."
        )


    if (
        provenance.experiment_id
        !=
        EXPECTED_EXPERIMENT_ID
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Unexpected provenance experiment identity."
        )


    if (
        provenance.artifact_id
        !=
        artifact.artifact_id
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Artifact/provenance identity mismatch."
        )


    if (
        provenance.artifact_family
        !=
        artifact.artifact_family
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Artifact/provenance family mismatch."
        )


    if (
        artifact.artifact_family
        !=
        "adapter"
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital QLoRA evaluation requires adapter artifact."
        )


    if (
        provenance.producer_family
        !=
        "llm_adaptation"
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital QLoRA evaluation requires llm_adaptation provenance."
        )


    # ========================================================
    # LOAD ONLY REPORT + RECEIPT
    # ========================================================

    report = _load_json_object(
        evaluation_report_path
    )

    receipt = _load_json_object(
        evaluation_receipt_path
    )


    report_sha = _sha256(
        evaluation_report_path
    )

    receipt_sha = _sha256(
        evaluation_receipt_path
    )


    # ========================================================
    # REPORT CONTRACT
    # ========================================================

    if (
        report.get(
            "rule_version"
        )
        !=
        EXPECTED_REPORT_RULE_VERSION
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Unexpected Hospital evaluation report rule."
        )


    scoring = report.get(
        "scoring"
    )


    if not isinstance(
        scoring,
        dict,
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital report scoring object is missing."
        )


    if (
        scoring.get(
            "rule_version"
        )
        !=
        EXPECTED_SCORING_RULE_VERSION
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Unexpected Hospital scoring rule."
        )


    if not isinstance(
        report.get(
            "promotion_eligible"
        ),
        bool,
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Report promotion_eligible must be boolean."
        )


    if not isinstance(
        scoring.get(
            "promotion_eligible"
        ),
        bool,
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Scoring promotion_eligible must be boolean."
        )


    if (
        report[
            "promotion_eligible"
        ]
        !=
        scoring[
            "promotion_eligible"
        ]
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Report/scoring promotion decision mismatch."
        )


    report_case_count = (
        scoring.get(
            "case_count"
        )
    )


    if (
        isinstance(
            report_case_count,
            bool,
        )
        or
        not isinstance(
            report_case_count,
            int,
        )
        or
        report_case_count <= 0
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital report case_count is invalid."
        )


    # ========================================================
    # RECEIPT CONTRACT
    # ========================================================

    if (
        receipt.get(
            "rule_version"
        )
        !=
        EXPECTED_RECEIPT_RULE_VERSION
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Unexpected Hospital evaluation receipt rule."
        )


    if (
        receipt.get(
            "status"
        )
        !=
        "completed"
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital evaluation is not completed."
        )


    if not isinstance(
        receipt.get(
            "promotion_eligible"
        ),
        bool,
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Receipt promotion_eligible must be boolean."
        )


    if (
        receipt[
            "promotion_eligible"
        ]
        !=
        report[
            "promotion_eligible"
        ]
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Report/receipt promotion decision mismatch."
        )


    if (
        receipt.get(
            "report_sha256"
        )
        !=
        report_sha
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital receipt does not bind report bytes."
        )


    if (
        receipt.get(
            "case_count"
        )
        !=
        report_case_count
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital report/receipt case_count mismatch."
        )


    if (
        receipt.get(
            "protocol_sha256"
        )
        !=
        report.get(
            "protocol_sha256"
        )
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital protocol authority mismatch."
        )


    if (
        receipt.get(
            "scorer_sha256"
        )
        !=
        report.get(
            "scorer_sha256"
        )
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital scorer authority mismatch."
        )


    if (
        receipt.get(
            "execution_authority_commit"
        )
        !=
        report.get(
            "execution_authority_commit"
        )
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital execution authority mismatch."
        )


    if (
        receipt.get(
            "claimed_at_utc"
        )
        !=
        report.get(
            "claimed_at_utc"
        )
    ):
        raise QLoRAHospitalEvaluationProjectionError(
            "Hospital claimed timestamp mismatch."
        )


    completed_at_utc = _required_text(
        receipt.get(
            "completed_at_utc"
        ),
        field_name="receipt.completed_at_utc",
    )


    # ========================================================
    # IMMUTABLE DECISION EVIDENCE
    # ========================================================

    report_binding = (
        ModelLifecycleDecisionEvidenceBinding(
            evidence_kind=
                "evaluation_report",

            authority_id=(
                EXPECTED_EXPERIMENT_ID
                +
                ":"
                +
                HOSPITAL_EVALUATION_ID
                +
                ":report"
            ),

            authority_sha256=
                report_sha,
        )
    )


    receipt_binding = (
        ModelLifecycleDecisionEvidenceBinding(
            evidence_kind=
                "evaluation_receipt",

            authority_id=(
                EXPECTED_EXPERIMENT_ID
                +
                ":"
                +
                HOSPITAL_EVALUATION_ID
                +
                ":receipt"
            ),

            authority_sha256=
                receipt_sha,
        )
    )


    promotion_eligible = (
        receipt[
            "promotion_eligible"
        ]
    )


    promotion_decision = (
        "promoted"
        if promotion_eligible
        else
        "not_promoted"
    )


    decision_id = (
        build_model_lifecycle_evaluation_decision_id(
            artifact_id=
                artifact.artifact_id,

            evaluation_id=
                HOSPITAL_EVALUATION_ID,

            source_receipt_sha256=
                receipt_sha,
        )
    )


    return (
        ModelLifecycleEvaluationDecisionRecord(
            decision_id=
                decision_id,

            artifact_id=
                artifact.artifact_id,

            provenance_id=
                provenance.provenance_id,

            experiment_id=
                artifact.experiment_id,

            evaluation_id=
                HOSPITAL_EVALUATION_ID,

            evaluation_status=
                "completed",

            promotion_eligible=
                promotion_eligible,

            promotion_decision=
                promotion_decision,

            evaluated_at_utc=
                completed_at_utc,

            source_receipt_sha256=
                receipt_sha,

            evidence=(
                report_binding,
                receipt_binding,
            ),
        )
    )

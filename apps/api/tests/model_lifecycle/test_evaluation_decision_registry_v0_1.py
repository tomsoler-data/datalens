from __future__ import annotations


import hashlib
import json
import shutil
import sqlite3
import tempfile


from pathlib import Path


from app.model_lifecycle.evaluation_decision_contracts import (
    ModelLifecycleDecisionEvidenceBinding,
    ModelLifecycleEvaluationDecisionRecord,
    build_model_lifecycle_evaluation_decision_id,
)

from app.model_lifecycle.evaluation_decision_registry import (
    MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_RULE_VERSION,
    ModelLifecycleEvaluationDecisionRegistryConflictError,
    ModelLifecycleEvaluationDecisionRegistryCorruptionError,
    ModelLifecycleEvaluationDecisionRegistryError,
    get_model_lifecycle_evaluation_decision,
    list_model_lifecycle_evaluation_decisions,
    register_model_lifecycle_evaluation_decision,
    resolve_model_lifecycle_evaluation_decision_registry_path,
)

from app.model_lifecycle.qlora_hospital_evaluation_projection import (
    project_qlora_hospital_evaluation_decision,
)

from app.model_lifecycle.qlora_projection import (
    project_qlora_adapter_artifact,
)

from app.model_lifecycle.qlora_provenance_projection import (
    project_qlora_adapter_provenance,
)

from app.model_lifecycle.registry import (
    register_model_lifecycle_entry,
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


HOSPITAL_REPORT = (
    API_ROOT
    /
    "artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_report.json"
)


HOSPITAL_RECEIPT = (
    API_ROOT
    /
    "artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_receipt.json"
)


# ============================================================
# HELPERS
# ============================================================


def sha256(
    path: Path,
) -> str:

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


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


def official_decision(
    artifact,
    provenance,
):

    return (
        project_qlora_hospital_evaluation_decision(
            artifact=
                artifact,

            provenance=
                provenance,

            evaluation_report_path=
                HOSPITAL_REPORT,

            evaluation_receipt_path=
                HOSPITAL_RECEIPT,
        )
    )


def expect_registry_error(
    function,
) -> None:

    try:
        function()

    except ModelLifecycleEvaluationDecisionRegistryError:
        return


    raise AssertionError(
        "Expected decision registry failure."
    )


# ============================================================
# ACCEPTANCE
# ============================================================


def main(
) -> None:

    assert (
        MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_RULE_VERSION
        ==
        "model_lifecycle_evaluation_decision_registry_v0.1"
    )


    (
        artifact,
        provenance,
    ) = lifecycle_identity()


    decision = official_decision(
        artifact,
        provenance,
    )


    with tempfile.TemporaryDirectory() as temp_dir:

        root = Path(
            temp_dir
        )


        identity_registry = (
            root
            /
            "identity.sqlite3"
        )


        decision_registry = (
            root
            /
            "decisions.sqlite3"
        )


        # ====================================================
        # BUILD TEMP IDENTITY AUTHORITY
        # ====================================================

        register_model_lifecycle_entry(
            artifact=
                artifact,

            provenance=
                provenance,

            registry_path=
                identity_registry,
        )


        identity_sha_before = sha256(
            identity_registry
        )


        print(
            "Temporary identity registry            PASS"
        )


        # ====================================================
        # SEPARATE STORE PATH
        # ====================================================

        resolved = (
            resolve_model_lifecycle_evaluation_decision_registry_path(
                decision_registry
            )
        )


        assert (
            resolved
            ==
            decision_registry
        )


        expect_registry_error(
            lambda:
                register_model_lifecycle_evaluation_decision(
                    decision=
                        decision,

                    lifecycle_registry_path=
                        identity_registry,

                    decision_registry_path=
                        identity_registry,
                )
        )


        print(
            "Identity/decision store separation      PASS"
        )


        # ====================================================
        # REGISTER
        # ====================================================

        registered = (
            register_model_lifecycle_evaluation_decision(
                decision=
                    decision,

                lifecycle_registry_path=
                    identity_registry,

                decision_registry_path=
                    decision_registry,
            )
        )


        assert (
            registered
            ==
            decision
        )


        assert decision_registry.is_file()


        print(
            "Metadata-only decision registration     PASS"
        )


        # ====================================================
        # IDENTITY REGISTRY REMAINS BYTE IDENTICAL
        # ====================================================

        identity_sha_after = sha256(
            identity_registry
        )


        assert (
            identity_sha_after
            ==
            identity_sha_before
        )


        print(
            "Identity registry remains byte-identical PASS"
        )


        # ====================================================
        # IDEMPOTENT REGISTRATION
        # ====================================================

        repeated = (
            register_model_lifecycle_evaluation_decision(
                decision=
                    decision,

                lifecycle_registry_path=
                    identity_registry,

                decision_registry_path=
                    decision_registry,
            )
        )


        assert (
            repeated
            ==
            decision
        )


        connection = sqlite3.connect(
            decision_registry
        )


        try:
            count = (
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM model_lifecycle_evaluation_decisions
                    """
                )
                .fetchone()[
                    0
                ]
            )

        finally:
            connection.close()


        assert count == 1


        print(
            "Idempotent decision registration        PASS"
        )


        # ====================================================
        # GET
        # ====================================================

        fetched = (
            get_model_lifecycle_evaluation_decision(
                decision.decision_id,
                decision_registry_path=
                    decision_registry,
            )
        )


        assert (
            fetched
            ==
            decision
        )


        print(
            "Decision get semantics                  PASS"
        )


        # ====================================================
        # LIST
        # ====================================================

        listed = (
            list_model_lifecycle_evaluation_decisions(
                decision_registry_path=
                    decision_registry,
            )
        )


        assert (
            listed
            ==
            (
                decision,
            )
        )


        print(
            "Deterministic decision list             PASS"
        )


        # ====================================================
        # UNREGISTERED IDENTITY REJECTED
        # ====================================================

        missing_identity = (
            root
            /
            "missing_identity.sqlite3"
        )


        expect_registry_error(
            lambda:
                register_model_lifecycle_evaluation_decision(
                    decision=
                        decision,

                    lifecycle_registry_path=
                        missing_identity,

                    decision_registry_path=
                        (
                            root
                            /
                            "missing_identity_decisions.sqlite3"
                        ),
                )
        )


        print(
            "Unregistered artifact rejected          PASS"
        )


        # ====================================================
        # CONFLICT FOR SAME ARTIFACT + EVALUATION
        # ====================================================

        fake_receipt_sha = (
            "0"
            *
            64
        )


        fake_report_sha = (
            "1"
            *
            64
        )


        conflicting_decision = (
            ModelLifecycleEvaluationDecisionRecord(
                decision_id=
                    build_model_lifecycle_evaluation_decision_id(
                        artifact_id=
                            artifact.artifact_id,

                        evaluation_id=
                            decision.evaluation_id,

                        source_receipt_sha256=
                            fake_receipt_sha,
                    ),

                artifact_id=
                    artifact.artifact_id,

                provenance_id=
                    provenance.provenance_id,

                experiment_id=
                    artifact.experiment_id,

                evaluation_id=
                    decision.evaluation_id,

                evaluation_status=
                    "completed",

                promotion_eligible=
                    False,

                promotion_decision=
                    "not_promoted",

                evaluated_at_utc=
                    decision.evaluated_at_utc,

                source_receipt_sha256=
                    fake_receipt_sha,

                evidence=(
                    ModelLifecycleDecisionEvidenceBinding(
                        evidence_kind=
                            "evaluation_report",

                        authority_id=
                            "conflicting:report",

                        authority_sha256=
                            fake_report_sha,
                    ),

                    ModelLifecycleDecisionEvidenceBinding(
                        evidence_kind=
                            "evaluation_receipt",

                        authority_id=
                            "conflicting:receipt",

                        authority_sha256=
                            fake_receipt_sha,
                    ),
                ),
            )
        )


        conflict_seen = False


        try:
            register_model_lifecycle_evaluation_decision(
                decision=
                    conflicting_decision,

                lifecycle_registry_path=
                    identity_registry,

                decision_registry_path=
                    decision_registry,
            )

        except ModelLifecycleEvaluationDecisionRegistryConflictError:
            conflict_seen = True


        assert conflict_seen


        print(
            "Conflicting evaluation decision rejected PASS"
        )


        # ====================================================
        # CORRUPTION FAIL CLOSED
        # ====================================================

        corrupt_registry = (
            root
            /
            "corrupt_decisions.sqlite3"
        )


        shutil.copyfile(
            decision_registry,
            corrupt_registry,
        )


        connection = sqlite3.connect(
            corrupt_registry
        )


        try:
            connection.execute(
                """
                UPDATE model_lifecycle_evaluation_decisions
                SET decision_json = ?
                WHERE decision_id = ?
                """,
                (
                    json.dumps(
                        {}
                    ),
                    decision.decision_id,
                ),
            )

            connection.commit()

        finally:
            connection.close()


        corruption_seen = False


        try:
            get_model_lifecycle_evaluation_decision(
                decision.decision_id,
                decision_registry_path=
                    corrupt_registry,
            )

        except ModelLifecycleEvaluationDecisionRegistryCorruptionError:
            corruption_seen = True


        assert corruption_seen


        print(
            "Decision registry corruption fail-closed PASS"
        )


        # ====================================================
        # ORIGINAL DECISION STILL VALID
        # ====================================================

        final = (
            get_model_lifecycle_evaluation_decision(
                decision.decision_id,
                decision_registry_path=
                    decision_registry,
            )
        )


        assert final == decision


        assert (
            final.promotion_decision
            ==
            "not_promoted"
        )


        assert (
            final.promotion_eligible
            is False
        )


        print(
            "Official decision semantics preserved   PASS"
        )


    print()
    print(
        "Decision ID:",
        decision.decision_id,
    )

    print(
        "Promotion decision:",
        decision.promotion_decision,
    )

    print()
    print(
        "Model Lifecycle Evaluation Decision Registry v0.1: PASS"
    )


if __name__ == "__main__":
    main()

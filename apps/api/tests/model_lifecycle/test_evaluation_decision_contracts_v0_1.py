from __future__ import annotations


from pydantic import ValidationError


from app.model_lifecycle.evaluation_decision_contracts import (
    MODEL_LIFECYCLE_EVALUATION_DECISION_RULE_VERSION,
    ModelLifecycleDecisionEvidenceBinding,
    ModelLifecycleEvaluationDecisionRecord,
    build_model_lifecycle_evaluation_decision_id,
)


ARTIFACT_ID = (
    "artifact:adapter:"
    "0351980df6d86096195c0971deb30c72"
)


PROVENANCE_ID = (
    "provenance:llm_adaptation:"
    "9fd6ec28f4c2953010e43e2a3414d9af"
)


EXPERIMENT_ID = (
    "datalens-semantic-qlora-v0.4"
)


EVALUATION_ID = (
    "hospital-independent-evaluation-v0.1"
)


REPORT_SHA = (
    "6129dd58c818d59ff888daf06592ba13"
    "be90b8a40a2acf9aadbab5de5c8f07b1"
)


RECEIPT_SHA = (
    "4e23a1df35298750429f39363b7cf119"
    "a6ad057b29b7bb4fbe83decbe92923cc"
)


def decision_id(
) -> str:

    return (
        build_model_lifecycle_evaluation_decision_id(
            artifact_id=
                ARTIFACT_ID,

            evaluation_id=
                EVALUATION_ID,

            source_receipt_sha256=
                RECEIPT_SHA,
        )
    )


def evidence(
) -> tuple[
    ModelLifecycleDecisionEvidenceBinding,
    ...,
]:

    return (
        ModelLifecycleDecisionEvidenceBinding(
            evidence_kind=
                "evaluation_report",

            authority_id=(
                "datalens-semantic-qlora-v0.4:"
                "hospital-independent-evaluation-v0.1:"
                "report"
            ),

            authority_sha256=
                REPORT_SHA,
        ),

        ModelLifecycleDecisionEvidenceBinding(
            evidence_kind=
                "evaluation_receipt",

            authority_id=(
                "datalens-semantic-qlora-v0.4:"
                "hospital-independent-evaluation-v0.1:"
                "receipt"
            ),

            authority_sha256=
                RECEIPT_SHA,
        ),
    )


def build_record(
    *,
    promotion_eligible: bool = False,
    promotion_decision: str = "not_promoted",
) -> ModelLifecycleEvaluationDecisionRecord:

    return (
        ModelLifecycleEvaluationDecisionRecord(
            decision_id=
                decision_id(),

            artifact_id=
                ARTIFACT_ID,

            provenance_id=
                PROVENANCE_ID,

            experiment_id=
                EXPERIMENT_ID,

            evaluation_id=
                EVALUATION_ID,

            evaluation_status=
                "completed",

            promotion_eligible=
                promotion_eligible,

            promotion_decision=
                promotion_decision,

            evaluated_at_utc=
                "2026-09-12T15:26:45Z",

            source_receipt_sha256=
                RECEIPT_SHA,

            evidence=
                evidence(),
        )
    )


def expect_validation_error(
    **kwargs,
) -> None:

    try:
        ModelLifecycleEvaluationDecisionRecord(
            **kwargs
        )

    except ValidationError:
        return


    raise AssertionError(
        "Expected validation failure."
    )


def main(
) -> None:

    assert (
        MODEL_LIFECYCLE_EVALUATION_DECISION_RULE_VERSION
        ==
        "model_lifecycle_evaluation_decision_v0.1"
    )


    record = build_record()


    assert (
        record.artifact_id
        ==
        ARTIFACT_ID
    )

    assert (
        record.provenance_id
        ==
        PROVENANCE_ID
    )

    assert (
        record.experiment_id
        ==
        EXPERIMENT_ID
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


    print(
        "Completed evaluation construction        PASS"
    )

    print(
        "Not-promoted decision semantics          PASS"
    )


    first = decision_id()
    second = decision_id()


    assert first == second

    assert first.startswith(
        "decision:evaluation:"
    )

    assert (
        len(first)
        ==
        len(
            "decision:evaluation:"
        )
        +
        32
    )


    print(
        "Deterministic decision identity          PASS"
    )


    immutable = False


    try:
        record.promotion_eligible = True

    except ValidationError:
        immutable = True


    assert immutable


    print(
        "Evaluation decision is immutable         PASS"
    )


    mismatch_payload = (
        record.model_dump(
            mode="python"
        )
    )

    mismatch_payload[
        "promotion_decision"
    ] = "promoted"


    expect_validation_error(
        **mismatch_payload
    )


    print(
        "Promotion decision consistency           PASS"
    )


    no_receipt_payload = (
        record.model_dump(
            mode="python"
        )
    )

    no_receipt_payload[
        "evidence"
    ] = (
        no_receipt_payload[
            "evidence"
        ][
            :1
        ]
    )


    expect_validation_error(
        **no_receipt_payload
    )


    print(
        "Receipt SHA evidence binding             PASS"
    )


    duplicate_payload = (
        record.model_dump(
            mode="python"
        )
    )

    duplicate_payload[
        "evidence"
    ] = (
        ModelLifecycleDecisionEvidenceBinding(
            evidence_kind=
                "evaluation_receipt",

            authority_id=
                "receipt:a",

            authority_sha256=
                RECEIPT_SHA,
        ),

        ModelLifecycleDecisionEvidenceBinding(
            evidence_kind=
                "evaluation_receipt",

            authority_id=
                "receipt:b",

            authority_sha256=
                RECEIPT_SHA,
        ),
    )


    expect_validation_error(
        **duplicate_payload
    )


    print(
        "Duplicate decision evidence rejected     PASS"
    )


    wrong_id_payload = (
        record.model_dump(
            mode="python"
        )
    )

    wrong_id_payload[
        "decision_id"
    ] = (
        "decision:evaluation:"
        +
        "0" * 32
    )


    expect_validation_error(
        **wrong_id_payload
    )


    print(
        "Decision identity mismatch rejected      PASS"
    )


    invalid_artifact_payload = (
        record.model_dump(
            mode="python"
        )
    )

    invalid_artifact_payload[
        "artifact_id"
    ] = "not-an-artifact-id"


    expect_validation_error(
        **invalid_artifact_payload
    )


    print(
        "Invalid artifact identity rejected       PASS"
    )


    invalid_provenance_payload = (
        record.model_dump(
            mode="python"
        )
    )

    invalid_provenance_payload[
        "provenance_id"
    ] = "not-a-provenance-id"


    expect_validation_error(
        **invalid_provenance_payload
    )


    print(
        "Invalid provenance identity rejected     PASS"
    )


    invalid_time_payload = (
        record.model_dump(
            mode="python"
        )
    )

    invalid_time_payload[
        "evaluated_at_utc"
    ] = "2026-09-12T15:26:45+02:00"


    expect_validation_error(
        **invalid_time_payload
    )


    print(
        "Non-UTC lifecycle timestamp rejected     PASS"
    )


    extra_payload = (
        record.model_dump(
            mode="python"
        )
    )

    extra_payload[
        "gold_relation"
    ] = "forbidden"


    expect_validation_error(
        **extra_payload
    )


    print(
        "Unknown decision fields fail closed      PASS"
    )


    payload_keys = set(
        record.model_dump(
            mode="json"
        )
        .keys()
    )


    forbidden_payload_fields = {
        "predictions",
        "gold",
        "gold_relation",
        "gold_reason",
        "model_outputs",
        "holdout_cases",
    }


    assert (
        payload_keys
        &
        forbidden_payload_fields
        ==
        set()
    )


    print(
        "No protected/evaluation payload owned    PASS"
    )


    print()
    print(
        "Decision ID:",
        record.decision_id,
    )

    print()
    print(
        "Model Lifecycle Evaluation Decision Contract v0.1: PASS"
    )


if __name__ == "__main__":
    main()

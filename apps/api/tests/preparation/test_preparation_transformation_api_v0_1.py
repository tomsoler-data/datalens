from __future__ import annotations

from contextlib import (
    contextmanager,
)

from types import (
    SimpleNamespace,
)

import pandas as pd

from fastapi import (
    HTTPException,
)

import app.api.preparation_transformation as transformation_api

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationStageStatus,
)

from app.preparation.transformation_approval import (
    TransformationApprovalCommand,
    TransformationApprovalDecision,
)


WORKFLOW_ID = (
    "workflow-transformation-api-test"
)


# ============================================================
# PATCH HELPER
# ============================================================

@contextmanager
def patched_globals(
    **replacements,
):

    originals = {}


    try:

        for (
            name,
            replacement,
        ) in replacements.items():

            originals[
                name
            ] = getattr(
                transformation_api,
                name,
            )


            setattr(
                transformation_api,
                name,
                replacement,
            )


        yield


    finally:

        for (
            name,
            original,
        ) in originals.items():

            setattr(
                transformation_api,
                name,
                original,
            )


# ============================================================
# DATA
# ============================================================

def source_dataframe() -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],
            }
        )
    )


def transformed_dataframe() -> pd.DataFrame:

    dataframe = (
        source_dataframe()
    )


    dataframe[
        "amount_x2"
    ] = (
        dataframe[
            "amount"
        ]
        *
        2
    )


    return (
        dataframe
    )


# ============================================================
# INTENT
# ============================================================

def arithmetic_intent_payload():

    return {
        "request_id":
            "derive-amount-x2",

        "operation":
            "derive_arithmetic",

        "output_column":
            "amount_x2",

        "left": {
            "kind":
                "column",

            "column":
                "amount",
        },

        "operator":
            "multiply",

        "right": {
            "kind":
                "literal",

            "value":
                2,
        },
    }


# ============================================================
# SESSION
# ============================================================

def session(
    *,
    clean_status=(
        PreparationStageStatus.PASSED
    ),
):

    return (
        SimpleNamespace(
            selected_analysis_dataset_ids=[
                "sales",
            ],

            snapshot=(
                SimpleNamespace(
                    stages=[
                        SimpleNamespace(
                            stage=(
                                PreparationStage.CLEAN
                            ),

                            status=(
                                clean_status
                            ),
                        ),
                    ]
                )
            ),
        )
    )


# ============================================================
# ARTIFACT
# ============================================================

def source_artifact():

    return (
        SimpleNamespace(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),

            dataset_filename=(
                "sales.csv"
            ),

            stage=(
                "clean"
            ),

            dataframe=(
                source_dataframe()
            ),

            parent_dataset_ids=(
                "sales",
            ),

            evidence_refs=(
                "cleaning:test",
            ),
        )
    )


# ============================================================
# PLAN
# ============================================================

def transformation_plan(
    *,
    human_approval_required_count: int = 1,
):

    return (
        SimpleNamespace(
            dataset_id=(
                "sales"
            ),

            dataset_filename=(
                "sales.csv"
            ),

            request_count=(
                1
            ),

            step_count=(
                1
            ),

            validated_count=(
                0
            ),

            review_required_count=(
                human_approval_required_count
            ),

            human_approval_required_count=(
                human_approval_required_count
            ),

            ready_for_approval=(
                True
            ),

            steps=[],

            notes=[],

            rule_version=(
                "transformation_planner_test_v0.1"
            ),
        )
    )


# ============================================================
# APPROVAL
# ============================================================

def approved_plan(
    *,
    ready_for_execution: bool = True,
    executable_step_count: int = 1,
    pending_count: int = 0,
    deferred_count: int = 0,
    blocked_dependency_count: int = 0,
    rejected_count: int = 0,
):

    return (
        SimpleNamespace(
            dataset_id=(
                "sales"
            ),

            dataset_filename=(
                "sales.csv"
            ),

            total_step_count=(
                1
            ),

            automatic_count=(
                0
            ),

            approved_count=(
                1
                if executable_step_count > 0
                else 0
            ),

            rejected_count=(
                rejected_count
            ),

            deferred_count=(
                deferred_count
            ),

            pending_count=(
                pending_count
            ),

            blocked_dependency_count=(
                blocked_dependency_count
            ),

            executable_step_count=(
                executable_step_count
            ),

            ready_for_execution=(
                ready_for_execution
            ),

            steps=[],

            notes=[],

            rule_version=(
                "transformation_approval_test_v0.1"
            ),
        )
    )


# ============================================================
# EXECUTION
# ============================================================

def execution_result():

    report = (
        SimpleNamespace(
            status=(
                "success"
            ),

            dataset_id=(
                "sales"
            ),

            dataset_filename=(
                "sales.csv"
            ),

            source_rows_before=(
                3
            ),

            source_rows_after=(
                3
            ),

            source_columns_before=(
                1
            ),

            source_columns_after=(
                2
            ),

            source_fingerprint_before=(
                "before"
            ),

            source_fingerprint_after=(
                "after"
            ),

            source_data_changed=(
                True
            ),

            total_step_count=(
                1
            ),

            executable_step_count=(
                1
            ),

            applied_count=(
                1
            ),

            skipped_count=(
                0
            ),

            derived_dataset_count=(
                0
            ),

            derived_dataset_ids=[],

            steps=[],

            notes=[],

            rule_version=(
                "transformation_executor_test_v0.1"
            ),
        )
    )


    return (
        SimpleNamespace(
            dataframe=(
                transformed_dataframe()
            ),

            derived_datasets={},

            report=(
                report
            ),
        )
    )


# ============================================================
# VALIDATION
# ============================================================

def validation_report(
    *,
    valid: bool = True,
):

    return (
        SimpleNamespace(
            status=(
                "passed"
                if valid
                else "failed"
            ),

            dataset_id=(
                "sales"
            ),

            dataset_filename=(
                "sales.csv"
            ),

            valid_for_downstream=(
                valid
            ),

            passed_check_count=(
                5
                if valid
                else 4
            ),

            failed_check_count=(
                0
                if valid
                else 1
            ),

            warning_count=(
                0
            ),

            validated_step_count=(
                1
            ),

            derived_dataset_count=(
                0
            ),

            step_validations=[],

            checks=[],

            notes=[],

            rule_version=(
                "post_transformation_validation_test_v0.1"
            ),
        )
    )


# ============================================================
# MATERIALIZATION
# ============================================================

def materialization_report():

    return (
        SimpleNamespace(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataset_id=(
                "sales"
            ),

            persisted_dataset_ids=(
                "sales",
            ),

            derived_dataset_ids=(),

            artifact_count=(
                1
            ),

            source_data_changed=(
                True
            ),

            materialization_kind=(
                "source_transformed"
            ),

            bridge_version=(
                "transformation_artifact_bridge_v0.1"
            ),
        )
    )


# ============================================================
# RESPONSE
# ============================================================

def fake_response(
    **kwargs,
):

    return (
        SimpleNamespace(
            **kwargs
        )
    )


# ============================================================
# COMMON PATCHES
# ============================================================

def common_patches(
    *,
    clean_status=(
        PreparationStageStatus.PASSED
    ),
):

    return {
        "get_preparation_session":
            (
                lambda workflow_id:
                    session(
                        clean_status=(
                            clean_status
                        )
                    )
            ),

        "get_preparation_artifact":
            (
                lambda **kwargs:
                    source_artifact()
            ),

        "list_preparation_artifacts":
            (
                lambda **kwargs:
                    [
                        SimpleNamespace(
                            dataset_id=(
                                "sales"
                            )
                        ),
                    ]
            ),
    }


# ============================================================
# 1. PLAN USES SERVER ARTIFACT
# ============================================================

def test_plan_uses_server_artifact() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    current_plan = (
        transformation_plan()
    )


    def fake_planner(
        *,
        dataframe,
        dataset_id,
        dataset_filename,
        intents,
    ):

        assert (
            dataframe
            .equals(
                source_dataframe()
            )
        )


        assert (
            dataset_id
            ==
            "sales"
        )


        assert (
            dataset_filename
            ==
            "sales.csv"
        )


        assert (
            len(
                intents
            )
            ==
            1
        )


        assert (
            intents[
                0
            ]
            .dataset_id
            ==
            "sales"
        )


        assert (
            intents[
                0
            ]
            .dataset_filename
            ==
            "sales.csv"
        )


        events.append(
            "plan"
        )


        return (
            current_plan
        )


    def fake_record(
        **kwargs,
    ):

        assert (
            events
            ==
            [
                "plan",
            ]
        )


        events.append(
            "record"
        )


    patches.update(
        {
            "plan_transformations":
                fake_planner,

            "_record_transformation_plan_stage":
                fake_record,
        }
    )


    request = (
        transformation_api
        .TransformationPlanRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),

            intents=[
                arithmetic_intent_payload(),
            ],
        )
    )


    with patched_globals(
        **patches
    ):

        result = (
            transformation_api.build_preparation_transformation_plan(
                request
            )
        )


    assert (
        result
        is current_plan
    )


    assert (
        events
        ==
        [
            "plan",
            "record",
        ]
    )


    print(
        "Transformation plan uses server-owned artifact: PASS"
    )


# ============================================================
# 2. CLEAN PRECONDITION
# ============================================================

def test_unresolved_clean_blocks_transform() -> None:

    patches = (
        common_patches(
            clean_status=(
                PreparationStageStatus
                .REVIEW_REQUIRED
            )
        )
    )


    request = (
        transformation_api
        .TransformationPlanRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),

            intents=[
                arithmetic_intent_payload(),
            ],
        )
    )


    with patched_globals(
        **patches
    ):

        try:

            transformation_api.build_preparation_transformation_plan(
                request
            )


        except HTTPException as error:

            assert (
                error.status_code
                ==
                409
            )


        else:

            raise AssertionError(
                "TRANSFORM must not bypass unresolved CLEAN."
            )


    print(
        "Unresolved CLEAN blocks TRANSFORM: PASS"
    )


# ============================================================
# 3. EMPTY PLAN = SKIPPED
# ============================================================

def test_empty_transform_marks_stage_skipped() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    def fake_record(
        **kwargs,
    ):

        assert (
            kwargs[
                "plan"
            ]
            .request_count
            ==
            0
        )


        events.append(
            "skip"
        )


    patches.update(
        {
            "_record_transformation_plan_stage":
                fake_record,
        }
    )


    request = (
        transformation_api
        .TransformationPlanRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),

            intents=[],
        )
    )


    with patched_globals(
        **patches
    ):

        result = (
            transformation_api.build_preparation_transformation_plan(
                request
            )
        )


    assert (
        result.request_count
        ==
        0
    )


    assert (
        events
        ==
        [
            "skip",
        ]
    )


    print(
        "No Transformation request resolves TRANSFORM "
        "as skipped: PASS"
    )


# ============================================================
# 4. FULL APPLY ORDER
# ============================================================

def test_full_apply_order() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    current_plan = (
        transformation_plan()
    )


    current_approved_plan = (
        approved_plan()
    )


    current_execution = (
        execution_result()
    )


    current_validation = (
        validation_report(
            valid=True
        )
    )


    current_materialization = (
        materialization_report()
    )


    def fake_plan(
        **kwargs,
    ):

        events.append(
            "plan"
        )


        return (
            current_plan
        )


    def fake_approval(
        **kwargs,
    ):

        assert (
            events[
                -1
            ]
            ==
            "plan"
        )


        events.append(
            "approve"
        )


        return (
            current_approved_plan
        )


    def fake_execute(
        **kwargs,
    ):

        assert (
            events[
                -1
            ]
            ==
            "approve"
        )


        events.append(
            "execute"
        )


        return (
            current_execution
        )


    def fake_validate(
        **kwargs,
    ):

        assert (
            events[
                -1
            ]
            ==
            "execute"
        )


        events.append(
            "validate"
        )


        return (
            current_validation
        )


    def fake_materialize(
        **kwargs,
    ):

        assert (
            events[
                -1
            ]
            ==
            "validate"
        )


        events.append(
            "materialize"
        )


        return (
            current_materialization
        )


    def fake_record_passed(
        **kwargs,
    ):

        assert (
            events[
                -1
            ]
            ==
            "materialize"
        )


        events.append(
            "record-passed"
        )


    patches.update(
        {
            "plan_transformations":
                fake_plan,

            "apply_transformation_approvals":
                fake_approval,

            "execute_transformation_plan":
                fake_execute,

            "validate_transformation_execution":
                fake_validate,

            "materialize_transformation_artifacts":
                fake_materialize,

            "_record_transformation_passed":
                fake_record_passed,

            "_materialization_view":
                (
                    lambda report:
                        report
                ),

            "TransformationApplyResponse":
                fake_response,
        }
    )


    request = (
        transformation_api
        .TransformationApplyRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),

            intents=[
                arithmetic_intent_payload(),
            ],

            approval_commands=[
                TransformationApprovalCommand(
                    request_id=(
                        "derive-amount-x2"
                    ),

                    decision=(
                        TransformationApprovalDecision
                        .APPROVE
                    ),
                ),
            ],
        )
    )


    with patched_globals(
        **patches
    ):

        result = (
            transformation_api.apply_preparation_transformation(
                request
            )
        )


    assert (
        result.status
        ==
        "ready"
    )


    assert (
        events
        ==
        [
            "plan",
            "approve",
            "execute",
            "validate",
            "materialize",
            "record-passed",
        ]
    )


    print(
        "Transformation full execution order: PASS"
    )


# ============================================================
# 5. UNRESOLVED APPROVAL DOES NOT EXECUTE
# ============================================================

def test_unresolved_approval_does_not_execute() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    current_plan = (
        transformation_plan()
    )


    unresolved_plan = (
        approved_plan(
            ready_for_execution=(
                False
            ),

            executable_step_count=(
                0
            ),

            pending_count=(
                1
            ),
        )
    )


    def fake_plan(
        **kwargs,
    ):

        events.append(
            "plan"
        )


        return (
            current_plan
        )


    def fake_approval(
        **kwargs,
    ):

        events.append(
            "approve-unresolved"
        )


        return (
            unresolved_plan
        )


    def forbidden_execute(
        **kwargs,
    ):

        raise AssertionError(
            "Unresolved Transformation approval must "
            "not execute."
        )


    def fake_record(
        **kwargs,
    ):

        events.append(
            "record-unresolved"
        )


    patches.update(
        {
            "plan_transformations":
                fake_plan,

            "apply_transformation_approvals":
                fake_approval,

            "execute_transformation_plan":
                forbidden_execute,

            "_record_transformation_approval_unresolved":
                fake_record,

            "TransformationApplyResponse":
                fake_response,
        }
    )


    request = (
        transformation_api
        .TransformationApplyRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),

            intents=[
                arithmetic_intent_payload(),
            ],

            approval_commands=[],
        )
    )


    with patched_globals(
        **patches
    ):

        result = (
            transformation_api.apply_preparation_transformation(
                request
            )
        )


    assert (
        result.status
        ==
        "approval_required"
    )


    assert (
        events
        ==
        [
            "plan",
            "approve-unresolved",
            "record-unresolved",
        ]
    )


    print(
        "Unresolved Transformation approval prevents "
        "execution: PASS"
    )


# ============================================================
# 6. FAILED VALIDATION DOES NOT MATERIALIZE
# ============================================================

def test_failed_validation_does_not_materialize() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    current_plan = (
        transformation_plan()
    )


    current_approved_plan = (
        approved_plan()
    )


    current_execution = (
        execution_result()
    )


    failed_validation = (
        validation_report(
            valid=False
        )
    )


    def forbidden_materialize(
        **kwargs,
    ):

        raise AssertionError(
            "Failed post-validation must not materialize "
            "Transformation artifacts."
        )


    def fake_record_failed(
        **kwargs,
    ):

        events.append(
            "record-failed"
        )


    patches.update(
        {
            "plan_transformations":
                (
                    lambda **kwargs:
                        current_plan
                ),

            "apply_transformation_approvals":
                (
                    lambda **kwargs:
                        current_approved_plan
                ),

            "execute_transformation_plan":
                (
                    lambda **kwargs:
                        current_execution
                ),

            "validate_transformation_execution":
                (
                    lambda **kwargs:
                        failed_validation
                ),

            "materialize_transformation_artifacts":
                forbidden_materialize,

            "_record_transformation_validation_failed":
                fake_record_failed,

            "TransformationApplyResponse":
                fake_response,
        }
    )


    request = (
        transformation_api
        .TransformationApplyRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),

            intents=[
                arithmetic_intent_payload(),
            ],

            approval_commands=[
                TransformationApprovalCommand(
                    request_id=(
                        "derive-amount-x2"
                    ),

                    decision=(
                        TransformationApprovalDecision
                        .APPROVE
                    ),
                ),
            ],
        )
    )


    with patched_globals(
        **patches
    ):

        result = (
            transformation_api.apply_preparation_transformation(
                request
            )
        )


    assert (
        result.status
        ==
        "validation_failed"
    )


    assert (
        events
        ==
        [
            "record-failed",
        ]
    )


    print(
        "Failed Transformation validation prevents "
        "materialization: PASS"
    )


# ============================================================
# 7. MATERIALIZATION FAILURE PREVENTS PASSED
# ============================================================

def test_materialization_failure_prevents_passed() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    current_plan = (
        transformation_plan()
    )


    current_approved_plan = (
        approved_plan()
    )


    current_execution = (
        execution_result()
    )


    current_validation = (
        validation_report(
            valid=True
        )
    )


    def failing_materialize(
        **kwargs,
    ):

        events.append(
            "materialize-failed"
        )


        raise RuntimeError(
            "Synthetic Transformation Artifact Store failure."
        )


    def forbidden_passed(
        **kwargs,
    ):

        events.append(
            "record-passed"
        )


        raise AssertionError(
            "TRANSFORM must not become PASSED when artifact "
            "persistence fails."
        )


    patches.update(
        {
            "plan_transformations":
                (
                    lambda **kwargs:
                        current_plan
                ),

            "apply_transformation_approvals":
                (
                    lambda **kwargs:
                        current_approved_plan
                ),

            "execute_transformation_plan":
                (
                    lambda **kwargs:
                        current_execution
                ),

            "validate_transformation_execution":
                (
                    lambda **kwargs:
                        current_validation
                ),

            "materialize_transformation_artifacts":
                failing_materialize,

            "_record_transformation_passed":
                forbidden_passed,
        }
    )


    request = (
        transformation_api
        .TransformationApplyRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),

            intents=[
                arithmetic_intent_payload(),
            ],

            approval_commands=[
                TransformationApprovalCommand(
                    request_id=(
                        "derive-amount-x2"
                    ),

                    decision=(
                        TransformationApprovalDecision
                        .APPROVE
                    ),
                ),
            ],
        )
    )


    with patched_globals(
        **patches
    ):

        try:

            transformation_api.apply_preparation_transformation(
                request
            )


        except HTTPException as error:

            assert (
                error.status_code
                ==
                500
            )


        else:

            raise AssertionError(
                "Transformation materialization failure "
                "must return HTTP 500."
            )


    assert (
        events
        ==
        [
            "materialize-failed",
        ]
    )


    assert (
        "record-passed"
        not in events
    )


    print(
        "Transformation artifact failure prevents "
        "TRANSFORM PASSED: PASS"
    )


# ============================================================
# 8. BROWSER CANNOT REDIRECT DATASET
# ============================================================

def test_browser_cannot_redirect_dataset() -> None:

    patches = (
        common_patches()
    )


    payload = (
        arithmetic_intent_payload()
    )


    payload[
        "dataset_id"
    ] = (
        "another-dataset"
    )


    request = (
        transformation_api
        .TransformationPlanRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),

            intents=[
                payload,
            ],
        )
    )


    with patched_globals(
        **patches
    ):

        try:

            transformation_api.build_preparation_transformation_plan(
                request
            )


        except HTTPException as error:

            assert (
                error.status_code
                ==
                422
            )


        else:

            raise AssertionError(
                "Browser-supplied intent must not redirect "
                "Transformation to another dataset."
            )


    print(
        "Browser cannot redirect Transformation dataset: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS PREPARATION TRANSFORMATION API v0.1 ==="
    )

    print()


    test_plan_uses_server_artifact()

    test_unresolved_clean_blocks_transform()

    test_empty_transform_marks_stage_skipped()

    test_full_apply_order()

    test_unresolved_approval_does_not_execute()

    test_failed_validation_does_not_materialize()

    test_materialization_failure_prevents_passed()

    test_browser_cannot_redirect_dataset()


    print()


    print(
        "Preparation Transformation API v0.1: PASS"
    )


if __name__ == "__main__":
    main()
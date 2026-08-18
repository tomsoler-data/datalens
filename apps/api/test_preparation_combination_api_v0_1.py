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

import app.api.preparation_combination as combination_api

from app.preparation.join_approval import (
    JoinApprovalCommand,
    JoinApprovalDecision,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationStageStatus,
)


WORKFLOW_ID = (
    "workflow-combination-api-test"
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
                combination_api,
                name,
            )


            setattr(
                combination_api,
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
                combination_api,
                name,
                original,
            )


# ============================================================
# DATA
# ============================================================

def sales_frame() -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "sale_id": [
                    1,
                    2,
                    3,
                ],

                "customer_id": [
                    "c1",
                    "c1",
                    "c2",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],
            }
        )
    )


def customers_frame() -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                ],

                "segment": [
                    "A",
                    "B",
                ],
            }
        )
    )


def joined_frame() -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "sale_id": [
                    1,
                    2,
                    3,
                ],

                "customer_id": [
                    "c1",
                    "c1",
                    "c2",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],

                "segment": [
                    "A",
                    "A",
                    "B",
                ],
            }
        )
    )


# ============================================================
# JOIN INTENT
# ============================================================

def join_intent_payload():

    return {
        "request_id":
            "join-sales-customers",

        "left_dataset_id":
            "sales",

        "right_dataset_id":
            "customers",

        "join_type":
            "left",

        "keys": [
            {
                "left_column":
                    "customer_id",

                "right_column":
                    "customer_id",
            },
        ],

        "expected_cardinality":
            "many_to_one",

        "output_dataset_id":
            "sales_customers",

        "output_dataset_filename":
            "sales_customers.csv",

        "left_suffix":
            "_sales",

        "right_suffix":
            "_customer",
    }


# ============================================================
# SESSION
# ============================================================

def session(
    *,
    transform_status=(
        PreparationStageStatus.PASSED
    ),
):

    return (
        SimpleNamespace(
            selected_analysis_dataset_ids=[
                "sales",
                "customers",
            ],

            snapshot=(
                SimpleNamespace(
                    stages=[
                        SimpleNamespace(
                            stage=(
                                PreparationStage.TRANSFORM
                            ),

                            status=(
                                transform_status
                            ),

                            dataset_ids=[
                                "sales",
                                "customers",
                            ],
                        ),
                    ]
                )
            ),
        )
    )


# ============================================================
# ARTIFACTS
# ============================================================

def sales_artifact():

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
                "transform"
            ),

            dataframe=(
                sales_frame()
            ),

            parent_dataset_ids=(
                "sales",
            ),

            evidence_refs=(),
        )
    )


def customers_artifact():

    return (
        SimpleNamespace(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "customers"
            ),

            dataset_filename=(
                "customers.csv"
            ),

            stage=(
                "clean"
            ),

            dataframe=(
                customers_frame()
            ),

            parent_dataset_ids=(
                "customers",
            ),

            evidence_refs=(),
        )
    )


# ============================================================
# PLAN
# ============================================================

def join_plan(
    *,
    ready_for_approval: bool = True,
    blocked_count: int = 0,
):

    return (
        SimpleNamespace(
            request_count=(
                1
            ),

            join_count=(
                1
            ),

            review_required_count=(
                1
                if ready_for_approval
                else 0
            ),

            blocked_count=(
                blocked_count
            ),

            ready_for_approval=(
                ready_for_approval
            ),

            joins=[],

            notes=[],

            rule_version=(
                "join_contract_test_v0.1"
            ),
        )
    )


# ============================================================
# APPROVAL
# ============================================================

def approved_join_plan(
    *,
    ready_for_execution: bool = True,
    executable_join_count: int = 1,
    pending_count: int = 0,
    deferred_count: int = 0,
    rejected_count: int = 0,
):

    return (
        SimpleNamespace(
            total_join_count=(
                1
            ),

            approved_count=(
                1
                if executable_join_count
                >
                0
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

            executable_join_count=(
                executable_join_count
            ),

            ready_for_execution=(
                ready_for_execution
            ),

            joins=[],

            notes=[],

            rule_version=(
                "join_approval_test_v0.1"
            ),
        )
    )


# ============================================================
# EXECUTION
# ============================================================

def execution_result():

    return (
        SimpleNamespace(
            joined_datasets={
                "sales_customers":
                    joined_frame(),
            },

            report=(
                SimpleNamespace(
                    status=(
                        "success"
                    ),

                    total_join_count=(
                        1
                    ),

                    executable_join_count=(
                        1
                    ),

                    applied_count=(
                        1
                    ),

                    skipped_count=(
                        0
                    ),

                    output_dataset_count=(
                        1
                    ),

                    output_dataset_ids=[
                        "sales_customers",
                    ],

                    steps=[],

                    notes=[],

                    rule_version=(
                        "join_executor_test_v0.1"
                    ),
                )
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

            valid_for_downstream=(
                valid
            ),

            total_join_count=(
                1
            ),

            validated_join_count=(
                1
            ),

            passed_check_count=(
                10
                if valid
                else 9
            ),

            failed_check_count=(
                0
                if valid
                else 1
            ),

            warning_count=(
                0
            ),

            output_dataset_count=(
                1
            ),

            join_validations=[],

            checks=[],

            notes=[],

            rule_version=(
                "post_join_validation_test_v0.1"
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

            output_dataset_ids=(
                "sales_customers",
            ),

            artifact_count=(
                1
            ),

            bridge_version=(
                "join_artifact_bridge_v0.1"
            ),
        )
    )


# ============================================================
# FAKE RESPONSE
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
    transform_status=(
        PreparationStageStatus.PASSED
    ),
):

    def fake_get_artifact(
        *,
        workflow_id,
        dataset_id,
    ):

        if (
            dataset_id
            ==
            "sales"
        ):

            return (
                sales_artifact()
            )


        if (
            dataset_id
            ==
            "customers"
        ):

            return (
                customers_artifact()
            )


        raise AssertionError(
            "Unexpected dataset requested by COMBINE test: "
            f"{dataset_id}"
        )


    return {
        "get_preparation_session":
            (
                lambda workflow_id:
                    session(
                        transform_status=(
                            transform_status
                        )
                    )
            ),

        "get_preparation_artifact":
            fake_get_artifact,

        "list_preparation_artifacts":
            (
                lambda **kwargs:
                    [
                        SimpleNamespace(
                            dataset_id=(
                                "sales"
                            )
                        ),

                        SimpleNamespace(
                            dataset_id=(
                                "customers"
                            )
                        ),
                    ]
            ),
    }


# ============================================================
# 1. PLAN USES SERVER ARTIFACTS
# ============================================================

def test_plan_uses_server_artifacts() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    current_plan = (
        join_plan()
    )


    def fake_plan(
        *,
        datasets,
        intents,
    ):

        assert (
            set(
                datasets.keys()
            )
            ==
            {
                "sales",
                "customers",
            }
        )


        assert (
            datasets[
                "sales"
            ]
            .equals(
                sales_frame()
            )
        )


        assert (
            datasets[
                "customers"
            ]
            .equals(
                customers_frame()
            )
        )


        assert (
            len(
                intents
            )
            ==
            1
        )


        intent = (
            intents[
                0
            ]
        )


        assert (
            intent.left_dataset_filename
            ==
            "sales.csv"
        )


        assert (
            intent.right_dataset_filename
            ==
            "customers.csv"
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
            "plan_joins":
                fake_plan,

            "_record_combination_plan_stage":
                fake_record,
        }
    )


    request = (
        combination_api.CombinationPlanRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            intents=[
                join_intent_payload(),
            ],
        )
    )


    with patched_globals(
        **patches
    ):

        result = (
            combination_api
            .build_preparation_combination_plan(
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
        "COMBINE plan uses server-owned artifacts: PASS"
    )


# ============================================================
# 2. TRANSFORM PRECONDITION
# ============================================================

def test_unresolved_transform_blocks_combine() -> None:

    patches = (
        common_patches(
            transform_status=(
                PreparationStageStatus
                .REVIEW_REQUIRED
            )
        )
    )


    request = (
        combination_api.CombinationPlanRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            intents=[
                join_intent_payload(),
            ],
        )
    )


    with patched_globals(
        **patches
    ):

        try:

            combination_api.build_preparation_combination_plan(
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
                "COMBINE must not bypass unresolved "
                "TRANSFORM."
            )


    print(
        "Unresolved TRANSFORM blocks COMBINE: PASS"
    )


# ============================================================
# 3. BROWSER CANNOT USE UNAUTHORIZED DATASET
# ============================================================

def test_browser_cannot_use_unauthorized_dataset() -> None:

    patches = (
        common_patches()
    )


    payload = (
        join_intent_payload()
    )


    payload[
        "right_dataset_id"
    ] = (
        "private_dataset"
    )


    request = (
        combination_api.CombinationPlanRequest(
            workflow_id=(
                WORKFLOW_ID
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

            combination_api.build_preparation_combination_plan(
                request
            )


        except HTTPException as error:

            assert (
                error.status_code
                ==
                403
            )


        else:

            raise AssertionError(
                "Browser must not redirect COMBINE to an "
                "unauthorized dataset."
            )


    print(
        "Browser cannot inject unauthorized JOIN dataset: PASS"
    )


# ============================================================
# 4. NO JOIN = SKIPPED
# ============================================================

def test_empty_combination_is_skipped() -> None:

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
            "_record_combination_plan_stage":
                fake_record,
        }
    )


    request = (
        combination_api.CombinationPlanRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            intents=[],
        )
    )


    with patched_globals(
        **patches
    ):

        result = (
            combination_api
            .build_preparation_combination_plan(
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
        "No JOIN request resolves COMBINE as skipped: PASS"
    )


# ============================================================
# 5. BLOCKED PLAN NEVER ENTERS APPROVAL
# ============================================================

def test_blocked_plan_never_enters_approval() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    blocked_plan = (
        join_plan(
            ready_for_approval=(
                False
            ),

            blocked_count=(
                1
            ),
        )
    )


    def fake_plan(
        **kwargs,
    ):

        events.append(
            "plan-blocked"
        )


        return (
            blocked_plan
        )


    def forbidden_approval(
        **kwargs,
    ):

        raise AssertionError(
            "Blocked JOIN plan must never enter approval."
        )


    def fake_record(
        **kwargs,
    ):

        events.append(
            "record-blocked"
        )


    patches.update(
        {
            "plan_joins":
                fake_plan,

            "apply_join_approvals":
                forbidden_approval,

            "_record_combination_plan_stage":
                fake_record,

            "CombinationApplyResponse":
                fake_response,
        }
    )


    request = (
        combination_api.CombinationApplyRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            intents=[
                join_intent_payload(),
            ],

            approval_commands=[],
        )
    )


    with patched_globals(
        **patches
    ):

        result = (
            combination_api
            .apply_preparation_combination(
                request
            )
        )


    assert (
        result.status
        ==
        "blocked"
    )


    assert (
        events
        ==
        [
            "plan-blocked",
            "record-blocked",
        ]
    )


    print(
        "Blocked JOIN never enters approval: PASS"
    )


# ============================================================
# 6. FULL APPLY ORDER
# ============================================================

def test_full_apply_order() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    current_plan = (
        join_plan()
    )


    current_approved_plan = (
        approved_join_plan()
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
            "plan_joins":
                fake_plan,

            "apply_join_approvals":
                fake_approval,

            "execute_join_plan":
                fake_execute,

            "validate_join_execution":
                fake_validate,

            "materialize_join_artifacts":
                fake_materialize,

            "_record_combination_passed":
                fake_record_passed,

            "_materialization_view":
                (
                    lambda report:
                        report
                ),

            "CombinationApplyResponse":
                fake_response,
        }
    )


    request = (
        combination_api.CombinationApplyRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            intents=[
                join_intent_payload(),
            ],

            approval_commands=[
                JoinApprovalCommand(
                    request_id=(
                        "join-sales-customers"
                    ),

                    decision=(
                        JoinApprovalDecision
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
            combination_api
            .apply_preparation_combination(
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
        "COMBINE full execution order: PASS"
    )


# ============================================================
# 7. UNRESOLVED APPROVAL DOES NOT EXECUTE
# ============================================================

def test_unresolved_approval_does_not_execute() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    current_plan = (
        join_plan()
    )


    unresolved_plan = (
        approved_join_plan(
            ready_for_execution=(
                False
            ),

            executable_join_count=(
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

        return (
            current_plan
        )


    def fake_approval(
        **kwargs,
    ):

        events.append(
            "approval-unresolved"
        )


        return (
            unresolved_plan
        )


    def forbidden_execute(
        **kwargs,
    ):

        raise AssertionError(
            "Unresolved JOIN approval must not execute."
        )


    def fake_record(
        **kwargs,
    ):

        events.append(
            "record-unresolved"
        )


    patches.update(
        {
            "plan_joins":
                fake_plan,

            "apply_join_approvals":
                fake_approval,

            "execute_join_plan":
                forbidden_execute,

            "_record_combination_approval_unresolved":
                fake_record,

            "CombinationApplyResponse":
                fake_response,
        }
    )


    request = (
        combination_api.CombinationApplyRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            intents=[
                join_intent_payload(),
            ],

            approval_commands=[],
        )
    )


    with patched_globals(
        **patches
    ):

        result = (
            combination_api
            .apply_preparation_combination(
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
            "approval-unresolved",
            "record-unresolved",
        ]
    )


    print(
        "Unresolved JOIN approval prevents execution: PASS"
    )


# ============================================================
# 8. FAILED VALIDATION DOES NOT MATERIALIZE
# ============================================================

def test_failed_validation_does_not_materialize() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    current_plan = (
        join_plan()
    )


    current_approved_plan = (
        approved_join_plan()
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
            "Failed Post-Join Validation must not "
            "materialize JOIN artifacts."
        )


    def fake_record_failed(
        **kwargs,
    ):

        events.append(
            "record-failed"
        )


    patches.update(
        {
            "plan_joins":
                (
                    lambda **kwargs:
                        current_plan
                ),

            "apply_join_approvals":
                (
                    lambda **kwargs:
                        current_approved_plan
                ),

            "execute_join_plan":
                (
                    lambda **kwargs:
                        current_execution
                ),

            "validate_join_execution":
                (
                    lambda **kwargs:
                        failed_validation
                ),

            "materialize_join_artifacts":
                forbidden_materialize,

            "_record_combination_validation_failed":
                fake_record_failed,

            "CombinationApplyResponse":
                fake_response,
        }
    )


    request = (
        combination_api.CombinationApplyRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            intents=[
                join_intent_payload(),
            ],

            approval_commands=[
                JoinApprovalCommand(
                    request_id=(
                        "join-sales-customers"
                    ),

                    decision=(
                        JoinApprovalDecision
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
            combination_api
            .apply_preparation_combination(
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
        "Failed Post-Join Validation prevents "
        "materialization: PASS"
    )


# ============================================================
# 9. MATERIALIZATION FAILURE PREVENTS COMBINE PASSED
# ============================================================

def test_materialization_failure_prevents_passed() -> None:

    events: list[
        str
    ] = []


    patches = (
        common_patches()
    )


    current_plan = (
        join_plan()
    )


    current_approved_plan = (
        approved_join_plan()
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
            "Synthetic JOIN Artifact Store failure."
        )


    def forbidden_passed(
        **kwargs,
    ):

        events.append(
            "record-passed"
        )


        raise AssertionError(
            "COMBINE must not become PASSED when JOIN "
            "artifact persistence fails."
        )


    patches.update(
        {
            "plan_joins":
                (
                    lambda **kwargs:
                        current_plan
                ),

            "apply_join_approvals":
                (
                    lambda **kwargs:
                        current_approved_plan
                ),

            "execute_join_plan":
                (
                    lambda **kwargs:
                        current_execution
                ),

            "validate_join_execution":
                (
                    lambda **kwargs:
                        current_validation
                ),

            "materialize_join_artifacts":
                failing_materialize,

            "_record_combination_passed":
                forbidden_passed,
        }
    )


    request = (
        combination_api.CombinationApplyRequest(
            workflow_id=(
                WORKFLOW_ID
            ),

            intents=[
                join_intent_payload(),
            ],

            approval_commands=[
                JoinApprovalCommand(
                    request_id=(
                        "join-sales-customers"
                    ),

                    decision=(
                        JoinApprovalDecision
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

            combination_api.apply_preparation_combination(
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
                "JOIN artifact failure must return HTTP 500."
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
        "JOIN artifact failure prevents COMBINE PASSED: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS PREPARATION COMBINATION API v0.1 ==="
    )

    print()


    test_plan_uses_server_artifacts()

    test_unresolved_transform_blocks_combine()

    test_browser_cannot_use_unauthorized_dataset()

    test_empty_combination_is_skipped()

    test_blocked_plan_never_enters_approval()

    test_full_apply_order()

    test_unresolved_approval_does_not_execute()

    test_failed_validation_does_not_materialize()

    test_materialization_failure_prevents_passed()


    print()


    print(
        "Preparation Combination API v0.1: PASS"
    )


if __name__ == "__main__":
    main()
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

import app.api.preparation_cleaning as cleaning_api


# ============================================================
# HELPERS
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
                cleaning_api,
                name,
            )


            setattr(
                cleaning_api,
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
                cleaning_api,
                name,
                original,
            )


def source_dataframe(
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                    "c3",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],
            }
        )
    )


def source_records(
) -> list[
    dict
]:

    return [
        {
            "dataset_id":
                "sales",

            "filename":
                "sales.csv",

            "dataframe":
                source_dataframe(),
        }
    ]


def plan(
    *,
    action_count: int,
    protected_issue_count: int = 0,
):

    return (
        SimpleNamespace(
            action_count=(
                action_count
            ),

            protected_issue_count=(
                protected_issue_count
            ),

            rule_version=(
                "cleaning_plan_test_v0.1"
            ),
        )
    )


def execution(
    *,
    blocked_action_count: int = 0,
):

    return (
        SimpleNamespace(
            blocked_action_count=(
                blocked_action_count
            ),

            applied_action_count=(
                1
            ),

            skipped_action_count=(
                0
            ),

            rule_version=(
                "cleaning_executor_test_v0.1"
            ),

            provenance=[
                SimpleNamespace(
                    dataset_id=(
                        "sales"
                    ),

                    rows_before=(
                        3
                    ),

                    rows_after=(
                        2
                    ),

                    columns_before=(
                        2
                    ),

                    columns_after=(
                        2
                    ),

                    source_fingerprint=(
                        "source-fingerprint"
                    ),

                    derived_fingerprint=(
                        "derived-fingerprint"
                    ),

                    applied_action_ids=[
                        "action-1",
                    ],

                    skipped_action_ids=[],
                )
            ],
        )
    )


def cleaned_dataframe(
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                ],

                "amount": [
                    10.0,
                    20.0,
                ],
            }
        )
    )


def fake_response(
    **kwargs,
):

    return (
        SimpleNamespace(
            **kwargs
        )
    )


def fake_preview(
    **kwargs,
):

    return (
        SimpleNamespace(
            **kwargs
        )
    )


# ============================================================
# 1. PLAN — SOURCE MATERIALIZED BEFORE SESSION UPDATE
# ============================================================

def test_plan_materializes_before_session_update(
) -> None:

    events: list[
        str
    ] = []


    records = (
        source_records()
    )


    current_plan = (
        plan(
            action_count=0,

            # A semantic issue may remain.
            # Materialization is still useful and does not
            # itself authorize downstream analysis.
            protected_issue_count=1,
        )
    )


    def fake_build(
        dataset_files,
    ):

        return (
            records,
            {
                "sales":
                    records[
                        0
                    ][
                        "dataframe"
                    ],
            },
            SimpleNamespace(),
            current_plan,
        )


    def fake_scope(
        *,
        workflow_id,
        source_dataset_records,
    ):

        return [
            "sales",
        ]


    def fake_materialize(
        **kwargs,
    ):

        events.append(
            "materialize"
        )


        return (
            SimpleNamespace()
        )


    def fake_record(
        **kwargs,
    ):

        assert (
            events
            ==
            [
                "materialize",
            ]
        )


        events.append(
            "record"
        )


    with patched_globals(
        _build_quality_and_plan=(
            fake_build
        ),

        _validate_session_dataset_scope=(
            fake_scope
        ),

        materialize_skipped_cleaning_artifacts=(
            fake_materialize
        ),

        _record_cleaning_plan_stage=(
            fake_record
        ),
    ):

        result = (
            cleaning_api
            .build_uploaded_cleaning_plan(
                dataset_files=[],

                workflow_id=(
                    "workflow-test"
                ),
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
            "materialize",
            "record",
        ]
    )


    print(
        "Cleaning plan materializes source before "
        "session update: PASS"
    )


# ============================================================
# 2. PLAN WITH ACTIONS DOES NOT PRETEND SOURCE IS FINAL
# ============================================================

def test_plan_with_actions_does_not_materialize(
) -> None:

    events: list[
        str
    ] = []


    records = (
        source_records()
    )


    current_plan = (
        plan(
            action_count=1
        )
    )


    def fake_build(
        dataset_files,
    ):

        return (
            records,
            {
                "sales":
                    records[
                        0
                    ][
                        "dataframe"
                    ],
            },
            SimpleNamespace(),
            current_plan,
        )


    def fake_scope(
        **kwargs,
    ):

        return [
            "sales",
        ]


    def forbidden_materialize(
        **kwargs,
    ):

        raise AssertionError(
            "Source passthrough must not be materialized "
            "when deterministic cleaning actions exist."
        )


    def fake_record(
        **kwargs,
    ):

        events.append(
            "record"
        )


    with patched_globals(
        _build_quality_and_plan=(
            fake_build
        ),

        _validate_session_dataset_scope=(
            fake_scope
        ),

        materialize_skipped_cleaning_artifacts=(
            forbidden_materialize
        ),

        _record_cleaning_plan_stage=(
            fake_record
        ),
    ):

        cleaning_api.build_uploaded_cleaning_plan(
            dataset_files=[],

            workflow_id=(
                "workflow-test"
            ),
        )


    assert (
        events
        ==
        [
            "record",
        ]
    )


    print(
        "Cleaning plan with actions waits for execution: PASS"
    )


# ============================================================
# 3. APPLY — DERIVED MATERIALIZED BEFORE SESSION UPDATE
# ============================================================

def test_apply_materializes_before_session_update(
) -> None:

    events: list[
        str
    ] = []


    records = (
        source_records()
    )


    current_plan = (
        plan(
            action_count=1,
            protected_issue_count=1,
        )
    )


    current_execution = (
        execution(
            blocked_action_count=0
        )
    )


    cleaned = (
        cleaned_dataframe()
    )


    def fake_build(
        dataset_files,
    ):

        return (
            records,
            {
                "sales":
                    records[
                        0
                    ][
                        "dataframe"
                    ],
            },
            SimpleNamespace(),
            current_plan,
        )


    def fake_scope(
        **kwargs,
    ):

        return [
            "sales",
        ]


    def fake_execute(
        **kwargs,
    ):

        return (
            {
                "sales":
                    cleaned,
            },
            current_execution,
        )


    def fake_materialize(
        **kwargs,
    ):

        assert (
            kwargs[
                "derived_frames"
            ][
                "sales"
            ]
            .equals(
                cleaned
            )
        )


        events.append(
            "materialize"
        )


        return (
            SimpleNamespace()
        )


    def fake_record(
        **kwargs,
    ):

        assert (
            events
            ==
            [
                "materialize",
            ]
        )


        events.append(
            "record"
        )


    with patched_globals(
        _build_quality_and_plan=(
            fake_build
        ),

        _validate_session_dataset_scope=(
            fake_scope
        ),

        _parse_approved_action_ids=(
            lambda raw_value:
                {
                    "action-1",
                }
        ),

        execute_cleaning_plan=(
            fake_execute
        ),

        materialize_cleaning_execution_artifacts=(
            fake_materialize
        ),

        _record_cleaning_execution_stage=(
            fake_record
        ),

        CleaningApplyResponse=(
            fake_response
        ),

        DerivedDatasetPreview=(
            fake_preview
        ),
    ):

        cleaning_api.apply_uploaded_cleaning_plan(
            dataset_files=[],

            approved_action_ids_json=(
                '["action-1"]'
            ),

            workflow_id=(
                "workflow-test"
            ),
        )


    assert (
        events
        ==
        [
            "materialize",
            "record",
        ]
    )


    print(
        "Cleaning execution materializes derived frame "
        "before session update: PASS"
    )


# ============================================================
# 4. BLOCKED EXECUTION IS NOT MATERIALIZED
# ============================================================

def test_blocked_execution_not_materialized(
) -> None:

    events: list[
        str
    ] = []


    records = (
        source_records()
    )


    current_plan = (
        plan(
            action_count=1
        )
    )


    current_execution = (
        execution(
            blocked_action_count=1
        )
    )


    cleaned = (
        cleaned_dataframe()
    )


    def fake_build(
        dataset_files,
    ):

        return (
            records,
            {
                "sales":
                    records[
                        0
                    ][
                        "dataframe"
                    ],
            },
            SimpleNamespace(),
            current_plan,
        )


    def fake_execute(
        **kwargs,
    ):

        return (
            {
                "sales":
                    cleaned,
            },
            current_execution,
        )


    def forbidden_materialize(
        **kwargs,
    ):

        raise AssertionError(
            "Blocked Cleaning execution must not become "
            "the current material artifact."
        )


    def fake_record(
        **kwargs,
    ):

        events.append(
            "record-blocked"
        )


    with patched_globals(
        _build_quality_and_plan=(
            fake_build
        ),

        _validate_session_dataset_scope=(
            lambda **kwargs:
                [
                    "sales",
                ]
        ),

        _parse_approved_action_ids=(
            lambda raw_value:
                {
                    "action-1",
                }
        ),

        execute_cleaning_plan=(
            fake_execute
        ),

        materialize_cleaning_execution_artifacts=(
            forbidden_materialize
        ),

        _record_cleaning_execution_stage=(
            fake_record
        ),

        CleaningApplyResponse=(
            fake_response
        ),

        DerivedDatasetPreview=(
            fake_preview
        ),
    ):

        cleaning_api.apply_uploaded_cleaning_plan(
            dataset_files=[],

            approved_action_ids_json=(
                '["action-1"]'
            ),

            workflow_id=(
                "workflow-test"
            ),
        )


    assert (
        events
        ==
        [
            "record-blocked",
        ]
    )


    print(
        "Blocked Cleaning execution is not materialized: PASS"
    )


# ============================================================
# 5. MATERIALIZATION FAILURE PREVENTS SESSION COMPLETION
# ============================================================

def test_materialization_failure_prevents_session_update(
) -> None:

    events: list[
        str
    ] = []


    records = (
        source_records()
    )


    current_plan = (
        plan(
            action_count=1
        )
    )


    current_execution = (
        execution(
            blocked_action_count=0
        )
    )


    def fake_build(
        dataset_files,
    ):

        return (
            records,
            {
                "sales":
                    records[
                        0
                    ][
                        "dataframe"
                    ],
            },
            SimpleNamespace(),
            current_plan,
        )


    def fake_execute(
        **kwargs,
    ):

        return (
            {
                "sales":
                    cleaned_dataframe(),
            },
            current_execution,
        )


    def failing_materialize(
        **kwargs,
    ):

        events.append(
            "materialize-failed"
        )


        raise RuntimeError(
            "Synthetic Artifact Store failure."
        )


    def forbidden_record(
        **kwargs,
    ):

        events.append(
            "record"
        )


        raise AssertionError(
            "PreparationSession must not be updated after "
            "Artifact Store persistence failed."
        )


    with patched_globals(
        _build_quality_and_plan=(
            fake_build
        ),

        _validate_session_dataset_scope=(
            lambda **kwargs:
                [
                    "sales",
                ]
        ),

        _parse_approved_action_ids=(
            lambda raw_value:
                {
                    "action-1",
                }
        ),

        execute_cleaning_plan=(
            fake_execute
        ),

        materialize_cleaning_execution_artifacts=(
            failing_materialize
        ),

        _record_cleaning_execution_stage=(
            forbidden_record
        ),
    ):

        try:

            cleaning_api.apply_uploaded_cleaning_plan(
                dataset_files=[],

                approved_action_ids_json=(
                    '["action-1"]'
                ),

                workflow_id=(
                    "workflow-test"
                ),
            )


        except HTTPException as error:

            assert (
                error.status_code
                ==
                500
            )


        else:

            raise AssertionError(
                "Artifact Store failure must propagate as "
                "an HTTP 500 from Cleaning workflow."
            )


    assert (
        events
        ==
        [
            "materialize-failed",
        ]
    )


    print(
        "Artifact persistence failure prevents CLEAN "
        "session update: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS CLEANING → ARTIFACT STORE WIRING v0.1 ==="
    )

    print()


    test_plan_materializes_before_session_update()

    test_plan_with_actions_does_not_materialize()

    test_apply_materializes_before_session_update()

    test_blocked_execution_not_materialized()

    test_materialization_failure_prevents_session_update()


    print()

    print(
        "Cleaning → Artifact Store Wiring v0.1: PASS"
    )


if __name__ == "__main__":
    main()
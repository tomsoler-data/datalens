from __future__ import annotations

from types import (
    SimpleNamespace,
)

import pandas as pd

from app.preparation.cleaning_artifacts import (
    CLEANING_ARTIFACT_BRIDGE_VERSION,
    materialize_cleaning_execution_artifacts,
    materialize_skipped_cleaning_artifacts,
)

from app.preparation.preparation_artifact_store import (
    get_preparation_artifact,
    list_preparation_artifacts,
    reset_preparation_artifact_store_for_tests,
)


WORKFLOW_ID = (
    "workflow-cleaning-artifacts-test"
)


# ============================================================
# FIXTURES
# ============================================================

def source_dataframe() -> pd.DataFrame:

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


def source_records() -> list[
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


def skipped_plan():

    return (
        SimpleNamespace(
            action_count=(
                0
            ),

            rule_version=(
                "cleaning_plan_test_v0.1"
            ),
        )
    )


def executable_plan():

    return (
        SimpleNamespace(
            action_count=(
                1
            ),

            rule_version=(
                "cleaning_plan_test_v0.1"
            ),
        )
    )


def execution_for(
    dataframe: pd.DataFrame,
):

    provenance = (
        SimpleNamespace(
            dataset_id=(
                "sales"
            ),

            rows_before=(
                3
            ),

            rows_after=(
                int(
                    dataframe.shape[
                        0
                    ]
                )
            ),

            columns_before=(
                2
            ),

            columns_after=(
                int(
                    dataframe.shape[
                        1
                    ]
                )
            ),

            source_fingerprint=(
                "source-fingerprint"
            ),

            derived_fingerprint=(
                "derived-fingerprint"
            ),

            applied_action_ids=[
                "drop_duplicate_rows:sales",
            ],
        )
    )


    return (
        SimpleNamespace(
            rule_version=(
                "cleaning_executor_test_v0.1"
            ),

            provenance=[
                provenance,
            ],
        )
    )


# ============================================================
# 1. VERSION
# ============================================================

def test_version() -> None:

    assert (
        CLEANING_ARTIFACT_BRIDGE_VERSION
        == "cleaning_artifact_bridge_v0.1"
    )


    print(
        "Cleaning artifact bridge version: PASS"
    )


# ============================================================
# 2. SKIPPED CLEANING MATERIALIZES SOURCE
# ============================================================

def test_skipped_cleaning_materializes_source() -> None:

    reset_preparation_artifact_store_for_tests()


    records = (
        source_records()
    )


    report = (
        materialize_skipped_cleaning_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataset_records=(
                records
            ),

            cleaning_plan=(
                skipped_plan()
            ),
        )
    )


    assert (
        report.materialization_kind
        == "source_passthrough"
    )


    assert (
        report.artifact_count
        == 1
    )


    artifact = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),
        )
    )


    assert (
        artifact.stage
        == "source"
    )


    assert (
        artifact
        .dataframe
        .equals(
            source_dataframe()
        )
    )


    assert (
        "cleaning:skipped"
        in artifact.evidence_refs
    )


    print(
        "Skipped Cleaning materializes source artifact: PASS"
    )


# ============================================================
# 3. SKIP CANNOT BYPASS REQUIRED CLEANING
# ============================================================

def test_skip_cannot_bypass_actions() -> None:

    reset_preparation_artifact_store_for_tests()


    try:

        materialize_skipped_cleaning_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataset_records=(
                source_records()
            ),

            cleaning_plan=(
                executable_plan()
            ),
        )


    except ValueError:

        assert (
            list_preparation_artifacts(
                workflow_id=(
                    WORKFLOW_ID
                )
            )
            == []
        )


        print(
            "Cleaning skip cannot bypass required actions: PASS"
        )


    else:

        raise AssertionError(
            "Cleaning with actions must not be materialized "
            "as skipped."
        )


# ============================================================
# 4. EXECUTED CLEANING MATERIALIZES DERIVED FRAME
# ============================================================

def test_executed_cleaning_materializes_derived() -> None:

    reset_preparation_artifact_store_for_tests()


    cleaned = (
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


    execution = (
        execution_for(
            cleaned
        )
    )


    report = (
        materialize_cleaning_execution_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataset_records=(
                source_records()
            ),

            cleaning_plan=(
                executable_plan()
            ),

            derived_frames={
                "sales":
                    cleaned,
            },

            execution=(
                execution
            ),
        )
    )


    assert (
        report.materialization_kind
        == "cleaned"
    )


    assert (
        report.dataset_ids
        == (
            "sales",
        )
    )


    artifact = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),
        )
    )


    assert (
        artifact.stage
        == "clean"
    )


    assert (
        artifact
        .dataframe
        .equals(
            cleaned
        )
    )


    assert (
        artifact.parent_dataset_ids
        == (
            "sales",
        )
    )


    assert (
        (
            "cleaning_execution:"
            "cleaning_executor_test_v0.1"
        )
        in artifact.evidence_refs
    )


    assert (
        (
            "cleaning_action:"
            "drop_duplicate_rows:sales"
        )
        in artifact.evidence_refs
    )


    print(
        "Executed Cleaning materializes derived artifact: PASS"
    )


# ============================================================
# 5. MISSING DERIVED DATASET IS REJECTED
# ============================================================

def test_missing_derived_dataset_rejected() -> None:

    reset_preparation_artifact_store_for_tests()


    cleaned = (
        source_dataframe()
    )


    try:

        materialize_cleaning_execution_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataset_records=(
                source_records()
            ),

            cleaning_plan=(
                executable_plan()
            ),

            derived_frames={},

            execution=(
                execution_for(
                    cleaned
                )
            ),
        )


    except ValueError:

        assert (
            list_preparation_artifacts(
                workflow_id=(
                    WORKFLOW_ID
                )
            )
            == []
        )


        print(
            "Missing Cleaning derived dataset rejected "
            "before persistence: PASS"
        )


    else:

        raise AssertionError(
            "Missing derived dataset must be rejected."
        )


# ============================================================
# 6. UNKNOWN DERIVED DATASET IS REJECTED
# ============================================================

def test_unknown_derived_dataset_rejected() -> None:

    reset_preparation_artifact_store_for_tests()


    cleaned = (
        source_dataframe()
    )


    try:

        materialize_cleaning_execution_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataset_records=(
                source_records()
            ),

            cleaning_plan=(
                executable_plan()
            ),

            derived_frames={
                "sales":
                    cleaned,

                "invented_dataset":
                    cleaned,
            },

            execution=(
                execution_for(
                    cleaned
                )
            ),
        )


    except ValueError:

        assert (
            list_preparation_artifacts(
                workflow_id=(
                    WORKFLOW_ID
                )
            )
            == []
        )


        print(
            "Unknown Cleaning derived dataset rejected "
            "before persistence: PASS"
        )


    else:

        raise AssertionError(
            "Unknown derived dataset must be rejected."
        )


# ============================================================
# 7. MISSING PROVENANCE IS REJECTED
# ============================================================

def test_missing_provenance_rejected() -> None:

    reset_preparation_artifact_store_for_tests()


    cleaned = (
        source_dataframe()
    )


    execution = (
        SimpleNamespace(
            rule_version=(
                "cleaning_executor_test_v0.1"
            ),

            provenance=[],
        )
    )


    try:

        materialize_cleaning_execution_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataset_records=(
                source_records()
            ),

            cleaning_plan=(
                executable_plan()
            ),

            derived_frames={
                "sales":
                    cleaned,
            },

            execution=(
                execution
            ),
        )


    except ValueError:

        assert (
            list_preparation_artifacts(
                workflow_id=(
                    WORKFLOW_ID
                )
            )
            == []
        )


        print(
            "Missing Cleaning provenance rejected before "
            "persistence: PASS"
        )


    else:

        raise AssertionError(
            "Missing Cleaning provenance must be rejected."
        )


# ============================================================
# 8. ROW-COUNT MISMATCH IS REJECTED
# ============================================================

def test_row_count_mismatch_rejected() -> None:

    reset_preparation_artifact_store_for_tests()


    cleaned = (
        source_dataframe()
    )


    bad_execution = (
        SimpleNamespace(
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
                        999
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
                        "some-action",
                    ],
                )
            ],
        )
    )


    try:

        materialize_cleaning_execution_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataset_records=(
                source_records()
            ),

            cleaning_plan=(
                executable_plan()
            ),

            derived_frames={
                "sales":
                    cleaned,
            },

            execution=(
                bad_execution
            ),
        )


    except ValueError:

        assert (
            list_preparation_artifacts(
                workflow_id=(
                    WORKFLOW_ID
                )
            )
            == []
        )


        print(
            "Cleaning provenance/frame mismatch rejected "
            "before persistence: PASS"
        )


    else:

        raise AssertionError(
            "Mismatched executor provenance must be rejected."
        )


# ============================================================
# 9. INPUT COPY ISOLATION REMAINS TRUE
# ============================================================

def test_materialized_cleaning_isolated_from_input() -> None:

    reset_preparation_artifact_store_for_tests()


    cleaned = (
        source_dataframe()
    )


    materialize_cleaning_execution_artifacts(
        workflow_id=(
            WORKFLOW_ID
        ),

        source_dataset_records=(
            source_records()
        ),

        cleaning_plan=(
            executable_plan()
        ),

        derived_frames={
            "sales":
                cleaned,
        },

        execution=(
            execution_for(
                cleaned
            )
        ),
    )


    cleaned.loc[
        0,
        "amount",
    ] = 9999.0


    artifact = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),
        )
    )


    assert (
        artifact
        .dataframe
        .loc[
            0,
            "amount",
        ]
        == 10.0
    )


    print(
        "Cleaning artifact remains isolated from caller: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS CLEANING ARTIFACT BRIDGE v0.1 ==="
    )

    print()


    test_version()

    test_skipped_cleaning_materializes_source()

    test_skip_cannot_bypass_actions()

    test_executed_cleaning_materializes_derived()

    test_missing_derived_dataset_rejected()

    test_unknown_derived_dataset_rejected()

    test_missing_provenance_rejected()

    test_row_count_mismatch_rejected()

    test_materialized_cleaning_isolated_from_input()


    print()


    print(
        "Cleaning Artifact Bridge v0.1: PASS"
    )


if __name__ == "__main__":
    main()
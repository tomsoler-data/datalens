from __future__ import annotations

from types import (
    SimpleNamespace,
)

import pandas as pd

from app.preparation.preparation_artifact_store import (
    get_preparation_artifact,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)

from app.preparation.semantic_cleaning_artifacts import (
    SEMANTIC_CLEANING_ARTIFACT_BRIDGE_VERSION,
    materialize_semantic_cleaning_artifacts,
)


WORKFLOW_ID = (
    "workflow-semantic-artifacts-test"
)


# ============================================================
# FIXTURES
# ============================================================

def deterministic_frame() -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                    "c3",
                ],

                "city": [
                    "New York",
                    "NY",
                    "Boston",
                ],
            }
        )
    )


def semantic_frame() -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                    "c3",
                ],

                "city": [
                    "New York",
                    "New York",
                    "Boston",
                ],
            }
        )
    )


def semantic_plan(
    *,
    action_count: int = 1,
):

    return (
        SimpleNamespace(
            action_count=(
                action_count
            ),

            rule_version=(
                "semantic_cleaning_engine_test_v0.1"
            ),
        )
    )


def semantic_execution_changed():

    return (
        SimpleNamespace(
            status=(
                "success"
            ),

            dataset_count=(
                1
            ),

            applied_action_count=(
                1
            ),

            skipped_action_count=(
                0
            ),

            changed_cell_count=(
                1
            ),

            rule_version=(
                "semantic_cleaning_engine_test_v0.1"
            ),

            provenance=[
                SimpleNamespace(
                    dataset_id=(
                        "sales"
                    ),

                    dataset_filename=(
                        "sales.csv"
                    ),

                    rows_before=(
                        3
                    ),

                    rows_after=(
                        3
                    ),

                    source_fingerprint=(
                        "source-fingerprint"
                    ),

                    derived_fingerprint=(
                        "derived-fingerprint"
                    ),

                    applied_action_ids=[
                        "semantic-action-1",
                    ],

                    changed_cell_count=(
                        1
                    ),
                )
            ],
        )
    )


def semantic_execution_no_change():

    return (
        SimpleNamespace(
            status=(
                "success"
            ),

            dataset_count=(
                1
            ),

            applied_action_count=(
                0
            ),

            skipped_action_count=(
                1
            ),

            changed_cell_count=(
                0
            ),

            rule_version=(
                "semantic_cleaning_engine_test_v0.1"
            ),

            provenance=[
                SimpleNamespace(
                    dataset_id=(
                        "sales"
                    ),

                    dataset_filename=(
                        "sales.csv"
                    ),

                    rows_before=(
                        3
                    ),

                    rows_after=(
                        3
                    ),

                    source_fingerprint=(
                        "same-fingerprint"
                    ),

                    derived_fingerprint=(
                        "same-fingerprint"
                    ),

                    applied_action_ids=[],

                    changed_cell_count=(
                        0
                    ),
                )
            ],
        )
    )


def put_current_clean_artifact(
    *,
    stage: str = "clean",
) -> None:

    put_preparation_artifact(
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
            stage
        ),

        dataframe=(
            deterministic_frame()
        ),

        parent_dataset_ids=[
            "sales",
        ],

        evidence_refs=[
            "cleaning_execution:test",
        ],
    )


# ============================================================
# 1. VERSION
# ============================================================

def test_version() -> None:

    assert (
        SEMANTIC_CLEANING_ARTIFACT_BRIDGE_VERSION
        ==
        "semantic_cleaning_artifact_bridge_v0.1"
    )


    print(
        "Semantic Cleaning artifact bridge version: PASS"
    )


# ============================================================
# 2. REAL MUTATION IS MATERIALIZED
# ============================================================

def test_real_semantic_mutation_materialized() -> None:

    reset_preparation_artifact_store_for_tests()


    put_current_clean_artifact()


    report = (
        materialize_semantic_cleaning_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            deterministic_frames={
                "sales":
                    deterministic_frame(),
            },

            derived_frames={
                "sales":
                    semantic_frame(),
            },

            semantic_plan=(
                semantic_plan()
            ),

            execution=(
                semantic_execution_changed()
            ),
        )
    )


    assert (
        report.materialization_kind
        ==
        "semantic_cleaned"
    )


    assert (
        report.artifact_count
        ==
        1
    )


    assert (
        report.persisted_dataset_ids
        ==
        (
            "sales",
        )
    )


    assert (
        report.changed_cell_count
        ==
        1
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
        ==
        "clean"
    )


    assert (
        artifact
        .dataframe
        .equals(
            semantic_frame()
        )
    )


    assert (
        "cleaning_execution:test"
        in artifact.evidence_refs
    )


    assert (
        (
            "semantic_cleaning_action:"
            "semantic-action-1"
        )
        in artifact.evidence_refs
    )


    assert (
        (
            "semantic_cleaning_execution:"
            "semantic_cleaning_engine_test_v0.1"
        )
        in artifact.evidence_refs
    )


    print(
        "Real semantic mutation materialized: PASS"
    )


# ============================================================
# 3. NON-MUTATING CONFIRMATION DOES NOT REWRITE ARTIFACT
# ============================================================

def test_no_change_does_not_rewrite_artifact() -> None:

    reset_preparation_artifact_store_for_tests()


    put_preparation_artifact(
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
            "source"
        ),

        dataframe=(
            deterministic_frame()
        ),

        evidence_refs=[
            "cleaning:skipped",
        ],
    )


    before = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),
        )
    )


    report = (
        materialize_semantic_cleaning_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            deterministic_frames={
                "sales":
                    deterministic_frame(),
            },

            derived_frames={
                "sales":
                    deterministic_frame(),
            },

            semantic_plan=(
                semantic_plan(
                    action_count=1
                )
            ),

            execution=(
                semantic_execution_no_change()
            ),
        )
    )


    after = (
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
        report.materialization_kind
        ==
        "no_change"
    )


    assert (
        report.artifact_count
        ==
        0
    )


    assert (
        report.persisted_dataset_ids
        ==
        ()
    )


    assert (
        after.stage
        ==
        before.stage
        ==
        "source"
    )


    assert (
        after.evidence_refs
        ==
        before.evidence_refs
    )


    assert (
        after
        .dataframe
        .equals(
            before.dataframe
        )
    )


    print(
        "Non-mutating semantic confirmation does not "
        "rewrite artifact: PASS"
    )


# ============================================================
# 4. STALE / DIFFERENT BROWSER INPUT IS REJECTED
# ============================================================

def test_stale_deterministic_input_rejected() -> None:

    reset_preparation_artifact_store_for_tests()


    put_current_clean_artifact()


    stale = (
        deterministic_frame()
    )


    stale.loc[
        0,
        "city",
    ] = (
        "CHANGED IN BROWSER"
    )


    before = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),
        )
    )


    try:

        materialize_semantic_cleaning_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            deterministic_frames={
                "sales":
                    stale,
            },

            derived_frames={
                "sales":
                    semantic_frame(),
            },

            semantic_plan=(
                semantic_plan()
            ),

            execution=(
                semantic_execution_changed()
            ),
        )


    except ValueError:

        after = (
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
            after
            .dataframe
            .equals(
                before.dataframe
            )
        )


        print(
            "Stale/browser-modified deterministic input "
            "rejected: PASS"
        )


    else:

        raise AssertionError(
            "Semantic Cleaning must reject deterministic "
            "input that differs from the server artifact."
        )


# ============================================================
# 5. DERIVED SCOPE MISMATCH
# ============================================================

def test_derived_scope_mismatch_rejected() -> None:

    reset_preparation_artifact_store_for_tests()


    put_current_clean_artifact()


    try:

        materialize_semantic_cleaning_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            deterministic_frames={
                "sales":
                    deterministic_frame(),
            },

            derived_frames={
                "invented_dataset":
                    semantic_frame(),
            },

            semantic_plan=(
                semantic_plan()
            ),

            execution=(
                semantic_execution_changed()
            ),
        )


    except ValueError:

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
            .equals(
                deterministic_frame()
            )
        )


        print(
            "Semantic derived dataset scope mismatch "
            "rejected before persistence: PASS"
        )


    else:

        raise AssertionError(
            "Unknown Semantic Cleaning output dataset "
            "must be rejected."
        )


# ============================================================
# 6. PROVENANCE SCOPE MISMATCH
# ============================================================

def test_provenance_scope_mismatch_rejected() -> None:

    reset_preparation_artifact_store_for_tests()


    put_current_clean_artifact()


    bad_execution = (
        SimpleNamespace(
            status=(
                "success"
            ),

            dataset_count=(
                1
            ),

            applied_action_count=(
                0
            ),

            skipped_action_count=(
                0
            ),

            changed_cell_count=(
                0
            ),

            rule_version=(
                "semantic_cleaning_engine_test_v0.1"
            ),

            provenance=[],
        )
    )


    try:

        materialize_semantic_cleaning_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            deterministic_frames={
                "sales":
                    deterministic_frame(),
            },

            derived_frames={
                "sales":
                    deterministic_frame(),
            },

            semantic_plan=(
                semantic_plan()
            ),

            execution=(
                bad_execution
            ),
        )


    except ValueError:

        print(
            "Semantic provenance scope mismatch rejected "
            "before persistence: PASS"
        )


    else:

        raise AssertionError(
            "Missing Semantic Cleaning provenance must "
            "be rejected."
        )


# ============================================================
# 7. CHANGED CELL COUNT MISMATCH
# ============================================================

def test_changed_cell_count_mismatch_rejected() -> None:

    reset_preparation_artifact_store_for_tests()


    put_current_clean_artifact()


    bad_execution = (
        semantic_execution_changed()
    )


    bad_execution.changed_cell_count = (
        999
    )


    try:

        materialize_semantic_cleaning_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            deterministic_frames={
                "sales":
                    deterministic_frame(),
            },

            derived_frames={
                "sales":
                    semantic_frame(),
            },

            semantic_plan=(
                semantic_plan()
            ),

            execution=(
                bad_execution
            ),
        )


    except ValueError:

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
            .equals(
                deterministic_frame()
            )
        )


        print(
            "Semantic changed-cell reconciliation "
            "enforced: PASS"
        )


    else:

        raise AssertionError(
            "Mismatched Semantic Cleaning execution counts "
            "must be rejected."
        )


# ============================================================
# 8. LATER STAGE CANNOT BE OVERWRITTEN
# ============================================================

def test_later_stage_cannot_be_overwritten() -> None:

    reset_preparation_artifact_store_for_tests()


    put_current_clean_artifact(
        stage=(
            "transform"
        )
    )


    try:

        materialize_semantic_cleaning_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            deterministic_frames={
                "sales":
                    deterministic_frame(),
            },

            derived_frames={
                "sales":
                    semantic_frame(),
            },

            semantic_plan=(
                semantic_plan()
            ),

            execution=(
                semantic_execution_changed()
            ),
        )


    except ValueError:

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
            ==
            "transform"
        )


        print(
            "Semantic Cleaning cannot overwrite later "
            "Preparation stage: PASS"
        )


    else:

        raise AssertionError(
            "Semantic Cleaning must not overwrite a "
            "TRANSFORM/COMBINE artifact."
        )


# ============================================================
# 9. MATERIALIZED RESULT IS COPY-ISOLATED
# ============================================================

def test_materialized_semantic_result_isolated() -> None:

    reset_preparation_artifact_store_for_tests()


    put_current_clean_artifact()


    derived = (
        semantic_frame()
    )


    materialize_semantic_cleaning_artifacts(
        workflow_id=(
            WORKFLOW_ID
        ),

        deterministic_frames={
            "sales":
                deterministic_frame(),
        },

        derived_frames={
            "sales":
                derived,
        },

        semantic_plan=(
            semantic_plan()
        ),

        execution=(
            semantic_execution_changed()
        ),
    )


    derived.loc[
        0,
        "city",
    ] = (
        "MODIFIED AFTER PERSISTENCE"
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
        artifact
        .dataframe
        .loc[
            0,
            "city",
        ]
        ==
        "New York"
    )


    print(
        "Semantic artifact remains isolated from caller: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS SEMANTIC CLEANING ARTIFACT BRIDGE v0.1 ==="
    )

    print()


    test_version()

    test_real_semantic_mutation_materialized()

    test_no_change_does_not_rewrite_artifact()

    test_stale_deterministic_input_rejected()

    test_derived_scope_mismatch_rejected()

    test_provenance_scope_mismatch_rejected()

    test_changed_cell_count_mismatch_rejected()

    test_later_stage_cannot_be_overwritten()

    test_materialized_semantic_result_isolated()


    print()


    print(
        "Semantic Cleaning Artifact Bridge v0.1: PASS"
    )


if __name__ == "__main__":
    main()
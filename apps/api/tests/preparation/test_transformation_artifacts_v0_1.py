from __future__ import annotations

from types import (
    SimpleNamespace,
)

import pandas as pd

from app.preparation.preparation_artifact_store import (
    get_preparation_artifact,
    list_preparation_artifacts,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)

from app.preparation.transformation_artifacts import (
    TRANSFORMATION_ARTIFACT_BRIDGE_VERSION,
    materialize_transformation_artifacts,
)


WORKFLOW_ID = (
    "workflow-transformation-artifacts-test"
)


# ============================================================
# DATA
# ============================================================

def source_frame() -> pd.DataFrame:

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


def transformed_frame() -> pd.DataFrame:

    dataframe = (
        source_frame()
    )


    dataframe[
        "amount_with_tax"
    ] = (
        dataframe[
            "amount"
        ]
        *
        1.2
    )


    return (
        dataframe
    )


def aggregate_frame() -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "total_amount": [
                    60.0,
                ],
            }
        )
    )


# ============================================================
# CURRENT ARTIFACT
# ============================================================

def put_clean_source() -> None:

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
            "clean"
        ),

        dataframe=(
            source_frame()
        ),

        parent_dataset_ids=[
            "sales",
        ],

        evidence_refs=[
            "cleaning_execution:test",
        ],
    )


# ============================================================
# CONTRACT FIXTURES
# ============================================================

def approved_plan(
    *,
    source_changed: bool = True,
    derived: bool = False,
):

    steps = []


    if (
        source_changed
    ):

        steps.append(
            SimpleNamespace(
                step_id=(
                    "step-derived-column"
                ),

                executable=(
                    True
                ),

                output_dataset_id=(
                    None
                ),

                output_dataset_filename=(
                    None
                ),
            )
        )


    if (
        derived
    ):

        steps.append(
            SimpleNamespace(
                step_id=(
                    "step-aggregate"
                ),

                executable=(
                    True
                ),

                output_dataset_id=(
                    "sales_summary"
                ),

                output_dataset_filename=(
                    "sales_summary.csv"
                ),
            )
        )


    return (
        SimpleNamespace(
            dataset_id=(
                "sales"
            ),

            dataset_filename=(
                "sales.csv"
            ),

            steps=(
                steps
            ),

            rule_version=(
                "transformation_approval_test_v0.1"
            ),
        )
    )


def execution_result(
    *,
    source_changed: bool = True,
    derived: bool = False,
):

    dataframe = (
        transformed_frame()
        if source_changed
        else source_frame()
    )


    derived_datasets = (
        {
            "sales_summary":
                aggregate_frame(),
        }
        if derived
        else {}
    )


    before_fingerprint = (
        "before-fingerprint"
    )


    after_fingerprint = (
        "after-fingerprint"
        if source_changed
        else before_fingerprint
    )


    step_reports = []


    if (
        source_changed
    ):

        step_reports.append(
            SimpleNamespace(
                step_id=(
                    "step-derived-column"
                ),

                status=(
                    "applied"
                ),
            )
        )


    if (
        derived
    ):

        step_reports.append(
            SimpleNamespace(
                step_id=(
                    "step-aggregate"
                ),

                status=(
                    "applied"
                ),
            )
        )


    report = (
        SimpleNamespace(
            dataset_id=(
                "sales"
            ),

            dataset_filename=(
                "sales.csv"
            ),

            source_data_changed=(
                source_changed
            ),

            source_fingerprint_before=(
                before_fingerprint
            ),

            source_fingerprint_after=(
                after_fingerprint
            ),

            derived_dataset_count=(
                len(
                    derived_datasets
                )
            ),

            derived_dataset_ids=list(
                derived_datasets.keys()
            ),

            steps=(
                step_reports
            ),

            rule_version=(
                "transformation_executor_test_v0.1"
            ),
        )
    )


    return (
        SimpleNamespace(
            dataframe=(
                dataframe
            ),

            derived_datasets=(
                derived_datasets
            ),

            report=(
                report
            ),
        )
    )


def passed_validation():

    return (
        SimpleNamespace(
            status=(
                "passed"
            ),

            dataset_id=(
                "sales"
            ),

            dataset_filename=(
                "sales.csv"
            ),

            valid_for_downstream=(
                True
            ),

            rule_version=(
                "post_transformation_validation_test_v0.1"
            ),
        )
    )


# ============================================================
# 1. VERSION
# ============================================================

def test_version() -> None:

    assert (
        TRANSFORMATION_ARTIFACT_BRIDGE_VERSION
        ==
        "transformation_artifact_bridge_v0.1"
    )


    print(
        "Transformation artifact bridge version: PASS"
    )


# ============================================================
# 2. MAIN DATASET MUTATION
# ============================================================

def test_source_transformation_materialized() -> None:

    reset_preparation_artifact_store_for_tests()

    put_clean_source()


    report = (
        materialize_transformation_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataframe=(
                source_frame()
            ),

            approved_plan=(
                approved_plan(
                    source_changed=True,
                    derived=False,
                )
            ),

            execution=(
                execution_result(
                    source_changed=True,
                    derived=False,
                )
            ),

            validation=(
                passed_validation()
            ),
        )
    )


    assert (
        report.materialization_kind
        ==
        "source_transformed"
    )


    assert (
        report.persisted_dataset_ids
        ==
        (
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
        ==
        "transform"
    )


    assert (
        artifact
        .dataframe
        .equals(
            transformed_frame()
        )
    )


    assert (
        "cleaning_execution:test"
        in artifact.evidence_refs
    )


    print(
        "Transformed source artifact materialized: PASS"
    )


# ============================================================
# 3. AGGREGATION DERIVED DATASET
# ============================================================

def test_derived_aggregation_materialized() -> None:

    reset_preparation_artifact_store_for_tests()

    put_clean_source()


    report = (
        materialize_transformation_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataframe=(
                source_frame()
            ),

            approved_plan=(
                approved_plan(
                    source_changed=False,
                    derived=True,
                )
            ),

            execution=(
                execution_result(
                    source_changed=False,
                    derived=True,
                )
            ),

            validation=(
                passed_validation()
            ),
        )
    )


    assert (
        report.materialization_kind
        ==
        "derived_only"
    )


    assert (
        report.persisted_dataset_ids
        ==
        (
            "sales_summary",
        )
    )


    source_artifact = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),
        )
    )


    derived_artifact = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales_summary"
            ),
        )
    )


    # Source was not mutated.
    assert (
        source_artifact.stage
        ==
        "clean"
    )


    assert (
        derived_artifact.stage
        ==
        "transform"
    )


    assert (
        derived_artifact.parent_dataset_ids
        ==
        (
            "sales",
        )
    )


    assert (
        derived_artifact
        .dataframe
        .equals(
            aggregate_frame()
        )
    )


    print(
        "Transformation aggregate artifact materialized: PASS"
    )


# ============================================================
# 4. SOURCE + DERIVED
# ============================================================

def test_source_and_derived_materialized() -> None:

    reset_preparation_artifact_store_for_tests()

    put_clean_source()


    report = (
        materialize_transformation_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataframe=(
                source_frame()
            ),

            approved_plan=(
                approved_plan(
                    source_changed=True,
                    derived=True,
                )
            ),

            execution=(
                execution_result(
                    source_changed=True,
                    derived=True,
                )
            ),

            validation=(
                passed_validation()
            ),
        )
    )


    assert (
        report.materialization_kind
        ==
        "source_and_derived"
    )


    assert (
        set(
            report.persisted_dataset_ids
        )
        ==
        {
            "sales",
            "sales_summary",
        }
    )


    print(
        "Source + derived Transformation artifacts: PASS"
    )


# ============================================================
# 5. NO CHANGE
# ============================================================

def test_no_change_does_not_rewrite_source() -> None:

    reset_preparation_artifact_store_for_tests()

    put_clean_source()


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
        materialize_transformation_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataframe=(
                source_frame()
            ),

            approved_plan=(
                approved_plan(
                    source_changed=False,
                    derived=False,
                )
            ),

            execution=(
                execution_result(
                    source_changed=False,
                    derived=False,
                )
            ),

            validation=(
                passed_validation()
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
        before.stage
        ==
        after.stage
        ==
        "clean"
    )


    assert (
        before.evidence_refs
        ==
        after.evidence_refs
    )


    print(
        "No-op Transformation does not rewrite artifact: PASS"
    )


# ============================================================
# 6. FAILED POST-VALIDATION
# ============================================================

def test_failed_validation_rejected() -> None:

    reset_preparation_artifact_store_for_tests()

    put_clean_source()


    failed_validation = (
        SimpleNamespace(
            status=(
                "failed"
            ),

            dataset_id=(
                "sales"
            ),

            dataset_filename=(
                "sales.csv"
            ),

            valid_for_downstream=(
                False
            ),

            rule_version=(
                "post_transformation_validation_test_v0.1"
            ),
        )
    )


    try:

        materialize_transformation_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataframe=(
                source_frame()
            ),

            approved_plan=(
                approved_plan()
            ),

            execution=(
                execution_result()
            ),

            validation=(
                failed_validation
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
            "clean"
        )


        print(
            "Failed Transformation validation rejected "
            "before persistence: PASS"
        )


    else:

        raise AssertionError(
            "Failed post-transformation validation must "
            "never be materialized."
        )


# ============================================================
# 7. STALE SOURCE
# ============================================================

def test_stale_source_rejected() -> None:

    reset_preparation_artifact_store_for_tests()

    put_clean_source()


    stale = (
        source_frame()
    )


    stale.loc[
        0,
        "amount",
    ] = (
        999.0
    )


    try:

        materialize_transformation_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataframe=(
                stale
            ),

            approved_plan=(
                approved_plan()
            ),

            execution=(
                execution_result()
            ),

            validation=(
                passed_validation()
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
                source_frame()
            )
        )


        print(
            "Stale Transformation source rejected: PASS"
        )


    else:

        raise AssertionError(
            "Transformation must consume the current "
            "server-owned artifact."
        )


# ============================================================
# 8. INVENTED DERIVED OUTPUT
# ============================================================

def test_invented_derived_output_rejected() -> None:

    reset_preparation_artifact_store_for_tests()

    put_clean_source()


    execution = (
        execution_result(
            source_changed=False,
            derived=False,
        )
    )


    execution.derived_datasets[
        "invented"
    ] = (
        aggregate_frame()
    )


    execution.report.derived_dataset_count = (
        1
    )


    execution.report.derived_dataset_ids = [
        "invented",
    ]


    try:

        materialize_transformation_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataframe=(
                source_frame()
            ),

            approved_plan=(
                approved_plan(
                    source_changed=False,
                    derived=False,
                )
            ),

            execution=(
                execution
            ),

            validation=(
                passed_validation()
            ),
        )


    except ValueError:

        items = (
            list_preparation_artifacts(
                workflow_id=(
                    WORKFLOW_ID
                )
            )
        )


        assert (
            len(
                items
            )
            ==
            1
        )


        assert (
            items[
                0
            ]
            .dataset_id
            ==
            "sales"
        )


        print(
            "Invented Transformation output rejected: PASS"
        )


    else:

        raise AssertionError(
            "Executor output not authorized by the "
            "Transformation plan must be rejected."
        )


# ============================================================
# 9. EXECUTOR CHANGE FLAG MISMATCH
# ============================================================

def test_source_change_flag_mismatch_rejected() -> None:

    reset_preparation_artifact_store_for_tests()

    put_clean_source()


    execution = (
        execution_result(
            source_changed=False,
            derived=False,
        )
    )


    execution.dataframe = (
        transformed_frame()
    )


    try:

        materialize_transformation_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataframe=(
                source_frame()
            ),

            approved_plan=(
                approved_plan(
                    source_changed=False,
                    derived=False,
                )
            ),

            execution=(
                execution
            ),

            validation=(
                passed_validation()
            ),
        )


    except ValueError:

        print(
            "Transformation source-change reconciliation "
            "enforced: PASS"
        )


    else:

        raise AssertionError(
            "source_data_changed must match the actual "
            "DataFrame result."
        )


# ============================================================
# 10. LATER STAGE CANNOT BE RE-TRANSFORMED
# ============================================================

def test_later_stage_rejected() -> None:

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
            "combine"
        ),

        dataframe=(
            source_frame()
        ),

        parent_dataset_ids=[
            "sales",
            "customers",
        ],
    )


    try:

        materialize_transformation_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_dataframe=(
                source_frame()
            ),

            approved_plan=(
                approved_plan()
            ),

            execution=(
                execution_result()
            ),

            validation=(
                passed_validation()
            ),
        )


    except ValueError:

        print(
            "Transformation cannot overwrite later stage: PASS"
        )


    else:

        raise AssertionError(
            "TRANSFORM must not overwrite COMBINE output."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS TRANSFORMATION ARTIFACT BRIDGE v0.1 ==="
    )

    print()


    test_version()

    test_source_transformation_materialized()

    test_derived_aggregation_materialized()

    test_source_and_derived_materialized()

    test_no_change_does_not_rewrite_source()

    test_failed_validation_rejected()

    test_stale_source_rejected()

    test_invented_derived_output_rejected()

    test_source_change_flag_mismatch_rejected()

    test_later_stage_rejected()


    print()


    print(
        "Transformation Artifact Bridge v0.1: PASS"
    )


if __name__ == "__main__":
    main()
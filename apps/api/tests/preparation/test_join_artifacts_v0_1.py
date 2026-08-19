from __future__ import annotations

from types import (
    SimpleNamespace,
)

import pandas as pd

from app.preparation.join_artifacts import (
    JOIN_ARTIFACT_BRIDGE_VERSION,
    materialize_join_artifacts,
)

from app.preparation.preparation_artifact_store import (
    get_preparation_artifact,
    list_preparation_artifacts,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)


WORKFLOW_ID = (
    "workflow-join-artifact-test"
)


# ============================================================
# DATA
# ============================================================

def customers_frame() -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                    "c3",
                ],

                "segment": [
                    "A",
                    "B",
                    "A",
                ],
            }
        )
    )


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
# STORE FIXTURE
# ============================================================

def put_sources(
    *,
    sales_stage: str = "clean",
    customers_stage: str = "clean",
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
            sales_stage
        ),

        dataframe=(
            sales_frame()
        ),

        parent_dataset_ids=[
            "sales",
        ],

        evidence_refs=[
            "cleaning:sales",
        ],
    )


    put_preparation_artifact(
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
            customers_stage
        ),

        dataframe=(
            customers_frame()
        ),

        parent_dataset_ids=[
            "customers",
        ],

        evidence_refs=[
            "cleaning:customers",
        ],
    )


# ============================================================
# JOIN FIXTURES
# ============================================================

def approved_join():

    return (
        SimpleNamespace(
            request_id=(
                "join-sales-customers"
            ),

            left_dataset_id=(
                "sales"
            ),

            left_dataset_filename=(
                "sales.csv"
            ),

            right_dataset_id=(
                "customers"
            ),

            right_dataset_filename=(
                "customers.csv"
            ),

            join_type=(
                SimpleNamespace(
                    value=(
                        "left"
                    )
                )
            ),

            keys=[
                SimpleNamespace(
                    left_column=(
                        "customer_id"
                    ),

                    right_column=(
                        "customer_id"
                    ),
                )
            ],

            detected_cardinality=(
                SimpleNamespace(
                    value=(
                        "many_to_one"
                    )
                )
            ),

            output_dataset_id=(
                "sales_customers"
            ),

            output_dataset_filename=(
                "sales_customers.csv"
            ),

            executable=(
                True
            ),
        )
    )


def approved_plan():

    return (
        SimpleNamespace(
            total_join_count=(
                1
            ),

            approved_count=(
                1
            ),

            rejected_count=(
                0
            ),

            deferred_count=(
                0
            ),

            pending_count=(
                0
            ),

            executable_join_count=(
                1
            ),

            ready_for_execution=(
                True
            ),

            joins=[
                approved_join(),
            ],

            rule_version=(
                "join_approval_test_v0.1"
            ),
        )
    )


def execution_result():

    return (
        SimpleNamespace(
            joined_datasets={
                "sales_customers":
                    joined_frame(),
            },

            report=(
                SimpleNamespace(
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

                    rule_version=(
                        "join_executor_test_v0.1"
                    ),
                )
            ),
        )
    )


def passed_validation():

    return (
        SimpleNamespace(
            status=(
                "passed"
            ),

            valid_for_downstream=(
                True
            ),

            total_join_count=(
                1
            ),

            validated_join_count=(
                1
            ),

            passed_check_count=(
                10
            ),

            failed_check_count=(
                0
            ),

            warning_count=(
                0
            ),

            output_dataset_count=(
                1
            ),

            join_validations=[],

            checks=[],

            rule_version=(
                "post_join_validation_test_v0.1"
            ),
        )
    )


# ============================================================
# 1. VERSION
# ============================================================

def test_version() -> None:

    assert (
        JOIN_ARTIFACT_BRIDGE_VERSION
        ==
        "join_artifact_bridge_v0.1"
    )


    print(
        "Join artifact bridge version: PASS"
    )


# ============================================================
# 2. VALIDATED JOIN MATERIALIZED
# ============================================================

def test_join_materialized() -> None:

    reset_preparation_artifact_store_for_tests()

    put_sources()


    report = (
        materialize_join_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_datasets={
                "sales":
                    sales_frame(),

                "customers":
                    customers_frame(),
            },

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
    )


    assert (
        report.output_dataset_ids
        ==
        (
            "sales_customers",
        )
    )


    assert (
        report.artifact_count
        ==
        1
    )


    artifact = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales_customers"
            ),
        )
    )


    assert (
        artifact.stage
        ==
        "combine"
    )


    assert (
        artifact.parent_dataset_ids
        ==
        (
            "sales",
            "customers",
        )
    )


    assert (
        artifact
        .dataframe
        .equals(
            joined_frame()
        )
    )


    assert (
        (
            "join_request:"
            "join-sales-customers"
        )
        in artifact.evidence_refs
    )


    assert (
        (
            "join_key:"
            "customer_id=customer_id"
        )
        in artifact.evidence_refs
    )


    print(
        "Validated JOIN artifact materialized: PASS"
    )


# ============================================================
# 3. SOURCE ARTIFACTS ARE PRESERVED
# ============================================================

def test_sources_are_preserved() -> None:

    reset_preparation_artifact_store_for_tests()

    put_sources()


    materialize_join_artifacts(
        workflow_id=(
            WORKFLOW_ID
        ),

        source_datasets={
            "sales":
                sales_frame(),

            "customers":
                customers_frame(),
        },

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


    sales = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),
        )
    )


    customers = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "customers"
            ),
        )
    )


    assert (
        sales.stage
        ==
        "clean"
    )


    assert (
        customers.stage
        ==
        "clean"
    )


    assert (
        sales
        .dataframe
        .equals(
            sales_frame()
        )
    )


    assert (
        customers
        .dataframe
        .equals(
            customers_frame()
        )
    )


    print(
        "JOIN preserves source artifacts: PASS"
    )


# ============================================================
# 4. FAILED VALIDATION
# ============================================================

def test_failed_validation_rejected() -> None:

    reset_preparation_artifact_store_for_tests()

    put_sources()


    failed = (
        passed_validation()
    )


    failed.status = (
        "failed"
    )


    failed.valid_for_downstream = (
        False
    )


    failed.failed_check_count = (
        1
    )


    try:

        materialize_join_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_datasets={
                "sales":
                    sales_frame(),

                "customers":
                    customers_frame(),
            },

            approved_plan=(
                approved_plan()
            ),

            execution=(
                execution_result()
            ),

            validation=(
                failed
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
            {
                item.dataset_id
                for item in items
            }
            ==
            {
                "sales",
                "customers",
            }
        )


        print(
            "Failed Post-Join Validation rejected before "
            "persistence: PASS"
        )


    else:

        raise AssertionError(
            "Failed JOIN validation must never be "
            "materialized."
        )


# ============================================================
# 5. STALE SOURCE
# ============================================================

def test_stale_source_rejected() -> None:

    reset_preparation_artifact_store_for_tests()

    put_sources()


    stale_sales = (
        sales_frame()
    )


    stale_sales.loc[
        0,
        "amount",
    ] = (
        999.0
    )


    try:

        materialize_join_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_datasets={
                "sales":
                    stale_sales,

                "customers":
                    customers_frame(),
            },

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
            "Stale JOIN source rejected: PASS"
        )


    else:

        raise AssertionError(
            "JOIN must consume the current server-owned "
            "Preparation artifacts."
        )


# ============================================================
# 6. INVENTED OUTPUT
# ============================================================

def test_invented_output_rejected() -> None:

    reset_preparation_artifact_store_for_tests()

    put_sources()


    execution = (
        execution_result()
    )


    execution.joined_datasets[
        "invented"
    ] = (
        joined_frame()
    )


    execution.report.output_dataset_count = (
        2
    )


    execution.report.output_dataset_ids = [
        "sales_customers",
        "invented",
    ]


    try:

        materialize_join_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_datasets={
                "sales":
                    sales_frame(),

                "customers":
                    customers_frame(),
            },

            approved_plan=(
                approved_plan()
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
            "Invented JOIN output rejected: PASS"
        )


    else:

        raise AssertionError(
            "Executor output absent from approved JOIN plan "
            "must be rejected."
        )


# ============================================================
# 7. MISSING OUTPUT
# ============================================================

def test_missing_output_rejected() -> None:

    reset_preparation_artifact_store_for_tests()

    put_sources()


    execution = (
        execution_result()
    )


    execution.joined_datasets = {}


    execution.report.output_dataset_count = (
        0
    )


    execution.report.output_dataset_ids = []


    try:

        materialize_join_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_datasets={
                "sales":
                    sales_frame(),

                "customers":
                    customers_frame(),
            },

            approved_plan=(
                approved_plan()
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
            "Missing approved JOIN output rejected: PASS"
        )


    else:

        raise AssertionError(
            "Approved executable JOIN must produce its "
            "declared output dataset."
        )


# ============================================================
# 8. EXISTING OUTPUT COLLISION
# ============================================================

def test_existing_output_collision_rejected() -> None:

    reset_preparation_artifact_store_for_tests()

    put_sources()


    put_preparation_artifact(
        workflow_id=(
            WORKFLOW_ID
        ),

        dataset_id=(
            "sales_customers"
        ),

        dataset_filename=(
            "old.csv"
        ),

        stage=(
            "transform"
        ),

        dataframe=(
            joined_frame()
        ),
    )


    try:

        materialize_join_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_datasets={
                "sales":
                    sales_frame(),

                "customers":
                    customers_frame(),
            },

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

        existing = (
            get_preparation_artifact(
                workflow_id=(
                    WORKFLOW_ID
                ),

                dataset_id=(
                    "sales_customers"
                ),
            )
        )


        assert (
            existing.stage
            ==
            "transform"
        )


        assert (
            existing.dataset_filename
            ==
            "old.csv"
        )


        print(
            "Existing JOIN output collision rejected: PASS"
        )


    else:

        raise AssertionError(
            "JOIN must not overwrite an existing artifact."
        )


# ============================================================
# 9. TRANSFORM SOURCE IS ALLOWED
# ============================================================

def test_transform_source_allowed() -> None:

    reset_preparation_artifact_store_for_tests()


    put_sources(
        sales_stage=(
            "transform"
        ),

        customers_stage=(
            "transform"
        ),
    )


    materialize_join_artifacts(
        workflow_id=(
            WORKFLOW_ID
        ),

        source_datasets={
            "sales":
                sales_frame(),

            "customers":
                customers_frame(),
        },

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


    artifact = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales_customers"
            ),
        )
    )


    assert (
        artifact.stage
        ==
        "combine"
    )


    print(
        "JOIN accepts validated TRANSFORM inputs: PASS"
    )


# ============================================================
# 10. COMBINE SOURCE IS NOT REUSED IN V0.1
# ============================================================

def test_existing_combine_source_rejected() -> None:

    reset_preparation_artifact_store_for_tests()


    put_sources(
        sales_stage=(
            "combine"
        ),

        customers_stage=(
            "clean"
        ),
    )


    try:

        materialize_join_artifacts(
            workflow_id=(
                WORKFLOW_ID
            ),

            source_datasets={
                "sales":
                    sales_frame(),

                "customers":
                    customers_frame(),
            },

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
            "JOIN v0.1 rejects recursive COMBINE input: PASS"
        )


    else:

        raise AssertionError(
            "JOIN v0.1 must not silently chain existing "
            "COMBINE artifacts."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS JOIN ARTIFACT BRIDGE v0.1 ==="
    )

    print()


    test_version()

    test_join_materialized()

    test_sources_are_preserved()

    test_failed_validation_rejected()

    test_stale_source_rejected()

    test_invented_output_rejected()

    test_missing_output_rejected()

    test_existing_output_collision_rejected()

    test_transform_source_allowed()

    test_existing_combine_source_rejected()


    print()

    print(
        "Join Artifact Bridge v0.1: PASS"
    )


if __name__ == "__main__":
    main()
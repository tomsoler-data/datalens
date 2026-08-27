from __future__ import annotations

import pandas as pd

from app.preparation.preparation_artifact_store import (
    PREPARATION_ARTIFACT_STORE_VERSION,
    PreparationArtifactDatasetNotFoundError,
    PreparationArtifactWorkflowNotFoundError,
    get_preparation_artifact,
    get_preparation_dataframe,
    get_preparation_dataframe_map,
    list_preparation_artifacts,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)


WORKFLOW_ID = (
    "workflow-artifact-test"
)


# ============================================================
# FIXTURES
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


# ============================================================
# 1. VERSION
# ============================================================

def test_version() -> None:

    assert (
        PREPARATION_ARTIFACT_STORE_VERSION
        == "preparation_artifact_store_v0.4"
    )


    print(
        "Artifact store version: PASS"
    )


# ============================================================
# 2. PUT + GET
# ============================================================

def test_put_get() -> None:

    reset_preparation_artifact_store_for_tests()


    dataframe = (
        source_frame()
    )


    info = (
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
                dataframe
            ),

            evidence_refs=[
                "ingestion:sales",
            ],
        )
    )


    assert (
        info.dataset_id
        == "sales"
    )


    assert (
        info.rows
        == 3
    )


    assert (
        info.columns
        == 2
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
            dataframe
        )
    )


    print(
        "Artifact put/get: PASS"
    )


# ============================================================
# 3. WRITE COPY ISOLATION
# ============================================================

def test_write_copy_isolation() -> None:

    reset_preparation_artifact_store_for_tests()


    dataframe = (
        source_frame()
    )


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
            dataframe
        ),
    )


    dataframe.loc[
        0,
        "amount",
    ] = 999.0


    stored = (
        get_preparation_dataframe(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),
        )
    )


    assert (
        stored.loc[
            0,
            "amount",
        ]
        == 10.0
    )


    print(
        "Artifact write isolation: PASS"
    )


# ============================================================
# 4. READ COPY ISOLATION
# ============================================================

def test_read_copy_isolation() -> None:

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
            source_frame()
        ),
    )


    first_read = (
        get_preparation_dataframe(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),
        )
    )


    first_read.loc[
        0,
        "amount",
    ] = 999.0


    second_read = (
        get_preparation_dataframe(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales"
            ),
        )
    )


    assert (
        second_read.loc[
            0,
            "amount",
        ]
        == 10.0
    )


    print(
        "Artifact read isolation: PASS"
    )


# ============================================================
# 5. STAGE REPLACEMENT
# ============================================================

def test_stage_replacement() -> None:

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
            source_frame()
        ),
    )


    cleaned = (
        source_frame()
    )


    cleaned[
        "amount"
    ] = (
        cleaned[
            "amount"
        ]
        * 2
    )


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
            cleaned
        ),

        parent_dataset_ids=[
            "sales",
        ],

        evidence_refs=[
            "cleaning:execution:v0.1",
        ],
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
        .dataframe[
            "amount"
        ]
        .tolist()
        == [
            20.0,
            40.0,
            60.0,
        ]
    )


    assert (
        artifact.parent_dataset_ids
        == (
            "sales",
        )
    )


    print(
        "Artifact stage replacement: PASS"
    )


# ============================================================
# 6. DERIVED DATASET
# ============================================================

def test_derived_dataset_lineage() -> None:

    reset_preparation_artifact_store_for_tests()


    aggregate = (
        pd.DataFrame(
            {
                "total_amount": [
                    60.0,
                ]
            }
        )
    )


    put_preparation_artifact(
        workflow_id=(
            WORKFLOW_ID
        ),

        dataset_id=(
            "sales_summary"
        ),

        dataset_filename=(
            "sales_summary.csv"
        ),

        stage=(
            "transform"
        ),

        dataframe=(
            aggregate
        ),

        parent_dataset_ids=[
            "sales",
        ],

        evidence_refs=[
            "transformation:aggregate",
        ],
    )


    artifact = (
        get_preparation_artifact(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "sales_summary"
            ),
        )
    )


    assert (
        artifact.parent_dataset_ids
        == (
            "sales",
        )
    )


    assert (
        artifact.stage
        == "transform"
    )


    print(
        "Derived dataset lineage: PASS"
    )


# ============================================================
# 7. DATAFRAME MAP
# ============================================================

def test_dataframe_map() -> None:

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
            "clean"
        ),

        dataframe=(
            source_frame()
        ),
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
            "clean"
        ),

        dataframe=(
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
        ),
    )


    datasets = (
        get_preparation_dataframe_map(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_ids=[
                "sales",
                "customers",
            ],
        )
    )


    assert (
        set(
            datasets.keys()
        )
        == {
            "sales",
            "customers",
        }
    )


    assert (
        datasets[
            "sales"
        ]
        .shape
        == (
            3,
            2,
        )
    )


    print(
        "Artifact dataframe map: PASS"
    )


# ============================================================
# 8. LIST
# ============================================================

def test_list_artifacts() -> None:

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
            source_frame()
        ),
    )


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
        == 1
    )


    assert (
        items[
            0
        ]
        .dataset_id
        == "sales"
    )


    print(
        "Artifact listing: PASS"
    )


# ============================================================
# 9. UNKNOWN DATASET IN EXISTING WORKFLOW
# ============================================================

def test_unknown_dataset() -> None:
    """
    Distinguish:

        existing workflow
        + unknown dataset

    from an entirely unknown workflow.
    """

    reset_preparation_artifact_store_for_tests()


    # Create the workflow namespace by storing one known
    # dataset first.
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
            source_frame()
        ),
    )


    try:

        get_preparation_dataframe(
            workflow_id=(
                WORKFLOW_ID
            ),

            dataset_id=(
                "missing"
            ),
        )


    except (
        PreparationArtifactDatasetNotFoundError
    ):

        print(
            "Unknown dataset in existing workflow "
            "rejected: PASS"
        )


    else:

        raise AssertionError(
            "Unknown dataset must not be returned "
            "from an existing workflow."
        )


# ============================================================
# 10. UNKNOWN WORKFLOW
# ============================================================

def test_unknown_workflow() -> None:
    """
    An unknown workflow is a different failure mode from an
    unknown dataset inside an existing workflow.
    """

    reset_preparation_artifact_store_for_tests()


    try:

        get_preparation_dataframe(
            workflow_id=(
                "workflow-does-not-exist"
            ),

            dataset_id=(
                "sales"
            ),
        )


    except (
        PreparationArtifactWorkflowNotFoundError
    ):

        print(
            "Unknown artifact workflow rejected: PASS"
        )


    else:

        raise AssertionError(
            "Unknown workflow must not expose artifacts."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS PREPARATION ARTIFACT STORE v0.1 ==="
    )

    print()


    test_version()

    test_put_get()

    test_write_copy_isolation()

    test_read_copy_isolation()

    test_stage_replacement()

    test_derived_dataset_lineage()

    test_dataframe_map()

    test_list_artifacts()

    test_unknown_dataset()

    test_unknown_workflow()


    print()


    print(
        "Preparation Artifact Store v0.1: PASS"
    )


if __name__ == "__main__":
    main()
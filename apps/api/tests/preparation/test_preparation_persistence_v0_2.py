from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from app.preparation.preparation_artifact_store import (
    PREPARATION_ARTIFACT_STORE_VERSION,
    PreparationArtifactStore,
    PreparationArtifactWorkflowNotFoundError,
    get_preparation_dataframe,
    list_preparation_artifacts,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)
from app.preparation.preparation_session import (
    PREPARATION_SESSION_RULE_VERSION,
    PREPARATION_SESSION_STORE_VERSION,
    PreparationSessionStore,
    create_preparation_session,
    get_preparation_session,
    record_required_stage_signal,
    reset_preparation_session_store_for_tests,
)
from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# ISOLATED STORE PATHS
# ============================================================

_TEMP_DIRECTORY = tempfile.TemporaryDirectory(
    prefix="datalens-preparation-persistence-"
)

_TEST_ROOT = Path(
    _TEMP_DIRECTORY.name
)

os.environ[
    "DATALENS_PREPARATION_SESSION_STORE_PATH"
] = str(
    _TEST_ROOT
    /
    "sessions.json"
)

os.environ[
    "DATALENS_PREPARATION_ARTIFACT_STORE_PATH"
] = str(
    _TEST_ROOT
    /
    "artifacts"
)


# ============================================================
# HELPERS
# ============================================================


def reset_state() -> None:
    reset_preparation_session_store_for_tests()
    reset_preparation_artifact_store_for_tests()


def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [
                "O1",
                "O2",
                "O3",
            ],
            "amount": pd.Series(
                [
                    10,
                    None,
                    30,
                ],
                dtype="Int64",
            ),
            "order_date": (
                pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-03",
                    ]
                )
                .astype(
                    "datetime64[us]"
                )
            ),
            "is_priority": pd.Series(
                [
                    True,
                    False,
                    None,
                ],
                dtype="boolean",
            ),
        }
    )


# ============================================================
# TESTS
# ============================================================


def test_session_survives_fresh_store_instance() -> None:
    reset_state()

    session = create_preparation_session(
        selected_analysis_dataset_ids=[
            "dataset:orders",
        ]
    )

    updated = record_required_stage_signal(
        workflow_id=session.workflow_id,
        stage=PreparationStage.IMPORT,
        completed=True,
        dataset_ids=[
            "dataset:orders",
        ],
        evidence_refs=[
            "test:import",
        ],
        blocking_reasons=[],
    )

    fresh_store = PreparationSessionStore()

    restored = fresh_store.get(
        session.workflow_id
    )

    assert restored.workflow_id == session.workflow_id
    assert restored.revision == updated.revision
    assert restored.selected_analysis_dataset_ids == [
        "dataset:orders",
    ]
    assert restored.import_stage.completed is True

    public_restored = get_preparation_session(
        session.workflow_id
    )

    assert public_restored.revision == updated.revision

    print(
        "[PASS] Preparation session survives a fresh store instance"
    )


def test_artifact_survives_fresh_store_instance() -> None:
    reset_state()

    workflow_id = "prep:persistence-artifact-test"
    dataset_id = "dataset:orders"
    expected = source_frame()

    put_preparation_artifact(
        workflow_id=workflow_id,
        dataset_id=dataset_id,
        dataset_filename="orders.csv",
        stage="source",
        dataframe=expected,
        parent_dataset_ids=[],
        evidence_refs=[
            "test:source",
        ],
    )

    fresh_store = PreparationArtifactStore()

    restored = fresh_store.get_dataframe(
        workflow_id=workflow_id,
        dataset_id=dataset_id,
    )

    pd.testing.assert_frame_equal(
        restored,
        expected,
    )


    assert str(
        restored[
            "order_date"
        ].dtype
    ) == "datetime64[us]"


    print(
        "[PASS] Exact datetime resolution survives persistence"
    )


    restored.iloc[
        0,
        0,
    ] = "MUTATED"

    second_read = get_preparation_dataframe(
        workflow_id=workflow_id,
        dataset_id=dataset_id,
    )

    assert second_read.iloc[
        0,
        0,
    ] == "O1"

    print(
        "[PASS] Preparation artifact survives a fresh store instance"
    )
    print(
        "[PASS] Artifact read remains copy-isolated"
    )


def test_artifact_manifest_survives_and_preserves_lineage() -> None:
    reset_state()

    workflow_id = "prep:persistence-lineage-test"

    put_preparation_artifact(
        workflow_id=workflow_id,
        dataset_id="dataset:orders",
        dataset_filename="orders.csv",
        stage="source",
        dataframe=source_frame(),
        evidence_refs=[
            "test:source",
        ],
    )

    derived = source_frame().copy(
        deep=True
    )

    derived[
        "amount_filled"
    ] = derived[
        "amount"
    ].fillna(
        0
    )

    put_preparation_artifact(
        workflow_id=workflow_id,
        dataset_id="transform:orders:filled",
        dataset_filename="orders__filled.csv",
        stage="transform",
        dataframe=derived,
        parent_dataset_ids=[
            "dataset:orders",
        ],
        evidence_refs=[
            "test:transform",
        ],
    )

    fresh_store = PreparationArtifactStore()

    infos = fresh_store.list(
        workflow_id=workflow_id
    )

    by_id = {
        item.dataset_id: item
        for item in infos
    }

    assert set(
        by_id
    ) == {
        "dataset:orders",
        "transform:orders:filled",
    }

    assert (
        by_id[
            "transform:orders:filled"
        ].parent_dataset_ids
        == (
            "dataset:orders",
        )
    )

    assert (
        by_id[
            "transform:orders:filled"
        ].rows
        == 3
    )

    print(
        "[PASS] Persistent artifact manifest preserves lineage"
    )


def test_delete_workflow_is_persistent() -> None:
    reset_state()

    workflow_id = "prep:persistence-delete-test"

    put_preparation_artifact(
        workflow_id=workflow_id,
        dataset_id="dataset:orders",
        dataset_filename="orders.csv",
        stage="source",
        dataframe=source_frame(),
    )

    store = PreparationArtifactStore()

    store.delete_workflow(
        workflow_id=workflow_id
    )

    fresh_store = PreparationArtifactStore()

    try:
        fresh_store.dataframe_map(
            workflow_id=workflow_id
        )
    except PreparationArtifactWorkflowNotFoundError:
        pass
    else:
        raise AssertionError(
            "Deleted workflow must stay deleted after a fresh store instance."
        )

    assert list_preparation_artifacts(
        workflow_id=workflow_id
    ) == []

    print(
        "[PASS] Artifact workflow deletion is durable"
    )


def test_versions() -> None:
    # SQLITE_SESSION_PERSISTENCE_EXPECTATION_V0_1
    # Persistence changed; Preparation business rules did not.
    assert PREPARATION_SESSION_RULE_VERSION == (
        "preparation_session_v0.2"
    )

    assert PREPARATION_SESSION_STORE_VERSION == (
        "preparation_session_sqlite_store_v0.1"
    )

    assert PREPARATION_ARTIFACT_STORE_VERSION == (
        "preparation_artifact_store_v0.4"
    )

    print(
        "[PASS] Persistence rule versions"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=== DATALENS PREPARATION PERSISTENCE v0.2 ==="
    )
    print()

    test_session_survives_fresh_store_instance()
    test_artifact_survives_fresh_store_instance()
    test_artifact_manifest_survives_and_preserves_lineage()
    test_delete_workflow_is_persistent()
    test_versions()

    print()
    print(
        "PASS - Preparation persistence v0.2"
    )


if __name__ == "__main__":
    main()

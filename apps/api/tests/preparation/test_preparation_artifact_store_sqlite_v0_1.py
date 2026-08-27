from __future__ import annotations

import json
import os
import sqlite3
import tempfile

from pathlib import Path

import pandas as pd


# ============================================================
# ISOLATED ENVIRONMENT
# ============================================================


_TEMP_DIRECTORY = tempfile.TemporaryDirectory(
    prefix=
        "datalens-artifact-store-sqlite-"
)

_ROOT = Path(
    _TEMP_DIRECTORY.name
)

_DATABASE_PATH = (
    _ROOT
    /
    "datalens.sqlite3"
)

_ARTIFACT_ROOT = (
    _ROOT
    /
    "artifacts"
)


os.environ[
    "DATALENS_SQLITE_PATH"
] = str(
    _DATABASE_PATH
)

os.environ[
    "DATALENS_PREPARATION_ARTIFACT_STORE_PATH"
] = str(
    _ARTIFACT_ROOT
)


from app.persistence.sqlite_database import (
    sqlite_schema_version,
)

from app.preparation.preparation_artifact_index import (
    preparation_artifact_store_scope,
)

from app.preparation.preparation_artifact_store import (
    PREPARATION_ARTIFACT_MANIFEST_VERSION,
    PREPARATION_ARTIFACT_STORE_VERSION,
    PreparationArtifactStore,
    PreparationArtifactWorkflowNotFoundError,
    get_preparation_dataframe,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)


# ============================================================
# DATA
# ============================================================


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [
                "O1",
                "O2",
                "O3",
            ],

            "amount":
                pd.Series(
                    [
                        10,
                        None,
                        30,
                    ],
                    dtype=
                        "Int64",
                ),

            "order_date":
                (
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
        }
    )


def sqlite_rows() -> list[
    tuple
]:
    connection = sqlite3.connect(
        str(
            _DATABASE_PATH
        )
    )

    try:
        return list(
            connection.execute(
                """
                SELECT
                    workflow_id,
                    dataset_id,
                    dataset_filename,
                    stage,
                    rows,
                    columns,
                    data_path
                FROM preparation_artifacts
                WHERE store_root = ?
                ORDER BY
                    workflow_id,
                    dataset_id
                """,
                (
                    preparation_artifact_store_scope(
                        _ARTIFACT_ROOT
                    ),
                ),
            )
        )

    finally:
        connection.close()


def reset_state() -> None:
    reset_preparation_artifact_store_for_tests()


# ============================================================
# 1. FRESH SQLITE STORE
# ============================================================


def test_new_store_uses_sqlite() -> None:
    reset_state()


    put_preparation_artifact(
        workflow_id=
            "prep:sqlite",

        dataset_id=
            "dataset:orders",

        dataset_filename=
            "orders.csv",

        stage=
            "source",

        dataframe=
            frame(),

        evidence_refs=[
            "test:source",
        ],
    )


    assert (
        sqlite_schema_version()
        ==
        3
    )


    rows = sqlite_rows()

    assert len(
        rows
    ) == 1

    assert rows[
        0
    ][
        0
    ] == "prep:sqlite"

    assert rows[
        0
    ][
        1
    ] == "dataset:orders"


    # Fresh SQLite-backed stores no longer need to create
    # manifest.json.
    assert (
        (
            _ARTIFACT_ROOT
            /
            "manifest.json"
        ).exists()
        is False
    )


    data_files = list(
        (
            _ARTIFACT_ROOT
            /
            "data"
        ).glob(
            "*.json.gz"
        )
    )

    assert len(
        data_files
    ) == 1


    print(
        "[PASS] new store uses SQLite metadata + filesystem data"
    )


# ============================================================
# 2. FRESH STORE / DATETIME FIDELITY
# ============================================================


def test_fresh_store_restores_sqlite_metadata(
) -> None:
    fresh_store = (
        PreparationArtifactStore()
    )


    restored = (
        fresh_store.get_dataframe(
            workflow_id=
                "prep:sqlite",

            dataset_id=
                "dataset:orders",
        )
    )


    pd.testing.assert_frame_equal(
        restored,
        frame(),
    )


    assert (
        str(
            restored[
                "order_date"
            ].dtype
        )
        ==
        "datetime64[us]"
    )


    print(
        "[PASS] fresh store restores SQLite metadata"
    )

    print(
        "[PASS] exact datetime dtype survives SQLite metadata"
    )


# ============================================================
# 3. REPLACE
# ============================================================


def test_replace_semantics() -> None:
    before = sqlite_rows()

    assert len(
        before
    ) == 1


    old_relative_path = (
        before[
            0
        ][
            6
        ]
    )

    old_path = (
        _ARTIFACT_ROOT
        /
        old_relative_path
    )

    assert old_path.exists()


    replacement = frame()

    replacement[
        "amount"
    ] = pd.Series(
        [
            100,
            200,
            300,
        ],
        dtype=
            "Int64",
    )


    put_preparation_artifact(
        workflow_id=
            "prep:sqlite",

        dataset_id=
            "dataset:orders",

        dataset_filename=
            "orders-v2.csv",

        stage=
            "clean",

        dataframe=
            replacement,

        evidence_refs=[
            "test:replacement",
        ],

        replace=
            True,
    )


    after = sqlite_rows()

    assert len(
        after
    ) == 1

    assert (
        after[
            0
        ][
            2
        ]
        ==
        "orders-v2.csv"
    )

    assert (
        after[
            0
        ][
            3
        ]
        ==
        "clean"
    )


    # Metadata commit happened before cleanup of previous data.
    assert (
        old_path.exists()
        is False
    )


    restored = (
        get_preparation_dataframe(
            workflow_id=
                "prep:sqlite",

            dataset_id=
                "dataset:orders",
        )
    )


    assert (
        restored[
            "amount"
        ].tolist()
        ==
        [
            100,
            200,
            300,
        ]
    )


    print(
        "[PASS] replace=True updates SQLite and removes old data file"
    )


# ============================================================
# 4. LEGACY MANIFEST IMPORT EXACTLY ONCE
# ============================================================


def test_legacy_manifest_import_once() -> None:
    reset_state()


    _ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_directory = (
        _ARTIFACT_ROOT
        /
        "data"
    )

    data_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    legacy_data_path = (
        data_directory
        /
        "legacy.json.gz"
    )


    legacy_frame = frame()

    legacy_frame.to_json(
        legacy_data_path,
        orient=
            "table",
        date_format=
            "iso",
        force_ascii=
            False,
        compression=
            "gzip",
        index=
            True,
    )


    manifest = {
        "manifest_version":
            PREPARATION_ARTIFACT_MANIFEST_VERSION,

        "workflows": {
            "prep:legacy": {
                "dataset:legacy": {
                    "workflow_id":
                        "prep:legacy",

                    "dataset_id":
                        "dataset:legacy",

                    "dataset_filename":
                        "legacy.csv",

                    "stage":
                        "source",

                    "rows":
                        3,

                    "columns":
                        3,

                    "parent_dataset_ids":
                        [],

                    "evidence_refs": [
                        "test:legacy",
                    ],

                    "datetime_dtypes": [
                        {
                            "position":
                                2,

                            "name":
                                "order_date",

                            "dtype":
                                "datetime64[us]",
                        },
                    ],

                    "data_path":
                        "data/legacy.json.gz",
                },
            },
        },
    }


    manifest_path = (
        _ARTIFACT_ROOT
        /
        "manifest.json"
    )


    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=
                False,
            indent=
                2,
        ),
        encoding=
            "utf-8",
    )


    first_store = (
        PreparationArtifactStore()
    )


    first = (
        first_store.get_dataframe(
            workflow_id=
                "prep:legacy",

            dataset_id=
                "dataset:legacy",
        )
    )


    assert (
        first[
            "order_id"
        ].tolist()
        ==
        [
            "O1",
            "O2",
            "O3",
        ]
    )

    assert (
        str(
            first[
                "order_date"
            ].dtype
        )
        ==
        "datetime64[us]"
    )


    # Change legacy JSON after successful import.
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version":
                    PREPARATION_ARTIFACT_MANIFEST_VERSION,

                "workflows":
                    {},
            }
        ),
        encoding=
            "utf-8",
    )


    second_store = (
        PreparationArtifactStore()
    )


    second = (
        second_store.get_dataframe(
            workflow_id=
                "prep:legacy",

            dataset_id=
                "dataset:legacy",
        )
    )


    assert (
        second[
            "order_id"
        ].tolist()
        ==
        [
            "O1",
            "O2",
            "O3",
        ]
    )


    assert len(
        sqlite_rows()
    ) == 1


    print(
        "[PASS] legacy manifest imports exactly once"
    )

    print(
        "[PASS] SQLite remains authoritative after legacy JSON changes"
    )


# ============================================================
# 5. WORKFLOW DELETE
# ============================================================


def test_workflow_delete() -> None:
    reset_state()


    put_preparation_artifact(
        workflow_id=
            "prep:delete",

        dataset_id=
            "dataset:orders",

        dataset_filename=
            "orders.csv",

        stage=
            "source",

        dataframe=
            frame(),
    )


    store = (
        PreparationArtifactStore()
    )


    store.delete_workflow(
        workflow_id=
            "prep:delete"
    )


    assert (
        sqlite_rows()
        ==
        []
    )


    assert (
        list(
            (
                _ARTIFACT_ROOT
                /
                "data"
            ).glob(
                "*.json.gz"
            )
        )
        ==
        []
    )


    fresh_store = (
        PreparationArtifactStore()
    )


    try:
        fresh_store.dataframe_map(
            workflow_id=
                "prep:delete"
        )

    except PreparationArtifactWorkflowNotFoundError:
        pass

    else:
        raise AssertionError(
            (
                "Deleted workflow must not "
                "reappear from SQLite."
            )
        )


    print(
        "[PASS] workflow delete is durable in SQLite"
    )


# ============================================================
# 6. VERSION
# ============================================================


def test_version() -> None:
    assert (
        PREPARATION_ARTIFACT_STORE_VERSION
        ==
        "preparation_artifact_store_v0.4"
    )


    print(
        "[PASS] Preparation Artifact Store v0.4"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print()

    print(
        "=== DATALENS PREPARATION ARTIFACT STORE SQLITE v0.1 ==="
    )

    print()


    test_new_store_uses_sqlite()

    test_fresh_store_restores_sqlite_metadata()

    test_replace_semantics()

    test_legacy_manifest_import_once()

    test_workflow_delete()

    test_version()


    print()

    print(
        "PASS - Preparation Artifact Store SQLite v0.1"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sqlite3
import tempfile

from pathlib import (
    Path,
)


_TEMP_DIRECTORY = (
    tempfile.TemporaryDirectory(
        prefix=
            "datalens-artifact-index-sqlite-"
    )
)

_ROOT = Path(
    _TEMP_DIRECTORY.name
)

_DATABASE_PATH = (
    _ROOT
    /
    "datalens.sqlite3"
)

_STORE_A = (
    _ROOT
    /
    "artifacts-a"
)

_STORE_B = (
    _ROOT
    /
    "artifacts-b"
)


os.environ[
    "DATALENS_SQLITE_PATH"
] = str(
    _DATABASE_PATH
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_schema_version,
)

from app.preparation.preparation_artifact_index import (
    PREPARATION_ARTIFACT_INDEX_VERSION,
    import_legacy_preparation_artifact_manifest_if_needed,
    load_preparation_artifact_index,
    preparation_artifact_index_is_initialized,
    replace_preparation_artifact_index,
)


MANIFEST_VERSION = (
    "preparation_artifact_manifest_v0.2"
)


def legacy_manifest(
    *,
    workflow_id: str,
    suffix: str,
) -> dict:
    return {
        "manifest_version":
            MANIFEST_VERSION,

        "workflows": {
            workflow_id: {
                "dataset:orders": {
                    "workflow_id":
                        workflow_id,

                    "dataset_id":
                        "dataset:orders",

                    "dataset_filename":
                        f"orders-{suffix}.csv",

                    "stage":
                        "source",

                    "rows":
                        3,

                    "columns":
                        3,

                    "parent_dataset_ids":
                        [],

                    "evidence_refs": [
                        "test:source",
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
                        (
                            "data/"
                            f"artifact-{suffix}.json.gz"
                        ),
                },
            },
        },
    }


def test_schema_v3() -> None:
    assert (
        sqlite_schema_version()
        ==
        SQLITE_SCHEMA_VERSION
        ==
        5
    )


    connection = sqlite3.connect(
        str(
            _DATABASE_PATH
        )
    )

    try:
        tables = {
            row[
                0
            ]

            for row
            in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

    finally:
        connection.close()


    assert (
        "preparation_artifacts"
        in tables
    )

    assert (
        "preparation_artifact_store_state"
        in tables
    )


    print(
        "[PASS] SQLite schema v3 artifact tables"
    )


def test_legacy_manifest_import() -> None:
    _STORE_A.mkdir(
        parents=True,
        exist_ok=True,
    )


    manifest_path = (
        _STORE_A
        /
        "manifest.json"
    )


    manifest_path.write_text(
        json.dumps(
            legacy_manifest(
                workflow_id=
                    "prep:legacy",

                suffix=
                    "legacy",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    imported = (
        import_legacy_preparation_artifact_manifest_if_needed(
            root=
                _STORE_A,

            manifest_path=
                manifest_path,

            fallback_manifest_version=
                MANIFEST_VERSION,
        )
    )


    assert imported is True


    restored = (
        load_preparation_artifact_index(
            root=
                _STORE_A,

            manifest_version=
                MANIFEST_VERSION,
        )
    )


    entry = (
        restored[
            "workflows"
        ][
            "prep:legacy"
        ][
            "dataset:orders"
        ]
    )


    assert (
        entry[
            "dataset_filename"
        ]
        ==
        "orders-legacy.csv"
    )

    assert (
        entry[
            "parent_dataset_ids"
        ]
        ==
        []
    )

    assert (
        entry[
            "evidence_refs"
        ]
        ==
        [
            "test:source",
        ]
    )

    assert (
        entry[
            "datetime_dtypes"
        ][
            0
        ][
            "dtype"
        ]
        ==
        "datetime64[us]"
    )


    print(
        "[PASS] legacy manifest -> SQLite metadata"
    )


def test_manifest_is_imported_once() -> None:
    manifest_path = (
        _STORE_A
        /
        "manifest.json"
    )


    manifest_path.write_text(
        json.dumps(
            legacy_manifest(
                workflow_id=
                    "prep:CHANGED",

                suffix=
                    "changed",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    imported = (
        import_legacy_preparation_artifact_manifest_if_needed(
            root=
                _STORE_A,

            manifest_path=
                manifest_path,

            fallback_manifest_version=
                MANIFEST_VERSION,
        )
    )


    assert imported is False


    restored = (
        load_preparation_artifact_index(
            root=
                _STORE_A,

            manifest_version=
                MANIFEST_VERSION,
        )
    )


    assert (
        "prep:legacy"
        in
        restored[
            "workflows"
        ]
    )

    assert (
        "prep:CHANGED"
        not in
        restored[
            "workflows"
        ]
    )


    print(
        "[PASS] legacy manifest cannot re-import"
    )


def test_store_root_isolation() -> None:
    replace_preparation_artifact_index(
        root=
            _STORE_B,

        manifest=
            legacy_manifest(
                workflow_id=
                    "prep:legacy",

                suffix=
                    "store-b",
            ),
    )


    a = load_preparation_artifact_index(
        root=
            _STORE_A,

        manifest_version=
            MANIFEST_VERSION,
    )


    b = load_preparation_artifact_index(
        root=
            _STORE_B,

        manifest_version=
            MANIFEST_VERSION,
    )


    assert (
        a[
            "workflows"
        ][
            "prep:legacy"
        ][
            "dataset:orders"
        ][
            "dataset_filename"
        ]
        ==
        "orders-legacy.csv"
    )


    assert (
        b[
            "workflows"
        ][
            "prep:legacy"
        ][
            "dataset:orders"
        ][
            "dataset_filename"
        ]
        ==
        "orders-store-b.csv"
    )


    print(
        "[PASS] artifact store_root isolation"
    )


def test_exact_scope_replacement() -> None:
    replacement = {
        "manifest_version":
            MANIFEST_VERSION,

        "workflows": {
            "prep:new": {
                "dataset:new": {
                    "workflow_id":
                        "prep:new",

                    "dataset_id":
                        "dataset:new",

                    "dataset_filename":
                        "new.csv",

                    "stage":
                        "transform",

                    "rows":
                        10,

                    "columns":
                        4,

                    "parent_dataset_ids": [
                        "dataset:orders",
                    ],

                    "evidence_refs": [
                        "test:transform",
                    ],

                    "datetime_dtypes":
                        [],

                    "data_path":
                        "data/new.json.gz",
                },
            },
        },
    }


    replace_preparation_artifact_index(
        root=
            _STORE_A,

        manifest=
            replacement,
    )


    restored = (
        load_preparation_artifact_index(
            root=
                _STORE_A,

            manifest_version=
                MANIFEST_VERSION,
        )
    )


    assert set(
        restored[
            "workflows"
        ]
    ) == {
        "prep:new",
    }


    entry = (
        restored[
            "workflows"
        ][
            "prep:new"
        ][
            "dataset:new"
        ]
    )


    assert (
        entry[
            "parent_dataset_ids"
        ]
        ==
        [
            "dataset:orders",
        ]
    )


    print(
        "[PASS] transactional scope replacement"
    )


def test_version() -> None:
    assert (
        PREPARATION_ARTIFACT_INDEX_VERSION
        ==
        "preparation_artifact_sqlite_index_v0.1"
    )


    assert (
        preparation_artifact_index_is_initialized(
            root=
                _STORE_A
        )
        is True
    )


    print(
        "[PASS] artifact SQLite index version"
    )


def main() -> None:
    print()
    print(
        "=== DATALENS PREPARATION ARTIFACT SQLITE INDEX v0.1 ==="
    )
    print()


    test_schema_v3()

    test_legacy_manifest_import()

    test_manifest_is_imported_once()

    test_store_root_isolation()

    test_exact_scope_replacement()

    test_version()


    print()
    print(
        "PASS - Preparation Artifact SQLite Index v0.1"
    )


if __name__ == "__main__":
    main()

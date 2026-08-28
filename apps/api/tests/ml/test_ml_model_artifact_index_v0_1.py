from __future__ import annotations


import os
import tempfile


from contextlib import (
    contextmanager,
)


from pathlib import (
    Path,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_artifact_index import (
    MLModelArtifactIndexError,
    ML_MODEL_ARTIFACT_INDEX_VERSION,
    delete_ml_model_artifact_index_entry,
    delete_ml_model_artifact_index_workflow,
    get_ml_model_artifact_index_entry,
    load_ml_model_artifact_index_scope,
    load_ml_model_artifact_index_workflow,
    upsert_ml_model_artifact_index_entry,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_connection,
    sqlite_schema_version,
)


# ============================================================
# ENVIRONMENT
# ============================================================


@contextmanager
def isolated_sqlite_database(
):
    previous = os.environ.get(
        "DATALENS_SQLITE_PATH"
    )


    with tempfile.TemporaryDirectory() as root:
        database_path = (
            Path(
                root
            )
            /
            "datalens.sqlite3"
        )


        os.environ[
            "DATALENS_SQLITE_PATH"
        ] = str(
            database_path
        )


        try:
            yield (
                database_path
            )

        finally:
            if previous is None:
                os.environ.pop(
                    "DATALENS_SQLITE_PATH",
                    None,
                )

            else:
                os.environ[
                    "DATALENS_SQLITE_PATH"
                ] = previous


# ============================================================
# FIXTURES
# ============================================================


def artifact_entry(
    *,
    model_id: str,
    workflow_id: str = "prep:ml-index",
    dataset_id: str = "dataset:validated",
    created_at_utc: str = (
        "2026-08-28T18:00:00+00:00"
    ),
    mae: float = 12.5,
    model_path: str = (
        "data/model.joblib"
    ),
    model_sha256: str = (
        "a"
        *
        64
    ),
) -> dict:

    contract = (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
                "tenure",
            ],

            estimator_key=
                "linear_regression",
        )
    )


    record = (
        MLModelArtifactRecord(
            model_id=
                model_id,

            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            training_contract=
                contract,

            metrics={
                "mae":
                    mae,

                "r2":
                    0.82,
            },

            train_rows=
                80,

            test_rows=
                20,

            created_at_utc=
                created_at_utc,

            model_path=
                model_path,

            model_file_bytes=
                128,

            model_sha256=
                model_sha256,
        )
    )


    return (
        record.model_dump(
            mode="json"
        )
    )


# ============================================================
# HELPERS
# ============================================================


def expect_index_error(
    callback,
) -> None:
    try:
        callback()

    except MLModelArtifactIndexError:
        return


    raise AssertionError(
        (
            "Expected "
            "MLModelArtifactIndexError."
        )
    )


# ============================================================
# SCHEMA V8
# ============================================================


def test_schema_v8_model_artifact_table_remains_present_under_v9(
) -> None:
    with isolated_sqlite_database():

        assert (
            SQLITE_SCHEMA_VERSION
            ==
            9
        )


        assert (
            sqlite_schema_version()
            ==
            9
        )


        with sqlite_connection(
            write=False
        ) as connection:

            migration = (
                connection.execute(
                    """
                    SELECT
                        version,
                        name

                    FROM schema_migrations

                    WHERE
                        version = 8
                    """
                )
                .fetchone()
            )


            assert (
                migration
                is not None
            )


            assert (
                int(
                    migration[
                        "version"
                    ]
                )
                ==
                8
            )


            assert (
                str(
                    migration[
                        "name"
                    ]
                )
                ==
                "ml_model_artifact_metadata_index"
            )


            table = (
                connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master

                    WHERE
                        type = 'table'
                        AND
                        name = 'ml_model_artifacts'
                    """
                )
                .fetchone()
            )


            assert (
                table
                is not None
            )


# ============================================================
# UPSERT / GET
# ============================================================


def test_upsert_and_get_model_artifact(
) -> None:
    with isolated_sqlite_database() as root:

        store_path = (
            root.parent
            /
            "model_artifacts.json"
        )


        source = (
            artifact_entry(
                model_id=
                    "model:001"
            )
        )


        upserted = (
            upsert_ml_model_artifact_index_entry(
                store_path=
                    store_path,

                entry=
                    source,
            )
        )


        assert (
            upserted[
                "model_id"
            ]
            ==
            "model:001"
        )


        restored = (
            get_ml_model_artifact_index_entry(
                store_path=
                    store_path,

                model_id=
                    "model:001",
            )
        )


        assert (
            restored
            is not None
        )


        assert (
            restored[
                "workflow_id"
            ]
            ==
            "prep:ml-index"
        )


        assert (
            restored[
                "dataset_id"
            ]
            ==
            "dataset:validated"
        )


        assert (
            restored[
                "training_contract"
            ][
                "target_column"
            ]
            ==
            "revenue"
        )


        assert (
            restored[
                "metrics"
            ][
                "mae"
            ]
            ==
            12.5
        )


# ============================================================
# STORE SCOPE ISOLATION
# ============================================================


def test_store_scope_isolation(
) -> None:
    with isolated_sqlite_database() as root:

        first_store = (
            root.parent
            /
            "first"
            /
            "model_artifacts.json"
        )


        second_store = (
            root.parent
            /
            "second"
            /
            "model_artifacts.json"
        )


        entry = (
            artifact_entry(
                model_id=
                    "model:shared"
            )
        )


        upsert_ml_model_artifact_index_entry(
            store_path=
                first_store,

            entry=
                entry,
        )


        assert (
            get_ml_model_artifact_index_entry(
                store_path=
                    first_store,

                model_id=
                    "model:shared",
            )
            is not None
        )


        assert (
            get_ml_model_artifact_index_entry(
                store_path=
                    second_store,

                model_id=
                    "model:shared",
            )
            is None
        )


# ============================================================
# WORKFLOW ORDERING
# ============================================================


def test_workflow_listing_is_deterministic(
) -> None:
    with isolated_sqlite_database() as root:

        store_path = (
            root.parent
            /
            "model_artifacts.json"
        )


        later = (
            artifact_entry(
                model_id=
                    "model:002",

                created_at_utc=
                    "2026-08-28T19:00:00+00:00",
            )
        )


        earlier = (
            artifact_entry(
                model_id=
                    "model:001",

                created_at_utc=
                    "2026-08-28T18:00:00+00:00",
            )
        )


        upsert_ml_model_artifact_index_entry(
            store_path=
                store_path,

            entry=
                later,
        )


        upsert_ml_model_artifact_index_entry(
            store_path=
                store_path,

            entry=
                earlier,
        )


        entries = (
            load_ml_model_artifact_index_workflow(
                store_path=
                    store_path,

                workflow_id=
                    "prep:ml-index",
            )
        )


        assert (
            [
                entry[
                    "model_id"
                ]
                for entry
                in entries
            ]
            ==
            [
                "model:001",
                "model:002",
            ]
        )


# ============================================================
# UPSERT REPLACEMENT
# ============================================================


def test_upsert_replaces_same_model_identity(
) -> None:
    with isolated_sqlite_database() as root:

        store_path = (
            root.parent
            /
            "model_artifacts.json"
        )


        first = (
            artifact_entry(
                model_id=
                    "model:001",

                mae=
                    12.5,

                model_path=
                    "data/model-first.joblib",

                model_sha256=
                    "a"
                    *
                    64,
            )
        )


        second = (
            artifact_entry(
                model_id=
                    "model:001",

                mae=
                    8.0,

                model_path=
                    "data/model-second.joblib",

                model_sha256=
                    "b"
                    *
                    64,
            )
        )


        upsert_ml_model_artifact_index_entry(
            store_path=
                store_path,

            entry=
                first,
        )


        upsert_ml_model_artifact_index_entry(
            store_path=
                store_path,

            entry=
                second,
        )


        restored = (
            get_ml_model_artifact_index_entry(
                store_path=
                    store_path,

                model_id=
                    "model:001",
            )
        )


        assert (
            restored
            is not None
        )


        assert (
            restored[
                "metrics"
            ][
                "mae"
            ]
            ==
            8.0
        )


        assert (
            restored[
                "model_path"
            ]
            ==
            "data/model-second.joblib"
        )


        scope = (
            load_ml_model_artifact_index_scope(
                store_path=
                    store_path
            )
        )


        assert (
            len(
                scope
            )
            ==
            1
        )


# ============================================================
# DELETE ONE
# ============================================================


def test_delete_one_model_artifact(
) -> None:
    with isolated_sqlite_database() as root:

        store_path = (
            root.parent
            /
            "model_artifacts.json"
        )


        upsert_ml_model_artifact_index_entry(
            store_path=
                store_path,

            entry=
                artifact_entry(
                    model_id=
                        "model:001"
                ),
        )


        deleted = (
            delete_ml_model_artifact_index_entry(
                store_path=
                    store_path,

                model_id=
                    "model:001",
            )
        )


        assert (
            deleted
            is True
        )


        assert (
            get_ml_model_artifact_index_entry(
                store_path=
                    store_path,

                model_id=
                    "model:001",
            )
            is None
        )


        deleted_again = (
            delete_ml_model_artifact_index_entry(
                store_path=
                    store_path,

                model_id=
                    "model:001",
            )
        )


        assert (
            deleted_again
            is False
        )


# ============================================================
# DELETE WORKFLOW
# ============================================================


def test_delete_workflow_scope(
) -> None:
    with isolated_sqlite_database() as root:

        store_path = (
            root.parent
            /
            "model_artifacts.json"
        )


        upsert_ml_model_artifact_index_entry(
            store_path=
                store_path,

            entry=
                artifact_entry(
                    model_id=
                        "model:001",

                    workflow_id=
                        "prep:first",
                ),
        )


        upsert_ml_model_artifact_index_entry(
            store_path=
                store_path,

            entry=
                artifact_entry(
                    model_id=
                        "model:002",

                    workflow_id=
                        "prep:first",
                ),
        )


        upsert_ml_model_artifact_index_entry(
            store_path=
                store_path,

            entry=
                artifact_entry(
                    model_id=
                        "model:003",

                    workflow_id=
                        "prep:second",
                ),
        )


        deleted = (
            delete_ml_model_artifact_index_workflow(
                store_path=
                    store_path,

                workflow_id=
                    "prep:first",
            )
        )


        assert (
            deleted
            ==
            2
        )


        assert (
            load_ml_model_artifact_index_workflow(
                store_path=
                    store_path,

                workflow_id=
                    "prep:first",
            )
            ==
            []
        )


        remaining = (
            load_ml_model_artifact_index_workflow(
                store_path=
                    store_path,

                workflow_id=
                    "prep:second",
            )
        )


        assert (
            len(
                remaining
            )
            ==
            1
        )


        assert (
            remaining[
                0
            ][
                "model_id"
            ]
            ==
            "model:003"
        )


# ============================================================
# INVALID ENTRY
# ============================================================


def test_invalid_metadata_is_blocked_before_sqlite(
) -> None:
    with isolated_sqlite_database() as root:

        store_path = (
            root.parent
            /
            "model_artifacts.json"
        )


        invalid = (
            artifact_entry(
                model_id=
                    "model:001"
            )
        )


        invalid[
            "model_sha256"
        ] = (
            "not-a-sha256"
        )


        def write_invalid(
        ) -> None:
            upsert_ml_model_artifact_index_entry(
                store_path=
                    store_path,

                entry=
                    invalid,
            )


        expect_index_error(
            write_invalid
        )


        assert (
            load_ml_model_artifact_index_scope(
                store_path=
                    store_path
            )
            ==
            []
        )


# ============================================================
# VERSION
# ============================================================


def test_index_rule_version(
) -> None:
    assert (
        ML_MODEL_ARTIFACT_INDEX_VERSION
        ==
        "ml_model_artifact_sqlite_index_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        (
            "=== DATALENS ML MODEL ARTIFACT "
            "SQLITE INDEX v0.1 ==="
        )
    )

    print()


    test_schema_v8_model_artifact_table_remains_present_under_v9()

    print(
        "SQLite schema v8 Model Artifact migration preserved under v9: PASS"
    )


    test_upsert_and_get_model_artifact()

    print(
        "Model Artifact upsert / restore: PASS"
    )


    test_store_scope_isolation()

    print(
        "Model Artifact store scope isolation: PASS"
    )


    test_workflow_listing_is_deterministic()

    print(
        "Workflow Model Artifact ordering: PASS"
    )


    test_upsert_replaces_same_model_identity()

    print(
        "Stable Model Artifact identity upsert: PASS"
    )


    test_delete_one_model_artifact()

    print(
        "Single Model Artifact index deletion: PASS"
    )


    test_delete_workflow_scope()

    print(
        "Workflow Model Artifact index deletion: PASS"
    )


    test_invalid_metadata_is_blocked_before_sqlite()

    print(
        "Invalid Model Artifact metadata is blocked: PASS"
    )


    test_index_rule_version()

    print(
        "Model Artifact SQLite index rule version: PASS"
    )


    print()

    print(
        "ML Model Artifact SQLite Index v0.1: PASS"
    )


if __name__ == "__main__":
    main()
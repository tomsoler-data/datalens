from __future__ import annotations


import os
import sqlite3
import tempfile


from pathlib import Path


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    build_ml_model_experiment_provenance,
)


from app.ml.model_artifact_index import (
    get_ml_model_artifact_index_entry,
    upsert_ml_model_artifact_index_entry,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.persistence.sqlite_database import (
    DATALENS_SQLITE_PATH_ENV,
    SQLITE_SCHEMA_VERSION,
    sqlite_connection,
)


def use_database(
    path: Path,
) -> None:

    os.environ[
        DATALENS_SQLITE_PATH_ENV
    ] = str(
        path
    )


def build_artifact(
    contract,
    *,
    model_character: str,
    metrics: dict[str, float],
) -> MLModelArtifactRecord:

    model_id = (
        "model:"
        +
        model_character
        *
        32
    )


    provenance = (
        build_ml_model_experiment_provenance(
            training_contract=
                contract,
            preparation_session_revision=
                19,
            model_id=
                model_id,
            train_rows=
                80,
            test_rows=
                20,
            metrics=
                metrics,
        )
    )


    return (
        MLModelArtifactRecord(
            model_id=
                model_id,
            workflow_id=
                contract.workflow_id,
            dataset_id=
                contract.dataset_id,
            training_contract=
                contract,
            experiment_provenance=
                provenance,
            metrics=
                metrics,
            train_rows=
                80,
            test_rows=
                20,
            created_at_utc=
                "2026-09-07T00:00:00+00:00",
            serialization_format=
                "pytorch_bundle",
            model_path=
                (
                    "models/"
                    +
                    model_id
                    +
                    ".ptbundle"
                ),
            model_file_bytes=
                1234,
            model_sha256=
                model_character
                *
                64,
        )
    )


def create_v14_fixture(
    path: Path,
) -> None:

    connection = sqlite3.connect(
        str(
            path
        )
    )


    try:

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )


        connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )


        connection.execute(
            """
            INSERT INTO schema_migrations (
                version,
                name,
                applied_at
            )
            VALUES (
                14,
                'ml_model_artifact_multi_format',
                '2026-09-06T00:00:00+00:00'
            )
            """
        )


        connection.execute(
            """
            CREATE TABLE ml_model_artifacts (
                store_root TEXT NOT NULL,
                model_id TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,

                problem_type TEXT NOT NULL
                    CHECK (
                        problem_type
                        IN (
                            'regression',
                            'classification'
                        )
                    ),

                target_column TEXT NOT NULL,
                estimator_key TEXT NOT NULL,
                training_contract_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,

                train_rows INTEGER NOT NULL
                    CHECK (train_rows > 0),

                test_rows INTEGER NOT NULL
                    CHECK (test_rows > 0),

                created_at_utc TEXT NOT NULL,

                serialization_format TEXT NOT NULL
                    CHECK (
                        serialization_format
                        IN (
                            'joblib',
                            'pytorch_bundle'
                        )
                    ),

                rule_version TEXT NOT NULL,
                model_path TEXT NOT NULL,

                model_file_bytes INTEGER NOT NULL
                    CHECK (model_file_bytes > 0),

                model_sha256 TEXT NOT NULL
                    CHECK (length(model_sha256) = 64),

                experiment_id TEXT,
                experiment_provenance_json TEXT,

                PRIMARY KEY (
                    store_root,
                    model_id
                )
            )
            """
        )


        connection.execute(
            """
            CREATE TABLE legacy_model_child (
                store_root TEXT NOT NULL,
                model_id TEXT NOT NULL,
                payload TEXT NOT NULL,

                PRIMARY KEY (
                    store_root,
                    model_id
                ),

                FOREIGN KEY (
                    store_root,
                    model_id
                )
                REFERENCES ml_model_artifacts (
                    store_root,
                    model_id
                )
                ON DELETE CASCADE
            )
            """
        )


        model_id = (
            "model:"
            +
            "a"
            *
            32
        )


        connection.execute(
            """
            INSERT INTO ml_model_artifacts (
                store_root,
                model_id,
                workflow_id,
                dataset_id,
                problem_type,
                target_column,
                estimator_key,
                training_contract_json,
                metrics_json,
                train_rows,
                test_rows,
                created_at_utc,
                serialization_format,
                rule_version,
                model_path,
                model_file_bytes,
                model_sha256,
                experiment_id,
                experiment_provenance_json
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                "/tmp/store-v14",
                model_id,
                "workflow:v14-legacy",
                "dataset:v14-legacy",
                "regression",
                "target",
                "tabular_mlp_regressor",
                "{}",
                '{"rmse":0.25}',
                80,
                20,
                "2026-09-06T00:00:00+00:00",
                "pytorch_bundle",
                "ml_model_artifact_v0.1",
                "models/" + model_id + ".ptbundle",
                1234,
                "a" * 64,
                None,
                None,
            ),
        )


        connection.execute(
            """
            INSERT INTO legacy_model_child (
                store_root,
                model_id,
                payload
            )
            VALUES (?, ?, ?)
            """,
            (
                "/tmp/store-v14",
                model_id,
                "preserve-me",
            ),
        )


        connection.commit()


    finally:

        connection.close()


print(
    "=== DATALENS SQLITE v15 MODEL ARTIFACT ANOMALY v0.1 ==="
)

print()


assert SQLITE_SCHEMA_VERSION == 15

print(
    "SQLite schema version 15 authority: PASS"
)


# ============================================================
# REAL v14 -> v15
# ============================================================


with tempfile.TemporaryDirectory() as directory:

    database_path = (
        Path(directory)
        /
        "legacy-v14.sqlite3"
    )


    create_v14_fixture(
        database_path
    )


    use_database(
        database_path
    )


    with sqlite_connection(
        write=False
    ) as connection:

        version = int(
            connection.execute(
                "SELECT MAX(version) FROM schema_migrations"
            )
            .fetchone()[0]
        )


        assert version == 15


        row = (
            connection.execute(
                """
                SELECT
                    problem_type,
                    target_column
                FROM ml_model_artifacts
                """
            )
            .fetchone()
        )


        assert row["problem_type"] == "regression"
        assert row["target_column"] == "target"


        child = (
            connection.execute(
                """
                SELECT payload
                FROM legacy_model_child
                """
            )
            .fetchone()
        )


        assert child["payload"] == "preserve-me"


        assert (
            connection.execute(
                "PRAGMA foreign_key_check"
            )
            .fetchall()
            ==
            []
        )


print(
    "Real SQLite v14 -> v15 migration: PASS"
)

print(
    "Historical Model Artifact row: PASS"
)

print(
    "Child foreign-key preservation: PASS"
)


# ============================================================
# FRESH v15
# ============================================================


with tempfile.TemporaryDirectory() as directory:

    database_path = (
        Path(directory)
        /
        "fresh-v15.sqlite3"
    )


    use_database(
        database_path
    )


    store_path = (
        Path(directory)
        /
        "artifact-store"
    )


    supervised = (
        MLTrainingContract(
            workflow_id=
                "workflow:v15-supervised",
            dataset_id=
                "dataset:v15-supervised",
            problem_type=
                "regression",
            target_column=
                "target",
            feature_columns=[
                "x1",
            ],
            estimator_key=
                "tabular_mlp_regressor",
        )
    )


    anomaly = (
        MLAnomalyTrainingContract(
            workflow_id=
                "workflow:v15-anomaly",
            dataset_id=
                "dataset:v15-anomaly",
            feature_columns=[
                "x1",
                "x2",
            ],
            threshold_quantile=
                0.975,
        )
    )


    supervised_artifact = (
        build_artifact(
            supervised,
            model_character=
                "b",
            metrics={
                "rmse":
                    0.20,
            },
        )
    )


    anomaly_artifact = (
        build_artifact(
            anomaly,
            model_character=
                "c",
            metrics={
                "train_reconstruction_mse":
                    0.10,
                "test_reconstruction_mse":
                    0.24,
                "anomaly_threshold":
                    0.42,
                "test_anomaly_rate":
                    0.05,
            },
        )
    )


    upsert_ml_model_artifact_index_entry(
        store_path=
            store_path,
        entry=
            supervised_artifact.model_dump(
                mode="json"
            ),
    )


    upsert_ml_model_artifact_index_entry(
        store_path=
            store_path,
        entry=
            anomaly_artifact.model_dump(
                mode="json"
            ),
    )


    restored_supervised = (
        get_ml_model_artifact_index_entry(
            store_path=
                store_path,
            model_id=
                supervised_artifact.model_id,
        )
    )


    restored_anomaly = (
        get_ml_model_artifact_index_entry(
            store_path=
                store_path,
            model_id=
                anomaly_artifact.model_id,
        )
    )


    assert restored_supervised is not None
    assert restored_anomaly is not None

    assert (
        restored_supervised["target_column"]
        ==
        "target"
    )

    assert (
        restored_anomaly["problem_type"]
        ==
        "anomaly_detection"
    )

    assert (
        restored_anomaly["target_column"]
        is None
    )

    assert (
        restored_anomaly["estimator_key"]
        ==
        "tabular_autoencoder"
    )


    print(
        "Real supervised SQLite persistence: PASS"
    )

    print(
        "Real anomaly SQLite persistence: PASS"
    )


    with sqlite_connection(
        write=False
    ) as connection:

        table_info = (
            connection.execute(
                "PRAGMA table_info(ml_model_artifacts)"
            )
            .fetchall()
        )


        target_info = next(
            row
            for row in table_info
            if row["name"] == "target_column"
        )


        assert int(target_info["notnull"]) == 0


        assert (
            connection.execute(
                "PRAGMA foreign_key_check"
            )
            .fetchall()
            ==
            []
        )


    print(
        "Nullable target_column schema: PASS"
    )


    raw_connection = sqlite3.connect(
        str(
            database_path
        ),
        isolation_level=None,
    )


    try:

        try:

            raw_connection.execute(
                """
                UPDATE ml_model_artifacts
                SET target_column = 'invented-target'
                WHERE model_id = ?
                """,
                (
                    anomaly_artifact.model_id,
                ),
            )

        except sqlite3.IntegrityError:
            pass

        else:

            raise AssertionError(
                "SQLite accepted anomaly target."
            )


        try:

            raw_connection.execute(
                """
                UPDATE ml_model_artifacts
                SET target_column = NULL
                WHERE model_id = ?
                """,
                (
                    supervised_artifact.model_id,
                ),
            )

        except sqlite3.IntegrityError:
            pass

        else:

            raise AssertionError(
                "SQLite accepted NULL supervised target."
            )


    finally:

        raw_connection.close()


    print(
        "Problem/target relational CHECK: PASS"
    )


print(
    "SQLite foreign-key integrity: PASS"
)

print()

print(
    "PASS - DataLens SQLite v15 Model Artifact Anomaly v0.1"
)

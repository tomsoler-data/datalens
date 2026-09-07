from __future__ import annotations


import os
import sqlite3
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


from app.ml.model_artifact_data_plane import (
    ml_model_artifact_data_root,
)


from app.ml.model_artifact_store import (
    MLModelArtifactStoreError,
    get_ml_model_artifact,
    load_ml_model_artifact_binary,
    register_ml_model_artifact,
)


from app.ml.model_loader import (
    MLModelLoaderArtifactError,
    load_trusted_ml_model,
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
def isolated_environment(
):

    previous_sqlite = os.environ.get(
        "DATALENS_SQLITE_PATH"
    )


    previous_store = os.environ.get(
        "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
    )


    with tempfile.TemporaryDirectory(
        prefix=
            "datalens-ml-multiformat-"
    ) as root:

        root_path = Path(
            root
        )


        database = (
            root_path
            /
            "datalens.sqlite3"
        )


        store = (
            root_path
            /
            "ml"
            /
            "model_artifacts.json"
        )


        os.environ[
            "DATALENS_SQLITE_PATH"
        ] = str(
            database
        )


        os.environ[
            "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
        ] = str(
            store
        )


        try:

            yield (
                root_path,
                database,
                store,
            )


        finally:

            if previous_sqlite is None:

                os.environ.pop(
                    "DATALENS_SQLITE_PATH",
                    None,
                )

            else:

                os.environ[
                    "DATALENS_SQLITE_PATH"
                ] = previous_sqlite


            if previous_store is None:

                os.environ.pop(
                    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH",
                    None,
                )

            else:

                os.environ[
                    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
                ] = previous_store


# ============================================================
# LEGACY V13 FIXTURE
# ============================================================


def build_legacy_v13_database(
    database: Path,
) -> None:

    connection = sqlite3.connect(
        str(
            database
        ),
        isolation_level=None,
    )


    connection.row_factory = (
        sqlite3.Row
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
                13,
                'ml_performance_evaluation_metadata',
                '2026-09-06T12:00:00+00:00'
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
                        problem_type IN (
                            'regression',
                            'classification'
                        )
                    ),

                target_column TEXT NOT NULL,
                estimator_key TEXT NOT NULL,

                training_contract_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,

                train_rows INTEGER NOT NULL
                    CHECK (
                        train_rows > 0
                    ),

                test_rows INTEGER NOT NULL
                    CHECK (
                        test_rows > 0
                    ),

                created_at_utc TEXT NOT NULL,

                serialization_format TEXT NOT NULL
                    CHECK (
                        serialization_format
                        =
                        'joblib'
                    ),

                rule_version TEXT NOT NULL,
                model_path TEXT NOT NULL,

                model_file_bytes INTEGER NOT NULL
                    CHECK (
                        model_file_bytes > 0
                    ),

                model_sha256 TEXT NOT NULL
                    CHECK (
                        length(
                            model_sha256
                        )
                        =
                        64
                    ),

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
            CREATE INDEX
            idx_ml_model_artifacts_scope_workflow

            ON ml_model_artifacts (
                store_root,
                workflow_id,
                created_at_utc,
                model_id
            )
            """
        )


        connection.execute(
            """
            CREATE INDEX
            idx_ml_model_artifacts_scope_dataset

            ON ml_model_artifacts (
                store_root,
                workflow_id,
                dataset_id,
                created_at_utc,
                model_id
            )
            """
        )


        connection.execute(
            """
            CREATE UNIQUE INDEX
            idx_ml_model_artifacts_scope_experiment

            ON ml_model_artifacts (
                store_root,
                experiment_id
            )

            WHERE
                experiment_id
                IS NOT NULL
            """
        )


        connection.execute(
            """
            CREATE TABLE ml_monitoring_profiles (
                store_root TEXT NOT NULL,
                model_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,

                PRIMARY KEY (
                    store_root,
                    profile_id
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


        connection.execute(
            """
            CREATE TABLE ml_performance_evaluations (
                store_root TEXT NOT NULL,
                model_id TEXT NOT NULL,
                evaluation_id TEXT NOT NULL,

                PRIMARY KEY (
                    store_root,
                    evaluation_id
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
                'legacy-store',
                'model:legacy',
                'prep:legacy',
                'dataset:validated',
                'regression',
                'target',
                'linear_regression',
                '{}',
                '{"rmse":1.0}',
                80,
                20,
                '2026-09-06T12:00:00+00:00',
                'joblib',
                'ml_model_artifact_v0.1',
                'data/legacy.joblib',
                100,
                ?,
                'experiment:legacy',
                '{}'
            )
            """,
            (
                "a"
                *
                64,
            ),
        )


        connection.execute(
            """
            INSERT INTO ml_monitoring_profiles (
                store_root,
                model_id,
                profile_id
            )
            VALUES (
                'legacy-store',
                'model:legacy',
                'profile:legacy'
            )
            """
        )


        connection.execute(
            """
            INSERT INTO ml_performance_evaluations (
                store_root,
                model_id,
                evaluation_id
            )
            VALUES (
                'legacy-store',
                'model:legacy',
                'performance:legacy'
            )
            """
        )


    finally:

        connection.close()


# ============================================================
# V13 -> CURRENT / V14 MULTI-FORMAT PRESERVATION
# ============================================================


def test_real_v13_to_current_migration_preserves_v14_multiformat(
) -> None:

    with isolated_environment() as (
        _,
        database,
        _,
    ):

        build_legacy_v13_database(
            database
        )


        assert (
            sqlite_schema_version()
            ==
            15
        )


        with sqlite_connection(
            write=False
        ) as connection:

            row = (
                connection.execute(
                    """
                    SELECT
                        serialization_format,
                        model_path

                    FROM ml_model_artifacts

                    WHERE
                        model_id =
                        'model:legacy'
                    """
                )
                .fetchone()
            )


            assert (
                row is not None
            )


            assert (
                row[
                    "serialization_format"
                ]
                ==
                "joblib"
            )


            assert (
                row[
                    "model_path"
                ]
                ==
                "data/legacy.joblib"
            )


            assert (
                connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()
                ==
                []
            )


            table_sql = str(
                connection.execute(
                    """
                    SELECT sql
                    FROM sqlite_master

                    WHERE
                        type = 'table'
                        AND
                        name = 'ml_model_artifacts'
                    """
                )
                .fetchone()[
                    "sql"
                ]
            )


            assert (
                "'joblib'"
                in
                table_sql
            )


            assert (
                "'pytorch_bundle'"
                in
                table_sql
            )


            monitoring_fk = (
                connection.execute(
                    """
                    PRAGMA foreign_key_list(
                        ml_monitoring_profiles
                    )
                    """
                ).fetchall()
            )


            performance_fk = (
                connection.execute(
                    """
                    PRAGMA foreign_key_list(
                        ml_performance_evaluations
                    )
                    """
                ).fetchall()
            )


            assert {
                str(
                    item[
                        "table"
                    ]
                )

                for item
                in monitoring_fk
            } == {
                "ml_model_artifacts"
            }


            assert {
                str(
                    item[
                        "table"
                    ]
                )

                for item
                in performance_fk
            } == {
                "ml_model_artifacts"
            }


        with sqlite_connection(
            write=True
        ) as connection:

            connection.execute(
                """
                DELETE FROM ml_model_artifacts
                WHERE model_id = 'model:legacy'
                """
            )


        with sqlite_connection(
            write=False
        ) as connection:

            assert (
                int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM ml_monitoring_profiles
                        """
                    ).fetchone()[0]
                )
                ==
                0
            )


            assert (
                int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM ml_performance_evaluations
                        """
                    ).fetchone()[0]
                )
                ==
                0
            )


# ============================================================
# PREPARATION AUTHORITY
# ============================================================


def seed_preparation_authority(
    *,
    workflow_id: str,
    dataset_id: str,
) -> None:

    with sqlite_connection(
        write=True
    ) as connection:

        connection.execute(
            """
            INSERT INTO preparation_sessions (
                workflow_id,
                revision,
                payload_json,
                created_at,
                updated_at
            )
            VALUES (
                ?, ?, ?, ?, ?
            )
            """,
            (
                workflow_id,
                0,
                "{}",
                "2026-09-06T12:00:00+00:00",
                "2026-09-06T12:00:00+00:00",
            ),
        )


        connection.execute(
            """
            INSERT INTO preparation_artifacts (
                store_root,
                workflow_id,
                dataset_id,
                dataset_filename,
                stage,
                rows,
                columns,
                parent_dataset_ids_json,
                evidence_refs_json,
                datetime_dtypes_json,
                data_path
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                "test-preparation-root",
                workflow_id,
                dataset_id,
                "validated.csv",
                "source",
                100,
                2,
                "[]",
                "[]",
                "[]",
                "data/validated.json.gz",
            ),
        )


# ============================================================
# CONTRACTS
# ============================================================


def classical_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:multiformat",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "feature",
            ],

            estimator_key=
                "linear_regression",
        )
    )


def mlp_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:multiformat",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "feature",
            ],

            estimator_key=
                "tabular_mlp_regressor",
        )
    )


# ============================================================
# STORE MULTI-FORMAT REGISTRATION
# ============================================================


def test_store_registers_both_formats(
) -> None:

    with isolated_environment() as (
        _,
        _,
        store,
    ):

        seed_preparation_authority(
            workflow_id=
                "prep:multiformat",

            dataset_id=
                "dataset:validated",
        )


        classical = (
            register_ml_model_artifact(
                training_contract=
                    classical_contract(),

                metrics={
                    "rmse":
                        1.0,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    b"classical-joblib-bytes",

                preparation_session_revision=
                    0,

                created_at_utc=
                    "2026-09-06T12:10:00+00:00",
            )
        )


        assert (
            classical.serialization_format
            ==
            "joblib"
        )


        assert (
            classical.model_path
            .endswith(
                ".joblib"
            )
        )


        mlp = (
            register_ml_model_artifact(
                training_contract=
                    mlp_contract(),

                metrics={
                    "rmse":
                        0.8,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    b"pytorch-bundle-bytes",

                serialization_format=
                    "pytorch_bundle",

                preparation_session_revision=
                    0,

                created_at_utc=
                    "2026-09-06T12:11:00+00:00",
            )
        )


        assert (
            mlp.serialization_format
            ==
            "pytorch_bundle"
        )


        assert (
            mlp.model_path
            .endswith(
                ".ptbundle"
            )
        )


        restored = (
            get_ml_model_artifact(
                model_id=
                    mlp.model_id,

                workflow_id=
                    mlp.workflow_id,
            )
        )


        assert (
            restored
            ==
            mlp
        )


        binary = (
            load_ml_model_artifact_binary(
                model_id=
                    mlp.model_id,

                workflow_id=
                    mlp.workflow_id,
            )
        )


        assert (
            binary
            ==
            b"pytorch-bundle-bytes"
        )


        data_root = (
            ml_model_artifact_data_root(
                store
            )
        )


        assert (
            len(
                list(
                    data_root.rglob(
                        "*.joblib"
                    )
                )
            )
            ==
            1
        )


        assert (
            len(
                list(
                    data_root.rglob(
                        "*.ptbundle"
                    )
                )
            )
            ==
            1
        )


# ============================================================
# CLASSICAL LOADER FAILS CLOSED ON PYTORCH
# ============================================================


def test_classical_loader_refuses_pytorch_bundle(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:multiformat",

            dataset_id=
                "dataset:validated",
        )


        mlp = (
            register_ml_model_artifact(
                training_contract=
                    mlp_contract(),

                metrics={
                    "rmse":
                        0.8,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    b"trusted-pytorch-bundle",

                serialization_format=
                    "pytorch_bundle",

                preparation_session_revision=
                    0,
            )
        )


        try:

            load_trusted_ml_model(
                workflow_id=
                    mlp.workflow_id,

                model_id=
                    mlp.model_id,
            )

        except MLModelLoaderArtifactError as error:

            assert (
                "unsupported serialization format"
                in
                str(
                    error
                )
            )

        else:

            raise AssertionError(
                (
                    "Classical trusted loader must "
                    "refuse pytorch_bundle artifacts."
                )
            )


# ============================================================
# UNKNOWN STORE FORMAT FAILS CLOSED
# ============================================================


def test_unknown_store_format_fails_before_binary_write(
) -> None:

    with isolated_environment() as (
        _,
        _,
        store,
    ):

        seed_preparation_authority(
            workflow_id=
                "prep:multiformat",

            dataset_id=
                "dataset:validated",
        )


        try:

            register_ml_model_artifact(
                training_contract=
                    mlp_contract(),

                metrics={
                    "rmse":
                        0.8,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    b"must-not-be-written",

                serialization_format=
                    "unknown",
            )

        except MLModelArtifactStoreError:
            pass

        else:

            raise AssertionError(
                "Unknown serialization format must fail closed."
            )


        data_root = (
            ml_model_artifact_data_root(
                store
            )
        )


        assert (
            not data_root.exists()
            or
            list(
                data_root.rglob(
                    "*"
                )
            )
            ==
            []
        )


# ============================================================
# SCHEMA VERSION
# ============================================================


def test_schema_version(
) -> None:

    assert (
        SQLITE_SCHEMA_VERSION
        ==
        15
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS SQLITE V14 / MULTI-FORMAT MODEL ARTIFACT STORE v0.1 ==="
    )

    print()


    test_real_v13_to_current_migration_preserves_v14_multiformat()

    print(
        "Real v13 -> current migration preserves v14 multi-format: PASS"
    )


    test_store_registers_both_formats()

    print(
        "Joblib + PyTorch store registration: PASS"
    )


    test_classical_loader_refuses_pytorch_bundle()

    print(
        "Classical loader format isolation: PASS"
    )


    test_unknown_store_format_fails_before_binary_write()

    print(
        "Unknown store format fail-closed guard: PASS"
    )


    test_schema_version()

    print(
        "SQLite current schema version 15: PASS"
    )


    print()

    print(
        "PASS - DataLens SQLite v14 / Multi-Format Model Artifact Store v0.1"
    )


if __name__ == "__main__":
    main()

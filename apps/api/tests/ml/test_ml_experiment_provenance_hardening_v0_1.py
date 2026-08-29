from __future__ import annotations


import json
import os
import sqlite3
import tempfile


from pathlib import (
    Path,
)


from pydantic import (
    ValidationError,
)


import app.ml.model_artifact_store as model_artifact_store_module


from app.ml.classical_executor import (
    ClassicalMLExecutorError,
    execute_classical_ml,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
    build_ml_experiment_provenance,
)


from app.ml.model_artifact_index import (
    get_ml_model_artifact_index_entry,
)


from app.ml.model_artifact_store import (
    list_ml_model_artifacts,
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_connection,
    sqlite_schema_version,
)


from tests.ml.test_classical_ml_executor_v0_1 import (
    isolated_environment,
    patched_handoff,
    regression_contract,
    regression_dataframe,
    seed_preparation_authority,
)


# ============================================================
# STRICT RULE VERSION
# ============================================================


def test_wrong_provenance_rule_version_is_blocked(
) -> None:

    contract = (
        regression_contract()
    )


    valid = (
        build_ml_experiment_provenance(
            training_contract=
                contract,

            preparation_session_revision=
                0,

            model_id=
                "model:test",

            train_rows=
                80,

            test_rows=
                20,

            metrics={
                "rmse":
                    5.0,

                "mae":
                    4.0,

                "r2":
                    0.9,
            },
        )
    )


    payload = (
        valid.model_dump(
            mode="json"
        )
    )


    payload[
        "rule_version"
    ] = (
        "ml_experiment_provenance_v999"
    )


    try:
        MLExperimentProvenanceRecord.model_validate(
            payload
        )

    except ValidationError:
        return


    raise AssertionError(
        (
            "Unknown Experiment Provenance "
            "rule version must be blocked."
        )
    )


# ============================================================
# ATOMIC LATE REVISION RACE
# ============================================================


def test_late_preparation_revision_race_is_atomic_fail_closed(
) -> None:

    original_upsert = (
        model_artifact_store_module
        .upsert_ml_model_artifact_index_entry
    )


    with isolated_environment():

        contract = (
            regression_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        race_triggered = False


        def upsert_after_revision_change(
            *,
            store_path,
            entry,
            expected_preparation_session_revision=None,
        ):

            nonlocal race_triggered


            if not race_triggered:

                race_triggered = True


                # The first authority check in
                # register_ml_model_artifact() has already
                # succeeded at this point.
                #
                # Simulate Preparation changing immediately
                # before the SQLite Model Artifact upsert.

                with sqlite_connection(
                    write=True
                ) as connection:

                    connection.execute(
                        """
                        UPDATE preparation_sessions

                        SET
                            revision =
                                revision + 1

                        WHERE
                            workflow_id = ?
                        """,
                        (
                            contract.workflow_id,
                        ),
                    )


            return (
                original_upsert(
                    store_path=
                        store_path,

                    entry=
                        entry,

                    expected_preparation_session_revision=(
                        expected_preparation_session_revision
                    ),
                )
            )


        model_artifact_store_module.upsert_ml_model_artifact_index_entry = (
            upsert_after_revision_change
        )


        try:

            with patched_handoff(
                dataframe=
                    regression_dataframe(),

                workflow_id=
                    contract.workflow_id,

                dataset_id=
                    contract.dataset_id,
            ):

                try:
                    execute_classical_ml(
                        training_contract=
                            contract
                    )

                except ClassicalMLExecutorError:
                    pass

                else:
                    raise AssertionError(
                        (
                            "Late Preparation revision race "
                            "must fail Classical ML closed."
                        )
                    )


        finally:
            model_artifact_store_module.upsert_ml_model_artifact_index_entry = (
                original_upsert
            )


        assert (
            race_triggered
            is True
        )


        assert (
            list_ml_model_artifacts(
                workflow_id=
                    contract.workflow_id
            )
            ==
            []
        )


# ============================================================
# REAL V8 -> V9 MIGRATION
# ============================================================


def test_real_v8_model_artifact_survives_current_schema_migration(
) -> None:

    previous_sqlite = os.environ.get(
        "DATALENS_SQLITE_PATH"
    )


    with tempfile.TemporaryDirectory(
        prefix=
            "datalens-v8-to-v9-"
    ) as temporary_directory:

        root = Path(
            temporary_directory
        )


        database_path = (
            root
            /
            "legacy.sqlite3"
        )


        store_path = (
            root
            /
            "model_artifacts.json"
        )


        os.environ[
            "DATALENS_SQLITE_PATH"
        ] = str(
            database_path
        )


        contract = (
            regression_contract()
        )


        training_contract_json = (
            json.dumps(
                contract.model_dump(
                    mode="json"
                ),
                ensure_ascii=False,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            )
        )


        metrics_json = (
            json.dumps(
                {
                    "mae":
                        8.0,

                    "rmse":
                        10.0,

                    "r2":
                        0.75,
                },
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            )
        )


        store_root = str(
            store_path
            .expanduser()
            .resolve()
        )


        raw_connection = (
            sqlite3.connect(
                str(
                    database_path
                )
            )
        )


        try:

            raw_connection.execute(
                """
                CREATE TABLE schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )


            raw_connection.execute(
                """
                INSERT INTO schema_migrations (
                    version,
                    name,
                    applied_at
                )
                VALUES (
                    8,
                    'ml_model_artifact_metadata_index',
                    '2026-08-28T18:00:00+00:00'
                )
                """
            )


            raw_connection.execute(
                """
                CREATE TABLE ml_model_artifacts (
                    store_root TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    problem_type TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    estimator_key TEXT NOT NULL,
                    training_contract_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    train_rows INTEGER NOT NULL,
                    test_rows INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    serialization_format TEXT NOT NULL,
                    rule_version TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    model_file_bytes INTEGER NOT NULL,
                    model_sha256 TEXT NOT NULL,

                    PRIMARY KEY (
                        store_root,
                        model_id
                    )
                )
                """
            )


            raw_connection.execute(
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
                    model_sha256
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    store_root,
                    "model:legacy-v8",
                    contract.workflow_id,
                    contract.dataset_id,
                    contract.problem_type,
                    contract.target_column,
                    contract.estimator_key,
                    training_contract_json,
                    metrics_json,
                    80,
                    20,
                    "2026-08-28T19:00:00+00:00",
                    "joblib",
                    "ml_model_artifact_v0.1",
                    "data/legacy-v8.joblib",
                    128,
                    (
                        "a"
                        *
                        64
                    ),
                ),
            )


            raw_connection.commit()


        finally:
            raw_connection.close()


        try:

            # Opening DataLens SQLite upgrades the
            # pre-existing schema-8 database through all
            # historical migrations up to the current schema.
            #
            # This includes the schema-9 Experiment Provenance
            # migration and any later additive migrations.
            assert (
                sqlite_schema_version()
                ==
                SQLITE_SCHEMA_VERSION
            )


            restored = (
                get_ml_model_artifact_index_entry(
                    store_path=
                        store_path,

                    model_id=
                        "model:legacy-v8",
                )
            )


            assert (
                restored
                is not None
            )


            assert (
                restored[
                    "model_id"
                ]
                ==
                "model:legacy-v8"
            )


            assert (
                restored[
                    "experiment_id"
                ]
                is None
            )


            assert (
                restored[
                    "experiment_provenance"
                ]
                is None
            )


            with sqlite_connection(
                write=False
            ) as connection:

                row = (
                    connection.execute(
                        """
                        SELECT
                            experiment_id,
                            experiment_provenance_json

                        FROM ml_model_artifacts

                        WHERE
                            model_id =
                            'model:legacy-v8'
                        """
                    )
                    .fetchone()
                )


                assert (
                    row
                    is not None
                )


                assert (
                    row[
                        "experiment_id"
                    ]
                    is None
                )


                assert (
                    row[
                        "experiment_provenance_json"
                    ]
                    is None
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


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML EXPERIMENT "
            "PROVENANCE HARDENING v0.1 ==="
        )
    )

    print()


    test_wrong_provenance_rule_version_is_blocked()

    print(
        "Unknown provenance rule version is blocked: PASS"
    )


    test_late_preparation_revision_race_is_atomic_fail_closed()

    print(
        "Atomic late Preparation revision race guard: PASS"
    )


    test_real_v8_model_artifact_survives_current_schema_migration()

    print(
        "Real SQLite v8 -> current schema legacy artifact migration: PASS"
    )


    print()

    print(
        "ML Experiment Provenance Hardening v0.1: PASS"
    )


if __name__ == "__main__":
    main()

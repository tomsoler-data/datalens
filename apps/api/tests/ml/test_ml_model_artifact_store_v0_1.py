from __future__ import annotations


import inspect
import os
import tempfile


from contextlib import (
    contextmanager,
)


from pathlib import (
    Path,
)


import app.ml.model_artifact_store as store_module


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_artifact_data_plane import (
    ml_model_artifact_data_root,
)


from app.ml.model_artifact_store import (
    MLModelArtifactAuthorityError,
    MLModelArtifactStoreError,
    MLModelArtifactWorkflowMismatchError,
    ML_MODEL_ARTIFACT_STORE_RULE_VERSION,
    get_ml_model_artifact,
    list_ml_model_artifacts,
    load_ml_model_artifact_binary,
    register_ml_model_artifact,
    resolve_ml_model_artifact_store_path,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


# ============================================================
# FIXTURES
# ============================================================


MODEL_BYTES = (
    b"DATALENS-SERVER-OWNED-MODEL-v0.1"
)


# ============================================================
# ISOLATION
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
            "datalens-ml-model-store-"
    ) as temporary_directory:

        root = Path(
            temporary_directory
        )


        database = (
            root
            /
            "datalens.sqlite3"
        )


        model_store = (
            root
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
            model_store
        )


        try:
            yield (
                root,
                model_store,
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
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                workflow_id,
                0,
                "{}",
                "2026-08-28T18:00:00+00:00",
                "2026-08-28T18:00:00+00:00",
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
                "test-preparation-root",
                workflow_id,
                dataset_id,
                "validated.csv",
                "source",
                100,
                3,
                "[]",
                "[]",
                "[]",
                "data/validated.json.gz",
            ),
        )


# ============================================================
# CONTRACT
# ============================================================


def training_contract(
    *,
    workflow_id: str = "prep:ml-store",
    dataset_id: str = "dataset:validated",
) -> MLTrainingContract:

    return (
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


# ============================================================
# SERVER-OWNED INPUT SURFACE
# ============================================================


def test_registration_does_not_accept_identity_or_path(
) -> None:

    parameters = (
        inspect.signature(
            register_ml_model_artifact
        )
        .parameters
    )


    assert (
        "model_id"
        not in
        parameters
    )


    assert (
        "model_path"
        not in
        parameters
    )


# ============================================================
# SUCCESSFUL REGISTRATION
# ============================================================


def test_register_and_restore_server_owned_model(
) -> None:

    with isolated_environment() as (
        _,
        model_store,
    ):

        seed_preparation_authority(
            workflow_id=
                "prep:ml-store",

            dataset_id=
                "dataset:validated",
        )


        record = (
            register_ml_model_artifact(
                training_contract=
                    training_contract(),

                metrics={
                    "mae":
                        10.5,

                    "r2":
                        0.81,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    MODEL_BYTES,

                created_at_utc=
                    "2026-08-28T19:00:00+00:00",
            )
        )


        assert (
            record.model_id
            .startswith(
                "model:"
            )
        )


        assert (
            record.workflow_id
            ==
            "prep:ml-store"
        )


        assert (
            record.dataset_id
            ==
            "dataset:validated"
        )


        assert (
            record.model_path
            .startswith(
                "data/"
            )
        )


        assert (
            record.model_path
            .endswith(
                ".joblib"
            )
        )


        restored = (
            get_ml_model_artifact(
                model_id=
                    record.model_id,

                workflow_id=
                    "prep:ml-store",
            )
        )


        assert (
            restored
            ==
            record
        )


        binary = (
            load_ml_model_artifact_binary(
                model_id=
                    record.model_id,

                workflow_id=
                    "prep:ml-store",
            )
        )


        assert (
            binary
            ==
            MODEL_BYTES
        )


        path = (
            ml_model_artifact_data_root(
                model_store
            )
            /
            record.model_path
        )


        assert (
            path.is_file()
        )


# ============================================================
# GENERATED IDENTITIES
# ============================================================


def test_model_identities_are_server_generated_and_unique(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-store",

            dataset_id=
                "dataset:validated",
        )


        first = (
            register_ml_model_artifact(
                training_contract=
                    training_contract(),

                metrics={
                    "mae":
                        10.0,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    MODEL_BYTES,
            )
        )


        second = (
            register_ml_model_artifact(
                training_contract=
                    training_contract(),

                metrics={
                    "mae":
                        9.0,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    MODEL_BYTES,
            )
        )


        assert (
            first.model_id
            !=
            second.model_id
        )


        assert (
            first.model_path
            !=
            second.model_path
        )


# ============================================================
# WORKFLOW AUTHORITY
# ============================================================


def test_unknown_workflow_is_blocked_before_model_write(
) -> None:

    with isolated_environment() as (
        _,
        model_store,
    ):

        try:
            register_ml_model_artifact(
                training_contract=
                    training_contract(
                        workflow_id=
                            "prep:invented"
                    ),

                metrics={
                    "mae":
                        10.0,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    MODEL_BYTES,
            )

        except MLModelArtifactAuthorityError:
            pass

        else:
            raise AssertionError(
                (
                    "Invented Preparation workflow "
                    "should have been rejected."
                )
            )


        data_root = (
            ml_model_artifact_data_root(
                model_store
            )
        )


        assert (
            not data_root.exists()
            or
            list(
                data_root.rglob(
                    "*.joblib"
                )
            )
            ==
            []
        )


# ============================================================
# DATASET AUTHORITY
# ============================================================


def test_dataset_must_belong_to_workflow(
) -> None:

    with isolated_environment() as (
        _,
        model_store,
    ):

        seed_preparation_authority(
            workflow_id=
                "prep:ml-store",

            dataset_id=
                "dataset:real",
        )


        try:
            register_ml_model_artifact(
                training_contract=
                    training_contract(
                        dataset_id=
                            "dataset:invented"
                    ),

                metrics={
                    "mae":
                        10.0,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    MODEL_BYTES,
            )

        except MLModelArtifactAuthorityError:
            pass

        else:
            raise AssertionError(
                (
                    "Invented Preparation dataset "
                    "should have been rejected."
                )
            )


        data_root = (
            ml_model_artifact_data_root(
                model_store
            )
        )


        assert (
            not data_root.exists()
            or
            list(
                data_root.rglob(
                    "*.joblib"
                )
            )
            ==
            []
        )


# ============================================================
# WORKFLOW READ SCOPE
# ============================================================


def test_workflow_mismatch_is_blocked(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:first",

            dataset_id=
                "dataset:validated",
        )


        record = (
            register_ml_model_artifact(
                training_contract=
                    training_contract(
                        workflow_id=
                            "prep:first"
                    ),

                metrics={
                    "mae":
                        10.0,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    MODEL_BYTES,
            )
        )


        try:
            get_ml_model_artifact(
                model_id=
                    record.model_id,

                workflow_id=
                    "prep:second",
            )

        except MLModelArtifactWorkflowMismatchError:
            pass

        else:
            raise AssertionError(
                (
                    "Cross-workflow Model Artifact "
                    "read should have been rejected."
                )
            )


# ============================================================
# LISTING DOES NOT OPEN BINARIES
# ============================================================


def test_workflow_listing_is_metadata_only(
) -> None:

    with isolated_environment() as (
        _,
        model_store,
    ):

        seed_preparation_authority(
            workflow_id=
                "prep:ml-store",

            dataset_id=
                "dataset:validated",
        )


        record = (
            register_ml_model_artifact(
                training_contract=
                    training_contract(),

                metrics={
                    "mae":
                        10.0,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    MODEL_BYTES,
            )
        )


        binary_path = (
            ml_model_artifact_data_root(
                model_store
            )
            /
            record.model_path
        )


        original = (
            binary_path.read_bytes()
        )


        tampered = bytearray(
            original
        )


        tampered[
            0
        ] = (
            tampered[
                0
            ]
            ^
            0x01
        )


        binary_path.write_bytes(
            bytes(
                tampered
            )
        )


        listed = (
            list_ml_model_artifacts(
                workflow_id=
                    "prep:ml-store"
            )
        )


        assert (
            len(
                listed
            )
            ==
            1
        )


        assert (
            listed[
                0
            ].model_id
            ==
            record.model_id
        )


        try:
            load_ml_model_artifact_binary(
                model_id=
                    record.model_id,

                workflow_id=
                    "prep:ml-store",
            )

        except MLModelArtifactStoreError:
            pass

        else:
            raise AssertionError(
                (
                    "Tampered Model Artifact binary "
                    "should fail verified loading."
                )
            )


# ============================================================
# SQLITE FAILURE COMPENSATION
# ============================================================


def test_index_failure_removes_new_binary(
) -> None:

    with isolated_environment() as (
        _,
        model_store,
    ):

        seed_preparation_authority(
            workflow_id=
                "prep:ml-store",

            dataset_id=
                "dataset:validated",
        )


        original_upsert = (
            store_module
            .upsert_ml_model_artifact_index_entry
        )


        def fail_index_write(
            *,
            store_path,
            entry,
        ):
            raise RuntimeError(
                "injected SQLite index failure"
            )


        store_module.upsert_ml_model_artifact_index_entry = (
            fail_index_write
        )


        try:
            try:
                register_ml_model_artifact(
                    training_contract=
                        training_contract(),

                    metrics={
                        "mae":
                            10.0,
                    },

                    train_rows=
                        80,

                    test_rows=
                        20,

                    model_bytes=
                        MODEL_BYTES,
                )

            except MLModelArtifactStoreError:
                pass

            else:
                raise AssertionError(
                    (
                        "Injected index failure "
                        "should propagate through "
                        "the store."
                    )
                )

        finally:
            store_module.upsert_ml_model_artifact_index_entry = (
                original_upsert
            )


        data_root = (
            ml_model_artifact_data_root(
                model_store
            )
        )


        assert (
            not data_root.exists()
            or
            list(
                data_root.rglob(
                    "*.joblib"
                )
            )
            ==
            []
        )


        with sqlite_connection(
            write=False
        ) as connection:

            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM ml_model_artifacts
                    """
                )
                .fetchone()[0]
            )


        assert (
            count
            ==
            0
        )


# ============================================================
# STORE PATH
# ============================================================


def test_configured_store_path_is_authoritative(
) -> None:

    with isolated_environment() as (
        _,
        model_store,
    ):

        assert (
            resolve_ml_model_artifact_store_path()
            ==
            model_store.resolve()
        )


# ============================================================
# VERSION
# ============================================================


def test_model_artifact_store_version(
) -> None:

    assert (
        ML_MODEL_ARTIFACT_STORE_RULE_VERSION
        ==
        "ml_model_artifact_store_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS SERVER-OWNED "
            "ML MODEL ARTIFACT STORE v0.1 ==="
        )
    )

    print()


    test_registration_does_not_accept_identity_or_path()

    print(
        "Registration identity/path are server-owned: PASS"
    )


    test_register_and_restore_server_owned_model()

    print(
        "Model Artifact register / restore / binary load: PASS"
    )


    test_model_identities_are_server_generated_and_unique()

    print(
        "Server-generated Model Artifact identities: PASS"
    )


    test_unknown_workflow_is_blocked_before_model_write()

    print(
        "Unknown Preparation workflow is blocked: PASS"
    )


    test_dataset_must_belong_to_workflow()

    print(
        "Dataset ownership is enforced: PASS"
    )


    test_workflow_mismatch_is_blocked()

    print(
        "Cross-workflow Model Artifact read is blocked: PASS"
    )


    test_workflow_listing_is_metadata_only()

    print(
        "Workflow listing remains metadata-only: PASS"
    )


    test_index_failure_removes_new_binary()

    print(
        "SQLite failure compensates filesystem write: PASS"
    )


    test_configured_store_path_is_authoritative()

    print(
        "Configured Model Artifact store scope: PASS"
    )


    test_model_artifact_store_version()

    print(
        "Model Artifact Store rule version: PASS"
    )


    print()

    print(
        "Server-owned ML Model Artifact Store v0.1: PASS"
    )


if __name__ == "__main__":
    main()
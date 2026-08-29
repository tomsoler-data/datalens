from __future__ import annotations


import json
import os


from contextlib import (
    contextmanager,
)


from pathlib import (
    Path,
)


from tempfile import (
    TemporaryDirectory,
)


import pandas as pd


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_artifact_store import (
    ML_MODEL_ARTIFACT_STORE_PATH_ENV,
    register_ml_model_artifact,
    resolve_ml_model_artifact_store_path,
)


from app.ml.monitoring_profile_builder import (
    build_ml_monitoring_profile,
)


from app.ml.monitoring_profile_store import (
    ML_MONITORING_PROFILE_STORE_RULE_VERSION,
    MLMonitoringProfileAlreadyExistsError,
    MLMonitoringProfileAuthorityError,
    MLMonitoringProfileNotFoundError,
    MLMonitoringProfileWorkflowMismatchError,
    get_ml_monitoring_profile,
    list_ml_monitoring_profiles,
    register_ml_monitoring_profile,
)


from app.persistence.sqlite_database import (
    DATALENS_SQLITE_PATH_ENV,
    SQLITE_SCHEMA_VERSION,
    sqlite_connection,
    sqlite_schema_version,
)


# ============================================================
# HELPERS
# ============================================================


@contextmanager
def isolated_environment(
):

    previous_sqlite = os.environ.get(
        DATALENS_SQLITE_PATH_ENV
    )


    previous_model_store = os.environ.get(
        ML_MODEL_ARTIFACT_STORE_PATH_ENV
    )


    with TemporaryDirectory() as directory:

        root = Path(
            directory
        )


        os.environ[
            DATALENS_SQLITE_PATH_ENV
        ] = str(
            root
            /
            "datalens.sqlite3"
        )


        os.environ[
            ML_MODEL_ARTIFACT_STORE_PATH_ENV
        ] = str(
            root
            /
            "ml"
            /
            "model_artifacts.json"
        )


        try:
            yield root

        finally:
            if previous_sqlite is None:
                os.environ.pop(
                    DATALENS_SQLITE_PATH_ENV,
                    None,
                )

            else:
                os.environ[
                    DATALENS_SQLITE_PATH_ENV
                ] = previous_sqlite


            if previous_model_store is None:
                os.environ.pop(
                    ML_MODEL_ARTIFACT_STORE_PATH_ENV,
                    None,
                )

            else:
                os.environ[
                    ML_MODEL_ARTIFACT_STORE_PATH_ENV
                ] = previous_model_store


def contract(
    *,
    workflow_id: str = "prep:monitoring-store",
    dataset_id: str = "dataset:validated",
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            problem_type=
                "classification",

            target_column=
                "target",

            feature_columns=[
                "age",
                "segment",
            ],

            categorical_feature_columns=[
                "segment"
            ],

            estimator_key=
                "logistic_regression",
        )
    )


def training_frame(
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "age": [
                    20.0,
                    25.0,
                    30.0,
                    35.0,
                    40.0,
                    45.0,
                ],

                "segment": [
                    "standard",
                    "premium",
                    "standard",
                    "premium",
                    "standard",
                    "premium",
                ],
            }
        )
    )


def seed_preparation_authority(
    *,
    training_contract: MLTrainingContract,
    revision: int = 7,
) -> None:

    now = (
        "2026-08-29T12:00:00+00:00"
    )


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
                training_contract.workflow_id,
                revision,
                "{}",
                now,
                now,
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
                "test-preparation-store",
                training_contract.workflow_id,
                training_contract.dataset_id,
                "monitoring.csv",
                "clean",
                8,
                3,
                "[]",
                "[]",
                "{}",
                "data/monitoring.parquet",
            ),
        )


def persisted_artifact_and_profile(
):

    frame = (
        training_frame()
    )


    training_contract = (
        contract()
    )


    seed_preparation_authority(
        training_contract=
            training_contract
    )


    artifact = (
        register_ml_model_artifact(
            training_contract=
                training_contract,

            metrics={
                "accuracy":
                    0.75,

                "balanced_accuracy":
                    0.75,

                "f1_macro":
                    0.75,
            },

            train_rows=
                len(
                    frame
                ),

            test_rows=
                2,

            model_bytes=
                b"trusted-model-bytes",

            preparation_session_revision=
                7,

            created_at_utc=
                "2026-08-29T12:01:00+00:00",
        )
    )


    profile = (
        build_ml_monitoring_profile(
            x_train=
                frame,

            model_artifact=
                artifact,
        )
    )


    return (
        artifact,
        profile,
    )


def expect_error(
    error_type,
    factory,
) -> None:

    try:
        factory()

    except error_type:
        return


    raise AssertionError(
        (
            "Expected "
            f"{error_type.__name__}."
        )
    )


# ============================================================
# TESTS
# ============================================================


def test_sqlite_schema_v10(
) -> None:

    with isolated_environment():

        # Schema v10 introduced the Monitoring Profile.
        # Later schemas must preserve that migration.
        assert (
            SQLITE_SCHEMA_VERSION
            >=
            10
        )


        assert (
            sqlite_schema_version()
            ==
            SQLITE_SCHEMA_VERSION
        )


        with sqlite_connection(
            write=False
        ) as connection:

            tables = {
                str(
                    row[
                        "name"
                    ]
                )

                for row
                in connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    """
                ).fetchall()
            }


        assert (
            "ml_monitoring_profiles"
            in
            tables
        )


def test_register_and_get_roundtrip(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        persisted = (
            register_ml_monitoring_profile(
                profile=
                    profile
            )
        )


        restored = (
            get_ml_monitoring_profile(
                model_id=
                    profile.model_id,

                workflow_id=
                    profile.workflow_id,
            )
        )


        assert (
            persisted
            ==
            profile
        )


        assert (
            restored
            ==
            profile
        )


def test_workflow_list_returns_profile(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        register_ml_monitoring_profile(
            profile=
                profile
        )


        profiles = (
            list_ml_monitoring_profiles(
                workflow_id=
                    profile.workflow_id
            )
        )


        assert (
            profiles
            ==
            [
                profile
            ]
        )


def test_duplicate_model_profile_is_blocked(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        register_ml_monitoring_profile(
            profile=
                profile
        )


        expect_error(
            MLMonitoringProfileAlreadyExistsError,

            lambda:
                register_ml_monitoring_profile(
                    profile=
                        profile
                ),
        )


def test_invented_model_is_blocked(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        invented = (
            profile.model_copy(
                update={
                    "model_id":
                        (
                            "model:"
                            +
                            (
                                "f"
                                *
                                32
                            )
                        )
                }
            )
        )


        expect_error(
            MLMonitoringProfileAuthorityError,

            lambda:
                register_ml_monitoring_profile(
                    profile=
                        invented
                ),
        )


def test_workflow_mismatch_is_blocked(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        tampered = (
            profile.model_copy(
                update={
                    "workflow_id":
                        "prep:other"
                }
            )
        )


        expect_error(
            MLMonitoringProfileAuthorityError,

            lambda:
                register_ml_monitoring_profile(
                    profile=
                        tampered
                ),
        )


def test_training_fingerprint_mismatch_is_blocked(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        tampered = (
            profile.model_copy(
                update={
                    "training_contract_sha256":
                        (
                            "b"
                            *
                            64
                        )
                }
            )
        )


        expect_error(
            MLMonitoringProfileAuthorityError,

            lambda:
                register_ml_monitoring_profile(
                    profile=
                        tampered
                ),
        )


def test_preparation_revision_mismatch_is_blocked(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        tampered = (
            profile.model_copy(
                update={
                    "preparation_session_revision":
                        8
                }
            )
        )


        expect_error(
            MLMonitoringProfileAuthorityError,

            lambda:
                register_ml_monitoring_profile(
                    profile=
                        tampered
                ),
        )


def test_reference_row_count_mismatch_is_blocked(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        payload = (
            profile.model_dump(
                mode="json"
            )
        )


        # ----------------------------------------------------
        # Build a STRUCTURALLY VALID five-row monitoring
        # profile.
        #
        # The persisted Model Artifact still owns a six-row
        # training split.
        #
        # This lets the test reach the Store authority layer
        # instead of being rejected earlier by the monitoring
        # contract itself.
        # ----------------------------------------------------


        categorical_payload = (
            payload[
                "feature_profiles"
            ][
                1
            ]
        )


        tracked_hashes = [
            item[
                "value_sha256"
            ]

            for item
            in categorical_payload[
                "tracked_categories"
            ]
        ]


        assert (
            len(
                tracked_hashes
            )
            ==
            2
        )


        payload[
            "reference_row_count"
        ] = 5


        payload[
            "feature_profiles"
        ] = [
            {
                "feature_name":
                    "age",

                "kind":
                    "numeric",

                "total_count":
                    5,

                "non_missing_count":
                    5,

                "missing_count":
                    0,

                "missing_rate":
                    0.0,

                "mean":
                    30.0,

                "std":
                    7.0710678118654755,

                "minimum":
                    20.0,

                "q25":
                    25.0,

                "median":
                    30.0,

                "q75":
                    35.0,

                "maximum":
                    40.0,

                "histogram_edges":
                    [],

                "histogram_counts":
                    [
                        5
                    ],

                "histogram_rates":
                    [
                        1.0
                    ],
            },

            {
                "feature_name":
                    "segment",

                "kind":
                    "categorical",

                "category_identity":
                    "sha256",

                "total_count":
                    5,

                "non_missing_count":
                    5,

                "missing_count":
                    0,

                "missing_rate":
                    0.0,

                "distinct_count":
                    2,

                "tracked_categories":
                    [
                        {
                            "value_sha256":
                                tracked_hashes[
                                    0
                                ],

                            "count":
                                3,

                            "rate":
                                0.6,
                        },

                        {
                            "value_sha256":
                                tracked_hashes[
                                    1
                                ],

                            "count":
                                2,

                            "rate":
                                0.4,
                        },
                    ],

                "other_count":
                    0,

                "other_rate":
                    0.0,
            },
        ]


        tampered = (
            profile.__class__
            .model_validate(
                payload
            )
        )


        assert (
            tampered.reference_row_count
            ==
            5
        )


        assert all(
            feature.total_count
            ==
            5

            for feature
            in tampered.feature_profiles
        )


        expect_error(
            MLMonitoringProfileAuthorityError,

            lambda:
                register_ml_monitoring_profile(
                    profile=
                        tampered
                ),
        )


def test_current_preparation_revision_does_not_rewrite_history(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        with sqlite_connection(
            write=True
        ) as connection:

            connection.execute(
                """
                UPDATE preparation_sessions

                SET revision = ?

                WHERE workflow_id = ?
                """,
                (
                    8,
                    profile.workflow_id,
                ),
            )


        persisted = (
            register_ml_monitoring_profile(
                profile=
                    profile
            )
        )


        assert (
            persisted.preparation_session_revision
            ==
            7
        )


def test_get_workflow_mismatch_is_blocked(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        register_ml_monitoring_profile(
            profile=
                profile
        )


        expect_error(
            MLMonitoringProfileWorkflowMismatchError,

            lambda:
                get_ml_monitoring_profile(
                    model_id=
                        profile.model_id,

                    workflow_id=
                        "prep:other",
                ),
        )


def test_missing_profile_is_explicit(
) -> None:

    with isolated_environment():

        _ = (
            sqlite_schema_version()
        )


        expect_error(
            MLMonitoringProfileNotFoundError,

            lambda:
                get_ml_monitoring_profile(
                    model_id=(
                        "model:"
                        +
                        (
                            "0"
                            *
                            32
                        )
                    )
                ),
        )


def test_persisted_payload_is_aggregate_only(
) -> None:

    with isolated_environment():

        (
            _,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        register_ml_monitoring_profile(
            profile=
                profile
        )


        store_root = str(
            resolve_ml_model_artifact_store_path()
            .expanduser()
            .resolve()
        )


        with sqlite_connection(
            write=False
        ) as connection:

            row = (
                connection.execute(
                    """
                    SELECT payload_json
                    FROM ml_monitoring_profiles

                    WHERE
                        store_root = ?
                        AND
                        model_id = ?
                    """,
                    (
                        store_root,
                        profile.model_id,
                    ),
                )
                .fetchone()
            )


        assert (
            row
            is not None
        )


        raw_payload = str(
            row[
                "payload_json"
            ]
        )


        payload = json.loads(
            raw_payload
        )


        assert (
            "premium"
            not in
            raw_payload
        )


        assert (
            "standard"
            not in
            raw_payload
        )


        assert (
            "raw_rows"
            not in
            payload
        )


        assert (
            "predictions"
            not in
            payload
        )


        assert (
            payload[
                "privacy_scope"
            ]
            ==
            "aggregate_only"
        )


def test_model_delete_cascades_monitoring_profile(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        register_ml_monitoring_profile(
            profile=
                profile
        )


        store_root = str(
            resolve_ml_model_artifact_store_path()
            .expanduser()
            .resolve()
        )


        with sqlite_connection(
            write=True
        ) as connection:

            connection.execute(
                """
                DELETE FROM ml_model_artifacts

                WHERE
                    store_root = ?
                    AND
                    model_id = ?
                """,
                (
                    store_root,
                    artifact.model_id,
                ),
            )


        expect_error(
            MLMonitoringProfileNotFoundError,

            lambda:
                get_ml_monitoring_profile(
                    model_id=
                        profile.model_id
                ),
        )


def test_store_rule_version(
) -> None:

    assert (
        ML_MONITORING_PROFILE_STORE_RULE_VERSION
        ==
        "ml_monitoring_profile_store_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MONITORING PROFILE STORE v0.1 ==="
    )


    tests = [
        (
            "SQLite schema v10",
            test_sqlite_schema_v10,
        ),
        (
            "Register / get roundtrip",
            test_register_and_get_roundtrip,
        ),
        (
            "Workflow profile list",
            test_workflow_list_returns_profile,
        ),
        (
            "Duplicate model profile blocked",
            test_duplicate_model_profile_is_blocked,
        ),
        (
            "Invented model blocked",
            test_invented_model_is_blocked,
        ),
        (
            "Workflow authority binding",
            test_workflow_mismatch_is_blocked,
        ),
        (
            "Training fingerprint binding",
            test_training_fingerprint_mismatch_is_blocked,
        ),
        (
            "Preparation revision binding",
            test_preparation_revision_mismatch_is_blocked,
        ),
        (
            "Reference row count binding",
            test_reference_row_count_mismatch_is_blocked,
        ),
        (
            "Historical Preparation snapshot preserved",
            test_current_preparation_revision_does_not_rewrite_history,
        ),
        (
            "Read workflow isolation",
            test_get_workflow_mismatch_is_blocked,
        ),
        (
            "Explicit missing profile",
            test_missing_profile_is_explicit,
        ),
        (
            "Aggregate-only persistence",
            test_persisted_payload_is_aggregate_only,
        ),
        (
            "Model deletion cascade",
            test_model_delete_cascades_monitoring_profile,
        ),
        (
            "Monitoring Profile Store rule version",
            test_store_rule_version,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        "PASS - ML Monitoring Profile Store v0.1"
    )


if __name__ == "__main__":
    main()

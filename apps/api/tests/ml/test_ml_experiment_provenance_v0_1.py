from __future__ import annotations


import math
import os
import tempfile


from contextlib import (
    contextmanager,
)


from pathlib import (
    Path,
)


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    ML_EXPERIMENT_PROVENANCE_RULE_VERSION,
    MLExperimentProvenanceRecord,
    build_ml_experiment_provenance,
    canonical_ml_training_contract_json,
    ml_training_contract_sha256,
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_connection,
    sqlite_schema_version,
)


# ============================================================
# CONTRACT
# ============================================================


def training_contract(
    *,
    estimator_key: str = (
        "linear_regression"
    ),
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:experiment-provenance",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
                "tenure",
            ],

            estimator_key=
                estimator_key,
        )
    )


# ============================================================
# SQLITE ISOLATION
# ============================================================


@contextmanager
def isolated_sqlite_database(
):

    previous = os.environ.get(
        "DATALENS_SQLITE_PATH"
    )


    with tempfile.TemporaryDirectory(
        prefix=
            "datalens-experiment-provenance-"
    ) as root:

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
            yield database_path

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
# CANONICAL CONTRACT
# ============================================================


def test_training_contract_serialization_is_canonical(
) -> None:

    contract = (
        training_contract()
    )


    first = (
        canonical_ml_training_contract_json(
            contract
        )
    )


    second = (
        canonical_ml_training_contract_json(
            MLTrainingContract.model_validate(
                contract.model_dump(
                    mode="json"
                )
            )
        )
    )


    assert (
        first
        ==
        second
    )


# ============================================================
# DETERMINISTIC FINGERPRINT
# ============================================================


def test_training_contract_sha256_is_deterministic(
) -> None:

    first = (
        ml_training_contract_sha256(
            training_contract()
        )
    )


    second = (
        ml_training_contract_sha256(
            training_contract()
        )
    )


    assert (
        first
        ==
        second
    )


    assert (
        len(
            first
        )
        ==
        64
    )


    int(
        first,
        16,
    )


def test_training_contract_change_changes_fingerprint(
) -> None:

    linear = (
        ml_training_contract_sha256(
            training_contract(
                estimator_key=
                    "linear_regression"
            )
        )
    )


    ridge = (
        ml_training_contract_sha256(
            training_contract(
                estimator_key=
                    "ridge_regression"
            )
        )
    )


    assert (
        linear
        !=
        ridge
    )


# ============================================================
# SERVER-OWNED EXPERIMENT ID
# ============================================================


def test_experiment_identity_is_server_generated(
) -> None:

    contract = (
        training_contract()
    )


    first = (
        build_ml_experiment_provenance(
            training_contract=
                contract,

            preparation_session_revision=
                7,

            model_id=
                "model:first",

            train_rows=
                80,

            test_rows=
                20,

            metrics={
                "rmse":
                    12.0,

                "mae":
                    10.0,

                "r2":
                    0.82,
            },
        )
    )


    second = (
        build_ml_experiment_provenance(
            training_contract=
                contract,

            preparation_session_revision=
                7,

            model_id=
                "model:second",

            train_rows=
                80,

            test_rows=
                20,

            metrics={
                "rmse":
                    12.0,

                "mae":
                    10.0,

                "r2":
                    0.82,
            },
        )
    )


    assert (
        first.experiment_id
        .startswith(
            "experiment:"
        )
    )


    assert (
        first.experiment_id
        !=
        second.experiment_id
    )


    assert (
        first.workflow_id
        ==
        contract.workflow_id
    )


    assert (
        first.dataset_id
        ==
        contract.dataset_id
    )


    assert (
        first.preparation_session_revision
        ==
        7
    )


    assert (
        first.training_contract_sha256
        ==
        ml_training_contract_sha256(
            contract
        )
    )


# ============================================================
# PRIVACY-MINIMAL RESULT
# ============================================================


def test_provenance_contains_no_raw_data_surface(
) -> None:

    provenance = (
        build_ml_experiment_provenance(
            training_contract=
                training_contract(),

            preparation_session_revision=
                3,

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
        provenance.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "dataframe",
        "rows",
        "predictions",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "model_bytes",
        "model_path",
    }


    assert (
        forbidden
        .isdisjoint(
            payload
        )
    )


# ============================================================
# FINITE METRIC GUARD
# ============================================================


def test_non_finite_metric_is_blocked(
) -> None:

    try:
        build_ml_experiment_provenance(
            training_contract=
                training_contract(),

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
                    float(
                        "nan"
                    )
            },
        )

    except ValidationError:
        return


    raise AssertionError(
        (
            "Experiment provenance must block "
            "non-finite metrics."
        )
    )


# ============================================================
# UNKNOWN FIELD GUARD
# ============================================================


def test_unknown_provenance_field_is_blocked(
) -> None:

    valid = (
        build_ml_experiment_provenance(
            training_contract=
                training_contract(),

            preparation_session_revision=
                1,

            model_id=
                "model:test",

            train_rows=
                80,

            test_rows=
                20,

            metrics={
                "rmse":
                    3.0
            },
        )
    )


    payload = (
        valid.model_dump(
            mode="json"
        )
    )


    payload[
        "raw_predictions"
    ] = [
        1.0,
        2.0,
    ]


    try:
        MLExperimentProvenanceRecord.model_validate(
            payload
        )

    except ValidationError:
        return


    raise AssertionError(
        (
            "Unknown provenance fields must "
            "be blocked."
        )
    )


# ============================================================
# SQLITE V9
# ============================================================


def test_sqlite_schema_v9_experiment_provenance(
) -> None:

    with isolated_sqlite_database():

        # Schema v9 introduced Experiment Provenance.
        # Newer application schemas must preserve that
        # historical migration and its guarantees.
        assert (
            SQLITE_SCHEMA_VERSION
            >=
            9
        )


        assert (
            sqlite_schema_version()
            ==
            SQLITE_SCHEMA_VERSION
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
                        version = 9
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
                9
            )


            assert (
                str(
                    migration[
                        "name"
                    ]
                )
                ==
                "ml_experiment_provenance_metadata"
            )


            columns = {
                str(
                    row[
                        "name"
                    ]
                ):
                    row

                for row
                in connection.execute(
                    """
                    PRAGMA table_info(
                        ml_model_artifacts
                    )
                    """
                ).fetchall()
            }


            assert (
                "experiment_id"
                in
                columns
            )


            assert (
                "experiment_provenance_json"
                in
                columns
            )


            # Nullable by design for legacy v8 artifacts.
            assert (
                int(
                    columns[
                        "experiment_id"
                    ][
                        "notnull"
                    ]
                )
                ==
                0
            )


            assert (
                int(
                    columns[
                        "experiment_provenance_json"
                    ][
                        "notnull"
                    ]
                )
                ==
                0
            )


            index = (
                connection.execute(
                    """
                    SELECT sql
                    FROM sqlite_master

                    WHERE
                        type = 'index'
                        AND
                        name =
                        'idx_ml_model_artifacts_scope_experiment'
                    """
                )
                .fetchone()
            )


            assert (
                index
                is not None
            )


            index_sql = str(
                index[
                    "sql"
                ]
            )


            assert (
                "experiment_id"
                in
                index_sql
            )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_EXPERIMENT_PROVENANCE_RULE_VERSION
        ==
        "ml_experiment_provenance_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML EXPERIMENT PROVENANCE v0.1 ==="
    )

    print()


    test_training_contract_serialization_is_canonical()

    print(
        "Canonical ML Training Contract serialization: PASS"
    )


    test_training_contract_sha256_is_deterministic()

    print(
        "Deterministic Training Contract SHA-256: PASS"
    )


    test_training_contract_change_changes_fingerprint()

    print(
        "Contract change changes provenance fingerprint: PASS"
    )


    test_experiment_identity_is_server_generated()

    print(
        "Server-generated experiment identity: PASS"
    )


    test_provenance_contains_no_raw_data_surface()

    print(
        "Privacy-minimal provenance surface: PASS"
    )


    test_non_finite_metric_is_blocked()

    print(
        "Non-finite provenance metric is blocked: PASS"
    )


    test_unknown_provenance_field_is_blocked()

    print(
        "Unknown provenance fields are blocked: PASS"
    )


    test_sqlite_schema_v9_experiment_provenance()

    print(
        "SQLite schema v9 experiment provenance migration: PASS"
    )


    test_rule_version()

    print(
        "ML Experiment Provenance rule version: PASS"
    )


    print()

    print(
        "ML Experiment Provenance v0.1: PASS"
    )


if __name__ == "__main__":
    main()

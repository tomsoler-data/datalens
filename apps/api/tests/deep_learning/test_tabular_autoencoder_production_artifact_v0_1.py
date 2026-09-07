from __future__ import annotations


import hashlib
import os
import tempfile


from contextlib import (
    contextmanager,
)


from dataclasses import (
    dataclass,
)


from pathlib import (
    Path,
)


import numpy as np
import pandas as pd


import app.deep_learning.anomaly_model_lab_executor as executor_module


from app.deep_learning.anomaly_model_lab_executor import (
    TabularAutoencoderExecutionResult,
    execute_tabular_autoencoder,
)


from app.deep_learning.autoencoder_bundle import (
    deserialize_trusted_tabular_autoencoder_bundle,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.model_artifact_store import (
    get_ml_model_artifact,
    load_ml_model_artifact_binary,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


# ============================================================
# FAKE PREPARATION HANDOFF
# ============================================================


@dataclass(
    frozen=True,
)
class FakeAnalysisInputHandoff:

    workflow_id: str

    session_revision: int

    dataset_ids: tuple[
        str,
        ...,
    ]

    dataset_records: tuple[
        dict,
        ...,
    ]


@contextmanager
def patched_production_environment(
    *,
    dataframe: pd.DataFrame,
    contract: MLAnomalyTrainingContract,
    revision: int = 37,
):

    original_handoff = (
        executor_module
        .load_validated_analysis_input
    )


    previous_sqlite = os.environ.get(
        "DATALENS_SQLITE_PATH"
    )


    previous_store = os.environ.get(
        "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
    )


    with tempfile.TemporaryDirectory(
        prefix=
            "datalens-autoencoder-production-"
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


        def fake_load_validated_analysis_input(
            *,
            workflow_id: str,
        ):

            assert (
                workflow_id
                ==
                contract.workflow_id
            )


            return (
                FakeAnalysisInputHandoff(
                    workflow_id=
                        workflow_id,

                    session_revision=
                        revision,

                    dataset_ids=(
                        contract.dataset_id,
                    ),

                    dataset_records=(
                        {
                            "dataset_id":
                                contract.dataset_id,

                            "dataframe":
                                dataframe.copy(
                                    deep=True
                                ),
                        },
                    ),
                )
            )


        executor_module.load_validated_analysis_input = (
            fake_load_validated_analysis_input
        )


        try:

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
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        contract.workflow_id,
                        revision,
                        "{}",
                        "2026-09-07T14:30:00+00:00",
                        "2026-09-07T14:30:00+00:00",
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
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "test-preparation-root",
                        contract.workflow_id,
                        contract.dataset_id,
                        "validated.csv",
                        "source",
                        int(
                            len(
                                dataframe
                            )
                        ),
                        int(
                            len(
                                dataframe.columns
                            )
                        ),
                        "[]",
                        "[]",
                        "[]",
                        "data/validated.json.gz",
                    ),
                )


            yield


        finally:

            executor_module.load_validated_analysis_input = (
                original_handoff
            )


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
# FIXTURE
# ============================================================


def build_dataframe(
) -> pd.DataFrame:

    rows = 80


    amount = np.linspace(
        10.0,
        200.0,
        rows,
        dtype=np.float64,
    )


    frequency = np.linspace(
        1.0,
        40.0,
        rows,
        dtype=np.float64,
    )


    margin = (
        0.30
        *
        amount
        +
        1.5
        *
        frequency
    )


    return (
        pd.DataFrame(
            {
                "amount":
                    amount,

                "frequency":
                    frequency,

                "margin":
                    margin,
            },
            index=[
                70_000
                +
                index
                *
                7

                for index
                in range(
                    rows
                )
            ],
        )
    )


def build_contract(
) -> MLAnomalyTrainingContract:

    return (
        MLAnomalyTrainingContract(
            workflow_id=
                "prep:autoencoder-production",

            dataset_id=
                "dataset:autoencoder-production",

            feature_columns=[
                "amount",
                "frequency",
                "margin",
            ],

            estimator_hyperparameters={
                "kind":
                    "tabular_autoencoder",

                "hidden_features":
                    8,

                "latent_features":
                    3,

                "epochs":
                    8,

                "batch_size":
                    16,

                "learning_rate":
                    0.002,
            },

            preprocessing={
                "numeric_imputation":
                    "error",

                "categorical_imputation":
                    "error",

                "categorical_encoding":
                    "one_hot",

                "handle_unknown_categories":
                    "ignore",

                "scale_numeric":
                    True,
            },

            split={
                "strategy":
                    "holdout",

                "test_size":
                    0.25,

                "random_seed":
                    43,

                "shuffle":
                    True,

                "stratify":
                    False,
            },

            threshold_quantile=
                0.95,
        )
    )


# ============================================================
# PRODUCTION ARTIFACT / SINGLE EXECUTION
# ============================================================


def test_production_executor_persists_exact_core_state_once(
) -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract()
    )


    original_core = (
        executor_module
        .execute_tabular_autoencoder_core
    )


    original_serializer = (
        executor_module
        .serialize_tabular_autoencoder_bundle
    )


    original_register = (
        executor_module
        .register_ml_model_artifact
    )


    counts = {
        "core":
            0,

        "serialize":
            0,

        "register":
            0,
    }


    captured = {}


    def counted_core(
        **kwargs,
    ):

        counts[
            "core"
        ] += 1


        result = (
            original_core(
                **kwargs
            )
        )


        captured[
            "core_result"
        ] = result


        return result


    def counted_serializer(
        *,
        model,
        preprocessor,
        training_contract,
        threshold,
    ):

        counts[
            "serialize"
        ] += 1


        core_result = (
            captured[
                "core_result"
            ]
        )


        assert (
            model
            is
            core_result.model
        )


        assert (
            preprocessor
            is
            core_result.preprocessor
        )


        assert (
            threshold
            is
            core_result.threshold
        )


        assert (
            training_contract
            ==
            contract
        )


        bundle = (
            original_serializer(
                model=
                    model,

                preprocessor=
                    preprocessor,

                training_contract=
                    training_contract,

                threshold=
                    threshold,
            )
        )


        captured[
            "bundle"
        ] = bundle


        return bundle


    def counted_register(
        **kwargs,
    ):

        counts[
            "register"
        ] += 1


        assert (
            kwargs[
                "training_contract"
            ]
            ==
            contract
        )


        assert (
            kwargs[
                "model_bytes"
            ]
            ==
            captured[
                "bundle"
            ]
        )


        assert (
            kwargs[
                "serialization_format"
            ]
            ==
            "pytorch_bundle"
        )


        core_result = (
            captured[
                "core_result"
            ]
        )


        assert (
            kwargs[
                "train_rows"
            ]
            ==
            core_result.train_rows
        )


        assert (
            kwargs[
                "test_rows"
            ]
            ==
            core_result.test_rows
        )


        artifact = (
            original_register(
                **kwargs
            )
        )


        captured[
            "artifact"
        ] = artifact


        return artifact


    executor_module.execute_tabular_autoencoder_core = (
        counted_core
    )


    executor_module.serialize_tabular_autoencoder_bundle = (
        counted_serializer
    )


    executor_module.register_ml_model_artifact = (
        counted_register
    )


    try:

        with patched_production_environment(
            dataframe=
                dataframe,

            contract=
                contract,

            revision=
                37,
        ):

            result = (
                execute_tabular_autoencoder(
                    training_contract=
                        contract,

                    expected_preparation_session_revision=
                        37,

                    execution_device=
                        "cpu",
                )
            )


            assert (
                counts
                ==
                {
                    "core":
                        1,

                    "serialize":
                        1,

                    "register":
                        1,
                }
            )


            assert isinstance(
                result,
                TabularAutoencoderExecutionResult,
            )


            core_result = (
                captured[
                    "core_result"
                ]
            )


            artifact = (
                captured[
                    "artifact"
                ]
            )


            assert (
                result.model_artifact
                ==
                artifact
            )


            assert (
                result.model_artifact.training_contract
                ==
                contract
            )


            assert (
                result.model_artifact.serialization_format
                ==
                "pytorch_bundle"
            )


            assert (
                result.model_artifact.model_path
                .endswith(
                    ".ptbundle"
                )
            )


            assert (
                result.experiment_provenance
                ==
                result
                .model_artifact
                .experiment_provenance
            )


            assert (
                result.experiment_provenance.model_id
                ==
                result.model_artifact.model_id
            )


            assert (
                result.workflow_id
                ==
                contract.workflow_id
            )


            assert (
                result.dataset_id
                ==
                contract.dataset_id
            )


            assert (
                result.preparation_session_revision
                ==
                37
            )


            assert (
                result.train_rows
                ==
                core_result.train_rows
            )


            assert (
                result.test_rows
                ==
                core_result.test_rows
            )


            assert (
                result.purged_rows
                ==
                core_result.purged_rows
            )


            assert (
                result.train_reconstruction_mse
                ==
                float(
                    core_result
                    .train_evaluation
                    .mean_loss
                )
            )


            assert (
                result.test_reconstruction_mse
                ==
                float(
                    core_result
                    .test_evaluation
                    .mean_loss
                )
            )


            assert (
                result.threshold_quantile
                ==
                float(
                    core_result
                    .threshold
                    .quantile
                )
            )


            assert (
                result.anomaly_threshold
                ==
                float(
                    core_result
                    .threshold
                    .threshold
                )
            )


            restored_artifact = (
                get_ml_model_artifact(
                    model_id=
                        result
                        .model_artifact
                        .model_id,

                    workflow_id=
                        result.workflow_id,
                )
            )


            assert (
                restored_artifact
                ==
                result.model_artifact
            )


            binary = (
                load_ml_model_artifact_binary(
                    model_id=
                        result
                        .model_artifact
                        .model_id,

                    workflow_id=
                        result.workflow_id,
                )
            )


            assert (
                len(
                    binary
                )
                ==
                result
                .model_artifact
                .model_file_bytes
            )


            assert (
                hashlib.sha256(
                    binary
                )
                .hexdigest()
                ==
                result
                .model_artifact
                .model_sha256
            )


            components = (
                deserialize_trusted_tabular_autoencoder_bundle(
                    trusted_bundle_bytes=
                        binary,

                    training_contract=
                        contract,
                )
            )


            assert (
                components.threshold
                ==
                core_result.threshold
            )


            assert (
                components.model.input_features
                ==
                core_result.model.input_features
            )


            assert (
                components.model.hidden_features
                ==
                core_result.model.hidden_features
            )


            assert (
                components.model.latent_features
                ==
                core_result.model.latent_features
            )


            public_payload = (
                result.model_dump(
                    mode="json"
                )
            )


            forbidden_public_fields = {
                "model",
                "preprocessor",
                "threshold",
                "train_flags",
                "test_flags",
                "train_errors",
                "test_errors",
                "reconstruction_errors",
                "epoch_losses",
            }


            assert (
                forbidden_public_fields
                .isdisjoint(
                    public_payload
                )
            )


    finally:

        executor_module.execute_tabular_autoencoder_core = (
            original_core
        )


        executor_module.serialize_tabular_autoencoder_bundle = (
            original_serializer
        )


        executor_module.register_ml_model_artifact = (
            original_register
        )


# ============================================================
# RULE / STATIC SINGLE-CALL AUTHORITY
# ============================================================


def test_production_persistence_call_surface(
) -> None:

    import ast
    import inspect


    source = (
        inspect.getsource(
            executor_module
            .execute_tabular_autoencoder
        )
    )


    tree = ast.parse(
        source
    )


    calls = []


    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue


        if isinstance(
            node.func,
            ast.Name,
        ):

            calls.append(
                node.func.id
            )


        elif isinstance(
            node.func,
            ast.Attribute,
        ):

            calls.append(
                node.func.attr
            )


    assert (
        calls.count(
            "execute_tabular_autoencoder_core"
        )
        ==
        1
    )


    assert (
        calls.count(
            "serialize_tabular_autoencoder_bundle"
        )
        ==
        1
    )


    assert (
        calls.count(
            "register_ml_model_artifact"
        )
        ==
        1
    )


    for forbidden in (
        "fit",
        "fit_transform",
        "fit_reconstruction_error_threshold",
        "train_reconstruction_epochs",
        "train_reconstruction_batch",
    ):

        assert (
            forbidden
            not in
            calls
        )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS TABULAR AUTOENCODER "
            "PRODUCTION ARTIFACT v0.1 ==="
        )
    )

    print()


    test_production_executor_persists_exact_core_state_once()

    print(
        "Single production core execution: PASS"
    )

    print(
        "Exact trained-state bundle reuse: PASS"
    )

    print(
        "Single server-owned Artifact registration: PASS"
    )

    print(
        "Artifact SHA / size verification: PASS"
    )

    print(
        "Experiment Provenance exposure: PASS"
    )

    print(
        "Privacy-minimal production result: PASS"
    )


    test_production_persistence_call_surface()

    print(
        "No retraining / refit persistence path: PASS"
    )


    print()

    print(
        (
            "PASS - DataLens Tabular Autoencoder "
            "Production Artifact v0.1"
        )
    )


if __name__ == "__main__":
    main()

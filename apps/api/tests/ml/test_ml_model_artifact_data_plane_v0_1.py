from __future__ import annotations


import math
import tempfile


from pathlib import (
    Path,
)


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
    ML_MODEL_ARTIFACT_RULE_VERSION,
)


from app.ml.model_artifact_data_plane import (
    MLModelArtifactDataPlaneError,
    ML_MODEL_ARTIFACT_DATA_PLANE_VERSION,
    delete_ml_model_binary,
    ml_model_artifact_data_root,
    read_ml_model_binary,
    write_ml_model_binary,
)


# ============================================================
# FIXTURE
# ============================================================


MODEL_BYTES = (
    b"DATALENS-TEST-MODEL-BINARY-v0.1"
)


def training_contract(
) -> MLTrainingContract:
    return (
        MLTrainingContract(
            workflow_id="prep:ml-artifact",
            dataset_id="dataset:validated",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
                "tenure",
            ],
            estimator_key="linear_regression",
        )
    )


# ============================================================
# HELPERS
# ============================================================


def expect_validation_error(
    callback,
) -> None:
    try:
        callback()

    except ValidationError:
        return


    raise AssertionError(
        "Expected Pydantic ValidationError."
    )


def expect_data_plane_error(
    callback,
) -> None:
    try:
        callback()

    except MLModelArtifactDataPlaneError:
        return


    raise AssertionError(
        (
            "Expected "
            "MLModelArtifactDataPlaneError."
        )
    )


# ============================================================
# DATA ROOT
# ============================================================


def test_data_root_is_derived_from_store_scope(
) -> None:
    with tempfile.TemporaryDirectory() as root:
        store_path = (
            Path(
                root
            )
            /
            "ml"
            /
            "model_artifacts.json"
        )


        expected = (
            store_path.parent
            /
            "model_artifacts"
        ).resolve()


        assert (
            ml_model_artifact_data_root(
                store_path
            )
            ==
            expected
        )


# ============================================================
# WRITE / READ
# ============================================================


def test_model_binary_round_trip(
) -> None:
    with tempfile.TemporaryDirectory() as root:
        store_path = (
            Path(
                root
            )
            /
            "model_artifacts.json"
        )


        info = (
            write_ml_model_binary(
                store_path=
                    store_path,

                model_id=
                    "model:001",

                model_bytes=
                    MODEL_BYTES,
            )
        )


        assert (
            info[
                "model_file_bytes"
            ]
            ==
            len(
                MODEL_BYTES
            )
        )


        assert (
            len(
                info[
                    "model_sha256"
                ]
            )
            ==
            64
        )


        assert (
            info[
                "model_path"
            ]
            .startswith(
                "data/"
            )
        )


        assert (
            info[
                "model_path"
            ]
            .endswith(
                ".joblib"
            )
        )


        restored = (
            read_ml_model_binary(
                store_path=
                    store_path,

                entry=
                    info,
            )
        )


        assert (
            restored
            ==
            MODEL_BYTES
        )


# ============================================================
# CHECKSUM DETERMINISM
# ============================================================


def test_same_binary_has_same_sha256(
) -> None:
    with tempfile.TemporaryDirectory() as root:
        store_path = (
            Path(
                root
            )
            /
            "model_artifacts.json"
        )


        first = (
            write_ml_model_binary(
                store_path=
                    store_path,

                model_id=
                    "model:001",

                model_bytes=
                    MODEL_BYTES,
            )
        )


        second = (
            write_ml_model_binary(
                store_path=
                    store_path,

                model_id=
                    "model:002",

                model_bytes=
                    MODEL_BYTES,
            )
        )


        assert (
            first[
                "model_sha256"
            ]
            ==
            second[
                "model_sha256"
            ]
        )


        assert (
            first[
                "model_path"
            ]
            !=
            second[
                "model_path"
            ]
        )


# ============================================================
# TAMPER DETECTION
# ============================================================


def test_binary_tampering_is_detected(
) -> None:
    with tempfile.TemporaryDirectory() as root:
        store_path = (
            Path(
                root
            )
            /
            "model_artifacts.json"
        )


        info = (
            write_ml_model_binary(
                store_path=
                    store_path,

                model_id=
                    "model:001",

                model_bytes=
                    MODEL_BYTES,
            )
        )


        data_root = (
            ml_model_artifact_data_root(
                store_path
            )
        )


        model_file = (
            data_root
            /
            info[
                "model_path"
            ]
        )


        original_size = (
            model_file.stat().st_size
        )


        tampered = bytearray(
            model_file.read_bytes()
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


        assert (
            len(
                tampered
            )
            ==
            original_size
        )


        model_file.write_bytes(
            bytes(
                tampered
            )
        )


        def read_tampered(
        ) -> None:
            read_ml_model_binary(
                store_path=
                    store_path,

                entry=
                    info,
            )


        expect_data_plane_error(
            read_tampered
        )


# ============================================================
# SIZE MISMATCH
# ============================================================


def test_binary_size_mismatch_is_detected(
) -> None:
    with tempfile.TemporaryDirectory() as root:
        store_path = (
            Path(
                root
            )
            /
            "model_artifacts.json"
        )


        info = (
            write_ml_model_binary(
                store_path=
                    store_path,

                model_id=
                    "model:001",

                model_bytes=
                    MODEL_BYTES,
            )
        )


        corrupted_metadata = {
            **info,

            "model_file_bytes":
                info[
                    "model_file_bytes"
                ]
                +
                1,
        }


        def read_invalid_size(
        ) -> None:
            read_ml_model_binary(
                store_path=
                    store_path,

                entry=
                    corrupted_metadata,
            )


        expect_data_plane_error(
            read_invalid_size
        )


# ============================================================
# PATH ESCAPE
# ============================================================


def test_path_escape_is_blocked(
) -> None:
    with tempfile.TemporaryDirectory() as root:
        store_path = (
            Path(
                root
            )
            /
            "model_artifacts.json"
        )


        def read_escape(
        ) -> None:
            read_ml_model_binary(
                store_path=
                    store_path,

                entry={
                    "model_path":
                        "../outside.joblib",

                    "model_file_bytes":
                        1,

                    "model_sha256":
                        "0"
                        *
                        64,
                },
            )


        expect_data_plane_error(
            read_escape
        )


# ============================================================
# EMPTY MODEL
# ============================================================


def test_empty_model_binary_is_blocked(
) -> None:
    with tempfile.TemporaryDirectory() as root:
        store_path = (
            Path(
                root
            )
            /
            "model_artifacts.json"
        )


        def write_empty(
        ) -> None:
            write_ml_model_binary(
                store_path=
                    store_path,

                model_id=
                    "model:001",

                model_bytes=
                    b"",
            )


        expect_data_plane_error(
            write_empty
        )


# ============================================================
# DELETE
# ============================================================


def test_model_binary_delete_is_idempotent(
) -> None:
    with tempfile.TemporaryDirectory() as root:
        store_path = (
            Path(
                root
            )
            /
            "model_artifacts.json"
        )


        info = (
            write_ml_model_binary(
                store_path=
                    store_path,

                model_id=
                    "model:001",

                model_bytes=
                    MODEL_BYTES,
            )
        )


        delete_ml_model_binary(
            store_path=
                store_path,

            model_path=
                info[
                    "model_path"
                ],
        )


        delete_ml_model_binary(
            store_path=
                store_path,

            model_path=
                info[
                    "model_path"
                ],
        )


        def read_deleted(
        ) -> None:
            read_ml_model_binary(
                store_path=
                    store_path,

                entry=
                    info,
            )


        expect_data_plane_error(
            read_deleted
        )


# ============================================================
# VALID ARTIFACT RECORD
# ============================================================


def test_valid_model_artifact_record(
) -> None:
    with tempfile.TemporaryDirectory() as root:
        store_path = (
            Path(
                root
            )
            /
            "model_artifacts.json"
        )


        info = (
            write_ml_model_binary(
                store_path=
                    store_path,

                model_id=
                    "model:001",

                model_bytes=
                    MODEL_BYTES,
            )
        )


        record = (
            MLModelArtifactRecord(
                model_id=
                    " model:001 ",

                workflow_id=
                    " prep:ml-artifact ",

                dataset_id=
                    " dataset:validated ",

                training_contract=
                    training_contract(),

                metrics={
                    "mae":
                        12.5,

                    "r2":
                        0.82,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                created_at_utc=
                    "2026-08-28T18:00:00+00:00",

                **info,
            )
        )


        assert (
            record.model_id
            ==
            "model:001"
        )


        assert (
            record.workflow_id
            ==
            "prep:ml-artifact"
        )


        assert (
            record.dataset_id
            ==
            "dataset:validated"
        )


        assert (
            record.metrics[
                "mae"
            ]
            ==
            12.5
        )


        assert (
            record.serialization_format
            ==
            "joblib"
        )


        assert (
            record.rule_version
            ==
            ML_MODEL_ARTIFACT_RULE_VERSION
        )


# ============================================================
# PROVENANCE MISMATCH
# ============================================================


def test_workflow_provenance_mismatch_is_blocked(
) -> None:

    def build(
    ) -> None:
        MLModelArtifactRecord(
            model_id="model:001",
            workflow_id="prep:WRONG",
            dataset_id="dataset:validated",
            training_contract=
                training_contract(),
            metrics={
                "mae":
                    10.0,
            },
            train_rows=80,
            test_rows=20,
            created_at_utc=
                "2026-08-28T18:00:00+00:00",
            model_path=
                "data/model.joblib",
            model_file_bytes=100,
            model_sha256=
                "a"
                *
                64,
        )


    expect_validation_error(
        build
    )


def test_dataset_provenance_mismatch_is_blocked(
) -> None:

    def build(
    ) -> None:
        MLModelArtifactRecord(
            model_id="model:001",
            workflow_id=
                "prep:ml-artifact",
            dataset_id=
                "dataset:WRONG",
            training_contract=
                training_contract(),
            metrics={
                "mae":
                    10.0,
            },
            train_rows=80,
            test_rows=20,
            created_at_utc=
                "2026-08-28T18:00:00+00:00",
            model_path=
                "data/model.joblib",
            model_file_bytes=100,
            model_sha256=
                "a"
                *
                64,
        )


    expect_validation_error(
        build
    )


# ============================================================
# NON-FINITE METRIC
# ============================================================


def test_non_finite_metric_is_blocked(
) -> None:

    def build(
    ) -> None:
        MLModelArtifactRecord(
            model_id="model:001",
            workflow_id=
                "prep:ml-artifact",
            dataset_id=
                "dataset:validated",
            training_contract=
                training_contract(),
            metrics={
                "mae":
                    math.nan,
            },
            train_rows=80,
            test_rows=20,
            created_at_utc=
                "2026-08-28T18:00:00+00:00",
            model_path=
                "data/model.joblib",
            model_file_bytes=100,
            model_sha256=
                "a"
                *
                64,
        )


    expect_validation_error(
        build
    )


# ============================================================
# ARTIFACT PATH VALIDATION
# ============================================================


def test_artifact_record_path_escape_is_blocked(
) -> None:

    def build(
    ) -> None:
        MLModelArtifactRecord(
            model_id="model:001",
            workflow_id=
                "prep:ml-artifact",
            dataset_id=
                "dataset:validated",
            training_contract=
                training_contract(),
            metrics={
                "mae":
                    10.0,
            },
            train_rows=80,
            test_rows=20,
            created_at_utc=
                "2026-08-28T18:00:00+00:00",
            model_path=
                "../model.joblib",
            model_file_bytes=100,
            model_sha256=
                "a"
                *
                64,
        )


    expect_validation_error(
        build
    )


# ============================================================
# VERSIONS
# ============================================================


def test_model_artifact_versions(
) -> None:
    assert (
        ML_MODEL_ARTIFACT_RULE_VERSION
        ==
        "ml_model_artifact_v0.1"
    )


    assert (
        ML_MODEL_ARTIFACT_DATA_PLANE_VERSION
        ==
        "ml_model_artifact_data_plane_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        (
            "=== DATALENS ML MODEL ARTIFACT "
            "DATA PLANE v0.1 ==="
        )
    )

    print()


    test_data_root_is_derived_from_store_scope()

    print(
        "Model Artifact data root is scoped: PASS"
    )


    test_model_binary_round_trip()

    print(
        "Opaque model binary write/read round-trip: PASS"
    )


    test_same_binary_has_same_sha256()

    print(
        "Model binary SHA-256 is deterministic: PASS"
    )


    test_binary_tampering_is_detected()

    print(
        "Model binary tampering is detected: PASS"
    )


    test_binary_size_mismatch_is_detected()

    print(
        "Model binary size mismatch is detected: PASS"
    )


    test_path_escape_is_blocked()

    print(
        "Filesystem path escape is blocked: PASS"
    )


    test_empty_model_binary_is_blocked()

    print(
        "Empty model binary is blocked: PASS"
    )


    test_model_binary_delete_is_idempotent()

    print(
        "Model binary deletion is idempotent: PASS"
    )


    test_valid_model_artifact_record()

    print(
        "Valid server-owned Model Artifact record: PASS"
    )


    test_workflow_provenance_mismatch_is_blocked()

    print(
        "Workflow provenance mismatch is blocked: PASS"
    )


    test_dataset_provenance_mismatch_is_blocked()

    print(
        "Dataset provenance mismatch is blocked: PASS"
    )


    test_non_finite_metric_is_blocked()

    print(
        "Non-finite evaluation metrics are blocked: PASS"
    )


    test_artifact_record_path_escape_is_blocked()

    print(
        "Artifact metadata path escape is blocked: PASS"
    )


    test_model_artifact_versions()

    print(
        "Model Artifact rule versions: PASS"
    )


    print()

    print(
        "ML Model Artifact Data Plane v0.1: PASS"
    )


if __name__ == "__main__":
    main()
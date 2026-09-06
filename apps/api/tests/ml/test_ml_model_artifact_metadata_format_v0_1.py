from __future__ import annotations


import sys


from pathlib import (
    Path,
)


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_artifact_formats import (
    ML_MODEL_ARTIFACT_FORMAT_RULE_VERSION,
)


from app.ml.model_artifacts import (
    ML_MODEL_ARTIFACT_RULE_VERSION,
    MLModelArtifactRecord,
)


# ============================================================
# CONTRACT FIXTURES
# ============================================================


def classical_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:artifact-format",

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
                "prep:artifact-format",

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


def common_fields(
    contract: MLTrainingContract,
) -> dict:

    return {
        "model_id":
            "model:artifact-format",

        "workflow_id":
            contract.workflow_id,

        "dataset_id":
            contract.dataset_id,

        "training_contract":
            contract,

        "metrics": {
            "mae":
                1.0,

            "rmse":
                1.25,

            "r2":
                0.80,
        },

        "train_rows":
            80,

        "test_rows":
            20,

        "created_at_utc":
            "2026-09-06T12:00:00+00:00",

        "model_file_bytes":
            123,

        "model_sha256":
            "a"
            *
            64,
    }


# ============================================================
# LEGACY JOBLIB DEFAULT
# ============================================================


def test_legacy_joblib_metadata_default(
) -> None:

    contract = (
        classical_contract()
    )


    record = (
        MLModelArtifactRecord(
            **common_fields(
                contract
            ),

            model_path=
                "data/model.joblib",
        )
    )


    assert (
        record.serialization_format
        ==
        "joblib"
    )


    assert (
        record.model_path
        ==
        "data/model.joblib"
    )


# ============================================================
# PYTORCH BUNDLE METADATA
# ============================================================


def test_pytorch_bundle_metadata(
) -> None:

    contract = (
        mlp_contract()
    )


    record = (
        MLModelArtifactRecord(
            **common_fields(
                contract
            ),

            serialization_format=
                "pytorch_bundle",

            model_path=
                "data/model.ptbundle",
        )
    )


    assert (
        record.serialization_format
        ==
        "pytorch_bundle"
    )


    assert (
        record.model_path
        ==
        "data/model.ptbundle"
    )


    assert (
        record.training_contract.estimator_key
        ==
        "tabular_mlp_regressor"
    )


# ============================================================
# FORMAT / SUFFIX FAIL-CLOSED
# ============================================================


def expect_validation_error(
    builder,
) -> None:

    try:

        builder()

    except ValidationError:
        return


    raise AssertionError(
        "Expected Pydantic ValidationError."
    )


def test_joblib_with_pytorch_suffix_is_blocked(
) -> None:

    contract = (
        classical_contract()
    )


    def build(
    ) -> None:

        MLModelArtifactRecord(
            **common_fields(
                contract
            ),

            serialization_format=
                "joblib",

            model_path=
                "data/model.ptbundle",
        )


    expect_validation_error(
        build
    )


def test_pytorch_with_joblib_suffix_is_blocked(
) -> None:

    contract = (
        mlp_contract()
    )


    def build(
    ) -> None:

        MLModelArtifactRecord(
            **common_fields(
                contract
            ),

            serialization_format=
                "pytorch_bundle",

            model_path=
                "data/model.joblib",
        )


    expect_validation_error(
        build
    )


def test_unknown_format_is_blocked(
) -> None:

    contract = (
        mlp_contract()
    )


    def build(
    ) -> None:

        MLModelArtifactRecord(
            **common_fields(
                contract
            ),

            serialization_format=
                "unknown",

            model_path=
                "data/model.ptbundle",
        )


    expect_validation_error(
        build
    )


# ============================================================
# PATH AUTHORITY
# ============================================================


def test_path_escape_remains_blocked(
) -> None:

    contract = (
        mlp_contract()
    )


    def build(
    ) -> None:

        MLModelArtifactRecord(
            **common_fields(
                contract
            ),

            serialization_format=
                "pytorch_bundle",

            model_path=
                "../model.ptbundle",
        )


    expect_validation_error(
        build
    )


# ============================================================
# TORCH-FREE METADATA
# ============================================================


def test_artifact_metadata_is_torch_free(
) -> None:

    assert (
        "torch"
        not in
        sys.modules
    )


    root = (
        Path(__file__)
        .parents[
            2
        ]
    )


    for relative_path in (
        "app/ml/model_artifact_formats.py",
        "app/ml/model_artifacts.py",
    ):

        source = (
            root
            /
            relative_path
        ).read_text(
            encoding="utf-8"
        )


        for token in (
            "import torch",
            "from torch",
            "torch.",
        ):

            assert (
                token
                not in
                source
            )


# ============================================================
# RULE VERSIONS
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        ML_MODEL_ARTIFACT_RULE_VERSION
        ==
        "ml_model_artifact_v0.1"
    )


    assert (
        ML_MODEL_ARTIFACT_FORMAT_RULE_VERSION
        ==
        "ml_model_artifact_format_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS MULTI-FORMAT MODEL ARTIFACT METADATA v0.1 ==="
    )

    print()


    test_legacy_joblib_metadata_default()

    print(
        "Legacy joblib metadata default: PASS"
    )


    test_pytorch_bundle_metadata()

    print(
        "PyTorch bundle metadata: PASS"
    )


    test_joblib_with_pytorch_suffix_is_blocked()

    test_pytorch_with_joblib_suffix_is_blocked()

    print(
        "Format / suffix metadata guard: PASS"
    )


    test_unknown_format_is_blocked()

    print(
        "Unknown metadata format fail-closed guard: PASS"
    )


    test_path_escape_remains_blocked()

    print(
        "Metadata path containment guard: PASS"
    )


    test_artifact_metadata_is_torch_free()

    print(
        "Torch-free artifact metadata: PASS"
    )


    test_rule_versions()

    print(
        "Artifact metadata rule versions: PASS"
    )


    print()

    print(
        "PASS - DataLens Multi-Format Model Artifact Metadata v0.1"
    )


if __name__ == "__main__":
    main()

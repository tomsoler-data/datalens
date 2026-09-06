from __future__ import annotations


import tempfile


from pathlib import (
    Path,
)


from app.ml.model_artifact_data_plane import (
    MLModelArtifactDataPlaneError,
    read_ml_model_binary,
    write_ml_model_binary,
)


from app.ml.model_artifact_formats import (
    ML_MODEL_ARTIFACT_FORMAT_RULE_VERSION,
    ml_model_serialization_suffix,
    normalize_ml_model_serialization_format,
)


# ============================================================
# FIXTURE
# ============================================================


MODEL_BYTES = (
    b"datalens-model-artifact-format-boundary-v0.1"
)


def store_path(
    root: str,
) -> Path:

    return (
        Path(
            root
        )
        /
        "model_artifacts.json"
    )


# ============================================================
# LEGACY JOBLIB DEFAULT
# ============================================================


def test_joblib_default_is_preserved(
) -> None:

    with tempfile.TemporaryDirectory() as root:

        path = (
            store_path(
                root
            )
        )


        info = (
            write_ml_model_binary(
                store_path=
                    path,

                model_id=
                    "model:legacy-joblib",

                model_bytes=
                    MODEL_BYTES,
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
                    path,

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
# PYTORCH BUNDLE DATA PLANE
# ============================================================


def test_pytorch_bundle_round_trip(
) -> None:

    with tempfile.TemporaryDirectory() as root:

        path = (
            store_path(
                root
            )
        )


        info = (
            write_ml_model_binary(
                store_path=
                    path,

                model_id=
                    "model:pytorch-bundle",

                model_bytes=
                    MODEL_BYTES,

                serialization_format=
                    "pytorch_bundle",
            )
        )


        assert (
            info[
                "model_path"
            ]
            .endswith(
                ".ptbundle"
            )
        )


        entry = {
            **info,
            "serialization_format":
                "pytorch_bundle",
        }


        restored = (
            read_ml_model_binary(
                store_path=
                    path,

                entry=
                    entry,
            )
        )


        assert (
            restored
            ==
            MODEL_BYTES
        )


# ============================================================
# FORMAT / PATH CONSISTENCY
# ============================================================


def test_format_suffix_mismatch_fails_closed(
) -> None:

    with tempfile.TemporaryDirectory() as root:

        path = (
            store_path(
                root
            )
        )


        info = (
            write_ml_model_binary(
                store_path=
                    path,

                model_id=
                    "model:mismatch",

                model_bytes=
                    MODEL_BYTES,
            )
        )


        malformed_entry = {
            **info,
            "serialization_format":
                "pytorch_bundle",
        }


        try:

            read_ml_model_binary(
                store_path=
                    path,

                entry=
                    malformed_entry,
            )

        except MLModelArtifactDataPlaneError as error:

            assert (
                "suffix"
                in
                str(
                    error
                )
            )

            return


    raise AssertionError(
        (
            "Format/path mismatch "
            "must fail closed."
        )
    )


# ============================================================
# UNKNOWN FORMAT
# ============================================================


def test_unknown_format_is_rejected_before_write(
) -> None:

    with tempfile.TemporaryDirectory() as root:

        path = (
            store_path(
                root
            )
        )


        try:

            write_ml_model_binary(
                store_path=
                    path,

                model_id=
                    "model:unknown-format",

                model_bytes=
                    MODEL_BYTES,

                serialization_format=
                    "unknown",
            )

        except MLModelArtifactDataPlaneError:
            pass

        else:

            raise AssertionError(
                (
                    "Unknown artifact format "
                    "must fail closed."
                )
            )


        data_root = (
            path.parent
            /
            path.stem
            /
            "data"
        )


        if data_root.exists():

            assert (
                list(
                    data_root.iterdir()
                )
                ==
                []
            )


# ============================================================
# FORMAT POLICY
# ============================================================


def test_format_policy(
) -> None:

    assert (
        normalize_ml_model_serialization_format(
            "joblib"
        )
        ==
        "joblib"
    )


    assert (
        normalize_ml_model_serialization_format(
            "pytorch_bundle"
        )
        ==
        "pytorch_bundle"
    )


    assert (
        ml_model_serialization_suffix(
            "joblib"
        )
        ==
        ".joblib"
    )


    assert (
        ml_model_serialization_suffix(
            "pytorch_bundle"
        )
        ==
        ".ptbundle"
    )


# ============================================================
# TORCH-FREE SHARED DATA PLANE
# ============================================================


def test_shared_artifact_boundary_is_torch_free(
) -> None:

    root = (
        Path(__file__)
        .parents[
            2
        ]
    )


    for relative_path in (
        "app/ml/model_artifact_formats.py",
        "app/ml/model_artifact_data_plane.py",
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
# CLASSICAL LOADER REMAINS JOBLIB-ONLY
# ============================================================


def test_classical_loader_remains_joblib_only(
) -> None:

    path = (
        Path(__file__)
        .parents[
            2
        ]
        /
        "app"
        /
        "ml"
        /
        "model_loader.py"
    )


    source = path.read_text(
        encoding="utf-8"
    )


    assert (
        "artifact_before.serialization_format"
        in
        source
    )


    assert (
        '"joblib"'
        in
        source
    )


    assert (
        "joblib.load("
        in
        source
    )


    for token in (
        "torch.load",
        "import torch",
        "from torch",
    ):

        assert (
            token
            not in
            source
        )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

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
        "=== DATALENS MODEL ARTIFACT FORMAT BOUNDARY v0.1 ==="
    )

    print()


    test_joblib_default_is_preserved()

    print(
        "Legacy joblib default: PASS"
    )


    test_pytorch_bundle_round_trip()

    print(
        "PyTorch bundle binary round-trip: PASS"
    )


    test_format_suffix_mismatch_fails_closed()

    print(
        "Format / suffix consistency guard: PASS"
    )


    test_unknown_format_is_rejected_before_write()

    print(
        "Unknown format fail-closed guard: PASS"
    )


    test_format_policy()

    print(
        "Artifact format policy: PASS"
    )


    test_shared_artifact_boundary_is_torch_free()

    print(
        "Torch-free shared artifact data plane: PASS"
    )


    test_classical_loader_remains_joblib_only()

    print(
        "Classical loader isolation: PASS"
    )


    test_rule_version()

    print(
        "Format boundary rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Model Artifact Format Boundary v0.1"
    )


if __name__ == "__main__":
    main()

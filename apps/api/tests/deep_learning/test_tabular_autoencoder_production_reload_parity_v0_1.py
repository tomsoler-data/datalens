from __future__ import annotations


import ast
import inspect


import numpy as np
import pandas as pd
import torch


import app.deep_learning.anomaly_model_lab_executor as executor_module


from app.deep_learning.anomaly_model_lab_executor import (
    execute_tabular_autoencoder,
)


from app.deep_learning.autoencoder_model_loader import (
    LoadedTabularAutoencoderModel,
    load_trusted_tabular_autoencoder_model,
)


from app.deep_learning.autoencoder_threshold import (
    apply_reconstruction_error_threshold,
)


from tests.deep_learning.test_tabular_autoencoder_production_artifact_v0_1 import (
    build_contract,
    build_dataframe,
    patched_production_environment,
)


# ============================================================
# ORIGINAL IN-MEMORY INFERENCE
# ============================================================


def reference_detect_from_core(
    *,
    features: pd.DataFrame,
    core_result,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    transformed = (
        np.asarray(
            core_result
            .preprocessor
            .transform(
                features
            ),
            dtype=np.float32,
        )
    )


    if (
        transformed.ndim
        !=
        2
    ):

        raise AssertionError(
            "Reference preprocessor did not return 2D features."
        )


    if (
        transformed.shape[
            1
        ]
        !=
        core_result.model.input_features
    ):

        raise AssertionError(
            "Reference transformed feature count is invalid."
        )


    if not (
        np.isfinite(
            transformed
        )
        .all()
    ):

        raise AssertionError(
            "Reference preprocessing produced non-finite values."
        )


    tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                transformed,
                dtype=np.float32,
            )
        )
    )


    core_result.model.eval()


    with torch.inference_mode():

        reconstruction = (
            core_result.model(
                tensor
            )
        )


        errors = (
            torch.mean(
                (
                    reconstruction
                    -
                    tensor
                )
                **
                2,
                dim=1,
            )
            .detach()
            .cpu()
            .to(
                dtype=torch.float32
            )
            .contiguous()
        )


    flags = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                errors,

            threshold=
                core_result.threshold,
        )
    )


    return (
        errors
        .numpy()
        .astype(
            np.float64,
            copy=True,
        ),

        flags
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.bool_,
            copy=True,
        ),
    )


# ============================================================
# PRODUCTION -> PERSISTENCE -> TRUSTED RELOAD
# ============================================================


def test_production_reload_preserves_exact_inference(
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


    core_calls = 0

    captured = {}


    def counted_core(
        **kwargs,
    ):

        nonlocal core_calls

        core_calls += 1


        result = (
            original_core(
                **kwargs
            )
        )


        captured[
            "core_result"
        ] = result


        return result


    executor_module.execute_tabular_autoencoder_core = (
        counted_core
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

            production_result = (
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
                core_calls
                ==
                1
            )


            core_result = (
                captured[
                    "core_result"
                ]
            )


            loaded = (
                load_trusted_tabular_autoencoder_model(
                    workflow_id=
                        production_result.workflow_id,

                    model_id=
                        production_result
                        .model_artifact
                        .model_id,
                )
            )


            assert isinstance(
                loaded,
                LoadedTabularAutoencoderModel,
            )


            assert (
                loaded.artifact
                ==
                production_result.model_artifact
            )


            assert (
                loaded.artifact.training_contract
                ==
                contract
            )


            assert (
                loaded.threshold
                ==
                core_result.threshold
            )


            inference_features = (
                dataframe.loc[
                    :,
                    contract.feature_columns,
                ]
                .copy(
                    deep=True
                )
            )


            populations = {
                "single_row":
                    inference_features
                    .iloc[
                        [
                            0
                        ]
                    ]
                    .copy(
                        deep=True
                    ),

                "small_batch":
                    inference_features
                    .iloc[
                        :7
                    ]
                    .copy(
                        deep=True
                    ),

                "full_population":
                    inference_features
                    .copy(
                        deep=True
                    ),
            }


            for (
                population_name,
                population,
            ) in populations.items():

                (
                    expected_errors,
                    expected_flags,
                ) = (
                    reference_detect_from_core(
                        features=
                            population,

                        core_result=
                            core_result,
                    )
                )


                actual = (
                    loaded.detect(
                        population
                    )
                )


                np.testing.assert_array_equal(
                    actual.reconstruction_errors,
                    expected_errors,
                    err_msg=(
                        "Reloaded reconstruction errors "
                        "differ for "
                        f"{population_name}."
                    ),
                )


                np.testing.assert_array_equal(
                    actual.anomaly_flags,
                    expected_flags,
                    err_msg=(
                        "Reloaded anomaly decisions "
                        "differ for "
                        f"{population_name}."
                    ),
                )


                assert (
                    actual.reconstruction_errors.shape
                    ==
                    expected_errors.shape
                )


                assert (
                    actual.anomaly_flags.shape
                    ==
                    expected_flags.shape
                )


            # Trusted reload must not execute the training core.
            assert (
                core_calls
                ==
                1
            )


    finally:

        executor_module.execute_tabular_autoencoder_core = (
            original_core
        )


# ============================================================
# STATIC NO-RETRAINING RELOAD BOUNDARY
# ============================================================


def test_trusted_reload_has_no_training_surface(
) -> None:

    import app.deep_learning.autoencoder_model_loader as loader_module


    source = (
        inspect.getsource(
            loader_module
            .load_trusted_tabular_autoencoder_model
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


    for forbidden in (
        "execute_tabular_autoencoder_core",
        "fit",
        "fit_transform",
        "fit_reconstruction_error_threshold",
        "train_reconstruction_epochs",
        "train_reconstruction_batch",
        "backward",
        "step",
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
            "PRODUCTION RELOAD PARITY v0.1 ==="
        )
    )

    print()


    test_production_reload_preserves_exact_inference()

    print(
        "Production Artifact trusted reload: PASS"
    )

    print(
        "Single-row reconstruction parity: PASS"
    )

    print(
        "Single-row anomaly-decision parity: PASS"
    )

    print(
        "Batch reconstruction parity: PASS"
    )

    print(
        "Batch anomaly-decision parity: PASS"
    )

    print(
        "Full-population reconstruction parity: PASS"
    )

    print(
        "Full-population anomaly-decision parity: PASS"
    )

    print(
        "Frozen TRAIN threshold parity: PASS"
    )

    print(
        "Artifact / Training Contract identity: PASS"
    )


    test_trusted_reload_has_no_training_surface()

    print(
        "No retraining during trusted reload: PASS"
    )


    print()

    print(
        (
            "PASS - DataLens Tabular Autoencoder "
            "Production Reload Parity v0.1"
        )
    )


if __name__ == "__main__":
    main()

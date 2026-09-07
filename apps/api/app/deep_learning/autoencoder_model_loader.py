from __future__ import annotations


from dataclasses import (
    dataclass,
)


import numpy as np
import pandas as pd
import torch


from app.deep_learning.autoencoder_bundle import (
    TabularAutoencoderBundleComponents,
    TabularAutoencoderBundleError,
    deserialize_trusted_tabular_autoencoder_bundle,
)


from app.deep_learning.autoencoder_network import (
    TabularAutoencoder,
)


from app.deep_learning.autoencoder_threshold import (
    ReconstructionErrorThreshold,
    apply_reconstruction_error_threshold,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.model_artifact_store import (
    MLModelArtifactStoreError,
    get_ml_model_artifact,
    load_ml_model_artifact_binary,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.preprocessing import (
    MLPreprocessingRuntimeError,
    validate_ml_feature_frame,
)


# ============================================================
# AUTHORITY
# ============================================================


DL_TRUSTED_AUTOENCODER_MODEL_LOADER_RULE_VERSION = (
    "dl_trusted_autoencoder_model_loader_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class DLTrustedAutoencoderLoaderError(
    RuntimeError
):
    pass


class DLTrustedAutoencoderArtifactError(
    DLTrustedAutoencoderLoaderError
):
    pass


class DLTrustedAutoencoderRaceError(
    DLTrustedAutoencoderLoaderError
):
    pass


class DLTrustedAutoencoderDeserializationError(
    DLTrustedAutoencoderLoaderError
):
    pass


class DLTrustedAutoencoderInferenceError(
    DLTrustedAutoencoderLoaderError
):
    pass


# ============================================================
# INFERENCE RESULT
# ============================================================


@dataclass(
    frozen=True,
)
class TabularAutoencoderInferenceResult:

    reconstruction_errors: np.ndarray

    anomaly_flags: np.ndarray


# ============================================================
# LOADED MODEL
# ============================================================


@dataclass(
    frozen=True,
)
class LoadedTabularAutoencoderModel:
    """
    Trusted in-memory Autoencoder restored from one
    server-owned DataLens Model Artifact.

    Raw bundle bytes are never exposed.
    """

    artifact: MLModelArtifactRecord

    model: TabularAutoencoder

    preprocessor: object

    threshold: ReconstructionErrorThreshold


    def detect(
        self,
        features: pd.DataFrame,
    ) -> TabularAutoencoderInferenceResult:
        """
        Execute trusted CPU anomaly inference using exactly the
        fitted preprocessing state, learned network parameters
        and frozen TRAIN threshold stored in the Artifact.

        No split, fit, training or threshold fitting occurs.
        """

        contract = (
            self
            .artifact
            .training_contract
        )


        if not isinstance(
            contract,
            MLAnomalyTrainingContract,
        ):

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder Artifact "
                        "does not contain an anomaly "
                        "Training Contract."
                    )
                )
            )


        try:

            validated_features = (
                validate_ml_feature_frame(
                    features=
                        features,
                    contract=
                        contract,
                )
            )

        except (
            MLPreprocessingRuntimeError,
            ValueError,
            TypeError,
        ) as error:

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder input "
                        "validation failed."
                    )
                )
            ) from error


        try:

            transformed = (
                self
                .preprocessor
                .transform(
                    validated_features
                )
            )


            transformed_array = (
                np.asarray(
                    transformed,
                    dtype=np.float32,
                )
            )

        except Exception as error:

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder "
                        "preprocessing failed."
                    )
                )
            ) from error


        if (
            transformed_array.ndim
            !=
            2
        ):

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder preprocessor "
                        "returned an invalid feature matrix."
                    )
                )
            )


        if (
            transformed_array.shape[
                0
            ]
            !=
            len(
                validated_features
            )
        ):

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder transformed "
                        "row count is invalid."
                    )
                )
            )


        if (
            transformed_array.shape[
                1
            ]
            !=
            self.model.input_features
        ):

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder transformed "
                        "feature count does not match "
                        "the restored network."
                    )
                )
            )


        if not (
            np.isfinite(
                transformed_array
            )
            .all()
        ):

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder preprocessing "
                        "produced non-finite values."
                    )
                )
            )


        tensor = (
            torch.from_numpy(
                np.ascontiguousarray(
                    transformed_array,
                    dtype=np.float32,
                )
            )
        )


        self.model.eval()


        try:

            with torch.inference_mode():

                reconstruction = (
                    self.model(
                        tensor
                    )
                )


                reconstruction_errors = (
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

        except Exception as error:

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder inference "
                        "failed."
                    )
                )
            ) from error


        if (
            reconstruction_errors.ndim
            !=
            1
            or
            reconstruction_errors.numel()
            !=
            len(
                validated_features
            )
        ):

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder reconstruction "
                        "error shape is invalid."
                    )
                )
            )


        if not bool(
            torch.isfinite(
                reconstruction_errors
            )
            .all()
            .item()
        ):

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder produced "
                        "non-finite reconstruction errors."
                    )
                )
            )


        if bool(
            (
                reconstruction_errors
                <
                0.0
            )
            .any()
            .item()
        ):

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder produced "
                        "negative reconstruction errors."
                    )
                )
            )


        try:

            anomaly_flags = (
                apply_reconstruction_error_threshold(
                    reconstruction_errors=
                        reconstruction_errors,
                    threshold=
                        self.threshold,
                )
            )

        except (
            TypeError,
            ValueError,
        ) as error:

            raise (
                DLTrustedAutoencoderInferenceError(
                    (
                        "Trusted Autoencoder frozen "
                        "threshold application failed."
                    )
                )
            ) from error


        errors_array = (
            reconstruction_errors
            .numpy()
            .astype(
                np.float64,
                copy=True,
            )
        )


        flags_array = (
            anomaly_flags
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.bool_,
                copy=True,
            )
        )


        errors_array.setflags(
            write=False
        )


        flags_array.setflags(
            write=False
        )


        return (
            TabularAutoencoderInferenceResult(
                reconstruction_errors=
                    errors_array,
                anomaly_flags=
                    flags_array,
            )
        )


# ============================================================
# TRUSTED LOAD
# ============================================================


def load_trusted_tabular_autoencoder_model(
    *,
    workflow_id: str,
    model_id: str,
) -> LoadedTabularAutoencoderModel:
    """
    Restore one trusted server-owned Tabular Autoencoder.

    Security boundary:

    - callers provide only workflow_id + model_id;
    - arbitrary filesystem paths and raw bytes are not accepted;
    - outer binary size/SHA validation belongs to the Model
      Artifact Store;
    - metadata is checked before and after binary retrieval;
    - only pytorch_bundle / tabular_autoencoder /
      anomaly_detection is accepted;
    - bundle-internal contract/component integrity is validated
      before learned state is restored;
    - no retraining, preprocessing fit or threshold fit occurs.
    """

    # ========================================================
    # METADATA BEFORE VERIFIED BINARY READ
    # ========================================================


    try:

        artifact_before = (
            get_ml_model_artifact(
                model_id=
                    model_id,
                workflow_id=
                    workflow_id,
            )
        )

    except (
        MLModelArtifactStoreError,
        ValueError,
    ) as error:

        raise (
            DLTrustedAutoencoderArtifactError(
                (
                    "Trusted Autoencoder Model Artifact "
                    "metadata lookup failed."
                )
            )
        ) from error


    if (
        artifact_before.serialization_format
        !=
        "pytorch_bundle"
    ):

        raise (
            DLTrustedAutoencoderArtifactError(
                (
                    "Trusted Autoencoder Artifact uses "
                    "an unsupported serialization format. "
                    "format="
                    f"{artifact_before.serialization_format}"
                )
            )
        )


    contract_before = (
        artifact_before
        .training_contract
    )


    if not isinstance(
        contract_before,
        MLAnomalyTrainingContract,
    ):

        raise (
            DLTrustedAutoencoderArtifactError(
                (
                    "Trusted Autoencoder Artifact does not "
                    "contain an anomaly Training Contract."
                )
            )
        )


    if (
        contract_before.estimator_key
        !=
        "tabular_autoencoder"
    ):

        raise (
            DLTrustedAutoencoderArtifactError(
                (
                    "Trusted Model Artifact is not "
                    "a Tabular Autoencoder Artifact."
                )
            )
        )


    if (
        contract_before.problem_type
        !=
        "anomaly_detection"
    ):

        raise (
            DLTrustedAutoencoderArtifactError(
                (
                    "Trusted Tabular Autoencoder Artifact "
                    "must use anomaly_detection."
                )
            )
        )


    if hasattr(
        contract_before,
        "target_column",
    ):

        raise (
            DLTrustedAutoencoderArtifactError(
                (
                    "Trusted Autoencoder Artifact must "
                    "remain target-free."
                )
            )
        )


    # ========================================================
    # VERIFIED SERVER-OWNED BINARY
    # ========================================================


    try:

        bundle_bytes = (
            load_ml_model_artifact_binary(
                model_id=
                    model_id,
                workflow_id=
                    workflow_id,
            )
        )

    except (
        MLModelArtifactStoreError,
        ValueError,
    ) as error:

        raise (
            DLTrustedAutoencoderArtifactError(
                (
                    "Trusted Autoencoder Artifact binary "
                    "could not be loaded and verified."
                )
            )
        ) from error


    # ========================================================
    # METADATA STABILITY / RACE GUARD
    # ========================================================


    try:

        artifact_after = (
            get_ml_model_artifact(
                model_id=
                    model_id,
                workflow_id=
                    workflow_id,
            )
        )

    except (
        MLModelArtifactStoreError,
        ValueError,
    ) as error:

        raise (
            DLTrustedAutoencoderArtifactError(
                (
                    "Trusted Autoencoder metadata could "
                    "not be revalidated after binary "
                    "verification."
                )
            )
        ) from error


    if (
        artifact_before
        !=
        artifact_after
    ):

        raise (
            DLTrustedAutoencoderRaceError(
                (
                    "Trusted Autoencoder Model Artifact "
                    "metadata changed while the model "
                    "was being restored."
                )
            )
        )


    # ========================================================
    # TRUSTED BUNDLE DESERIALIZATION
    # ========================================================


    try:

        components: TabularAutoencoderBundleComponents = (
            deserialize_trusted_tabular_autoencoder_bundle(
                trusted_bundle_bytes=
                    bundle_bytes,
                training_contract=
                    artifact_after
                    .training_contract,
            )
        )

    except TabularAutoencoderBundleError as error:

        raise (
            DLTrustedAutoencoderDeserializationError(
                (
                    "Trusted Autoencoder Model Artifact "
                    "bundle could not be restored."
                )
            )
        ) from error


    return (
        LoadedTabularAutoencoderModel(
            artifact=
                artifact_after,
            model=
                components.model,
            preprocessor=
                components.preprocessor,
            threshold=
                components.threshold,
        )
    )

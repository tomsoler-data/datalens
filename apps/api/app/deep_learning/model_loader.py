from __future__ import annotations


from dataclasses import (
    dataclass,
)


import numpy as np
import pandas as pd
import torch


from app.deep_learning.model_bundle import (
    TabularMLPBundleComponents,
    TabularMLPBundleError,
    deserialize_trusted_tabular_mlp_bundle,
)


from app.deep_learning.networks import (
    FeedForwardRegressor,
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
# VERSION
# ============================================================


DL_TRUSTED_MODEL_LOADER_RULE_VERSION = (
    "dl_trusted_model_loader_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class DLTrustedModelLoaderError(
    RuntimeError
):
    pass


class DLTrustedModelArtifactError(
    DLTrustedModelLoaderError
):
    pass


class DLTrustedModelRaceError(
    DLTrustedModelLoaderError
):
    pass


class DLTrustedModelDeserializationError(
    DLTrustedModelLoaderError
):
    pass


class DLTrustedModelInferenceError(
    DLTrustedModelLoaderError
):
    pass


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True,
)
class LoadedTabularMLPModel:
    """
    Trusted in-memory Tabular MLP restored from one
    server-owned Model Artifact.

    Raw bundle bytes are never exposed.
    """

    artifact: MLModelArtifactRecord

    model: FeedForwardRegressor

    preprocessor: object


    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """
        Execute trusted CPU inference using the exact fitted
        preprocessing state stored with the network weights.
        """

        try:

            validated_features = (
                validate_ml_feature_frame(
                    features=
                        features,

                    contract=
                        self
                        .artifact
                        .training_contract,
                )
            )

        except (
            MLPreprocessingRuntimeError,
            ValueError,
            TypeError,
        ) as error:

            raise (
                DLTrustedModelInferenceError(
                    (
                        "Trusted Tabular MLP input "
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
                DLTrustedModelInferenceError(
                    (
                        "Trusted Tabular MLP "
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
                DLTrustedModelInferenceError(
                    (
                        "Trusted Tabular MLP "
                        "preprocessor returned an "
                        "invalid feature matrix."
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
                DLTrustedModelInferenceError(
                    (
                        "Trusted Tabular MLP transformed "
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
                DLTrustedModelInferenceError(
                    (
                        "Trusted Tabular MLP "
                        "preprocessing produced "
                        "non-finite values."
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

                predictions = (
                    self.model(
                        tensor
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

        except Exception as error:

            raise (
                DLTrustedModelInferenceError(
                    (
                        "Trusted Tabular MLP "
                        "inference failed."
                    )
                )
            ) from error


        predictions = (
            np.asarray(
                predictions,
                dtype=np.float64,
            )
        )


        if (
            predictions.ndim
            !=
            1
            or
            predictions.shape[
                0
            ]
            !=
            len(
                validated_features
            )
        ):
            raise (
                DLTrustedModelInferenceError(
                    (
                        "Trusted Tabular MLP "
                        "prediction shape is invalid."
                    )
                )
            )


        if not (
            np.isfinite(
                predictions
            )
            .all()
        ):
            raise (
                DLTrustedModelInferenceError(
                    (
                        "Trusted Tabular MLP produced "
                        "non-finite predictions."
                    )
                )
            )


        return predictions


# ============================================================
# TRUSTED LOAD
# ============================================================


def load_trusted_tabular_mlp_model(
    *,
    workflow_id: str,
    model_id: str,
) -> LoadedTabularMLPModel:
    """
    Restore one trusted server-owned Tabular MLP artifact.

    Security boundary:

    - callers provide only server-owned identifiers;
    - filesystem paths and arbitrary bytes are never accepted;
    - outer binary size/SHA verification belongs to the shared
      Model Artifact Store;
    - metadata is checked before and after binary retrieval;
    - only pytorch_bundle / tabular_mlp_regressor is accepted;
    - bundle-internal contract/component integrity is validated
      before learned state is restored.
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
            DLTrustedModelArtifactError(
                (
                    "Trusted Deep Learning model "
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
            DLTrustedModelArtifactError(
                (
                    "Trusted Deep Learning Model Artifact "
                    "uses an unsupported serialization "
                    "format. format="
                    f"{artifact_before.serialization_format}"
                )
            )
        )


    if (
        artifact_before
        .training_contract
        .estimator_key
        !=
        "tabular_mlp_regressor"
    ):
        raise (
            DLTrustedModelArtifactError(
                (
                    "Trusted Deep Learning Model Artifact "
                    "is not a Tabular MLP artifact."
                )
            )
        )


    if (
        artifact_before
        .training_contract
        .problem_type
        !=
        "regression"
    ):
        raise (
            DLTrustedModelArtifactError(
                (
                    "Trusted Tabular MLP Artifact "
                    "must use regression."
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
            DLTrustedModelArtifactError(
                (
                    "Trusted Deep Learning Model "
                    "Artifact binary could not be "
                    "loaded and verified."
                )
            )
        ) from error


    # ========================================================
    # METADATA STABILITY
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
            DLTrustedModelArtifactError(
                (
                    "Trusted Deep Learning metadata "
                    "could not be revalidated after "
                    "binary verification."
                )
            )
        ) from error


    if (
        artifact_before
        !=
        artifact_after
    ):
        raise (
            DLTrustedModelRaceError(
                (
                    "Trusted Deep Learning Model "
                    "Artifact metadata changed while "
                    "the model was being restored."
                )
            )
        )


    # ========================================================
    # TRUSTED BUNDLE DESERIALIZATION
    # ========================================================


    try:

        components: TabularMLPBundleComponents = (
            deserialize_trusted_tabular_mlp_bundle(
                trusted_bundle_bytes=
                    bundle_bytes,

                training_contract=
                    artifact_after
                    .training_contract,
            )
        )

    except TabularMLPBundleError as error:

        raise (
            DLTrustedModelDeserializationError(
                (
                    "Trusted PyTorch Model Artifact "
                    "bundle could not be restored."
                )
            )
        ) from error


    return (
        LoadedTabularMLPModel(
            artifact=
                artifact_after,

            model=
                components.model,

            preprocessor=
                components.preprocessor,
        )
    )

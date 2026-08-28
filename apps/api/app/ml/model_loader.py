from __future__ import annotations

import io

from dataclasses import dataclass
from typing import Any

import joblib

from app.ml.model_artifact_store import (
    MLModelArtifactStoreError,
    get_ml_model_artifact,
    load_ml_model_artifact_binary,
)

from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_LOADER_RULE_VERSION = (
    "ml_model_loader_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLModelLoaderError(
    RuntimeError
):
    pass


class MLModelLoaderArtifactError(
    MLModelLoaderError
):
    pass


class MLModelLoaderRaceError(
    MLModelLoaderError
):
    pass


class MLModelDeserializationError(
    MLModelLoaderError
):
    pass


class MLModelInterfaceError(
    MLModelLoaderError
):
    pass


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True
)
class LoadedMLModel:
    """
    Trusted in-memory model restored from one server-owned
    DataLens Model Artifact.

    The raw serialized bytes and filesystem path are deliberately
    not exposed by this result object.
    """

    artifact: MLModelArtifactRecord
    estimator: Any

    def predict(
        self,
        features: Any,
    ) -> Any:
        predict = getattr(
            self.estimator,
            "predict",
            None,
        )

        if not callable(
            predict
        ):
            raise (
                MLModelInterfaceError(
                    (
                        "Loaded ML estimator no longer exposes "
                        "a callable predict() interface."
                    )
                )
            )

        return predict(
            features
        )


# ============================================================
# TRUSTED LOAD
# ============================================================


def load_trusted_ml_model(
    *,
    workflow_id: str,
    model_id: str,
) -> LoadedMLModel:
    """
    Restore one trusted server-owned ML Model Artifact.

    Security boundary
    -----------------

    joblib/pickle deserialization may execute Python code.

    Therefore this function MUST NOT accept:

    - arbitrary model bytes;
    - arbitrary filesystem paths;
    - client-uploaded serialized objects.

    The only accepted inputs are server-owned identifiers.

    The serialized bytes are obtained exclusively through the
    Model Artifact Store, which validates:

    - model ownership;
    - workflow scope;
    - server-owned metadata;
    - filesystem containment;
    - expected file size;
    - SHA-256 integrity.

    A metadata stability check is also performed before
    deserialization so that a concurrent artifact mutation cannot
    silently change the trusted provenance during restoration.
    """

    # ========================================================
    # METADATA BEFORE VERIFIED READ
    # ========================================================

    try:
        artifact_before = (
            get_ml_model_artifact(
                model_id=model_id,
                workflow_id=workflow_id,
            )
        )

    except (
        MLModelArtifactStoreError,
        ValueError,
    ) as error:
        raise (
            MLModelLoaderArtifactError(
                (
                    "Trusted ML model metadata lookup failed."
                )
            )
        ) from error

    if (
        artifact_before.serialization_format
        !=
        "joblib"
    ):
        raise (
            MLModelLoaderArtifactError(
                (
                    "Trusted ML Model Artifact uses an "
                    "unsupported serialization format. "
                    f"format={artifact_before.serialization_format}"
                )
            )
        )

    # ========================================================
    # VERIFIED SERVER-OWNED BINARY
    # ========================================================

    try:
        model_bytes = (
            load_ml_model_artifact_binary(
                model_id=model_id,
                workflow_id=workflow_id,
            )
        )

    except (
        MLModelArtifactStoreError,
        ValueError,
    ) as error:
        raise (
            MLModelLoaderArtifactError(
                (
                    "Trusted ML Model Artifact binary could "
                    "not be loaded and verified."
                )
            )
        ) from error

    # ========================================================
    # METADATA STABILITY
    # ========================================================

    try:
        artifact_after = (
            get_ml_model_artifact(
                model_id=model_id,
                workflow_id=workflow_id,
            )
        )

    except (
        MLModelArtifactStoreError,
        ValueError,
    ) as error:
        raise (
            MLModelLoaderArtifactError(
                (
                    "Trusted ML model metadata could not be "
                    "revalidated after binary verification."
                )
            )
        ) from error

    if (
        artifact_before
        !=
        artifact_after
    ):
        raise (
            MLModelLoaderRaceError(
                (
                    "Trusted ML Model Artifact metadata changed "
                    "while the model was being restored."
                )
            )
        )

    # ========================================================
    # DESERIALIZATION
    # ========================================================

    try:
        estimator = (
            joblib.load(
                io.BytesIO(
                    model_bytes
                )
            )
        )

    except Exception as error:
        raise (
            MLModelDeserializationError(
                (
                    "Trusted server-owned ML Model Artifact "
                    "could not be deserialized."
                )
            )
        ) from error

    # ========================================================
    # PREDICTOR INTERFACE
    # ========================================================

    predict = getattr(
        estimator,
        "predict",
        None,
    )

    if not callable(
        predict
    ):
        raise (
            MLModelInterfaceError(
                (
                    "Deserialized ML Model Artifact does not "
                    "expose the required predict() interface."
                )
            )
        )

    return (
        LoadedMLModel(
            artifact=artifact_after,
            estimator=estimator,
        )
    )
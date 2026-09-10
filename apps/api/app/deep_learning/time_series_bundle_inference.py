from __future__ import annotations


from dataclasses import (
    dataclass,
)


import numpy as np
import torch


from app.deep_learning.time_series_bundle import (
    DLTimeSeriesNeuralBundleComponents,
)


from app.deep_learning.time_series_scaling import (
    DLTimeSeriesScalingError,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_BUNDLE_INFERENCE_RULE_VERSION = (
    "dl_time_series_bundle_inference_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class DLTimeSeriesBundleInferenceError(
    RuntimeError
):
    pass


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True
)
class DLTimeSeriesBundleInferenceResult:
    """
    Target-blind inference result from one trusted restored
    temporal neural Artifact.

    Targets and evaluation metrics are deliberately absent.
    """

    estimator_key: str

    predictions: np.ndarray

    rule_version: str = (
        DL_TIME_SERIES_BUNDLE_INFERENCE_RULE_VERSION
    )


    @property
    def sample_count(
        self,
    ) -> int:

        return int(
            self.predictions.size
        )


# ============================================================
# INPUT AUTHORITY
# ============================================================


def _validated_inputs(
    *,
    inputs: object,
    lookback: int,
) -> np.ndarray:

    try:

        values = np.asarray(
            inputs,
            dtype=np.float64,
        )

    except Exception as error:

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact inference inputs "
                "could not be converted to float64."
            )
        ) from error


    if values.ndim != 2:

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact inference inputs "
                "must be two-dimensional."
            )
        )


    if values.shape[0] <= 0:

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact inference requires "
                "at least one forecasting sample."
            )
        )


    if values.shape[1] != lookback:

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact inference window width "
                "must match bundle lookback."
            )
        )


    if not np.isfinite(
        values
    ).all():

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact inference inputs "
                "contain non-finite values."
            )
        )


    return np.ascontiguousarray(
        values,
        dtype=np.float64,
    )


# ============================================================
# PUBLIC INFERENCE
# ============================================================


def predict_trusted_time_series_bundle(
    *,
    components: DLTimeSeriesNeuralBundleComponents,
    inputs: object,
) -> DLTimeSeriesBundleInferenceResult:
    """
    Execute target-blind inference from one already trusted and
    reconstructed temporal neural bundle.

    No:

    - training;
    - optimizer;
    - loss;
    - targets;
    - evaluation metrics;
    - parameter updates.

    are involved.
    """

    if not isinstance(
        components,
        DLTimeSeriesNeuralBundleComponents,
    ):

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact inference requires "
                "trusted bundle components."
            )
        )


    lookback = int(
        components.manifest.lookback
    )


    values = _validated_inputs(
        inputs=
            inputs,

        lookback=
            lookback,
    )


    try:

        scaled_inputs = (
            components.standardizer
            .transform(
                values
            )
        )

    except DLTimeSeriesScalingError as error:

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact inference scaling "
                "failed."
            )
        ) from error


    try:

        scaled_float32 = np.asarray(
            scaled_inputs,
            dtype=np.float32,
        )

    except Exception as error:

        raise DLTimeSeriesBundleInferenceError(
            (
                "Scaled temporal Artifact inputs "
                "could not be converted to float32."
            )
        ) from error


    if (
        scaled_float32.ndim
        !=
        2
    ):

        raise DLTimeSeriesBundleInferenceError(
            (
                "Scaled temporal Artifact inputs "
                "must remain two-dimensional."
            )
        )


    if not np.isfinite(
        scaled_float32
    ).all():

        raise DLTimeSeriesBundleInferenceError(
            (
                "Scaled temporal Artifact inputs "
                "contain non-finite values."
            )
        )


    scaled_float32 = (
        np.ascontiguousarray(
            scaled_float32,
            dtype=np.float32,
        )
    )


    model = (
        components.model
    )


    for parameter in (
        model.parameters()
    ):

        if (
            parameter.device.type
            !=
            "cpu"
        ):

            raise DLTimeSeriesBundleInferenceError(
                (
                    "Trusted temporal Artifact inference "
                    "requires CPU-owned restored model."
                )
            )


    model.eval()


    tensor_inputs = torch.from_numpy(
        scaled_float32
    )


    with torch.inference_mode():

        scaled_predictions_tensor = (
            model(
                tensor_inputs
            )
        )


    if not isinstance(
        scaled_predictions_tensor,
        torch.Tensor,
    ):

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact model did not return "
                "a tensor."
            )
        )


    if (
        scaled_predictions_tensor.ndim
        !=
        1
    ):

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact predictions must "
                "be one-dimensional."
            )
        )


    if (
        int(
            scaled_predictions_tensor.shape[
                0
            ]
        )
        !=
        int(
            values.shape[
                0
            ]
        )
    ):

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact prediction count "
                "does not match input sample count."
            )
        )


    scaled_predictions = (
        scaled_predictions_tensor
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=True,
        )
    )


    if not np.isfinite(
        scaled_predictions
    ).all():

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact model produced "
                "non-finite predictions."
            )
        )


    try:

        predictions = (
            components.standardizer
            .inverse_transform(
                scaled_predictions
            )
        )

    except DLTimeSeriesScalingError as error:

        raise DLTimeSeriesBundleInferenceError(
            (
                "Temporal Artifact prediction "
                "inverse-transform failed."
            )
        ) from error


    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    ).copy()


    if (
        predictions.ndim
        !=
        1
    ):

        raise DLTimeSeriesBundleInferenceError(
            (
                "Restored temporal predictions must "
                "be one-dimensional."
            )
        )


    if not np.isfinite(
        predictions
    ).all():

        raise DLTimeSeriesBundleInferenceError(
            (
                "Restored temporal predictions contain "
                "non-finite values."
            )
        )


    predictions.setflags(
        write=False
    )


    return (
        DLTimeSeriesBundleInferenceResult(
            estimator_key=
                components.manifest.estimator_key,

            predictions=
                predictions,
        )
    )

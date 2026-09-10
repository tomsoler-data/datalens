from __future__ import annotations


import hashlib
import io
import json
import math
import zipfile


from dataclasses import (
    dataclass,
)


from typing import (
    Mapping,
)


import torch


from pydantic import (
    ValidationError,
)


from app.deep_learning.networks import (
    FeedForwardRegressor,
)


from app.deep_learning.time_series_bundle_contracts import (
    DLTimeSeriesNeuralBundleManifest,
    DLTimeSeriesStandardizerSnapshot,
)


from app.deep_learning.time_series_lstm_network import (
    TimeSeriesLSTMRegressor,
)


from app.deep_learning.time_series_rnn_network import (
    TimeSeriesRNNRegressor,
)


from app.deep_learning.time_series_scaling import (
    DLTimeSeriesStandardizer,
)


from app.ml.experiment_provenance import (
    ml_model_training_contract_sha256,
)


from app.ml.time_series_model_training_contracts import (
    MLTimeSeriesModelTrainingContract,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_NEURAL_BUNDLE_CODEC_RULE_VERSION = (
    "dl_time_series_neural_bundle_codec_v0.1"
)


# ============================================================
# MEMBERS
# ============================================================


_MANIFEST_MEMBER = (
    "manifest.json"
)


_STATE_DICT_MEMBER = (
    "model_state.pt"
)


_STANDARDIZER_MEMBER = (
    "standardizer.json"
)


_EXPECTED_MEMBERS = {
    _MANIFEST_MEMBER,
    _STATE_DICT_MEMBER,
    _STANDARDIZER_MEMBER,
}


# ============================================================
# SIZE BOUNDARIES
# ============================================================


_MAX_BUNDLE_BYTES = (
    128
    *
    1024
    *
    1024
)


_MAX_MEMBER_BYTES = (
    128
    *
    1024
    *
    1024
)


_MAX_TOTAL_UNCOMPRESSED_BYTES = (
    129
    *
    1024
    *
    1024
)


# ============================================================
# ERRORS
# ============================================================


class DLTimeSeriesNeuralBundleError(
    RuntimeError
):
    pass


class DLTimeSeriesNeuralBundleContractError(
    DLTimeSeriesNeuralBundleError
):
    pass


class DLTimeSeriesNeuralBundleIntegrityError(
    DLTimeSeriesNeuralBundleError
):
    pass


class DLTimeSeriesNeuralBundleDeserializationError(
    DLTimeSeriesNeuralBundleError
):
    pass


# ============================================================
# MODEL TYPE
# ============================================================


DLTimeSeriesNeuralModel = (
    FeedForwardRegressor
    |
    TimeSeriesRNNRegressor
    |
    TimeSeriesLSTMRegressor
)


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True
)
class DLTimeSeriesNeuralBundleComponents:
    """
    Trusted reconstructed temporal neural bundle.

    The model is CPU-owned and in evaluation mode.

    No optimizer, training loop, raw rows, targets or
    predictions are reconstructed.
    """

    manifest: DLTimeSeriesNeuralBundleManifest

    model: DLTimeSeriesNeuralModel

    standardizer: DLTimeSeriesStandardizer


# ============================================================
# HASHING
# ============================================================


def _sha256(
    payload: bytes,
) -> str:

    return (
        hashlib.sha256(
            payload
        )
        .hexdigest()
    )


# ============================================================
# STRICT JSON
# ============================================================


def _reject_duplicate_json_pairs(
    pairs,
):

    result = {}


    for (
        key,
        value,
    ) in pairs:

        if key in result:

            raise ValueError(
                (
                    "Duplicate JSON key is forbidden. "
                    f"key={key!r}"
                )
            )


        result[
            key
        ] = value


    return result


def _reject_json_constant(
    value: str,
):

    raise ValueError(
        (
            "Non-standard JSON numeric constant "
            f"is forbidden: {value!r}"
        )
    )


def _strict_json_loads(
    payload: bytes,
    *,
    label: str,
):

    if not payload:

        raise DLTimeSeriesNeuralBundleDeserializationError(
            f"{label} cannot be empty."
        )


    try:

        text = payload.decode(
            "utf-8"
        )


        value = json.loads(
            text,
            object_pairs_hook=
                _reject_duplicate_json_pairs,
            parse_constant=
                _reject_json_constant,
        )

    except Exception as error:

        raise DLTimeSeriesNeuralBundleDeserializationError(
            f"{label} is not valid strict UTF-8 JSON."
        ) from error


    if not isinstance(
        value,
        dict,
    ):

        raise DLTimeSeriesNeuralBundleDeserializationError(
            f"{label} must contain a JSON object."
        )


    return value


def _canonical_model_bytes(
    model,
) -> bytes:

    payload = (
        model.model_dump(
            mode="json"
        )
    )


    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        )
        .encode(
            "utf-8"
        )
    )


# ============================================================
# TRAINING CONTRACT AUTHORITY
# ============================================================


def _require_training_contract(
    training_contract: object,
) -> tuple[
    MLTimeSeriesModelTrainingContract,
    str,
    str,
    int,
]:

    try:

        contract = (
            MLTimeSeriesModelTrainingContract
            .model_validate(
                training_contract
            )
        )

    except Exception as error:

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Temporal neural bundle requires a valid "
                "forecast Model Training Contract."
            )
        ) from error


    estimator_key = (
        contract.estimator_key
    )


    hyperparameters = (
        contract
        .effective_estimator_hyperparameters
    )


    if (
        getattr(
            hyperparameters,
            "kind",
            None,
        )
        !=
        estimator_key
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Forecast Training Contract estimator "
                "identity is inconsistent."
            )
        )


    if (
        estimator_key
        ==
        "time_series_mlp_regressor"
    ):

        network_class = (
            "FeedForwardRegressor"
        )


        hidden_width = getattr(
            hyperparameters,
            "hidden_features",
            None,
        )

    elif (
        estimator_key
        ==
        "time_series_rnn_regressor"
    ):

        network_class = (
            "TimeSeriesRNNRegressor"
        )


        hidden_width = getattr(
            hyperparameters,
            "hidden_size",
            None,
        )

    elif (
        estimator_key
        ==
        "time_series_lstm_regressor"
    ):

        network_class = (
            "TimeSeriesLSTMRegressor"
        )


        hidden_width = getattr(
            hyperparameters,
            "hidden_size",
            None,
        )

    else:

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Unsupported temporal neural estimator. "
                f"estimator_key={estimator_key!r}"
            )
        )


    if isinstance(
        hidden_width,
        bool,
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            "Temporal neural hidden width is invalid."
        )


    if not isinstance(
        hidden_width,
        int,
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            "Temporal neural hidden width is invalid."
        )


    if hidden_width <= 0:

        raise DLTimeSeriesNeuralBundleContractError(
            "Temporal neural hidden width must be positive."
        )


    if contract.lookback <= 0:

        raise DLTimeSeriesNeuralBundleContractError(
            "Temporal neural lookback must be positive."
        )


    return (
        contract,
        estimator_key,
        network_class,
        hidden_width,
    )


# ============================================================
# MODEL STRUCTURE AUTHORITY
# ============================================================


def _validate_model_structure(
    *,
    model: object,
    contract: MLTimeSeriesModelTrainingContract,
    estimator_key: str,
    hidden_width: int,
) -> DLTimeSeriesNeuralModel:

    lookback = int(
        contract.lookback
    )


    if (
        estimator_key
        ==
        "time_series_mlp_regressor"
    ):

        if not isinstance(
            model,
            FeedForwardRegressor,
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal MLP bundle requires "
                    "FeedForwardRegressor."
                )
            )


        if (
            int(
                model.input_features
            )
            !=
            lookback
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal MLP input_features must "
                    "match forecasting lookback."
                )
            )


        if (
            int(
                model.hidden_features
            )
            !=
            hidden_width
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal MLP hidden_features must "
                    "match the Training Contract."
                )
            )


        return model


    if (
        estimator_key
        ==
        "time_series_rnn_regressor"
    ):

        if not isinstance(
            model,
            TimeSeriesRNNRegressor,
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal RNN bundle requires "
                    "TimeSeriesRNNRegressor."
                )
            )


        if (
            int(
                model.lookback
            )
            !=
            lookback
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal RNN lookback must match "
                    "the Training Contract."
                )
            )


        if (
            int(
                model.hidden_size
            )
            !=
            hidden_width
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal RNN hidden_size must match "
                    "the Training Contract."
                )
            )


        return model


    if (
        estimator_key
        ==
        "time_series_lstm_regressor"
    ):

        if not isinstance(
            model,
            TimeSeriesLSTMRegressor,
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal LSTM bundle requires "
                    "TimeSeriesLSTMRegressor."
                )
            )


        if (
            int(
                model.lookback
            )
            !=
            lookback
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal LSTM lookback must match "
                    "the Training Contract."
                )
            )


        if (
            int(
                model.hidden_size
            )
            !=
            hidden_width
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal LSTM hidden_size must match "
                    "the Training Contract."
                )
            )


        return model


    raise DLTimeSeriesNeuralBundleContractError(
        "Unsupported temporal neural estimator."
    )


# ============================================================
# STANDARDIZER AUTHORITY
# ============================================================


def _standardizer_snapshot(
    standardizer: object,
) -> DLTimeSeriesStandardizerSnapshot:

    if not isinstance(
        standardizer,
        DLTimeSeriesStandardizer,
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Temporal neural bundle requires "
                "DLTimeSeriesStandardizer."
            )
        )


    if not math.isfinite(
        float(
            standardizer.mean
        )
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            "Temporal standardizer mean must be finite."
        )


    if not math.isfinite(
        float(
            standardizer.standard_deviation
        )
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Temporal standardizer standard deviation "
                "must be finite."
            )
        )


    if (
        float(
            standardizer.standard_deviation
        )
        <=
        0.0
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Temporal standardizer standard deviation "
                "must be positive."
            )
        )


    if isinstance(
        standardizer.fitted_observation_count,
        bool,
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Temporal standardizer fitted observation "
                "count is invalid."
            )
        )


    if not isinstance(
        standardizer.fitted_observation_count,
        int,
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Temporal standardizer fitted observation "
                "count is invalid."
            )
        )


    if (
        standardizer.fitted_observation_count
        <
        2
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Temporal standardizer requires at least "
                "two fitted observations."
            )
        )


    if (
        standardizer.rule_version
        !=
        "dl_time_series_scaling_v0.1"
    ):

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Unsupported temporal standardizer "
                "rule version."
            )
        )


    try:

        return (
            DLTimeSeriesStandardizerSnapshot(
                mean=
                    float(
                        standardizer.mean
                    ),

                standard_deviation=
                    float(
                        standardizer
                        .standard_deviation
                    ),

                fitted_observation_count=
                    int(
                        standardizer
                        .fitted_observation_count
                    ),

                scaling_rule_version=
                    standardizer.rule_version,
            )
        )

    except ValidationError as error:

        raise DLTimeSeriesNeuralBundleContractError(
            (
                "Temporal standardizer state does not "
                "satisfy persistence contract."
            )
        ) from error


def _restore_standardizer(
    snapshot: DLTimeSeriesStandardizerSnapshot,
) -> DLTimeSeriesStandardizer:

    return (
        DLTimeSeriesStandardizer(
            mean=
                float(
                    snapshot.mean
                ),

            standard_deviation=
                float(
                    snapshot.standard_deviation
                ),

            fitted_observation_count=
                int(
                    snapshot.fitted_observation_count
                ),

            rule_version=
                snapshot.scaling_rule_version,
        )
    )


# ============================================================
# STATE DICT VALIDATION
# ============================================================


def _validate_model_state_finite(
    state_dict: Mapping[
        str,
        object,
    ],
) -> None:

    if not state_dict:

        raise DLTimeSeriesNeuralBundleContractError(
            "Temporal neural state_dict cannot be empty."
        )


    for (
        name,
        value,
    ) in state_dict.items():

        if not isinstance(
            name,
            str,
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal neural state_dict keys "
                    "must be strings."
                )
            )


        if not isinstance(
            value,
            torch.Tensor,
        ):

            raise DLTimeSeriesNeuralBundleContractError(
                (
                    "Temporal neural state_dict values "
                    "must be tensors."
                )
            )


        if (
            value.is_floating_point()
            or
            value.is_complex()
        ):

            if not bool(
                torch.isfinite(
                    value
                )
                .all()
                .item()
            ):

                raise DLTimeSeriesNeuralBundleContractError(
                    (
                        "Temporal neural state_dict "
                        "contains non-finite values."
                    )
                )


# ============================================================
# ZIP WRITER
# ============================================================


def _write_zip_member(
    archive: zipfile.ZipFile,
    *,
    name: str,
    payload: bytes,
) -> None:

    info = zipfile.ZipInfo(
        filename=
            name,

        date_time=(
            1980,
            1,
            1,
            0,
            0,
            0,
        ),
    )


    info.compress_type = (
        zipfile.ZIP_DEFLATED
    )


    info.external_attr = (
        0o600
        <<
        16
    )


    archive.writestr(
        info,
        payload,
    )


# ============================================================
# MODEL CONSTRUCTION
# ============================================================


def _build_model_from_contract(
    *,
    contract: MLTimeSeriesModelTrainingContract,
    estimator_key: str,
    hidden_width: int,
) -> DLTimeSeriesNeuralModel:

    lookback = int(
        contract.lookback
    )


    if (
        estimator_key
        ==
        "time_series_mlp_regressor"
    ):

        return (
            FeedForwardRegressor(
                input_features=
                    lookback,

                hidden_features=
                    hidden_width,
            )
        )


    if (
        estimator_key
        ==
        "time_series_rnn_regressor"
    ):

        return (
            TimeSeriesRNNRegressor(
                lookback=
                    lookback,

                hidden_size=
                    hidden_width,
            )
        )


    if (
        estimator_key
        ==
        "time_series_lstm_regressor"
    ):

        return (
            TimeSeriesLSTMRegressor(
                lookback=
                    lookback,

                hidden_size=
                    hidden_width,
            )
        )


    raise DLTimeSeriesNeuralBundleContractError(
        "Unsupported temporal neural estimator."
    )


# ============================================================
# SERIALIZE
# ============================================================


def serialize_time_series_neural_bundle(
    *,
    model: DLTimeSeriesNeuralModel,
    standardizer: DLTimeSeriesStandardizer,
    training_contract: MLTimeSeriesModelTrainingContract,
) -> bytes:
    """
    Serialize one DataLens-owned temporal neural model.

    The binary contains only:

    - a privacy-minimal structural manifest;
    - CPU state_dict tensors;
    - JSON-safe TRAIN-only standardizer state.

    The Training Contract itself remains outside the binary and
    is bound by its canonical SHA-256 fingerprint.
    """

    (
        contract,
        estimator_key,
        network_class,
        hidden_width,
    ) = _require_training_contract(
        training_contract
    )


    validated_model = (
        _validate_model_structure(
            model=
                model,

            contract=
                contract,

            estimator_key=
                estimator_key,

            hidden_width=
                hidden_width,
        )
    )


    snapshot = (
        _standardizer_snapshot(
            standardizer
        )
    )


    cpu_state_dict = {
        name:
            value
            .detach()
            .cpu()
            .clone()

        for (
            name,
            value,
        ) in validated_model.state_dict().items()
    }


    _validate_model_state_finite(
        cpu_state_dict
    )


    state_buffer = io.BytesIO()


    try:

        torch.save(
            cpu_state_dict,
            state_buffer,
        )

    except Exception as error:

        raise DLTimeSeriesNeuralBundleError(
            (
                "Temporal neural state_dict "
                "serialization failed."
            )
        ) from error


    state_bytes = (
        state_buffer
        .getvalue()
    )


    standardizer_bytes = (
        _canonical_model_bytes(
            snapshot
        )
    )


    if not state_bytes:

        raise DLTimeSeriesNeuralBundleError(
            "Serialized temporal state_dict is empty."
        )


    if not standardizer_bytes:

        raise DLTimeSeriesNeuralBundleError(
            "Serialized temporal standardizer is empty."
        )


    manifest = (
        DLTimeSeriesNeuralBundleManifest(
            estimator_key=
                estimator_key,

            network_class=
                network_class,

            lookback=
                int(
                    contract.lookback
                ),

            hidden_width=
                hidden_width,

            training_contract_sha256=
                ml_model_training_contract_sha256(
                    contract
                ),

            state_dict_sha256=
                _sha256(
                    state_bytes
                ),

            standardizer_sha256=
                _sha256(
                    standardizer_bytes
                ),
        )
    )


    manifest_bytes = (
        _canonical_model_bytes(
            manifest
        )
    )


    output = io.BytesIO()


    try:

        with zipfile.ZipFile(
            output,
            mode="w",
        ) as archive:

            _write_zip_member(
                archive,
                name=
                    _MANIFEST_MEMBER,
                payload=
                    manifest_bytes,
            )


            _write_zip_member(
                archive,
                name=
                    _STATE_DICT_MEMBER,
                payload=
                    state_bytes,
            )


            _write_zip_member(
                archive,
                name=
                    _STANDARDIZER_MEMBER,
                payload=
                    standardizer_bytes,
            )

    except Exception as error:

        raise DLTimeSeriesNeuralBundleError(
            "Temporal neural bundle creation failed."
        ) from error


    bundle_bytes = (
        output.getvalue()
    )


    if not bundle_bytes:

        raise DLTimeSeriesNeuralBundleError(
            "Temporal neural bundle cannot be empty."
        )


    if len(
        bundle_bytes
    ) > _MAX_BUNDLE_BYTES:

        raise DLTimeSeriesNeuralBundleError(
            (
                "Temporal neural bundle exceeds "
                "the allowed size boundary."
            )
        )


    return bundle_bytes


# ============================================================
# ZIP READER
# ============================================================


def _read_bundle_members(
    trusted_bundle_bytes: bytes,
) -> dict[
    str,
    bytes,
]:

    if not isinstance(
        trusted_bundle_bytes,
        bytes,
    ):

        raise DLTimeSeriesNeuralBundleDeserializationError(
            "Trusted temporal bundle must be bytes."
        )


    if not trusted_bundle_bytes:

        raise DLTimeSeriesNeuralBundleDeserializationError(
            "Trusted temporal bundle cannot be empty."
        )


    if len(
        trusted_bundle_bytes
    ) > _MAX_BUNDLE_BYTES:

        raise DLTimeSeriesNeuralBundleDeserializationError(
            (
                "Trusted temporal bundle exceeds "
                "the allowed size boundary."
            )
        )


    try:

        archive = zipfile.ZipFile(
            io.BytesIO(
                trusted_bundle_bytes
            ),
            mode="r",
        )

    except Exception as error:

        raise DLTimeSeriesNeuralBundleDeserializationError(
            "Trusted temporal bundle is not a valid ZIP archive."
        ) from error


    with archive:

        infos = (
            archive.infolist()
        )


        names = tuple(
            info.filename
            for info
            in infos
        )


        if (
            len(
                names
            )
            !=
            len(
                set(
                    names
                )
            )
        ):

            raise DLTimeSeriesNeuralBundleIntegrityError(
                (
                    "Trusted temporal bundle contains "
                    "duplicate ZIP members."
                )
            )


        if set(
            names
        ) != _EXPECTED_MEMBERS:

            raise DLTimeSeriesNeuralBundleIntegrityError(
                (
                    "Trusted temporal bundle member set "
                    "does not match DataLens authority."
                )
            )


        total_uncompressed = 0


        for info in infos:

            if info.is_dir():

                raise DLTimeSeriesNeuralBundleIntegrityError(
                    (
                        "Trusted temporal bundle cannot "
                        "contain directories."
                    )
                )


            if (
                info.flag_bits
                &
                0x1
            ):

                raise DLTimeSeriesNeuralBundleIntegrityError(
                    (
                        "Encrypted ZIP members are "
                        "not supported."
                    )
                )


            if (
                info.file_size
                <=
                0
            ):

                raise DLTimeSeriesNeuralBundleIntegrityError(
                    (
                        "Trusted temporal bundle contains "
                        "an empty member."
                    )
                )


            if (
                info.file_size
                >
                _MAX_MEMBER_BYTES
            ):

                raise DLTimeSeriesNeuralBundleIntegrityError(
                    (
                        "Trusted temporal bundle member "
                        "exceeds size boundary."
                    )
                )


            total_uncompressed += int(
                info.file_size
            )


        if (
            total_uncompressed
            >
            _MAX_TOTAL_UNCOMPRESSED_BYTES
        ):

            raise DLTimeSeriesNeuralBundleIntegrityError(
                (
                    "Trusted temporal bundle uncompressed "
                    "size exceeds boundary."
                )
            )


        try:

            return {
                name:
                    archive.read(
                        name
                    )

                for name
                in names
            }

        except Exception as error:

            raise DLTimeSeriesNeuralBundleDeserializationError(
                (
                    "Trusted temporal bundle members "
                    "could not be read."
                )
            ) from error


# ============================================================
# TRUSTED DESERIALIZATION
# ============================================================


def deserialize_trusted_time_series_neural_bundle(
    *,
    trusted_bundle_bytes: bytes,
    training_contract: MLTimeSeriesModelTrainingContract,
) -> DLTimeSeriesNeuralBundleComponents:
    """
    Decode one trusted DataLens-owned temporal neural bundle.

    IMPORTANT:

    Callers must only supply bytes restored from DataLens-owned
    Artifact storage after the outer Artifact SHA-256 has been
    verified.

    Internal PyTorch state restoration uses weights_only=True.
    """

    (
        contract,
        estimator_key,
        network_class,
        hidden_width,
    ) = _require_training_contract(
        training_contract
    )


    members = (
        _read_bundle_members(
            trusted_bundle_bytes
        )
    )


    manifest_payload = (
        _strict_json_loads(
            members[
                _MANIFEST_MEMBER
            ],
            label=
                "Temporal bundle manifest",
        )
    )


    try:

        manifest = (
            DLTimeSeriesNeuralBundleManifest
            .model_validate(
                manifest_payload
            )
        )

    except ValidationError as error:

        raise DLTimeSeriesNeuralBundleDeserializationError(
            (
                "Temporal bundle manifest violates "
                "its persistence contract."
            )
        ) from error


    expected_contract_sha = (
        ml_model_training_contract_sha256(
            contract
        )
    )


    if (
        manifest.training_contract_sha256
        !=
        expected_contract_sha
    ):

        raise DLTimeSeriesNeuralBundleIntegrityError(
            (
                "Temporal bundle Training Contract "
                "fingerprint mismatch."
            )
        )


    if (
        manifest.estimator_key
        !=
        estimator_key
    ):

        raise DLTimeSeriesNeuralBundleIntegrityError(
            (
                "Temporal bundle estimator identity "
                "does not match Training Contract."
            )
        )


    if (
        manifest.network_class
        !=
        network_class
    ):

        raise DLTimeSeriesNeuralBundleIntegrityError(
            (
                "Temporal bundle network identity "
                "does not match Training Contract."
            )
        )


    if (
        manifest.lookback
        !=
        int(
            contract.lookback
        )
    ):

        raise DLTimeSeriesNeuralBundleIntegrityError(
            (
                "Temporal bundle lookback does not "
                "match Training Contract."
            )
        )


    if (
        manifest.hidden_width
        !=
        hidden_width
    ):

        raise DLTimeSeriesNeuralBundleIntegrityError(
            (
                "Temporal bundle hidden width does not "
                "match Training Contract."
            )
        )


    state_bytes = members[
        _STATE_DICT_MEMBER
    ]


    standardizer_bytes = members[
        _STANDARDIZER_MEMBER
    ]


    if (
        _sha256(
            state_bytes
        )
        !=
        manifest.state_dict_sha256
    ):

        raise DLTimeSeriesNeuralBundleIntegrityError(
            "Temporal bundle state_dict SHA-256 mismatch."
        )


    if (
        _sha256(
            standardizer_bytes
        )
        !=
        manifest.standardizer_sha256
    ):

        raise DLTimeSeriesNeuralBundleIntegrityError(
            "Temporal bundle standardizer SHA-256 mismatch."
        )


    standardizer_payload = (
        _strict_json_loads(
            standardizer_bytes,
            label=
                "Temporal bundle standardizer",
        )
    )


    try:

        standardizer_snapshot = (
            DLTimeSeriesStandardizerSnapshot
            .model_validate(
                standardizer_payload
            )
        )

    except ValidationError as error:

        raise DLTimeSeriesNeuralBundleDeserializationError(
            (
                "Temporal bundle standardizer violates "
                "its persistence contract."
            )
        ) from error


    standardizer = (
        _restore_standardizer(
            standardizer_snapshot
        )
    )


    # ========================================================
    # STATE DICT
    #
    # weights_only=True prevents generic Python object
    # reconstruction by the PyTorch loader.
    # ========================================================

    try:

        loaded_state = torch.load(
            io.BytesIO(
                state_bytes
            ),
            map_location=
                "cpu",
            weights_only=
                True,
        )

    except Exception as error:

        raise DLTimeSeriesNeuralBundleDeserializationError(
            (
                "Temporal neural state_dict "
                "deserialization failed."
            )
        ) from error


    if not isinstance(
        loaded_state,
        Mapping,
    ):

        raise DLTimeSeriesNeuralBundleDeserializationError(
            (
                "Temporal bundle state payload "
                "must be a state_dict mapping."
            )
        )


    _validate_model_state_finite(
        loaded_state
    )


    model = (
        _build_model_from_contract(
            contract=
                contract,

            estimator_key=
                estimator_key,

            hidden_width=
                hidden_width,
        )
    )


    try:

        model.load_state_dict(
            loaded_state,
            strict=True,
        )

    except Exception as error:

        raise DLTimeSeriesNeuralBundleIntegrityError(
            (
                "Temporal bundle state_dict does not "
                "match expected network structure."
            )
        ) from error


    model.eval()


    return (
        DLTimeSeriesNeuralBundleComponents(
            manifest=
                manifest,

            model=
                model,

            standardizer=
                standardizer,
        )
    )

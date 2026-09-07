from __future__ import annotations


import hashlib
import io
import json
import math
import zipfile


from collections.abc import (
    Mapping,
)


from dataclasses import (
    dataclass,
)


import joblib
import torch


from pydantic import (
    ValidationError,
)


from sklearn.compose import (
    ColumnTransformer,
)


from sklearn.utils.validation import (
    check_is_fitted,
)


from app.deep_learning.autoencoder_bundle_contracts import (
    TabularAutoencoderBundleManifest,
)


from app.deep_learning.autoencoder_network import (
    TabularAutoencoder,
)


from app.deep_learning.autoencoder_threshold import (
    AUTOENCODER_THRESHOLD_COMPARISON,
    AUTOENCODER_THRESHOLD_METHOD,
    AUTOENCODER_THRESHOLD_RULE_VERSION,
    ReconstructionErrorThreshold,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.experiment_provenance import (
    ml_model_training_contract_sha256,
)


# ============================================================
# AUTHORITY
# ============================================================


DL_TABULAR_AUTOENCODER_BUNDLE_CODEC_RULE_VERSION = (
    "dl_tabular_autoencoder_bundle_codec_v0.1"
)


_MANIFEST_MEMBER = (
    "manifest.json"
)


_STATE_DICT_MEMBER = (
    "model_state.pt"
)


_PREPROCESSOR_MEMBER = (
    "preprocessor.joblib"
)


_EXPECTED_MEMBERS = {
    _MANIFEST_MEMBER,
    _STATE_DICT_MEMBER,
    _PREPROCESSOR_MEMBER,
}


_MAX_BUNDLE_BYTES = (
    64
    *
    1024
    *
    1024
)


# ============================================================
# ERRORS
# ============================================================


class TabularAutoencoderBundleError(
    RuntimeError
):
    pass


class TabularAutoencoderBundleContractError(
    TabularAutoencoderBundleError
):
    pass


class TabularAutoencoderBundleIntegrityError(
    TabularAutoencoderBundleError
):
    pass


class TabularAutoencoderBundleDeserializationError(
    TabularAutoencoderBundleError
):
    pass


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True,
)
class TabularAutoencoderBundleComponents:

    manifest: TabularAutoencoderBundleManifest

    model: TabularAutoencoder

    preprocessor: ColumnTransformer

    threshold: ReconstructionErrorThreshold


# ============================================================
# HELPERS
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


def _require_autoencoder_contract(
    training_contract: MLAnomalyTrainingContract,
) -> tuple[
    MLAnomalyTrainingContract,
    int,
    int,
]:

    try:

        contract = (
            MLAnomalyTrainingContract
            .model_validate(
                training_contract
            )
        )

    except ValidationError as error:

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Autoencoder bundle requires a valid "
                    "anomaly Training Contract."
                )
            )
        ) from error


    if (
        contract.problem_type
        !=
        "anomaly_detection"
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Autoencoder bundle requires "
                    "problem_type=anomaly_detection."
                )
            )
        )


    if (
        contract.estimator_key
        !=
        "tabular_autoencoder"
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Autoencoder bundle requires "
                    "estimator_key=tabular_autoencoder."
                )
            )
        )


    hidden_features = int(
        contract
        .estimator_hyperparameters
        .hidden_features
    )


    latent_features = int(
        contract
        .estimator_hyperparameters
        .latent_features
    )


    return (
        contract,
        hidden_features,
        latent_features,
    )


def _validate_threshold(
    *,
    threshold: object,
    contract: MLAnomalyTrainingContract,
) -> ReconstructionErrorThreshold:

    if not isinstance(
        threshold,
        ReconstructionErrorThreshold,
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Autoencoder bundle requires a frozen "
                    "ReconstructionErrorThreshold."
                )
            )
        )


    if (
        not math.isfinite(
            float(
                threshold.threshold
            )
        )
        or
        float(
            threshold.threshold
        )
        <
        0.0
    ):

        raise (
            TabularAutoencoderBundleContractError(
                "Anomaly threshold value is invalid."
            )
        )


    if (
        not math.isfinite(
            float(
                threshold.quantile
            )
        )
        or
        not (
            0.0
            <
            float(
                threshold.quantile
            )
            <
            1.0
        )
    ):

        raise (
            TabularAutoencoderBundleContractError(
                "Anomaly threshold quantile is invalid."
            )
        )


    if (
        float(
            threshold.quantile
        )
        !=
        float(
            contract.threshold_quantile
        )
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Frozen anomaly threshold quantile "
                    "does not match the Training Contract."
                )
            )
        )


    if (
        isinstance(
            threshold.train_rows,
            bool,
        )
        or
        not isinstance(
            threshold.train_rows,
            int,
        )
        or
        threshold.train_rows
        <
        2
    ):

        raise (
            TabularAutoencoderBundleContractError(
                "Anomaly threshold train_rows is invalid."
            )
        )


    if (
        threshold.method
        !=
        AUTOENCODER_THRESHOLD_METHOD
    ):

        raise (
            TabularAutoencoderBundleContractError(
                "Anomaly threshold method is invalid."
            )
        )


    if (
        threshold.comparison_operator
        !=
        AUTOENCODER_THRESHOLD_COMPARISON
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Anomaly threshold comparison "
                    "operator is invalid."
                )
            )
        )


    return threshold


def _validate_preprocessor(
    preprocessor: object,
) -> ColumnTransformer:

    if not isinstance(
        preprocessor,
        ColumnTransformer,
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Autoencoder bundle requires "
                    "a fitted ColumnTransformer."
                )
            )
        )


    try:

        check_is_fitted(
            preprocessor
        )

    except Exception as error:

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Autoencoder bundle requires "
                    "a fitted preprocessor."
                )
            )
        ) from error


    transform = getattr(
        preprocessor,
        "transform",
        None,
    )


    if not callable(
        transform
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Fitted Autoencoder preprocessor "
                    "does not expose transform()."
                )
            )
        )


    return preprocessor


def _preprocessor_output_features(
    preprocessor: ColumnTransformer,
) -> int:

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        count = int(
            len(
                feature_names
            )
        )

    except Exception as error:

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Fitted Autoencoder preprocessor "
                    "output feature count could not "
                    "be resolved."
                )
            )
        ) from error


    if count <= 0:

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Fitted Autoencoder preprocessor "
                    "must produce at least one feature."
                )
            )
        )


    return count


def _validate_model_state_finite(
    state_dict: Mapping[
        str,
        object,
    ],
) -> None:

    if not state_dict:

        raise (
            TabularAutoencoderBundleContractError(
                "Autoencoder state_dict cannot be empty."
            )
        )


    for name, value in state_dict.items():

        if (
            not isinstance(
                name,
                str,
            )
            or
            not name
        ):

            raise (
                TabularAutoencoderBundleContractError(
                    (
                        "Autoencoder state_dict contains "
                        "an invalid parameter name."
                    )
                )
            )


        if not isinstance(
            value,
            torch.Tensor,
        ):

            raise (
                TabularAutoencoderBundleContractError(
                    (
                        "Autoencoder state_dict values "
                        "must be tensors."
                    )
                )
            )


        if value.numel() <= 0:

            raise (
                TabularAutoencoderBundleContractError(
                    (
                        "Autoencoder state_dict contains "
                        "an empty tensor."
                    )
                )
            )


        if not bool(
            torch.isfinite(
                value
            )
            .all()
            .item()
        ):

            raise (
                TabularAutoencoderBundleContractError(
                    (
                        "Autoencoder state_dict contains "
                        "non-finite parameters."
                    )
                )
            )


def _canonical_manifest_bytes(
    manifest: TabularAutoencoderBundleManifest,
) -> bytes:

    payload = (
        manifest.model_dump(
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
        )
        .encode(
            "utf-8"
        )
    )


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


    info.create_system = 3


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
# SERIALIZE
# ============================================================


def serialize_tabular_autoencoder_bundle(
    *,
    model: TabularAutoencoder,
    preprocessor: ColumnTransformer,
    training_contract: MLAnomalyTrainingContract,
    threshold: ReconstructionErrorThreshold,
) -> bytes:

    (
        contract,
        contract_hidden_features,
        contract_latent_features,
    ) = (
        _require_autoencoder_contract(
            training_contract
        )
    )


    if not isinstance(
        model,
        TabularAutoencoder,
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Autoencoder bundle requires "
                    "TabularAutoencoder."
                )
            )
        )


    fitted_preprocessor = (
        _validate_preprocessor(
            preprocessor
        )
    )


    output_features = (
        _preprocessor_output_features(
            fitted_preprocessor
        )
    )


    if (
        int(
            model.input_features
        )
        !=
        output_features
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Model input_features do not match "
                    "fitted preprocessor output."
                )
            )
        )


    if (
        int(
            model.hidden_features
        )
        !=
        contract_hidden_features
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Model hidden_features do not match "
                    "the Training Contract."
                )
            )
        )


    if (
        int(
            model.latent_features
        )
        !=
        contract_latent_features
    ):

        raise (
            TabularAutoencoderBundleContractError(
                (
                    "Model latent_features do not match "
                    "the Training Contract."
                )
            )
        )


    frozen_threshold = (
        _validate_threshold(
            threshold=
                threshold,
            contract=
                contract,
        )
    )


    cpu_state_dict = {
        name:
            value
            .detach()
            .cpu()
            .clone()

        for name, value
        in model.state_dict().items()
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

        raise (
            TabularAutoencoderBundleError(
                (
                    "Autoencoder state_dict "
                    "serialization failed."
                )
            )
        ) from error


    state_bytes = (
        state_buffer
        .getvalue()
    )


    preprocessor_buffer = (
        io.BytesIO()
    )


    try:

        joblib.dump(
            fitted_preprocessor,
            preprocessor_buffer,
        )

    except Exception as error:

        raise (
            TabularAutoencoderBundleError(
                (
                    "Autoencoder fitted preprocessor "
                    "serialization failed."
                )
            )
        ) from error


    preprocessor_bytes = (
        preprocessor_buffer
        .getvalue()
    )


    if (
        not state_bytes
        or
        not preprocessor_bytes
    ):

        raise (
            TabularAutoencoderBundleError(
                (
                    "Autoencoder bundle components "
                    "cannot be empty."
                )
            )
        )


    manifest = (
        TabularAutoencoderBundleManifest(
            estimator_key=
                "tabular_autoencoder",

            network_class=
                "TabularAutoencoder",

            state_dict_format=
                "torch_state_dict",

            preprocessor_format=
                "joblib",

            input_features=
                int(
                    model.input_features
                ),

            hidden_features=
                int(
                    model.hidden_features
                ),

            latent_features=
                int(
                    model.latent_features
                ),

            training_contract_sha256=
                ml_model_training_contract_sha256(
                    contract
                ),

            threshold_rule_version=
                AUTOENCODER_THRESHOLD_RULE_VERSION,

            threshold_quantile=
                float(
                    frozen_threshold.quantile
                ),

            anomaly_threshold=
                float(
                    frozen_threshold.threshold
                ),

            threshold_train_rows=
                int(
                    frozen_threshold.train_rows
                ),

            threshold_method=
                frozen_threshold.method,

            threshold_comparison_operator=
                frozen_threshold
                .comparison_operator,

            state_dict_sha256=
                _sha256(
                    state_bytes
                ),

            preprocessor_sha256=
                _sha256(
                    preprocessor_bytes
                ),
        )
    )


    manifest_bytes = (
        _canonical_manifest_bytes(
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
                    _PREPROCESSOR_MEMBER,
                payload=
                    preprocessor_bytes,
            )

    except Exception as error:

        raise (
            TabularAutoencoderBundleError(
                "Autoencoder bundle creation failed."
            )
        ) from error


    bundle_bytes = (
        output.getvalue()
    )


    if (
        not bundle_bytes
        or
        len(
            bundle_bytes
        )
        >
        _MAX_BUNDLE_BYTES
    ):

        raise (
            TabularAutoencoderBundleError(
                (
                    "Autoencoder bundle size is "
                    "outside the allowed boundary."
                )
            )
        )


    return bundle_bytes


# ============================================================
# TRUSTED DESERIALIZATION
# ============================================================


def deserialize_trusted_tabular_autoencoder_bundle(
    *,
    trusted_bundle_bytes: bytes,
    training_contract: MLAnomalyTrainingContract,
) -> TabularAutoencoderBundleComponents:
    """
    Decode one trusted DataLens-owned Autoencoder bundle.

    joblib/pickle decoding is intentionally delayed until
    archive structure, Training Contract binding and component
    SHA-256 checks have succeeded.

    Callers must pass bytes obtained from the server-owned
    Artifact Store after outer artifact size/SHA validation.
    """

    (
        contract,
        contract_hidden_features,
        contract_latent_features,
    ) = (
        _require_autoencoder_contract(
            training_contract
        )
    )


    if (
        not isinstance(
            trusted_bundle_bytes,
            bytes,
        )
        or
        not trusted_bundle_bytes
        or
        len(
            trusted_bundle_bytes
        )
        >
        _MAX_BUNDLE_BYTES
    ):

        raise (
            TabularAutoencoderBundleIntegrityError(
                "Invalid trusted Autoencoder bundle bytes."
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

        raise (
            TabularAutoencoderBundleIntegrityError(
                (
                    "Autoencoder bundle is not "
                    "a valid ZIP archive."
                )
            )
        ) from error


    with archive:

        infos = (
            archive.infolist()
        )


        members = [
            info.filename
            for info in infos
        ]


        if (
            len(
                members
            )
            !=
            len(
                _EXPECTED_MEMBERS
            )
            or
            set(
                members
            )
            !=
            _EXPECTED_MEMBERS
        ):

            raise (
                TabularAutoencoderBundleIntegrityError(
                    (
                        "Autoencoder bundle member "
                        "surface is invalid."
                    )
                )
            )


        total_uncompressed_bytes = 0


        for info in infos:

            if (
                info.is_dir()
                or
                info.file_size
                <=
                0
                or
                info.file_size
                >
                _MAX_BUNDLE_BYTES
            ):

                raise (
                    TabularAutoencoderBundleIntegrityError(
                        (
                            "Autoencoder bundle member "
                            "size is invalid."
                        )
                    )
                )


            total_uncompressed_bytes += int(
                info.file_size
            )


        if (
            total_uncompressed_bytes
            >
            _MAX_BUNDLE_BYTES
        ):

            raise (
                TabularAutoencoderBundleIntegrityError(
                    (
                        "Autoencoder bundle expanded "
                        "size exceeds the allowed boundary."
                    )
                )
            )


        try:

            manifest_bytes = (
                archive.read(
                    _MANIFEST_MEMBER
                )
            )


            state_bytes = (
                archive.read(
                    _STATE_DICT_MEMBER
                )
            )


            preprocessor_bytes = (
                archive.read(
                    _PREPROCESSOR_MEMBER
                )
            )

        except Exception as error:

            raise (
                TabularAutoencoderBundleIntegrityError(
                    (
                        "Autoencoder bundle members "
                        "could not be read."
                    )
                )
            ) from error


    try:

        manifest_payload = (
            json.loads(
                manifest_bytes.decode(
                    "utf-8"
                )
            )
        )


        manifest = (
            TabularAutoencoderBundleManifest
            .model_validate(
                manifest_payload
            )
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
    ) as error:

        raise (
            TabularAutoencoderBundleIntegrityError(
                (
                    "Autoencoder bundle manifest "
                    "is invalid."
                )
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

        raise (
            TabularAutoencoderBundleIntegrityError(
                (
                    "Autoencoder bundle Training Contract "
                    "fingerprint mismatch."
                )
            )
        )


    if (
        manifest.hidden_features
        !=
        contract_hidden_features
        or
        manifest.latent_features
        !=
        contract_latent_features
    ):

        raise (
            TabularAutoencoderBundleIntegrityError(
                (
                    "Autoencoder bundle architecture "
                    "does not match the Training Contract."
                )
            )
        )


    if (
        float(
            manifest.threshold_quantile
        )
        !=
        float(
            contract.threshold_quantile
        )
    ):

        raise (
            TabularAutoencoderBundleIntegrityError(
                (
                    "Autoencoder bundle threshold policy "
                    "does not match the Training Contract."
                )
            )
        )


    if (
        manifest.state_dict_sha256
        !=
        _sha256(
            state_bytes
        )
    ):

        raise (
            TabularAutoencoderBundleIntegrityError(
                (
                    "Autoencoder bundle state_dict "
                    "SHA-256 mismatch."
                )
            )
        )


    if (
        manifest.preprocessor_sha256
        !=
        _sha256(
            preprocessor_bytes
        )
    ):

        raise (
            TabularAutoencoderBundleIntegrityError(
                (
                    "Autoencoder bundle preprocessor "
                    "SHA-256 mismatch."
                )
            )
        )


    # ========================================================
    # PREPROCESSOR
    #
    # joblib/pickle decoding occurs only after structural,
    # contract and SHA guards have passed.
    # ========================================================


    try:

        preprocessor = (
            joblib.load(
                io.BytesIO(
                    preprocessor_bytes
                )
            )
        )

    except Exception as error:

        raise (
            TabularAutoencoderBundleDeserializationError(
                (
                    "Trusted Autoencoder preprocessor "
                    "could not be deserialized."
                )
            )
        ) from error


    try:

        fitted_preprocessor = (
            _validate_preprocessor(
                preprocessor
            )
        )

    except TabularAutoencoderBundleContractError as error:

        raise (
            TabularAutoencoderBundleDeserializationError(
                str(
                    error
                )
            )
        ) from error


    output_features = (
        _preprocessor_output_features(
            fitted_preprocessor
        )
    )


    if (
        output_features
        !=
        manifest.input_features
    ):

        raise (
            TabularAutoencoderBundleIntegrityError(
                (
                    "Autoencoder fitted preprocessor "
                    "output does not match model input."
                )
            )
        )


    # ========================================================
    # STATE DICT
    #
    # weights_only=True prevents generic PyTorch pickle object
    # restoration.
    # ========================================================


    try:

        loaded_state = (
            torch.load(
                io.BytesIO(
                    state_bytes
                ),
                map_location=
                    "cpu",
                weights_only=
                    True,
            )
        )

    except Exception as error:

        raise (
            TabularAutoencoderBundleDeserializationError(
                (
                    "Trusted Autoencoder state_dict "
                    "could not be deserialized."
                )
            )
        ) from error


    if not isinstance(
        loaded_state,
        Mapping,
    ):

        raise (
            TabularAutoencoderBundleDeserializationError(
                (
                    "Trusted Autoencoder state payload "
                    "is not a state_dict mapping."
                )
            )
        )


    try:

        _validate_model_state_finite(
            loaded_state
        )

    except TabularAutoencoderBundleContractError as error:

        raise (
            TabularAutoencoderBundleDeserializationError(
                str(
                    error
                )
            )
        ) from error


    model = (
        TabularAutoencoder(
            input_features=
                manifest.input_features,
            hidden_features=
                manifest.hidden_features,
            latent_features=
                manifest.latent_features,
        )
    )


    try:

        model.load_state_dict(
            loaded_state,
            strict=True,
        )

    except Exception as error:

        raise (
            TabularAutoencoderBundleDeserializationError(
                (
                    "Trusted Autoencoder state_dict "
                    "does not match TabularAutoencoder."
                )
            )
        ) from error


    model.eval()


    threshold = (
        ReconstructionErrorThreshold(
            quantile=
                float(
                    manifest.threshold_quantile
                ),
            threshold=
                float(
                    manifest.anomaly_threshold
                ),
            train_rows=
                int(
                    manifest.threshold_train_rows
                ),
            method=
                manifest.threshold_method,
            comparison_operator=
                manifest
                .threshold_comparison_operator,
        )
    )


    return (
        TabularAutoencoderBundleComponents(
            manifest=
                manifest,
            model=
                model,
            preprocessor=
                fitted_preprocessor,
            threshold=
                threshold,
        )
    )

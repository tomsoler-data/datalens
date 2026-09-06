from __future__ import annotations


import hashlib
import io
import json
import zipfile


from dataclasses import (
    dataclass,
)


from typing import (
    Literal,
    Mapping,
)


import joblib
import torch


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)


from sklearn.compose import (
    ColumnTransformer,
)


from sklearn.utils.validation import (
    check_is_fitted,
)


from app.deep_learning.networks import (
    FeedForwardRegressor,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


# ============================================================
# VERSION
# ============================================================


DL_TABULAR_MLP_BUNDLE_RULE_VERSION = (
    "dl_tabular_mlp_bundle_v0.1"
)


# ============================================================
# BUNDLE MEMBERS
# ============================================================


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
    128
    *
    1024
    *
    1024
)


# ============================================================
# ERRORS
# ============================================================


class TabularMLPBundleError(
    RuntimeError
):
    pass


class TabularMLPBundleContractError(
    TabularMLPBundleError
):
    pass


class TabularMLPBundleIntegrityError(
    TabularMLPBundleError
):
    pass


class TabularMLPBundleDeserializationError(
    TabularMLPBundleError
):
    pass


# ============================================================
# MANIFEST
# ============================================================


class TabularMLPBundleManifest(
    BaseModel
):
    """
    Privacy-minimal structural manifest.

    It deliberately contains no raw rows, predictions,
    target values or learned preprocessing statistics.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    bundle_version: Literal[
        "dl_tabular_mlp_bundle_v0.1"
    ] = DL_TABULAR_MLP_BUNDLE_RULE_VERSION


    estimator_key: Literal[
        "tabular_mlp_regressor"
    ]


    network_class: Literal[
        "FeedForwardRegressor"
    ]


    state_dict_format: Literal[
        "torch_state_dict"
    ]


    preprocessor_format: Literal[
        "joblib"
    ]


    input_features: int = Field(
        gt=0,
    )


    hidden_features: int = Field(
        gt=0,
    )


    training_contract_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    state_dict_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    preprocessor_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True,
)
class TabularMLPBundleComponents:
    manifest: TabularMLPBundleManifest
    model: FeedForwardRegressor
    preprocessor: ColumnTransformer


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


def _require_mlp_contract(
    training_contract: MLTrainingContract,
) -> tuple[
    MLTrainingContract,
    int,
]:

    contract = (
        MLTrainingContract
        .model_validate(
            training_contract
        )
    )


    if (
        contract.problem_type
        !=
        "regression"
    ):
        raise (
            TabularMLPBundleContractError(
                (
                    "Tabular MLP bundle requires "
                    "a regression Training Contract."
                )
            )
        )


    if (
        contract.estimator_key
        !=
        "tabular_mlp_regressor"
    ):
        raise (
            TabularMLPBundleContractError(
                (
                    "Tabular MLP bundle requires "
                    "estimator_key="
                    "tabular_mlp_regressor."
                )
            )
        )


    hyperparameters = (
        contract
        .effective_estimator_hyperparameters
    )


    if (
        hyperparameters
        is None
        or
        getattr(
            hyperparameters,
            "kind",
            None,
        )
        !=
        "tabular_mlp_regressor"
    ):
        raise (
            TabularMLPBundleContractError(
                (
                    "Tabular MLP hyperparameter "
                    "authority could not be resolved."
                )
            )
        )


    hidden_features = getattr(
        hyperparameters,
        "hidden_features",
        None,
    )


    if (
        isinstance(
            hidden_features,
            bool,
        )
        or
        not isinstance(
            hidden_features,
            int,
        )
        or
        hidden_features
        <=
        0
    ):
        raise (
            TabularMLPBundleContractError(
                (
                    "Tabular MLP hidden_features "
                    "authority is invalid."
                )
            )
        )


    return (
        contract,
        hidden_features,
    )


def _validate_preprocessor(
    preprocessor: object,
) -> ColumnTransformer:

    if not isinstance(
        preprocessor,
        ColumnTransformer,
    ):
        raise (
            TabularMLPBundleContractError(
                (
                    "Tabular MLP bundle requires "
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
            TabularMLPBundleContractError(
                (
                    "Tabular MLP bundle requires "
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
            TabularMLPBundleContractError(
                (
                    "Fitted Tabular MLP preprocessor "
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
            TabularMLPBundleContractError(
                (
                    "Fitted Tabular MLP preprocessor "
                    "output feature count could not "
                    "be resolved."
                )
            )
        ) from error


    if count <= 0:

        raise (
            TabularMLPBundleContractError(
                (
                    "Fitted Tabular MLP preprocessor "
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
            TabularMLPBundleContractError(
                "Tabular MLP state_dict cannot be empty."
            )
        )


    for (
        name,
        value,
    ) in state_dict.items():

        if not isinstance(
            name,
            str,
        ):

            raise (
                TabularMLPBundleContractError(
                    (
                        "Tabular MLP state_dict "
                        "keys must be strings."
                    )
                )
            )


        if not isinstance(
            value,
            torch.Tensor,
        ):

            raise (
                TabularMLPBundleContractError(
                    (
                        "Tabular MLP state_dict "
                        "values must be tensors."
                    )
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

                raise (
                    TabularMLPBundleContractError(
                        (
                            "Tabular MLP state_dict "
                            "contains non-finite values."
                        )
                    )
                )


def _canonical_manifest_bytes(
    manifest: TabularMLPBundleManifest,
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


def _zip_member(
    archive: zipfile.ZipFile,
    *,
    name: str,
    payload: bytes,
) -> None:

    info = zipfile.ZipInfo(
        filename=name,
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
# SERIALIZE
# ============================================================


def serialize_tabular_mlp_bundle(
    *,
    model: FeedForwardRegressor,
    preprocessor: ColumnTransformer,
    training_contract: MLTrainingContract,
) -> bytes:

    (
        contract,
        contract_hidden_features,
    ) = (
        _require_mlp_contract(
            training_contract
        )
    )


    if not isinstance(
        model,
        FeedForwardRegressor,
    ):
        raise (
            TabularMLPBundleContractError(
                (
                    "Tabular MLP bundle requires "
                    "FeedForwardRegressor."
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
            TabularMLPBundleContractError(
                (
                    "Model input_features do not "
                    "match fitted preprocessor output."
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
            TabularMLPBundleContractError(
                (
                    "Model hidden_features do not "
                    "match the Training Contract."
                )
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
        ) in model.state_dict().items()
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
            TabularMLPBundleError(
                (
                    "Tabular MLP state_dict "
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
            TabularMLPBundleError(
                (
                    "Tabular MLP fitted preprocessor "
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
            TabularMLPBundleError(
                (
                    "Tabular MLP bundle components "
                    "cannot be empty."
                )
            )
        )


    manifest = (
        TabularMLPBundleManifest(
            estimator_key=
                "tabular_mlp_regressor",

            network_class=
                "FeedForwardRegressor",

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

            training_contract_sha256=
                ml_training_contract_sha256(
                    contract
                ),

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

            _zip_member(
                archive,
                name=
                    _MANIFEST_MEMBER,
                payload=
                    manifest_bytes,
            )


            _zip_member(
                archive,
                name=
                    _STATE_DICT_MEMBER,
                payload=
                    state_bytes,
            )


            _zip_member(
                archive,
                name=
                    _PREPROCESSOR_MEMBER,
                payload=
                    preprocessor_bytes,
            )


    except Exception as error:

        raise (
            TabularMLPBundleError(
                "Tabular MLP bundle creation failed."
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
            TabularMLPBundleError(
                (
                    "Tabular MLP bundle size is "
                    "outside the allowed boundary."
                )
            )
        )


    return bundle_bytes


# ============================================================
# TRUSTED DESERIALIZATION
# ============================================================


def deserialize_trusted_tabular_mlp_bundle(
    *,
    trusted_bundle_bytes: bytes,
    training_contract: MLTrainingContract,
) -> TabularMLPBundleComponents:
    """
    Decode one trusted DataLens-owned PyTorch bundle.

    IMPORTANT:
    The fitted preprocessor uses joblib/pickle serialization.
    Therefore callers MUST only pass bytes obtained from the
    server-owned Model Artifact Store after size/SHA validation.

    The future trusted DL loader owns that security boundary.
    """

    (
        contract,
        contract_hidden_features,
    ) = (
        _require_mlp_contract(
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
            TabularMLPBundleIntegrityError(
                "Invalid trusted PyTorch bundle bytes."
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
            TabularMLPBundleIntegrityError(
                "PyTorch bundle is not a valid ZIP archive."
            )
        ) from error


    with archive:

        members = (
            archive.namelist()
        )


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
                TabularMLPBundleIntegrityError(
                    (
                        "PyTorch bundle member surface "
                        "is invalid."
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
                TabularMLPBundleIntegrityError(
                    (
                        "PyTorch bundle members "
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
            TabularMLPBundleManifest
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
            TabularMLPBundleIntegrityError(
                "PyTorch bundle manifest is invalid."
            )
        ) from error


    expected_contract_sha = (
        ml_training_contract_sha256(
            contract
        )
    )


    if (
        manifest.training_contract_sha256
        !=
        expected_contract_sha
    ):
        raise (
            TabularMLPBundleIntegrityError(
                (
                    "PyTorch bundle Training Contract "
                    "fingerprint mismatch."
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
            TabularMLPBundleIntegrityError(
                (
                    "PyTorch bundle state_dict "
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
            TabularMLPBundleIntegrityError(
                (
                    "PyTorch bundle preprocessor "
                    "SHA-256 mismatch."
                )
            )
        )


    if (
        manifest.hidden_features
        !=
        contract_hidden_features
    ):
        raise (
            TabularMLPBundleIntegrityError(
                (
                    "PyTorch bundle hidden_features "
                    "do not match the Training Contract."
                )
            )
        )


    # ========================================================
    # PREPROCESSOR
    #
    # joblib/pickle decoding occurs only after all structural,
    # contract and component integrity guards above.
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
            TabularMLPBundleDeserializationError(
                (
                    "Trusted PyTorch bundle preprocessor "
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

    except TabularMLPBundleContractError as error:

        raise (
            TabularMLPBundleDeserializationError(
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
            TabularMLPBundleIntegrityError(
                (
                    "PyTorch bundle fitted preprocessor "
                    "output does not match model input."
                )
            )
        )


    # ========================================================
    # STATE DICT
    #
    # weights_only=True deliberately prevents generic
    # torch pickle object restoration.
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
            TabularMLPBundleDeserializationError(
                (
                    "Trusted PyTorch state_dict "
                    "could not be deserialized."
                )
            )
        ) from error


    if not isinstance(
        loaded_state,
        Mapping,
    ):
        raise (
            TabularMLPBundleDeserializationError(
                (
                    "Trusted PyTorch state payload "
                    "is not a state_dict mapping."
                )
            )
        )


    _validate_model_state_finite(
        loaded_state
    )


    model = (
        FeedForwardRegressor(
            input_features=
                manifest.input_features,

            hidden_features=
                manifest.hidden_features,
        )
    )


    try:

        model.load_state_dict(
            loaded_state,
            strict=True,
        )

    except Exception as error:

        raise (
            TabularMLPBundleDeserializationError(
                (
                    "Trusted PyTorch state_dict "
                    "does not match FeedForwardRegressor."
                )
            )
        ) from error


    model.eval()


    return (
        TabularMLPBundleComponents(
            manifest=
                manifest,

            model=
                model,

            preprocessor=
                fitted_preprocessor,
        )
    )

from __future__ import annotations


import ast
import io
import json
from pathlib import Path
import zipfile


import torch


from app.deep_learning.networks import (
    FeedForwardRegressor,
)


from app.deep_learning.time_series_bundle import (
    DL_TIME_SERIES_NEURAL_BUNDLE_CODEC_RULE_VERSION,
    DLTimeSeriesNeuralBundleContractError,
    DLTimeSeriesNeuralBundleIntegrityError,
    deserialize_trusted_time_series_neural_bundle,
    serialize_time_series_neural_bundle,
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


from app.deep_learning.time_series_contracts import (
    DLTimeSeriesLSTMRegressorHyperparameters,
    DLTimeSeriesMLPRegressorHyperparameters,
    DLTimeSeriesRNNRegressorHyperparameters,
)


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.experiment_provenance import (
    ml_model_training_contract_sha256,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_model_training_contracts import (
    MLTimeSeriesModelTrainingContract,
)


# ============================================================
# HELPERS
# ============================================================


EXPECTED_MEMBERS = {
    "manifest.json",
    "model_state.pt",
    "standardizer.json",
}


def require_bundle_error(
    callback,
    *,
    label: str,
) -> None:

    try:

        callback()

    except (
        DLTimeSeriesNeuralBundleContractError,
        DLTimeSeriesNeuralBundleIntegrityError,
    ):

        return


    raise AssertionError(
        (
            "Expected temporal bundle failure: "
            f"{label}"
        )
    )


def archive_members(
    bundle_bytes: bytes,
) -> dict[
    str,
    bytes,
]:

    with zipfile.ZipFile(
        io.BytesIO(
            bundle_bytes
        ),
        mode="r",
    ) as archive:

        return {
            name:
                archive.read(
                    name
                )

            for name
            in archive.namelist()
        }


def build_archive(
    members: dict[
        str,
        bytes,
    ],
) -> bytes:

    output = io.BytesIO()


    with zipfile.ZipFile(
        output,
        mode="w",
    ) as archive:

        for (
            name,
            payload,
        ) in members.items():

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


            archive.writestr(
                info,
                payload,
            )


    return output.getvalue()


def assert_state_equal(
    left,
    right,
) -> None:

    left_state = (
        left.state_dict()
    )


    right_state = (
        right.state_dict()
    )


    assert (
        tuple(
            left_state
            .keys()
        )
        ==
        tuple(
            right_state
            .keys()
        )
    )


    for name in left_state:

        assert torch.equal(
            left_state[
                name
            ],
            right_state[
                name
            ],
        )


def contract_for(
    estimator_hyperparameters,
) -> MLTimeSeriesModelTrainingContract:

    task = (
        MLTimeSeriesForecastingContract(
            workflow_id=
                "workflow:bundle",

            dataset_id=
                "dataset:bundle",

            target_column=
                "revenue",

            lookback=
                4,

            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "order_date",

                    test_size=
                        0.25,
                ),
        )
    )


    return (
        MLTimeSeriesModelTrainingContract(
            task_contract=
                task,

            estimator_hyperparameters=
                estimator_hyperparameters,
        )
    )


# ============================================================
# 1. CONTROLLED FIXTURES
# ============================================================


mlp_contract = contract_for(
    DLTimeSeriesMLPRegressorHyperparameters(
        hidden_features=
            6,

        epochs=
            10,

        batch_size=
            8,

        learning_rate=
            0.01,
    )
)


rnn_contract = contract_for(
    DLTimeSeriesRNNRegressorHyperparameters(
        hidden_size=
            5,

        epochs=
            10,

        batch_size=
            8,

        learning_rate=
            0.01,
    )
)


lstm_contract = contract_for(
    DLTimeSeriesLSTMRegressorHyperparameters(
        hidden_size=
            5,

        epochs=
            10,

        batch_size=
            8,

        learning_rate=
            0.01,
    )
)


torch.manual_seed(
    123
)


mlp = FeedForwardRegressor(
    input_features=
        4,

    hidden_features=
        6,
)


rnn = TimeSeriesRNNRegressor(
    lookback=
        4,

    hidden_size=
        5,
)


lstm = TimeSeriesLSTMRegressor(
    lookback=
        4,

    hidden_size=
        5,
)


standardizer = (
    DLTimeSeriesStandardizer(
        mean=
            64.5,

        standard_deviation=
            8.65544144839919,

        fitted_observation_count=
            30,
    )
)


print(
    "[PASS] controlled MLP/RNN/LSTM bundle fixtures"
)


# ============================================================
# 2. SERIALIZE + TRUSTED RESTORE ALL THREE FAMILIES
# ============================================================


cases = (
    (
        mlp_contract,
        mlp,
        FeedForwardRegressor,
        "FeedForwardRegressor",
        6,
    ),
    (
        rnn_contract,
        rnn,
        TimeSeriesRNNRegressor,
        "TimeSeriesRNNRegressor",
        5,
    ),
    (
        lstm_contract,
        lstm,
        TimeSeriesLSTMRegressor,
        "TimeSeriesLSTMRegressor",
        5,
    ),
)


bundles = {}


for (
    contract,
    model,
    expected_type,
    expected_network_class,
    expected_hidden_width,
) in cases:

    state_before = {
        name:
            value.detach().clone()

        for (
            name,
            value,
        ) in model.state_dict().items()
    }


    bundle = (
        serialize_time_series_neural_bundle(
            model=
                model,

            standardizer=
                standardizer,

            training_contract=
                contract,
        )
    )


    assert isinstance(
        bundle,
        bytes,
    )


    assert len(
        bundle
    ) > 0


    bundles[
        contract.estimator_key
    ] = bundle


    members = archive_members(
        bundle
    )


    assert set(
        members
    ) == EXPECTED_MEMBERS


    manifest_payload = json.loads(
        members[
            "manifest.json"
        ]
        .decode(
            "utf-8"
        )
    )


    assert (
        manifest_payload[
            "estimator_key"
        ]
        ==
        contract.estimator_key
    )


    assert (
        manifest_payload[
            "network_class"
        ]
        ==
        expected_network_class
    )


    assert (
        manifest_payload[
            "lookback"
        ]
        ==
        4
    )


    assert (
        manifest_payload[
            "hidden_width"
        ]
        ==
        expected_hidden_width
    )


    assert (
        manifest_payload[
            "training_contract_sha256"
        ]
        ==
        ml_model_training_contract_sha256(
            contract
        )
    )


    restored = (
        deserialize_trusted_time_series_neural_bundle(
            trusted_bundle_bytes=
                bundle,

            training_contract=
                contract,
        )
    )


    assert isinstance(
        restored.model,
        expected_type,
    )


    assert (
        restored.standardizer
        ==
        standardizer
    )


    assert (
        restored.model.training
        is False
    )


    for parameter in (
        restored.model.parameters()
    ):

        assert (
            parameter.device.type
            ==
            "cpu"
        )


    assert_state_equal(
        model,
        restored.model,
    )


    for name, value in model.state_dict().items():

        assert torch.equal(
            value,
            state_before[
                name
            ],
        )


print(
    "[PASS] MLP/RNN/LSTM serialize and trusted-restore exact state"
)


# ============================================================
# 3. STANDARDIZER MEMBER IS DETERMINISTIC JSON
# ============================================================


mlp_bundle_second = (
    serialize_time_series_neural_bundle(
        model=
            mlp,

        standardizer=
            standardizer,

        training_contract=
            mlp_contract,
    )
)


first_standardizer_bytes = (
    archive_members(
        bundles[
            "time_series_mlp_regressor"
        ]
    )[
        "standardizer.json"
    ]
)


second_standardizer_bytes = (
    archive_members(
        mlp_bundle_second
    )[
        "standardizer.json"
    ]
)


assert (
    first_standardizer_bytes
    ==
    second_standardizer_bytes
)


standardizer_payload = json.loads(
    first_standardizer_bytes
    .decode(
        "utf-8"
    )
)


assert set(
    standardizer_payload
) == {
    "snapshot_version",
    "mean",
    "standard_deviation",
    "fitted_observation_count",
    "scaling_rule_version",
}


print(
    "[PASS] TRAIN-only scaler persists as deterministic minimal JSON"
)


# ============================================================
# 4. WRONG TRAINING CONTRACT FAILS CLOSED
# ============================================================


mlp_bundle = bundles[
    "time_series_mlp_regressor"
]


require_bundle_error(
    lambda:
        deserialize_trusted_time_series_neural_bundle(
            trusted_bundle_bytes=
                mlp_bundle,

            training_contract=
                rnn_contract,
        ),
    label=
        "MLP bundle restored with RNN contract",
)


print(
    "[PASS] Training Contract fingerprint mismatch fails closed"
)


# ============================================================
# 5. STANDARDIZER MEMBER TAMPERING FAILS CLOSED
# ============================================================


tampered_standardizer_members = (
    archive_members(
        mlp_bundle
    )
)


tampered_standardizer_members[
    "standardizer.json"
] = (
    b'{"mean":999999.0}'
)


tampered_standardizer_bundle = (
    build_archive(
        tampered_standardizer_members
    )
)


require_bundle_error(
    lambda:
        deserialize_trusted_time_series_neural_bundle(
            trusted_bundle_bytes=
                tampered_standardizer_bundle,

            training_contract=
                mlp_contract,
        ),
    label=
        "tampered standardizer member",
)


print(
    "[PASS] standardizer SHA-256 detects member tampering"
)


# ============================================================
# 6. STATE MEMBER TAMPERING FAILS CLOSED
# ============================================================


tampered_state_members = (
    archive_members(
        mlp_bundle
    )
)


tampered_state_members[
    "model_state.pt"
] = (
    tampered_state_members[
        "model_state.pt"
    ]
    +
    b"tamper"
)


tampered_state_bundle = (
    build_archive(
        tampered_state_members
    )
)


require_bundle_error(
    lambda:
        deserialize_trusted_time_series_neural_bundle(
            trusted_bundle_bytes=
                tampered_state_bundle,

            training_contract=
                mlp_contract,
        ),
    label=
        "tampered state member",
)


print(
    "[PASS] state_dict SHA-256 detects member tampering"
)


# ============================================================
# 7. EXTRA ZIP MEMBER FAILS CLOSED
# ============================================================


extra_member_payload = (
    archive_members(
        mlp_bundle
    )
)


extra_member_payload[
    "unexpected.bin"
] = b"forbidden"


extra_member_bundle = (
    build_archive(
        extra_member_payload
    )
)


require_bundle_error(
    lambda:
        deserialize_trusted_time_series_neural_bundle(
            trusted_bundle_bytes=
                extra_member_bundle,

            training_contract=
                mlp_contract,
        ),
    label=
        "unexpected ZIP member",
)


print(
    "[PASS] exact ZIP member authority fails closed"
)


# ============================================================
# 8. NON-FINITE LEARNED STATE FAILS BEFORE SERIALIZATION
# ============================================================


bad_model = FeedForwardRegressor(
    input_features=
        4,

    hidden_features=
        6,
)


with torch.no_grad():

    first_parameter = next(
        bad_model.parameters()
    )


    first_parameter.view(
        -1
    )[
        0
    ] = float(
        "nan"
    )


require_bundle_error(
    lambda:
        serialize_time_series_neural_bundle(
            model=
                bad_model,

            standardizer=
                standardizer,

            training_contract=
                mlp_contract,
        ),
    label=
        "non-finite model state",
)


print(
    "[PASS] non-finite learned state is blocked before persistence"
)


# ============================================================
# 9. INVALID STANDARDIZER FAILS BEFORE SERIALIZATION
# ============================================================


invalid_standardizer = (
    DLTimeSeriesStandardizer(
        mean=
            10.0,

        standard_deviation=
            0.0,

        fitted_observation_count=
            30,
    )
)


require_bundle_error(
    lambda:
        serialize_time_series_neural_bundle(
            model=
                mlp,

            standardizer=
                invalid_standardizer,

            training_contract=
                mlp_contract,
        ),
    label=
        "zero standard deviation",
)


print(
    "[PASS] invalid TRAIN-only scaler is blocked before persistence"
)


# ============================================================
# 10. MODEL / CONTRACT STRUCTURE MISMATCH FAILS CLOSED
# ============================================================


wrong_width_model = (
    FeedForwardRegressor(
        input_features=
            3,

        hidden_features=
            6,
    )
)


require_bundle_error(
    lambda:
        serialize_time_series_neural_bundle(
            model=
                wrong_width_model,

            standardizer=
                standardizer,

            training_contract=
                mlp_contract,
        ),
    label=
        "MLP input width does not match lookback",
)


print(
    "[PASS] model structure must match exact Forecast Training Contract"
)


# ============================================================
# 11. STATIC SECURITY BOUNDARY
# ============================================================


source_path = Path(
    "app/deep_learning/time_series_bundle.py"
)


source = source_path.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source
)


imports = []


for node in ast.walk(
    tree
):

    if isinstance(
        node,
        ast.Import,
    ):

        imports.extend(
            alias.name
            for alias
            in node.names
        )


    elif isinstance(
        node,
        ast.ImportFrom,
    ):

        if node.module:

            imports.append(
                node.module
            )


assert "joblib" not in imports
assert "pickle" not in imports


torch_load_calls = []


for node in ast.walk(
    tree
):

    if not isinstance(
        node,
        ast.Call,
    ):

        continue


    function = node.func


    if not isinstance(
        function,
        ast.Attribute,
    ):

        continue


    if not isinstance(
        function.value,
        ast.Name,
    ):

        continue


    if (
        function.value.id
        ==
        "torch"
        and
        function.attr
        ==
        "load"
    ):

        torch_load_calls.append(
            node
        )


assert len(
    torch_load_calls
) == 1


weights_only_keywords = [
    keyword
    for keyword
    in torch_load_calls[
        0
    ].keywords
    if keyword.arg
    ==
    "weights_only"
]


assert len(
    weights_only_keywords
) == 1


assert isinstance(
    weights_only_keywords[
        0
    ].value,
    ast.Constant,
)


assert (
    weights_only_keywords[
        0
    ].value.value
    is True
)


assert (
    "strict=True"
    in
    source
)


for forbidden_import in (
    "app.deep_learning.training",
    "app.deep_learning.time_series_mlp_executor",
    "app.deep_learning.time_series_rnn_executor",
    "app.deep_learning.time_series_lstm_executor",
):

    assert forbidden_import not in imports


print(
    "[PASS] trusted reload uses weights_only=True and contains no training authority"
)


# ============================================================
# 12. BUNDLE DOES NOT PERSIST EVALUATION EVIDENCE
# ============================================================


for bundle in bundles.values():

    members = (
        archive_members(
            bundle
        )
    )


    manifest_text = (
        members[
            "manifest.json"
        ]
        .decode(
            "utf-8"
        )
    )


    standardizer_text = (
        members[
            "standardizer.json"
        ]
        .decode(
            "utf-8"
        )
    )


    searchable = (
        manifest_text
        +
        standardizer_text
    )


    for forbidden_term in (
        "predictions",
        "targets",
        "raw_rows",
        "epoch_losses",
        "optimizer_state",
        "execution_device",
    ):

        assert (
            forbidden_term
            not in
            searchable
        )


print(
    "[PASS] bundle metadata remains privacy-minimal and evaluation-free"
)


print()
print("=" * 80)
print("DL-5-A11-V2-P2 FINAL VERDICT")
print("=" * 80)
print()

print("Shared MLP/RNN/LSTM .ptbundle codec         PASS")
print("Exact three-member ZIP authority            PASS")
print("Training Contract SHA-256 binding           PASS")
print("Model/network structural binding            PASS")
print("TRAIN-only standardizer JSON                PASS")
print("State-dict SHA-256                          PASS")
print("Standardizer SHA-256                        PASS")
print("Non-finite learned-state guard              PASS")
print("Strict ZIP member authority                 PASS")
print("weights_only trusted PyTorch restore         PASS")
print("strict state_dict restore                    PASS")
print("CPU-owned reconstructed models              PASS")
print("No joblib / pickle preprocessing            PASS")
print("No training during trusted restore           PASS")
print("Privacy-minimal bundle metadata              PASS")

print()
print(
    "DL-5-A11-V2-P2 - TEMPORAL NEURAL PTBUNDLE CODEC: PASS"
)

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from app.adaptation.qlora_runtime_v0_4 import (
    load_pinned_tokenizer,
    local_text_checkpoint_path,
    validate_static_authority,
)
from app.adaptation.training_dataset_canonicalizer_v0_4 import (
    build_user_message,
)
from app.semantics.normalizer import (
    normalize_dataset_semantics,
)
from app.semantics.profiler import (
    build_column_context,
    build_deterministic_structural_profile,
    build_fallback_profile,
)
from app.semantics.schemas import (
    ColumnSemanticProfile,
    DatasetSemanticProfile,
)


GREENHOUSE_FINAL_ACCEPTANCE_RUNNER_RULE_VERSION = (
    "qlora_v0.4_greenhouse_final_acceptance_runner_v0.1"
)

GREENHOUSE_DETERMINISTIC_PROFILE_WRAPPER_RULE_VERSION = (
    "greenhouse_deterministic_semantic_profile_wrapper_v0.1"
)

GREENHOUSE_DESCRIPTION_SERIALIZER_RULE_VERSION = (
    "greenhouse_normalized_semantic_"
    "profile_description_serializer_v0.1"
)


EXPECTED_SOURCE_GIT_COMMIT = (
    "db7ae1ab0d4d9d8f58f7c3a8c5d5ab6ae874cba0"
)

EXPECTED_GREENHOUSE_PROTOCOL_SHA256 = (
    "056272cca075a6f8c48c38f5296c0dda"
    "caaeb29e64f08048a43a80ea3dab0b16"
)

EXPECTED_INPUT_SERIALIZER_DECISION_SHA256 = (
    "df54af3e32dc97b5fd43573c8c2a8caf"
    "1335976fde3b39faa51afc48ab5244e3"
)

EXPECTED_TASK_INTERPRETATION_SHA256 = (
    "9f097f8b8e7e2dd182cc86cc5ed12fc7"
    "4ded18262bac2af3639d2bf07bdf61a2"
)

EXPECTED_AIRPORT_RUNNER_SHA256 = (
    "fda934f45a98c10905cd857109df6c3f"
    "927d0ae025e9fb41150d85872001fe23"
)

EXPECTED_SHARED_RUNTIME_SHA256 = (
    "20e41ab00606296893276a84e53746c0"
    "6618b8cabca74fef77cb743c5e80ab7c"
)

EXPECTED_CANONICALIZER_SHA256 = (
    "075d6c22cf4473b414221a0766ca831f"
    "4da1e19597a0a5b2b5efd3e5755c9356"
)

EXPECTED_PROFILER_SHA256 = (
    "54542f6e0d8709ea5990596427e9c1d9"
    "9996db1a27e68ccbd9b2c40c9e0d39e7"
)

EXPECTED_NORMALIZER_SHA256 = (
    "90ce9442abc041649d9f72683b5b5cabb"
    "a120d097ec5a6354cd0ea261962d3e2"
)

EXPECTED_SEMANTIC_SCHEMAS_SHA256 = (
    "1aab9c75bdbebad424c01b065ce256d6"
    "2223a638dd6333f5d4dcbf59a7b49776"
)

EXPECTED_ADAPTER_BUNDLE_SHA256 = (
    "0351980df6d86096195c0971deb30c725"
    "e155c71aa5de8054b2b37fa42090716"
)

EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "7de1c58e208eda46e9c7f86397df37ec"
    "49883aeece39fb961e0a6b24088dd3c4"
)


BASE_MODEL_REPOSITORY = (
    "google/gemma-3-4b-it"
)

BASE_MODEL_REVISION = (
    "093f9f388b31de276ce2de164bdc2081324b9767"
)

GREENHOUSE_DOMAIN = (
    "commercial_greenhouse_operations"
)


EXPECTED_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


SERIALIZED_PROFILE_FIELDS = (
    "data_type",
    "concept",
    "semantic_group",
    "variant",
    "measure_kind",
    "unit_kind",
    "quantity_dimension",
    "quantity_unit",
    "entity_role",
    "qualifiers",
)


ALLOWED_PROFILE_SOURCES = {
    "deterministic",
    "deterministic_fallback",
}


MAX_NEW_TOKENS = 64

EOS_TOKEN_IDS = (
    1,
    106,
)

PAD_TOKEN_ID = 0


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / (
        "datalens_semantic_qlora_v0.4_"
        "greenhouse_final_acceptance_protocol_v0.1.json"
    )
)


INPUT_SERIALIZER_DECISION_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / (
        "datalens_semantic_qlora_v0.4_"
        "greenhouse_deterministic_input_"
        "serializer_decision_v0.1.json"
    )
)


TASK_INTERPRETATION_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / (
        "datalens_semantic_qlora_v0.4_"
        "greenhouse_gate_task_"
        "interpretation_decision_v0.1.json"
    )
)


ADAPTER_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "training"
    / "datalens-semantic-qlora-v0.4-training-v0.1"
    / "adapter"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def _require_exact_sha(
    *,
    path: Path,
    expected_sha256: str,
    label: str,
) -> None:

    if not path.is_file():
        raise RuntimeError(
            f"{label} missing: {path}"
        )

    actual = sha256_file(
        path
    )

    if actual != expected_sha256:
        raise RuntimeError(
            (
                f"{label} SHA changed.\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual}"
            )
        )


def validate_non_protected_runner_authorities(
) -> None:

    _require_exact_sha(
        path=
            PROTOCOL_PATH,
        expected_sha256=
            EXPECTED_GREENHOUSE_PROTOCOL_SHA256,
        label=
            "Greenhouse protocol",
    )

    _require_exact_sha(
        path=
            INPUT_SERIALIZER_DECISION_PATH,
        expected_sha256=
            EXPECTED_INPUT_SERIALIZER_DECISION_SHA256,
        label=
            "Greenhouse deterministic-input decision",
    )

    _require_exact_sha(
        path=
            TASK_INTERPRETATION_PATH,
        expected_sha256=
            EXPECTED_TASK_INTERPRETATION_SHA256,
        label=
            "Greenhouse task interpretation",
    )


def _validate_dataframe_columns(
    dataframe: pd.DataFrame,
) -> tuple[str, ...]:

    columns = list(
        dataframe.columns
    )

    if not all(
        isinstance(
            column,
            str,
        )
        for column in columns
    ):
        raise RuntimeError(
            "All dataframe column names must be strings."
        )

    if (
        len(columns)
        !=
        len(
            set(columns)
        )
    ):
        raise RuntimeError(
            "Dataframe column names must be unique."
        )

    return tuple(
        columns
    )


def build_greenhouse_deterministic_profile(
    *,
    dataset_id: str,
    filename: str,
    dataframe: pd.DataFrame,
) -> DatasetSemanticProfile:

    columns = (
        _validate_dataframe_columns(
            dataframe
        )
    )

    profiles: list[
        ColumnSemanticProfile
    ] = []


    for column in columns:

        context = build_column_context(
            dataset_id=
                dataset_id,
            filename=
                filename,
            column=
                column,
            series=
                dataframe[
                    column
                ],
            peer_columns=
                list(columns),
        )


        profile = (
            build_deterministic_structural_profile(
                context=
                    context,
            )
        )


        if profile is None:

            profile = build_fallback_profile(
                context=
                    context,
            )


        if (
            profile.source
            not in
            ALLOWED_PROFILE_SOURCES
        ):
            raise RuntimeError(
                (
                    "Greenhouse deterministic profile "
                    "used a forbidden source: "
                    f"{profile.source!r}"
                )
            )


        profiles.append(
            profile
        )


    raw_profile = DatasetSemanticProfile(
        dataset_id=
            dataset_id,
        filename=
            filename,
        columns=
            profiles,
    )


    normalized = normalize_dataset_semantics(
        raw_profile
    )


    if (
        normalized.dataset_id
        !=
        dataset_id
    ):
        raise RuntimeError(
            "Normalization changed dataset identity."
        )


    if (
        normalized.filename
        !=
        filename
    ):
        raise RuntimeError(
            "Normalization changed filename identity."
        )


    if (
        len(
            normalized.columns
        )
        !=
        len(columns)
    ):
        raise RuntimeError(
            "Normalization changed column count."
        )


    for expected_column, profile in zip(
        columns,
        normalized.columns,
    ):

        if (
            profile.column
            !=
            expected_column
        ):
            raise RuntimeError(
                "Normalization changed column order."
            )

        if (
            profile.source
            not in
            ALLOWED_PROFILE_SOURCES
        ):
            raise RuntimeError(
                (
                    "Normalized profile has forbidden source: "
                    f"{profile.source!r}"
                )
            )


    return normalized


def build_profile_index(
    profiles: Sequence[
        DatasetSemanticProfile
    ],
) -> dict[
    tuple[str, str],
    ColumnSemanticProfile,
]:

    output: dict[
        tuple[str, str],
        ColumnSemanticProfile,
    ] = {}


    for dataset_profile in profiles:

        for column_profile in (
            dataset_profile.columns
        ):

            key = (
                column_profile.dataset_id,
                column_profile.column,
            )

            if key in output:
                raise RuntimeError(
                    (
                        "Duplicate semantic-profile identity: "
                        f"{key!r}"
                    )
                )

            output[
                key
            ] = column_profile


    return output


def serialize_normalized_profile(
    profile: ColumnSemanticProfile,
) -> str:

    if (
        profile.source
        not in
        ALLOWED_PROFILE_SOURCES
    ):
        raise RuntimeError(
            (
                "Cannot serialize profile from source "
                f"{profile.source!r}."
            )
        )


    values: dict[
        str,
        Any,
    ] = {}


    for field in SERIALIZED_PROFILE_FIELDS:

        if not hasattr(
            profile,
            field,
        ):
            raise RuntimeError(
                (
                    "Required serializer field missing: "
                    f"{field}"
                )
            )

        values[
            field
        ] = getattr(
            profile,
            field,
        )


    qualifiers = values[
        "qualifiers"
    ]


    if not isinstance(
        qualifiers,
        list,
    ):
        raise RuntimeError(
            "Profile qualifiers must be a list."
        )


    if not all(
        isinstance(
            qualifier,
            str,
        )
        for qualifier in qualifiers
    ):
        raise RuntimeError(
            "Profile qualifiers must contain strings only."
        )


    values[
        "qualifiers"
    ] = sorted(
        qualifiers
    )


    for key, value in values.items():

        if key == "qualifiers":
            continue

        if not isinstance(
            value,
            str,
        ):
            raise RuntimeError(
                (
                    "Serializer value must be a string: "
                    f"{key}"
                )
            )


    return json.dumps(
        values,
        ensure_ascii=True,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def build_label_blind_prompt_record(
    *,
    case_identity: Mapping[
        str,
        Any,
    ],
    profile_index: Mapping[
        tuple[str, str],
        ColumnSemanticProfile,
    ],
) -> dict[
    str,
    str,
]:

    required_identity_fields = (
        "left_dataset_id",
        "right_dataset_id",
        "left_column",
        "right_column",
    )


    for field in required_identity_fields:

        value = case_identity.get(
            field
        )

        if not isinstance(
            value,
            str,
        ) or not value:

            raise RuntimeError(
                (
                    "Invalid case identity field: "
                    f"{field}"
                )
            )


    left_key = (
        case_identity[
            "left_dataset_id"
        ],
        case_identity[
            "left_column"
        ],
    )

    right_key = (
        case_identity[
            "right_dataset_id"
        ],
        case_identity[
            "right_column"
        ],
    )


    left_profile = profile_index.get(
        left_key
    )

    right_profile = profile_index.get(
        right_key
    )


    if left_profile is None:
        raise RuntimeError(
            (
                "Missing left semantic profile: "
                f"{left_key!r}"
            )
        )


    if right_profile is None:
        raise RuntimeError(
            (
                "Missing right semantic profile: "
                f"{right_key!r}"
            )
        )


    return {
        "domain":
            GREENHOUSE_DOMAIN,
        "left_metric":
            case_identity[
                "left_column"
            ],
        "left_description":
            serialize_normalized_profile(
                left_profile
            ),
        "right_metric":
            case_identity[
                "right_column"
            ],
        "right_description":
            serialize_normalized_profile(
                right_profile
            ),
    }


def build_label_blind_user_message(
    *,
    case_identity: Mapping[
        str,
        Any,
    ],
    profile_index: Mapping[
        tuple[str, str],
        ColumnSemanticProfile,
    ],
) -> str:

    record = (
        build_label_blind_prompt_record(
            case_identity=
                case_identity,
            profile_index=
                profile_index,
        )
    )


    return build_user_message(
        record
    )


def _invalid_output(
    *,
    decoded_output: str,
    invalid_reason: str,
    terminal_stop_token_id: int | None,
    generation_budget_exhausted: bool,
) -> dict[
    str,
    Any,
]:

    normalized = (
        decoded_output.strip()
    )

    return {
        "strict_json_valid":
            False,
        "relation":
            None,
        "reason":
            None,
        "invalid_reason":
            invalid_reason,
        "decoded_output":
            normalized,
        "decoded_output_sha256":
            hashlib.sha256(
                normalized.encode(
                    "utf-8"
                )
            ).hexdigest(),
        "terminal_stop_token_id":
            terminal_stop_token_id,
        "generation_budget_exhausted":
            generation_budget_exhausted,
    }


def parse_generated_output(
    *,
    decoded_output: str,
    terminal_stop_token_id: int | None = None,
    generation_budget_exhausted: bool = False,
) -> dict[
    str,
    Any,
]:

    normalized = (
        decoded_output.strip()
    )


    if generation_budget_exhausted:

        return _invalid_output(
            decoded_output=
                normalized,
            invalid_reason=
                "generation_budget_exhausted",
            terminal_stop_token_id=
                terminal_stop_token_id,
            generation_budget_exhausted=
                True,
        )


    try:

        payload = json.loads(
            normalized
        )

    except Exception:

        return _invalid_output(
            decoded_output=
                normalized,
            invalid_reason=
                "invalid_json",
            terminal_stop_token_id=
                terminal_stop_token_id,
            generation_budget_exhausted=
                False,
        )


    if not isinstance(
        payload,
        dict,
    ):

        return _invalid_output(
            decoded_output=
                normalized,
            invalid_reason=
                "json_value_not_object",
            terminal_stop_token_id=
                terminal_stop_token_id,
            generation_budget_exhausted=
                False,
        )


    if (
        set(payload)
        !=
        {
            "relation",
            "reason",
        }
    ):

        return _invalid_output(
            decoded_output=
                normalized,
            invalid_reason=
                "invalid_key_set",
            terminal_stop_token_id=
                terminal_stop_token_id,
            generation_budget_exhausted=
                False,
        )


    relation = payload[
        "relation"
    ]

    reason = payload[
        "reason"
    ]


    if (
        relation
        not in
        EXPECTED_RELATIONS
    ):

        return _invalid_output(
            decoded_output=
                normalized,
            invalid_reason=
                "invalid_relation",
            terminal_stop_token_id=
                terminal_stop_token_id,
            generation_budget_exhausted=
                False,
        )


    if not isinstance(
        reason,
        str,
    ):

        return _invalid_output(
            decoded_output=
                normalized,
            invalid_reason=
                "reason_not_string",
            terminal_stop_token_id=
                terminal_stop_token_id,
            generation_budget_exhausted=
                False,
        )


    word_count = len(
        reason.split()
    )


    if word_count < 6:

        return _invalid_output(
            decoded_output=
                normalized,
            invalid_reason=
                "reason_too_short",
            terminal_stop_token_id=
                terminal_stop_token_id,
            generation_budget_exhausted=
                False,
        )


    if word_count > 45:

        return _invalid_output(
            decoded_output=
                normalized,
            invalid_reason=
                "reason_too_long",
            terminal_stop_token_id=
                terminal_stop_token_id,
            generation_budget_exhausted=
                False,
        )


    return {
        "strict_json_valid":
            True,
        "relation":
            relation,
        "reason":
            reason,
        "invalid_reason":
            None,
        "decoded_output":
            normalized,
        "decoded_output_sha256":
            hashlib.sha256(
                normalized.encode(
                    "utf-8"
                )
            ).hexdigest(),
        "terminal_stop_token_id":
            terminal_stop_token_id,
        "generation_budget_exhausted":
            False,
    }


def process_generated_token_ids(
    *,
    tokenizer: Any,
    generated_token_ids: Sequence[
        int
    ],
) -> dict[
    str,
    Any,
]:

    token_ids = [
        int(
            token_id
        )
        for token_id
        in generated_token_ids
    ]


    terminal_stop_token_id: (
        int
        |
        None
    ) = None


    if (
        token_ids
        and
        token_ids[-1]
        in
        EOS_TOKEN_IDS
    ):

        terminal_stop_token_id = (
            token_ids.pop()
        )


    generation_budget_exhausted = (
        len(
            token_ids
        )
        >=
        MAX_NEW_TOKENS
        and
        terminal_stop_token_id
        is None
    )


    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
    )


    return parse_generated_output(
        decoded_output=
            decoded,
        terminal_stop_token_id=
            terminal_stop_token_id,
        generation_budget_exhausted=
            generation_budget_exhausted,
    )


def generate_label_blind_case(
    *,
    model: Any,
    tokenizer: Any,
    case_identity: Mapping[
        str,
        Any,
    ],
    profile_index: Mapping[
        tuple[str, str],
        ColumnSemanticProfile,
    ],
    torch_module: Any,
) -> dict[
    str,
    Any,
]:

    user_message = (
        build_label_blind_user_message(
            case_identity=
                case_identity,
            profile_index=
                profile_index,
        )
    )


    messages = [
        {
            "role":
                "user",
            "content":
                user_message,
        }
    ]


    encoded = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )
    )


    input_ids = getattr(
        encoded,
        "input_ids",
        encoded,
    )


    if (
        getattr(
            input_ids,
            "ndim",
            None
        )
        !=
        2
    ):

        raise RuntimeError(
            "Expected rank-2 tokenized prompt."
        )


    input_ids = input_ids.to(
        model.device
    )


    attention_mask = (
        torch_module.ones_like(
            input_ids
        )
    )


    prompt_token_count = int(
        input_ids.shape[-1]
    )


    with torch_module.inference_mode():

        output = model.generate(
            input_ids=
                input_ids,
            attention_mask=
                attention_mask,
            do_sample=
                False,
            num_beams=
                1,
            max_new_tokens=
                MAX_NEW_TOKENS,
            eos_token_id=
                list(
                    EOS_TOKEN_IDS
                ),
            pad_token_id=
                PAD_TOKEN_ID,
        )


    generated = output[
        0,
        prompt_token_count:
    ]


    generated_ids = (
        generated
        .detach()
        .cpu()
        .tolist()
    )


    result = (
        process_generated_token_ids(
            tokenizer=
                tokenizer,
            generated_token_ids=
                generated_ids,
        )
    )


    result[
        "prompt_token_count"
    ] = (
        prompt_token_count
    )

    result[
        "generated_token_count"
    ] = len(
        generated_ids
    )


    return result


def cuda_barrier(
    *,
    torch_module: Any,
    minimum_free_cuda_bytes: int,
) -> dict[
    str,
    int,
]:

    if not torch_module.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for Greenhouse runtime preflight."
        )


    free_bytes, total_bytes = (
        torch_module.cuda.mem_get_info()
    )


    if (
        free_bytes
        <
        minimum_free_cuda_bytes
    ):

        raise RuntimeError(
            (
                "Insufficient free CUDA memory. "
                f"free={free_bytes / 1024**3:.2f} GiB "
                "required="
                f"{minimum_free_cuda_bytes / 1024**3:.2f} GiB"
            )
        )


    return {
        "free_bytes":
            int(
                free_bytes
            ),
        "total_bytes":
            int(
                total_bytes
            ),
    }


def load_base_model(
    *,
    torch_module: Any,
    authority: Any,
) -> Any:

    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )


    quantization = (
        authority.contract
        .quantization
    )


    if (
        quantization.load_in_4bit
        is not True
    ):
        raise RuntimeError(
            "Frozen model is no longer 4-bit."
        )


    if (
        quantization.quantization_type
        !=
        "nf4"
    ):
        raise RuntimeError(
            "Frozen quantization is no longer NF4."
        )


    if (
        quantization.use_double_quantization
        is not True
    ):
        raise RuntimeError(
            "Frozen double quantization changed."
        )


    if (
        quantization.compute_dtype
        !=
        "bfloat16"
    ):
        raise RuntimeError(
            "Frozen compute dtype changed."
        )


    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=
                torch_module.bfloat16,
        )
    )


    checkpoint_path = (
        local_text_checkpoint_path(
            authority=
                authority,
        )
    )


    model = (
        AutoModelForCausalLM
        .from_pretrained(
            str(
                checkpoint_path
            ),
            device_map={
                "":
                    0,
            },
            dtype=
                torch_module.bfloat16,
            quantization_config=
                quantization_config,
            trust_remote_code=
                False,
            local_files_only=
                True,
        )
    )


    model.eval()

    return model


def attach_adapter(
    *,
    model: Any,
) -> Any:

    from peft import (
        PeftModel,
    )


    adapted_model = (
        PeftModel
        .from_pretrained(
            model,
            str(
                ADAPTER_PATH
            ),
            is_trainable=False,
        )
    )


    adapted_model.eval()

    return adapted_model


def validate_tokenizer_authority(
    *,
    tokenizer: Any,
) -> None:

    if not isinstance(
        tokenizer.chat_template,
        str,
    ):
        raise RuntimeError(
            "Tokenizer has no chat template."
        )


    template_sha = hashlib.sha256(
        tokenizer.chat_template.encode(
            "utf-8"
        )
    ).hexdigest()


    if (
        template_sha
        !=
        EXPECTED_CHAT_TEMPLATE_SHA256
    ):
        raise RuntimeError(
            (
                "Chat-template SHA changed.\n"
                f"Expected: {EXPECTED_CHAT_TEMPLATE_SHA256}\n"
                f"Actual:   {template_sha}"
            )
        )


    if (
        tokenizer.pad_token_id
        !=
        PAD_TOKEN_ID
    ):
        raise RuntimeError(
            (
                "Tokenizer pad-token authority changed. "
                f"Expected={PAD_TOKEN_ID}, "
                f"actual={tokenizer.pad_token_id}"
            )
        )


def prepare_runtime_authority(
) -> tuple[
    Any,
    Any,
]:

    validate_non_protected_runner_authorities()


    authority = (
        validate_static_authority(
            repository_root_value=
                ROOT,
        )
    )


    tokenizer = (
        load_pinned_tokenizer(
            authority=
                authority,
        )
    )


    validate_tokenizer_authority(
        tokenizer=
            tokenizer,
    )


    return (
        authority,
        tokenizer,
    )


# IMPORTANT
# ------------------------------------------------------------
# This v0.1 source deliberately contains no function that opens
# Greenhouse protected dataset/case/independence material.
#
# Official single-use consumption will only be added/frozen
# after this deterministic input/runtime core has passed static
# and synthetic tests.
#
# Therefore importing or testing this module cannot consume
# Greenhouse.

from __future__ import annotations


import gc
import hashlib
import json


from pathlib import Path


from typing import Literal


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)




from app.model_lifecycle.trusted_artifact_resolver import (
    resolve_trusted_peft_adapter_artifact,
)


# ============================================================
# VERSION / FROZEN SMOKE PARAMETERS
# ============================================================


TRUSTED_PEFT_SMOKE_RULE_VERSION = (
    "trusted_peft_smoke_v0.1"
)


SMOKE_RANDOM_SEED = 42

SMOKE_MAX_NEW_TOKENS = 64

SMOKE_EOS_TOKEN_IDS = (
    1,
    106,
)

SMOKE_PAD_TOKEN_ID = 0

MINIMUM_FREE_CUDA_BYTES = (
    5
    *
    1024**3
)


# ============================================================
# RESULT
# ============================================================


class TrustedPEFTSmokeResult(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    artifact_id: str = Field(
        min_length=1,
    )


    provenance_id: str = Field(
        min_length=1,
    )


    experiment_id: str = Field(
        min_length=1,
    )


    base_model_repository: str = Field(
        min_length=1,
    )


    base_model_revision: str = Field(
        min_length=1,
    )


    base_model_checkpoint_path: str = Field(
        min_length=1,
    )


    adapter_path: str = Field(
        min_length=1,
    )


    prompt_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    output_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    generated_token_ids_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    decoded_output: str = Field(
        min_length=1,
    )


    prompt_token_count: int = Field(
        gt=0,
    )


    generated_token_count: int = Field(
        gt=0,
        le=SMOKE_MAX_NEW_TOKENS,
    )


    deterministic_repeat_equal: bool


    cuda_device_name: str = Field(
        min_length=1,
    )


    cuda_compute_capability: str = Field(
        min_length=1,
    )


    runtime_versions: dict[
        str,
        str,
    ]


    rule_version: Literal[
        "trusted_peft_smoke_v0.1"
    ] = (
        TRUSTED_PEFT_SMOKE_RULE_VERSION
    )


# ============================================================
# SYNTHETIC LABEL-BLIND PROMPT
# ============================================================


def smoke_prompt_record(
) -> dict[
    str,
    str,
]:

    return {
        "domain":
            "warehouse_fulfillment_runtime_smoke",

        "left_metric":
            "orders_packed",

        "left_description":
            (
                "Count of customer orders for which "
                "all requested items were packed and "
                "sealed during the reporting period."
            ),

        "right_metric":
            "orders_shipped",

        "right_description":
            (
                "Count of customer orders handed to "
                "a carrier and marked as shipped "
                "during the same reporting period."
            ),
    }


# ============================================================
# HELPERS
# ============================================================


def _sha256_text(
    value: str,
) -> str:

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256_token_ids(
    values: tuple[
        int,
        ...,
    ],
) -> str:

    encoded = json.dumps(
        list(
            values
        ),
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )


    return hashlib.sha256(
        encoded
    ).hexdigest()


def _load_adapter_config(
    adapter_path: Path,
) -> dict[
    str,
    object,
]:

    path = (
        adapter_path
        /
        "adapter_config.json"
    )


    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            (
                "Trusted adapter_config.json "
                "could not be loaded."
            )
        ) from error


    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "adapter_config.json must be an object."
        )


    return payload


def _validate_adapter_config(
    *,
    adapter_config: dict[
        str,
        object,
    ],
    checkpoint: Path,
    authority,
    expected_target_suffixes: frozenset[
        str
    ],
) -> None:

    raw_base = adapter_config.get(
        "base_model_name_or_path"
    )


    if not isinstance(
        raw_base,
        str,
    ):
        raise RuntimeError(
            (
                "PEFT adapter has no "
                "base_model_name_or_path."
            )
        )


    adapter_base = (
        Path(
            raw_base
        )
        .expanduser()
        .resolve()
    )


    if (
        adapter_base
        !=
        checkpoint
    ):
        raise RuntimeError(
            (
                "PEFT adapter base-model path "
                "does not match frozen text checkpoint."
            )
        )


    if (
        str(
            adapter_config.get(
                "peft_type"
            )
        ).upper()
        !=
        "LORA"
    ):
        raise RuntimeError(
            "PEFT type is not LORA."
        )


    if (
        adapter_config.get(
            "task_type"
        )
        !=
        "CAUSAL_LM"
    ):
        raise RuntimeError(
            "PEFT task type is not CAUSAL_LM."
        )


    lora = (
        authority.contract.lora
    )


    required = {
        "r":
            lora.rank,

        "lora_alpha":
            lora.alpha,

        "lora_dropout":
            lora.dropout,

        "bias":
            lora.bias,

        "inference_mode":
            True,
    }


    for key, expected in (
        required.items()
    ):

        if (
            adapter_config.get(
                key
            )
            !=
            expected
        ):
            raise RuntimeError(
                (
                    "PEFT adapter configuration "
                    f"changed: {key}"
                )
            )


    targets = adapter_config.get(
        "target_modules"
    )


    if not isinstance(
        targets,
        list,
    ):
        raise RuntimeError(
            "PEFT target_modules missing."
        )


    if (
        frozenset(
            str(
                value
            )
            for value
            in targets
        )
        !=
        expected_target_suffixes
    ):
        raise RuntimeError(
            (
                "PEFT target-module surface "
                "does not match frozen QLoRA runtime."
            )
        )


def _generate_once(
    *,
    model,
    tokenizer,
    torch_module,
    input_ids,
    attention_mask,
    prompt_token_count: int,
) -> tuple[
    int,
    ...,
]:

    with torch_module.inference_mode():

        generated = (
            model.generate(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,

                do_sample=
                    False,

                num_beams=
                    1,

                max_new_tokens=
                    SMOKE_MAX_NEW_TOKENS,

                eos_token_id=
                    list(
                        SMOKE_EOS_TOKEN_IDS
                    ),

                pad_token_id=
                    SMOKE_PAD_TOKEN_ID,
            )
        )


    if (
        getattr(
            generated,
            "ndim",
            None,
        )
        !=
        2
        or
        int(
            generated.shape[
                0
            ]
        )
        !=
        1
    ):
        raise RuntimeError(
            "Unexpected generate() output shape."
        )


    values = tuple(
        int(
            value
        )

        for value
        in (
            generated[
                0,
                prompt_token_count:
            ]
            .detach()
            .cpu()
            .tolist()
        )
    )


    if (
        len(
            values
        )
        < 1
        or
        len(
            values
        )
        >
        SMOKE_MAX_NEW_TOKENS
    ):
        raise RuntimeError(
            (
                "Unexpected deterministic smoke "
                "generation length."
            )
        )


    return values


# ============================================================
# TRUSTED LOAD + SMOKE
# ============================================================


def run_trusted_peft_smoke(
    artifact_id: str,
    *,
    registry_path: Path | str | None = None,
    api_root: Path | str | None = None,
) -> TrustedPEFTSmokeResult:
    """
    Resolve, verify, load and smoke-test one registered PEFT
    adapter.

    The trusted artifact resolver executes before any heavy
    model runtime is imported or any model/adapter is loaded.
    """

    # --------------------------------------------------------
    # 1. TRUST FIRST
    # --------------------------------------------------------


    trusted = (
        resolve_trusted_peft_adapter_artifact(
            artifact_id,
            registry_path=
                registry_path,

            api_root=
                api_root,
        )
    )

    # --------------------------------------------------------
    # 2. ADAPTATION AUTHORITIES ONLY AFTER TRUST
    # --------------------------------------------------------


    from app.adaptation.qlora_runtime_v0_4 import (
        BASE_MODEL_REPOSITORY,
        BASE_MODEL_REVISION,
        EXPECTED_CHAT_TEMPLATE_SHA256,
        EXPECTED_TARGET_SUFFIXES,
        load_pinned_tokenizer,
        local_text_checkpoint_path,
        runtime_versions,
        validate_static_authority,
    )

    from app.adaptation.training_dataset_canonicalizer_v0_4 import (
        build_user_message,
    )



    root = (
        Path(
            api_root
        ).resolve()

        if api_root is not None

        else (
            Path(
                __file__
            )
            .resolve()
            .parents[
                2
            ]
        )
    )


    # --------------------------------------------------------
    # 2. FROZEN NON-PROTECTED AUTHORITIES
    # --------------------------------------------------------


    authority = (
        validate_static_authority(
            repository_root_value=
                root,
        )
    )


    versions = (
        runtime_versions()
    )


    if (
        authority.contract
        .base_model
        .repository
        !=
        BASE_MODEL_REPOSITORY
    ):
        raise RuntimeError(
            "Frozen base-model repository changed."
        )


    if (
        authority.contract
        .base_model
        .revision
        !=
        BASE_MODEL_REVISION
    ):
        raise RuntimeError(
            "Frozen base-model revision changed."
        )


    checkpoint = (
        local_text_checkpoint_path()
        .resolve()
    )


    if not checkpoint.is_dir():
        raise RuntimeError(
            (
                "Frozen converted text-only "
                "checkpoint is missing."
            )
        )


    adapter_path = (
        Path(
            trusted.physical_adapter_path
        )
        .resolve()
    )


    adapter_config = (
        _load_adapter_config(
            adapter_path
        )
    )


    _validate_adapter_config(
        adapter_config=
            adapter_config,

        checkpoint=
            checkpoint,

        authority=
            authority,

        expected_target_suffixes=
            EXPECTED_TARGET_SUFFIXES,
    )


    # --------------------------------------------------------
    # 3. HEAVY RUNTIME IMPORTS ONLY AFTER TRUST
    # --------------------------------------------------------


    import torch


    from peft import (
        PeftModel,
    )


    from transformers import (
        BitsAndBytesConfig,
        Gemma3ForCausalLM,
    )


    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for trusted PEFT smoke."
        )


    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "CUDA BF16 support is required."
        )


    torch.cuda.empty_cache()


    (
        free_bytes,
        total_bytes,
    ) = (
        torch.cuda.mem_get_info()
    )


    if (
        free_bytes
        <
        MINIMUM_FREE_CUDA_BYTES
    ):
        raise RuntimeError(
            (
                "Insufficient free CUDA memory. "
                f"free={free_bytes / 1024**3:.2f} GiB "
                "required="
                f"{MINIMUM_FREE_CUDA_BYTES / 1024**3:.2f} GiB"
            )
        )


    torch.manual_seed(
        SMOKE_RANDOM_SEED
    )

    torch.cuda.manual_seed_all(
        SMOKE_RANDOM_SEED
    )

    torch.cuda.reset_peak_memory_stats()


    tokenizer = None
    base_model = None
    adapted_model = None


    try:

        # ----------------------------------------------------
        # 4. PINNED TOKENIZER
        # ----------------------------------------------------


        tokenizer = (
            load_pinned_tokenizer(
                authority=
                    authority,
            )
        )


        tokenizer.padding_side = (
            "right"
        )


        if (
            tokenizer.pad_token_id
            !=
            SMOKE_PAD_TOKEN_ID
        ):
            raise RuntimeError(
                "Pinned tokenizer PAD changed."
            )


        if not isinstance(
            tokenizer.chat_template,
            str,
        ):
            raise RuntimeError(
                "Pinned tokenizer chat template missing."
            )


        chat_template_sha = (
            hashlib.sha256(
                tokenizer.chat_template.encode(
                    "utf-8"
                )
            )
            .hexdigest()
        )


        if (
            chat_template_sha
            !=
            EXPECTED_CHAT_TEMPLATE_SHA256
        ):
            raise RuntimeError(
                "Pinned chat-template SHA changed."
            )


        # ----------------------------------------------------
        # 5. EXACT INFERENCE BASE MODEL
        # ----------------------------------------------------


        quantization = (
            authority.contract.quantization
        )


        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=
                    quantization.load_in_4bit,

                bnb_4bit_quant_type=
                    quantization.quantization_type,

                bnb_4bit_use_double_quant=
                    quantization.use_double_quantization,

                bnb_4bit_compute_dtype=
                    torch.bfloat16,
            )
        )


        base_model = (
            Gemma3ForCausalLM
            .from_pretrained(
                str(
                    checkpoint
                ),

                quantization_config=
                    quantization_config,

                device_map={
                    "":
                        0,
                },

                dtype=
                    torch.bfloat16,

                trust_remote_code=
                    False,

                local_files_only=
                    True,
            )
        )


        base_model.eval()


        # ----------------------------------------------------
        # 6. EXACT TRUSTED ADAPTER
        # ----------------------------------------------------


        adapted_model = (
            PeftModel
            .from_pretrained(
                base_model,
                str(
                    adapter_path
                ),
                is_trainable=
                    False,
            )
        )


        adapted_model.eval()


        if any(
            parameter.requires_grad

            for parameter
            in adapted_model.parameters()
        ):
            raise RuntimeError(
                (
                    "Inference adapter unexpectedly "
                    "contains trainable parameters."
                )
            )


        # ----------------------------------------------------
        # 7. LABEL-BLIND SYNTHETIC PROMPT
        # ----------------------------------------------------


        prompt_record = (
            smoke_prompt_record()
        )


        user_message = (
            build_user_message(
                prompt_record
            )
        )


        prompt_sha = (
            _sha256_text(
                user_message
            )
        )


        input_ids = (
            tokenizer.apply_chat_template(
                [
                    {
                        "role":
                            "user",

                        "content":
                            user_message,
                    }
                ],

                tokenize=
                    True,

                add_generation_prompt=
                    True,

                truncation=
                    False,

                return_tensors=
                    "pt",

                return_dict=
                    False,
            )
        )


        if (
            getattr(
                input_ids,
                "ndim",
                None,
            )
            !=
            2
            or
            int(
                input_ids.shape[
                    0
                ]
            )
            !=
            1
        ):
            raise RuntimeError(
                "Unexpected prompt tensor shape."
            )


        input_ids = (
            input_ids.to(
                "cuda"
            )
        )


        attention_mask = (
            torch.ones_like(
                input_ids,
                dtype=
                    torch.long,
                device=
                    "cuda",
            )
        )


        prompt_token_count = int(
            input_ids.shape[
                1
            ]
        )


        model_limit = int(
            getattr(
                adapted_model.config,
                "max_position_embeddings",
                0,
            )
        )


        if model_limit <= 0:
            raise RuntimeError(
                "Model context limit unavailable."
            )


        if (
            prompt_token_count
            +
            SMOKE_MAX_NEW_TOKENS
            >
            model_limit
        ):
            raise RuntimeError(
                (
                    "Synthetic smoke prompt exceeds "
                    "model context."
                )
            )


        # ----------------------------------------------------
        # 8. TWO GREEDY PASSES
        # ----------------------------------------------------


        first_ids = (
            _generate_once(
                model=
                    adapted_model,

                tokenizer=
                    tokenizer,

                torch_module=
                    torch,

                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,

                prompt_token_count=
                    prompt_token_count,
            )
        )


        second_ids = (
            _generate_once(
                model=
                    adapted_model,

                tokenizer=
                    tokenizer,

                torch_module=
                    torch,

                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,

                prompt_token_count=
                    prompt_token_count,
            )
        )


        if (
            first_ids
            !=
            second_ids
        ):
            raise RuntimeError(
                (
                    "Repeated greedy smoke generation "
                    "was not token-identical."
                )
            )


        body_ids = (
            first_ids[
                :-1
            ]

            if (
                first_ids
                and
                first_ids[
                    -1
                ]
                in
                SMOKE_EOS_TOKEN_IDS
            )

            else
            first_ids
        )


        decoded_output = (
            tokenizer.decode(
                list(
                    body_ids
                ),
                skip_special_tokens=
                    False,

                clean_up_tokenization_spaces=
                    False,
            )
            .strip()
        )


        if not decoded_output:
            raise RuntimeError(
                "Synthetic smoke output is empty."
            )


        output_sha = (
            _sha256_text(
                decoded_output
            )
        )


        token_ids_sha = (
            _sha256_token_ids(
                first_ids
            )
        )


        device_name = (
            torch.cuda.get_device_name(
                0
            )
        )


        capability = (
            torch.cuda.get_device_capability(
                0
            )
        )


        return TrustedPEFTSmokeResult(
            artifact_id=
                trusted.artifact_id,

            provenance_id=
                trusted.provenance_id,

            experiment_id=
                trusted.experiment_id,

            base_model_repository=
                BASE_MODEL_REPOSITORY,

            base_model_revision=
                BASE_MODEL_REVISION,

            base_model_checkpoint_path=
                str(
                    checkpoint
                ),

            adapter_path=
                str(
                    adapter_path
                ),

            prompt_sha256=
                prompt_sha,

            output_sha256=
                output_sha,

            generated_token_ids_sha256=
                token_ids_sha,

            decoded_output=
                decoded_output,

            prompt_token_count=
                prompt_token_count,

            generated_token_count=
                len(
                    first_ids
                ),

            deterministic_repeat_equal=
                True,

            cuda_device_name=
                device_name,

            cuda_compute_capability=(
                f"{capability[0]}.{capability[1]}"
            ),

            runtime_versions=
                versions,
        )


    finally:

        adapted_model = None
        base_model = None
        tokenizer = None

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

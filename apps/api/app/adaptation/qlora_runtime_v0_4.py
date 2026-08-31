from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from app.adaptation.assistant_masking import (
    ASSISTANT_ONLY_MASKING_RULE_VERSION,
    build_assistant_only_training_example,
    collate_assistant_only_examples,
)
from app.adaptation.contracts import (
    QLoRAExperimentContract,
)
from app.adaptation.target_resolver import (
    resolve_qlora_target_modules,
)


QLORA_V04_SHARED_RUNTIME_RULE_VERSION = (
    "qlora_v0.4_shared_runtime_v0.1"
)


# ============================================================
# FROZEN AUTHORITIES
# ============================================================


EXPERIMENT_ID = (
    "datalens-semantic-qlora-v0.4"
)


CONTRACT_RELATIVE_PATH = (
    "artifacts/adaptation/experiments/"
    "datalens_semantic_qlora_v0.4_contract.json"
)


CONTRACT_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/experiments/"
    "datalens_semantic_qlora_v0.4_contract_freeze.json"
)


OPTIMIZATION_POLICY_RELATIVE_PATH = (
    "artifacts/adaptation/design/"
    "datalens_semantic_qlora_v0.4_"
    "optimization_policy_v0.1.json"
)


TOKEN_AUDIT_RELATIVE_PATH = (
    "artifacts/adaptation/probes/"
    "datalens_semantic_training_v0.4_"
    "token_length_audit.json"
)


DATASET_RELATIVE_PATH = (
    "artifacts/adaptation/datasets/"
    "datalens_semantic_training_v0.4.jsonl"
)


EXPECTED_CONTRACT_SHA256 = (
    "22f3c38d4165dc34d9a210605ebfb65c"
    "1c50c5839ed6607a21206fbe8313a287"
)


EXPECTED_CONTRACT_FREEZE_SHA256 = (
    "922cd45c7a84d1c5da54a86ede3b6792"
    "d4abe14b48fbbdd485cf737fcd148321"
)


EXPECTED_OPTIMIZATION_POLICY_SHA256 = (
    "01a8fc993a8699e1ae6511f5ce73c642"
    "c7b1c1bf1d974b147ba5e6542d48824d"
)


EXPECTED_TOKEN_AUDIT_SHA256 = (
    "add94bd50fc89120a7626fa7533af299"
    "6765d25c7c64d2d866dd7178674d40c1"
)


EXPECTED_DATASET_SHA256 = (
    "4fd00586f2d53d6de57f5cbc5f1d7bfb"
    "2e512960e60b30c28596aaefbac322b7"
)


# ============================================================
# MODEL / TOKENIZER
# ============================================================


BASE_MODEL_REPOSITORY = (
    "google/gemma-3-4b-it"
)


BASE_MODEL_REVISION = (
    "093f9f388b31de276ce2de164bdc2081324b9767"
)


EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "7de1c58e208eda46e9c7f86397df37ec"
    "49883aeece39fb961e0a6b24088dd3c4"
)


LOCAL_TEXT_CHECKPOINT_DIRECTORY = (
    "gemma-3-4b-it-text-"
    "093f9f388b31de276ce2de164bdc2081324b9767"
)


# ============================================================
# EXACT MODEL SURFACE
# ============================================================


EXPECTED_TARGET_COUNT = 238

EXPECTED_TARGET_SUFFIXES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)


EXPECTED_TRAINABLE_PARAMETERS = 29_802_496

EXPECTED_TRAINABLE_TENSORS = 476


# ============================================================
# DATASET / MASKING
# ============================================================


EXPECTED_EXAMPLE_COUNT = 230

EXPECTED_MAX_SEQUENCE_LENGTH = 256

EXPECTED_TOTAL_FULL_TOKENS = 43_799

EXPECTED_TOTAL_SUPERVISED_TOKENS = 7_821

EXPECTED_MAX_EXAMPLE_TOKENS = 206


# ============================================================
# RUNTIME VERSIONS
# ============================================================


EXPECTED_RUNTIME_VERSIONS = {
    "torch":
        "2.11.0+cu128",

    "transformers":
        "5.16.1",

    "peft":
        "0.20.0",

    "bitsandbytes":
        "0.50.2",
}


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass(
    frozen=True
)
class RuntimeAuthority:
    contract: QLoRAExperimentContract

    contract_payload: Mapping[str, Any]

    contract_freeze: Mapping[str, Any]

    optimization_policy: Mapping[str, Any]

    token_audit: Mapping[str, Any]

    dataset_path: Path


@dataclass(
    frozen=True
)
class PreparedDataset:
    records: tuple[Mapping[str, Any], ...]

    examples: tuple[Any, ...]

    total_full_tokens: int

    total_supervised_tokens: int

    max_example_tokens: int


@dataclass(
    frozen=True
)
class PreparedQLoRAModel:
    model: Any

    target_modules: tuple[str, ...]

    trainable_parameters: tuple[Any, ...]

    trainable_names: tuple[str, ...]

    trainable_parameter_count: int

    trainable_tensor_count: int


# ============================================================
# GENERIC HELPERS
# ============================================================


def repository_root(
    value: Path | None = None,
) -> Path:
    root = (
        Path.cwd()
        if value is None
        else value
    )

    root = root.expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(
            root
        )

    return root


def sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def require_file_sha256(
    *,
    repository_root_value: Path,
    relative_path: str,
    expected_sha256: str,
) -> Path:
    path = (
        repository_root_value
        /
        relative_path
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual = sha256_file(
        path
    )

    if actual != expected_sha256:
        raise RuntimeError(
            (
                "Frozen authority SHA mismatch.\n"
                f"Path:     {relative_path}\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual}"
            )
        )

    return path


def local_text_checkpoint_path() -> Path:
    return (
        Path.home()
        /
        ".cache"
        /
        "datalens"
        /
        "adaptation"
        /
        "base-models"
        /
        LOCAL_TEXT_CHECKPOINT_DIRECTORY
    ).resolve()


# ============================================================
# STATIC AUTHORITY
# ============================================================


def runtime_versions() -> Dict[str, str]:
    versions: Dict[
        str,
        str,
    ] = {}

    for (
        distribution,
        expected,
    ) in EXPECTED_RUNTIME_VERSIONS.items():
        actual = (
            importlib.metadata.version(
                distribution
            )
        )

        if actual != expected:
            raise RuntimeError(
                (
                    "Runtime dependency version changed.\n"
                    f"{distribution}: "
                    f"expected={expected!r}, "
                    f"actual={actual!r}"
                )
            )

        versions[
            distribution
        ] = actual

    return versions


def validate_static_authority(
    *,
    repository_root_value: Path,
) -> RuntimeAuthority:
    root = repository_root(
        repository_root_value
    )

    contract_path = (
        require_file_sha256(
            repository_root_value=
                root,

            relative_path=
                CONTRACT_RELATIVE_PATH,

            expected_sha256=
                EXPECTED_CONTRACT_SHA256,
        )
    )

    contract_freeze_path = (
        require_file_sha256(
            repository_root_value=
                root,

            relative_path=
                CONTRACT_FREEZE_RELATIVE_PATH,

            expected_sha256=
                EXPECTED_CONTRACT_FREEZE_SHA256,
        )
    )

    optimization_path = (
        require_file_sha256(
            repository_root_value=
                root,

            relative_path=
                OPTIMIZATION_POLICY_RELATIVE_PATH,

            expected_sha256=
                EXPECTED_OPTIMIZATION_POLICY_SHA256,
        )
    )

    token_audit_path = (
        require_file_sha256(
            repository_root_value=
                root,

            relative_path=
                TOKEN_AUDIT_RELATIVE_PATH,

            expected_sha256=
                EXPECTED_TOKEN_AUDIT_SHA256,
        )
    )

    dataset_path = (
        require_file_sha256(
            repository_root_value=
                root,

            relative_path=
                DATASET_RELATIVE_PATH,

            expected_sha256=
                EXPECTED_DATASET_SHA256,
        )
    )

    contract_payload = (
        load_json(
            contract_path
        )
    )

    contract = (
        QLoRAExperimentContract
        .model_validate(
            contract_payload
        )
    )

    freeze = load_json(
        contract_freeze_path
    )

    optimization = load_json(
        optimization_path
    )

    token_audit = load_json(
        token_audit_path
    )

    if (
        contract.experiment_id
        !=
        EXPERIMENT_ID
    ):
        raise RuntimeError(
            "Experiment ID mismatch."
        )

    if (
        contract.base_model.repository
        !=
        BASE_MODEL_REPOSITORY
    ):
        raise RuntimeError(
            "Base-model repository changed."
        )

    if (
        contract.base_model.revision
        !=
        BASE_MODEL_REVISION
    ):
        raise RuntimeError(
            "Base-model revision changed."
        )

    if (
        contract.base_model.tokenizer_revision
        !=
        BASE_MODEL_REVISION
    ):
        raise RuntimeError(
            "Tokenizer revision changed."
        )

    if (
        contract.base_model.modality_scope
        !=
        "text_only"
    ):
        raise RuntimeError(
            "Base-model modality scope changed."
        )

    if (
        contract.base_model.use_multimodal_inputs
        is not False
    ):
        raise RuntimeError(
            "Multimodal inputs became enabled."
        )

    if (
        contract.base_model.trust_remote_code
        is not False
    ):
        raise RuntimeError(
            "trust_remote_code became enabled."
        )

    if (
        freeze[
            "contract_sha256"
        ]
        !=
        EXPECTED_CONTRACT_SHA256
    ):
        raise RuntimeError(
            "Contract-freeze binding changed."
        )

    if (
        freeze[
            "frozen_before_resource_preflight"
        ]
        is not True
    ):
        raise RuntimeError(
            "Contract not frozen before preflight."
        )

    if (
        freeze[
            "frozen_before_training"
        ]
        is not True
    ):
        raise RuntimeError(
            "Contract not frozen before training."
        )

    if (
        freeze[
            "training_started_at_contract_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "Training had started at contract freeze."
        )

    if (
        freeze[
            "airport_evaluated_at_contract_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "Airport had been evaluated."
        )

    if (
        freeze[
            "final_acceptance_loaded_at_contract_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance had been loaded."
        )

    if (
        optimization[
            "training"
        ][
            "max_sequence_length"
        ]
        !=
        EXPECTED_MAX_SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            "Optimization sequence length changed."
        )

    if (
        optimization[
            "training"
        ][
            "assistant_only_loss"
        ]
        is not True
    ):
        raise RuntimeError(
            "Assistant-only loss changed."
        )

    if (
        optimization[
            "training"
        ][
            "gradient_checkpoint_use_reentrant"
        ]
        is not False
    ):
        raise RuntimeError(
            "Gradient-checkpoint reentrant policy changed."
        )

    if (
        optimization[
            "training"
        ][
            "micro_batch_size"
        ]
        !=
        1
    ):
        raise RuntimeError(
            "Micro-batch size changed."
        )

    if (
        optimization[
            "training"
        ][
            "gradient_accumulation_steps"
        ]
        !=
        8
    ):
        raise RuntimeError(
            "Gradient accumulation changed."
        )

    accumulation = (
        optimization[
            "accumulation"
        ]
    )

    required_accumulation = {
        "micro_batches_per_epoch":
            230,

        "full_groups_per_epoch":
            28,

        "partial_group_size":
            6,

        "optimizer_steps_per_epoch":
            29,

        "total_micro_batches":
            460,

        "total_optimizer_steps":
            58,

        "example_presentations":
            460,

        "discarded_example_presentations":
            0,

        "cross_epoch_accumulation":
            False,

        "policy":
            "flush_partial_group_at_epoch_end",
    }

    for key, expected in required_accumulation.items():
        if (
            accumulation[
                key
            ]
            !=
            expected
        ):
            raise RuntimeError(
                f"Accumulation authority changed: {key}"
            )

    if (
        token_audit[
            "dataset"
        ][
            "dataset_sha256"
        ]
        !=
        EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "Token audit dataset binding changed."
        )

    if (
        token_audit[
            "dataset"
        ][
            "example_count"
        ]
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Token audit example count changed."
        )

    if (
        token_audit[
            "tokenizer"
        ][
            "chat_template_sha256"
        ]
        !=
        EXPECTED_CHAT_TEMPLATE_SHA256
    ):
        raise RuntimeError(
            "Token-audit chat template changed."
        )

    if (
        token_audit[
            "recommendation"
        ][
            "recommended_max_sequence_length"
        ]
        !=
        EXPECTED_MAX_SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            "Token audit no longer recommends seq=256."
        )

    if (
        token_audit[
            "recommendation"
        ][
            "truncated_examples"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Token audit is no longer lossless."
        )

    if (
        ASSISTANT_ONLY_MASKING_RULE_VERSION
        !=
        "assistant_only_masking_v0.1"
    ):
        raise RuntimeError(
            "Assistant masking rule changed."
        )

    runtime_versions()

    return RuntimeAuthority(
        contract=
            contract,

        contract_payload=
            contract_payload,

        contract_freeze=
            freeze,

        optimization_policy=
            optimization,

        token_audit=
            token_audit,

        dataset_path=
            dataset_path,
    )


# ============================================================
# TOKENIZER
# ============================================================


def load_pinned_tokenizer(
    *,
    authority: RuntimeAuthority,
) -> Any:
    from transformers import (
        AutoTokenizer,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            authority.contract
            .base_model
            .repository,

            revision=
                authority.contract
                .base_model
                .tokenizer_revision,

            trust_remote_code=
                False,

            local_files_only=
                True,
        )
    )

    if not isinstance(
        tokenizer.chat_template,
        str,
    ):
        raise RuntimeError(
            "Pinned tokenizer has no chat template."
        )

    template_sha = sha256_bytes(
        tokenizer.chat_template.encode(
            "utf-8"
        )
    )

    if (
        template_sha
        !=
        EXPECTED_CHAT_TEMPLATE_SHA256
    ):
        raise RuntimeError(
            (
                "Tokenizer chat-template SHA changed.\n"
                f"Expected: {EXPECTED_CHAT_TEMPLATE_SHA256}\n"
                f"Actual:   {template_sha}"
            )
        )

    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Tokenizer has no pad token."
        )

    return tokenizer


# ============================================================
# DATASET
# ============================================================


def prepare_training_dataset(
    *,
    authority: RuntimeAuthority,
    tokenizer: Any,
) -> PreparedDataset:
    records = []

    with authority.dataset_path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue

            records.append(
                json.loads(
                    line
                )
            )

    if (
        len(
            records
        )
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Training example count changed."
        )

    examples = []

    for record in records:
        messages = record.get(
            "messages"
        )

        if not isinstance(
            messages,
            list,
        ):
            raise RuntimeError(
                "Training record has no messages list."
            )

        example = (
            build_assistant_only_training_example(
                tokenizer=
                    tokenizer,

                messages=
                    messages,

                max_sequence_length=
                    EXPECTED_MAX_SEQUENCE_LENGTH,
            )
        )

        examples.append(
            example
        )

    total_full_tokens = sum(
        example.total_token_count
        for example
        in examples
    )

    total_supervised_tokens = sum(
        example.supervised_token_count
        for example
        in examples
    )

    max_example_tokens = max(
        example.total_token_count
        for example
        in examples
    )

    if (
        total_full_tokens
        !=
        EXPECTED_TOTAL_FULL_TOKENS
    ):
        raise RuntimeError(
            (
                "Full-token total changed.\n"
                f"Expected: {EXPECTED_TOTAL_FULL_TOKENS}\n"
                f"Actual:   {total_full_tokens}"
            )
        )

    if (
        total_supervised_tokens
        !=
        EXPECTED_TOTAL_SUPERVISED_TOKENS
    ):
        raise RuntimeError(
            (
                "Supervised-token total changed.\n"
                f"Expected: {EXPECTED_TOTAL_SUPERVISED_TOKENS}\n"
                f"Actual:   {total_supervised_tokens}"
            )
        )

    if (
        max_example_tokens
        !=
        EXPECTED_MAX_EXAMPLE_TOKENS
    ):
        raise RuntimeError(
            (
                "Maximum example length changed.\n"
                f"Expected: {EXPECTED_MAX_EXAMPLE_TOKENS}\n"
                f"Actual:   {max_example_tokens}"
            )
        )

    # build_assistant_only_training_example() is
    # fail-closed when a rendered example exceeds the
    # configured sequence limit. AssistantOnlyTrainingExample
    # therefore has no synthetic ``truncated`` attribute.
    #
    # The frozen token audit independently proves that the
    # v0.4 dataset requires zero truncation. At runtime we
    # preserve that invariant through the exact observed
    # maximum token count.
    if (
        max_example_tokens
        >
        EXPECTED_MAX_SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            "Training example exceeds sequence limit."
        )

    return PreparedDataset(
        records=
            tuple(
                records
            ),

        examples=
            tuple(
                examples
            ),

        total_full_tokens=
            total_full_tokens,

        total_supervised_tokens=
            total_supervised_tokens,

        max_example_tokens=
            max_example_tokens,
    )


def longest_training_example_index(
    prepared:
        PreparedDataset,
) -> int:
    if not prepared.examples:
        raise RuntimeError(
            "No prepared training examples."
        )

    return max(
        range(
            len(
                prepared.examples
            )
        ),
        key=lambda index:
            (
                prepared.examples[
                    index
                ].total_token_count,
                prepared.examples[
                    index
                ].supervised_token_count,
                -index,
            ),
    )


# ============================================================
# MODEL LOAD + LORA ATTACH
# ============================================================


def prepare_qlora_model(
    *,
    authority: RuntimeAuthority,
) -> PreparedQLoRAModel:
    import torch

    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    from transformers import (
        BitsAndBytesConfig,
        Gemma3ForCausalLM,
    )

    checkpoint = (
        local_text_checkpoint_path()
    )

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            (
                "Converted text-only checkpoint not found: "
                f"{checkpoint}"
            )
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "CUDA device does not support BF16."
        )

    quantization = (
        authority.contract
        .quantization
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

    model = (
        Gemma3ForCausalLM.from_pretrained(
            checkpoint,

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

    model.config.use_cache = False

    model = (
        prepare_model_for_kbit_training(
            model,

            use_gradient_checkpointing=
                True,

            gradient_checkpointing_kwargs={
                "use_reentrant":
                    False,
            },
        )
    )

    model.config.use_cache = False

    resolution = (
        resolve_qlora_target_modules(
            model=
                model,

            base_model=
                authority.contract
                .base_model,

            lora=
                authority.contract
                .lora,
        )
    )

    target_modules = tuple(
        resolution.target_modules
    )

    if (
        len(
            target_modules
        )
        !=
        EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            (
                "Unexpected LoRA target count.\n"
                f"Expected: {EXPECTED_TARGET_COUNT}\n"
                f"Actual:   {len(target_modules)}"
            )
        )

    suffixes = frozenset(
        target.split(
            "."
        )[
            -1
        ]

        for target
        in target_modules
    )

    if (
        suffixes
        !=
        EXPECTED_TARGET_SUFFIXES
    ):
        raise RuntimeError(
            (
                "Unexpected LoRA target surface: "
                f"{sorted(suffixes)}"
            )
        )

    for target in target_modules:
        lowered = (
            target.casefold()
        )

        if any(
            fragment in lowered

            for fragment in (
                "vision",
                "projector",
                "multi_modal",
                "lm_head",
            )
        ):
            raise RuntimeError(
                f"Forbidden LoRA target: {target}"
            )

    lora = (
        authority.contract
        .lora
    )

    lora_config = (
        LoraConfig(
            r=
                lora.rank,

            lora_alpha=
                lora.alpha,

            lora_dropout=
                lora.dropout,

            bias=
                lora.bias,

            task_type=
                lora.task_type,

            target_modules=
                list(
                    target_modules
                ),
        )
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    trainable_parameters = tuple(
        parameter

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )

    trainable_names = tuple(
        name

        for (
            name,
            parameter,
        ) in model.named_parameters()

        if parameter.requires_grad
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in trainable_parameters
    )

    tensor_count = len(
        trainable_parameters
    )

    if (
        parameter_count
        !=
        EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError(
            (
                "Unexpected trainable parameter count.\n"
                f"Expected: {EXPECTED_TRAINABLE_PARAMETERS}\n"
                f"Actual:   {parameter_count}"
            )
        )

    if (
        tensor_count
        !=
        EXPECTED_TRAINABLE_TENSORS
    ):
        raise RuntimeError(
            (
                "Unexpected trainable tensor count.\n"
                f"Expected: {EXPECTED_TRAINABLE_TENSORS}\n"
                f"Actual:   {tensor_count}"
            )
        )

    if any(
        "lora_"
        not in
        name.casefold()

        for name
        in trainable_names
    ):
        raise RuntimeError(
            "Non-LoRA trainable parameter detected."
        )

    model.train()

    return PreparedQLoRAModel(
        model=
            model,

        target_modules=
            target_modules,

        trainable_parameters=
            trainable_parameters,

        trainable_names=
            trainable_names,

        trainable_parameter_count=
            parameter_count,

        trainable_tensor_count=
            tensor_count,
    )


# ============================================================
# TENSOR BATCH
# ============================================================


def tensor_batch_from_example(
    *,
    example: Any,
    tokenizer: Any,
    torch_module: Any,
    device: Any,
) -> Dict[str, Any]:
    batch = (
        collate_assistant_only_examples(
            examples=[
                example
            ],

            pad_token_id=
                tokenizer.pad_token_id,
        )
    )

    return {
        "input_ids":
            torch_module.tensor(
                batch.input_ids,
                dtype=
                    torch_module.long,
                device=
                    device,
            ),

        "attention_mask":
            torch_module.tensor(
                batch.attention_mask,
                dtype=
                    torch_module.long,
                device=
                    device,
            ),

        "labels":
            torch_module.tensor(
                batch.labels,
                dtype=
                    torch_module.long,
                device=
                    device,
            ),
    }


# ============================================================
# MODEL / GRADIENT FINGERPRINT
# ============================================================


def trainable_parameter_fingerprint(
    *,
    model: Any,
    torch_module: Any,
) -> str:
    digest = hashlib.sha256()

    for (
        name,
        parameter,
    ) in model.named_parameters():
        if not parameter.requires_grad:
            continue

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        detached = (
            parameter.detach()
            .float()
            .cpu()
            .contiguous()
        )

        digest.update(
            detached.numpy().tobytes()
        )

    return digest.hexdigest()


def gradient_statistics(
    *,
    model: Any,
    torch_module: Any,
) -> Dict[str, Any]:
    gradient_tensor_count = 0

    nonfinite_gradient_count = 0

    gradient_square_sum = 0.0

    for parameter in model.parameters():
        if (
            not parameter.requires_grad
            or
            parameter.grad is None
        ):
            continue

        gradient_tensor_count += 1

        gradient = (
            parameter.grad.detach()
        )

        if not torch_module.isfinite(
            gradient
        ).all():
            nonfinite_gradient_count += 1

        gradient_square_sum += (
            gradient
            .float()
            .pow(
                2
            )
            .sum()
            .item()
        )

    gradient_norm = math.sqrt(
        gradient_square_sum
    )

    if (
        gradient_tensor_count
        !=
        EXPECTED_TRAINABLE_TENSORS
    ):
        raise RuntimeError(
            (
                "Unexpected gradient tensor count.\n"
                f"Expected: {EXPECTED_TRAINABLE_TENSORS}\n"
                f"Actual:   {gradient_tensor_count}"
            )
        )

    if nonfinite_gradient_count:
        raise RuntimeError(
            "Non-finite LoRA gradients detected."
        )

    if (
        not math.isfinite(
            gradient_norm
        )
        or
        gradient_norm <= 0.0
    ):
        raise RuntimeError(
            "Invalid LoRA gradient norm."
        )

    return {
        "gradient_tensor_count":
            gradient_tensor_count,

        "nonfinite_gradient_count":
            nonfinite_gradient_count,

        "gradient_norm":
            gradient_norm,
    }


# ============================================================
# CLI STATIC VALIDATION
# ============================================================


def print_static_validation(
    *,
    repository_root_value: Path,
) -> None:
    authority = (
        validate_static_authority(
            repository_root_value=
                repository_root_value
        )
    )

    versions = runtime_versions()

    print(
        "=== DATALENS QLORA v0.4 SHARED RUNTIME STATIC VALIDATION v0.1 ==="
    )

    print()

    print(
        "AUTHORITIES"
    )

    print(
        "  Experiment contract: PASS"
    )

    print(
        "  Experiment contract freeze: PASS"
    )

    print(
        "  Optimization policy: PASS"
    )

    print(
        "  Token-length evidence: PASS"
    )

    print(
        "  Training dataset: PASS"
    )

    print()

    print(
        "MODEL CONTRACT"
    )

    print(
        (
            "  Repository: "
            f"{authority.contract.base_model.repository}"
        )
    )

    print(
        (
            "  Revision: "
            f"{authority.contract.base_model.revision}"
        )
    )

    print(
        "  text_only: True"
    )

    print(
        "  trust_remote_code: False"
    )

    print(
        "  NF4 + double quant + BF16: True"
    )

    print(
        f"  Expected LoRA targets: {EXPECTED_TARGET_COUNT}"
    )

    print(
        (
            "  Expected trainable parameters: "
            f"{EXPECTED_TRAINABLE_PARAMETERS}"
        )
    )

    print(
        (
            "  Expected trainable tensors: "
            f"{EXPECTED_TRAINABLE_TENSORS}"
        )
    )

    print()

    print(
        "DATASET CONTRACT"
    )

    print(
        f"  Examples: {EXPECTED_EXAMPLE_COUNT}"
    )

    print(
        f"  Sequence length: {EXPECTED_MAX_SEQUENCE_LENGTH}"
    )

    print(
        f"  Full tokens: {EXPECTED_TOTAL_FULL_TOKENS}"
    )

    print(
        (
            "  Supervised assistant tokens: "
            f"{EXPECTED_TOTAL_SUPERVISED_TOKENS}"
        )
    )

    print(
        (
            "  Maximum example tokens: "
            f"{EXPECTED_MAX_EXAMPLE_TOKENS}"
        )
    )

    print()

    print(
        "RUNTIME VERSIONS"
    )

    for key, value in versions.items():
        print(
            f"  {key}: {value}"
        )

    print()

    print(
        "SAFETY"
    )

    print(
        "  Heavy ML modules imported by validation: False"
    )

    print(
        "  Tokenizer loaded: False"
    )

    print(
        "  Model loaded: False"
    )

    print(
        "  CUDA requested: False"
    )

    print(
        "  Forward executed: False"
    )

    print(
        "  Backward executed: False"
    )

    print(
        "  Optimizer created: False"
    )

    print(
        "  optimizer.step(): False"
    )

    print(
        "  Training executed: False"
    )

    print(
        "  Airport case content opened: False"
    )

    print(
        "  Airport evaluated: False"
    )

    print(
        "  Final Acceptance opened: False"
    )

    print(
        "  Final Acceptance evaluated: False"
    )

    print()

    print(
        "DATALENS QLORA v0.4 SHARED RUNTIME STATIC VALIDATION: PASS"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=(
            "validate-static",
        ),
    )

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )

    args = parser.parse_args()

    if args.command == "validate-static":
        print_static_validation(
            repository_root_value=
                args.repository_root
        )

        return

    raise RuntimeError(
        "Unsupported runtime command."
    )


if __name__ == "__main__":
    main()

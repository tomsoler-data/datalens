from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile

from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any, Dict, Mapping

from app.adaptation import (
    qlora_runtime_v0_4 as runtime,
)


RESOURCE_PREFLIGHT_RULE_VERSION = (
    "qlora_v0.4_resource_preflight_v0.1"
)

MEMORY_PROBE_RULE_VERSION = (
    "qlora_v0.4_one_batch_memory_probe_v0.1"
)

OPTIMIZER_PROBE_RULE_VERSION = (
    "qlora_v0.4_optimizer_state_memory_probe_v0.1"
)

ASSISTANT_PROBE_RULE_VERSION = (
    "qlora_v0.4_assistant_only_gpu_probe_v0.1"
)


EXPECTED_SHARED_RUNTIME_SHA256 = (
    "20e41ab00606296893276a84e53746c0"
    "6618b8cabca74fef77cb743c5e80ab7c"
)


MEMORY_PROBE_RELATIVE_PATH = (
    "artifacts/adaptation/probes/"
    "datalens_semantic_qlora_v0.4_memory_probe.json"
)

OPTIMIZER_PROBE_RELATIVE_PATH = (
    "artifacts/adaptation/probes/"
    "datalens_semantic_qlora_v0.4_"
    "optimizer_state_memory_probe.json"
)

ASSISTANT_PROBE_RELATIVE_PATH = (
    "artifacts/adaptation/probes/"
    "datalens_semantic_qlora_v0.4_assistant_only_probe.json"
)


# ============================================================
# EXPLICIT EXECUTION CONTROLS
# ============================================================


OPTIMIZER_BETAS = (
    0.9,
    0.999,
)

OPTIMIZER_EPS = 1e-8

OPTIMIZER_AMSGRAD = False

OPTIMIZER_MIN_8BIT_SIZE = 4096


# Resource acceptance floor.
#
# These are preflight execution controls, not training-contract
# hyperparameters. They are committed before GPU execution and
# must be bound by the later Training Execution Manifest.
MINIMUM_CUDA_FREE_BYTES = (
    512
    *
    1024
    *
    1024
)

MINIMUM_PEAK_RESERVED_HEADROOM_BYTES = (
    512
    *
    1024
    *
    1024
)


# ============================================================
# BASIC HELPERS
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
                8
                *
                1024
                *
                1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def utc_now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def canonical_json_bytes(
    payload: Mapping[str, Any],
) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def preflight_source_path() -> Path:
    return Path(
        __file__
    ).resolve()


def preflight_source_sha256() -> str:
    return sha256_file(
        preflight_source_path()
    )


def shared_runtime_path() -> Path:
    return (
        preflight_source_path()
        .with_name(
            "qlora_runtime_v0_4.py"
        )
    )


def validate_shared_runtime_identity() -> str:
    path = shared_runtime_path()

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual = sha256_file(
        path
    )

    if (
        actual
        !=
        EXPECTED_SHARED_RUNTIME_SHA256
    ):
        raise RuntimeError(
            (
                "Shared QLoRA runtime identity changed.\n"
                f"Expected: {EXPECTED_SHARED_RUNTIME_SHA256}\n"
                f"Actual:   {actual}"
            )
        )

    return actual


# ============================================================
# GIT SAFETY
# ============================================================


def git_output(
    *,
    repository_root_value: Path,
    args: list[str],
) -> str:
    result = subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=repository_root_value,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr
        )

    return result.stdout


def current_git_head(
    *,
    repository_root_value: Path,
) -> str:
    return git_output(
        repository_root_value=
            repository_root_value,

        args=[
            "rev-parse",
            "HEAD",
        ],
    ).strip()


def require_clean_working_tree(
    *,
    repository_root_value: Path,
) -> None:
    status = git_output(
        repository_root_value=
            repository_root_value,

        args=[
            "status",
            "--porcelain",
        ],
    )

    if status.strip():
        raise RuntimeError(
            (
                "GPU preflight requires a clean "
                "working tree.\n"
                f"{status}"
            )
        )


def require_tracked_file(
    *,
    repository_root_value: Path,
    relative_path: str,
) -> None:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            relative_path,
        ],
        cwd=repository_root_value,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            (
                "Required upstream probe is not "
                "committed/tracked yet: "
                f"{relative_path}"
            )
        )


# ============================================================
# ATOMIC ARTIFACT PUBLICATION
# ============================================================


def publish_new_json(
    *,
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    if path.exists():
        raise FileExistsError(
            (
                "Fail-closed artifact publication: "
                f"{path} already exists."
            )
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = canonical_json_bytes(
        payload
    )

    temporary_path: Path | None = None

    try:
        file_descriptor, raw_path = (
            tempfile.mkstemp(
                prefix=(
                    "."
                    +
                    path.name
                    +
                    "."
                ),
                suffix=".tmp",
                dir=str(
                    path.parent
                ),
            )
        )

        temporary_path = Path(
            raw_path
        )

        with os.fdopen(
            file_descriptor,
            "wb",
        ) as handle:
            handle.write(
                data
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary_path,
            path,
        )

        temporary_path = None

    finally:
        if (
            temporary_path
            is not None
            and
            temporary_path.exists()
        ):
            temporary_path.unlink()

    return sha256_bytes(
        data
    )


# ============================================================
# FROZEN AUTHORITY
# ============================================================


def validate_static(
    *,
    repository_root_value: Path,
) -> runtime.RuntimeAuthority:
    root = repository_root(
        repository_root_value
    )

    validate_shared_runtime_identity()

    authority = (
        runtime.validate_static_authority(
            repository_root_value=
                root
        )
    )

    training = (
        authority.contract.training
    )

    if (
        training.max_sequence_length
        !=
        256
    ):
        raise RuntimeError(
            "Unexpected sequence length."
        )

    if (
        training.per_device_train_batch_size
        !=
        1
    ):
        raise RuntimeError(
            "Unexpected micro-batch size."
        )

    if (
        training.gradient_accumulation_steps
        !=
        8
    ):
        raise RuntimeError(
            "Unexpected gradient accumulation."
        )

    if (
        training.learning_rate
        !=
        0.0002
    ):
        raise RuntimeError(
            "Unexpected learning rate."
        )

    if (
        training.weight_decay
        !=
        0.0
    ):
        raise RuntimeError(
            "Unexpected weight decay."
        )

    if (
        training.optimizer
        !=
        "paged_adamw_8bit"
    ):
        raise RuntimeError(
            "Unexpected optimizer."
        )

    accumulation = (
        authority.optimization_policy[
            "accumulation"
        ]
    )

    required = {
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

    for key, expected in required.items():
        if (
            accumulation[
                key
            ]
            !=
            expected
        ):
            raise RuntimeError(
                (
                    "Optimization execution plan changed: "
                    f"{key}"
                )
            )

    if (
        OPTIMIZER_BETAS
        !=
        (
            0.9,
            0.999,
        )
    ):
        raise RuntimeError(
            "Optimizer beta controls changed."
        )

    if (
        OPTIMIZER_EPS
        !=
        1e-8
    ):
        raise RuntimeError(
            "Optimizer epsilon control changed."
        )

    if (
        OPTIMIZER_AMSGRAD
        is not False
    ):
        raise RuntimeError(
            "AMSGrad became enabled."
        )

    if (
        OPTIMIZER_MIN_8BIT_SIZE
        !=
        4096
    ):
        raise RuntimeError(
            "min_8bit_size control changed."
        )

    if (
        MINIMUM_CUDA_FREE_BYTES
        !=
        536_870_912
    ):
        raise RuntimeError(
            "CUDA free-memory floor changed."
        )

    if (
        MINIMUM_PEAK_RESERVED_HEADROOM_BYTES
        !=
        536_870_912
    ):
        raise RuntimeError(
            "Reserved-memory headroom floor changed."
        )

    return authority


# ============================================================
# PROTECTED EVALUATION BOUNDARIES
# ============================================================


def protected_evaluation_boundaries(
    *,
    authority: runtime.RuntimeAuthority,
) -> Dict[str, Any]:
    airport = (
        authority
        .contract_freeze[
            "airport_independent_holdout"
        ]
    )

    final_acceptance = (
        authority.contract
        .final_acceptance_holdout
    )

    return {
        "airport_holdout_freeze_sha256":
            airport[
                "freeze_sha256"
            ],

        "airport_evaluation_executed":
            False,

        "airport_results_observed":
            False,

        "final_acceptance_freeze_sha256":
            final_acceptance.sha256,

        "final_acceptance_loaded":
            False,

        "final_acceptance_evaluated":
            False,
    }


# ============================================================
# ENVIRONMENT / DEVICE
# ============================================================


def environment_evidence() -> Dict[str, Any]:
    return {
        "python":
            platform.python_version(),

        **runtime.runtime_versions(),
    }


def device_evidence(
    *,
    torch_module: Any,
    device: Any,
) -> Dict[str, Any]:
    properties = (
        torch_module
        .cuda
        .get_device_properties(
            device
        )
    )

    free_bytes, total_visible_bytes = (
        torch_module
        .cuda
        .mem_get_info(
            device
        )
    )

    return {
        "name":
            properties.name,

        "compute_capability":
            (
                f"{properties.major}."
                f"{properties.minor}"
            ),

        "total_vram_bytes":
            int(
                properties.total_memory
            ),

        "cuda_visible_total_bytes":
            int(
                total_visible_bytes
            ),

        "free_vram_before_load_bytes":
            int(
                free_bytes
            ),
    }


def cuda_snapshot(
    *,
    torch_module: Any,
    device: Any,
) -> Dict[str, int]:
    free_bytes, _ = (
        torch_module
        .cuda
        .mem_get_info(
            device
        )
    )

    return {
        "allocated_bytes":
            int(
                torch_module
                .cuda
                .memory_allocated(
                    device
                )
            ),

        "reserved_bytes":
            int(
                torch_module
                .cuda
                .memory_reserved(
                    device
                )
            ),

        "free_bytes":
            int(
                free_bytes
            ),
    }


# ============================================================
# TRAINING EXAMPLE IDENTITY
# ============================================================


def record_identifier(
    *,
    record: Mapping[str, Any],
    index: int,
) -> str:
    for key in (
        "example_id",
        "record_id",
        "id",
    ):
        value = record.get(
            key
        )

        if (
            isinstance(
                value,
                str,
            )
            and
            value.strip()
        ):
            return value

    return (
        "dataset-index:"
        f"{index:04d}"
    )


def example_evidence(
    *,
    prepared: runtime.PreparedDataset,
    index: int,
) -> Dict[str, Any]:
    example = (
        prepared.examples[
            index
        ]
    )

    record = (
        prepared.records[
            index
        ]
    )

    return {
        "example_id":
            record_identifier(
                record=
                    record,

                index=
                    index,
            ),

        "dataset_index":
            index,

        "total_token_count":
            example.total_token_count,

        "supervised_assistant_token_count":
            example.supervised_token_count,

        "masked_token_count":
            (
                example.total_token_count
                -
                example.supervised_token_count
            ),

        "sequence_limit":
            runtime.EXPECTED_MAX_SEQUENCE_LENGTH,

        # Training example construction is fail-closed on
        # sequence overflow. A successfully prepared v0.4
        # example therefore represents a non-truncated example.
        "truncated":
            False,
    }


# ============================================================
# CUDA ACCEPTANCE
# ============================================================


def validate_memory_headroom(
    *,
    torch_module: Any,
    device: Any,
) -> Dict[str, int]:
    total_vram = int(
        torch_module
        .cuda
        .get_device_properties(
            device
        )
        .total_memory
    )

    peak_allocated = int(
        torch_module
        .cuda
        .max_memory_allocated(
            device
        )
    )

    peak_reserved = int(
        torch_module
        .cuda
        .max_memory_reserved(
            device
        )
    )

    current_free, _ = (
        torch_module
        .cuda
        .mem_get_info(
            device
        )
    )

    current_free = int(
        current_free
    )

    peak_allocated_headroom = (
        total_vram
        -
        peak_allocated
    )

    peak_reserved_headroom = (
        total_vram
        -
        peak_reserved
    )

    if (
        current_free
        <
        MINIMUM_CUDA_FREE_BYTES
    ):
        raise RuntimeError(
            (
                "Insufficient CUDA free-memory floor.\n"
                f"Required: {MINIMUM_CUDA_FREE_BYTES}\n"
                f"Observed: {current_free}"
            )
        )

    if (
        peak_reserved_headroom
        <
        MINIMUM_PEAK_RESERVED_HEADROOM_BYTES
    ):
        raise RuntimeError(
            (
                "Insufficient peak-reserved headroom.\n"
                "Required: "
                f"{MINIMUM_PEAK_RESERVED_HEADROOM_BYTES}\n"
                "Observed: "
                f"{peak_reserved_headroom}"
            )
        )

    return {
        "peak_allocated_bytes":
            peak_allocated,

        "peak_reserved_bytes":
            peak_reserved,

        "peak_allocated_headroom_bytes":
            peak_allocated_headroom,

        "peak_reserved_headroom_bytes":
            peak_reserved_headroom,

        "minimum_required_cuda_free_bytes":
            MINIMUM_CUDA_FREE_BYTES,

        "minimum_required_peak_reserved_headroom_bytes":
            MINIMUM_PEAK_RESERVED_HEADROOM_BYTES,

        "free_at_acceptance_bytes":
            current_free,
    }


# ============================================================
# PROBE CHAIN VALIDATION
# ============================================================


def probe_path(
    *,
    repository_root_value: Path,
    relative_path: str,
) -> Path:
    return (
        repository_root_value
        /
        relative_path
    ).resolve()


def validate_probe_common(
    *,
    payload: Mapping[str, Any],
    expected_rule_version: str,
) -> None:
    if (
        payload.get(
            "passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Upstream probe did not pass."
        )

    if (
        payload.get(
            "experiment_id"
        )
        !=
        runtime.EXPERIMENT_ID
    ):
        raise RuntimeError(
            "Upstream probe experiment changed."
        )

    if (
        payload.get(
            "rule_version"
        )
        !=
        expected_rule_version
    ):
        raise RuntimeError(
            "Upstream probe rule changed."
        )

    if (
        payload.get(
            "experiment_contract_sha256"
        )
        !=
        runtime.EXPECTED_CONTRACT_SHA256
    ):
        raise RuntimeError(
            "Upstream contract binding changed."
        )

    if (
        payload.get(
            "experiment_contract_freeze_sha256"
        )
        !=
        runtime.EXPECTED_CONTRACT_FREEZE_SHA256
    ):
        raise RuntimeError(
            "Upstream contract-freeze binding changed."
        )

    if (
        payload.get(
            "training_dataset_sha256"
        )
        !=
        runtime.EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "Upstream dataset binding changed."
        )

    if (
        payload.get(
            "token_length_audit_sha256"
        )
        !=
        runtime.EXPECTED_TOKEN_AUDIT_SHA256
    ):
        raise RuntimeError(
            "Upstream token evidence changed."
        )

    if (
        payload.get(
            "optimization_policy_sha256"
        )
        !=
        runtime.EXPECTED_OPTIMIZATION_POLICY_SHA256
    ):
        raise RuntimeError(
            "Upstream optimization policy changed."
        )

    implementation = payload.get(
        "implementation"
    )

    if not isinstance(
        implementation,
        Mapping,
    ):
        raise RuntimeError(
            "Missing probe implementation metadata."
        )

    if (
        implementation.get(
            "shared_runtime_sha256"
        )
        !=
        EXPECTED_SHARED_RUNTIME_SHA256
    ):
        raise RuntimeError(
            "Upstream shared runtime changed."
        )


def load_committed_probe(
    *,
    repository_root_value: Path,
    relative_path: str,
    expected_rule_version: str,
) -> tuple[Mapping[str, Any], str]:
    require_tracked_file(
        repository_root_value=
            repository_root_value,

        relative_path=
            relative_path,
    )

    path = probe_path(
        repository_root_value=
            repository_root_value,

        relative_path=
            relative_path,
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    payload = load_json(
        path
    )

    validate_probe_common(
        payload=
            payload,

        expected_rule_version=
            expected_rule_version,
    )

    return (
        payload,
        sha256_file(
            path
        ),
    )


# ============================================================
# OPTIMIZER STATE SUMMARY
# ============================================================


def optimizer_state_summary(
    *,
    optimizer: Any,
    torch_module: Any,
) -> Dict[str, Any]:
    parameter_state_count = 0

    paged_parameter_state_count = 0

    regular_parameter_state_count = 0

    unique_tensor_ids = set()

    unique_state_tensor_count = 0

    paged_state_tensor_count = 0

    nonpaged_state_tensor_count = 0

    logical_state_bytes = 0

    logical_paged_state_bytes = 0

    logical_nonpaged_state_bytes = 0

    logical_uint8_state_bytes = 0

    logical_fp32_state_bytes = 0

    largest_paged_parameter_numel = 0

    for parameter, state in optimizer.state.items():
        if not state:
            continue

        parameter_state_count += 1

        parameter_has_paged_state = False

        for value in state.values():
            if not isinstance(
                value,
                torch_module.Tensor,
            ):
                continue

            identifier = id(
                value
            )

            if identifier in unique_tensor_ids:
                continue

            unique_tensor_ids.add(
                identifier
            )

            unique_state_tensor_count += 1

            logical_bytes = (
                int(
                    value.numel()
                )
                *
                int(
                    value.element_size()
                )
            )

            logical_state_bytes += (
                logical_bytes
            )

            is_paged = bool(
                getattr(
                    value,
                    "is_paged",
                    False,
                )
            )

            if is_paged:
                parameter_has_paged_state = True

                paged_state_tensor_count += 1

                logical_paged_state_bytes += (
                    logical_bytes
                )

            else:
                nonpaged_state_tensor_count += 1

                logical_nonpaged_state_bytes += (
                    logical_bytes
                )

            if (
                value.dtype
                ==
                torch_module.uint8
            ):
                logical_uint8_state_bytes += (
                    logical_bytes
                )

            if (
                value.dtype
                ==
                torch_module.float32
            ):
                logical_fp32_state_bytes += (
                    logical_bytes
                )

        if parameter_has_paged_state:
            paged_parameter_state_count += 1

            largest_paged_parameter_numel = max(
                largest_paged_parameter_numel,
                int(
                    parameter.numel()
                ),
            )

        else:
            regular_parameter_state_count += 1

    return {
        "parameter_state_count":
            parameter_state_count,

        "paged_parameter_state_count":
            paged_parameter_state_count,

        "regular_parameter_state_count":
            regular_parameter_state_count,

        "unique_state_tensor_count":
            unique_state_tensor_count,

        "paged_state_tensor_count":
            paged_state_tensor_count,

        "nonpaged_state_tensor_count":
            nonpaged_state_tensor_count,

        "logical_state_bytes":
            logical_state_bytes,

        "logical_paged_state_bytes":
            logical_paged_state_bytes,

        "logical_nonpaged_state_bytes":
            logical_nonpaged_state_bytes,

        "logical_uint8_state_bytes":
            logical_uint8_state_bytes,

        "logical_fp32_state_bytes":
            logical_fp32_state_bytes,

        "largest_paged_parameter_numel":
            largest_paged_parameter_numel,
    }


# ============================================================
# COMMON PROBE METADATA
# ============================================================


def base_probe_payload(
    *,
    authority: runtime.RuntimeAuthority,
    repository_root_value: Path,
    probe_id: str,
    rule_version: str,
) -> Dict[str, Any]:
    return {
        "probe_id":
            probe_id,

        "rule_version":
            rule_version,

        "resource_preflight_rule_version":
            RESOURCE_PREFLIGHT_RULE_VERSION,

        "experiment_id":
            runtime.EXPERIMENT_ID,

        "git_head":
            current_git_head(
                repository_root_value=
                    repository_root_value
            ),

        "experiment_contract_sha256":
            runtime.EXPECTED_CONTRACT_SHA256,

        "experiment_contract_freeze_sha256":
            runtime.EXPECTED_CONTRACT_FREEZE_SHA256,

        "training_dataset_sha256":
            runtime.EXPECTED_DATASET_SHA256,

        "token_length_audit_sha256":
            runtime.EXPECTED_TOKEN_AUDIT_SHA256,

        "optimization_policy_sha256":
            runtime.EXPECTED_OPTIMIZATION_POLICY_SHA256,

        "implementation": {
            "preflight_source_sha256":
                preflight_source_sha256(),

            "shared_runtime_sha256":
                validate_shared_runtime_identity(),

            "shared_runtime_rule_version":
                runtime.QLORA_V04_SHARED_RUNTIME_RULE_VERSION,
        },

        "environment":
            environment_evidence(),

        "protected_evaluation":
            protected_evaluation_boundaries(
                authority=
                    authority
            ),
    }


# ============================================================
# MEMORY PROBE
# ============================================================


def run_memory_probe(
    *,
    repository_root_value: Path,
) -> str:
    root = repository_root(
        repository_root_value
    )

    require_clean_working_tree(
        repository_root_value=
            root
    )

    authority = validate_static(
        repository_root_value=
            root
    )

    output_path = probe_path(
        repository_root_value=
            root,

        relative_path=
            MEMORY_PROBE_RELATIVE_PATH,
    )

    if output_path.exists():
        raise FileExistsError(
            output_path
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "CUDA BF16 is unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    random_seed = (
        authority.contract
        .training
        .random_seed
    )

    random.seed(
        random_seed
    )

    torch.manual_seed(
        random_seed
    )

    torch.cuda.manual_seed_all(
        random_seed
    )

    torch.cuda.empty_cache()

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    device_info = device_evidence(
        torch_module=
            torch,

        device=
            device,
    )

    tokenizer = (
        runtime.load_pinned_tokenizer(
            authority=
                authority
        )
    )

    prepared = (
        runtime.prepare_training_dataset(
            authority=
                authority,

            tokenizer=
                tokenizer,
        )
    )

    selected_index = (
        runtime.longest_training_example_index(
            prepared
        )
    )

    selected_example = (
        prepared.examples[
            selected_index
        ]
    )

    prepared_model = (
        runtime.prepare_qlora_model(
            authority=
                authority
        )
    )

    model = prepared_model.model

    after_lora = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    fingerprint_before = (
        runtime.trainable_parameter_fingerprint(
            model=
                model,

            torch_module=
                torch,
        )
    )

    batch = (
        runtime.tensor_batch_from_example(
            example=
                selected_example,

            tokenizer=
                tokenizer,

            torch_module=
                torch,

            device=
                device,
        )
    )

    before_forward = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    outputs = model(
        input_ids=
            batch[
                "input_ids"
            ],

        attention_mask=
            batch[
                "attention_mask"
            ],

        labels=
            batch[
                "labels"
            ],

        use_cache=
            False,
    )

    loss = outputs.loss

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Non-finite memory-probe loss."
        )

    loss_value = float(
        loss.detach().cpu()
    )

    after_forward = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    loss.backward()

    torch.cuda.synchronize()

    after_backward = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    gradients = (
        runtime.gradient_statistics(
            model=
                model,

            torch_module=
                torch,
        )
    )

    fingerprint_after = (
        runtime.trainable_parameter_fingerprint(
            model=
                model,

            torch_module=
                torch,
        )
    )

    weights_unchanged = (
        fingerprint_before
        ==
        fingerprint_after
    )

    if not weights_unchanged:
        raise RuntimeError(
            "Memory probe modified LoRA weights."
        )

    memory_acceptance = (
        validate_memory_headroom(
            torch_module=
                torch,

            device=
                device,
        )
    )

    model.zero_grad(
        set_to_none=True
    )

    torch.cuda.synchronize()

    after_zero_grad = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    payload = base_probe_payload(
        authority=
            authority,

        repository_root_value=
            root,

        probe_id=
            (
                "datalens-semantic-qlora-v0.4-"
                "memory-probe"
            ),

        rule_version=
            MEMORY_PROBE_RULE_VERSION,
    )

    payload.update(
        {
            "created_at":
                utc_now(),

            "configuration": {
                "quantization":
                    "nf4",

                "compute_dtype":
                    "bfloat16",

                "sequence_length":
                    runtime.EXPECTED_MAX_SEQUENCE_LENGTH,

                "actual_sequence_length":
                    selected_example.total_token_count,

                "micro_batch_size":
                    1,

                "gradient_accumulation_steps":
                    8,

                "gradient_checkpointing":
                    True,

                "gradient_checkpoint_use_reentrant":
                    False,

                "lora_rank":
                    authority.contract.lora.rank,

                "lora_alpha":
                    authority.contract.lora.alpha,

                "lora_dropout":
                    authority.contract.lora.dropout,

                "target_module_count":
                    prepared_model.trainable_tensor_count
                    //
                    2,

                "trainable_parameter_count":
                    prepared_model.trainable_parameter_count,

                "trainable_tensor_count":
                    prepared_model.trainable_tensor_count,
            },

            "selected_training_example":
                example_evidence(
                    prepared=
                        prepared,

                    index=
                        selected_index,
                ),

            "device":
                device_info,

            "memory_bytes": {
                "allocated_after_lora_attach":
                    after_lora[
                        "allocated_bytes"
                    ],

                "reserved_after_lora_attach":
                    after_lora[
                        "reserved_bytes"
                    ],

                "free_after_lora_attach":
                    after_lora[
                        "free_bytes"
                    ],

                "allocated_before_forward":
                    before_forward[
                        "allocated_bytes"
                    ],

                "reserved_before_forward":
                    before_forward[
                        "reserved_bytes"
                    ],

                "free_before_forward":
                    before_forward[
                        "free_bytes"
                    ],

                "allocated_after_forward":
                    after_forward[
                        "allocated_bytes"
                    ],

                "reserved_after_forward":
                    after_forward[
                        "reserved_bytes"
                    ],

                "free_after_forward":
                    after_forward[
                        "free_bytes"
                    ],

                "allocated_after_backward":
                    after_backward[
                        "allocated_bytes"
                    ],

                "reserved_after_backward":
                    after_backward[
                        "reserved_bytes"
                    ],

                "free_after_backward":
                    after_backward[
                        "free_bytes"
                    ],

                "allocated_after_zero_grad":
                    after_zero_grad[
                        "allocated_bytes"
                    ],

                "reserved_after_zero_grad":
                    after_zero_grad[
                        "reserved_bytes"
                    ],

                "free_after_zero_grad":
                    after_zero_grad[
                        "free_bytes"
                    ],

                **memory_acceptance,
            },

            "memory_classification":
                "healthy_v0_4_pre_optimizer_headroom",

            "result": {
                "loss":
                    loss_value,

                "loss_finite":
                    True,

                **gradients,

                "weights_unchanged":
                    weights_unchanged,
            },

            "safety": {
                "real_training_labels_used":
                    True,

                "optimizer_created":
                    False,

                "optimizer_state_initialized":
                    False,

                "optimizer_step_executed":
                    False,

                "optimizer_update_step_called":
                    False,

                "scheduler_created":
                    False,

                "trainer_created":
                    False,

                "trainer_train_called":
                    False,

                "model_weights_saved":
                    False,

                "adapter_saved":
                    False,

                "training_executed":
                    False,

                "pre_adaptation_holdouts_executed":
                    False,

                "airport_cases_loaded":
                    False,

                "airport_evaluated":
                    False,

                "final_acceptance_cases_loaded":
                    False,

                "final_acceptance_evaluated":
                    False,
            },

            "passed":
                True,
        }
    )

    artifact_sha = publish_new_json(
        path=
            output_path,

        payload=
            payload,
    )

    print(
        "=== DATALENS QLORA v0.4 MEMORY PROBE ==="
    )

    print()

    print(
        f"Artifact: {output_path}"
    )

    print(
        f"Artifact SHA256: {artifact_sha}"
    )

    print(
        (
            "Example tokens: "
            f"{selected_example.total_token_count}"
        )
    )

    print(
        f"Loss: {loss_value:.6f}"
    )

    print(
        (
            "Gradient norm: "
            f"{gradients['gradient_norm']:.6f}"
        )
    )

    print(
        (
            "Peak allocated: "
            f"{memory_acceptance['peak_allocated_bytes'] / 1024**2:.2f} MiB"
        )
    )

    print(
        (
            "Peak reserved: "
            f"{memory_acceptance['peak_reserved_bytes'] / 1024**2:.2f} MiB"
        )
    )

    print(
        (
            "Free at acceptance: "
            f"{memory_acceptance['free_at_acceptance_bytes'] / 1024**2:.2f} MiB"
        )
    )

    print()

    print(
        "Optimizer created: False"
    )

    print(
        "optimizer.step(): False"
    )

    print(
        "Weights unchanged: True"
    )

    print(
        "Training executed: False"
    )

    print()

    print(
        "DATALENS QLORA v0.4 MEMORY PROBE: PASS"
    )

    return artifact_sha


# ============================================================
# OPTIMIZER-STATE MEMORY PROBE
# ============================================================


def run_optimizer_probe(
    *,
    repository_root_value: Path,
) -> str:
    root = repository_root(
        repository_root_value
    )

    require_clean_working_tree(
        repository_root_value=
            root
    )

    authority = validate_static(
        repository_root_value=
            root
    )

    (
        memory_probe,
        memory_probe_sha,
    ) = load_committed_probe(
        repository_root_value=
            root,

        relative_path=
            MEMORY_PROBE_RELATIVE_PATH,

        expected_rule_version=
            MEMORY_PROBE_RULE_VERSION,
    )

    output_path = probe_path(
        repository_root_value=
            root,

        relative_path=
            OPTIMIZER_PROBE_RELATIVE_PATH,
    )

    if output_path.exists():
        raise FileExistsError(
            output_path
        )

    import bitsandbytes as bnb
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "CUDA BF16 is unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    seed = (
        authority.contract
        .training
        .random_seed
    )

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    torch.cuda.empty_cache()

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    device_info = device_evidence(
        torch_module=
            torch,

        device=
            device,
    )

    tokenizer = (
        runtime.load_pinned_tokenizer(
            authority=
                authority
        )
    )

    prepared = (
        runtime.prepare_training_dataset(
            authority=
                authority,

            tokenizer=
                tokenizer,
        )
    )

    selected_index = (
        runtime.longest_training_example_index(
            prepared
        )
    )

    selected_example = (
        prepared.examples[
            selected_index
        ]
    )

    prepared_model = (
        runtime.prepare_qlora_model(
            authority=
                authority
        )
    )

    model = prepared_model.model

    after_lora = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    optimizer = (
        bnb.optim.PagedAdamW8bit(
            prepared_model.trainable_parameters,

            lr=
                authority.contract
                .training
                .learning_rate,

            betas=
                OPTIMIZER_BETAS,

            eps=
                OPTIMIZER_EPS,

            weight_decay=
                authority.contract
                .training
                .weight_decay,

            amsgrad=
                OPTIMIZER_AMSGRAD,

            min_8bit_size=
                OPTIMIZER_MIN_8BIT_SIZE,
        )
    )

    if (
        optimizer.__class__.__name__
        !=
        "PagedAdamW8bit"
    ):
        raise RuntimeError(
            "Unexpected optimizer class."
        )

    if (
        getattr(
            optimizer,
            "is_paged",
            None,
        )
        is not True
    ):
        raise RuntimeError(
            "Optimizer is not paged."
        )

    after_optimizer_create = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    fingerprint_before = (
        runtime.trainable_parameter_fingerprint(
            model=
                model,

            torch_module=
                torch,
        )
    )

    batch = (
        runtime.tensor_batch_from_example(
            example=
                selected_example,

            tokenizer=
                tokenizer,

            torch_module=
                torch,

            device=
                device,
        )
    )

    outputs = model(
        input_ids=
            batch[
                "input_ids"
            ],

        attention_mask=
            batch[
                "attention_mask"
            ],

        labels=
            batch[
                "labels"
            ],

        use_cache=
            False,
    )

    loss = outputs.loss

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Non-finite optimizer-probe loss."
        )

    loss_value = float(
        loss.detach().cpu()
    )

    after_forward = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    loss.backward()

    torch.cuda.synchronize()

    after_backward = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    gradients = (
        runtime.gradient_statistics(
            model=
                model,

            torch_module=
                torch,
        )
    )

    page_manager = (
        optimizer.page_mng
    )

    paged_before = len(
        page_manager.paged_tensors
    )

    # --------------------------------------------------------
    # Reproduce ONLY the initialization/prefetch portion of
    # bitsandbytes Optimizer8bit.step().
    #
    # Explicitly excluded:
    #   optimizer.step()
    #   optimizer.update_step()
    # --------------------------------------------------------

    optimizer.check_overrides()

    optimizer.to_gpu()

    optimizer.initialized = True

    initialized_parameter_count = 0

    prefetched_parameter_count = 0

    for (
        group_index,
        group,
    ) in enumerate(
        optimizer.param_groups
    ):
        for (
            parameter_index,
            parameter,
        ) in enumerate(
            group[
                "params"
            ]
        ):
            if parameter.grad is None:
                continue

            state = optimizer.state[
                parameter
            ]

            if len(
                state
            ) == 0:
                optimizer.init_state(
                    group,
                    parameter,
                    group_index,
                    parameter_index,
                )

                initialized_parameter_count += 1

            optimizer.prefetch_state(
                parameter
            )

            prefetched_parameter_count += 1

    torch.cuda.synchronize()

    after_state_init = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    paged_after = len(
        page_manager.paged_tensors
    )

    state_summary = optimizer_state_summary(
        optimizer=
            optimizer,

        torch_module=
            torch,
    )

    if (
        initialized_parameter_count
        !=
        runtime.EXPECTED_TRAINABLE_TENSORS
    ):
        raise RuntimeError(
            (
                "Not every LoRA tensor received "
                "optimizer state.\n"
                f"Expected: {runtime.EXPECTED_TRAINABLE_TENSORS}\n"
                f"Actual:   {initialized_parameter_count}"
            )
        )

    if (
        prefetched_parameter_count
        !=
        runtime.EXPECTED_TRAINABLE_TENSORS
    ):
        raise RuntimeError(
            "Not every LoRA tensor was prefetched."
        )

    if (
        state_summary[
            "parameter_state_count"
        ]
        !=
        runtime.EXPECTED_TRAINABLE_TENSORS
    ):
        raise RuntimeError(
            "Optimizer state parameter count mismatch."
        )

    if (
        state_summary[
            "paged_state_tensor_count"
        ]
        <=
        0
    ):
        raise RuntimeError(
            "Paged optimizer state was not created."
        )

    if (
        paged_after
        <=
        paged_before
    ):
        raise RuntimeError(
            "Global page manager did not gain paged tensors."
        )

    fingerprint_after = (
        runtime.trainable_parameter_fingerprint(
            model=
                model,

            torch_module=
                torch,
        )
    )

    weights_unchanged = (
        fingerprint_before
        ==
        fingerprint_after
    )

    if not weights_unchanged:
        raise RuntimeError(
            "Optimizer state initialization modified weights."
        )

    memory_acceptance = (
        validate_memory_headroom(
            torch_module=
                torch,

            device=
                device,
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    torch.cuda.synchronize()

    after_zero_grad = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    payload = base_probe_payload(
        authority=
            authority,

        repository_root_value=
            root,

        probe_id=
            (
                "datalens-semantic-qlora-v0.4-"
                "optimizer-state-memory-probe"
            ),

        rule_version=
            OPTIMIZER_PROBE_RULE_VERSION,
    )

    payload.update(
        {
            "created_at":
                utc_now(),

            "source_memory_probe_sha256":
                memory_probe_sha,

            "configuration": {
                "sequence_length":
                    runtime.EXPECTED_MAX_SEQUENCE_LENGTH,

                "actual_sequence_length":
                    selected_example.total_token_count,

                "micro_batch_size":
                    1,

                "gradient_accumulation_steps":
                    8,

                "optimizer":
                    "paged_adamw_8bit",

                "learning_rate":
                    authority.contract
                    .training
                    .learning_rate,

                "optimizer_betas":
                    list(
                        OPTIMIZER_BETAS
                    ),

                "optimizer_eps":
                    OPTIMIZER_EPS,

                "weight_decay":
                    authority.contract
                    .training
                    .weight_decay,

                "amsgrad":
                    OPTIMIZER_AMSGRAD,

                "optimizer_min_8bit_size":
                    OPTIMIZER_MIN_8BIT_SIZE,

                "lora_rank":
                    authority.contract
                    .lora
                    .rank,

                "trainable_parameter_count":
                    prepared_model
                    .trainable_parameter_count,

                "trainable_tensor_count":
                    prepared_model
                    .trainable_tensor_count,
            },

            "selected_training_example":
                example_evidence(
                    prepared=
                        prepared,

                    index=
                        selected_index,
                ),

            "device":
                device_info,

            "memory_bytes": {
                "allocated_after_lora_attach":
                    after_lora[
                        "allocated_bytes"
                    ],

                "reserved_after_lora_attach":
                    after_lora[
                        "reserved_bytes"
                    ],

                "free_after_lora_attach":
                    after_lora[
                        "free_bytes"
                    ],

                "allocated_after_optimizer_create":
                    after_optimizer_create[
                        "allocated_bytes"
                    ],

                "reserved_after_optimizer_create":
                    after_optimizer_create[
                        "reserved_bytes"
                    ],

                "free_after_optimizer_create":
                    after_optimizer_create[
                        "free_bytes"
                    ],

                "allocated_after_forward":
                    after_forward[
                        "allocated_bytes"
                    ],

                "reserved_after_forward":
                    after_forward[
                        "reserved_bytes"
                    ],

                "free_after_forward":
                    after_forward[
                        "free_bytes"
                    ],

                "allocated_after_backward":
                    after_backward[
                        "allocated_bytes"
                    ],

                "reserved_after_backward":
                    after_backward[
                        "reserved_bytes"
                    ],

                "free_after_backward":
                    after_backward[
                        "free_bytes"
                    ],

                "allocated_after_state_init":
                    after_state_init[
                        "allocated_bytes"
                    ],

                "reserved_after_state_init":
                    after_state_init[
                        "reserved_bytes"
                    ],

                "free_after_state_init":
                    after_state_init[
                        "free_bytes"
                    ],

                "allocated_after_zero_grad":
                    after_zero_grad[
                        "allocated_bytes"
                    ],

                "reserved_after_zero_grad":
                    after_zero_grad[
                        "reserved_bytes"
                    ],

                "free_after_zero_grad":
                    after_zero_grad[
                        "free_bytes"
                    ],

                **memory_acceptance,
            },

            "optimizer_state":
                state_summary,

            "page_manager": {
                "paged_tensor_count_before_state_init":
                    paged_before,

                "paged_tensor_count_after_state_init":
                    paged_after,

                "new_paged_tensor_count":
                    (
                        paged_after
                        -
                        paged_before
                    ),
            },

            "result": {
                "loss":
                    loss_value,

                "loss_finite":
                    True,

                **gradients,

                "optimizer_state_initialized":
                    True,

                "initialized_parameter_count":
                    initialized_parameter_count,

                "prefetched_parameter_count":
                    prefetched_parameter_count,

                "largest_paged_state_prefetched":
                    True,

                "weights_unchanged_after_state_init":
                    weights_unchanged,

                "weights_unchanged_final":
                    weights_unchanged,
            },

            "memory_classification":
                "healthy_v0_4_optimizer_state_preflight",

            "safety": {
                "optimizer_created":
                    True,

                "optimizer_state_initialized":
                    True,

                "optimizer_prefetch_executed":
                    True,

                "optimizer_step_executed":
                    False,

                "optimizer_update_step_called":
                    False,

                "scheduler_created":
                    False,

                "scheduler_step_executed":
                    False,

                "trainer_created":
                    False,

                "trainer_train_called":
                    False,

                "model_weights_saved":
                    False,

                "adapter_saved":
                    False,

                "training_executed":
                    False,

                "pre_adaptation_holdouts_executed":
                    False,

                "airport_cases_loaded":
                    False,

                "airport_evaluated":
                    False,

                "final_acceptance_cases_loaded":
                    False,

                "final_acceptance_evaluated":
                    False,
            },

            "passed":
                True,
        }
    )

    artifact_sha = publish_new_json(
        path=
            output_path,

        payload=
            payload,
    )

    print(
        "=== DATALENS QLORA v0.4 OPTIMIZER-STATE MEMORY PROBE ==="
    )

    print()

    print(
        f"Source memory probe SHA256: {memory_probe_sha}"
    )

    print(
        f"Artifact SHA256: {artifact_sha}"
    )

    print(
        f"Loss: {loss_value:.6f}"
    )

    print(
        (
            "Initialized parameter states: "
            f"{initialized_parameter_count}"
        )
    )

    print(
        (
            "Paged state tensors: "
            f"{state_summary['paged_state_tensor_count']}"
        )
    )

    print(
        (
            "New page-manager tensors: "
            f"{paged_after - paged_before}"
        )
    )

    print(
        (
            "Peak reserved: "
            f"{memory_acceptance['peak_reserved_bytes'] / 1024**2:.2f} MiB"
        )
    )

    print()

    print(
        "optimizer.step(): False"
    )

    print(
        "optimizer.update_step(): False"
    )

    print(
        "Weights unchanged: True"
    )

    print(
        "Training executed: False"
    )

    print()

    print(
        "DATALENS QLORA v0.4 OPTIMIZER-STATE MEMORY PROBE: PASS"
    )

    return artifact_sha


# ============================================================
# ASSISTANT-ONLY GPU PROBE
# ============================================================


def run_assistant_probe(
    *,
    repository_root_value: Path,
) -> str:
    root = repository_root(
        repository_root_value
    )

    require_clean_working_tree(
        repository_root_value=
            root
    )

    authority = validate_static(
        repository_root_value=
            root
    )

    (
        optimizer_probe,
        optimizer_probe_sha,
    ) = load_committed_probe(
        repository_root_value=
            root,

        relative_path=
            OPTIMIZER_PROBE_RELATIVE_PATH,

        expected_rule_version=
            OPTIMIZER_PROBE_RULE_VERSION,
    )

    output_path = probe_path(
        repository_root_value=
            root,

        relative_path=
            ASSISTANT_PROBE_RELATIVE_PATH,
    )

    if output_path.exists():
        raise FileExistsError(
            output_path
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "CUDA BF16 is unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    seed = (
        authority.contract
        .training
        .random_seed
    )

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    torch.cuda.empty_cache()

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    device_info = device_evidence(
        torch_module=
            torch,

        device=
            device,
    )

    tokenizer = (
        runtime.load_pinned_tokenizer(
            authority=
                authority
        )
    )

    prepared = (
        runtime.prepare_training_dataset(
            authority=
                authority,

            tokenizer=
                tokenizer,
        )
    )

    selected_index = (
        runtime.longest_training_example_index(
            prepared
        )
    )

    selected_example = (
        prepared.examples[
            selected_index
        ]
    )

    prepared_model = (
        runtime.prepare_qlora_model(
            authority=
                authority
        )
    )

    model = prepared_model.model

    fingerprint_before = (
        runtime.trainable_parameter_fingerprint(
            model=
                model,

            torch_module=
                torch,
        )
    )

    batch = (
        runtime.tensor_batch_from_example(
            example=
                selected_example,

            tokenizer=
                tokenizer,

            torch_module=
                torch,

            device=
                device,
        )
    )

    labels = batch[
        "labels"
    ]

    supervised_token_count = int(
        (
            labels
            !=
            -100
        )
        .sum()
        .item()
    )

    masked_token_count = int(
        (
            labels
            ==
            -100
        )
        .sum()
        .item()
    )

    if (
        supervised_token_count
        !=
        selected_example.supervised_token_count
    ):
        raise RuntimeError(
            "Assistant supervision count changed."
        )

    if supervised_token_count <= 0:
        raise RuntimeError(
            "No assistant tokens supervised."
        )

    if masked_token_count <= 0:
        raise RuntimeError(
            "Prompt tokens are not masked."
        )

    if (
        supervised_token_count
        +
        masked_token_count
        !=
        selected_example.total_token_count
    ):
        raise RuntimeError(
            "Assistant masking token accounting failed."
        )

    outputs = model(
        input_ids=
            batch[
                "input_ids"
            ],

        attention_mask=
            batch[
                "attention_mask"
            ],

        labels=
            labels,

        use_cache=
            False,
    )

    loss = outputs.loss

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Non-finite assistant probe loss."
        )

    loss_value = float(
        loss.detach().cpu()
    )

    after_forward = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    loss.backward()

    torch.cuda.synchronize()

    after_backward = cuda_snapshot(
        torch_module=
            torch,

        device=
            device,
    )

    gradients = (
        runtime.gradient_statistics(
            model=
                model,

            torch_module=
                torch,
        )
    )

    fingerprint_after = (
        runtime.trainable_parameter_fingerprint(
            model=
                model,

            torch_module=
                torch,
        )
    )

    weights_unchanged = (
        fingerprint_before
        ==
        fingerprint_after
    )

    if not weights_unchanged:
        raise RuntimeError(
            "Assistant probe modified LoRA weights."
        )

    memory_acceptance = (
        validate_memory_headroom(
            torch_module=
                torch,

            device=
                device,
        )
    )

    payload = base_probe_payload(
        authority=
            authority,

        repository_root_value=
            root,

        probe_id=
            (
                "datalens-semantic-qlora-v0.4-"
                "assistant-only-gpu-probe"
            ),

        rule_version=
            ASSISTANT_PROBE_RULE_VERSION,
    )

    payload.update(
        {
            "created_at":
                utc_now(),

            "optimizer_memory_probe_sha256":
                optimizer_probe_sha,

            "configuration": {
                "configured_max_sequence_length":
                    runtime.EXPECTED_MAX_SEQUENCE_LENGTH,

                "actual_sequence_length":
                    selected_example.total_token_count,

                "micro_batch_size":
                    1,

                "gradient_checkpointing":
                    True,

                "gradient_checkpoint_use_reentrant":
                    False,

                "lora_rank":
                    authority.contract
                    .lora
                    .rank,

                "lora_alpha":
                    authority.contract
                    .lora
                    .alpha,

                "lora_dropout":
                    authority.contract
                    .lora
                    .dropout,

                "target_module_count":
                    len(
                        prepared_model.target_modules
                    ),

                "trainable_parameter_count":
                    prepared_model
                    .trainable_parameter_count,

                "trainable_tensor_count":
                    prepared_model
                    .trainable_tensor_count,
            },

            "assistant_masking": {
                "rule_version":
                    runtime
                    .ASSISTANT_ONLY_MASKING_RULE_VERSION,

                "ignore_index":
                    -100,

                "assistant_supervised":
                    True,

                "prompt_supervised":
                    False,

                "padding_supervised":
                    False,

                "silent_truncation_allowed":
                    False,

                "supervised_token_count":
                    supervised_token_count,

                "masked_token_count":
                    masked_token_count,
            },

            "selected_training_example":
                example_evidence(
                    prepared=
                        prepared,

                    index=
                        selected_index,
                ),

            "device":
                device_info,

            "memory_bytes": {
                "allocated_after_forward":
                    after_forward[
                        "allocated_bytes"
                    ],

                "reserved_after_forward":
                    after_forward[
                        "reserved_bytes"
                    ],

                "free_after_forward":
                    after_forward[
                        "free_bytes"
                    ],

                "allocated_after_backward":
                    after_backward[
                        "allocated_bytes"
                    ],

                "reserved_after_backward":
                    after_backward[
                        "reserved_bytes"
                    ],

                "free_after_backward":
                    after_backward[
                        "free_bytes"
                    ],

                **memory_acceptance,
            },

            "memory_classification":
                "healthy_v0_4_assistant_only_headroom",

            "result": {
                "loss":
                    loss_value,

                "loss_finite":
                    True,

                **gradients,

                "weights_unchanged":
                    weights_unchanged,
            },

            "safety": {
                "real_training_labels_used":
                    True,

                "optimizer_created":
                    False,

                "optimizer_step_executed":
                    False,

                "optimizer_update_step_called":
                    False,

                "scheduler_created":
                    False,

                "trainer_created":
                    False,

                "trainer_train_called":
                    False,

                "model_weights_saved":
                    False,

                "adapter_saved":
                    False,

                "training_executed":
                    False,

                "pre_adaptation_holdouts_executed":
                    False,

                "airport_cases_loaded":
                    False,

                "airport_evaluated":
                    False,

                "final_acceptance_cases_loaded":
                    False,

                "final_acceptance_evaluated":
                    False,
            },

            "passed":
                True,
        }
    )

    artifact_sha = publish_new_json(
        path=
            output_path,

        payload=
            payload,
    )

    print(
        "=== DATALENS QLORA v0.4 ASSISTANT-ONLY GPU PROBE ==="
    )

    print()

    print(
        (
            "Source optimizer probe SHA256: "
            f"{optimizer_probe_sha}"
        )
    )

    print(
        f"Artifact SHA256: {artifact_sha}"
    )

    print(
        (
            "Total tokens: "
            f"{selected_example.total_token_count}"
        )
    )

    print(
        (
            "Supervised assistant tokens: "
            f"{supervised_token_count}"
        )
    )

    print(
        (
            "Masked tokens: "
            f"{masked_token_count}"
        )
    )

    print(
        f"Loss: {loss_value:.6f}"
    )

    print(
        (
            "Gradient norm: "
            f"{gradients['gradient_norm']:.6f}"
        )
    )

    print()

    print(
        "Optimizer created: False"
    )

    print(
        "optimizer.step(): False"
    )

    print(
        "Weights unchanged: True"
    )

    print(
        "Training executed: False"
    )

    print()

    print(
        "DATALENS QLORA v0.4 ASSISTANT-ONLY GPU PROBE: PASS"
    )

    return artifact_sha


# ============================================================
# VERIFICATION
# ============================================================


def verify_probe(
    *,
    repository_root_value: Path,
    relative_path: str,
    expected_rule_version: str,
) -> str:
    root = repository_root(
        repository_root_value
    )

    path = probe_path(
        repository_root_value=
            root,

        relative_path=
            relative_path,
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    payload = load_json(
        path
    )

    validate_probe_common(
        payload=
            payload,

        expected_rule_version=
            expected_rule_version,
    )

    return sha256_file(
        path
    )


# ============================================================
# STATIC CLI
# ============================================================


def print_static_validation(
    *,
    repository_root_value: Path,
) -> None:
    authority = validate_static(
        repository_root_value=
            repository_root_value
    )

    print(
        "=== DATALENS QLORA v0.4 RESOURCE PREFLIGHT STATIC VALIDATION v0.1 ==="
    )

    print()

    print(
        "AUTHORITY"
    )

    print(
        "  Frozen experiment contract: PASS"
    )

    print(
        "  Frozen optimization policy: PASS"
    )

    print(
        "  Frozen training dataset: PASS"
    )

    print(
        "  Token-length evidence: PASS"
    )

    print(
        "  Shared runtime identity: PASS"
    )

    print()

    print(
        "EXECUTION PLAN"
    )

    print(
        "  Dataset examples: 230"
    )

    print(
        "  Sequence length: 256"
    )

    print(
        "  Micro batch: 1"
    )

    print(
        "  Gradient accumulation: 8"
    )

    print(
        "  Full groups / epoch: 28"
    )

    print(
        "  Partial group: 6"
    )

    print(
        "  Optimizer steps / epoch: 29"
    )

    print(
        "  Total optimizer steps: 58"
    )

    print()

    print(
        "OPTIMIZER IMPLEMENTATION CONTROLS"
    )

    print(
        "  PagedAdamW8bit: True"
    )

    print(
        (
            "  betas: "
            f"{list(OPTIMIZER_BETAS)}"
        )
    )

    print(
        f"  eps: {OPTIMIZER_EPS}"
    )

    print(
        f"  amsgrad: {OPTIMIZER_AMSGRAD}"
    )

    print(
        (
            "  min_8bit_size: "
            f"{OPTIMIZER_MIN_8BIT_SIZE}"
        )
    )

    print(
        "  implicit optimizer defaults: False"
    )

    print()

    print(
        "RESOURCE ACCEPTANCE"
    )

    print(
        (
            "  CUDA free floor: "
            f"{MINIMUM_CUDA_FREE_BYTES}"
        )
    )

    print(
        (
            "  Peak reserved headroom floor: "
            f"{MINIMUM_PEAK_RESERVED_HEADROOM_BYTES}"
        )
    )

    print()

    print(
        "PROBE CHAIN"
    )

    print(
        "  memory"
    )

    print(
        "  memory -> optimizer"
    )

    print(
        "  optimizer -> assistant"
    )

    print()

    print(
        "SAFETY"
    )

    print(
        "  GPU used by static validation: False"
    )

    print(
        "  Model loaded: False"
    )

    print(
        "  Optimizer created: False"
    )

    print(
        "  optimizer.step(): False"
    )

    print(
        "  optimizer.update_step(): False"
    )

    print(
        "  Training executed: False"
    )

    print(
        "  Airport evaluated: False"
    )

    print(
        "  Final Acceptance evaluated: False"
    )

    print()

    print(
        "DATALENS QLORA v0.4 RESOURCE PREFLIGHT STATIC VALIDATION: PASS"
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=(
            "validate-static",
            "memory",
            "optimizer",
            "assistant",
            "verify-memory",
            "verify-optimizer",
            "verify-assistant",
        ),
    )

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )

    args = parser.parse_args()

    root = repository_root(
        args.repository_root
    )

    if args.command == "validate-static":
        print_static_validation(
            repository_root_value=
                root
        )

        return

    if args.command == "memory":
        run_memory_probe(
            repository_root_value=
                root
        )

        return

    if args.command == "optimizer":
        run_optimizer_probe(
            repository_root_value=
                root
        )

        return

    if args.command == "assistant":
        run_assistant_probe(
            repository_root_value=
                root
        )

        return

    if args.command == "verify-memory":
        sha = verify_probe(
            repository_root_value=
                root,

            relative_path=
                MEMORY_PROBE_RELATIVE_PATH,

            expected_rule_version=
                MEMORY_PROBE_RULE_VERSION,
        )

        print(
            f"Memory probe SHA256: {sha}"
        )

        print(
            "Memory probe verification: PASS"
        )

        return

    if args.command == "verify-optimizer":
        sha = verify_probe(
            repository_root_value=
                root,

            relative_path=
                OPTIMIZER_PROBE_RELATIVE_PATH,

            expected_rule_version=
                OPTIMIZER_PROBE_RULE_VERSION,
        )

        print(
            f"Optimizer probe SHA256: {sha}"
        )

        print(
            "Optimizer probe verification: PASS"
        )

        return

    if args.command == "verify-assistant":
        sha = verify_probe(
            repository_root_value=
                root,

            relative_path=
                ASSISTANT_PROBE_RELATIVE_PATH,

            expected_rule_version=
                ASSISTANT_PROBE_RULE_VERSION,
        )

        print(
            f"Assistant probe SHA256: {sha}"
        )

        print(
            "Assistant probe verification: PASS"
        )

        return

    raise RuntimeError(
        "Unsupported command."
    )


if __name__ == "__main__":
    main()

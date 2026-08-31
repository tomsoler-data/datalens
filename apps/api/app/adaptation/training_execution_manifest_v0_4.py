from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess

from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# RULES / IDENTITIES
# ============================================================


BUILDER_RULE_VERSION = (
    "training_execution_manifest_builder_v0.4_v0.1"
)

MANIFEST_RULE_VERSION = (
    "training_execution_manifest_v0.2"
)

FREEZE_RULE_VERSION = (
    "training_execution_manifest_freeze_v0.2"
)


EXPERIMENT_ID = (
    "datalens-semantic-qlora-v0.4"
)

MANIFEST_ID = (
    "training-manifest:datalens-semantic-qlora:v0.4"
)

TRAINING_RUN_ID = (
    "training-run:datalens-semantic-qlora:v0.4:0001"
)


MANIFEST_RELATIVE_PATH = (
    "artifacts/adaptation/training/"
    "datalens_semantic_qlora_v0.4_"
    "training_v0.1_manifest.json"
)

FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/training/"
    "datalens_semantic_qlora_v0.4_"
    "training_v0.1_manifest_freeze.json"
)

FUTURE_RUNNER_RELATIVE_PATH = (
    "app/adaptation/training_runner_v0_4.py"
)

BUILDER_RELATIVE_PATH = (
    "app/adaptation/"
    "training_execution_manifest_v0_4.py"
)


# ============================================================
# FROZEN AUTHORITY PATHS
# ============================================================


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

OPTIMIZATION_POLICY_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/design/"
    "datalens_semantic_qlora_v0.4_"
    "optimization_policy_v0.1_freeze.json"
)

DATASET_RELATIVE_PATH = (
    "artifacts/adaptation/datasets/"
    "datalens_semantic_training_v0.4.jsonl"
)

DATASET_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/datasets/"
    "datalens_semantic_training_v0.4_freeze.json"
)

TOKEN_AUDIT_RELATIVE_PATH = (
    "artifacts/adaptation/probes/"
    "datalens_semantic_training_v0.4_"
    "token_length_audit.json"
)

SHARED_RUNTIME_RELATIVE_PATH = (
    "app/adaptation/qlora_runtime_v0_4.py"
)

RESOURCE_PREFLIGHT_RELATIVE_PATH = (
    "app/adaptation/resource_preflight_v0_4.py"
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
    "datalens_semantic_qlora_v0.4_"
    "assistant_only_probe.json"
)

AIRPORT_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/holdouts/"
    "datalens_semantic_qlora_v0.4_"
    "airport_ground_operations_holdout_v0.1_freeze.json"
)

FINAL_ACCEPTANCE_FREEZE_RELATIVE_PATH = (
    "artifacts/evaluation/holdouts/"
    "greenhouse_operations_final_acceptance_v0.1_freeze.json"
)


# ============================================================
# FROZEN SHAS
# ============================================================


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

EXPECTED_OPTIMIZATION_POLICY_FREEZE_SHA256 = (
    "3a5cf4123cefa25afe597ab057e86307"
    "e4b51505f4e7d0d98927f9a42476259c"
)

EXPECTED_DATASET_SHA256 = (
    "4fd00586f2d53d6de57f5cbc5f1d7bfb"
    "2e512960e60b30c28596aaefbac322b7"
)

EXPECTED_DATASET_FREEZE_SHA256 = (
    "c9ae4421becea37ee07bf964f82e2e4f"
    "ffd9328f2920cda7d371fe57e0eb1f70"
)

EXPECTED_TOKEN_AUDIT_SHA256 = (
    "add94bd50fc89120a7626fa7533af299"
    "6765d25c7c64d2d866dd7178674d40c1"
)

EXPECTED_SHARED_RUNTIME_SHA256 = (
    "20e41ab00606296893276a84e53746c0"
    "6618b8cabca74fef77cb743c5e80ab7c"
)

EXPECTED_RESOURCE_PREFLIGHT_SHA256 = (
    "2c7cd7803e41d4af12a9faa254f6f9cc"
    "7802b915907d3f266899695748da9ac2"
)

EXPECTED_MEMORY_PROBE_SHA256 = (
    "d0b4d49518d2b974ede2e621e8931d54"
    "1183476294cff0e671e304967a71bf85"
)

EXPECTED_OPTIMIZER_PROBE_SHA256 = (
    "f12dcbd2ed0bda1cf0136d3b512ffea3"
    "fe481e8fdd35a112d759adbc40b2d9cb"
)

EXPECTED_ASSISTANT_PROBE_SHA256 = (
    "f89f8a4a07ddbfb4b997a78ec823ea62"
    "6eb6427b60d7c19517410f8f26ad7895"
)

EXPECTED_AIRPORT_FREEZE_SHA256 = (
    "46accf23eeae32f0fdc926f7b0a9e731"
    "15a1413ddbc28efa84472f0571ee2f09"
)

EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256 = (
    "11615352547f152b91e592066142c343"
    "1b557c5f620018380b71b931cb77a736"
)


# ============================================================
# EXECUTION CONTROLS
# ============================================================


OPTIMIZER_BETAS = [
    0.9,
    0.999,
]

OPTIMIZER_EPS = 1e-8

OPTIMIZER_AMSGRAD = False

OPTIMIZER_MIN_8BIT_SIZE = 4096


EXPECTED_EXAMPLES = 230

EXPECTED_EPOCHS = 2

EXPECTED_MICRO_BATCH_SIZE = 1

EXPECTED_ACCUMULATION_STEPS = 8

EXPECTED_FULL_GROUPS_PER_EPOCH = 28

EXPECTED_PARTIAL_GROUPS_PER_EPOCH = 1

EXPECTED_PARTIAL_GROUP_SIZE = 6

EXPECTED_GROUPS_PER_EPOCH = 29

EXPECTED_OPTIMIZER_STEPS_PER_EPOCH = 29

EXPECTED_TOTAL_MICRO_BATCHES = 460

EXPECTED_TOTAL_OPTIMIZER_STEPS = 58

EXPECTED_EXAMPLE_PRESENTATIONS = 460

EXPECTED_DISCARDED_PRESENTATIONS = 0

EXPECTED_WARMUP_STEPS = 2


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

    return root.expanduser().resolve()


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


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


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


def git_output(
    *,
    root: Path,
    args: list[str],
) -> str:
    result = subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=root,
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


def git_head(
    *,
    root: Path,
) -> str:
    return git_output(
        root=root,
        args=[
            "rev-parse",
            "HEAD",
        ],
    ).strip()


def git_commit_for_path(
    *,
    root: Path,
    relative_path: str,
) -> str:
    value = git_output(
        root=root,
        args=[
            "log",
            "-1",
            "--format=%H",
            "--",
            relative_path,
        ],
    ).strip()

    if len(value) != 40:
        raise RuntimeError(
            (
                "Unable to resolve source commit for "
                f"{relative_path}"
            )
        )

    return value


def require_clean_tree(
    *,
    root: Path,
) -> None:
    status = git_output(
        root=root,
        args=[
            "status",
            "--porcelain",
        ],
    )

    if status.strip():
        raise RuntimeError(
            (
                "Manifest publication requires "
                "a clean working tree.\n"
                f"{status}"
            )
        )


def require_tracked(
    *,
    root: Path,
    relative_path: str,
) -> None:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            relative_path,
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            (
                "Required authority is not committed: "
                f"{relative_path}"
            )
        )


def path_for(
    *,
    root: Path,
    relative_path: str,
) -> Path:
    return (
        root
        /
        relative_path
    ).resolve()


def require_sha(
    *,
    root: Path,
    relative_path: str,
    expected: str,
) -> Path:
    path = path_for(
        root=root,
        relative_path=
            relative_path,
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual = sha256_file(
        path
    )

    if actual != expected:
        raise RuntimeError(
            (
                f"Authority SHA changed: {relative_path}\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )
        )

    return path


# ============================================================
# STATIC AUTHORITY VALIDATION
# ============================================================


def validate_static_authority(
    *,
    root: Path,
) -> dict[str, Any]:
    root = repository_root(
        root
    )

    authority_specs = {
        CONTRACT_RELATIVE_PATH:
            EXPECTED_CONTRACT_SHA256,

        CONTRACT_FREEZE_RELATIVE_PATH:
            EXPECTED_CONTRACT_FREEZE_SHA256,

        OPTIMIZATION_POLICY_RELATIVE_PATH:
            EXPECTED_OPTIMIZATION_POLICY_SHA256,

        OPTIMIZATION_POLICY_FREEZE_RELATIVE_PATH:
            EXPECTED_OPTIMIZATION_POLICY_FREEZE_SHA256,

        DATASET_RELATIVE_PATH:
            EXPECTED_DATASET_SHA256,

        DATASET_FREEZE_RELATIVE_PATH:
            EXPECTED_DATASET_FREEZE_SHA256,

        TOKEN_AUDIT_RELATIVE_PATH:
            EXPECTED_TOKEN_AUDIT_SHA256,

        SHARED_RUNTIME_RELATIVE_PATH:
            EXPECTED_SHARED_RUNTIME_SHA256,

        RESOURCE_PREFLIGHT_RELATIVE_PATH:
            EXPECTED_RESOURCE_PREFLIGHT_SHA256,

        MEMORY_PROBE_RELATIVE_PATH:
            EXPECTED_MEMORY_PROBE_SHA256,

        OPTIMIZER_PROBE_RELATIVE_PATH:
            EXPECTED_OPTIMIZER_PROBE_SHA256,

        ASSISTANT_PROBE_RELATIVE_PATH:
            EXPECTED_ASSISTANT_PROBE_SHA256,

        AIRPORT_FREEZE_RELATIVE_PATH:
            EXPECTED_AIRPORT_FREEZE_SHA256,

        FINAL_ACCEPTANCE_FREEZE_RELATIVE_PATH:
            EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256,
    }

    for relative_path, expected_sha in authority_specs.items():
        require_sha(
            root=root,
            relative_path=
                relative_path,
            expected=
                expected_sha,
        )

    contract = load_json(
        path_for(
            root=root,
            relative_path=
                CONTRACT_RELATIVE_PATH,
        )
    )

    contract_freeze = load_json(
        path_for(
            root=root,
            relative_path=
                CONTRACT_FREEZE_RELATIVE_PATH,
        )
    )

    optimization = load_json(
        path_for(
            root=root,
            relative_path=
                OPTIMIZATION_POLICY_RELATIVE_PATH,
        )
    )

    dataset_freeze = load_json(
        path_for(
            root=root,
            relative_path=
                DATASET_FREEZE_RELATIVE_PATH,
        )
    )

    memory_probe = load_json(
        path_for(
            root=root,
            relative_path=
                MEMORY_PROBE_RELATIVE_PATH,
        )
    )

    optimizer_probe = load_json(
        path_for(
            root=root,
            relative_path=
                OPTIMIZER_PROBE_RELATIVE_PATH,
        )
    )

    assistant_probe = load_json(
        path_for(
            root=root,
            relative_path=
                ASSISTANT_PROBE_RELATIVE_PATH,
        )
    )

    airport_freeze = load_json(
        path_for(
            root=root,
            relative_path=
                AIRPORT_FREEZE_RELATIVE_PATH,
        )
    )

    final_acceptance_freeze = load_json(
        path_for(
            root=root,
            relative_path=
                FINAL_ACCEPTANCE_FREEZE_RELATIVE_PATH,
        )
    )

    if (
        contract[
            "experiment_id"
        ]
        !=
        EXPERIMENT_ID
    ):
        raise RuntimeError(
            "Experiment ID changed."
        )

    training = contract[
        "training"
    ]

    expected_contract_training = {
        "random_seed":
            42,

        "max_sequence_length":
            256,

        "per_device_train_batch_size":
            1,

        "gradient_accumulation_steps":
            8,

        "num_train_epochs":
            2.0,

        "learning_rate":
            0.0002,

        "warmup_ratio":
            0.03,

        "weight_decay":
            0.0,

        "optimizer":
            "paged_adamw_8bit",

        "scheduler":
            "cosine",

        "gradient_checkpointing":
            True,

        "bf16":
            True,

        "fp16":
            False,
    }

    for key, expected in expected_contract_training.items():
        if (
            training[
                key
            ]
            !=
            expected
        ):
            raise RuntimeError(
                (
                    "Experiment contract training value changed: "
                    f"{key}"
                )
            )

    optimization_training = (
        optimization[
            "training"
        ]
    )

    expected_optimization_training = {
        "assistant_only_loss":
            True,

        "bf16":
            True,

        "fp16":
            False,

        "gradient_accumulation_steps":
            8,

        "gradient_checkpoint_use_reentrant":
            False,

        "gradient_checkpointing":
            True,

        "initial_epoch_budget":
            2,

        "learning_rate":
            0.0002,

        "max_sequence_length":
            256,

        "micro_batch_size":
            1,

        "optimizer":
            "paged_adamw_8bit",

        "packing":
            False,

        "random_seed":
            42,

        "scheduler":
            "cosine",

        "sequence_truncation":
            "fail_closed",

        "warmup_ratio":
            0.03,

        "weight_decay":
            0.0,
    }

    for key, expected in expected_optimization_training.items():
        if (
            optimization_training[
                key
            ]
            !=
            expected
        ):
            raise RuntimeError(
                (
                    "Optimization training authority changed: "
                    f"{key}"
                )
            )

    accumulation = (
        optimization[
            "accumulation"
        ]
    )

    expected_accumulation = {
        "cross_epoch_accumulation":
            False,

        "discard_incomplete_group":
            False,

        "discarded_example_presentations":
            0,

        "example_presentations":
            460,

        "full_groups_per_epoch":
            28,

        "gradient_accumulation_weighting":
            "supervised_token_weighted",

        "micro_batch_model_loss_reduction":
            "mean_over_supervised_assistant_tokens",

        "micro_batches_per_epoch":
            230,

        "nominal_effective_batch_size":
            8,

        "optimizer_steps_per_epoch":
            29,

        "partial_group_count_per_epoch":
            1,

        "partial_group_effective_batch_size":
            6,

        "partial_group_size":
            6,

        "policy":
            "flush_partial_group_at_epoch_end",

        "total_micro_batches":
            460,

        "total_optimizer_steps":
            58,
    }

    for key, expected in expected_accumulation.items():
        if (
            accumulation[
                key
            ]
            !=
            expected
        ):
            raise RuntimeError(
                (
                    "Accumulation authority changed: "
                    f"{key}"
                )
            )

    if (
        dataset_freeze[
            "contamination_match_count"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Training dataset contamination detected."
        )

    if (
        dataset_freeze[
            "provenance_violation_count"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Training dataset provenance violation detected."
        )

    if (
        dataset_freeze[
            "final_acceptance_tuning_input"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance contaminated training input."
        )

    if (
        dataset_freeze[
            "frozen_before_training"
        ]
        is not True
    ):
        raise RuntimeError(
            "Dataset was not frozen before training."
        )

    if (
        contract_freeze[
            "airport_independent_holdout"
        ][
            "freeze_sha256"
        ]
        !=
        EXPECTED_AIRPORT_FREEZE_SHA256
    ):
        raise RuntimeError(
            "Airport holdout freeze binding changed."
        )

    if (
        memory_probe[
            "passed"
        ]
        is not True
    ):
        raise RuntimeError(
            "Memory probe did not pass."
        )

    if (
        optimizer_probe[
            "passed"
        ]
        is not True
    ):
        raise RuntimeError(
            "Optimizer probe did not pass."
        )

    if (
        assistant_probe[
            "passed"
        ]
        is not True
    ):
        raise RuntimeError(
            "Assistant probe did not pass."
        )

    if (
        optimizer_probe[
            "source_memory_probe_sha256"
        ]
        !=
        EXPECTED_MEMORY_PROBE_SHA256
    ):
        raise RuntimeError(
            "Optimizer probe chain changed."
        )

    if (
        assistant_probe[
            "optimizer_memory_probe_sha256"
        ]
        !=
        EXPECTED_OPTIMIZER_PROBE_SHA256
    ):
        raise RuntimeError(
            "Assistant probe chain changed."
        )

    if (
        assistant_probe[
            "assistant_masking"
        ][
            "rule_version"
        ]
        !=
        "assistant_only_masking_v0.1"
    ):
        raise RuntimeError(
            "Assistant masking rule changed."
        )

    if (
        airport_freeze[
            "evaluation_executed"
        ]
        is not False
    ):
        raise RuntimeError(
            "Airport evaluation has already executed."
        )

    if (
        airport_freeze[
            "results_observed"
        ]
        is not False
    ):
        raise RuntimeError(
            "Airport results have already been observed."
        )

    if (
        airport_freeze[
            "used_for_training"
        ]
        is not False
    ):
        raise RuntimeError(
            "Airport holdout was used for training."
        )

    if (
        final_acceptance_freeze[
            "adaptation_tuning_input"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance was used for tuning."
        )

    if (
        final_acceptance_freeze[
            "training_started_at_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance freeze was not pre-training."
        )

    return {
        "contract":
            contract,

        "contract_freeze":
            contract_freeze,

        "optimization":
            optimization,

        "dataset_freeze":
            dataset_freeze,

        "memory_probe":
            memory_probe,

        "optimizer_probe":
            optimizer_probe,

        "assistant_probe":
            assistant_probe,

        "airport_freeze":
            airport_freeze,

        "final_acceptance_freeze":
            final_acceptance_freeze,
    }


# ============================================================
# MANIFEST COMPOSITION
# ============================================================


def build_manifest_payload(
    *,
    root: Path,
    builder_source_commit: str,
    build_git_head: str,
) -> dict[str, Any]:
    authority = validate_static_authority(
        root=root
    )

    contract = authority[
        "contract"
    ]

    optimization = authority[
        "optimization"
    ]

    dataset_freeze = authority[
        "dataset_freeze"
    ]

    memory_probe = authority[
        "memory_probe"
    ]

    optimizer_probe = authority[
        "optimizer_probe"
    ]

    assistant_probe = authority[
        "assistant_probe"
    ]

    preflight_commit = git_commit_for_path(
        root=root,
        relative_path=
            RESOURCE_PREFLIGHT_RELATIVE_PATH,
    )

    runtime_commit = git_commit_for_path(
        root=root,
        relative_path=
            SHARED_RUNTIME_RELATIVE_PATH,
    )

    training_dataset = contract[
        "training_dataset"
    ]

    base_model = contract[
        "base_model"
    ]

    quantization = contract[
        "quantization"
    ]

    lora = contract[
        "lora"
    ]

    evaluation = contract[
        "evaluation"
    ]

    final_acceptance = contract[
        "final_acceptance_holdout"
    ]

    return {
        "rule_version":
            MANIFEST_RULE_VERSION,

        "manifest_id":
            MANIFEST_ID,

        "training_run_id":
            TRAINING_RUN_ID,

        "experiment": {
            "experiment_id":
                EXPERIMENT_ID,

            "experiment_contract_relative_path":
                CONTRACT_RELATIVE_PATH,

            "experiment_contract_sha256":
                EXPECTED_CONTRACT_SHA256,

            "experiment_contract_freeze_relative_path":
                CONTRACT_FREEZE_RELATIVE_PATH,

            "experiment_contract_freeze_sha256":
                EXPECTED_CONTRACT_FREEZE_SHA256,

            "base_model_repository":
                base_model[
                    "repository"
                ],

            "base_model_revision":
                base_model[
                    "revision"
                ],

            "base_model_family":
                base_model[
                    "model_family"
                ],

            "modality_scope":
                base_model[
                    "modality_scope"
                ],

            "quantization":
                quantization,

            "lora":
                lora,
        },

        "dataset": {
            "dataset_id":
                training_dataset[
                    "dataset_id"
                ],

            "dataset_version":
                training_dataset[
                    "dataset_version"
                ],

            "relative_path":
                DATASET_RELATIVE_PATH,

            "dataset_sha256":
                EXPECTED_DATASET_SHA256,

            "freeze_relative_path":
                DATASET_FREEZE_RELATIVE_PATH,

            "freeze_sha256":
                EXPECTED_DATASET_FREEZE_SHA256,

            "example_count":
                EXPECTED_EXAMPLES,

            "contamination_match_count":
                dataset_freeze[
                    "contamination_match_count"
                ],

            "provenance_violation_count":
                dataset_freeze[
                    "provenance_violation_count"
                ],

            "final_acceptance_tuning_input":
                False,
        },

        "optimization_authority": {
            "relative_path":
                OPTIMIZATION_POLICY_RELATIVE_PATH,

            "sha256":
                EXPECTED_OPTIMIZATION_POLICY_SHA256,

            "freeze_relative_path":
                OPTIMIZATION_POLICY_FREEZE_RELATIVE_PATH,

            "freeze_sha256":
                EXPECTED_OPTIMIZATION_POLICY_FREEZE_SHA256,
        },

        "training": {
            "adaptation_method":
                "qlora",

            "max_sequence_length":
                256,

            "micro_batch_size":
                EXPECTED_MICRO_BATCH_SIZE,

            "gradient_accumulation_steps":
                EXPECTED_ACCUMULATION_STEPS,

            "nominal_effective_batch_size":
                8,

            "epochs":
                EXPECTED_EPOCHS,

            "micro_batches_per_epoch":
                EXPECTED_EXAMPLES,

            "full_accumulation_groups_per_epoch":
                EXPECTED_FULL_GROUPS_PER_EPOCH,

            "partial_accumulation_groups_per_epoch":
                EXPECTED_PARTIAL_GROUPS_PER_EPOCH,

            "terminal_partial_group_size":
                EXPECTED_PARTIAL_GROUP_SIZE,

            "accumulation_groups_per_epoch":
                EXPECTED_GROUPS_PER_EPOCH,

            "optimizer_steps_per_epoch":
                EXPECTED_OPTIMIZER_STEPS_PER_EPOCH,

            "total_micro_batches":
                EXPECTED_TOTAL_MICRO_BATCHES,

            "total_optimizer_steps":
                EXPECTED_TOTAL_OPTIMIZER_STEPS,

            "example_presentations":
                EXPECTED_EXAMPLE_PRESENTATIONS,

            "discarded_example_presentations":
                EXPECTED_DISCARDED_PRESENTATIONS,

            "cross_epoch_accumulation":
                False,

            "discard_incomplete_group":
                False,

            "accumulation_group_policy":
                "full_groups_plus_terminal_partial_group",

            "partial_group_policy":
                "flush_partial_group_at_epoch_end",

            "partial_group_optimizer_step":
                True,

            "gradient_accumulation_weighting":
                "supervised_token_weighted",

            "gradient_accumulation_scale_formula":
                (
                    "micro_batch_loss * "
                    "micro_batch_supervised_tokens / "
                    "accumulation_group_supervised_tokens"
                ),

            "partial_group_loss_denominator":
                "actual_accumulation_group_supervised_tokens",

            "micro_batch_model_loss_reduction":
                "mean_over_supervised_assistant_tokens",

            "effective_objective":
                (
                    "mean_cross_entropy_over_all_"
                    "supervised_assistant_tokens_in_"
                    "accumulation_group"
                ),

            "optimizer":
                "paged_adamw_8bit",

            "optimizer_implementation":
                "bitsandbytes.optim.PagedAdamW8bit",

            "learning_rate":
                0.0002,

            "optimizer_betas":
                OPTIMIZER_BETAS,

            "optimizer_eps":
                OPTIMIZER_EPS,

            "weight_decay":
                0.0,

            "optimizer_amsgrad":
                OPTIMIZER_AMSGRAD,

            "optimizer_min_8bit_size":
                OPTIMIZER_MIN_8BIT_SIZE,

            "optimizer_zero_grad_policy":
                "set_to_none_true_after_each_optimizer_step",

            "scheduler":
                "cosine",

            "scheduler_implementation":
                "transformers.get_cosine_schedule_with_warmup",

            "scheduler_step_order":
                "optimizer_then_scheduler",

            "warmup_ratio":
                0.03,

            "warmup_rounding_policy":
                "ceil_with_minimum_one",

            "warmup_steps":
                EXPECTED_WARMUP_STEPS,

            "seed":
                42,

            "shuffle":
                True,

            "shuffle_policy":
                "deterministic_epoch_seed",

            "epoch_seed_policy":
                "seed_plus_zero_based_epoch",

            "gradient_checkpointing":
                True,

            "gradient_checkpoint_use_reentrant":
                False,

            "gradient_clipping":
                None,

            "bf16":
                True,

            "fp16":
                False,

            "packing":
                False,

            "drop_last":
                False,

            "dataloader_num_workers":
                0,

            "silent_truncation_allowed":
                False,
        },

        "supervision": {
            "assistant_only":
                True,

            "masking_rule_version":
                assistant_probe[
                    "assistant_masking"
                ][
                    "rule_version"
                ],

            "ignore_index":
                -100,

            "assistant_supervised":
                True,

            "prompt_supervised":
                False,

            "padding_supervised":
                False,

            "training_loss_is_acceptance_evidence":
                False,
        },

        "preflight_evidence": {
            "resource_preflight_relative_path":
                RESOURCE_PREFLIGHT_RELATIVE_PATH,

            "resource_preflight_sha256":
                EXPECTED_RESOURCE_PREFLIGHT_SHA256,

            "resource_preflight_source_git_commit":
                preflight_commit,

            "token_audit_relative_path":
                TOKEN_AUDIT_RELATIVE_PATH,

            "token_audit_sha256":
                EXPECTED_TOKEN_AUDIT_SHA256,

            "sequence_memory_probe": {
                "relative_path":
                    MEMORY_PROBE_RELATIVE_PATH,

                "sha256":
                    EXPECTED_MEMORY_PROBE_SHA256,

                "execution_git_commit":
                    memory_probe[
                        "git_head"
                    ],

                "passed":
                    True,
            },

            "optimizer_state_memory_probe": {
                "relative_path":
                    OPTIMIZER_PROBE_RELATIVE_PATH,

                "sha256":
                    EXPECTED_OPTIMIZER_PROBE_SHA256,

                "execution_git_commit":
                    optimizer_probe[
                        "git_head"
                    ],

                "source_memory_probe_sha256":
                    EXPECTED_MEMORY_PROBE_SHA256,

                "passed":
                    True,
            },

            "assistant_only_gpu_probe": {
                "relative_path":
                    ASSISTANT_PROBE_RELATIVE_PATH,

                "sha256":
                    EXPECTED_ASSISTANT_PROBE_SHA256,

                "execution_git_commit":
                    assistant_probe[
                        "git_head"
                    ],

                "source_optimizer_probe_sha256":
                    EXPECTED_OPTIMIZER_PROBE_SHA256,

                "passed":
                    True,
            },
        },

        "protected_evaluation": {
            "airport_independent_holdout": {
                "freeze_relative_path":
                    AIRPORT_FREEZE_RELATIVE_PATH,

                "freeze_sha256":
                    EXPECTED_AIRPORT_FREEZE_SHA256,

                "independent_holdout":
                    True,

                "used_for_training":
                    False,

                "used_for_hyperparameter_tuning":
                    False,

                "evaluation_executed_before_training":
                    False,

                "results_observed_before_training":
                    False,
            },

            "final_acceptance": {
                "artifact_id":
                    final_acceptance[
                        "artifact_id"
                    ],

                "freeze_relative_path":
                    FINAL_ACCEPTANCE_FREEZE_RELATIVE_PATH,

                "freeze_sha256":
                    EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256,

                "used_for_training":
                    False,

                "used_for_hyperparameter_tuning":
                    False,

                "cases_loaded_before_training":
                    False,

                "evaluation_executed_before_training":
                    False,
            },
        },

        "evaluation_policy":
            evaluation,

        "authorization": {
            "optimizer_step_authorized_at_manifest_creation":
                False,

            "training_execution_authorized_at_manifest_creation":
                False,

            "optimizer_step_requires_runtime_execution_gates":
                True,

            "runner_must_bind_exact_manifest":
                True,

            "runner_must_bind_exact_manifest_freeze":
                True,

            "runner_must_bind_exact_shared_runtime":
                True,

            "runner_must_bind_exact_probe_chain":
                True,

            "manifest_commit_must_be_ancestor_of_training_start_head":
                True,

            "resource_preflight_source_commit_must_be_ancestor_of_training_start_head":
                True,

            "training_runner_commit_must_equal_training_start_head":
                True,

            "airport_evaluation_authorized_before_training":
                False,

            "final_acceptance_evaluation_authorized_before_training":
                False,
        },

        "execution_safety": {
            "training_started_at_manifest_creation":
                False,

            "optimizer_step_executed_at_manifest_creation":
                False,

            "model_weights_modified_at_manifest_creation":
                False,

            "adapter_saved_at_manifest_creation":
                False,

            "airport_evaluated_at_manifest_creation":
                False,

            "final_acceptance_evaluated_at_manifest_creation":
                False,
        },

        "git_execution_binding": {
            "manifest_builder_relative_path":
                BUILDER_RELATIVE_PATH,

            "manifest_builder_source_git_commit":
                builder_source_commit,

            "manifest_build_git_head":
                build_git_head,

            "shared_runtime_relative_path":
                SHARED_RUNTIME_RELATIVE_PATH,

            "shared_runtime_sha256":
                EXPECTED_SHARED_RUNTIME_SHA256,

            "shared_runtime_source_git_commit":
                runtime_commit,

            "training_runner_relative_path":
                FUTURE_RUNNER_RELATIVE_PATH,

            "training_runner_existed_at_manifest_creation":
                False,

            "manifest_must_be_committed_before_training_runner_execution":
                True,

            "clean_working_tree_required_at_training_start":
                True,
        },

        "planned_outputs": {
            "adapter_directory":
                (
                    "artifacts/adaptation/adapters/"
                    "datalens_semantic_qlora_v0.4_adapter"
                ),

            "training_report":
                (
                    "artifacts/adaptation/training/"
                    "datalens_semantic_qlora_v0.4_"
                    "training_v0.1_report.json"
                ),

            "training_receipt":
                (
                    "artifacts/adaptation/training/"
                    "datalens_semantic_qlora_v0.4_"
                    "training_v0.1_receipt.json"
                ),
        },
    }


# ============================================================
# MANIFEST VALIDATION
# ============================================================


def validate_manifest_payload(
    payload: Mapping[str, Any],
) -> None:
    if (
        payload[
            "rule_version"
        ]
        !=
        MANIFEST_RULE_VERSION
    ):
        raise RuntimeError(
            "Manifest rule changed."
        )

    if (
        payload[
            "manifest_id"
        ]
        !=
        MANIFEST_ID
    ):
        raise RuntimeError(
            "Manifest ID changed."
        )

    if (
        payload[
            "training_run_id"
        ]
        !=
        TRAINING_RUN_ID
    ):
        raise RuntimeError(
            "Training run ID changed."
        )

    training = payload[
        "training"
    ]

    exact_training = {
        "micro_batch_size":
            1,

        "gradient_accumulation_steps":
            8,

        "epochs":
            2,

        "micro_batches_per_epoch":
            230,

        "full_accumulation_groups_per_epoch":
            28,

        "partial_accumulation_groups_per_epoch":
            1,

        "terminal_partial_group_size":
            6,

        "accumulation_groups_per_epoch":
            29,

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

        "discard_incomplete_group":
            False,

        "partial_group_policy":
            "flush_partial_group_at_epoch_end",

        "partial_group_loss_denominator":
            "actual_accumulation_group_supervised_tokens",

        "gradient_accumulation_weighting":
            "supervised_token_weighted",

        "optimizer":
            "paged_adamw_8bit",

        "optimizer_betas":
            [
                0.9,
                0.999,
            ],

        "optimizer_eps":
            1e-8,

        "optimizer_amsgrad":
            False,

        "optimizer_min_8bit_size":
            4096,

        "learning_rate":
            0.0002,

        "weight_decay":
            0.0,

        "scheduler":
            "cosine",

        "warmup_ratio":
            0.03,

        "warmup_rounding_policy":
            "ceil_with_minimum_one",

        "warmup_steps":
            2,

        "gradient_checkpointing":
            True,

        "gradient_checkpoint_use_reentrant":
            False,

        "bf16":
            True,

        "fp16":
            False,

        "packing":
            False,

        "silent_truncation_allowed":
            False,
    }

    for key, expected in exact_training.items():
        if (
            training[
                key
            ]
            !=
            expected
        ):
            raise RuntimeError(
                (
                    "Manifest training semantic changed: "
                    f"{key}"
                )
            )

    if (
        training[
            "full_accumulation_groups_per_epoch"
        ]
        *
        training[
            "gradient_accumulation_steps"
        ]
        +
        training[
            "terminal_partial_group_size"
        ]
        !=
        training[
            "micro_batches_per_epoch"
        ]
    ):
        raise RuntimeError(
            "Accumulation arithmetic failed."
        )

    if (
        training[
            "accumulation_groups_per_epoch"
        ]
        *
        training[
            "epochs"
        ]
        !=
        training[
            "total_optimizer_steps"
        ]
    ):
        raise RuntimeError(
            "Optimizer-step arithmetic failed."
        )

    calculated_warmup = max(
        1,
        math.ceil(
            training[
                "warmup_ratio"
            ]
            *
            training[
                "total_optimizer_steps"
            ]
        ),
    )

    if (
        calculated_warmup
        !=
        training[
            "warmup_steps"
        ]
    ):
        raise RuntimeError(
            "Warmup-step derivation failed."
        )

    if (
        payload[
            "authorization"
        ][
            "optimizer_step_authorized_at_manifest_creation"
        ]
        is not False
    ):
        raise RuntimeError(
            "Manifest prematurely authorizes optimizer.step()."
        )

    if (
        payload[
            "authorization"
        ][
            "training_execution_authorized_at_manifest_creation"
        ]
        is not False
    ):
        raise RuntimeError(
            "Manifest prematurely authorizes training."
        )

    for key in (
        "sequence_memory_probe",
        "optimizer_state_memory_probe",
        "assistant_only_gpu_probe",
    ):
        if (
            payload[
                "preflight_evidence"
            ][
                key
            ][
                "passed"
            ]
            is not True
        ):
            raise RuntimeError(
                f"Preflight evidence failed: {key}"
            )


# ============================================================
# FREEZE COMPOSITION
# ============================================================


def build_freeze_payload(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    frozen_at: str,
) -> dict[str, Any]:
    return {
        "rule_version":
            FREEZE_RULE_VERSION,

        "freeze_id":
            "training-manifest-freeze:datalens-semantic-qlora:v0.4",

        "manifest_id":
            MANIFEST_ID,

        "training_run_id":
            TRAINING_RUN_ID,

        "manifest_relative_path":
            MANIFEST_RELATIVE_PATH,

        "manifest_sha256":
            manifest_sha256,

        "frozen_at":
            frozen_at,

        "frozen_before_training":
            True,

        "experiment_contract_sha256":
            EXPECTED_CONTRACT_SHA256,

        "experiment_contract_freeze_sha256":
            EXPECTED_CONTRACT_FREEZE_SHA256,

        "optimization_policy_sha256":
            EXPECTED_OPTIMIZATION_POLICY_SHA256,

        "optimization_policy_freeze_sha256":
            EXPECTED_OPTIMIZATION_POLICY_FREEZE_SHA256,

        "training_dataset_sha256":
            EXPECTED_DATASET_SHA256,

        "training_dataset_freeze_sha256":
            EXPECTED_DATASET_FREEZE_SHA256,

        "token_audit_sha256":
            EXPECTED_TOKEN_AUDIT_SHA256,

        "shared_runtime_sha256":
            EXPECTED_SHARED_RUNTIME_SHA256,

        "resource_preflight_sha256":
            EXPECTED_RESOURCE_PREFLIGHT_SHA256,

        "sequence_memory_probe_sha256":
            EXPECTED_MEMORY_PROBE_SHA256,

        "optimizer_state_memory_probe_sha256":
            EXPECTED_OPTIMIZER_PROBE_SHA256,

        "assistant_only_gpu_probe_sha256":
            EXPECTED_ASSISTANT_PROBE_SHA256,

        "airport_holdout_freeze_sha256":
            EXPECTED_AIRPORT_FREEZE_SHA256,

        "final_acceptance_freeze_sha256":
            EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256,

        "assistant_masking_rule_version":
            "assistant_only_masking_v0.1",

        "gradient_accumulation_weighting":
            "supervised_token_weighted",

        "partial_group_policy":
            "flush_partial_group_at_epoch_end",

        "partial_group_size":
            6,

        "optimizer_steps_per_epoch":
            29,

        "total_optimizer_steps":
            58,

        "warmup_steps":
            2,

        "optimizer_betas":
            [
                0.9,
                0.999,
            ],

        "optimizer_eps":
            1e-8,

        "optimizer_amsgrad":
            False,

        "optimizer_min_8bit_size":
            4096,

        "manifest_builder_source_git_commit":
            manifest[
                "git_execution_binding"
            ][
                "manifest_builder_source_git_commit"
            ],

        "manifest_build_git_head":
            manifest[
                "git_execution_binding"
            ][
                "manifest_build_git_head"
            ],

        "resource_preflight_source_git_commit":
            manifest[
                "preflight_evidence"
            ][
                "resource_preflight_source_git_commit"
            ],

        "shared_runtime_source_git_commit":
            manifest[
                "git_execution_binding"
            ][
                "shared_runtime_source_git_commit"
            ],

        "git_binding_policy":
            "manifest_commit_ancestor_and_runner_commit_equals_training_start_head",

        "manifest_commit_must_be_ancestor_of_training_start_head":
            True,

        "preflight_source_commit_must_be_ancestor_of_training_start_head":
            True,

        "training_runner_relative_path":
            FUTURE_RUNNER_RELATIVE_PATH,

        "training_runner_existed_at_freeze":
            False,

        "training_runner_commit_must_equal_training_start_head":
            True,

        "optimizer_step_requires_runtime_execution_gates":
            True,

        "optimizer_step_authorized_at_freeze":
            False,

        "optimizer_step_executed_at_freeze":
            False,

        "training_started_at_freeze":
            False,

        "model_weights_modified_at_freeze":
            False,

        "adapter_saved_at_freeze":
            False,

        "airport_cases_loaded_at_freeze":
            False,

        "airport_evaluated_at_freeze":
            False,

        "airport_results_observed_at_freeze":
            False,

        "final_acceptance_cases_loaded_at_freeze":
            False,

        "final_acceptance_evaluated_at_freeze":
            False,
    }


def validate_freeze_payload(
    payload: Mapping[str, Any],
) -> None:
    if (
        payload[
            "rule_version"
        ]
        !=
        FREEZE_RULE_VERSION
    ):
        raise RuntimeError(
            "Freeze rule changed."
        )

    if (
        payload[
            "frozen_before_training"
        ]
        is not True
    ):
        raise RuntimeError(
            "Manifest freeze is not pre-training."
        )

    required_false = (
        "training_runner_existed_at_freeze",
        "optimizer_step_authorized_at_freeze",
        "optimizer_step_executed_at_freeze",
        "training_started_at_freeze",
        "model_weights_modified_at_freeze",
        "adapter_saved_at_freeze",
        "airport_cases_loaded_at_freeze",
        "airport_evaluated_at_freeze",
        "airport_results_observed_at_freeze",
        "final_acceptance_cases_loaded_at_freeze",
        "final_acceptance_evaluated_at_freeze",
    )

    for key in required_false:
        if (
            payload[
                key
            ]
            is not False
        ):
            raise RuntimeError(
                (
                    "Freeze safety invariant failed: "
                    f"{key}"
                )
            )

    if (
        payload[
            "total_optimizer_steps"
        ]
        !=
        58
    ):
        raise RuntimeError(
            "Frozen optimizer-step count changed."
        )

    if (
        payload[
            "partial_group_size"
        ]
        !=
        6
    ):
        raise RuntimeError(
            "Frozen partial-group size changed."
        )

    if (
        payload[
            "warmup_steps"
        ]
        !=
        2
    ):
        raise RuntimeError(
            "Frozen warmup-step count changed."
        )


# ============================================================
# PUBLICATION
# ============================================================


def publish_new_file(
    *,
    path: Path,
    payload: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "xb"
    ) as handle:
        handle.write(
            payload
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )


def build_artifacts(
    *,
    root: Path,
) -> tuple[str, str]:
    root = repository_root(
        root
    )

    require_clean_tree(
        root=root
    )

    for relative_path in (
        BUILDER_RELATIVE_PATH,
        CONTRACT_RELATIVE_PATH,
        CONTRACT_FREEZE_RELATIVE_PATH,
        OPTIMIZATION_POLICY_RELATIVE_PATH,
        OPTIMIZATION_POLICY_FREEZE_RELATIVE_PATH,
        DATASET_RELATIVE_PATH,
        DATASET_FREEZE_RELATIVE_PATH,
        TOKEN_AUDIT_RELATIVE_PATH,
        SHARED_RUNTIME_RELATIVE_PATH,
        RESOURCE_PREFLIGHT_RELATIVE_PATH,
        MEMORY_PROBE_RELATIVE_PATH,
        OPTIMIZER_PROBE_RELATIVE_PATH,
        ASSISTANT_PROBE_RELATIVE_PATH,
        AIRPORT_FREEZE_RELATIVE_PATH,
        FINAL_ACCEPTANCE_FREEZE_RELATIVE_PATH,
    ):
        require_tracked(
            root=root,
            relative_path=
                relative_path,
        )

    future_runner = path_for(
        root=root,
        relative_path=
            FUTURE_RUNNER_RELATIVE_PATH,
    )

    if future_runner.exists():
        raise RuntimeError(
            (
                "v0.4 training runner already exists. "
                "Manifest must freeze before runner creation."
            )
        )

    manifest_path = path_for(
        root=root,
        relative_path=
            MANIFEST_RELATIVE_PATH,
    )

    freeze_path = path_for(
        root=root,
        relative_path=
            FREEZE_RELATIVE_PATH,
    )

    if manifest_path.exists():
        raise FileExistsError(
            manifest_path
        )

    if freeze_path.exists():
        raise FileExistsError(
            freeze_path
        )

    build_head = git_head(
        root=root
    )

    builder_commit = git_commit_for_path(
        root=root,
        relative_path=
            BUILDER_RELATIVE_PATH,
    )

    manifest = build_manifest_payload(
        root=root,
        builder_source_commit=
            builder_commit,

        build_git_head=
            build_head,
    )

    validate_manifest_payload(
        manifest
    )

    manifest_bytes = canonical_json_bytes(
        manifest
    )

    manifest_sha = sha256_bytes(
        manifest_bytes
    )

    freeze = build_freeze_payload(
        manifest=
            manifest,

        manifest_sha256=
            manifest_sha,

        frozen_at=
            utc_now(),
    )

    validate_freeze_payload(
        freeze
    )

    freeze_bytes = canonical_json_bytes(
        freeze
    )

    freeze_sha = sha256_bytes(
        freeze_bytes
    )

    manifest_written = False

    try:
        publish_new_file(
            path=
                manifest_path,

            payload=
                manifest_bytes,
        )

        manifest_written = True

        publish_new_file(
            path=
                freeze_path,

            payload=
                freeze_bytes,
        )

    except Exception:
        if (
            manifest_written
            and
            manifest_path.is_file()
            and
            sha256_file(
                manifest_path
            )
            ==
            manifest_sha
        ):
            manifest_path.unlink()

        raise

    return (
        manifest_sha,
        freeze_sha,
    )


# ============================================================
# STATIC OUTPUT
# ============================================================


def print_static_validation(
    *,
    root: Path,
) -> None:
    validate_static_authority(
        root=root
    )

    preflight_commit = git_commit_for_path(
        root=root,
        relative_path=
            RESOURCE_PREFLIGHT_RELATIVE_PATH,
    )

    manifest = build_manifest_payload(
        root=root,
        builder_source_commit=
            "0" * 40,

        build_git_head=
            "1" * 40,
    )

    validate_manifest_payload(
        manifest
    )

    manifest_sha = sha256_bytes(
        canonical_json_bytes(
            manifest
        )
    )

    freeze = build_freeze_payload(
        manifest=
            manifest,

        manifest_sha256=
            manifest_sha,

        frozen_at=
            "2000-01-01T00:00:00Z",
    )

    validate_freeze_payload(
        freeze
    )

    print(
        "=== DATALENS QLORA v0.4 TRAINING EXECUTION MANIFEST STATIC VALIDATION v0.1 ==="
    )

    print()

    print(
        "AUTHORITIES"
    )

    print(
        "  Experiment contract: PASS"
    )

    print(
        "  Optimization policy: PASS"
    )

    print(
        "  Dataset freeze: PASS"
    )

    print(
        "  Token audit: PASS"
    )

    print(
        "  Shared runtime: PASS"
    )

    print(
        "  Resource preflight: PASS"
    )

    print(
        "  Memory probe: PASS"
    )

    print(
        "  Optimizer-state probe: PASS"
    )

    print(
        "  Assistant-only probe: PASS"
    )

    print()

    print(
        "TRAINING PLAN"
    )

    print(
        "  Examples / epoch: 230"
    )

    print(
        "  Epochs: 2"
    )

    print(
        "  Full groups / epoch: 28 x 8"
    )

    print(
        "  Terminal partial group / epoch: 6"
    )

    print(
        "  Optimizer steps / epoch: 29"
    )

    print(
        "  Total optimizer steps: 58"
    )

    print(
        "  Example presentations: 460"
    )

    print(
        "  Discarded presentations: 0"
    )

    print(
        "  Cross-epoch accumulation: False"
    )

    print()

    print(
        "OPTIMIZER"
    )

    print(
        "  PagedAdamW8bit: PASS"
    )

    print(
        "  betas=[0.9, 0.999]: PASS"
    )

    print(
        "  eps=1e-8: PASS"
    )

    print(
        "  amsgrad=False: PASS"
    )

    print(
        "  min_8bit_size=4096: PASS"
    )

    print()

    print(
        "OBJECTIVE"
    )

    print(
        "  Assistant-only: PASS"
    )

    print(
        "  Supervised-token weighted: PASS"
    )

    print(
        "  Partial denominator=actual group tokens: PASS"
    )

    print(
        "  Silent truncation=False: PASS"
    )

    print()

    print(
        "SCHEDULER"
    )

    print(
        "  cosine: PASS"
    )

    print(
        "  warmup_ratio=0.03: PASS"
    )

    print(
        "  warmup rounding=ceil_with_minimum_one: PASS"
    )

    print(
        "  warmup_steps=2: PASS"
    )

    print()

    print(
        "GIT / AUTHORIZATION"
    )

    print(
        f"  Preflight source commit: {preflight_commit}"
    )

    print(
        "  Future runner exists: False"
    )

    print(
        "  optimizer.step() authorized now: False"
    )

    print(
        "  Training authorized now: False"
    )

    print(
        "  Airport evaluation authorized now: False"
    )

    print(
        "  Final Acceptance authorized now: False"
    )

    print()

    print(
        "SAFETY"
    )

    print(
        "  Heavy ML imports: False"
    )

    print(
        "  CUDA requested: False"
    )

    print(
        "  Model loaded: False"
    )

    print(
        "  Optimizer created: False"
    )

    print(
        "  Training executed: False"
    )

    print()

    print(
        "DATALENS QLORA v0.4 TRAINING EXECUTION MANIFEST STATIC VALIDATION: PASS"
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
            "build",
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

    if (
        args.command
        ==
        "validate-static"
    ):
        print_static_validation(
            root=root
        )

        return

    if (
        args.command
        ==
        "build"
    ):
        manifest_sha, freeze_sha = (
            build_artifacts(
                root=root
            )
        )

        print(
            "=== DATALENS QLORA v0.4 TRAINING EXECUTION MANIFEST BUILD ==="
        )

        print()

        print(
            f"Manifest SHA256: {manifest_sha}"
        )

        print(
            f"Freeze SHA256:   {freeze_sha}"
        )

        print()

        print(
            "optimizer.step() authorized: False"
        )

        print(
            "Training executed: False"
        )

        print(
            "Airport evaluated: False"
        )

        print(
            "Final Acceptance evaluated: False"
        )

        print()

        print(
            "DATALENS QLORA v0.4 TRAINING EXECUTION MANIFEST BUILD: PASS"
        )

        return

    raise RuntimeError(
        "Unsupported command."
    )


if __name__ == "__main__":
    main()

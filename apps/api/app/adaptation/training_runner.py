from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import (
    Any,
    Iterable,
    Sequence,
)


RUNNER_RULE_VERSION = (
    "qlora_training_runner_v0.1"
)

TRAINING_RUN_ID = (
    "datalens-semantic-qlora-v0.3-training-v0.1"
)

EXPECTED_MANIFEST_SHA256 = (
    "5c6edc1f91f1c5fe53a5402d01873f9"
    "f0e3b9ce7a7ef4cb0aa4087a926552664"
)

EXPECTED_MANIFEST_FREEZE_SHA256 = (
    "096759f95ebea93df270628157afc904"
    "ee3b7c46ba2170e823a61f117f700a6e"
)

EXPECTED_PREFLIGHT_SOURCE_COMMIT = (
    "280fef029d6d77584d0e00412e8f217ca7d89174"
)

EXPECTED_CONTRACT_SHA256 = (
    "609954fe4f06ace47000475053dcb011a"
    "4337e36122678860fb744eff645f92e"
)

EXPECTED_DATASET_SHA256 = (
    "4d7f1d8defeeb956448e31d776e531079"
    "5326b08ec5ddaf24091bd493c42f892"
)

EXPECTED_MEMORY_PROBE_SHA256 = (
    "74d4927e3c25077de5228c68db71ad404"
    "a2d008c4eb1ffe4a75f9349d121138a"
)

EXPECTED_OPTIMIZER_PROBE_SHA256 = (
    "16f36e609e506518822b0a6f95d2f281"
    "14cb4608208a29f8eff120965faf0719"
)

EXPECTED_ASSISTANT_PROBE_SHA256 = (
    "a56e14d637677fb5f56eed7cd56113a0"
    "db05314af6b090ba69055ede5651ea68"
)

EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256 = (
    "11615352547f152b91e592066142c343"
    "1b557c5f620018380b71b931cb77a736"
)

EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "7de1c58e208eda46e9c7f86397df37ec"
    "49883aeece39fb961e0a6b24088dd3c4"
)

EXPECTED_BASE_CONFIG_SHA256 = (
    "92d1975005be74a96582b41f59e846fe"
    "5f9ab7bd829d9b14a9afbe26708cba8f"
)

EXPECTED_BASE_INDEX_SHA256 = (
    "2cc9b808d15ac36a4ef5d12dc1df774"
    "c5bfa5c186a108671612a85265d3d2b64"
)

EXPECTED_BASE_SHARD_SHA256 = {
    (
        "bacf289ba8e7cb2d5a1d154a51d706b9"
        "7ed1277b713e38d72797c6c654335b96"
    ),
    (
        "0e412c790323463c40ccd6f3fd5ec028"
        "a02ecff7c6fe2269ab0e436afca42606"
    ),
}

EXPECTED_TORCH_VERSION = (
    "2.11.0+cu128"
)

EXPECTED_TRANSFORMERS_VERSION = (
    "5.16.1"
)

EXPECTED_PEFT_VERSION = (
    "0.20.0"
)

EXPECTED_BITSANDBYTES_VERSION = (
    "0.50.2"
)

EXPECTED_EXAMPLE_COUNT = 40

EXPECTED_TOTAL_TOKENS = 2769

EXPECTED_SUPERVISED_TOKENS = 1476

EXPECTED_MAX_EXAMPLE_TOKENS = 85

EXPECTED_TARGET_COUNT = 238

EXPECTED_TRAINABLE_PARAMETERS = (
    29_802_496
)

EXPECTED_TRAINABLE_TENSORS = 476

EXPECTED_TOTAL_MICRO_BATCHES = 80

EXPECTED_TOTAL_OPTIMIZER_STEPS = 10


def api_root() -> Path:
    return (
        Path(
            __file__
        )
        .resolve()
        .parents[
            2
        ]
    )


def git_root() -> Path:
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "--show-toplevel",
        ],
        cwd=api_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return Path(
        result.stdout.strip()
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


def canonical_json_bytes(
    payload: dict[str, Any],
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
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return payload


def git_relative_path(
    path: Path,
) -> str:
    return (
        path
        .resolve()
        .relative_to(
            git_root().resolve()
        )
        .as_posix()
    )


def run_git_text(
    arguments: Sequence[str],
) -> str:
    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=git_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            (
                "Git command failed: "
                f"git {' '.join(arguments)}\n"
                f"{result.stderr}"
            )
        )

    return result.stdout.strip()


def git_blob_bytes(
    *,
    ref: str,
    path: Path,
) -> bytes:
    relative = git_relative_path(
        path
    )

    result = subprocess.run(
        [
            "git",
            "show",
            f"{ref}:{relative}",
        ],
        cwd=git_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            (
                "Unable to read committed Git blob: "
                f"{ref}:{relative}\n"
                f"{result.stderr.decode(errors='replace')}"
            )
        )

    return result.stdout


def git_blob_sha256(
    *,
    ref: str,
    path: Path,
) -> str:
    return sha256_bytes(
        git_blob_bytes(
            ref=ref,
            path=path,
        )
    )


def git_is_clean() -> bool:
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        cwd=git_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return not bool(
        result.stdout.strip()
    )


def git_path_is_tracked(
    path: Path,
) -> bool:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            git_relative_path(
                path
            ),
        ],
        cwd=git_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    return (
        result.returncode
        ==
        0
    )


def git_latest_commit_for_path(
    path: Path,
) -> str:
    return run_git_text(
        [
            "log",
            "-1",
            "--format=%H",
            "--",
            git_relative_path(
                path
            ),
        ]
    )


def git_is_ancestor(
    ancestor: str,
    descendant: str,
) -> bool:
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        cwd=git_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode == 0:
        return True

    if result.returncode == 1:
        return False

    raise RuntimeError(
        "git merge-base --is-ancestor failed."
    )


def manifest_path(
    root: Path | None = None,
) -> Path:
    root = (
        root
        or
        api_root()
    )

    return (
        root
        / "artifacts"
        / "adaptation"
        / "training"
        / (
            "datalens_semantic_qlora_v0.3_"
            "training_v0.1_manifest.json"
        )
    )


def manifest_freeze_path(
    root: Path | None = None,
) -> Path:
    root = (
        root
        or
        api_root()
    )

    return (
        root
        / "artifacts"
        / "adaptation"
        / "training"
        / (
            "datalens_semantic_qlora_v0.3_"
            "training_v0.1_manifest_freeze.json"
        )
    )


def resolve_api_relative_path(
    relative_path: str,
) -> Path:
    path = (
        api_root()
        /
        relative_path
    ).resolve()

    try:
        path.relative_to(
            api_root().resolve()
        )

    except ValueError as error:
        raise RuntimeError(
            "Manifest path escapes API root."
        ) from error

    return path


def deterministic_epoch_order(
    *,
    example_count: int,
    seed: int,
    zero_based_epoch: int,
) -> list[int]:
    if example_count <= 0:
        raise ValueError(
            "example_count must be positive."
        )

    if zero_based_epoch < 0:
        raise ValueError(
            "zero_based_epoch must be non-negative."
        )

    order = list(
        range(
            example_count
        )
    )

    rng = random.Random(
        seed
        +
        zero_based_epoch
    )

    rng.shuffle(
        order
    )

    return order


def accumulation_groups(
    *,
    order: Sequence[int],
    accumulation_steps: int,
) -> list[list[int]]:
    if accumulation_steps <= 0:
        raise ValueError(
            "accumulation_steps must be positive."
        )

    if (
        len(
            order
        )
        %
        accumulation_steps
        !=
        0
    ):
        raise ValueError(
            "Order does not end on an exact "
            "accumulation boundary."
        )

    return [
        list(
            order[
                start:
                start
                +
                accumulation_steps
            ]
        )

        for start in range(
            0,
            len(
                order
            ),
            accumulation_steps,
        )
    ]


def supervised_token_loss_scales(
    token_counts: Sequence[int],
) -> list[float]:
    if not token_counts:
        raise ValueError(
            "token_counts must not be empty."
        )

    if any(
        count <= 0
        for count in token_counts
    ):
        raise ValueError(
            "Every supervised token count "
            "must be positive."
        )

    total = sum(
        token_counts
    )

    scales = [
        count
        /
        total
        for count in token_counts
    ]

    if not math.isclose(
        sum(
            scales
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Loss scales do not sum to 1."
        )

    return scales


def weighted_group_loss(
    *,
    losses: Sequence[float],
    token_counts: Sequence[int],
) -> float:
    if (
        len(
            losses
        )
        !=
        len(
            token_counts
        )
    ):
        raise ValueError(
            "losses/token_counts length mismatch."
        )

    scales = supervised_token_loss_scales(
        token_counts
    )

    return sum(
        loss
        *
        scale

        for loss, scale
        in zip(
            losses,
            scales,
        )
    )


def record_id(
    record: dict[str, Any],
    index: int,
) -> str:
    for key in (
        "example_id",
        "id",
        "training_example_id",
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
            return value.strip()

    return (
        f"record:{index:04d}"
    )


def order_sha256(
    identifiers: Sequence[str],
) -> str:
    payload = json.dumps(
        list(
            identifiers
        ),
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
    ).encode(
        "utf-8"
    )

    return sha256_bytes(
        payload
    )


def validate_file_binding(
    *,
    path: Path,
    expected_sha256: str,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} missing: {path}"
        )

    actual = sha256_file(
        path
    )

    if (
        actual
        !=
        expected_sha256
    ):
        raise RuntimeError(
            (
                f"{label} SHA256 mismatch. "
                f"Expected={expected_sha256}, "
                f"Observed={actual}"
            )
        )


def validate_static_contract(
) -> dict[str, Any]:
    root = api_root()

    manifest_file = manifest_path(
        root
    )

    freeze_file = manifest_freeze_path(
        root
    )

    validate_file_binding(
        path=manifest_file,
        expected_sha256=
            EXPECTED_MANIFEST_SHA256,
        label=
            "Training manifest",
    )

    validate_file_binding(
        path=freeze_file,
        expected_sha256=
            EXPECTED_MANIFEST_FREEZE_SHA256,
        label=
            "Training manifest freeze",
    )

    manifest = load_json(
        manifest_file
    )

    freeze = load_json(
        freeze_file
    )

    if (
        freeze[
            "manifest_sha256"
        ]
        !=
        EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Manifest freeze binding mismatch."
        )

    if (
        manifest[
            "training_run_id"
        ]
        !=
        TRAINING_RUN_ID
    ):
        raise RuntimeError(
            "Unexpected training run ID."
        )

    if (
        manifest[
            "preflight_source_git_commit"
        ]
        !=
        EXPECTED_PREFLIGHT_SOURCE_COMMIT
    ):
        raise RuntimeError(
            "Unexpected preflight Git baseline."
        )

    if (
        freeze[
            "preflight_source_git_commit"
        ]
        !=
        EXPECTED_PREFLIGHT_SOURCE_COMMIT
    ):
        raise RuntimeError(
            "Freeze preflight Git binding mismatch."
        )

    if (
        freeze[
            "frozen_before_training"
        ]
        is not True
    ):
        raise RuntimeError(
            "Manifest was not frozen before training."
        )

    if (
        freeze[
            "training_started_at_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "Manifest freeze indicates "
            "training already started."
        )

    if (
        freeze[
            "optimizer_step_executed_at_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "Manifest freeze indicates "
            "optimizer.step()."
        )

    if (
        freeze[
            "optimizer_step_authorized_at_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "Manifest freeze prematurely "
            "authorized optimizer.step()."
        )

    training = manifest[
        "training"
    ]

    expected_training = {
        "max_sequence_length":
            256,

        "micro_batch_size":
            1,

        "gradient_accumulation_steps":
            8,

        "effective_batch_size":
            8,

        "epochs":
            2,

        "micro_batches_per_epoch":
            40,

        "optimizer_steps_per_epoch":
            5,

        "total_micro_batches":
            EXPECTED_TOTAL_MICRO_BATCHES,

        "total_optimizer_steps":
            EXPECTED_TOTAL_OPTIMIZER_STEPS,

        "learning_rate":
            0.0002,

        "optimizer":
            "paged_adamw_8bit",

        "weight_decay":
            0.0,

        "scheduler":
            "cosine",

        "warmup_ratio":
            0.03,

        "warmup_steps":
            1,

        "seed":
            42,

        "bf16":
            True,

        "gradient_checkpointing":
            True,

        "gradient_checkpoint_use_reentrant":
            False,

        "shuffle":
            True,

        "drop_last":
            False,

        "dataloader_num_workers":
            0,

        "gradient_accumulation_weighting":
            "supervised_token_weighted",

        "micro_batch_model_loss_reduction":
            "mean_over_supervised_assistant_tokens",

        "scheduler_step_order":
            "optimizer_then_scheduler",
    }

    for key, expected in expected_training.items():
        observed = training.get(
            key
        )

        if observed != expected:
            raise RuntimeError(
                (
                    "Training manifest field mismatch. "
                    f"{key}: "
                    f"expected={expected!r}, "
                    f"observed={observed!r}"
                )
            )

    if (
        manifest[
            "dataset"
        ][
            "example_count"
        ]
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Unexpected training example count."
        )

    if (
        manifest[
            "dataset"
        ][
            "dataset_sha256"
        ]
        !=
        EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "Dataset manifest binding mismatch."
        )

    if (
        manifest[
            "supervision"
        ][
            "prompt_loss"
        ]
        !=
        "masked"
    ):
        raise RuntimeError(
            "Prompt-loss policy changed."
        )

    if (
        manifest[
            "supervision"
        ][
            "assistant_loss"
        ]
        !=
        "supervised"
    ):
        raise RuntimeError(
            "Assistant-loss policy changed."
        )

    if (
        manifest[
            "supervision"
        ][
            "padding_loss"
        ]
        !=
        "masked"
    ):
        raise RuntimeError(
            "Padding-loss policy changed."
        )

    if (
        manifest[
            "supervision"
        ][
            "packing"
        ]
        is not False
    ):
        raise RuntimeError(
            "Packing must remain disabled."
        )

    if (
        manifest[
            "supervision"
        ][
            "silent_truncation"
        ]
        is not False
    ):
        raise RuntimeError(
            "Silent truncation must remain forbidden."
        )

    if (
        manifest[
            "evaluation_policy"
        ][
            "evaluation_during_training"
        ]
        is not False
    ):
        raise RuntimeError(
            "Evaluation during training is forbidden."
        )

    if (
        manifest[
            "evaluation_policy"
        ][
            "training_loss_is_acceptance_evidence"
        ]
        is not False
    ):
        raise RuntimeError(
            "Training loss cannot become "
            "acceptance evidence."
        )

    if (
        manifest[
            "final_acceptance"
        ][
            "cases_may_be_loaded_during_training"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance access policy changed."
        )

    if (
        manifest[
            "authorization"
        ][
            "optimizer_step_authorized_at_manifest_creation"
        ]
        is not False
    ):
        raise RuntimeError(
            "Manifest prematurely authorized training."
        )

    if (
        manifest[
            "authorization"
        ][
            "final_acceptance_access_authorized"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance authorization changed."
        )

    contract_path = resolve_api_relative_path(
        manifest[
            "experiment"
        ][
            "contract_relative_path"
        ]
    )

    validate_file_binding(
        path=contract_path,
        expected_sha256=
            EXPECTED_CONTRACT_SHA256,
        label=
            "QLoRA v0.3 contract",
    )

    dataset_path = resolve_api_relative_path(
        manifest[
            "dataset"
        ][
            "relative_path"
        ]
    )

    validate_file_binding(
        path=dataset_path,
        expected_sha256=
            EXPECTED_DATASET_SHA256,
        label=
            "Training dataset",
    )

    sequence_probe = (
        manifest[
            "preflight_evidence"
        ][
            "sequence_memory_probe"
        ]
    )

    optimizer_probe = (
        manifest[
            "preflight_evidence"
        ][
            "optimizer_state_memory_probe"
        ]
    )

    assistant_probe = (
        manifest[
            "preflight_evidence"
        ][
            "assistant_only_gpu_probe"
        ]
    )

    if (
        sequence_probe[
            "sha256"
        ]
        !=
        EXPECTED_MEMORY_PROBE_SHA256
    ):
        raise RuntimeError(
            "Sequence-memory probe manifest "
            "binding changed."
        )

    if (
        optimizer_probe[
            "sha256"
        ]
        !=
        EXPECTED_OPTIMIZER_PROBE_SHA256
    ):
        raise RuntimeError(
            "Optimizer-memory probe manifest "
            "binding changed."
        )

    if (
        assistant_probe[
            "sha256"
        ]
        !=
        EXPECTED_ASSISTANT_PROBE_SHA256
    ):
        raise RuntimeError(
            "Assistant-only probe manifest "
            "binding changed."
        )

    validate_file_binding(
        path=resolve_api_relative_path(
            sequence_probe[
                "relative_path"
            ]
        ),
        expected_sha256=
            EXPECTED_MEMORY_PROBE_SHA256,
        label=
            "Sequence-memory preflight",
    )

    validate_file_binding(
        path=resolve_api_relative_path(
            optimizer_probe[
                "relative_path"
            ]
        ),
        expected_sha256=
            EXPECTED_OPTIMIZER_PROBE_SHA256,
        label=
            "Optimizer-state preflight",
    )

    validate_file_binding(
        path=resolve_api_relative_path(
            assistant_probe[
                "relative_path"
            ]
        ),
        expected_sha256=
            EXPECTED_ASSISTANT_PROBE_SHA256,
        label=
            "Assistant-only GPU preflight",
    )

    final_acceptance_path = (
        resolve_api_relative_path(
            manifest[
                "final_acceptance"
            ][
                "freeze_relative_path"
            ]
        )
    )

    validate_file_binding(
        path=final_acceptance_path,
        expected_sha256=
            EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256,
        label=
            "Final Acceptance freeze",
    )

    masking_source_path = (
        resolve_api_relative_path(
            manifest[
                "supervision"
            ][
                "masking_source_relative_path"
            ]
        )
    )

    masking_test_path = (
        resolve_api_relative_path(
            manifest[
                "supervision"
            ][
                "masking_test_relative_path"
            ]
        )
    )

    current_head = run_git_text(
        [
            "rev-parse",
            "HEAD",
        ]
    )

    if (
        git_blob_sha256(
            ref=current_head,
            path=masking_source_path,
        )
        !=
        manifest[
            "supervision"
        ][
            "masking_source_git_blob_sha256"
        ]
    ):
        raise RuntimeError(
            "Committed assistant masking "
            "source changed."
        )

    if (
        git_blob_sha256(
            ref=current_head,
            path=masking_test_path,
        )
        !=
        manifest[
            "supervision"
        ][
            "masking_test_git_blob_sha256"
        ]
    ):
        raise RuntimeError(
            "Committed assistant masking "
            "test changed."
        )

    if not git_path_is_tracked(
        manifest_file
    ):
        raise RuntimeError(
            "Training manifest is not committed."
        )

    manifest_commit = (
        git_latest_commit_for_path(
            manifest_file
        )
    )

    if not git_is_ancestor(
        EXPECTED_PREFLIGHT_SOURCE_COMMIT,
        current_head,
    ):
        raise RuntimeError(
            "Preflight baseline is not an "
            "ancestor of current HEAD."
        )

    if not git_is_ancestor(
        manifest_commit,
        current_head,
    ):
        raise RuntimeError(
            "Manifest commit is not an "
            "ancestor of current HEAD."
        )

    return {
        "manifest":
            manifest,

        "freeze":
            freeze,

        "manifest_commit":
            manifest_commit,

        "current_head":
            current_head,

        "manifest_sha256":
            EXPECTED_MANIFEST_SHA256,

        "manifest_freeze_sha256":
            EXPECTED_MANIFEST_FREEZE_SHA256,
    }


def planned_output_paths(
    manifest: dict[str, Any],
) -> dict[str, Path]:
    adapter_identity = (
        manifest[
            "planned_outputs"
        ][
            "adapter_storage_relative_identity"
        ]
    )

    adapter_path = (
        Path.home()
        / ".cache"
        / adapter_identity
    )

    report_path = resolve_api_relative_path(
        manifest[
            "planned_outputs"
        ][
            "repository_training_report_relative_path"
        ]
    )

    receipt_path = resolve_api_relative_path(
        manifest[
            "planned_outputs"
        ][
            "repository_training_receipt_relative_path"
        ]
    )

    return {
        "adapter":
            adapter_path,

        "report":
            report_path,

        "receipt":
            receipt_path,
    }


def collect_runtime_authorization(
) -> dict[str, Any]:
    static = validate_static_contract()

    manifest = static[
        "manifest"
    ]

    binding = manifest[
        "git_execution_binding"
    ]

    head = run_git_text(
        [
            "rev-parse",
            "HEAD",
        ]
    )

    runner_path = (
        api_root()
        /
        binding[
            "training_runner_relative_path"
        ]
    )

    runner_test_path = (
        api_root()
        /
        binding[
            "training_runner_test_relative_path"
        ]
    )

    manifest_file = manifest_path()

    manifest_commit = (
        git_latest_commit_for_path(
            manifest_file
        )
    )

    runner_exists = (
        runner_path.is_file()
    )

    runner_test_exists = (
        runner_test_path.is_file()
    )

    runner_tracked = (
        runner_exists
        and
        git_path_is_tracked(
            runner_path
        )
    )

    runner_test_tracked = (
        runner_test_exists
        and
        git_path_is_tracked(
            runner_test_path
        )
    )

    runner_commit = (
        git_latest_commit_for_path(
            runner_path
        )
        if runner_tracked
        else None
    )

    runner_test_commit = (
        git_latest_commit_for_path(
            runner_test_path
        )
        if runner_test_tracked
        else None
    )

    output_paths = planned_output_paths(
        manifest
    )

    gates = {
        "working_tree_clean":
            git_is_clean(),

        "preflight_source_is_ancestor":
            git_is_ancestor(
                EXPECTED_PREFLIGHT_SOURCE_COMMIT,
                head,
            ),

        "manifest_is_committed":
            git_path_is_tracked(
                manifest_file
            ),

        "manifest_commit_is_ancestor":
            git_is_ancestor(
                manifest_commit,
                head,
            ),

        "runner_exists":
            runner_exists,

        "runner_is_committed":
            runner_tracked,

        "runner_test_exists":
            runner_test_exists,

        "runner_test_is_committed":
            runner_test_tracked,

        "runner_commit_equals_training_start_head":
            (
                runner_commit
                ==
                head
            ),

        "runner_test_commit_equals_training_start_head":
            (
                runner_test_commit
                ==
                head
            ),

        "adapter_output_absent":
            not output_paths[
                "adapter"
            ].exists(),

        "report_output_absent":
            not output_paths[
                "report"
            ].exists(),

        "receipt_output_absent":
            not output_paths[
                "receipt"
            ].exists(),
    }

    authorized = all(
        gates.values()
    )

    runner_blob_sha256 = (
        git_blob_sha256(
            ref=head,
            path=runner_path,
        )
        if (
            runner_tracked
            and
            runner_commit
            ==
            head
        )
        else None
    )

    runner_test_blob_sha256 = (
        git_blob_sha256(
            ref=head,
            path=runner_test_path,
        )
        if (
            runner_test_tracked
            and
            runner_test_commit
            ==
            head
        )
        else None
    )

    return {
        "authorized":
            authorized,

        "gates":
            gates,

        "training_start_head":
            head,

        "manifest_commit":
            manifest_commit,

        "runner_commit":
            runner_commit,

        "runner_test_commit":
            runner_test_commit,

        "runner_blob_sha256":
            runner_blob_sha256,

        "runner_test_blob_sha256":
            runner_test_blob_sha256,

        "output_paths":
            output_paths,

        "static":
            static,
    }


def require_runtime_authorization(
) -> dict[str, Any]:
    authorization = (
        collect_runtime_authorization()
    )

    if authorization[
        "authorized"
    ]:
        return authorization

    failed = [
        name
        for name, value
        in authorization[
            "gates"
        ].items()
        if not value
    ]

    raise RuntimeError(
        (
            "Training runtime authorization "
            "FAILED. optimizer.step() remains "
            "forbidden.\nFailed gates:\n  - "
            +
            "\n  - ".join(
                failed
            )
        )
    )


def extract_target_modules(
    resolution: Any,
) -> list[str]:
    payload = (
        resolution.model_dump()
        if hasattr(
            resolution,
            "model_dump",
        )
        else vars(
            resolution
        )
    )

    preferred_keys = (
        "target_modules",
        "resolved_target_modules",
        "module_names",
        "targets",
    )

    for key in preferred_keys:
        value = payload.get(
            key
        )

        if (
            isinstance(
                value,
                (
                    list,
                    tuple,
                ),
            )
            and
            value
            and
            all(
                isinstance(
                    item,
                    str,
                )
                for item in value
            )
        ):
            return list(
                value
            )

    candidates = []

    for value in payload.values():
        if (
            isinstance(
                value,
                (
                    list,
                    tuple,
                ),
            )
            and
            len(
                value
            )
            ==
            EXPECTED_TARGET_COUNT
            and
            all(
                isinstance(
                    item,
                    str,
                )
                for item in value
            )
        ):
            candidates.append(
                list(
                    value
                )
            )

    if len(
        candidates
    ) != 1:
        raise RuntimeError(
            "Unable to identify resolved "
            "LoRA target modules safely."
        )

    return candidates[
        0
    ]


def trainable_parameter_fingerprint(
    *,
    model: Any,
    torch_module: Any,
) -> str:
    digest = hashlib.sha256()

    count = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        count += 1

        tensor = (
            parameter
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                parameter.dtype
            ).encode(
                "ascii"
            )
        )

        digest.update(
            str(
                tuple(
                    parameter.shape
                )
            ).encode(
                "ascii"
            )
        )

        digest.update(
            tensor
            .view(
                torch_module.uint8
            )
            .numpy()
            .tobytes()
        )

    if (
        count
        !=
        EXPECTED_TRAINABLE_TENSORS
    ):
        raise RuntimeError(
            "Unexpected trainable tensor count "
            "during fingerprint."
        )

    return digest.hexdigest()


def adapter_bundle_metadata(
    *,
    adapter_path: Path,
    safe_open_function: Any,
) -> dict[str, Any]:
    if not adapter_path.is_dir():
        raise RuntimeError(
            "Temporary adapter directory missing."
        )

    files = sorted(
        path
        for path in adapter_path.rglob(
            "*"
        )
        if path.is_file()
    )

    if not files:
        raise RuntimeError(
            "Adapter output contains no files."
        )

    forbidden_names = {
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    }

    if any(
        path.name
        in
        forbidden_names
        for path in files
    ):
        raise RuntimeError(
            "Base-model weight file detected "
            "inside adapter output."
        )

    adapter_weight_path = (
        adapter_path
        /
        "adapter_model.safetensors"
    )

    adapter_config_path = (
        adapter_path
        /
        "adapter_config.json"
    )

    if not adapter_weight_path.is_file():
        raise RuntimeError(
            "adapter_model.safetensors missing."
        )

    if not adapter_config_path.is_file():
        raise RuntimeError(
            "adapter_config.json missing."
        )

    with safe_open_function(
        str(
            adapter_weight_path
        ),
        framework="pt",
        device="cpu",
    ) as handle:
        tensor_keys = sorted(
            handle.keys()
        )

    if (
        len(
            tensor_keys
        )
        !=
        EXPECTED_TRAINABLE_TENSORS
    ):
        raise RuntimeError(
            (
                "Unexpected adapter tensor count. "
                f"Observed={len(tensor_keys)}"
            )
        )

    if any(
        "lora_"
        not in
        key.lower()
        for key in tensor_keys
    ):
        raise RuntimeError(
            "Non-LoRA tensor detected "
            "inside saved adapter."
        )

    file_records = []

    bundle_digest = hashlib.sha256()

    total_bytes = 0

    for path in files:
        relative = (
            path
            .relative_to(
                adapter_path
            )
            .as_posix()
        )

        size = path.stat().st_size

        sha256 = sha256_file(
            path
        )

        total_bytes += size

        file_records.append(
            {
                "relative_path":
                    relative,

                "size_bytes":
                    size,

                "sha256":
                    sha256,
            }
        )

        bundle_digest.update(
            relative.encode(
                "utf-8"
            )
        )

        bundle_digest.update(
            b"\0"
        )

        bundle_digest.update(
            sha256.encode(
                "ascii"
            )
        )

        bundle_digest.update(
            b"\n"
        )

    return {
        "file_count":
            len(
                file_records
            ),

        "total_bytes":
            total_bytes,

        "adapter_tensor_count":
            len(
                tensor_keys
            ),

        "bundle_sha256":
            bundle_digest.hexdigest(),

        "files":
            file_records,
    }


def execute_training() -> None:
    authorization = (
        require_runtime_authorization()
    )

    manifest = authorization[
        "static"
    ][
        "manifest"
    ]

    print(
        "=== DATALENS OFFICIAL QLORA TRAINING v0.1 ==="
    )

    print()

    print(
        "RUNTIME AUTHORIZATION: PASS"
    )

    print(
        (
            "Training start HEAD: "
            f"{authorization['training_start_head']}"
        )
    )

    print(
        (
            "Runner Git blob SHA256: "
            f"{authorization['runner_blob_sha256']}"
        )
    )

    print(
        "Final Acceptance access: FORBIDDEN"
    )

    print()

    # Heavy dependencies are deliberately imported
    # only after runtime authorization succeeds.
    import bitsandbytes as bnb
    import numpy as np
    import peft
    import torch
    import transformers

    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    from safetensors import (
        safe_open,
    )

    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        Gemma3ForCausalLM,
        get_cosine_schedule_with_warmup,
    )

    from app.adaptation import (
        resolve_qlora_target_modules,
    )

    from app.adaptation.assistant_masking import (
        ASSISTANT_ONLY_MASKING_RULE_VERSION,
        build_assistant_only_training_example,
        collate_assistant_only_examples,
    )

    from app.adaptation.contracts import (
        QLoRAExperimentContract,
    )

    # ========================================================
    # ENVIRONMENT GATE
    # ========================================================

    versions = {
        "torch":
            torch.__version__,

        "transformers":
            transformers.__version__,

        "peft":
            peft.__version__,

        "bitsandbytes":
            bnb.__version__,
    }

    expected_versions = {
        "torch":
            EXPECTED_TORCH_VERSION,

        "transformers":
            EXPECTED_TRANSFORMERS_VERSION,

        "peft":
            EXPECTED_PEFT_VERSION,

        "bitsandbytes":
            EXPECTED_BITSANDBYTES_VERSION,
    }

    for key, expected in expected_versions.items():
        if (
            versions[
                key
            ]
            !=
            expected
        ):
            raise RuntimeError(
                (
                    "Adaptation environment version "
                    f"mismatch for {key}. "
                    f"Expected={expected}, "
                    f"Observed={versions[key]}"
                )
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda:0"
    )

    properties = (
        torch.cuda.get_device_properties(
            device
        )
    )

    free_before_load, total_vram = (
        torch.cuda.mem_get_info(
            device
        )
    )

    if (
        free_before_load
        <
        int(
            6.5
            *
            1024 ** 3
        )
    ):
        raise RuntimeError(
            (
                "Insufficient free VRAM before "
                "official training. "
                f"Free={free_before_load / 1024**3:.2f} GiB"
            )
        )

    print(
        "Adaptation environment: PASS"
    )

    print(
        (
            "CUDA device: "
            f"{properties.name}"
        )
    )

    print(
        (
            "Free VRAM before load: "
            f"{free_before_load / 1024**3:.2f} GiB"
        )
    )

    # ========================================================
    # BASE CHECKPOINT
    # ========================================================

    base_model_path = (
        Path.home()
        / ".cache"
        / "datalens"
        / "adaptation"
        / "base-models"
        / (
            "gemma-3-4b-it-text-"
            "093f9f388b31de276ce2de164bdc2081324b9767"
        )
    )

    if not base_model_path.is_dir():
        raise FileNotFoundError(
            "Converted Gemma text checkpoint missing."
        )

    validate_file_binding(
        path=
            base_model_path
            /
            "config.json",
        expected_sha256=
            EXPECTED_BASE_CONFIG_SHA256,
        label=
            "Converted Gemma config",
    )

    validate_file_binding(
        path=
            base_model_path
            /
            "model.safetensors.index.json",
        expected_sha256=
            EXPECTED_BASE_INDEX_SHA256,
        label=
            "Converted Gemma index",
    )

    shard_hashes = {
        sha256_file(
            path
        )

        for path in base_model_path.glob(
            "*.safetensors"
        )
    }

    if (
        shard_hashes
        !=
        EXPECTED_BASE_SHARD_SHA256
    ):
        raise RuntimeError(
            "Converted Gemma shard binding mismatch."
        )

    # ========================================================
    # EXPERIMENT CONTRACT
    # ========================================================

    contract_path = (
        resolve_api_relative_path(
            manifest[
                "experiment"
            ][
                "contract_relative_path"
            ]
        )
    )

    contract = (
        QLoRAExperimentContract.model_validate(
            load_json(
                contract_path
            )
        )
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            contract.base_model.repository,
            revision=
                contract.base_model.tokenizer_revision,
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
            "Pinned tokenizer chat template missing."
        )

    if (
        sha256_bytes(
            tokenizer.chat_template.encode(
                "utf-8"
            )
        )
        !=
        EXPECTED_CHAT_TEMPLATE_SHA256
    ):
        raise RuntimeError(
            "Pinned chat-template SHA256 mismatch."
        )

    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Tokenizer has no pad_token_id."
        )

    if (
        ASSISTANT_ONLY_MASKING_RULE_VERSION
        !=
        manifest[
            "supervision"
        ][
            "rule_version"
        ]
    ):
        raise RuntimeError(
            "Assistant masking runtime "
            "version mismatch."
        )

    # ========================================================
    # BUILD EXACT TRAINING EXAMPLES
    # ========================================================

    dataset_path = (
        resolve_api_relative_path(
            manifest[
                "dataset"
            ][
                "relative_path"
            ]
        )
    )

    records = []

    with dataset_path.open(
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

    identifiers = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        example = (
            build_assistant_only_training_example(
                tokenizer=
                    tokenizer,
                messages=
                    record[
                        "messages"
                    ],
                max_sequence_length=
                    manifest[
                        "training"
                    ][
                        "max_sequence_length"
                    ],
            )
        )

        examples.append(
            example
        )

        identifiers.append(
            record_id(
                record,
                index,
            )
        )

    total_full_tokens = sum(
        example.total_token_count
        for example in examples
    )

    total_supervised_tokens = sum(
        example.supervised_token_count
        for example in examples
    )

    max_example_tokens = max(
        example.total_token_count
        for example in examples
    )

    if (
        total_full_tokens
        !=
        EXPECTED_TOTAL_TOKENS
    ):
        raise RuntimeError(
            "Training full-token total changed."
        )

    if (
        total_supervised_tokens
        !=
        EXPECTED_SUPERVISED_TOKENS
    ):
        raise RuntimeError(
            "Training supervised-token total changed."
        )

    if (
        max_example_tokens
        !=
        EXPECTED_MAX_EXAMPLE_TOKENS
    ):
        raise RuntimeError(
            "Training max example length changed."
        )

    # ========================================================
    # DETERMINISTIC EPOCH ORDERS
    # ========================================================

    seed = manifest[
        "training"
    ][
        "seed"
    ]

    epoch_orders = []

    epoch_order_hashes = []

    epoch_count = manifest[
        "training"
    ][
        "epochs"
    ]

    for epoch in range(
        epoch_count
    ):
        order = deterministic_epoch_order(
            example_count=
                EXPECTED_EXAMPLE_COUNT,
            seed=
                seed,
            zero_based_epoch=
                epoch,
        )

        if (
            sorted(
                order
            )
            !=
            list(
                range(
                    EXPECTED_EXAMPLE_COUNT
                )
            )
        ):
            raise RuntimeError(
                "Epoch order is not a full permutation."
            )

        epoch_orders.append(
            order
        )

        epoch_order_hashes.append(
            order_sha256(
                [
                    identifiers[
                        index
                    ]
                    for index in order
                ]
            )
        )

    # ========================================================
    # RNG
    # ========================================================

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    # ========================================================
    # NF4 MODEL LOAD
    # ========================================================

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=
                contract.quantization.load_in_4bit,
            bnb_4bit_quant_type=
                contract.quantization.quantization_type,
            bnb_4bit_use_double_quant=
                contract.quantization.use_double_quantization,
            bnb_4bit_compute_dtype=
                torch.bfloat16,
        )
    )

    print()

    print(
        "Loading pinned Gemma text checkpoint in NF4..."
    )

    model = (
        Gemma3ForCausalLM.from_pretrained(
            base_model_path,
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

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=
            True,
        gradient_checkpointing_kwargs={
            "use_reentrant":
                False,
        },
    )

    model.config.use_cache = False

    # ========================================================
    # SERVER-OWNED LORA TARGET RESOLUTION
    # ========================================================

    resolution = resolve_qlora_target_modules(
        model=model,
        base_model=
            contract.base_model,
        lora=
            contract.lora,
    )

    target_modules = (
        extract_target_modules(
            resolution
        )
    )

    if (
        len(
            target_modules
        )
        !=
        EXPECTED_TARGET_COUNT
    ):
        raise RuntimeError(
            "Unexpected LoRA target count."
        )

    for target in target_modules:
        lowered = target.lower()

        if any(
            fragment in lowered
            for fragment in (
                "vision",
                "projector",
                "lm_head",
            )
        ):
            raise RuntimeError(
                "Forbidden LoRA target: "
                f"{target}"
            )

    lora_config = LoraConfig(
        r=
            contract.lora.rank,
        lora_alpha=
            contract.lora.alpha,
        lora_dropout=
            contract.lora.dropout,
        bias=
            contract.lora.bias,
        task_type=
            contract.lora.task_type,
        target_modules=
            target_modules,
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    trainable_parameters = [
        parameter

        for parameter
        in model.parameters()

        if parameter.requires_grad
    ]

    trainable_names = [
        name

        for name, parameter
        in model.named_parameters()

        if parameter.requires_grad
    ]

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    if (
        trainable_parameter_count
        !=
        EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError(
            "Unexpected trainable parameter count."
        )

    if (
        len(
            trainable_names
        )
        !=
        EXPECTED_TRAINABLE_TENSORS
    ):
        raise RuntimeError(
            "Unexpected trainable tensor count."
        )

    if any(
        "lora_"
        not in
        name.lower()
        for name in trainable_names
    ):
        raise RuntimeError(
            "Non-LoRA trainable parameter detected."
        )

    model.train()

    # ========================================================
    # OPTIMIZER + SCHEDULER
    # ========================================================

    optimizer = bnb.optim.PagedAdamW8bit(
        trainable_parameters,
        lr=
            manifest[
                "training"
            ][
                "learning_rate"
            ],
        betas=tuple(
            manifest[
                "training"
            ][
                "optimizer_betas"
            ]
        ),
        eps=
            manifest[
                "training"
            ][
                "optimizer_eps"
            ],
        weight_decay=
            manifest[
                "training"
            ][
                "weight_decay"
            ],
        amsgrad=False,
        min_8bit_size=4096,
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

    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=
                manifest[
                    "training"
                ][
                    "warmup_steps"
                ],
            num_training_steps=
                EXPECTED_TOTAL_OPTIMIZER_STEPS,
        )
    )

    # ========================================================
    # TRAINING START
    # ========================================================

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    training_started_at = utc_now()

    started_monotonic = time.monotonic()

    fingerprint_before = (
        trainable_parameter_fingerprint(
            model=model,
            torch_module=torch,
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    global_micro_batch = 0

    global_optimizer_step = 0

    step_records = []

    supervised_tokens_seen = 0

    optimizer_step_executed = False

    print()

    print(
        "OFFICIAL TRAINING START"
    )

    print(
        (
            "  Epochs: "
            f"{epoch_count}"
        )
    )

    print(
        (
            "  Planned micro-batches: "
            f"{EXPECTED_TOTAL_MICRO_BATCHES}"
        )
    )

    print(
        (
            "  Planned optimizer steps: "
            f"{EXPECTED_TOTAL_OPTIMIZER_STEPS}"
        )
    )

    print(
        "  Final Acceptance access: False"
    )

    # ========================================================
    # EXACT TRAINING LOOP
    # ========================================================

    for epoch in range(
        epoch_count
    ):
        order = epoch_orders[
            epoch
        ]

        groups = accumulation_groups(
            order=order,
            accumulation_steps=
                manifest[
                    "training"
                ][
                    "gradient_accumulation_steps"
                ],
        )

        if len(groups) != 5:
            raise RuntimeError(
                "Expected 5 accumulation groups "
                "per epoch."
            )

        print()

        print(
            (
                f"EPOCH {epoch + 1}/"
                f"{epoch_count}"
            )
        )

        for group_index, group in enumerate(
            groups,
            start=1,
        ):
            token_counts = [
                examples[
                    index
                ].supervised_token_count

                for index in group
            ]

            group_supervised_tokens = sum(
                token_counts
            )

            scales = supervised_token_loss_scales(
                token_counts
            )

            micro_losses = []

            optimizer.zero_grad(
                set_to_none=True
            )

            for (
                example_index,
                loss_scale,
            ) in zip(
                group,
                scales,
            ):
                example = examples[
                    example_index
                ]

                batch = (
                    collate_assistant_only_examples(
                        examples=[
                            example
                        ],
                        pad_token_id=
                            tokenizer.pad_token_id,
                    )
                )

                input_ids = torch.tensor(
                    batch.input_ids,
                    dtype=torch.long,
                    device=device,
                )

                attention_mask = torch.tensor(
                    batch.attention_mask,
                    dtype=torch.long,
                    device=device,
                )

                labels = torch.tensor(
                    batch.labels,
                    dtype=torch.long,
                    device=device,
                )

                outputs = model(
                    input_ids=
                        input_ids,
                    attention_mask=
                        attention_mask,
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
                        "Non-finite training loss."
                    )

                loss_value = float(
                    loss.detach().cpu()
                )

                micro_losses.append(
                    loss_value
                )

                (
                    loss
                    *
                    loss_scale
                ).backward()

                global_micro_batch += 1

                supervised_tokens_seen += (
                    example.supervised_token_count
                )

                del outputs
                del loss
                del input_ids
                del attention_mask
                del labels

            # -----------------------------------------------
            # GRADIENT VALIDATION BEFORE OFFICIAL UPDATE
            # -----------------------------------------------

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

                if not torch.isfinite(
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

            if (
                gradient_tensor_count
                !=
                EXPECTED_TRAINABLE_TENSORS
            ):
                raise RuntimeError(
                    (
                        "Unexpected gradient tensor "
                        f"count before step. "
                        f"Observed={gradient_tensor_count}"
                    )
                )

            if nonfinite_gradient_count:
                raise RuntimeError(
                    "Non-finite LoRA gradients detected."
                )

            gradient_norm = math.sqrt(
                gradient_square_sum
            )

            if (
                not math.isfinite(
                    gradient_norm
                )
                or
                gradient_norm
                <=
                0.0
            ):
                raise RuntimeError(
                    "Invalid LoRA gradient norm."
                )

            group_loss = weighted_group_loss(
                losses=
                    micro_losses,
                token_counts=
                    token_counts,
            )

            lr_before_step = float(
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            )

            # ===============================================
            # OFFICIAL MODEL UPDATE
            # ===============================================

            optimizer.step()

            optimizer_step_executed = True

            global_optimizer_step += 1

            scheduler.step()

            lr_after_scheduler = float(
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            torch.cuda.synchronize()

            step_record = {
                "epoch":
                    epoch
                    +
                    1,

                "optimizer_step_in_epoch":
                    group_index,

                "global_optimizer_step":
                    global_optimizer_step,

                "global_micro_batch":
                    global_micro_batch,

                "group_example_count":
                    len(
                        group
                    ),

                "group_supervised_tokens":
                    group_supervised_tokens,

                "group_weighted_loss":
                    group_loss,

                "gradient_tensor_count":
                    gradient_tensor_count,

                "gradient_norm":
                    gradient_norm,

                "learning_rate_before_optimizer_step":
                    lr_before_step,

                "learning_rate_after_scheduler_step":
                    lr_after_scheduler,

                "cuda_allocated_bytes":
                    torch.cuda.memory_allocated(
                        device
                    ),

                "cuda_reserved_bytes":
                    torch.cuda.memory_reserved(
                        device
                    ),
            }

            step_records.append(
                step_record
            )

            print(
                (
                    "  step "
                    f"{global_optimizer_step:02d}/"
                    f"{EXPECTED_TOTAL_OPTIMIZER_STEPS} "
                    f"loss={group_loss:.6f} "
                    f"tokens={group_supervised_tokens} "
                    f"grad={gradient_norm:.6f} "
                    f"lr={lr_before_step:.8f}"
                )
            )

    # ========================================================
    # POST-TRAINING VALIDATION
    # ========================================================

    torch.cuda.synchronize()

    training_finished_at = utc_now()

    duration_seconds = (
        time.monotonic()
        -
        started_monotonic
    )

    if (
        global_micro_batch
        !=
        EXPECTED_TOTAL_MICRO_BATCHES
    ):
        raise RuntimeError(
            "Training micro-batch count mismatch."
        )

    if (
        global_optimizer_step
        !=
        EXPECTED_TOTAL_OPTIMIZER_STEPS
    ):
        raise RuntimeError(
            "Training optimizer-step count mismatch."
        )

    if (
        supervised_tokens_seen
        !=
        EXPECTED_SUPERVISED_TOKENS
        *
        epoch_count
    ):
        raise RuntimeError(
            "Supervised-token presentation count "
            "mismatch."
        )

    if not optimizer_step_executed:
        raise RuntimeError(
            "No optimizer step was executed."
        )

    if len(
        step_records
    ) != EXPECTED_TOTAL_OPTIMIZER_STEPS:
        raise RuntimeError(
            "Training step record count mismatch."
        )

    fingerprint_after = (
        trainable_parameter_fingerprint(
            model=model,
            torch_module=torch,
        )
    )

    weights_changed = (
        fingerprint_before
        !=
        fingerprint_after
    )

    if not weights_changed:
        raise RuntimeError(
            "LoRA weights did not change "
            "after official training."
        )

    for parameter in trainable_parameters:
        if not torch.isfinite(
            parameter.detach()
        ).all():
            raise RuntimeError(
                "Non-finite trained LoRA parameter."
            )

    peak_allocated = (
        torch.cuda.max_memory_allocated(
            device
        )
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved(
            device
        )
    )

    # ========================================================
    # ADAPTER TEMPORARY SAVE
    # ========================================================

    output_paths = authorization[
        "output_paths"
    ]

    adapter_path = output_paths[
        "adapter"
    ]

    report_path = output_paths[
        "report"
    ]

    receipt_path = output_paths[
        "receipt"
    ]

    adapter_parent = (
        adapter_path.parent
    )

    adapter_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique_suffix = uuid.uuid4().hex

    temp_adapter_path = (
        adapter_parent
        /
        (
            f".{adapter_path.name}."
            f"{unique_suffix}.tmp"
        )
    )

    temp_report_path = (
        report_path.parent
        /
        (
            f".{report_path.name}."
            f"{unique_suffix}.tmp"
        )
    )

    temp_receipt_path = (
        receipt_path.parent
        /
        (
            f".{receipt_path.name}."
            f"{unique_suffix}.tmp"
        )
    )

    for path in (
        temp_adapter_path,
        temp_report_path,
        temp_receipt_path,
    ):
        if path.exists():
            raise RuntimeError(
                "Unexpected temporary-output collision."
            )

    print()

    print(
        "Saving adapter-only artifact to temporary path..."
    )

    model.save_pretrained(
        temp_adapter_path,
        safe_serialization=True,
    )

    adapter_metadata = (
        adapter_bundle_metadata(
            adapter_path=
                temp_adapter_path,
            safe_open_function=
                safe_open,
        )
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "report_id":
            "training-report:datalens-semantic-qlora:v0.1",

        "rule_version":
            "qlora_training_report_v0.1",

        "training_run_id":
            TRAINING_RUN_ID,

        "experiment_id":
            manifest[
                "experiment"
            ][
                "experiment_id"
            ],

        "manifest_sha256":
            EXPECTED_MANIFEST_SHA256,

        "manifest_freeze_sha256":
            EXPECTED_MANIFEST_FREEZE_SHA256,

        "training_start_git_commit":
            authorization[
                "training_start_head"
            ],

        "runner_git_blob_sha256":
            authorization[
                "runner_blob_sha256"
            ],

        "runner_test_git_blob_sha256":
            authorization[
                "runner_test_blob_sha256"
            ],

        "environment":
            versions,

        "device": {
            "name":
                properties.name,

            "compute_capability":
                (
                    f"{properties.major}."
                    f"{properties.minor}"
                ),

            "total_vram_bytes":
                total_vram,

            "free_vram_before_load_bytes":
                free_before_load,

            "peak_allocated_bytes":
                peak_allocated,

            "peak_reserved_bytes":
                peak_reserved,
        },

        "dataset": {
            "example_count":
                EXPECTED_EXAMPLE_COUNT,

            "full_tokens_per_epoch":
                EXPECTED_TOTAL_TOKENS,

            "supervised_tokens_per_epoch":
                EXPECTED_SUPERVISED_TOKENS,

            "supervised_token_presentations":
                supervised_tokens_seen,

            "training_text_recorded":
                False,
        },

        "execution": {
            "started_at":
                training_started_at,

            "finished_at":
                training_finished_at,

            "duration_seconds":
                duration_seconds,

            "epochs":
                epoch_count,

            "micro_batches_executed":
                global_micro_batch,

            "optimizer_steps_executed":
                global_optimizer_step,

            "gradient_accumulation_steps":
                manifest[
                    "training"
                ][
                    "gradient_accumulation_steps"
                ],

            "gradient_accumulation_weighting":
                "supervised_token_weighted",

            "gradient_checkpoint_use_reentrant":
                False,

            "assistant_only_loss":
                True,

            "training_executed":
                True,
        },

        "epoch_order_sha256": [
            {
                "epoch":
                    index
                    +
                    1,

                "sha256":
                    value,
            }

            for index, value
            in enumerate(
                epoch_order_hashes
            )
        ],

        "optimizer_steps":
            step_records,

        "model_change": {
            "trainable_tensor_count":
                EXPECTED_TRAINABLE_TENSORS,

            "trainable_parameter_count":
                EXPECTED_TRAINABLE_PARAMETERS,

            "weights_changed":
                weights_changed,
        },

        "adapter": {
            "storage_scope":
                "local_cache",

            "published_path":
                str(
                    adapter_path
                ),

            **adapter_metadata,
        },

        "evaluation": {
            "evaluation_during_training":
                False,

            "training_loss_is_acceptance_evidence":
                False,

            "pre_adaptation_holdouts_executed":
                False,

            "final_acceptance_cases_loaded":
                False,

            "final_acceptance_evaluated":
                False,
        },

        "passed":
            True,
    }

    report_bytes = canonical_json_bytes(
        report
    )

    report_sha256 = sha256_bytes(
        report_bytes
    )

    # ========================================================
    # RECEIPT
    # ========================================================

    receipt = {
        "receipt_id":
            "training-receipt:datalens-semantic-qlora:v0.1",

        "rule_version":
            "qlora_training_receipt_v0.1",

        "training_run_id":
            TRAINING_RUN_ID,

        "manifest_sha256":
            EXPECTED_MANIFEST_SHA256,

        "manifest_freeze_sha256":
            EXPECTED_MANIFEST_FREEZE_SHA256,

        "preflight_source_git_commit":
            EXPECTED_PREFLIGHT_SOURCE_COMMIT,

        "manifest_commit":
            authorization[
                "manifest_commit"
            ],

        "training_start_git_commit":
            authorization[
                "training_start_head"
            ],

        "training_runner_commit":
            authorization[
                "runner_commit"
            ],

        "training_runner_git_blob_sha256":
            authorization[
                "runner_blob_sha256"
            ],

        "training_runner_test_commit":
            authorization[
                "runner_test_commit"
            ],

        "training_runner_test_git_blob_sha256":
            authorization[
                "runner_test_blob_sha256"
            ],

        "report_relative_path":
            report_path
            .relative_to(
                api_root()
            )
            .as_posix(),

        "report_sha256":
            report_sha256,

        "adapter_path":
            str(
                adapter_path
            ),

        "adapter_bundle_sha256":
            adapter_metadata[
                "bundle_sha256"
            ],

        "adapter_tensor_count":
            adapter_metadata[
                "adapter_tensor_count"
            ],

        "optimizer_steps_executed":
            global_optimizer_step,

        "model_weights_modified":
            True,

        "base_model_weights_saved":
            False,

        "merged_model_saved":
            False,

        "training_executed":
            True,

        "final_acceptance_cases_loaded":
            False,

        "final_acceptance_evaluated":
            False,

        "created_at":
            utc_now(),
    }

    receipt_bytes = canonical_json_bytes(
        receipt
    )

    receipt_sha256 = sha256_bytes(
        receipt_bytes
    )

    # ========================================================
    # TEMPORARY REPOSITORY ARTIFACTS
    # ========================================================

    with temp_report_path.open(
        "xb"
    ) as handle:
        handle.write(
            report_bytes
        )

    with temp_receipt_path.open(
        "xb"
    ) as handle:
        handle.write(
            receipt_bytes
        )

    if (
        sha256_file(
            temp_report_path
        )
        !=
        report_sha256
    ):
        raise RuntimeError(
            "Temporary report integrity failure."
        )

    if (
        sha256_file(
            temp_receipt_path
        )
        !=
        receipt_sha256
    ):
        raise RuntimeError(
            "Temporary receipt integrity failure."
        )

    # ========================================================
    # ATOMIC FINAL PUBLICATION WITH ROLLBACK
    # ========================================================

    published_adapter = False

    published_report = False

    published_receipt = False

    try:
        os.replace(
            temp_adapter_path,
            adapter_path,
        )

        published_adapter = True

        os.replace(
            temp_report_path,
            report_path,
        )

        published_report = True

        os.replace(
            temp_receipt_path,
            receipt_path,
        )

        published_receipt = True

    except Exception:
        if (
            published_receipt
            and
            receipt_path.exists()
        ):
            receipt_path.unlink()

        if (
            published_report
            and
            report_path.exists()
        ):
            report_path.unlink()

        if (
            published_adapter
            and
            adapter_path.exists()
        ):
            shutil.rmtree(
                adapter_path
            )

        raise

    finally:
        if temp_adapter_path.exists():
            shutil.rmtree(
                temp_adapter_path
            )

        for path in (
            temp_report_path,
            temp_receipt_path,
        ):
            if path.exists():
                path.unlink()

    # ========================================================
    # FINAL PUBLISHED VALIDATION
    # ========================================================

    if (
        sha256_file(
            report_path
        )
        !=
        report_sha256
    ):
        raise RuntimeError(
            "Published training report "
            "SHA256 mismatch."
        )

    if (
        sha256_file(
            receipt_path
        )
        !=
        receipt_sha256
    ):
        raise RuntimeError(
            "Published training receipt "
            "SHA256 mismatch."
        )

    published_adapter_metadata = (
        adapter_bundle_metadata(
            adapter_path=
                adapter_path,
            safe_open_function=
                safe_open,
        )
    )

    if (
        published_adapter_metadata[
            "bundle_sha256"
        ]
        !=
        adapter_metadata[
            "bundle_sha256"
        ]
    ):
        raise RuntimeError(
            "Published adapter bundle "
            "SHA256 mismatch."
        )

    print()

    print(
        "OFFICIAL TRAINING RESULT"
    )

    print(
        (
            "  Micro-batches executed: "
            f"{global_micro_batch}"
        )
    )

    print(
        (
            "  optimizer.step() executed: "
            f"{global_optimizer_step}"
        )
    )

    print(
        (
            "  LoRA weights changed: "
            f"{weights_changed}"
        )
    )

    print(
        (
            "  Peak allocated: "
            f"{peak_allocated / 1024**2:.2f} MiB"
        )
    )

    print(
        (
            "  Peak reserved: "
            f"{peak_reserved / 1024**2:.2f} MiB"
        )
    )

    print(
        (
            "  Adapter bundle SHA256: "
            f"{adapter_metadata['bundle_sha256']}"
        )
    )

    print(
        (
            "  Report SHA256: "
            f"{report_sha256}"
        )
    )

    print(
        (
            "  Receipt SHA256: "
            f"{receipt_sha256}"
        )
    )

    print(
        "  Final Acceptance loaded: False"
    )

    print(
        "  Final Acceptance evaluated: False"
    )

    print()

    print(
        "DATALENS OFFICIAL QLORA TRAINING v0.1: PASS"
    )


def print_static_validation() -> None:
    result = validate_static_contract()

    manifest = result[
        "manifest"
    ]

    print(
        "=== DATALENS TRAINING RUNNER STATIC VALIDATION v0.1 ==="
    )

    print()

    print(
        "Training manifest: PASS"
    )

    print(
        "Manifest freeze: PASS"
    )

    print(
        "Preflight Git ancestry: PASS"
    )

    print(
        "QLoRA contract: PASS"
    )

    print(
        "Frozen dataset: PASS"
    )

    print(
        "Preflight evidence: PASS"
    )

    print(
        "Assistant-only Git blob binding: PASS"
    )

    print(
        "Final Acceptance isolation contract: PASS"
    )

    print()

    print(
        "TRAINING PLAN"
    )

    print(
        (
            "  Examples: "
            f"{manifest['dataset']['example_count']}"
        )
    )

    print(
        (
            "  Epochs: "
            f"{manifest['training']['epochs']}"
        )
    )

    print(
        (
            "  Micro-batches: "
            f"{manifest['training']['total_micro_batches']}"
        )
    )

    print(
        (
            "  Optimizer steps: "
            f"{manifest['training']['total_optimizer_steps']}"
        )
    )

    print(
        "  Token-weighted accumulation: True"
    )

    print(
        "  use_reentrant=False: True"
    )

    print()

    print(
        "SAFETY"
    )

    print(
        "  Heavy ML dependencies imported: False"
    )

    print(
        "  GPU computation: False"
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
        "  Final Acceptance loaded: False"
    )

    print()

    print(
        "DATALENS TRAINING RUNNER STATIC VALIDATION: PASS"
    )


def print_runtime_authorization() -> None:
    authorization = (
        collect_runtime_authorization()
    )

    print(
        "=== DATALENS TRAINING RUNTIME AUTHORIZATION v0.1 ==="
    )

    print()

    for name, value in authorization[
        "gates"
    ].items():
        print(
            f"  {name}: {value}"
        )

    print()

    print(
        (
            "Authorized: "
            f"{authorization['authorized']}"
        )
    )

    print(
        "GPU computation: False"
    )

    print(
        "Optimizer created: False"
    )

    print(
        "optimizer.step(): False"
    )

    print(
        "Training executed: False"
    )

    print(
        "Final Acceptance loaded: False"
    )

    if not authorization[
        "authorized"
    ]:
        raise RuntimeError(
            "Runtime training authorization failed."
        )

    print()

    print(
        "DATALENS TRAINING RUNTIME AUTHORIZATION: PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DataLens controlled QLoRA "
            "training runner v0.1"
        )
    )

    parser.add_argument(
        "mode",
        choices=(
            "validate-static",
            "authorize-only",
            "execute",
        ),
    )

    arguments = parser.parse_args()

    if (
        arguments.mode
        ==
        "validate-static"
    ):
        print_static_validation()

        return

    if (
        arguments.mode
        ==
        "authorize-only"
    ):
        print_runtime_authorization()

        return

    if (
        arguments.mode
        ==
        "execute"
    ):
        execute_training()

        return

    raise RuntimeError(
        "Unsupported runner mode."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


QLORA_V04_OPTIMIZATION_POLICY_RULE_VERSION = (
    "qlora_v0.4_optimization_policy_v0.1"
)

QLORA_V04_OPTIMIZATION_POLICY_FREEZE_RULE_VERSION = (
    "qlora_v0.4_optimization_policy_freeze_v0.1"
)

EXPERIMENT_ID = "datalens-semantic-qlora-v0.4"

POLICY_ID = (
    "optimization-policy:datalens-semantic-qlora:v0.4"
)


POLICY_RELATIVE_PATH = (
    "artifacts/adaptation/design/"
    "datalens_semantic_qlora_v0.4_"
    "optimization_policy_v0.1.json"
)

FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/design/"
    "datalens_semantic_qlora_v0.4_"
    "optimization_policy_v0.1_freeze.json"
)


DESIGN_RELATIVE_PATH = (
    "artifacts/adaptation/design/"
    "datalens_semantic_qlora_v0.4_"
    "experiment_design_v0.1.json"
)

DESIGN_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/design/"
    "datalens_semantic_qlora_v0.4_"
    "experiment_design_v0.1_freeze.json"
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

V03_CONTRACT_RELATIVE_PATH = (
    "artifacts/adaptation/experiments/"
    "datalens_semantic_qlora_v0.3_contract.json"
)

V03_CONTRACT_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/experiments/"
    "datalens_semantic_qlora_v0.3_contract_freeze.json"
)


EXPECTED_DESIGN_SHA256 = (
    "dd36103a01cadc49101dfeffa006bba9"
    "e9cf6cdfc3599a3f0967beff14765cf9"
)

EXPECTED_DESIGN_FREEZE_SHA256 = (
    "84d64a1406ce0e6648de54e4830eb96d"
    "d99b87a980108085906eb57875ef978a"
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

EXPECTED_V03_CONTRACT_SHA256 = (
    "609954fe4f06ace47000475053dcb011a"
    "4337e36122678860fb744eff645f92e"
)

EXPECTED_V03_CONTRACT_FREEZE_SHA256 = (
    "9e7eaedefe9823bf070e1c4c25574ee4"
    "2cc0081bd8bf3f1810255609cda4559c"
)


EXAMPLE_COUNT = 230
EPOCHS = 2
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 0.0002
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0

OPTIMIZER = "paged_adamw_8bit"
SCHEDULER = "cosine"

RANDOM_SEED = 42
MAX_SEQUENCE_LENGTH = 256

GRADIENT_CHECKPOINTING = True
BF16 = True
FP16 = False

PACKING = False


def _root(
    repository_root: Path | None,
) -> Path:
    root = (
        Path.cwd()
        if repository_root is None
        else repository_root
    )

    root = root.expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)

    return root


def _sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(
    path: Path,
) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_sha(
    *,
    repository_root: Path,
    relative_path: str,
    expected_sha256: str,
) -> Path:
    path = (
        repository_root
        /
        relative_path
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    actual = _sha256_file(path)

    if actual != expected_sha256:
        raise RuntimeError(
            "Evidence identity changed.\n"
            f"Path:     {relative_path}\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual}"
        )

    return path


def _canonical_json_bytes(
    value: Dict[str, Any],
) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _git_head(
    *,
    repository_root: Path,
) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_optimization_policy(
    *,
    repository_root: Path,
) -> Dict[str, Any]:
    repository_root = _root(repository_root)

    _require_sha(
        repository_root=repository_root,
        relative_path=DESIGN_RELATIVE_PATH,
        expected_sha256=EXPECTED_DESIGN_SHA256,
    )

    _require_sha(
        repository_root=repository_root,
        relative_path=DESIGN_FREEZE_RELATIVE_PATH,
        expected_sha256=EXPECTED_DESIGN_FREEZE_SHA256,
    )

    _require_sha(
        repository_root=repository_root,
        relative_path=DATASET_RELATIVE_PATH,
        expected_sha256=EXPECTED_DATASET_SHA256,
    )

    _require_sha(
        repository_root=repository_root,
        relative_path=DATASET_FREEZE_RELATIVE_PATH,
        expected_sha256=EXPECTED_DATASET_FREEZE_SHA256,
    )

    _require_sha(
        repository_root=repository_root,
        relative_path=TOKEN_AUDIT_RELATIVE_PATH,
        expected_sha256=EXPECTED_TOKEN_AUDIT_SHA256,
    )

    _require_sha(
        repository_root=repository_root,
        relative_path=V03_CONTRACT_RELATIVE_PATH,
        expected_sha256=EXPECTED_V03_CONTRACT_SHA256,
    )

    _require_sha(
        repository_root=repository_root,
        relative_path=V03_CONTRACT_FREEZE_RELATIVE_PATH,
        expected_sha256=EXPECTED_V03_CONTRACT_FREEZE_SHA256,
    )

    design = json.loads(
        (
            repository_root
            /
            DESIGN_RELATIVE_PATH
        ).read_text(
            encoding="utf-8-sig"
        )
    )

    strategy = design["training_strategy"]
    adaptation = design["model_adaptation"]

    required_design_values = {
        "assistant_only_loss": True,
        "gradient_checkpointing": True,
        "initial_epoch_budget": 2,
        "max_sequence_length": 256,
        "packing": False,
        "sequence_truncation": "fail_closed",
        "training_seed": 42,
    }

    for key, expected in required_design_values.items():
        actual = strategy.get(key)

        if actual != expected:
            raise RuntimeError(
                "Frozen experiment design changed.\n"
                f"{key}: expected={expected!r}, "
                f"actual={actual!r}"
            )

    if adaptation.get("lora_rank") != 16:
        raise RuntimeError("Frozen LoRA rank changed.")

    if adaptation.get("lora_alpha") != 32:
        raise RuntimeError("Frozen LoRA alpha changed.")

    if adaptation.get("lora_dropout") != 0.05:
        raise RuntimeError("Frozen LoRA dropout changed.")

    if (
        adaptation.get("quantization")
        !=
        "nf4_double_quant_bf16"
    ):
        raise RuntimeError(
            "Frozen quantization policy changed."
        )

    token_audit = json.loads(
        (
            repository_root
            /
            TOKEN_AUDIT_RELATIVE_PATH
        ).read_text(
            encoding="utf-8-sig"
        )
    )

    if (
        token_audit["dataset"]["example_count"]
        !=
        EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Token evidence example count changed."
        )

    if (
        token_audit["recommendation"][
            "recommended_max_sequence_length"
        ]
        !=
        MAX_SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            "Token evidence no longer supports seq=256."
        )

    if (
        token_audit["recommendation"][
            "truncated_examples"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Token evidence is no longer lossless."
        )

    micro_batches_per_epoch = math.ceil(
        EXAMPLE_COUNT / MICRO_BATCH_SIZE
    )

    full_groups_per_epoch = (
        micro_batches_per_epoch
        //
        GRADIENT_ACCUMULATION_STEPS
    )

    partial_group_size = (
        micro_batches_per_epoch
        %
        GRADIENT_ACCUMULATION_STEPS
    )

    if partial_group_size <= 0:
        raise RuntimeError(
            "v0.4 policy unexpectedly has no "
            "partial accumulation group."
        )

    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch
        /
        GRADIENT_ACCUMULATION_STEPS
    )

    total_micro_batches = (
        micro_batches_per_epoch
        *
        EPOCHS
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        *
        EPOCHS
    )

    example_presentations = (
        EXAMPLE_COUNT
        *
        EPOCHS
    )

    warmup_steps = max(
        1,
        math.ceil(
            total_optimizer_steps
            *
            WARMUP_RATIO
        ),
    )

    return {
        "policy_id":
            POLICY_ID,

        "rule_version":
            QLORA_V04_OPTIMIZATION_POLICY_RULE_VERSION,

        "experiment_id":
            EXPERIMENT_ID,

        "status":
            "pre_training_policy",

        "authority": {
            "policy_role":
                (
                    "Explicitly authorize v0.4 "
                    "optimization parameters before "
                    "resource preflight and GPU training."
                ),

            "implicit_pydantic_defaults_allowed":
                False,

            "implicit_v0_3_inheritance_allowed":
                False,

            "continuity_reference":
                {
                    "experiment_id":
                        "datalens-semantic-qlora-v0.3",

                    "contract_relative_path":
                        V03_CONTRACT_RELATIVE_PATH,

                    "contract_sha256":
                        EXPECTED_V03_CONTRACT_SHA256,

                    "contract_freeze_relative_path":
                        V03_CONTRACT_FREEZE_RELATIVE_PATH,

                    "contract_freeze_sha256":
                        EXPECTED_V03_CONTRACT_FREEZE_SHA256,
                },

            "decision":
                (
                    "Re-authorize the v0.3 optimizer "
                    "hyperparameters explicitly for "
                    "v0.4 to preserve experimental "
                    "comparability while changing the "
                    "training dataset and supervision "
                    "objective."
                ),
        },

        "evidence": {
            "experiment_design": {
                "relative_path":
                    DESIGN_RELATIVE_PATH,

                "sha256":
                    EXPECTED_DESIGN_SHA256,

                "freeze_relative_path":
                    DESIGN_FREEZE_RELATIVE_PATH,

                "freeze_sha256":
                    EXPECTED_DESIGN_FREEZE_SHA256,
            },

            "training_dataset": {
                "relative_path":
                    DATASET_RELATIVE_PATH,

                "sha256":
                    EXPECTED_DATASET_SHA256,

                "freeze_relative_path":
                    DATASET_FREEZE_RELATIVE_PATH,

                "freeze_sha256":
                    EXPECTED_DATASET_FREEZE_SHA256,

                "example_count":
                    EXAMPLE_COUNT,
            },

            "token_length": {
                "relative_path":
                    TOKEN_AUDIT_RELATIVE_PATH,

                "sha256":
                    EXPECTED_TOKEN_AUDIT_SHA256,

                "maximum_observed_tokens":
                    206,

                "recommended_max_sequence_length":
                    256,

                "truncated_examples":
                    0,
            },
        },

        "training": {
            "random_seed":
                RANDOM_SEED,

            "max_sequence_length":
                MAX_SEQUENCE_LENGTH,

            "micro_batch_size":
                MICRO_BATCH_SIZE,

            "gradient_accumulation_steps":
                GRADIENT_ACCUMULATION_STEPS,

            "initial_epoch_budget":
                EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "warmup_ratio":
                WARMUP_RATIO,

            "weight_decay":
                WEIGHT_DECAY,

            "optimizer":
                OPTIMIZER,

            "scheduler":
                SCHEDULER,

            "gradient_checkpointing":
                GRADIENT_CHECKPOINTING,

            "gradient_checkpoint_use_reentrant":
                False,

            "bf16":
                BF16,

            "fp16":
                FP16,

            "packing":
                PACKING,

            "sequence_truncation":
                "fail_closed",

            "assistant_only_loss":
                True,
        },

        "accumulation": {
            "policy":
                "flush_partial_group_at_epoch_end",

            "cross_epoch_accumulation":
                False,

            "discard_incomplete_group":
                False,

            "micro_batches_per_epoch":
                micro_batches_per_epoch,

            "full_groups_per_epoch":
                full_groups_per_epoch,

            "partial_group_count_per_epoch":
                1,

            "partial_group_size":
                partial_group_size,

            "optimizer_steps_per_epoch":
                optimizer_steps_per_epoch,

            "total_micro_batches":
                total_micro_batches,

            "total_optimizer_steps":
                total_optimizer_steps,

            "example_presentations":
                example_presentations,

            "discarded_example_presentations":
                0,

            "nominal_effective_batch_size":
                (
                    MICRO_BATCH_SIZE
                    *
                    GRADIENT_ACCUMULATION_STEPS
                ),

            "partial_group_effective_batch_size":
                (
                    MICRO_BATCH_SIZE
                    *
                    partial_group_size
                ),

            "gradient_accumulation_weighting":
                "supervised_token_weighted",

            "micro_batch_model_loss_reduction":
                "mean_over_supervised_assistant_tokens",

            "effective_group_objective":
                (
                    "mean_cross_entropy_over_all_"
                    "supervised_assistant_tokens_"
                    "in_accumulation_group"
                ),
        },

        "schedule": {
            "total_optimizer_steps":
                total_optimizer_steps,

            "warmup_step_rounding":
                "ceil_with_minimum_one",

            "warmup_steps":
                warmup_steps,

            "scheduler":
                SCHEDULER,
        },

        "adaptation": {
            "quantization":
                "nf4_double_quant_bf16",

            "lora_rank":
                16,

            "lora_alpha":
                32,

            "lora_dropout":
                0.05,

            "target_strategy":
                "language_model_all_linear",
        },

        "evaluation_boundary": {
            "training_loss_is_acceptance_evidence":
                False,

            "early_stopping_from_independent_holdout":
                False,

            "airport_used_for_training":
                False,

            "airport_evaluated_before_training":
                False,

            "final_acceptance_loaded_before_training":
                False,

            "final_acceptance_evaluated_before_training":
                False,
        },

        "implementation_requirement": {
            "historical_training_runner_v0_1_may_be_modified":
                False,

            "v0_4_runner_must_support_partial_accumulation":
                True,

            "v0_4_runner_must_flush_partial_group_per_epoch":
                True,

            "v0_4_runner_must_not_carry_gradients_across_epochs":
                True,

            "v0_4_runner_must_present_all_examples_each_epoch":
                True,
        },

        "safety": {
            "model_loaded":
                False,

            "adapter_loaded":
                False,

            "cuda_requested":
                False,

            "optimizer_created":
                False,

            "backward_executed":
                False,

            "training_executed":
                False,

            "airport_case_content_read":
                False,

            "airport_evaluated":
                False,

            "final_acceptance_loaded":
                False,

            "final_acceptance_evaluated":
                False,
        },
    }


def optimization_policy_bytes(
    *,
    repository_root: Path,
) -> bytes:
    return _canonical_json_bytes(
        build_optimization_policy(
            repository_root=repository_root
        )
    )


def optimization_policy_sha256(
    *,
    repository_root: Path,
) -> str:
    return _sha256_bytes(
        optimization_policy_bytes(
            repository_root=repository_root
        )
    )


def validate_optimization_policy(
    *,
    repository_root: Path,
) -> Dict[str, Any]:
    first = build_optimization_policy(
        repository_root=repository_root
    )

    second = build_optimization_policy(
        repository_root=repository_root
    )

    first_bytes = _canonical_json_bytes(first)
    second_bytes = _canonical_json_bytes(second)

    if first_bytes != second_bytes:
        raise RuntimeError(
            "Optimization policy is not deterministic."
        )

    accumulation = first["accumulation"]

    expected = {
        "micro_batches_per_epoch": 230,
        "full_groups_per_epoch": 28,
        "partial_group_count_per_epoch": 1,
        "partial_group_size": 6,
        "optimizer_steps_per_epoch": 29,
        "total_micro_batches": 460,
        "total_optimizer_steps": 58,
        "example_presentations": 460,
        "discarded_example_presentations": 0,
        "nominal_effective_batch_size": 8,
        "partial_group_effective_batch_size": 6,
    }

    for key, value in expected.items():
        if accumulation[key] != value:
            raise RuntimeError(
                f"Accumulation invariant changed: {key}"
            )

    if first["schedule"]["warmup_steps"] != 2:
        raise RuntimeError(
            "Expected two warmup steps for 58 steps."
        )

    return first


def _publish_new_bundle(
    *,
    outputs: Dict[Path, bytes],
) -> None:
    for path in outputs:
        if path.exists():
            raise FileExistsError(path)

    temporary = {}

    try:
        for path, payload in outputs.items():
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temp = (
                path.parent
                /
                (
                    "."
                    + path.name
                    + "."
                    + uuid.uuid4().hex
                    + ".tmp"
                )
            )

            with temp.open("xb") as handle:
                handle.write(payload)

            if _sha256_file(temp) != _sha256_bytes(payload):
                raise RuntimeError(
                    "Temporary artifact SHA mismatch."
                )

            temporary[path] = temp

        published = []

        try:
            for path, temp in temporary.items():
                os.replace(temp, path)
                published.append(path)

        except Exception:
            for path in reversed(published):
                if path.exists():
                    path.unlink()
            raise

    finally:
        for temp in temporary.values():
            if temp.exists():
                temp.unlink()


def freeze_optimization_policy(
    *,
    repository_root: Path,
) -> Dict[str, Any]:
    repository_root = _root(repository_root)

    policy = validate_optimization_policy(
        repository_root=repository_root
    )

    policy_bytes = _canonical_json_bytes(policy)
    policy_sha256 = _sha256_bytes(policy_bytes)

    policy_path = (
        repository_root
        /
        POLICY_RELATIVE_PATH
    ).resolve()

    freeze_path = (
        repository_root
        /
        FREEZE_RELATIVE_PATH
    ).resolve()

    freeze = {
        "freeze_id":
            (
                "optimization-policy-freeze:"
                "datalens-semantic-qlora:v0.4"
            ),

        "freeze_rule_version":
            QLORA_V04_OPTIMIZATION_POLICY_FREEZE_RULE_VERSION,

        "policy_rule_version":
            QLORA_V04_OPTIMIZATION_POLICY_RULE_VERSION,

        "experiment_id":
            EXPERIMENT_ID,

        "status":
            "frozen",

        "policy_relative_path":
            POLICY_RELATIVE_PATH,

        "policy_sha256":
            policy_sha256,

        "source_git_commit":
            _git_head(
                repository_root=repository_root
            ),

        "experiment_design_freeze_sha256":
            EXPECTED_DESIGN_FREEZE_SHA256,

        "training_dataset_freeze_sha256":
            EXPECTED_DATASET_FREEZE_SHA256,

        "token_length_audit_sha256":
            EXPECTED_TOKEN_AUDIT_SHA256,

        "supersedes_optimization_reference": {
            "experiment_id":
                "datalens-semantic-qlora-v0.3",

            "contract_sha256":
                EXPECTED_V03_CONTRACT_SHA256,

            "contract_freeze_sha256":
                EXPECTED_V03_CONTRACT_FREEZE_SHA256,
        },

        "frozen_before_resource_preflight":
            True,

        "frozen_before_v0_4_training":
            True,

        "training_started_at_freeze":
            False,

        "model_loaded_at_freeze":
            False,

        "optimizer_created_at_freeze":
            False,

        "airport_evaluated_at_freeze":
            False,

        "final_acceptance_loaded_at_freeze":
            False,

        "final_acceptance_evaluated_at_freeze":
            False,

        "frozen_at":
            _utc_now(),
    }

    freeze_bytes = _canonical_json_bytes(freeze)

    _publish_new_bundle(
        outputs={
            policy_path:
                policy_bytes,

            freeze_path:
                freeze_bytes,
        }
    )

    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=(
            "validate",
            "freeze",
        ),
    )

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )

    args = parser.parse_args()

    root = _root(args.repository_root)

    if args.command == "validate":
        policy_path = root / POLICY_RELATIVE_PATH
        freeze_path = root / FREEZE_RELATIVE_PATH

        if policy_path.exists() or freeze_path.exists():
            raise RuntimeError(
                "Official optimization-policy "
                "artifacts already exist."
            )

        policy = validate_optimization_policy(
            repository_root=root
        )

        print(
            "=== DATALENS QLORA v0.4 OPTIMIZATION POLICY v0.1 ==="
        )

        print()

        print(
            f"Policy ID: {policy['policy_id']}"
        )

        print(
            (
                "Future policy SHA256: "
                f"{optimization_policy_sha256(repository_root=root)}"
            )
        )

        print()

        print(
            "OPTIMIZATION"
        )

        training = policy["training"]

        for key in (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "initial_epoch_budget",
            "learning_rate",
            "warmup_ratio",
            "weight_decay",
            "optimizer",
            "scheduler",
            "max_sequence_length",
        ):
            print(
                f"  {key}: {training[key]}"
            )

        print()

        print(
            "ACCUMULATION"
        )

        accumulation = policy["accumulation"]

        for key in (
            "micro_batches_per_epoch",
            "full_groups_per_epoch",
            "partial_group_size",
            "optimizer_steps_per_epoch",
            "total_micro_batches",
            "total_optimizer_steps",
            "example_presentations",
            "discarded_example_presentations",
        ):
            print(
                f"  {key}: {accumulation[key]}"
            )

        print(
            (
                "  policy: "
                f"{accumulation['policy']}"
            )
        )

        print(
            "  cross_epoch_accumulation: False"
        )

        print()

        print(
            "SCHEDULE"
        )

        print(
            (
                "  total_optimizer_steps: "
                f"{policy['schedule']['total_optimizer_steps']}"
            )
        )

        print(
            (
                "  warmup_steps: "
                f"{policy['schedule']['warmup_steps']}"
            )
        )

        print()

        print(
            "SAFETY"
        )

        print(
            "  Official artifacts written: False"
        )

        print(
            "  Historical runner modified: False"
        )

        print(
            "  Model loaded: False"
        )

        print(
            "  CUDA requested: False"
        )

        print(
            "  Training executed: False"
        )

        print()

        print(
            "DATALENS QLORA v0.4 OPTIMIZATION POLICY v0.1: PASS"
        )

        return

    freeze_optimization_policy(
        repository_root=root
    )

    print(
        "DATALENS QLORA v0.4 OPTIMIZATION POLICY FREEZE: PASS"
    )


if __name__ == "__main__":
    main()

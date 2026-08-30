from __future__ import annotations

import ast
import inspect

from pathlib import Path


print(
    "=== DATALENS QLORA v0.4 OPTIMIZATION POLICY TEST v0.1 ==="
)

print()


ROOT = Path.cwd().resolve()


from app.adaptation.optimization_policy_v0_4 import (
    FREEZE_RELATIVE_PATH,
    POLICY_RELATIVE_PATH,
    QLORA_V04_OPTIMIZATION_POLICY_RULE_VERSION,
    build_optimization_policy,
    optimization_policy_sha256,
    validate_optimization_policy,
)


if (
    QLORA_V04_OPTIMIZATION_POLICY_RULE_VERSION
    !=
    "qlora_v0.4_optimization_policy_v0.1"
):
    raise RuntimeError(
        "Optimization policy rule mismatch."
    )


for relative_path in (
    POLICY_RELATIVE_PATH,
    FREEZE_RELATIVE_PATH,
):
    if (
        ROOT
        /
        relative_path
    ).exists():
        raise RuntimeError(
            "Official optimization artifact "
            "already exists."
        )


policy = validate_optimization_policy(
    repository_root=ROOT
)


if build_optimization_policy(
    repository_root=ROOT
) != policy:
    raise RuntimeError(
        "Policy recomputation changed."
    )


training = policy["training"]
accumulation = policy["accumulation"]
schedule = policy["schedule"]


expected_training = {
    "random_seed": 42,
    "max_sequence_length": 256,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "initial_epoch_budget": 2,
    "learning_rate": 0.0002,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "optimizer": "paged_adamw_8bit",
    "scheduler": "cosine",
    "gradient_checkpointing": True,
    "gradient_checkpoint_use_reentrant": False,
    "bf16": True,
    "fp16": False,
    "packing": False,
    "sequence_truncation": "fail_closed",
    "assistant_only_loss": True,
}


for key, expected in expected_training.items():
    if training[key] != expected:
        raise RuntimeError(
            f"Training policy mismatch: {key}"
        )


expected_accumulation = {
    "policy": "flush_partial_group_at_epoch_end",
    "cross_epoch_accumulation": False,
    "discard_incomplete_group": False,
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


for key, expected in expected_accumulation.items():
    if accumulation[key] != expected:
        raise RuntimeError(
            f"Accumulation policy mismatch: {key}"
        )


if schedule["warmup_steps"] != 2:
    raise RuntimeError(
        "Warmup step count mismatch."
    )


if (
    policy["authority"][
        "implicit_pydantic_defaults_allowed"
    ]
    is not False
):
    raise RuntimeError(
        "Implicit Pydantic defaults are not allowed."
    )


if (
    policy["authority"][
        "implicit_v0_3_inheritance_allowed"
    ]
    is not False
):
    raise RuntimeError(
        "Implicit v0.3 inheritance is not allowed."
    )


implementation = policy[
    "implementation_requirement"
]


for key in (
    "v0_4_runner_must_support_partial_accumulation",
    "v0_4_runner_must_flush_partial_group_per_epoch",
    "v0_4_runner_must_not_carry_gradients_across_epochs",
    "v0_4_runner_must_present_all_examples_each_epoch",
):
    if implementation[key] is not True:
        raise RuntimeError(
            f"Implementation requirement failed: {key}"
        )


if (
    implementation[
        "historical_training_runner_v0_1_may_be_modified"
    ]
    is not False
):
    raise RuntimeError(
        "Historical runner must remain immutable."
    )


module = __import__(
    "app.adaptation.optimization_policy_v0_4",
    fromlist=["*"],
)


source = inspect.getsource(module)
tree = ast.parse(source)


for forbidden in (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "AutoModel",
    "PeftModel",
    "airport_ground_operations_holdout",
    "greenhouse_operations_final_acceptance",
):
    if forbidden in source:
        raise RuntimeError(
            f"Forbidden dependency: {forbidden}"
        )


print(
    "Rule version: PASS"
)

print(
    "Explicit optimization policy: PASS"
)

print(
    "Implicit defaults disabled: PASS"
)

print(
    "Implicit v0.3 inheritance disabled: PASS"
)

print(
    "230-example accumulation arithmetic: PASS"
)

print(
    "Partial group 6/8: PASS"
)

print(
    "29 optimizer steps / epoch: PASS"
)

print(
    "58 total optimizer steps: PASS"
)

print(
    "460 example presentations: PASS"
)

print(
    "Zero discarded examples: PASS"
)

print(
    "Cross-epoch accumulation disabled: PASS"
)

print(
    "Historical runner immutable: PASS"
)

print(
    "Warmup steps = 2: PASS"
)

print(
    (
        "Future policy SHA256: "
        f"{optimization_policy_sha256(repository_root=ROOT)}"
    )
)

print()

print(
    "SAFETY"
)

print(
    "  Heavy ML imports: False"
)

print(
    "  Official artifacts written: False"
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
    "DATALENS QLORA v0.4 OPTIMIZATION POLICY TEST v0.1: PASS"
)

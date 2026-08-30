from __future__ import annotations

import ast
import inspect

from pathlib import Path


print(
    "=== DATALENS QLORA v0.4 EXPERIMENT CONTRACT TEST v0.1 ==="
)

print()


ROOT = Path.cwd().resolve()


from app.adaptation.experiment_contract_v0_4 import (
    CONTRACT_RELATIVE_PATH,
    FREEZE_RELATIVE_PATH,
    EXPECTED_AIRPORT_FREEZE_SHA256,
    EXPECTED_DATASET_FREEZE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256,
    EXPECTED_OPTIMIZATION_POLICY_FREEZE_SHA256,
    EXPECTED_OPTIMIZATION_POLICY_SHA256,
    EXPECTED_TOKEN_AUDIT_SHA256,
    EXPERIMENT_ID,
    build_contract_freeze,
    build_experiment_contract,
    contract_bytes,
    contract_dict,
    contract_sha256,
    validate_contract,
)


if (
    EXPERIMENT_ID
    !=
    "datalens-semantic-qlora-v0.4"
):
    raise RuntimeError(
        "Experiment ID mismatch."
    )


for relative_path in (
    CONTRACT_RELATIVE_PATH,
    FREEZE_RELATIVE_PATH,
):
    if (
        ROOT
        /
        relative_path
    ).exists():
        raise RuntimeError(
            "Official contract artifact "
            "already exists."
        )


first = validate_contract(
    repository_root=
        ROOT
)


second = validate_contract(
    repository_root=
        ROOT
)


if first != second:
    raise RuntimeError(
        "Contract object recomputation differs."
    )


first_bytes = contract_bytes(
    repository_root=
        ROOT
)


second_bytes = contract_bytes(
    repository_root=
        ROOT
)


if first_bytes != second_bytes:
    raise RuntimeError(
        "Contract byte recomputation differs."
    )


contract = build_experiment_contract(
    repository_root=
        ROOT
)


if (
    contract.model_dump(
        mode="json"
    )
    !=
    contract_dict(
        repository_root=
            ROOT
    )
):
    raise RuntimeError(
        "Pydantic contract serialization differs."
    )


if (
    first[
        "training_dataset"
    ][
        "dataset_sha256"
    ]
    !=
    EXPECTED_DATASET_SHA256
):
    raise RuntimeError(
        "Training dataset SHA mismatch."
    )


if (
    first[
        "training_dataset"
    ][
        "example_count"
    ]
    !=
    230
):
    raise RuntimeError(
        "Training dataset count mismatch."
    )


training = first[
    "training"
]


expected_training = {
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


for key, expected in expected_training.items():
    if training[
        key
    ] != expected:
        raise RuntimeError(
            f"Training mismatch: {key}"
        )


if len(
    first[
        "regression_baselines"
    ]
) != 1:
    raise RuntimeError(
        "Regression count mismatch."
    )


if len(
    first[
        "pre_adaptation_holdouts"
    ]
) != 5:
    raise RuntimeError(
        "Pre-adaptation count mismatch."
    )


if any(
    "airport"
    in
    item[
        "relative_path"
    ].casefold()

    for item
    in first[
        "pre_adaptation_holdouts"
    ]
):
    raise RuntimeError(
        "Airport leaked into pre-adaptation holdouts."
    )


freeze_preview = build_contract_freeze(
    repository_root=
        ROOT,

    contract_sha256_value=
        contract_sha256(
            repository_root=
                ROOT
        ),
)


if (
    freeze_preview[
        "authorities"
    ][
        "optimization_policy"
    ][
        "sha256"
    ]
    !=
    EXPECTED_OPTIMIZATION_POLICY_SHA256
):
    raise RuntimeError(
        "Optimization policy binding mismatch."
    )


if (
    freeze_preview[
        "authorities"
    ][
        "optimization_policy"
    ][
        "freeze_sha256"
    ]
    !=
    EXPECTED_OPTIMIZATION_POLICY_FREEZE_SHA256
):
    raise RuntimeError(
        "Optimization freeze binding mismatch."
    )


if (
    freeze_preview[
        "authorities"
    ][
        "token_length_evidence"
    ][
        "sha256"
    ]
    !=
    EXPECTED_TOKEN_AUDIT_SHA256
):
    raise RuntimeError(
        "Token evidence binding mismatch."
    )


if (
    freeze_preview[
        "training_dataset"
    ][
        "freeze_sha256"
    ]
    !=
    EXPECTED_DATASET_FREEZE_SHA256
):
    raise RuntimeError(
        "Dataset freeze binding mismatch."
    )


if (
    freeze_preview[
        "airport_independent_holdout"
    ][
        "freeze_sha256"
    ]
    !=
    EXPECTED_AIRPORT_FREEZE_SHA256
):
    raise RuntimeError(
        "Airport freeze binding mismatch."
    )


if (
    freeze_preview[
        "final_acceptance"
    ][
        "freeze_sha256"
    ]
    !=
    EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256
):
    raise RuntimeError(
        "Final Acceptance binding mismatch."
    )


execution = freeze_preview[
    "optimization_execution_plan"
]


expected_execution = {
    "micro_batches_per_epoch":
        230,

    "full_accumulation_groups_per_epoch":
        28,

    "partial_accumulation_group_size":
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

    "partial_group_policy":
        "flush_partial_group_at_epoch_end",
}


for key, expected in expected_execution.items():
    if execution[
        key
    ] != expected:
        raise RuntimeError(
            f"Execution-plan mismatch: {key}"
        )


if (
    freeze_preview[
        "airport_independent_holdout"
    ][
        "used_for_training"
    ]
    is not False
):
    raise RuntimeError(
        "Airport became training input."
    )


if (
    freeze_preview[
        "airport_independent_holdout"
    ][
        "used_for_hyperparameter_tuning"
    ]
    is not False
):
    raise RuntimeError(
        "Airport became tuning input."
    )


if (
    freeze_preview[
        "final_acceptance"
    ][
        "tuning_input"
    ]
    is not False
):
    raise RuntimeError(
        "Final Acceptance became tuning input."
    )


module = __import__(
    "app.adaptation.experiment_contract_v0_4",
    fromlist=[
        "*",
    ],
)


source = inspect.getsource(
    module
)


ast.parse(
    source
)


for forbidden in (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "AutoModel",
    "PeftModel",
):
    if forbidden in source:
        raise RuntimeError(
            (
                "Forbidden heavy ML dependency "
                f"in contract builder: {forbidden}"
            )
        )


if (
    "airport_ground_operations_holdout_v0.1_cases.json"
    in
    source
):
    raise RuntimeError(
        "Contract builder must not define or open "
        "the Airport cases path."
    )


print(
    "QLoRAExperimentContract v0.1 reuse: PASS"
)

print(
    "Contract deterministic serialization: PASS"
)

print(
    "Training dataset v0.4 / 230: PASS"
)

print(
    "Optimization policy mapping: PASS"
)

print(
    "Regression baseline count=1: PASS"
)

print(
    "Pre-adaptation holdout count=5: PASS"
)

print(
    "Airport excluded from pre-adaptation list: PASS"
)

print(
    "Airport freeze governance binding: PASS"
)

print(
    "Final Acceptance freeze binding: PASS"
)

print(
    "Partial accumulation execution plan: PASS"
)

print(
    "460 example presentations: PASS"
)

print(
    "58 optimizer steps: PASS"
)

print(
    "Heavy ML dependencies absent: PASS"
)

print(
    "Airport cases path absent: PASS"
)

print(
    (
        "Future contract SHA256: "
        f"{contract_sha256(repository_root=ROOT)}"
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
    "  Airport cases opened: False"
)

print(
    "  Airport evaluated: False"
)

print(
    "  Final Acceptance cases opened: False"
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
    "DATALENS QLORA v0.4 EXPERIMENT CONTRACT TEST v0.1: PASS"
)

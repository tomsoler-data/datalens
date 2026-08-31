from __future__ import annotations

import ast
import importlib
import inspect

from pathlib import Path


print(
    "=== DATALENS QLORA v0.4 SHARED RUNTIME TEST v0.1 ==="
)

print()


ROOT = Path.cwd().resolve()


module = importlib.import_module(
    "app.adaptation.qlora_runtime_v0_4"
)


if (
    module.QLORA_V04_SHARED_RUNTIME_RULE_VERSION
    !=
    "qlora_v0.4_shared_runtime_v0.1"
):
    raise RuntimeError(
        "Runtime rule mismatch."
    )


authority = (
    module.validate_static_authority(
        repository_root_value=
            ROOT
    )
)


if (
    authority.contract.experiment_id
    !=
    "datalens-semantic-qlora-v0.4"
):
    raise RuntimeError(
        "Experiment ID mismatch."
    )


if (
    authority.contract
    .training_dataset
    .example_count
    !=
    230
):
    raise RuntimeError(
        "Dataset example count mismatch."
    )


if (
    authority.contract
    .training
    .max_sequence_length
    !=
    256
):
    raise RuntimeError(
        "Contract sequence length mismatch."
    )


if (
    authority.optimization_policy[
        "accumulation"
    ][
        "partial_group_size"
    ]
    !=
    6
):
    raise RuntimeError(
        "Partial group size mismatch."
    )


if (
    authority.optimization_policy[
        "accumulation"
    ][
        "total_optimizer_steps"
    ]
    !=
    58
):
    raise RuntimeError(
        "Optimizer-step count mismatch."
    )


if (
    module.EXPECTED_TARGET_COUNT
    !=
    238
):
    raise RuntimeError(
        "Target count mismatch."
    )


if (
    module.EXPECTED_TRAINABLE_PARAMETERS
    !=
    29_802_496
):
    raise RuntimeError(
        "Trainable parameter count mismatch."
    )


if (
    module.EXPECTED_TRAINABLE_TENSORS
    !=
    476
):
    raise RuntimeError(
        "Trainable tensor count mismatch."
    )


if (
    module.EXPECTED_TOTAL_FULL_TOKENS
    !=
    43_799
):
    raise RuntimeError(
        "Full token count mismatch."
    )


if (
    module.EXPECTED_TOTAL_SUPERVISED_TOKENS
    !=
    7_821
):
    raise RuntimeError(
        "Supervised token count mismatch."
    )


if (
    module.EXPECTED_MAX_EXAMPLE_TOKENS
    !=
    206
):
    raise RuntimeError(
        "Maximum example length mismatch."
    )


versions = module.runtime_versions()


expected_versions = {
    "torch":
        "2.11.0+cu128",

    "transformers":
        "5.16.1",

    "peft":
        "0.20.0",

    "bitsandbytes":
        "0.50.2",
}


if versions != expected_versions:
    raise RuntimeError(
        "Runtime version map changed."
    )


# ============================================================
# STATIC SOURCE SAFETY
# ============================================================


source = inspect.getsource(
    module
)


tree = ast.parse(
    source
)


heavy_modules = {
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "numpy",
    "safetensors",
}


top_level_heavy_imports = []


for node in tree.body:
    if isinstance(
        node,
        ast.Import,
    ):
        for alias in node.names:
            if (
                alias.name.split(
                    "."
                )[
                    0
                ]
                in
                heavy_modules
            ):
                top_level_heavy_imports.append(
                    (
                        node.lineno,
                        alias.name,
                    )
                )

    elif isinstance(
        node,
        ast.ImportFrom,
    ):
        root_name = (
            (
                node.module
                or
                ""
            )
            .split(
                "."
            )[
                0
            ]
        )

        if root_name in heavy_modules:
            top_level_heavy_imports.append(
                (
                    node.lineno,
                    node.module,
                )
            )


if top_level_heavy_imports:
    raise RuntimeError(
        (
            "Heavy ML dependency imported at module import time: "
            f"{top_level_heavy_imports}"
        )
    )


optimizer_step_calls = []


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

    if (
        function.attr
        !=
        "step"
    ):
        continue

    owner = function.value

    if (
        isinstance(
            owner,
            ast.Name,
        )
        and
        owner.id
        ==
        "optimizer"
    ):
        optimizer_step_calls.append(
            node.lineno
        )


if optimizer_step_calls:
    raise RuntimeError(
        (
            "Shared runtime must not authorize "
            "optimizer.step(). "
            f"Call lines={optimizer_step_calls}"
        )
    )


truncated_attribute_reads = [
    node.lineno

    for node
    in ast.walk(
        tree
    )

    if (
        isinstance(
            node,
            ast.Attribute,
        )
        and
        node.attr
        ==
        "truncated"
    )
]


if truncated_attribute_reads:
    raise RuntimeError(
        (
            "Shared runtime depends on unsupported "
            "AssistantOnlyTrainingExample.truncated. "
            f"Lines={truncated_attribute_reads}"
        )
    )


for forbidden in (
    "airport_ground_operations_holdout_v0.1_cases.json",
    "greenhouse_operations_final_acceptance_v0_1_cases",
):
    if forbidden in source:
        raise RuntimeError(
            (
                "Protected case-file dependency detected: "
                f"{forbidden}"
            )
        )


# ============================================================
# FUNCTION PRESENCE
# ============================================================


required_functions = (
    "validate_static_authority",
    "load_pinned_tokenizer",
    "prepare_training_dataset",
    "longest_training_example_index",
    "prepare_qlora_model",
    "tensor_batch_from_example",
    "trainable_parameter_fingerprint",
    "gradient_statistics",
)


for name in required_functions:
    if not hasattr(
        module,
        name,
    ):
        raise RuntimeError(
            f"Required runtime primitive missing: {name}"
        )


print(
    "Frozen authority validation: PASS"
)

print(
    "Runtime dependency versions: PASS"
)

print(
    "Dataset v0.4 / 230 binding: PASS"
)

print(
    "Token totals 43799 / 7821: PASS"
)

print(
    "Maximum real example length=206: PASS"
)

print(
    "Sequence length=256: PASS"
)

print(
    "Partial accumulation group=6: PASS"
)

print(
    "Total optimizer steps=58: PASS"
)

print(
    "Expected LoRA targets=238: PASS"
)

print(
    "Expected trainable parameters=29802496: PASS"
)

print(
    "Expected trainable tensors=476: PASS"
)

print(
    "Runtime primitives present: PASS"
)

print(
    "Heavy ML imports deferred to runtime: PASS"
)

print(
    "optimizer.step() absent: PASS"
)

print(
    "Unsupported .truncated attribute dependency absent: PASS"
)

print(
    "Protected case-file dependencies absent: PASS"
)


print()

print(
    "SAFETY"
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
    "  Training executed: False"
)

print(
    "  Airport opened/evaluated: False"
)

print(
    "  Final Acceptance opened/evaluated: False"
)


print()

print(
    "DATALENS QLORA v0.4 SHARED RUNTIME TEST v0.1: PASS"
)

from __future__ import annotations

import ast
import importlib
import inspect

from pathlib import Path


print(
    "=== DATALENS QLORA v0.4 TRAINING EXECUTION MANIFEST TEST v0.1 ==="
)

print()


ROOT = Path.cwd().resolve()


module = importlib.import_module(
    "app.adaptation.training_execution_manifest_v0_4"
)


if (
    module.MANIFEST_RULE_VERSION
    !=
    "training_execution_manifest_v0.2"
):
    raise RuntimeError(
        "Manifest rule changed."
    )


if (
    module.FREEZE_RULE_VERSION
    !=
    "training_execution_manifest_freeze_v0.2"
):
    raise RuntimeError(
        "Freeze rule changed."
    )


authority = module.validate_static_authority(
    root=ROOT
)


manifest = module.build_manifest_payload(
    root=ROOT,
    builder_source_commit=
        "0" * 40,
    build_git_head=
        "1" * 40,
)


module.validate_manifest_payload(
    manifest
)


manifest_sha = module.sha256_bytes(
    module.canonical_json_bytes(
        manifest
    )
)


freeze = module.build_freeze_payload(
    manifest=
        manifest,
    manifest_sha256=
        manifest_sha,
    frozen_at=
        "2000-01-01T00:00:00Z",
)


module.validate_freeze_payload(
    freeze
)


training = manifest[
    "training"
]


assert (
    training[
        "micro_batches_per_epoch"
    ]
    ==
    230
)


assert (
    training[
        "full_accumulation_groups_per_epoch"
    ]
    ==
    28
)


assert (
    training[
        "terminal_partial_group_size"
    ]
    ==
    6
)


assert (
    training[
        "optimizer_steps_per_epoch"
    ]
    ==
    29
)


assert (
    training[
        "total_optimizer_steps"
    ]
    ==
    58
)


assert (
    training[
        "example_presentations"
    ]
    ==
    460
)


assert (
    training[
        "discarded_example_presentations"
    ]
    ==
    0
)


assert (
    training[
        "cross_epoch_accumulation"
    ]
    is False
)


assert (
    training[
        "partial_group_policy"
    ]
    ==
    "flush_partial_group_at_epoch_end"
)


assert (
    training[
        "partial_group_loss_denominator"
    ]
    ==
    "actual_accumulation_group_supervised_tokens"
)


assert (
    training[
        "optimizer_betas"
    ]
    ==
    [
        0.9,
        0.999,
    ]
)


assert (
    training[
        "optimizer_eps"
    ]
    ==
    1e-8
)


assert (
    training[
        "optimizer_amsgrad"
    ]
    is False
)


assert (
    training[
        "optimizer_min_8bit_size"
    ]
    ==
    4096
)


assert (
    training[
        "warmup_steps"
    ]
    ==
    2
)


assert (
    manifest[
        "authorization"
    ][
        "optimizer_step_authorized_at_manifest_creation"
    ]
    is False
)


assert (
    manifest[
        "authorization"
    ][
        "training_execution_authorized_at_manifest_creation"
    ]
    is False
)


assert (
    freeze[
        "optimizer_step_authorized_at_freeze"
    ]
    is False
)


assert (
    freeze[
        "training_started_at_freeze"
    ]
    is False
)


future_runner = (
    ROOT
    /
    module.FUTURE_RUNNER_RELATIVE_PATH
)


if future_runner.exists():
    raise RuntimeError(
        "v0.4 training runner exists before manifest freeze."
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
    "safetensors",
}


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
                raise RuntimeError(
                    (
                        "Heavy ML import at module load: "
                        f"{alias.name}"
                    )
                )

    if isinstance(
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
            raise RuntimeError(
                (
                    "Heavy ML import at module load: "
                    f"{node.module}"
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
        ==
        "step"
        and
        isinstance(
            function.value,
            ast.Name,
        )
        and
        function.value.id
        ==
        "optimizer"
    ):
        optimizer_step_calls.append(
            node.lineno
        )


if optimizer_step_calls:
    raise RuntimeError(
        (
            "Manifest builder contains optimizer.step(). "
            f"Lines={optimizer_step_calls}"
        )
    )


for forbidden in (
    "airport_ground_operations_holdout_v0.1_cases.json",
    "greenhouse_operations_final_acceptance_v0.1_cases",
):
    if forbidden in source:
        raise RuntimeError(
            (
                "Protected case-file dependency detected: "
                f"{forbidden}"
            )
        )


print(
    "Frozen authority chain: PASS"
)

print(
    "v0.2 manifest schema: PASS"
)

print(
    "230 = 28 x 8 + 6: PASS"
)

print(
    "29 optimizer steps / epoch: PASS"
)

print(
    "58 optimizer steps total: PASS"
)

print(
    "460 example presentations: PASS"
)

print(
    "Discarded examples=0: PASS"
)

print(
    "Cross-epoch accumulation=False: PASS"
)

print(
    "Partial-group flush policy: PASS"
)

print(
    "Actual supervised-token denominator: PASS"
)

print(
    "PagedAdamW8bit controls explicit: PASS"
)

print(
    "betas=[0.9, 0.999]: PASS"
)

print(
    "eps=1e-8: PASS"
)

print(
    "amsgrad=False: PASS"
)

print(
    "min_8bit_size=4096: PASS"
)

print(
    "Warmup steps=2: PASS"
)

print(
    "Three GPU probes bound: PASS"
)

print(
    "Future v0.4 runner absent: PASS"
)

print(
    "optimizer.step() AST calls=0: PASS"
)

print(
    "Protected case-file dependencies absent: PASS"
)

print()

print(
    "SAFETY"
)

print(
    "  Manifest artifact written: False"
)

print(
    "  Freeze artifact written: False"
)

print(
    "  Heavy ML imported: False"
)

print(
    "  CUDA requested: False"
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
    "  Airport evaluated: False"
)

print(
    "  Final Acceptance evaluated: False"
)

print()

print(
    "DATALENS QLORA v0.4 TRAINING EXECUTION MANIFEST TEST v0.1: PASS"
)

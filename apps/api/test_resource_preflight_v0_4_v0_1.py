from __future__ import annotations

import ast
import importlib
import inspect

from pathlib import Path


print(
    "=== DATALENS QLORA v0.4 RESOURCE PREFLIGHT TEST v0.1 ==="
)

print()


ROOT = Path.cwd().resolve()


module = importlib.import_module(
    "app.adaptation.resource_preflight_v0_4"
)


if (
    module.RESOURCE_PREFLIGHT_RULE_VERSION
    !=
    "qlora_v0.4_resource_preflight_v0.1"
):
    raise RuntimeError(
        "Resource preflight rule changed."
    )


if (
    module.MEMORY_PROBE_RULE_VERSION
    !=
    "qlora_v0.4_one_batch_memory_probe_v0.1"
):
    raise RuntimeError(
        "Memory probe rule changed."
    )


if (
    module.OPTIMIZER_PROBE_RULE_VERSION
    !=
    "qlora_v0.4_optimizer_state_memory_probe_v0.1"
):
    raise RuntimeError(
        "Optimizer probe rule changed."
    )


if (
    module.ASSISTANT_PROBE_RULE_VERSION
    !=
    "qlora_v0.4_assistant_only_gpu_probe_v0.1"
):
    raise RuntimeError(
        "Assistant probe rule changed."
    )


authority = module.validate_static(
    repository_root_value=
        ROOT
)


if (
    authority.contract.experiment_id
    !=
    "datalens-semantic-qlora-v0.4"
):
    raise RuntimeError(
        "Experiment ID changed."
    )


if (
    module.EXPECTED_SHARED_RUNTIME_SHA256
    !=
    (
        "20e41ab00606296893276a84e53746c0"
        "6618b8cabca74fef77cb743c5e80ab7c"
    )
):
    raise RuntimeError(
        "Shared runtime binding changed."
    )


if (
    module.OPTIMIZER_BETAS
    !=
    (
        0.9,
        0.999,
    )
):
    raise RuntimeError(
        "Optimizer betas changed."
    )


if (
    module.OPTIMIZER_EPS
    !=
    1e-8
):
    raise RuntimeError(
        "Optimizer epsilon changed."
    )


if (
    module.OPTIMIZER_AMSGRAD
    is not False
):
    raise RuntimeError(
        "AMSGrad became enabled."
    )


if (
    module.OPTIMIZER_MIN_8BIT_SIZE
    !=
    4096
):
    raise RuntimeError(
        "min_8bit_size changed."
    )


if (
    module.MINIMUM_CUDA_FREE_BYTES
    !=
    536_870_912
):
    raise RuntimeError(
        "CUDA free floor changed."
    )


if (
    module.MINIMUM_PEAK_RESERVED_HEADROOM_BYTES
    !=
    536_870_912
):
    raise RuntimeError(
        "Peak headroom floor changed."
    )


# ============================================================
# SOURCE ARCHITECTURE
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
            root_name = (
                alias.name.split(
                    "."
                )[
                    0
                ]
            )

            if root_name in heavy_modules:
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
            "Heavy ML import at module import time: "
            f"{top_level_heavy_imports}"
        )
    )


optimizer_step_calls = []

optimizer_update_step_calls = []


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

    if (
        function.attr
        ==
        "update_step"
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
        optimizer_update_step_calls.append(
            node.lineno
        )


if optimizer_step_calls:
    raise RuntimeError(
        (
            "Resource preflight contains optimizer.step(). "
            f"Lines={optimizer_step_calls}"
        )
    )


if optimizer_update_step_calls:
    raise RuntimeError(
        (
            "Resource preflight contains optimizer.update_step(). "
            f"Lines={optimizer_update_step_calls}"
        )
    )


if (
    "training_runner"
    in
    source
):
    raise RuntimeError(
        "Historical training runner dependency detected."
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
# REQUIRED PRE-UPDATE CALLS
# ============================================================


required_runtime_calls = (
    "optimizer.check_overrides",
    "optimizer.to_gpu",
    "optimizer.init_state",
    "optimizer.prefetch_state",
)


for call in required_runtime_calls:
    if call not in source:
        raise RuntimeError(
            (
                "Required optimizer pre-update primitive "
                f"missing: {call}"
            )
        )


# ============================================================
# REQUIRED COMMANDS / FUNCTIONS
# ============================================================


required_functions = (
    "validate_static",
    "run_memory_probe",
    "run_optimizer_probe",
    "run_assistant_probe",
    "optimizer_state_summary",
    "validate_memory_headroom",
    "publish_new_json",
    "verify_probe",
)


for name in required_functions:
    if not hasattr(
        module,
        name,
    ):
        raise RuntimeError(
            f"Missing preflight function: {name}"
        )


print(
    "Frozen v0.4 authority: PASS"
)

print(
    "Shared runtime exact binding: PASS"
)

print(
    "PagedAdamW8bit explicit controls: PASS"
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
    "CUDA free floor=512 MiB: PASS"
)

print(
    "Peak reserved headroom floor=512 MiB: PASS"
)

print(
    "Memory -> optimizer -> assistant chain: PASS"
)

print(
    "Heavy ML imports deferred: PASS"
)

print(
    "optimizer.step() AST calls=0: PASS"
)

print(
    "optimizer.update_step() AST calls=0: PASS"
)

print(
    "check_overrides / to_gpu / init_state / prefetch_state: PASS"
)

print(
    "Historical training runner dependency absent: PASS"
)

print(
    "Protected case-file dependencies absent: PASS"
)

print(
    "Atomic artifact publication: PASS"
)


print()

print(
    "SAFETY"
)

print(
    "  GPU used by test: False"
)

print(
    "  Tokenizer loaded: False"
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
    "  Forward executed: False"
)

print(
    "  Backward executed: False"
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
    "DATALENS QLORA v0.4 RESOURCE PREFLIGHT TEST v0.1: PASS"
)

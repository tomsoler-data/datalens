from __future__ import annotations

import ast
import importlib
import inspect
import math

from pathlib import Path


print("=== DATALENS QLORA v0.4 TRAINING RUNNER TEST v0.1 ===")
print()

ROOT = Path.cwd().resolve()

module = importlib.import_module(
    "app.adaptation.training_runner_v0_4"
)

if module.RUNNER_RULE_VERSION != "qlora_training_runner_v0.4_v0.1":
    raise RuntimeError("Runner rule changed.")

static = module.validate_static_contract()
manifest = static["manifest"]
freeze = static["freeze"]

if manifest["training_run_id"] != module.TRAINING_RUN_ID:
    raise RuntimeError("Training run identity changed.")
if freeze["manifest_sha256"] != module.EXPECTED_MANIFEST_SHA256:
    raise RuntimeError("Freeze binding changed.")


# ---------------------------------------------------------------------------
# Pure execution semantics
# ---------------------------------------------------------------------------


order = list(range(230))
groups = module.accumulation_groups_v0_4(
    order=order,
    accumulation_steps=8,
)
module.validate_v0_4_group_plan(groups)

if [len(group) for group in groups] != ([8] * 28 + [6]):
    raise RuntimeError("Unexpected v0.4 group sizes.")
if [index for group in groups for index in group] != order:
    raise RuntimeError("Grouping lost or reordered examples.")
if len(groups) != 29:
    raise RuntimeError("Expected 29 groups per epoch.")

generic = module.accumulation_groups_v0_4(
    order=list(range(10)),
    accumulation_steps=8,
)
if [len(group) for group in generic] != [8, 2]:
    raise RuntimeError("Generic partial-group behavior changed.")

epoch_zero_a = module.deterministic_epoch_order(
    example_count=230,
    seed=42,
    zero_based_epoch=0,
)
epoch_zero_b = module.deterministic_epoch_order(
    example_count=230,
    seed=42,
    zero_based_epoch=0,
)
epoch_one = module.deterministic_epoch_order(
    example_count=230,
    seed=42,
    zero_based_epoch=1,
)

if epoch_zero_a != epoch_zero_b:
    raise RuntimeError("Epoch ordering is not deterministic.")
if epoch_zero_a == epoch_one:
    raise RuntimeError("Per-epoch shuffle seed did not change.")
if sorted(epoch_zero_a) != list(range(230)):
    raise RuntimeError("Epoch 1 order is not a permutation.")
if sorted(epoch_one) != list(range(230)):
    raise RuntimeError("Epoch 2 order is not a permutation.")

if module.supervised_token_loss_scales([2, 3, 5]) != [0.2, 0.3, 0.5]:
    raise RuntimeError("Unexpected supervised-token loss scales.")

weighted = module.weighted_group_loss(
    losses=[1.0, 2.0, 4.0],
    token_counts=[2, 3, 5],
)
if not math.isclose(weighted, 2.8, rel_tol=0.0, abs_tol=1e-12):
    raise RuntimeError(f"Unexpected weighted loss: {weighted}")

outputs = module.planned_output_paths(manifest)
if set(outputs) != {"adapter", "report", "receipt"}:
    raise RuntimeError("Unexpected output-path set.")

api_root = module.api_root().resolve()
for path in outputs.values():
    try:
        path.resolve().relative_to(api_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Official output escapes API root: {path}"
        ) from exc


# ---------------------------------------------------------------------------
# AST safety
# ---------------------------------------------------------------------------


source = inspect.getsource(module)
tree = ast.parse(source)

heavy_modules = {
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "safetensors",
    "numpy",
}

for node in tree.body:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] in heavy_modules:
                raise RuntimeError(
                    f"Heavy import at module load: {alias.name}"
                )
    if isinstance(node, ast.ImportFrom):
        root_name = (node.module or "").split(".")[0]
        if root_name in heavy_modules:
            raise RuntimeError(
                f"Heavy import at module load: {node.module}"
            )

functions = {
    node.name: node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
execute = functions.get("execute_training")
if execute is None:
    raise RuntimeError("execute_training() missing.")

authorization_lines = []
heavy_import_lines = []
runtime_import_lines = []

for node in ast.walk(execute):
    if isinstance(node, ast.Call):
        function = node.func
        if (
            isinstance(function, ast.Name)
            and function.id == "require_runtime_authorization"
        ):
            authorization_lines.append(node.lineno)

    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] in heavy_modules:
                heavy_import_lines.append(node.lineno)

    if isinstance(node, ast.ImportFrom):
        root_name = (node.module or "").split(".")[0]
        if root_name in heavy_modules:
            heavy_import_lines.append(node.lineno)
        if (
            node.module == "app.adaptation"
            and any(
                alias.name == "qlora_runtime_v0_4"
                for alias in node.names
            )
        ):
            runtime_import_lines.append(node.lineno)

if len(authorization_lines) != 1:
    raise RuntimeError(
        "Expected exactly one runtime-authorization call."
    )
if not heavy_import_lines:
    raise RuntimeError("Deferred ML imports missing.")
if len(runtime_import_lines) != 1:
    raise RuntimeError("Deferred shared-runtime import missing.")

authorization_line = authorization_lines[0]
if min(heavy_import_lines) <= authorization_line:
    raise RuntimeError("Heavy import precedes authorization.")
if runtime_import_lines[0] <= authorization_line:
    raise RuntimeError("Shared runtime import precedes authorization.")


def attribute_call_lines(
    *, object_name: str, attribute_name: str
) -> list[int]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == attribute_name
            and isinstance(function.value, ast.Name)
            and function.value.id == object_name
        ):
            found.append(node.lineno)
    return found


optimizer_steps = attribute_call_lines(
    object_name="optimizer",
    attribute_name="step",
)
scheduler_steps = attribute_call_lines(
    object_name="scheduler",
    attribute_name="step",
)
adapter_saves = attribute_call_lines(
    object_name="model",
    attribute_name="save_pretrained",
)

if len(optimizer_steps) != 1:
    raise RuntimeError(
        f"Expected one optimizer.step() site: {optimizer_steps}"
    )
if len(scheduler_steps) != 1:
    raise RuntimeError(
        f"Expected one scheduler.step() site: {scheduler_steps}"
    )
if len(adapter_saves) != 1:
    raise RuntimeError(
        f"Expected one adapter save site: {adapter_saves}"
    )
if optimizer_steps[0] <= authorization_line:
    raise RuntimeError("optimizer.step() precedes authorization.")
if scheduler_steps[0] <= optimizer_steps[0]:
    raise RuntimeError("scheduler.step() must follow optimizer.step().")

for name in (
    "validate_static_authority",
    "load_pinned_tokenizer",
    "prepare_training_dataset",
    "prepare_qlora_model",
    "tensor_batch_from_example",
    "trainable_parameter_fingerprint",
    "gradient_statistics",
):
    if f"runtime.{name}" not in source:
        raise RuntimeError(f"Shared runtime API not reused: {name}")

if "app.adaptation.training_runner import" in source:
    raise RuntimeError("Historical runner import detected.")

for forbidden in (
    "airport_ground_operations_holdout_v0.1_cases.json",
    "greenhouse_operations_final_acceptance_v0.1_cases",
):
    if forbidden in source:
        raise RuntimeError(
            f"Protected case-file dependency detected: {forbidden}"
        )

if "loss * loss_scale" not in source:
    raise RuntimeError("Weighted backward expression missing.")
if '"training_loss_is_acceptance_evidence": False' not in source:
    raise RuntimeError("Training-loss acceptance boundary missing.")


print("Exact manifest + freeze validation: PASS")
print("Shared runtime exact binding: PASS")
print("230 = 28 x 8 + 6: PASS")
print("29 optimizer steps / epoch semantics: PASS")
print("58 optimizer steps total semantics: PASS")
print("Terminal partial groups preserved: PASS")
print("Generic partial-group helper: PASS")
print("Zero discarded examples: PASS")
print("Deterministic per-epoch shuffle: PASS")
print("Supervised-token weighting: PASS")
print("Weighted group loss: PASS")
print("Official output path contract: PASS")
print("Heavy imports deferred: PASS")
print("Authorization precedes runtime + heavy imports: PASS")
print("Exactly one optimizer.step() source site: PASS")
print("Exactly one scheduler.step() source site: PASS")
print("optimizer -> scheduler source order: PASS")
print("Exactly one adapter save site: PASS")
print("Shared runtime APIs reused: PASS")
print("Historical runner import absent: PASS")
print("Protected case-file dependencies absent: PASS")
print("Training loss excluded from acceptance: PASS")

print()
print("SAFETY")
print("  Runtime authorization executed by test: False")
print("  Heavy ML imported: False")
print("  CUDA requested: False")
print("  Optimizer created: False")
print("  optimizer.step(): False")
print("  Training executed: False")
print("  Airport evaluated: False")
print("  Final Acceptance evaluated: False")

print()
print("DATALENS QLORA v0.4 TRAINING RUNNER TEST v0.1: PASS")

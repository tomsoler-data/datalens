from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


RUNNER_RULE_VERSION = "qlora_training_runner_v0.4_v0.1"
REPORT_RULE_VERSION = "qlora_training_report_v0.4_v0.1"
RECEIPT_RULE_VERSION = "qlora_training_receipt_v0.4_v0.1"

EXPERIMENT_ID = "datalens-semantic-qlora-v0.4"
TRAINING_RUN_ID = "training-run:datalens-semantic-qlora:v0.4:0001"

RUNNER_RELATIVE_PATH = "app/adaptation/training_runner_v0_4.py"
RUNNER_TEST_RELATIVE_PATH = "test_training_runner_v0_4_v0_1.py"
MANIFEST_RELATIVE_PATH = (
    "artifacts/adaptation/training/"
    "datalens_semantic_qlora_v0.4_training_v0.1_manifest.json"
)
FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/training/"
    "datalens_semantic_qlora_v0.4_training_v0.1_manifest_freeze.json"
)
SHARED_RUNTIME_RELATIVE_PATH = "app/adaptation/qlora_runtime_v0_4.py"

EXPECTED_MANIFEST_SHA256 = (
    "824770d7827a732df78121fe4fc515f8"
    "3bfeb8314f524188ce5fe89f43730756"
)
EXPECTED_FREEZE_SHA256 = (
    "b8fc28e8f1b99f981570243a00609677"
    "536e7ccc7997d6d74db4b427274b6de0"
)
EXPECTED_SHARED_RUNTIME_SHA256 = (
    "20e41ab00606296893276a84e53746c0"
    "6618b8cabca74fef77cb743c5e80ab7c"
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

EXPECTED_EXAMPLES = 230
EXPECTED_EPOCHS = 2
EXPECTED_ACCUMULATION_STEPS = 8
EXPECTED_FULL_GROUPS_PER_EPOCH = 28
EXPECTED_PARTIAL_GROUP_SIZE = 6
EXPECTED_GROUPS_PER_EPOCH = 29
EXPECTED_TOTAL_MICRO_BATCHES = 460
EXPECTED_TOTAL_OPTIMIZER_STEPS = 58
EXPECTED_SUPERVISED_TOKENS_PER_EPOCH = 7821
EXPECTED_TOTAL_SUPERVISED_TOKENS = 15642
EXPECTED_TARGET_MODULE_COUNT = 238
EXPECTED_TRAINABLE_PARAMETER_COUNT = 29_802_496
EXPECTED_TRAINABLE_TENSOR_COUNT = 476


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def api_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=api_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return Path(result.stdout.strip()).resolve()


def run_git_text(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=git_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed:\n{result.stderr}"
        )
    return result.stdout.strip()


def git_relative_path(path: Path) -> str:
    return path.resolve().relative_to(git_root()).as_posix()


def git_is_clean() -> bool:
    return run_git_text(["status", "--porcelain"]) == ""


def git_path_is_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", git_relative_path(path)],
        cwd=git_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0


def git_latest_commit_for_path(path: Path) -> str:
    value = run_git_text(
        ["log", "-1", "--format=%H", "--", git_relative_path(path)]
    )
    if len(value) != 40:
        raise RuntimeError(f"Cannot resolve commit for {path}")
    return value


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=git_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError("git merge-base --is-ancestor failed.")


def git_blob_bytes(*, ref: str, path: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{ref}:{git_relative_path(path)}"],
        cwd=git_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout


def git_blob_sha256(*, ref: str, path: Path) -> str:
    return hashlib.sha256(git_blob_bytes(ref=ref, path=path)).hexdigest()


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def resolve_api_relative_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise RuntimeError("Invalid API-relative path.")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise RuntimeError("Expected relative path.")
    root = api_root().resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Path escapes API root.") from exc
    return resolved


def require_exact_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"SHA mismatch for {path}\n"
            f"Expected: {expected_sha256}\nActual:   {actual}"
        )


def manifest_path() -> Path:
    return resolve_api_relative_path(MANIFEST_RELATIVE_PATH)


def manifest_freeze_path() -> Path:
    return resolve_api_relative_path(FREEZE_RELATIVE_PATH)


def runner_path() -> Path:
    return resolve_api_relative_path(RUNNER_RELATIVE_PATH)


def runner_test_path() -> Path:
    return resolve_api_relative_path(RUNNER_TEST_RELATIVE_PATH)


def shared_runtime_path() -> Path:
    return resolve_api_relative_path(SHARED_RUNTIME_RELATIVE_PATH)


def planned_output_paths(manifest: Mapping[str, Any]) -> dict[str, Path]:
    outputs = manifest["planned_outputs"]
    paths = {
        "adapter": resolve_api_relative_path(outputs["adapter_directory"]),
        "report": resolve_api_relative_path(outputs["training_report"]),
        "receipt": resolve_api_relative_path(outputs["training_receipt"]),
    }
    if len({path.resolve() for path in paths.values()}) != 3:
        raise RuntimeError("Training output paths are not distinct.")
    return paths


# ---------------------------------------------------------------------------
# Frozen v0.4 authority
# ---------------------------------------------------------------------------


def validate_static_contract() -> dict[str, Any]:
    manifest_file = manifest_path()
    freeze_file = manifest_freeze_path()
    runtime_file = shared_runtime_path()

    require_exact_file(manifest_file, EXPECTED_MANIFEST_SHA256)
    require_exact_file(freeze_file, EXPECTED_FREEZE_SHA256)
    require_exact_file(runtime_file, EXPECTED_SHARED_RUNTIME_SHA256)

    manifest = load_json(manifest_file)
    freeze = load_json(freeze_file)

    if manifest["rule_version"] != "training_execution_manifest_v0.2":
        raise RuntimeError("Unexpected manifest rule.")
    if freeze["rule_version"] != "training_execution_manifest_freeze_v0.2":
        raise RuntimeError("Unexpected freeze rule.")
    if manifest["training_run_id"] != TRAINING_RUN_ID:
        raise RuntimeError("Unexpected training run ID.")
    if freeze["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Freeze does not bind exact manifest.")
    if freeze["frozen_before_training"] is not True:
        raise RuntimeError("Manifest was not frozen before training.")

    training = manifest["training"]
    expected_training = {
        "max_sequence_length": 256,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "nominal_effective_batch_size": 8,
        "epochs": 2,
        "micro_batches_per_epoch": 230,
        "full_accumulation_groups_per_epoch": 28,
        "partial_accumulation_groups_per_epoch": 1,
        "terminal_partial_group_size": 6,
        "accumulation_groups_per_epoch": 29,
        "optimizer_steps_per_epoch": 29,
        "total_micro_batches": 460,
        "total_optimizer_steps": 58,
        "example_presentations": 460,
        "discarded_example_presentations": 0,
        "cross_epoch_accumulation": False,
        "discard_incomplete_group": False,
        "partial_group_policy": "flush_partial_group_at_epoch_end",
        "partial_group_optimizer_step": True,
        "gradient_accumulation_weighting": "supervised_token_weighted",
        "gradient_accumulation_scale_formula":
            "micro_batch_loss * micro_batch_supervised_tokens / "
            "accumulation_group_supervised_tokens",
        "partial_group_loss_denominator":
            "actual_accumulation_group_supervised_tokens",
        "micro_batch_model_loss_reduction":
            "mean_over_supervised_assistant_tokens",
        "effective_objective":
            "mean_cross_entropy_over_all_supervised_assistant_tokens_in_"
            "accumulation_group",
        "optimizer": "paged_adamw_8bit",
        "optimizer_implementation": "bitsandbytes.optim.PagedAdamW8bit",
        "learning_rate": 0.0002,
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1e-8,
        "weight_decay": 0.0,
        "optimizer_amsgrad": False,
        "optimizer_min_8bit_size": 4096,
        "optimizer_zero_grad_policy":
            "set_to_none_true_after_each_optimizer_step",
        "scheduler": "cosine",
        "scheduler_implementation":
            "transformers.get_cosine_schedule_with_warmup",
        "scheduler_step_order": "optimizer_then_scheduler",
        "warmup_ratio": 0.03,
        "warmup_rounding_policy": "ceil_with_minimum_one",
        "warmup_steps": 2,
        "seed": 42,
        "shuffle": True,
        "shuffle_policy": "deterministic_epoch_seed",
        "epoch_seed_policy": "seed_plus_zero_based_epoch",
        "gradient_checkpointing": True,
        "gradient_checkpoint_use_reentrant": False,
        "gradient_clipping": None,
        "bf16": True,
        "fp16": False,
        "packing": False,
        "drop_last": False,
        "dataloader_num_workers": 0,
        "silent_truncation_allowed": False,
    }
    for key, expected in expected_training.items():
        actual = training.get(key)
        if actual != expected:
            raise RuntimeError(
                f"Training semantic changed: {key}\n"
                f"Expected: {expected!r}\nActual:   {actual!r}"
            )

    if (
        training["full_accumulation_groups_per_epoch"]
        * training["gradient_accumulation_steps"]
        + training["terminal_partial_group_size"]
        != training["micro_batches_per_epoch"]
    ):
        raise RuntimeError("Accumulation arithmetic changed.")

    if (
        training["optimizer_steps_per_epoch"] * training["epochs"]
        != training["total_optimizer_steps"]
    ):
        raise RuntimeError("Optimizer-step arithmetic changed.")

    expected_warmup = max(
        1,
        math.ceil(
            training["warmup_ratio"] * training["total_optimizer_steps"]
        ),
    )
    if training["warmup_steps"] != expected_warmup:
        raise RuntimeError("Warmup derivation changed.")

    preflight = manifest["preflight_evidence"]
    expected_probe_bindings = {
        "sequence_memory_probe": EXPECTED_MEMORY_PROBE_SHA256,
        "optimizer_state_memory_probe": EXPECTED_OPTIMIZER_PROBE_SHA256,
        "assistant_only_gpu_probe": EXPECTED_ASSISTANT_PROBE_SHA256,
    }
    for name, expected_sha in expected_probe_bindings.items():
        if preflight[name]["sha256"] != expected_sha:
            raise RuntimeError(f"Probe binding changed: {name}")
        if preflight[name]["passed"] is not True:
            raise RuntimeError(f"Probe did not pass: {name}")

    if (
        preflight["optimizer_state_memory_probe"][
            "source_memory_probe_sha256"
        ]
        != EXPECTED_MEMORY_PROBE_SHA256
    ):
        raise RuntimeError("Optimizer -> memory probe chain changed.")
    if (
        preflight["assistant_only_gpu_probe"][
            "source_optimizer_probe_sha256"
        ]
        != EXPECTED_OPTIMIZER_PROBE_SHA256
    ):
        raise RuntimeError("Assistant -> optimizer probe chain changed.")

    authorization = manifest["authorization"]
    for key in (
        "optimizer_step_requires_runtime_execution_gates",
        "runner_must_bind_exact_manifest",
        "runner_must_bind_exact_manifest_freeze",
        "runner_must_bind_exact_shared_runtime",
        "runner_must_bind_exact_probe_chain",
        "manifest_commit_must_be_ancestor_of_training_start_head",
        "resource_preflight_source_commit_must_be_ancestor_of_training_start_head",
        "training_runner_commit_must_equal_training_start_head",
    ):
        if authorization[key] is not True:
            raise RuntimeError(f"Required runtime gate disabled: {key}")

    for key in (
        "optimizer_step_authorized_at_manifest_creation",
        "training_execution_authorized_at_manifest_creation",
        "airport_evaluation_authorized_before_training",
        "final_acceptance_evaluation_authorized_before_training",
    ):
        if authorization[key] is not False:
            raise RuntimeError(f"Premature authorization: {key}")

    for key in (
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
    ):
        if freeze[key] is not False:
            raise RuntimeError(f"Freeze invariant changed: {key}")

    if freeze["training_runner_relative_path"] != RUNNER_RELATIVE_PATH:
        raise RuntimeError("Freeze runner path changed.")
    if freeze["training_runner_commit_must_equal_training_start_head"] is not True:
        raise RuntimeError("Freeze runner/HEAD gate disabled.")

    protected = manifest["protected_evaluation"]
    for key in (
        "used_for_training",
        "used_for_hyperparameter_tuning",
        "evaluation_executed_before_training",
        "results_observed_before_training",
    ):
        if protected["airport_independent_holdout"][key] is not False:
            raise RuntimeError(f"Airport boundary failed: {key}")

    for key in (
        "used_for_training",
        "used_for_hyperparameter_tuning",
        "cases_loaded_before_training",
        "evaluation_executed_before_training",
    ):
        if protected["final_acceptance"][key] is not False:
            raise RuntimeError(f"Final Acceptance boundary failed: {key}")

    return {"manifest": manifest, "freeze": freeze}


# ---------------------------------------------------------------------------
# Pure training semantics
# ---------------------------------------------------------------------------


def deterministic_epoch_order(
    *, example_count: int, seed: int, zero_based_epoch: int
) -> list[int]:
    if example_count <= 0:
        raise ValueError("example_count must be positive.")
    if zero_based_epoch < 0:
        raise ValueError("zero_based_epoch must be non-negative.")
    order = list(range(example_count))
    random.Random(seed + zero_based_epoch).shuffle(order)
    if sorted(order) != list(range(example_count)):
        raise RuntimeError("Epoch order is not a permutation.")
    return order


def accumulation_groups_v0_4(
    *, order: Sequence[int], accumulation_steps: int
) -> list[list[int]]:
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive.")
    if not order:
        raise ValueError("order must not be empty.")

    groups = [
        list(order[start:start + accumulation_steps])
        for start in range(0, len(order), accumulation_steps)
    ]
    if any(not group or len(group) > accumulation_steps for group in groups):
        raise RuntimeError("Invalid accumulation grouping.")
    if [index for group in groups for index in group] != list(order):
        raise RuntimeError("Grouping lost or reordered examples.")
    return groups


def validate_v0_4_group_plan(groups: Sequence[Sequence[int]]) -> None:
    expected_sizes = (
        [EXPECTED_ACCUMULATION_STEPS] * EXPECTED_FULL_GROUPS_PER_EPOCH
        + [EXPECTED_PARTIAL_GROUP_SIZE]
    )
    sizes = [len(group) for group in groups]
    if sizes != expected_sizes:
        raise RuntimeError(
            f"Unexpected v0.4 group sizes.\n"
            f"Expected: {expected_sizes}\nActual:   {sizes}"
        )
    if len(groups) != EXPECTED_GROUPS_PER_EPOCH:
        raise RuntimeError("Unexpected group count per epoch.")


def supervised_token_loss_scales(
    token_counts: Sequence[int],
) -> list[float]:
    if not token_counts:
        raise ValueError("token_counts must not be empty.")
    if any(count <= 0 for count in token_counts):
        raise ValueError("Every supervised token count must be positive.")
    total = sum(token_counts)
    scales = [count / total for count in token_counts]
    if not math.isclose(sum(scales), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Loss scales do not sum to 1.")
    return scales


def weighted_group_loss(
    *, losses: Sequence[float], token_counts: Sequence[int]
) -> float:
    if len(losses) != len(token_counts):
        raise ValueError("losses/token_counts length mismatch.")
    return sum(
        loss * scale
        for loss, scale in zip(
            losses,
            supervised_token_loss_scales(token_counts),
        )
    )


# ---------------------------------------------------------------------------
# Runtime authorization
# ---------------------------------------------------------------------------


def collect_runtime_authorization() -> dict[str, Any]:
    static = validate_static_contract()
    manifest = static["manifest"]
    head = run_git_text(["rev-parse", "HEAD"])

    manifest_file = manifest_path()
    freeze_file = manifest_freeze_path()
    current_runner = runner_path()
    current_test = runner_test_path()

    manifest_commit = (
        git_latest_commit_for_path(manifest_file)
        if git_path_is_tracked(manifest_file)
        else None
    )
    freeze_commit = (
        git_latest_commit_for_path(freeze_file)
        if git_path_is_tracked(freeze_file)
        else None
    )
    runner_commit = (
        git_latest_commit_for_path(current_runner)
        if current_runner.is_file() and git_path_is_tracked(current_runner)
        else None
    )
    runner_test_commit = (
        git_latest_commit_for_path(current_test)
        if current_test.is_file() and git_path_is_tracked(current_test)
        else None
    )

    preflight_commit = manifest["preflight_evidence"][
        "resource_preflight_source_git_commit"
    ]
    builder_commit = manifest["git_execution_binding"][
        "manifest_builder_source_git_commit"
    ]
    outputs = planned_output_paths(manifest)

    gates = {
        "working_tree_clean": git_is_clean(),
        "manifest_is_committed": manifest_commit is not None,
        "manifest_freeze_is_committed": freeze_commit is not None,
        "manifest_and_freeze_same_commit":
            manifest_commit is not None and manifest_commit == freeze_commit,
        "manifest_commit_is_ancestor":
            manifest_commit is not None
            and git_is_ancestor(manifest_commit, head),
        "manifest_builder_commit_is_ancestor":
            git_is_ancestor(builder_commit, head),
        "preflight_source_is_ancestor":
            git_is_ancestor(preflight_commit, head),
        "runner_exists": current_runner.is_file(),
        "runner_is_committed": runner_commit is not None,
        "runner_test_exists": current_test.is_file(),
        "runner_test_is_committed": runner_test_commit is not None,
        "runner_commit_equals_training_start_head": runner_commit == head,
        "runner_test_commit_equals_training_start_head":
            runner_test_commit == head,
        "shared_runtime_exact":
            sha256_file(shared_runtime_path())
            == EXPECTED_SHARED_RUNTIME_SHA256,
        "adapter_output_absent": not outputs["adapter"].exists(),
        "report_output_absent": not outputs["report"].exists(),
        "receipt_output_absent": not outputs["receipt"].exists(),
    }

    return {
        "authorized": all(gates.values()),
        "gates": gates,
        "training_start_head": head,
        "manifest_commit": manifest_commit,
        "manifest_freeze_commit": freeze_commit,
        "runner_commit": runner_commit,
        "runner_test_commit": runner_test_commit,
        "runner_blob_sha256":
            git_blob_sha256(ref=head, path=current_runner)
            if runner_commit == head
            else None,
        "runner_test_blob_sha256":
            git_blob_sha256(ref=head, path=current_test)
            if runner_test_commit == head
            else None,
        "output_paths": outputs,
        "static": static,
    }


def require_runtime_authorization() -> dict[str, Any]:
    authorization = collect_runtime_authorization()
    if authorization["authorized"]:
        return authorization
    failed = [
        name
        for name, value in authorization["gates"].items()
        if not value
    ]
    raise RuntimeError(
        "Training runtime authorization FAILED. "
        "optimizer.step() remains forbidden.\nFailed gates:\n  - "
        + "\n  - ".join(failed)
    )


# ---------------------------------------------------------------------------
# Output evidence
# ---------------------------------------------------------------------------


def adapter_bundle_metadata(
    *, adapter_path: Path, safe_open_function: Any
) -> dict[str, Any]:
    if not adapter_path.is_dir():
        raise RuntimeError("Adapter directory missing.")

    discovered = sorted(
        path for path in adapter_path.rglob("*") if path.is_file()
    )
    names = {
        path.relative_to(adapter_path).as_posix()
        for path in discovered
    }
    if "adapter_config.json" not in names:
        raise RuntimeError("adapter_config.json missing.")
    if "adapter_model.safetensors" not in names:
        raise RuntimeError("adapter_model.safetensors missing.")
    if names & {"model.safetensors", "pytorch_model.bin", "model.bin"}:
        raise RuntimeError("Full-model artifact detected.")

    files = [
        {
            "relative_path": path.relative_to(adapter_path).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in discovered
    ]

    with safe_open_function(
        str(adapter_path / "adapter_model.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        tensor_keys = sorted(handle.keys())

    if len(tensor_keys) != EXPECTED_TRAINABLE_TENSOR_COUNT:
        raise RuntimeError(
            f"Unexpected adapter tensor count: {len(tensor_keys)}"
        )

    bundle_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {"files": files, "safetensor_keys": tensor_keys}
        )
    ).hexdigest()

    return {
        "file_count": len(files),
        "files": files,
        "safetensor_tensor_count": len(tensor_keys),
        "bundle_sha256": bundle_sha256,
    }


def remove_path_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def publish_outputs_atomically(
    *,
    model: Any,
    report: Mapping[str, Any],
    receipt_builder: Any,
    output_paths: Mapping[str, Path],
    safe_open_function: Any,
    training_start_head: str,
) -> tuple[dict[str, Any], str, str]:
    adapter_path = output_paths["adapter"]
    report_path = output_paths["report"]
    receipt_path = output_paths["receipt"]

    for path in output_paths.values():
        if path.exists():
            raise RuntimeError(f"Official output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    suffix = training_start_head[:12]
    temp_adapter = adapter_path.with_name(
        f".{adapter_path.name}.tmp-{suffix}"
    )
    temp_report = report_path.with_name(
        f".{report_path.name}.tmp-{suffix}"
    )
    temp_receipt = receipt_path.with_name(
        f".{receipt_path.name}.tmp-{suffix}"
    )

    for path in (temp_adapter, temp_report, temp_receipt):
        if path.exists():
            raise RuntimeError(f"Temporary-output collision: {path}")

    published: list[Path] = []

    try:
        model.save_pretrained(temp_adapter, safe_serialization=True)
        adapter_metadata = adapter_bundle_metadata(
            adapter_path=temp_adapter,
            safe_open_function=safe_open_function,
        )

        report_bytes = canonical_json_bytes(report)
        report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        receipt = receipt_builder(
            adapter_metadata=adapter_metadata,
            report_sha256=report_sha256,
        )
        receipt_bytes = canonical_json_bytes(receipt)
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()

        for path, payload in (
            (temp_report, report_bytes),
            (temp_receipt, receipt_bytes),
        ):
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

        if sha256_file(temp_report) != report_sha256:
            raise RuntimeError("Temporary report SHA mismatch.")
        if sha256_file(temp_receipt) != receipt_sha256:
            raise RuntimeError("Temporary receipt SHA mismatch.")

        for source, target in (
            (temp_adapter, adapter_path),
            (temp_report, report_path),
            (temp_receipt, receipt_path),
        ):
            os.replace(source, target)
            published.append(target)

        if sha256_file(report_path) != report_sha256:
            raise RuntimeError("Published report SHA mismatch.")
        if sha256_file(receipt_path) != receipt_sha256:
            raise RuntimeError("Published receipt SHA mismatch.")
        if (
            adapter_bundle_metadata(
                adapter_path=adapter_path,
                safe_open_function=safe_open_function,
            )
            != adapter_metadata
        ):
            raise RuntimeError("Published adapter metadata changed.")

        return adapter_metadata, report_sha256, receipt_sha256

    except BaseException:
        for path in reversed(published):
            remove_path_if_exists(path)
        for path in (temp_adapter, temp_report, temp_receipt):
            remove_path_if_exists(path)
        raise


# ---------------------------------------------------------------------------
# Static / authorization modes
# ---------------------------------------------------------------------------


def print_static_validation() -> None:
    validate_static_contract()
    groups = accumulation_groups_v0_4(
        order=list(range(EXPECTED_EXAMPLES)),
        accumulation_steps=EXPECTED_ACCUMULATION_STEPS,
    )
    validate_v0_4_group_plan(groups)
    if supervised_token_loss_scales([2, 3, 5]) != [0.2, 0.3, 0.5]:
        raise RuntimeError("Supervised-token scaling changed.")

    print("=== DATALENS QLORA v0.4 TRAINING RUNNER STATIC VALIDATION ===")
    print("Manifest exact: PASS")
    print("Freeze exact: PASS")
    print("Shared runtime exact: PASS")
    print("Three GPU probes bound: PASS")
    print("230 = 28 x 8 + 6: PASS")
    print("29 optimizer steps / epoch: PASS")
    print("58 optimizer steps total: PASS")
    print("460 micro-batches total: PASS")
    print("Cross-epoch accumulation=False: PASS")
    print("Discarded presentations=0: PASS")
    print("Supervised-token weighted objective: PASS")
    print("Partial denominator=actual group tokens: PASS")
    print("PagedAdamW8bit controls manifest-owned: PASS")
    print("warmup_steps=2: PASS")
    print("Heavy ML imported: False")
    print("optimizer.step() executed: False")
    print("Training executed: False")
    print("Protected evaluation executed: False")
    print("DATALENS QLORA v0.4 TRAINING RUNNER STATIC VALIDATION: PASS")


def print_runtime_authorization() -> None:
    authorization = collect_runtime_authorization()
    print("=== DATALENS QLORA v0.4 TRAINING RUNTIME AUTHORIZATION ===")
    print(f"Authorized: {authorization['authorized']}")
    print(f"Training-start HEAD: {authorization['training_start_head']}")
    for name, value in authorization["gates"].items():
        print(f"  {name}: {value}")
    print("Heavy ML imported: False")
    print("GPU requested: False")
    print("Optimizer created: False")
    print("optimizer.step(): False")
    print("Training executed: False")

    if not authorization["authorized"]:
        failed = [
            name
            for name, value in authorization["gates"].items()
            if not value
        ]
        raise RuntimeError(
            "Runtime authorization failed. Failed gates:\n  - "
            + "\n  - ".join(failed)
        )

    print("DATALENS QLORA v0.4 TRAINING RUNTIME AUTHORIZATION: PASS")


# ---------------------------------------------------------------------------
# Official execution
# ---------------------------------------------------------------------------


def execute_training() -> None:
    # Trust boundary: nothing from the ML stack may load before this succeeds.
    authorization = require_runtime_authorization()

    import bitsandbytes as bnb
    import numpy as np
    import torch

    from safetensors import safe_open
    from transformers import get_cosine_schedule_with_warmup

    from app.adaptation import qlora_runtime_v0_4 as runtime

    manifest = authorization["static"]["manifest"]
    training = manifest["training"]
    seed = int(training["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official v0.4 training.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")

    runtime_authority = runtime.validate_static_authority(
        repository_root_value=api_root()
    )
    tokenizer = runtime.load_pinned_tokenizer(authority=runtime_authority)
    prepared = runtime.prepare_training_dataset(
        authority=runtime_authority,
        tokenizer=tokenizer,
    )

    if len(prepared.examples) != EXPECTED_EXAMPLES:
        raise RuntimeError("Prepared example count changed.")
    if prepared.total_supervised_tokens != EXPECTED_SUPERVISED_TOKENS_PER_EPOCH:
        raise RuntimeError("Prepared supervised-token total changed.")
    if prepared.max_example_tokens > training["max_sequence_length"]:
        raise RuntimeError("Prepared dataset exceeds sequence limit.")

    prepared_model = runtime.prepare_qlora_model(
        authority=runtime_authority
    )
    if len(prepared_model.target_modules) != EXPECTED_TARGET_MODULE_COUNT:
        raise RuntimeError("Target module count changed.")
    if (
        prepared_model.trainable_parameter_count
        != EXPECTED_TRAINABLE_PARAMETER_COUNT
    ):
        raise RuntimeError("Trainable parameter count changed.")
    if (
        prepared_model.trainable_tensor_count
        != EXPECTED_TRAINABLE_TENSOR_COUNT
    ):
        raise RuntimeError("Trainable tensor count changed.")

    model = prepared_model.model
    model.train()

    optimizer = bnb.optim.PagedAdamW8bit(
        prepared_model.trainable_parameters,
        lr=training["learning_rate"],
        betas=tuple(training["optimizer_betas"]),
        eps=training["optimizer_eps"],
        weight_decay=training["weight_decay"],
        amsgrad=training["optimizer_amsgrad"],
        min_8bit_size=training["optimizer_min_8bit_size"],
    )
    if optimizer.__class__.__name__ != "PagedAdamW8bit":
        raise RuntimeError("Unexpected optimizer class.")
    if getattr(optimizer, "is_paged", None) is not True:
        raise RuntimeError("Optimizer is not paged.")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training["warmup_steps"],
        num_training_steps=training["total_optimizer_steps"],
    )

    started_at = utc_now()
    started_monotonic = time.monotonic()
    fingerprint_before = runtime.trainable_parameter_fingerprint(
        model=model,
        torch_module=torch,
    )

    global_micro_batch = 0
    global_optimizer_step = 0
    scheduler_steps = 0
    supervised_tokens_seen = 0
    step_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []

    optimizer.zero_grad(set_to_none=True)

    for zero_based_epoch in range(training["epochs"]):
        order = deterministic_epoch_order(
            example_count=EXPECTED_EXAMPLES,
            seed=seed,
            zero_based_epoch=zero_based_epoch,
        )
        groups = accumulation_groups_v0_4(
            order=order,
            accumulation_steps=training["gradient_accumulation_steps"],
        )
        validate_v0_4_group_plan(groups)

        epoch_micro_batches = 0
        epoch_optimizer_steps = 0
        epoch_supervised_tokens = 0

        for group_index, group in enumerate(groups, start=1):
            token_counts = [
                prepared.examples[index].supervised_token_count
                for index in group
            ]
            scales = supervised_token_loss_scales(token_counts)
            group_supervised_tokens = sum(token_counts)
            micro_losses: list[float] = []

            optimizer.zero_grad(set_to_none=True)

            for example_index, loss_scale in zip(group, scales):
                example = prepared.examples[example_index]
                batch = runtime.tensor_batch_from_example(
                    example=example,
                    tokenizer=tokenizer,
                    torch_module=torch,
                    device=device,
                )
                outputs = model(**batch)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model returned no training loss.")
                if not torch.isfinite(loss).all():
                    raise RuntimeError("Non-finite micro-batch loss.")

                micro_losses.append(
                    float(loss.detach().float().cpu().item())
                )
                (loss * loss_scale).backward()

                global_micro_batch += 1
                epoch_micro_batches += 1
                supervised_tokens_seen += example.supervised_token_count
                epoch_supervised_tokens += example.supervised_token_count

            gradient_stats = runtime.gradient_statistics(
                model=model,
                torch_module=torch,
            )
            gradient_tensor_count = int(
                gradient_stats["gradient_tensor_count"]
            )
            nonfinite_gradient_count = int(
                gradient_stats["nonfinite_gradient_count"]
            )
            gradient_norm = float(gradient_stats["gradient_norm"])

            if gradient_tensor_count != EXPECTED_TRAINABLE_TENSOR_COUNT:
                raise RuntimeError("Gradient tensor count changed.")
            if nonfinite_gradient_count != 0:
                raise RuntimeError("Non-finite LoRA gradients detected.")
            if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
                raise RuntimeError("Invalid LoRA gradient norm.")

            group_loss = weighted_group_loss(
                losses=micro_losses,
                token_counts=token_counts,
            )
            if not math.isfinite(group_loss):
                raise RuntimeError("Non-finite weighted group loss.")

            lr_before = float(optimizer.param_groups[0]["lr"])

            # The only official model-update site in this runner.
            optimizer.step()
            global_optimizer_step += 1
            epoch_optimizer_steps += 1

            scheduler.step()
            scheduler_steps += 1
            lr_after = float(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad(set_to_none=True)

            step_records.append(
                {
                    "epoch": zero_based_epoch + 1,
                    "optimizer_step_in_epoch": epoch_optimizer_steps,
                    "global_optimizer_step": global_optimizer_step,
                    "group_index": group_index,
                    "group_size": len(group),
                    "partial_group":
                        len(group)
                        < training["gradient_accumulation_steps"],
                    "group_supervised_tokens": group_supervised_tokens,
                    "micro_losses": micro_losses,
                    "weighted_group_loss": group_loss,
                    "gradient_tensor_count": gradient_tensor_count,
                    "nonfinite_gradient_count": nonfinite_gradient_count,
                    "gradient_norm": gradient_norm,
                    "learning_rate_before_optimizer_step": lr_before,
                    "learning_rate_after_scheduler_step": lr_after,
                }
            )

        if epoch_micro_batches != EXPECTED_EXAMPLES:
            raise RuntimeError("Epoch micro-batch count changed.")
        if epoch_optimizer_steps != EXPECTED_GROUPS_PER_EPOCH:
            raise RuntimeError("Epoch optimizer-step count changed.")
        if epoch_supervised_tokens != EXPECTED_SUPERVISED_TOKENS_PER_EPOCH:
            raise RuntimeError("Epoch supervised-token count changed.")

        epoch_records.append(
            {
                "epoch": zero_based_epoch + 1,
                "micro_batches": epoch_micro_batches,
                "optimizer_steps": epoch_optimizer_steps,
                "supervised_tokens": epoch_supervised_tokens,
                "terminal_group_size": len(groups[-1]),
            }
        )

    if global_micro_batch != EXPECTED_TOTAL_MICRO_BATCHES:
        raise RuntimeError("Total micro-batch count changed.")
    if global_optimizer_step != EXPECTED_TOTAL_OPTIMIZER_STEPS:
        raise RuntimeError("Total optimizer-step count changed.")
    if scheduler_steps != EXPECTED_TOTAL_OPTIMIZER_STEPS:
        raise RuntimeError("Total scheduler-step count changed.")
    if supervised_tokens_seen != EXPECTED_TOTAL_SUPERVISED_TOKENS:
        raise RuntimeError("Supervised-token presentations changed.")

    partial_records = [
        record for record in step_records if record["partial_group"]
    ]
    if len(partial_records) != EXPECTED_EPOCHS:
        raise RuntimeError("Expected one partial group per epoch.")
    if any(
        record["group_size"] != EXPECTED_PARTIAL_GROUP_SIZE
        for record in partial_records
    ):
        raise RuntimeError("Partial group size changed.")

    fingerprint_after = runtime.trainable_parameter_fingerprint(
        model=model,
        torch_module=torch,
    )
    weights_changed = fingerprint_before != fingerprint_after
    if not weights_changed:
        raise RuntimeError("LoRA weights did not change.")

    torch.cuda.synchronize()
    finished_at = utc_now()

    report = {
        "report_id":
            "training-report:datalens-semantic-qlora:v0.4:0001",
        "rule_version": REPORT_RULE_VERSION,
        "training_run_id": TRAINING_RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_freeze_sha256": EXPECTED_FREEZE_SHA256,
        "shared_runtime_sha256": EXPECTED_SHARED_RUNTIME_SHA256,
        "training_start_head": authorization["training_start_head"],
        "runner_blob_sha256": authorization["runner_blob_sha256"],
        "runner_test_blob_sha256":
            authorization["runner_test_blob_sha256"],
        "training_started_at": started_at,
        "training_finished_at": finished_at,
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "execution": {
            "epochs": EXPECTED_EPOCHS,
            "micro_batches_executed": global_micro_batch,
            "optimizer_steps_executed": global_optimizer_step,
            "scheduler_steps_executed": scheduler_steps,
            "supervised_tokens_seen": supervised_tokens_seen,
            "discarded_example_presentations": 0,
            "cross_epoch_accumulation": False,
            "partial_group_policy":
                "flush_partial_group_at_epoch_end",
            "partial_group_size": EXPECTED_PARTIAL_GROUP_SIZE,
            "partial_group_count": len(partial_records),
            "gradient_accumulation_weighting":
                "supervised_token_weighted",
            "training_loss_is_acceptance_evidence": False,
            "weights_changed": weights_changed,
            "fingerprint_before": fingerprint_before,
            "fingerprint_after": fingerprint_after,
        },
        "epochs": epoch_records,
        "optimizer_steps": step_records,
        "protected_evaluation": {
            "airport_loaded": False,
            "airport_evaluated": False,
            "final_acceptance_loaded": False,
            "final_acceptance_evaluated": False,
        },
        "passed": True,
    }

    def receipt_builder(
        *,
        adapter_metadata: Mapping[str, Any],
        report_sha256: str,
    ) -> dict[str, Any]:
        return {
            "receipt_id":
                "training-receipt:datalens-semantic-qlora:v0.4:0001",
            "rule_version": RECEIPT_RULE_VERSION,
            "training_run_id": TRAINING_RUN_ID,
            "experiment_id": EXPERIMENT_ID,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "manifest_freeze_sha256": EXPECTED_FREEZE_SHA256,
            "shared_runtime_sha256": EXPECTED_SHARED_RUNTIME_SHA256,
            "training_start_head": authorization["training_start_head"],
            "manifest_commit": authorization["manifest_commit"],
            "manifest_freeze_commit":
                authorization["manifest_freeze_commit"],
            "runner_commit": authorization["runner_commit"],
            "runner_test_commit": authorization["runner_test_commit"],
            "runner_blob_sha256": authorization["runner_blob_sha256"],
            "runner_test_blob_sha256":
                authorization["runner_test_blob_sha256"],
            "adapter": dict(adapter_metadata),
            "training_report_sha256": report_sha256,
            "optimizer_steps_executed": global_optimizer_step,
            "scheduler_steps_executed": scheduler_steps,
            "micro_batches_executed": global_micro_batch,
            "weights_changed": weights_changed,
            "training_loss_is_acceptance_evidence": False,
            "airport_evaluated": False,
            "final_acceptance_evaluated": False,
            "created_at": utc_now(),
            "passed": True,
        }

    adapter_metadata, report_sha256, receipt_sha256 = (
        publish_outputs_atomically(
            model=model,
            report=report,
            receipt_builder=receipt_builder,
            output_paths=authorization["output_paths"],
            safe_open_function=safe_open,
            training_start_head=authorization["training_start_head"],
        )
    )

    print("=== DATALENS QLORA v0.4 OFFICIAL TRAINING RESULT ===")
    print(f"Training-start HEAD: {authorization['training_start_head']}")
    print(f"Micro-batches executed: {global_micro_batch}")
    print(f"optimizer.step() executed: {global_optimizer_step}")
    print(f"scheduler.step() executed: {scheduler_steps}")
    print(f"Supervised tokens seen: {supervised_tokens_seen}")
    print(f"Partial groups executed: {len(partial_records)}")
    print(f"LoRA weights changed: {weights_changed}")
    print(f"Adapter bundle SHA256: {adapter_metadata['bundle_sha256']}")
    print(f"Training report SHA256: {report_sha256}")
    print(f"Training receipt SHA256: {receipt_sha256}")
    print("Airport evaluated: False")
    print("Final Acceptance evaluated: False")
    print("Training loss used as acceptance evidence: False")
    print("DATALENS QLORA v0.4 OFFICIAL TRAINING: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DataLens controlled QLoRA training runner v0.4"
    )
    parser.add_argument(
        "mode",
        choices=("validate-static", "authorize-only", "execute"),
    )
    args = parser.parse_args()

    if args.mode == "validate-static":
        print_static_validation()
        return
    if args.mode == "authorize-only":
        print_runtime_authorization()
        return
    if args.mode == "execute":
        execute_training()
        return
    raise RuntimeError("Unsupported runner mode.")


if __name__ == "__main__":
    main()

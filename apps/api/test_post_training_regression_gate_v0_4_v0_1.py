from __future__ import annotations

import ast

from pathlib import Path


print(
    "=== DATALENS QLORA v0.4 S3 POST-TRAINING GATE RUNNER STATIC TEST v0.1 ==="
)


ROOT = Path(
    __file__
).resolve().parent


RUNNER_PATH = (
    ROOT
    /
    "app"
    /
    "adaptation"
    /
    "post_training_regression_gate_v0_4.py"
)


source = RUNNER_PATH.read_text(
    encoding="utf-8-sig"
)


tree = ast.parse(
    source
)


def expect(
    condition,
    message,
):
    if not condition:
        raise AssertionError(
            message
        )


# ============================================================
# FILE / VERSION SURFACE
# ============================================================


expect(
    "qlora_v0.4_s3_post_training_gate_runner_v0.1"
    in source,
    "Runner rule version missing.",
)

expect(
    "qlora_v0.4_s3_post_training_gate_manifest_v0.1"
    in source,
    "Manifest rule version missing.",
)

expect(
    "qlora_v0.4_s3_post_training_gate_freeze_v0.1"
    in source,
    "Freeze rule version missing.",
)

expect(
    "semantic-s3-regression-249"
    in source,
    "Frozen S3 baseline binding missing.",
)

expect(
    all(
        fragment in source
        for fragment in (
            "fdb5510e9426b857aa9e52feb4d3282f",
            "367e10af1d8ae4335c727673506960ac",
        )
    ),
    "Frozen S3 baseline SHA binding missing.",
)

expect(
    "f5ec307f134eadbfc70f282a840fef3e7d5987a4"
    in source,
    "Official training evidence commit binding missing.",
)

expect(
    'EXPECTED_MODEL = "gemma3:4b"'
    in source,
    "Production S3 model binding missing.",
)


# ============================================================
# NO ADAPTER INJECTION
# ============================================================


for forbidden in (
    "PeftModel",
    "AutoModelForCausalLM",
    "BitsAndBytesConfig",
    "prepare_qlora_model",
    "reasoning_evaluation_runner",
):
    expect(
        forbidden
        not in source,
        (
            "S3 system gate must not inject "
            f"adaptation runtime: {forbidden}"
        ),
    )


# ============================================================
# DEFERRED HEAVY IMPORTS
# ============================================================


top_level_imports = []


for node in tree.body:
    if isinstance(
        node,
        ast.Import,
    ):
        top_level_imports.extend(
            alias.name
            for alias in node.names
        )

    elif isinstance(
        node,
        ast.ImportFrom,
    ):
        top_level_imports.append(
            node.module
            or
            ""
        )


for forbidden_prefix in (
    "pandas",
    "app.",
    "torch",
    "transformers",
    "peft",
):
    expect(
        not any(
            value.startswith(
                forbidden_prefix
            )
            for value in top_level_imports
        ),
        (
            "Heavy/application dependency imported "
            "before runtime authorization: "
            f"{forbidden_prefix}"
        ),
    )


# ============================================================
# EXACT EVALUATION WIRING
# ============================================================


expect(
    "run_semantic_benchmark_registry"
    in source,
    "Existing semantic registry runner is not reused.",
)

expect(
    'split="regression"'
    in "".join(
        source.split()
    ),
    "Regression split is not explicit.",
)

for benchmark in (
    "DWFA_BENCHMARK_ID",
    "ECOMMERCE_BENCHMARK_ID",
    "MANUFACTURING_BENCHMARK_ID",
    "LOGISTICS_BENCHMARK_ID",
    "CLOUD_BENCHMARK_ID",
    "ELECTRIC_MOBILITY_BENCHMARK_ID",
):
    expect(
        benchmark
        in source,
        (
            "Expected S3 regression benchmark "
            f"missing: {benchmark}"
        ),
    )


for filename in (
    "BasicAndSafelyManagedDrinkingWaterServices.csv",
    "MortalityRateAttributedToWater.csv",
    "PoliticalStability.csv",
    "Population.csv",
    "RegionCountry.csv",
):
    expect(
        filename
        in source,
        (
            "DWFA source binding missing: "
            f"{filename}"
        ),
    )


# ============================================================
# GATE SEMANTICS
# ============================================================


for requirement in (
    '"normalized_failure_count"',
    '"regression_gate_passed"',
    '"safety_gate_passed"',
    '"false_positive_count"',
    '"unclassified_count"',
    '"normalized_micro_accuracy"',
    '"normalized_macro_accuracy"',
):
    expect(
        requirement
        in source,
        (
            "Required gate field missing: "
            f"{requirement}"
        ),
    )


expect(
    "EXPECTED_ASSERTION_COUNT = 249"
    in source,
    "249-assertion authority missing.",
)

expect(
    "EXPECTED_SUITE_COUNT = 6"
    in source,
    "Six-suite authority missing.",
)


# ============================================================
# OUTPUT PROTECTION
# ============================================================


expect(
    "validate_output_absence"
    in source,
    "Output overwrite protection missing.",
)

expect(
    "atomic_write_bytes"
    in source,
    "Atomic output publication missing.",
)


# ============================================================
# CURRENT IMMUTABLE EVIDENCE
# ============================================================


import importlib.util
import sys


MODULE_NAME = (
    "datalens_post_training_regression_gate_v0_4_static"
)


expect(
    "app.adaptation" not in sys.modules,
    (
        "app.adaptation was imported before the "
        "isolated runner load."
    ),
)


spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    RUNNER_PATH,
)


expect(
    spec is not None,
    "Could not create isolated runner module spec.",
)

expect(
    spec.loader is not None,
    "Isolated runner module loader is missing.",
)


runner_module = importlib.util.module_from_spec(
    spec
)


spec.loader.exec_module(
    runner_module
)


expect(
    "app.adaptation" not in sys.modules,
    (
        "Isolated runner load unexpectedly imported "
        "the app.adaptation package."
    ),
)


for forbidden_module in (
    "huggingface_hub",
    "torch",
    "transformers",
    "peft",
):
    expect(
        forbidden_module not in sys.modules,
        (
            "Static runner import unexpectedly loaded "
            f"{forbidden_module}."
        ),
    )


BASELINE_PATH = (
    runner_module.BASELINE_PATH
)

BASELINE_SHA256 = (
    runner_module.BASELINE_SHA256
)

REPORT_PATH = (
    runner_module.REPORT_PATH
)

RECEIPT_PATH = (
    runner_module.RECEIPT_PATH
)

TRAINING_RECEIPT_PATH = (
    runner_module.TRAINING_RECEIPT_PATH
)

TRAINING_RECEIPT_SHA256 = (
    runner_module.TRAINING_RECEIPT_SHA256
)

TRAINING_REPORT_PATH = (
    runner_module.TRAINING_REPORT_PATH
)

TRAINING_REPORT_SHA256 = (
    runner_module.TRAINING_REPORT_SHA256
)

evaluate_gate_payload = (
    runner_module.evaluate_gate_payload
)

sha256_file = (
    runner_module.sha256_file
)


expect(
    BASELINE_PATH.is_file(),
    "Frozen baseline missing.",
)

expect(
    sha256_file(
        BASELINE_PATH
    )
    ==
    BASELINE_SHA256,
    "Frozen baseline SHA mismatch.",
)

expect(
    TRAINING_RECEIPT_PATH.is_file(),
    "Training receipt missing.",
)

expect(
    sha256_file(
        TRAINING_RECEIPT_PATH
    )
    ==
    TRAINING_RECEIPT_SHA256,
    "Training receipt SHA mismatch.",
)

expect(
    TRAINING_REPORT_PATH.is_file(),
    "Training report missing.",
)

expect(
    sha256_file(
        TRAINING_REPORT_PATH
    )
    ==
    TRAINING_REPORT_SHA256,
    "Training report SHA mismatch.",
)

expect(
    not REPORT_PATH.exists(),
    "S3 post-training report already exists.",
)

expect(
    not RECEIPT_PATH.exists(),
    "S3 post-training receipt already exists.",
)


# ============================================================
# PURE GATE TEST
# ============================================================


baseline = {
    "result": {
        "normalized_micro_accuracy":
            1.0,

        "normalized_macro_accuracy":
            1.0,

        "normalized_safety_decisions": {
            "false_positive_count":
                0,

            "unclassified_count":
                0,
        },
    },
}


candidate = {
    "suite_count":
        6,

    "domain_count":
        6,

    "normalized_assertion_count":
        249,

    "normalized_failure_count":
        0,

    "regression_gate_passed":
        True,

    "safety_gate_passed":
        True,

    "normalized_micro_accuracy":
        1.0,

    "normalized_macro_accuracy":
        1.0,

    "normalized_safety_decisions": {
        "false_positive_count":
            0,

        "unclassified_count":
            0,
    },
}


passed = evaluate_gate_payload(
    result_payload=
        candidate,

    baseline=
        baseline,
)


expect(
    passed[
        "passed"
    ]
    is True,
    "Valid S3 gate candidate did not pass.",
)


dangerous = dict(
    candidate
)

dangerous[
    "normalized_safety_decisions"
] = {
    "false_positive_count":
        1,

    "unclassified_count":
        0,
}


failed = evaluate_gate_payload(
    result_payload=
        dangerous,

    baseline=
        baseline,
)


expect(
    failed[
        "passed"
    ]
    is False,
    "Dangerous FP increase was not blocked.",
)

expect(
    failed[
        "dangerous_false_positive_delta"
    ]
    ==
    1,
    "Dangerous FP delta is incorrect.",
)


print(
    "Frozen S3 baseline binding: PASS"
)

print(
    "Official training evidence binding: PASS"
)

print(
    "No PEFT/HF adapter injection: PASS"
)

print(
    "Application imports deferred: PASS"
)

print(
    "Six regression-suite wiring: PASS"
)

print(
    "249-assertion gate authority: PASS"
)

print(
    "Dangerous FP increase blocked: PASS"
)

print(
    "Output overwrite protection: PASS"
)

print(
    "Inference executed: False"
)

print(
    "DATALENS QLORA v0.4 S3 POST-TRAINING GATE RUNNER STATIC TEST v0.1: PASS"
)

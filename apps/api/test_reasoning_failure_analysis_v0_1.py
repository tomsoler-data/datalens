from __future__ import annotations

import ast
import hashlib

from pathlib import Path


from app.adaptation.reasoning_failure_analysis import (
    ARTIFACT_PATH,
    FAILURE_ANALYSIS_RULE_VERSION,
    REPORT_PATH,
    SOURCE_REPORT_SHA256,
    analyze_report,
    load_json_object,
    sha256_file,
    validate_artifact,
)


print(
    "=== DATALENS QLORA FAILURE ANALYSIS ARTIFACT v0.1 ==="
)

print()


assert (
    FAILURE_ANALYSIS_RULE_VERSION
    ==
    "qlora_reasoning_failure_analysis_v0.1"
)


assert (
    sha256_file(
        REPORT_PATH
    )
    ==
    SOURCE_REPORT_SHA256
)


report = load_json_object(
    REPORT_PATH
)


artifact = load_json_object(
    ARTIFACT_PATH
)


validate_artifact(
    artifact=
        artifact,

    report=
        report,
)


print(
    "Source report binding: PASS"
)

print(
    "Deterministic recomputation: PASS"
)


analysis = analyze_report(
    report=
        report,
)


observed = analysis[
    "observed_result"
]


assert (
    observed[
        "base_correct_count"
    ]
    ==
    6
)


assert (
    observed[
        "adapted_correct_count"
    ]
    ==
    3
)


assert (
    observed[
        "accuracy_delta"
    ]
    ==
    -0.157894
)


assert (
    observed[
        "macro_accuracy_delta"
    ]
    ==
    -0.076923
)


assert (
    observed[
        "adapted_only_correct"
    ]
    ==
    0
)


assert (
    observed[
        "base_only_correct"
    ]
    ==
    3
)


assert (
    observed[
        "preregistered_signal"
    ]
    ==
    "negative_signal"
)


print(
    "Observed negative signal: PASS"
)


margins = analysis[
    "expected_margin_analysis"
]


assert (
    margins[
        "base_mean"
    ]
    ==
    -0.31935363
)


assert (
    margins[
        "adapted_mean"
    ]
    ==
    -0.12632614
)


assert (
    margins[
        "mean_delta"
    ]
    ==
    0.19302749
)


assert (
    margins[
        "improved_count"
    ]
    ==
    11
)


assert (
    margins[
        "worsened_count"
    ]
    ==
    8
)


assert (
    margins[
        "negative_or_zero_to_positive_count"
    ]
    ==
    0
)


assert (
    margins[
        "positive_to_non_positive_count"
    ]
    ==
    3
)


print(
    "Margin movement evidence: PASS"
)


same_metric = analysis[
    "by_expected_relation"
][
    "same_metric_different_state"
]


assert (
    same_metric[
        "case_count"
    ]
    ==
    5
)


assert (
    same_metric[
        "base_correct"
    ]
    ==
    0
)


assert (
    same_metric[
        "adapted_correct"
    ]
    ==
    0
)


assert (
    same_metric[
        "margin_improved_count"
    ]
    ==
    4
)


assert (
    same_metric[
        "mean_margin_delta"
    ]
    ==
    0.469801
)


print(
    "same_metric learning-direction evidence: PASS"
)


relative = analysis[
    "relative_candidate_preference_shift_mean"
]


assert (
    relative[
        "same_metric_different_state"
    ]
    ==
    0.32178972
)


assert (
    relative[
        "same_process_different_stage"
    ]
    ==
    -0.36505324
)


assert (
    relative[
        "related_distinct_metric"
    ]
    ==
    0.04326352
)


print(
    "Relative preference shift evidence: PASS"
)


expected_shift = analysis[
    "expected_answer_score_shift"
]


assert (
    expected_shift[
        "increased_count"
    ]
    ==
    19
)


assert (
    expected_shift[
        "decreased_count"
    ]
    ==
    0
)


assert (
    expected_shift[
        "mean_delta"
    ]
    ==
    7.29228282
)


print(
    "Absolute expected-score shift evidence: PASS"
)


changes = analysis[
    "argmax_changes"
]


assert len(
    changes
) == 3


assert all(
    item[
        "base_correct"
    ]
    is True
    and
    item[
        "adapted_correct"
    ]
    is False

    for item
    in changes
)


print(
    "All argmax changes are regressions: PASS"
)


assert (
    artifact[
        "conclusions"
    ][
        "diagnostic_gate_passed"
    ]
    is False
)


assert (
    artifact[
        "conclusions"
    ][
        "final_acceptance_eligible"
    ]
    is False
)


assert (
    artifact[
        "conclusions"
    ][
        "causal_root_cause_established"
    ]
    is False
)


assert (
    artifact[
        "conclusions"
    ][
        "learning_signal_observed"
    ]
    is True
)


print(
    "Interpretation boundaries: PASS"
)


assert (
    artifact[
        "evaluation_policy"
    ][
        "hotel_reusable_as_new_independent_holdout"
    ]
    is False
)


assert (
    artifact[
        "evaluation_policy"
    ][
        "new_independent_holdout_required_before_next_training"
    ]
    is True
)


assert (
    artifact[
        "evaluation_policy"
    ][
        "final_acceptance_remains_closed"
    ]
    is True
)


print(
    "Future evaluation policy: PASS"
)


source_path = (
    Path(__file__)
    .resolve()
    .parent
    / "app"
    / "adaptation"
    / "reasoning_failure_analysis.py"
)


source = source_path.read_text(
    encoding="utf-8-sig"
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


for node in ast.walk(
    tree
):
    if isinstance(
        node,
        ast.Import,
    ):
        roots = {
            alias.name.split(
                "."
            )[0]

            for alias
            in node.names
        }

        assert not (
            roots
            &
            heavy_modules
        )

    elif isinstance(
        node,
        ast.ImportFrom,
    ):
        module = (
            node.module
            or
            ""
        )

        root = module.split(
            "."
        )[0]

        assert (
            root
            not in
            heavy_modules
        )


print(
    "Offline-only implementation: PASS"
)


safety = artifact[
    "safety"
]


assert all(
    safety[
        key
    ]
    is False

    for key
    in (
        "adapter_loaded",
        "benchmark_modified",
        "cuda_requested",
        "final_acceptance_evaluated",
        "final_acceptance_loaded",
        "free_generation_used",
        "inference_executed",
        "llm_judge_used",
        "model_loaded",
        "new_evaluation_executed",
        "training_executed",
    )
)


print()

print(
    "SAFETY"
)

print(
    "  Model loaded: False"
)

print(
    "  Adapter loaded: False"
)

print(
    "  CUDA requested: False"
)

print(
    "  Inference executed: False"
)

print(
    "  New evaluation executed: False"
)

print(
    "  Training executed: False"
)

print(
    "  Benchmark modified: False"
)

print(
    "  Final Acceptance loaded: False"
)

print(
    "  Final Acceptance evaluated: False"
)


print()

print(
    (
        "Artifact SHA256: "
        f"{sha256_file(ARTIFACT_PATH)}"
    )
)


print()

print(
    "DATALENS QLORA FAILURE ANALYSIS ARTIFACT v0.1: PASS"
)

from __future__ import annotations

import ast
import hashlib
import json
import math

from collections import Counter
from pathlib import Path


from app.adaptation.reasoning_benchmark import (
    ADAPTED_REASONING_BENCHMARK_RULE_VERSION,
    ADAPTED_REASONING_CASE_ARTIFACT_RULE_VERSION,
    ADAPTED_REASONING_SCORING_RULE_VERSION,
    ALLOWED_RELATIONS,
    BENCHMARK_ID,
    BENCHMARK_VERSION,
    CANDIDATE_SURFACES,
    EXPECTED_CASE_COUNT,
    EXPECTED_LABEL_COUNTS,
    AdaptedReasoningCase,
    candidate_surface_sha256,
    classification_accuracy,
    label_distribution,
    load_frozen_reasoning_cases,
    prediction_is_correct,
    prompt_template_sha256,
    render_reasoning_prompt,
    select_relation_from_scores,
)


print(
    "=== DATALENS ADAPTED REASONING BENCHMARK v0.1 ==="
)

print()


ROOT = Path(__file__).resolve().parent


CASES_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / "datalens_semantic_qlora_v0.3_"
      "reasoning_benchmark_v0.1_cases.json"
)


FREEZE_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / "datalens_semantic_qlora_v0.3_"
      "reasoning_benchmark_v0.1_freeze.json"
)


MODULE_PATH = (
    ROOT
    / "app"
    / "adaptation"
    / "reasoning_benchmark.py"
)


HOTEL_SOURCE_PATH = (
    ROOT
    / "app"
    / "evaluation"
    / "benchmarks"
    / "hotel_operations_holdout.py"
)


HOTEL_FREEZE_PATH = (
    ROOT
    / "artifacts"
    / "evaluation"
    / "holdouts"
    / "hotel_operations_holdout_v0.1_freeze.json"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def extract_hotel_cases(
) -> list[
    dict[
        str,
        str,
    ]
]:
    source = HOTEL_SOURCE_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
    )

    calls = []

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if not (
            isinstance(
                node.func,
                ast.Name,
            )
            and
            node.func.id
            ==
            "HotelOperationsRelationCase"
        ):
            continue

        values = {}

        for keyword in node.keywords:
            if keyword.arg is None:
                continue

            values[
                keyword.arg
            ] = ast.literal_eval(
                keyword.value
            )

        calls.append(
            (
                node.lineno,
                values,
            )
        )

    calls.sort(
        key=lambda item:
            item[
                0
            ]
    )

    return [
        {
            "case_id":
                values[
                    "case_id"
                ],

            "expected_relation":
                values[
                    "relation"
                ],

            "left_column":
                values[
                    "left_column"
                ],

            "right_column":
                values[
                    "right_column"
                ],
        }

        for _line, values
        in calls
    ]


cases = load_frozen_reasoning_cases(
    path=
        CASES_PATH,
)


assert (
    ADAPTED_REASONING_BENCHMARK_RULE_VERSION
    ==
    "adapted_semantic_reasoning_benchmark_v0.1"
)

assert (
    ADAPTED_REASONING_CASE_ARTIFACT_RULE_VERSION
    ==
    "adapted_semantic_reasoning_case_artifact_v0.1"
)

assert (
    ADAPTED_REASONING_SCORING_RULE_VERSION
    ==
    "adapted_semantic_reasoning_scoring_v0.1"
)


print(
    "Rule versions: PASS"
)


assert (
    BENCHMARK_ID
    ==
    "adaptation:semantic_reasoning:hotel:v0.1"
)

assert (
    BENCHMARK_VERSION
    ==
    "datalens_adapted_semantic_reasoning_hotel_v0.1"
)


print(
    "Benchmark identity: PASS"
)


assert len(
    cases
) == EXPECTED_CASE_COUNT


assert (
    label_distribution(
        cases
    )
    ==
    dict(
        EXPECTED_LABEL_COUNTS
    )
)


print(
    "Frozen 19-case set: PASS"
)

print(
    (
        "Label distribution: "
        f"{label_distribution(cases)}"
    )
)


source_cases = extract_hotel_cases()


artifact_payload = json.loads(
    CASES_PATH.read_text(
        encoding="utf-8-sig"
    )
)


assert (
    artifact_payload[
        "cases"
    ]
    ==
    source_cases
)


print(
    "Frozen cases match Hotel source: PASS"
)


hotel_freeze = json.loads(
    HOTEL_FREEZE_PATH.read_text(
        encoding="utf-8-sig"
    )
)


assert (
    hotel_freeze[
        "holdout_frozen"
    ]
    is True
)

assert (
    hotel_freeze[
        "relation_case_count"
    ]
    ==
    19
)

assert (
    sha256_file(
        HOTEL_SOURCE_PATH
    )
    ==
    hotel_freeze[
        "source_sha256"
    ]
)


print(
    "Hotel holdout freeze integrity: PASS"
)


first_prompt = render_reasoning_prompt(
    cases[
        0
    ]
)


second_prompt = render_reasoning_prompt(
    cases[
        0
    ]
)


assert (
    first_prompt
    ==
    second_prompt
)

assert (
    cases[
        0
    ].left_column
    in
    first_prompt
)

assert (
    cases[
        0
    ].right_column
    in
    first_prompt
)


print(
    "Prompt rendering determinism: PASS"
)


assert len(
    CANDIDATE_SURFACES
) == 3

assert (
    set(
        CANDIDATE_SURFACES
    )
    ==
    set(
        ALLOWED_RELATIONS
    )
)

assert len(
    set(
        CANDIDATE_SURFACES.values()
    )
) == 3


print(
    "Frozen candidate surfaces: PASS"
)


assert len(
    prompt_template_sha256()
) == 64

assert len(
    candidate_surface_sha256()
) == 64


print(
    "Prompt/candidate hashes: PASS"
)


selected = select_relation_from_scores(
    scores={
        "same_metric_different_state":
            -0.10,

        "same_process_different_stage":
            -0.40,

        "related_distinct_metric":
            -0.25,
    }
)


assert (
    selected
    ==
    "same_metric_different_state"
)


print(
    "Argmax score selection: PASS"
)


try:
    select_relation_from_scores(
        scores={
            "same_metric_different_state":
                -0.10,

            "same_process_different_stage":
                -0.10,

            "related_distinct_metric":
                -0.30,
        }
    )

except ValueError:
    pass

else:
    raise AssertionError(
        "Score ties must fail closed."
    )


print(
    "Score tie fail-closed: PASS"
)


try:
    select_relation_from_scores(
        scores={
            "same_metric_different_state":
                math.nan,

            "same_process_different_stage":
                -0.20,

            "related_distinct_metric":
                -0.30,
        }
    )

except ValueError:
    pass

else:
    raise AssertionError(
        "NaN score must fail closed."
    )


print(
    "Non-finite score fail-closed: PASS"
)


sample_case = AdaptedReasoningCase(
    case_id=
        "test:0001",

    left_column=
        "planned demand",

    right_column=
        "observed demand",

    expected_relation=
        "same_metric_different_state",
)


assert (
    prediction_is_correct(
        case=
            sample_case,

        scores={
            "same_metric_different_state":
                -0.10,

            "same_process_different_stage":
                -0.50,

            "related_distinct_metric":
                -0.25,
        },
    )
    is True
)


print(
    "Case correctness evaluation: PASS"
)


perfect_predictions = {
    case.case_id:
        case.expected_relation

    for case
    in cases
}


assert (
    classification_accuracy(
        cases=
            cases,

        predictions=
            perfect_predictions,
    )
    ==
    1.0
)


print(
    "Classification accuracy: PASS"
)


freeze = json.loads(
    FREEZE_PATH.read_text(
        encoding="utf-8-sig"
    )
)


assert (
    freeze[
        "status"
    ]
    ==
    "frozen"
)

assert (
    freeze[
        "acceptance_authority"
    ]
    is False
)

assert (
    freeze[
        "frozen_before_first_post_training_inference"
    ]
    is True
)

assert (
    freeze[
        "first_post_training_inference_completed"
    ]
    is False
)

assert (
    freeze[
        "case_artifact_sha256"
    ]
    ==
    sha256_file(
        CASES_PATH
    )
)

assert (
    freeze[
        "module_sha256"
    ]
    ==
    sha256_file(
        MODULE_PATH
    )
)

assert (
    freeze[
        "case_source"
    ][
        "hotel_holdout_freeze_sha256"
    ]
    ==
    sha256_file(
        HOTEL_FREEZE_PATH
    )
)

assert (
    freeze[
        "coverage"
    ][
        "covered_training_families"
    ]
    ==
    3
)

assert (
    freeze[
        "coverage"
    ][
        "total_training_families"
    ]
    ==
    5
)

assert (
    freeze[
        "coverage"
    ][
        "not_covered"
    ]
    ==
    [
        "safe_uncertainty",
        "unit_and_quantity_reasoning",
    ]
)


print(
    "Frozen protocol artifact: PASS"
)

print(
    "Coverage declaration 3/5: PASS"
)


source = MODULE_PATH.read_text(
    encoding="utf-8-sig"
)

tree = ast.parse(
    source
)


for node in ast.walk(
    tree
):
    if isinstance(
        node,
        ast.Import,
    ):
        modules = [
            alias.name.split(
                "."
            )[
                0
            ]

            for alias
            in node.names
        ]

    elif isinstance(
        node,
        ast.ImportFrom,
    ):
        modules = [
            (
                node.module.split(
                    "."
                )[
                    0
                ]
                if node.module
                else ""
            )
        ]

    else:
        continue

    if set(
        modules
    ) & {
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "safetensors",
    }:
        raise AssertionError(
            "Benchmark contract module must not "
            "load heavy ML dependencies."
        )


assert (
    "final_acceptance"
    not in
    source
)


print(
    "Static benchmark module safety: PASS"
)


print()

print(
    "SAFETY"
)

print(
    "  Model loaded by benchmark test: False"
)

print(
    "  Adapter loaded by benchmark test: False"
)

print(
    "  CUDA requested by benchmark test: False"
)

print(
    "  Inference requested by benchmark test: False"
)

print(
    "  Generation requested by benchmark test: False"
)

print(
    "  LLM judge used: False"
)

print(
    "  Final Acceptance imported: False"
)

print(
    "  Final Acceptance executed: False"
)


print()

print(
    "DATALENS ADAPTED REASONING BENCHMARK v0.1: PASS"
)

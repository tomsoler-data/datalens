from __future__ import annotations

import ast
import hashlib
import json

from pathlib import Path


from app.adaptation.reasoning_evaluation_runner_v0_2 import (
    ADAPTER_PATH,
    CASES_PATH,
    CONVERTED_MODEL_PATH,
    MANIFEST_FREEZE_PATH,
    MANIFEST_PATH,
    REASONING_EVALUATION_MANIFEST_RULE_VERSION,
    REASONING_EVALUATION_RECEIPT_RULE_VERSION,
    REASONING_EVALUATION_RUNNER_RULE_VERSION,
    REPORT_PATH,
    RECEIPT_PATH,
    ROOT,
    _paired_comparison,
    directory_snapshot,
    load_json_object,
    sha256_file,
    validate_static_contract,
)


print(
    "=== DATALENS REASONING EVALUATION RUNNER v0.2 ==="
)

print()


assert (
    REASONING_EVALUATION_RUNNER_RULE_VERSION
    ==
    "adapted_reasoning_evaluation_runner_v0.2"
)

assert (
    REASONING_EVALUATION_MANIFEST_RULE_VERSION
    ==
    "adapted_reasoning_evaluation_manifest_v0.2"
)

assert (
    REASONING_EVALUATION_RECEIPT_RULE_VERSION
    ==
    "adapted_reasoning_evaluation_receipt_v0.2"
)


print(
    "Rule versions: PASS"
)


manifest = validate_static_contract()


assert (
    manifest[
        "benchmark"
    ][
        "case_count"
    ]
    ==
    19
)


assert (
    manifest[
        "benchmark"
    ][
        "covered_training_families"
    ]
    ==
    3
)


assert (
    manifest[
        "benchmark"
    ][
        "total_training_families"
    ]
    ==
    5
)


print(
    "Frozen benchmark binding: PASS"
)


assert (
    directory_snapshot(
        ADAPTER_PATH
    )
    ==
    manifest[
        "adapter"
    ][
        "files"
    ]
)


assert (
    directory_snapshot(
        CONVERTED_MODEL_PATH
    )
    ==
    manifest[
        "converted_base_model"
    ][
        "files"
    ]
)


print(
    "Local model/adapter byte binding: PASS"
)


assert not REPORT_PATH.exists()

assert not RECEIPT_PATH.exists()


print(
    "No prior evaluation evidence: PASS"
)


freeze = load_json_object(
    MANIFEST_FREEZE_PATH
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
        "first_post_training_inference_completed"
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
        "manifest_sha256"
    ]
    ==
    sha256_file(
        MANIFEST_PATH
    )
)


print(
    "Execution manifest freeze: PASS"
)


# ============================================================
# STATIC AST SAFETY
# ============================================================


runner_path = (
    ROOT
    / "app"
    / "adaptation"
    / "reasoning_evaluation_runner_v0_2.py"
)


source = runner_path.read_text(
    encoding="utf-8-sig"
)


tree = ast.parse(
    source
)


heavy = {
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "safetensors",
}


top_level_heavy_imports = []


for node in tree.body:
    if isinstance(
        node,
        ast.Import,
    ):
        for alias in node.names:
            root = alias.name.split(
                "."
            )[0]

            if root in heavy:
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
        module = (
            node.module
            or
            ""
        )

        root = module.split(
            "."
        )[0]

        if root in heavy:
            top_level_heavy_imports.append(
                (
                    node.lineno,
                    module,
                )
            )


assert not top_level_heavy_imports


print(
    "No top-level heavy ML imports: PASS"
)


execute_nodes = [
    node
    for node in tree.body
    if (
        isinstance(
            node,
            ast.FunctionDef,
        )
        and
        node.name
        ==
        "execute_evaluation"
    )
]


assert len(
    execute_nodes
) == 1


execute_node = execute_nodes[
    0
]


authorization_lines = []

heavy_import_lines = []


for node in ast.walk(
    execute_node
):
    if isinstance(
        node,
        ast.Call,
    ):
        if (
            isinstance(
                node.func,
                ast.Name,
            )
            and
            node.func.id
            ==
            "authorize_execution"
        ):
            authorization_lines.append(
                node.lineno
            )

    elif isinstance(
        node,
        ast.Import,
    ):
        for alias in node.names:
            if (
                alias.name.split(
                    "."
                )[0]
                in
                heavy
            ):
                heavy_import_lines.append(
                    node.lineno
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

        if (
            module.split(
                "."
            )[0]
            in
            heavy
        ):
            heavy_import_lines.append(
                node.lineno
            )


assert len(
    authorization_lines
) == 1


assert heavy_import_lines


assert (
    authorization_lines[
        0
    ]
    <
    min(
        heavy_import_lines
    )
)


print(
    "Authorization precedes heavy ML imports: PASS"
)


for forbidden in (
    ".generate(",
    ".backward(",
    "optimizer.step(",
    "trainer.train(",
):
    assert (
        forbidden
        not in
        source.lower()
    )


print(
    "No generation/backward/optimizer/trainer calls: PASS"
)


for node in ast.walk(
    tree
):
    if isinstance(
        node,
        ast.ImportFrom,
    ):
        assert (
            "final_acceptance"
            not in
            (
                node.module
                or
                ""
            ).lower()
        )

    elif isinstance(
        node,
        ast.Import,
    ):
        for alias in node.names:
            assert (
                "final_acceptance"
                not in
                alias.name.lower()
            )


print(
    "No Final Acceptance runtime import: PASS"
)


# ============================================================
# TOKENIZER RETURN CONTRACT FIX
# ============================================================


chat_template_calls = [
    node
    for node in ast.walk(
        tree
    )
    if (
        isinstance(
            node,
            ast.Call,
        )
        and
        isinstance(
            node.func,
            ast.Attribute,
        )
        and
        node.func.attr
        ==
        "apply_chat_template"
    )
]


assert len(
    chat_template_calls
) == 2


for call in chat_template_calls:
    keyword_by_name = {
        keyword.arg:
            keyword.value

        for keyword
        in call.keywords

        if keyword.arg
        is not None
    }

    assert (
        "return_dict"
        in
        keyword_by_name
    )

    return_dict_value = (
        keyword_by_name[
            "return_dict"
        ]
    )

    assert isinstance(
        return_dict_value,
        ast.Constant,
    )

    assert (
        return_dict_value.value
        is False
    )


print(
    "Explicit chat-template list return contract: PASS"
)


# ============================================================
# PURE PAIRED-COMPARISON TEST
# ============================================================


base = {
    "accuracy":
        0.5,

    "macro_accuracy":
        0.5,

    "results": [
        {
            "case_id":
                "a",

            "correct":
                True,

            "predicted_relation":
                "same_metric_different_state",
        },

        {
            "case_id":
                "b",

            "correct":
                False,

            "predicted_relation":
                "related_distinct_metric",
        },
    ],
}


adapted = {
    "accuracy":
        1.0,

    "macro_accuracy":
        1.0,

    "results": [
        {
            "case_id":
                "a",

            "correct":
                True,

            "predicted_relation":
                "same_metric_different_state",
        },

        {
            "case_id":
                "b",

            "correct":
                True,

            "predicted_relation":
                "same_process_different_stage",
        },
    ],
}


paired = _paired_comparison(
    base=
        base,

    adapted=
        adapted,
)


assert (
    paired[
        "accuracy_delta"
    ]
    ==
    0.5
)


assert (
    paired[
        "macro_accuracy_delta"
    ]
    ==
    0.5
)


assert (
    paired[
        "adapted_only_correct"
    ]
    ==
    1
)


assert (
    paired[
        "base_only_correct"
    ]
    ==
    0
)


assert (
    paired[
        "preregistered_signal"
    ]
    ==
    "positive_signal"
)


print(
    "Paired comparison semantics: PASS"
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
    "  Generation executed: False"
)

print(
    "  Final Acceptance imported: False"
)

print(
    "  Final Acceptance executed: False"
)


print()

print(
    "DATALENS REASONING EVALUATION RUNNER v0.2: PASS"
)

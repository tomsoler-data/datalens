from __future__ import annotations

import ast
import inspect

from pathlib import Path


print(
    "=== DATALENS QLORA v0.4 TOKEN AUDIT RUNNER TEST v0.1 ==="
)

print()


ROOT = Path.cwd().resolve()


from app.adaptation.training_dataset_token_audit_v0_4 import (
    ADAPTATION_TOKEN_LENGTH_AUDIT_RULE_VERSION,
    AUDIT_RELATIVE_PATH,
    EXPECTED_CHAT_TEMPLATE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_EXAMPLE_COUNT,
    EXPECTED_MASKING_GIT_BLOB_SHA256,
    EXPECTED_MAX_TOTAL_TOKENS,
    EXPECTED_P95_TOTAL_TOKENS,
    EXPECTED_SEQUENCE_HEADROOM,
    EXPECTED_SUPERVISED_ASSISTANT_TOKENS,
    EXPECTED_TOTAL_TOKENS,
    EXPECTED_MASKED_PROMPT_TOKENS,
    MAX_SEQUENCE_LENGTH,
    build_token_length_audit,
    token_length_audit_bytes,
    token_length_audit_sha256,
)


if (
    ADAPTATION_TOKEN_LENGTH_AUDIT_RULE_VERSION
    !=
    "adaptation_token_length_audit_v0.2"
):
    raise RuntimeError(
        "Token audit rule version mismatch."
    )


print(
    "Rule version: PASS"
)


output_path = (
    ROOT
    /
    AUDIT_RELATIVE_PATH
)


if output_path.exists():
    raise RuntimeError(
        (
            "Official v0.4 token audit artifact "
            "must not exist before runner commit."
        )
    )


print(
    "Official artifact absent: PASS"
)


first = build_token_length_audit(
    repository_root=
        ROOT,
)


second = build_token_length_audit(
    repository_root=
        ROOT,
)


first_bytes = token_length_audit_bytes(
    repository_root=
        ROOT,
)


second_bytes = token_length_audit_bytes(
    repository_root=
        ROOT,
)


if first != second:
    raise RuntimeError(
        "Token audit object recomputation differs."
    )


if first_bytes != second_bytes:
    raise RuntimeError(
        "Token audit byte recomputation differs."
    )


if (
    first[
        "dataset"
    ][
        "dataset_sha256"
    ]
    !=
    EXPECTED_DATASET_SHA256
):
    raise RuntimeError(
        "Dataset binding mismatch."
    )


if (
    first[
        "dataset"
    ][
        "example_count"
    ]
    !=
    EXPECTED_EXAMPLE_COUNT
):
    raise RuntimeError(
        "Dataset example count mismatch."
    )


if (
    first[
        "tokenizer"
    ][
        "chat_template_sha256"
    ]
    !=
    EXPECTED_CHAT_TEMPLATE_SHA256
):
    raise RuntimeError(
        "Chat-template binding mismatch."
    )


if (
    first[
        "assistant_only_masking"
    ][
        "git_blob_sha256"
    ]
    !=
    EXPECTED_MASKING_GIT_BLOB_SHA256
):
    raise RuntimeError(
        "Assistant masking Git binding mismatch."
    )


distribution = first[
    "distribution"
][
    "full_example"
]


if (
    distribution[
        "maximum"
    ]
    !=
    EXPECTED_MAX_TOTAL_TOKENS
):
    raise RuntimeError(
        "Maximum token count mismatch."
    )


if (
    distribution[
        "p95"
    ]
    !=
    EXPECTED_P95_TOTAL_TOKENS
):
    raise RuntimeError(
        "p95 token count mismatch."
    )


if (
    distribution[
        "total_tokens"
    ]
    !=
    EXPECTED_TOTAL_TOKENS
):
    raise RuntimeError(
        "Total token count mismatch."
    )


if (
    first[
        "corpus_accounting"
    ][
        "masked_prompt_tokens"
    ]
    !=
    EXPECTED_MASKED_PROMPT_TOKENS
):
    raise RuntimeError(
        "Masked prompt token total mismatch."
    )


if (
    first[
        "corpus_accounting"
    ][
        "supervised_assistant_tokens"
    ]
    !=
    EXPECTED_SUPERVISED_ASSISTANT_TOKENS
):
    raise RuntimeError(
        "Supervised token total mismatch."
    )


if (
    first[
        "recommendation"
    ][
        "recommended_max_sequence_length"
    ]
    !=
    MAX_SEQUENCE_LENGTH
):
    raise RuntimeError(
        "Sequence-length recommendation mismatch."
    )


if (
    first[
        "recommendation"
    ][
        "sequence_headroom_tokens"
    ]
    !=
    EXPECTED_SEQUENCE_HEADROOM
):
    raise RuntimeError(
        "Sequence headroom mismatch."
    )


if (
    first[
        "recommendation"
    ][
        "truncated_examples"
    ]
    !=
    0
):
    raise RuntimeError(
        "Truncation count must be zero."
    )


if (
    first[
        "masking_invariants"
    ][
        "prompt_supervision_leaks"
    ]
    !=
    0
):
    raise RuntimeError(
        "Prompt supervision leak detected."
    )


if (
    first[
        "masking_invariants"
    ][
        "zero_supervision_examples"
    ]
    !=
    0
):
    raise RuntimeError(
        "Zero-supervision example detected."
    )


if (
    first[
        "passed"
    ]
    is not True
):
    raise RuntimeError(
        "Token audit must pass."
    )


artifact_sha256 = token_length_audit_sha256(
    repository_root=
        ROOT,
)


print()

print(
    "OBSERVED EVIDENCE"
)

print(
    (
        "  Examples: "
        f"{first['dataset']['example_count']}"
    )
)

print(
    (
        "  Max total tokens: "
        f"{distribution['maximum']}"
    )
)

print(
    (
        "  p95 total tokens: "
        f"{distribution['p95']}"
    )
)

print(
    (
        "  Total tokens: "
        f"{distribution['total_tokens']}"
    )
)

print(
    (
        "  Masked prompt tokens: "
        f"{first['corpus_accounting']['masked_prompt_tokens']}"
    )
)

print(
    (
        "  Supervised assistant tokens: "
        f"{first['corpus_accounting']['supervised_assistant_tokens']}"
    )
)

print(
    (
        "  Sequence headroom: "
        f"{first['recommendation']['sequence_headroom_tokens']}"
    )
)

print(
    (
        "  Future artifact SHA256: "
        f"{artifact_sha256}"
    )
)


module = __import__(
    (
        "app.adaptation."
        "training_dataset_token_audit_v0_4"
    ),
    fromlist=[
        "*",
    ],
)


module_source = inspect.getsource(
    module
)


tree = ast.parse(
    module_source
)


for forbidden in (
    "AutoModel",
    "AutoModelForCausalLM",
    "PeftModel",
    "bitsandbytes",
    "torch.cuda",
    "airport_ground_operations_holdout",
    "greenhouse_operations_final_acceptance",
):
    if forbidden in module_source:
        raise RuntimeError(
            (
                "Forbidden runtime dependency "
                f"in token audit runner: {forbidden}"
            )
        )


print()

print(
    "SAFETY"
)

print(
    "  Tokenizer loaded: True"
)

print(
    "  Model-loaded API dependency: False"
)

print(
    "  Adapter dependency: False"
)

print(
    "  CUDA dependency: False"
)

print(
    "  Holdout definition dependency: False"
)

print(
    "  Official artifact written: False"
)

print(
    "  Training executed: False"
)


print()

print(
    "DATALENS QLORA v0.4 TOKEN AUDIT RUNNER TEST v0.1: PASS"
)

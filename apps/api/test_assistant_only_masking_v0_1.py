from __future__ import annotations

import hashlib
import json

from pathlib import Path

from transformers import AutoTokenizer

from app.adaptation.assistant_masking import (
    ASSISTANT_ONLY_MASKING_RULE_VERSION,
    IGNORE_INDEX,
    build_assistant_only_training_example,
    collate_assistant_only_examples,
)


print(
    "=== DATALENS ASSISTANT-ONLY MASKING v0.1 ==="
)

print()


ROOT = Path.cwd()


DATASET_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "datasets"
    / "datalens_semantic_training_v0.1.jsonl"
)


EXPECTED_DATASET_SHA256 = (
    "4d7f1d8defeeb956448e31d776e5310795326b08ec5ddaf24091bd493c42f892"
)


MODEL_REPOSITORY = (
    "google/gemma-3-4b-it"
)


MODEL_REVISION = (
    "093f9f388b31de276ce2de164bdc2081324b9767"
)


EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"
)


EXPECTED_EXAMPLE_COUNT = 40

EXPECTED_MAX_SEQUENCE_LENGTH = 256

EXPECTED_TOTAL_TOKENS = 2769

EXPECTED_MASKED_PROMPT_TOKENS = 1293

EXPECTED_SUPERVISED_TOKENS = 1476

EXPECTED_MIN_SUPERVISED = 29

EXPECTED_MAX_SUPERVISED = 46

EXPECTED_MAX_EXAMPLE_TOKENS = 85


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


if (
    ASSISTANT_ONLY_MASKING_RULE_VERSION
    !=
    "assistant_only_masking_v0.1"
):
    raise RuntimeError(
        "Unexpected masking rule version."
    )


if (
    sha256_file(
        DATASET_PATH
    )
    !=
    EXPECTED_DATASET_SHA256
):
    raise RuntimeError(
        "Frozen training dataset SHA256 mismatch."
    )


print(
    "Frozen training dataset: PASS"
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_REPOSITORY,
        revision=
            MODEL_REVISION,
        trust_remote_code=
            False,
        local_files_only=
            True,
    )
)


if not isinstance(
    tokenizer.chat_template,
    str,
):
    raise RuntimeError(
        "Tokenizer has no string chat template."
    )


chat_template_sha256 = (
    hashlib.sha256(
        tokenizer.chat_template.encode(
            "utf-8"
        )
    ).hexdigest()
)


if (
    chat_template_sha256
    !=
    EXPECTED_CHAT_TEMPLATE_SHA256
):
    raise RuntimeError(
        "Chat-template SHA256 mismatch."
    )


print(
    "Pinned tokenizer: PASS"
)


records = []


with DATASET_PATH.open(
    "r",
    encoding="utf-8-sig",
) as handle:
    for line in handle:
        if not line.strip():
            continue

        records.append(
            json.loads(
                line
            )
        )


if (
    len(
        records
    )
    !=
    EXPECTED_EXAMPLE_COUNT
):
    raise RuntimeError(
        "Unexpected example count."
    )


examples = []


for record in records:
    example = (
        build_assistant_only_training_example(
            tokenizer=tokenizer,
            messages=
                record[
                    "messages"
                ],
            max_sequence_length=
                EXPECTED_MAX_SEQUENCE_LENGTH,
        )
    )

    examples.append(
        example
    )


if (
    len(
        examples
    )
    !=
    EXPECTED_EXAMPLE_COUNT
):
    raise RuntimeError(
        "Not all examples were converted."
    )


print(
    "All 40 examples converted: PASS"
)


total_tokens = sum(
    example.total_token_count
    for example in examples
)


total_prompt_tokens = sum(
    example.prompt_token_count
    for example in examples
)


total_supervised_tokens = sum(
    example.supervised_token_count
    for example in examples
)


minimum_supervised = min(
    example.supervised_token_count
    for example in examples
)


maximum_supervised = max(
    example.supervised_token_count
    for example in examples
)


maximum_example_length = max(
    example.total_token_count
    for example in examples
)


if (
    total_tokens
    !=
    EXPECTED_TOTAL_TOKENS
):
    raise RuntimeError(
        "Total token count changed."
    )


if (
    total_prompt_tokens
    !=
    EXPECTED_MASKED_PROMPT_TOKENS
):
    raise RuntimeError(
        "Masked prompt token count changed."
    )


if (
    total_supervised_tokens
    !=
    EXPECTED_SUPERVISED_TOKENS
):
    raise RuntimeError(
        "Supervised token count changed."
    )


if (
    minimum_supervised
    !=
    EXPECTED_MIN_SUPERVISED
):
    raise RuntimeError(
        "Minimum supervised token count changed."
    )


if (
    maximum_supervised
    !=
    EXPECTED_MAX_SUPERVISED
):
    raise RuntimeError(
        "Maximum supervised token count changed."
    )


if (
    maximum_example_length
    !=
    EXPECTED_MAX_EXAMPLE_TOKENS
):
    raise RuntimeError(
        "Maximum example length changed."
    )


print(
    "Frozen supervision statistics: PASS"
)


for index, example in enumerate(
    examples,
    start=1,
):
    if (
        example.total_token_count
        >
        EXPECTED_MAX_SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            f"Example {index} exceeds seq=256."
        )

    if (
        example.supervised_token_count
        <=
        0
    ):
        raise RuntimeError(
            f"Example {index} has zero supervision."
        )

    prompt_labels = (
        example.labels[
            :example.prompt_token_count
        ]
    )

    assistant_labels = (
        example.labels[
            example.prompt_token_count:
        ]
    )

    if any(
        label != IGNORE_INDEX
        for label in prompt_labels
    ):
        raise RuntimeError(
            "Prompt token is supervised."
        )

    if any(
        label == IGNORE_INDEX
        for label in assistant_labels
    ):
        raise RuntimeError(
            "Assistant token was unexpectedly masked."
        )

    if (
        assistant_labels
        !=
        example.input_ids[
            example.prompt_token_count:
        ]
    ):
        raise RuntimeError(
            "Assistant labels do not match "
            "assistant input tokens."
        )


print(
    "Prompt masking: PASS"
)

print(
    "Assistant supervision: PASS"
)


pad_token_id = (
    tokenizer.pad_token_id
)


if pad_token_id is None:
    raise RuntimeError(
        "Tokenizer has no pad_token_id."
    )


batch = collate_assistant_only_examples(
    examples=[
        examples[0],
        examples[-1],
    ],
    pad_token_id=
        pad_token_id,
)


if (
    len(
        batch.input_ids
    )
    !=
    2
):
    raise RuntimeError(
        "Unexpected batch size."
    )


batch_width = len(
    batch.input_ids[
        0
    ]
)


if (
    batch_width
    !=
    max(
        examples[0].total_token_count,
        examples[-1].total_token_count,
    )
):
    raise RuntimeError(
        "Unexpected padded batch width."
    )


for row_index in range(
    2
):
    real_length = (
        batch.total_token_counts[
            row_index
        ]
    )

    if any(
        value != 0
        for value in batch.attention_mask[
            row_index
        ][
            real_length:
        ]
    ):
        raise RuntimeError(
            "Padding attention mask is not zero."
        )

    if any(
        value != IGNORE_INDEX
        for value in batch.labels[
            row_index
        ][
            real_length:
        ]
    ):
        raise RuntimeError(
            "Padding labels are not masked."
        )


print(
    "Batch padding mask: PASS"
)


example_again = (
    build_assistant_only_training_example(
        tokenizer=tokenizer,
        messages=
            records[
                0
            ][
                "messages"
            ],
        max_sequence_length=
            EXPECTED_MAX_SEQUENCE_LENGTH,
    )
)


if (
    example_again
    !=
    examples[
        0
    ]
):
    raise RuntimeError(
        "Masking is not deterministic."
    )


print(
    "Determinism: PASS"
)


try:
    build_assistant_only_training_example(
        tokenizer=tokenizer,
        messages=
            records[
                0
            ][
                "messages"
            ],
        max_sequence_length=
            10,
    )

except ValueError:
    pass

else:
    raise RuntimeError(
        "Silent truncation was not rejected."
    )


print(
    "Silent truncation rejected: PASS"
)


try:
    build_assistant_only_training_example(
        tokenizer=tokenizer,
        messages=[
            {
                "role":
                    "user",
                "content":
                    "Synthetic test prompt.",
            },
        ],
        max_sequence_length=
            EXPECTED_MAX_SEQUENCE_LENGTH,
    )

except ValueError:
    pass

else:
    raise RuntimeError(
        "Missing assistant response was "
        "not rejected."
    )


print(
    "Missing assistant rejected: PASS"
)


print()

print(
    "SUPERVISION SUMMARY"
)


print(
    (
        "  Examples: "
        f"{len(examples)}"
    )
)


print(
    (
        "  Full tokens: "
        f"{total_tokens}"
    )
)


print(
    (
        "  Masked prompt tokens: "
        f"{total_prompt_tokens}"
    )
)


print(
    (
        "  Supervised assistant tokens: "
        f"{total_supervised_tokens}"
    )
)


print(
    (
        "  Supervised fraction: "
        f"{total_supervised_tokens / total_tokens:.4f}"
    )
)


print(
    (
        "  Min supervised/example: "
        f"{minimum_supervised}"
    )
)


print(
    (
        "  Max supervised/example: "
        f"{maximum_supervised}"
    )
)


print(
    (
        "  Max full example: "
        f"{maximum_example_length}"
    )
)


print()

print(
    "Training executed: False"
)


print(
    "Final Acceptance loaded: False"
)


print()

print(
    "DATALENS ASSISTANT-ONLY MASKING v0.1: PASS"
)

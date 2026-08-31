from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from transformers import AutoTokenizer

from app.adaptation.assistant_masking import (
    ASSISTANT_ONLY_MASKING_RULE_VERSION,
    IGNORE_INDEX,
    build_assistant_only_training_example,
    collate_assistant_only_examples,
)


ADAPTATION_TOKEN_LENGTH_AUDIT_RULE_VERSION = (
    "adaptation_token_length_audit_v0.2"
)


AUDIT_ID = (
    "datalens-semantic-training-v0.4-token-length-audit"
)


DATASET_ID = (
    "adaptation:datalens-semantic:training:v0.4"
)


DATASET_VERSION = (
    "datalens_semantic_adaptation_training_v0.4"
)


DATASET_RELATIVE_PATH = (
    "artifacts/adaptation/datasets/"
    "datalens_semantic_training_v0.4.jsonl"
)


AUDIT_RELATIVE_PATH = (
    "artifacts/adaptation/probes/"
    "datalens_semantic_training_v0.4_token_length_audit.json"
)


EXPECTED_DATASET_SHA256 = (
    "4fd00586f2d53d6de57f5cbc5f1d7bfb"
    "2e512960e60b30c28596aaefbac322b7"
)


MODEL_REPOSITORY = (
    "google/gemma-3-4b-it"
)


MODEL_REVISION = (
    "093f9f388b31de276ce2de164bdc2081324b9767"
)


EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "7de1c58e208eda46e9c7f86397df37ec"
    "49883aeece39fb961e0a6b24088dd3c4"
)


MASKING_GIT_PATH = (
    "apps/api/app/adaptation/"
    "assistant_masking.py"
)


MASKING_WORKING_RELATIVE_PATH = (
    "app/adaptation/"
    "assistant_masking.py"
)


EXPECTED_MASKING_GIT_BLOB_SHA256 = (
    "eafa9ed7a207395e3698ffc5ccfdf8a5"
    "dc3e3dcbafe554d98037ecc3140c6483"
)


EXPECTED_MASKING_RULE_VERSION = (
    "assistant_only_masking_v0.1"
)


EXPECTED_EXAMPLE_COUNT = 230


MAX_SEQUENCE_LENGTH = 256


EXPECTED_MAX_TOTAL_TOKENS = 206


EXPECTED_P95_TOTAL_TOKENS = 200


EXPECTED_TOTAL_TOKENS = 43799


EXPECTED_MASKED_PROMPT_TOKENS = 35978


EXPECTED_SUPERVISED_ASSISTANT_TOKENS = 7821


EXPECTED_SEQUENCE_HEADROOM = 50


EXPECTED_RELATION_COUNTS = Counter(
    {
        "same_metric_different_state":
            50,

        "same_process_different_stage":
            50,

        "related_distinct_metric":
            50,

        "unrelated":
            40,

        "uncertain":
            40,
    }
)


THRESHOLDS = (
    192,
    224,
    256,
    384,
    512,
    768,
    1024,
)


def _root(
    repository_root: Path | None,
) -> Path:
    root = (
        Path.cwd()
        if repository_root is None
        else repository_root
    )

    root = (
        root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise NotADirectoryError(
            root
        )

    return root


def _sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def _sha256_file(
    path: Path,
) -> str:
    return _sha256_bytes(
        path.read_bytes()
    )


def _normalize_newlines(
    payload: bytes,
) -> bytes:
    return (
        payload
        .replace(
            b"\r\n",
            b"\n",
        )
        .replace(
            b"\r",
            b"\n",
        )
    )


def _masking_git_blob(
    *,
    repository_root: Path,
) -> bytes:
    result = subprocess.run(
        [
            "git",
            "show",
            f"HEAD:{MASKING_GIT_PATH}",
        ],
        cwd=
            repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode(
                errors="replace"
            )
        )

    return result.stdout


def _verify_masking_binding(
    *,
    repository_root: Path,
) -> str:
    git_blob = _masking_git_blob(
        repository_root=
            repository_root,
    )

    blob_sha256 = _sha256_bytes(
        git_blob
    )

    if (
        blob_sha256
        !=
        EXPECTED_MASKING_GIT_BLOB_SHA256
    ):
        raise RuntimeError(
            (
                "Assistant masking Git blob changed.\n"
                f"Expected: "
                f"{EXPECTED_MASKING_GIT_BLOB_SHA256}\n"
                f"Actual:   {blob_sha256}"
            )
        )

    working_path = (
        repository_root
        /
        MASKING_WORKING_RELATIVE_PATH
    )

    if not working_path.is_file():
        raise FileNotFoundError(
            working_path
        )

    working_normalized = (
        _normalize_newlines(
            working_path.read_bytes()
        )
    )

    if working_normalized != git_blob:
        raise RuntimeError(
            (
                "Working assistant_masking.py differs "
                "from HEAD beyond newline normalization."
            )
        )

    if (
        ASSISTANT_ONLY_MASKING_RULE_VERSION
        !=
        EXPECTED_MASKING_RULE_VERSION
    ):
        raise RuntimeError(
            "Assistant masking rule version changed."
        )

    if IGNORE_INDEX != -100:
        raise RuntimeError(
            "Assistant masking IGNORE_INDEX changed."
        )

    return blob_sha256


def _load_dataset(
    *,
    repository_root: Path,
) -> Tuple[
    Mapping[
        str,
        Any,
    ],
    ...,
]:
    path = (
        repository_root
        /
        DATASET_RELATIVE_PATH
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual_sha256 = _sha256_file(
        path
    )

    if (
        actual_sha256
        !=
        EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            (
                "Frozen v0.4 dataset SHA changed.\n"
                f"Expected: {EXPECTED_DATASET_SHA256}\n"
                f"Actual:   {actual_sha256}"
            )
        )

    raw_bytes = path.read_bytes()

    if b"\r\n" in raw_bytes:
        raise RuntimeError(
            "Frozen canonical dataset contains CRLF."
        )

    records = []

    for line_number, line in enumerate(
        raw_bytes.decode(
            "utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            raise RuntimeError(
                (
                    "Blank canonical JSONL line: "
                    f"{line_number}"
                )
            )

        value = json.loads(
            line
        )

        if not isinstance(
            value,
            dict,
        ):
            raise RuntimeError(
                (
                    "Canonical record is not an object: "
                    f"{line_number}"
                )
            )

        records.append(
            value
        )

    if len(
        records
    ) != EXPECTED_EXAMPLE_COUNT:
        raise RuntimeError(
            (
                "Frozen example count changed.\n"
                f"Expected: {EXPECTED_EXAMPLE_COUNT}\n"
                f"Actual:   {len(records)}"
            )
        )

    relations = Counter()

    for record in records:
        messages = record[
            "messages"
        ]

        if (
            len(
                messages
            )
            !=
            2
            or
            messages[
                0
            ][
                "role"
            ]
            !=
            "user"
            or
            messages[
                1
            ][
                "role"
            ]
            !=
            "assistant"
        ):
            raise RuntimeError(
                "Canonical chat shape changed."
            )

        assistant_payload = json.loads(
            messages[
                1
            ][
                "content"
            ]
        )

        relation = assistant_payload[
            "relation"
        ]

        relations[
            relation
        ] += 1

    if relations != EXPECTED_RELATION_COUNTS:
        raise RuntimeError(
            (
                "Frozen relation distribution changed.\n"
                f"Actual: {relations}"
            )
        )

    return tuple(
        records
    )


def _load_tokenizer() -> Any:
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
            "Pinned tokenizer has no chat template."
        )

    template_sha256 = _sha256_bytes(
        tokenizer.chat_template.encode(
            "utf-8"
        )
    )

    if (
        template_sha256
        !=
        EXPECTED_CHAT_TEMPLATE_SHA256
    ):
        raise RuntimeError(
            (
                "Pinned Gemma chat template changed.\n"
                f"Expected: "
                f"{EXPECTED_CHAT_TEMPLATE_SHA256}\n"
                f"Actual:   {template_sha256}"
            )
        )

    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Pinned tokenizer has no pad_token_id."
        )

    return tokenizer


def _nearest_rank_percentile(
    values: Sequence[int],
    percentile: float,
) -> int:
    if not values:
        raise ValueError(
            "values must not be empty."
        )

    ordered = sorted(
        values
    )

    rank = math.ceil(
        percentile
        *
        len(
            ordered
        )
    )

    rank = max(
        1,
        min(
            rank,
            len(
                ordered
            ),
        ),
    )

    return ordered[
        rank - 1
    ]


def _distribution(
    values: Sequence[int],
) -> Dict[
    str,
    float | int,
]:
    return {
        "minimum":
            min(
                values
            ),

        "mean":
            statistics.mean(
                values
            ),

        "median":
            statistics.median(
                values
            ),

        "p75":
            _nearest_rank_percentile(
                values,
                0.75,
            ),

        "p90":
            _nearest_rank_percentile(
                values,
                0.90,
            ),

        "p95":
            _nearest_rank_percentile(
                values,
                0.95,
            ),

        "p99":
            _nearest_rank_percentile(
                values,
                0.99,
            ),

        "maximum":
            max(
                values
            ),

        "total_tokens":
            sum(
                values
            ),
    }


def _threshold_summary(
    *,
    token_counts: Sequence[int],
    limit: int,
) -> Dict[
    str,
    float | int,
]:
    covered_examples = sum(
        1

        for count
        in token_counts

        if count <= limit
    )

    truncated_examples = (
        len(
            token_counts
        )
        -
        covered_examples
    )

    tokens_above_limit = sum(
        max(
            0,
            count - limit,
        )

        for count
        in token_counts
    )

    return {
        "coverage_fraction":
            covered_examples
            /
            len(
                token_counts
            ),

        "covered_examples":
            covered_examples,

        "tokens_above_limit":
            tokens_above_limit,

        "truncated_examples":
            truncated_examples,
    }


def _artifact_bytes(
    artifact: Mapping[
        str,
        Any,
    ],
) -> bytes:
    serialized = json.dumps(
        artifact,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    )

    return (
        serialized
        +
        "\n"
    ).encode(
        "utf-8"
    )


def build_token_length_audit(
    *,
    repository_root: Path,
) -> Dict[
    str,
    Any,
]:
    repository_root = _root(
        repository_root
    )

    masking_blob_sha256 = (
        _verify_masking_binding(
            repository_root=
                repository_root,
        )
    )

    records = _load_dataset(
        repository_root=
            repository_root,
    )

    tokenizer = _load_tokenizer()

    examples = []

    relation_names = []

    example_rows = []

    for record in records:
        tokenized = (
            build_assistant_only_training_example(
                tokenizer=
                    tokenizer,

                messages=
                    record[
                        "messages"
                    ],

                max_sequence_length=
                    MAX_SEQUENCE_LENGTH,
            )
        )

        if (
            tokenized.total_token_count
            >
            MAX_SEQUENCE_LENGTH
        ):
            raise RuntimeError(
                (
                    "Tokenized example exceeds seq=256: "
                    f"{record['example_id']}"
                )
            )

        if (
            tokenized.supervised_token_count
            <=
            0
        ):
            raise RuntimeError(
                (
                    "Tokenized example has no "
                    "assistant supervision: "
                    f"{record['example_id']}"
                )
            )

        if (
            tokenized.prompt_token_count
            +
            tokenized.supervised_token_count
            !=
            tokenized.total_token_count
        ):
            raise RuntimeError(
                "Token accounting mismatch."
            )

        if (
            len(
                tokenized.input_ids
            )
            !=
            tokenized.total_token_count
            or
            len(
                tokenized.attention_mask
            )
            !=
            tokenized.total_token_count
            or
            len(
                tokenized.labels
            )
            !=
            tokenized.total_token_count
        ):
            raise RuntimeError(
                "Tokenized sequence length mismatch."
            )

        if any(
            value != 1

            for value
            in tokenized.attention_mask
        ):
            raise RuntimeError(
                "Non-padding attention mask changed."
            )

        prompt_labels = tokenized.labels[
            :tokenized.prompt_token_count
        ]

        assistant_labels = tokenized.labels[
            tokenized.prompt_token_count:
        ]

        assistant_inputs = tokenized.input_ids[
            tokenized.prompt_token_count:
        ]

        if any(
            value != IGNORE_INDEX

            for value
            in prompt_labels
        ):
            raise RuntimeError(
                "Prompt supervision leak detected."
            )

        if any(
            value == IGNORE_INDEX

            for value
            in assistant_labels
        ):
            raise RuntimeError(
                "Assistant supervision masking detected."
            )

        if assistant_labels != assistant_inputs:
            raise RuntimeError(
                "Assistant labels differ from input IDs."
            )

        assistant_payload = json.loads(
            record[
                "messages"
            ][
                1
            ][
                "content"
            ]
        )

        relation = assistant_payload[
            "relation"
        ]

        relation_names.append(
            relation
        )

        examples.append(
            tokenized
        )

        example_rows.append(
            {
                "example_id":
                    record[
                        "example_id"
                    ],

                "prompt_token_count":
                    tokenized.prompt_token_count,

                "supervised_token_count":
                    tokenized.supervised_token_count,

                "total_token_count":
                    tokenized.total_token_count,
            }
        )

    prompt_counts = [
        example.prompt_token_count

        for example
        in examples
    ]

    supervised_counts = [
        example.supervised_token_count

        for example
        in examples
    ]

    total_counts = [
        example.total_token_count

        for example
        in examples
    ]

    prompt_distribution = _distribution(
        prompt_counts
    )

    supervised_distribution = _distribution(
        supervised_counts
    )

    total_distribution = _distribution(
        total_counts
    )

    if (
        total_distribution[
            "maximum"
        ]
        !=
        EXPECTED_MAX_TOTAL_TOKENS
    ):
        raise RuntimeError(
            (
                "Observed maximum token count changed.\n"
                f"Expected: {EXPECTED_MAX_TOTAL_TOKENS}\n"
                f"Actual:   "
                f"{total_distribution['maximum']}"
            )
        )

    if (
        total_distribution[
            "p95"
        ]
        !=
        EXPECTED_P95_TOTAL_TOKENS
    ):
        raise RuntimeError(
            (
                "Observed p95 token count changed.\n"
                f"Expected: {EXPECTED_P95_TOTAL_TOKENS}\n"
                f"Actual:   "
                f"{total_distribution['p95']}"
            )
        )

    if (
        sum(
            total_counts
        )
        !=
        EXPECTED_TOTAL_TOKENS
    ):
        raise RuntimeError(
            "Observed corpus token total changed."
        )

    if (
        sum(
            prompt_counts
        )
        !=
        EXPECTED_MASKED_PROMPT_TOKENS
    ):
        raise RuntimeError(
            "Observed masked prompt total changed."
        )

    if (
        sum(
            supervised_counts
        )
        !=
        EXPECTED_SUPERVISED_ASSISTANT_TOKENS
    ):
        raise RuntimeError(
            "Observed assistant token total changed."
        )

    sequence_headroom = (
        MAX_SEQUENCE_LENGTH
        -
        max(
            total_counts
        )
    )

    if (
        sequence_headroom
        !=
        EXPECTED_SEQUENCE_HEADROOM
    ):
        raise RuntimeError(
            "Observed sequence headroom changed."
        )

    shortest_index = min(
        range(
            len(
                examples
            )
        ),
        key=lambda index:
            examples[
                index
            ].total_token_count,
    )

    longest_index = max(
        range(
            len(
                examples
            )
        ),
        key=lambda index:
            examples[
                index
            ].total_token_count,
    )

    batch = collate_assistant_only_examples(
        examples=[
            examples[
                shortest_index
            ],
            examples[
                longest_index
            ],
        ],
        pad_token_id=
            tokenizer.pad_token_id,
    )

    expected_width = max(
        examples[
            shortest_index
        ].total_token_count,
        examples[
            longest_index
        ].total_token_count,
    )

    for row_index in range(
        2
    ):
        real_length = (
            batch.total_token_counts[
                row_index
            ]
        )

        if (
            len(
                batch.input_ids[
                    row_index
                ]
            )
            !=
            expected_width
        ):
            raise RuntimeError(
                "Collator input width mismatch."
            )

        if any(
            value != 0

            for value
            in batch.attention_mask[
                row_index
            ][
                real_length:
            ]
        ):
            raise RuntimeError(
                "Collator attention padding changed."
            )

        if any(
            value != IGNORE_INDEX

            for value
            in batch.labels[
                row_index
            ][
                real_length:
            ]
        ):
            raise RuntimeError(
                "Collator label padding changed."
            )

    recomputed_first = (
        build_assistant_only_training_example(
            tokenizer=
                tokenizer,

            messages=
                records[
                    0
                ][
                    "messages"
                ],

            max_sequence_length=
                MAX_SEQUENCE_LENGTH,
        )
    )

    if recomputed_first != examples[
        0
    ]:
        raise RuntimeError(
            "Assistant masking is not deterministic."
        )

    try:
        build_assistant_only_training_example(
            tokenizer=
                tokenizer,

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
        truncation_rejected = True

    else:
        truncation_rejected = False

    if not truncation_rejected:
        raise RuntimeError(
            "Silent truncation was not rejected."
        )

    relation_values = defaultdict(
        lambda: {
            "total":
                [],
            "supervised":
                [],
        }
    )

    for relation, example in zip(
        relation_names,
        examples,
    ):
        relation_values[
            relation
        ][
            "total"
        ].append(
            example.total_token_count
        )

        relation_values[
            relation
        ][
            "supervised"
        ].append(
            example.supervised_token_count
        )

    relation_summary = {}

    for relation in EXPECTED_RELATION_COUNTS:
        totals = relation_values[
            relation
        ][
            "total"
        ]

        supervised = relation_values[
            relation
        ][
            "supervised"
        ]

        relation_summary[
            relation
        ] = {
            "example_count":
                len(
                    totals
                ),

            "maximum_total_tokens":
                max(
                    totals
                ),

            "mean_supervised_tokens":
                statistics.mean(
                    supervised
                ),

            "mean_total_tokens":
                statistics.mean(
                    totals
                ),
        }

    total_tokens = sum(
        total_counts
    )

    masked_tokens = sum(
        prompt_counts
    )

    supervised_tokens = sum(
        supervised_counts
    )

    threshold_summary = {
        str(
            limit
        ):
            _threshold_summary(
                token_counts=
                    total_counts,
                limit=
                    limit,
            )

        for limit
        in THRESHOLDS
    }

    if (
        threshold_summary[
            str(
                MAX_SEQUENCE_LENGTH
            )
        ][
            "truncated_examples"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "seq=256 is not lossless."
        )

    artifact = {
        "audit_id":
            AUDIT_ID,

        "assistant_only_masking": {
            "git_blob_sha256":
                masking_blob_sha256,

            "ignore_index":
                IGNORE_INDEX,

            "rule_version":
                ASSISTANT_ONLY_MASKING_RULE_VERSION,

            "supervision_scope":
                "assistant_tokens_only",
        },

        "collator": {
            "label_padding_value":
                IGNORE_INDEX,

            "padding_attention_value":
                0,

            "shortest_longest_batch_passed":
                True,
        },

        "corpus_accounting": {
            "masked_fraction":
                masked_tokens
                /
                total_tokens,

            "masked_prompt_tokens":
                masked_tokens,

            "supervised_assistant_tokens":
                supervised_tokens,

            "supervised_fraction":
                supervised_tokens
                /
                total_tokens,

            "total_tokens":
                total_tokens,
        },

        "dataset": {
            "dataset_id":
                DATASET_ID,

            "dataset_sha256":
                EXPECTED_DATASET_SHA256,

            "dataset_version":
                DATASET_VERSION,

            "example_count":
                EXPECTED_EXAMPLE_COUNT,
        },

        "distribution": {
            "full_example":
                total_distribution,

            "prompt_masked":
                prompt_distribution,

            "supervised_assistant":
                supervised_distribution,
        },

        "examples":
            example_rows,

        "fail_closed": {
            "deliberate_seq_10_truncation_rejected":
                True,

            "deterministic_recomputation":
                True,

            "silent_truncation_permitted":
                False,
        },

        "masking_invariants": {
            "assistant_label_input_mismatches":
                0,

            "assistant_masked_token_failures":
                0,

            "attention_mask_failures":
                0,

            "prompt_supervision_leaks":
                0,

            "token_accounting_failures":
                0,

            "zero_supervision_examples":
                0,
        },

        "passed":
            True,

        "per_relation":
            relation_summary,

        "recommendation": {
            "lossless_for_frozen_dataset":
                True,

            "reason": (
                "All 230 frozen QLoRA v0.4 training "
                "examples fit within 256 tokens after "
                "the pinned Gemma chat template and "
                "assistant-only masking path. The "
                "observed maximum is 206 tokens, "
                "leaving 50 tokens of sequence "
                "headroom with zero truncation."
            ),

            "recommended_max_sequence_length":
                MAX_SEQUENCE_LENGTH,

            "sequence_headroom_tokens":
                sequence_headroom,

            "tokens_removed":
                0,

            "truncated_examples":
                0,
        },

        "rule_version":
            ADAPTATION_TOKEN_LENGTH_AUDIT_RULE_VERSION,

        "safety": {
            "adapter_loaded":
                False,

            "airport_case_content_read":
                False,

            "airport_evaluated":
                False,

            "backward_executed":
                False,

            "cuda_requested":
                False,

            "final_acceptance_loaded":
                False,

            "final_acceptance_evaluated":
                False,

            "forward_executed":
                False,

            "gpu_computation":
                False,

            "model_loaded":
                False,

            "optimizer_created":
                False,

            "tokenizer_loaded":
                True,

            "training_dataset_content_embedded":
                False,

            "training_executed":
                False,
        },

        "thresholds":
            threshold_summary,

        "tokenizer": {
            "base_vocab_size":
                tokenizer.vocab_size,

            "chat_template_sha256":
                EXPECTED_CHAT_TEMPLATE_SHA256,

            "pad_token_id":
                tokenizer.pad_token_id,

            "repository":
                MODEL_REPOSITORY,

            "revision":
                MODEL_REVISION,

            "tokenization_path":
                (
                    "apply_chat_template(tokenize=True) "
                    "via assistant_only_masking_v0.1"
                ),

            "tokenizer_length":
                len(
                    tokenizer
                ),
        },
    }

    return artifact


def token_length_audit_bytes(
    *,
    repository_root: Path,
) -> bytes:
    artifact = build_token_length_audit(
        repository_root=
            repository_root,
    )

    return _artifact_bytes(
        artifact
    )


def token_length_audit_sha256(
    *,
    repository_root: Path,
) -> str:
    return _sha256_bytes(
        token_length_audit_bytes(
            repository_root=
                repository_root,
        )
    )


def validate_token_length_audit(
    *,
    repository_root: Path,
) -> None:
    repository_root = _root(
        repository_root
    )

    first = build_token_length_audit(
        repository_root=
            repository_root,
    )

    second = build_token_length_audit(
        repository_root=
            repository_root,
    )

    first_bytes = _artifact_bytes(
        first
    )

    second_bytes = _artifact_bytes(
        second
    )

    if first_bytes != second_bytes:
        raise RuntimeError(
            "Token audit is not deterministic."
        )

    artifact_sha256 = _sha256_bytes(
        first_bytes
    )

    print(
        "=== DATALENS QLORA v0.4 TOKEN-LENGTH EVIDENCE v0.2 ==="
    )

    print()

    print(
        f"Audit ID: {first['audit_id']}"
    )

    print(
        (
            "Dataset SHA256: "
            f"{first['dataset']['dataset_sha256']}"
        )
    )

    print(
        (
            "Examples: "
            f"{first['dataset']['example_count']}"
        )
    )

    print()

    print(
        "TOKENIZER"
    )

    print(
        (
            "  Repository: "
            f"{first['tokenizer']['repository']}"
        )
    )

    print(
        (
            "  Revision: "
            f"{first['tokenizer']['revision']}"
        )
    )

    print(
        (
            "  Chat-template SHA256: "
            f"{first['tokenizer']['chat_template_sha256']}"
        )
    )

    print()

    print(
        "ASSISTANT-ONLY MASKING"
    )

    print(
        (
            "  Rule: "
            f"{first['assistant_only_masking']['rule_version']}"
        )
    )

    print(
        (
            "  Git blob SHA256: "
            f"{first['assistant_only_masking']['git_blob_sha256']}"
        )
    )

    print(
        "  Prompt supervision leaks: 0"
    )

    print(
        "  Assistant masked-token failures: 0"
    )

    print(
        "  Zero-supervision examples: 0"
    )

    print()

    print(
        "FULL EXAMPLE DISTRIBUTION"
    )

    distribution = first[
        "distribution"
    ][
        "full_example"
    ]

    for key in (
        "minimum",
        "mean",
        "median",
        "p75",
        "p90",
        "p95",
        "p99",
        "maximum",
        "total_tokens",
    ):
        print(
            (
                f"  {key}: "
                f"{distribution[key]}"
            )
        )

    print()

    print(
        "CORPUS ACCOUNTING"
    )

    for key in (
        "total_tokens",
        "masked_prompt_tokens",
        "supervised_assistant_tokens",
        "masked_fraction",
        "supervised_fraction",
    ):
        print(
            (
                f"  {key}: "
                f"{first['corpus_accounting'][key]}"
            )
        )

    print()

    print(
        "RECOMMENDATION"
    )

    print(
        (
            "  max_sequence_length: "
            f"{first['recommendation']['recommended_max_sequence_length']}"
        )
    )

    print(
        (
            "  observed maximum: "
            f"{distribution['maximum']}"
        )
    )

    print(
        (
            "  sequence headroom: "
            f"{first['recommendation']['sequence_headroom_tokens']}"
        )
    )

    print(
        "  truncated examples: 0"
    )

    print(
        "  lossless: True"
    )

    print()

    print(
        "EVIDENCE"
    )

    print(
        (
            "  Rule version: "
            f"{first['rule_version']}"
        )
    )

    print(
        (
            "  Future artifact SHA256: "
            f"{artifact_sha256}"
        )
    )

    print(
        "  Deterministic recomputation: PASS"
    )

    print()

    print(
        "SAFETY"
    )

    print(
        "  Official artifact written: False"
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
        "DATALENS QLORA v0.4 TOKEN-LENGTH EVIDENCE v0.2: PASS"
    )


def write_token_length_audit(
    *,
    repository_root: Path,
) -> str:
    repository_root = _root(
        repository_root
    )

    output_path = (
        repository_root
        /
        AUDIT_RELATIVE_PATH
    ).resolve()

    if output_path.exists():
        raise FileExistsError(
            output_path
        )

    payload = token_length_audit_bytes(
        repository_root=
            repository_root,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        payload
    )

    actual_sha256 = _sha256_file(
        output_path
    )

    expected_sha256 = _sha256_bytes(
        payload
    )

    if actual_sha256 != expected_sha256:
        try:
            output_path.unlink()
        except OSError:
            pass

        raise RuntimeError(
            "Written token audit SHA mismatch."
        )

    return actual_sha256


def verify_token_length_audit(
    *,
    repository_root: Path,
) -> str:
    repository_root = _root(
        repository_root
    )

    output_path = (
        repository_root
        /
        AUDIT_RELATIVE_PATH
    ).resolve()

    if not output_path.is_file():
        raise FileNotFoundError(
            output_path
        )

    expected_bytes = token_length_audit_bytes(
        repository_root=
            repository_root,
    )

    actual_bytes = output_path.read_bytes()

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            (
                "Committed token audit does not match "
                "deterministic recomputation."
            )
        )

    return _sha256_bytes(
        actual_bytes
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=(
            "validate",
            "write",
            "verify",
        ),
    )

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )

    arguments = parser.parse_args()

    repository_root = _root(
        arguments.repository_root
    )

    if arguments.command == "validate":
        output_path = (
            repository_root
            /
            AUDIT_RELATIVE_PATH
        )

        if output_path.exists():
            raise RuntimeError(
                (
                    "Official token-length artifact "
                    "already exists."
                )
            )

        validate_token_length_audit(
            repository_root=
                repository_root,
        )

        return

    if arguments.command == "write":
        sha256 = write_token_length_audit(
            repository_root=
                repository_root,
        )

        print(
            "=== DATALENS QLORA v0.4 TOKEN-LENGTH ARTIFACT WRITE v0.2 ==="
        )

        print()

        print(
            f"Path: {AUDIT_RELATIVE_PATH}"
        )

        print(
            f"SHA256: {sha256}"
        )

        print()

        print(
            "DATALENS QLORA v0.4 TOKEN-LENGTH ARTIFACT WRITE v0.2: PASS"
        )

        return

    if arguments.command == "verify":
        sha256 = verify_token_length_audit(
            repository_root=
                repository_root,
        )

        print(
            "=== DATALENS QLORA v0.4 TOKEN-LENGTH ARTIFACT VERIFY v0.2 ==="
        )

        print()

        print(
            f"SHA256: {sha256}"
        )

        print(
            "Deterministic evidence identity: PASS"
        )

        print()

        print(
            "DATALENS QLORA v0.4 TOKEN-LENGTH ARTIFACT VERIFY v0.2: PASS"
        )

        return

    raise RuntimeError(
        "Unsupported command."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess

from collections import Counter
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import (
    Any,
    Mapping,
)


from app.adaptation.reasoning_benchmark import (
    ADAPTED_REASONING_SCORING_RULE_VERSION,
    ALLOWED_RELATIONS,
    BENCHMARK_ID,
    BENCHMARK_VERSION,
    AdaptedReasoningCase,
    candidate_continuation,
    load_frozen_reasoning_cases,
    render_reasoning_prompt,
    select_relation_from_scores,
)


REASONING_EVALUATION_RUNNER_RULE_VERSION = (
    "adapted_reasoning_evaluation_runner_v0.2"
)

REASONING_EVALUATION_MANIFEST_RULE_VERSION = (
    "adapted_reasoning_evaluation_manifest_v0.2"
)

REASONING_EVALUATION_RECEIPT_RULE_VERSION = (
    "adapted_reasoning_evaluation_receipt_v0.2"
)


ROOT = Path(__file__).resolve().parents[2]


ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
)


MANIFEST_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.3_"
      "reasoning_evaluation_v0.2_manifest.json"
)


MANIFEST_FREEZE_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.3_"
      "reasoning_evaluation_v0.2_manifest_freeze.json"
)


CASES_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.3_"
      "reasoning_benchmark_v0.1_cases.json"
)


PROTOCOL_FREEZE_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.3_"
      "reasoning_benchmark_v0.1_freeze.json"
)


REPORT_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.3_"
      "reasoning_evaluation_v0.2_report.json"
)


RECEIPT_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.3_"
      "reasoning_evaluation_v0.2_receipt.json"
)


ADAPTER_PATH = (
    Path.home()
    / ".cache"
    / "datalens"
    / "adaptation"
    / "runs"
    / "datalens-semantic-qlora-v0.3-training-v0.1"
    / "adapter"
)


CONVERTED_MODEL_PATH = (
    Path.home()
    / ".cache"
    / "datalens"
    / "adaptation"
    / "base-models"
    / (
        "gemma-3-4b-it-text-"
        "093f9f388b31de276ce2de164bdc2081324b9767"
    )
)


BASE_MODEL_REPOSITORY = (
    "google/gemma-3-4b-it"
)


BASE_MODEL_REVISION = (
    "093f9f388b31de276ce2de164bdc2081324b9767"
)


EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "7de1c58e208eda46e9c7f86397df37ec"
    "49883aeece39fb961e0a6b24088dd3c4"
)


MAX_SEQUENCE_LENGTH = 256

MINIMUM_FREE_CUDA_BYTES = (
    5
    * 1024
    * 1024
    * 1024
)


RUNNER_REPO_PATH = (
    "apps/api/app/adaptation/"
    "reasoning_evaluation_runner_v0_2.py"
)


TEST_REPO_PATH = (
    "apps/api/"
    "test_reasoning_evaluation_runner_v0_2.py"
)


MANIFEST_REPO_PATH = (
    "apps/api/artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.3_"
    "reasoning_evaluation_v0.2_manifest.json"
)


MANIFEST_FREEZE_REPO_PATH = (
    "apps/api/artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.3_"
    "reasoning_evaluation_v0.2_manifest_freeze.json"
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


def directory_snapshot(
    path: Path,
) -> dict[
    str,
    str,
]:
    result = {}

    for candidate in sorted(
        path.rglob(
            "*"
        )
    ):
        if not candidate.is_file():
            continue

        result[
            candidate
            .relative_to(
                path
            )
            .as_posix()
        ] = sha256_file(
            candidate
        )

    return result


def load_json_object(
    path: Path,
) -> dict[
    str,
    Any,
]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return payload


def atomic_write_json(
    *,
    path: Path,
    payload: object,
) -> None:
    if path.exists():
        raise FileExistsError(
            (
                "Refusing to overwrite immutable "
                f"evaluation evidence: {path}"
            )
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode(
            "utf-8"
        )
        +
        b"\n"
    )

    temporary = path.with_name(
        path.name
        +
        ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    temporary.write_bytes(
        data
    )

    os.replace(
        temporary,
        path,
    )


def git_output(
    *arguments: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=
            ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            (
                "Git command failed: "
                f"git {' '.join(arguments)}\n"
                f"{result.stderr}"
            )
        )

    return result.stdout.strip()


def git_head(
) -> str:
    return git_output(
        "rev-parse",
        "HEAD",
    )


def git_worktree_clean(
) -> bool:
    return (
        git_output(
            "status",
            "--porcelain",
        )
        ==
        ""
    )


def git_is_ancestor(
    ancestor: str,
    descendant: str,
) -> bool:
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        cwd=
            ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    return (
        result.returncode
        ==
        0
    )


def git_blob_bytes(
    path: str,
) -> bytes:
    result = subprocess.run(
        [
            "git",
            "show",
            f"HEAD:{path}",
        ],
        cwd=
            ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            (
                "Unable to read committed Git blob: "
                f"{path}"
            )
        )

    return result.stdout


def git_blob_sha256(
    path: str,
) -> str:
    return hashlib.sha256(
        git_blob_bytes(
            path
        )
    ).hexdigest()


def validate_static_contract(
) -> dict[
    str,
    Any,
]:
    manifest = load_json_object(
        MANIFEST_PATH
    )

    freeze = load_json_object(
        MANIFEST_FREEZE_PATH
    )

    protocol_freeze = load_json_object(
        PROTOCOL_FREEZE_PATH
    )

    if (
        manifest[
            "manifest_rule_version"
        ]
        !=
        REASONING_EVALUATION_MANIFEST_RULE_VERSION
    ):
        raise RuntimeError(
            "Evaluation manifest rule mismatch."
        )

    if (
        freeze[
            "status"
        ]
        !=
        "frozen"
    ):
        raise RuntimeError(
            "Evaluation execution manifest is not frozen."
        )

    if (
        freeze[
            "first_post_training_inference_completed"
        ]
        is not False
    ):
        raise RuntimeError(
            "Manifest freeze claims inference already occurred."
        )

    if (
        protocol_freeze[
            "first_post_training_inference_completed"
        ]
        is not False
    ):
        raise RuntimeError(
            "Reasoning protocol claims inference already occurred."
        )

    if (
        protocol_freeze[
            "frozen_before_first_post_training_inference"
        ]
        is not True
    ):
        raise RuntimeError(
            "Reasoning protocol was not frozen pre-inference."
        )

    if (
        manifest[
            "benchmark"
        ][
            "benchmark_id"
        ]
        !=
        BENCHMARK_ID
    ):
        raise RuntimeError(
            "Benchmark ID mismatch."
        )

    if (
        manifest[
            "benchmark"
        ][
            "benchmark_version"
        ]
        !=
        BENCHMARK_VERSION
    ):
        raise RuntimeError(
            "Benchmark version mismatch."
        )

    if (
        manifest[
            "scoring"
        ][
            "rule_version"
        ]
        !=
        ADAPTED_REASONING_SCORING_RULE_VERSION
    ):
        raise RuntimeError(
            "Scoring rule mismatch."
        )

    if (
        manifest[
            "scoring"
        ][
            "max_sequence_length"
        ]
        !=
        MAX_SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            "Evaluation sequence-length contract mismatch."
        )

    if (
        manifest[
            "scoring"
        ][
            "generation_used"
        ]
        is not False
    ):
        raise RuntimeError(
            "Free generation is forbidden."
        )

    if (
        manifest[
            "scoring"
        ][
            "llm_judge_used"
        ]
        is not False
    ):
        raise RuntimeError(
            "LLM judge is forbidden."
        )

    expected_adapter_files = manifest[
        "adapter"
    ][
        "files"
    ]

    observed_adapter_files = directory_snapshot(
        ADAPTER_PATH
    )

    if (
        observed_adapter_files
        !=
        expected_adapter_files
    ):
        raise RuntimeError(
            "Local adapter files changed after manifest freeze."
        )

    expected_model_files = manifest[
        "converted_base_model"
    ][
        "files"
    ]

    observed_model_files = directory_snapshot(
        CONVERTED_MODEL_PATH
    )

    if (
        observed_model_files
        !=
        expected_model_files
    ):
        raise RuntimeError(
            "Converted base checkpoint changed after manifest freeze."
        )

    protocol_hashes = manifest[
        "protocol"
    ][
        "sha256"
    ]

    protocol_paths = {
        "module":
            (
                ROOT
                / "app"
                / "adaptation"
                / "reasoning_benchmark.py"
            ),

        "test":
            (
                ROOT
                / "test_adapted_reasoning_benchmark_v0_1.py"
            ),

        "cases":
            CASES_PATH,

        "freeze":
            PROTOCOL_FREEZE_PATH,
    }

    for key, path in protocol_paths.items():
        if (
            sha256_file(
                path
            )
            !=
            protocol_hashes[
                key
            ]
        ):
            raise RuntimeError(
                (
                    "Frozen reasoning protocol changed: "
                    f"{key}"
                )
            )

    if REPORT_PATH.exists():
        raise RuntimeError(
            "Evaluation report already exists."
        )

    if RECEIPT_PATH.exists():
        raise RuntimeError(
            "Evaluation receipt already exists."
        )

    return manifest


def authorize_execution(
) -> dict[
    str,
    Any,
]:
    manifest = validate_static_contract()

    if not git_worktree_clean():
        raise RuntimeError(
            "Working tree must be clean before evaluation."
        )

    head = git_head()

    protocol_commit = manifest[
        "protocol"
    ][
        "git_commit"
    ]

    if not git_is_ancestor(
        protocol_commit,
        head,
    ):
        raise RuntimeError(
            "Frozen benchmark commit is not an ancestor of HEAD."
        )

    committed_hashes = {
        "runner":
            git_blob_sha256(
                RUNNER_REPO_PATH
            ),

        "test":
            git_blob_sha256(
                TEST_REPO_PATH
            ),

        "manifest":
            git_blob_sha256(
                MANIFEST_REPO_PATH
            ),

        "manifest_freeze":
            git_blob_sha256(
                MANIFEST_FREEZE_REPO_PATH
            ),
    }

    expected_hashes = {
        "runner":
            manifest[
                "execution_code"
            ][
                "runner_sha256"
            ],

        "test":
            manifest[
                "execution_code"
            ][
                "test_sha256"
            ],

        "manifest":
            freeze_sha_binding(
                manifest_path=
                    MANIFEST_PATH,
            ),

        "manifest_freeze":
            sha256_file(
                MANIFEST_FREEZE_PATH
            ),
    }

    for key in (
        "runner",
        "test",
    ):
        if (
            committed_hashes[
                key
            ]
            !=
            expected_hashes[
                key
            ]
        ):
            raise RuntimeError(
                (
                    "Committed evaluation code differs "
                    f"from frozen manifest: {key}"
                )
            )

    if (
        committed_hashes[
            "manifest"
        ]
        !=
        sha256_file(
            MANIFEST_PATH
        )
    ):
        raise RuntimeError(
            "Committed manifest differs from working tree."
        )

    if (
        committed_hashes[
            "manifest_freeze"
        ]
        !=
        sha256_file(
            MANIFEST_FREEZE_PATH
        )
    ):
        raise RuntimeError(
            "Committed manifest freeze differs from working tree."
        )

    return manifest


def freeze_sha_binding(
    *,
    manifest_path: Path,
) -> str:
    freeze = load_json_object(
        MANIFEST_FREEZE_PATH
    )

    actual = sha256_file(
        manifest_path
    )

    expected = freeze[
        "manifest_sha256"
    ]

    if actual != expected:
        raise RuntimeError(
            "Manifest SHA does not match manifest freeze."
        )

    return expected


def _to_token_list(
    value: Any,
) -> list[
    int,
]:
    if hasattr(
        value,
        "tolist",
    ):
        value = value.tolist()

    if (
        isinstance(
            value,
            list,
        )
        and
        value
        and
        isinstance(
            value[
                0
            ],
            list,
        )
    ):
        if len(
            value
        ) != 1:
            raise RuntimeError(
                "Expected one token sequence."
            )

        value = value[
            0
        ]

    if not isinstance(
        value,
        list,
    ):
        raise TypeError(
            "Tokenizer did not return a token list."
        )

    if not all(
        isinstance(
            token,
            int,
        )
        for token
        in value
    ):
        raise TypeError(
            "Tokenizer returned non-integer token IDs."
        )

    return list(
        value
    )


def _encode_candidate(
    *,
    tokenizer: Any,
    case: AdaptedReasoningCase,
    relation: str,
) -> dict[
    str,
    Any,
]:
    prompt = render_reasoning_prompt(
        case
    )

    continuation = (
        candidate_continuation(
            relation
        )
    )

    prompt_tokens = _to_token_list(
        tokenizer.apply_chat_template(
            [
                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },
            ],

            tokenize=True,

            add_generation_prompt=True,

            return_dict=False,
        )
    )

    full_tokens = _to_token_list(
        tokenizer.apply_chat_template(
            [
                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },

                {
                    "role":
                        "assistant",

                    "content":
                        continuation,
                },
            ],

            tokenize=True,

            add_generation_prompt=False,

            return_dict=False,
        )
    )

    if (
        full_tokens[
            :len(
                prompt_tokens
            )
        ]
        !=
        prompt_tokens
    ):
        raise RuntimeError(
            (
                "Gemma full conversation does not "
                "preserve the generation-prompt prefix."
            )
        )

    if (
        len(
            full_tokens
        )
        >
        MAX_SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            (
                "Reasoning candidate exceeds frozen "
                "sequence-length limit. "
                f"case={case.case_id} "
                f"relation={relation} "
                f"tokens={len(full_tokens)}"
            )
        )

    assistant_token_count = (
        len(
            full_tokens
        )
        -
        len(
            prompt_tokens
        )
    )

    if assistant_token_count <= 0:
        raise RuntimeError(
            "Assistant continuation has no scored tokens."
        )

    return {
        "relation":
            relation,

        "prompt_token_count":
            len(
                prompt_tokens
            ),

        "assistant_token_count":
            assistant_token_count,

        "full_token_count":
            len(
                full_tokens
            ),

        "input_ids":
            full_tokens,
    }


def _score_case(
    *,
    model: Any,
    tokenizer: Any,
    case: AdaptedReasoningCase,
    torch_module: Any,
) -> dict[
    str,
    float,
]:
    encoded = [
        _encode_candidate(
            tokenizer=
                tokenizer,

            case=
                case,

            relation=
                relation,
        )

        for relation
        in ALLOWED_RELATIONS
    ]

    pad_token_id = (
        tokenizer.pad_token_id
    )

    if pad_token_id is None:
        raise RuntimeError(
            "Tokenizer pad_token_id is unavailable."
        )

    max_length = max(
        item[
            "full_token_count"
        ]
        for item
        in encoded
    )

    input_ids = torch_module.full(
        (
            len(
                encoded
            ),
            max_length,
        ),

        int(
            pad_token_id
        ),

        dtype=
            torch_module.long,

        device=
            "cuda",
    )

    attention_mask = torch_module.zeros(
        (
            len(
                encoded
            ),
            max_length,
        ),

        dtype=
            torch_module.long,

        device=
            "cuda",
    )

    for row, item in enumerate(
        encoded
    ):
        token_ids = torch_module.tensor(
            item[
                "input_ids"
            ],

            dtype=
                torch_module.long,

            device=
                "cuda",
        )

        length = token_ids.shape[
            0
        ]

        input_ids[
            row,
            :length,
        ] = token_ids

        attention_mask[
            row,
            :length,
        ] = 1

    with torch_module.inference_mode():
        output = model(
            input_ids=
                input_ids,

            attention_mask=
                attention_mask,

            use_cache=False,
        )

        log_probs = (
            torch_module.log_softmax(
                output.logits.float(),
                dim=-1,
            )
        )

    scores = {}

    for row, item in enumerate(
        encoded
    ):
        start = item[
            "prompt_token_count"
        ]

        end = item[
            "full_token_count"
        ]

        if start < 1:
            raise RuntimeError(
                "Prompt must contain at least one token."
            )

        prediction_log_probs = (
            log_probs[
                row,
                start - 1:
                end - 1,
                :,
            ]
        )

        targets = input_ids[
            row,
            start:
            end,
        ]

        if (
            prediction_log_probs.shape[
                0
            ]
            !=
            targets.shape[
                0
            ]
        ):
            raise RuntimeError(
                "Teacher-forced score alignment failed."
            )

        token_scores = (
            prediction_log_probs
            .gather(
                1,
                targets.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )

        mean_score = float(
            token_scores.mean().item()
        )

        if not math.isfinite(
            mean_score
        ):
            raise RuntimeError(
                "Non-finite candidate log probability."
            )

        scores[
            item[
                "relation"
            ]
        ] = mean_score

    # Reuse the frozen fail-closed selector here.
    select_relation_from_scores(
        scores=
            scores,
    )

    return scores


def _evaluate_candidate_model(
    *,
    model: Any,
    tokenizer: Any,
    cases: tuple[
        AdaptedReasoningCase,
        ...,
    ],
    torch_module: Any,
) -> dict[
    str,
    Any,
]:
    case_results = []

    correct_count = 0

    per_class_total = Counter()

    per_class_correct = Counter()

    for index, case in enumerate(
        cases,
        start=1,
    ):
        scores = _score_case(
            model=
                model,

            tokenizer=
                tokenizer,

            case=
                case,

            torch_module=
                torch_module,
        )

        predicted = (
            select_relation_from_scores(
                scores=
                    scores,
            )
        )

        correct = (
            predicted
            ==
            case.expected_relation
        )

        per_class_total[
            case.expected_relation
        ] += 1

        if correct:
            correct_count += 1

            per_class_correct[
                case.expected_relation
            ] += 1

        case_results.append(
            {
                "case_id":
                    case.case_id,

                "correct":
                    correct,

                "expected_relation":
                    case.expected_relation,

                "predicted_relation":
                    predicted,

                "scores":
                    {
                        relation:
                            round(
                                scores[
                                    relation
                                ],
                                8,
                            )

                        for relation
                        in ALLOWED_RELATIONS
                    },
            }
        )

        print(
            (
                f"  case {index:02d}/"
                f"{len(cases):02d} "
                f"{case.case_id}: "
                f"{predicted}"
            )
        )

    accuracy = (
        correct_count
        /
        len(
            cases
        )
    )

    per_class_accuracy = {
        relation:
            (
                per_class_correct[
                    relation
                ]
                /
                per_class_total[
                    relation
                ]
            )

        for relation
        in ALLOWED_RELATIONS
    }

    macro_accuracy = (
        sum(
            per_class_accuracy.values()
        )
        /
        len(
            ALLOWED_RELATIONS
        )
    )

    return {
        "accuracy":
            round(
                accuracy,
                6,
            ),

        "case_count":
            len(
                cases
            ),

        "correct_count":
            correct_count,

        "macro_accuracy":
            round(
                macro_accuracy,
                6,
            ),

        "per_class_accuracy":
            {
                relation:
                    round(
                        value,
                        6,
                    )

                for relation, value
                in per_class_accuracy.items()
            },

        "results":
            case_results,
    }


def _paired_comparison(
    *,
    base: dict[
        str,
        Any,
    ],
    adapted: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    base_by_id = {
        item[
            "case_id"
        ]:
            item

        for item
        in base[
            "results"
        ]
    }

    adapted_by_id = {
        item[
            "case_id"
        ]:
            item

        for item
        in adapted[
            "results"
        ]
    }

    if (
        set(
            base_by_id
        )
        !=
        set(
            adapted_by_id
        )
    ):
        raise RuntimeError(
            "Base/adapted case IDs differ."
        )

    outcomes = Counter()

    changed_predictions = 0

    for case_id in sorted(
        base_by_id
    ):
        base_case = base_by_id[
            case_id
        ]

        adapted_case = adapted_by_id[
            case_id
        ]

        if (
            base_case[
                "predicted_relation"
            ]
            !=
            adapted_case[
                "predicted_relation"
            ]
        ):
            changed_predictions += 1

        if (
            base_case[
                "correct"
            ]
            and
            adapted_case[
                "correct"
            ]
        ):
            outcomes[
                "both_correct"
            ] += 1

        elif (
            not
            base_case[
                "correct"
            ]
            and
            not
            adapted_case[
                "correct"
            ]
        ):
            outcomes[
                "both_wrong"
            ] += 1

        elif base_case[
            "correct"
        ]:
            outcomes[
                "base_only_correct"
            ] += 1

        else:
            outcomes[
                "adapted_only_correct"
            ] += 1

    accuracy_delta = round(
        (
            adapted[
                "accuracy"
            ]
            -
            base[
                "accuracy"
            ]
        ),
        6,
    )

    macro_delta = round(
        (
            adapted[
                "macro_accuracy"
            ]
            -
            base[
                "macro_accuracy"
            ]
        ),
        6,
    )

    if (
        accuracy_delta < 0
        or
        macro_delta < 0
    ):
        signal = (
            "negative_signal"
        )

    elif (
        accuracy_delta == 0
        and
        macro_delta == 0
    ):
        signal = (
            "neutral_signal"
        )

    else:
        signal = (
            "positive_signal"
        )

    return {
        "accuracy_delta":
            accuracy_delta,

        "adapted_only_correct":
            outcomes[
                "adapted_only_correct"
            ],

        "base_only_correct":
            outcomes[
                "base_only_correct"
            ],

        "both_correct":
            outcomes[
                "both_correct"
            ],

        "both_wrong":
            outcomes[
                "both_wrong"
            ],

        "changed_prediction_count":
            changed_predictions,

        "macro_accuracy_delta":
            macro_delta,

        "preregistered_signal":
            signal,
    }


def execute_evaluation(
) -> dict[
    str,
    Any,
]:
    # CRITICAL:
    # Runtime authorization occurs before all heavy ML imports.
    manifest = authorize_execution()

    import torch

    from peft import (
        PeftModel,
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )


    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the frozen evaluation."
        )

    free_bytes, total_bytes = (
        torch.cuda.mem_get_info()
    )

    if (
        free_bytes
        <
        MINIMUM_FREE_CUDA_BYTES
    ):
        raise RuntimeError(
            (
                "Insufficient free CUDA memory. "
                f"free={free_bytes / 1024**3:.2f} GiB "
                f"required="
                f"{MINIMUM_FREE_CUDA_BYTES / 1024**3:.2f} GiB"
            )
        )

    print()

    print(
        "=== DATALENS ADAPTED REASONING EVALUATION v0.1 ==="
    )

    print(
        (
            "CUDA free before load: "
            f"{free_bytes / 1024**3:.2f} GiB"
        )
    )


    torch.manual_seed(
        42
    )

    torch.cuda.manual_seed_all(
        42
    )

    torch.cuda.reset_peak_memory_stats()


    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL_REPOSITORY,

            revision=
                BASE_MODEL_REVISION,

            trust_remote_code=
                False,

            local_files_only=
                True,
        )
    )


    if tokenizer.chat_template is None:
        raise RuntimeError(
            "Gemma tokenizer chat template is unavailable."
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
            (
                "Gemma chat template hash mismatch. "
                f"Observed={chat_template_sha256}"
            )
        )


    tokenizer.padding_side = (
        "right"
    )


    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Gemma tokenizer has no pad token."
        )


    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=
                True,

            bnb_4bit_quant_type=
                "nf4",

            bnb_4bit_use_double_quant=
                True,

            bnb_4bit_compute_dtype=
                torch.bfloat16,
        )
    )


    model = (
        AutoModelForCausalLM
        .from_pretrained(
            str(
                CONVERTED_MODEL_PATH
            ),

            device_map={
                "":
                    0,
            },

            dtype=
                torch.bfloat16,

            quantization_config=
                quantization_config,

            trust_remote_code=
                False,

            local_files_only=
                True,
        )
    )


    model.eval()

    model.config.use_cache = (
        False
    )


    cases = (
        load_frozen_reasoning_cases(
            path=
                CASES_PATH,
        )
    )


    print()

    print(
        "BASE GEMMA"
    )


    base_result = (
        _evaluate_candidate_model(
            model=
                model,

            tokenizer=
                tokenizer,

            cases=
                cases,

            torch_module=
                torch,
        )
    )


    print()

    print(
        "ATTACHING FROZEN QLoRA ADAPTER"
    )


    adapted_model = (
        PeftModel
        .from_pretrained(
            model,

            str(
                ADAPTER_PATH
            ),

            is_trainable=
                False,
        )
    )


    adapted_model.eval()


    print()

    print(
        "ADAPTED GEMMA"
    )


    adapted_result = (
        _evaluate_candidate_model(
            model=
                adapted_model,

            tokenizer=
                tokenizer,

            cases=
                cases,

            torch_module=
                torch,
        )
    )


    paired = _paired_comparison(
        base=
            base_result,

        adapted=
            adapted_result,
    )


    peak_allocated = (
        torch.cuda.max_memory_allocated()
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
    )


    created_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


    report = {
        "acceptance_authority":
            False,

        "adapter_bundle_sha256":
            manifest[
                "adapter"
            ][
                "official_bundle_sha256"
            ],

        "adapted":
            adapted_result,

        "base":
            base_result,

        "benchmark_id":
            BENCHMARK_ID,

        "benchmark_version":
            BENCHMARK_VERSION,

        "created_at":
            created_at,

        "execution_git_commit":
            git_head(),

        "final_acceptance_evaluated":
            False,

        "final_acceptance_loaded":
            False,

        "generation_used":
            False,

        "llm_judge_used":
            False,

        "manifest_sha256":
            sha256_file(
                MANIFEST_PATH
            ),

        "memory": {
            "cuda_total_bytes":
                int(
                    total_bytes
                ),

            "free_cuda_bytes_before_load":
                int(
                    free_bytes
                ),

            "peak_allocated_bytes":
                int(
                    peak_allocated
                ),

            "peak_reserved_bytes":
                int(
                    peak_reserved
                ),
        },

        "paired":
            paired,

        "protocol_commit":
            manifest[
                "protocol"
            ][
                "git_commit"
            ],

        "protocol_freeze_sha256":
            manifest[
                "protocol"
            ][
                "sha256"
            ][
                "freeze"
            ],

        "runner_rule_version":
            REASONING_EVALUATION_RUNNER_RULE_VERSION,

        "scoring": {
            "assistant_span_definition":
                (
                    "all full-chat token IDs after the "
                    "exact add_generation_prompt=True "
                    "user-only prefix"
                ),

            "candidate_selection":
                "argmax_mean_log_probability",

            "generation_used":
                False,

            "llm_judge_used":
                False,

            "max_sequence_length":
                MAX_SEQUENCE_LENGTH,

            "rule_version":
                ADAPTED_REASONING_SCORING_RULE_VERSION,

            "silent_truncation_allowed":
                False,

            "ties":
                "fail_closed",
        },

        "status":
            "completed",

        "training_loss_used_as_acceptance_evidence":
            False,
    }


    atomic_write_json(
        path=
            REPORT_PATH,

        payload=
            report,
    )


    report_sha256 = sha256_file(
        REPORT_PATH
    )


    receipt = {
        "adapter_bundle_sha256":
            manifest[
                "adapter"
            ][
                "official_bundle_sha256"
            ],

        "benchmark_id":
            BENCHMARK_ID,

        "created_at":
            created_at,

        "evaluation_report_sha256":
            report_sha256,

        "final_acceptance_evaluated":
            False,

        "final_acceptance_loaded":
            False,

        "manifest_sha256":
            sha256_file(
                MANIFEST_PATH
            ),

        "receipt_rule_version":
            REASONING_EVALUATION_RECEIPT_RULE_VERSION,

        "runner_rule_version":
            REASONING_EVALUATION_RUNNER_RULE_VERSION,

        "status":
            "completed",
    }


    atomic_write_json(
        path=
            RECEIPT_PATH,

        payload=
            receipt,
    )


    print()

    print(
        "RESULT"
    )

    print(
        (
            "  Base accuracy: "
            f"{base_result['accuracy']:.6f}"
        )
    )

    print(
        (
            "  Adapted accuracy: "
            f"{adapted_result['accuracy']:.6f}"
        )
    )

    print(
        (
            "  Accuracy delta: "
            f"{paired['accuracy_delta']:+.6f}"
        )
    )

    print(
        (
            "  Base macro accuracy: "
            f"{base_result['macro_accuracy']:.6f}"
        )
    )

    print(
        (
            "  Adapted macro accuracy: "
            f"{adapted_result['macro_accuracy']:.6f}"
        )
    )

    print(
        (
            "  Macro delta: "
            f"{paired['macro_accuracy_delta']:+.6f}"
        )
    )

    print(
        (
            "  Preregistered signal: "
            f"{paired['preregistered_signal']}"
        )
    )

    print(
        (
            "  Report SHA256: "
            f"{report_sha256}"
        )
    )

    print(
        (
            "  Receipt SHA256: "
            f"{sha256_file(RECEIPT_PATH)}"
        )
    )


    print()

    print(
        "SAFETY"
    )

    print(
        "  Training executed: False"
    )

    print(
        "  Optimizer created: False"
    )

    print(
        "  Backward executed: False"
    )

    print(
        "  Free generation used: False"
    )

    print(
        "  LLM judge used: False"
    )

    print(
        "  Final Acceptance loaded: False"
    )

    print(
        "  Final Acceptance evaluated: False"
    )


    print()

    print(
        "DATALENS ADAPTED REASONING EVALUATION v0.1: PASS"
    )


    return report


def main(
) -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",

        choices=(
            "validate-static",
            "authorize-only",
            "execute",
        ),
    )

    arguments = parser.parse_args()


    if (
        arguments.mode
        ==
        "validate-static"
    ):
        validate_static_contract()

        print(
            "DATALENS REASONING EVALUATION STATIC VALIDATION: PASS"
        )

        return


    if (
        arguments.mode
        ==
        "authorize-only"
    ):
        authorize_execution()

        print(
            "DATALENS REASONING EVALUATION AUTHORIZATION: PASS"
        )

        return


    execute_evaluation()


if __name__ == "__main__":
    main()

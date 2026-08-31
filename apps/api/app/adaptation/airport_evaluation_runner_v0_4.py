from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.adaptation.airport_ground_operations_holdout import (
    RELATIONS,
    validate_cases,
    validate_freeze,
)
from app.adaptation.qlora_runtime_v0_4 import (
    QLORA_V04_SHARED_RUNTIME_RULE_VERSION,
    load_pinned_tokenizer,
    local_text_checkpoint_path,
    runtime_versions,
    validate_static_authority,
)
from app.adaptation.training_dataset_canonicalizer_v0_4 import (
    build_user_message,
)


AIRPORT_EVALUATION_RUNNER_RULE_VERSION = (
    "qlora_v0.4_airport_evaluation_runner_v0.1"
)

AIRPORT_EVALUATION_MANIFEST_RULE_VERSION = (
    "qlora_v0.4_airport_evaluation_manifest_v0.1"
)

AIRPORT_EVALUATION_MANIFEST_FREEZE_RULE_VERSION = (
    "qlora_v0.4_airport_evaluation_manifest_freeze_v0.1"
)

AIRPORT_EVALUATION_RECEIPT_RULE_VERSION = (
    "qlora_v0.4_airport_evaluation_receipt_v0.1"
)

AIRPORT_CONSUMPTION_MARKER_RULE_VERSION = (
    "qlora_v0.4_airport_holdout_consumption_v0.1"
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
)


PROTOCOL_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.4_"
      "airport_evaluation_protocol_v0.1.json"
)

MANIFEST_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.4_"
      "airport_evaluation_v0.1_manifest.json"
)

MANIFEST_FREEZE_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.4_"
      "airport_evaluation_v0.1_manifest_freeze.json"
)

REPORT_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.4_"
      "airport_evaluation_v0.1_report.json"
)

RECEIPT_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.4_"
      "airport_evaluation_v0.1_receipt.json"
)

CONSUMPTION_MARKER_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.4_"
      "airport_evaluation_v0.1_consumption.json"
)


AIRPORT_CASES_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "holdouts"
    / "datalens_semantic_qlora_v0.4_"
      "airport_ground_operations_holdout_v0.1_cases.json"
)

AIRPORT_FREEZE_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "holdouts"
    / "datalens_semantic_qlora_v0.4_"
      "airport_ground_operations_holdout_v0.1_freeze.json"
)

AIRPORT_MODULE_PATH = (
    ROOT
    / "app"
    / "adaptation"
    / "airport_ground_operations_holdout.py"
)

AIRPORT_MODULE_TEST_PATH = (
    ROOT
    / "test_airport_ground_operations_holdout_v0_1.py"
)


POST_HOTEL_DECISION_PATH = (
    ARTIFACT_DIR
    / "datalens_semantic_qlora_v0.4_"
      "post_hotel_decision_v0.1.json"
)

TRAINING_REPORT_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "training"
    / "datalens_semantic_qlora_v0.4_"
      "training_v0.1_report.json"
)

TRAINING_RECEIPT_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "training"
    / "datalens_semantic_qlora_v0.4_"
      "training_v0.1_receipt.json"
)

ADAPTER_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "adapters"
    / "datalens_semantic_qlora_v0.4_adapter"
)

CONVERTED_MODEL_PATH = (
    local_text_checkpoint_path()
)


RUNNER_REPO_PATH = (
    "apps/api/app/adaptation/"
    "airport_evaluation_runner_v0_4.py"
)

TEST_REPO_PATH = (
    "apps/api/"
    "test_airport_evaluation_runner_v0_4_v0_1.py"
)

PROTOCOL_REPO_PATH = (
    "apps/api/artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "airport_evaluation_protocol_v0.1.json"
)

MANIFEST_REPO_PATH = (
    "apps/api/artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "airport_evaluation_v0.1_manifest.json"
)

MANIFEST_FREEZE_REPO_PATH = (
    "apps/api/artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "airport_evaluation_v0.1_manifest_freeze.json"
)


EXPECTED_PROTOCOL_GIT_COMMIT = (
    "f6b8063808badc05a4cc93abe80323d1d745f17d"
)

EXPECTED_PROTOCOL_SHA256 = (
    "a88cb86565c0a3d4daa4d5b2bea3beee"
    "d79a7df81155ed4d4ed0c7bfef2e00d1"
)

EXPECTED_POST_HOTEL_DECISION_SHA256 = (
    "54a8a880c37359d9734fb2e08dce3418"
    "a94014d8dde5a1fd7f49e26ec4173c6f"
)

EXPECTED_AIRPORT_FREEZE_SHA256 = (
    "46accf23eeae32f0fdc926f7b0a9e731"
    "15a1413ddbc28efa84472f0571ee2f09"
)

EXPECTED_AIRPORT_CASES_SHA256 = (
    "7d9454d0d59dd047050bd72d195feb139"
    "0ca0edc2c83584af0cbfec951cf3939"
)

EXPECTED_AIRPORT_MODULE_SHA256 = (
    "4326564a101321f49ec375aca1d67e057"
    "35841c298132843ff774f8f013ec961"
)

EXPECTED_AIRPORT_TEST_SHA256 = (
    "54e426a2d2b432395f4660e5e591070c"
    "b48b8b1e3deed0faf75ac4bb9cb1b9bc"
)

EXPECTED_SHARED_RUNTIME_SHA256 = (
    "20e41ab00606296893276a84e53746c0"
    "6618b8cabca74fef77cb743c5e80ab7c"
)

EXPECTED_CANONICALIZER_SHA256 = (
    "075d6c22cf4473b414221a0766ca831f"
    "4da1e19597a0a5b2b5efd3e5755c9356"
)

EXPECTED_TRAINING_REPORT_SHA256 = (
    "759ba4957806daab8b7a14d3aeb2b068"
    "59e0bcd6193d30cb877b63748617e04d"
)

EXPECTED_TRAINING_RECEIPT_SHA256 = (
    "f412062f78432d7c432d4b36beed9d84"
    "d527d5990279240030a0a31227dffaee"
)

EXPECTED_S3_REPORT_SHA256 = (
    "b0b662c31f7bbc013968cd7a69968aba"
    "21621c13cfe8de72bef2f4ddde0c1e6c"
)

EXPECTED_S3_RECEIPT_SHA256 = (
    "fb4d81feb1c20c77e8fe3eaaf491ca7"
    "f831c32c96a6dcf29389e7b06dae28b96"
)

EXPECTED_ADAPTER_BUNDLE_SHA256 = (
    "0351980df6d86096195c0971deb30c725"
    "e155c71aa5de8054b2b37fa42090716"
)

EXPECTED_ADAPTER_FILES = {
    "README.md":
        (
            "6ecdbb662eaed8010ab0e012a2b95b79"
            "543884cf294406dc6da2cde64f98389d"
        ),

    "adapter_config.json":
        (
            "3ae14896612f6bf74ee7786a450e2ac0"
            "f08f3da9f33391505cb1a7dc823dcdb8"
        ),

    "adapter_model.safetensors":
        (
            "4f145b0bf37f67841c09f02b86679634"
            "a9532491d2f560b0e7c5c328009e4610"
        ),
}

EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "7de1c58e208eda46e9c7f86397df37ec"
    "49883aeece39fb961e0a6b24088dd3c4"
)


BASE_MODEL_REPOSITORY = (
    "google/gemma-3-4b-it"
)

BASE_MODEL_REVISION = (
    "093f9f388b31de276ce2de164bdc2081324b9767"
)


EXPECTED_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


MAX_NEW_TOKENS = 64

EOS_TOKEN_IDS = (
    1,
    106,
)

PAD_TOKEN_ID = 0

MINIMUM_FREE_CUDA_BYTES = (
    5
    *
    1024**3
)

ROUND_DECIMAL_PLACES = 6


# ============================================================
# GENERIC FILE / GIT HELPERS
# ============================================================


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda:
                handle.read(
                    8
                    *
                    1024
                    *
                    1024
                ),
            b"",
        ):
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
    if not path.is_dir():
        raise NotADirectoryError(
            path
        )

    result: dict[
        str,
        str,
    ] = {}

    for item in sorted(
        path.iterdir(),
        key=lambda value:
            value.name,
    ):
        if not item.is_file():
            raise RuntimeError(
                (
                    "Unexpected non-file in "
                    f"frozen directory: {item}"
                )
            )

        result[
            item.name
        ] = sha256_file(
            item
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
            encoding=
                "utf-8-sig"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            (
                "Expected JSON object: "
                f"{path}"
            )
        )

    return payload


def atomic_write_json(
    *,
    path: Path,
    payload: object,
) -> None:
    if path.exists():
        raise FileExistsError(
            path
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        +
        ".tmp"
    )

    if temporary.exists():
        raise FileExistsError(
            temporary
        )

    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )

    try:
        with temporary.open(
            "xb"
        ) as handle:
            handle.write(
                encoded
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary,
            path,
        )

    finally:
        if temporary.exists():
            temporary.unlink()


def git_output(
    *arguments: str,
    binary: bool = False,
) -> Any:
    repository = subprocess.run(
        [
            "git",
            "rev-parse",
            "--show-toplevel",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if repository.returncode != 0:
        raise RuntimeError(
            repository.stderr
        )

    repository_root = Path(
        repository.stdout.strip()
    ).resolve()

    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=
            not binary,
        check=False,
    )

    if result.returncode != 0:
        error = (
            result.stderr.decode(
                errors="replace"
            )
            if binary
            else result.stderr
        )

        raise RuntimeError(
            error
        )

    return result.stdout


def git_head(
) -> str:
    return str(
        git_output(
            "rev-parse",
            "HEAD",
        )
    ).strip()


def git_worktree_clean(
) -> bool:
    return not bool(
        str(
            git_output(
                "status",
                "--porcelain",
            )
        ).strip()
    )


def git_is_ancestor(
    ancestor: str,
    descendant: str,
) -> bool:
    repository = subprocess.run(
        [
            "git",
            "rev-parse",
            "--show-toplevel",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        cwd=
            Path(
                repository
                .stdout
                .strip()
            ).resolve(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
    return bytes(
        git_output(
            "show",
            f"HEAD:{path}",
            binary=True,
        )
    )


def git_blob_sha256(
    path: str,
) -> str:
    return hashlib.sha256(
        git_blob_bytes(
            path
        )
    ).hexdigest()


# ============================================================
# STATIC AUTHORITY
# ============================================================


def _require_exact_sha(
    path: Path,
    expected: str,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual = sha256_file(
        path
    )

    if actual != expected:
        raise RuntimeError(
            (
                f"{label} SHA mismatch.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )
        )


def _find_unique_json_by_sha(
    *,
    directory: Path,
    expected_sha256: str,
    label: str,
) -> Path:
    matches = []

    for path in sorted(
        directory.glob(
            "*.json"
        )
    ):
        if not path.is_file():
            continue

        if (
            sha256_file(
                path
            )
            ==
            expected_sha256
        ):
            matches.append(
                path
            )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            (
                f"{label} SHA authority "
                "resolved to "
                f"{len(matches)} files."
            )
        )

    return matches[
        0
    ]


def _validate_prerequisite_evidence(
) -> None:
    _require_exact_sha(
        POST_HOTEL_DECISION_PATH,
        EXPECTED_POST_HOTEL_DECISION_SHA256,
        "Post-Hotel Decision",
    )

    _require_exact_sha(
        AIRPORT_FREEZE_PATH,
        EXPECTED_AIRPORT_FREEZE_SHA256,
        "Airport freeze",
    )

    _require_exact_sha(
        AIRPORT_MODULE_PATH,
        EXPECTED_AIRPORT_MODULE_SHA256,
        "Airport module",
    )

    _require_exact_sha(
        AIRPORT_MODULE_TEST_PATH,
        EXPECTED_AIRPORT_TEST_SHA256,
        "Airport holdout test",
    )

    _require_exact_sha(
        TRAINING_REPORT_PATH,
        EXPECTED_TRAINING_REPORT_SHA256,
        "Training report",
    )

    _require_exact_sha(
        TRAINING_RECEIPT_PATH,
        EXPECTED_TRAINING_RECEIPT_SHA256,
        "Training receipt",
    )

    s3_report_path = (
        _find_unique_json_by_sha(
            directory=
                ARTIFACT_DIR,

            expected_sha256=
                EXPECTED_S3_REPORT_SHA256,

            label=
                "S3 report",
        )
    )

    s3_receipt_path = (
        _find_unique_json_by_sha(
            directory=
                ARTIFACT_DIR,

            expected_sha256=
                EXPECTED_S3_RECEIPT_SHA256,

            label=
                "S3 receipt",
        )
    )

    _require_exact_sha(
        PROTOCOL_PATH,
        EXPECTED_PROTOCOL_SHA256,
        "Airport evaluation protocol",
    )

    _require_exact_sha(
        (
            ROOT
            / "app"
            / "adaptation"
            / "qlora_runtime_v0_4.py"
        ),
        EXPECTED_SHARED_RUNTIME_SHA256,
        "Shared QLoRA runtime",
    )

    _require_exact_sha(
        (
            ROOT
            / "app"
            / "adaptation"
            / "training_dataset_canonicalizer_v0_4.py"
        ),
        EXPECTED_CANONICALIZER_SHA256,
        "Training canonicalizer",
    )

    decision = load_json_object(
        POST_HOTEL_DECISION_PATH
    )

    if (
        decision.get(
            "decision",
            {},
        ).get(
            "action"
        )
        !=
        "proceed_to_airport_independent_holdout"
    ):
        raise RuntimeError(
            (
                "Post-Hotel Decision does "
                "not authorize Airport."
            )
        )

    # The exact S3 bytes and the committed Post-Hotel Decision
    # are the authority here. We deliberately do not reinterpret
    # the completed S3 evidence with a new post-hoc schema.
    _ = s3_report_path
    _ = s3_receipt_path

    training_receipt = (
        load_json_object(
            TRAINING_RECEIPT_PATH
        )
    )

    adapter = (
        training_receipt.get(
            "adapter"
        )
    )

    if not isinstance(
        adapter,
        Mapping,
    ):
        raise RuntimeError(
            (
                "Training receipt has "
                "no adapter authority."
            )
        )

    if (
        adapter.get(
            "bundle_sha256"
        )
        !=
        EXPECTED_ADAPTER_BUNDLE_SHA256
    ):
        raise RuntimeError(
            (
                "Official adapter bundle "
                "SHA changed."
            )
        )

    observed_adapter_files = (
        directory_snapshot(
            ADAPTER_PATH
        )
    )

    if (
        observed_adapter_files
        !=
        EXPECTED_ADAPTER_FILES
    ):
        raise RuntimeError(
            (
                "Official adapter directory "
                "bytes changed."
            )
        )

    freeze = load_json_object(
        AIRPORT_FREEZE_PATH
    )

    validate_freeze(
        cases_sha256=
            EXPECTED_AIRPORT_CASES_SHA256,

        freeze=
            freeze,
    )

    if (
        tuple(
            RELATIONS
        )
        !=
        EXPECTED_RELATIONS
    ):
        raise RuntimeError(
            (
                "Airport relation authority "
                "changed."
            )
        )

    if any(
        path.exists()
        for path
        in (
            REPORT_PATH,
            RECEIPT_PATH,
            CONSUMPTION_MARKER_PATH,
        )
    ):
        raise RuntimeError(
            (
                "Airport execution output already "
                "exists; single-use evaluation is closed."
            )
        )


def _validate_protocol_contract(
    protocol: Mapping[
        str,
        Any,
    ],
) -> None:
    if (
        protocol.get(
            "record_rule_version"
        )
        !=
        "qlora_v0.4_airport_evaluation_protocol_v0.1"
    ):
        raise RuntimeError(
            "Airport protocol rule mismatch."
        )

    if (
        protocol.get(
            "status"
        )
        !=
        "preregistered_before_holdout_consumption"
    ):
        raise RuntimeError(
            (
                "Airport protocol was not "
                "preregistered pre-consumption."
            )
        )

    generation = (
        protocol.get(
            "generation"
        )
    )

    if not isinstance(
        generation,
        Mapping,
    ):
        raise RuntimeError(
            "Airport generation contract missing."
        )

    expected_generation = {
        "mode":
            "deterministic_greedy",

        "do_sample":
            False,

        "num_beams":
            1,

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "eos_token_ids":
            list(
                EOS_TOKEN_IDS
            ),

        "input_truncation_permitted":
            False,

        "generation_budget_exhaustion":
            "fail_closed",
    }

    for key, expected in (
        expected_generation.items()
    ):
        if (
            generation.get(
                key
            )
            !=
            expected
        ):
            raise RuntimeError(
                (
                    "Airport generation contract "
                    f"changed: {key}="
                    f"{generation.get(key)!r}"
                )
            )

    output = (
        protocol.get(
            "task",
            {},
        ).get(
            "output_contract"
        )
    )

    if not isinstance(
        output,
        Mapping,
    ):
        raise RuntimeError(
            "Airport output contract missing."
        )

    if (
        output.get(
            "required_keys"
        )
        !=
        [
            "relation",
            "reason",
        ]
    ):
        raise RuntimeError(
            "Airport JSON key contract changed."
        )

    if (
        output.get(
            "additional_properties"
        )
        is not False
    ):
        raise RuntimeError(
            (
                "Airport JSON additional-property "
                "rule changed."
            )
        )

    if (
        output.get(
            "relation_allowed_values"
        )
        !=
        list(
            EXPECTED_RELATIONS
        )
    ):
        raise RuntimeError(
            (
                "Airport relation output "
                "order changed."
            )
        )

    if (
        output.get(
            "reason_word_count_minimum"
        )
        !=
        6
    ):
        raise RuntimeError(
            "Airport reason minimum changed."
        )

    if (
        output.get(
            "reason_word_count_maximum"
        )
        !=
        45
    ):
        raise RuntimeError(
            "Airport reason maximum changed."
        )

    processing = (
        protocol.get(
            "generated_output_processing"
        )
    )

    if not isinstance(
        processing,
        Mapping,
    ):
        raise RuntimeError(
            (
                "Airport output-processing "
                "contract missing."
            )
        )

    if (
        processing.get(
            "decode_scope"
        )
        !=
        "new_tokens_only"
    ):
        raise RuntimeError(
            "Airport decode scope changed."
        )

    if (
        processing.get(
            "decode_skip_special_tokens"
        )
        is not False
    ):
        raise RuntimeError(
            (
                "Airport special-token visibility "
                "changed."
            )
        )

    if (
        processing.get(
            "parser"
        )
        !=
        "json.loads"
    ):
        raise RuntimeError(
            "Airport parser changed."
        )

    gates = protocol.get(
        "acceptance_gates"
    )

    if not isinstance(
        gates,
        Mapping,
    ):
        raise RuntimeError(
            "Airport gate contract missing."
        )

    absolute = gates.get(
        "pretraining_frozen_airport_absolute_gates"
    )

    non_regression = gates.get(
        "post_hotel_non_regression_gates"
    )

    if absolute != {
        "adapted_accuracy_minimum":
            0.70,

        "adapted_macro_accuracy_minimum":
            0.70,

        "adapted_per_relation_accuracy_minimum":
            0.50,

        "adapted_uncertain_accuracy_minimum":
            0.666667,

        "adapted_strict_json_validity_rate":
            1.0,

        "training_loss_is_acceptance_evidence":
            False,
    }:
        raise RuntimeError(
            "Airport absolute gates changed."
        )

    if non_regression != {
        "accuracy_delta_minimum":
            0.0,

        "macro_accuracy_delta_minimum":
            0.0,
    }:
        raise RuntimeError(
            (
                "Airport non-regression "
                "gates changed."
            )
        )

    if (
        gates.get(
            "all_gates_required"
        )
        is not True
    ):
        raise RuntimeError(
            (
                "Airport gates are no "
                "longer conjunctive."
            )
        )

    if (
        gates.get(
            "failure_action"
        )
        !=
        "stop_v0.4_before_greenhouse"
    ):
        raise RuntimeError(
            "Airport failure action changed."
        )


def validate_static_contract(
) -> dict[
    str,
    Any,
]:
    _validate_prerequisite_evidence()

    # Reuse the already-frozen v0.4 runtime authority rather
    # than duplicating its 4-bit/tokenizer/contract checks.
    authority = (
        validate_static_authority(
            repository_root_value=
                ROOT
        )
    )

    if (
        authority.contract
        .base_model
        .repository
        !=
        BASE_MODEL_REPOSITORY
    ):
        raise RuntimeError(
            "Base-model repository changed."
        )

    if (
        authority.contract
        .base_model
        .revision
        !=
        BASE_MODEL_REVISION
    ):
        raise RuntimeError(
            "Base-model revision changed."
        )

    if (
        authority.contract
        .base_model
        .tokenizer_revision
        !=
        BASE_MODEL_REVISION
    ):
        raise RuntimeError(
            "Tokenizer revision changed."
        )

    if (
        authority.contract
        .training
        .random_seed
        !=
        42
    ):
        raise RuntimeError(
            "Frozen random seed changed."
        )

    if (
        QLORA_V04_SHARED_RUNTIME_RULE_VERSION
        !=
        "qlora_v0.4_shared_runtime_v0.1"
    ):
        raise RuntimeError(
            (
                "Shared runtime rule "
                "version changed."
            )
        )

    runtime_versions()

    protocol = load_json_object(
        PROTOCOL_PATH
    )

    _validate_protocol_contract(
        protocol
    )

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            MANIFEST_PATH
        )

    if not MANIFEST_FREEZE_PATH.is_file():
        raise FileNotFoundError(
            MANIFEST_FREEZE_PATH
        )

    manifest = load_json_object(
        MANIFEST_PATH
    )

    freeze = load_json_object(
        MANIFEST_FREEZE_PATH
    )

    if (
        manifest.get(
            "manifest_rule_version"
        )
        !=
        AIRPORT_EVALUATION_MANIFEST_RULE_VERSION
    ):
        raise RuntimeError(
            "Airport manifest rule mismatch."
        )

    if (
        manifest.get(
            "experiment_id"
        )
        !=
        "datalens-semantic-qlora-v0.4"
    ):
        raise RuntimeError(
            (
                "Airport manifest experiment "
                "mismatch."
            )
        )

    if (
        manifest.get(
            "protocol",
            {},
        ).get(
            "sha256"
        )
        !=
        EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            (
                "Manifest Airport protocol "
                "SHA mismatch."
            )
        )

    if (
        manifest.get(
            "protocol",
            {},
        ).get(
            "git_commit"
        )
        !=
        EXPECTED_PROTOCOL_GIT_COMMIT
    ):
        raise RuntimeError(
            (
                "Manifest Airport protocol "
                "commit mismatch."
            )
        )

    execution_code = (
        manifest.get(
            "execution_code"
        )
    )

    if not isinstance(
        execution_code,
        Mapping,
    ):
        raise RuntimeError(
            (
                "Airport execution-code "
                "binding missing."
            )
        )

    if (
        execution_code.get(
            "runner_repo_path"
        )
        !=
        RUNNER_REPO_PATH
    ):
        raise RuntimeError(
            "Airport runner repo path changed."
        )

    if (
        execution_code.get(
            "test_repo_path"
        )
        !=
        TEST_REPO_PATH
    ):
        raise RuntimeError(
            (
                "Airport runner test "
                "repo path changed."
            )
        )

    holdout = manifest.get(
        "holdout"
    )

    if not isinstance(
        holdout,
        Mapping,
    ):
        raise RuntimeError(
            (
                "Airport holdout manifest "
                "section missing."
            )
        )

    if (
        holdout.get(
            "cases_sha256"
        )
        !=
        EXPECTED_AIRPORT_CASES_SHA256
    ):
        raise RuntimeError(
            (
                "Manifest Airport cases "
                "SHA mismatch."
            )
        )

    if (
        holdout.get(
            "freeze_sha256"
        )
        !=
        EXPECTED_AIRPORT_FREEZE_SHA256
    ):
        raise RuntimeError(
            (
                "Manifest Airport freeze "
                "SHA mismatch."
            )
        )

    if (
        holdout.get(
            "case_count"
        )
        !=
        30
    ):
        raise RuntimeError(
            (
                "Manifest Airport case "
                "count changed."
            )
        )

    outputs = manifest.get(
        "outputs"
    )

    expected_outputs = {
        "consumption_marker":
            str(
                CONSUMPTION_MARKER_PATH
                .relative_to(
                    ROOT
                )
            ).replace(
                "\\",
                "/",
            ),

        "report":
            str(
                REPORT_PATH
                .relative_to(
                    ROOT
                )
            ).replace(
                "\\",
                "/",
            ),

        "receipt":
            str(
                RECEIPT_PATH
                .relative_to(
                    ROOT
                )
            ).replace(
                "\\",
                "/",
            ),
    }

    if outputs != expected_outputs:
        raise RuntimeError(
            "Airport output paths changed."
        )

    if (
        freeze.get(
            "freeze_rule_version"
        )
        !=
        AIRPORT_EVALUATION_MANIFEST_FREEZE_RULE_VERSION
    ):
        raise RuntimeError(
            (
                "Airport manifest-freeze "
                "rule mismatch."
            )
        )

    if (
        freeze.get(
            "status"
        )
        !=
        "frozen"
    ):
        raise RuntimeError(
            (
                "Airport execution manifest "
                "is not frozen."
            )
        )

    if (
        freeze.get(
            "manifest_sha256"
        )
        !=
        sha256_file(
            MANIFEST_PATH
        )
    ):
        raise RuntimeError(
            (
                "Airport manifest freeze "
                "SHA binding changed."
            )
        )

    for key in (
        "frozen_before_airport_case_consumption",
        "frozen_before_airport_evaluation",
        "greenhouse_closed",
    ):
        if (
            freeze.get(
                key
            )
            is not True
        ):
            raise RuntimeError(
                (
                    "Airport manifest-freeze "
                    f"governance changed: {key}"
                )
            )

    for key in (
        "airport_cases_consumed",
        "airport_evaluation_executed",
        "airport_results_observed",
    ):
        if (
            freeze.get(
                key
            )
            is not False
        ):
            raise RuntimeError(
                (
                    "Airport pre-execution "
                    f"state changed: {key}"
                )
            )

    if (
        REPORT_PATH.exists()
        or
        RECEIPT_PATH.exists()
        or
        CONSUMPTION_MARKER_PATH.exists()
    ):
        raise RuntimeError(
            (
                "Airport official execution "
                "has already started."
            )
        )

    return manifest


def authorize_execution(
) -> dict[
    str,
    Any,
]:
    manifest = (
        validate_static_contract()
    )

    if not git_worktree_clean():
        raise RuntimeError(
            (
                "Working tree must be clean "
                "before Airport evaluation."
            )
        )

    head = git_head()

    if not git_is_ancestor(
        EXPECTED_PROTOCOL_GIT_COMMIT,
        head,
    ):
        raise RuntimeError(
            (
                "Frozen Airport protocol commit "
                "is not an ancestor of HEAD."
            )
        )

    execution_code = (
        manifest[
            "execution_code"
        ]
    )

    committed = {
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

        "protocol":
            git_blob_sha256(
                PROTOCOL_REPO_PATH
            ),
    }

    expected = {
        "runner":
            execution_code[
                "runner_sha256"
            ],

        "test":
            execution_code[
                "test_sha256"
            ],

        "manifest":
            sha256_file(
                MANIFEST_PATH
            ),

        "manifest_freeze":
            sha256_file(
                MANIFEST_FREEZE_PATH
            ),

        "protocol":
            EXPECTED_PROTOCOL_SHA256,
    }

    if committed != expected:
        raise RuntimeError(
            (
                "Committed Airport execution "
                "authority differs from "
                "frozen manifest."
            )
        )

    return manifest


# ============================================================
# OUTPUT PARSING
# ============================================================


def _round_metric(
    value: float,
) -> float:
    return round(
        float(
            value
        ),
        ROUND_DECIMAL_PLACES,
    )


def _invalid_output(
    *,
    decoded_output: str,
    invalid_reason: str,
    terminal_stop_token_id: (
        int
        | None
    ),
    generation_budget_exhausted: bool,
) -> dict[
    str,
    Any,
]:
    return {
        "strict_json_valid":
            False,

        "predicted_relation":
            None,

        "reason":
            None,

        "reason_word_count":
            None,

        "invalid_reason":
            invalid_reason,

        "decoded_output":
            decoded_output,

        "decoded_output_sha256":
            hashlib.sha256(
                decoded_output.encode(
                    "utf-8"
                )
            ).hexdigest(),

        "terminal_stop_token_id":
            terminal_stop_token_id,

        "generation_budget_exhausted":
            generation_budget_exhausted,
    }


def parse_generated_output(
    *,
    decoded_output: str,
    terminal_stop_token_id: (
        int
        | None
    ) = None,
    generation_budget_exhausted: bool = False,
) -> dict[
    str,
    Any,
]:
    normalized = (
        decoded_output.strip()
    )

    if generation_budget_exhausted:
        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "generation_budget_exhausted",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                True,
        )

    try:
        payload = json.loads(
            normalized
        )

    except json.JSONDecodeError:
        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "json_parse_failed",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )

    if not isinstance(
        payload,
        dict,
    ):
        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "json_value_not_object",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )

    if set(
        payload
    ) != {
        "relation",
        "reason",
    }:
        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "json_key_set_mismatch",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )

    relation = payload[
        "relation"
    ]

    reason = payload[
        "reason"
    ]

    if (
        not isinstance(
            relation,
            str,
        )
        or
        relation
        not in
        EXPECTED_RELATIONS
    ):
        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "relation_not_allowed",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )

    if not isinstance(
        reason,
        str,
    ):
        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "reason_not_string",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )

    word_count = len(
        reason.split()
    )

    if (
        word_count < 6
        or
        word_count > 45
    ):
        return _invalid_output(
            decoded_output=
                normalized,

            invalid_reason=
                "reason_word_count_out_of_range",

            terminal_stop_token_id=
                terminal_stop_token_id,

            generation_budget_exhausted=
                False,
        )

    return {
        "strict_json_valid":
            True,

        "predicted_relation":
            relation,

        "reason":
            reason,

        "reason_word_count":
            word_count,

        "invalid_reason":
            None,

        "decoded_output":
            normalized,

        "decoded_output_sha256":
            hashlib.sha256(
                normalized.encode(
                    "utf-8"
                )
            ).hexdigest(),

        "terminal_stop_token_id":
            terminal_stop_token_id,

        "generation_budget_exhausted":
            False,
    }


def process_generated_token_ids(
    *,
    tokenizer: Any,
    generated_token_ids: Sequence[
        int
    ],
) -> dict[
    str,
    Any,
]:
    token_ids = [
        int(
            value
        )
        for value
        in generated_token_ids
    ]

    if (
        len(
            token_ids
        )
        >
        MAX_NEW_TOKENS
    ):
        raise RuntimeError(
            (
                "Model generated more tokens "
                "than frozen budget."
            )
        )

    if not token_ids:
        return _invalid_output(
            decoded_output="",
            invalid_reason=
                "empty_generation",
            terminal_stop_token_id=None,
            generation_budget_exhausted=False,
        )

    terminal_stop_token_id: (
        int
        | None
    ) = None

    body_ids = token_ids

    if (
        token_ids[
            -1
        ]
        in
        EOS_TOKEN_IDS
    ):
        terminal_stop_token_id = (
            token_ids[
                -1
            ]
        )

        body_ids = token_ids[
            :-1
        ]

        budget_exhausted = False

    elif (
        len(
            token_ids
        )
        ==
        MAX_NEW_TOKENS
    ):
        budget_exhausted = True

    else:
        # generate() should stop early only on one of the
        # frozen EOS/EOT authorities. Anything else fails closed.
        return _invalid_output(
            decoded_output="",
            invalid_reason=
                "missing_terminal_stop_token",
            terminal_stop_token_id=None,
            generation_budget_exhausted=False,
        )

    decoded = tokenizer.decode(
        body_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return parse_generated_output(
        decoded_output=
            decoded,

        terminal_stop_token_id=
            terminal_stop_token_id,

        generation_budget_exhausted=
            budget_exhausted,
    )


# ============================================================
# PROMPT / GENERATION
# ============================================================


def _build_prompt_record(
    *,
    domain: str,
    case: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    return {
        "domain":
            domain,

        "left_metric":
            case[
                "left_metric"
            ],

        "left_description":
            case[
                "left_description"
            ],

        "right_metric":
            case[
                "right_metric"
            ],

        "right_description":
            case[
                "right_description"
            ],
    }


def _generate_case(
    *,
    model: Any,
    tokenizer: Any,
    domain: str,
    case: Mapping[
        str,
        Any,
    ],
    torch_module: Any,
) -> dict[
    str,
    Any,
]:
    user_message = build_user_message(
        _build_prompt_record(
            domain=
                domain,

            case=
                case,
        )
    )

    input_ids = (
        tokenizer.apply_chat_template(
            [
                {
                    "role":
                        "user",

                    "content":
                        user_message,
                }
            ],

            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            return_tensors="pt",
        )
    )

    if (
        getattr(
            input_ids,
            "ndim",
            None,
        )
        !=
        2
        or
        int(
            input_ids.shape[
                0
            ]
        )
        !=
        1
    ):
        raise RuntimeError(
            (
                "Unexpected chat-template "
                "tensor shape."
            )
        )

    input_ids = input_ids.to(
        "cuda"
    )

    attention_mask = (
        torch_module.ones_like(
            input_ids,
            dtype=
                torch_module.long,
            device=
                "cuda",
        )
    )

    prompt_token_count = int(
        input_ids.shape[
            1
        ]
    )

    model_limit = int(
        getattr(
            model.config,
            "max_position_embeddings",
            0,
        )
    )

    if model_limit <= 0:
        raise RuntimeError(
            (
                "Model context limit "
                "unavailable."
            )
        )

    if (
        prompt_token_count
        +
        MAX_NEW_TOKENS
        >
        model_limit
    ):
        raise RuntimeError(
            (
                "Airport prompt exceeds model "
                "context with frozen generation "
                "budget."
            )
        )

    with torch_module.inference_mode():
        generated = model.generate(
            input_ids=
                input_ids,

            attention_mask=
                attention_mask,

            do_sample=False,

            num_beams=1,

            max_new_tokens=
                MAX_NEW_TOKENS,

            eos_token_id=
                list(
                    EOS_TOKEN_IDS
                ),

            pad_token_id=
                PAD_TOKEN_ID,
        )

    if (
        getattr(
            generated,
            "ndim",
            None,
        )
        !=
        2
        or
        int(
            generated.shape[
                0
            ]
        )
        !=
        1
    ):
        raise RuntimeError(
            (
                "Unexpected generate() "
                "output shape."
            )
        )

    new_ids = (
        generated[
            0,
            prompt_token_count:
        ]
        .detach()
        .cpu()
        .tolist()
    )

    processed = (
        process_generated_token_ids(
            tokenizer=
                tokenizer,

            generated_token_ids=
                new_ids,
        )
    )

    processed[
        "prompt_token_count"
    ] = prompt_token_count

    processed[
        "generated_token_count"
    ] = len(
        new_ids
    )

    return processed


# ============================================================
# METRICS
# ============================================================


def evaluate_model(
    *,
    model: Any,
    tokenizer: Any,
    domain: str,
    cases: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    torch_module: Any,
) -> dict[
    str,
    Any,
]:
    if len(
        cases
    ) != 30:
        raise RuntimeError(
            (
                "Official Airport evaluation "
                "requires exactly 30 cases."
            )
        )

    relation_totals = Counter(
        str(
            case[
                "expected_relation"
            ]
        )
        for case
        in cases
    )

    expected_counts = {
        relation:
            6
        for relation
        in EXPECTED_RELATIONS
    }

    if (
        dict(
            relation_totals
        )
        !=
        expected_counts
    ):
        raise RuntimeError(
            (
                "Airport official relation counts "
                "are not 6 per class."
            )
        )

    correct_count = 0
    valid_count = 0

    relation_correct = Counter()

    results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        expected = str(
            case[
                "expected_relation"
            ]
        )

        generated = _generate_case(
            model=
                model,

            tokenizer=
                tokenizer,

            domain=
                domain,

            case=
                case,

            torch_module=
                torch_module,
        )

        predicted = generated[
            "predicted_relation"
        ]

        correct = bool(
            generated[
                "strict_json_valid"
            ]
            and
            predicted
            ==
            expected
        )

        if generated[
            "strict_json_valid"
        ]:
            valid_count += 1

        if correct:
            correct_count += 1

            relation_correct[
                expected
            ] += 1

        case_result = {
            "case_id":
                case[
                    "case_id"
                ],

            "expected_relation":
                expected,

            "predicted_relation":
                predicted,

            "correct":
                correct,

            **generated,
        }

        results.append(
            case_result
        )

        print(
            (
                f"  [{index:02d}/30] "
                f"{case['case_id']}: "
                "valid="
                f"{generated['strict_json_valid']} "
                "predicted="
                f"{predicted!r} "
                f"correct={correct}"
            )
        )

    per_relation_accuracy = {
        relation:
            _round_metric(
                relation_correct[
                    relation
                ]
                /
                relation_totals[
                    relation
                ]
            )
        for relation
        in EXPECTED_RELATIONS
    }

    raw_macro = (
        sum(
            relation_correct[
                relation
            ]
            /
            relation_totals[
                relation
            ]
            for relation
            in EXPECTED_RELATIONS
        )
        /
        len(
            EXPECTED_RELATIONS
        )
    )

    return {
        "case_count":
            len(
                cases
            ),

        "correct_count":
            correct_count,

        "strict_json_valid_count":
            valid_count,

        "accuracy":
            _round_metric(
                correct_count
                /
                len(
                    cases
                )
            ),

        "macro_accuracy":
            _round_metric(
                raw_macro
            ),

        "strict_json_validity_rate":
            _round_metric(
                valid_count
                /
                len(
                    cases
                )
            ),

        "per_relation_accuracy":
            per_relation_accuracy,

        "uncertain_accuracy":
            per_relation_accuracy[
                "uncertain"
            ],

        "cases":
            results,
    }


def paired_comparison(
    *,
    base: Mapping[
        str,
        Any,
    ],
    adapted: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    base_cases = base[
        "cases"
    ]

    adapted_cases = adapted[
        "cases"
    ]

    if (
        len(
            base_cases
        )
        !=
        30
        or
        len(
            adapted_cases
        )
        !=
        30
    ):
        raise RuntimeError(
            (
                "Paired Airport comparison "
                "requires 30+30 results."
            )
        )

    base_by_id = {
        item[
            "case_id"
        ]:
            item
        for item
        in base_cases
    }

    adapted_by_id = {
        item[
            "case_id"
        ]:
            item
        for item
        in adapted_cases
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
            (
                "Base/adapted Airport "
                "case IDs differ."
            )
        )

    both_correct = 0
    base_only_correct = 0
    adapted_only_correct = 0
    both_wrong = 0
    changed_predictions = 0

    for case_id in base_by_id:
        base_case = (
            base_by_id[
                case_id
            ]
        )

        adapted_case = (
            adapted_by_id[
                case_id
            ]
        )

        if (
            base_case[
                "expected_relation"
            ]
            !=
            adapted_case[
                "expected_relation"
            ]
        ):
            raise RuntimeError(
                (
                    "Base/adapted expected "
                    "labels differ."
                )
            )

        base_correct = bool(
            base_case[
                "correct"
            ]
        )

        adapted_correct = bool(
            adapted_case[
                "correct"
            ]
        )

        if (
            base_correct
            and
            adapted_correct
        ):
            both_correct += 1

        elif base_correct:
            base_only_correct += 1

        elif adapted_correct:
            adapted_only_correct += 1

        else:
            both_wrong += 1

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

    return {
        "accuracy_delta":
            _round_metric(
                float(
                    adapted[
                        "accuracy"
                    ]
                )
                -
                float(
                    base[
                        "accuracy"
                    ]
                )
            ),

        "macro_accuracy_delta":
            _round_metric(
                float(
                    adapted[
                        "macro_accuracy"
                    ]
                )
                -
                float(
                    base[
                        "macro_accuracy"
                    ]
                )
            ),

        "strict_json_validity_delta":
            _round_metric(
                float(
                    adapted[
                        "strict_json_validity_rate"
                    ]
                )
                -
                float(
                    base[
                        "strict_json_validity_rate"
                    ]
                )
            ),

        "both_correct":
            both_correct,

        "base_only_correct":
            base_only_correct,

        "adapted_only_correct":
            adapted_only_correct,

        "both_wrong":
            both_wrong,

        "changed_predictions":
            changed_predictions,
    }


def evaluate_acceptance_gates(
    *,
    base: Mapping[
        str,
        Any,
    ],
    adapted: Mapping[
        str,
        Any,
    ],
    paired: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    per_relation = adapted[
        "per_relation_accuracy"
    ]

    absolute = {
        "adapted_accuracy_minimum":
            (
                float(
                    adapted[
                        "accuracy"
                    ]
                )
                >=
                0.70
            ),

        "adapted_macro_accuracy_minimum":
            (
                float(
                    adapted[
                        "macro_accuracy"
                    ]
                )
                >=
                0.70
            ),

        "adapted_per_relation_accuracy_minimum":
            all(
                float(
                    per_relation[
                        relation
                    ]
                )
                >=
                0.50
                for relation
                in EXPECTED_RELATIONS
            ),

        "adapted_uncertain_accuracy_minimum":
            (
                float(
                    adapted[
                        "uncertain_accuracy"
                    ]
                )
                >=
                0.666667
            ),

        "adapted_strict_json_validity_rate":
            (
                float(
                    adapted[
                        "strict_json_validity_rate"
                    ]
                )
                ==
                1.0
            ),
    }

    non_regression = {
        "accuracy_delta_minimum":
            (
                float(
                    paired[
                        "accuracy_delta"
                    ]
                )
                >=
                0.0
            ),

        "macro_accuracy_delta_minimum":
            (
                float(
                    paired[
                        "macro_accuracy_delta"
                    ]
                )
                >=
                0.0
            ),
    }

    all_passed = (
        all(
            absolute.values()
        )
        and
        all(
            non_regression.values()
        )
    )

    return {
        "absolute":
            absolute,

        "non_regression":
            non_regression,

        "all_passed":
            all_passed,

        "failure_action":
            (
                None
                if all_passed
                else
                "stop_v0.4_before_greenhouse"
            ),

        "greenhouse_opened":
            False,

        "greenhouse_authorized_by_runner":
            False,

        (
            "greenhouse_may_be_considered_"
            "after_airport_evidence_commit"
        ):
            all_passed,
    }


# ============================================================
# SINGLE-USE HOLDOUT BOUNDARY
# ============================================================


def _load_official_holdout(
) -> tuple[
    str,
    tuple[
        dict[
            str,
            Any,
        ],
        ...,
    ],
]:
    # CRITICAL:
    # This function is the ONLY intentional Airport
    # case-consumption boundary in the runner.

    actual_cases_sha = (
        sha256_file(
            AIRPORT_CASES_PATH
        )
    )

    if (
        actual_cases_sha
        !=
        EXPECTED_AIRPORT_CASES_SHA256
    ):
        raise RuntimeError(
            (
                "Airport cases SHA changed before "
                "single-use evaluation."
            )
        )

    payload = load_json_object(
        AIRPORT_CASES_PATH
    )

    validate_cases(
        payload
    )

    freeze = load_json_object(
        AIRPORT_FREEZE_PATH
    )

    validate_freeze(
        cases_sha256=
            actual_cases_sha,

        freeze=
            freeze,
    )

    domain = payload.get(
        "domain"
    )

    cases = payload.get(
        "cases"
    )

    if (
        not isinstance(
            domain,
            str,
        )
        or
        not domain
    ):
        raise RuntimeError(
            "Airport holdout domain missing."
        )

    if (
        not isinstance(
            cases,
            list,
        )
        or
        len(
            cases
        )
        !=
        30
    ):
        raise RuntimeError(
            "Airport holdout cases missing."
        )

    normalized = []

    for case in cases:
        if not isinstance(
            case,
            dict,
        ):
            raise RuntimeError(
                (
                    "Airport case is not "
                    "a JSON object."
                )
            )

        normalized.append(
            dict(
                case
            )
        )

    return (
        domain,
        tuple(
            normalized
        ),
    )


# ============================================================
# RUNTIME
# ============================================================


def _cuda_barrier(
    torch_module: Any,
) -> dict[
    str,
    int,
]:
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            (
                "CUDA is required for frozen "
                "Airport evaluation."
            )
        )

    (
        free_bytes,
        total_bytes,
    ) = (
        torch_module
        .cuda
        .mem_get_info()
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
                "required="
                f"{MINIMUM_FREE_CUDA_BYTES / 1024**3:.2f} GiB"
            )
        )

    return {
        "free_bytes":
            int(
                free_bytes
            ),

        "total_bytes":
            int(
                total_bytes
            ),
    }


def _load_base_model(
    *,
    torch_module: Any,
    authority: Any,
) -> Any:
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    quantization = (
        authority.contract
        .quantization
    )

    if (
        quantization.load_in_4bit
        is not True
    ):
        raise RuntimeError(
            (
                "Frozen v0.4 model is "
                "no longer 4-bit."
            )
        )

    if (
        quantization.quantization_type
        !=
        "nf4"
    ):
        raise RuntimeError(
            (
                "Frozen v0.4 quantization "
                "is no longer NF4."
            )
        )

    if (
        quantization.use_double_quantization
        is not True
    ):
        raise RuntimeError(
            (
                "Frozen v0.4 double "
                "quantization changed."
            )
        )

    if (
        quantization.compute_dtype
        !=
        "bfloat16"
    ):
        raise RuntimeError(
            (
                "Frozen v0.4 compute "
                "dtype changed."
            )
        )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=
                torch_module.bfloat16,
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
                torch_module.bfloat16,

            quantization_config=
                quantization_config,

            trust_remote_code=
                False,

            local_files_only=
                True,
        )
    )

    model.eval()

    return model


def _attach_adapter(
    *,
    model: Any,
) -> Any:
    from peft import (
        PeftModel,
    )

    adapted_model = (
        PeftModel
        .from_pretrained(
            model,
            str(
                ADAPTER_PATH
            ),
            is_trainable=False,
        )
    )

    adapted_model.eval()

    return adapted_model


def _synthetic_case(
) -> dict[
    str,
    str,
]:
    return {
        "case_id":
            "synthetic:airport-runtime-preflight",

        "left_metric":
            "departed_flights",

        "left_description":
            (
                "Count of flights that have completed "
                "departure from the airport during the "
                "reporting period after leaving the "
                "assigned gate."
            ),

        "right_metric":
            "scheduled_departures",

        "right_description":
            (
                "Count of flights planned to depart "
                "from the airport during the same "
                "reporting period according to the "
                "published schedule."
            ),

        "expected_relation":
            "related_distinct_metric",
    }


def preflight_runtime(
) -> None:
    manifest = (
        authorize_execution()
    )

    import torch

    memory = _cuda_barrier(
        torch
    )

    torch.manual_seed(
        42
    )

    torch.cuda.manual_seed_all(
        42
    )

    torch.cuda.reset_peak_memory_stats()

    authority = (
        validate_static_authority(
            repository_root_value=
                ROOT
        )
    )

    tokenizer = (
        load_pinned_tokenizer(
            authority=
                authority
        )
    )

    tokenizer.padding_side = (
        "right"
    )

    if (
        tokenizer.pad_token_id
        !=
        PAD_TOKEN_ID
    ):
        raise RuntimeError(
            (
                "Pinned tokenizer pad "
                "token changed."
            )
        )

    if not isinstance(
        tokenizer.chat_template,
        str,
    ):
        raise RuntimeError(
            (
                "Pinned tokenizer chat "
                "template missing."
            )
        )

    if (
        hashlib.sha256(
            tokenizer.chat_template.encode(
                "utf-8"
            )
        ).hexdigest()
        !=
        EXPECTED_CHAT_TEMPLATE_SHA256
    ):
        raise RuntimeError(
            (
                "Pinned tokenizer chat-template "
                "SHA changed."
            )
        )

    model = _load_base_model(
        torch_module=
            torch,

        authority=
            authority,
    )

    synthetic = _synthetic_case()

    base_probe = _generate_case(
        model=
            model,

        tokenizer=
            tokenizer,

        domain=
            "airport_ground_operations_synthetic_preflight",

        case=
            synthetic,

        torch_module=
            torch,
    )

    adapted_model = (
        _attach_adapter(
            model=
                model
        )
    )

    adapted_probe = _generate_case(
        model=
            adapted_model,

        tokenizer=
            tokenizer,

        domain=
            "airport_ground_operations_synthetic_preflight",

        case=
            synthetic,

        torch_module=
            torch,
    )

    print(
        "=== DATALENS QLORA v0.4 AIRPORT RUNTIME PREFLIGHT v0.1 ==="
    )

    print(
        (
            "CUDA free before load: "
            f"{memory['free_bytes'] / 1024**3:.2f} GiB"
        )
    )

    print(
        (
            "Base synthetic generated tokens: "
            f"{base_probe['generated_token_count']}"
        )
    )

    print(
        (
            "Adapted synthetic generated tokens: "
            f"{adapted_probe['generated_token_count']}"
        )
    )

    print(
        (
            "Base synthetic strict JSON: "
            f"{base_probe['strict_json_valid']}"
        )
    )

    print(
        (
            "Adapted synthetic strict JSON: "
            f"{adapted_probe['strict_json_valid']}"
        )
    )

    print(
        "Airport cases opened: False"
    )

    print(
        "Airport results observed: False"
    )

    print(
        "Greenhouse opened: False"
    )

    print(
        "Training executed: False"
    )

    print(
        "Optimizer created: False"
    )

    print(
        "Backward executed: False"
    )

    print(
        "DATALENS QLORA v0.4 AIRPORT RUNTIME PREFLIGHT v0.1: PASS"
    )

    del adapted_model
    del model

    torch.cuda.empty_cache()

    _ = manifest


# ============================================================
# CONSUMPTION MARKER
# ============================================================


def _write_consumption_marker(
    *,
    manifest: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    marker = {
        "marker_rule_version":
            AIRPORT_CONSUMPTION_MARKER_RULE_VERSION,

        "experiment_id":
            "datalens-semantic-qlora-v0.4",

        "holdout_id":
            (
                "adaptation:"
                "datalens-semantic-qlora-v0.4:"
                "airport-ground-operations:"
                "holdout:v0.1"
            ),

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_head":
            git_head(),

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "manifest_sha256":
            sha256_file(
                MANIFEST_PATH
            ),

        "cases_sha256_expected":
            EXPECTED_AIRPORT_CASES_SHA256,

        "cases_opened_before_marker":
            False,

        "results_observed_before_marker":
            False,

        "single_use_consumption_started":
            True,

        "reexecution_permitted":
            False,

        "greenhouse_opened":
            False,

        "status":
            "consumption_started",
    }

    atomic_write_json(
        path=
            CONSUMPTION_MARKER_PATH,

        payload=
            marker,
    )

    _ = manifest

    return marker


# ============================================================
# OFFICIAL EXECUTION
# ============================================================


def execute_evaluation(
) -> dict[
    str,
    Any,
]:
    # Authorization and all normal runtime checks happen before
    # any Airport case byte is consumed.
    manifest = (
        authorize_execution()
    )

    import torch

    memory = _cuda_barrier(
        torch
    )

    torch.manual_seed(
        42
    )

    torch.cuda.manual_seed_all(
        42
    )

    torch.cuda.reset_peak_memory_stats()

    authority = (
        validate_static_authority(
            repository_root_value=
                ROOT
        )
    )

    tokenizer = (
        load_pinned_tokenizer(
            authority=
                authority
        )
    )

    tokenizer.padding_side = (
        "right"
    )

    if (
        tokenizer.pad_token_id
        !=
        PAD_TOKEN_ID
    ):
        raise RuntimeError(
            (
                "Pinned tokenizer pad "
                "token changed."
            )
        )

    if (
        hashlib.sha256(
            tokenizer.chat_template.encode(
                "utf-8"
            )
        ).hexdigest()
        !=
        EXPECTED_CHAT_TEMPLATE_SHA256
    ):
        raise RuntimeError(
            (
                "Pinned tokenizer chat-template "
                "SHA changed."
            )
        )

    # Load the base model before consuming the protected holdout.
    # The separate preflight mode also exercises adapter loading
    # without touching Airport.
    model = _load_base_model(
        torch_module=
            torch,

        authority=
            authority,
    )

    # From this point onward any failure is terminal for the v0.4
    # Airport holdout. The marker prevents an accidental rerun.
    marker = (
        _write_consumption_marker(
            manifest=
                manifest
        )
    )

    (
        domain,
        cases,
    ) = (
        _load_official_holdout()
    )

    print(
        "=== DATALENS QLORA v0.4 AIRPORT INDEPENDENT HOLDOUT v0.1 ==="
    )

    print(
        (
            "CUDA free before load: "
            f"{memory['free_bytes'] / 1024**3:.2f} GiB"
        )
    )

    print(
        "Airport holdout consumed: True"
    )

    print(
        "Airport single-use marker written: True"
    )

    print()
    print(
        "BASE GEMMA"
    )

    base_result = (
        evaluate_model(
            model=
                model,

            tokenizer=
                tokenizer,

            domain=
                domain,

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
        _attach_adapter(
            model=
                model
        )
    )

    print()
    print(
        "ADAPTED GEMMA"
    )

    adapted_result = (
        evaluate_model(
            model=
                adapted_model,

            tokenizer=
                tokenizer,

            domain=
                domain,

            cases=
                cases,

            torch_module=
                torch,
        )
    )

    paired = paired_comparison(
        base=
            base_result,

        adapted=
            adapted_result,
    )

    gates = evaluate_acceptance_gates(
        base=
            base_result,

        adapted=
            adapted_result,

        paired=
            paired,
    )

    peak_allocated = int(
        torch.cuda.max_memory_allocated()
    )

    peak_reserved = int(
        torch.cuda.max_memory_reserved()
    )

    created_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    report = {
        "report_rule_version":
            "qlora_v0.4_airport_evaluation_report_v0.1",

        "experiment_id":
            "datalens-semantic-qlora-v0.4",

        "holdout_id":
            marker[
                "holdout_id"
            ],

        "created_at":
            created_at,

        "git_head":
            git_head(),

        "status":
            "completed",

        "acceptance_authority":
            True,

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "manifest_sha256":
            sha256_file(
                MANIFEST_PATH
            ),

        "consumption_marker_sha256":
            sha256_file(
                CONSUMPTION_MARKER_PATH
            ),

        "cases_sha256":
            EXPECTED_AIRPORT_CASES_SHA256,

        "adapter_bundle_sha256":
            EXPECTED_ADAPTER_BUNDLE_SHA256,

        "evaluation_order":
            [
                "pinned_base",
                "same_base_plus_frozen_candidate_adapter",
            ],

        "generation": {
            "mode":
                "deterministic_greedy",

            "do_sample":
                False,

            "num_beams":
                1,

            "max_new_tokens":
                MAX_NEW_TOKENS,

            "eos_token_ids":
                list(
                    EOS_TOKEN_IDS
                ),

            "pad_token_id":
                PAD_TOKEN_ID,

            "input_truncation_permitted":
                False,

            "decode_skip_special_tokens":
                False,

            "llm_judge_used":
                False,
        },

        "base":
            base_result,

        "adapted":
            adapted_result,

        "paired":
            paired,

        "acceptance_gates":
            gates,

        "runtime": {
            "versions":
                runtime_versions(),

            "cuda_free_bytes_before_load":
                memory[
                    "free_bytes"
                ],

            "cuda_total_bytes":
                memory[
                    "total_bytes"
                ],

            "cuda_peak_allocated_bytes":
                peak_allocated,

            "cuda_peak_reserved_bytes":
                peak_reserved,
        },

        "safety": {
            "training_executed":
                False,

            "optimizer_created":
                False,

            "backward_executed":
                False,

            "airport_used_for_tuning":
                False,

            "airport_used_for_model_selection":
                False,

            "hotel_reexecuted":
                False,

            "greenhouse_opened":
                False,

            "greenhouse_evaluated":
                False,
        },
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
        "receipt_rule_version":
            AIRPORT_EVALUATION_RECEIPT_RULE_VERSION,

        "runner_rule_version":
            AIRPORT_EVALUATION_RUNNER_RULE_VERSION,

        "experiment_id":
            "datalens-semantic-qlora-v0.4",

        "holdout_id":
            marker[
                "holdout_id"
            ],

        "status":
            "completed",

        "git_head":
            report[
                "git_head"
            ],

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "manifest_sha256":
            report[
                "manifest_sha256"
            ],

        "consumption_marker_sha256":
            report[
                "consumption_marker_sha256"
            ],

        "airport_cases_sha256":
            EXPECTED_AIRPORT_CASES_SHA256,

        "adapter_bundle_sha256":
            EXPECTED_ADAPTER_BUNDLE_SHA256,

        "report_sha256":
            report_sha256,

        "base_accuracy":
            base_result[
                "accuracy"
            ],

        "adapted_accuracy":
            adapted_result[
                "accuracy"
            ],

        "base_macro_accuracy":
            base_result[
                "macro_accuracy"
            ],

        "adapted_macro_accuracy":
            adapted_result[
                "macro_accuracy"
            ],

        "accuracy_delta":
            paired[
                "accuracy_delta"
            ],

        "macro_accuracy_delta":
            paired[
                "macro_accuracy_delta"
            ],

        "adapted_strict_json_validity_rate":
            adapted_result[
                "strict_json_validity_rate"
            ],

        "all_airport_gates_passed":
            gates[
                "all_passed"
            ],

        "failure_action":
            gates[
                "failure_action"
            ],

        "greenhouse_opened":
            False,

        "greenhouse_evaluated":
            False,

        "reexecution_permitted":
            False,

        "training_loss_used_as_acceptance_evidence":
            False,
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
            "  Base accuracy:                 "
            f"{base_result['accuracy']:.6f}"
        )
    )

    print(
        (
            "  Adapted accuracy:              "
            f"{adapted_result['accuracy']:.6f}"
        )
    )

    print(
        (
            "  Accuracy delta:                "
            f"{paired['accuracy_delta']:+.6f}"
        )
    )

    print(
        (
            "  Base macro accuracy:           "
            f"{base_result['macro_accuracy']:.6f}"
        )
    )

    print(
        (
            "  Adapted macro accuracy:        "
            f"{adapted_result['macro_accuracy']:.6f}"
        )
    )

    print(
        (
            "  Macro delta:                   "
            f"{paired['macro_accuracy_delta']:+.6f}"
        )
    )

    print(
        (
            "  Adapted strict JSON validity:  "
            f"{adapted_result['strict_json_validity_rate']:.6f}"
        )
    )

    print(
        (
            "  Airport gates passed:          "
            f"{gates['all_passed']}"
        )
    )

    print(
        (
            "  Report SHA256:                 "
            f"{report_sha256}"
        )
    )

    print(
        (
            "  Receipt SHA256:                "
            f"{sha256_file(RECEIPT_PATH)}"
        )
    )

    print(
        (
            "  Consumption marker SHA256:     "
            f"{sha256_file(CONSUMPTION_MARKER_PATH)}"
        )
    )

    print()
    print(
        "GOVERNANCE"
    )

    print(
        "  Airport reexecution allowed: False"
    )

    print(
        "  Airport tuning allowed: False"
    )

    print(
        "  Hotel reexecution allowed: False"
    )

    print(
        "  Greenhouse opened: False"
    )

    print(
        (
            "  Greenhouse may be considered "
            "after evidence commit: "
            f"{gates['greenhouse_may_be_considered_after_airport_evidence_commit']}"
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
        "DATALENS QLORA v0.4 AIRPORT INDEPENDENT HOLDOUT v0.1: COMPLETED"
    )

    return report


# ============================================================
# CLI
# ============================================================


def main(
) -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=(
            "validate-static",
            "authorize-only",
            "preflight",
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
            "DATALENS QLORA v0.4 AIRPORT STATIC VALIDATION: PASS"
        )

        return

    if (
        arguments.mode
        ==
        "authorize-only"
    ):
        authorize_execution()

        print(
            "DATALENS QLORA v0.4 AIRPORT AUTHORIZATION: PASS"
        )

        return

    if (
        arguments.mode
        ==
        "preflight"
    ):
        preflight_runtime()

        return

    execute_evaluation()


if __name__ == "__main__":
    main()

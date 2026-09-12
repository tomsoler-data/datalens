from __future__ import annotations


import hashlib
import json
import os
import re
import subprocess


from pathlib import Path
from typing import Any, Mapping


from app.adaptation.hospital_independent_evaluation_runner_v0_4_v0_1 import (
    EXPECTED_HOLDOUT_CASES_SHA256,
    EXPECTED_HOLDOUT_FREEZE_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    validate_pre_execution_authorities,
)


HOSPITAL_INDEPENDENT_EVALUATION_PREFLIGHT_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_preflight_v0.1"
)

HOSPITAL_CONSUMPTION_MARKER_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_consumption_v0.1"
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

REPO_ROOT = (
    ROOT
    .parents[1]
)

ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
)


D4_C_B_FOUNDATION_COMMIT = (
    "67bf0bccaabf63afc42a832e56ebea47ca559acd"
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


EXPECTED_ADAPTER_BUNDLE_SHA256 = (
    "0351980df6d86096195c0971deb30c725"
    "e155c71aa5de8054b2b37fa42090716"
)

EXPECTED_ADAPTER_CONFIG_SHA256 = (
    "3ae14896612f6bf74ee7786a450e2ac0"
    "f08f3da9f33391505cb1a7dc823dcdb8"
)

EXPECTED_ADAPTER_WEIGHTS_SHA256 = (
    "4f145b0bf37f67841c09f02b86679634"
    "a9532491d2f560b0e7c5c328009e4610"
)

EXPECTED_ADAPTER_README_SHA256 = (
    "6ecdbb662eaed8010ab0e012a2b95b79"
    "543884cf294406dc6da2cde64f98389d"
)


EXPECTED_TRAINING_REPORT_SHA256 = (
    "759ba4957806daab8b7a14d3aeb2b068"
    "59e0bcd6193d30cb877b63748617e04d"
)

EXPECTED_TRAINING_RECEIPT_SHA256 = (
    "f412062f78432d7c432d4b36beed9d84"
    "d527d5990279240030a0a31227dffaee"
)


SOURCE_AUTHORITIES = {
    (
        "apps/api/app/adaptation/"
        "qlora_runtime_v0_4.py"
    ):
        (
            "20e41ab00606296893276a84e53746c0"
            "6618b8cabca74fef77cb743c5e80ab7c"
        ),

    (
        "apps/api/app/adaptation/"
        "hospital_independent_evaluation_runner_v0_4_v0_1.py"
    ):
        (
            "4f1f69ff01b9a9c096dd42fc694a6f9"
            "f11f0ba30a23f6fc2a8dddc92deff900d"
        ),

    (
        "apps/api/tests/adaptation/"
        "test_hospital_independent_evaluation_runner_static_v0_1.py"
    ):
        (
            "81ce3b70801b118e52ea5ddad0832444"
            "e9f985da3771c2b103a0fc32a8f5445c"
        ),

    (
        "apps/api/app/adaptation/"
        "hospital_independent_evaluation_execution_v0_1.py"
    ):
        (
            "28566b79ad34e2587ced709f68ca19b1"
            "dfaa5d4074c07ba769db3a11306ee1c1"
        ),

    (
        "apps/api/tests/adaptation/"
        "test_hospital_independent_evaluation_execution_v0_1.py"
    ):
        (
            "a952c44393b1765f69c7bb9c628a6b0"
            "ab2162bcfdea15c754739a477c81d4808"
        ),
}


ADAPTER_DIR = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "adapters"
    / "datalens_semantic_qlora_v0.4_adapter"
)

TRAINING_REPORT_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "training"
    / "datalens_semantic_qlora_v0.4_training_v0.1_report.json"
)

TRAINING_RECEIPT_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "training"
    / "datalens_semantic_qlora_v0.4_training_v0.1_receipt.json"
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


CONSUMPTION_MARKER_FILENAME = (
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_consumption.json"
)

PREDICTIONS_FILENAME = (
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_predictions.json"
)

REPORT_FILENAME = (
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_report.json"
)

RECEIPT_FILENAME = (
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_receipt.json"
)


EXPECTED_CASE_COUNT = 30


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


def load_json_object(
    path: Path,
) -> dict[
    str,
    Any,
]:

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):

        raise RuntimeError(
            (
                "Expected JSON object: "
                f"{path}"
            )
        )

    return payload


def require_exact_sha(
    *,
    path: Path,
    expected_sha256: str,
    label: str,
) -> None:

    if not path.is_file():

        raise RuntimeError(
            (
                f"{label} missing: "
                f"{path}"
            )
        )

    actual = sha256_file(
        path
    )

    if actual != expected_sha256:

        raise RuntimeError(
            (
                f"{label} SHA changed.\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual}"
            )
        )


def normalize_source_eol(
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


def git_committed_bytes(
    *,
    repo_relative_path: str,
    revision: str = "HEAD",
) -> bytes:

    if (
        not isinstance(
            repo_relative_path,
            str,
        )
        or
        not repo_relative_path.strip()
    ):

        raise TypeError(
            "repo_relative_path must be a non-empty string."
        )

    if (
        not isinstance(
            revision,
            str,
        )
        or
        not revision.strip()
    ):

        raise TypeError(
            "revision must be a non-empty string."
        )

    result = subprocess.run(
        [
            "git",
            "show",
            (
                f"{revision}:"
                f"{repo_relative_path}"
            ),
        ],
        cwd=
            REPO_ROOT,
        stdout=
            subprocess.PIPE,
        stderr=
            subprocess.PIPE,
        check=
            False,
    )

    if result.returncode != 0:

        error = (
            result.stderr.decode(
                "utf-8",
                errors="replace",
            )
            .strip()
        )

        raise RuntimeError(
            (
                "Unable to resolve committed source authority.\n"
                f"Revision: {revision}\n"
                f"Path: {repo_relative_path}\n"
                f"Git: {error}"
            )
        )

    return result.stdout


def git_committed_sha256(
    *,
    repo_relative_path: str,
    revision: str = "HEAD",
) -> str:

    return hashlib.sha256(
        git_committed_bytes(
            repo_relative_path=
                repo_relative_path,

            revision=
                revision,
        )
    ).hexdigest()


def validate_source_authorities(
) -> None:

    for (
        relative_path,
        expected_sha,
    ) in SOURCE_AUTHORITIES.items():

        working_path = (
            REPO_ROOT
            /
            relative_path
        )

        if not working_path.is_file():

            raise RuntimeError(
                (
                    "Source authority missing from working tree: "
                    f"{relative_path}"
                )
            )

        committed_bytes = (
            git_committed_bytes(
                repo_relative_path=
                    relative_path,
            )
        )

        committed_sha = hashlib.sha256(
            committed_bytes
        ).hexdigest()

        if committed_sha != expected_sha:

            raise RuntimeError(
                (
                    "Committed source authority SHA changed.\n"
                    f"Path:     {relative_path}\n"
                    f"Expected: {expected_sha}\n"
                    f"Actual:   {committed_sha}"
                )
            )

        working_bytes = (
            working_path.read_bytes()
        )

        if (
            normalize_source_eol(
                working_bytes
            )
            !=
            normalize_source_eol(
                committed_bytes
            )
        ):

            raise RuntimeError(
                (
                    "Working source differs from committed "
                    "authority beyond EOL normalization.\n"
                    f"Path: {relative_path}"
                )
            )


def validate_candidate_authorities(
) -> None:

    require_exact_sha(
        path=
            TRAINING_REPORT_PATH,

        expected_sha256=
            EXPECTED_TRAINING_REPORT_SHA256,

        label=
            "Official QLoRA v0.4 training report",
    )

    require_exact_sha(
        path=
            TRAINING_RECEIPT_PATH,

        expected_sha256=
            EXPECTED_TRAINING_RECEIPT_SHA256,

        label=
            "Official QLoRA v0.4 training receipt",
    )

    adapter_files = {
        "adapter_config.json":
            EXPECTED_ADAPTER_CONFIG_SHA256,

        "adapter_model.safetensors":
            EXPECTED_ADAPTER_WEIGHTS_SHA256,

        "README.md":
            EXPECTED_ADAPTER_README_SHA256,
    }

    for (
        filename,
        expected_sha,
    ) in adapter_files.items():

        require_exact_sha(
            path=
                ADAPTER_DIR
                /
                filename,

            expected_sha256=
                expected_sha,

            label=
                (
                    "Official QLoRA v0.4 adapter "
                    f"{filename}"
                ),
        )

    receipt = load_json_object(
        TRAINING_RECEIPT_PATH
    )

    adapter_evidence = (
        receipt.get(
            "adapter"
        )
    )

    if not isinstance(
        adapter_evidence,
        dict,
    ):

        raise RuntimeError(
            "Training receipt adapter evidence is malformed."
        )

    if (
        adapter_evidence.get(
            "bundle_sha256"
        )
        !=
        EXPECTED_ADAPTER_BUNDLE_SHA256
    ):

        raise RuntimeError(
            "Official adapter bundle SHA changed."
        )


def validate_converted_checkpoint_metadata(
) -> None:

    if not CONVERTED_MODEL_PATH.is_dir():

        raise RuntimeError(
            (
                "Converted Gemma checkpoint missing: "
                f"{CONVERTED_MODEL_PATH}"
            )
        )

    config_path = (
        CONVERTED_MODEL_PATH
        /
        "config.json"
    )

    index_path = (
        CONVERTED_MODEL_PATH
        /
        "model.safetensors.index.json"
    )

    if not config_path.is_file():

        raise RuntimeError(
            "Converted Gemma config is missing."
        )

    if not index_path.is_file():

        raise RuntimeError(
            "Converted Gemma shard index is missing."
        )

    config = load_json_object(
        config_path
    )

    if (
        config.get(
            "model_type"
        )
        !=
        "gemma3_text"
    ):

        raise RuntimeError(
            "Converted Gemma model_type changed."
        )

    architectures = (
        config.get(
            "architectures"
        )
    )

    if (
        not isinstance(
            architectures,
            list,
        )
        or
        "Gemma3ForCausalLM"
        not in
        architectures
    ):

        raise RuntimeError(
            "Converted Gemma architecture changed."
        )

    index = load_json_object(
        index_path
    )

    weight_map = (
        index.get(
            "weight_map"
        )
    )

    if not isinstance(
        weight_map,
        dict,
    ):

        raise RuntimeError(
            "Converted Gemma weight map is malformed."
        )

    shards = sorted(
        set(
            weight_map.values()
        )
    )

    if not shards:

        raise RuntimeError(
            "Converted Gemma checkpoint has zero shards."
        )

    for shard in shards:

        shard_path = (
            CONVERTED_MODEL_PATH
            /
            str(
                shard
            )
        )

        if not shard_path.is_file():

            raise RuntimeError(
                (
                    "Converted Gemma shard missing: "
                    f"{shard}"
                )
            )

    adapter_config = load_json_object(
        ADAPTER_DIR
        /
        "adapter_config.json"
    )

    if (
        adapter_config.get(
            "base_model_name_or_path"
        )
        !=
        str(
            CONVERTED_MODEL_PATH
        )
    ):

        raise RuntimeError(
            (
                "Adapter no longer references the exact "
                "converted Gemma checkpoint."
            )
        )


def single_use_artifact_paths(
    *,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[
    str,
    Path,
]:

    return {
        "consumption_marker":
            artifact_dir
            /
            CONSUMPTION_MARKER_FILENAME,

        "predictions":
            artifact_dir
            /
            PREDICTIONS_FILENAME,

        "report":
            artifact_dir
            /
            REPORT_FILENAME,

        "receipt":
            artifact_dir
            /
            RECEIPT_FILENAME,
    }


def assert_single_use_available(
    *,
    artifact_dir: Path = ARTIFACT_DIR,
) -> None:

    paths = single_use_artifact_paths(
        artifact_dir=
            artifact_dir
    )

    existing = [
        (
            label,
            path,
        )

        for (
            label,
            path,
        ) in paths.items()

        if path.exists()
    ]

    if existing:

        details = "\n".join(
            (
                f"- {label}: {path}"
            )

            for (
                label,
                path,
            ) in existing
        )

        raise RuntimeError(
            (
                "Hospital benchmark is not eligible for "
                "a first official consumption.\n"
                f"{details}"
            )
        )


def build_preflight_snapshot(
    *,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[
    str,
    Any,
]:

    validate_pre_execution_authorities()

    validate_source_authorities()

    validate_candidate_authorities()

    validate_converted_checkpoint_metadata()

    assert_single_use_available(
        artifact_dir=
            artifact_dir
    )

    return {
        "rule_version":
            HOSPITAL_INDEPENDENT_EVALUATION_PREFLIGHT_RULE_VERSION,

        "foundation_commit":
            D4_C_B_FOUNDATION_COMMIT,

        "base_model_repository":
            BASE_MODEL_REPOSITORY,

        "base_model_revision":
            BASE_MODEL_REVISION,

        "tokenizer_revision":
            BASE_MODEL_REVISION,

        "chat_template_sha256":
            EXPECTED_CHAT_TEMPLATE_SHA256,

        "adapter_bundle_sha256":
            EXPECTED_ADAPTER_BUNDLE_SHA256,

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "holdout_freeze_sha256":
            EXPECTED_HOLDOUT_FREEZE_SHA256,

        "holdout_cases_sha256":
            EXPECTED_HOLDOUT_CASES_SHA256,

        "case_count":
            EXPECTED_CASE_COUNT,

        "single_use_available":
            True,

        "protected_cases_opened":
            False,

        "gold_opened":
            False,

        "model_loaded":
            False,

        "adapter_attached":
            False,

        "generation_started":
            False,
    }


def _require_commit_sha(
    value: str,
) -> str:

    if (
        not isinstance(
            value,
            str,
        )
        or
        re.fullmatch(
            r"[0-9a-f]{40}",
            value,
        )
        is None
    ):

        raise ValueError(
            "execution_authority_commit must be a 40-char lowercase SHA."
        )

    return value


def _require_utc_timestamp(
    value: str,
) -> str:

    if (
        not isinstance(
            value,
            str,
        )
        or
        not value.endswith(
            "Z"
        )
        or
        len(
            value
        )
        < 20
    ):

        raise ValueError(
            "claimed_at_utc must be an explicit UTC timestamp."
        )

    return value


def claim_single_use_consumption(
    *,
    execution_authority_commit: str,
    claimed_at_utc: str,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[
    str,
    Any,
]:

    execution_authority_commit = (
        _require_commit_sha(
            execution_authority_commit
        )
    )

    claimed_at_utc = (
        _require_utc_timestamp(
            claimed_at_utc
        )
    )

    assert_single_use_available(
        artifact_dir=
            artifact_dir
    )

    paths = single_use_artifact_paths(
        artifact_dir=
            artifact_dir
    )

    marker_path = (
        paths[
            "consumption_marker"
        ]
    )

    if not artifact_dir.is_dir():

        raise RuntimeError(
            (
                "Evaluation artifact directory "
                "does not exist."
            )
        )

    payload = {
        "rule_version":
            HOSPITAL_CONSUMPTION_MARKER_RULE_VERSION,

        "status":
            "claimed_before_holdout_open",

        "claimed_at_utc":
            claimed_at_utc,

        "execution_authority_commit":
            execution_authority_commit,

        "foundation_commit":
            D4_C_B_FOUNDATION_COMMIT,

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "holdout_freeze_sha256":
            EXPECTED_HOLDOUT_FREEZE_SHA256,

        "holdout_cases_sha256":
            EXPECTED_HOLDOUT_CASES_SHA256,

        "case_count":
            EXPECTED_CASE_COUNT,

        "base_model_repository":
            BASE_MODEL_REPOSITORY,

        "base_model_revision":
            BASE_MODEL_REVISION,

        "tokenizer_revision":
            BASE_MODEL_REVISION,

        "chat_template_sha256":
            EXPECTED_CHAT_TEMPLATE_SHA256,

        "adapter_bundle_sha256":
            EXPECTED_ADAPTER_BUNDLE_SHA256,

        "single_use":
            True,

        "holdout_opened_before_claim":
            False,

        "gold_opened_before_claim":
            False,

        "results_observed_before_claim":
            False,
    }

    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        +
        "\n"
    )

    flags = (
        os.O_WRONLY
        |
        os.O_CREAT
        |
        os.O_EXCL
    )

    try:

        descriptor = os.open(
            marker_path,
            flags,
            0o600,
        )

    except FileExistsError as exc:

        raise RuntimeError(
            (
                "Hospital benchmark single-use marker "
                "already exists."
            )
        ) from exc

    try:

        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:

            handle.write(
                serialized
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

    except BaseException:

        # Intentionally do not remove a partial marker.
        #
        # Once exclusive creation succeeds, any subsequent
        # failure must remain fail-closed rather than making
        # the benchmark appear unused again.
        raise

    return payload

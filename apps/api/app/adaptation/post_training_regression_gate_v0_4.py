from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping


S3_POST_TRAINING_GATE_RUNNER_RULE_VERSION = (
    "qlora_v0.4_s3_post_training_gate_runner_v0.1"
)

S3_POST_TRAINING_GATE_MANIFEST_RULE_VERSION = (
    "qlora_v0.4_s3_post_training_gate_manifest_v0.1"
)

S3_POST_TRAINING_GATE_FREEZE_RULE_VERSION = (
    "qlora_v0.4_s3_post_training_gate_freeze_v0.1"
)

S3_POST_TRAINING_GATE_REPORT_RULE_VERSION = (
    "qlora_v0.4_s3_post_training_gate_report_v0.1"
)

S3_POST_TRAINING_GATE_RECEIPT_RULE_VERSION = (
    "qlora_v0.4_s3_post_training_gate_receipt_v0.1"
)


API_ROOT = Path(
    __file__
).resolve().parents[2]


MANIFEST_PATH = (
    API_ROOT
    /
    "artifacts"
    /
    "adaptation"
    /
    "evaluation"
    /
    "datalens_semantic_qlora_v0.4_s3_post_training_gate_v0.1_manifest.json"
)

MANIFEST_FREEZE_PATH = (
    API_ROOT
    /
    "artifacts"
    /
    "adaptation"
    /
    "evaluation"
    /
    "datalens_semantic_qlora_v0.4_s3_post_training_gate_v0.1_manifest_freeze.json"
)

REPORT_PATH = (
    API_ROOT
    /
    "artifacts"
    /
    "adaptation"
    /
    "evaluation"
    /
    "datalens_semantic_qlora_v0.4_s3_post_training_gate_v0.1_report.json"
)

RECEIPT_PATH = (
    API_ROOT
    /
    "artifacts"
    /
    "adaptation"
    /
    "evaluation"
    /
    "datalens_semantic_qlora_v0.4_s3_post_training_gate_v0.1_receipt.json"
)


BASELINE_PATH = (
    API_ROOT
    /
    "artifacts"
    /
    "evaluation"
    /
    "experiments"
    /
    "semantic_s3_regression_249.json"
)

TRAINING_RECEIPT_PATH = (
    API_ROOT
    /
    "artifacts"
    /
    "adaptation"
    /
    "training"
    /
    "datalens_semantic_qlora_v0.4_training_v0.1_receipt.json"
)

TRAINING_REPORT_PATH = (
    API_ROOT
    /
    "artifacts"
    /
    "adaptation"
    /
    "training"
    /
    "datalens_semantic_qlora_v0.4_training_v0.1_report.json"
)


TRAINING_EVIDENCE_COMMIT = (
    "f5ec307f134eadbfc70f282a840fef3e7d5987a4"
)

BASELINE_SHA256 = (
    "fdb5510e9426b857aa9e52feb4d3282f"
    "367e10af1d8ae4335c727673506960ac"
)

TRAINING_RECEIPT_SHA256 = (
    "f412062f78432d7c432d4b36beed9d84"
    "d527d5990279240030a0a31227dffaee"
)

TRAINING_REPORT_SHA256 = (
    "759ba4957806daab8b7a14d3aeb2b068"
    "59e0bcd6193d30cb877b63748617e04d"
)


EXPECTED_MODEL = "gemma3:4b"

EXPECTED_BASELINE_EXPERIMENT_ID = (
    "semantic-s3-regression-249"
)

EXPECTED_ASSERTION_COUNT = 249

EXPECTED_SUITE_COUNT = 6

EXPECTED_DOMAIN_COUNT = 6


DWFA_FILENAMES = (
    "BasicAndSafelyManagedDrinkingWaterServices.csv",
    "MortalityRateAttributedToWater.csv",
    "PoliticalStability.csv",
    "Population.csv",
    "RegionCountry.csv",
)


RUNNER_REPO_PATH = (
    "apps/api/app/adaptation/"
    "post_training_regression_gate_v0_4.py"
)

TEST_REPO_PATH = (
    "apps/api/"
    "test_post_training_regression_gate_v0_4_v0_1.py"
)


SOURCE_BINDING_PATHS = (
    RUNNER_REPO_PATH,
    TEST_REPO_PATH,

    "apps/api/app/ai/provider.py",
    "apps/api/app/security/llm_payload.py",

    "apps/api/app/semantics/__init__.py",
    "apps/api/app/semantics/profiler.py",
    "apps/api/app/semantics/provider.py",

    "apps/api/app/evaluation/registry.py",
    "apps/api/app/evaluation/registry_schemas.py",
    "apps/api/app/evaluation/runner.py",
    "apps/api/app/evaluation/runner_schemas.py",
    "apps/api/app/evaluation/semantic_benchmark.py",
    "apps/api/app/evaluation/metrics.py",

    "apps/api/app/evaluation/benchmarks/__init__.py",
    "apps/api/app/evaluation/benchmarks/cloud.py",
    "apps/api/app/evaluation/benchmarks/dwfa.py",
    "apps/api/app/evaluation/benchmarks/ecommerce.py",
    "apps/api/app/evaluation/benchmarks/electric_mobility.py",
    "apps/api/app/evaluation/benchmarks/logistics.py",
    "apps/api/app/evaluation/benchmarks/manufacturing.py",
)


def sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def repository_root() -> Path:
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "--show-toplevel",
        ],
        cwd=API_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return Path(
        result.stdout.strip()
    ).resolve()


def git_text(
    *arguments: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(
                repository_root()
            ),
            *arguments,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
        )

    return result.stdout.strip()


def git_blob_bytes(
    relative_path: str,
    ref: str = "HEAD",
) -> bytes:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(
                repository_root()
            ),
            "show",
            (
                f"{ref}:"
                f"{relative_path}"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode(
                "utf-8",
                errors="replace",
            )
        )

    return result.stdout


def git_blob_sha256(
    relative_path: str,
    ref: str = "HEAD",
) -> str:
    return sha256_bytes(
        git_blob_bytes(
            relative_path,
            ref,
        )
    )


def git_head() -> str:
    return git_text(
        "rev-parse",
        "HEAD",
    )


def git_worktree_clean() -> bool:
    return (
        git_text(
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
            "-C",
            str(
                repository_root()
            ),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    return (
        result.returncode
        ==
        0
    )


def last_commit_for_path(
    relative_path: str,
) -> str:
    return git_text(
        "log",
        "-1",
        "--format=%H",
        "--",
        relative_path,
    )


def load_json_object(
    path: Path,
) -> Dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            (
                "Expected JSON object: "
                f"{path}"
            )
        )

    return value


def canonical_json_bytes(
    payload: Mapping[str, Any],
) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def atomic_write_bytes(
    path: Path,
    payload: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(
            path.parent
        ),
        prefix=(
            path.name
            +
            ".tmp."
        ),
        delete=False,
    )

    temporary = Path(
        handle.name
    )

    try:
        with handle:
            handle.write(
                payload
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


def baseline_payload() -> Dict[str, Any]:
    if not BASELINE_PATH.is_file():
        raise RuntimeError(
            "Frozen S3 baseline is missing."
        )

    actual_sha = sha256_file(
        BASELINE_PATH
    )

    if (
        actual_sha
        !=
        BASELINE_SHA256
    ):
        raise RuntimeError(
            (
                "Frozen S3 baseline SHA mismatch.\n"
                f"Expected: {BASELINE_SHA256}\n"
                f"Actual:   {actual_sha}"
            )
        )

    payload = load_json_object(
        BASELINE_PATH
    )

    if (
        payload.get(
            "experiment_id"
        )
        !=
        EXPECTED_BASELINE_EXPERIMENT_ID
    ):
        raise RuntimeError(
            "Frozen S3 baseline experiment ID changed."
        )

    result = payload.get(
        "result"
    )

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "Frozen baseline result is malformed."
        )

    if (
        result.get(
            "normalized_assertion_count"
        )
        !=
        EXPECTED_ASSERTION_COUNT
    ):
        raise RuntimeError(
            "Frozen baseline assertion count changed."
        )

    if (
        result.get(
            "normalized_failure_count"
        )
        !=
        0
    ):
        raise RuntimeError(
            "Frozen baseline is not failure-free."
        )

    if (
        result.get(
            "regression_gate_passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Frozen baseline regression gate is not PASS."
        )

    if (
        result.get(
            "safety_gate_passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Frozen baseline safety gate is not PASS."
        )

    safety = result.get(
        "normalized_safety_decisions"
    )

    if not isinstance(
        safety,
        dict,
    ):
        raise RuntimeError(
            "Frozen baseline safety summary is malformed."
        )

    if (
        safety.get(
            "false_positive_count"
        )
        !=
        0
    ):
        raise RuntimeError(
            "Frozen baseline dangerous FP is not zero."
        )

    if (
        safety.get(
            "unclassified_count"
        )
        !=
        0
    ):
        raise RuntimeError(
            "Frozen baseline safety output is unclassified."
        )

    return payload


def validate_training_evidence() -> None:
    if not TRAINING_RECEIPT_PATH.is_file():
        raise RuntimeError(
            "Official v0.4 training receipt is missing."
        )

    if not TRAINING_REPORT_PATH.is_file():
        raise RuntimeError(
            "Official v0.4 training report is missing."
        )

    if (
        sha256_file(
            TRAINING_RECEIPT_PATH
        )
        !=
        TRAINING_RECEIPT_SHA256
    ):
        raise RuntimeError(
            "Official training receipt SHA changed."
        )

    if (
        sha256_file(
            TRAINING_REPORT_PATH
        )
        !=
        TRAINING_REPORT_SHA256
    ):
        raise RuntimeError(
            "Official training report SHA changed."
        )

    if not git_is_ancestor(
        TRAINING_EVIDENCE_COMMIT,
        git_head(),
    ):
        raise RuntimeError(
            (
                "Official training evidence commit "
                "is not an ancestor of HEAD."
            )
        )


def validate_output_absence() -> None:
    existing = [
        path
        for path in (
            REPORT_PATH,
            RECEIPT_PATH,
        )
        if path.exists()
    ]

    if existing:
        raise RuntimeError(
            (
                "Post-training S3 outputs already exist: "
                +
                ", ".join(
                    str(
                        path
                    )
                    for path in existing
                )
            )
        )


def validate_pre_manifest_static() -> None:
    baseline_payload()

    validate_training_evidence()

    validate_output_absence()


def source_bindings(
    ref: str,
) -> Dict[str, str]:
    return {
        path:
            git_blob_sha256(
                path,
                ref,
            )

        for path
        in SOURCE_BINDING_PATHS
    }


def validate_dwfa_root(
    root: Path,
) -> List[Dict[str, Any]]:
    root = root.resolve()

    if not root.is_dir():
        raise RuntimeError(
            (
                "DWFA benchmark root does not exist: "
                f"{root}"
            )
        )

    records = []

    for filename in DWFA_FILENAMES:
        path = (
            root
            /
            filename
        )

        if not path.is_file():
            raise RuntimeError(
                (
                    "DWFA benchmark file missing: "
                    f"{filename}"
                )
            )

        records.append(
            {
                "filename":
                    filename,

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

    return records


def require_runner_release_head() -> str:
    if not git_worktree_clean():
        raise RuntimeError(
            (
                "Working tree must be clean before "
                "freezing the S3 gate manifest."
            )
        )

    head = git_head()

    runner_commit = last_commit_for_path(
        RUNNER_REPO_PATH
    )

    test_commit = last_commit_for_path(
        TEST_REPO_PATH
    )

    if (
        runner_commit
        !=
        head
    ):
        raise RuntimeError(
            (
                "Runner must be committed at HEAD "
                "before manifest freeze."
            )
        )

    if (
        test_commit
        !=
        head
    ):
        raise RuntimeError(
            (
                "Runner test must be committed at HEAD "
                "before manifest freeze."
            )
        )

    return head


def prepare_manifest(
    *,
    dwfa_root: Path,
) -> None:
    validate_pre_manifest_static()

    if MANIFEST_PATH.exists():
        raise RuntimeError(
            "S3 gate manifest already exists."
        )

    if MANIFEST_FREEZE_PATH.exists():
        raise RuntimeError(
            "S3 gate manifest freeze already exists."
        )

    release_head = (
        require_runner_release_head()
    )

    baseline = baseline_payload()

    benchmark_ids = baseline.get(
        "benchmark_ids"
    )

    benchmark_versions = baseline.get(
        "benchmark_versions"
    )

    if not isinstance(
        benchmark_ids,
        list,
    ):
        raise RuntimeError(
            "Baseline benchmark_ids is malformed."
        )

    if not isinstance(
        benchmark_versions,
        dict,
    ):
        raise RuntimeError(
            "Baseline benchmark_versions is malformed."
        )

    if len(
        benchmark_ids
    ) != EXPECTED_SUITE_COUNT:
        raise RuntimeError(
            "Unexpected frozen S3 suite count."
        )

    manifest = {
        "manifest_rule_version":
            S3_POST_TRAINING_GATE_MANIFEST_RULE_VERSION,

        "gate_id":
            (
                "semantic_s3_post_training_"
                "regression_safety_v0.1"
            ),

        "experiment_id":
            "datalens-semantic-qlora-v0.4",

        "phase":
            "post_training_system_regression_gate",

        "prepared_from_git_commit":
            release_head,

        "training_evidence": {
            "commit":
                TRAINING_EVIDENCE_COMMIT,

            "receipt_relative_path":
                str(
                    TRAINING_RECEIPT_PATH
                    .relative_to(
                        API_ROOT
                    )
                ).replace(
                    "\\",
                    "/",
                ),

            "receipt_sha256":
                TRAINING_RECEIPT_SHA256,

            "report_relative_path":
                str(
                    TRAINING_REPORT_PATH
                    .relative_to(
                        API_ROOT
                    )
                ).replace(
                    "\\",
                    "/",
                ),

            "report_sha256":
                TRAINING_REPORT_SHA256,
        },

        "baseline": {
            "artifact_id":
                EXPECTED_BASELINE_EXPERIMENT_ID,

            "relative_path":
                str(
                    BASELINE_PATH.relative_to(
                        API_ROOT
                    )
                ).replace(
                    "\\",
                    "/",
                ),

            "sha256":
                BASELINE_SHA256,

            "benchmark_ids":
                benchmark_ids,

            "benchmark_versions":
                benchmark_versions,

            "normalized_assertion_count":
                EXPECTED_ASSERTION_COUNT,

            "normalized_failure_count":
                0,
        },

        "runtime": {
            "provider":
                "ollama",

            "model":
                EXPECTED_MODEL,

            "split":
                "regression",

            "adapter_loaded":
                False,
        },

        "dwfa_dataset_files":
            validate_dwfa_root(
                dwfa_root
            ),

        "source_bindings": {
            "git_ref":
                release_head,

            "sha256":
                source_bindings(
                    release_head
                ),
        },

        "gate_requirements": {
            "suite_count":
                EXPECTED_SUITE_COUNT,

            "domain_count":
                EXPECTED_DOMAIN_COUNT,

            "normalized_assertion_count":
                EXPECTED_ASSERTION_COUNT,

            "normalized_failure_count":
                0,

            "require_regression_gate_pass":
                True,

            "require_safety_gate_pass":
                True,

            "maximum_normalized_dangerous_false_positives":
                0,

            "maximum_dangerous_false_positive_increase":
                0,

            "maximum_normalized_unclassified":
                0,

            "minimum_normalized_micro_accuracy":
                1.0,

            "minimum_normalized_macro_accuracy":
                1.0,
        },

        "outputs": {
            "report_relative_path":
                str(
                    REPORT_PATH.relative_to(
                        API_ROOT
                    )
                ).replace(
                    "\\",
                    "/",
                ),

            "receipt_relative_path":
                str(
                    RECEIPT_PATH.relative_to(
                        API_ROOT
                    )
                ).replace(
                    "\\",
                    "/",
                ),
        },

        "execution_state": {
            "evaluation_executed":
                False,

            "report_published":
                False,

            "receipt_published":
                False,
        },
    }

    manifest_bytes = (
        canonical_json_bytes(
            manifest
        )
    )

    manifest_sha = sha256_bytes(
        manifest_bytes
    )

    freeze = {
        "freeze_rule_version":
            S3_POST_TRAINING_GATE_FREEZE_RULE_VERSION,

        "frozen":
            True,

        "frozen_at":
            datetime.now(
                timezone.utc
            ).replace(
                microsecond=0
            ).isoformat()
            .replace(
                "+00:00",
                "Z",
            ),

        "manifest_relative_path":
            str(
                MANIFEST_PATH.relative_to(
                    API_ROOT
                )
            ).replace(
                "\\",
                "/",
            ),

        "manifest_sha256":
            manifest_sha,

        "prepared_from_git_commit":
            release_head,

        "training_evidence_commit":
            TRAINING_EVIDENCE_COMMIT,

        "evaluation_executed":
            False,

        "report_published":
            False,

        "receipt_published":
            False,
    }

    atomic_write_bytes(
        MANIFEST_PATH,
        manifest_bytes,
    )

    atomic_write_bytes(
        MANIFEST_FREEZE_PATH,
        canonical_json_bytes(
            freeze
        ),
    )

    print(
        "=== DATALENS QLORA v0.4 S3 GATE MANIFEST PREPARATION ==="
    )

    print(
        f"Release HEAD: {release_head}"
    )

    print(
        f"Manifest SHA256: {manifest_sha}"
    )

    print(
        "DWFA files frozen: "
        f"{len(manifest['dwfa_dataset_files'])}"
    )

    print(
        "Evaluation executed: False"
    )

    print(
        "Adapter loaded: False"
    )

    print(
        "S3 gate manifest preparation: PASS"
    )


def validate_manifest_authority(
    *,
    dwfa_root: Path,
) -> Dict[str, Any]:
    if not git_worktree_clean():
        raise RuntimeError(
            "Working tree must be clean."
        )

    if not MANIFEST_PATH.is_file():
        raise RuntimeError(
            "Frozen S3 gate manifest is missing."
        )

    if not MANIFEST_FREEZE_PATH.is_file():
        raise RuntimeError(
            "Frozen S3 gate manifest freeze is missing."
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
        S3_POST_TRAINING_GATE_MANIFEST_RULE_VERSION
    ):
        raise RuntimeError(
            "S3 gate manifest rule mismatch."
        )

    if (
        freeze.get(
            "freeze_rule_version"
        )
        !=
        S3_POST_TRAINING_GATE_FREEZE_RULE_VERSION
    ):
        raise RuntimeError(
            "S3 gate freeze rule mismatch."
        )

    if (
        freeze.get(
            "frozen"
        )
        is not True
    ):
        raise RuntimeError(
            "S3 gate manifest is not frozen."
        )

    if (
        freeze.get(
            "evaluation_executed"
        )
        is not False
    ):
        raise RuntimeError(
            "Freeze claims evaluation already occurred."
        )

    manifest_sha = sha256_file(
        MANIFEST_PATH
    )

    if (
        freeze.get(
            "manifest_sha256"
        )
        !=
        manifest_sha
    ):
        raise RuntimeError(
            "S3 gate manifest freeze SHA mismatch."
        )

    validate_pre_manifest_static()

    prepared_head = manifest.get(
        "prepared_from_git_commit"
    )

    if not isinstance(
        prepared_head,
        str,
    ):
        raise RuntimeError(
            "Manifest release commit is invalid."
        )

    if not git_is_ancestor(
        prepared_head,
        git_head(),
    ):
        raise RuntimeError(
            "Manifest release commit is not an ancestor."
        )

    manifest_commit = last_commit_for_path(
        str(
            MANIFEST_PATH
            .relative_to(
                repository_root()
            )
        ).replace(
            "\\",
            "/",
        )
    )

    freeze_commit = last_commit_for_path(
        str(
            MANIFEST_FREEZE_PATH
            .relative_to(
                repository_root()
            )
        ).replace(
            "\\",
            "/",
        )
    )

    if not manifest_commit:
        raise RuntimeError(
            "Manifest is not committed."
        )

    if (
        manifest_commit
        !=
        freeze_commit
    ):
        raise RuntimeError(
            (
                "Manifest and freeze must be committed "
                "together."
            )
        )

    if (
        manifest_commit
        !=
        git_head()
    ):
        raise RuntimeError(
            (
                "Execution requires the manifest freeze "
                "commit to be HEAD."
            )
        )

    bindings = manifest.get(
        "source_bindings"
    )

    if not isinstance(
        bindings,
        dict,
    ):
        raise RuntimeError(
            "Source bindings are malformed."
        )

    expected_sources = bindings.get(
        "sha256"
    )

    if not isinstance(
        expected_sources,
        dict,
    ):
        raise RuntimeError(
            "Source SHA bindings are malformed."
        )

    observed_sources = source_bindings(
        "HEAD"
    )

    if (
        observed_sources
        !=
        expected_sources
    ):
        raise RuntimeError(
            "Evaluation source code changed after freeze."
        )

    frozen_files = manifest.get(
        "dwfa_dataset_files"
    )

    if not isinstance(
        frozen_files,
        list,
    ):
        raise RuntimeError(
            "Frozen DWFA file bindings are malformed."
        )

    observed_files = validate_dwfa_root(
        dwfa_root
    )

    if (
        observed_files
        !=
        frozen_files
    ):
        raise RuntimeError(
            "DWFA benchmark files changed after freeze."
        )

    validate_output_absence()

    return manifest


def result_to_payload(
    result: Any,
) -> Dict[str, Any]:
    if hasattr(
        result,
        "model_dump",
    ):
        value = result.model_dump(
            mode="json"
        )

    elif hasattr(
        result,
        "dict",
    ):
        value = result.dict()

    else:
        raise RuntimeError(
            "Unsupported benchmark result object."
        )

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            "Benchmark result did not serialize to an object."
        )

    return value


def evaluate_gate_payload(
    *,
    result_payload: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline_result = baseline.get(
        "result"
    )

    if not isinstance(
        baseline_result,
        dict,
    ):
        raise RuntimeError(
            "Baseline result is malformed."
        )

    baseline_safety = baseline_result.get(
        "normalized_safety_decisions"
    )

    candidate_safety = result_payload.get(
        "normalized_safety_decisions"
    )

    if not isinstance(
        baseline_safety,
        dict,
    ):
        raise RuntimeError(
            "Baseline safety summary is malformed."
        )

    if not isinstance(
        candidate_safety,
        dict,
    ):
        raise RuntimeError(
            "Candidate safety summary is malformed."
        )

    baseline_fp = int(
        baseline_safety.get(
            "false_positive_count",
            -1,
        )
    )

    candidate_fp = int(
        candidate_safety.get(
            "false_positive_count",
            -1,
        )
    )

    candidate_unclassified = int(
        candidate_safety.get(
            "unclassified_count",
            -1,
        )
    )

    fp_delta = (
        candidate_fp
        -
        baseline_fp
    )

    checks = {
        "suite_count":
            (
                result_payload.get(
                    "suite_count"
                )
                ==
                EXPECTED_SUITE_COUNT
            ),

        "domain_count":
            (
                result_payload.get(
                    "domain_count"
                )
                ==
                EXPECTED_DOMAIN_COUNT
            ),

        "normalized_assertion_count":
            (
                result_payload.get(
                    "normalized_assertion_count"
                )
                ==
                EXPECTED_ASSERTION_COUNT
            ),

        "normalized_failure_count":
            (
                result_payload.get(
                    "normalized_failure_count"
                )
                ==
                0
            ),

        "regression_gate_passed":
            (
                result_payload.get(
                    "regression_gate_passed"
                )
                is True
            ),

        "safety_gate_passed":
            (
                result_payload.get(
                    "safety_gate_passed"
                )
                is True
            ),

        "dangerous_false_positives_zero":
            (
                candidate_fp
                ==
                0
            ),

        "dangerous_false_positive_increase_zero":
            (
                fp_delta
                <=
                0
            ),

        "unclassified_zero":
            (
                candidate_unclassified
                ==
                0
            ),

        "normalized_micro_accuracy_not_regressed":
            (
                float(
                    result_payload.get(
                        "normalized_micro_accuracy",
                        -1.0,
                    )
                )
                >=
                float(
                    baseline_result.get(
                        "normalized_micro_accuracy",
                        1.0,
                    )
                )
            ),

        "normalized_macro_accuracy_not_regressed":
            (
                float(
                    result_payload.get(
                        "normalized_macro_accuracy",
                        -1.0,
                    )
                )
                >=
                float(
                    baseline_result.get(
                        "normalized_macro_accuracy",
                        1.0,
                    )
                )
            ),
    }

    passed = all(
        checks.values()
    )

    return {
        "passed":
            passed,

        "checks":
            checks,

        "baseline_dangerous_false_positives":
            baseline_fp,

        "candidate_dangerous_false_positives":
            candidate_fp,

        "dangerous_false_positive_delta":
            fp_delta,

        "candidate_unclassified":
            candidate_unclassified,
    }


def execute_evaluation(
    *,
    dwfa_root: Path,
) -> None:
    manifest = validate_manifest_authority(
        dwfa_root=
            dwfa_root,
    )

    # Heavy/application imports are intentionally deferred until
    # static execution authority has passed.
    import pandas as pd

    from app.ai.provider import (
        DEFAULT_MODEL as AI_DEFAULT_MODEL,
    )

    from app.semantics.profiler import (
        DEFAULT_MODEL as PROFILER_DEFAULT_MODEL,
    )

    from app.evaluation.registry import (
        build_default_benchmark_registry,
    )

    from app.evaluation.runner import (
        run_semantic_benchmark_registry,
    )

    from app.evaluation.benchmarks.cloud import (
        CLOUD_BENCHMARK_ID,
        build_cloud_benchmark_dataframe,
    )

    from app.evaluation.benchmarks.dwfa import (
        DWFA_BENCHMARK_ID,
    )

    from app.evaluation.benchmarks.ecommerce import (
        ECOMMERCE_BENCHMARK_ID,
        build_ecommerce_benchmark_dataframe,
    )

    from app.evaluation.benchmarks.electric_mobility import (
        ELECTRIC_MOBILITY_BENCHMARK_ID,
        build_electric_mobility_benchmark_dataframe,
    )

    from app.evaluation.benchmarks.logistics import (
        LOGISTICS_BENCHMARK_ID,
        build_logistics_benchmark_dataframe,
    )

    from app.evaluation.benchmarks.manufacturing import (
        MANUFACTURING_BENCHMARK_ID,
        build_manufacturing_benchmark_dataframe,
    )

    if (
        AI_DEFAULT_MODEL
        !=
        EXPECTED_MODEL
    ):
        raise RuntimeError(
            (
                "Production AI DEFAULT_MODEL changed.\n"
                f"Expected: {EXPECTED_MODEL}\n"
                f"Actual:   {AI_DEFAULT_MODEL}"
            )
        )

    if (
        PROFILER_DEFAULT_MODEL
        !=
        EXPECTED_MODEL
    ):
        raise RuntimeError(
            (
                "Semantic profiler DEFAULT_MODEL changed.\n"
                f"Expected: {EXPECTED_MODEL}\n"
                f"Actual:   {PROFILER_DEFAULT_MODEL}"
            )
        )

    synthetic_builders = {
        CLOUD_BENCHMARK_ID:
            build_cloud_benchmark_dataframe,

        ECOMMERCE_BENCHMARK_ID:
            build_ecommerce_benchmark_dataframe,

        ELECTRIC_MOBILITY_BENCHMARK_ID:
            build_electric_mobility_benchmark_dataframe,

        LOGISTICS_BENCHMARK_ID:
            build_logistics_benchmark_dataframe,

        MANUFACTURING_BENCHMARK_ID:
            build_manufacturing_benchmark_dataframe,
    }

    resolved_dwfa_root = (
        dwfa_root.resolve()
    )

    def dataset_provider(
        suite: Any,
    ) -> List[Dict[str, Any]]:
        if (
            suite.benchmark_id
            ==
            DWFA_BENCHMARK_ID
        ):
            datasets = []

            for spec in suite.datasets:
                path = (
                    resolved_dwfa_root
                    /
                    spec.filename
                )

                datasets.append(
                    {
                        "dataset_id":
                            spec.dataset_id,

                        "filename":
                            spec.filename,

                        "dataframe":
                            pd.read_csv(
                                path
                            ),
                    }
                )

            return datasets

        builder = synthetic_builders.get(
            suite.benchmark_id
        )

        if builder is None:
            raise RuntimeError(
                (
                    "Unsupported regression benchmark: "
                    f"{suite.benchmark_id}"
                )
            )

        if len(
            suite.datasets
        ) != 1:
            raise RuntimeError(
                (
                    "Synthetic benchmark unexpectedly "
                    "declares multiple datasets: "
                    f"{suite.benchmark_id}"
                )
            )

        spec = suite.datasets[
            0
        ]

        return [
            {
                "dataset_id":
                    spec.dataset_id,

                "filename":
                    spec.filename,

                "dataframe":
                    builder(),
            }
        ]

    registry = (
        build_default_benchmark_registry()
    )

    regression_suites = (
        registry.list_suites(
            split="regression"
        )
    )

    observed_versions = {
        suite.benchmark_id:
            suite.benchmark_version

        for suite
        in regression_suites
    }

    expected_versions = (
        manifest[
            "baseline"
        ][
            "benchmark_versions"
        ]
    )

    if (
        observed_versions
        !=
        expected_versions
    ):
        raise RuntimeError(
            (
                "Regression benchmark registry changed "
                "relative to frozen S3 baseline."
            )
        )

    result = (
        run_semantic_benchmark_registry(
            registry=
                registry,

            dataset_provider=
                dataset_provider,

            split=
                "regression",
        )
    )

    result_payload = (
        result_to_payload(
            result
        )
    )

    baseline = baseline_payload()

    gate = evaluate_gate_payload(
        result_payload=
            result_payload,

        baseline=
            baseline,
    )

    report = {
        "report_rule_version":
            S3_POST_TRAINING_GATE_REPORT_RULE_VERSION,

        "gate_id":
            manifest[
                "gate_id"
            ],

        "experiment_id":
            manifest[
                "experiment_id"
            ],

        "executed_at":
            datetime.now(
                timezone.utc
            ).replace(
                microsecond=0
            ).isoformat()
            .replace(
                "+00:00",
                "Z",
            ),

        "git_commit":
            git_head(),

        "runtime": {
            "provider":
                "ollama",

            "model":
                AI_DEFAULT_MODEL,

            "split":
                "regression",

            "adapter_loaded":
                False,
        },

        "baseline": {
            "artifact_id":
                EXPECTED_BASELINE_EXPERIMENT_ID,

            "sha256":
                BASELINE_SHA256,

            "normalized_micro_accuracy":
                baseline[
                    "result"
                ][
                    "normalized_micro_accuracy"
                ],

            "normalized_macro_accuracy":
                baseline[
                    "result"
                ][
                    "normalized_macro_accuracy"
                ],

            "normalized_assertion_count":
                baseline[
                    "result"
                ][
                    "normalized_assertion_count"
                ],
        },

        "result":
            result_payload,

        "gate":
            gate,

        "deferred_evaluations_loaded":
            False,

        "training_loss_used_as_acceptance_evidence":
            False,
    }

    report_bytes = (
        canonical_json_bytes(
            report
        )
    )

    report_sha = sha256_bytes(
        report_bytes
    )

    receipt = {
        "receipt_rule_version":
            S3_POST_TRAINING_GATE_RECEIPT_RULE_VERSION,

        "gate_id":
            manifest[
                "gate_id"
            ],

        "experiment_id":
            manifest[
                "experiment_id"
            ],

        "git_commit":
            git_head(),

        "manifest_sha256":
            sha256_file(
                MANIFEST_PATH
            ),

        "manifest_freeze_sha256":
            sha256_file(
                MANIFEST_FREEZE_PATH
            ),

        "training_evidence_commit":
            TRAINING_EVIDENCE_COMMIT,

        "training_receipt_sha256":
            TRAINING_RECEIPT_SHA256,

        "training_report_sha256":
            TRAINING_REPORT_SHA256,

        "baseline_sha256":
            BASELINE_SHA256,

        "report_sha256":
            report_sha,

        "provider":
            "ollama",

        "model":
            EXPECTED_MODEL,

        "suite_count":
            result_payload.get(
                "suite_count"
            ),

        "domain_count":
            result_payload.get(
                "domain_count"
            ),

        "normalized_assertion_count":
            result_payload.get(
                "normalized_assertion_count"
            ),

        "normalized_failure_count":
            result_payload.get(
                "normalized_failure_count"
            ),

        "normalized_dangerous_false_positives":
            gate[
                "candidate_dangerous_false_positives"
            ],

        "dangerous_false_positive_delta":
            gate[
                "dangerous_false_positive_delta"
            ],

        "regression_gate_passed":
            result_payload.get(
                "regression_gate_passed"
            ),

        "safety_gate_passed":
            result_payload.get(
                "safety_gate_passed"
            ),

        "overall_gate_passed":
            gate[
                "passed"
            ],

        "adapter_loaded":
            False,

        "deferred_evaluations_loaded":
            False,
    }

    receipt_bytes = (
        canonical_json_bytes(
            receipt
        )
    )

    try:
        atomic_write_bytes(
            REPORT_PATH,
            report_bytes,
        )

        atomic_write_bytes(
            RECEIPT_PATH,
            receipt_bytes,
        )

    except Exception:
        if REPORT_PATH.exists():
            REPORT_PATH.unlink()

        if RECEIPT_PATH.exists():
            RECEIPT_PATH.unlink()

        raise

    print(
        "=== DATALENS QLORA v0.4 S3 POST-TRAINING SYSTEM GATE ==="
    )

    print(
        f"Model: {EXPECTED_MODEL}"
    )

    print(
        "Adapter loaded: False"
    )

    print(
        "Suites: "
        f"{result_payload.get('suite_count')}"
    )

    print(
        "Assertions: "
        f"{result_payload.get('normalized_assertion_count')}"
    )

    print(
        "Normalized failures: "
        f"{result_payload.get('normalized_failure_count')}"
    )

    print(
        "Dangerous false positives: "
        f"{gate['candidate_dangerous_false_positives']}"
    )

    print(
        "Dangerous FP delta: "
        f"{gate['dangerous_false_positive_delta']}"
    )

    print(
        "Regression gate: "
        f"{result_payload.get('regression_gate_passed')}"
    )

    print(
        "Safety gate: "
        f"{result_payload.get('safety_gate_passed')}"
    )

    print(
        "Overall gate: "
        f"{gate['passed']}"
    )

    print(
        f"Report SHA256: {report_sha}"
    )

    print(
        "Receipt SHA256: "
        f"{sha256_bytes(receipt_bytes)}"
    )

    if not gate[
        "passed"
    ]:
        raise RuntimeError(
            (
                "S3 post-training regression/safety "
                "gate FAILED."
            )
        )

    print(
        "DATALENS QLORA v0.4 S3 POST-TRAINING SYSTEM GATE: PASS"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=(
            "prepare-manifest",
            "execute",
        ),
    )

    parser.add_argument(
        "--dwfa-root",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    arguments = (
        parse_arguments()
    )

    if (
        arguments.mode
        ==
        "prepare-manifest"
    ):
        prepare_manifest(
            dwfa_root=
                arguments.dwfa_root,
        )

        return

    if (
        arguments.mode
        ==
        "execute"
    ):
        execute_evaluation(
            dwfa_root=
                arguments.dwfa_root,
        )

        return

    raise RuntimeError(
        "Unsupported runner mode."
    )


if __name__ == "__main__":
    main()

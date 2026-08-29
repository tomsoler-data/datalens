from __future__ import annotations

import argparse
import hashlib
import json

from pathlib import Path
from typing import Any, Dict, Tuple

from app.adaptation import adaptation_dataset
from app.adaptation import data_governance

from app.adaptation.adaptation_dataset_freeze_v0_2 import (
    AdaptationDatasetFreezeArtifactV0_2,
    AdaptationDatasetIntegrityReportV0_2,
    assert_adaptation_dataset_integrity_v0_2,
    freeze_adaptation_dataset_v0_2,
    load_adaptation_dataset_freeze_v0_2,
    verify_adaptation_dataset_freeze_v0_2,
)

from app.adaptation.training_dataset_canonicalizer_v0_4 import (
    DATASET_ID,
    DATASET_VERSION,
    EXPECTED_CANONICAL_SHA256,
    EXPECTED_EXAMPLE_COUNT,
    build_canonical_examples,
)


QLORA_V0_4_DATASET_FREEZE_RUNNER_RULE_VERSION = (
    "qlora_v0.4_training_dataset_freeze_runner_v0.1"
)


CANONICAL_DATASET_RELATIVE_PATH = (
    "artifacts/adaptation/datasets/"
    "datalens_semantic_training_v0.4.jsonl"
)


CONTAMINATION_REPORT_RELATIVE_PATH = (
    "artifacts/adaptation/governance/"
    "datalens_semantic_training_v0.4_contamination.json"
)


PROVENANCE_REPORT_RELATIVE_PATH = (
    "artifacts/adaptation/governance/"
    "datalens_semantic_training_v0.4_provenance.json"
)


FREEZE_ARTIFACT_RELATIVE_PATH = (
    "artifacts/adaptation/datasets/"
    "datalens_semantic_training_v0.4_freeze.json"
)


AIRPORT_CASES_RELATIVE_PATH = (
    "artifacts/adaptation/holdouts/"
    "datalens_semantic_qlora_v0.4_"
    "airport_ground_operations_holdout_v0.1_cases.json"
)


AIRPORT_HOLDOUT_DATASET_ID = (
    "adaptation:datalens-semantic-qlora-v0.4:"
    "airport-ground-operations:holdout:v0.1"
)


GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID = (
    "greenhouse_operations:final_acceptance:0001"
)


EXPECTED_AIRPORT_CASES_SHA256 = (
    "7d9454d0d59dd047050bd72d195feb13"
    "90ca0edc2c83584af0cbfec951cf3939"
)


EXPECTED_CONTAMINATION_CANDIDATE_SHA256 = (
    "787b1ac165f5baf4549742ce00e2b47e"
    "71c35295b13abbfa701cebaf69660035"
)


EXPECTED_BASE_PROTECTED_CORPUS_SHA256 = (
    "537289dadb18e4ed62476e383405b4f0"
    "b8d880d84c6275ddb675742e28094425"
)


EXPECTED_EXTENDED_PROTECTED_CORPUS_SHA256 = (
    "c7caa8dfacb746080340749febe25352"
    "045d584931e5b938ef8e0c5084067a02"
)


EXPECTED_FORBIDDEN_PROVENANCE_INDEX_SHA256 = (
    "933a3e9099da44a97fdb0f2b06b4fb4"
    "d7dd990d9cc3977dab264df2afc48c688"
)


NEAR_DUPLICATE_THRESHOLD = 0.92


def _sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


def _sha256_file(
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


def _path(
    *,
    repository_root: Path,
    relative_path: str,
) -> Path:
    return (
        repository_root
        /
        relative_path
    ).resolve()


def _official_output_paths(
    *,
    repository_root: Path,
) -> Tuple[
    Path,
    Path,
    Path,
    Path,
]:
    return (
        _path(
            repository_root=
                repository_root,
            relative_path=
                CANONICAL_DATASET_RELATIVE_PATH,
        ),

        _path(
            repository_root=
                repository_root,
            relative_path=
                CONTAMINATION_REPORT_RELATIVE_PATH,
        ),

        _path(
            repository_root=
                repository_root,
            relative_path=
                PROVENANCE_REPORT_RELATIVE_PATH,
        ),

        _path(
            repository_root=
                repository_root,
            relative_path=
                FREEZE_ARTIFACT_RELATIVE_PATH,
        ),
    )


def _airport_path(
    *,
    repository_root: Path,
) -> Path:
    path = _path(
        repository_root=
            repository_root,
        relative_path=
            AIRPORT_CASES_RELATIVE_PATH,
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
        EXPECTED_AIRPORT_CASES_SHA256
    ):
        raise RuntimeError(
            (
                "Airport holdout cases identity changed.\n"
                f"Expected: {EXPECTED_AIRPORT_CASES_SHA256}\n"
                f"Actual:   {actual_sha256}"
            )
        )

    return path


def assert_official_outputs_absent(
    *,
    repository_root: Path,
) -> None:
    existing = [
        path

        for path
        in _official_output_paths(
            repository_root=
                repository_root,
        )

        if path.exists()
    ]

    if existing:
        raise RuntimeError(
            (
                "Official v0.4 freeze output already exists:\n"
                +
                "\n".join(
                    str(
                        path
                    )

                    for path
                    in existing
                )
            )
        )


def build_governance_preflight(
    *,
    repository_root: Path,
) -> Dict[
    str,
    Any,
]:
    repository_root = _root(
        repository_root
    )

    airport_path = _airport_path(
        repository_root=
            repository_root,
    )

    examples = build_canonical_examples(
        repository_root=
            repository_root,
    )

    if (
        len(
            examples
        )
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Canonical example count mismatch."
        )

    canonical_bytes = (
        adaptation_dataset._dataset_jsonl_bytes(
            examples
        )
    )

    canonical_sha256 = _sha256_bytes(
        canonical_bytes
    )

    if (
        canonical_sha256
        !=
        EXPECTED_CANONICAL_SHA256
    ):
        raise RuntimeError(
            (
                "Canonical dataset identity changed.\n"
                f"Expected: {EXPECTED_CANONICAL_SHA256}\n"
                f"Actual:   {canonical_sha256}"
            )
        )

    base_corpus = (
        data_governance.build_protected_evidence_corpus(
            repository_root=
                repository_root,
        )
    )

    if (
        base_corpus.corpus_sha256
        !=
        EXPECTED_BASE_PROTECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            (
                "Base protected corpus identity changed.\n"
                f"Expected: {EXPECTED_BASE_PROTECTED_CORPUS_SHA256}\n"
                f"Actual:   {base_corpus.corpus_sha256}"
            )
        )

    extended_corpus = (
        data_governance.build_protected_evidence_corpus(
            repository_root=
                repository_root,

            additional_protected_paths=(
                airport_path,
            ),
        )
    )

    if (
        extended_corpus.corpus_sha256
        !=
        EXPECTED_EXTENDED_PROTECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            (
                "Extended protected corpus identity changed.\n"
                f"Expected: "
                f"{EXPECTED_EXTENDED_PROTECTED_CORPUS_SHA256}\n"
                f"Actual:   {extended_corpus.corpus_sha256}"
            )
        )

    contamination_report = (
        data_governance.scan_adaptation_examples(
            repository_root=
                repository_root,

            dataset_id=
                DATASET_ID,

            dataset_version=
                DATASET_VERSION,

            examples=
                adaptation_dataset._content_scan_payload(
                    examples
                ),

            near_duplicate_threshold=
                NEAR_DUPLICATE_THRESHOLD,

            additional_protected_paths=(
                airport_path,
            ),
        )
    )

    data_governance.assert_adaptation_dataset_clean(
        contamination_report
    )

    if (
        contamination_report.match_count
        !=
        0
        or
        contamination_report.contaminated
        is not False
        or
        contamination_report.passed
        is not True
    ):
        raise RuntimeError(
            "Official contamination preflight failed."
        )

    if (
        contamination_report.candidate_dataset_sha256
        !=
        EXPECTED_CONTAMINATION_CANDIDATE_SHA256
    ):
        raise RuntimeError(
            (
                "Contamination candidate identity changed.\n"
                f"Expected: "
                f"{EXPECTED_CONTAMINATION_CANDIDATE_SHA256}\n"
                f"Actual:   "
                f"{contamination_report.candidate_dataset_sha256}"
            )
        )

    if (
        contamination_report.protected_corpus_sha256
        !=
        EXPECTED_EXTENDED_PROTECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            "Contamination report corpus binding changed."
        )

    forbidden_index = (
        adaptation_dataset.build_forbidden_provenance_index(
            repository_root=
                repository_root,

            additional_forbidden_dataset_ids=(
                AIRPORT_HOLDOUT_DATASET_ID,
                GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID,
            ),

            additional_forbidden_artifact_paths=(
                AIRPORT_CASES_RELATIVE_PATH,
            ),
        )
    )

    if (
        forbidden_index.protected_corpus_sha256
        !=
        EXPECTED_BASE_PROTECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            "Forbidden index base-corpus binding changed."
        )

    if (
        forbidden_index.index_sha256
        !=
        EXPECTED_FORBIDDEN_PROVENANCE_INDEX_SHA256
    ):
        raise RuntimeError(
            (
                "Forbidden provenance index identity changed.\n"
                f"Expected: "
                f"{EXPECTED_FORBIDDEN_PROVENANCE_INDEX_SHA256}\n"
                f"Actual:   {forbidden_index.index_sha256}"
            )
        )

    provenance_report = (
        adaptation_dataset.scan_adaptation_provenance(
            dataset_id=
                DATASET_ID,

            dataset_version=
                DATASET_VERSION,

            examples=
                examples,

            forbidden_index=
                forbidden_index,

            approved_development_artifact_paths=
                (),
        )
    )

    adaptation_dataset.assert_adaptation_provenance_clean(
        provenance_report
    )

    if (
        provenance_report.violation_count
        !=
        0
        or
        provenance_report.passed
        is not True
    ):
        raise RuntimeError(
            "Official provenance preflight failed."
        )

    if (
        provenance_report.forbidden_index_sha256
        !=
        EXPECTED_FORBIDDEN_PROVENANCE_INDEX_SHA256
    ):
        raise RuntimeError(
            "Provenance report index binding changed."
        )

    return {
        "examples":
            examples,

        "canonical_sha256":
            canonical_sha256,

        "base_corpus_sha256":
            base_corpus.corpus_sha256,

        "extended_corpus_sha256":
            extended_corpus.corpus_sha256,

        "contamination_candidate_sha256":
            contamination_report.candidate_dataset_sha256,

        "contamination_match_count":
            contamination_report.match_count,

        "provenance_violation_count":
            provenance_report.violation_count,

        "forbidden_index_sha256":
            forbidden_index.index_sha256,

        "protected_source_count_base":
            base_corpus.source_count,

        "protected_source_count_extended":
            extended_corpus.source_count,

        "protected_text_fingerprint_count_extended":
            extended_corpus.text_fingerprint_count,

        "protected_structured_fingerprint_count_extended":
            extended_corpus.structured_fingerprint_count,
    }


def run_official_freeze(
    *,
    repository_root: Path,
) -> Tuple[
    AdaptationDatasetFreezeArtifactV0_2,
    AdaptationDatasetIntegrityReportV0_2,
]:
    repository_root = _root(
        repository_root
    )

    assert_official_outputs_absent(
        repository_root=
            repository_root,
    )

    preflight = build_governance_preflight(
        repository_root=
            repository_root,
    )

    (
        dataset_path,
        contamination_path,
        provenance_path,
        freeze_path,
    ) = _official_output_paths(
        repository_root=
            repository_root,
    )

    airport_path = _airport_path(
        repository_root=
            repository_root,
    )

    artifact = (
        freeze_adaptation_dataset_v0_2(
            repository_root=
                repository_root,

            dataset_id=
                DATASET_ID,

            dataset_version=
                DATASET_VERSION,

            examples=
                preflight[
                    "examples"
                ],

            dataset_output_path=
                dataset_path,

            contamination_report_output_path=
                contamination_path,

            provenance_report_output_path=
                provenance_path,

            freeze_output_path=
                freeze_path,

            additional_protected_paths=(
                airport_path,
            ),

            approved_development_artifact_paths=
                (),

            additional_forbidden_dataset_ids=(
                AIRPORT_HOLDOUT_DATASET_ID,
                GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID,
            ),

            additional_forbidden_artifact_paths=(
                AIRPORT_CASES_RELATIVE_PATH,
            ),

            near_duplicate_threshold=
                NEAR_DUPLICATE_THRESHOLD,

            training_has_started=
                False,
        )
    )

    if (
        artifact.dataset.dataset_sha256
        !=
        EXPECTED_CANONICAL_SHA256
    ):
        raise RuntimeError(
            "Published canonical dataset SHA mismatch."
        )

    if (
        artifact.dataset.example_count
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Published example count mismatch."
        )

    if (
        artifact.contamination_protected_corpus_sha256
        !=
        EXPECTED_EXTENDED_PROTECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            "Published contamination corpus binding mismatch."
        )

    if (
        artifact.provenance_protected_corpus_sha256
        !=
        EXPECTED_BASE_PROTECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            "Published provenance corpus binding mismatch."
        )

    if (
        artifact.forbidden_provenance_index_sha256
        !=
        EXPECTED_FORBIDDEN_PROVENANCE_INDEX_SHA256
    ):
        raise RuntimeError(
            "Published provenance index binding mismatch."
        )

    if (
        artifact.contamination_match_count
        !=
        0
        or
        artifact.provenance_violation_count
        !=
        0
    ):
        raise RuntimeError(
            "Published governance evidence is not clean."
        )

    integrity = (
        verify_adaptation_dataset_freeze_v0_2(
            repository_root=
                repository_root,

            freeze_path=
                freeze_path,
        )
    )

    assert_adaptation_dataset_integrity_v0_2(
        integrity
    )

    if not integrity.passed:
        raise RuntimeError(
            "Official v0.4 freeze integrity failed."
        )

    verify_official_freeze(
        repository_root=
            repository_root,
    )

    return (
        artifact,
        integrity,
    )


def verify_official_freeze(
    *,
    repository_root: Path,
) -> AdaptationDatasetIntegrityReportV0_2:
    repository_root = _root(
        repository_root
    )

    (
        dataset_path,
        contamination_path,
        provenance_path,
        freeze_path,
    ) = _official_output_paths(
        repository_root=
            repository_root,
    )

    for path in (
        dataset_path,
        contamination_path,
        provenance_path,
        freeze_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    artifact = (
        load_adaptation_dataset_freeze_v0_2(
            freeze_path
        )
    )

    if (
        _sha256_file(
            dataset_path
        )
        !=
        EXPECTED_CANONICAL_SHA256
    ):
        raise RuntimeError(
            "Official canonical dataset bytes changed."
        )

    if (
        artifact.dataset.dataset_sha256
        !=
        EXPECTED_CANONICAL_SHA256
    ):
        raise RuntimeError(
            "Freeze artifact dataset binding mismatch."
        )

    if (
        artifact.dataset.example_count
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Freeze artifact example count mismatch."
        )

    if (
        artifact.contamination_protected_corpus_sha256
        !=
        EXPECTED_EXTENDED_PROTECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            "Official contamination corpus binding mismatch."
        )

    if (
        artifact.provenance_protected_corpus_sha256
        !=
        EXPECTED_BASE_PROTECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            "Official provenance corpus binding mismatch."
        )

    if (
        artifact.forbidden_provenance_index_sha256
        !=
        EXPECTED_FORBIDDEN_PROVENANCE_INDEX_SHA256
    ):
        raise RuntimeError(
            "Official forbidden provenance index mismatch."
        )

    contamination_payload = json.loads(
        contamination_path.read_text(
            encoding="utf-8-sig"
        )
    )

    contamination_report = (
        data_governance.AdaptationContaminationReport
        .model_validate(
            contamination_payload
        )
    )

    if (
        contamination_report.match_count
        !=
        0
        or
        contamination_report.passed
        is not True
        or
        contamination_report.candidate_example_count
        !=
        EXPECTED_EXAMPLE_COUNT
        or
        contamination_report.candidate_dataset_sha256
        !=
        EXPECTED_CONTAMINATION_CANDIDATE_SHA256
        or
        contamination_report.protected_corpus_sha256
        !=
        EXPECTED_EXTENDED_PROTECTED_CORPUS_SHA256
    ):
        raise RuntimeError(
            "Official contamination report verification failed."
        )

    provenance_payload = json.loads(
        provenance_path.read_text(
            encoding="utf-8-sig"
        )
    )

    provenance_report = (
        adaptation_dataset.AdaptationProvenanceReport
        .model_validate(
            provenance_payload
        )
    )

    if (
        provenance_report.violation_count
        !=
        0
        or
        provenance_report.passed
        is not True
        or
        provenance_report.example_count
        !=
        EXPECTED_EXAMPLE_COUNT
        or
        provenance_report.forbidden_index_sha256
        !=
        EXPECTED_FORBIDDEN_PROVENANCE_INDEX_SHA256
    ):
        raise RuntimeError(
            "Official provenance report verification failed."
        )

    integrity = (
        verify_adaptation_dataset_freeze_v0_2(
            repository_root=
                repository_root,

            freeze_path=
                freeze_path,
        )
    )

    assert_adaptation_dataset_integrity_v0_2(
        integrity
    )

    return integrity


def print_preflight(
    *,
    repository_root: Path,
) -> None:
    result = build_governance_preflight(
        repository_root=
            repository_root,
    )

    print(
        "=== DATALENS QLORA v0.4 OFFICIAL DATASET FREEZE PREFLIGHT v0.1 ==="
    )

    print()

    print(
        f"Canonical examples: {len(result['examples'])}"
    )

    print(
        (
            "Canonical dataset SHA256: "
            f"{result['canonical_sha256']}"
        )
    )

    print()

    print(
        "CONTAMINATION"
    )

    print(
        (
            "  Candidate SHA256: "
            f"{result['contamination_candidate_sha256']}"
        )
    )

    print(
        (
            "  Base protected sources: "
            f"{result['protected_source_count_base']}"
        )
    )

    print(
        (
            "  Extended protected sources: "
            f"{result['protected_source_count_extended']}"
        )
    )

    print(
        (
            "  Extended text fingerprints: "
            f"{result['protected_text_fingerprint_count_extended']}"
        )
    )

    print(
        (
            "  Extended structured fingerprints: "
            f"{result['protected_structured_fingerprint_count_extended']}"
        )
    )

    print(
        (
            "  Extended corpus SHA256: "
            f"{result['extended_corpus_sha256']}"
        )
    )

    print(
        (
            "  Match count: "
            f"{result['contamination_match_count']}"
        )
    )

    print()

    print(
        "PROVENANCE"
    )

    print(
        (
            "  Base corpus SHA256: "
            f"{result['base_corpus_sha256']}"
        )
    )

    print(
        (
            "  Forbidden index SHA256: "
            f"{result['forbidden_index_sha256']}"
        )
    )

    print(
        (
            "  Violation count: "
            f"{result['provenance_violation_count']}"
        )
    )

    print()

    print(
        "SAFETY"
    )

    print(
        "  Official dataset written: False"
    )

    print(
        "  Official reports written: False"
    )

    print(
        "  Airport evidence read for governance: True"
    )

    print(
        "  Airport evaluated: False"
    )

    print(
        "  Airport results observed: False"
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
        "  Evaluation executed: False"
    )

    print(
        "  Training executed: False"
    )

    print(
        "  Final Acceptance evaluated: False"
    )

    print()

    print(
        "DATALENS QLORA v0.4 OFFICIAL DATASET FREEZE PREFLIGHT v0.1: PASS"
    )


def print_official_freeze(
    *,
    repository_root: Path,
) -> None:
    artifact, integrity = run_official_freeze(
        repository_root=
            repository_root,
    )

    print(
        "=== DATALENS QLORA v0.4 OFFICIAL DATASET FREEZE v0.1 ==="
    )

    print()

    print(
        f"Dataset ID: {artifact.dataset.dataset_id}"
    )

    print(
        f"Dataset version: {artifact.dataset.dataset_version}"
    )

    print(
        f"Examples: {artifact.dataset.example_count}"
    )

    print(
        (
            "Dataset SHA256: "
            f"{artifact.dataset.dataset_sha256}"
        )
    )

    print(
        (
            "Contamination report SHA256: "
            f"{artifact.contamination_report_sha256}"
        )
    )

    print(
        (
            "Provenance report SHA256: "
            f"{artifact.provenance_report_sha256}"
        )
    )

    print(
        (
            "Contamination protected corpus SHA256: "
            f"{artifact.contamination_protected_corpus_sha256}"
        )
    )

    print(
        (
            "Provenance protected corpus SHA256: "
            f"{artifact.provenance_protected_corpus_sha256}"
        )
    )

    print(
        (
            "Forbidden provenance index SHA256: "
            f"{artifact.forbidden_provenance_index_sha256}"
        )
    )

    print(
        (
            "Freeze artifact rule: "
            f"{artifact.freeze_rule_version}"
        )
    )

    print(
        f"Frozen at: {artifact.frozen_at}"
    )

    print()

    print(
        "GOVERNANCE"
    )

    print(
        (
            "  Contamination matches: "
            f"{artifact.contamination_match_count}"
        )
    )

    print(
        (
            "  Provenance violations: "
            f"{artifact.provenance_violation_count}"
        )
    )

    print(
        "  Final Acceptance material: False"
    )

    print(
        "  Frozen before training: True"
    )

    print()

    print(
        "INTEGRITY"
    )

    print(
        (
            "  Dataset hash: "
            f"{integrity.dataset_hash_valid}"
        )
    )

    print(
        (
            "  Contamination report hash: "
            f"{integrity.contamination_report_hash_valid}"
        )
    )

    print(
        (
            "  Provenance report hash: "
            f"{integrity.provenance_report_hash_valid}"
        )
    )

    print(
        (
            "  Contamination gate: "
            f"{integrity.contamination_gate_valid}"
        )
    )

    print(
        (
            "  Provenance gate: "
            f"{integrity.provenance_gate_valid}"
        )
    )

    print(
        (
            "  Contamination corpus binding: "
            f"{integrity.contamination_corpus_binding_valid}"
        )
    )

    print(
        (
            "  Provenance index binding: "
            f"{integrity.provenance_index_binding_valid}"
        )
    )

    print(
        (
            "  Evidence consistent: "
            f"{integrity.evidence_consistent}"
        )
    )

    print(
        (
            "  Passed: "
            f"{integrity.passed}"
        )
    )

    print()

    print(
        "SAFETY"
    )

    print(
        "  Airport evaluated: False"
    )

    print(
        "  Airport results observed: False"
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
        "  Evaluation executed: False"
    )

    print(
        "  Training executed: False"
    )

    print(
        "  Final Acceptance evaluated: False"
    )

    print()

    print(
        "DATALENS QLORA v0.4 OFFICIAL DATASET FREEZE v0.1: PASS"
    )


def print_verification(
    *,
    repository_root: Path,
) -> None:
    integrity = verify_official_freeze(
        repository_root=
            repository_root,
    )

    print(
        "=== DATALENS QLORA v0.4 OFFICIAL DATASET FREEZE VERIFY v0.1 ==="
    )

    print()

    print(
        (
            "Dataset hash valid: "
            f"{integrity.dataset_hash_valid}"
        )
    )

    print(
        (
            "Contamination report hash valid: "
            f"{integrity.contamination_report_hash_valid}"
        )
    )

    print(
        (
            "Provenance report hash valid: "
            f"{integrity.provenance_report_hash_valid}"
        )
    )

    print(
        (
            "Contamination gate valid: "
            f"{integrity.contamination_gate_valid}"
        )
    )

    print(
        (
            "Provenance gate valid: "
            f"{integrity.provenance_gate_valid}"
        )
    )

    print(
        (
            "Contamination corpus binding valid: "
            f"{integrity.contamination_corpus_binding_valid}"
        )
    )

    print(
        (
            "Provenance index binding valid: "
            f"{integrity.provenance_index_binding_valid}"
        )
    )

    print(
        (
            "Evidence consistent: "
            f"{integrity.evidence_consistent}"
        )
    )

    print(
        (
            "Passed: "
            f"{integrity.passed}"
        )
    )

    print()

    print(
        "DATALENS QLORA v0.4 OFFICIAL DATASET FREEZE VERIFY v0.1: PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "freeze",
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

    if arguments.command == "preflight":
        assert_official_outputs_absent(
            repository_root=
                repository_root,
        )

        print_preflight(
            repository_root=
                repository_root,
        )

        return

    if arguments.command == "freeze":
        print_official_freeze(
            repository_root=
                repository_root,
        )

        return

    if arguments.command == "verify":
        print_verification(
            repository_root=
                repository_root,
        )

        return

    raise RuntimeError(
        "Unsupported command."
    )


if __name__ == "__main__":
    main()

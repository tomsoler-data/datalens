from __future__ import annotations

import json

from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from app.adaptation import adaptation_dataset as _v1
from app.adaptation.contracts import AdaptationDatasetEvidence
from app.adaptation.data_governance import (
    AdaptationContaminationReport,
    build_protected_evidence_corpus,
    scan_adaptation_examples,
    assert_adaptation_dataset_clean,
)


ADAPTATION_DATASET_FREEZE_V0_2_RULE_VERSION = (
    "adaptation_dataset_freeze_v0.2"
)

ADAPTATION_DATASET_INTEGRITY_V0_2_RULE_VERSION = (
    "adaptation_dataset_integrity_v0.2"
)


class AdaptationDatasetFreezeArtifactV0_2(
    BaseModel
):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    status: Literal[
        "frozen"
    ] = "frozen"

    dataset: AdaptationDatasetEvidence

    dataset_relative_path: str
    contamination_report_relative_path: str
    provenance_report_relative_path: str

    contamination_report_sha256: str
    provenance_report_sha256: str

    contamination_protected_corpus_sha256: str
    provenance_protected_corpus_sha256: str

    forbidden_provenance_index_sha256: str

    additional_protected_paths: Tuple[
        str,
        ...,
    ] = ()

    additional_forbidden_dataset_ids: Tuple[
        str,
        ...,
    ] = ()

    additional_forbidden_artifact_paths: Tuple[
        str,
        ...,
    ] = ()

    approved_development_artifact_paths: Tuple[
        str,
        ...,
    ] = ()

    near_duplicate_threshold: float = 0.92

    contamination_match_count: Literal[
        0
    ] = 0

    provenance_violation_count: Literal[
        0
    ] = 0

    contains_final_acceptance_material: Literal[
        False
    ] = False

    final_acceptance_tuning_input: Literal[
        False
    ] = False

    frozen_before_training: Literal[
        True
    ] = True

    training_started_at_freeze: Literal[
        False
    ] = False

    frozen_at: str

    builder_rule_version: Literal[
        "adaptation_dataset_builder_v0.1"
    ] = "adaptation_dataset_builder_v0.1"

    freeze_rule_version: Literal[
        "adaptation_dataset_freeze_v0.2"
    ] = ADAPTATION_DATASET_FREEZE_V0_2_RULE_VERSION

    @model_validator(
        mode="after",
    )
    def validate_alignment(
        self,
    ) -> "AdaptationDatasetFreezeArtifactV0_2":
        if (
            self.dataset.contamination_report_sha256
            !=
            self.contamination_report_sha256
        ):
            raise ValueError(
                "Dataset evidence contamination report hash "
                "does not match the freeze artifact."
            )

        if not self.dataset.frozen:
            raise ValueError(
                "Dataset evidence must be frozen."
            )

        if not (
            0.0
            <=
            self.near_duplicate_threshold
            <=
            1.0
        ):
            raise ValueError(
                "near_duplicate_threshold must be "
                "between 0 and 1."
            )

        for value in (
            self.dataset_relative_path,
            self.contamination_report_relative_path,
            self.provenance_report_relative_path,
        ):
            _assert_safe_relative_path(
                value
            )

        for values in (
            self.additional_protected_paths,
            self.additional_forbidden_artifact_paths,
            self.approved_development_artifact_paths,
        ):
            for value in values:
                _assert_safe_relative_path(
                    value
                )

        _assert_sorted_unique(
            self.additional_protected_paths,
            label="additional_protected_paths",
        )

        _assert_sorted_unique(
            self.additional_forbidden_dataset_ids,
            label="additional_forbidden_dataset_ids",
        )

        _assert_sorted_unique(
            self.additional_forbidden_artifact_paths,
            label="additional_forbidden_artifact_paths",
        )

        _assert_sorted_unique(
            self.approved_development_artifact_paths,
            label="approved_development_artifact_paths",
        )

        if not self.frozen_at.strip():
            raise ValueError(
                "frozen_at must not be empty."
            )

        return self


class AdaptationDatasetIntegrityReportV0_2(
    BaseModel
):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    dataset_hash_valid: bool
    contamination_report_hash_valid: bool
    provenance_report_hash_valid: bool

    contamination_gate_valid: bool
    provenance_gate_valid: bool

    contamination_corpus_binding_valid: bool
    provenance_index_binding_valid: bool

    evidence_consistent: bool

    passed: bool

    rule_version: Literal[
        "adaptation_dataset_integrity_v0.2"
    ] = ADAPTATION_DATASET_INTEGRITY_V0_2_RULE_VERSION

    @model_validator(
        mode="after",
    )
    def validate_status(
        self,
    ) -> "AdaptationDatasetIntegrityReportV0_2":
        expected = all(
            (
                self.dataset_hash_valid,
                self.contamination_report_hash_valid,
                self.provenance_report_hash_valid,
                self.contamination_gate_valid,
                self.provenance_gate_valid,
                self.contamination_corpus_binding_valid,
                self.provenance_index_binding_valid,
                self.evidence_consistent,
            )
        )

        if self.passed != expected:
            raise ValueError(
                "v0.2 dataset integrity status "
                "is inconsistent."
            )

        return self


def _assert_safe_relative_path(
    value: str,
) -> None:
    normalized = value.replace(
        "\\",
        "/",
    ).strip()

    if not normalized:
        raise ValueError(
            "Relative path must not be empty."
        )

    path = PurePosixPath(
        normalized
    )

    if path.is_absolute():
        raise ValueError(
            f"Absolute path is forbidden: {value}"
        )

    if ".." in path.parts:
        raise ValueError(
            f"Parent traversal is forbidden: {value}"
        )


def _assert_sorted_unique(
    values: Sequence[str],
    *,
    label: str,
) -> None:
    expected = tuple(
        sorted(
            set(
                values
            )
        )
    )

    if tuple(
        values
    ) != expected:
        raise ValueError(
            f"{label} must be sorted and unique."
        )


def _normalize_string_sequence(
    values: Sequence[str],
    *,
    label: str,
) -> Tuple[str, ...]:
    normalized = []

    for value in values:
        item = value.strip()

        if not item:
            raise ValueError(
                f"{label} contains an empty value."
            )

        normalized.append(
            item
        )

    return tuple(
        sorted(
            set(
                normalized
            )
        )
    )


def _normalize_relative_paths(
    values: Sequence[str],
    *,
    label: str,
) -> Tuple[str, ...]:
    normalized = []

    for value in values:
        item = _v1._normalize_relative_string(
            value
        )

        _assert_safe_relative_path(
            item
        )

        normalized.append(
            item
        )

    result = tuple(
        sorted(
            set(
                normalized
            )
        )
    )

    _assert_sorted_unique(
        result,
        label=label,
    )

    return result


def _resolve_additional_protected_paths(
    *,
    repository_root: Path,
    values: Sequence[Path],
) -> Tuple[
    Tuple[Path, ...],
    Tuple[str, ...],
]:
    resolved_by_relative = {}

    for value in values:
        candidate = value.expanduser()

        if not candidate.is_absolute():
            candidate = (
                repository_root
                /
                candidate
            )

        candidate = candidate.resolve()

        if not candidate.is_file():
            raise FileNotFoundError(
                candidate
            )

        relative = _v1._relative_path(
            repository_root=
                repository_root,
            path=
                candidate,
        )

        relative = (
            relative
            .replace(
                "\\",
                "/",
            )
        )

        _assert_safe_relative_path(
            relative
        )

        resolved_by_relative[
            relative
        ] = candidate

    relative_paths = tuple(
        sorted(
            resolved_by_relative
        )
    )

    resolved_paths = tuple(
        resolved_by_relative[
            relative
        ]

        for relative
        in relative_paths
    )

    return (
        resolved_paths,
        relative_paths,
    )


def _resolve_output_path(
    *,
    repository_root: Path,
    path: Path,
) -> Tuple[
    Path,
    str,
]:
    resolved = (
        path
        .expanduser()
    )

    if not resolved.is_absolute():
        resolved = (
            repository_root
            /
            resolved
        )

    resolved = resolved.resolve()

    relative = _v1._relative_path(
        repository_root=
            repository_root,
        path=
            resolved,
    )

    relative = relative.replace(
        "\\",
        "/",
    )

    _assert_safe_relative_path(
        relative
    )

    return (
        resolved,
        relative,
    )


def _artifact_path(
    *,
    repository_root: Path,
    relative_path: str,
) -> Path:
    _assert_safe_relative_path(
        relative_path
    )

    resolved = (
        repository_root
        /
        Path(
            relative_path
        )
    ).resolve()

    _v1._relative_path(
        repository_root=
            repository_root,
        path=
            resolved,
    )

    return resolved


def freeze_adaptation_dataset_v0_2(
    *,
    repository_root: Path,
    dataset_id: str,
    dataset_version: str,
    examples: Sequence[
        _v1.AdaptationTrainingExample
        |
        Mapping[
            str,
            Any,
        ]
    ],
    dataset_output_path: Path,
    contamination_report_output_path: Path,
    provenance_report_output_path: Path,
    freeze_output_path: Path,
    additional_protected_paths: Sequence[
        Path
    ] = (),
    approved_development_artifact_paths: Sequence[
        str
    ] = (),
    additional_forbidden_dataset_ids: Sequence[
        str
    ] = (),
    additional_forbidden_artifact_paths: Sequence[
        str
    ] = (),
    near_duplicate_threshold: float = 0.92,
    training_has_started: bool = False,
    frozen_at: Optional[str] = None,
) -> AdaptationDatasetFreezeArtifactV0_2:
    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )

    if not repository_root.is_dir():
        raise NotADirectoryError(
            repository_root
        )

    if training_has_started:
        raise RuntimeError(
            "Adaptation dataset must be frozen "
            "before training starts."
        )

    dataset_id = dataset_id.strip()
    dataset_version = dataset_version.strip()

    if not dataset_id:
        raise ValueError(
            "dataset_id must not be empty."
        )

    if not dataset_version:
        raise ValueError(
            "dataset_version must not be empty."
        )

    if not (
        0.0
        <=
        near_duplicate_threshold
        <=
        1.0
    ):
        raise ValueError(
            "near_duplicate_threshold must be "
            "between 0 and 1."
        )

    normalized_examples = (
        _v1._coerce_examples(
            examples
        )
    )

    if not normalized_examples:
        raise ValueError(
            "At least one adaptation example is required."
        )

    (
        dataset_output_path,
        dataset_relative_path,
    ) = _resolve_output_path(
        repository_root=
            repository_root,
        path=
            dataset_output_path,
    )

    (
        contamination_report_output_path,
        contamination_report_relative_path,
    ) = _resolve_output_path(
        repository_root=
            repository_root,
        path=
            contamination_report_output_path,
    )

    (
        provenance_report_output_path,
        provenance_report_relative_path,
    ) = _resolve_output_path(
        repository_root=
            repository_root,
        path=
            provenance_report_output_path,
    )

    (
        freeze_output_path,
        freeze_relative_path,
    ) = _resolve_output_path(
        repository_root=
            repository_root,
        path=
            freeze_output_path,
    )

    output_paths = (
        dataset_output_path,
        contamination_report_output_path,
        provenance_report_output_path,
        freeze_output_path,
    )

    if len(
        set(
            output_paths
        )
    ) != len(
        output_paths
    ):
        raise ValueError(
            "Freeze output paths must be distinct."
        )

    for output_path in output_paths:
        if output_path.exists():
            raise FileExistsError(
                output_path
            )

    (
        resolved_protected_paths,
        protected_relative_paths,
    ) = _resolve_additional_protected_paths(
        repository_root=
            repository_root,
        values=
            additional_protected_paths,
    )

    approved_paths = (
        _normalize_relative_paths(
            approved_development_artifact_paths,
            label=
                "approved_development_artifact_paths",
        )
    )

    forbidden_dataset_ids = (
        _normalize_string_sequence(
            additional_forbidden_dataset_ids,
            label=
                "additional_forbidden_dataset_ids",
        )
    )

    forbidden_artifact_paths = (
        _normalize_relative_paths(
            additional_forbidden_artifact_paths,
            label=
                "additional_forbidden_artifact_paths",
        )
    )

    forbidden_index = (
        _v1.build_forbidden_provenance_index(
            repository_root=
                repository_root,
            additional_forbidden_dataset_ids=
                forbidden_dataset_ids,
            additional_forbidden_artifact_paths=
                forbidden_artifact_paths,
        )
    )

    provenance_report = (
        _v1.scan_adaptation_provenance(
            dataset_id=
                dataset_id,
            dataset_version=
                dataset_version,
            examples=
                normalized_examples,
            forbidden_index=
                forbidden_index,
            approved_development_artifact_paths=
                approved_paths,
        )
    )

    _v1.assert_adaptation_provenance_clean(
        provenance_report
    )

    contamination_report = (
        scan_adaptation_examples(
            repository_root=
                repository_root,
            dataset_id=
                dataset_id,
            dataset_version=
                dataset_version,
            examples=
                _v1._content_scan_payload(
                    normalized_examples
                ),
            near_duplicate_threshold=
                near_duplicate_threshold,
            additional_protected_paths=
                resolved_protected_paths,
        )
    )

    assert_adaptation_dataset_clean(
        contamination_report
    )

    dataset_bytes = (
        _v1._dataset_jsonl_bytes(
            normalized_examples
        )
    )

    contamination_bytes = (
        _v1._pretty_model_json_bytes(
            contamination_report
        )
    )

    provenance_bytes = (
        _v1._pretty_model_json_bytes(
            provenance_report
        )
    )

    dataset_sha256 = (
        _v1._sha256_bytes(
            dataset_bytes
        )
    )

    contamination_sha256 = (
        _v1._sha256_bytes(
            contamination_bytes
        )
    )

    provenance_sha256 = (
        _v1._sha256_bytes(
            provenance_bytes
        )
    )

    evidence = AdaptationDatasetEvidence(
        dataset_id=
            dataset_id,
        dataset_version=
            dataset_version,
        dataset_sha256=
            dataset_sha256,
        example_count=
            len(
                normalized_examples
            ),
        contamination_report_sha256=
            contamination_sha256,
        frozen=
            True,
        contains_regression_expected_answers=
            False,
        contains_pre_adaptation_holdout_material=
            False,
        contains_rag_holdout_material=
            False,
    )

    frozen_at_value = (
        frozen_at.strip()
        if
        frozen_at is not None
        else
        _v1._utc_now_iso8601()
    )

    if not frozen_at_value:
        raise ValueError(
            "frozen_at must not be empty."
        )

    artifact = (
        AdaptationDatasetFreezeArtifactV0_2(
            dataset=
                evidence,
            dataset_relative_path=
                dataset_relative_path,
            contamination_report_relative_path=
                contamination_report_relative_path,
            provenance_report_relative_path=
                provenance_report_relative_path,
            contamination_report_sha256=
                contamination_sha256,
            provenance_report_sha256=
                provenance_sha256,
            contamination_protected_corpus_sha256=
                contamination_report.protected_corpus_sha256,
            provenance_protected_corpus_sha256=
                forbidden_index.protected_corpus_sha256,
            forbidden_provenance_index_sha256=
                forbidden_index.index_sha256,
            additional_protected_paths=
                protected_relative_paths,
            additional_forbidden_dataset_ids=
                forbidden_dataset_ids,
            additional_forbidden_artifact_paths=
                forbidden_artifact_paths,
            approved_development_artifact_paths=
                approved_paths,
            near_duplicate_threshold=
                near_duplicate_threshold,
            contamination_match_count=
                0,
            provenance_violation_count=
                0,
            contains_final_acceptance_material=
                False,
            final_acceptance_tuning_input=
                False,
            frozen_before_training=
                True,
            training_started_at_freeze=
                False,
            frozen_at=
                frozen_at_value,
        )
    )

    freeze_bytes = (
        _v1._pretty_model_json_bytes(
            artifact
        )
    )

    _v1._publish_new_bundle(
        {
            dataset_output_path:
                dataset_bytes,
            contamination_report_output_path:
                contamination_bytes,
            provenance_report_output_path:
                provenance_bytes,
            freeze_output_path:
                freeze_bytes,
        }
    )

    return artifact


def load_adaptation_dataset_freeze_v0_2(
    path: Path,
) -> AdaptationDatasetFreezeArtifactV0_2:
    payload = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    return (
        AdaptationDatasetFreezeArtifactV0_2
        .model_validate(
            payload
        )
    )


def verify_adaptation_dataset_freeze_v0_2(
    *,
    repository_root: Path,
    freeze_path: Path,
) -> AdaptationDatasetIntegrityReportV0_2:
    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )

    artifact = (
        load_adaptation_dataset_freeze_v0_2(
            freeze_path
        )
    )

    dataset_path = _artifact_path(
        repository_root=
            repository_root,
        relative_path=
            artifact.dataset_relative_path,
    )

    contamination_path = _artifact_path(
        repository_root=
            repository_root,
        relative_path=
            artifact.contamination_report_relative_path,
    )

    provenance_path = _artifact_path(
        repository_root=
            repository_root,
        relative_path=
            artifact.provenance_report_relative_path,
    )

    dataset_hash_valid = (
        dataset_path.is_file()
        and
        _v1._sha256_file(
            dataset_path
        )
        ==
        artifact.dataset.dataset_sha256
    )

    contamination_report_hash_valid = (
        contamination_path.is_file()
        and
        _v1._sha256_file(
            contamination_path
        )
        ==
        artifact.contamination_report_sha256
    )

    provenance_report_hash_valid = (
        provenance_path.is_file()
        and
        _v1._sha256_file(
            provenance_path
        )
        ==
        artifact.provenance_report_sha256
    )

    contamination_gate_valid = False

    if contamination_report_hash_valid:
        try:
            contamination_payload = json.loads(
                contamination_path.read_text(
                    encoding="utf-8-sig"
                )
            )

            contamination_report = (
                AdaptationContaminationReport
                .model_validate(
                    contamination_payload
                )
            )

            contamination_gate_valid = (
                contamination_report.passed
                and
                contamination_report.match_count
                ==
                0
                and
                contamination_report.dataset_id
                ==
                artifact.dataset.dataset_id
                and
                contamination_report.dataset_version
                ==
                artifact.dataset.dataset_version
                and
                contamination_report.protected_corpus_sha256
                ==
                artifact.contamination_protected_corpus_sha256
                and
                contamination_report.near_duplicate_threshold
                ==
                artifact.near_duplicate_threshold
            )

        except Exception:
            contamination_gate_valid = False

    provenance_gate_valid = False

    if provenance_report_hash_valid:
        try:
            provenance_payload = json.loads(
                provenance_path.read_text(
                    encoding="utf-8-sig"
                )
            )

            provenance_report = (
                _v1.AdaptationProvenanceReport
                .model_validate(
                    provenance_payload
                )
            )

            provenance_gate_valid = (
                provenance_report.passed
                and
                provenance_report.violation_count
                ==
                0
                and
                provenance_report.dataset_id
                ==
                artifact.dataset.dataset_id
                and
                provenance_report.dataset_version
                ==
                artifact.dataset.dataset_version
                and
                provenance_report.forbidden_index_sha256
                ==
                artifact.forbidden_provenance_index_sha256
            )

        except Exception:
            provenance_gate_valid = False

    contamination_corpus_binding_valid = False

    try:
        protected_paths = tuple(
            _artifact_path(
                repository_root=
                    repository_root,
                relative_path=
                    value,
            )

            for value
            in artifact.additional_protected_paths
        )

        rebuilt_contamination_corpus = (
            build_protected_evidence_corpus(
                repository_root=
                    repository_root,
                additional_protected_paths=
                    protected_paths,
            )
        )

        contamination_corpus_binding_valid = (
            rebuilt_contamination_corpus.corpus_sha256
            ==
            artifact.contamination_protected_corpus_sha256
        )

    except Exception:
        contamination_corpus_binding_valid = False

    provenance_index_binding_valid = False

    try:
        rebuilt_forbidden_index = (
            _v1.build_forbidden_provenance_index(
                repository_root=
                    repository_root,
                additional_forbidden_dataset_ids=
                    artifact.additional_forbidden_dataset_ids,
                additional_forbidden_artifact_paths=
                    artifact.additional_forbidden_artifact_paths,
            )
        )

        provenance_index_binding_valid = (
            rebuilt_forbidden_index.index_sha256
            ==
            artifact.forbidden_provenance_index_sha256
            and
            rebuilt_forbidden_index.protected_corpus_sha256
            ==
            artifact.provenance_protected_corpus_sha256
        )

    except Exception:
        provenance_index_binding_valid = False

    evidence_consistent = (
        artifact.status
        ==
        "frozen"
        and
        artifact.dataset.frozen
        and
        artifact.dataset.contamination_report_sha256
        ==
        artifact.contamination_report_sha256
        and
        artifact.contamination_match_count
        ==
        0
        and
        artifact.provenance_violation_count
        ==
        0
        and
        artifact.contains_final_acceptance_material
        is False
        and
        artifact.final_acceptance_tuning_input
        is False
        and
        artifact.frozen_before_training
        is True
        and
        artifact.training_started_at_freeze
        is False
    )

    passed = all(
        (
            dataset_hash_valid,
            contamination_report_hash_valid,
            provenance_report_hash_valid,
            contamination_gate_valid,
            provenance_gate_valid,
            contamination_corpus_binding_valid,
            provenance_index_binding_valid,
            evidence_consistent,
        )
    )

    return AdaptationDatasetIntegrityReportV0_2(
        dataset_hash_valid=
            dataset_hash_valid,
        contamination_report_hash_valid=
            contamination_report_hash_valid,
        provenance_report_hash_valid=
            provenance_report_hash_valid,
        contamination_gate_valid=
            contamination_gate_valid,
        provenance_gate_valid=
            provenance_gate_valid,
        contamination_corpus_binding_valid=
            contamination_corpus_binding_valid,
        provenance_index_binding_valid=
            provenance_index_binding_valid,
        evidence_consistent=
            evidence_consistent,
        passed=
            passed,
    )


def assert_adaptation_dataset_integrity_v0_2(
    report: AdaptationDatasetIntegrityReportV0_2,
) -> None:
    if not report.passed:
        raise RuntimeError(
            "Adaptation dataset v0.2 integrity "
            "verification failed."
        )

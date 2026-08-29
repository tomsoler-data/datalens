from __future__ import annotations


import ast
import hashlib
import json
import os
import uuid


from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
    PurePosixPath,
)

from typing import (
    Any,
    Literal,
    Mapping,
    Sequence,
    Tuple,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


from app.adaptation.contracts import (
    ADAPTATION_DATA_GOVERNANCE_RULE_VERSION,
    AdaptationDatasetEvidence,
)

from app.adaptation.data_governance import (
    AdaptationContaminationReport,
    build_protected_evidence_corpus,
    scan_adaptation_examples,
)


# ============================================================
# VERSIONS
# ============================================================


ADAPTATION_DATASET_BUILDER_RULE_VERSION = (
    "adaptation_dataset_builder_v0.1"
)

ADAPTATION_DATASET_FREEZE_RULE_VERSION = (
    "adaptation_dataset_freeze_v0.1"
)

ADAPTATION_PROVENANCE_GUARD_RULE_VERSION = (
    "adaptation_provenance_guard_v0.1"
)

ADAPTATION_DATASET_INTEGRITY_RULE_VERSION = (
    "adaptation_dataset_integrity_v0.1"
)


# ============================================================
# TYPES
# ============================================================


AdaptationMessageRole = Literal[
    "user",
    "assistant",
]


AdaptationProvenanceOrigin = Literal[
    "independently_authored",
    "approved_development_source",
]


AdaptationProvenanceViolationKind = Literal[
    "forbidden_dataset_id",
    "forbidden_artifact_path",
    "unapproved_development_source",
]


# ============================================================
# EXAMPLES
# ============================================================


class AdaptationMessage(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    role: AdaptationMessageRole

    content: str = Field(
        min_length=1,
    )


class AdaptationExampleProvenance(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    origin: AdaptationProvenanceOrigin

    source_ids: Tuple[
        str,
        ...,
    ] = ()

    source_dataset_ids: Tuple[
        str,
        ...,
    ] = ()

    source_artifact_paths: Tuple[
        str,
        ...,
    ] = ()


    @model_validator(
        mode="after",
    )
    def validate_origin(
        self,
    ) -> "AdaptationExampleProvenance":
        if (
            self.origin
            ==
            "independently_authored"
        ):
            if (
                self.source_dataset_ids
                or
                self.source_artifact_paths
            ):
                raise ValueError(
                    "Independently authored examples may not "
                    "declare source datasets or source artifacts."
                )


        if (
            self.origin
            ==
            "approved_development_source"
        ):
            if not self.source_artifact_paths:
                raise ValueError(
                    "Approved development examples must "
                    "declare at least one source artifact."
                )


        return self


class AdaptationTrainingExample(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    example_id: str = Field(
        min_length=1,
    )

    messages: Tuple[
        AdaptationMessage,
        ...,
    ] = Field(
        min_length=2,
        max_length=2,
    )

    provenance: AdaptationExampleProvenance

    tags: Tuple[
        str,
        ...,
    ] = ()


    @model_validator(
        mode="after",
    )
    def validate_chat_shape(
        self,
    ) -> "AdaptationTrainingExample":
        roles = tuple(
            message.role

            for message
            in self.messages
        )


        if (
            roles
            !=
            (
                "user",
                "assistant",
            )
        ):
            raise ValueError(
                "Adaptation examples must contain exactly "
                "one user message followed by exactly one "
                "assistant message."
            )


        if (
            len(
                self.tags
            )
            !=
            len(
                set(
                    self.tags
                )
            )
        ):
            raise ValueError(
                "Adaptation example tags must be unique."
            )


        return self


# ============================================================
# FORBIDDEN PROVENANCE
# ============================================================


class ForbiddenProvenanceIndex(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    protected_corpus_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    protected_source_count: int = Field(
        ge=1,
    )

    forbidden_dataset_ids: Tuple[
        str,
        ...,
    ] = ()

    forbidden_artifact_paths: Tuple[
        str,
        ...,
    ] = ()

    forbidden_dataset_id_count: int = Field(
        ge=0,
    )

    forbidden_artifact_path_count: int = Field(
        ge=0,
    )

    index_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    rule_version: Literal[
        "adaptation_provenance_guard_v0.1"
    ] = ADAPTATION_PROVENANCE_GUARD_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_counts(
        self,
    ) -> "ForbiddenProvenanceIndex":
        if (
            self.forbidden_dataset_id_count
            !=
            len(
                self.forbidden_dataset_ids
            )
        ):
            raise ValueError(
                "Forbidden dataset ID count mismatch."
            )


        if (
            self.forbidden_artifact_path_count
            !=
            len(
                self.forbidden_artifact_paths
            )
        ):
            raise ValueError(
                "Forbidden artifact path count mismatch."
            )


        return self


class AdaptationProvenanceViolation(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    example_id: str = Field(
        min_length=1,
    )

    kind: AdaptationProvenanceViolationKind

    value: str = Field(
        min_length=1,
    )


class AdaptationProvenanceReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    dataset_id: str = Field(
        min_length=1,
    )

    dataset_version: str = Field(
        min_length=1,
    )

    example_count: int = Field(
        gt=0,
    )

    forbidden_index_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    approved_development_artifact_paths: Tuple[
        str,
        ...,
    ] = ()

    violations: Tuple[
        AdaptationProvenanceViolation,
        ...,
    ] = ()

    violation_count: int = Field(
        ge=0,
    )

    passed: bool

    rule_version: Literal[
        "adaptation_provenance_guard_v0.1"
    ] = ADAPTATION_PROVENANCE_GUARD_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_status(
        self,
    ) -> "AdaptationProvenanceReport":
        if (
            self.violation_count
            !=
            len(
                self.violations
            )
        ):
            raise ValueError(
                "Adaptation provenance violation "
                "count mismatch."
            )


        if (
            self.passed
            !=
            (
                self.violation_count
                ==
                0
            )
        ):
            raise ValueError(
                "Adaptation provenance pass status "
                "is inconsistent."
            )


        return self


# ============================================================
# FREEZE
# ============================================================


class AdaptationDatasetFreezeArtifact(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    status: Literal[
        "frozen"
    ] = "frozen"

    dataset: AdaptationDatasetEvidence

    dataset_relative_path: str = Field(
        min_length=1,
    )

    contamination_report_relative_path: str = Field(
        min_length=1,
    )

    provenance_report_relative_path: str = Field(
        min_length=1,
    )

    contamination_report_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    provenance_report_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    protected_corpus_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    forbidden_provenance_index_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

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

    frozen_at: str = Field(
        min_length=1,
    )

    builder_rule_version: Literal[
        "adaptation_dataset_builder_v0.1"
    ] = ADAPTATION_DATASET_BUILDER_RULE_VERSION

    freeze_rule_version: Literal[
        "adaptation_dataset_freeze_v0.1"
    ] = ADAPTATION_DATASET_FREEZE_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_evidence_alignment(
        self,
    ) -> "AdaptationDatasetFreezeArtifact":
        if (
            self.dataset.contamination_report_sha256
            !=
            self.contamination_report_sha256
        ):
            raise ValueError(
                "Adaptation dataset evidence contamination "
                "report hash does not match the freeze artifact."
            )


        if not self.dataset.frozen:
            raise ValueError(
                "Adaptation dataset evidence must be frozen."
            )


        return self


class AdaptationDatasetIntegrityReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    dataset_hash_valid: bool

    contamination_report_hash_valid: bool

    provenance_report_hash_valid: bool

    contamination_gate_valid: bool

    provenance_gate_valid: bool

    evidence_consistent: bool

    passed: bool

    rule_version: Literal[
        "adaptation_dataset_integrity_v0.1"
    ] = ADAPTATION_DATASET_INTEGRITY_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_status(
        self,
    ) -> "AdaptationDatasetIntegrityReport":
        expected = all(
            (
                self.dataset_hash_valid,
                self.contamination_report_hash_valid,
                self.provenance_report_hash_valid,
                self.contamination_gate_valid,
                self.provenance_gate_valid,
                self.evidence_consistent,
            )
        )


        if (
            self.passed
            !=
            expected
        ):
            raise ValueError(
                "Adaptation dataset integrity status "
                "is inconsistent."
            )


        return self


# ============================================================
# HASHING
# ============================================================


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
        "rb",
    ) as handle:
        while True:
            chunk = handle.read(
                8
                *
                1024
                *
                1024
            )


            if not chunk:
                break


            digest.update(
                chunk
            )


    return digest.hexdigest()


def _canonical_json_bytes(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )


def _canonical_sha256(
    value: object,
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            value
        )
    )


def _utc_now_iso8601() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


# ============================================================
# PATH HELPERS
# ============================================================


def _relative_path(
    *,
    repository_root: Path,
    path: Path,
) -> str:
    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )

    path = (
        path
        .expanduser()
        .resolve()
    )


    try:
        relative = path.relative_to(
            repository_root
        )

    except ValueError as error:
        raise ValueError(
            "Adaptation dataset artifacts must "
            "live inside the repository."
        ) from error


    return relative.as_posix()


def _normalize_relative_string(
    value: str,
) -> str:
    normalized = (
        value
        .replace(
            "\\",
            "/",
        )
        .strip()
    )


    if not normalized:
        raise ValueError(
            "Repository-relative artifact path "
            "may not be empty."
        )


    path = PurePosixPath(
        normalized
    )


    if (
        path.is_absolute()
        or
        ".."
        in
        path.parts
    ):
        raise ValueError(
            "Repository-relative artifact path may "
            "not be absolute or escape the repository."
        )


    while normalized.startswith(
        "./"
    ):
        normalized = normalized[
            2:
        ]


    return str(
        PurePosixPath(
            normalized
        )
    )


# ============================================================
# PROTECTED SOURCE EXTRACTION
# ============================================================


def _collect_dataset_ids_from_json(
    value: Any,
) -> set[
    str
]:
    result: set[
        str
    ] = set()


    if isinstance(
        value,
        dict,
    ):
        for (
            key,
            child,
        ) in value.items():
            lowered = str(
                key
            ).lower()


            if (
                lowered
                in {
                    "dataset_id",
                    "left_dataset_id",
                    "right_dataset_id",
                }
            ):
                if (
                    isinstance(
                        child,
                        str,
                    )
                    and
                    child.strip()
                ):
                    result.add(
                        child.strip()
                    )


            result.update(
                _collect_dataset_ids_from_json(
                    child
                )
            )


    elif isinstance(
        value,
        list,
    ):
        for child in value:
            result.update(
                _collect_dataset_ids_from_json(
                    child
                )
            )


    return result


def _collect_artifact_paths_from_json(
    value: Any,
) -> set[
    str
]:
    result: set[
        str
    ] = set()


    if isinstance(
        value,
        dict,
    ):
        for (
            key,
            child,
        ) in value.items():
            lowered = str(
                key
            ).lower()


            if (
                lowered
                ==
                "relative_path"
            ):
                if (
                    isinstance(
                        child,
                        str,
                    )
                    and
                    child.strip()
                ):
                    try:
                        result.add(
                            _normalize_relative_string(
                                child
                            )
                        )

                    except ValueError:
                        pass


            result.update(
                _collect_artifact_paths_from_json(
                    child
                )
            )


    elif isinstance(
        value,
        list,
    ):
        for child in value:
            result.update(
                _collect_artifact_paths_from_json(
                    child
                )
            )


    return result


def _assignment_names(
    node: ast.AST,
) -> list[
    str
]:
    names: list[
        str
    ] = []


    if isinstance(
        node,
        ast.Assign,
    ):
        for target in node.targets:
            if isinstance(
                target,
                ast.Name,
            ):
                names.append(
                    target.id
                )


    elif isinstance(
        node,
        ast.AnnAssign,
    ):
        if isinstance(
            node.target,
            ast.Name,
        ):
            names.append(
                node.target.id
            )


    return names


def _collect_dataset_ids_from_python(
    source: str,
) -> set[
    str
]:
    result: set[
        str
    ] = set()

    tree = ast.parse(
        source
    )


    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            names = (
                _assignment_names(
                    node
                )
            )

            value_node = getattr(
                node,
                "value",
                None,
            )


            if (
                value_node
                is not None
                and
                any(
                    "DATASET_ID"
                    in
                    name.upper()

                    for name
                    in names
                )
            ):
                try:
                    literal = ast.literal_eval(
                        value_node
                    )

                except Exception:
                    literal = None


                if (
                    isinstance(
                        literal,
                        str,
                    )
                    and
                    literal.strip()
                ):
                    result.add(
                        literal.strip()
                    )


        if isinstance(
            node,
            ast.Dict,
        ):
            for (
                key_node,
                value_node,
            ) in zip(
                node.keys,
                node.values,
            ):
                try:
                    key = (
                        ast.literal_eval(
                            key_node
                        )
                        if key_node
                        is not None
                        else
                        None
                    )

                except Exception:
                    key = None


                if (
                    key
                    not in {
                        "dataset_id",
                        "left_dataset_id",
                        "right_dataset_id",
                    }
                ):
                    continue


                try:
                    value = ast.literal_eval(
                        value_node
                    )

                except Exception:
                    value = None


                if (
                    isinstance(
                        value,
                        str,
                    )
                    and
                    value.strip()
                ):
                    result.add(
                        value.strip()
                    )


    return result


# ============================================================
# FORBIDDEN PROVENANCE INDEX
# ============================================================


def build_forbidden_provenance_index(
    *,
    repository_root: Path,
    additional_forbidden_dataset_ids: Sequence[
        str
    ] = (),
    additional_forbidden_artifact_paths: Sequence[
        str
    ] = (),
) -> ForbiddenProvenanceIndex:
    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )


    if not repository_root.is_dir():
        raise FileNotFoundError(
            "Repository root does not exist."
        )


    corpus = (
        build_protected_evidence_corpus(
            repository_root=
                repository_root,
        )
    )


    dataset_ids = {
        value.strip()

        for value
        in additional_forbidden_dataset_ids

        if value.strip()
    }


    artifact_paths = {
        _normalize_relative_string(
            value
        )

        for value
        in additional_forbidden_artifact_paths
    }


    for source in corpus.sources:
        artifact_paths.add(
            _normalize_relative_string(
                source.relative_path
            )
        )


        path = (
            repository_root
            /
            Path(
                source.relative_path
            )
        )


        if (
            source.source_type
            ==
            "json_artifact"
        ):
            try:
                payload = json.loads(
                    path.read_text(
                        encoding="utf-8-sig"
                    )
                )

            except Exception as error:
                raise RuntimeError(
                    "Protected JSON evidence could not "
                    "be read for provenance indexing: "
                    f"{source.relative_path}"
                ) from error


            dataset_ids.update(
                _collect_dataset_ids_from_json(
                    payload
                )
            )


            artifact_paths.update(
                _collect_artifact_paths_from_json(
                    payload
                )
            )


        elif (
            source.source_type
            ==
            "python_definition"
        ):
            try:
                source_text = path.read_text(
                    encoding="utf-8-sig"
                )

                dataset_ids.update(
                    _collect_dataset_ids_from_python(
                        source_text
                    )
                )

            except Exception as error:
                raise RuntimeError(
                    "Protected Python evidence could not "
                    "be read for provenance indexing: "
                    f"{source.relative_path}"
                ) from error


        else:
            raise RuntimeError(
                "Unexpected protected source type: "
                f"{source.source_type}"
            )


    ordered_dataset_ids = tuple(
        sorted(
            dataset_ids
        )
    )


    ordered_artifact_paths = tuple(
        sorted(
            artifact_paths
        )
    )


    identity = {
        "rule_version":
            ADAPTATION_PROVENANCE_GUARD_RULE_VERSION,
        "protected_corpus_sha256":
            corpus.corpus_sha256,
        "protected_source_count":
            corpus.source_count,
        "forbidden_dataset_ids":
            ordered_dataset_ids,
        "forbidden_artifact_paths":
            ordered_artifact_paths,
    }


    return ForbiddenProvenanceIndex(
        protected_corpus_sha256=
            corpus.corpus_sha256,
        protected_source_count=
            corpus.source_count,
        forbidden_dataset_ids=
            ordered_dataset_ids,
        forbidden_artifact_paths=
            ordered_artifact_paths,
        forbidden_dataset_id_count=
            len(
                ordered_dataset_ids
            ),
        forbidden_artifact_path_count=
            len(
                ordered_artifact_paths
            ),
        index_sha256=
            _canonical_sha256(
                identity
            ),
    )


# ============================================================
# EXAMPLE NORMALIZATION
# ============================================================


def _coerce_examples(
    examples: Sequence[
        AdaptationTrainingExample
        |
        Mapping[
            str,
            Any,
        ]
    ],
) -> Tuple[
    AdaptationTrainingExample,
    ...,
]:
    result = tuple(
        (
            example

            if isinstance(
                example,
                AdaptationTrainingExample,
            )

            else AdaptationTrainingExample.model_validate(
                example
            )
        )

        for example
        in examples
    )


    if not result:
        raise ValueError(
            "Adaptation dataset must contain "
            "at least one example."
        )


    example_ids = [
        example.example_id

        for example
        in result
    ]


    if (
        len(
            example_ids
        )
        !=
        len(
            set(
                example_ids
            )
        )
    ):
        raise ValueError(
            "Adaptation example IDs must be unique."
        )


    return result


# ============================================================
# PROVENANCE SCAN
# ============================================================


def scan_adaptation_provenance(
    *,
    dataset_id: str,
    dataset_version: str,
    examples: Sequence[
        AdaptationTrainingExample
        |
        Mapping[
            str,
            Any,
        ]
    ],
    forbidden_index: ForbiddenProvenanceIndex,
    approved_development_artifact_paths: Sequence[
        str
    ] = (),
) -> AdaptationProvenanceReport:
    normalized_examples = (
        _coerce_examples(
            examples
        )
    )


    approved = {
        _normalize_relative_string(
            path
        )

        for path
        in approved_development_artifact_paths
    }


    forbidden_dataset_ids = set(
        forbidden_index.forbidden_dataset_ids
    )

    forbidden_artifact_paths = set(
        forbidden_index.forbidden_artifact_paths
    )


    violations: list[
        AdaptationProvenanceViolation
    ] = []


    seen: set[
        tuple[
            str,
            str,
            str,
        ]
    ] = set()


    def append_violation(
        *,
        example_id: str,
        kind: AdaptationProvenanceViolationKind,
        value: str,
    ) -> None:
        key = (
            example_id,
            kind,
            value,
        )


        if key in seen:
            return


        seen.add(
            key
        )


        violations.append(
            AdaptationProvenanceViolation(
                example_id=
                    example_id,
                kind=
                    kind,
                value=
                    value,
            )
        )


    for example in normalized_examples:
        provenance = (
            example.provenance
        )


        declared_dataset_like_ids = {
            value.strip()

            for value
            in (
                tuple(
                    provenance.source_ids
                )
                +
                tuple(
                    provenance.source_dataset_ids
                )
            )

            if value.strip()
        }


        for value in (
            declared_dataset_like_ids
        ):
            if (
                value
                in
                forbidden_dataset_ids
            ):
                append_violation(
                    example_id=
                        example.example_id,
                    kind=
                        "forbidden_dataset_id",
                    value=
                        value,
                )


        normalized_paths = tuple(
            _normalize_relative_string(
                path
            )

            for path
            in provenance.source_artifact_paths
        )


        for path in normalized_paths:
            if (
                path
                in
                forbidden_artifact_paths
            ):
                append_violation(
                    example_id=
                        example.example_id,
                    kind=
                        "forbidden_artifact_path",
                    value=
                        path,
                )


        if (
            provenance.origin
            ==
            "approved_development_source"
        ):
            for path in normalized_paths:
                if (
                    path
                    not in
                    approved
                ):
                    append_violation(
                        example_id=
                            example.example_id,
                        kind=
                            "unapproved_development_source",
                        value=
                            path,
                    )


    violations = sorted(
        violations,
        key=lambda item: (
            item.example_id,
            item.kind,
            item.value,
        ),
    )


    return AdaptationProvenanceReport(
        dataset_id=
            dataset_id.strip(),
        dataset_version=
            dataset_version.strip(),
        example_count=
            len(
                normalized_examples
            ),
        forbidden_index_sha256=
            forbidden_index.index_sha256,
        approved_development_artifact_paths=
            tuple(
                sorted(
                    approved
                )
            ),
        violations=
            tuple(
                violations
            ),
        violation_count=
            len(
                violations
            ),
        passed=(
            len(
                violations
            )
            ==
            0
        ),
    )


def assert_adaptation_provenance_clean(
    report: AdaptationProvenanceReport,
) -> None:
    if not report.passed:
        raise RuntimeError(
            "Adaptation dataset provenance gate failed. "
            f"Detected {report.violation_count} violation(s)."
        )


# ============================================================
# CONTENT SCAN PAYLOAD
# ============================================================


def _content_scan_payload(
    examples: Sequence[
        AdaptationTrainingExample
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    return [
        {
            "example_id":
                example.example_id,
            "messages": [
                message.model_dump(
                    mode="json"
                )

                for message
                in example.messages
            ],
        }

        for example
        in examples
    ]


# ============================================================
# SERIALIZATION
# ============================================================


def _dataset_jsonl_bytes(
    examples: Sequence[
        AdaptationTrainingExample
    ],
) -> bytes:
    lines = [
        json.dumps(
            example.model_dump(
                mode="json"
            ),
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=True,
        )

        for example
        in examples
    ]


    return (
        "\n".join(
            lines
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def _pretty_model_json_bytes(
    model: BaseModel,
) -> bytes:
    return (
        json.dumps(
            model.model_dump(
                mode="json"
            ),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


# ============================================================
# ATOMIC PUBLISH
# ============================================================


def _publish_new_bundle(
    files: Mapping[
        Path,
        bytes,
    ],
) -> None:
    resolved = {
        path.expanduser().resolve():
            payload

        for (
            path,
            payload,
        )
        in files.items()
    }


    for path in resolved:
        if path.exists():
            raise FileExistsError(
                "Adaptation freeze output already "
                f"exists: {path}"
            )


    temp_paths: dict[
        Path,
        Path,
    ] = {}

    published: list[
        Path
    ] = []


    try:
        for (
            path,
            payload,
        ) in resolved.items():
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            temp_path = path.with_name(
                (
                    f".{path.name}."
                    f"{uuid.uuid4().hex}.tmp"
                )
            )


            with temp_path.open(
                "wb"
            ) as handle:
                handle.write(
                    payload
                )

                handle.flush()

                os.fsync(
                    handle.fileno()
                )


            temp_paths[
                path
            ] = temp_path


        for (
            path,
            temp_path,
        ) in temp_paths.items():
            os.replace(
                temp_path,
                path,
            )

            published.append(
                path
            )


    except Exception:
        for path in published:
            try:
                path.unlink()

            except FileNotFoundError:
                pass


        for temp_path in (
            temp_paths.values()
        ):
            try:
                temp_path.unlink()

            except FileNotFoundError:
                pass


        raise


# ============================================================
# FREEZE BUILDER
# ============================================================


def freeze_adaptation_dataset(
    *,
    repository_root: Path,
    dataset_id: str,
    dataset_version: str,
    examples: Sequence[
        AdaptationTrainingExample
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
    frozen_at: str | None = None,
) -> AdaptationDatasetFreezeArtifact:
    if training_has_started:
        raise RuntimeError(
            "Adaptation dataset must be frozen "
            "before training starts."
        )


    if not dataset_id.strip():
        raise ValueError(
            "dataset_id is required."
        )


    if not dataset_version.strip():
        raise ValueError(
            "dataset_version is required."
        )


    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )


    if not repository_root.is_dir():
        raise FileNotFoundError(
            "Repository root does not exist."
        )


    normalized_examples = (
        _coerce_examples(
            examples
        )
    )


    output_paths = (
        dataset_output_path,
        contamination_report_output_path,
        provenance_report_output_path,
        freeze_output_path,
    )


    relative_outputs = [
        _relative_path(
            repository_root=
                repository_root,
            path=
                path,
        )

        for path
        in output_paths
    ]


    if (
        len(
            relative_outputs
        )
        !=
        len(
            set(
                relative_outputs
            )
        )
    ):
        raise ValueError(
            "Adaptation dataset output paths "
            "must be unique."
        )


    forbidden_index = (
        build_forbidden_provenance_index(
            repository_root=
                repository_root,
            additional_forbidden_dataset_ids=
                additional_forbidden_dataset_ids,
            additional_forbidden_artifact_paths=
                additional_forbidden_artifact_paths,
        )
    )


    provenance_report = (
        scan_adaptation_provenance(
            dataset_id=
                dataset_id,
            dataset_version=
                dataset_version,
            examples=
                normalized_examples,
            forbidden_index=
                forbidden_index,
            approved_development_artifact_paths=
                approved_development_artifact_paths,
        )
    )


    assert_adaptation_provenance_clean(
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
                _content_scan_payload(
                    normalized_examples
                ),
            near_duplicate_threshold=
                near_duplicate_threshold,
        )
    )


    if not contamination_report.passed:
        raise RuntimeError(
            "Adaptation dataset contamination gate failed. "
            f"Detected {contamination_report.match_count} "
            "match(es)."
        )


    dataset_bytes = (
        _dataset_jsonl_bytes(
            normalized_examples
        )
    )

    contamination_bytes = (
        _pretty_model_json_bytes(
            contamination_report
        )
    )

    provenance_bytes = (
        _pretty_model_json_bytes(
            provenance_report
        )
    )


    dataset_sha256 = (
        _sha256_bytes(
            dataset_bytes
        )
    )

    contamination_sha256 = (
        _sha256_bytes(
            contamination_bytes
        )
    )

    provenance_sha256 = (
        _sha256_bytes(
            provenance_bytes
        )
    )


    evidence = AdaptationDatasetEvidence(
        dataset_id=
            dataset_id.strip(),
        dataset_version=
            dataset_version.strip(),
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
        governance_rule_version=
            ADAPTATION_DATA_GOVERNANCE_RULE_VERSION,
    )


    artifact = AdaptationDatasetFreezeArtifact(
        dataset=
            evidence,
        dataset_relative_path=
            relative_outputs[
                0
            ],
        contamination_report_relative_path=
            relative_outputs[
                1
            ],
        provenance_report_relative_path=
            relative_outputs[
                2
            ],
        contamination_report_sha256=
            contamination_sha256,
        provenance_report_sha256=
            provenance_sha256,
        protected_corpus_sha256=
            forbidden_index.protected_corpus_sha256,
        forbidden_provenance_index_sha256=
            forbidden_index.index_sha256,
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
        frozen_at=(
            frozen_at
            if frozen_at is not None
            else
            _utc_now_iso8601()
        ),
    )


    freeze_bytes = (
        _pretty_model_json_bytes(
            artifact
        )
    )


    _publish_new_bundle(
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


# ============================================================
# LOAD / VERIFY
# ============================================================


def load_adaptation_dataset_freeze(
    path: Path,
) -> AdaptationDatasetFreezeArtifact:
    path = (
        path
        .expanduser()
        .resolve()
    )


    if not path.is_file():
        raise FileNotFoundError(
            "Adaptation dataset freeze artifact "
            f"is missing: {path}"
        )


    payload = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


    return AdaptationDatasetFreezeArtifact.model_validate(
        payload
    )


def verify_adaptation_dataset_freeze(
    *,
    repository_root: Path,
    freeze_path: Path,
) -> AdaptationDatasetIntegrityReport:
    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )


    artifact = (
        load_adaptation_dataset_freeze(
            freeze_path
        )
    )


    dataset_path = (
        repository_root
        /
        Path(
            artifact.dataset_relative_path
        )
    )

    contamination_path = (
        repository_root
        /
        Path(
            artifact.contamination_report_relative_path
        )
    )

    provenance_path = (
        repository_root
        /
        Path(
            artifact.provenance_report_relative_path
        )
    )


    dataset_hash_valid = (
        dataset_path.is_file()
        and
        _sha256_file(
            dataset_path
        )
        ==
        artifact.dataset.dataset_sha256
    )


    contamination_report_hash_valid = (
        contamination_path.is_file()
        and
        _sha256_file(
            contamination_path
        )
        ==
        artifact.contamination_report_sha256
    )


    provenance_report_hash_valid = (
        provenance_path.is_file()
        and
        _sha256_file(
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
                AdaptationContaminationReport.model_validate(
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
                AdaptationProvenanceReport.model_validate(
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
            evidence_consistent,
        )
    )


    return AdaptationDatasetIntegrityReport(
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
        evidence_consistent=
            evidence_consistent,
        passed=
            passed,
    )


def assert_adaptation_dataset_integrity(
    report: AdaptationDatasetIntegrityReport,
) -> None:
    if not report.passed:
        raise RuntimeError(
            "Adaptation dataset integrity gate failed."
        )

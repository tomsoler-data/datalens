from __future__ import annotations


import ast
import hashlib
import json
import re
import unicodedata


from dataclasses import (
    dataclass,
)

from difflib import (
    SequenceMatcher,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
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
)


# ============================================================
# VERSION
# ============================================================


PROTECTED_EVIDENCE_CORPUS_RULE_VERSION = (
    "protected_evidence_corpus_v0.1"
)


CONTAMINATION_REPORT_RULE_VERSION = (
    "adaptation_contamination_report_v0.1"
)


# ============================================================
# POLICY
# ============================================================


MIN_PROTECTED_TEXT_CHARS = 24

MIN_PROTECTED_TEXT_TOKENS = 4

DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.92


ProtectedSourceClassification = Literal[
    "protected_regression",
    "protected_pre_adaptation_holdout",
    "protected_holdout",
    "protected_code_evaluation",
    "protected_final_acceptance_holdout",
]


ProtectedSourceType = Literal[
    "json_artifact",
    "python_definition",
]


ContaminationMatchKind = Literal[
    "exact_raw_text",
    "exact_normalized_text",
    "exact_structured_record",
    "near_duplicate_text",
]


CONTENT_FIELD_FRAGMENTS = frozenset(
    {
        "case",
        "cases",
        "prompt",
        "question",
        "query",
        "input",
        "output",
        "answer",
        "expected",
        "target",
        "label",
        "ground_truth",
        "relation",
        "relations",
        "document",
        "documents",
        "text",
        "request",
        "response",
        "passage",
        "finding",
        "reason",
        "explanation",
        "statement",
        "evidence_quote",
        "column",
        "concept",
        "accepted_values",
        "claims",
    }
)


EXCLUDED_METADATA_KEYS = frozenset(
    {
        "id",
        "case_id",
        "dataset_id",
        "document_id",
        "chunk_id",
        "filename",
        "source_locator",
        "artifact",
        "path",
        "status",
        "system_label",
        "model",
        "model_name",
        "domain",
        "version",
        "benchmark_id",
        "benchmark_version",
        "holdout_version",
        "rule_version",
        "freeze_version",
    }
)


STRUCTURED_CONTAINER_NAMES = frozenset(
    {
        "cases",
        "relations",
        "column_cases",
        "pair_cases",
        "s4_quantity_family_cases",
        "claims",
        "records",
        "scenarios",
    }
)


PYTHON_DATA_ASSIGNMENT_FRAGMENTS = (
    "CASE",
    "CASES",
    "EXPECTED",
    "RELATION",
    "RELATIONS",
    "SCENARIO",
    "SCENARIOS",
    "BENCHMARK",
)


# ============================================================
# PUBLIC MODELS
# ============================================================


class ProtectedEvidenceSource(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    relative_path: str = Field(
        min_length=1,
    )


    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    classification: ProtectedSourceClassification


    source_type: ProtectedSourceType


class ProtectedEvidenceCorpus(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    sources: Tuple[
        ProtectedEvidenceSource,
        ...,
    ]


    source_count: int = Field(
        ge=1,
    )


    text_fingerprint_count: int = Field(
        ge=1,
    )


    structured_fingerprint_count: int = Field(
        ge=0,
    )


    corpus_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    rule_version: Literal[
        "protected_evidence_corpus_v0.1"
    ] = PROTECTED_EVIDENCE_CORPUS_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_counts(
        self,
    ) -> "ProtectedEvidenceCorpus":
        if (
            self.source_count
            !=
            len(
                self.sources
            )
        ):
            raise ValueError(
                "Protected evidence source count mismatch."
            )


        return self


class ContaminationMatch(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    candidate_example_id: str = Field(
        min_length=1,
    )


    candidate_locator: str = Field(
        min_length=1,
    )


    match_kind: ContaminationMatchKind


    candidate_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    protected_source_path: str = Field(
        min_length=1,
    )


    protected_locator: str = Field(
        min_length=1,
    )


    protected_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    similarity: Optional[
        float
    ] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class AdaptationContaminationReport(
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


    candidate_dataset_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    candidate_example_count: int = Field(
        ge=1,
    )


    protected_corpus_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    protected_source_count: int = Field(
        ge=1,
    )


    protected_text_fingerprint_count: int = Field(
        ge=1,
    )


    protected_structured_fingerprint_count: int = Field(
        ge=0,
    )


    near_duplicate_threshold: float = Field(
        gt=0.0,
        le=1.0,
    )


    matches: Tuple[
        ContaminationMatch,
        ...,
    ] = ()


    match_count: int = Field(
        ge=0,
    )


    contaminated: bool


    passed: bool


    governance_rule_version: Literal[
        "adaptation_data_governance_v0.1"
    ] = ADAPTATION_DATA_GOVERNANCE_RULE_VERSION


    report_rule_version: Literal[
        "adaptation_contamination_report_v0.1"
    ] = CONTAMINATION_REPORT_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_result(
        self,
    ) -> "AdaptationContaminationReport":
        if (
            self.match_count
            !=
            len(
                self.matches
            )
        ):
            raise ValueError(
                "Contamination match count mismatch."
            )


        expected_contaminated = (
            self.match_count
            >
            0
        )


        if (
            self.contaminated
            !=
            expected_contaminated
        ):
            raise ValueError(
                "Contamination status is inconsistent "
                "with the match count."
            )


        if (
            self.passed
            ==
            self.contaminated
        ):
            raise ValueError(
                "A contaminated dataset may not pass "
                "the governance gate."
            )


        return self


# ============================================================
# INTERNAL RECORDS
# ============================================================


@dataclass(
    frozen=True,
)
class _ProtectedTextRecord:
    source_path: str

    locator: str

    raw_sha256: str

    normalized_sha256: str

    normalized_text: str


@dataclass(
    frozen=True,
)
class _ProtectedStructuredRecord:
    source_path: str

    locator: str

    canonical_sha256: str


@dataclass(
    frozen=True,
)
class _ProtectedIndex:
    corpus: ProtectedEvidenceCorpus

    text_records: Tuple[
        _ProtectedTextRecord,
        ...,
    ]

    structured_records: Tuple[
        _ProtectedStructuredRecord,
        ...,
    ]


# ============================================================
# HASHING / NORMALIZATION
# ============================================================


def _sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


def _sha256_text(
    value: str,
) -> str:
    return _sha256_bytes(
        value.encode(
            "utf-8"
        )
    )


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
    value: Any,
) -> bytes:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    )


    return serialized.encode(
        "utf-8"
    )


def _canonical_sha256(
    value: Any,
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            value
        )
    )


def normalize_evidence_text(
    value: str,
) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    )


    normalized = normalized.translate(
        str.maketrans(
            {
                "\u2018":
                    "'",
                "\u2019":
                    "'",
                "\u201c":
                    '"',
                "\u201d":
                    '"',
                "\u2013":
                    "-",
                "\u2014":
                    "-",
                "\u00a0":
                    " ",
            }
        )
    )


    normalized = normalized.casefold()


    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )


    return normalized.strip()


def _text_token_count(
    value: str,
) -> int:
    return len(
        re.findall(
            r"\w+",
            value,
            flags=re.UNICODE,
        )
    )


def _qualifies_for_text_fingerprint(
    value: str,
) -> bool:
    normalized = normalize_evidence_text(
        value
    )


    if (
        len(
            normalized
        )
        <
        MIN_PROTECTED_TEXT_CHARS
    ):
        return False


    if (
        _text_token_count(
            normalized
        )
        <
        MIN_PROTECTED_TEXT_TOKENS
    ):
        return False


    return True


# ============================================================
# PATH HELPERS
# ============================================================


def _relative_path(
    *,
    root: Path,
    path: Path,
) -> str:
    root = root.resolve()

    path = path.resolve()


    try:
        relative = path.relative_to(
            root
        )

    except ValueError as error:
        raise ValueError(
            "Protected evidence must live inside "
            "the repository root."
        ) from error


    return relative.as_posix()


def _locator_leaf(
    locator: str,
) -> str:
    if not locator:
        return ""


    segment = locator.split(
        "."
    )[
        -1
    ]


    segment = re.sub(
        r"\[\d+\]$",
        "",
        segment,
    )


    return segment


def _is_excluded_metadata_locator(
    locator: str,
) -> bool:
    leaf = (
        _locator_leaf(
            locator
        )
        .lower()
    )


    if (
        leaf
        in
        EXCLUDED_METADATA_KEYS
    ):
        return True


    if (
        "sha256"
        in leaf
    ):
        return True


    if leaf.endswith(
        "_id"
    ):
        return True


    if leaf.endswith(
        "_version"
    ):
        return True


    return False


def _is_content_bearing_locator(
    locator: str,
) -> bool:
    lowered = locator.lower()


    if (
        _is_excluded_metadata_locator(
            locator
        )
    ):
        return False


    return any(
        fragment
        in
        lowered

        for fragment
        in CONTENT_FIELD_FRAGMENTS
    )


def _is_structured_record_locator(
    locator: str,
) -> bool:
    if not locator:
        return False


    pattern = (
        r"(?:^|\.)("
        +
        "|".join(
            sorted(
                re.escape(
                    name
                )

                for name
                in STRUCTURED_CONTAINER_NAMES
            )
        )
        +
        r")\[\d+\]$"
    )


    return bool(
        re.search(
            pattern,
            locator,
        )
    )


# ============================================================
# SOURCE CLASSIFICATION
# ============================================================


def _load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def _classify_json_artifact(
    *,
    repository_root: Path,
    path: Path,
    payload: Any,
) -> Optional[
    ProtectedSourceClassification
]:
    relative = (
        _relative_path(
            root=repository_root,
            path=path,
        )
        .lower()
    )


    phase = None


    if isinstance(
        payload,
        dict,
    ):
        phase = payload.get(
            "phase"
        )


    if (
        phase
        ==
        "regression_baseline"
    ):
        return (
            "protected_regression"
        )


    if (
        phase
        ==
        "pre_adaptation_holdout"
    ):
        return (
            "protected_pre_adaptation_holdout"
        )


    if (
        "development"
        in relative
    ):
        return None


    if (
        "holdout"
        in relative
    ):
        return (
            "protected_holdout"
        )


    if (
        "regression"
        in relative
    ):
        return (
            "protected_regression"
        )


    return None


def _discover_protected_json_sources(
    *,
    repository_root: Path,
) -> Dict[
    Path,
    ProtectedSourceClassification,
]:
    artifact_root = (
        repository_root
        /
        "artifacts"
        /
        "evaluation"
    )


    if not artifact_root.is_dir():
        raise RuntimeError(
            "Evaluation artifact root is missing: "
            f"{artifact_root}"
        )


    result: Dict[
        Path,
        ProtectedSourceClassification,
    ] = {}


    for path in sorted(
        artifact_root.rglob(
            "*.json"
        )
    ):
        if not path.is_file():
            continue


        try:
            payload = _load_json(
                path
            )

        except Exception:
            continue


        classification = (
            _classify_json_artifact(
                repository_root=
                    repository_root,
                path=path,
                payload=payload,
            )
        )


        if (
            classification
            is not None
        ):
            result[
                path.resolve()
            ] = classification


    return result


def _discover_protected_python_sources(
    *,
    repository_root: Path,
) -> Dict[
    Path,
    ProtectedSourceClassification,
]:
    result: Dict[
        Path,
        ProtectedSourceClassification,
    ] = {}


    benchmark_root = (
        repository_root
        /
        "app"
        /
        "evaluation"
        /
        "benchmarks"
    )


    if benchmark_root.is_dir():
        for path in sorted(
            benchmark_root.glob(
                "*.py"
            )
        ):
            if (
                not path.is_file()
                or
                path.name
                ==
                "__init__.py"
            ):
                continue


            if (
                "development"
                in
                path.name.lower()
            ):
                continue


            result[
                path.resolve()
            ] = (
                "protected_code_evaluation"
            )


    evaluation_root = (
        repository_root
        /
        "app"
        /
        "evaluation"
    )


    if evaluation_root.is_dir():
        for path in sorted(
            evaluation_root.rglob(
                "*holdout*.py"
            )
        ):
            if (
                path.is_file()
                and
                path.name
                !=
                "__init__.py"
            ):
                result[
                    path.resolve()
                ] = (
                    "protected_code_evaluation"
                )


    evals_root = (
        repository_root
        /
        "app"
        /
        "evals"
    )


    if evals_root.is_dir():
        for path in sorted(
            evals_root.rglob(
                "*.py"
            )
        ):
            if (
                not path.is_file()
                or
                path.name
                ==
                "__init__.py"
            ):
                continue


            relative_parts = {
                part.lower()

                for part
                in path.relative_to(
                    evals_root
                ).parts
            }


            if (
                "benchmark"
                in
                path.name.lower()
                or
                "scenarios"
                in
                relative_parts
            ):
                result[
                    path.resolve()
                ] = (
                    "protected_code_evaluation"
                )


    return result


# ============================================================
# JSON FINGERPRINT EXTRACTION
# ============================================================


def _walk_json_nodes(
    value: Any,
    *,
    locator: str,
) -> Iterable[
    Tuple[
        str,
        Any,
    ]
]:
    yield (
        locator,
        value,
    )


    if isinstance(
        value,
        dict,
    ):
        for (
            key,
            child,
        ) in value.items():
            child_locator = (
                f"{locator}.{key}"
                if locator
                else
                str(
                    key
                )
            )


            yield from _walk_json_nodes(
                child,
                locator=
                    child_locator,
            )


    elif isinstance(
        value,
        list,
    ):
        for (
            index,
            child,
        ) in enumerate(
            value
        ):
            child_locator = (
                f"{locator}[{index}]"
                if locator
                else
                f"[{index}]"
            )


            yield from _walk_json_nodes(
                child,
                locator=
                    child_locator,
            )


def _extract_json_records(
    *,
    source_path: str,
    payload: Any,
) -> Tuple[
    List[
        _ProtectedTextRecord
    ],
    List[
        _ProtectedStructuredRecord
    ],
]:
    text_records: List[
        _ProtectedTextRecord
    ] = []


    structured_records: List[
        _ProtectedStructuredRecord
    ] = []


    for (
        locator,
        value,
    ) in _walk_json_nodes(
        payload,
        locator="",
    ):
        if isinstance(
            value,
            str,
        ):
            if not _is_content_bearing_locator(
                locator
            ):
                continue


            if not _qualifies_for_text_fingerprint(
                value
            ):
                continue


            normalized = (
                normalize_evidence_text(
                    value
                )
            )


            text_records.append(
                _ProtectedTextRecord(
                    source_path=
                        source_path,
                    locator=(
                        locator
                        or
                        "<root>"
                    ),
                    raw_sha256=
                        _sha256_text(
                            value
                        ),
                    normalized_sha256=
                        _sha256_text(
                            normalized
                        ),
                    normalized_text=
                        normalized,
                )
            )


        elif isinstance(
            value,
            dict,
        ):
            if not _is_structured_record_locator(
                locator
            ):
                continue


            structured_records.append(
                _ProtectedStructuredRecord(
                    source_path=
                        source_path,
                    locator=locator,
                    canonical_sha256=
                        _canonical_sha256(
                            value
                        ),
                )
            )


    return (
        text_records,
        structured_records,
    )


# ============================================================
# PYTHON FINGERPRINT EXTRACTION
# ============================================================


def _assignment_names(
    node: Any,
) -> List[
    str
]:
    names: List[
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


def _is_data_assignment_name(
    name: str,
) -> bool:
    upper = name.upper()


    return any(
        fragment
        in
        upper

        for fragment
        in PYTHON_DATA_ASSIGNMENT_FRAGMENTS
    )


def _docstring_node_ids(
    tree: ast.AST,
) -> Set[
    int
]:
    result: Set[
        int
    ] = set()


    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            (
                ast.Module,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue


        body = getattr(
            node,
            "body",
            None,
        )


        if not body:
            continue


        first = body[
            0
        ]


        if (
            isinstance(
                first,
                ast.Expr,
            )
            and
            isinstance(
                first.value,
                ast.Constant,
            )
            and
            isinstance(
                first.value.value,
                str,
            )
        ):
            result.add(
                id(
                    first.value
                )
            )


    return result


def _walk_literal_structures(
    value: Any,
    *,
    locator: str,
) -> Iterable[
    Tuple[
        str,
        Any,
    ]
]:
    yield (
        locator,
        value,
    )


    if isinstance(
        value,
        dict,
    ):
        for (
            key,
            child,
        ) in value.items():
            child_locator = (
                f"{locator}.{key}"
                if locator
                else
                str(
                    key
                )
            )


            yield from _walk_literal_structures(
                child,
                locator=
                    child_locator,
            )


    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        for (
            index,
            child,
        ) in enumerate(
            value
        ):
            yield from _walk_literal_structures(
                child,
                locator=(
                    f"{locator}[{index}]"
                ),
            )


def _extract_python_records(
    *,
    source_path: str,
    source: str,
) -> Tuple[
    List[
        _ProtectedTextRecord
    ],
    List[
        _ProtectedStructuredRecord
    ],
]:
    tree = ast.parse(
        source
    )


    docstring_ids = (
        _docstring_node_ids(
            tree
        )
    )


    text_records: List[
        _ProtectedTextRecord
    ] = []


    structured_records: List[
        _ProtectedStructuredRecord
    ] = []


    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Constant,
        ):
            continue


        if (
            id(
                node
            )
            in
            docstring_ids
        ):
            continue


        if not isinstance(
            node.value,
            str,
        ):
            continue


        if not _qualifies_for_text_fingerprint(
            node.value
        ):
            continue


        normalized = (
            normalize_evidence_text(
                node.value
            )
        )


        locator = (
            "python:"
            f"{getattr(node, 'lineno', 0)}:"
            f"{getattr(node, 'col_offset', 0)}"
        )


        text_records.append(
            _ProtectedTextRecord(
                source_path=
                    source_path,
                locator=locator,
                raw_sha256=
                    _sha256_text(
                        node.value
                    ),
                normalized_sha256=
                    _sha256_text(
                        normalized
                    ),
                normalized_text=
                    normalized,
            )
        )


    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue


        names = (
            _assignment_names(
                node
            )
        )


        if not any(
            _is_data_assignment_name(
                name
            )

            for name
            in names
        ):
            continue


        value_node = getattr(
            node,
            "value",
            None,
        )


        if value_node is None:
            continue


        try:
            literal = ast.literal_eval(
                value_node
            )

        except Exception:
            continue


        assignment_name = (
            names[
                0
            ]
            if names
            else
            "<assignment>"
        )


        for (
            locator,
            value,
        ) in _walk_literal_structures(
            literal,
            locator=
                assignment_name,
        ):
            if not isinstance(
                value,
                dict,
            ):
                continue


            structured_records.append(
                _ProtectedStructuredRecord(
                    source_path=
                        source_path,
                    locator=locator,
                    canonical_sha256=
                        _canonical_sha256(
                            value
                        ),
                )
            )


    return (
        text_records,
        structured_records,
    )


# ============================================================
# PROTECTED CORPUS
# ============================================================


def _build_protected_index(
    *,
    repository_root: Path,
    additional_protected_paths: Sequence[
        Path
    ] = (),
) -> _ProtectedIndex:
    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )


    if not repository_root.is_dir():
        raise RuntimeError(
            "Repository root does not exist."
        )


    json_sources = (
        _discover_protected_json_sources(
            repository_root=
                repository_root,
        )
    )


    python_sources = (
        _discover_protected_python_sources(
            repository_root=
                repository_root,
        )
    )


    all_sources: Dict[
        Path,
        Tuple[
            ProtectedSourceClassification,
            ProtectedSourceType,
        ],
    ] = {}


    for (
        path,
        classification,
    ) in json_sources.items():
        all_sources[
            path
        ] = (
            classification,
            "json_artifact",
        )


    for (
        path,
        classification,
    ) in python_sources.items():
        all_sources[
            path
        ] = (
            classification,
            "python_definition",
        )


    for raw_path in additional_protected_paths:
        path = (
            raw_path
            .expanduser()
            .resolve()
        )


        _relative_path(
            root=repository_root,
            path=path,
        )


        if not path.is_file():
            raise RuntimeError(
                "Additional protected evidence file "
                f"is missing: {path}"
            )


        if (
            path.suffix.lower()
            ==
            ".json"
        ):
            source_type: ProtectedSourceType = (
                "json_artifact"
            )

        elif (
            path.suffix.lower()
            ==
            ".py"
        ):
            source_type = (
                "python_definition"
            )

        else:
            raise ValueError(
                "Additional protected evidence must "
                "be JSON or Python."
            )


        all_sources[
            path
        ] = (
            "protected_final_acceptance_holdout",
            source_type,
        )


    if not all_sources:
        raise RuntimeError(
            "No protected evidence sources were discovered."
        )


    source_models: List[
        ProtectedEvidenceSource
    ] = []


    text_records: List[
        _ProtectedTextRecord
    ] = []


    structured_records: List[
        _ProtectedStructuredRecord
    ] = []


    for path in sorted(
        all_sources,
        key=lambda item: str(
            item
        ),
    ):
        (
            classification,
            source_type,
        ) = all_sources[
            path
        ]


        relative = (
            _relative_path(
                root=repository_root,
                path=path,
            )
        )


        source_sha256 = (
            _sha256_file(
                path
            )
        )


        source_models.append(
            ProtectedEvidenceSource(
                relative_path=
                    relative,
                sha256=
                    source_sha256,
                classification=
                    classification,
                source_type=
                    source_type,
            )
        )


        if (
            source_type
            ==
            "json_artifact"
        ):
            payload = _load_json(
                path
            )


            (
                source_text_records,
                source_structured_records,
            ) = _extract_json_records(
                source_path=
                    relative,
                payload=payload,
            )


        else:
            source = path.read_text(
                encoding="utf-8-sig"
            )


            (
                source_text_records,
                source_structured_records,
            ) = _extract_python_records(
                source_path=
                    relative,
                source=source,
            )


        text_records.extend(
            source_text_records
        )


        structured_records.extend(
            source_structured_records
        )


    if not text_records:
        raise RuntimeError(
            "Protected evidence corpus contains "
            "no meaningful text fingerprints."
        )


    text_records = sorted(
        set(
            text_records
        ),
        key=lambda item: (
            item.source_path,
            item.locator,
            item.normalized_sha256,
        ),
    )


    structured_records = sorted(
        set(
            structured_records
        ),
        key=lambda item: (
            item.source_path,
            item.locator,
            item.canonical_sha256,
        ),
    )


    corpus_identity = {
        "rule_version":
            PROTECTED_EVIDENCE_CORPUS_RULE_VERSION,
        "sources": [
            source.model_dump(
                mode="json"
            )

            for source
            in source_models
        ],
        "text_fingerprints": [
            {
                "source_path":
                    record.source_path,
                "locator":
                    record.locator,
                "raw_sha256":
                    record.raw_sha256,
                "normalized_sha256":
                    record.normalized_sha256,
            }

            for record
            in text_records
        ],
        "structured_fingerprints": [
            {
                "source_path":
                    record.source_path,
                "locator":
                    record.locator,
                "canonical_sha256":
                    record.canonical_sha256,
            }

            for record
            in structured_records
        ],
    }


    corpus = ProtectedEvidenceCorpus(
        sources=
            tuple(
                source_models
            ),
        source_count=
            len(
                source_models
            ),
        text_fingerprint_count=
            len(
                text_records
            ),
        structured_fingerprint_count=
            len(
                structured_records
            ),
        corpus_sha256=
            _canonical_sha256(
                corpus_identity
            ),
    )


    return _ProtectedIndex(
        corpus=corpus,
        text_records=
            tuple(
                text_records
            ),
        structured_records=
            tuple(
                structured_records
            ),
    )


def build_protected_evidence_corpus(
    *,
    repository_root: Path,
    additional_protected_paths: Sequence[
        Path
    ] = (),
) -> ProtectedEvidenceCorpus:
    return _build_protected_index(
        repository_root=
            repository_root,
        additional_protected_paths=
            additional_protected_paths,
    ).corpus


# ============================================================
# CANDIDATE DATA EXTRACTION
# ============================================================


def _candidate_example_id(
    *,
    example: Mapping[
        str,
        Any,
    ],
    index: int,
) -> str:
    for key in (
        "example_id",
        "id",
    ):
        value = example.get(
            key
        )


        if (
            isinstance(
                value,
                str,
            )
            and
            value.strip()
        ):
            return value.strip()


    return (
        f"example:{index:04d}"
    )


def _candidate_text_records(
    *,
    example: Mapping[
        str,
        Any,
    ],
) -> Iterable[
    Tuple[
        str,
        str,
    ]
]:
    for (
        locator,
        value,
    ) in _walk_json_nodes(
        example,
        locator="",
    ):
        if not isinstance(
            value,
            str,
        ):
            continue


        if (
            _is_excluded_metadata_locator(
                locator
            )
        ):
            continue


        if not _qualifies_for_text_fingerprint(
            value
        ):
            continue


        yield (
            locator
            or
            "<root>",
            value,
        )


def _candidate_structured_records(
    *,
    example: Mapping[
        str,
        Any,
    ],
) -> Iterable[
    Tuple[
        str,
        str,
    ]
]:
    for (
        locator,
        value,
    ) in _walk_json_nodes(
        example,
        locator="",
    ):
        if not isinstance(
            value,
            dict,
        ):
            continue


        yield (
            locator
            or
            "<root>",
            _canonical_sha256(
                value
            ),
        )


# ============================================================
# CONTAMINATION SCAN
# ============================================================


def scan_adaptation_examples(
    *,
    repository_root: Path,
    dataset_id: str,
    dataset_version: str,
    examples: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    near_duplicate_threshold: float = (
        DEFAULT_NEAR_DUPLICATE_THRESHOLD
    ),
    additional_protected_paths: Sequence[
        Path
    ] = (),
) -> AdaptationContaminationReport:
    if not dataset_id.strip():
        raise ValueError(
            "dataset_id is required."
        )


    if not dataset_version.strip():
        raise ValueError(
            "dataset_version is required."
        )


    if not examples:
        raise ValueError(
            "Adaptation dataset must contain "
            "at least one example."
        )


    if (
        near_duplicate_threshold
        <=
        0.0
        or
        near_duplicate_threshold
        >
        1.0
    ):
        raise ValueError(
            "near_duplicate_threshold must be "
            "within (0, 1]."
        )


    protected_index = (
        _build_protected_index(
            repository_root=
                repository_root,
            additional_protected_paths=
                additional_protected_paths,
        )
    )


    raw_text_lookup: Dict[
        str,
        List[
            _ProtectedTextRecord
        ],
    ] = {}


    normalized_text_lookup: Dict[
        str,
        List[
            _ProtectedTextRecord
        ],
    ] = {}


    for record in (
        protected_index.text_records
    ):
        raw_text_lookup.setdefault(
            record.raw_sha256,
            [],
        ).append(
            record
        )


        normalized_text_lookup.setdefault(
            record.normalized_sha256,
            [],
        ).append(
            record
        )


    structured_lookup: Dict[
        str,
        List[
            _ProtectedStructuredRecord
        ],
    ] = {}


    for record in (
        protected_index.structured_records
    ):
        structured_lookup.setdefault(
            record.canonical_sha256,
            [],
        ).append(
            record
        )


    matches: List[
        ContaminationMatch
    ] = []


    seen_match_keys: Set[
        Tuple[
            str,
            str,
            str,
            str,
        ]
    ] = set()


    def append_text_match(
        *,
        example_id: str,
        candidate_locator: str,
        candidate_sha256: str,
        match_kind: ContaminationMatchKind,
        protected_record: _ProtectedTextRecord,
        similarity: Optional[
            float
        ] = None,
    ) -> None:
        dedupe_key = (
            example_id,
            candidate_locator,
            match_kind,
            protected_record.normalized_sha256,
        )


        if (
            dedupe_key
            in
            seen_match_keys
        ):
            return


        seen_match_keys.add(
            dedupe_key
        )


        matches.append(
            ContaminationMatch(
                candidate_example_id=
                    example_id,
                candidate_locator=
                    candidate_locator,
                match_kind=
                    match_kind,
                candidate_sha256=
                    candidate_sha256,
                protected_source_path=
                    protected_record.source_path,
                protected_locator=
                    protected_record.locator,
                protected_sha256=
                    protected_record.normalized_sha256,
                similarity=
                    similarity,
            )
        )


    for (
        example_index,
        example,
    ) in enumerate(
        examples
    ):
        if not isinstance(
            example,
            Mapping,
        ):
            raise TypeError(
                "Every adaptation example must "
                "be a mapping."
            )


        example_id = (
            _candidate_example_id(
                example=example,
                index=example_index,
            )
        )


        for (
            locator,
            candidate_text,
        ) in _candidate_text_records(
            example=example,
        ):
            raw_sha256 = (
                _sha256_text(
                    candidate_text
                )
            )


            normalized_text = (
                normalize_evidence_text(
                    candidate_text
                )
            )


            normalized_sha256 = (
                _sha256_text(
                    normalized_text
                )
            )


            exact_raw_records = (
                raw_text_lookup.get(
                    raw_sha256,
                    [],
                )
            )


            if exact_raw_records:
                for record in (
                    exact_raw_records
                ):
                    append_text_match(
                        example_id=
                            example_id,
                        candidate_locator=
                            locator,
                        candidate_sha256=
                            raw_sha256,
                        match_kind=
                            "exact_raw_text",
                        protected_record=
                            record,
                        similarity=1.0,
                    )


                continue


            exact_normalized_records = (
                normalized_text_lookup.get(
                    normalized_sha256,
                    [],
                )
            )


            if exact_normalized_records:
                for record in (
                    exact_normalized_records
                ):
                    append_text_match(
                        example_id=
                            example_id,
                        candidate_locator=
                            locator,
                        candidate_sha256=
                            normalized_sha256,
                        match_kind=
                            "exact_normalized_text",
                        protected_record=
                            record,
                        similarity=1.0,
                    )


                continue


            best_record: Optional[
                _ProtectedTextRecord
            ] = None


            best_similarity = 0.0


            candidate_length = len(
                normalized_text
            )


            for protected_record in (
                protected_index.text_records
            ):
                protected_text = (
                    protected_record.normalized_text
                )


                protected_length = len(
                    protected_text
                )


                if (
                    candidate_length
                    ==
                    0
                    or
                    protected_length
                    ==
                    0
                ):
                    continue


                length_ratio = (
                    min(
                        candidate_length,
                        protected_length,
                    )
                    /
                    max(
                        candidate_length,
                        protected_length,
                    )
                )


                if (
                    length_ratio
                    <
                    0.70
                ):
                    continue


                similarity = (
                    SequenceMatcher(
                        None,
                        normalized_text,
                        protected_text,
                        autojunk=False,
                    )
                    .ratio()
                )


                if (
                    similarity
                    >
                    best_similarity
                ):
                    best_similarity = (
                        similarity
                    )

                    best_record = (
                        protected_record
                    )


            if (
                best_record
                is not None
                and
                best_similarity
                >=
                near_duplicate_threshold
            ):
                append_text_match(
                    example_id=
                        example_id,
                    candidate_locator=
                        locator,
                    candidate_sha256=
                        normalized_sha256,
                    match_kind=
                        "near_duplicate_text",
                    protected_record=
                        best_record,
                    similarity=
                        round(
                            best_similarity,
                            6,
                        ),
                )


        for (
            locator,
            canonical_sha256,
        ) in _candidate_structured_records(
            example=example,
        ):
            protected_records = (
                structured_lookup.get(
                    canonical_sha256,
                    [],
                )
            )


            for protected_record in (
                protected_records
            ):
                dedupe_key = (
                    example_id,
                    locator,
                    "exact_structured_record",
                    canonical_sha256,
                )


                if (
                    dedupe_key
                    in
                    seen_match_keys
                ):
                    continue


                seen_match_keys.add(
                    dedupe_key
                )


                matches.append(
                    ContaminationMatch(
                        candidate_example_id=
                            example_id,
                        candidate_locator=
                            locator,
                        match_kind=
                            "exact_structured_record",
                        candidate_sha256=
                            canonical_sha256,
                        protected_source_path=
                            protected_record.source_path,
                        protected_locator=
                            protected_record.locator,
                        protected_sha256=
                            protected_record.canonical_sha256,
                        similarity=1.0,
                    )
                )


    matches = sorted(
        matches,
        key=lambda match: (
            match.candidate_example_id,
            match.candidate_locator,
            match.match_kind,
            match.protected_source_path,
            match.protected_locator,
        ),
    )


    dataset_payload = [
        dict(
            example
        )

        for example
        in examples
    ]


    contaminated = bool(
        matches
    )


    return AdaptationContaminationReport(
        dataset_id=
            dataset_id.strip(),
        dataset_version=
            dataset_version.strip(),
        candidate_dataset_sha256=
            _canonical_sha256(
                dataset_payload
            ),
        candidate_example_count=
            len(
                examples
            ),
        protected_corpus_sha256=
            protected_index.corpus.corpus_sha256,
        protected_source_count=
            protected_index.corpus.source_count,
        protected_text_fingerprint_count=
            protected_index.corpus.text_fingerprint_count,
        protected_structured_fingerprint_count=
            protected_index.corpus.structured_fingerprint_count,
        near_duplicate_threshold=
            near_duplicate_threshold,
        matches=
            tuple(
                matches
            ),
        match_count=
            len(
                matches
            ),
        contaminated=
            contaminated,
        passed=
            not contaminated,
    )


def assert_adaptation_dataset_clean(
    report: AdaptationContaminationReport,
) -> None:
    if not report.passed:
        raise RuntimeError(
            "Adaptation dataset contamination gate failed. "
            f"Detected {report.match_count} protected-evidence "
            "match(es)."
        )


# ============================================================
# REPORT FREEZE
# ============================================================


def write_contamination_report(
    *,
    report: AdaptationContaminationReport,
    output_path: Path,
) -> str:
    output_path = (
        output_path
        .expanduser()
        .resolve()
    )


    if output_path.exists():
        raise FileExistsError(
            "Contamination report already exists: "
            f"{output_path}"
        )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = (
        report.model_dump(
            mode="json"
        )
    )


    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    )


    output_path.write_text(
        serialized
        +
        "\n",
        encoding="utf-8",
        newline="\n",
    )


    return _sha256_file(
        output_path
    )

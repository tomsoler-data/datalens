from __future__ import annotations

from enum import Enum
import hashlib
import json
import re
from typing import Any
from urllib.error import (
    HTTPError,
    URLError,
)
from urllib.request import (
    Request,
)

from app.security.llm_egress import (
    open_local_llm_request,
)

from app.security.llm_payload import (
    LLMPayloadClass,
)

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from app.preparation.data_quality import (
    DataQualityReport,
    QualityIssue,
    QualityIssueKind,
)


SEMANTIC_REVIEW_RULE_VERSION = (
    "semantic_review_v0.3"
)


SEMANTIC_CANONICALIZATION_RULE_VERSION = (
    "semantic_canonicalization_v0.1"
)


DEFAULT_SEMANTIC_REVIEW_MODEL = (
    "gemma3:4b"
)


DEFAULT_OLLAMA_CHAT_URL = (
    "http://127.0.0.1:11434/api/chat"
)


SEMANTIC_MERGE_CANDIDATE_KINDS = {
    QualityIssueKind
    .CATEGORY_FORMAT_VARIANTS,

    QualityIssueKind
    .POSSIBLE_SEMANTIC_ALIASES,
}


class SemanticVerdict(
    str,
    Enum,
):
    MERGE_VALUES = (
        "merge_values"
    )

    KEEP_SEPARATE = (
        "keep_separate"
    )

    FLAG_FOR_REVIEW = (
        "flag_for_review"
    )

    CONTEXTUALIZE = (
        "contextualize"
    )

    NO_CHANGE = (
        "no_change"
    )

    ABSTAIN = (
        "abstain"
    )


class SemanticReviewCandidate(
    BaseModel,
):
    issue_id: str
    dataset_id: str
    dataset_filename: str
    column: str | None
    kind: QualityIssueKind
    severity: str

    title: str
    explanation: str

    observed_count: int
    affected_ratio: float

    examples: list[str] = Field(
        default_factory=list,
    )

    candidate_values: list[
        str
    ] = Field(
        default_factory=list,
    )

    candidate_groups: list[
        list[
            str
        ]
    ] = Field(
        default_factory=list,
    )

    context: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class RawSemanticDecision(
    BaseModel,
):
    issue_id: str

    verdict: SemanticVerdict

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    rationale: str = Field(
        min_length=1,
        max_length=800,
    )

    source_values: list[
        str
    ] = Field(
        default_factory=list,
    )

    canonical_value: str | None = None

    user_message: str = Field(
        min_length=1,
        max_length=500,
    )


class RawSemanticReviewResponse(
    BaseModel,
):
    decisions: list[
        RawSemanticDecision
    ]


class ValidatedSemanticDecision(
    BaseModel,
):
    issue_id: str
    dataset_id: str
    dataset_filename: str
    column: str | None
    kind: QualityIssueKind

    verdict: SemanticVerdict

    confidence: float
    rationale: str

    source_values: list[
        str
    ] = Field(
        default_factory=list,
    )

    canonical_value: str | None = None

    user_message: str

    python_validated: bool = True

    executable: bool = False

    requires_user_confirmation: bool = True

    validation_notes: list[
        str
    ] = Field(
        default_factory=list,
    )


class SemanticReviewReport(
    BaseModel,
):
    status: str

    model: str

    candidate_count: int

    decision_count: int

    merge_proposal_count: int

    abstention_count: int

    decisions: list[
        ValidatedSemanticDecision
    ]

    notes: list[str]

    rule_version: str = (
        SEMANTIC_REVIEW_RULE_VERSION
    )


def _normalize_token(
    value: Any,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(
            value
        ).strip(),
    ).casefold()


def _safe_dtype(
    dataframe: pd.DataFrame,
    column: str | None,
) -> str | None:
    if (
        column is None
        or
        column not in dataframe.columns
    ):
        return None

    return str(
        dataframe[
            column
        ].dtype
    )


def _safe_unique_values(
    dataframe: pd.DataFrame,
    column: str | None,
    *,
    limit: int = 20,
) -> list[str]:
    if (
        column is None
        or
        column not in dataframe.columns
    ):
        return []

    output: list[
        str
    ] = []

    for value in (
        dataframe[
            column
        ]
        .dropna()
        .astype(
            str
        )
    ):
        if value in output:
            continue

        output.append(
            value
        )

        if (
            len(
                output
            )
            >=
            limit
        ):
            break

    return output


def _exact_values_for_alias_issue(
    *,
    issue: QualityIssue,
    dataframe: pd.DataFrame,
) -> tuple[
    list[
        str
    ],
    list[
        list[
            str
        ]
    ],
]:
    if (
        issue.column is None
        or
        issue.column not in dataframe.columns
    ):
        return (
            [],
            [],
        )

    pairs = (
        issue.evidence
        .details
        .get(
            "candidate_pairs",
            [],
        )
    )

    observed_values = (
        dataframe[
            issue.column
        ]
        .dropna()
        .astype(
            str
        )
        .unique()
        .tolist()
    )

    groups: list[
        list[
            str
        ]
    ] = []

    flat: list[
        str
    ] = []

    for pair in pairs:
        if (
            not isinstance(
                pair,
                (
                    list,
                    tuple,
                ),
            )
            or
            len(
                pair
            )
            !=
            2
        ):
            continue

        normalized_pair = {
            _normalize_token(
                pair[
                    0
                ]
            ),
            _normalize_token(
                pair[
                    1
                ]
            ),
        }

        group = [
            value

            for value
            in observed_values

            if (
                _normalize_token(
                    value
                )
                in
                normalized_pair
            )
        ]

        unique_group: list[
            str
        ] = []

        for value in group:
            if value not in unique_group:
                unique_group.append(
                    value
                )

        if (
            len(
                unique_group
            )
            <
            2
        ):
            continue

        # Avoid duplicated groups when multiple deterministic
        # pairs reconstruct the same exact observed aliases.
        group_key = frozenset(
            unique_group
        )

        if any(
            frozenset(
                existing
            )
            ==
            group_key

            for existing
            in groups
        ):
            continue

        groups.append(
            unique_group
        )

        for value in unique_group:
            if value not in flat:
                flat.append(
                    value
                )

    return (
        flat,
        groups,
    )


def _stable_alias_candidate_id(
    *,
    source_issue_id: str,
    group: list[str],
    group_index: int,
) -> str:
    """
    Preserve the historical issue_id for the first alias group.

    Additional groups receive deterministic synthetic IDs so
    one quality issue may safely produce several independent
    semantic decisions without breaking duplicate-ID guards.
    """

    if (
        group_index
        ==
        0
    ):
        return source_issue_id

    payload = {
        "source_issue_id":
            source_issue_id,

        "group":
            sorted(
                group
            ),
    }

    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )
    ).hexdigest()[
        :12
    ]

    return (
        f"{source_issue_id}:alias:{digest}"
    )


def _candidate_from_issue(
    *,
    issue: QualityIssue,
    dataframe: pd.DataFrame,
    issue_id: str,
    candidate_values: list[str],
    candidate_groups: list[
        list[
            str
        ]
    ],
    source_issue_id: str,
    alias_group_index: int | None,
) -> SemanticReviewCandidate:
    context: dict[
        str,
        Any,
    ] = {
        "column_dtype":
            _safe_dtype(
                dataframe,
                issue.column,
            ),

        "sample_unique_values":
            _safe_unique_values(
                dataframe,
                issue.column,
                limit=12,
            ),

        "deterministic_details":
            issue.evidence
            .details,

        "source_quality_issue_id":
            source_issue_id,

        "alias_group_index":
            alias_group_index,

        "semantic_canonicalization_rule_version":
            SEMANTIC_CANONICALIZATION_RULE_VERSION,
    }

    return (
        SemanticReviewCandidate(
            issue_id=
                issue_id,

            dataset_id=
                issue.dataset_id,

            dataset_filename=
                issue
                .dataset_filename,

            column=
                issue.column,

            kind=
                issue.kind,

            severity=
                issue.severity.value,

            title=
                issue.title,

            explanation=
                issue.explanation,

            observed_count=
                issue.evidence
                .observed_count,

            affected_ratio=
                issue.evidence
                .affected_ratio,

            examples=
                list(
                    issue.evidence
                    .examples[
                        :8
                    ]
                ),

            candidate_values=
                candidate_values,

            candidate_groups=
                candidate_groups,

            context=
                context,
        )
    )


def build_semantic_review_candidates(
    *,
    quality_report: DataQualityReport,
    dataset_frames: dict[
        str,
        pd.DataFrame,
    ],
) -> list[
    SemanticReviewCandidate
]:
    """
    Convert semantic-review quality signals into minimal,
    privacy-preserving candidates.

    Alias issues are reviewed at GROUP granularity.

    This fixes an important ambiguity in the previous contract:
    one quality issue may contain several independent alias
    groups, while one semantic decision can describe only one
    canonical merge.

    Backward compatibility:
    - the first alias group keeps the original quality issue_id;
    - additional groups receive stable derived issue IDs;
    - non-alias issues keep their historical IDs unchanged.
    """

    candidates: list[
        SemanticReviewCandidate
    ] = []

    for issue in (
        quality_report.issues
    ):
        if not (
            issue
            .semantic_review_recommended
        ):
            continue

        dataframe = (
            dataset_frames.get(
                issue.dataset_id
            )
        )

        if dataframe is None:
            raise KeyError(
                "Missing dataframe for semantic "
                f"review dataset_id {issue.dataset_id}."
            )

        if (
            issue.kind
            in
            SEMANTIC_MERGE_CANDIDATE_KINDS
        ):
            (
                candidate_values,
                candidate_groups,
            ) = (
                _exact_values_for_alias_issue(
                    issue=
                        issue,
                    dataframe=
                        dataframe,
                )
            )

            if candidate_groups:
                for (
                    group_index,
                    group,
                ) in enumerate(
                    candidate_groups
                ):
                    candidate_id = (
                        _stable_alias_candidate_id(
                            source_issue_id=
                                issue.issue_id,
                            group=
                                group,
                            group_index=
                                group_index,
                        )
                    )

                    candidates.append(
                        _candidate_from_issue(
                            issue=
                                issue,
                            dataframe=
                                dataframe,
                            issue_id=
                                candidate_id,
                            candidate_values=
                                list(
                                    group
                                ),
                            candidate_groups=[
                                list(
                                    group
                                )
                            ],
                            source_issue_id=
                                issue.issue_id,
                            alias_group_index=
                                group_index,
                        )
                    )

                continue

            # Safe fallback when the quality signal exists but
            # no exact observed alias group can be rebuilt.
            candidates.append(
                _candidate_from_issue(
                    issue=
                        issue,
                    dataframe=
                        dataframe,
                    issue_id=
                        issue.issue_id,
                    candidate_values=
                        candidate_values,
                    candidate_groups=
                        candidate_groups,
                    source_issue_id=
                        issue.issue_id,
                    alias_group_index=
                        None,
                )
            )

            continue

        candidates.append(
            _candidate_from_issue(
                issue=
                    issue,
                dataframe=
                    dataframe,
                issue_id=
                    issue.issue_id,
                candidate_values=
                    list(
                        issue.evidence
                        .examples[
                            :8
                        ]
                    ),
                candidate_groups=
                    [],
                source_issue_id=
                    issue.issue_id,
                alias_group_index=
                    None,
            )
        )

    return candidates


def _system_prompt(
    candidate: SemanticReviewCandidate,
) -> str:
    base = (
        "You are the guarded semantic-review layer of DataLens. "
        "Python owns factual validation and data execution. "
        "You only interpret ONE ambiguous data-quality candidate "
        "at a time. "
        "Never invent values, columns, datasets, mappings, "
        "business rules, numeric bounds, replacement values, "
        "or missing-value imputations. "
        "Prefer abstain when context is insufficient. "
        "Return exactly one structured decision for the supplied "
        "issue_id and nothing else. "
    )

    if (
        candidate.kind
        in
        SEMANTIC_MERGE_CANDIDATE_KINDS
    ):
        return (
            base
            +
            "This candidate is one possible semantic alias group. "
            "If and only if the supplied exact values clearly "
            "represent the same category, use merge_values. "
            "For merge_values: "
            "source_values MUST contain at least two exact strings "
            "copied from candidate_values, and canonical_value MUST "
            "be one exact string from source_values. "
            "Do not normalize spelling yourself. "
            "Do not omit source_values. "
            "If the evidence is insufficient, abstain."
        )

    if (
        candidate.kind
        in {
            QualityIssueKind
            .NUMERIC_OUTLIERS,

            QualityIssueKind
            .INVALID_NUMERIC_VALUES,

            QualityIssueKind
            .INVALID_DATES,

            QualityIssueKind
            .MISSING_VALUES,

            QualityIssueKind
            .MISSING_IDENTIFIER,
        }
    ):
        return (
            base
            +
            "This candidate is NOT an alias merge. "
            "Do not use merge_values. "
            "You may flag_for_review, contextualize, no_change "
            "or abstain. "
            "Do not invent a corrected value or business threshold."
        )

    return (
        base
        +
        "Use the most conservative verdict supported by the "
        "provided evidence. Do not use merge_values unless the "
        "candidate kind is possible_semantic_aliases."
    )


def _user_prompt(
    candidate: SemanticReviewCandidate,
) -> str:
    payload = (
        candidate.model_dump(
            mode="json"
        )
    )

    if (
        candidate.kind
        in
        SEMANTIC_MERGE_CANDIDATE_KINDS
    ):
        instruction = (
            "Evaluate whether the exact supplied candidate values "
            "represent the same semantic category. "
            "If you choose merge_values, copy the exact strings "
            "into source_values and choose one exact existing value "
            "as canonical_value."
        )

    else:
        instruction = (
            "Interpret the quality signal conservatively. "
            "Do not propose a data replacement."
        )

    return (
        instruction
        +
        "\n\nCandidate:\n"
        +
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


def _single_decision_schema() -> dict[
    str,
    Any,
]:
    return (
        RawSemanticDecision
        .model_json_schema()
    )


def _ollama_chat_one(
    *,
    model: str,
    candidate: SemanticReviewCandidate,
    ollama_chat_url: str,
    timeout_seconds: float,
) -> RawSemanticDecision:
    request_payload = {
        "model":
            model,

        "messages": [
            {
                "role":
                    "system",

                "content":
                    _system_prompt(
                        candidate
                    ),
            },
            {
                "role":
                    "user",

                "content":
                    _user_prompt(
                        candidate
                    ),
            },
        ],

        "stream":
            False,

        "format":
            _single_decision_schema(),

        "options": {
            "temperature":
                0.0,
        },
    }

    request = Request(
        ollama_chat_url,
        data=json.dumps(
            request_payload,
            ensure_ascii=False,
        ).encode(
            "utf-8"
        ),
        headers={
            "Content-Type":
                "application/json",
        },
        method=
            "POST",
    )

    try:
        with open_local_llm_request(
            request,
            payload_class=(
                LLMPayloadClass
                .SEMANTIC_VALUE_SAMPLE
            ),
            timeout=
                timeout_seconds,
        ) as response:
            response_payload = json.loads(
                response
                .read()
                .decode(
                    "utf-8"
                )
            )

    except HTTPError as error:
        raise RuntimeError(
            "Local model semantic review request failed."
        ) from error

    except URLError as error:
        raise RuntimeError(
            "Ollama semantic review is unavailable. "
            "Verify that the local Ollama service is running."
        ) from error

    except TimeoutError as error:
        raise RuntimeError(
            "Ollama semantic review timed out."
        ) from error

    message = (
        response_payload
        .get(
            "message",
            {}
        )
    )

    content = (
        message.get(
            "content"
        )
    )

    if not isinstance(
        content,
        str,
    ):
        raise RuntimeError(
            "Ollama semantic review returned "
            "no textual JSON content."
        )

    try:
        parsed_content = json.loads(
            content
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Ollama semantic review returned "
            "invalid JSON content."
        ) from error

    try:
        decision = (
            RawSemanticDecision
            .model_validate(
                parsed_content
            )
        )

    except ValidationError as error:
        raise RuntimeError(
            "Ollama semantic review did not "
            "respect the single-decision schema."
        ) from error

    if (
        decision.issue_id
        !=
        candidate.issue_id
    ):
        raise RuntimeError(
            "Ollama semantic review returned "
            "the wrong issue_id for the current candidate."
        )

    return decision


def _observed_exact_values(
    *,
    dataframe: pd.DataFrame,
    column: str | None,
) -> set[
    str
]:
    if (
        column is None
        or
        column not in dataframe.columns
    ):
        return set()

    return set(
        dataframe[
            column
        ]
        .dropna()
        .astype(
            str
        )
        .tolist()
    )


def _abstain_decision(
    *,
    candidate: SemanticReviewCandidate,
    rationale: str,
    user_message: str,
    validation_notes: list[
        str
    ] | None = None,
) -> ValidatedSemanticDecision:
    return (
        ValidatedSemanticDecision(
            issue_id=
                candidate.issue_id,

            dataset_id=
                candidate.dataset_id,

            dataset_filename=
                candidate.dataset_filename,

            column=
                candidate.column,

            kind=
                candidate.kind,

            verdict=
                SemanticVerdict.ABSTAIN,

            confidence=
                0.0,

            rationale=
                rationale,

            source_values=
                [],

            canonical_value=
                None,

            user_message=
                user_message,

            python_validated=
                True,

            executable=
                False,

            requires_user_confirmation=
                True,

            validation_notes=
                list(
                    validation_notes
                    or
                    []
                ),
        )
    )


def _strict_normalization_group(
    candidate: SemanticReviewCandidate,
) -> list[str] | None:
    """
    Return the alias group only when every exact observed value
    collapses to ONE non-empty token under the conservative
    normalization already used by DataLens:

        trim
        collapse whitespace
        Unicode casefold

    No spelling correction or fuzzy semantic inference occurs.
    """

    if (
        candidate.kind
        not in
        SEMANTIC_MERGE_CANDIDATE_KINDS
    ):
        return None

    if (
        len(
            candidate.candidate_groups
        )
        !=
        1
    ):
        return None

    group = list(
        dict.fromkeys(
            candidate.candidate_groups[
                0
            ]
        )
    )

    if (
        len(
            group
        )
        <
        2
    ):
        return None

    normalized = {
        _normalize_token(
            value
        )
        for value in group
    }

    if (
        len(
            normalized
        )
        !=
        1
    ):
        return None

    token = next(
        iter(
            normalized
        )
    )

    if not token:
        return None

    return group


def _canonical_existing_value(
    *,
    dataframe: pd.DataFrame,
    column: str | None,
    source_values: list[str],
) -> str | None:
    """
    Choose an EXISTING exact value deterministically.

    Order:
    1. highest exact observed frequency;
    2. first appearance in the current prepared DataFrame.

    No value is invented and no formatting is synthesized.
    """

    if (
        column is None
        or
        column not in dataframe.columns
        or
        not source_values
    ):
        return None

    source_set = set(
        source_values
    )

    counts = {
        value:
            0
        for value in
        source_values
    }

    first_position = {
        value:
            float(
                "inf"
            )
        for value in
        source_values
    }

    for (
        position,
        cell,
    ) in enumerate(
        dataframe[
            column
        ].tolist()
    ):
        if (
            not isinstance(
                cell,
                str,
            )
            or
            cell
            not in
            source_set
        ):
            continue

        counts[
            cell
        ] += 1

        if (
            first_position[
                cell
            ]
            ==
            float(
                "inf"
            )
        ):
            first_position[
                cell
            ] = position

    observed = [
        value
        for value in source_values
        if (
            counts[
                value
            ]
            >
            0
        )
    ]

    if not observed:
        return None

    observed.sort(
        key=lambda value: (
            -counts[
                value
            ],
            first_position[
                value
            ],
        )
    )

    return observed[
        0
    ]


def _deterministic_strict_alias_decision(
    *,
    candidate: SemanticReviewCandidate,
    dataframe: pd.DataFrame,
) -> ValidatedSemanticDecision | None:
    """
    Python may PROPOSE a merge when the only differences are
    whitespace and/or letter case.

    The proposal remains non-executable and still requires
    explicit user confirmation before Semantic Cleaning.
    """

    source_values = (
        _strict_normalization_group(
            candidate
        )
    )

    if (
        source_values
        is None
    ):
        return None

    canonical_value = (
        _canonical_existing_value(
            dataframe=
                dataframe,

            column=
                candidate.column,

            source_values=
                source_values,
        )
    )

    if (
        canonical_value
        is None
    ):
        return None

    return (
        ValidatedSemanticDecision(
            issue_id=
                candidate.issue_id,

            dataset_id=
                candidate.dataset_id,

            dataset_filename=
                candidate.dataset_filename,

            column=
                candidate.column,

            kind=
                candidate.kind,

            verdict=
                SemanticVerdict
                .MERGE_VALUES,

            confidence=
                1.0,

            rationale=(
                "Python detected exact category variants that "
                "become identical after conservative whitespace "
                "normalization and Unicode casefold."
            ),

            source_values=
                source_values,

            canonical_value=
                canonical_value,

            user_message=(
                "Variantes de casse ou d'espacement détectées "
                "par Python. Vérifiez puis confirmez la fusion."
            ),

            python_validated=
                True,

            executable=
                False,

            requires_user_confirmation=
                True,

            validation_notes=[
                (
                    "Deterministic strict-normalization proposal."
                ),
                (
                    "No spelling correction, fuzzy matching or "
                    "business inference was used."
                ),
                (
                    "Canonical value is one exact observed value."
                ),
                (
                    "User confirmation remains mandatory."
                ),
                (
                    "Rule: "
                    f"{SEMANTIC_CANONICALIZATION_RULE_VERSION}."
                ),
            ],
        )
    )


def validate_semantic_review_response(
    *,
    raw_response: RawSemanticReviewResponse,
    candidates: list[
        SemanticReviewCandidate
    ],
    dataset_frames: dict[
        str,
        pd.DataFrame,
    ],
) -> list[
    ValidatedSemanticDecision
]:
    """
    Validate every semantic decision against deterministic
    evidence.

    Unknown and duplicate IDs fail the protocol.

    Invalid proposals for known candidates are downgraded to
    ABSTAIN rather than executed.

    No semantic proposal becomes executable here.
    """

    candidate_map = {
        candidate.issue_id:
            candidate
        for candidate
        in candidates
    }

    seen_ids: set[
        str
    ] = set()

    decisions: list[
        ValidatedSemanticDecision
    ] = []

    for raw in (
        raw_response.decisions
    ):
        if (
            raw.issue_id
            in seen_ids
        ):
            raise ValueError(
                "Semantic review returned "
                "a duplicate issue_id: "
                f"{raw.issue_id}"
            )

        seen_ids.add(
            raw.issue_id
        )

        candidate = (
            candidate_map.get(
                raw.issue_id
            )
        )

        if candidate is None:
            raise ValueError(
                "Semantic review invented or "
                "referenced an unknown issue_id: "
                f"{raw.issue_id}"
            )

        dataframe = (
            dataset_frames.get(
                candidate.dataset_id
            )
        )

        if dataframe is None:
            raise KeyError(
                "Missing dataframe during semantic "
                f"validation for {candidate.dataset_id}."
            )

        notes: list[
            str
        ] = []

        source_values = list(
            dict.fromkeys(
                raw.source_values
            )
        )

        canonical_value = (
            raw.canonical_value
        )

        if (
            raw.verdict
            ==
            SemanticVerdict
            .MERGE_VALUES
        ):
            rejection_reasons: list[
                str
            ] = []

            if (
                candidate.kind
                not in
                SEMANTIC_MERGE_CANDIDATE_KINDS
            ):
                rejection_reasons.append(
                    "merge_values is allowed only for "
                    "deterministic category-format variants "
                    "or possible semantic aliases."
                )

            if (
                len(
                    source_values
                )
                <
                2
            ):
                rejection_reasons.append(
                    "merge_values requires at least "
                    "two exact source_values."
                )

            allowed_values = set(
                candidate
                .candidate_values
            )

            invented_values = (
                set(
                    source_values
                )
                -
                allowed_values
            )

            if invented_values:
                rejection_reasons.append(
                    "The model referenced value(s) "
                    "outside the deterministic candidate: "
                    +
                    ", ".join(
                        sorted(
                            invented_values
                        )
                    )
                    +
                    "."
                )

            if (
                canonical_value
                is None
                or
                canonical_value
                not in source_values
            ):
                rejection_reasons.append(
                    "canonical_value must be one "
                    "exact proposed source value."
                )

            observed_values = (
                _observed_exact_values(
                    dataframe=
                        dataframe,
                    column=
                        candidate.column,
                )
            )

            missing_observed = (
                set(
                    source_values
                )
                -
                observed_values
            )

            if missing_observed:
                rejection_reasons.append(
                    "The model referenced value(s) "
                    "not present in the current "
                    "derived dataset: "
                    +
                    ", ".join(
                        sorted(
                            missing_observed
                        )
                    )
                    +
                    "."
                )

            group_sets = [
                set(
                    group
                )
                for group in
                candidate
                .candidate_groups
            ]

            if (
                source_values
                and
                not any(
                    set(
                        source_values
                    ).issubset(
                        group
                    )
                    for group in
                    group_sets
                )
            ):
                rejection_reasons.append(
                    "The proposed merge crosses "
                    "deterministic candidate groups."
                )

            if rejection_reasons:
                decisions.append(
                    _abstain_decision(
                        candidate=
                            candidate,

                        rationale=(
                            "Python rejected the model's "
                            "merge proposal because it did "
                            "not satisfy deterministic "
                            "semantic-cleaning guards."
                        ),

                        user_message=(
                            "Proposition sémantique rejetée "
                            "par les garde-fous Python. "
                            "Conserver pour revue manuelle."
                        ),

                        validation_notes=[
                            (
                                "Original model verdict: "
                                f"{raw.verdict.value}."
                            ),
                            (
                                "Original model confidence: "
                                f"{raw.confidence:.3f}."
                            ),
                            *rejection_reasons,
                        ],
                    )
                )

                continue

            notes.append(
                "Exact alias values verified "
                "against the current derived dataset."
            )

            notes.append(
                "Canonical value is an existing "
                "observed value, not an invention."
            )

        else:
            rejection_reasons: list[
                str
            ] = []

            if source_values:
                allowed_values = set(
                    candidate
                    .candidate_values
                )

                invented_values = (
                    set(
                        source_values
                    )
                    -
                    allowed_values
                )

                if invented_values:
                    rejection_reasons.append(
                        "The model referenced unsupported "
                        "source value(s): "
                        +
                        ", ".join(
                            sorted(
                                invented_values
                            )
                        )
                        +
                        "."
                    )

            if (
                canonical_value
                is not None
            ):
                rejection_reasons.append(
                    "canonical_value is allowed only "
                    "with merge_values."
                )

            if rejection_reasons:
                decisions.append(
                    _abstain_decision(
                        candidate=
                            candidate,

                        rationale=(
                            "Python rejected fields that "
                            "were incompatible with the "
                            "model's semantic verdict."
                        ),

                        user_message=(
                            "Décision sémantique conservée "
                            "pour revue manuelle."
                        ),

                        validation_notes=[
                            (
                                "Original model verdict: "
                                f"{raw.verdict.value}."
                            ),
                            (
                                "Original model confidence: "
                                f"{raw.confidence:.3f}."
                            ),
                            *rejection_reasons,
                        ],
                    )
                )

                continue

        decisions.append(
            ValidatedSemanticDecision(
                issue_id=
                    candidate.issue_id,

                dataset_id=
                    candidate.dataset_id,

                dataset_filename=
                    candidate
                    .dataset_filename,

                column=
                    candidate.column,

                kind=
                    candidate.kind,

                verdict=
                    raw.verdict,

                confidence=
                    raw.confidence,

                rationale=
                    raw.rationale,

                source_values=
                    source_values,

                canonical_value=
                    canonical_value,

                user_message=
                    raw.user_message,

                python_validated=
                    True,

                executable=
                    False,

                requires_user_confirmation=
                    True,

                validation_notes=
                    notes,
            )
        )

    missing_ids = (
        set(
            candidate_map
        )
        -
        seen_ids
    )

    for issue_id in sorted(
        missing_ids
    ):
        candidate = (
            candidate_map[
                issue_id
            ]
        )

        decisions.append(
            _abstain_decision(
                candidate=
                    candidate,

                rationale=(
                    "The local model omitted this "
                    "semantic-review candidate."
                ),

                user_message=(
                    "Aucune décision fiable du modèle. "
                    "Conserver pour revue manuelle."
                ),

                validation_notes=[
                    (
                        "The candidate was omitted "
                        "from the model response and "
                        "was converted to ABSTAIN "
                        "by Python."
                    )
                ],
            )
        )

    decision_map = {
        decision.issue_id:
            decision
        for decision in
        decisions
    }

    return [
        decision_map[
            candidate.issue_id
        ]
        for candidate in
        candidates
    ]


def review_quality_semantics(
    *,
    quality_report: DataQualityReport,
    dataset_frames: dict[
        str,
        pd.DataFrame,
    ],
    model: str = (
        DEFAULT_SEMANTIC_REVIEW_MODEL
    ),
    ollama_chat_url: str = (
        DEFAULT_OLLAMA_CHAT_URL
    ),
    timeout_seconds: float = 90.0,
) -> SemanticReviewReport:
    candidates = (
        build_semantic_review_candidates(
            quality_report=
                quality_report,
            dataset_frames=
                dataset_frames,
        )
    )

    if not candidates:
        return (
            SemanticReviewReport(
                status=
                    "ready",

                model=
                    model,

                candidate_count=
                    0,

                decision_count=
                    0,

                merge_proposal_count=
                    0,

                abstention_count=
                    0,

                decisions=
                    [],

                notes=[
                    (
                        "No deterministic quality "
                        "issue required semantic review."
                    ),
                ],
            )
        )

    validated_decisions: list[
        ValidatedSemanticDecision
    ] = []

    llm_failure_count = 0
    deterministic_proposal_count = 0

    for candidate in candidates:
        dataframe = (
            dataset_frames.get(
                candidate.dataset_id
            )
        )

        if dataframe is None:
            raise KeyError(
                "Missing dataframe during semantic "
                f"review for {candidate.dataset_id}."
            )

        deterministic_decision = (
            _deterministic_strict_alias_decision(
                candidate=
                    candidate,
                dataframe=
                    dataframe,
            )
        )

        if (
            deterministic_decision
            is not None
        ):
            deterministic_proposal_count += 1

            validated_decisions.append(
                deterministic_decision
            )

            continue

        try:
            raw_decision = (
                _ollama_chat_one(
                    model=
                        model,

                    candidate=
                        candidate,

                    ollama_chat_url=
                        ollama_chat_url,

                    timeout_seconds=
                        timeout_seconds,
                )
            )

            validated = (
                validate_semantic_review_response(
                    raw_response=
                        RawSemanticReviewResponse(
                            decisions=[
                                raw_decision
                            ]
                        ),

                    candidates=[
                        candidate
                    ],

                    dataset_frames=
                        dataset_frames,
                )
            )

            validated_decisions.extend(
                validated
            )

        except (
            RuntimeError,
            ValueError,
        ):
            llm_failure_count += 1

            validated_decisions.append(
                _abstain_decision(
                    candidate=
                        candidate,

                    rationale=(
                        "The local model did not "
                        "produce a usable decision "
                        "for this candidate."
                    ),

                    user_message=(
                        "Aucune décision fiable du modèle. "
                        "Conserver pour revue manuelle."
                    ),

                    validation_notes=[
                        (
                            "Candidate-level semantic review "
                            "failed; internal model error "
                            "details were suppressed."
                        )
                    ],
                )
            )

    merge_proposal_count = sum(
        decision.verdict
        ==
        SemanticVerdict
        .MERGE_VALUES
        for decision
        in validated_decisions
    )

    abstention_count = sum(
        decision.verdict
        ==
        SemanticVerdict
        .ABSTAIN
        for decision
        in validated_decisions
    )

    return (
        SemanticReviewReport(
            status=
                "ready",

            model=
                model,

            candidate_count=
                len(
                    candidates
                ),

            decision_count=
                len(
                    validated_decisions
                ),

            merge_proposal_count=
                merge_proposal_count,

            abstention_count=
                abstention_count,

            decisions=
                validated_decisions,

            notes=[
                (
                    "Category-format variants and semantic-alias "
                    "quality issues are split into independent "
                    "deterministic groups before semantic review."
                ),
                (
                    f"{deterministic_proposal_count} strict "
                    "case/whitespace alias group(s) were "
                    "proposed directly by Python."
                ),
                (
                    "Strict deterministic proposals remain "
                    "non-executable until user confirmation."
                ),
                (
                    "The local LLM received one remaining "
                    "ambiguous quality candidate at a time "
                    "with limited evidence."
                ),
                (
                    "Alias candidates sent to the model use "
                    "a specialized prompt requiring exact "
                    "source_values when merge_values is proposed."
                ),
                (
                    "Every model-returned decision was validated "
                    "against deterministic dataset evidence."
                ),
                (
                    f"{llm_failure_count} candidate-level "
                    "model response(s) were safely "
                    "downgraded to ABSTAIN."
                ),
                (
                    "No semantic proposal is executable in "
                    f"{SEMANTIC_REVIEW_RULE_VERSION}."
                ),
                (
                    "Canonicalization component: "
                    f"{SEMANTIC_CANONICALIZATION_RULE_VERSION}."
                ),
            ],
        )
    )

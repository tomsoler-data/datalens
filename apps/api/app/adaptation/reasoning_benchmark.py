from __future__ import annotations

import hashlib
import json
import math

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Literal,
    Mapping,
)


ADAPTED_REASONING_BENCHMARK_RULE_VERSION = (
    "adapted_semantic_reasoning_benchmark_v0.1"
)

ADAPTED_REASONING_CASE_ARTIFACT_RULE_VERSION = (
    "adapted_semantic_reasoning_case_artifact_v0.1"
)

ADAPTED_REASONING_SCORING_RULE_VERSION = (
    "adapted_semantic_reasoning_scoring_v0.1"
)


BENCHMARK_ID = (
    "adaptation:semantic_reasoning:hotel:v0.1"
)

BENCHMARK_VERSION = (
    "datalens_adapted_semantic_reasoning_hotel_v0.1"
)


ReasoningRelation = Literal[
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
]


ALLOWED_RELATIONS: tuple[
    ReasoningRelation,
    ...,
] = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
)


EXPECTED_CASE_COUNT = 19


EXPECTED_LABEL_COUNTS = MappingProxyType(
    {
        "same_metric_different_state":
            5,

        "same_process_different_stage":
            1,

        "related_distinct_metric":
            13,
    }
)


PROMPT_TEMPLATE = (
    "Classify the semantic relationship between two "
    "analytical metrics.\n"
    "Left metric: {left_column}\n"
    "Right metric: {right_column}\n"
    "Select the single best interpretation. "
    "Respond with the interpretation itself."
)


_CANDIDATE_SURFACES = {
    "same_metric_different_state":
        (
            "These metrics represent the same underlying "
            "measure in different states."
        ),

    "same_process_different_stage":
        (
            "These metrics represent different stages "
            "of the same operational process."
        ),

    "related_distinct_metric":
        (
            "These metrics are related, but they "
            "represent distinct measures."
        ),
}


CANDIDATE_SURFACES: Mapping[
    str,
    str,
] = MappingProxyType(
    _CANDIDATE_SURFACES
)


@dataclass(
    frozen=True,
)
class AdaptedReasoningCase:
    case_id: str

    left_column: str

    right_column: str

    expected_relation: ReasoningRelation


def canonical_json_bytes(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )


def sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def prompt_template_sha256(
) -> str:
    return sha256_bytes(
        PROMPT_TEMPLATE.encode(
            "utf-8"
        )
    )


def candidate_surface_payload(
) -> dict[
    str,
    str,
]:
    return {
        relation:
            CANDIDATE_SURFACES[
                relation
            ]

        for relation
        in ALLOWED_RELATIONS
    }


def candidate_surface_sha256(
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            candidate_surface_payload()
        )
    )


def render_reasoning_prompt(
    case: AdaptedReasoningCase,
) -> str:
    _validate_case(
        case
    )

    return PROMPT_TEMPLATE.format(
        left_column=
            case.left_column,

        right_column=
            case.right_column,
    )


def candidate_continuation(
    relation: ReasoningRelation,
) -> str:
    if relation not in ALLOWED_RELATIONS:
        raise ValueError(
            (
                "Unsupported reasoning relation: "
                f"{relation}"
            )
        )

    return CANDIDATE_SURFACES[
        relation
    ]


def _validate_text(
    *,
    value: str,
    field_name: str,
) -> None:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field_name} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{field_name} must not be empty."
        )


def _validate_case(
    case: AdaptedReasoningCase,
) -> None:
    _validate_text(
        value=
            case.case_id,

        field_name=
            "case_id",
    )

    _validate_text(
        value=
            case.left_column,

        field_name=
            "left_column",
    )

    _validate_text(
        value=
            case.right_column,

        field_name=
            "right_column",
    )

    if (
        case.left_column
        ==
        case.right_column
    ):
        raise ValueError(
            "Reasoning pair columns must be distinct."
        )

    if (
        case.expected_relation
        not in
        ALLOWED_RELATIONS
    ):
        raise ValueError(
            (
                "Unsupported expected relation: "
                f"{case.expected_relation}"
            )
        )


def case_to_record(
    case: AdaptedReasoningCase,
) -> dict[
    str,
    str,
]:
    _validate_case(
        case
    )

    return {
        "case_id":
            case.case_id,

        "expected_relation":
            case.expected_relation,

        "left_column":
            case.left_column,

        "right_column":
            case.right_column,
    }


def label_distribution(
    cases: tuple[
        AdaptedReasoningCase,
        ...,
    ],
) -> dict[
    str,
    int,
]:
    counts = Counter(
        case.expected_relation
        for case
        in cases
    )

    return {
        relation:
            counts[
                relation
            ]

        for relation
        in ALLOWED_RELATIONS
    }


def _validate_case_collection(
    cases: tuple[
        AdaptedReasoningCase,
        ...,
    ],
) -> None:
    if (
        len(
            cases
        )
        !=
        EXPECTED_CASE_COUNT
    ):
        raise ValueError(
            (
                "Expected exactly "
                f"{EXPECTED_CASE_COUNT} reasoning cases."
            )
        )

    case_ids = [
        case.case_id
        for case
        in cases
    ]

    if (
        len(
            set(
                case_ids
            )
        )
        !=
        len(
            case_ids
        )
    ):
        raise ValueError(
            "Reasoning case IDs must be unique."
        )

    for case in cases:
        _validate_case(
            case
        )

    observed = label_distribution(
        cases
    )

    expected = dict(
        EXPECTED_LABEL_COUNTS
    )

    if observed != expected:
        raise ValueError(
            (
                "Frozen reasoning label distribution "
                "does not match the benchmark contract. "
                f"Observed={observed} "
                f"Expected={expected}"
            )
        )


def load_frozen_reasoning_cases(
    *,
    path: Path,
) -> tuple[
    AdaptedReasoningCase,
    ...,
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
            "Reasoning case artifact must be a JSON object."
        )

    required_top_level = {
        "benchmark_id",
        "benchmark_version",
        "candidate_surface_sha256",
        "candidate_surfaces",
        "case_artifact_rule_version",
        "case_source",
        "cases",
        "prompt_template_sha256",
    }

    if (
        set(
            payload
        )
        !=
        required_top_level
    ):
        raise ValueError(
            (
                "Unexpected reasoning case artifact schema. "
                f"Keys={sorted(payload)}"
            )
        )

    if (
        payload[
            "benchmark_id"
        ]
        !=
        BENCHMARK_ID
    ):
        raise ValueError(
            "Reasoning benchmark ID mismatch."
        )

    if (
        payload[
            "benchmark_version"
        ]
        !=
        BENCHMARK_VERSION
    ):
        raise ValueError(
            "Reasoning benchmark version mismatch."
        )

    if (
        payload[
            "case_artifact_rule_version"
        ]
        !=
        ADAPTED_REASONING_CASE_ARTIFACT_RULE_VERSION
    ):
        raise ValueError(
            "Reasoning case artifact rule mismatch."
        )

    if (
        payload[
            "prompt_template_sha256"
        ]
        !=
        prompt_template_sha256()
    ):
        raise ValueError(
            "Reasoning prompt template hash mismatch."
        )

    if (
        payload[
            "candidate_surface_sha256"
        ]
        !=
        candidate_surface_sha256()
    ):
        raise ValueError(
            "Reasoning candidate surface hash mismatch."
        )

    if (
        payload[
            "candidate_surfaces"
        ]
        !=
        candidate_surface_payload()
    ):
        raise ValueError(
            "Reasoning candidate surfaces mismatch."
        )

    source = payload[
        "case_source"
    ]

    if not isinstance(
        source,
        dict,
    ):
        raise TypeError(
            "Reasoning case source must be an object."
        )

    if (
        source.get(
            "benchmark_id"
        )
        !=
        "semantic:hotel_operations:holdout:v0.1"
    ):
        raise ValueError(
            "Unexpected source holdout benchmark."
        )

    if (
        source.get(
            "benchmark_version"
        )
        !=
        "hotel_operations_semantic_holdout_v0.1"
    ):
        raise ValueError(
            "Unexpected source holdout version."
        )

    if (
        source.get(
            "holdout_frozen"
        )
        is not True
    ):
        raise ValueError(
            "Source Hotel holdout must be frozen."
        )

    raw_cases = payload[
        "cases"
    ]

    if not isinstance(
        raw_cases,
        list,
    ):
        raise TypeError(
            "Reasoning cases must be a list."
        )

    cases = []

    expected_case_keys = {
        "case_id",
        "expected_relation",
        "left_column",
        "right_column",
    }

    for record in raw_cases:
        if not isinstance(
            record,
            dict,
        ):
            raise TypeError(
                "Reasoning case must be an object."
            )

        if (
            set(
                record
            )
            !=
            expected_case_keys
        ):
            raise ValueError(
                (
                    "Unexpected reasoning case schema. "
                    f"Keys={sorted(record)}"
                )
            )

        relation = record[
            "expected_relation"
        ]

        if relation not in ALLOWED_RELATIONS:
            raise ValueError(
                (
                    "Unsupported frozen relation: "
                    f"{relation}"
                )
            )

        cases.append(
            AdaptedReasoningCase(
                case_id=
                    str(
                        record[
                            "case_id"
                        ]
                    ),

                left_column=
                    str(
                        record[
                            "left_column"
                        ]
                    ),

                right_column=
                    str(
                        record[
                            "right_column"
                        ]
                    ),

                expected_relation=
                    relation,
            )
        )

    result = tuple(
        cases
    )

    _validate_case_collection(
        result
    )

    return result


def select_relation_from_scores(
    *,
    scores: Mapping[
        str,
        float,
    ],
) -> ReasoningRelation:
    if (
        set(
            scores
        )
        !=
        set(
            ALLOWED_RELATIONS
        )
    ):
        raise ValueError(
            (
                "Candidate score keys must exactly match "
                "the frozen reasoning relations."
            )
        )

    normalized = {}

    for relation in ALLOWED_RELATIONS:
        score = float(
            scores[
                relation
            ]
        )

        if not math.isfinite(
            score
        ):
            raise ValueError(
                (
                    "Candidate score must be finite: "
                    f"{relation}={score}"
                )
            )

        normalized[
            relation
        ] = score

    best_score = max(
        normalized.values()
    )

    winners = [
        relation
        for relation, score
        in normalized.items()
        if score == best_score
    ]

    if len(
        winners
    ) != 1:
        raise ValueError(
            (
                "Candidate scoring tie is forbidden "
                "and must fail closed."
            )
        )

    return winners[
        0
    ]


def prediction_is_correct(
    *,
    case: AdaptedReasoningCase,
    scores: Mapping[
        str,
        float,
    ],
) -> bool:
    predicted = (
        select_relation_from_scores(
            scores=
                scores,
        )
    )

    return (
        predicted
        ==
        case.expected_relation
    )


def classification_accuracy(
    *,
    cases: tuple[
        AdaptedReasoningCase,
        ...,
    ],
    predictions: Mapping[
        str,
        str,
    ],
) -> float:
    _validate_case_collection(
        cases
    )

    expected_ids = {
        case.case_id
        for case
        in cases
    }

    if (
        set(
            predictions
        )
        !=
        expected_ids
    ):
        raise ValueError(
            "Prediction IDs must exactly match case IDs."
        )

    correct = 0

    for case in cases:
        predicted = predictions[
            case.case_id
        ]

        if predicted not in ALLOWED_RELATIONS:
            raise ValueError(
                (
                    "Unsupported predicted relation: "
                    f"{predicted}"
                )
            )

        if (
            predicted
            ==
            case.expected_relation
        ):
            correct += 1

    return round(
        (
            correct
            /
            len(
                cases
            )
        ),
        6,
    )

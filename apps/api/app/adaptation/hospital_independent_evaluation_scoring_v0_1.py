from __future__ import annotations


import re


from collections.abc import (
    Mapping,
    Sequence,
)

from typing import Any


HOSPITAL_INDEPENDENT_EVALUATION_SCORING_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_scoring_v0.1"
)


SCORING_FOUNDATION_COMMIT = (
    "66e151caf93a7d22b201347ca1862906e0e8cc5d"
)


EXPECTED_PROTOCOL_SHA256 = (
    "0e958d67a6294f8666485a6274b1eee2"
    "1b9f06fe2e300ba93241a7ffaba3127d"
)


EXPECTED_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


EXPECTED_CASE_COUNT = 30
EXPECTED_CASES_PER_RELATION = 6
DECIMAL_PLACES = 6


ADAPTED_ACCURACY_MINIMUM = 0.700000
ADAPTED_MACRO_ACCURACY_MINIMUM = 0.700000
PER_RELATION_ACCURACY_MINIMUM = 0.500000
UNCERTAIN_ACCURACY_MINIMUM = 0.666667
STRICT_JSON_VALIDITY_RATE_REQUIRED = 1.000000

ACCURACY_DELTA_MINIMUM = 0.000000
MACRO_ACCURACY_DELTA_MINIMUM = 0.000000

CORRECT_CASE_COUNT_DELTA_MINIMUM = 1
EQUIVALENT_ACCURACY_DELTA_MINIMUM = 0.033333


def _round_metric(
    value: float,
) -> float:

    return round(
        float(
            value
        ),
        DECIMAL_PLACES,
    )


def _require_sequence(
    *,
    value: Any,
    label: str,
) -> Sequence[Any]:

    if (
        not isinstance(
            value,
            Sequence,
        )
        or
        isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        )
    ):

        raise TypeError(
            f"{label} must be a sequence."
        )

    return value


def _require_case_id(
    value: Any,
) -> str:

    if (
        not isinstance(
            value,
            str,
        )
        or
        not value.strip()
    ):

        raise TypeError(
            "case_id must be a non-empty string."
        )

    return value


def _require_prompt_sha256(
    value: Any,
) -> str:

    if (
        not isinstance(
            value,
            str,
        )
        or
        re.fullmatch(
            r"[0-9a-f]{64}",
            value,
        )
        is None
    ):

        raise ValueError(
            "prompt_sha256 must be a lowercase SHA256."
        )

    return value


def normalize_gold_records(
    gold_records: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> tuple[
    dict[
        str,
        str,
    ],
    ...,
]:

    sequence = _require_sequence(
        value=
            gold_records,

        label=
            "gold_records",
    )

    if (
        len(
            sequence
        )
        !=
        EXPECTED_CASE_COUNT
    ):

        raise RuntimeError(
            (
                "Expected exactly "
                f"{EXPECTED_CASE_COUNT} gold records."
            )
        )

    normalized = []

    for record in sequence:

        if not isinstance(
            record,
            Mapping,
        ):

            raise TypeError(
                "Gold record must be a mapping."
            )

        if (
            set(
                record
            )
            !=
            {
                "case_id",
                "gold_relation",
            }
        ):

            raise RuntimeError(
                (
                    "Gold scoring boundary changed. "
                    "Expected exactly case_id + gold_relation."
                )
            )

        case_id = _require_case_id(
            record[
                "case_id"
            ]
        )

        relation = record[
            "gold_relation"
        ]

        if relation not in EXPECTED_RELATIONS:

            raise ValueError(
                (
                    "Unexpected gold relation: "
                    f"{relation!r}"
                )
            )

        normalized.append(
            {
                "case_id":
                    case_id,

                "gold_relation":
                    relation,
            }
        )

    case_ids = [
        record[
            "case_id"
        ]

        for record
        in normalized
    ]

    if (
        len(
            case_ids
        )
        !=
        len(
            set(
                case_ids
            )
        )
    ):

        raise RuntimeError(
            "Gold case IDs are not unique."
        )

    relation_counts = {
        relation:
            0

        for relation
        in EXPECTED_RELATIONS
    }

    for record in normalized:

        relation_counts[
            record[
                "gold_relation"
            ]
        ] += 1

    for relation in EXPECTED_RELATIONS:

        if (
            relation_counts[
                relation
            ]
            !=
            EXPECTED_CASES_PER_RELATION
        ):

            raise RuntimeError(
                (
                    "Gold benchmark is no longer balanced. "
                    f"relation={relation!r}, "
                    "expected="
                    f"{EXPECTED_CASES_PER_RELATION}, "
                    "actual="
                    f"{relation_counts[relation]}"
                )
            )

    return tuple(
        normalized
    )


def _normalize_model_results(
    *,
    model_label: str,
    results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    gold_records: Sequence[
        Mapping[
            str,
            str,
        ]
    ],
) -> tuple[
    dict[
        str,
        Any,
    ],
    ...,
]:

    if model_label not in {
        "base",
        "adapted",
    }:

        raise ValueError(
            "model_label must be base or adapted."
        )

    sequence = _require_sequence(
        value=
            results,

        label=
            (
                f"{model_label}_results"
            ),
    )

    if (
        len(
            sequence
        )
        !=
        EXPECTED_CASE_COUNT
    ):

        raise RuntimeError(
            (
                "Expected exactly "
                f"{EXPECTED_CASE_COUNT} "
                f"{model_label} results."
            )
        )

    required_keys = {
        "case_id",
        "model_label",
        "prompt_sha256",
        "strict_json_valid",
        "relation",
    }

    by_case_id = {}

    for record in sequence:

        if not isinstance(
            record,
            Mapping,
        ):

            raise TypeError(
                (
                    f"{model_label} result "
                    "must be a mapping."
                )
            )

        if not required_keys.issubset(
            set(
                record
            )
        ):

            missing = sorted(
                required_keys
                -
                set(
                    record
                )
            )

            raise RuntimeError(
                (
                    f"{model_label} result missing "
                    f"required keys: {missing}"
                )
            )

        case_id = _require_case_id(
            record[
                "case_id"
            ]
        )

        if case_id in by_case_id:

            raise RuntimeError(
                (
                    f"Duplicate {model_label} "
                    f"case_id: {case_id}"
                )
            )

        if (
            record[
                "model_label"
            ]
            !=
            model_label
        ):

            raise RuntimeError(
                (
                    "Model label mismatch. "
                    f"Expected={model_label!r}, "
                    "actual="
                    f"{record['model_label']!r}"
                )
            )

        prompt_sha256 = (
            _require_prompt_sha256(
                record[
                    "prompt_sha256"
                ]
            )
        )

        strict_json_valid = record[
            "strict_json_valid"
        ]

        if not isinstance(
            strict_json_valid,
            bool,
        ):

            raise TypeError(
                "strict_json_valid must be boolean."
            )

        relation = record[
            "relation"
        ]

        if strict_json_valid:

            if relation not in EXPECTED_RELATIONS:

                raise RuntimeError(
                    (
                        "Strict-valid result has "
                        "an invalid relation."
                    )
                )

        else:

            if relation is not None:

                raise RuntimeError(
                    (
                        "Strict-invalid result must "
                        "have relation=None."
                    )
                )

        by_case_id[
            case_id
        ] = {
            "case_id":
                case_id,

            "model_label":
                model_label,

            "prompt_sha256":
                prompt_sha256,

            "strict_json_valid":
                strict_json_valid,

            "relation":
                relation,
        }

    expected_case_ids = {
        record[
            "case_id"
        ]

        for record
        in gold_records
    }

    actual_case_ids = set(
        by_case_id
    )

    if actual_case_ids != expected_case_ids:

        missing = sorted(
            expected_case_ids
            -
            actual_case_ids
        )

        extra = sorted(
            actual_case_ids
            -
            expected_case_ids
        )

        raise RuntimeError(
            (
                f"{model_label} case identity mismatch. "
                f"missing={missing}, extra={extra}"
            )
        )

    return tuple(
        by_case_id[
            gold[
                "case_id"
            ]
        ]

        for gold
        in gold_records
    )


def _validate_prompt_identity(
    *,
    base_results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    adapted_results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:

    for (
        base,
        adapted,
    ) in zip(
        base_results,
        adapted_results,
        strict=True,
    ):

        if (
            base[
                "case_id"
            ]
            !=
            adapted[
                "case_id"
            ]
        ):

            raise RuntimeError(
                "Base/Adapted case order changed."
            )

        if (
            base[
                "prompt_sha256"
            ]
            !=
            adapted[
                "prompt_sha256"
            ]
        ):

            raise RuntimeError(
                (
                    "Base/Adapted prompt identity "
                    "changed."
                )
            )


def _score_normalized_model(
    *,
    model_label: str,
    results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    gold_records: Sequence[
        Mapping[
            str,
            str,
        ]
    ],
) -> dict[
    str,
    Any,
]:

    correct_count = 0
    strict_json_valid_count = 0

    relation_total_count = {
        relation:
            0

        for relation
        in EXPECTED_RELATIONS
    }

    relation_correct_count = {
        relation:
            0

        for relation
        in EXPECTED_RELATIONS
    }

    for (
        result,
        gold,
    ) in zip(
        results,
        gold_records,
        strict=True,
    ):

        gold_relation = gold[
            "gold_relation"
        ]

        relation_total_count[
            gold_relation
        ] += 1

        if result[
            "strict_json_valid"
        ]:

            strict_json_valid_count += 1

            if (
                result[
                    "relation"
                ]
                ==
                gold_relation
            ):

                correct_count += 1

                relation_correct_count[
                    gold_relation
                ] += 1

    for relation in EXPECTED_RELATIONS:

        if (
            relation_total_count[
                relation
            ]
            !=
            EXPECTED_CASES_PER_RELATION
        ):

            raise RuntimeError(
                (
                    "Scoring denominator changed: "
                    f"{relation}"
                )
            )

    raw_accuracy = (
        correct_count
        /
        EXPECTED_CASE_COUNT
    )

    raw_strict_json_rate = (
        strict_json_valid_count
        /
        EXPECTED_CASE_COUNT
    )

    raw_per_relation_accuracy = {
        relation:
            (
                relation_correct_count[
                    relation
                ]
                /
                relation_total_count[
                    relation
                ]
            )

        for relation
        in EXPECTED_RELATIONS
    }

    raw_macro_accuracy = (
        sum(
            raw_per_relation_accuracy.values()
        )
        /
        len(
            EXPECTED_RELATIONS
        )
    )

    per_relation_accuracy = {
        relation:
            _round_metric(
                raw_per_relation_accuracy[
                    relation
                ]
            )

        for relation
        in EXPECTED_RELATIONS
    }

    return {
        "model_label":
            model_label,

        "case_count":
            EXPECTED_CASE_COUNT,

        "strict_json_valid_count":
            strict_json_valid_count,

        "strict_json_validity_rate":
            _round_metric(
                raw_strict_json_rate
            ),

        "correct_count":
            correct_count,

        "accuracy":
            _round_metric(
                raw_accuracy
            ),

        "per_relation_total_count":
            dict(
                relation_total_count
            ),

        "per_relation_correct_count":
            dict(
                relation_correct_count
            ),

        "per_relation_accuracy":
            per_relation_accuracy,

        "macro_accuracy":
            _round_metric(
                raw_macro_accuracy
            ),

        "uncertain_accuracy":
            per_relation_accuracy[
                "uncertain"
            ],
    }


def _macro_delta_from_counts(
    *,
    base_score: Mapping[
        str,
        Any,
    ],
    adapted_score: Mapping[
        str,
        Any,
    ],
) -> float:

    raw_delta = (
        sum(
            (
                adapted_score[
                    "per_relation_correct_count"
                ][
                    relation
                ]
                -
                base_score[
                    "per_relation_correct_count"
                ][
                    relation
                ]
            )
            /
            EXPECTED_CASES_PER_RELATION

            for relation
            in EXPECTED_RELATIONS
        )
        /
        len(
            EXPECTED_RELATIONS
        )
    )

    return _round_metric(
        raw_delta
    )


def score_official_comparison(
    *,
    gold_records: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    base_results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    adapted_results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    Any,
]:

    gold = normalize_gold_records(
        gold_records
    )

    base = _normalize_model_results(
        model_label=
            "base",

        results=
            base_results,

        gold_records=
            gold,
    )

    adapted = _normalize_model_results(
        model_label=
            "adapted",

        results=
            adapted_results,

        gold_records=
            gold,
    )

    _validate_prompt_identity(
        base_results=
            base,

        adapted_results=
            adapted,
    )

    base_score = _score_normalized_model(
        model_label=
            "base",

        results=
            base,

        gold_records=
            gold,
    )

    adapted_score = _score_normalized_model(
        model_label=
            "adapted",

        results=
            adapted,

        gold_records=
            gold,
    )

    correct_count_delta = (
        adapted_score[
            "correct_count"
        ]
        -
        base_score[
            "correct_count"
        ]
    )

    accuracy_delta = _round_metric(
        correct_count_delta
        /
        EXPECTED_CASE_COUNT
    )

    macro_accuracy_delta = (
        _macro_delta_from_counts(
            base_score=
                base_score,

            adapted_score=
                adapted_score,
        )
    )

    per_relation_gate = {
        relation:
            (
                adapted_score[
                    "per_relation_accuracy"
                ][
                    relation
                ]
                >=
                PER_RELATION_ACCURACY_MINIMUM
            )

        for relation
        in EXPECTED_RELATIONS
    }

    absolute_gates = {
        "accuracy":
            (
                adapted_score[
                    "accuracy"
                ]
                >=
                ADAPTED_ACCURACY_MINIMUM
            ),

        "macro_accuracy":
            (
                adapted_score[
                    "macro_accuracy"
                ]
                >=
                ADAPTED_MACRO_ACCURACY_MINIMUM
            ),

        "per_relation_accuracy":
            per_relation_gate,

        "uncertain_accuracy":
            (
                adapted_score[
                    "uncertain_accuracy"
                ]
                >=
                UNCERTAIN_ACCURACY_MINIMUM
            ),

        "strict_json_validity_rate":
            (
                adapted_score[
                    "strict_json_validity_rate"
                ]
                ==
                STRICT_JSON_VALIDITY_RATE_REQUIRED
            ),
    }

    all_absolute_gates_pass = (
        absolute_gates[
            "accuracy"
        ]
        and
        absolute_gates[
            "macro_accuracy"
        ]
        and
        all(
            per_relation_gate.values()
        )
        and
        absolute_gates[
            "uncertain_accuracy"
        ]
        and
        absolute_gates[
            "strict_json_validity_rate"
        ]
    )

    non_regression_gates = {
        "accuracy_delta":
            (
                accuracy_delta
                >=
                ACCURACY_DELTA_MINIMUM
            ),

        "macro_accuracy_delta":
            (
                macro_accuracy_delta
                >=
                MACRO_ACCURACY_DELTA_MINIMUM
            ),
    }

    all_non_regression_gates_pass = all(
        non_regression_gates.values()
    )

    promotion_signal_gates = {
        "correct_case_count_delta":
            (
                correct_count_delta
                >=
                CORRECT_CASE_COUNT_DELTA_MINIMUM
            ),

        "equivalent_accuracy_delta":
            (
                accuracy_delta
                >=
                EQUIVALENT_ACCURACY_DELTA_MINIMUM
            ),
    }

    promotion_signal_pass = all(
        promotion_signal_gates.values()
    )

    promotion_eligible = (
        all_absolute_gates_pass
        and
        all_non_regression_gates_pass
        and
        promotion_signal_pass
    )

    return {
        "rule_version":
            HOSPITAL_INDEPENDENT_EVALUATION_SCORING_RULE_VERSION,

        "case_count":
            EXPECTED_CASE_COUNT,

        "decimal_places":
            DECIMAL_PLACES,

        "base":
            base_score,

        "adapted":
            adapted_score,

        "comparison":
            {
                "accuracy_delta":
                    accuracy_delta,

                "macro_accuracy_delta":
                    macro_accuracy_delta,

                "correct_case_count_delta":
                    correct_count_delta,
            },

        "gates":
            {
                "adapted_absolute":
                    absolute_gates,

                "all_absolute_gates_pass":
                    all_absolute_gates_pass,

                "non_regression":
                    non_regression_gates,

                "all_non_regression_gates_pass":
                    all_non_regression_gates_pass,

                "promotion_signal":
                    promotion_signal_gates,

                "promotion_signal_pass":
                    promotion_signal_pass,
            },

        "promotion_eligible":
            promotion_eligible,
    }

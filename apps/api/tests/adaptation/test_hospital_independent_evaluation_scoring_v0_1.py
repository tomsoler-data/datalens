from __future__ import annotations


import hashlib


from pathlib import Path


from app.adaptation import (
    hospital_independent_evaluation_preflight_v0_1
    as preflight
)

from app.adaptation import (
    hospital_independent_evaluation_scoring_v0_1
    as scoring
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

SCORER_PATH = (
    ROOT
    /
    "app"
    /
    "adaptation"
    /
    "hospital_independent_evaluation_scoring_v0_1.py"
)


def synthetic_gold(
) -> tuple[
    dict,
    ...,
]:

    records = []

    for relation in scoring.EXPECTED_RELATIONS:

        for index in range(
            scoring.EXPECTED_CASES_PER_RELATION
        ):

            records.append(
                {
                    "case_id":
                        (
                            f"{relation}"
                            f"::{index}"
                        ),

                    "gold_relation":
                        relation,
                }
            )

    return tuple(
        records
    )


def wrong_relation(
    relation: str,
) -> str:

    position = (
        scoring.EXPECTED_RELATIONS
        .index(
            relation
        )
    )

    return scoring.EXPECTED_RELATIONS[
        (
            position
            +
            1
        )
        %
        len(
            scoring.EXPECTED_RELATIONS
        )
    ]


def synthetic_results(
    *,
    model_label: str,
    correct_counts: dict[
        str,
        int,
    ],
    invalid_case_ids: frozenset[
        str
    ] = frozenset(),
) -> tuple[
    dict,
    ...,
]:

    records = []

    for gold in synthetic_gold():

        case_id = gold[
            "case_id"
        ]

        relation = gold[
            "gold_relation"
        ]

        relation_index = int(
            case_id.rsplit(
                "::",
                1,
            )[
                1
            ]
        )

        prompt_sha = hashlib.sha256(
            (
                "synthetic-prompt::"
                f"{case_id}"
            )
            .encode(
                "utf-8"
            )
        ).hexdigest()

        if case_id in invalid_case_ids:

            strict_json_valid = False
            predicted_relation = None

        else:

            strict_json_valid = True

            if (
                relation_index
                <
                correct_counts[
                    relation
                ]
            ):

                predicted_relation = relation

            else:

                predicted_relation = wrong_relation(
                    relation
                )

        records.append(
            {
                "case_id":
                    case_id,

                "model_label":
                    model_label,

                "prompt_sha256":
                    prompt_sha,

                "strict_json_valid":
                    strict_json_valid,

                "relation":
                    predicted_relation,

                "reason":
                    (
                        "Synthetic deterministic reason "
                        "used only for scorer testing."
                    ),
            }
        )

    return tuple(
        records
    )


def counts(
    *,
    default: int,
    **overrides: int,
) -> dict[
    str,
    int,
]:

    result = {
        relation:
            default

        for relation
        in scoring.EXPECTED_RELATIONS
    }

    result.update(
        overrides
    )

    return result


def test_frozen_scoring_authorities(
) -> None:

    assert (
        scoring.SCORING_FOUNDATION_COMMIT
        ==
        "66e151caf93a7d22b201347ca1862906e0e8cc5d"
    )

    assert (
        scoring.EXPECTED_PROTOCOL_SHA256
        ==
        "0e958d67a6294f8666485a6274b1eee21b9f06fe2e300ba93241a7ffaba3127d"
    )

    assert (
        scoring.EXPECTED_CASE_COUNT
        ==
        30
    )

    assert (
        scoring.EXPECTED_CASES_PER_RELATION
        ==
        6
    )

    assert (
        scoring.DECIMAL_PLACES
        ==
        6
    )


def test_exact_thresholds_pass_with_plus_one_correct_case(
) -> None:

    gold = synthetic_gold()

    base = synthetic_results(
        model_label=
            "base",

        correct_counts=
            counts(
                default=4
            ),
    )

    adapted_counts = counts(
        default=4
    )

    adapted_counts[
        "same_metric_different_state"
    ] = 5

    adapted = synthetic_results(
        model_label=
            "adapted",

        correct_counts=
            adapted_counts,
    )

    result = scoring.score_official_comparison(
        gold_records=
            gold,

        base_results=
            base,

        adapted_results=
            adapted,
    )

    assert (
        result[
            "base"
        ][
            "correct_count"
        ]
        ==
        20
    )

    assert (
        result[
            "adapted"
        ][
            "correct_count"
        ]
        ==
        21
    )

    assert (
        result[
            "adapted"
        ][
            "accuracy"
        ]
        ==
        0.7
    )

    assert (
        result[
            "adapted"
        ][
            "macro_accuracy"
        ]
        ==
        0.7
    )

    assert (
        result[
            "adapted"
        ][
            "uncertain_accuracy"
        ]
        ==
        0.666667
    )

    assert (
        result[
            "comparison"
        ][
            "accuracy_delta"
        ]
        ==
        0.033333
    )

    assert (
        result[
            "comparison"
        ][
            "macro_accuracy_delta"
        ]
        ==
        0.033333
    )

    assert (
        result[
            "comparison"
        ][
            "correct_case_count_delta"
        ]
        ==
        1
    )

    assert (
        result[
            "gates"
        ][
            "all_absolute_gates_pass"
        ]
        is True
    )

    assert (
        result[
            "gates"
        ][
            "all_non_regression_gates_pass"
        ]
        is True
    )

    assert (
        result[
            "gates"
        ][
            "promotion_signal_pass"
        ]
        is True
    )

    assert (
        result[
            "promotion_eligible"
        ]
        is True
    )


def test_four_of_six_uncertain_rounds_to_gate(
) -> None:

    base = synthetic_results(
        model_label=
            "base",

        correct_counts=
            counts(
                default=4
            ),
    )

    adapted_counts = counts(
        default=4
    )

    adapted_counts[
        "same_metric_different_state"
    ] = 5

    adapted = synthetic_results(
        model_label=
            "adapted",

        correct_counts=
            adapted_counts,
    )

    result = scoring.score_official_comparison(
        gold_records=
            synthetic_gold(),

        base_results=
            base,

        adapted_results=
            adapted,
    )

    assert (
        result[
            "adapted"
        ][
            "uncertain_accuracy"
        ]
        ==
        0.666667
    )

    assert (
        result[
            "gates"
        ][
            "adapted_absolute"
        ][
            "uncertain_accuracy"
        ]
        is True
    )


def test_strict_invalid_output_counts_incorrect_and_fails_gate(
) -> None:

    gold = synthetic_gold()

    base = synthetic_results(
        model_label=
            "base",

        correct_counts=
            counts(
                default=4
            ),
    )

    adapted_counts = counts(
        default=4
    )

    adapted_counts[
        "same_metric_different_state"
    ] = 5

    invalid_case = (
        "same_metric_different_state::0"
    )

    adapted = synthetic_results(
        model_label=
            "adapted",

        correct_counts=
            adapted_counts,

        invalid_case_ids=
            frozenset(
                {
                    invalid_case,
                }
            ),
    )

    result = scoring.score_official_comparison(
        gold_records=
            gold,

        base_results=
            base,

        adapted_results=
            adapted,
    )

    assert (
        result[
            "adapted"
        ][
            "strict_json_valid_count"
        ]
        ==
        29
    )

    assert (
        result[
            "adapted"
        ][
            "strict_json_validity_rate"
        ]
        ==
        0.966667
    )

    assert (
        result[
            "gates"
        ][
            "adapted_absolute"
        ][
            "strict_json_validity_rate"
        ]
        is False
    )

    assert (
        result[
            "promotion_eligible"
        ]
        is False
    )


def test_per_relation_gate_can_fail_even_when_global_accuracy_passes(
) -> None:

    base = synthetic_results(
        model_label=
            "base",

        correct_counts=
            counts(
                default=4
            ),
    )

    adapted_counts = counts(
        default=6
    )

    adapted_counts[
        "same_metric_different_state"
    ] = 2

    adapted = synthetic_results(
        model_label=
            "adapted",

        correct_counts=
            adapted_counts,
    )

    result = scoring.score_official_comparison(
        gold_records=
            synthetic_gold(),

        base_results=
            base,

        adapted_results=
            adapted,
    )

    assert (
        result[
            "adapted"
        ][
            "accuracy"
        ]
        ==
        0.866667
    )

    assert (
        result[
            "gates"
        ][
            "adapted_absolute"
        ][
            "accuracy"
        ]
        is True
    )

    assert (
        result[
            "gates"
        ][
            "adapted_absolute"
        ][
            "per_relation_accuracy"
        ][
            "same_metric_different_state"
        ]
        is False
    )

    assert (
        result[
            "gates"
        ][
            "all_absolute_gates_pass"
        ]
        is False
    )

    assert (
        result[
            "promotion_eligible"
        ]
        is False
    )


def test_regression_blocks_promotion(
) -> None:

    base = synthetic_results(
        model_label=
            "base",

        correct_counts=
            counts(
                default=6
            ),
    )

    adapted_counts = counts(
        default=6
    )

    adapted_counts[
        "same_metric_different_state"
    ] = 5

    adapted = synthetic_results(
        model_label=
            "adapted",

        correct_counts=
            adapted_counts,
    )

    result = scoring.score_official_comparison(
        gold_records=
            synthetic_gold(),

        base_results=
            base,

        adapted_results=
            adapted,
    )

    assert (
        result[
            "adapted"
        ][
            "accuracy"
        ]
        ==
        0.966667
    )

    assert (
        result[
            "gates"
        ][
            "all_absolute_gates_pass"
        ]
        is True
    )

    assert (
        result[
            "comparison"
        ][
            "accuracy_delta"
        ]
        ==
        -0.033333
    )

    assert (
        result[
            "gates"
        ][
            "all_non_regression_gates_pass"
        ]
        is False
    )

    assert (
        result[
            "promotion_eligible"
        ]
        is False
    )


def test_tie_is_not_promotion_signal(
) -> None:

    base = synthetic_results(
        model_label=
            "base",

        correct_counts=
            counts(
                default=6
            ),
    )

    adapted = synthetic_results(
        model_label=
            "adapted",

        correct_counts=
            counts(
                default=6
            ),
    )

    result = scoring.score_official_comparison(
        gold_records=
            synthetic_gold(),

        base_results=
            base,

        adapted_results=
            adapted,
    )

    assert (
        result[
            "gates"
        ][
            "all_absolute_gates_pass"
        ]
        is True
    )

    assert (
        result[
            "gates"
        ][
            "all_non_regression_gates_pass"
        ]
        is True
    )

    assert (
        result[
            "comparison"
        ][
            "correct_case_count_delta"
        ]
        ==
        0
    )

    assert (
        result[
            "gates"
        ][
            "promotion_signal_pass"
        ]
        is False
    )

    assert (
        result[
            "promotion_eligible"
        ]
        is False
    )


def test_gold_boundary_rejects_gold_reason(
) -> None:

    gold = list(
        synthetic_gold()
    )

    gold[
        0
    ] = {
        **gold[
            0
        ],
        "gold_reason":
            "Synthetic forbidden extra gold field.",
    }

    try:

        scoring.normalize_gold_records(
            gold
        )

    except RuntimeError:

        pass

    else:

        raise AssertionError(
            "gold_reason crossed scoring boundary."
        )


def test_case_identity_mismatch_fails_closed(
) -> None:

    base = list(
        synthetic_results(
            model_label=
                "base",

            correct_counts=
                counts(
                    default=4
                ),
        )
    )

    base[
        0
    ] = {
        **base[
            0
        ],
        "case_id":
            "unknown-case",
    }

    adapted = synthetic_results(
        model_label=
            "adapted",

        correct_counts=
            counts(
                default=4
            ),
    )

    try:

        scoring.score_official_comparison(
            gold_records=
                synthetic_gold(),

            base_results=
                base,

            adapted_results=
                adapted,
        )

    except RuntimeError:

        pass

    else:

        raise AssertionError(
            "Case identity mismatch was accepted."
        )


def test_prompt_identity_mismatch_fails_closed(
) -> None:

    base = synthetic_results(
        model_label=
            "base",

        correct_counts=
            counts(
                default=4
            ),
    )

    adapted = list(
        synthetic_results(
            model_label=
                "adapted",

            correct_counts=
                counts(
                    default=4
                ),
        )
    )

    adapted[
        0
    ] = {
        **adapted[
            0
        ],
        "prompt_sha256":
            (
                "0"
                *
                64
            ),
    }

    try:

        scoring.score_official_comparison(
            gold_records=
                synthetic_gold(),

            base_results=
                base,

            adapted_results=
                adapted,
        )

    except RuntimeError:

        pass

    else:

        raise AssertionError(
            "Prompt identity mismatch was accepted."
        )


def test_scorer_has_no_protected_file_or_runtime_authority(
) -> None:

    source = SCORER_PATH.read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "HOLDOUT_CASES_PATH",
        "gold_reason",
        "Path(",
        ".open(",
        "open(",
        ".read_text(",
        ".read_bytes(",
        ".write_text(",
        ".write_bytes(",
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        ".generate(",
        ".from_pretrained(",
    ):

        assert (
            forbidden
            not in
            source
        )


def test_real_evaluation_directory_remains_unconsumed(
) -> None:

    preflight.assert_single_use_available()

    paths = (
        preflight.single_use_artifact_paths()
    )

    assert all(
        not path.exists()

        for path
        in paths.values()
    )


TESTS = (
    test_frozen_scoring_authorities,
    test_exact_thresholds_pass_with_plus_one_correct_case,
    test_four_of_six_uncertain_rounds_to_gate,
    test_strict_invalid_output_counts_incorrect_and_fails_gate,
    test_per_relation_gate_can_fail_even_when_global_accuracy_passes,
    test_regression_blocks_promotion,
    test_tie_is_not_promotion_signal,
    test_gold_boundary_rejects_gold_reason,
    test_case_identity_mismatch_fails_closed,
    test_prompt_identity_mismatch_fails_closed,
    test_scorer_has_no_protected_file_or_runtime_authority,
    test_real_evaluation_directory_remains_unconsumed,
)


def main(
) -> None:

    print(
        "=== R8 HOSPITAL DETERMINISTIC OFFICIAL SCORER v0.1 ==="
    )

    print()

    for test in TESTS:

        test()

        print(
            f"[PASS] {test.__name__}"
        )

    print()

    print(
        "PASS - hospital deterministic official scorer v0.1"
    )


if __name__ == "__main__":

    main()

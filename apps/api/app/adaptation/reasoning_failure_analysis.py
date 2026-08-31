from __future__ import annotations

import argparse
import hashlib
import json
import statistics

from collections import (
    Counter,
    defaultdict,
)
from pathlib import Path
from typing import Any


FAILURE_ANALYSIS_RULE_VERSION = (
    "qlora_reasoning_failure_analysis_v0.1"
)


ARTIFACT_ID = (
    "adaptation:datalens-semantic-qlora-v0.3:"
    "reasoning-failure-analysis:v0.1"
)


SOURCE_EVIDENCE_COMMIT = (
    "87bfdb0ce7ccec60539517968d6d95a03e12d87f"
)


SOURCE_EXECUTION_COMMIT = (
    "2cb1dff0a771f6c89f72a9e24a0b17994d325391"
)


SOURCE_BENCHMARK_COMMIT = (
    "60e50bb435f8e3f8bc8b2d093d711676ed645b6c"
)


SOURCE_REPORT_SHA256 = (
    "1194baabef17891f84f2879a135586c7"
    "da95363cbac49c4258a8133d81c6206b"
)


SOURCE_ADAPTER_BUNDLE_SHA256 = (
    "ed10297a1d0e6ce504189985d91f46b2"
    "f1f37a5a67fe84f0d62a9f09c3940ffe"
)


RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
)


ROOT = Path(__file__).resolve().parents[2]


REPORT_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / (
        "datalens_semantic_qlora_v0.3_"
        "reasoning_evaluation_v0.2_report.json"
    )
)


ARTIFACT_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / (
        "datalens_semantic_qlora_v0.3_"
        "reasoning_failure_analysis_v0.1.json"
    )
)


def sha256_file(
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


def load_json_object(
    path: Path,
) -> dict[
    str,
    Any,
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
            f"Expected JSON object: {path}"
        )

    return payload


def write_json_lf(
    *,
    path: Path,
    payload: object,
) -> None:
    if path.exists():
        raise FileExistsError(
            (
                "Refusing to overwrite immutable "
                f"derived evidence: {path}"
            )
        )

    data = (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode(
            "utf-8"
        )
        +
        b"\n"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        data
    )


def mean(
    values,
) -> float:
    values = list(
        values
    )

    if not values:
        raise ValueError(
            "Cannot compute mean of empty sequence."
        )

    return statistics.mean(
        values
    )


def rounded(
    value: float,
    digits: int = 8,
) -> float:
    return round(
        float(
            value
        ),
        digits,
    )


def _validate_result_sets(
    *,
    report: dict[
        str,
        Any,
    ],
) -> tuple[
    dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
    dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
]:
    base = report[
        "base"
    ][
        "results"
    ]

    adapted = report[
        "adapted"
    ][
        "results"
    ]

    if (
        len(
            base
        )
        !=
        19
        or
        len(
            adapted
        )
        !=
        19
    ):
        raise RuntimeError(
            "Expected exactly 19 base and adapted cases."
        )

    base_by_id = {
        item[
            "case_id"
        ]:
            item

        for item
        in base
    }

    adapted_by_id = {
        item[
            "case_id"
        ]:
            item

        for item
        in adapted
    }

    if (
        len(
            base_by_id
        )
        !=
        19
    ):
        raise RuntimeError(
            "Base case IDs are not unique."
        )

    if (
        set(
            base_by_id
        )
        !=
        set(
            adapted_by_id
        )
    ):
        raise RuntimeError(
            "Base/adapted case sets differ."
        )

    return (
        base_by_id,
        adapted_by_id,
    )


def _scores(
    item: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    float,
]:
    scores = item[
        "scores"
    ]

    if (
        set(
            scores
        )
        !=
        set(
            RELATIONS
        )
    ):
        raise RuntimeError(
            (
                "Unexpected candidate score set for "
                f"{item['case_id']}."
            )
        )

    return {
        relation:
            float(
                scores[
                    relation
                ]
            )

        for relation
        in RELATIONS
    }


def _expected_margin(
    item: dict[
        str,
        Any,
    ],
) -> float:
    expected = item[
        "expected_relation"
    ]

    scores = _scores(
        item
    )

    return (
        scores[
            expected
        ]
        -
        max(
            score

            for relation, score
            in scores.items()

            if relation
            !=
            expected
        )
    )


def _centered_scores(
    item: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    float,
]:
    scores = _scores(
        item
    )

    center = mean(
        scores.values()
    )

    return {
        relation:
            (
                scores[
                    relation
                ]
                -
                center
            )

        for relation
        in RELATIONS
    }


def _pairwise_gap(
    *,
    item: dict[
        str,
        Any,
    ],
    left: str,
    right: str,
) -> float:
    scores = _scores(
        item
    )

    return (
        scores[
            left
        ]
        -
        scores[
            right
        ]
    )


def _distribution(
    values,
) -> dict[
    str,
    int,
]:
    counter = Counter(
        values
    )

    return {
        relation:
            counter[
                relation
            ]

        for relation
        in RELATIONS

        if counter[
            relation
        ]
        >
        0
    }


def analyze_report(
    *,
    report: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    if (
        report[
            "adapter_bundle_sha256"
        ]
        !=
        SOURCE_ADAPTER_BUNDLE_SHA256
    ):
        raise RuntimeError(
            "Unexpected adapter bundle."
        )

    if (
        report[
            "execution_git_commit"
        ]
        !=
        SOURCE_EXECUTION_COMMIT
    ):
        raise RuntimeError(
            "Unexpected evaluation execution commit."
        )

    if (
        report[
            "protocol_commit"
        ]
        !=
        SOURCE_BENCHMARK_COMMIT
    ):
        raise RuntimeError(
            "Unexpected benchmark commit."
        )

    if (
        report[
            "final_acceptance_loaded"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance was loaded."
        )

    if (
        report[
            "final_acceptance_evaluated"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance was evaluated."
        )

    base_by_id, adapted_by_id = (
        _validate_result_sets(
            report=
                report,
        )
    )

    case_ids = list(
        base_by_id
    )

    rows = []

    for case_id in case_ids:
        base = base_by_id[
            case_id
        ]

        adapted = adapted_by_id[
            case_id
        ]

        expected = base[
            "expected_relation"
        ]

        if (
            adapted[
                "expected_relation"
            ]
            !=
            expected
        ):
            raise RuntimeError(
                (
                    "Expected relation changed for "
                    f"{case_id}."
                )
            )

        base_scores = _scores(
            base
        )

        adapted_scores = _scores(
            adapted
        )

        base_centered = (
            _centered_scores(
                base
            )
        )

        adapted_centered = (
            _centered_scores(
                adapted
            )
        )

        base_margin = (
            _expected_margin(
                base
            )
        )

        adapted_margin = (
            _expected_margin(
                adapted
            )
        )

        rows.append(
            {
                "adapted_correct":
                    bool(
                        adapted[
                            "correct"
                        ]
                    ),

                "adapted_margin":
                    adapted_margin,

                "adapted_prediction":
                    adapted[
                        "predicted_relation"
                    ],

                "base_correct":
                    bool(
                        base[
                            "correct"
                        ]
                    ),

                "base_margin":
                    base_margin,

                "base_prediction":
                    base[
                        "predicted_relation"
                    ],

                "candidate_score_deltas": {
                    relation:
                        (
                            adapted_scores[
                                relation
                            ]
                            -
                            base_scores[
                                relation
                            ]
                        )

                    for relation
                    in RELATIONS
                },

                "case_id":
                    case_id,

                "expected":
                    expected,

                "expected_score_delta":
                    (
                        adapted_scores[
                            expected
                        ]
                        -
                        base_scores[
                            expected
                        ]
                    ),

                "margin_delta":
                    (
                        adapted_margin
                        -
                        base_margin
                    ),

                "relative_preference_deltas": {
                    relation:
                        (
                            adapted_centered[
                                relation
                            ]
                            -
                            base_centered[
                                relation
                            ]
                        )

                    for relation
                    in RELATIONS
                },
            }
        )

    expected_distribution = (
        _distribution(
            row[
                "expected"
            ]

            for row
            in rows
        )
    )

    base_prediction_distribution = (
        _distribution(
            row[
                "base_prediction"
            ]

            for row
            in rows
        )
    )

    adapted_prediction_distribution = (
        _distribution(
            row[
                "adapted_prediction"
            ]

            for row
            in rows
        )
    )

    raw_candidate_score_shifts = {}

    for relation in RELATIONS:
        values = [
            row[
                "candidate_score_deltas"
            ][
                relation
            ]

            for row
            in rows
        ]

        raw_candidate_score_shifts[
            relation
        ] = {
            "max":
                rounded(
                    max(
                        values
                    )
                ),

            "mean":
                rounded(
                    mean(
                        values
                    )
                ),

            "median":
                rounded(
                    statistics.median(
                        values
                    )
                ),

            "min":
                rounded(
                    min(
                        values
                    )
                ),
        }

    relative_preference_shift = {
        relation:
            rounded(
                mean(
                    row[
                        "relative_preference_deltas"
                    ][
                        relation
                    ]

                    for row
                    in rows
                )
            )

        for relation
        in RELATIONS
    }

    pairwise_pairs = (
        (
            "same_process_different_stage",
            "related_distinct_metric",
        ),

        (
            "same_process_different_stage",
            "same_metric_different_state",
        ),

        (
            "related_distinct_metric",
            "same_metric_different_state",
        ),
    )

    pairwise_shifts = {}

    for left, right in pairwise_pairs:
        deltas = []

        for case_id in case_ids:
            before = (
                _pairwise_gap(
                    item=
                        base_by_id[
                            case_id
                        ],

                    left=
                        left,

                    right=
                        right,
                )
            )

            after = (
                _pairwise_gap(
                    item=
                        adapted_by_id[
                            case_id
                        ],

                    left=
                        left,

                    right=
                        right,
                )
            )

            deltas.append(
                after
                -
                before
            )

        pairwise_shifts[
            f"{left}__minus__{right}"
        ] = rounded(
            mean(
                deltas
            )
        )

    margin_improved = [
        row
        for row
        in rows
        if row[
            "margin_delta"
        ]
        >
        0
    ]

    margin_worsened = [
        row
        for row
        in rows
        if row[
            "margin_delta"
        ]
        <
        0
    ]

    negative_to_positive = [
        row
        for row
        in rows
        if (
            row[
                "base_margin"
            ]
            <=
            0
            and
            row[
                "adapted_margin"
            ]
            >
            0
        )
    ]

    positive_to_non_positive = [
        row
        for row
        in rows
        if (
            row[
                "base_margin"
            ]
            >
            0
            and
            row[
                "adapted_margin"
            ]
            <=
            0
        )
    ]

    expected_score_shifts = [
        row[
            "expected_score_delta"
        ]

        for row
        in rows
    ]

    rows_by_expected = defaultdict(
        list
    )

    for row in rows:
        rows_by_expected[
            row[
                "expected"
            ]
        ].append(
            row
        )

    by_expected_relation = {}

    for relation in RELATIONS:
        relation_rows = (
            rows_by_expected[
                relation
            ]
        )

        by_expected_relation[
            relation
        ] = {
            "adapted_correct":
                sum(
                    row[
                        "adapted_correct"
                    ]

                    for row
                    in relation_rows
                ),

            "adapted_mean_margin":
                rounded(
                    mean(
                        row[
                            "adapted_margin"
                        ]

                        for row
                        in relation_rows
                    )
                ),

            "base_correct":
                sum(
                    row[
                        "base_correct"
                    ]

                    for row
                    in relation_rows
                ),

            "base_mean_margin":
                rounded(
                    mean(
                        row[
                            "base_margin"
                        ]

                        for row
                        in relation_rows
                    )
                ),

            "case_count":
                len(
                    relation_rows
                ),

            "mean_margin_delta":
                rounded(
                    mean(
                        row[
                            "margin_delta"
                        ]

                        for row
                        in relation_rows
                    )
                ),

            "margin_improved_count":
                sum(
                    row[
                        "margin_delta"
                    ]
                    >
                    0

                    for row
                    in relation_rows
                ),
        }

    transitions_counter = Counter(
        (
            row[
                "base_prediction"
            ],
            row[
                "adapted_prediction"
            ],
        )

        for row
        in rows
    )

    prediction_transitions = [
        {
            "adapted_prediction":
                adapted_prediction,

            "base_prediction":
                base_prediction,

            "count":
                count,
        }

        for (
            base_prediction,
            adapted_prediction,
        ), count
        in sorted(
            transitions_counter.items()
        )
    ]

    changed_predictions = [
        {
            "adapted_correct":
                row[
                    "adapted_correct"
                ],

            "adapted_margin":
                rounded(
                    row[
                        "adapted_margin"
                    ]
                ),

            "adapted_prediction":
                row[
                    "adapted_prediction"
                ],

            "base_correct":
                row[
                    "base_correct"
                ],

            "base_margin":
                rounded(
                    row[
                        "base_margin"
                    ]
                ),

            "base_prediction":
                row[
                    "base_prediction"
                ],

            "case_id":
                row[
                    "case_id"
                ],

            "expected_relation":
                row[
                    "expected"
                ],

            "margin_delta":
                rounded(
                    row[
                        "margin_delta"
                    ]
                ),
        }

        for row
        in rows

        if (
            row[
                "base_prediction"
            ]
            !=
            row[
                "adapted_prediction"
            ]
        )
    ]

    top_improvements = sorted(
        rows,
        key=lambda row: (
            -row[
                "margin_delta"
            ],
            row[
                "case_id"
            ],
        ),
    )[
        :5
    ]

    top_regressions = sorted(
        rows,
        key=lambda row: (
            row[
                "margin_delta"
            ],
            row[
                "case_id"
            ],
        ),
    )[
        :5
    ]

    def summarize_ranked(
        selected_rows,
    ):
        return [
            {
                "adapted_margin":
                    rounded(
                        row[
                            "adapted_margin"
                        ]
                    ),

                "adapted_prediction":
                    row[
                        "adapted_prediction"
                    ],

                "base_margin":
                    rounded(
                        row[
                            "base_margin"
                        ]
                    ),

                "base_prediction":
                    row[
                        "base_prediction"
                    ],

                "case_id":
                    row[
                        "case_id"
                    ],

                "expected_relation":
                    row[
                        "expected"
                    ],

                "margin_delta":
                    rounded(
                        row[
                            "margin_delta"
                        ]
                    ),
            }

            for row
            in selected_rows
        ]

    paired = report[
        "paired"
    ]

    analysis = {
        "argmax_changes":
            changed_predictions,

        "by_expected_relation":
            by_expected_relation,

        "distributions": {
            "adapted_predictions":
                adapted_prediction_distribution,

            "base_predictions":
                base_prediction_distribution,

            "expected":
                expected_distribution,
        },

        "expected_answer_score_shift": {
            "decreased_count":
                sum(
                    value
                    <
                    0

                    for value
                    in expected_score_shifts
                ),

            "increased_count":
                sum(
                    value
                    >
                    0

                    for value
                    in expected_score_shifts
                ),

            "mean_delta":
                rounded(
                    mean(
                        expected_score_shifts
                    )
                ),
        },

        "expected_margin_analysis": {
            "adapted_mean":
                rounded(
                    mean(
                        row[
                            "adapted_margin"
                        ]

                        for row
                        in rows
                    )
                ),

            "base_mean":
                rounded(
                    mean(
                        row[
                            "base_margin"
                        ]

                        for row
                        in rows
                    )
                ),

            "improved_count":
                len(
                    margin_improved
                ),

            "mean_delta":
                rounded(
                    mean(
                        row[
                            "margin_delta"
                        ]

                        for row
                        in rows
                    )
                ),

            "negative_or_zero_to_positive_count":
                len(
                    negative_to_positive
                ),

            "positive_to_non_positive_count":
                len(
                    positive_to_non_positive
                ),

            "unchanged_count":
                (
                    19
                    -
                    len(
                        margin_improved
                    )
                    -
                    len(
                        margin_worsened
                    )
                ),

            "worsened_count":
                len(
                    margin_worsened
                ),
        },

        "observed_result": {
            "adapted_accuracy":
                report[
                    "adapted"
                ][
                    "accuracy"
                ],

            "adapted_correct_count":
                report[
                    "adapted"
                ][
                    "correct_count"
                ],

            "adapted_macro_accuracy":
                report[
                    "adapted"
                ][
                    "macro_accuracy"
                ],

            "adapted_only_correct":
                paired[
                    "adapted_only_correct"
                ],

            "accuracy_delta":
                paired[
                    "accuracy_delta"
                ],

            "base_accuracy":
                report[
                    "base"
                ][
                    "accuracy"
                ],

            "base_correct_count":
                report[
                    "base"
                ][
                    "correct_count"
                ],

            "base_macro_accuracy":
                report[
                    "base"
                ][
                    "macro_accuracy"
                ],

            "base_only_correct":
                paired[
                    "base_only_correct"
                ],

            "both_correct":
                paired[
                    "both_correct"
                ],

            "both_wrong":
                paired[
                    "both_wrong"
                ],

            "changed_prediction_count":
                paired[
                    "changed_prediction_count"
                ],

            "macro_accuracy_delta":
                paired[
                    "macro_accuracy_delta"
                ],

            "preregistered_signal":
                paired[
                    "preregistered_signal"
                ],
        },

        "pairwise_preference_gap_shift_mean":
            pairwise_shifts,

        "prediction_transitions":
            prediction_transitions,

        "raw_candidate_score_shifts":
            raw_candidate_score_shifts,

        "relative_candidate_preference_shift_mean":
            relative_preference_shift,

        "top_margin_improvements":
            summarize_ranked(
                top_improvements
            ),

        "top_margin_regressions":
            summarize_ranked(
                top_regressions
            ),
    }

    return analysis


def build_artifact(
    *,
    report: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    analysis = analyze_report(
        report=
            report,
    )

    observed = analysis[
        "observed_result"
    ]

    margins = analysis[
        "expected_margin_analysis"
    ]

    same_metric = analysis[
        "by_expected_relation"
    ][
        "same_metric_different_state"
    ]

    return {
        "analysis":
            analysis,

        "artifact_id":
            ARTIFACT_ID,

        "conclusions": {
            "causal_root_cause_established":
                False,

            "diagnostic_gate_passed":
                False,

            "final_acceptance_eligible":
                False,

            "learning_signal_observed":
                (
                    margins[
                        "improved_count"
                    ]
                    >
                    margins[
                        "worsened_count"
                    ]
                    and
                    same_metric[
                        "mean_margin_delta"
                    ]
                    >
                    0
                ),

            "observed_failure_mode":
                (
                    "frozen_reasoning_benchmark_"
                    "decision_regression"
                ),

            "preregistered_signal":
                observed[
                    "preregistered_signal"
                ],
        },

        "evaluation_policy": {
            "final_acceptance_remains_closed":
                True,

            "hotel_role_for_future_adaptation":
                "diagnostic_regression_only",

            "hotel_reusable_as_new_independent_holdout":
                False,

            "new_independent_holdout_required_before_next_training":
                True,

            "training_loss_is_acceptance_evidence":
                False,
        },

        "interpretation": {
            "causal_hypotheses_are_not_proven":
                True,

            "observed_facts": [
                (
                    "The adapted candidate produced fewer "
                    "correct argmax decisions than the base "
                    "candidate on the frozen 19-case benchmark."
                ),

                (
                    "All three argmax changes were regressions: "
                    "three base-correct cases became adapted-wrong."
                ),

                (
                    "No base-wrong case became adapted-correct."
                ),

                (
                    "The expected-class margin improved on "
                    "11 of 19 cases but no non-positive margin "
                    "crossed into positive territory."
                ),

                (
                    "Three positive expected-class margins "
                    "crossed to non-positive after adaptation."
                ),

                (
                    "The same_metric_different_state expected "
                    "class improved its mean margin but remained "
                    "0-for-5 correct."
                ),

                (
                    "All expected-answer absolute scores "
                    "increased after adaptation."
                ),

                (
                    "Absolute score increases do not establish "
                    "improved relative classification quality."
                ),
            ],

            "permitted_hypotheses_for_next_experiment": [
                "small_dataset_or_overfitting",

                (
                    "training_evaluation_task_alignment_"
                    "mismatch"
                ),

                (
                    "insufficient_contrastive_"
                    "decision_boundary_supervision"
                ),

                (
                    "candidate_probability_"
                    "calibration_shift"
                ),
            ],
        },

        "rule_version":
            FAILURE_ANALYSIS_RULE_VERSION,

        "safety": {
            "adapter_loaded":
                False,

            "benchmark_modified":
                False,

            "cuda_requested":
                False,

            "final_acceptance_evaluated":
                False,

            "final_acceptance_loaded":
                False,

            "free_generation_used":
                False,

            "inference_executed":
                False,

            "llm_judge_used":
                False,

            "model_loaded":
                False,

            "new_evaluation_executed":
                False,

            "training_executed":
                False,
        },

        "source": {
            "adapter_bundle_sha256":
                SOURCE_ADAPTER_BUNDLE_SHA256,

            "benchmark_git_commit":
                SOURCE_BENCHMARK_COMMIT,

            "evaluation_evidence_git_commit":
                SOURCE_EVIDENCE_COMMIT,

            "evaluation_execution_git_commit":
                SOURCE_EXECUTION_COMMIT,

            "evaluation_report_relative_path":
                (
                    "artifacts/adaptation/evaluation/"
                    "datalens_semantic_qlora_v0.3_"
                    "reasoning_evaluation_v0.2_report.json"
                ),

            "evaluation_report_sha256":
                SOURCE_REPORT_SHA256,
        },

        "status":
            "derived_post_evaluation_analysis",
    }


def validate_artifact(
    *,
    artifact: dict[
        str,
        Any,
    ],
    report: dict[
        str,
        Any,
    ],
) -> None:
    expected = build_artifact(
        report=
            report,
    )

    if artifact != expected:
        raise RuntimeError(
            (
                "Failure Analysis artifact does not "
                "recompute exactly from source evidence."
            )
        )

    observed = artifact[
        "analysis"
    ][
        "observed_result"
    ]

    required_result = {
        "adapted_accuracy":
            0.157895,

        "adapted_correct_count":
            3,

        "adapted_macro_accuracy":
            0.384615,

        "adapted_only_correct":
            0,

        "accuracy_delta":
            -0.157894,

        "base_accuracy":
            0.315789,

        "base_correct_count":
            6,

        "base_macro_accuracy":
            0.461538,

        "base_only_correct":
            3,

        "both_correct":
            3,

        "both_wrong":
            13,

        "changed_prediction_count":
            3,

        "macro_accuracy_delta":
            -0.076923,

        "preregistered_signal":
            "negative_signal",
    }

    if observed != required_result:
        raise RuntimeError(
            "Observed result contract mismatch."
        )

    margins = artifact[
        "analysis"
    ][
        "expected_margin_analysis"
    ]

    if (
        margins[
            "improved_count"
        ]
        !=
        11
        or
        margins[
            "worsened_count"
        ]
        !=
        8
        or
        margins[
            "negative_or_zero_to_positive_count"
        ]
        !=
        0
        or
        margins[
            "positive_to_non_positive_count"
        ]
        !=
        3
    ):
        raise RuntimeError(
            "Margin diagnostic contract mismatch."
        )

    if (
        artifact[
            "conclusions"
        ][
            "final_acceptance_eligible"
        ]
        is not False
    ):
        raise RuntimeError(
            "Failed candidate cannot be Final Acceptance eligible."
        )

    if (
        artifact[
            "conclusions"
        ][
            "causal_root_cause_established"
        ]
        is not False
    ):
        raise RuntimeError(
            "Failure Analysis must not overclaim causality."
        )

    if (
        artifact[
            "evaluation_policy"
        ][
            "hotel_reusable_as_new_independent_holdout"
        ]
        is not False
    ):
        raise RuntimeError(
            "Hotel cannot be reused as a new independent holdout."
        )


def build(
) -> dict[
    str,
    Any,
]:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            REPORT_PATH
        )

    if (
        sha256_file(
            REPORT_PATH
        )
        !=
        SOURCE_REPORT_SHA256
    ):
        raise RuntimeError(
            "Source evaluation report SHA mismatch."
        )

    report = load_json_object(
        REPORT_PATH
    )

    artifact = build_artifact(
        report=
            report,
    )

    validate_artifact(
        artifact=
            artifact,

        report=
            report,
    )

    write_json_lf(
        path=
            ARTIFACT_PATH,

        payload=
            artifact,
    )

    return artifact


def validate_existing(
) -> dict[
    str,
    Any,
]:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            REPORT_PATH
        )

    if not ARTIFACT_PATH.is_file():
        raise FileNotFoundError(
            ARTIFACT_PATH
        )

    if (
        sha256_file(
            REPORT_PATH
        )
        !=
        SOURCE_REPORT_SHA256
    ):
        raise RuntimeError(
            "Source evaluation report SHA mismatch."
        )

    report = load_json_object(
        REPORT_PATH
    )

    artifact = load_json_object(
        ARTIFACT_PATH
    )

    validate_artifact(
        artifact=
            artifact,

        report=
            report,
    )

    return artifact


def main(
) -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=(
            "build",
            "validate",
        ),
    )

    args = parser.parse_args()

    if args.mode == "build":
        artifact = build()

        print(
            "DATALENS QLORA FAILURE ANALYSIS BUILD: PASS"
        )

        print(
            (
                "Artifact SHA256: "
                f"{sha256_file(ARTIFACT_PATH)}"
            )
        )

        print(
            (
                "Preregistered signal: "
                f"{artifact['conclusions']['preregistered_signal']}"
            )
        )

        print(
            (
                "Final Acceptance eligible: "
                f"{artifact['conclusions']['final_acceptance_eligible']}"
            )
        )

        return

    validate_existing()

    print(
        "DATALENS QLORA FAILURE ANALYSIS VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()

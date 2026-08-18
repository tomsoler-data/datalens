from __future__ import annotations

from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.evals.analytical_planner_contract_v0_9 import (
    AnalyticalPlannerCandidate,
    AnalyticalRequirementPlan,
)

from app.evals.analytical_planner_input_v0_9 import (
    AnalyticalPlannerInput,
)

from app.evals.analytical_planner_validator_v0_9 import (
    validate_analytical_planner_candidate,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_SCORER_VERSION = (
    "analytical_planner_scorer_v0.9"
)


# ============================================================
# SCORE PRECISION
# ============================================================

SCORE_PRECISION = 12


def _score(
    value: float,
) -> float:

    normalized = round(
        float(
            value
        ),
        SCORE_PRECISION,
    )


    if normalized < 0.0:
        return 0.0


    if normalized > 1.0:
        return 1.0


    return normalized


# ============================================================
# WEIGHTS
# ============================================================

WEIGHTS = {
    "requirement_coverage":
        0.15,

    "intent":
        0.10,

    "family":
        0.10,

    "target_grain":
        0.10,

    "tool_sequence":
        0.20,

    "tool_arguments":
        0.20,

    "validator_acceptance":
        0.15,
}


assert (
    round(
        sum(
            WEIGHTS.values()
        ),
        12,
    )
    == 1.0
)


# ============================================================
# METRICS
# ============================================================

class AnalyticalPlannerScoreMetrics(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    requirement_coverage_f1: float

    intent_accuracy: float

    family_accuracy: float

    target_grain_accuracy: float

    tool_sequence_score: float

    tool_argument_score: float

    validator_acceptance: float


# ============================================================
# DIAGNOSTICS
# ============================================================

class AnalyticalPlannerScoreDiagnostics(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    missing_requirement_ids: list[
        str
    ]

    extra_requirement_ids: list[
        str
    ]

    invalid_requirement_ids: list[
        str
    ]

    validator_issue_codes: list[
        str
    ]

    exact_requirement_ids: list[
        str
    ]


# ============================================================
# RESULT
# ============================================================

class AnalyticalPlannerScore(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    scorer_version: str

    overall: float

    metrics: AnalyticalPlannerScoreMetrics

    diagnostics: AnalyticalPlannerScoreDiagnostics


    def as_dict(
        self,
    ) -> dict[str, Any]:

        return self.model_dump(
            mode="json",
        )


# ============================================================
# GENERIC HELPERS
# ============================================================

def _average(
    values: list[float],
) -> float:

    if not values:
        return 0.0


    return _score(
        sum(
            values
        )
        / len(
            values
        )
    )


def _set_f1(
    expected: set[str],
    actual: set[str],
) -> float:

    if (
        not expected
        and not actual
    ):
        return 1.0


    if (
        not expected
        or not actual
    ):
        return 0.0


    true_positive = len(
        expected
        & actual
    )


    precision = (
        true_positive
        / len(
            actual
        )
    )


    recall = (
        true_positive
        / len(
            expected
        )
    )


    if (
        precision
        + recall
        == 0.0
    ):
        return 0.0


    return _score(
        (
            2.0
            * precision
            * recall
        )
        / (
            precision
            + recall
        )
    )


# ============================================================
# LONGEST COMMON SUBSEQUENCE
# ============================================================

def _lcs_length(
    expected: list[str],
    actual: list[str],
) -> int:

    rows = (
        len(
            expected
        )
        + 1
    )


    columns = (
        len(
            actual
        )
        + 1
    )


    matrix = [
        [
            0
            for _ in range(
                columns
            )
        ]

        for _ in range(
            rows
        )
    ]


    for i in range(
        1,
        rows,
    ):

        for j in range(
            1,
            columns,
        ):

            if (
                expected[
                    i - 1
                ]
                == actual[
                    j - 1
                ]
            ):

                matrix[
                    i
                ][
                    j
                ] = (
                    matrix[
                        i - 1
                    ][
                        j - 1
                    ]
                    + 1
                )


            else:

                matrix[
                    i
                ][
                    j
                ] = max(
                    matrix[
                        i - 1
                    ][
                        j
                    ],

                    matrix[
                        i
                    ][
                        j - 1
                    ],
                )


    return (
        matrix[
            -1
        ][
            -1
        ]
    )


def _sequence_score(
    expected: list[str],
    actual: list[str],
) -> float:

    if (
        not expected
        and not actual
    ):
        return 1.0


    if (
        not expected
        or not actual
    ):
        return 0.0


    lcs = (
        _lcs_length(
            expected,
            actual,
        )
    )


    return _score(
        (
            2.0
            * lcs
        )
        / (
            len(
                expected
            )
            + len(
                actual
            )
        )
    )


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_text(
    value: Any,
) -> str:

    return (
        " ".join(
            str(
                value
            )
            .strip()
            .split()
        )
        .lower()
    )


def _normalize_list(
    value: Any,
) -> list[Any]:

    if value is None:
        return []


    if isinstance(
        value,
        list,
    ):
        return value


    return [
        value,
    ]


def _unordered_list_equal(
    expected: Any,
    actual: Any,
) -> bool:

    expected_values = {
        _normalize_text(
            value
        )

        for value
        in _normalize_list(
            expected
        )
    }


    actual_values = {
        _normalize_text(
            value
        )

        for value
        in _normalize_list(
            actual
        )
    }


    return (
        expected_values
        == actual_values
    )


# ============================================================
# TOOL ARGUMENT COMPARISON
# ============================================================

def _tool_argument_score(
    *,
    expected_action,
    actual_action,
) -> float:
    """
    Compare arguments for two actions of the same tool.

    Important semantic rules:

    - measure_association is symmetric;
    - aggregate metrics/group_by are unordered;
    - derive_metric inputs are unordered;
    - detect_entity_outliers metrics are unordered;
    - step IDs are deliberately ignored.
    """

    if (
        expected_action.name
        != actual_action.name
    ):
        return 0.0


    expected = (
        expected_action.model_dump(
            mode="json",
        )
    )


    actual = (
        actual_action.model_dump(
            mode="json",
        )
    )


    tool_name = (
        expected_action.name
    )


    # ========================================================
    # TOOL NAME ITSELF IS NOT AN ARGUMENT
    # ========================================================

    expected.pop(
        "name",
        None,
    )


    actual.pop(
        "name",
        None,
    )


    # ========================================================
    # ASSOCIATION
    #
    # target/value are analytically symmetric.
    # ========================================================

    if (
        tool_name
        == "measure_association"
    ):

        expected_pair = {
            _normalize_text(
                expected[
                    "target"
                ]
            ),

            _normalize_text(
                expected[
                    "value"
                ]
            ),
        }


        actual_pair = {
            _normalize_text(
                actual[
                    "target"
                ]
            ),

            _normalize_text(
                actual[
                    "value"
                ]
            ),
        }


        return (
            1.0
            if (
                expected_pair
                == actual_pair
            )
            else 0.0
        )


    # ========================================================
    # AGGREGATE
    # ========================================================

    if (
        tool_name
        == "aggregate"
    ):

        metric_match = (
            _unordered_list_equal(
                expected[
                    "metrics"
                ],
                actual[
                    "metrics"
                ],
            )
        )


        group_match = (
            _unordered_list_equal(
                expected.get(
                    "group_by"
                ),
                actual.get(
                    "group_by"
                ),
            )
        )


        return _score(
            (
                float(
                    metric_match
                )
                + float(
                    group_match
                )
            )
            / 2.0
        )


    # ========================================================
    # DERIVE METRIC
    # ========================================================

    if (
        tool_name
        == "derive_metric"
    ):

        input_match = (
            _unordered_list_equal(
                expected[
                    "inputs"
                ],
                actual[
                    "inputs"
                ],
            )
        )


        output_match = (
            _normalize_text(
                expected[
                    "output"
                ]
            )
            == _normalize_text(
                actual[
                    "output"
                ]
            )
        )


        formula_match = (
            _normalize_text(
                expected[
                    "formula"
                ]
            )
            == _normalize_text(
                actual[
                    "formula"
                ]
            )
        )


        return _score(
            (
                float(
                    input_match
                )
                + float(
                    output_match
                )
                + float(
                    formula_match
                )
            )
            / 3.0
        )


    # ========================================================
    # ENTITY OUTLIERS
    # ========================================================

    if (
        tool_name
        == "detect_entity_outliers"
    ):

        entity_match = (
            _normalize_text(
                expected[
                    "entity"
                ]
            )
            == _normalize_text(
                actual[
                    "entity"
                ]
            )
        )


        metrics_match = (
            _unordered_list_equal(
                expected[
                    "metrics"
                ],
                actual[
                    "metrics"
                ],
            )
        )


        return _score(
            (
                float(
                    entity_match
                )
                + float(
                    metrics_match
                )
            )
            / 2.0
        )


    # ========================================================
    # ALL OTHER TOOLS
    #
    # Compare each declared argument equally.
    # ========================================================

    expected_keys = set(
        expected
    )


    actual_keys = set(
        actual
    )


    all_keys = (
        expected_keys
        | actual_keys
    )


    if not all_keys:
        return 1.0


    matches = 0.0


    for key in all_keys:

        if (
            key
            not in expected
            or key
            not in actual
        ):
            continue


        expected_value = (
            expected[
                key
            ]
        )


        actual_value = (
            actual[
                key
            ]
        )


        if (
            _normalize_text(
                expected_value
            )
            == _normalize_text(
                actual_value
            )
        ):

            matches += 1.0


    return _score(
        matches
        / len(
            all_keys
        )
    )


# ============================================================
# STEP ARGUMENT SCORE
# ============================================================

def _plan_argument_score(
    *,
    expected_plan: AnalyticalRequirementPlan,
    actual_plan: AnalyticalRequirementPlan,
) -> float:

    denominator = max(
        len(
            expected_plan.steps
        ),
        len(
            actual_plan.steps
        ),
    )


    if denominator == 0:
        return 1.0


    total = 0.0


    for index in range(
        denominator
    ):

        if (
            index
            >= len(
                expected_plan.steps
            )
            or index
            >= len(
                actual_plan.steps
            )
        ):
            continue


        expected_step = (
            expected_plan.steps[
                index
            ]
        )


        actual_step = (
            actual_plan.steps[
                index
            ]
        )


        total += (
            _tool_argument_score(
                expected_action=(
                    expected_step.action
                ),

                actual_action=(
                    actual_step.action
                ),
            )
        )


    return _score(
        total
        / denominator
    )


# ============================================================
# EXACT REQUIREMENT MATCH
# ============================================================

def _requirement_exact(
    *,
    expected_plan: AnalyticalRequirementPlan,
    actual_plan: AnalyticalRequirementPlan,
) -> bool:

    if (
        expected_plan.intent
        != actual_plan.intent
    ):
        return False


    if (
        expected_plan.family
        != actual_plan.family
    ):
        return False


    if (
        expected_plan.target_grain
        != actual_plan.target_grain
    ):
        return False


    expected_actions = [
        step.action.name

        for step
        in expected_plan.steps
    ]


    actual_actions = [
        step.action.name

        for step
        in actual_plan.steps
    ]


    if (
        expected_actions
        != actual_actions
    ):
        return False


    return (
        _plan_argument_score(
            expected_plan=(
                expected_plan
            ),

            actual_plan=(
                actual_plan
            ),
        )
        == 1.0
    )


# ============================================================
# PUBLIC SCORER
# ============================================================

def score_analytical_planner_candidate(
    *,
    candidate: AnalyticalPlannerCandidate,
    expected: AnalyticalPlannerCandidate,
    planner_input: AnalyticalPlannerInput,
) -> AnalyticalPlannerScore:

    expected_by_requirement = {
        plan.requirement_id:
            plan

        for plan
        in expected.plans
    }


    actual_by_requirement = {
        plan.requirement_id:
            plan

        for plan
        in candidate.plans
    }


    expected_ids = set(
        expected_by_requirement
    )


    actual_ids = set(
        actual_by_requirement
    )


    # ========================================================
    # REQUIREMENT COVERAGE
    # ========================================================

    requirement_coverage_f1 = (
        _set_f1(
            expected_ids,
            actual_ids,
        )
    )


    # ========================================================
    # PER-EXPECTED-REQUIREMENT METRICS
    #
    # Missing plans receive zero.
    # ========================================================

    intent_scores: list[
        float
    ] = []


    family_scores: list[
        float
    ] = []


    grain_scores: list[
        float
    ] = []


    sequence_scores: list[
        float
    ] = []


    argument_scores: list[
        float
    ] = []


    exact_requirement_ids: list[
        str
    ] = []


    for requirement_id in sorted(
        expected_ids
    ):

        expected_plan = (
            expected_by_requirement[
                requirement_id
            ]
        )


        actual_plan = (
            actual_by_requirement.get(
                requirement_id
            )
        )


        if actual_plan is None:

            intent_scores.append(
                0.0
            )


            family_scores.append(
                0.0
            )


            grain_scores.append(
                0.0
            )


            sequence_scores.append(
                0.0
            )


            argument_scores.append(
                0.0
            )


            continue


        intent_scores.append(
            float(
                expected_plan.intent
                == actual_plan.intent
            )
        )


        family_scores.append(
            float(
                expected_plan.family
                == actual_plan.family
            )
        )


        grain_scores.append(
            float(
                expected_plan.target_grain
                == actual_plan.target_grain
            )
        )


        expected_actions = [
            step.action.name

            for step
            in expected_plan.steps
        ]


        actual_actions = [
            step.action.name

            for step
            in actual_plan.steps
        ]


        sequence_scores.append(
            _sequence_score(
                expected_actions,
                actual_actions,
            )
        )


        argument_scores.append(
            _plan_argument_score(
                expected_plan=(
                    expected_plan
                ),

                actual_plan=(
                    actual_plan
                ),
            )
        )


        if (
            _requirement_exact(
                expected_plan=(
                    expected_plan
                ),

                actual_plan=(
                    actual_plan
                ),
            )
        ):

            exact_requirement_ids.append(
                requirement_id
            )


    intent_accuracy = (
        _average(
            intent_scores
        )
    )


    family_accuracy = (
        _average(
            family_scores
        )
    )


    target_grain_accuracy = (
        _average(
            grain_scores
        )
    )


    tool_sequence_score = (
        _average(
            sequence_scores
        )
    )


    tool_argument_score = (
        _average(
            argument_scores
        )
    )


    # ========================================================
    # REAL PYTHON VALIDATOR
    # ========================================================

    validation = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    validator_acceptance = (
        1.0
        if validation.valid
        else 0.0
    )


    # ========================================================
    # OVERALL
    # ========================================================

    overall = _score(
        (
            requirement_coverage_f1
            * WEIGHTS[
                "requirement_coverage"
            ]
        )
        + (
            intent_accuracy
            * WEIGHTS[
                "intent"
            ]
        )
        + (
            family_accuracy
            * WEIGHTS[
                "family"
            ]
        )
        + (
            target_grain_accuracy
            * WEIGHTS[
                "target_grain"
            ]
        )
        + (
            tool_sequence_score
            * WEIGHTS[
                "tool_sequence"
            ]
        )
        + (
            tool_argument_score
            * WEIGHTS[
                "tool_arguments"
            ]
        )
        + (
            validator_acceptance
            * WEIGHTS[
                "validator_acceptance"
            ]
        )
    )


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    missing_requirement_ids = sorted(
        expected_ids
        - actual_ids
    )


    extra_requirement_ids = sorted(
        actual_ids
        - expected_ids
    )


    invalid_requirement_ids = sorted(
        {
            issue.requirement_id

            for issue
            in validation.issues

            if (
                issue.requirement_id
                is not None
            )
        }
    )


    validator_issue_codes = [
        issue.code

        for issue
        in validation.issues
    ]


    return (
        AnalyticalPlannerScore(
            scorer_version=(
                ANALYTICAL_PLANNER_SCORER_VERSION
            ),

            overall=(
                overall
            ),

            metrics=(
                AnalyticalPlannerScoreMetrics(
                    requirement_coverage_f1=(
                        requirement_coverage_f1
                    ),

                    intent_accuracy=(
                        intent_accuracy
                    ),

                    family_accuracy=(
                        family_accuracy
                    ),

                    target_grain_accuracy=(
                        target_grain_accuracy
                    ),

                    tool_sequence_score=(
                        tool_sequence_score
                    ),

                    tool_argument_score=(
                        tool_argument_score
                    ),

                    validator_acceptance=(
                        validator_acceptance
                    ),
                )
            ),

            diagnostics=(
                AnalyticalPlannerScoreDiagnostics(
                    missing_requirement_ids=(
                        missing_requirement_ids
                    ),

                    extra_requirement_ids=(
                        extra_requirement_ids
                    ),

                    invalid_requirement_ids=(
                        invalid_requirement_ids
                    ),

                    validator_issue_codes=(
                        validator_issue_codes
                    ),

                    exact_requirement_ids=(
                        sorted(
                            exact_requirement_ids
                        )
                    ),
                )
            ),
        )
    )
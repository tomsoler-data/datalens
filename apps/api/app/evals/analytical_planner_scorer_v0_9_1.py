from __future__ import annotations

import re

from typing import Any

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

from app.evals.analytical_planner_validator_v0_9_1 import (
    validate_analytical_planner_candidate,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_SCORER_VERSION = (
    "analytical_planner_scorer_v0.9.1"
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
        0.15,

    "tool_arguments":
        0.15,

    "validator_acceptance":
        0.15,

    "parsimony":
        0.10,
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

class AnalyticalPlannerScoreMetricsV091(
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

    parsimony_score: float


# ============================================================
# DIAGNOSTICS
# ============================================================

class AnalyticalPlannerScoreDiagnosticsV091(
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

    extra_step_count: int


# ============================================================
# RESULT
# ============================================================

class AnalyticalPlannerScoreV091(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    scorer_version: str

    overall: float

    metrics: (
        AnalyticalPlannerScoreMetricsV091
    )

    diagnostics: (
        AnalyticalPlannerScoreDiagnosticsV091
    )


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
# SET F1
# ============================================================

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
# LCS ALIGNMENT
# ============================================================

def _lcs_alignment(
    expected: list[str],
    actual: list[str],
) -> list[
    tuple[
        int,
        int,
    ]
]:
    """
    Return index pairs corresponding to one deterministic
    longest common subsequence.

    Example:

        expected:
            derive_metric
            compare_groups

        actual:
            aggregate
            derive_metric
            compare_groups

    returns:

        (0, 1)
        (1, 2)

    This allows argument scoring to compare corresponding
    analytical actions even when an extra action was inserted.
    """

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


    # ========================================================
    # BACKTRACK
    # ========================================================

    pairs: list[
        tuple[
            int,
            int,
        ]
    ] = []


    i = len(
        expected
    )


    j = len(
        actual
    )


    while (
        i > 0
        and j > 0
    ):

        if (
            expected[
                i - 1
            ]
            == actual[
                j - 1
            ]
        ):

            pairs.append(
                (
                    i - 1,
                    j - 1,
                )
            )


            i -= 1

            j -= 1


        elif (
            matrix[
                i - 1
            ][
                j
            ]
            >= matrix[
                i
            ][
                j - 1
            ]
        ):

            i -= 1


        else:

            j -= 1


    pairs.reverse()


    return pairs


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


    lcs_length = len(
        _lcs_alignment(
            expected,
            actual,
        )
    )


    return _score(
        (
            2.0
            * lcs_length
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
# FORMULA NORMALIZATION
# ============================================================

def _canonicalize_formula(
    formula: str,
    *,
    inputs: list[str],
) -> str:
    """
    Normalize a declarative derive_metric formula.

    The scorer deliberately treats:

        conversions / visits

    and:

        marketing.conversions / marketing.visits

    as equivalent when those qualified references correspond
    exactly to declared derive_metric inputs.

    This is evaluation normalization only.

    It does NOT parse or execute the formula.
    """

    normalized = (
        formula
        .strip()
        .lower()
    )


    # ========================================================
    # Replace known qualified analytical references with their
    # unqualified column names.
    #
    # Longest names first prevents accidental partial
    # replacements.
    # ========================================================

    qualified_inputs = sorted(
        {
            input_reference.strip().lower()

            for input_reference
            in inputs

            if (
                "."
                in input_reference
            )
        },
        key=len,
        reverse=True,
    )


    for qualified in qualified_inputs:

        unqualified = (
            qualified
            .rsplit(
                ".",
                1,
            )[
                -1
            ]
        )


        normalized = re.sub(
            (
                r"(?<![a-z0-9_])"
                + re.escape(
                    qualified
                )
                + r"(?![a-z0-9_])"
            ),
            unqualified,
            normalized,
        )


    # ========================================================
    # Whitespace is not analytically meaningful for the simple
    # declarative formulas currently supported.
    # ========================================================

    normalized = re.sub(
        r"\s+",
        "",
        normalized,
    )


    return normalized


def _formula_equal(
    *,
    expected_formula: str,
    actual_formula: str,
    expected_inputs: list[str],
    actual_inputs: list[str],
) -> bool:

    all_inputs = [
        *expected_inputs,
        *actual_inputs,
    ]


    expected_normalized = (
        _canonicalize_formula(
            expected_formula,
            inputs=all_inputs,
        )
    )


    actual_normalized = (
        _canonicalize_formula(
            actual_formula,
            inputs=all_inputs,
        )
    )


    return (
        expected_normalized
        == actual_normalized
    )


# ============================================================
# TOOL ARGUMENT SCORE
# ============================================================

def _tool_argument_score(
    *,
    expected_action,
    actual_action,
) -> float:

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


    expected.pop(
        "name",
        None,
    )


    actual.pop(
        "name",
        None,
    )


    # ========================================================
    # ASSOCIATION — SYMMETRIC
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
    # AGGREGATION — LIST ORDER IRRELEVANT
    # ========================================================

    if (
        tool_name
        == "aggregate"
    ):

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


        groups_match = (
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
                    metrics_match
                )
                + float(
                    groups_match
                )
            )
            / 2.0
        )


    # ========================================================
    # DERIVED METRIC
    # ========================================================

    if (
        tool_name
        == "derive_metric"
    ):

        expected_inputs = (
            expected[
                "inputs"
            ]
        )


        actual_inputs = (
            actual[
                "inputs"
            ]
        )


        inputs_match = (
            _unordered_list_equal(
                expected_inputs,
                actual_inputs,
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
            _formula_equal(
                expected_formula=(
                    expected[
                        "formula"
                    ]
                ),

                actual_formula=(
                    actual[
                        "formula"
                    ]
                ),

                expected_inputs=(
                    expected_inputs
                ),

                actual_inputs=(
                    actual_inputs
                ),
            )
        )


        return _score(
            (
                float(
                    inputs_match
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
    # ENTITY OUTLIERS — METRIC ORDER IRRELEVANT
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
    # OTHER TOOLS
    # ========================================================

    keys = (
        set(
            expected
        )
        | set(
            actual
        )
    )


    if not keys:
        return 1.0


    matches = 0.0


    for key in keys:

        if (
            key
            not in expected
            or key
            not in actual
        ):
            continue


        if (
            _normalize_text(
                expected[
                    key
                ]
            )
            == _normalize_text(
                actual[
                    key
                ]
            )
        ):

            matches += 1.0


    return _score(
        matches
        / len(
            keys
        )
    )


# ============================================================
# PLAN ARGUMENT SCORE — LCS ALIGNED
# ============================================================

def _plan_argument_score(
    *,
    expected_plan: AnalyticalRequirementPlan,
    actual_plan: AnalyticalRequirementPlan,
) -> float:
    """
    Argument scoring is performed only between corresponding
    tool calls selected by LCS alignment.

    Missing expected analytical actions receive zero credit.

    Extra actual actions are handled separately by:

    - tool_sequence_score
    - parsimony_score

    This prevents one extra action from shifting every later
    argument comparison.
    """

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


    if not expected_actions:

        return (
            1.0
            if not actual_actions
            else 0.0
        )


    alignment = (
        _lcs_alignment(
            expected_actions,
            actual_actions,
        )
    )


    matched_by_expected_index = {
        expected_index:
            actual_index

        for (
            expected_index,
            actual_index,
        )
        in alignment
    }


    scores: list[
        float
    ] = []


    for expected_index in range(
        len(
            expected_plan.steps
        )
    ):

        actual_index = (
            matched_by_expected_index.get(
                expected_index
            )
        )


        if actual_index is None:

            scores.append(
                0.0
            )


            continue


        scores.append(
            _tool_argument_score(
                expected_action=(
                    expected_plan
                    .steps[
                        expected_index
                    ]
                    .action
                ),

                actual_action=(
                    actual_plan
                    .steps[
                        actual_index
                    ]
                    .action
                ),
            )
        )


    return _average(
        scores
    )


# ============================================================
# PARSIMONY
# ============================================================

def _plan_parsimony_score(
    *,
    expected_plan: AnalyticalRequirementPlan,
    actual_plan: AnalyticalRequirementPlan,
) -> float:
    """
    Penalize only unnecessary extra actions.

    Under-planning is already penalized by tool sequence and
    argument coverage, so parsimony does not double-penalize
    missing steps.

    Examples:

        expected 2 / actual 2 -> 1.0
        expected 2 / actual 3 -> 0.667
        expected 2 / actual 4 -> 0.5
        expected 2 / actual 1 -> 1.0
    """

    expected_count = len(
        expected_plan.steps
    )


    actual_count = len(
        actual_plan.steps
    )


    if (
        actual_count
        <= expected_count
    ):
        return 1.0


    if actual_count == 0:
        return 1.0


    return _score(
        expected_count
        / actual_count
    )


# ============================================================
# EXACT REQUIREMENT
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


    if (
        _plan_argument_score(
            expected_plan=expected_plan,
            actual_plan=actual_plan,
        )
        != 1.0
    ):
        return False


    return True


# ============================================================
# PUBLIC SCORER
# ============================================================

def score_analytical_planner_candidate(
    *,
    candidate: AnalyticalPlannerCandidate,
    expected: AnalyticalPlannerCandidate,
    planner_input: AnalyticalPlannerInput,
) -> AnalyticalPlannerScoreV091:

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
    # PER-REQUIREMENT METRICS
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


    parsimony_scores: list[
        float
    ] = []


    exact_requirement_ids: list[
        str
    ] = []


    extra_step_count = 0


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


            parsimony_scores.append(
                1.0
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


        parsimony_scores.append(
            _plan_parsimony_score(
                expected_plan=(
                    expected_plan
                ),

                actual_plan=(
                    actual_plan
                ),
            )
        )


        extra_step_count += max(
            0,
            (
                len(
                    actual_plan.steps
                )
                - len(
                    expected_plan.steps
                )
            ),
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


    parsimony_score = (
        _average(
            parsimony_scores
        )
    )


    # ========================================================
    # VALIDATOR v0.9.1
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
        + (
            parsimony_score
            * WEIGHTS[
                "parsimony"
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
        AnalyticalPlannerScoreV091(
            scorer_version=(
                ANALYTICAL_PLANNER_SCORER_VERSION
            ),

            overall=(
                overall
            ),

            metrics=(
                AnalyticalPlannerScoreMetricsV091(
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

                    parsimony_score=(
                        parsimony_score
                    ),
                )
            ),

            diagnostics=(
                AnalyticalPlannerScoreDiagnosticsV091(
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

                    extra_step_count=(
                        extra_step_count
                    ),
                )
            ),
        )
    )
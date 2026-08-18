from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.evals.guardrails import (
    evaluate_guardrails,
)

from app.evals.schemas import (
    AnalyticalCandidate,
    AnalyticalEvalCase,
)


# ============================================================
# VERSION
# ============================================================

SCORER_RULE_VERSION = (
    "analytical_scorer_v0.2"
)


# ============================================================
# COLUMN ARGUMENTS
# ============================================================

COLUMN_ARGUMENT_KEYS = {
    "column",
    "columns",
    "target",
    "targets",
    "value",
    "values",
    "metric",
    "metrics",
    "group_by",
    "groupby",
    "by",
    "entity",
    "date",
    "date_column",
    "time_column",
    "inputs",
}


UNORDERED_LIST_ARGUMENTS = {
    "columns",
    "inputs",
    "metrics",
    "targets",
    "values",
    "group_by",
}


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_text(
    value: str | None,
) -> str:
    if value is None:
        return ""

    return " ".join(
        value
        .strip()
        .lower()
        .split()
    )


def _exact_score(
    expected: str | None,
    actual: str | None,
) -> float:
    return (
        1.0
        if _normalize_text(
            expected,
        )
        == _normalize_text(
            actual,
        )
        else 0.0
    )


# ============================================================
# SET F1
# ============================================================

def _set_f1(
    expected: Iterable[str],
    actual: Iterable[str],
) -> float:
    expected_set = {
        _normalize_text(
            value,
        )
        for value in expected
        if _normalize_text(
            value,
        )
    }

    actual_set = {
        _normalize_text(
            value,
        )
        for value in actual
        if _normalize_text(
            value,
        )
    }

    if (
        not expected_set
        and not actual_set
    ):
        return 1.0

    if (
        not expected_set
        or not actual_set
    ):
        return 0.0

    overlap = len(
        expected_set
        & actual_set
    )

    precision = (
        overlap
        / len(
            actual_set,
        )
    )

    recall = (
        overlap
        / len(
            expected_set,
        )
    )

    if (
        precision
        + recall
        == 0
    ):
        return 0.0

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


# ============================================================
# VALUE COMPARISON
# ============================================================

def _normalize_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        str,
    ):
        return _normalize_text(
            value,
        )

    if isinstance(
        value,
        list,
    ):
        return [
            _normalize_value(
                item,
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key,
            ): _normalize_value(
                item,
            )
            for key, item
            in value.items()
        }

    return value


def _values_equivalent(
    *,
    argument_name: str,
    expected: Any,
    actual: Any,
) -> bool:
    normalized_name = (
        _normalize_text(
            argument_name,
        )
    )

    normalized_expected = (
        _normalize_value(
            expected,
        )
    )

    normalized_actual = (
        _normalize_value(
            actual,
        )
    )

    if (
        normalized_name
        in UNORDERED_LIST_ARGUMENTS
        and isinstance(
            normalized_expected,
            list,
        )
        and isinstance(
            normalized_actual,
            list,
        )
    ):
        return (
            set(
                normalized_expected,
            )
            == set(
                normalized_actual,
            )
        )

    return (
        normalized_expected
        == normalized_actual
    )


# ============================================================
# TOOL ARGUMENT SCORE
# ============================================================

def _single_tool_argument_score(
    *,
    tool_name: str,
    expected_arguments: dict[
        str,
        Any,
    ],
    actual_arguments: dict[
        str,
        Any,
    ],
) -> float:
    if not expected_arguments:
        return 1.0

    normalized_tool = (
        _normalize_text(
            tool_name,
        )
    )

    matched_arguments = 0

    processed_arguments: set[
        str
    ] = set()


    # ========================================================
    # ASSOCIATION IS SYMMETRIC
    # ========================================================

    if (
        normalized_tool
        == "measure_association"

        and {
            "target",
            "value",
        }.issubset(
            expected_arguments,
        )

        and {
            "target",
            "value",
        }.issubset(
            actual_arguments,
        )
    ):
        expected_pair = {
            _normalize_text(
                str(
                    expected_arguments[
                        "target"
                    ]
                )
            ),
            _normalize_text(
                str(
                    expected_arguments[
                        "value"
                    ]
                )
            ),
        }

        actual_pair = {
            _normalize_text(
                str(
                    actual_arguments[
                        "target"
                    ]
                )
            ),
            _normalize_text(
                str(
                    actual_arguments[
                        "value"
                    ]
                )
            ),
        }

        if (
            expected_pair
            == actual_pair
        ):
            matched_arguments += 2

        processed_arguments.update(
            {
                "target",
                "value",
            }
        )


    # ========================================================
    # OTHER ARGUMENTS
    # ========================================================

    for (
        argument_name,
        expected_value,
    ) in expected_arguments.items():

        if (
            argument_name
            in processed_arguments
        ):
            continue


        # ----------------------------------------------------
        # IMPORTANT v0.2 FIX
        #
        # Some typed Pydantic tool models omit optional values
        # when converted with exclude_none=True.
        #
        # Therefore:
        #
        #     {"group_by": None}
        #
        # and
        #
        #     {}
        #
        # are semantically equivalent when the benchmark
        # explicitly expects None.
        # ----------------------------------------------------

        if (
            argument_name
            not in actual_arguments
        ):
            if expected_value is None:
                matched_arguments += 1

            continue


        actual_value = (
            actual_arguments[
                argument_name
            ]
        )


        if _values_equivalent(
            argument_name=(
                argument_name
            ),

            expected=(
                expected_value
            ),

            actual=(
                actual_value
            ),
        ):
            matched_arguments += 1


    return (
        matched_arguments
        / len(
            expected_arguments,
        )
    )


def _tool_argument_score(
    case: AnalyticalEvalCase,
    candidate: AnalyticalCandidate,
) -> float:
    requirements = (
        case
        .expected
        .required_tool_arguments
    )

    if not requirements:
        return 1.0


    scores: list[
        float
    ] = []


    for (
        expected_tool,
        expected_arguments,
    ) in requirements.items():

        matching_calls = [
            call
            for call
            in candidate.tool_calls
            if (
                _normalize_text(
                    call.name,
                )
                == _normalize_text(
                    expected_tool,
                )
            )
        ]


        if not matching_calls:
            scores.append(
                0.0,
            )

            continue


        best_call_score = max(
            _single_tool_argument_score(
                tool_name=(
                    expected_tool
                ),

                expected_arguments=(
                    expected_arguments
                ),

                actual_arguments=(
                    call.arguments
                ),
            )

            for call
            in matching_calls
        )


        scores.append(
            best_call_score,
        )


    return (
        sum(
            scores,
        )
        / len(
            scores,
        )
    )


# ============================================================
# COLUMN REFERENCES
# ============================================================

def _candidate_column_references(
    candidate: AnalyticalCandidate,
) -> set[str]:
    references = {
        _normalize_text(
            column,
        )
        for column
        in candidate.relevant_columns
        if _normalize_text(
            column,
        )
    }


    derived_outputs: set[
        str
    ] = set()


    for call in candidate.tool_calls:
        if (
            _normalize_text(
                call.name,
            )
            == "derive_metric"
        ):
            output = (
                call.arguments.get(
                    "output",
                )
            )

            if isinstance(
                output,
                str,
            ):
                derived_outputs.add(
                    _normalize_text(
                        output,
                    )
                )


    def walk(
        key: str | None,
        value: Any,
    ) -> None:
        normalized_key = (
            _normalize_text(
                key,
            )
            if key is not None
            else ""
        )


        if isinstance(
            value,
            dict,
        ):
            for (
                child_key,
                child_value,
            ) in value.items():
                walk(
                    str(
                        child_key,
                    ),
                    child_value,
                )

            return


        if isinstance(
            value,
            list,
        ):
            for item in value:
                walk(
                    key,
                    item,
                )

            return


        if (
            normalized_key
            in COLUMN_ARGUMENT_KEYS

            and isinstance(
                value,
                str,
            )
        ):
            references.add(
                _normalize_text(
                    value,
                )
            )


    for call in candidate.tool_calls:
        for (
            key,
            value,
        ) in call.arguments.items():
            walk(
                key,
                value,
            )


    references -= (
        derived_outputs
    )


    return {
        reference
        for reference
        in references
        if reference
    }


# ============================================================
# PLAN CONSISTENCY
# ============================================================

INTENT_FAMILY_MAP = {
    "aggregate_metric":
        "aggregation",

    "compare_groups":
        "group_comparison",

    "measure_relationship":
        "association",

    "time_series_analysis":
        "time_series",

    "distribution_analysis":
        "distribution",

    "entity_anomaly_analysis":
        "entity_outlier",

    "data_quality_analysis":
        "data_quality",
}


def _plan_consistency(
    candidate: AnalyticalCandidate,
) -> tuple[
    float,
    tuple[str, ...],
]:
    checks: list[
        bool
    ] = []

    issues: list[
        str
    ] = []


    normalized_intent = (
        _normalize_text(
            candidate.intent,
        )
    )

    normalized_family = (
        _normalize_text(
            candidate.family,
        )
    )


    # ========================================================
    # INTENT ↔ FAMILY
    # ========================================================

    expected_family = (
        INTENT_FAMILY_MAP.get(
            normalized_intent,
        )
    )

    if (
        expected_family
        is not None
    ):
        coherent = (
            normalized_family
            == expected_family
        )

        checks.append(
            coherent,
        )

        if not coherent:
            issues.append(
                "intent_family_mismatch"
            )


    # ========================================================
    # RELEVANT COLUMNS
    # ========================================================

    relevant_columns = {
        _normalize_text(
            column,
        )
        for column
        in candidate.relevant_columns
    }


    # ========================================================
    # TOOL-SPECIFIC CONSISTENCY
    # ========================================================

    for call in candidate.tool_calls:
        tool_name = (
            _normalize_text(
                call.name,
            )
        )

        arguments = (
            call.arguments
        )


        # ----------------------------------------------------
        # BUILD ENTITY VIEW
        # ----------------------------------------------------

        if (
            tool_name
            == "build_entity_view"
        ):
            tool_entity = (
                arguments.get(
                    "entity",
                )
            )

            entity_matches = (
                isinstance(
                    tool_entity,
                    str,
                )
                and _normalize_text(
                    candidate.entity,
                )
                == _normalize_text(
                    tool_entity,
                )
            )

            checks.append(
                entity_matches,
            )

            if not entity_matches:
                issues.append(
                    "entity_missing_or_inconsistent_with_build_entity_view"
                )


            target_exists = bool(
                _normalize_text(
                    candidate.target_grain,
                )
            )

            checks.append(
                target_exists,
            )

            if not target_exists:
                issues.append(
                    "target_grain_missing_after_build_entity_view"
                )


        # ----------------------------------------------------
        # ENTITY OUTLIERS
        # ----------------------------------------------------

        if (
            tool_name
            == "detect_entity_outliers"
        ):
            tool_entity = (
                arguments.get(
                    "entity",
                )
            )

            entity_matches = (
                isinstance(
                    tool_entity,
                    str,
                )
                and _normalize_text(
                    candidate.entity,
                )
                == _normalize_text(
                    tool_entity,
                )
            )

            checks.append(
                entity_matches,
            )

            if not entity_matches:
                issues.append(
                    "entity_missing_or_inconsistent_with_entity_outlier_tool"
                )


            target_exists = bool(
                _normalize_text(
                    candidate.target_grain,
                )
            )

            checks.append(
                target_exists,
            )

            if not target_exists:
                issues.append(
                    "target_grain_missing_for_entity_outlier_analysis"
                )


            metrics = (
                arguments.get(
                    "metrics",
                    [],
                )
            )

            if isinstance(
                metrics,
                list,
            ):
                metrics_are_relevant = all(
                    _normalize_text(
                        str(
                            metric,
                        )
                    )
                    in relevant_columns

                    for metric
                    in metrics
                )

                checks.append(
                    metrics_are_relevant,
                )

                if not metrics_are_relevant:
                    issues.append(
                        "entity_outlier_metric_not_declared_relevant"
                    )


        # ----------------------------------------------------
        # COMPARE GROUPS
        # ----------------------------------------------------

        if (
            tool_name
            == "compare_groups"
        ):
            target = (
                arguments.get(
                    "target",
                )
            )

            group_by = (
                arguments.get(
                    "group_by",
                )
            )

            references_are_relevant = (
                isinstance(
                    target,
                    str,
                )
                and isinstance(
                    group_by,
                    str,
                )
                and _normalize_text(
                    target,
                )
                in relevant_columns
                and _normalize_text(
                    group_by,
                )
                in relevant_columns
            )

            checks.append(
                references_are_relevant,
            )

            if not references_are_relevant:
                issues.append(
                    "compare_groups_arguments_not_declared_relevant"
                )


        # ----------------------------------------------------
        # ASSOCIATION
        # ----------------------------------------------------

        if (
            tool_name
            == "measure_association"
        ):
            target = (
                arguments.get(
                    "target",
                )
            )

            value = (
                arguments.get(
                    "value",
                )
            )

            references_are_relevant = (
                isinstance(
                    target,
                    str,
                )
                and isinstance(
                    value,
                    str,
                )
                and _normalize_text(
                    target,
                )
                in relevant_columns
                and _normalize_text(
                    value,
                )
                in relevant_columns
            )

            checks.append(
                references_are_relevant,
            )

            if not references_are_relevant:
                issues.append(
                    "association_arguments_not_declared_relevant"
                )


        # ----------------------------------------------------
        # TIME SERIES
        # ----------------------------------------------------

        if (
            tool_name
            == "analyze_time_series"
        ):
            date = (
                arguments.get(
                    "date",
                )
            )

            target = (
                arguments.get(
                    "target",
                )
            )

            references_are_relevant = (
                isinstance(
                    date,
                    str,
                )
                and isinstance(
                    target,
                    str,
                )
                and _normalize_text(
                    date,
                )
                in relevant_columns
                and _normalize_text(
                    target,
                )
                in relevant_columns
            )

            checks.append(
                references_are_relevant,
            )

            if not references_are_relevant:
                issues.append(
                    "time_series_arguments_not_declared_relevant"
                )


    # ========================================================
    # ENTITY ANALYSIS CONTRACT
    # ========================================================

    entity_analysis = (
        normalized_intent
        == "entity_anomaly_analysis"

        or normalized_family
        == "entity_outlier"
    )


    if entity_analysis:
        entity_present = bool(
            _normalize_text(
                candidate.entity,
            )
        )

        target_grain_present = bool(
            _normalize_text(
                candidate.target_grain,
            )
        )

        checks.append(
            entity_present,
        )

        checks.append(
            target_grain_present,
        )

        if not entity_present:
            issues.append(
                "entity_analysis_without_entity"
            )

        if not target_grain_present:
            issues.append(
                "entity_analysis_without_target_grain"
            )


    if not checks:
        return (
            1.0,
            tuple(),
        )


    score = (
        sum(
            1.0
            if check
            else 0.0

            for check
            in checks
        )
        / len(
            checks,
        )
    )


    return (
        score,
        tuple(
            sorted(
                set(
                    issues,
                )
            )
        ),
    )


# ============================================================
# PARSIMONY
# ============================================================

def _plan_parsimony(
    case: AnalyticalEvalCase,
    candidate: AnalyticalCandidate,
) -> tuple[
    float,
    tuple[str, ...],
    tuple[str, ...],
]:
    expected_tools = {
        _normalize_text(
            tool,
        )
        for tool
        in case.expected.acceptable_tools
    }


    actual_tools = [
        _normalize_text(
            call.name,
        )
        for call
        in candidate.tool_calls
    ]


    if not actual_tools:
        if not expected_tools:
            return (
                1.0,
                tuple(),
                tuple(),
            )

        return (
            0.0,
            tuple(),
            tuple(),
        )


    covered_required_tools = {
        tool
        for tool
        in expected_tools
        if tool
        in actual_tools
    }


    score = min(
        1.0,
        (
            len(
                covered_required_tools,
            )
            / len(
                actual_tools,
            )
        ),
    )


    extra_tool_calls = tuple(
        sorted(
            tool
            for tool
            in actual_tools
            if tool
            not in expected_tools
        )
    )


    seen: set[
        str
    ] = set()

    redundant_tool_calls: list[
        str
    ] = []


    for tool in actual_tools:
        if tool in seen:
            redundant_tool_calls.append(
                tool,
            )

        else:
            seen.add(
                tool,
            )


    return (
        score,
        extra_tool_calls,
        tuple(
            sorted(
                redundant_tool_calls,
            )
        ),
    )


# ============================================================
# RESULT
# ============================================================

@dataclass(
    frozen=True,
)
class AnalyticalScoreV02:
    scorer_rule_version: str

    case_id: str

    intent: float
    entity: float
    grain: float
    relevant_columns: float
    family: float

    tool_selection: float
    tool_arguments: float
    plan_consistency: float
    parsimony: float

    constraint_compliance: float
    guardrails: float

    invented_columns: tuple[
        str,
        ...,
    ]

    invented_tools: tuple[
        str,
        ...,
    ]

    forbidden_tools_used: tuple[
        str,
        ...,
    ]

    forbidden_assumptions_used: tuple[
        str,
        ...,
    ]

    extra_tool_calls: tuple[
        str,
        ...,
    ]

    redundant_tool_calls: tuple[
        str,
        ...,
    ]

    consistency_issues: tuple[
        str,
        ...,
    ]

    required_guardrails: tuple[
        str,
        ...,
    ]

    passed_guardrails: tuple[
        str,
        ...,
    ]

    failed_guardrails: tuple[
        str,
        ...,
    ]

    comprehension: float
    planning: float
    reliability: float

    overall: float


    def as_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "scorer_rule_version":
                self.scorer_rule_version,

            "case_id":
                self.case_id,

            "metrics": {
                "intent":
                    self.intent,

                "entity":
                    self.entity,

                "grain":
                    self.grain,

                "relevant_columns":
                    self.relevant_columns,

                "family":
                    self.family,

                "tool_selection":
                    self.tool_selection,

                "tool_arguments":
                    self.tool_arguments,

                "plan_consistency":
                    self.plan_consistency,

                "parsimony":
                    self.parsimony,

                "constraint_compliance":
                    self.constraint_compliance,

                "guardrails":
                    self.guardrails,
            },

            "capabilities": {
                "comprehension":
                    self.comprehension,

                "planning":
                    self.planning,

                "reliability":
                    self.reliability,
            },

            "diagnostics": {
                "invented_columns":
                    list(
                        self.invented_columns,
                    ),

                "invented_tools":
                    list(
                        self.invented_tools,
                    ),

                "forbidden_tools_used":
                    list(
                        self.forbidden_tools_used,
                    ),

                "forbidden_assumptions_used":
                    list(
                        self.forbidden_assumptions_used,
                    ),

                "extra_tool_calls":
                    list(
                        self.extra_tool_calls,
                    ),

                "redundant_tool_calls":
                    list(
                        self.redundant_tool_calls,
                    ),

                "consistency_issues":
                    list(
                        self.consistency_issues,
                    ),

                "required_guardrails":
                    list(
                        self.required_guardrails,
                    ),

                "passed_guardrails":
                    list(
                        self.passed_guardrails,
                    ),

                "failed_guardrails":
                    list(
                        self.failed_guardrails,
                    ),
            },

            "overall":
                self.overall,
        }


# ============================================================
# MAIN SCORER
# ============================================================

def score_candidate_v0_2(
    case: AnalyticalEvalCase,
    candidate: AnalyticalCandidate,
) -> AnalyticalScoreV02:
    expected = (
        case.expected
    )


    # ========================================================
    # COMPREHENSION
    # ========================================================

    intent_score = _exact_score(
        expected.intent,
        candidate.intent,
    )


    entity_score = _exact_score(
        expected.entity,
        candidate.entity,
    )


    current_grain_score = (
        _exact_score(
            expected.current_grain,
            candidate.current_grain,
        )
    )


    target_grain_score = (
        _exact_score(
            expected.target_grain,
            candidate.target_grain,
        )
    )


    grain_score = (
        current_grain_score
        + target_grain_score
    ) / 2


    relevant_columns_score = (
        _set_f1(
            expected.relevant_columns,
            candidate.relevant_columns,
        )
    )


    family_score = _exact_score(
        expected.family,
        candidate.family,
    )


    # ========================================================
    # TOOL SELECTION
    # ========================================================

    actual_tools = [
        call.name
        for call
        in candidate.tool_calls
    ]


    tool_selection_score = (
        _set_f1(
            expected.acceptable_tools,
            actual_tools,
        )
    )


    # ========================================================
    # TOOL ARGUMENTS
    # ========================================================

    tool_arguments_score = (
        _tool_argument_score(
            case,
            candidate,
        )
    )


    # ========================================================
    # INVENTED COLUMNS
    # ========================================================

    known_columns = {
        _normalize_text(
            column.name,
        )
        for dataset
        in case.datasets
        for column
        in dataset.columns
    }


    referenced_columns = (
        _candidate_column_references(
            candidate,
        )
    )


    invented_columns = tuple(
        sorted(
            referenced_columns
            - known_columns
        )
    )


    # ========================================================
    # INVENTED TOOLS
    # ========================================================

    available_tools = {
        _normalize_text(
            tool,
        )
        for tool
        in case.available_tools
    }


    invented_tools = tuple(
        sorted(
            {
                _normalize_text(
                    call.name,
                )
                for call
                in candidate.tool_calls
                if (
                    _normalize_text(
                        call.name,
                    )
                    not in available_tools
                )
            }
        )
    )


    # ========================================================
    # FORBIDDEN TOOLS
    # ========================================================

    forbidden_tool_set = {
        _normalize_text(
            tool,
        )
        for tool
        in expected.forbidden_tools
    }


    forbidden_tools_used = tuple(
        sorted(
            {
                _normalize_text(
                    call.name,
                )
                for call
                in candidate.tool_calls
                if (
                    _normalize_text(
                        call.name,
                    )
                    in forbidden_tool_set
                )
            }
        )
    )


    # ========================================================
    # FORBIDDEN ASSUMPTIONS
    # ========================================================

    forbidden_assumption_set = {
        _normalize_text(
            assumption,
        )
        for assumption
        in expected.forbidden_assumptions
    }


    forbidden_assumptions_used = tuple(
        sorted(
            {
                _normalize_text(
                    assumption,
                )
                for assumption
                in candidate.assumptions
                if (
                    _normalize_text(
                        assumption,
                    )
                    in forbidden_assumption_set
                )
            }
        )
    )


    # ========================================================
    # CONSTRAINT COMPLIANCE
    # ========================================================

    constraint_compliance = 1.0


    if invented_columns:
        constraint_compliance -= 0.25


    if invented_tools:
        constraint_compliance -= 0.25


    if forbidden_tools_used:
        constraint_compliance -= 0.25


    if forbidden_assumptions_used:
        constraint_compliance -= 0.25


    constraint_compliance = max(
        0.0,
        constraint_compliance,
    )


    # ========================================================
    # PLAN CONSISTENCY
    # ========================================================

    (
        plan_consistency_score,
        consistency_issues,
    ) = _plan_consistency(
        candidate,
    )


    # ========================================================
    # PARSIMONY
    # ========================================================

    (
        parsimony_score,
        extra_tool_calls,
        redundant_tool_calls,
    ) = _plan_parsimony(
        case,
        candidate,
    )


    # ========================================================
    # POSITIVE GUARDRAILS
    # ========================================================

    guardrail_assessment = (
        evaluate_guardrails(
            case=case,
            candidate=candidate,
        )
    )


    guardrail_score = (
        guardrail_assessment.score
    )


    # ========================================================
    # CAPABILITY VIEWS
    # ========================================================

    comprehension = (
        (
            intent_score
            + entity_score
            + grain_score
            + relevant_columns_score
            + family_score
        )
        / 5
    )


    planning = (
        (
            tool_selection_score
            + tool_arguments_score
            + plan_consistency_score
            + parsimony_score
        )
        / 4
    )


    reliability = (
        (
            constraint_compliance
            + guardrail_score
        )
        / 2
    )


    # ========================================================
    # WEIGHTED OVERALL
    # ========================================================

    weights = {
        "intent":
            0.10,

        "entity":
            0.07,

        "grain":
            0.08,

        "relevant_columns":
            0.12,

        "family":
            0.08,

        "tool_selection":
            0.12,

        "tool_arguments":
            0.10,

        "plan_consistency":
            0.08,

        "parsimony":
            0.05,

        "constraint_compliance":
            0.10,

        "guardrails":
            0.10,
    }


    overall = (
        intent_score
        * weights[
            "intent"
        ]

        + entity_score
        * weights[
            "entity"
        ]

        + grain_score
        * weights[
            "grain"
        ]

        + relevant_columns_score
        * weights[
            "relevant_columns"
        ]

        + family_score
        * weights[
            "family"
        ]

        + tool_selection_score
        * weights[
            "tool_selection"
        ]

        + tool_arguments_score
        * weights[
            "tool_arguments"
        ]

        + plan_consistency_score
        * weights[
            "plan_consistency"
        ]

        + parsimony_score
        * weights[
            "parsimony"
        ]

        + constraint_compliance
        * weights[
            "constraint_compliance"
        ]

        + guardrail_score
        * weights[
            "guardrails"
        ]
    )


    return AnalyticalScoreV02(
        scorer_rule_version=(
            SCORER_RULE_VERSION
        ),

        case_id=case.case_id,

        intent=intent_score,

        entity=entity_score,

        grain=grain_score,

        relevant_columns=(
            relevant_columns_score
        ),

        family=family_score,

        tool_selection=(
            tool_selection_score
        ),

        tool_arguments=(
            tool_arguments_score
        ),

        plan_consistency=(
            plan_consistency_score
        ),

        parsimony=(
            parsimony_score
        ),

        constraint_compliance=(
            constraint_compliance
        ),

        guardrails=(
            guardrail_score
        ),

        invented_columns=(
            invented_columns
        ),

        invented_tools=(
            invented_tools
        ),

        forbidden_tools_used=(
            forbidden_tools_used
        ),

        forbidden_assumptions_used=(
            forbidden_assumptions_used
        ),

        extra_tool_calls=(
            extra_tool_calls
        ),

        redundant_tool_calls=(
            redundant_tool_calls
        ),

        consistency_issues=(
            consistency_issues
        ),

        required_guardrails=(
            guardrail_assessment
            .required_rules
        ),

        passed_guardrails=(
            guardrail_assessment
            .passed_rules
        ),

        failed_guardrails=(
            guardrail_assessment
            .failed_rules
        ),

        comprehension=(
            comprehension
        ),

        planning=(
            planning
        ),

        reliability=(
            reliability
        ),

        overall=overall,
    )
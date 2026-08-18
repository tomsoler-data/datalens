from __future__ import annotations

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
    Iterable,
)

from app.evals.schemas import (
    AnalyticalCandidate,
    AnalyticalEvalCase,
)


# ============================================================
# COLUMN ARGUMENT DISCOVERY
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
# ARGUMENT NORMALIZATION
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


# ============================================================
# TOOL ARGUMENT SCORE
# ============================================================


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


        best_call_score = 0.0


        for call in matching_calls:

            if not expected_arguments:
                best_call_score = 1.0

                break


            matched_arguments = 0


            for (
                argument_name,
                expected_value,
            ) in expected_arguments.items():

                if (
                    argument_name
                    not in call.arguments
                ):
                    continue


                actual_value = (
                    call.arguments[
                        argument_name
                    ]
                )


                if (
                    _normalize_value(
                        actual_value,
                    )
                    == _normalize_value(
                        expected_value,
                    )
                ):
                    matched_arguments += 1


            call_score = (
                matched_arguments
                / len(
                    expected_arguments,
                )
            )


            best_call_score = max(
                best_call_score,
                call_score,
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


    return {
        reference
        for reference
        in references
        if reference
    }


# ============================================================
# SCORE RESULT
# ============================================================


@dataclass(
    frozen=True,
)
class AnalyticalScore:
    case_id: str

    intent: float
    entity: float
    grain: float
    relevant_columns: float
    family: float
    tool_selection: float
    tool_arguments: float
    safety: float

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

    overall: float


    def as_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
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

                "safety":
                    self.safety,
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
                        self
                        .forbidden_tools_used,
                    ),

                "forbidden_assumptions_used":
                    list(
                        self
                        .forbidden_assumptions_used,
                    ),
            },

            "overall":
                self.overall,
        }


# ============================================================
# MAIN SCORER
# ============================================================


def score_candidate(
    case: AnalyticalEvalCase,
    candidate: AnalyticalCandidate,
) -> AnalyticalScore:
    expected = (
        case.expected
    )


    # --------------------------------------------------------
    # INTENT
    # --------------------------------------------------------

    intent_score = (
        _exact_score(
            expected.intent,
            candidate.intent,
        )
    )


    # --------------------------------------------------------
    # ENTITY
    # --------------------------------------------------------

    entity_score = (
        _exact_score(
            expected.entity,
            candidate.entity,
        )
    )


    # --------------------------------------------------------
    # GRAIN
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # COLUMNS
    # --------------------------------------------------------

    relevant_columns_score = (
        _set_f1(
            expected.relevant_columns,
            candidate.relevant_columns,
        )
    )


    # --------------------------------------------------------
    # ANALYTICAL FAMILY
    # --------------------------------------------------------

    family_score = (
        _exact_score(
            expected.family,
            candidate.family,
        )
    )


    # --------------------------------------------------------
    # TOOL SELECTION
    # --------------------------------------------------------

    expected_tools = (
        expected
        .acceptable_tools
    )


    actual_tools = [
        call.name
        for call
        in candidate.tool_calls
    ]


    tool_selection_score = (
        _set_f1(
            expected_tools,
            actual_tools,
        )
    )


    # --------------------------------------------------------
    # TOOL ARGUMENTS
    # --------------------------------------------------------

    tool_arguments_score = (
        _tool_argument_score(
            case,
            candidate,
        )
    )


    # --------------------------------------------------------
    # INVENTED COLUMNS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # INVENTED TOOLS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FORBIDDEN TOOLS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FORBIDDEN ASSUMPTIONS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    safety_score = 1.0


    if invented_columns:
        safety_score -= 0.25


    if invented_tools:
        safety_score -= 0.25


    if forbidden_tools_used:
        safety_score -= 0.25


    if forbidden_assumptions_used:
        safety_score -= 0.25


    safety_score = max(
        0.0,
        safety_score,
    )


    # --------------------------------------------------------
    # WEIGHTED SCORE
    # --------------------------------------------------------

    weights = {
        "intent":
            0.15,

        "entity":
            0.10,

        "grain":
            0.10,

        "relevant_columns":
            0.15,

        "family":
            0.10,

        "tool_selection":
            0.20,

        "tool_arguments":
            0.10,

        "safety":
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

        + safety_score
        * weights[
            "safety"
        ]
    )


    return AnalyticalScore(
        case_id=
            case.case_id,

        intent=
            intent_score,

        entity=
            entity_score,

        grain=
            grain_score,

        relevant_columns=
            relevant_columns_score,

        family=
            family_score,

        tool_selection=
            tool_selection_score,

        tool_arguments=
            tool_arguments_score,

        safety=
            safety_score,

        invented_columns=
            invented_columns,

        invented_tools=
            invented_tools,

        forbidden_tools_used=
            forbidden_tools_used,

        forbidden_assumptions_used=
            forbidden_assumptions_used,

        overall=
            overall,
    )
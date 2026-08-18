from __future__ import annotations


# DataLens local-model performance benchmark v0.1
#
# Purpose:
# - measure quality AND latency for the current planner/tool-model pair;
# - keep analytical semantics unchanged;
# - use the same canonical HR cases for every compared model pair;
# - disable normal observability traces by default so benchmark runs do not
#   pollute the product observability screen.
#
# Architecture measured:
#   planner LLM -> Python validation -> tool LLM -> Python validation
#   -> deterministic execution.


import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time

from dataclasses import (
    asdict,
    dataclass,
)

from datetime import (
    datetime,
    timezone,
)

from io import (
    BytesIO,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
)


from fastapi import (
    UploadFile,
)


# ============================================================
# VERSION
# ============================================================

MODEL_PERFORMANCE_BENCHMARK_RULE_VERSION = (
    "model_performance_benchmark_v0.2"
)


# ============================================================
# API BOOTSTRAP
# ============================================================

HERE = (
    Path(
        __file__
    )
    .resolve()
    .parent
)


API_ROOT = (
    HERE
    .parents[
        1
    ]
)


if (
    str(
        API_ROOT
    )
    not in sys.path
):
    sys.path.insert(
        0,
        str(
            API_ROOT
        ),
    )


from app.api.analysis_run import (  # noqa: E402
    run_ai_native_pipeline,
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_DATASET = (
    HERE
    /
    "datalens_hr_benchmark.csv"
)


DEFAULT_CASES = (
    HERE
    /
    "cases_model_performance_v0_1.json"
)


DEFAULT_RESULTS_ROOT = (
    HERE
    /
    "results"
)


DEFAULT_PLANNER_MODEL = (
    "gemma3:4b"
)


DEFAULT_TOOL_MODEL = (
    "qwen2.5:1.5b-instruct"
)


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass
class BenchmarkResult:
    case_id: str
    repeat_index: int

    objective: str

    planner_model: str
    tool_model: str

    expected_families: str
    planner_status: str
    planner_family: str

    planner_outcome_correct: bool
    planner_family_correct: bool
    planner_bindings_correct: bool

    expected_tool: str
    requested_tool: str
    tool_selection_correct: bool
    tool_arguments_correct: bool

    should_execute: bool
    execution_observed: bool
    execution_correct: bool

    chart_type_expected: str
    chart_type_observed: str
    chart_type_correct: bool

    forbidden_column_safe: bool

    planner_attempt_count: int
    planner_retry_count: int
    planner_normalization_count: int
    planner_first_pass: bool

    tool_attempt_count: int
    tool_retry_count: int
    tool_first_pass: bool

    case_pass: bool

    wall_total_ms: float

    planner_total_ms: float
    planner_prompt_construction_ms: float
    planner_model_inference_ms: float
    planner_structured_parse_ms: float
    planner_python_validation_ms: float
    planner_retry_feedback_ms: float

    native_total_ms: float
    tool_prompt_construction_ms: float
    tool_model_inference_ms: float
    tool_response_parse_ms: float
    tool_python_validation_ms: float
    deterministic_execution_ms: float

    planner_inference_share_wall_pct: float
    tool_inference_share_wall_pct: float
    deterministic_execution_share_wall_pct: float

    planner_rule_version: str
    native_tool_rule_version: str
    pipeline_rule_version: str

    failure_reasons: str
    error: str


# ============================================================
# UTILITIES
# ============================================================

def load_cases(
    path: Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


    if (
        not isinstance(
            raw,
            list,
        )
        or
        not raw
    ):
        raise ValueError(
            "Le fichier de cas doit contenir une liste JSON non vide."
        )


    return raw


def make_upload(
    path: Path,
) -> UploadFile:
    return UploadFile(
        filename=(
            path.name
        ),
        file=BytesIO(
            path.read_bytes()
        ),
    )


def read_value(
    source: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(
        source,
        dict,
    ):
        return source.get(
            name,
            default,
        )


    return getattr(
        source,
        name,
        default,
    )


def binding_map(
    planner_item: Any | None,
) -> dict[
    str,
    str,
]:
    if planner_item is None:
        return {}


    contract = read_value(
        planner_item,
        "contract",
        None,
    )


    if contract is None:
        return {}


    return {
        str(
            read_value(
                binding,
                "role",
                "",
            )
        ):
            str(
                read_value(
                    binding,
                    "column",
                    "",
                )
            )

        for binding
        in (
            read_value(
                contract,
                "bindings",
                [],
            )
            or []
        )
    }


def expected_binding_match(
    actual: dict[
        str,
        str,
    ],
    acceptable: list[
        dict[
            str,
            str,
        ]
    ],
) -> bool:
    if not acceptable:
        return actual == {}


    return any(
        actual == expected

        for expected
        in acceptable
    )


def expected_tool_argument_maps(
    expected: dict[
        str,
        Any,
    ],
) -> list[
    dict[
        str,
        str,
    ]
]:
    families = (
        expected.get(
            "acceptable_families",
            [],
        )
        or []
    )


    family = (
        families[
            0
        ]
        if families
        else ""
    )


    result: list[
        dict[
            str,
            str,
        ]
    ] = []


    for bindings in (
        expected.get(
            "acceptable_bindings",
            [],
        )
        or []
    ):
        if family in {
            "quantitative_association",
            "categorical_association",
        }:
            result.append(
                {
                    "x_column":
                        bindings[
                            "x"
                        ],

                    "y_column":
                        bindings[
                            "y"
                        ],
                }
            )


        elif (
            family ==
            "group_comparison"
        ):
            result.append(
                {
                    "group_column":
                        bindings[
                            "group"
                        ],

                    "value_column":
                        bindings[
                            "value"
                        ],
                }
            )


        elif (
            family ==
            "distribution"
        ):
            result.append(
                {
                    "value_column":
                        bindings[
                            "value"
                        ],
                }
            )


        elif (
            family ==
            "time_series"
        ):
            result.append(
                {
                    "time_column":
                        bindings[
                            "time"
                        ],

                    "value_column":
                        bindings[
                            "value"
                        ],
                }
            )


    return result


def tool_column_arguments(
    arguments: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    str,
]:
    return {
        key:
            str(
                value
            )

        for (
            key,
            value,
        )
        in arguments.items()

        if (
            key !=
            "dataset_id"
            and
            value is not None
        )
    }


def select_planner_item(
    report: Any,
    acceptable_families: set[
        str
    ],
) -> Any | None:
    items = list(
        read_value(
            read_value(
                report,
                "planner",
                None,
            ),
            "items",
            [],
        )
        or []
    )


    for item in items:
        proposal = read_value(
            item,
            "proposal",
            None,
        )


        family = str(
            read_value(
                proposal,
                "family",
                "",
            )
        )


        if family in acceptable_families:
            return item


    return (
        items[
            0
        ]
        if items
        else None
    )


def select_native_item(
    report: Any,
    acceptable_families: set[
        str
    ],
) -> Any | None:
    items = list(
        read_value(
            report,
            "items",
            [],
        )
        or []
    )


    for item in items:
        family = str(
            read_value(
                item,
                "family",
                "",
            )
        )


        if family in acceptable_families:
            return item


    return (
        items[
            0
        ]
        if items
        else None
    )


def safe_float(
    value: Any,
) -> float:
    try:
        number = float(
            value
        )


    except (
        TypeError,
        ValueError,
    ):
        return 0.0


    if number != number:
        return 0.0


    return max(
        0.0,
        number,
    )


def safe_share(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0.0:
        return 0.0


    return (
        numerator
        /
        denominator
        *
        100.0
    )


def percentile(
    values: list[
        float
    ],
    q: float,
) -> float:
    if not values:
        return 0.0


    ordered = sorted(
        values
    )


    if len(
        ordered
    ) == 1:
        return ordered[
            0
        ]


    position = (
        (
            len(
                ordered
            )
            -
            1
        )
        *
        q
    )


    lower_index = int(
        position
    )

    upper_index = min(
        lower_index +
        1,
        len(
            ordered
        )
        -
        1,
    )


    fraction = (
        position
        -
        lower_index
    )


    return (
        ordered[
            lower_index
        ]
        +
        (
            ordered[
                upper_index
            ]
            -
            ordered[
                lower_index
            ]
        )
        *
        fraction
    )


def latency_stats(
    values: list[
        float
    ],
) -> dict[
    str,
    float,
]:
    if not values:
        return {
            "median_ms":
                0.0,

            "p95_ms":
                0.0,

            "mean_ms":
                0.0,

            "min_ms":
                0.0,

            "max_ms":
                0.0,
        }


    return {
        "median_ms":
            round(
                statistics.median(
                    values
                ),
                3,
            ),

        "p95_ms":
            round(
                percentile(
                    values,
                    0.95,
                ),
                3,
            ),

        "mean_ms":
            round(
                statistics.fmean(
                    values
                ),
                3,
            ),

        "min_ms":
            round(
                min(
                    values
                ),
                3,
            ),

        "max_ms":
            round(
                max(
                    values
                ),
                3,
            ),
    }


def rate(
    values: list[
        bool
    ],
) -> float:
    if not values:
        return 0.0


    return (
        sum(
            1
            for value
            in values
            if value
        )
        /
        len(
            values
        )
    )


# ============================================================
# SINGLE RUN
# ============================================================

def evaluate_case(
    *,
    case: dict[
        str,
        Any,
    ],
    repeat_index: int,
    dataset_path: Path,
    planner_model: str,
    tool_model: str,
    raw_dir: Path,
) -> BenchmarkResult:
    expected = (
        case[
            "expected"
        ]
    )


    acceptable_statuses = set(
        str(
            value
        )

        for value
        in (
            expected.get(
                "acceptable_planner_statuses",
                [],
            )
            or []
        )
    )


    acceptable_families = set(
        str(
            value
        )

        for value
        in (
            expected.get(
                "acceptable_families",
                [],
            )
            or []
        )
    )


    expected_tool = (
        str(
            expected.get(
                "tool"
            )
        )
        if expected.get(
            "tool"
        )
        is not None
        else ""
    )


    should_execute = bool(
        expected.get(
            "should_execute",
            False,
        )
    )


    wall_started_at = (
        time.perf_counter()
    )


    report = None
    error_message = ""


    try:
        report = (
            run_ai_native_pipeline(
                dataset_files=[
                    make_upload(
                        dataset_path
                    ),
                ],
                objective=str(
                    case[
                        "objective"
                    ]
                ),
                planner_model=(
                    planner_model
                ),
                tool_model=(
                    tool_model
                ),
            )
        )


    except Exception as error:
        error_message = (
            f"{type(error).__name__}: {error}"
        )


    wall_total_ms = (
        (
            time.perf_counter()
            -
            wall_started_at
        )
        *
        1000.0
    )


    raw_path = (
        raw_dir
        /
        (
            f"{case['case_id']}"
            f"__r{repeat_index:02d}.json"
        )
    )


    if report is not None:
        raw_path.write_text(
            report.model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )


    else:
        raw_path.write_text(
            json.dumps(
                {
                    "case_id":
                        case[
                            "case_id"
                        ],

                    "error":
                        error_message,
                },
                ensure_ascii=False,
                indent=2,
            )
            +
            "\n",
            encoding="utf-8",
        )


    if report is None:
        return BenchmarkResult(
            case_id=str(
                case[
                    "case_id"
                ]
            ),
            repeat_index=(
                repeat_index
            ),
            objective=str(
                case[
                    "objective"
                ]
            ),
            planner_model=(
                planner_model
            ),
            tool_model=(
                tool_model
            ),
            expected_families=",".join(
                sorted(
                    acceptable_families
                )
            ),
            planner_status="error",
            planner_family="error",
            planner_outcome_correct=False,
            planner_family_correct=False,
            planner_bindings_correct=False,
            expected_tool=(
                expected_tool
            ),
            requested_tool="",
            tool_selection_correct=False,
            tool_arguments_correct=False,
            should_execute=(
                should_execute
            ),
            execution_observed=False,
            execution_correct=(
                not should_execute
            ),
            chart_type_expected=str(
                expected.get(
                    "chart_type",
                    "",
                )
                or ""
            ),
            chart_type_observed="",
            chart_type_correct=False,
            forbidden_column_safe=False,
            planner_attempt_count=0,
            planner_retry_count=0,
            planner_normalization_count=0,
            planner_first_pass=False,
            tool_attempt_count=0,
            tool_retry_count=0,
            tool_first_pass=False,
            case_pass=False,
            wall_total_ms=round(
                wall_total_ms,
                3,
            ),
            planner_total_ms=0.0,
            planner_prompt_construction_ms=0.0,
            planner_model_inference_ms=0.0,
            planner_structured_parse_ms=0.0,
            planner_python_validation_ms=0.0,
            planner_retry_feedback_ms=0.0,
            native_total_ms=0.0,
            tool_prompt_construction_ms=0.0,
            tool_model_inference_ms=0.0,
            tool_response_parse_ms=0.0,
            tool_python_validation_ms=0.0,
            deterministic_execution_ms=0.0,
            planner_inference_share_wall_pct=0.0,
            tool_inference_share_wall_pct=0.0,
            deterministic_execution_share_wall_pct=0.0,
            planner_rule_version="",
            native_tool_rule_version="",
            pipeline_rule_version="",
            failure_reasons=(
                "pipeline_exception"
            ),
            error=(
                error_message
            ),
        )


    planner = (
        report
        .planner
    )


    planner_item = (
        select_planner_item(
            report,
            acceptable_families,
        )
    )


    proposal = (
        read_value(
            planner_item,
            "proposal",
            None,
        )
        if planner_item
        is not None
        else None
    )


    planner_status = str(
        read_value(
            planner_item,
            "validation_status",
            "missing",
        )
    )


    planner_family = str(
        read_value(
            proposal,
            "family",
            "missing",
        )
    )


    planner_outcome_correct = (
        planner_status
        in
        acceptable_statuses
    )


    planner_family_correct = (
        planner_family
        in
        acceptable_families
    )


    actual_bindings = (
        binding_map(
            planner_item
        )
        if planner_status ==
        "validated"
        else {}
    )


    planner_bindings_correct = (
        expected_binding_match(
            actual_bindings,
            (
                expected.get(
                    "acceptable_bindings",
                    [],
                )
                or []
            ),
        )
        if should_execute
        else True
    )


    native_item = (
        select_native_item(
            report,
            acceptable_families,
        )
    )


    native = (
        read_value(
            native_item,
            "native_tool",
            None,
        )
        if native_item
        is not None
        else None
    )


    requested_tool = str(
        read_value(
            native,
            "requested_tool",
            "",
        )
        or ""
    )


    tool_selection_correct = (
        requested_tool ==
        expected_tool
    )


    expected_argument_maps = (
        expected_tool_argument_maps(
            expected
        )
    )


    actual_tool_arguments = (
        tool_column_arguments(
            dict(
                read_value(
                    native,
                    "requested_arguments",
                    {},
                )
                or {}
            )
        )
        if native
        is not None
        else {}
    )


    tool_arguments_correct = (
        any(
            actual_tool_arguments ==
            expected_arguments

            for expected_arguments
            in expected_argument_maps
        )
        if expected_tool
        else (
            actual_tool_arguments ==
            {}
        )
    )


    execution_observed = (
        int(
            read_value(
                report,
                "executed_count",
                0,
            )
            or 0
        )
        >
        0
    )


    execution_correct = (
        execution_observed ==
        should_execute
    )


    chart_type_observed = ""


    execution = (
        read_value(
            native,
            "execution",
            None,
        )
        if native
        is not None
        else None
    )


    result = (
        read_value(
            execution,
            "result",
            None,
        )
        if execution
        is not None
        else None
    )


    if result is not None:
        chart_type_observed = str(
            read_value(
                result,
                "chart_type",
                "",
            )
            or ""
        )


    chart_type_expected = str(
        expected.get(
            "chart_type",
            "",
        )
        or ""
    )


    chart_type_correct = (
        (
            chart_type_observed ==
            chart_type_expected
        )
        if chart_type_expected
        else True
    )


    forbidden_columns = {
        str(
            value
        )
        .casefold()

        for value
        in (
            expected.get(
                "forbidden_columns",
                [],
            )
            or []
        )
    }


    used_values = {
        str(
            value
        )
        .casefold()

        for value
        in actual_tool_arguments.values()
    }


    forbidden_column_safe = (
        not (
            forbidden_columns
            &
            used_values
        )
    )


    failure_reasons: list[
        str
    ] = []


    checks = [
        (
            planner_outcome_correct,
            f"planner_status={planner_status}",
        ),
        (
            planner_family_correct,
            f"planner_family={planner_family}",
        ),
        (
            planner_bindings_correct,
            f"bindings={actual_bindings}",
        ),
        (
            tool_selection_correct,
            f"tool={requested_tool or 'none'}",
        ),
        (
            tool_arguments_correct,
            f"tool_args={actual_tool_arguments}",
        ),
        (
            execution_correct,
            f"execution={execution_observed}",
        ),
        (
            chart_type_correct,
            f"chart={chart_type_observed or 'none'}",
        ),
        (
            forbidden_column_safe,
            "forbidden_column_used",
        ),
    ]


    for passed, reason in checks:
        if not passed:
            failure_reasons.append(
                reason
            )


    planner_timing = (
        planner
        .timing
    )


    native_timing = (
        report
        .timing
    )


    planner_total_ms = safe_float(
        read_value(
            planner_timing,
            "total_ms",
            0.0,
        )
    )


    planner_prompt_construction_ms = safe_float(
        read_value(
            planner_timing,
            "prompt_construction_ms",
            0.0,
        )
    )


    planner_model_inference_ms = safe_float(
        read_value(
            planner_timing,
            "model_inference_ms",
            0.0,
        )
    )


    planner_structured_parse_ms = safe_float(
        read_value(
            planner_timing,
            "structured_parse_ms",
            0.0,
        )
    )


    planner_python_validation_ms = safe_float(
        read_value(
            planner_timing,
            "python_validation_ms",
            0.0,
        )
    )


    planner_retry_feedback_ms = safe_float(
        read_value(
            planner_timing,
            "retry_feedback_ms",
            0.0,
        )
    )


    native_total_ms = safe_float(
        read_value(
            native_timing,
            "total_ms",
            0.0,
        )
    )


    tool_prompt_construction_ms = safe_float(
        read_value(
            native_timing,
            "tool_prompt_construction_ms",
            0.0,
        )
    )


    tool_model_inference_ms = safe_float(
        read_value(
            native_timing,
            "tool_model_inference_ms",
            0.0,
        )
    )


    tool_response_parse_ms = safe_float(
        read_value(
            native_timing,
            "tool_response_parse_ms",
            0.0,
        )
    )


    tool_python_validation_ms = safe_float(
        read_value(
            native_timing,
            "tool_python_validation_ms",
            0.0,
        )
    )


    deterministic_execution_ms = safe_float(
        read_value(
            native_timing,
            "deterministic_execution_ms",
            0.0,
        )
    )


    planner_attempt_count = int(
        read_value(
            planner,
            "attempt_count",
            0,
        )
        or 0
    )


    planner_retry_count = int(
        read_value(
            planner,
            "retry_count",
            0,
        )
        or 0
    )


    planner_normalization_count = int(
        read_value(
            planner,
            "normalization_count",
            0,
        )
        or 0
    )


    tool_attempt_count = int(
        read_value(
            native,
            "attempt_count",
            0,
        )
        or 0
    )


    tool_retry_count = int(
        read_value(
            native,
            "retry_count",
            0,
        )
        or 0
    )


    planner_first_pass = (
        planner_attempt_count ==
        1
        and
        planner_retry_count ==
        0
    )


    tool_first_pass = (
        (
            tool_attempt_count ==
            1
            and
            tool_retry_count ==
            0
        )
        if expected_tool
        else True
    )


    native_tool_rule_version = str(
        read_value(
            native,
            "native_tool_rule_version",
            "",
        )
        or ""
    )


    case_pass = (
        len(
            failure_reasons
        )
        ==
        0
    )


    return BenchmarkResult(
        case_id=str(
            case[
                "case_id"
            ]
        ),
        repeat_index=(
            repeat_index
        ),
        objective=str(
            case[
                "objective"
            ]
        ),
        planner_model=(
            planner_model
        ),
        tool_model=(
            tool_model
        ),
        expected_families=",".join(
            sorted(
                acceptable_families
            )
        ),
        planner_status=(
            planner_status
        ),
        planner_family=(
            planner_family
        ),
        planner_outcome_correct=(
            planner_outcome_correct
        ),
        planner_family_correct=(
            planner_family_correct
        ),
        planner_bindings_correct=(
            planner_bindings_correct
        ),
        expected_tool=(
            expected_tool
        ),
        requested_tool=(
            requested_tool
        ),
        tool_selection_correct=(
            tool_selection_correct
        ),
        tool_arguments_correct=(
            tool_arguments_correct
        ),
        should_execute=(
            should_execute
        ),
        execution_observed=(
            execution_observed
        ),
        execution_correct=(
            execution_correct
        ),
        chart_type_expected=(
            chart_type_expected
        ),
        chart_type_observed=(
            chart_type_observed
        ),
        chart_type_correct=(
            chart_type_correct
        ),
        forbidden_column_safe=(
            forbidden_column_safe
        ),
        planner_attempt_count=(
            planner_attempt_count
        ),
        planner_retry_count=(
            planner_retry_count
        ),
        planner_normalization_count=(
            planner_normalization_count
        ),
        planner_first_pass=(
            planner_first_pass
        ),
        tool_attempt_count=(
            tool_attempt_count
        ),
        tool_retry_count=(
            tool_retry_count
        ),
        tool_first_pass=(
            tool_first_pass
        ),
        case_pass=(
            case_pass
        ),
        wall_total_ms=round(
            wall_total_ms,
            3,
        ),
        planner_total_ms=round(
            planner_total_ms,
            3,
        ),
        planner_prompt_construction_ms=round(
            planner_prompt_construction_ms,
            3,
        ),
        planner_model_inference_ms=round(
            planner_model_inference_ms,
            3,
        ),
        planner_structured_parse_ms=round(
            planner_structured_parse_ms,
            3,
        ),
        planner_python_validation_ms=round(
            planner_python_validation_ms,
            3,
        ),
        planner_retry_feedback_ms=round(
            planner_retry_feedback_ms,
            3,
        ),
        native_total_ms=round(
            native_total_ms,
            3,
        ),
        tool_prompt_construction_ms=round(
            tool_prompt_construction_ms,
            3,
        ),
        tool_model_inference_ms=round(
            tool_model_inference_ms,
            3,
        ),
        tool_response_parse_ms=round(
            tool_response_parse_ms,
            3,
        ),
        tool_python_validation_ms=round(
            tool_python_validation_ms,
            3,
        ),
        deterministic_execution_ms=round(
            deterministic_execution_ms,
            3,
        ),
        planner_inference_share_wall_pct=round(
            safe_share(
                planner_model_inference_ms,
                wall_total_ms,
            ),
            3,
        ),
        tool_inference_share_wall_pct=round(
            safe_share(
                tool_model_inference_ms,
                wall_total_ms,
            ),
            3,
        ),
        deterministic_execution_share_wall_pct=round(
            safe_share(
                deterministic_execution_ms,
                wall_total_ms,
            ),
            3,
        ),
        planner_rule_version=str(
            read_value(
                planner,
                "planner_rule_version",
                "",
            )
            or ""
        ),
        native_tool_rule_version=(
            native_tool_rule_version
        ),
        pipeline_rule_version=str(
            read_value(
                report,
                "pipeline_rule_version",
                "",
            )
            or ""
        ),
        failure_reasons=" | ".join(
            failure_reasons
        ),
        error=(
            error_message
        ),
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    *,
    results: list[
        BenchmarkResult
    ],
    planner_model: str,
    tool_model: str,
    dataset_path: Path,
    cases_path: Path,
    repeats: int,
    warmup: int,
) -> dict[
    str,
    Any,
]:
    executable = [
        result
        for result in results
        if result.should_execute
    ]


    successful = [
        result
        for result in results
        if result.error == ""
    ]


    case_summaries: list[
        dict[
            str,
            Any,
        ]
    ] = []


    case_ids = list(
        dict.fromkeys(
            result.case_id
            for result in results
        )
    )


    for case_id in case_ids:
        selected = [
            result
            for result in results
            if result.case_id ==
            case_id
        ]


        case_summaries.append(
            {
                "case_id":
                    case_id,

                "run_count":
                    len(
                        selected
                    ),

                "pass_rate":
                    rate(
                        [
                            result.case_pass
                            for result
                            in selected
                        ]
                    ),

                "wall_total_latency":
                    latency_stats(
                        [
                            result.wall_total_ms
                            for result
                            in selected
                        ]
                    ),

                "planner_model_inference_latency":
                    latency_stats(
                        [
                            result.planner_model_inference_ms
                            for result
                            in selected
                        ]
                    ),

                "tool_model_inference_latency":
                    latency_stats(
                        [
                            result.tool_model_inference_ms
                            for result
                            in selected
                            if result.should_execute
                        ]
                    ),
            }
        )


    return {
        "benchmark_rule_version":
            MODEL_PERFORMANCE_BENCHMARK_RULE_VERSION,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            )
            .isoformat(),

        "planner_model":
            planner_model,

        "tool_model":
            tool_model,

        "dataset":
            str(
                dataset_path
                .resolve()
            ),

        "cases":
            str(
                cases_path
                .resolve()
            ),

        "repeats":
            repeats,

        "warmup_runs":
            warmup,

        "run_count":
            len(
                results
            ),

        "successful_run_count":
            len(
                successful
            ),

        "case_pass_rate":
            rate(
                [
                    result.case_pass
                    for result
                    in results
                ]
            ),

        "planner_outcome_accuracy":
            rate(
                [
                    result.planner_outcome_correct
                    for result
                    in results
                ]
            ),

        "planner_family_accuracy":
            rate(
                [
                    result.planner_family_correct
                    for result
                    in results
                ]
            ),

        "planner_binding_accuracy":
            rate(
                [
                    result.planner_bindings_correct
                    for result
                    in executable
                ]
            )
            if executable
            else None,

        "planner_first_pass_rate":
            rate(
                [
                    result.planner_first_pass
                    for result
                    in results
                ]
            ),

        "planner_retry_rate":
            rate(
                [
                    result.planner_retry_count >
                    0
                    for result
                    in results
                ]
            ),

        "tool_selection_accuracy":
            rate(
                [
                    result.tool_selection_correct
                    for result
                    in executable
                ]
            )
            if executable
            else None,

        "tool_argument_accuracy":
            rate(
                [
                    result.tool_arguments_correct
                    for result
                    in executable
                ]
            )
            if executable
            else None,

        "tool_first_pass_rate":
            rate(
                [
                    result.tool_first_pass
                    for result
                    in executable
                ]
            )
            if executable
            else None,

        "tool_retry_rate":
            rate(
                [
                    result.tool_retry_count >
                    0
                    for result
                    in executable
                ]
            )
            if executable
            else None,

        "execution_accuracy":
            rate(
                [
                    result.execution_correct
                    for result
                    in results
                ]
            ),

        "wall_total_latency":
            latency_stats(
                [
                    result.wall_total_ms
                    for result
                    in results
                ]
            ),

        "planner_total_latency":
            latency_stats(
                [
                    result.planner_total_ms
                    for result
                    in results
                ]
            ),

        "planner_model_inference_latency":
            latency_stats(
                [
                    result.planner_model_inference_ms
                    for result
                    in results
                ]
            ),

        "planner_python_validation_latency":
            latency_stats(
                [
                    result.planner_python_validation_ms
                    for result
                    in results
                ]
            ),

        "native_total_latency":
            latency_stats(
                [
                    result.native_total_ms
                    for result
                    in executable
                ]
            ),

        "tool_model_inference_latency":
            latency_stats(
                [
                    result.tool_model_inference_ms
                    for result
                    in executable
                ]
            ),

        "tool_python_validation_latency":
            latency_stats(
                [
                    result.tool_python_validation_ms
                    for result
                    in executable
                ]
            ),

        "deterministic_execution_latency":
            latency_stats(
                [
                    result.deterministic_execution_ms
                    for result
                    in executable
                ]
            ),

        "planner_inference_share_wall_median_pct":
            round(
                statistics.median(
                    [
                        result.planner_inference_share_wall_pct
                        for result
                        in results
                    ]
                ),
                3,
            )
            if results
            else 0.0,

        "tool_inference_share_wall_median_pct":
            round(
                statistics.median(
                    [
                        result.tool_inference_share_wall_pct
                        for result
                        in executable
                    ]
                ),
                3,
            )
            if executable
            else 0.0,

        "deterministic_execution_share_wall_median_pct":
            round(
                statistics.median(
                    [
                        result.deterministic_execution_share_wall_pct
                        for result
                        in executable
                    ]
                ),
                3,
            )
            if executable
            else 0.0,

        "environment":
            {
                "python":
                    sys.version.split(
                        " "
                    )[
                        0
                    ],

                "platform":
                    platform.platform(),
            },

        "cases_summary":
            case_summaries,
    }


# ============================================================
# OUTPUT
# ============================================================

def write_csv(
    path: Path,
    results: list[
        BenchmarkResult
    ],
) -> None:
    if not results:
        return


    rows = [
        asdict(
            result
        )
        for result
        in results
    ]


    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = (
            csv.DictWriter(
                handle,
                fieldnames=list(
                    rows[
                        0
                    ]
                    .keys()
                ),
            )
        )


        writer.writeheader()
        writer.writerows(
            rows
        )


def print_summary(
    summary: dict[
        str,
        Any,
    ],
) -> None:
    def pct(
        value: Any,
    ) -> str:
        if value is None:
            return "n/a"


        return (
            f"{float(value) * 100:.1f}%"
        )


    def med(
        key: str,
    ) -> str:
        value = (
            summary[
                key
            ][
                "median_ms"
            ]
        )


        if value >= 1000:
            return (
                f"{value / 1000:.2f}s"
            )


        return (
            f"{value:.1f}ms"
        )


    print()
    print(
        "DataLens model performance benchmark v0.1"
    )
    print(
        "========================================="
    )
    print(
        f"Planner : {summary['planner_model']}"
    )
    print(
        f"Tool    : {summary['tool_model']}"
    )
    print(
        f"Runs    : {summary['run_count']}"
    )
    print()
    print(
        f"PASS global                 : {pct(summary['case_pass_rate'])}"
    )
    print(
        f"Planner family              : {pct(summary['planner_family_accuracy'])}"
    )
    print(
        f"Planner bindings            : {pct(summary['planner_binding_accuracy'])}"
    )
    print(
        f"Planner first-pass          : {pct(summary['planner_first_pass_rate'])}"
    )
    print(
        f"Tool selection              : {pct(summary['tool_selection_accuracy'])}"
    )
    print(
        f"Tool arguments              : {pct(summary['tool_argument_accuracy'])}"
    )
    print(
        f"Tool first-pass             : {pct(summary['tool_first_pass_rate'])}"
    )
    print(
        f"Execution                   : {pct(summary['execution_accuracy'])}"
    )
    print()
    print(
        f"Latence totale médiane      : {med('wall_total_latency')}"
    )
    print(
        f"Inférence planner médiane   : {med('planner_model_inference_latency')}"
    )
    print(
        f"Inférence tool médiane      : {med('tool_model_inference_latency')}"
    )
    print(
        f"Exécution Python médiane    : {med('deterministic_execution_latency')}"
    )
    print()
    print(
        "Part médiane de la latence totale :"
    )
    print(
        "  Planner inference : "
        f"{summary['planner_inference_share_wall_median_pct']:.1f}%"
    )
    print(
        "  Tool inference    : "
        f"{summary['tool_inference_share_wall_median_pct']:.1f}%"
    )
    print(
        "  Python execution  : "
        f"{summary['deterministic_execution_share_wall_median_pct']:.2f}%"
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local DataLens planner/tool models "
            "on canonical HR analytical cases."
        )
    )


    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            DEFAULT_DATASET
        ),
    )


    parser.add_argument(
        "--cases",
        type=Path,
        default=(
            DEFAULT_CASES
        ),
    )


    parser.add_argument(
        "--planner-model",
        default=(
            DEFAULT_PLANNER_MODEL
        ),
    )


    parser.add_argument(
        "--tool-model",
        default=(
            DEFAULT_TOOL_MODEL
        ),
    )


    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )


    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
    )


    parser.add_argument(
        "--cooldown-ms",
        type=int,
        default=100,
    )


    parser.add_argument(
        "--results-root",
        type=Path,
        default=(
            DEFAULT_RESULTS_ROOT
        ),
    )


    parser.add_argument(
        "--record-observability",
        action="store_true",
        help=(
            "Keep normal DataLens JSONL observability traces. "
            "Disabled by default for benchmark cleanliness."
        ),
    )


    return (
        parser.parse_args()
    )


def main() -> None:
    args = parse_args()


    dataset_path = (
        args
        .dataset
        .expanduser()
        .resolve()
    )


    cases_path = (
        args
        .cases
        .expanduser()
        .resolve()
    )


    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {dataset_path}"
        )


    if not cases_path.exists():
        raise FileNotFoundError(
            f"Fichier de cas introuvable : {cases_path}"
        )


    if args.repeats < 1:
        raise ValueError(
            "--repeats doit être >= 1."
        )


    if args.warmup < 0:
        raise ValueError(
            "--warmup doit être >= 0."
        )


    if not args.record_observability:
        os.environ[
            "DATALENS_AI_TRACE_ENABLED"
        ] = "0"


    cases = load_cases(
        cases_path
    )


    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%S"
        )
    )


    safe_planner = (
        args.planner_model
        .replace(
            ":",
            "_"
        )
        .replace(
            "/",
            "_"
        )
    )


    safe_tool = (
        args.tool_model
        .replace(
            ":",
            "_"
        )
        .replace(
            "/",
            "_"
        )
    )


    run_dir = (
        args
        .results_root
        .expanduser()
        .resolve()
        /
        (
            f"{timestamp}"
            f"__planner-{safe_planner}"
            f"__tool-{safe_tool}"
        )
    )


    raw_dir = (
        run_dir
        /
        "raw"
    )


    raw_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    metadata = {
        "benchmark_rule_version":
            MODEL_PERFORMANCE_BENCHMARK_RULE_VERSION,

        "planner_model":
            args.planner_model,

        "tool_model":
            args.tool_model,

        "dataset":
            str(
                dataset_path
            ),

        "cases":
            str(
                cases_path
            ),

        "repeats":
            args.repeats,

        "warmup":
            args.warmup,

        "observability_recording":
            bool(
                args.record_observability
            ),
    }


    (
        run_dir
        /
        "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )


    if args.warmup > 0:
        warmup_case = next(
            (
                case
                for case
                in cases
                if (
                    case[
                        "case_id"
                    ]
                    ==
                    "time_salary_snapshot"
                )
            ),
            cases[
                0
            ],
        )


        print(
            f"Warmup : {args.warmup} exécution(s)..."
        )


        for index in range(
            1,
            args.warmup +
            1,
        ):
            warmup_upload = (
                make_upload(
                    dataset_path
                )
            )


            warmup_error = ""


            try:
                run_ai_native_pipeline(
                    dataset_files=[
                        warmup_upload,
                    ],
                    objective=str(
                        warmup_case[
                            "objective"
                        ]
                    ),
                    planner_model=(
                        args.planner_model
                    ),
                    tool_model=(
                        args.tool_model
                    ),
                )


            except Exception as error:
                warmup_error = (
                    f"{type(error).__name__}: {error}"
                )


            finally:
                try:
                    warmup_upload.file.close()
                except Exception:
                    pass


            if warmup_error:
                print(
                    (
                        f"  warmup {index}/{args.warmup} "
                        "échoué — non mesuré"
                    )
                )

                print(
                    f"    {warmup_error}"
                )

                print(
                    (
                        "    Le benchmark continue : "
                        "un échec de warmup ne doit pas "
                        "masquer la qualité réelle du modèle."
                    )
                )


            else:
                print(
                    f"  warmup {index}/{args.warmup} terminé"
                )


    results: list[
        BenchmarkResult
    ] = []


    total_runs = (
        len(
            cases
        )
        *
        args.repeats
    )


    current_run = 0


    for repeat_index in range(
        1,
        args.repeats +
        1,
    ):
        for case in cases:
            current_run += 1


            print(
                (
                    f"[{current_run:02d}/{total_runs:02d}] "
                    f"{case['case_id']} "
                    f"(repeat {repeat_index})"
                )
            )


            result = evaluate_case(
                case=(
                    case
                ),
                repeat_index=(
                    repeat_index
                ),
                dataset_path=(
                    dataset_path
                ),
                planner_model=(
                    args.planner_model
                ),
                tool_model=(
                    args.tool_model
                ),
                raw_dir=(
                    raw_dir
                ),
            )


            results.append(
                result
            )


            status = (
                "PASS"
                if result.case_pass
                else "FAIL"
            )


            print(
                (
                    f"    {status} · "
                    f"{result.wall_total_ms / 1000:.2f}s · "
                    f"planner={result.planner_model_inference_ms / 1000:.2f}s · "
                    f"tool={result.tool_model_inference_ms / 1000:.2f}s"
                )
            )


            if result.failure_reasons:
                print(
                    f"    {result.failure_reasons}"
                )


            if args.cooldown_ms > 0:
                time.sleep(
                    args.cooldown_ms
                    /
                    1000.0
                )


    results_csv = (
        run_dir
        /
        "results.csv"
    )


    write_csv(
        results_csv,
        results,
    )


    summary = build_summary(
        results=(
            results
        ),
        planner_model=(
            args.planner_model
        ),
        tool_model=(
            args.tool_model
        ),
        dataset_path=(
            dataset_path
        ),
        cases_path=(
            cases_path
        ),
        repeats=(
            args.repeats
        ),
        warmup=(
            args.warmup
        ),
    )


    summary_path = (
        run_dir
        /
        "summary.json"
    )


    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )


    print_summary(
        summary
    )


    print()
    print(
        f"Résultats : {run_dir}"
    )
    print(
        f"CSV       : {results_csv}"
    )
    print(
        f"Résumé    : {summary_path}"
    )


if __name__ == "__main__":
    main()

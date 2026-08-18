from __future__ import annotations


# DataLens tool-use evaluation runner v0.8
#
# v0.8 keeps the v0.7 multilingual guardrail semantics and adds:
# - support for one or multiple dataset fixtures per eval case;
# - explicit dataset-selection accuracy when a case declares an
#   expected dataset filename.


import argparse
import csv
import json
import statistics
import sys
import time

from datetime import datetime

from dataclasses import (
    asdict,
    dataclass,
)

from io import BytesIO

from pathlib import Path

from typing import (
    Any,
)


from fastapi import (
    UploadFile,
)


# ============================================================
# API PACKAGE BOOTSTRAP
#
# When this file is executed directly from:
#   apps/api/evals/tool_use/run_tool_use_evals.py
#
# Python places `evals/tool_use` on sys.path, not `apps/api`.
# Add the API root explicitly so `import app...` works without
# requiring a manual PYTHONPATH environment variable.
# ============================================================

API_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        2
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


from app.api.analysis_run import (
    run_ai_native_pipeline,
)


# ============================================================
# PATHS
# ============================================================

HERE = Path(
    __file__
).resolve().parent


DEFAULT_CASES_PATH = (
    HERE
    /
    "cases.json"
)


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass
class EvalRunResult:
    case_id: str
    repeat_index: int

    objective: str

    expected_family: str

    planner_status: str
    planner_family: str

    planner_family_correct: bool
    planner_bindings_correct: bool
    planner_outcome_correct: bool

    dataset_selection_applicable: bool
    dataset_selection_correct: bool
    expected_dataset_filenames: str
    actual_dataset_filenames: str

    planner_attempt_count: int
    planner_retry_count: int
    planner_first_pass: bool
    planner_recovered_after_retry: bool
    planner_retry_class: str
    planner_normalization_count: int

    expected_tool: str
    requested_tool: str

    tool_selection_correct: bool
    tool_arguments_correct: bool

    tool_attempt_count: int
    tool_retry_count: int
    tool_first_pass: bool
    tool_recovered_after_retry: bool

    should_execute: bool
    execution_observed: bool
    execution_correct: bool

    chart_type_expected: str
    chart_type_observed: str
    chart_type_correct: bool

    guardrail_correct: bool

    case_pass: bool

    latency_seconds: float

    planner_rule_version: str
    native_tool_rule_version: str
    pipeline_rule_version: str

    failure_reasons: str


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


    if not isinstance(
        raw,
        list,
    ):
        raise ValueError(
            (
                "cases.json must contain "
                "a JSON array."
            )
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


def binding_map(
    item: Any,
) -> dict[
    str,
    str,
]:
    contract = getattr(
        item,
        "contract",
        None,
    )


    if contract is None:
        return {}


    return {
        binding.role:
            binding.column

        for binding
        in contract.bindings
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
        return (
            actual ==
            {}
        )


    return any(
        actual ==
        expected

        for expected
        in acceptable
    )


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


def acceptable_tool_argument_maps(
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
    acceptable = (
        expected.get(
            "acceptable_bindings",
            [],
        )
        or []
    )


    family = (
        expected.get(
            "family",
            ""
        )
    )


    result: list[
        dict[
            str,
            str,
        ]
    ] = []


    for bindings in (
        acceptable
    ):
        if (
            family
            in {
                "quantitative_association",
                "categorical_association",
            }
        ):
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


def select_planner_item(
    *,
    report: Any,
    expected_family: str,
) -> Any | None:
    family_matches = [
        item

        for item
        in report.planner.items

        if (
            item.proposal.family
            ==
            expected_family
        )
    ]


    if family_matches:
        return family_matches[
            0
        ]


    if report.planner.items:
        return (
            report
            .planner
            .items[
                0
            ]
        )


    return None


def select_native_item(
    *,
    report: Any,
    expected_family: str,
) -> Any | None:
    family_matches = [
        item

        for item
        in report.items

        if (
            item.family
            ==
            expected_family
        )
    ]


    if family_matches:
        return family_matches[
            0
        ]


    if report.items:
        return report.items[
            0
        ]


    return None


def fidelity_guard_seen(
    planner_item: Any | None,
) -> bool:
    if planner_item is None:
        return False


    text_parts = [
        *(
            getattr(
                planner_item,
                "errors",
                [],
            )
            or []
        ),
        *(
            getattr(
                getattr(
                    planner_item,
                    "proposal",
                    None,
                ),
                "blockers",
                [],
            )
            or []
        ),
        *(
            getattr(
                getattr(
                    planner_item,
                    "contract",
                    None,
                ),
                "blockers",
                [],
            )
            or []
        ),
    ]


    combined = (
        "\n".join(
            str(
                part
            )

            for part
            in text_parts
        )
        .casefold()
    )


    guardrail_markers = [
        # Explicit deterministic guard.
        "fidélité objectif",

        # French blocker wording.
        "n'existe pas",
        "n’existe pas",
        "absent",
        "absente",
        "inconnue",
        "inconnu",

        # English blocker wording.
        "does not exist",
        "doesn't exist",
        "doesn’t exist",
        "is missing",
        "missing column",
        "unknown column",
        "column is unknown",
        "not found",
    ]


    return any(
        marker
        in combined

        for marker
        in guardrail_markers
    )


def forbidden_column_executed(
    *,
    report: Any,
    forbidden_columns: list[
        str
    ],
) -> bool:
    forbidden = {
        value.casefold()

        for value
        in forbidden_columns
    }


    for item in (
        report.items
    ):
        native = (
            item.native_tool
        )


        if native is None:
            continue


        arguments = {
            str(
                value
            ).casefold()

            for value
            in native.requested_arguments.values()

            if value is not None
        }


        if (
            forbidden
            &
            arguments
        ):
            return True


    return False


# ============================================================
# SINGLE CASE
# ============================================================

def evaluate_case(
    *,
    case: dict[
        str,
        Any,
    ],
    repeat_index: int,
    planner_model: str,
    tool_model: str,
    raw_dir: Path,
) -> EvalRunResult:
    expected = (
        case[
            "expected"
        ]
    )


    expected_family = str(
        expected[
            "family"
        ]
    )


    fixture_values = (
        case.get(
            "fixtures"
        )
    )


    if fixture_values is None:
        fixture_values = [
            case[
                "fixture"
            ]
        ]


    if (
        not isinstance(
            fixture_values,
            list,
        )
        or
        not fixture_values
    ):
        raise ValueError(
            (
                f"Case {case['case_id']} : "
                "`fixtures` doit être une liste non vide."
            )
        )


    fixture_paths = [
        (
            HERE
            /
            str(
                fixture_value
            )
        ).resolve()

        for fixture_value
        in fixture_values
    ]


    uploads = [
        make_upload(
            fixture_path
        )

        for fixture_path
        in fixture_paths
    ]


    start = (
        time.perf_counter()
    )


    report = (
        run_ai_native_pipeline(
            dataset_files=(
                uploads
            ),
            objective=(
                str(
                    case[
                        "objective"
                    ]
                )
            ),
            planner_model=(
                planner_model
            ),
            tool_model=(
                tool_model
            ),
        )
    )


    latency = (
        time.perf_counter()
        -
        start
    )


    raw_path = (
        raw_dir
        /
        (
            f"{case['case_id']}"
            f"__r{repeat_index:02d}.json"
        )
    )


    raw_path.write_text(
        report.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )


    planner_item = (
        select_planner_item(
            report=(
                report
            ),
            expected_family=(
                expected_family
            ),
        )
    )


    planner_status = (
        planner_item
        .validation_status
        if planner_item
        is not None
        else "missing"
    )


    planner_family = (
        planner_item
        .proposal
        .family
        if planner_item
        is not None
        else "missing"
    )


    acceptable_statuses = set(
        expected.get(
            "acceptable_planner_statuses",
            [
                "validated",
            ],
        )
    )


    planner_outcome_correct = (
        planner_status
        in
        acceptable_statuses
    )


    acceptable_families = set(
        expected.get(
            "acceptable_families",
            [
                expected_family,
            ],
        )
        or [
            expected_family,
        ]
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
        if (
            planner_item
            is not None
            and
            planner_status ==
            "validated"
        )
        else {}
    )


    planner_bindings_correct = (
        expected_binding_match(
            actual=(
                actual_bindings
            ),
            acceptable=(
                expected.get(
                    "acceptable_bindings",
                    [],
                )
                or []
            ),
        )
        if (
            expected.get(
                "should_execute",
                False
            )
        )
        else True
    )


    expected_dataset_filename_values = (
        expected.get(
            "expected_dataset_filenames",
            [],
        )
        or []
    )


    expected_dataset_filename_set = {
        str(
            value
        )

        for value
        in expected_dataset_filename_values
    }


    dataset_selection_applicable = bool(
        expected_dataset_filename_set
    )


    actual_dataset_filename_set: set[
        str
    ] = set()


    if (
        planner_item
        is not None
        and
        planner_item.contract
        is not None
    ):
        actual_dataset_filename_set = {
            str(
                filename
            )

            for filename
            in (
                planner_item
                .contract
                .required_dataset_filenames
                or []
            )
        }


    dataset_selection_correct = (
        (
            actual_dataset_filename_set
            ==
            expected_dataset_filename_set
        )
        if dataset_selection_applicable
        else True
    )


    expected_dataset_filenames = (
        ", ".join(
            sorted(
                expected_dataset_filename_set
            )
        )
    )


    actual_dataset_filenames = (
        ", ".join(
            sorted(
                actual_dataset_filename_set
            )
        )
    )


    planner_attempt_count = int(
        report
        .planner
        .attempt_count
    )


    planner_retry_count = int(
        report
        .planner
        .retry_count
    )


    planner_first_pass = (
        planner_attempt_count ==
        1
        and
        planner_retry_count ==
        0
    )


    planner_recovered_after_retry = (
        planner_retry_count >
        0
        and
        planner_outcome_correct
        and
        planner_family_correct
    )


    if (
        planner_retry_count ==
        0
    ):
        planner_retry_class = (
            "none"
        )


    elif (
        not bool(
            expected.get(
                "should_execute",
                False
            )
        )
        and
        planner_status
        in {
            "blocked",
            "rejected",
            "ambiguous",
        }
    ):
        planner_retry_class = (
            "guardrail_abstention"
        )


    elif (
        bool(
            expected.get(
                "should_execute",
                False
            )
        )
        and
        planner_outcome_correct
        and
        planner_family_correct
    ):
        planner_retry_class = (
            "execution_recovery"
        )


    else:
        planner_retry_class = (
            "unresolved_retry"
        )


    planner_normalization_count = int(
        report
        .planner
        .normalization_count
    )


    native_item = (
        select_native_item(
            report=(
                report
            ),
            expected_family=(
                expected_family
            ),
        )
    )


    native = (
        native_item.native_tool
        if (
            native_item
            is not None
        )
        else None
    )


    expected_tool_value = (
        expected.get(
            "tool"
        )
    )


    expected_tool = (
        ""
        if expected_tool_value
        is None
        else str(
            expected_tool_value
        )
    )


    requested_tool = (
        (
            native.requested_tool
            or ""
        )
        if native
        is not None
        else ""
    )


    if expected_tool:
        tool_selection_correct = (
            requested_tool
            ==
            expected_tool
        )


        acceptable_argument_maps = (
            acceptable_tool_argument_maps(
                expected
            )
        )


        actual_tool_arguments = (
            tool_column_arguments(
                native.requested_arguments
            )
            if native
            is not None
            else {}
        )


        tool_arguments_correct = (
            any(
                actual_tool_arguments
                ==
                expected_arguments

                for expected_arguments
                in acceptable_argument_maps
            )
        )


    else:
        tool_selection_correct = (
            requested_tool ==
            ""
        )

        tool_arguments_correct = (
            native is None
            or
            native.requested_arguments
            ==
            {}
        )


    tool_attempt_count = (
        int(
            native.attempt_count
        )
        if native
        is not None
        else 0
    )


    tool_retry_count = (
        int(
            native.retry_count
        )
        if native
        is not None
        else 0
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


    tool_recovered_after_retry = (
        tool_retry_count >
        0
        and
        tool_selection_correct
        and
        (
            native is not None
            and
            native.validation_status ==
            "validated"
        )
    )


    should_execute = bool(
        expected.get(
            "should_execute",
            False
        )
    )


    execution_observed = (
        bool(
            report.executed_count
            >
            0
        )
    )


    execution_correct = (
        execution_observed
        ==
        should_execute
    )


    chart_type_expected = str(
        expected.get(
            "chart_type",
            "",
        )
        or ""
    )


    chart_type_observed = ""


    if (
        native is not None
        and
        native.execution is not None
        and
        native.execution.result is not None
    ):
        chart_type_observed = str(
            native
            .execution
            .result
            .chart_type
            or ""
        )


    chart_type_correct = (
        (
            chart_type_observed
            ==
            chart_type_expected
        )
        if chart_type_expected
        else True
    )


    guardrail_correct = True


    if bool(
        expected.get(
            "must_have_fidelity_guard",
            False
        )
    ):
        guardrail_correct = (
            fidelity_guard_seen(
                planner_item
            )
            and
            report
            .validated_contract_count
            ==
            0
            and
            report
            .executed_count
            ==
            0
            and
            not forbidden_column_executed(
                report=(
                    report
                ),
                forbidden_columns=(
                    expected.get(
                        "forbidden_columns",
                        [],
                    )
                    or []
                ),
            )
        )


    failure_reasons: list[
        str
    ] = []


    checks = [
        (
            planner_outcome_correct,
            (
                "planner_status="
                f"{planner_status}"
            ),
        ),
        (
            planner_family_correct,
            (
                "planner_family="
                f"{planner_family}; "
                "acceptable="
                f"{sorted(acceptable_families)}"
            ),
        ),
        (
            planner_bindings_correct,
            (
                "bindings="
                f"{actual_bindings}"
            ),
        ),
        (
            dataset_selection_correct,
            (
                "datasets="
                f"{sorted(actual_dataset_filename_set)}; "
                "expected="
                f"{sorted(expected_dataset_filename_set)}"
            ),
        ),
        (
            tool_selection_correct,
            (
                "tool="
                f"{requested_tool or 'none'}"
            ),
        ),
        (
            tool_arguments_correct,
            (
                "tool_args="
                f"{getattr(native, 'requested_arguments', {}) if native is not None else {}}"
            ),
        ),
        (
            execution_correct,
            (
                "execution_observed="
                f"{execution_observed}"
            ),
        ),
        (
            chart_type_correct,
            (
                "chart_type="
                f"{chart_type_observed or 'none'}"
            ),
        ),
        (
            guardrail_correct,
            "guardrail_failed",
        ),
    ]


    for (
        passed,
        reason,
    ) in checks:
        if not passed:
            failure_reasons.append(
                reason
            )


    case_pass = (
        len(
            failure_reasons
        )
        ==
        0
    )


    return (
        EvalRunResult(
            case_id=(
                str(
                    case[
                        "case_id"
                    ]
                )
            ),
            repeat_index=(
                repeat_index
            ),
            objective=(
                str(
                    case[
                        "objective"
                    ]
                )
            ),
            expected_family=(
                expected_family
            ),
            planner_status=(
                planner_status
            ),
            planner_family=(
                planner_family
            ),
            planner_family_correct=(
                planner_family_correct
            ),
            planner_bindings_correct=(
                planner_bindings_correct
            ),
            planner_outcome_correct=(
                planner_outcome_correct
            ),
            dataset_selection_applicable=(
                dataset_selection_applicable
            ),
            dataset_selection_correct=(
                dataset_selection_correct
            ),
            expected_dataset_filenames=(
                expected_dataset_filenames
            ),
            actual_dataset_filenames=(
                actual_dataset_filenames
            ),
            planner_attempt_count=(
                planner_attempt_count
            ),
            planner_retry_count=(
                planner_retry_count
            ),
            planner_first_pass=(
                planner_first_pass
            ),
            planner_recovered_after_retry=(
                planner_recovered_after_retry
            ),
            planner_retry_class=(
                planner_retry_class
            ),
            planner_normalization_count=(
                planner_normalization_count
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
            tool_attempt_count=(
                tool_attempt_count
            ),
            tool_retry_count=(
                tool_retry_count
            ),
            tool_first_pass=(
                tool_first_pass
            ),
            tool_recovered_after_retry=(
                tool_recovered_after_retry
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
            guardrail_correct=(
                guardrail_correct
            ),
            case_pass=(
                case_pass
            ),
            latency_seconds=(
                round(
                    latency,
                    4,
                )
            ),
            planner_rule_version=(
                report
                .planner
                .planner_rule_version
            ),
            native_tool_rule_version=(
                (
                    native
                    .native_tool_rule_version
                )
                if native
                is not None
                else ""
            ),
            pipeline_rule_version=(
                report
                .pipeline_rule_version
            ),
            failure_reasons=(
                " | ".join(
                    failure_reasons
                )
            ),
        )
    )


# ============================================================
# SUMMARY
# ============================================================

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


def build_summary(
    results: list[
        EvalRunResult
    ],
) -> dict[
    str,
    Any,
]:
    executable = [
        result

        for result
        in results

        if result.should_execute
    ]


    guardrails = [
        result

        for result
        in results

        if not result.should_execute
    ]


    dataset_selection_runs = [
        result

        for result
        in results

        if result.dataset_selection_applicable
    ]


    planner_retries = [
        result

        for result
        in results

        if (
            result.planner_retry_count
            >
            0
        )
    ]


    execution_retries = [
        result

        for result
        in results

        if (
            result.planner_retry_class
            ==
            "execution_recovery"
        )
    ]


    guardrail_retries = [
        result

        for result
        in results

        if (
            result.planner_retry_class
            ==
            "guardrail_abstention"
        )
    ]


    unresolved_retries = [
        result

        for result
        in results

        if (
            result.planner_retry_class
            ==
            "unresolved_retry"
        )
    ]


    tool_retries = [
        result

        for result
        in executable

        if (
            result.tool_retry_count
            >
            0
        )
    ]


    latencies = [
        result.latency_seconds

        for result
        in results
    ]


    return {
        "run_count":
            len(
                results
            ),

        "executable_run_count":
            len(
                executable
            ),

        "guardrail_run_count":
            len(
                guardrails
            ),

        "overall_pass_rate":
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
            (
                rate(
                    [
                        result.planner_bindings_correct

                        for result
                        in executable
                    ]
                )
                if executable
                else None
            ),

        "planner_dataset_selection_accuracy":
            (
                rate(
                    [
                        result.dataset_selection_correct

                        for result
                        in dataset_selection_runs
                    ]
                )
                if dataset_selection_runs
                else None
            ),

        "planner_first_pass_rate":
            rate(
                [
                    result.planner_first_pass

                    for result
                    in results
                ]
            ),

        "planner_first_pass_rate_executable":
            rate(
                [
                    result.planner_first_pass

                    for result
                    in executable
                ]
            )
            if executable
            else None,

        "planner_retry_rate":
            (
                len(
                    planner_retries
                )
                /
                len(
                    results
                )
                if results
                else 0.0
            ),

        "planner_retry_recovery_rate":
            rate(
                [
                    result.planner_recovered_after_retry

                    for result
                    in planner_retries
                ]
            )
            if planner_retries
            else None,

        "planner_execution_retry_rate":
            (
                len(
                    execution_retries
                )
                /
                len(
                    executable
                )
                if executable
                else 0.0
            ),

        "planner_guardrail_retry_rate":
            (
                len(
                    guardrail_retries
                )
                /
                len(
                    guardrails
                )
                if guardrails
                else None
            ),

        "planner_unresolved_retry_count":
            len(
                unresolved_retries
            ),

        "planner_normalization_rate":
            rate(
                [
                    (
                        result
                        .planner_normalization_count
                        >
                        0
                    )

                    for result
                    in results
                ]
            ),

        "native_tool_selection_accuracy":
            (
                rate(
                    [
                        result.tool_selection_correct

                        for result
                        in executable
                    ]
                )
                if executable
                else None
            ),

        "native_argument_accuracy":
            (
                rate(
                    [
                        result.tool_arguments_correct

                        for result
                        in executable
                    ]
                )
                if executable
                else None
            ),

        "native_first_pass_rate":
            (
                rate(
                    [
                        result.tool_first_pass

                        for result
                        in executable
                    ]
                )
                if executable
                else None
            ),

        "native_retry_rate":
            (
                len(
                    tool_retries
                )
                /
                len(
                    executable
                )
                if executable
                else None
            ),

        "native_retry_recovery_rate":
            rate(
                [
                    result.tool_recovered_after_retry

                    for result
                    in tool_retries
                ]
            )
            if tool_retries
            else None,

        "execution_success_rate":
            rate(
                [
                    result.execution_correct

                    for result
                    in results
                ]
            ),

        "chart_type_accuracy":
            (
                rate(
                    [
                        result.chart_type_correct

                        for result
                        in executable
                    ]
                )
                if executable
                else None
            ),

        "guardrail_accuracy":
            rate(
                [
                    result.guardrail_correct

                    for result
                    in guardrails
                ]
            )
            if guardrails
            else None,

        "latency_seconds_mean":
            (
                statistics.mean(
                    latencies
                )
                if latencies
                else 0.0
            ),

        "latency_seconds_median":
            (
                statistics.median(
                    latencies
                )
                if latencies
                else 0.0
            ),

        "failed_runs":
            [
                {
                    "case_id":
                        result.case_id,

                    "repeat_index":
                        result.repeat_index,

                    "failure_reasons":
                        result.failure_reasons,
                }

                for result
                in results

                if not result.case_pass
            ],
    }


def write_results_csv(
    *,
    results: list[
        EvalRunResult
    ],
    path: Path,
) -> None:
    rows = [
        asdict(
            result
        )

        for result
        in results
    ]


    if not rows:
        return


    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def format_rate(
    value: float | None,
) -> str:
    if value is None:
        return "n/a"


    return (
        f"{value * 100:.1f}%"
    )


def print_summary(
    summary: dict[
        str,
        Any,
    ],
) -> None:
    print()
    print(
        "=== DataLens tool-use eval summary ==="
    )

    print(
        f"Runs                    : {summary['run_count']}"
    )

    print(
        "Overall pass rate       : "
        f"{format_rate(summary['overall_pass_rate'])}"
    )

    print(
        "Planner family accuracy : "
        f"{format_rate(summary['planner_family_accuracy'])}"
    )

    print(
        "Planner binding accuracy: "
        f"{format_rate(summary['planner_binding_accuracy'])}"
    )

    print(
        "Planner dataset accuracy: "
        f"{format_rate(summary['planner_dataset_selection_accuracy'])}"
    )

    print(
        "Planner first-pass rate  : "
        f"{format_rate(summary['planner_first_pass_rate'])}"
    )

    print(
        "Planner first-pass exec. : "
        f"{format_rate(summary['planner_first_pass_rate_executable'])}"
    )

    print(
        "Planner retry rate       : "
        f"{format_rate(summary['planner_retry_rate'])}"
    )

    print(
        "Planner retry recovery   : "
        f"{format_rate(summary['planner_retry_recovery_rate'])}"
    )

    print(
        "Execution retry rate     : "
        f"{format_rate(summary['planner_execution_retry_rate'])}"
    )

    print(
        "Guardrail retry rate     : "
        f"{format_rate(summary['planner_guardrail_retry_rate'])}"
    )

    print(
        "Unresolved retries       : "
        f"{summary['planner_unresolved_retry_count']}"
    )

    print(
        "Planner normalization    : "
        f"{format_rate(summary['planner_normalization_rate'])}"
    )

    print(
        "Tool selection accuracy  : "
        f"{format_rate(summary['native_tool_selection_accuracy'])}"
    )

    print(
        "Tool argument accuracy   : "
        f"{format_rate(summary['native_argument_accuracy'])}"
    )

    print(
        "Tool first-pass rate     : "
        f"{format_rate(summary['native_first_pass_rate'])}"
    )

    print(
        "Tool retry rate          : "
        f"{format_rate(summary['native_retry_rate'])}"
    )

    print(
        "Tool retry recovery      : "
        f"{format_rate(summary['native_retry_recovery_rate'])}"
    )

    print(
        "Execution success rate   : "
        f"{format_rate(summary['execution_success_rate'])}"
    )

    print(
        "Chart type accuracy      : "
        f"{format_rate(summary['chart_type_accuracy'])}"
    )

    print(
        "Guardrail accuracy       : "
        f"{format_rate(summary['guardrail_accuracy'])}"
    )

    print(
        "Median latency           : "
        f"{summary['latency_seconds_median']:.2f}s"
    )


# ============================================================
# PER-CASE SUMMARY
# ============================================================

def build_case_summary(
    results: list[
        EvalRunResult
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    grouped: dict[
        str,
        list[
            EvalRunResult
        ],
    ] = {}


    for result in (
        results
    ):
        grouped.setdefault(
            result.case_id,
            [],
        ).append(
            result
        )


    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for case_id in sorted(
        grouped
    ):
        case_results = (
            grouped[
                case_id
            ]
        )


        latencies = [
            item.latency_seconds

            for item
            in case_results
        ]


        executable_case_results = [
            item

            for item
            in case_results

            if item.should_execute
        ]


        guardrail_case_results = [
            item

            for item
            in case_results

            if not item.should_execute
        ]


        dataset_selection_case_results = [
            item

            for item
            in case_results

            if item.dataset_selection_applicable
        ]


        planner_retry_runs = [
            item

            for item
            in case_results

            if (
                item.planner_retry_count
                >
                0
            )
        ]


        tool_retry_runs = [
            item

            for item
            in case_results

            if (
                item.tool_retry_count
                >
                0
            )
        ]


        rows.append(
            {
                "case_id":
                    case_id,

                "runs":
                    len(
                        case_results
                    ),

                "case_kind":
                    (
                        "executable"
                        if executable_case_results
                        else "guardrail"
                    ),

                "pass_rate":
                    rate(
                        [
                            item.case_pass

                            for item
                            in case_results
                        ]
                    ),

                "planner_family_accuracy":
                    rate(
                        [
                            item.planner_family_correct

                            for item
                            in case_results
                        ]
                    ),

                "planner_binding_accuracy":
                    (
                        rate(
                            [
                                item.planner_bindings_correct

                                for item
                                in executable_case_results
                            ]
                        )
                        if executable_case_results
                        else None
                    ),

                "planner_dataset_selection_accuracy":
                    (
                        rate(
                            [
                                item.dataset_selection_correct

                                for item
                                in dataset_selection_case_results
                            ]
                        )
                        if dataset_selection_case_results
                        else None
                    ),

                "planner_first_pass_rate":
                    rate(
                        [
                            item.planner_first_pass

                            for item
                            in case_results
                        ]
                    ),

                "planner_retry_rate":
                    (
                        len(
                            planner_retry_runs
                        )
                        /
                        len(
                            case_results
                        )
                    ),

                "planner_retry_classes":
                    ", ".join(
                        sorted(
                            set(
                                item.planner_retry_class

                                for item
                                in case_results
                            )
                        )
                    ),

                "planner_retry_recovery_rate":
                    (
                        rate(
                            [
                                item
                                .planner_recovered_after_retry

                                for item
                                in planner_retry_runs
                            ]
                        )
                        if planner_retry_runs
                        else None
                    ),

                "planner_normalization_rate":
                    rate(
                        [
                            (
                                item
                                .planner_normalization_count
                                >
                                0
                            )

                            for item
                            in case_results
                        ]
                    ),

                "tool_selection_accuracy":
                    (
                        rate(
                            [
                                item.tool_selection_correct

                                for item
                                in executable_case_results
                            ]
                        )
                        if executable_case_results
                        else None
                    ),

                "tool_argument_accuracy":
                    (
                        rate(
                            [
                                item.tool_arguments_correct

                                for item
                                in executable_case_results
                            ]
                        )
                        if executable_case_results
                        else None
                    ),

                "tool_first_pass_rate":
                    (
                        rate(
                            [
                                item.tool_first_pass

                                for item
                                in executable_case_results
                            ]
                        )
                        if executable_case_results
                        else None
                    ),

                "tool_retry_rate":
                    (
                        len(
                            tool_retry_runs
                        )
                        /
                        len(
                            executable_case_results
                        )
                        if executable_case_results
                        else None
                    ),

                "tool_retry_recovery_rate":
                    (
                        rate(
                            [
                                item
                                .tool_recovered_after_retry

                                for item
                                in tool_retry_runs
                            ]
                        )
                        if tool_retry_runs
                        else None
                    ),

                "execution_accuracy":
                    rate(
                        [
                            item.execution_correct

                            for item
                            in case_results
                        ]
                    ),

                "guardrail_accuracy":
                    rate(
                        [
                            item.guardrail_correct

                            for item
                            in case_results
                        ]
                    ),

                "latency_mean_seconds":
                    statistics.mean(
                        latencies
                    )
                    if latencies
                    else 0.0,

                "latency_median_seconds":
                    statistics.median(
                        latencies
                    )
                    if latencies
                    else 0.0,

                "planner_statuses":
                    ", ".join(
                        sorted(
                            set(
                                item.planner_status

                                for item
                                in case_results
                            )
                        )
                    ),

                "requested_tools":
                    ", ".join(
                        sorted(
                            set(
                                (
                                    item.requested_tool
                                    or
                                    "none"
                                )

                                for item
                                in case_results
                            )
                        )
                    ),

                "failure_count":
                    sum(
                        1

                        for item
                        in case_results

                        if not item.case_pass
                    ),
            }
        )


    return rows


def write_case_summary_csv(
    *,
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
    path: Path,
) -> None:
    if not rows:
        return


    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def print_case_summary(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    print()
    print(
        "=== Per-case stability ==="
    )


    for row in (
        rows
    ):
        print(
            (
                f"{row['case_id']:<30} "
                f"pass={format_rate(row['pass_rate']):>6} · "
                f"planner first={format_rate(row['planner_first_pass_rate']):>6} · "
                f"planner retry={format_rate(row['planner_retry_rate']):>6} · "
                f"retry kind={row['planner_retry_classes']:<20} · "
                f"tool first={format_rate(row['tool_first_pass_rate']):>6} · "
                f"median={row['latency_median_seconds']:.2f}s"
            )
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live DataLens planner + native tool-use "
            "evaluations against fixed ground-truth cases."
        )
    )


    parser.add_argument(
        "--cases",
        type=Path,
        default=(
            DEFAULT_CASES_PATH
        ),
    )


    parser.add_argument(
        "--planner-model",
        default="gemma3:4b",
    )


    parser.add_argument(
        "--tool-model",
        default="qwen2.5:1.5b-instruct",
    )


    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
    )


    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        default=[],
        help=(
            "Run only the given case_id. "
            "Can be passed multiple times."
        ),
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            HERE
            /
            "results"
        ),
    )


    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional stable name for this eval run. "
            "By default DataLens creates a timestamped run id."
        ),
    )


    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help=(
            "Exit with status 1 if any eval run fails."
        ),
    )


    return (
        parser.parse_args()
    )


def main() -> None:
    args = (
        parse_args()
    )


    if (
        args.repeat <
        1
    ):
        raise SystemExit(
            "--repeat must be >= 1"
        )


    cases = (
        load_cases(
            args.cases
        )
    )


    if args.case_ids:
        requested = set(
            args.case_ids
        )

        cases = [
            case

            for case
            in cases

            if (
                case[
                    "case_id"
                ]
                in requested
            )
        ]


    if not cases:
        raise SystemExit(
            "No evaluation cases selected."
        )


    output_root = (
        args.output_dir
        .resolve()
    )


    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    run_id = (
        str(
            args.run_id
        ).strip()
        if (
            args.run_id
            is not None
            and
            str(
                args.run_id
            ).strip()
        )
        else datetime.now().strftime(
            "%Y%m%dT%H%M%S_%f"
        )
    )


    output_dir = (
        output_root
        /
        "runs"
        /
        run_id
    )


    raw_dir = (
        output_dir
        /
        "raw"
    )


    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        output_root
        /
        "latest_run.txt"
    ).write_text(
        str(
            output_dir
        ),
        encoding="utf-8",
    )


    results: list[
        EvalRunResult
    ] = []


    total_runs = (
        len(
            cases
        )
        *
        args.repeat
    )


    run_index = 0


    for repeat_index in range(
        1,
        args.repeat + 1,
    ):
        for case in (
            cases
        ):
            run_index += 1


            print(
                (
                    f"[{run_index}/{total_runs}] "
                    f"{case['case_id']} "
                    f"(repeat {repeat_index})"
                )
            )


            result = (
                evaluate_case(
                    case=(
                        case
                    ),
                    repeat_index=(
                        repeat_index
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
            )


            results.append(
                result
            )


            print(
                (
                    "  "
                    +
                    (
                        "PASS"
                        if result.case_pass
                        else "FAIL"
                    )
                    +
                    f" · planner={result.planner_status}/"
                    f"{result.planner_family}"
                    +
                    (
                        f" · tool={result.requested_tool}"
                        if result.requested_tool
                        else
                        " · tool=none"
                    )
                    +
                    f" · {result.latency_seconds:.2f}s"
                )
            )


            if (
                result.failure_reasons
            ):
                print(
                    "  "
                    +
                    result.failure_reasons
                )


    summary = (
        build_summary(
            results
        )
    )


    case_summary = (
        build_case_summary(
            results
        )
    )


    summary[
        "by_case"
    ] = (
        case_summary
    )


    results_csv = (
        output_dir
        /
        "results.csv"
    )


    summary_json = (
        output_dir
        /
        "summary.json"
    )


    case_summary_csv = (
        output_dir
        /
        "summary_by_case.csv"
    )


    write_results_csv(
        results=(
            results
        ),
        path=(
            results_csv
        ),
    )


    write_case_summary_csv(
        rows=(
            case_summary
        ),
        path=(
            case_summary_csv
        ),
    )


    summary_json.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print_summary(
        summary
    )


    print_case_summary(
        case_summary
    )


    print()
    print(
        f"Run ID  : {run_id}"
    )

    print(
        f"Run dir : {output_dir}"
    )

    print(
        f"CSV     : {results_csv}"
    )

    print(
        f"Summary : {summary_json}"
    )

    print(
        f"By case : {case_summary_csv}"
    )

    print(
        f"Raw JSON: {raw_dir}"
    )


    if (
        args.fail_on_regression
        and
        summary[
            "overall_pass_rate"
        ]
        <
        1.0
    ):
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()

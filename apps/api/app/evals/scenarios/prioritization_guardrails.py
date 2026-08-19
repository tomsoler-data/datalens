from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)

from app.evals.analysis_benchmark import (
    AnalysisBenchmarkExpectation,
    AnalysisBenchmarkScenario,
    BenchmarkVariableExpectation,
)


EVAL_COVERAGE_RULE_VERSION = (
    "eval_coverage_v0.1"
)


DATASET_ID = (
    "dataset:prioritization_coverage"
)

DATASET_FILENAME = (
    "prioritization_coverage.csv"
)


@dataclass(frozen=True)
class ControlledPrioritizationEval:
    """
    One deterministic Prioritization eval.

    Discovery is intentionally controlled here.

    The goal is not to re-test Discovery. The goal is to place a
    precise DiscoveredAnalysis contract in front of
    prioritize_analysis_discovery() and verify its decision and
    reason code.

    Real Discovery + Prioritization integration remains covered by
    the core analysis scenarios.
    """

    scenario: AnalysisBenchmarkScenario
    discovery: AnalysisDiscoveryReport


# ============================================================
# GENERIC HELPERS
# ============================================================


def _variable(
    *,
    column: str,
    role: str,
    analysis_kind: str = "quantitative",
    semantic_role: str = "measure",
) -> DiscoveredVariable:
    return (
        DiscoveredVariable(
            dataset_id=
                DATASET_ID,

            dataset_filename=
                DATASET_FILENAME,

            column=
                column,

            role=
                role,

            analysis_kind=
                analysis_kind,

            semantic_role=
                semantic_role,

            concepts=[],
        )
    )


def _candidate(
    *,
    analysis_id: str,
    family: str,
    variables: list[
        DiscoveredVariable
    ],
    score: float = 90.0,
    readiness: str = "executable_now",
    observed_signals: dict | None = None,
    title: str | None = None,
) -> DiscoveredAnalysis:
    return (
        DiscoveredAnalysis(
            analysis_id=
                analysis_id,

            scope=
                "single_dataset",

            family=
                family,

            title=
                title
                or
                analysis_id,

            priority_score=
                score,

            readiness=
                readiness,

            datasets=[
                DATASET_FILENAME
            ],

            dataset_ids=[
                DATASET_ID
            ],

            variables=
                variables,

            chart_type=
                "eval",

            execution_strategy=
                "eval",

            why_interesting=[],

            limitations=[],

            relationship_status=
                None,

            relationship_score=
                None,

            join_keys={},

            observed_signals=
                observed_signals
                or {},

            redundancy_key=
                analysis_id,
        )
    )


def _discovery(
    candidates: list[
        DiscoveredAnalysis
    ],
) -> AnalysisDiscoveryReport:
    return (
        AnalysisDiscoveryReport(
            objective=
                None,

            dataset_count=
                1,

            candidate_count=
                len(
                    candidates
                ),

            single_dataset_candidate_count=
                len(
                    candidates
                ),

            cross_dataset_candidate_count=
                0,

            candidates=
                candidates,

            relationships=[],

            discovery_notes=[
                EVAL_COVERAGE_RULE_VERSION
            ],
        )
    )


def _dataset(
    dataframe: pd.DataFrame | None = None,
) -> tuple[
    dict,
    ...,
]:
    if dataframe is None:
        dataframe = pd.DataFrame(
            {
                "placeholder": [
                    1,
                    2,
                    3,
                    4,
                ]
            }
        )


    return (
        {
            "dataset_id":
                DATASET_ID,

            "filename":
                DATASET_FILENAME,

            "dataframe":
                dataframe,
        },
    )


def _scenario(
    *,
    scenario_id: str,
    description: str,
    datasets: tuple[
        dict,
        ...,
    ],
    expectation: AnalysisBenchmarkExpectation,
) -> AnalysisBenchmarkScenario:
    return (
        AnalysisBenchmarkScenario(
            scenario_id=
                scenario_id,

            description=
                description,

            split=
                "test",

            frozen=
                True,

            datasets=
                datasets,

            expectations=(
                expectation,
            ),

            min_discovered_count=
                1,

            max_selected_count=
                36,
        )
    )


# ============================================================
# 1 — IDENTIFIER MISUSE
# ============================================================


def identifier_misuse_eval(
) -> ControlledPrioritizationEval:
    candidate = (
        _candidate(
            analysis_id=
                "guardrail:identifier-misuse",

            family=
                "quantitative_association",

            variables=[
                _variable(
                    column=
                        "customer_id",

                    role=
                        "x",

                    analysis_kind=
                        "identifier",

                    semantic_role=
                        "identifier",
                ),

                _variable(
                    column=
                        "unit_price",

                    role=
                        "y",
                ),
            ],
        )
    )


    expectation = (
        AnalysisBenchmarkExpectation(
            expectation_id=
                "reject-identifier-misuse",

            description=(
                "An identifier used as an analytical measure must "
                "be rejected before execution."
            ),

            family=
                "quantitative_association",

            variables=[
                BenchmarkVariableExpectation(
                    role=
                        "x",

                    column=
                        "customer_id",
                ),

                BenchmarkVariableExpectation(
                    role=
                        "y",

                    column=
                        "unit_price",
                ),
            ],

            allowed_decisions=[
                "rejected"
            ],

            allowed_reason_codes=[
                "identifier_misuse"
            ],
        )
    )


    return (
        ControlledPrioritizationEval(
            scenario=
                _scenario(
                    scenario_id=
                        "guardrail_identifier_misuse_v0.1",

                    description=
                        "Identifier analytical misuse guard.",

                    datasets=
                        _dataset(),

                    expectation=
                        expectation,
                ),

            discovery=
                _discovery(
                    [
                        candidate
                    ]
                ),
        )
    )


# ============================================================
# 2 — RECORD LABEL DIMENSION
# ============================================================


def record_label_dimension_eval(
) -> ControlledPrioritizationEval:
    dataframe = pd.DataFrame(
        {
            "first_name": [
                f"Name_{index % 16}"

                for index
                in range(
                    40
                )
            ],

            "quantity": [
                index % 5 + 1

                for index
                in range(
                    40
                )
            ],
        }
    )


    candidate = (
        _candidate(
            analysis_id=
                "guardrail:record-label",

            family=
                "group_comparison",

            variables=[
                _variable(
                    column=
                        "first_name",

                    role=
                        "group",

                    analysis_kind=
                        "categorical",

                    semantic_role=
                        "category",
                ),

                _variable(
                    column=
                        "quantity",

                    role=
                        "value",
                ),
            ],
        )
    )


    expectation = (
        AnalysisBenchmarkExpectation(
            expectation_id=
                "defer-record-label",

            description=(
                "A high-cardinality record label must not consume "
                "the automatic exploratory budget."
            ),

            family=
                "group_comparison",

            variables=[
                BenchmarkVariableExpectation(
                    role=
                        "group",

                    column=
                        "first_name",
                ),

                BenchmarkVariableExpectation(
                    role=
                        "value",

                    column=
                        "quantity",
                ),
            ],

            allowed_decisions=[
                "deferred"
            ],

            allowed_reason_codes=[
                "record_label_dimension"
            ],
        )
    )


    return (
        ControlledPrioritizationEval(
            scenario=
                _scenario(
                    scenario_id=
                        "guardrail_record_label_v0.1",

                    description=
                        "High-cardinality record-label guard.",

                    datasets=
                        _dataset(
                            dataframe
                        ),

                    expectation=
                        expectation,
                ),

            discovery=
                _discovery(
                    [
                        candidate
                    ]
                ),
        )
    )


# ============================================================
# 3 — SPARSE CATEGORICAL STRUCTURE
# ============================================================


def sparse_categorical_eval(
) -> ControlledPrioritizationEval:
    candidate = (
        _candidate(
            analysis_id=
                "guardrail:sparse-categorical",

            family=
                "categorical_association",

            variables=[
                _variable(
                    column=
                        "left_category",

                    role=
                        "x",

                    analysis_kind=
                        "categorical",

                    semantic_role=
                        "category",
                ),

                _variable(
                    column=
                        "right_category",

                    role=
                        "y",

                    analysis_kind=
                        "categorical",

                    semantic_role=
                        "category",
                ),
            ],

            observed_signals={
                "valid_observations":
                    40,

                "left_levels":
                    10,

                "right_levels":
                    8,
            },
        )
    )


    expectation = (
        AnalysisBenchmarkExpectation(
            expectation_id=
                "defer-sparse-categorical",

            description=(
                "A categorical contingency structure that is too "
                "sparse must be deferred."
            ),

            family=
                "categorical_association",

            variables=[
                BenchmarkVariableExpectation(
                    role=
                        "x",

                    column=
                        "left_category",
                ),

                BenchmarkVariableExpectation(
                    role=
                        "y",

                    column=
                        "right_category",
                ),
            ],

            allowed_decisions=[
                "deferred"
            ],

            allowed_reason_codes=[
                "sparse_categorical_structure"
            ],
        )
    )


    return (
        ControlledPrioritizationEval(
            scenario=
                _scenario(
                    scenario_id=
                        "guardrail_sparse_categorical_v0.1",

                    description=
                        "Sparse contingency-table guard.",

                    datasets=
                        _dataset(),

                    expectation=
                        expectation,
                ),

            discovery=
                _discovery(
                    [
                        candidate
                    ]
                ),
        )
    )


# ============================================================
# 4 — NON EXECUTABLE
# ============================================================


def non_executable_eval(
) -> ControlledPrioritizationEval:
    candidate = (
        _candidate(
            analysis_id=
                "guardrail:not-executable",

            family=
                "distribution",

            variables=[
                _variable(
                    column=
                        "amount",

                    role=
                        "value",
                )
            ],

            score=
                95.0,

            readiness=
                "planned",
        )
    )


    expectation = (
        AnalysisBenchmarkExpectation(
            expectation_id=
                "defer-not-executable",

            description=(
                "A candidate that is not executable now must remain "
                "outside the execution shortlist."
            ),

            family=
                "distribution",

            variables=[
                BenchmarkVariableExpectation(
                    role=
                        "value",

                    column=
                        "amount",
                )
            ],

            allowed_decisions=[
                "deferred"
            ],

            allowed_reason_codes=[
                "not_executable_now"
            ],
        )
    )


    return (
        ControlledPrioritizationEval(
            scenario=
                _scenario(
                    scenario_id=
                        "guardrail_not_executable_v0.1",

                    description=
                        "Readiness guard.",

                    datasets=
                        _dataset(),

                    expectation=
                        expectation,
                ),

            discovery=
                _discovery(
                    [
                        candidate
                    ]
                ),
        )
    )


# ============================================================
# 5 — QUALITY GUARD
# ============================================================


def quality_guard_eval(
) -> ControlledPrioritizationEval:
    candidate = (
        _candidate(
            analysis_id=
                "guardrail:quality",

            family=
                "data_quality",

            variables=[
                _variable(
                    column=
                        "amount",

                    role=
                        "value",
                )
            ],

            score=
                5.0,
        )
    )


    expectation = (
        AnalysisBenchmarkExpectation(
            expectation_id=
                "preserve-quality-evidence",

            description=(
                "Data-quality evidence must remain selectable even "
                "when its discovery priority is low."
            ),

            family=
                "data_quality",

            variables=[
                BenchmarkVariableExpectation(
                    role=
                        "value",

                    column=
                        "amount",
                )
            ],

            allowed_decisions=[
                "selected"
            ],

            allowed_reason_codes=[
                "quality_guard"
            ],
        )
    )


    return (
        ControlledPrioritizationEval(
            scenario=
                _scenario(
                    scenario_id=
                        "guardrail_quality_preservation_v0.1",

                    description=
                        "Data-quality evidence preservation.",

                    datasets=
                        _dataset(),

                    expectation=
                        expectation,
                ),

            discovery=
                _discovery(
                    [
                        candidate
                    ]
                ),
        )
    )


# ============================================================
# 6 — PRIORITY THRESHOLD
# ============================================================


def priority_threshold_eval(
) -> ControlledPrioritizationEval:
    candidate = (
        _candidate(
            analysis_id=
                "guardrail:priority-threshold",

            family=
                "distribution",

            variables=[
                _variable(
                    column=
                        "amount",

                    role=
                        "value",
                )
            ],

            score=
                10.0,
        )
    )


    expectation = (
        AnalysisBenchmarkExpectation(
            expectation_id=
                "defer-low-priority",

            description=(
                "Low-value exploratory candidates must not consume "
                "the automatic execution budget."
            ),

            family=
                "distribution",

            variables=[
                BenchmarkVariableExpectation(
                    role=
                        "value",

                    column=
                        "amount",
                )
            ],

            allowed_decisions=[
                "deferred"
            ],

            allowed_reason_codes=[
                "priority_below_threshold"
            ],
        )
    )


    return (
        ControlledPrioritizationEval(
            scenario=
                _scenario(
                    scenario_id=
                        "guardrail_priority_threshold_v0.1",

                    description=
                        "Minimum execution-priority threshold.",

                    datasets=
                        _dataset(),

                    expectation=
                        expectation,
                ),

            discovery=
                _discovery(
                    [
                        candidate
                    ]
                ),
        )
    )


# ============================================================
# 7 — FAMILY BUDGET
# ============================================================


def family_budget_eval(
) -> ControlledPrioritizationEval:
    candidates: list[
        DiscoveredAnalysis
    ] = []


    for index in range(
        13
    ):
        candidates.append(
            _candidate(
                analysis_id=
                    f"family-budget:{index:02d}",

                family=
                    "quantitative_association",

                score=
                    99.0
                    -
                    (
                        index
                        *
                        0.5
                    ),

                variables=[
                    _variable(
                        column=
                            f"x_{index}",

                        role=
                            "x",
                    ),

                    _variable(
                        column=
                            f"y_{index}",

                        role=
                            "y",
                    ),
                ],
            )
        )


    expectation = (
        AnalysisBenchmarkExpectation(
            expectation_id=
                "defer-family-overflow",

            description=(
                "The thirteenth quantitative association must be "
                "deferred after the family budget is exhausted."
            ),

            family=
                "quantitative_association",

            variables=[
                BenchmarkVariableExpectation(
                    role=
                        "x",

                    column=
                        "x_12",
                ),

                BenchmarkVariableExpectation(
                    role=
                        "y",

                    column=
                        "y_12",
                ),
            ],

            allowed_decisions=[
                "deferred"
            ],

            allowed_reason_codes=[
                "family_budget_exhausted"
            ],
        )
    )


    return (
        ControlledPrioritizationEval(
            scenario=
                _scenario(
                    scenario_id=
                        "guardrail_family_budget_v0.1",

                    description=
                        "Per-family diversity budget.",

                    datasets=
                        _dataset(),

                    expectation=
                        expectation,
                ),

            discovery=
                _discovery(
                    candidates
                ),
        )
    )


# ============================================================
# 8 — VARIABLE BUDGET
# ============================================================


def variable_budget_eval(
) -> ControlledPrioritizationEval:
    candidates: list[
        DiscoveredAnalysis
    ] = []


    for index in range(
        10
    ):
        candidates.append(
            _candidate(
                analysis_id=
                    f"variable-budget:{index:02d}",

                family=
                    "quantitative_association",

                score=
                    99.0
                    -
                    (
                        index
                        *
                        0.5
                    ),

                variables=[
                    _variable(
                        column=
                            "revenue",

                        role=
                            "x",
                    ),

                    _variable(
                        column=
                            f"metric_{index}",

                        role=
                            "y",
                    ),
                ],
            )
        )


    expectation = (
        AnalysisBenchmarkExpectation(
            expectation_id=
                "defer-variable-overuse",

            description=(
                "The ninth use of the same analytical variable must "
                "be deferred by the variable-occurrence budget."
            ),

            family=
                "quantitative_association",

            variables=[
                BenchmarkVariableExpectation(
                    role=
                        "x",

                    column=
                        "revenue",
                ),

                BenchmarkVariableExpectation(
                    role=
                        "y",

                    column=
                        "metric_8",
                ),
            ],

            allowed_decisions=[
                "deferred"
            ],

            allowed_reason_codes=[
                "variable_budget_exhausted"
            ],
        )
    )


    return (
        ControlledPrioritizationEval(
            scenario=
                _scenario(
                    scenario_id=
                        "guardrail_variable_budget_v0.1",

                    description=
                        "Repeated-variable redundancy budget.",

                    datasets=
                        _dataset(),

                    expectation=
                        expectation,
                ),

            discovery=
                _discovery(
                    candidates
                ),
        )
    )


# ============================================================
# 9 — GLOBAL EXECUTION BUDGET
# ============================================================


def global_budget_eval(
) -> ControlledPrioritizationEval:
    candidates: list[
        DiscoveredAnalysis
    ] = []


    score = 100.0


    # 12 quantitative associations.
    for index in range(
        12
    ):
        candidates.append(
            _candidate(
                analysis_id=
                    f"global:assoc:{index:02d}",

                family=
                    "quantitative_association",

                score=
                    score,

                variables=[
                    _variable(
                        column=
                            f"assoc_x_{index}",

                        role=
                            "x",
                    ),

                    _variable(
                        column=
                            f"assoc_y_{index}",

                        role=
                            "y",
                    ),
                ],
            )
        )

        score -= 0.1


    # 8 group comparisons.
    for index in range(
        8
    ):
        candidates.append(
            _candidate(
                analysis_id=
                    f"global:group:{index:02d}",

                family=
                    "group_comparison",

                score=
                    score,

                variables=[
                    _variable(
                        column=
                            f"group_{index}",

                        role=
                            "group",

                        analysis_kind=
                            "categorical",

                        semantic_role=
                            "category",
                    ),

                    _variable(
                        column=
                            f"group_value_{index}",

                        role=
                            "value",
                    ),
                ],
            )
        )

        score -= 0.1


    # 6 categorical associations.
    for index in range(
        6
    ):
        candidates.append(
            _candidate(
                analysis_id=
                    f"global:categorical:{index:02d}",

                family=
                    "categorical_association",

                score=
                    score,

                variables=[
                    _variable(
                        column=
                            f"cat_left_{index}",

                        role=
                            "x",

                        analysis_kind=
                            "categorical",

                        semantic_role=
                            "category",
                    ),

                    _variable(
                        column=
                            f"cat_right_{index}",

                        role=
                            "y",

                        analysis_kind=
                            "categorical",

                        semantic_role=
                            "category",
                    ),
                ],

                observed_signals={
                    "valid_observations":
                        120,

                    "left_levels":
                        3,

                    "right_levels":
                        4,
                },
            )
        )

        score -= 0.1


    # 6 distributions.
    for index in range(
        6
    ):
        candidates.append(
            _candidate(
                analysis_id=
                    f"global:distribution:{index:02d}",

                family=
                    "distribution",

                score=
                    score,

                variables=[
                    _variable(
                        column=
                            f"distribution_value_{index}",

                        role=
                            "value",
                    )
                ],
            )
        )

        score -= 0.1


    # 4 selected time series = 36 total selected candidates.
    for index in range(
        4
    ):
        candidates.append(
            _candidate(
                analysis_id=
                    f"global:time:{index:02d}",

                family=
                    "time_series",

                score=
                    score,

                variables=[
                    _variable(
                        column=
                            f"time_{index}",

                        role=
                            "time",

                        analysis_kind=
                            "temporal",

                        semantic_role=
                            "time",
                    ),

                    _variable(
                        column=
                            f"time_value_{index}",

                        role=
                            "value",
                    ),
                ],
            )
        )

        score -= 0.1


    # Candidate 37:
    # family time_series still has room (cap = 5), variables are
    # unique, but the global budget is already full.
    candidates.append(
        _candidate(
            analysis_id=
                "global:overflow",

            family=
                "time_series",

            score=
                score,

            variables=[
                _variable(
                    column=
                        "overflow_time",

                    role=
                        "time",

                    analysis_kind=
                        "temporal",

                    semantic_role=
                        "time",
                ),

                _variable(
                    column=
                        "overflow_value",

                    role=
                        "value",
                ),
            ],
        )
    )


    expectation = (
        AnalysisBenchmarkExpectation(
            expectation_id=
                "defer-global-overflow",

            description=(
                "Candidate 37 must be deferred after the global "
                "execution budget reaches 36."
            ),

            family=
                "time_series",

            variables=[
                BenchmarkVariableExpectation(
                    role=
                        "time",

                    column=
                        "overflow_time",
                ),

                BenchmarkVariableExpectation(
                    role=
                        "value",

                    column=
                        "overflow_value",
                ),
            ],

            allowed_decisions=[
                "deferred"
            ],

            allowed_reason_codes=[
                "global_budget_exhausted"
            ],
        )
    )


    return (
        ControlledPrioritizationEval(
            scenario=
                _scenario(
                    scenario_id=
                        "guardrail_global_budget_v0.1",

                    description=
                        "Global automatic-execution budget.",

                    datasets=
                        _dataset(),

                    expectation=
                        expectation,
                ),

            discovery=
                _discovery(
                    candidates
                ),
        )
    )


# ============================================================
# REGISTRY
# ============================================================


def build_prioritization_guardrail_evals(
) -> tuple[
    ControlledPrioritizationEval,
    ...,
]:
    """
    Stable ordering is intentional.

    Combined with the three real-pipeline scenarios, this registry
    covers every PrioritizationReasonCode introduced by
    Analysis Prioritization v0.1 and Analytical Value Guard v0.1.
    """

    return (
        identifier_misuse_eval(),
        record_label_dimension_eval(),
        sparse_categorical_eval(),
        non_executable_eval(),
        quality_guard_eval(),
        priority_threshold_eval(),
        family_budget_eval(),
        variable_budget_eval(),
        global_budget_eval(),
    )

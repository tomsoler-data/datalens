from __future__ import annotations

from dataclasses import (
    dataclass,
)

import hashlib
import json

from typing import (
    Any,
    Literal,
)

import pandas as pd

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.analysis_prioritization import (
    ANALYSIS_PRIORITIZATION_RULE_VERSION,
    ANALYTICAL_VALUE_GUARD_RULE_VERSION,
    AnalysisPrioritizationDecision,
    AnalysisPrioritizationReport,
    PrioritizationDecision,
    PrioritizationReasonCode,
    prioritize_analysis_discovery,
)

from app.discovery import (
    discover_analyses,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
)

from app.evals.schemas import (
    EvalSplit,
)


ANALYSIS_BENCHMARK_RULE_VERSION = (
    "analysis_benchmark_v0.1"
)


BenchmarkFailureCode = Literal[
    "missing_candidate",
    "ambiguous_candidate",
    "missing_decision",
    "unexpected_decision",
    "unexpected_reason",
]


# ============================================================
# EXPECTATION CONTRACTS
# ============================================================


class BenchmarkVariableExpectation(
    BaseModel,
):
    """
    Structural variable matcher.

    Benchmarks intentionally match candidates by analytical
    structure rather than by analysis_id alone.

    This keeps evaluation stable when public candidate identity
    evolves while the underlying analytical meaning remains the
    same.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    role: str
    column: str


class AnalysisBenchmarkExpectation(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )

    expectation_id: str

    description: str = ""

    family: str

    variables: list[
        BenchmarkVariableExpectation
    ] = Field(
        default_factory=list
    )

    title_contains: str | None = None

    must_be_discovered: bool = True

    allowed_decisions: list[
        PrioritizationDecision
    ] = Field(
        default_factory=lambda: [
            "selected"
        ]
    )

    allowed_reason_codes: list[
        PrioritizationReasonCode
    ] = Field(
        default_factory=list
    )


@dataclass(frozen=True)
class AnalysisBenchmarkScenario:
    """
    Versioned Discovery/Prioritization eval scenario.

    This contract intentionally remains distinct from
    ``AnalyticalEvalCase``:

    - AnalyticalEvalCase evaluates planner / LLM reasoning from
      metadata-only DatasetContext objects.
    - AnalysisBenchmarkScenario evaluates deterministic Python
      Discovery + Prioritization against local DataFrames.

    Both systems share the same EvalSplit vocabulary and the
    same frozen-test discipline.

    DataFrames remain outside Pydantic on purpose so the
    benchmark can execute directly against the same local
    in-memory dataset contract used by DataLens.

    Dataset records are deep-copied before every benchmark run.
    """

    scenario_id: str
    description: str

    split: EvalSplit

    datasets: tuple[
        dict[
            str,
            Any,
        ],
        ...,
    ]

    expectations: tuple[
        AnalysisBenchmarkExpectation,
        ...,
    ]

    objective: str | None = None

    min_discovered_count: int | None = None

    max_selected_count: int | None = None

    frozen: bool = False


    def __post_init__(
        self,
    ) -> None:
        if (
            not str(
                self.scenario_id
            ).strip()
        ):
            raise ValueError(
                "scenario_id must not be empty."
            )


        if (
            self.split
            ==
            "test"
            and
            not self.frozen
        ):
            raise ValueError(
                "All test benchmark scenarios must be frozen."
            )


        if (
            not self.datasets
        ):
            raise ValueError(
                "A benchmark scenario must contain at least "
                "one dataset."
            )


# ============================================================
# RESULT CONTRACTS
# ============================================================


class AnalysisBenchmarkOutcome(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )

    expectation_id: str
    description: str

    passed: bool

    failure_code: (
        BenchmarkFailureCode
        | None
    ) = None

    matched_candidate_count: int

    matched_analysis_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    matched_title: str | None = None

    actual_decision: (
        PrioritizationDecision
        | None
    ) = None

    actual_reason_code: (
        PrioritizationReasonCode
        | None
    ) = None

    allowed_decisions: list[
        PrioritizationDecision
    ] = Field(
        default_factory=list
    )

    allowed_reason_codes: list[
        PrioritizationReasonCode
    ] = Field(
        default_factory=list
    )

    notes: list[str] = Field(
        default_factory=list
    )


class AnalysisBenchmarkMetrics(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )

    expectation_count: int

    passed_expectation_count: int

    failed_expectation_count: int

    expectation_accuracy: float

    required_discovery_count: int

    discovered_required_count: int

    discovery_recall: float

    required_selection_count: int

    selected_required_count: int

    selection_recall: float

    guardrail_expectation_count: int

    guardrail_pass_count: int

    guardrail_success_rate: float

    discovered_candidate_count: int

    selected_candidate_count: int

    deferred_candidate_count: int

    rejected_candidate_count: int

    deterministic: bool


class AnalysisBenchmarkReport(
    BaseModel,
):
    model_config = ConfigDict(
        extra="forbid"
    )

    scenario_id: str
    description: str

    split: EvalSplit
    frozen: bool

    passed: bool

    metrics: AnalysisBenchmarkMetrics

    outcomes: list[
        AnalysisBenchmarkOutcome
    ]

    discovery_fingerprint: str

    prioritization_fingerprint: str

    deterministic_run_count: int

    notes: list[str]

    benchmark_rule_version: str = (
        ANALYSIS_BENCHMARK_RULE_VERSION
    )

    prioritization_rule_version: str = (
        ANALYSIS_PRIORITIZATION_RULE_VERSION
    )

    analytical_value_guard_rule_version: str = (
        ANALYTICAL_VALUE_GUARD_RULE_VERSION
    )


# ============================================================
# DATASET CLONING
# ============================================================


def _clone_dataset_records(
    datasets: tuple[
        dict[
            str,
            Any,
        ],
        ...,
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    output: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for record in datasets:
        clone = dict(
            record
        )


        dataframe = (
            record.get(
                "dataframe"
            )
        )


        if (
            not isinstance(
                dataframe,
                pd.DataFrame,
            )
        ):
            raise TypeError(
                "Every benchmark dataset record must contain "
                "a pandas DataFrame under 'dataframe'."
            )


        clone[
            "dataframe"
        ] = dataframe.copy(
            deep=True
        )


        output.append(
            clone
        )


    return output


# ============================================================
# STRUCTURAL MATCHING
# ============================================================


def _candidate_variable_pairs(
    candidate: DiscoveredAnalysis,
) -> set[
    tuple[
        str,
        str,
    ]
]:
    return {
        (
            variable.role,
            variable.column,
        )

        for variable
        in candidate.variables
    }


def _expectation_variable_pairs(
    expectation: AnalysisBenchmarkExpectation,
) -> set[
    tuple[
        str,
        str,
    ]
]:
    return {
        (
            variable.role,
            variable.column,
        )

        for variable
        in expectation.variables
    }


def _candidate_matches_expectation(
    *,
    candidate: DiscoveredAnalysis,
    expectation: AnalysisBenchmarkExpectation,
) -> bool:
    if (
        candidate.family
        !=
        expectation.family
    ):
        return False


    expected_variables = (
        _expectation_variable_pairs(
            expectation
        )
    )


    candidate_variables = (
        _candidate_variable_pairs(
            candidate
        )
    )


    if (
        not expected_variables
        .issubset(
            candidate_variables
        )
    ):
        return False


    if (
        expectation.title_contains
        is not None
        and
        expectation.title_contains
        not in
        candidate.title
    ):
        return False


    return True


# ============================================================
# DECISION MATCHING
# ============================================================


def _decision_signature(
    decision: AnalysisPrioritizationDecision,
) -> tuple[
    str,
    str,
    tuple[
        str,
        ...,
    ],
]:
    return (
        decision.family,
        decision.title,
        tuple(
            sorted(
                decision.variable_keys
            )
        ),
    )


def _candidate_signature(
    candidate: DiscoveredAnalysis,
) -> tuple[
    str,
    str,
    tuple[
        str,
        ...,
    ],
]:
    variable_keys = tuple(
        sorted(
            f"{variable.dataset_id}:{variable.column}"

            for variable
            in candidate.variables
        )
    )


    return (
        candidate.family,
        candidate.title,
        variable_keys,
    )


def _decision_for_candidate(
    *,
    candidate: DiscoveredAnalysis,
    prioritization: AnalysisPrioritizationReport,
) -> (
    AnalysisPrioritizationDecision
    | None
):
    target = (
        _candidate_signature(
            candidate
        )
    )


    matches = [
        decision

        for decision
        in prioritization.decisions

        if (
            _decision_signature(
                decision
            )
            ==
            target
        )
    ]


    if (
        len(
            matches
        )
        !=
        1
    ):
        return None


    return matches[
        0
    ]


# ============================================================
# FINGERPRINTS
# ============================================================


def _stable_hash(
    payload: Any,
) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )


    return hashlib.sha256(
        encoded
    ).hexdigest()


def discovery_fingerprint(
    discovery: AnalysisDiscoveryReport,
) -> str:
    payload = [
        {
            "analysis_id":
                candidate.analysis_id,

            "family":
                candidate.family,

            "title":
                candidate.title,

            "priority_score":
                candidate.priority_score,

            "readiness":
                candidate.readiness,

            "variables": [
                {
                    "dataset_id":
                        variable.dataset_id,

                    "column":
                        variable.column,

                    "role":
                        variable.role,
                }

                for variable
                in candidate.variables
            ],

            "redundancy_key":
                candidate.redundancy_key,
        }

        for candidate
        in discovery.candidates
    ]


    return _stable_hash(
        payload
    )


def prioritization_fingerprint(
    report: AnalysisPrioritizationReport,
) -> str:
    payload = [
        {
            "analysis_id":
                decision.analysis_id,

            "family":
                decision.family,

            "title":
                decision.title,

            "decision":
                decision.decision,

            "reason_code":
                decision.reason_code,

            "execution_priority_score":
                decision
                .execution_priority_score,

            "variable_keys":
                list(
                    decision.variable_keys
                ),
        }

        for decision
        in report.decisions
    ]


    return _stable_hash(
        payload
    )


# ============================================================
# EXPECTATION EVALUATION
# ============================================================


def _evaluate_expectation(
    *,
    expectation: AnalysisBenchmarkExpectation,
    discovery: AnalysisDiscoveryReport,
    prioritization: AnalysisPrioritizationReport,
) -> AnalysisBenchmarkOutcome:
    matches = [
        candidate

        for candidate
        in discovery.candidates

        if (
            _candidate_matches_expectation(
                candidate=
                    candidate,

                expectation=
                    expectation,
            )
        )
    ]


    matched_ids = [
        candidate.analysis_id

        for candidate
        in matches
    ]


    if not matches:
        if (
            expectation.must_be_discovered
        ):
            return (
                AnalysisBenchmarkOutcome(
                    expectation_id=
                        expectation
                        .expectation_id,

                    description=
                        expectation
                        .description,

                    passed=
                        False,

                    failure_code=
                        "missing_candidate",

                    matched_candidate_count=
                        0,

                    matched_analysis_ids=
                        [],

                    allowed_decisions=
                        list(
                            expectation
                            .allowed_decisions
                        ),

                    allowed_reason_codes=
                        list(
                            expectation
                            .allowed_reason_codes
                        ),

                    notes=[
                        (
                            "No discovered candidate matched "
                            "the structural benchmark expectation."
                        )
                    ],
                )
            )


        return (
            AnalysisBenchmarkOutcome(
                expectation_id=
                    expectation
                    .expectation_id,

                description=
                    expectation
                    .description,

                passed=
                    True,

                matched_candidate_count=
                    0,

                matched_analysis_ids=
                    [],

                allowed_decisions=
                    list(
                        expectation
                        .allowed_decisions
                    ),

                allowed_reason_codes=
                    list(
                        expectation
                        .allowed_reason_codes
                    ),

                notes=[
                    (
                        "The candidate was optional and "
                        "was not discovered."
                    )
                ],
            )
        )


    if (
        len(
            matches
        )
        >
        1
    ):
        return (
            AnalysisBenchmarkOutcome(
                expectation_id=
                    expectation
                    .expectation_id,

                description=
                    expectation
                    .description,

                passed=
                    False,

                failure_code=
                    "ambiguous_candidate",

                matched_candidate_count=
                    len(
                        matches
                    ),

                matched_analysis_ids=
                    matched_ids,

                allowed_decisions=
                    list(
                        expectation
                        .allowed_decisions
                    ),

                allowed_reason_codes=
                    list(
                        expectation
                        .allowed_reason_codes
                    ),

                notes=[
                    (
                        "More than one Discovery candidate "
                        "matched the same benchmark structure."
                    )
                ],
            )
        )


    candidate = matches[
        0
    ]


    decision = (
        _decision_for_candidate(
            candidate=
                candidate,

            prioritization=
                prioritization,
        )
    )


    if (
        decision
        is None
    ):
        return (
            AnalysisBenchmarkOutcome(
                expectation_id=
                    expectation
                    .expectation_id,

                description=
                    expectation
                    .description,

                passed=
                    False,

                failure_code=
                    "missing_decision",

                matched_candidate_count=
                    1,

                matched_analysis_ids=[
                    candidate
                    .analysis_id
                ],

                matched_title=
                    candidate.title,

                allowed_decisions=
                    list(
                        expectation
                        .allowed_decisions
                    ),

                allowed_reason_codes=
                    list(
                        expectation
                        .allowed_reason_codes
                    ),

                notes=[
                    (
                        "The candidate was discovered but no "
                        "unique prioritization decision matched "
                        "its structural signature."
                    )
                ],
            )
        )


    if (
        decision.decision
        not in
        expectation.allowed_decisions
    ):
        return (
            AnalysisBenchmarkOutcome(
                expectation_id=
                    expectation
                    .expectation_id,

                description=
                    expectation
                    .description,

                passed=
                    False,

                failure_code=
                    "unexpected_decision",

                matched_candidate_count=
                    1,

                matched_analysis_ids=[
                    candidate
                    .analysis_id
                ],

                matched_title=
                    candidate.title,

                actual_decision=
                    decision.decision,

                actual_reason_code=
                    decision.reason_code,

                allowed_decisions=
                    list(
                        expectation
                        .allowed_decisions
                    ),

                allowed_reason_codes=
                    list(
                        expectation
                        .allowed_reason_codes
                    ),

                notes=[
                    (
                        "Prioritization returned a decision "
                        "outside the benchmark contract."
                    )
                ],
            )
        )


    if (
        expectation.allowed_reason_codes
        and
        decision.reason_code
        not in
        expectation.allowed_reason_codes
    ):
        return (
            AnalysisBenchmarkOutcome(
                expectation_id=
                    expectation
                    .expectation_id,

                description=
                    expectation
                    .description,

                passed=
                    False,

                failure_code=
                    "unexpected_reason",

                matched_candidate_count=
                    1,

                matched_analysis_ids=[
                    candidate
                    .analysis_id
                ],

                matched_title=
                    candidate.title,

                actual_decision=
                    decision.decision,

                actual_reason_code=
                    decision.reason_code,

                allowed_decisions=
                    list(
                        expectation
                        .allowed_decisions
                    ),

                allowed_reason_codes=
                    list(
                        expectation
                        .allowed_reason_codes
                    ),

                notes=[
                    (
                        "The decision was acceptable, but its "
                        "deterministic reason code did not match "
                        "the benchmark contract."
                    )
                ],
            )
        )


    return (
        AnalysisBenchmarkOutcome(
            expectation_id=
                expectation
                .expectation_id,

            description=
                expectation
                .description,

            passed=
                True,

            matched_candidate_count=
                1,

            matched_analysis_ids=[
                candidate
                .analysis_id
            ],

            matched_title=
                candidate.title,

            actual_decision=
                decision.decision,

            actual_reason_code=
                decision.reason_code,

            allowed_decisions=
                list(
                    expectation
                    .allowed_decisions
                ),

            allowed_reason_codes=
                list(
                    expectation
                    .allowed_reason_codes
                ),

            notes=[
                (
                    "Discovery and Prioritization matched "
                    "the benchmark expectation."
                )
            ],
        )
    )


# ============================================================
# METRICS
# ============================================================


def _safe_rate(
    numerator: int,
    denominator: int,
) -> float:
    if (
        denominator
        <=
        0
    ):
        return 1.0


    return round(
        numerator
        /
        denominator,
        4,
    )


def _build_metrics(
    *,
    expectations: tuple[
        AnalysisBenchmarkExpectation,
        ...,
    ],
    outcomes: list[
        AnalysisBenchmarkOutcome
    ],
    prioritization: AnalysisPrioritizationReport,
    deterministic: bool,
) -> AnalysisBenchmarkMetrics:
    passed = sum(
        outcome.passed

        for outcome
        in outcomes
    )


    required_discovery = [
        expectation

        for expectation
        in expectations

        if (
            expectation
            .must_be_discovered
        )
    ]


    discovered_required = sum(
        outcome.matched_candidate_count
        ==
        1

        for (
            expectation,
            outcome,
        ) in zip(
            expectations,
            outcomes,
        )

        if (
            expectation
            .must_be_discovered
        )
    )


    required_selection = [
        expectation

        for expectation
        in expectations

        if (
            expectation
            .allowed_decisions
            ==
            [
                "selected"
            ]
        )
    ]


    selected_required = sum(
        (
            outcome.actual_decision
            ==
            "selected"
            and
            outcome.passed
        )

        for (
            expectation,
            outcome,
        ) in zip(
            expectations,
            outcomes,
        )

        if (
            expectation
            .allowed_decisions
            ==
            [
                "selected"
            ]
        )
    )


    guardrail_expectations = [
        expectation

        for expectation
        in expectations

        if (
            "selected"
            not in
            expectation
            .allowed_decisions
        )
    ]


    guardrail_pass = sum(
        outcome.passed

        for (
            expectation,
            outcome,
        ) in zip(
            expectations,
            outcomes,
        )

        if (
            "selected"
            not in
            expectation
            .allowed_decisions
        )
    )


    return (
        AnalysisBenchmarkMetrics(
            expectation_count=
                len(
                    expectations
                ),

            passed_expectation_count=
                passed,

            failed_expectation_count=
                (
                    len(
                        expectations
                    )
                    -
                    passed
                ),

            expectation_accuracy=
                _safe_rate(
                    passed,
                    len(
                        expectations
                    ),
                ),

            required_discovery_count=
                len(
                    required_discovery
                ),

            discovered_required_count=
                discovered_required,

            discovery_recall=
                _safe_rate(
                    discovered_required,
                    len(
                        required_discovery
                    ),
                ),

            required_selection_count=
                len(
                    required_selection
                ),

            selected_required_count=
                selected_required,

            selection_recall=
                _safe_rate(
                    selected_required,
                    len(
                        required_selection
                    ),
                ),

            guardrail_expectation_count=
                len(
                    guardrail_expectations
                ),

            guardrail_pass_count=
                guardrail_pass,

            guardrail_success_rate=
                _safe_rate(
                    guardrail_pass,
                    len(
                        guardrail_expectations
                    ),
                ),

            discovered_candidate_count=
                prioritization
                .discovered_count,

            selected_candidate_count=
                prioritization
                .selected_count,

            deferred_candidate_count=
                prioritization
                .deferred_count,

            rejected_candidate_count=
                prioritization
                .rejected_count,

            deterministic=
                deterministic,
        )
    )


# ============================================================
# EVALUATE PRECOMPUTED REPORTS
# ============================================================


def evaluate_analysis_benchmark(
    *,
    scenario: AnalysisBenchmarkScenario,
    discovery: AnalysisDiscoveryReport,
    prioritization: AnalysisPrioritizationReport,
    deterministic: bool = True,
    deterministic_run_count: int = 1,
) -> AnalysisBenchmarkReport:
    outcomes = [
        _evaluate_expectation(
            expectation=
                expectation,

            discovery=
                discovery,

            prioritization=
                prioritization,
        )

        for expectation
        in scenario.expectations
    ]


    metrics = (
        _build_metrics(
            expectations=
                scenario
                .expectations,

            outcomes=
                outcomes,

            prioritization=
                prioritization,

            deterministic=
                deterministic,
        )
    )


    constraints_pass = True

    constraint_notes: list[
        str
    ] = []


    if (
        scenario.min_discovered_count
        is not None
        and
        discovery.candidate_count
        <
        scenario.min_discovered_count
    ):
        constraints_pass = False

        constraint_notes.append(
            (
                "Discovery count below scenario minimum: "
                f"{discovery.candidate_count} < "
                f"{scenario.min_discovered_count}."
            )
        )


    if (
        scenario.max_selected_count
        is not None
        and
        prioritization.selected_count
        >
        scenario.max_selected_count
    ):
        constraints_pass = False

        constraint_notes.append(
            (
                "Selected count above scenario maximum: "
                f"{prioritization.selected_count} > "
                f"{scenario.max_selected_count}."
            )
        )


    passed = (
        metrics.failed_expectation_count
        ==
        0
        and
        metrics.deterministic
        and
        constraints_pass
    )


    return (
        AnalysisBenchmarkReport(
            scenario_id=
                scenario.scenario_id,

            description=
                scenario.description,

            split=
                scenario.split,

            frozen=
                scenario.frozen,

            passed=
                passed,

            metrics=
                metrics,

            outcomes=
                outcomes,

            discovery_fingerprint=
                discovery_fingerprint(
                    discovery
                ),

            prioritization_fingerprint=
                prioritization_fingerprint(
                    prioritization
                ),

            deterministic_run_count=
                deterministic_run_count,

            notes=[
                (
                    "This benchmark shares EvalSplit and frozen "
                    "test-set semantics with DataLens planner evals, "
                    "while keeping a separate contract for "
                    "deterministic Discovery/Prioritization."
                ),
                (
                    "Expectations are matched by analytical "
                    "structure (family + variable roles/columns), "
                    "not only by analysis_id."
                ),
                (
                    "Discovery recall measures whether required "
                    "benchmark candidates were discovered."
                ),
                (
                    "Selection recall measures whether candidates "
                    "explicitly expected to be selected actually "
                    "entered the execution shortlist."
                ),
                (
                    "Guardrail success measures expectations that "
                    "explicitly forbid automatic selection."
                ),
                (
                    "Determinism compares complete Discovery and "
                    "Prioritization fingerprints across repeated "
                    "runs of the same scenario."
                ),
                *constraint_notes,
            ],
        )
    )


# ============================================================
# FULL PIPELINE BENCHMARK RUNNER
# ============================================================


def run_analysis_benchmark(
    scenario: AnalysisBenchmarkScenario,
    *,
    deterministic_runs: int = 2,
) -> AnalysisBenchmarkReport:
    if (
        deterministic_runs
        <
        1
    ):
        raise ValueError(
            "deterministic_runs must be at least 1."
        )


    discoveries: list[
        AnalysisDiscoveryReport
    ] = []

    prioritizations: list[
        AnalysisPrioritizationReport
    ] = []


    for _ in range(
        deterministic_runs
    ):
        datasets = (
            _clone_dataset_records(
                scenario.datasets
            )
        )


        discovery = (
            discover_analyses(
                datasets=
                    datasets,

                objective=
                    scenario.objective,
            )
        )


        prioritization = (
            prioritize_analysis_discovery(
                discovery,

                datasets=
                    datasets,
            )
        )


        discoveries.append(
            discovery
        )


        prioritizations.append(
            prioritization
        )


    discovery_hashes = [
        discovery_fingerprint(
            discovery
        )

        for discovery
        in discoveries
    ]


    prioritization_hashes = [
        prioritization_fingerprint(
            report
        )

        for report
        in prioritizations
    ]


    deterministic = (
        len(
            set(
                discovery_hashes
            )
        )
        ==
        1
        and
        len(
            set(
                prioritization_hashes
            )
        )
        ==
        1
    )


    return (
        evaluate_analysis_benchmark(
            scenario=
                scenario,

            discovery=
                discoveries[
                    0
                ],

            prioritization=
                prioritizations[
                    0
                ],

            deterministic=
                deterministic,

            deterministic_run_count=
                deterministic_runs,
        )
    )

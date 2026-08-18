from __future__ import annotations

from datetime import (
    datetime,
)

from pathlib import (
    Path,
)

from app.evaluation.experiment_schemas import (
    ExperimentPhase,
    SemanticExperimentSnapshot,
    SemanticSystemManifest,
)

from app.evaluation.registry import (
    SemanticBenchmarkRegistry,
)

from app.evaluation.runner_schemas import (
    SemanticGlobalBenchmarkResult,
)


# ============================================================
# BUILD SNAPSHOT
# ============================================================

def build_semantic_experiment_snapshot(
    *,
    experiment_id: str,
    phase: ExperimentPhase,
    registry: SemanticBenchmarkRegistry,
    result: SemanticGlobalBenchmarkResult,
    system: SemanticSystemManifest,
    notes: list[
        str
    ] | None = None,
) -> SemanticExperimentSnapshot:
    suites = (
        registry.list_suites(
            split=
                result.split,
        )
    )


    result_benchmark_ids = {
        suite.benchmark_id
        for suite
        in result.suites
    }


    registry_benchmark_ids = {
        suite.benchmark_id
        for suite
        in suites
    }


    if (
        result_benchmark_ids
        !=
        registry_benchmark_ids
    ):
        missing = (
            registry_benchmark_ids
            -
            result_benchmark_ids
        )


        unexpected = (
            result_benchmark_ids
            -
            registry_benchmark_ids
        )


        raise ValueError(
            "Experiment result does not match "
            f"registry split {result.split}. "
            f"Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}."
        )


    benchmark_versions = {
        suite.benchmark_id:
            suite.benchmark_version

        for suite
        in suites
    }


    benchmark_ids = sorted(
        registry_benchmark_ids
    )


    return SemanticExperimentSnapshot(
        experiment_id=
            experiment_id,

        created_at=
            datetime.now().astimezone(),

        phase=
            phase,

        split=
            result.split,

        benchmark_ids=
            benchmark_ids,

        benchmark_versions=
            benchmark_versions,

        system=
            system,

        result=
            result,

        notes=(
            notes
            if notes is not None
            else []
        ),
    )


# ============================================================
# SAVE SNAPSHOT
# ============================================================

def save_semantic_experiment_snapshot(
    *,
    snapshot: SemanticExperimentSnapshot,
    path: str | Path,
) -> Path:
    output_path = Path(
        path
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path.write_text(
        snapshot.model_dump_json(
            indent=2,
        ),
        encoding="utf-8",
    )


    return output_path


# ============================================================
# LOAD SNAPSHOT
# ============================================================

def load_semantic_experiment_snapshot(
    *,
    path: str | Path,
) -> SemanticExperimentSnapshot:
    input_path = Path(
        path
    )


    payload = input_path.read_text(
        encoding="utf-8",
    )


    return (
        SemanticExperimentSnapshot
        .model_validate_json(
            payload
        )
    )


# ============================================================
# SNAPSHOT SUMMARY
# ============================================================

def summarize_semantic_experiment_snapshot(
    snapshot: SemanticExperimentSnapshot,
) -> dict[
    str,
    object,
]:
    result = snapshot.result


    normalized_safety = (
        result.normalized_safety_decisions
    )


    raw_safety = (
        result.raw_safety_decisions
    )


    return {
        "experiment_id":
            snapshot.experiment_id,

        "created_at":
            snapshot.created_at.isoformat(),

        "phase":
            snapshot.phase,

        "split":
            snapshot.split,

        "model":
            snapshot.system.model_name,

        "system_label":
            snapshot.system.system_label,

        "benchmarks":
            snapshot.benchmark_ids,

        "suite_count":
            result.suite_count,

        "assertion_count":
            result.normalized_assertion_count,

        "raw_micro_accuracy":
            result.raw_micro_accuracy,

        "datalens_micro_accuracy":
            result.normalized_micro_accuracy,

        "micro_delta":
            result.micro_accuracy_delta,

        "raw_macro_accuracy":
            result.raw_macro_accuracy,

        "datalens_macro_accuracy":
            result.normalized_macro_accuracy,

        "macro_delta":
            result.macro_accuracy_delta,

        # Legacy counts remain visible.
        "raw_safety_assertion_errors":
            result.raw_safety_failure_count,

        "datalens_safety_assertion_errors":
            result.normalized_safety_failure_count,

        # Direction-aware v0.2 safety metrics.
        "raw_dangerous_false_positives": (
            raw_safety.false_positive_count
            if raw_safety is not None
            else None
        ),

        "raw_valid_operations_missed": (
            raw_safety.false_negative_count
            if raw_safety is not None
            else None
        ),

        "datalens_dangerous_false_positives": (
            normalized_safety.false_positive_count
            if normalized_safety is not None
            else None
        ),

        "datalens_valid_operations_missed": (
            normalized_safety.false_negative_count
            if normalized_safety is not None
            else None
        ),

        "datalens_safety_precision": (
            normalized_safety.precision
            if normalized_safety is not None
            else None
        ),

        "datalens_capability_recall": (
            normalized_safety.recall
            if normalized_safety is not None
            else None
        ),

        "normalized_failures":
            result.normalized_failure_count,

        "safety_gate_passed":
            result.safety_gate_passed,

        "regression_gate_passed":
            result.regression_gate_passed,

        "safety_gate_rule_version":
            result.safety_gate_rule_version,

        "runner_rule_version":
            result.runner_rule_version,

        "snapshot_rule_version":
            snapshot.snapshot_rule_version,
    }

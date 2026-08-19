from __future__ import annotations

from app.analysis_prioritization import (
    build_prioritized_execution_discovery,
    prioritize_analysis_discovery,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)


DATASET_ID = "dataset:test"


def variable(
    *,
    column: str,
    role: str,
    semantic_role: str,
    analysis_kind: str,
) -> DiscoveredVariable:
    return DiscoveredVariable(
        dataset_id=DATASET_ID,
        dataset_filename="test.csv",
        column=column,
        role=role,
        analysis_kind=analysis_kind,
        semantic_role=semantic_role,
        concepts=[],
    )


def time_candidate(
    *,
    title: str,
    time_column: str,
    readiness: str,
    score: float,
) -> DiscoveredAnalysis:
    """
    Reproduce the legacy Discovery collision:
    same dataset + same value + different time column
    while analysis_id stays identical.
    """

    return DiscoveredAnalysis(
        analysis_id=(
            f"{DATASET_ID}:time:quantity"
        ),
        scope="single_dataset",
        family="time_series",
        title=title,
        priority_score=score,
        readiness=readiness,
        datasets=["test.csv"],
        dataset_ids=[DATASET_ID],
        variables=[
            variable(
                column=time_column,
                role="time",
                semantic_role="time",
                analysis_kind="temporal",
            ),
            variable(
                column="quantity",
                role="value",
                semantic_role="measure",
                analysis_kind="quantitative",
            ),
        ],
        chart_type="line",
        execution_strategy="descriptive_time_series",
        why_interesting=[],
        limitations=[],
        relationship_status=None,
        relationship_score=None,
        join_keys={},
        observed_signals={
            "period_count": 12,
        },
        redundancy_key=(
            f"time:{DATASET_ID}:{time_column}:quantity"
        ),
    )


def build_collision_discovery(
) -> AnalysisDiscoveryReport:
    candidates = [
        time_candidate(
            title="Quantity selon order_date",
            time_column="order_date",
            readiness="executable_now",
            score=95.0,
        ),
        time_candidate(
            title="Quantity selon signup_date",
            time_column="signup_date",
            readiness="planned",
            score=94.0,
        ),
    ]

    return AnalysisDiscoveryReport(
        objective=None,
        dataset_count=1,
        candidate_count=2,
        single_dataset_candidate_count=2,
        cross_dataset_candidate_count=0,
        candidates=candidates,
        relationships=[],
        discovery_notes=[],
    )


def test_duplicate_analysis_id_does_not_leak_deferred_candidate(
) -> None:
    discovery = (
        build_collision_discovery()
    )

    report = (
        prioritize_analysis_discovery(
            discovery
        )
    )

    assert report.selected_count == 1
    assert report.deferred_count == 1

    # The fixture intentionally keeps the legacy collision.
    assert (
        report.decisions[0].analysis_id
        ==
        report.decisions[1].analysis_id
    )

    execution_discovery = (
        build_prioritized_execution_discovery(
            source_discovery=discovery,
            prioritization=report,
        )
    )

    assert (
        execution_discovery.candidate_count
        ==
        1
    )

    assert (
        len(
            execution_discovery.candidates
        )
        ==
        1
    )

    selected = (
        execution_discovery.candidates[0]
    )

    assert (
        selected.title
        ==
        "Quantity selon order_date"
    )

    assert (
        selected.variables[0].column
        ==
        "order_date"
    )

    assert (
        selected.readiness
        ==
        "executable_now"
    )

    print(
        "Duplicate legacy analysis_id cannot leak a deferred candidate into execution: PASS"
    )


def test_source_discovery_remains_unchanged(
) -> None:
    discovery = (
        build_collision_discovery()
    )

    original_titles = [
        item.title
        for item
        in discovery.candidates
    ]

    report = (
        prioritize_analysis_discovery(
            discovery
        )
    )

    _ = (
        build_prioritized_execution_discovery(
            source_discovery=discovery,
            prioritization=report,
        )
    )

    assert discovery.candidate_count == 2

    assert (
        [
            item.title
            for item
            in discovery.candidates
        ]
        ==
        original_titles
    )

    print(
        "Broad Discovery remains immutable after duplicate-ID-safe selection: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS PRIORITIZATION DUPLICATE ID SAFETY v0.1 ==="
    )

    print()

    test_duplicate_analysis_id_does_not_leak_deferred_candidate()
    test_source_discovery_remains_unchanged()

    print()

    print(
        "Prioritization Duplicate ID Safety v0.1: PASS"
    )


if __name__ == "__main__":
    main()

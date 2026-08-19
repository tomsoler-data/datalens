from __future__ import annotations

import inspect

from dataclasses import (
    dataclass,
    field,
)

import app.api.analysis_run as analysis_run

from app.analysis_prioritization import (
    ANALYSIS_PRIORITIZATION_RULE_VERSION,
    MAX_SELECTED_ANALYSES,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)


def variable(
    *,
    column: str,
    role: str,
    semantic_role: str,
) -> DiscoveredVariable:
    return DiscoveredVariable(
        dataset_id="dataset:test",
        dataset_filename="test.csv",
        column=column,
        role=role,
        analysis_kind="quantitative",
        semantic_role=semantic_role,
        concepts=[],
    )


def candidate(
    *,
    analysis_id: str,
    family: str,
    score: float,
    variables: list[DiscoveredVariable],
) -> DiscoveredAnalysis:
    return DiscoveredAnalysis(
        analysis_id=analysis_id,
        scope="single_dataset",
        family=family,
        title=analysis_id,
        priority_score=score,
        readiness="executable_now",
        datasets=["test.csv"],
        dataset_ids=["dataset:test"],
        variables=variables,
        chart_type="test",
        execution_strategy="test",
        why_interesting=[],
        limitations=[],
        relationship_status=None,
        relationship_score=None,
        join_keys={},
        observed_signals={},
        redundancy_key=analysis_id,
    )


def build_large_discovery() -> AnalysisDiscoveryReport:
    candidates: list[DiscoveredAnalysis] = []

    candidates.append(
        candidate(
            analysis_id="identifier-misuse",
            family="quantitative_association",
            score=100.0,
            variables=[
                variable(
                    column="order_id",
                    role="x",
                    semantic_role="identifier",
                ),
                variable(
                    column="price",
                    role="y",
                    semantic_role="measure",
                ),
            ],
        )
    )

    families = [
        "distribution",
        "quantitative_association",
        "group_comparison",
        "categorical_association",
        "time_series",
        "derived_gap",
        "entity_ranking",
        "geographic_comparison",
        "other_family",
    ]

    for index in range(
        1,
        147,
    ):
        family = families[
            index
            %
            len(
                families
            )
        ]

        candidates.append(
            candidate(
                analysis_id=
                    f"analysis-{index:03d}",
                family=
                    family,
                score=
                    99.0
                    -
                    (
                        index
                        *
                        0.1
                    ),
                variables=[
                    variable(
                        column=
                            f"metric_{index}",
                        role=
                            "value",
                        semantic_role=
                            "measure",
                    )
                ],
            )
        )

    return AnalysisDiscoveryReport(
        objective=None,
        dataset_count=1,
        candidate_count=len(
            candidates
        ),
        single_dataset_candidate_count=len(
            candidates
        ),
        cross_dataset_candidate_count=0,
        candidates=candidates,
        relationships=[],
        discovery_notes=[],
    )


@dataclass
class FakeInventory:
    discovered_analysis_count: int = 0


@dataclass
class FakeReport:
    inventory: FakeInventory = field(
        default_factory=FakeInventory
    )
    methodology_notes: list[str] = field(
        default_factory=list
    )


def test_exploratory_execution_uses_shortlist_but_report_keeps_broad_count(
) -> None:
    discovery = build_large_discovery()

    original_ids = [
        item.analysis_id
        for item
        in discovery.candidates
    ]

    calls: dict[str, object] = {}

    originals = {
        "execute_single_dataset_discovery":
            analysis_run.execute_single_dataset_discovery,
        "inject_aggregate_breakdown_execution":
            analysis_run.inject_aggregate_breakdown_execution,
        "execute_cross_dataset_discovery":
            analysis_run.execute_cross_dataset_discovery,
        "rank_unified_analysis":
            analysis_run.rank_unified_analysis,
        "apply_aggregate_breakdown_ranking_policy":
            analysis_run.apply_aggregate_breakdown_ranking_policy,
        "apply_feature_lineage_ranking_policy":
            analysis_run.apply_feature_lineage_ranking_policy,
        "compose_unified_report":
            analysis_run.compose_unified_report,
        "normalize_report_aggregate_families":
            analysis_run.normalize_report_aggregate_families,
    }

    try:
        def fake_single(
            *,
            discovery,
            datasets,
        ):
            calls[
                "single_discovery"
            ] = discovery
            return {
                "single":
                    True,
            }

        def fake_inject(
            single_execution,
            *,
            discovery,
            datasets,
        ):
            calls[
                "inject_discovery"
            ] = discovery
            return single_execution

        def fake_cross(
            *,
            discovery,
            datasets,
        ):
            calls[
                "cross_discovery"
            ] = discovery
            return {
                "cross":
                    True,
            }

        def fake_rank(
            *,
            discovery,
            single_execution,
            cross_execution,
            datasets,
        ):
            calls[
                "ranking_discovery"
            ] = discovery
            return {
                "ranking":
                    True,
            }

        def fake_aggregate_ranking(
            ranking,
            *,
            discovery,
        ):
            calls[
                "aggregate_ranking_discovery"
            ] = discovery
            return ranking

        def fake_lineage_ranking(
            ranking,
            *,
            discovery,
        ):
            calls[
                "lineage_ranking_discovery"
            ] = discovery
            return ranking

        def fake_compose(
            *,
            discovery,
            single_execution,
            cross_execution,
            ranking,
            datasets,
            title,
        ):
            calls[
                "compose_discovery"
            ] = discovery

            return FakeReport(
                inventory=
                    FakeInventory(
                        discovered_analysis_count=
                            discovery.candidate_count
                    )
            )

        def fake_normalize(
            report,
            *,
            discovery,
        ):
            calls[
                "normalize_discovery"
            ] = discovery
            return report

        analysis_run.execute_single_dataset_discovery = (
            fake_single
        )
        analysis_run.inject_aggregate_breakdown_execution = (
            fake_inject
        )
        analysis_run.execute_cross_dataset_discovery = (
            fake_cross
        )
        analysis_run.rank_unified_analysis = (
            fake_rank
        )
        analysis_run.apply_aggregate_breakdown_ranking_policy = (
            fake_aggregate_ranking
        )
        analysis_run.apply_feature_lineage_ranking_policy = (
            fake_lineage_ranking
        )
        analysis_run.compose_unified_report = (
            fake_compose
        )
        analysis_run.normalize_report_aggregate_families = (
            fake_normalize
        )

        report = (
            analysis_run
            .run_unified_analysis_from_prepared_records(
                source_dataset_records=[
                    {
                        "dataset_id":
                            "dataset:test",
                        "filename":
                            "test.csv",
                        "dataframe":
                            object(),
                    }
                ],
                discovery=discovery,
                analysis_datasets=[
                    {
                        "dataset_id":
                            "dataset:test",
                        "filename":
                            "test.csv",
                        "dataframe":
                            object(),
                    }
                ],
            )
        )

    finally:
        for name, function in originals.items():
            setattr(
                analysis_run,
                name,
                function,
            )

    shortlist = calls[
        "single_discovery"
    ]

    assert (
        shortlist.candidate_count
        <=
        MAX_SELECTED_ANALYSES
    )

    assert (
        shortlist.candidate_count
        <
        discovery.candidate_count
    )

    assert (
        "identifier-misuse"
        not in
        {
            item.analysis_id
            for item
            in shortlist.candidates
        }
    )

    for key in [
        "inject_discovery",
        "cross_discovery",
        "ranking_discovery",
        "aggregate_ranking_discovery",
        "lineage_ranking_discovery",
        "compose_discovery",
        "normalize_discovery",
    ]:
        assert (
            calls[
                key
            ].candidate_count
            ==
            shortlist.candidate_count
        )

    assert discovery.candidate_count == 147

    assert (
        [
            item.analysis_id
            for item
            in discovery.candidates
        ]
        ==
        original_ids
    )

    assert (
        report
        .inventory
        .discovered_analysis_count
        ==
        147
    )

    assert any(
        (
            "Priorisation exploratoire avant exécution"
            in
            note
        )
        and
        (
            ANALYSIS_PRIORITIZATION_RULE_VERSION
            in
            note
        )
        for note
        in report.methodology_notes
    )

    print(
        "Broad discovery preserved while exploratory execution uses shortlist: PASS"
    )


def test_requested_analysis_path_remains_separate_from_exploratory_budget(
) -> None:
    source = inspect.getsource(
        analysis_run
        .run_contextualized_dataset_analysis
    )

    requested_position = source.find(
        "execute_requested_analysis_plan"
    )

    exploratory_position = source.find(
        "run_unified_analysis_from_prepared_records"
    )

    assert requested_position != -1
    assert exploratory_position != -1

    assert (
        requested_position
        <
        exploratory_position
    )

    assert (
        "prioritize_analysis_discovery"
        not in
        source[
            :exploratory_position
        ]
    )

    print(
        "Requested Analysis remains outside exploratory prioritization budget: PASS"
    )


def test_prioritization_is_integrated_only_in_prepared_exploration(
) -> None:
    source = inspect.getsource(
        analysis_run
        .run_unified_analysis_from_prepared_records
    )

    assert (
        "prioritize_analysis_discovery"
        in
        source
    )

    assert (
        "build_prioritized_execution_discovery"
        in
        source
    )

    assert (
        "report.inventory.discovered_analysis_count"
        in
        source
    )

    print(
        "Prepared exploratory pipeline contains deterministic prioritization: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS ANALYSIS PRIORITIZATION INTEGRATION v0.1 ==="
    )

    print()

    test_exploratory_execution_uses_shortlist_but_report_keeps_broad_count()
    test_requested_analysis_path_remains_separate_from_exploratory_budget()
    test_prioritization_is_integrated_only_in_prepared_exploration()

    print()

    print(
        "Analysis Prioritization Integration v0.1: PASS"
    )


if __name__ == "__main__":
    main()

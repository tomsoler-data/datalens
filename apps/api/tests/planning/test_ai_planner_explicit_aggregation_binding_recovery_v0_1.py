from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    canonicalize_aggregation_ranking_intent,
    validate_ai_proposal,
)


EXPECTED_VERSION = (
    "ai_analytical_planner_v0.34"
)

DATASET_ID = (
    "dataset:0001"
)


def column(
    name: str,
    kind: str,
    *,
    unique_count: int,
) -> PlannerColumnProfile:
    return PlannerColumnProfile(
        name=name,
        dtype=(
            "float64"
            if kind == "quantitative"
            else "object"
        ),
        analysis_kind=kind,
        missing_ratio=0.0,
        unique_count=unique_count,
        unique_candidate=False,
    )


def catalog() -> PlannerCatalog:
    return PlannerCatalog(
        datasets=[
            PlannerDatasetProfile(
                dataset_id=DATASET_ID,
                filename="orders_prepared.csv",
                row_count=12,
                column_count=4,
                columns=[
                    column(
                        "order_id",
                        "categorical",
                        unique_count=12,
                    ),
                    column(
                        "customer_segment",
                        "categorical",
                        unique_count=3,
                    ),
                    column(
                        "amount",
                        "quantitative",
                        unique_count=12,
                    ),
                    column(
                        "quantity",
                        "quantitative",
                        unique_count=5,
                    ),
                ],
                is_derived=False,
            ),
        ]
    )


def malformed_wire(
    *,
    benchmark_reference=None,
    benchmark_operator=None,
    benchmark_selection=None,
) -> AIPlannerProposal:
    return AIPlannerProposal(
        decision="propose",
        title="Customer Segment Average Amount Analysis",

        family="aggregation",
        dataset_id=DATASET_ID,
        analytical_grain="customer_segment",

        x_column="customer_segment",
        y_column="sum_amount",

        group_column=None,
        value_column=None,

        time_column=None,
        dimension_column=None,
        entity_column=None,

        aggregation_function="sum",

        ranking_order="descending",
        ranking_limit=None,

        window_operation="none",
        window_size=None,

        benchmark_reference=benchmark_reference,
        benchmark_operator=benchmark_operator,
        benchmark_selection=benchmark_selection,

        blockers=[],
        reasons=[],
        confidence=0.95,
    )


def exact_benchmark_objective() -> str:
    return (
        "Quels customer_segment ont "
        "un amount moyen superieur "
        "a la moyenne globale ?"
    )


def normal_mean_objective() -> str:
    return (
        "Quel est le amount moyen "
        "par customer_segment ?"
    )


def assert_live_failure_recovered(
    test_catalog: PlannerCatalog,
) -> None:
    item = validate_ai_proposal(
        objective=(
            exact_benchmark_objective()
        ),
        proposal=(
            malformed_wire(
                benchmark_reference=(
                    "overall_aggregate"
                ),
                benchmark_operator="gt",
                benchmark_selection=(
                    "matching_only"
                ),
            )
        ),
        proposal_index=1,
        catalog=test_catalog,
    )

    assert (
        item.validation_status
        ==
        "validated"
    ), item.model_dump()

    proposal = item.proposal

    assert proposal.family == "aggregation"
    assert proposal.dataset_id == DATASET_ID

    assert (
        proposal.group_column
        ==
        "customer_segment"
    )

    assert (
        proposal.value_column
        ==
        "amount"
    )

    assert (
        proposal.aggregation_function
        ==
        "mean"
    )

    assert (
        proposal.ranking_order
        ==
        "none"
    )

    assert (
        proposal.ranking_limit
        is None
    )

    assert (
        proposal.benchmark_reference
        ==
        "overall_aggregate"
    )

    assert (
        proposal.benchmark_operator
        ==
        "gt"
    )

    assert (
        proposal.benchmark_selection
        ==
        "matching_only"
    )

    assert item.contract is not None
    assert item.contract.status == "validated"

    assert item.contract.ranking is None

    assert item.contract.benchmark is not None

    assert (
        item.contract.benchmark.reference
        ==
        "overall_aggregate"
    )

    assert (
        item.contract.benchmark.operator
        ==
        "gt"
    )

    assert (
        item.contract.benchmark.selection
        ==
        "matching_only"
    )

    bindings = {
        binding.role:
            binding.column
        for binding
        in item.contract.bindings
    }

    assert (
        bindings.get("group")
        ==
        "customer_segment"
    )

    assert (
        bindings.get("value")
        ==
        "amount"
    )


def assert_normal_mean_recovered(
    test_catalog: PlannerCatalog,
) -> None:
    item = validate_ai_proposal(
        objective=(
            normal_mean_objective()
        ),
        proposal=(
            malformed_wire()
        ),
        proposal_index=1,
        catalog=test_catalog,
    )

    assert (
        item.validation_status
        ==
        "validated"
    ), item.model_dump()

    assert (
        item.proposal.group_column
        ==
        "customer_segment"
    )

    assert (
        item.proposal.value_column
        ==
        "amount"
    )

    assert (
        item.proposal.aggregation_function
        ==
        "mean"
    )

    assert (
        item.proposal.ranking_order
        ==
        "none"
    )

    assert (
        item.proposal.benchmark_reference
        is None
    )

    assert item.contract is not None
    assert item.contract.ranking is None
    assert item.contract.benchmark is None


def assert_missing_benchmark_fails_closed(
    test_catalog: PlannerCatalog,
) -> None:
    item = validate_ai_proposal(
        objective=(
            exact_benchmark_objective()
        ),
        proposal=(
            malformed_wire()
        ),
        proposal_index=1,
        catalog=test_catalog,
    )

    assert (
        item.validation_status
        ==
        "rejected"
    ), item.model_dump()

    assert item.contract is None

    joined = "\n".join(
        item.errors
    ).lower()

    assert "benchmark" in joined


def assert_wrong_operator_fails_closed(
    test_catalog: PlannerCatalog,
) -> None:
    item = validate_ai_proposal(
        objective=(
            exact_benchmark_objective()
        ),
        proposal=(
            malformed_wire(
                benchmark_reference=(
                    "overall_aggregate"
                ),
                benchmark_operator="lt",
                benchmark_selection=(
                    "matching_only"
                ),
            )
        ),
        proposal_index=1,
        catalog=test_catalog,
    )

    assert (
        item.validation_status
        ==
        "rejected"
    ), item.model_dump()

    assert item.contract is None

    joined = "\n".join(
        item.errors
    ).lower()

    assert "benchmark" in joined


def assert_exact_descriptive_aggregation_clears_unsolicited_benchmark(
    test_catalog: PlannerCatalog,
) -> None:
    item = validate_ai_proposal(
        objective=(
            "Quel est le amount moyen "
            "par customer_segment ?"
        ),
        proposal=(
            malformed_wire(
                benchmark_reference=(
                    "overall_aggregate"
                ),
                benchmark_operator="gt",
                benchmark_selection=(
                    "matching_only"
                ),
            )
        ),
        proposal_index=1,
        catalog=test_catalog,
    )

    assert (
        item.validation_status
        ==
        "validated"
    ), item.model_dump()

    assert (
        item.proposal.family
        ==
        "aggregation"
    )

    assert (
        item.proposal.group_column
        ==
        "customer_segment"
    )

    assert (
        item.proposal.value_column
        ==
        "amount"
    )

    assert (
        item.proposal.aggregation_function
        ==
        "mean"
    )

    assert (
        item.proposal.ranking_order
        ==
        "none"
    )

    assert (
        item.proposal.ranking_limit
        is None
    )

    assert (
        item.proposal.benchmark_reference
        is None
    )

    assert (
        item.proposal.benchmark_operator
        is None
    )

    assert (
        item.proposal.benchmark_selection
        is None
    )

    assert (
        item.contract
        is not None
    )

    assert (
        item.contract.status
        ==
        "validated"
    )

    assert (
        item.contract.ranking
        is None
    )

    assert (
        item.contract.benchmark
        is None
    )

    assert any(
        (
            "removed an unsolicited benchmark"
            in
            normalization
        )

        for normalization
        in item.normalizations
    )


def assert_three_columns_do_not_recover(
    test_catalog: PlannerCatalog,
) -> None:
    normalized, notes = (
        canonicalize_aggregation_ranking_intent(
            objective=(
                "Quel est le amount moyen "
                "par customer_segment "
                "selon quantity ?"
            ),
            proposal=(
                malformed_wire()
            ),
            catalog=test_catalog,
        )
    )

    assert (
        normalized.value_column
        is None
    )

    assert (
        normalized.group_column
        is None
    )

    assert not any(
        (
            "recovered explicit aggregation bindings"
            in
            note
        )
        for note
        in notes
    )


def assert_vague_does_not_recover(
    test_catalog: PlannerCatalog,
) -> None:
    normalized, notes = (
        canonicalize_aggregation_ranking_intent(
            objective=(
                "Analyser la performance "
                "selon customer_segment."
            ),
            proposal=(
                malformed_wire()
            ),
            catalog=test_catalog,
        )
    )

    assert (
        normalized.value_column
        is None
    )

    assert not any(
        (
            "recovered explicit aggregation bindings"
            in
            note
        )
        for note
        in notes
    )


def assert_clean_ranking_preserved(
    test_catalog: PlannerCatalog,
) -> None:
    clean = AIPlannerProposal(
        decision="propose",
        title="Ranking",
        family="ranking",
        dataset_id=DATASET_ID,
        analytical_grain="customer_segment",

        x_column=None,
        y_column=None,
        group_column=None,
        value_column="amount",
        time_column=None,
        dimension_column="customer_segment",
        entity_column=None,

        aggregation_function="mean",

        ranking_order="descending",
        ranking_limit=3,

        window_operation="none",
        window_size=None,

        benchmark_reference=None,
        benchmark_operator=None,
        benchmark_selection=None,

        blockers=[],
        reasons=[],
        confidence=0.95,
    )

    item = validate_ai_proposal(
        objective=(
            "Donne les 3 customer_segment "
            "les plus performants selon "
            "le amount moyen."
        ),
        proposal=clean,
        proposal_index=1,
        catalog=test_catalog,
    )

    assert (
        item.validation_status
        ==
        "validated"
    ), item.model_dump()

    assert item.proposal.family == "ranking"

    assert (
        item.proposal.ranking_order
        ==
        "descending"
    )

    assert (
        item.proposal.ranking_limit
        ==
        3
    )

    assert item.contract is not None
    assert item.contract.ranking is not None
    assert item.contract.benchmark is None


def main() -> None:
    print(
        "=== DATALENS EXPLICIT AGGREGATION "
        "RECOVERY v0.1 ==="
    )


    assert (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
        ==
        EXPECTED_VERSION
    )


    print(
        "[PASS] planner version v0.34"
    )


    test_catalog = catalog()


    assert_live_failure_recovered(
        test_catalog
    )

    print(
        "[PASS] captured live benchmark "
        "failure is recovered"
    )


    assert_normal_mean_recovered(
        test_catalog
    )

    print(
        "[PASS] malformed normal mean recovered"
    )


    assert_missing_benchmark_fails_closed(
        test_catalog
    )

    print(
        "[PASS] missing requested benchmark "
        "fails closed"
    )


    assert_wrong_operator_fails_closed(
        test_catalog
    )

    print(
        "[PASS] wrong benchmark operator "
        "fails closed"
    )


    assert_exact_descriptive_aggregation_clears_unsolicited_benchmark(
        test_catalog
    )

    print(
        "[PASS] unsolicited benchmark "
        "fails closed"
    )


    assert_three_columns_do_not_recover(
        test_catalog
    )

    print(
        "[PASS] 3-column objective "
        "does not recover"
    )


    assert_vague_does_not_recover(
        test_catalog
    )

    print(
        "[PASS] vague objective "
        "does not recover"
    )


    assert_clean_ranking_preserved(
        test_catalog
    )

    print(
        "[PASS] explicit ranking preserved"
    )


    print()
    print(
        "PASS - explicit aggregation "
        "recovery v0.1"
    )


if __name__ == "__main__":
    main()

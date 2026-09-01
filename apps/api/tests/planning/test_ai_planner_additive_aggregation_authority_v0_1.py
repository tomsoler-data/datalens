from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    canonicalize_analytical_view_intent,
    canonicalize_categorical_additive_view_from_objective,
    validate_ai_proposal,
)


EXPECTED_VERSION = (
    "ai_analytical_planner_v0.34"
)


BASE_DATASET_ID = (
    "dataset:0001"
)


DERIVED_DATASET_ID = (
    "derived:"
    "dataset_0001:"
    "category:"
    "customer_segment:"
    "amount"
)


def column(
    name: str,
    kind: str,
    *,
    unique_count: int,
) -> PlannerColumnProfile:
    dtype = (
        "float64"
        if kind == "quantitative"
        else "object"
    )

    return PlannerColumnProfile(
        name=name,
        dtype=dtype,
        analysis_kind=kind,
        missing_ratio=0.0,
        unique_count=unique_count,
        unique_candidate=False,
    )


def catalog() -> PlannerCatalog:
    base = PlannerDatasetProfile(
        dataset_id=BASE_DATASET_ID,
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
    )

    derived = PlannerDatasetProfile(
        dataset_id=DERIVED_DATASET_ID,
        filename=(
            "orders_prepared"
            "__by_customer_segment_amount.derived"
        ),
        row_count=3,
        column_count=3,
        columns=[
            column(
                "customer_segment",
                "categorical",
                unique_count=3,
            ),
            column(
                "sum_amount",
                "quantitative",
                unique_count=3,
            ),
            column(
                "event_count",
                "quantitative",
                unique_count=3,
            ),
        ],
        is_derived=True,
        derivation_type=(
            "categorical_additive_measure"
        ),
        analytical_grain=(
            "customer_segment"
        ),
        operation=(
            "groupby_sum"
        ),
        aggregation=(
            "sum"
        ),
        group_column=(
            "customer_segment"
        ),
        source_measure_column=(
            "amount"
        ),
        target_measure_column=(
            "sum_amount"
        ),
        metric_semantics=(
            "additive monetary amount"
        ),
        measure_semantic_aliases=[
            "amount",
        ],
    )

    return PlannerCatalog(
        datasets=[
            base,
            derived,
        ]
    )


def proposal(
    *,
    family: str,
    dataset_id: str,
    aggregation: str,
    ranking_order: str,
    ranking_limit,
    benchmark_reference=None,
    benchmark_operator=None,
    benchmark_selection=None,
) -> AIPlannerProposal:
    return AIPlannerProposal(
        decision="propose",
        title="Controlled proposal",
        family=family,
        dataset_id=dataset_id,
        analytical_grain=(
            "customer_segment"
        ),
        x_column=(
            "customer_segment"
        ),
        y_column=(
            "amount"
        ),
        group_column=None,
        value_column=None,
        time_column=None,
        dimension_column=None,
        entity_column=None,
        aggregation_function=aggregation,
        ranking_order=ranking_order,
        ranking_limit=ranking_limit,
        window_operation="none",
        window_size=None,
        benchmark_reference=(
            benchmark_reference
        ),
        benchmark_operator=(
            benchmark_operator
        ),
        benchmark_selection=(
            benchmark_selection
        ),
        blockers=[],
        reasons=[],
        confidence=0.95,
    )


def assert_mean_does_not_select_sum_view(
    test_catalog: PlannerCatalog,
) -> None:
    raw = proposal(
        family="ranking",
        dataset_id=BASE_DATASET_ID,
        aggregation="mean",
        ranking_order="descending",
        ranking_limit=3,
    )

    normalized, notes = (
        canonicalize_categorical_additive_view_from_objective(
            objective=(
                "Donne les 3 customer_segment "
                "les plus performants selon "
                "le amount moyen."
            ),
            proposal=raw,
            catalog=test_catalog,
        )
    )

    assert (
        normalized.model_dump()
        ==
        raw.model_dump()
    )

    assert (
        notes
        ==
        []
    )

    assert (
        normalized.dataset_id
        ==
        BASE_DATASET_ID
    )


def assert_sum_still_selects_additive_view(
    test_catalog: PlannerCatalog,
) -> None:
    raw = proposal(
        family="aggregation",
        dataset_id=BASE_DATASET_ID,
        aggregation="sum",
        ranking_order="none",
        ranking_limit=None,
    )

    normalized, notes = (
        canonicalize_categorical_additive_view_from_objective(
            objective=(
                "Donne le total amount "
                "par customer_segment."
            ),
            proposal=raw,
            catalog=test_catalog,
        )
    )

    assert (
        normalized.dataset_id
        ==
        DERIVED_DATASET_ID
    )

    assert (
        normalized.family
        ==
        "aggregation"
    )

    assert (
        normalized.group_column
        ==
        "customer_segment"
    )

    assert (
        normalized.value_column
        ==
        "sum_amount"
    )

    assert (
        normalized.aggregation_function
        ==
        "sum"
    )

    assert (
        bool(
            notes
        )
    )


def assert_existing_derived_sum_view_not_promoted_for_mean(
    test_catalog: PlannerCatalog,
) -> None:
    raw = AIPlannerProposal(
        decision="propose",
        title="Wrong derived mean",
        family="aggregation",
        dataset_id=DERIVED_DATASET_ID,
        analytical_grain=(
            "customer_segment"
        ),
        x_column=None,
        y_column=None,
        group_column=(
            "customer_segment"
        ),
        value_column=(
            "sum_amount"
        ),
        time_column=None,
        dimension_column=None,
        entity_column=None,
        aggregation_function="sum",
        ranking_order="none",
        ranking_limit=None,
        window_operation="none",
        window_size=None,
        benchmark_reference=None,
        benchmark_operator=None,
        benchmark_selection=None,
        blockers=[],
        reasons=[],
        confidence=0.95,
    )

    normalized, notes = (
        canonicalize_analytical_view_intent(
            objective=(
                "Quel est le amount moyen "
                "par customer_segment ?"
            ),
            proposal=raw,
            catalog=test_catalog,
        )
    )

    assert (
        normalized.model_dump()
        ==
        raw.model_dump()
    )

    assert (
        notes
        ==
        []
    )


def assert_mean_ranking_validates_on_source_dataset(
    test_catalog: PlannerCatalog,
) -> None:
    raw = proposal(
        family="ranking",
        dataset_id=BASE_DATASET_ID,
        aggregation="mean",
        ranking_order="descending",
        ranking_limit=3,
    )

    item = validate_ai_proposal(
        objective=(
            "Donne les 3 customer_segment "
            "les plus performants selon "
            "le amount moyen."
        ),
        proposal=raw,
        proposal_index=1,
        catalog=test_catalog,
    )

    assert (
        item.validation_status
        ==
        "validated"
    ), (
        item.model_dump()
    )

    assert (
        item.proposal.dataset_id
        ==
        BASE_DATASET_ID
    )

    assert (
        item.proposal.family
        ==
        "ranking"
    )

    assert (
        item.proposal.dimension_column
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
        "descending"
    )

    assert (
        item.proposal.ranking_limit
        ==
        3
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
        item.contract.required_dataset_ids
        ==
        [
            BASE_DATASET_ID
        ]
    )

    assert (
        item.contract.ranking
        is not None
    )

    assert (
        item.contract.ranking.order
        ==
        "descending"
    )

    assert (
        item.contract.ranking.limit
        ==
        3
    )

    assert (
        item.contract.benchmark
        is None
    )


def assert_mean_benchmark_validates_on_source_dataset(
    test_catalog: PlannerCatalog,
) -> None:
    raw = proposal(
        family="aggregation",
        dataset_id=BASE_DATASET_ID,
        aggregation="mean",
        ranking_order="none",
        ranking_limit=None,
        benchmark_reference=(
            "overall_aggregate"
        ),
        benchmark_operator=(
            "gt"
        ),
        benchmark_selection=(
            "matching_only"
        ),
    )

    item = validate_ai_proposal(
        objective=(
            "Quels customer_segment ont "
            "un amount moyen superieur "
            "a la moyenne globale ?"
        ),
        proposal=raw,
        proposal_index=1,
        catalog=test_catalog,
    )

    assert (
        item.validation_status
        ==
        "validated"
    ), (
        item.model_dump()
    )

    assert (
        item.proposal.dataset_id
        ==
        BASE_DATASET_ID
    )

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
        item.contract
        is not None
    )

    assert (
        item.contract.ranking
        is None
    )

    assert (
        item.contract.benchmark
        is not None
    )

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


def main() -> None:
    print(
        "=== DATALENS ADDITIVE AGGREGATION "
        "AUTHORITY v0.1 ==="
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

    assert_mean_does_not_select_sum_view(
        test_catalog
    )

    print(
        "[PASS] explicit mean does not select "
        "categorical groupby_sum view"
    )

    assert_sum_still_selects_additive_view(
        test_catalog
    )

    print(
        "[PASS] explicit sum still selects "
        "categorical additive view"
    )

    assert_existing_derived_sum_view_not_promoted_for_mean(
        test_catalog
    )

    print(
        "[PASS] direct derived SUM selection "
        "cannot override explicit mean"
    )

    assert_mean_ranking_validates_on_source_dataset(
        test_catalog
    )

    print(
        "[PASS] mean ranking validates against "
        "source amount"
    )

    assert_mean_benchmark_validates_on_source_dataset(
        test_catalog
    )

    print(
        "[PASS] mean benchmark validates against "
        "source amount"
    )

    print()
    print(
        "PASS - additive aggregation authority v0.1"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    decision_coverage_errors,
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


NORMAL_OBJECTIVE = (
    "Quel est le amount moyen "
    "par customer_segment ?"
)


BENCHMARK_OBJECTIVE = (
    "Quels customer_segment ont "
    "un amount moyen superieur "
    "a la moyenne globale ?"
)


def proposal(
    *,
    canonical: bool,
    benchmark_reference=None,
    benchmark_operator=None,
    benchmark_selection=None,
) -> AIPlannerProposal:
    if canonical:
        return AIPlannerProposal(
            decision="propose",
            title="Canonical aggregation",

            family="aggregation",
            dataset_id=DATASET_ID,
            analytical_grain="customer_segment",

            x_column=None,
            y_column=None,

            group_column="customer_segment",
            value_column="amount",

            time_column=None,
            dimension_column=None,
            entity_column=None,

            aggregation_function="mean",

            ranking_order="none",
            ranking_limit=None,

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


    return AIPlannerProposal(
        decision="propose",
        title="Malformed aggregation",

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


def assert_low_level_guard_still_detects_noise() -> None:
    raw = proposal(
        canonical=True,
        benchmark_reference="overall_aggregate",
        benchmark_operator="gt",
        benchmark_selection="matching_only",
    )


    errors = decision_coverage_errors(
        objective=NORMAL_OBJECTIVE,
        proposal=raw,
    )


    assert (
        len(
            errors
        )
        ==
        1
    ), errors


    assert (
        "benchmark"
        in
        errors[
            0
        ].lower()
    )


def assert_live_78_shape_is_recovered(
    test_catalog: PlannerCatalog,
) -> None:
    raw = AIPlannerProposal(
        decision="propose",
        title="Captured live shape",

        family="aggregation",
        dataset_id=DATASET_ID,
        analytical_grain="customer_segment",

        x_column=None,
        y_column="amount",

        group_column="customer_segment",
        value_column=None,

        time_column=None,
        dimension_column=None,
        entity_column=None,

        aggregation_function="sum",

        ranking_order="none",
        ranking_limit=None,

        window_operation="none",
        window_size=None,

        benchmark_reference="overall_aggregate",
        benchmark_operator="gt",
        benchmark_selection="matching_only",

        blockers=[],
        reasons=[],
        confidence=0.95,
    )


    item = validate_ai_proposal(
        objective=NORMAL_OBJECTIVE,
        proposal=raw,
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
        item.contract.ranking
        is None
    )


    assert (
        item.contract.benchmark
        is None
    )


def assert_canonical_normal_noise_is_recovered(
    test_catalog: PlannerCatalog,
) -> None:
    item = validate_ai_proposal(
        objective=NORMAL_OBJECTIVE,
        proposal=(
            proposal(
                canonical=True,
                benchmark_reference="overall_aggregate",
                benchmark_operator="gt",
                benchmark_selection="matching_only",
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
        item.proposal.benchmark_reference
        is None
    )


    assert (
        item.contract
        is not None
    )


    assert (
        item.contract.benchmark
        is None
    )


def assert_correct_benchmark_preserved(
    test_catalog: PlannerCatalog,
) -> None:
    for canonical in (
        False,
        True,
    ):
        item = validate_ai_proposal(
            objective=BENCHMARK_OBJECTIVE,
            proposal=(
                proposal(
                    canonical=canonical,
                    benchmark_reference="overall_aggregate",
                    benchmark_operator="gt",
                    benchmark_selection="matching_only",
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
            item.proposal.benchmark_reference
            ==
            "overall_aggregate"
        )


        assert (
            item.proposal.benchmark_operator
            ==
            "gt"
        )


        assert (
            item.proposal.benchmark_selection
            ==
            "matching_only"
        )


        assert (
            item.contract
            is not None
        )


        assert (
            item.contract.benchmark
            is not
            None
        )


def assert_wrong_operator_is_path_independently_rejected(
    test_catalog: PlannerCatalog,
) -> None:
    for canonical in (
        False,
        True,
    ):
        item = validate_ai_proposal(
            objective=BENCHMARK_OBJECTIVE,
            proposal=(
                proposal(
                    canonical=canonical,
                    benchmark_reference="overall_aggregate",
                    benchmark_operator="lt",
                    benchmark_selection="matching_only",
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


        # The final rejected proposal must expose the
        # planner's real decision, not a silent lt -> gt fix.

        assert (
            item.proposal.benchmark_operator
            ==
            "lt"
        ), item.model_dump()


        assert (
            item.contract
            is None
        )


        joined = "\n".join(
            item.errors
        ).lower()


        assert (
            "benchmark"
            in
            joined
        )


def assert_missing_explicit_benchmark_stays_fail_closed(
    test_catalog: PlannerCatalog,
) -> None:
    item = validate_ai_proposal(
        objective=BENCHMARK_OBJECTIVE,
        proposal=(
            proposal(
                canonical=True,
                benchmark_reference=None,
                benchmark_operator=None,
                benchmark_selection=None,
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


    assert (
        item.proposal.benchmark_reference
        is None
    )


    assert (
        item.contract
        is None
    )


def assert_vague_request_not_recovered(
    test_catalog: PlannerCatalog,
) -> None:
    item = validate_ai_proposal(
        objective=(
            "Analyser la performance "
            "selon customer_segment."
        ),
        proposal=(
            proposal(
                canonical=True,
                benchmark_reference="overall_aggregate",
                benchmark_operator="gt",
                benchmark_selection="matching_only",
            )
        ),
        proposal_index=1,
        catalog=test_catalog,
    )


    assert (
        item.validation_status
        !=
        "validated"
    ), item.model_dump()


def main() -> None:
    print(
        "=== DATALENS BENCHMARK DECISION "
        "BOUNDARY v0.1 ==="
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


    assert_low_level_guard_still_detects_noise()

    print(
        "[PASS] Decision Coverage still detects "
        "unsolicited benchmark noise"
    )


    assert_live_78_shape_is_recovered(
        test_catalog
    )

    print(
        "[PASS] exact live-78 normal aggregation "
        "shape is recovered"
    )


    assert_canonical_normal_noise_is_recovered(
        test_catalog
    )

    print(
        "[PASS] canonical descriptive aggregation "
        "clears unsolicited benchmark"
    )


    assert_correct_benchmark_preserved(
        test_catalog
    )

    print(
        "[PASS] correct explicit benchmark preserved "
        "for malformed and canonical wires"
    )


    assert_wrong_operator_is_path_independently_rejected(
        test_catalog
    )

    print(
        "[PASS] wrong benchmark operator is rejected "
        "for malformed and canonical wires"
    )


    assert_missing_explicit_benchmark_stays_fail_closed(
        test_catalog
    )

    print(
        "[PASS] missing explicit benchmark "
        "remains fail-closed"
    )


    assert_vague_request_not_recovered(
        test_catalog
    )

    print(
        "[PASS] vague request receives "
        "no benchmark-noise recovery"
    )


    print()
    print(
        "PASS - benchmark decision boundary v0.1"
    )


if __name__ == "__main__":
    main()

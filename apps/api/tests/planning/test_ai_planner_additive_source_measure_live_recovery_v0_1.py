from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    derived_additive_binding_semantic_concept,
    derived_additive_source_measure_fidelity_satisfied,
    validate_ai_proposal,
)


from app.planning.objective_coverage import (
    build_objective_coverage,
)


BASE_DATASET_ID = (
    "dataset:0001"
)


CATEGORY_VIEW_ID = (
    "derived:"
    "dataset_0001:"
    "category:"
    "category:"
    "gross_amount"
)


SESSION_VIEW_ID = (
    "derived:"
    "dataset_0001:"
    "session:"
    "order_id:"
    "gross_amount"
)


OBJECTIVE = (
    "Calcule la somme de gross_amount par category."
)


def column(
    name: str,
    dtype: str,
    kind: str,
    unique_count: int,
    unique_candidate: bool = False,
) -> PlannerColumnProfile:

    return PlannerColumnProfile(
        name=name,
        dtype=dtype,
        analysis_kind=kind,
        missing_ratio=0.0,
        unique_count=unique_count,
        unique_candidate=unique_candidate,
    )


def catalog() -> PlannerCatalog:

    return PlannerCatalog(
        datasets=[

            PlannerDatasetProfile(
                dataset_id=
                    BASE_DATASET_ID,

                filename=
                    "sales_live.csv",

                row_count=
                    4,

                column_count=
                    3,

                columns=[
                    column(
                        "order_id",
                        "str",
                        "identifier",
                        4,
                        True,
                    ),

                    column(
                        "category",
                        "str",
                        "categorical",
                        2,
                    ),

                    column(
                        "gross_amount",
                        "float64",
                        "quantitative",
                        4,
                    ),
                ],

                is_derived=
                    False,
            ),


            PlannerDatasetProfile(
                dataset_id=
                    CATEGORY_VIEW_ID,

                filename=(
                    "sales_live"
                    "__by_category_gross_amount.derived"
                ),

                row_count=
                    2,

                column_count=
                    3,

                columns=[
                    column(
                        "category",
                        "str",
                        "categorical",
                        2,
                    ),

                    column(
                        "sum_gross_amount",
                        "float64",
                        "quantitative",
                        2,
                    ),

                    column(
                        "event_count",
                        "int64",
                        "quantitative",
                        1,
                    ),
                ],

                is_derived=
                    True,

                derivation_type=
                    "categorical_additive_measure",

                analytical_grain=
                    "category",

                operation=
                    "groupby_sum",

                aggregation=
                    "sum",

                group_column=
                    "category",

                source_measure_column=
                    "gross_amount",

                target_measure_column=
                    "sum_gross_amount",

                measure_semantic_aliases=[
                    "gross_amount",
                    "sum_gross_amount",
                ],
            ),


            PlannerDatasetProfile(
                dataset_id=
                    SESSION_VIEW_ID,

                filename=(
                    "sales_live"
                    "__sessions_gross_amount.derived"
                ),

                row_count=
                    4,

                column_count=
                    4,

                columns=[
                    column(
                        "order_id",
                        "str",
                        "identifier",
                        4,
                        True,
                    ),

                    column(
                        "basket_amount",
                        "float64",
                        "quantitative",
                        4,
                    ),

                    column(
                        "item_count",
                        "int64",
                        "quantitative",
                        1,
                    ),

                    column(
                        "category",
                        "str",
                        "categorical",
                        2,
                    ),
                ],

                is_derived=
                    True,

                derivation_type=
                    "entity_additive_measure",

                analytical_grain=
                    "order_id",

                operation=
                    "session_materialization",

                aggregation=
                    "sum",

                entity_column=
                    "order_id",

                source_measure_column=
                    "gross_amount",

                target_measure_column=
                    "basket_amount",

                measure_semantic_aliases=[
                    "gross_amount",
                    "basket_amount",
                ],
            ),
        ]
    )


def captured_live_proposal() -> AIPlannerProposal:

    return AIPlannerProposal(
        decision=
            "propose",

        title=(
            "Calculate the sum of "
            "gross_amount by category"
        ),

        family=
            "aggregation",

        dataset_id=
            BASE_DATASET_ID,

        analytical_grain=
            "category",

        x_column=
            None,

        y_column=
            "sum_gross_amount",

        group_column=
            "category",

        value_column=
            None,

        time_column=
            None,

        dimension_column=
            None,

        entity_column=
            None,

        aggregation_function=
            "sum",

        ranking_order=
            "none",

        ranking_limit=
            None,

        window_operation=
            "none",

        window_size=
            None,

        benchmark_reference=
            "overall_aggregate",

        benchmark_operator=
            "gt",

        benchmark_selection=
            "matching_only",

        blockers=
            [],

        reasons=[
            (
                "The categorical additive view "
                "exposes sum_gross_amount."
            )
        ],

        confidence=
            0.95,
    )


def assert_captured_live_attempt_validates(
    test_catalog: PlannerCatalog,
) -> None:

    item = validate_ai_proposal(
        objective=
            OBJECTIVE,

        proposal=
            captured_live_proposal(),

        proposal_index=
            1,

        catalog=
            test_catalog,
    )


    assert (
        item.validation_status
        ==
        "validated"
    ), item.model_dump()


    assert (
        item.proposal.dataset_id
        ==
        CATEGORY_VIEW_ID
    )


    assert (
        item.proposal.group_column
        ==
        "category"
    )


    assert (
        item.proposal.value_column
        ==
        "sum_gross_amount"
    )


    assert (
        item.proposal.benchmark_reference
        is None
    )


    assert (
        item.contract
        is not None
    )


    bindings = {
        binding.role:
            binding

        for binding
        in item.contract.bindings
    }


    value_binding = (
        bindings[
            "value"
        ]
    )


    assert (
        value_binding.column
        ==
        "sum_gross_amount"
    )


    assert (
        value_binding.dataset_id
        ==
        CATEGORY_VIEW_ID
    )


    assert (
        value_binding.semantic_concept
        ==
        "gross_amount"
    )


    group_binding = (
        bindings[
            "group"
        ]
    )


    assert (
        group_binding.semantic_concept
        is None
    )


def assert_objective_coverage_uses_contract_provenance(
    test_catalog: PlannerCatalog,
) -> None:

    item = validate_ai_proposal(
        objective=
            OBJECTIVE,

        proposal=
            captured_live_proposal(),

        proposal_index=
            1,

        catalog=
            test_catalog,
    )


    assert (
        item.contract
        is not None
    )


    report = build_objective_coverage(
        objective=
            OBJECTIVE,

        catalog=
            test_catalog,

        contracts=[
            item.contract
        ],
    )


    assert (
        report.status
        ==
        "complete"
    ), report.model_dump()


    assert (
        report.covered_count
        ==
        2
    )


    assert (
        report.missing_count
        ==
        0
    )


    by_concept = {
        requirement.concept:
            requirement

        for requirement
        in report.requirements
    }


    gross = (
        by_concept[
            "gross_amount"
        ]
    )


    # No global physical alias was added.
    assert (
        gross.candidate_columns
        ==
        [
            "gross_amount"
        ]
    )


    assert (
        gross.covered
        is True
    )


    assert (
        item.contract.contract_id
        in
        gross.covered_by_contract_ids
    )


def assert_semantic_provenance_is_categorical_only(
    test_catalog: PlannerCatalog,
) -> None:

    category_view = next(
        dataset

        for dataset
        in test_catalog.datasets

        if (
            dataset.dataset_id
            ==
            CATEGORY_VIEW_ID
        )
    )


    session_view = next(
        dataset

        for dataset
        in test_catalog.datasets

        if (
            dataset.dataset_id
            ==
            SESSION_VIEW_ID
        )
    )


    assert (
        derived_additive_binding_semantic_concept(
            dataset=
                category_view,

            column_name=
                "sum_gross_amount",
        )
        ==
        "gross_amount"
    )


    assert (
        derived_additive_binding_semantic_concept(
            dataset=
                category_view,

            column_name=
                "event_count",
        )
        is None
    )


    assert (
        derived_additive_binding_semantic_concept(
            dataset=
                session_view,

            column_name=
                "basket_amount",
        )
        is None
    )


def assert_wrong_source_does_not_pass_fidelity(
    test_catalog: PlannerCatalog,
) -> None:

    item = validate_ai_proposal(
        objective=
            OBJECTIVE,

        proposal=
            captured_live_proposal(),

        proposal_index=
            1,

        catalog=
            test_catalog,
    )


    assert (
        item.validation_status
        ==
        "validated"
    )


    category_view = next(
        dataset

        for dataset
        in test_catalog.datasets

        if (
            dataset.dataset_id
            ==
            CATEGORY_VIEW_ID
        )
    )


    assert (
        derived_additive_source_measure_fidelity_satisfied(
            mention=
                "net_amount",

            proposal=
                item.proposal,

            dataset=
                category_view,
        )
        is False
    )


def assert_wire_noise_is_repaired_not_preserved(
    test_catalog: PlannerCatalog,
) -> None:

    noisy = (
        captured_live_proposal()
        .model_copy(
            update={
                "y_column":
                    "event_count",
            }
        )
    )


    item = validate_ai_proposal(
        objective=
            OBJECTIVE,

        proposal=
            noisy,

        proposal_index=
            1,

        catalog=
            test_catalog,
    )


    # Existing deterministic authority deliberately ignores this
    # contradictory wire slot because the explicit objective and
    # unique server-owned additive view resolve unambiguously.
    assert (
        item.validation_status
        ==
        "validated"
    ), item.model_dump()


    assert (
        item.proposal.value_column
        ==
        "sum_gross_amount"
    )


    assert (
        item.proposal.y_column
        is None
    )


    assert (
        item.contract
        is not None
    )


    bound_columns = {
        binding.column

        for binding
        in item.contract.bindings
    }


    assert (
        "event_count"
        not in
        bound_columns
    )


    assert (
        "sum_gross_amount"
        in
        bound_columns
    )


def main() -> None:

    print(
        "=== DATALENS ADDITIVE SOURCE-MEASURE "
        "LIVE RECOVERY v0.2 ==="
    )


    test_catalog = (
        catalog()
    )


    assert_captured_live_attempt_validates(
        test_catalog
    )

    print(
        "[PASS] captured Gemma attempt "
        "validates"
    )


    assert_objective_coverage_uses_contract_provenance(
        test_catalog
    )

    print(
        "[PASS] Objective Coverage uses "
        "contract-local semantic provenance"
    )


    assert_semantic_provenance_is_categorical_only(
        test_catalog
    )

    print(
        "[PASS] semantic provenance is limited "
        "to categorical additive target measure"
    )


    assert_wrong_source_does_not_pass_fidelity(
        test_catalog
    )

    print(
        "[PASS] unrelated source concept "
        "remains fail-closed"
    )


    assert_wire_noise_is_repaired_not_preserved(
        test_catalog
    )

    print(
        "[PASS] unrelated wire slot is discarded, "
        "not promoted"
    )


    print()
    print(
        "PASS - additive source-measure "
        "live recovery v0.2"
    )


if __name__ == "__main__":
    main()

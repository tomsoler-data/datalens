from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    validate_ai_proposal,
)


def build_catalog() -> PlannerCatalog:
    return PlannerCatalog(
        datasets=[
            PlannerDatasetProfile(
                dataset_id="dataset:sales",
                filename="sales.csv",
                row_count=39,
                column_count=4,
                columns=[
                    PlannerColumnProfile(
                        name="category",
                        dtype="object",
                        analysis_kind="categorical",
                        missing_ratio=0.0,
                        unique_count=3,
                    ),
                    PlannerColumnProfile(
                        name="unit_cost",
                        dtype="float64",
                        analysis_kind="quantitative",
                        missing_ratio=0.0,
                        unique_count=36,
                    ),
                    PlannerColumnProfile(
                        name="unit_price",
                        dtype="float64",
                        analysis_kind="quantitative",
                        missing_ratio=0.0,
                        unique_count=36,
                    ),
                    PlannerColumnProfile(
                        name="list_price",
                        dtype="float64",
                        analysis_kind="quantitative",
                        missing_ratio=0.0,
                        unique_count=36,
                    ),
                ],
            )
        ]
    )


def build_proposal(
    *,
    family: str,
    x_column: str | None = None,
    y_column: str | None = None,
    group_column: str | None = None,
    value_column: str | None = None,
) -> AIPlannerProposal:
    return AIPlannerProposal(
        decision="propose",
        title="Test semantic fidelity",
        family=family,  # type: ignore[arg-type]
        dataset_id="dataset:sales",
        analytical_grain=None,
        x_column=x_column,
        y_column=y_column,
        group_column=group_column,
        value_column=value_column,
        time_column=None,
        dimension_column=None,
        entity_column=None,
        aggregation_function="none",
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


def test_mixed_semantic_pair_repairs_wrong_quantitative_binding() -> None:
    item = validate_ai_proposal(
        objective=(
            "Étudier la relation entre les coûts unitaires "
            "et la catégorie de produit."
        ),
        proposal=build_proposal(
            family="quantitative_association",
            x_column="unit_cost",
            y_column="unit_price",
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )

    assert item.validation_status == "validated"
    assert item.contract is not None
    assert item.contract.family == "group_comparison"

    bindings = {
        binding.role: binding.column
        for binding in item.contract.bindings
    }

    assert bindings == {
        "group": "category",
        "value": "unit_cost",
    }

    assert any(
        "incohérence sémantique" in normalization
        for normalization in item.normalizations
    )


def test_two_quantitative_semantic_columns_remain_quantitative_association() -> None:
    item = validate_ai_proposal(
        objective=(
            "Le coût unitaire est-il associé "
            "au prix unitaire ?"
        ),
        proposal=build_proposal(
            family="quantitative_association",
            x_column="unit_cost",
            y_column="list_price",
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )

    assert item.validation_status == "validated"
    assert item.contract is not None
    assert item.contract.family == "quantitative_association"

    bindings = {
        binding.role: binding.column
        for binding in item.contract.bindings
    }

    assert set(bindings.values()) == {
        "unit_cost",
        "unit_price",
    }


def test_existing_correct_group_comparison_is_preserved() -> None:
    item = validate_ai_proposal(
        objective=(
            "Compare les coûts unitaires "
            "entre les catégories de produits."
        ),
        proposal=build_proposal(
            family="group_comparison",
            group_column="category",
            value_column="unit_cost",
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )

    assert item.validation_status == "validated"
    assert item.contract is not None
    assert item.contract.family == "group_comparison"

    bindings = {
        binding.role: binding.column
        for binding in item.contract.bindings
    }

    assert bindings == {
        "group": "category",
        "value": "unit_cost",
    }


if __name__ == "__main__":
    print(
        "=== DATALENS AI PLANNER SEMANTIC "
        "BINDING FIDELITY v0.1 ==="
    )

    test_mixed_semantic_pair_repairs_wrong_quantitative_binding()
    print(
        "[PASS] unit_cost + category cannot become "
        "unit_cost + unit_price"
    )

    test_two_quantitative_semantic_columns_remain_quantitative_association()
    print(
        "[PASS] unit_cost + unit_price remains "
        "quantitative association"
    )

    test_existing_correct_group_comparison_is_preserved()
    print(
        "[PASS] correct group comparison preserved"
    )

    print(
        "PASS - AI planner semantic binding fidelity v0.1"
    )

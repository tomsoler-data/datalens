from __future__ import annotations


import pandas as pd


from app.ai.native_tool_calling import (
    NativeToolCallProposal,
    SUPPORTED_NATIVE_FAMILIES,
    expected_tool_arguments,
    validate_native_tool_call,
)

from app.ai.tool_orchestrator import (
    execute_validated_contract,
)

from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    validate_ai_proposal,
)


# ============================================================
# FIXTURES
# ============================================================

def build_catalog() -> PlannerCatalog:
    return PlannerCatalog(
        datasets=[
            PlannerDatasetProfile(
                dataset_id="dataset:sales",
                filename="sales.csv",
                row_count=6,
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
                        name="unit_price",
                        dtype="float64",
                        analysis_kind="quantitative",
                        missing_ratio=0.0,
                        unique_count=6,
                    ),
                    PlannerColumnProfile(
                        name="list_price",
                        dtype="float64",
                        analysis_kind="quantitative",
                        missing_ratio=0.0,
                        unique_count=6,
                    ),
                    PlannerColumnProfile(
                        name="unit_cost",
                        dtype="float64",
                        analysis_kind="quantitative",
                        missing_ratio=0.0,
                        unique_count=6,
                    ),
                ],
            )
        ]
    )


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
            ],
            "unit_price": [
                10.0,
                20.0,
                30.0,
                40.0,
                25.0,
                35.0,
            ],
            "list_price": [
                12.0,
                18.0,
                50.0,
                70.0,
                30.0,
                40.0,
            ],
            "unit_cost": [
                5.0,
                7.0,
                20.0,
                22.0,
                15.0,
                16.0,
            ],
        }
    )


def build_proposal(
    *,
    family: str,
    value_column: str,
    aggregation_function: str,
    group_column: str | None = "category",
    dimension_column: str | None = None,
    ranking_order: str = "none",
    ranking_limit: int | None = None,
) -> AIPlannerProposal:
    return AIPlannerProposal(
        decision="propose",
        title="Test analytique",
        family=family,  # type: ignore[arg-type]
        dataset_id="dataset:sales",
        analytical_grain="category",
        x_column=None,
        y_column=None,
        group_column=group_column,
        value_column=value_column,
        time_column=None,
        dimension_column=dimension_column,
        entity_column=None,
        aggregation_function=aggregation_function,  # type: ignore[arg-type]
        ranking_order=ranking_order,  # type: ignore[arg-type]
        ranking_limit=ranking_limit,
        window_operation="none",
        window_size=None,
        benchmark_reference=None,
        benchmark_operator=None,
        benchmark_selection=None,
        blockers=[],
        reasons=[],
        confidence=0.95,
    )


def dataset_records() -> list[dict]:
    return [
        {
            "dataset_id": "dataset:sales",
            "filename": "sales.csv",
            "dataframe": build_dataframe(),
        }
    ]


# ============================================================
# TESTS
# ============================================================

def test_highest_mean_becomes_ranking_top_1() -> None:
    item = validate_ai_proposal(
        objective=(
            "Quelle catégorie a le prix unitaire moyen "
            "le plus élevé ?"
        ),
        proposal=build_proposal(
            family="aggregation",
            value_column="unit_price",
            aggregation_function="mean",
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )


    assert item.validation_status == "validated"
    assert item.contract is not None
    assert item.contract.family == "ranking"
    assert item.contract.aggregation is not None
    assert item.contract.aggregation.function == "mean"
    assert item.contract.ranking is not None
    assert item.contract.ranking.order == "descending"
    assert item.contract.ranking.limit == 1


    args = expected_tool_arguments(
        item.contract
    )


    assert args.model_dump() == {
        "dataset_id": "dataset:sales",
        "dimension_column": "category",
        "aggregation_function": "mean",
        "source_column": "unit_price",
        "order": "descending",
        "limit": 1,
    }


    trace = execute_validated_contract(
        contract=item.contract,
        datasets=dataset_records(),
        call_index=1,
    )


    assert trace.execution_status == "executed"
    assert trace.tool_name == "run_ranking"
    assert trace.result is not None
    assert trace.result.family == "ranking"
    assert len(trace.result.chart_data) == 1
    assert trace.result.chart_data[0]["category"] == "B"
    assert trace.result.chart_data[0]["value"] == 35.0
    assert trace.result.chart_data[0]["rank"] == 1



def test_top_2_catalog_price_is_exact_top_2() -> None:
    item = validate_ai_proposal(
        objective=(
            "Donne-moi les deux catégories ayant le prix "
            "catalogue moyen le plus élevé."
        ),
        proposal=build_proposal(
            family="group_comparison",
            value_column="list_price",
            aggregation_function="none",
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )


    assert item.validation_status == "validated"
    assert item.contract is not None
    assert item.contract.family == "ranking"
    assert item.contract.aggregation is not None
    assert item.contract.aggregation.function == "mean"
    assert item.contract.ranking is not None
    assert item.contract.ranking.order == "descending"
    assert item.contract.ranking.limit == 2


    trace = execute_validated_contract(
        contract=item.contract,
        datasets=dataset_records(),
        call_index=1,
    )


    assert trace.execution_status == "executed"
    assert trace.result is not None
    assert [
        row["category"]
        for row
        in trace.result.chart_data
    ] == [
        "B",
        "C",
    ]
    assert [
        row["value"]
        for row
        in trace.result.chart_data
    ] == [
        60.0,
        35.0,
    ]



def test_explicit_mean_group_request_becomes_aggregation() -> None:
    item = validate_ai_proposal(
        objective=(
            "Compare le prix unitaire moyen entre les catégories."
        ),
        proposal=build_proposal(
            family="group_comparison",
            value_column="unit_price",
            aggregation_function="none",
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )


    assert item.validation_status == "validated"
    assert item.contract is not None
    assert item.contract.family == "aggregation"
    assert item.contract.aggregation is not None
    assert item.contract.aggregation.function == "mean"
    assert item.contract.aggregation.group_by_roles == [
        "group"
    ]


    trace = execute_validated_contract(
        contract=item.contract,
        datasets=dataset_records(),
        call_index=1,
    )


    assert trace.execution_status == "executed"
    assert trace.tool_name == "run_aggregation"
    assert trace.result is not None
    assert trace.result.chart_type == "bar"
    assert {
        row["category"]: row["value"]
        for row
        in trace.result.chart_data
    } == {
        "A": 15.0,
        "B": 35.0,
        "C": 30.0,
    }



def test_generic_price_remains_ambiguous() -> None:
    item = validate_ai_proposal(
        objective=(
            "Compare le prix moyen entre les catégories."
        ),
        proposal=build_proposal(
            family="group_comparison",
            value_column="unit_price",
            aggregation_function="none",
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )


    assert item.validation_status == "ambiguous"
    assert item.contract is not None
    assert item.contract.status == "ambiguous"
    assert any(
        "AMBIGUÏTÉ MÉTRIQUE"
        in blocker
        for blocker
        in item.contract.blockers
    )



def test_native_registry_exposes_new_families() -> None:
    assert "aggregation" in SUPPORTED_NATIVE_FAMILIES
    assert "ranking" in SUPPORTED_NATIVE_FAMILIES



def test_python_rejects_wrong_ranking_limit_from_tool_model() -> None:
    item = validate_ai_proposal(
        objective=(
            "Donne-moi les deux catégories ayant le prix "
            "catalogue moyen le plus élevé."
        ),
        proposal=build_proposal(
            family="ranking",
            value_column="list_price",
            aggregation_function="mean",
            group_column=None,
            dimension_column="category",
            ranking_order="descending",
            ranking_limit=2,
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )


    assert item.validation_status == "validated"
    assert item.contract is not None


    errors = validate_native_tool_call(
        contract=item.contract,
        proposal=NativeToolCallProposal(
            tool_name="run_ranking",
            arguments={
                "dataset_id": "dataset:sales",
                "dimension_column": "category",
                "aggregation_function": "mean",
                "source_column": "list_price",
                "order": "descending",
                "limit": 3,
            },
        ),
    )


    assert errors
    assert any(
        "limit"
        in error
        for error
        in errors
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

def main() -> None:
    tests = [
        (
            "highest mean -> ranking top 1",
            test_highest_mean_becomes_ranking_top_1,
        ),
        (
            "top 2 catalog price -> exact top 2",
            test_top_2_catalog_price_is_exact_top_2,
        ),
        (
            "explicit mean -> aggregation",
            test_explicit_mean_group_request_becomes_aggregation,
        ),
        (
            "generic price remains ambiguous",
            test_generic_price_remains_ambiguous,
        ),
        (
            "native registry exposes aggregation/ranking",
            test_native_registry_exposes_new_families,
        ),
        (
            "wrong ranking limit rejected",
            test_python_rejects_wrong_ranking_limit_from_tool_model,
        ),
    ]


    print(
        "=== DATALENS AI NATIVE AGGREGATION / RANKING v0.1 ==="
    )


    for (
        label,
        test,
    ) in tests:
        test()
        print(
            f"[PASS] {label}"
        )


    print(
        "PASS - AI native aggregation / ranking v0.1"
    )


if __name__ == "__main__":
    main()

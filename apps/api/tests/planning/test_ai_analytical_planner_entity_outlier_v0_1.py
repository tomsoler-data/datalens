from __future__ import annotations

from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    SYSTEM_PROMPT,
    expected_roles_for_family,
    validate_ai_proposal,
    validate_role_type,
)


def build_catalog() -> PlannerCatalog:
    return PlannerCatalog(
        datasets=[
            PlannerDatasetProfile(
                dataset_id="dataset:generic",
                filename="generic.csv",
                row_count=100,
                column_count=3,
                analytical_grain="entity_id",
                entity_column="entity_id",
                columns=[
                    PlannerColumnProfile(
                        name="entity_id",
                        dtype="object",
                        analysis_kind="categorical",
                        missing_ratio=0.0,
                        unique_count=100,
                        unique_candidate=True,
                    ),
                    PlannerColumnProfile(
                        name="metric_value",
                        dtype="float64",
                        analysis_kind="quantitative",
                        missing_ratio=0.0,
                        unique_count=80,
                        unique_candidate=False,
                    ),
                    PlannerColumnProfile(
                        name="segment",
                        dtype="object",
                        analysis_kind="categorical",
                        missing_ratio=0.0,
                        unique_count=4,
                        unique_candidate=False,
                    ),
                ],
            )
        ]
    )


def build_proposal(
    *,
    entity_column: str | None = "entity_id",
    value_column: str | None = "metric_value",
) -> AIPlannerProposal:
    return AIPlannerProposal(
        decision="propose",
        title="Generic entity outlier analysis",
        family="entity_outlier",
        dataset_id="dataset:generic",
        analytical_grain="entity_id",
        x_column=None,
        y_column=None,
        group_column=None,
        value_column=value_column,
        time_column=None,
        dimension_column=None,
        entity_column=entity_column,
        aggregation_function="none",
        ranking_order="none",
        ranking_limit=None,
        window_operation="none",
        window_size=None,
        benchmark_reference=None,
        benchmark_operator=None,
        benchmark_selection=None,
        blockers=[],
        reasons=[
            "The objective requests atypical entities."
        ],
        confidence=0.95,
    )


def test_system_prompt_exposes_entity_outlier_family() -> None:
    assert "9. entity_outlier" in SYSTEM_PROMPT
    assert "entity_column" in SYSTEM_PROMPT
    assert "value_column quantitative" in SYSTEM_PROMPT


def test_entity_outlier_expected_roles_are_generic() -> None:
    assert (
        expected_roles_for_family(
            "entity_outlier"
        )
        ==
        {
            "entity",
            "value",
        }
    )


def test_entity_outlier_quantitative_value_is_valid() -> None:
    column = PlannerColumnProfile(
        name="metric_value",
        dtype="float64",
        analysis_kind="quantitative",
        missing_ratio=0.0,
        unique_count=80,
        unique_candidate=False,
    )

    assert (
        validate_role_type(
            family="entity_outlier",
            role="value",
            column=column,
        )
        is None
    )


def test_entity_outlier_non_quantitative_value_is_rejected() -> None:
    column = PlannerColumnProfile(
        name="segment",
        dtype="object",
        analysis_kind="categorical",
        missing_ratio=0.0,
        unique_count=4,
        unique_candidate=False,
    )

    error = validate_role_type(
        family="entity_outlier",
        role="value",
        column=column,
    )

    assert error is not None
    assert "quantitative" in error


def test_exact_ai_entity_outlier_proposal_becomes_validated_contract() -> None:
    result = validate_ai_proposal(
        objective=(
            "Identifie les entity_id atypiques "
            "selon metric_value."
        ),
        proposal=build_proposal(),
        proposal_index=1,
        catalog=build_catalog(),
    )

    assert result.validation_status == "validated"
    assert result.contract is not None

    contract = result.contract

    assert contract.family == "entity_outlier"
    assert contract.status == "validated"

    bindings = {
        binding.role:
            binding.column
        for binding
        in contract.bindings
    }

    assert bindings == {
        "entity": "entity_id",
        "value": "metric_value",
    }


def test_entity_outlier_missing_entity_is_rejected() -> None:
    result = validate_ai_proposal(
        objective=(
            "Identifie les profils atypiques "
            "selon metric_value."
        ),
        proposal=build_proposal(
            entity_column=None,
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )

    assert result.validation_status == "rejected"
    assert result.contract is None


def test_entity_outlier_missing_value_is_rejected() -> None:
    result = validate_ai_proposal(
        objective=(
            "Identifie les entity_id atypiques."
        ),
        proposal=build_proposal(
            value_column=None,
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )

    assert result.validation_status == "rejected"
    assert result.contract is None


def test_ai_entity_outlier_does_not_replace_invalid_value_column() -> None:
    result = validate_ai_proposal(
        objective=(
            "Identifie les entity_id atypiques "
            "selon segment."
        ),
        proposal=build_proposal(
            value_column="segment",
        ),
        proposal_index=1,
        catalog=build_catalog(),
    )

    assert result.validation_status == "rejected"
    assert result.contract is None

    assert any(
        "quantitative"
        in error
        for error
        in result.errors
    )


def test_ai_planner_rule_version_v0_36() -> None:
    assert (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
        ==
        "ai_analytical_planner_v0.36"
    )

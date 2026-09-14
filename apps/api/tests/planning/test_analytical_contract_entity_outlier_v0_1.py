from __future__ import annotations

import pytest

from pydantic import ValidationError

from app.planning.analytical_contract import (
    ANALYTICAL_CONTRACT_RULE_VERSION,
    AnalyticalContract,
    VariableBinding,
)


def build_entity_outlier_contract(
    *,
    bindings: list[VariableBinding],
) -> AnalyticalContract:
    return AnalyticalContract(
        contract_id="contract:entity_outlier:test",
        origin="ai_planner",
        status="validated",
        title="Generic entity outlier analysis",
        request_text="Identify atypical entities.",
        family="entity_outlier",
        analytical_grain="entity_id",
        bindings=bindings,
        reasons=[
            (
                "The analytical planner selected a generic "
                "entity-level outlier analysis."
            )
        ],
        planner_confidence=0.95,
    )


def test_entity_outlier_contract_accepts_entity_and_value() -> None:
    contract = build_entity_outlier_contract(
        bindings=[
            VariableBinding(
                role="entity",
                column="entity_id",
                analysis_kind="identifier",
            ),
            VariableBinding(
                role="value",
                column="metric_value",
                analysis_kind="quantitative",
            ),
        ]
    )

    assert contract.family == "entity_outlier"

    assert contract.roles() == {
        "entity",
        "value",
    }

    assert (
        contract.contract_version
        ==
        ANALYTICAL_CONTRACT_RULE_VERSION
    )

    assert (
        ANALYTICAL_CONTRACT_RULE_VERSION
        ==
        "analytical_contract_v0.4"
    )


def test_entity_outlier_contract_requires_entity_role() -> None:
    with pytest.raises(
        ValidationError
    ):
        build_entity_outlier_contract(
            bindings=[
                VariableBinding(
                    role="value",
                    column="metric_value",
                    analysis_kind="quantitative",
                ),
            ]
        )


def test_entity_outlier_contract_requires_value_role() -> None:
    with pytest.raises(
        ValidationError
    ):
        build_entity_outlier_contract(
            bindings=[
                VariableBinding(
                    role="entity",
                    column="entity_id",
                    analysis_kind="identifier",
                ),
            ]
        )


def test_blocked_entity_outlier_contract_may_remain_incomplete() -> None:
    contract = AnalyticalContract(
        contract_id="contract:entity_outlier:blocked",
        origin="ai_planner",
        status="blocked",
        title="Blocked entity outlier analysis",
        request_text="Identify atypical entities.",
        family="entity_outlier",
        blockers=[
            "No executable entity/value binding was validated."
        ],
    )

    assert contract.status == "blocked"
    assert contract.bindings == []

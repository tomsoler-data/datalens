from __future__ import annotations

import json

from types import SimpleNamespace

import app.planning.ai_analytical_planner as planner


# ============================================================
# FIXTURE
# ============================================================


def build_catalog() -> planner.PlannerCatalog:
    return planner.PlannerCatalog(
        datasets=[
            planner.PlannerDatasetProfile(
                dataset_id="dataset:test",
                filename="test.csv",
                row_count=10,
                column_count=1,
                columns=[
                    planner.PlannerColumnProfile(
                        name="value",
                        dtype="float64",
                        analysis_kind="quantitative",
                        missing_ratio=0.0,
                        unique_count=10,
                        unique_candidate=False,
                    )
                ],
            )
        ]
    )


# ============================================================
# TEST
# ============================================================


def test_ai_planner_uses_8192_context_window() -> None:
    captured_kwargs: dict[str, object] = {}

    payload = {
        "proposals": [
            {
                "decision": "blocked",
                "title": "Context window contract test",
                "family": "unresolved",
                "dataset_id": None,
                "analytical_grain": None,
                "x_column": None,
                "y_column": None,
                "group_column": None,
                "value_column": None,
                "time_column": None,
                "dimension_column": None,
                "entity_column": None,
                "aggregation_function": "none",
                "ranking_order": "none",
                "ranking_limit": None,
                "window_operation": "none",
                "window_size": None,
                "benchmark_reference": None,
                "benchmark_operator": None,
                "benchmark_selection": None,
                "blockers": [
                    "Synthetic transport-only test."
                ],
                "reasons": [],
                "confidence": None,
            }
        ]
    }

    def fake_classified_llm_chat(
        chat_client,
        *,
        payload_class,
        **kwargs,
    ):
        _ = chat_client
        _ = payload_class

        captured_kwargs.update(
            kwargs
        )

        return SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps(
                    payload
                )
            )
        )

    original_chat = (
        planner.classified_llm_chat
    )

    planner.classified_llm_chat = (
        fake_classified_llm_chat
    )

    try:
        (
            raw_output,
            _,
            _,
            _,
        ) = (
            planner
            ._generate_raw_ai_plan_with_timing(
                objective=(
                    "Analyse la variable value."
                ),
                catalog=build_catalog(),
                model="qwen3.5:4b",
                validation_feedback=None,
            )
        )

    finally:
        planner.classified_llm_chat = (
            original_chat
        )

    assert len(
        raw_output.proposals
    ) == 1

    options = captured_kwargs.get(
        "options"
    )

    assert isinstance(
        options,
        dict,
    )

    assert (
        options.get(
            "num_ctx"
        )
        ==
        8192
    ), (
        "AI Planner context window regression: "
        f"expected num_ctx=8192, got "
        f"{options.get('num_ctx')!r}."
    )


if __name__ == "__main__":
    test_ai_planner_uses_8192_context_window()

    print(
        "AI Planner context window v0.1: PASS"
    )

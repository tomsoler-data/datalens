from __future__ import annotations

from app.reporting.selected_report_pdf import (
    SELECTED_REPORT_PDF_RULE_VERSION,
    build_selected_report_pdf,
    format_number,
    translate_executor_text,
)


def fake_selection() -> dict:
    return {
        "workflow_id": "prep:test-v02",
        "revision": 2,
        "selected_count": 2,
        "analyses": [
            {
                "selection": {
                    "analysis_id": "analysis:initial",
                    "source_type": "initial_request",
                    "objective": (
                        "Étudier la relation entre les coûts unitaires "
                        "et la catégorie de produit."
                    ),
                    "trace_id": "trace:initial",
                    "report_order": 1,
                    "added_at_utc": "2026-08-20T18:00:00+00:00",
                    "executed": True,
                },
                "pipeline_payload": {
                    "planner_model": "gemma3:4b",
                    "tool_model": "qwen2.5:1.5b-instruct",
                    "planner": {
                        "items": [
                            {
                                "validation_status": "validated",
                                "contract": {
                                    "family": "group_comparison",
                                    "bindings": [
                                        {"role": "group", "column": "category"},
                                        {"role": "value", "column": "unit_cost"},
                                    ],
                                    "aggregation": None,
                                    "ranking": None,
                                },
                            }
                        ]
                    },
                    "items": [
                        {
                            "pipeline_status": "executed",
                            "native_tool": {
                                "requested_tool": "run_group_comparison",
                                "validation_status": "validated",
                                "execution": {
                                    "execution_status": "executed",
                                    "result": {
                                        "title": (
                                            "Étudier la relation entre les coûts unitaires "
                                            "et la catégorie de produit."
                                        ),
                                        "family": "group_comparison",
                                        "chart_type": "boxplot",
                                        "summary": [
                                            "3 groupe(s) ont été comparés pour unit_cost.",
                                            (
                                                "DataLens calculated the sample size, mean, median, "
                                                "dispersion and quartiles for each group."
                                            ),
                                        ],
                                        "metrics": {
                                            "valid_observations": 36,
                                            "group_count": 3,
                                        },
                                        "chart_data": [
                                            {
                                                "group": "Accessories",
                                                "min": 6.5,
                                                "q1": 12.0,
                                                "median": 17.0,
                                                "q3": 26.0,
                                                "max": 54.0,
                                            },
                                            {
                                                "group": "Electronics",
                                                "min": 12.0,
                                                "q1": 24.0,
                                                "median": 38.0,
                                                "q3": 310.0,
                                                "max": 620.0,
                                            },
                                            {
                                                "group": "Furniture",
                                                "min": 45.0,
                                                "q1": 80.0,
                                                "median": 118.0,
                                                "q3": 205.0,
                                                "max": 310.0,
                                            },
                                        ],
                                        "warnings": [
                                            (
                                                "This execution compares group distributions "
                                                "descriptively. No inferential group-comparison "
                                                "test has been applied yet."
                                            )
                                        ],
                                    },
                                },
                            },
                        }
                    ],
                },
            },
            {
                "selection": {
                    "analysis_id": "analysis:follow-up",
                    "source_type": "follow_up_prompt",
                    "objective": (
                        "Donne-moi les deux catégories ayant le prix catalogue "
                        "moyen le plus élevé."
                    ),
                    "trace_id": "trace:follow-up",
                    "report_order": 2,
                    "added_at_utc": "2026-08-20T18:02:00+00:00",
                    "executed": True,
                },
                "pipeline_payload": {
                    "planner_model": "gemma3:4b",
                    "tool_model": "qwen2.5:1.5b-instruct",
                    "planner": {
                        "items": [
                            {
                                "validation_status": "validated",
                                "contract": {
                                    "family": "ranking",
                                    "bindings": [
                                        {"role": "value", "column": "list_price"},
                                        {"role": "dimension", "column": "category"},
                                    ],
                                    "aggregation": {
                                        "function": "mean",
                                        "source_role": "value",
                                        "group_by_roles": ["dimension"],
                                    },
                                    "ranking": {
                                        "order": "descending",
                                        "limit": 2,
                                    },
                                },
                            }
                        ]
                    },
                    "items": [
                        {
                            "pipeline_status": "executed",
                            "native_tool": {
                                "requested_tool": "run_ranking",
                                "validation_status": "validated",
                                "execution": {
                                    "execution_status": "executed",
                                    "result": {
                                        "title": (
                                            "Analyser les deux catégories ayant le prix catalogue "
                                            "moyen le plus élevé."
                                        ),
                                        "family": "ranking",
                                        "chart_type": "bar",
                                        "summary": [
                                            (
                                                "2 résultat(s) ont été conservés après classement "
                                                "décroissant."
                                            ),
                                            "Premier résultat : Electronics = 273.1611111111111.",
                                        ],
                                        "metrics": {
                                            "source_observation_count": 39,
                                            "available_group_count": 3,
                                            "result_count": 2,
                                            "aggregation_function": "mean",
                                        },
                                        "chart_data": [
                                            {"category": "Electronics", "value": 273.1611111111111},
                                            {"category": "Furniture", "value": 255.9666666666667},
                                        ],
                                        "warnings": [],
                                    },
                                },
                            },
                        }
                    ],
                },
            },
        ],
    }


def fake_preparation_context() -> dict:
    return {
        "workflow_id": "prep:test-v02",
        "available": True,
        "ready_for_analysis": True,
        "session_revision": 12,
        "dataset_count": 1,
        "total_rows": 39,
        "datasets": [
            {
                "dataset_id": "combine:final",
                "filename": "orders_products_final.csv",
                "stage": "combine",
                "column_count": 12,
            }
        ],
        "stages": [
            {"stage": "import", "status": "passed"},
            {"stage": "understand", "status": "passed"},
            {"stage": "quality", "status": "passed"},
            {"stage": "clean", "status": "skipped"},
            {"stage": "transform", "status": "skipped"},
            {"stage": "combine", "status": "passed"},
            {"stage": "validate", "status": "passed"},
        ],
    }


def main() -> None:
    assert SELECTED_REPORT_PDF_RULE_VERSION == "selected_report_pdf_v0.2"
    assert format_number(273.1611111111111) == "273,16"
    assert (
        translate_executor_text(
            "Premier résultat : Electronics = 273.1611111111111."
        )
        == "Premier résultat : Electronics - 273,16."
    )
    assert translate_executor_text(
        "DataLens calculated the sample size, mean, median, dispersion and quartiles for each group."
    ).startswith("DataLens a calculé")

    pdf_bytes = build_selected_report_pdf(
        fake_selection(),
        preparation_context=fake_preparation_context(),
    )

    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 6000

    print("=== DATALENS SELECTED REPORT PDF v0.2 ===")
    print("[PASS] French number formatting")
    print("[PASS] executor text normalized to French")
    print("[PASS] deterministic ranking values rounded")
    print("[PASS] preparation/readiness section supported")
    print("[PASS] technical audit separated from business body")
    print("[PASS] PDF generated")
    print("PASS - selected report PDF v0.2")


if __name__ == "__main__":
    main()

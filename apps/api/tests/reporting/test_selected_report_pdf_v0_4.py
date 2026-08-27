from __future__ import annotations

from app.reporting.selected_report_pdf import (
    SELECTED_REPORT_PDF_RULE_VERSION,
    build_selected_report_pdf,
    display_source_filename,
    family_label,
    normalize_business_summary,
    status_label,
)


def pipeline(
    *,
    objective: str,
    family: str,
    chart_type: str,
    summary: list[str],
    metrics: dict,
    chart_data: list[dict],
    source_filename: str,
) -> dict:
    return {
        "planner_model": "python-deterministic",
        "tool_model": "python-deterministic",
        "planner": {
            "items": [
                {
                    "validation_status": "validated",
                    "contract": {
                        "family": family,
                        "bindings": [],
                    },
                }
            ],
        },
        "items": [
            {
                "pipeline_status": "executed",
                "native_tool": {
                    "requested_tool": f"run_{family}",
                    "validation_status": "validated",
                    "execution": {
                        "result": {
                            "title": objective,
                            "family": family,
                            "chart_type": chart_type,
                            "summary": summary,
                            "metrics": metrics,
                            "chart_data": chart_data,
                            "warnings": [],
                        }
                    },
                },
            }
        ],
        "report_context": {
            "source_filename": source_filename,
            "page_number": 1,
        },
    }


def fake_selection() -> dict:
    periods = []
    for year, months in [(2021, range(3, 13)), (2022, range(1, 13)), (2023, range(1, 3))]:
        for month in months:
            periods.append(
                {
                    "period": f"{year}-{month:02d}-01T00:00:00",
                    "value": 5000 + (year - 2021) * 900 + month * 70,
                }
            )

    rows = [
        (
            "analysis:clients",
            "nombre de clients par mois",
            pipeline(
                objective="nombre de clients par mois",
                family="time_series",
                chart_type="line",
                summary=["Les clients distincts ont été comptés pour chaque mois."],
                metrics={
                    "valid_observations": 687534,
                    "period_count": 24,
                    "distinct_customers_total": 8600,
                },
                chart_data=periods,
                source_filename="Brief+de+l%27analyse.pdf",
            ),
        ),
        (
            "analysis:transactions",
            "nombre de transactions",
            pipeline(
                objective="nombre de transactions",
                family="descriptive_metric",
                chart_type="metric",
                summary=[
                    "687534 événement(s) transactionnel(s) sont présents dans le dataset préparé."
                ],
                metrics={
                    "valid_observations": 687534,
                    "transaction_count": 687534,
                },
                chart_data=[],
                source_filename="Brief+de+l%27analyse.pdf",
            ),
        ),
        (
            "analysis:products",
            "nombre de produits vendus",
            pipeline(
                objective="nombre de produits vendus",
                family="descriptive_metric",
                chart_type="metric",
                summary=[
                    "687534 occurrence(s) produit sont observées dans les événements transactionnels.",
                    "Elles concernent 3265 référence(s) produit distincte(s).",
                ],
                metrics={
                    "valid_observations": 687534,
                    "products_sold_count": 687534,
                    "distinct_products_sold": 3265,
                },
                chart_data=[],
                source_filename="Brief+de+l%27analyse.pdf",
            ),
        ),
    ]

    return {
        "workflow_id": "prep:lapage-v04",
        "revision": 4,
        "selected_count": 3,
        "analyses": [
            {
                "selection": {
                    "analysis_id": analysis_id,
                    "source_type": "document_request",
                    "objective": objective,
                    "trace_id": f"trace:{index}",
                    "report_order": index,
                    "added_at_utc": "2026-08-24T15:00:00+00:00",
                    "executed": True,
                },
                "pipeline_payload": payload,
            }
            for index, (analysis_id, objective, payload) in enumerate(rows, start=1)
        ],
    }


def fake_preparation_context() -> dict:
    return {
        "workflow_id": "prep:lapage-v04",
        "available": True,
        "ready_for_analysis": True,
        "session_revision": 13,
        "dataset_count": 1,
        "total_rows": 687534,
        "datasets": [
            {
                "dataset_id": "combine:final",
                "filename": "Transactions__row_id__customers__products.csv",
                "stage": "combine",
                "column_count": 9,
            }
        ],
        "stages": [
            {"stage": "import", "status": "passed", "required": True, "materialized": False},
            {"stage": "understand", "status": "passed", "required": True, "materialized": False},
            {"stage": "quality", "status": "passed", "required": True, "materialized": False},
            {"stage": "clean", "status": "passed", "required": True, "materialized": False},
            {"stage": "transform", "status": "passed", "required": True, "materialized": True},
            {"stage": "combine", "status": "passed", "required": True, "materialized": True},
            {"stage": "validate", "status": "passed", "required": True, "materialized": False},
        ],
    }


def main() -> None:
    assert SELECTED_REPORT_PDF_RULE_VERSION == "selected_report_pdf_v0.4"
    assert family_label("descriptive_metric") == "Indicateur descriptif"
    assert display_source_filename("Brief+de+l%27analyse.pdf") == "Brief de l'analyse.pdf"
    assert normalize_business_summary(
        "687534 événement(s) transactionnel(s) sont présents dans le dataset préparé."
    ) == "687 534 transactions sont présentes dans le jeu de données préparé."
    assert status_label(
        "passed",
        stage="clean",
        required=True,
        materialized=False,
    ) == "Validé sans modification"
    assert status_label(
        "passed",
        stage="transform",
        required=True,
        materialized=True,
    ) == "Appliqué"

    pdf = build_selected_report_pdf(
        fake_selection(),
        preparation_context=fake_preparation_context(),
    )

    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 7000

    print("=== DATALENS SELECTED REPORT PDF v0.4 ===")
    print("[PASS] descriptive_metric translated")
    print("[PASS] document filename decoded")
    print("[PASS] business summaries normalized")
    print("[PASS] preparation status distinguishes no-op vs applied")
    print("[PASS] line time-series chart supported")
    print("[PASS] compact descriptive findings supported")
    print("[PASS] PDF generated")
    print("PASS - selected report PDF v0.4")


if __name__ == "__main__":
    main()

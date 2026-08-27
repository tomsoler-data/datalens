from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import sys
from typing import Any

import httpx


API_URL = os.getenv("DATALENS_API_URL", "http://127.0.0.1:8000").rstrip("/")
WORKFLOW_ID = os.getenv(
    "DATALENS_WORKFLOW_ID",
    "prep:7f74010272c149bda1384a55d990d4f2",
)
PLANNER_MODEL = os.getenv("DATALENS_PLANNER_MODEL", "gemma3:4b")
TOOL_MODEL = os.getenv("DATALENS_TOOL_MODEL", "qwen2.5:1.5b-instruct")
TIMEOUT_SECONDS = float(os.getenv("DATALENS_E2E_TIMEOUT_SECONDS", "180"))


@dataclass(frozen=True)
class Case:
    name: str
    objective: str
    family: str
    tool: str
    proposal: dict[str, Any]
    bindings: dict[str, str]
    chart_types: set[str]
    result_statuses: set[str] = field(default_factory=lambda: {"complete"})
    derived_token: str | None = None
    source_dataset: bool = False


CASES: tuple[Case, ...] = (
    Case(
        "CA par catégorie",
        "CA par catégorie",
        "aggregation",
        "run_aggregation",
        {
            "analytical_grain": "category",
            "group_column": "category",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
        },
        {"group": "category", "value": "sum_gross_amount"},
        {"bar"},
        derived_token=":category:category:gross_amount",
    ),
    Case(
        "CA par pays",
        "CA par pays",
        "aggregation",
        "run_aggregation",
        {
            "analytical_grain": "country",
            "group_column": "country",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
        },
        {"group": "country", "value": "sum_gross_amount"},
        {"bar"},
        derived_token=":category:country:gross_amount",
    ),
    Case(
        "CA par marque",
        "CA par marque",
        "aggregation",
        "run_aggregation",
        {
            "analytical_grain": "brand",
            "group_column": "brand",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
        },
        {"group": "brand", "value": "sum_gross_amount"},
        {"bar"},
        derived_token=":category:brand:gross_amount",
    ),
    Case(
        "CA par segment",
        "CA par segment",
        "aggregation",
        "run_aggregation",
        {
            "analytical_grain": "segment",
            "group_column": "segment",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
        },
        {"group": "segment", "value": "sum_gross_amount"},
        {"bar"},
        derived_token=":category:segment:gross_amount",
    ),
    Case(
        "Top 3 marques par CA",
        "Top 3 marques par chiffre d’affaires",
        "ranking",
        "run_ranking",
        {
            "analytical_grain": "brand",
            "dimension_column": "brand",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
            "ranking_order": "descending",
            "ranking_limit": 3,
        },
        {"dimension": "brand", "value": "sum_gross_amount"},
        {"bar"},
        derived_token=":category:brand:gross_amount",
    ),
    Case(
        "Évolution mensuelle du CA",
        "Évolution mensuelle du chiffre d’affaires",
        "time_series",
        "run_time_series",
        {
            "analytical_grain": "month",
            "time_column": "month",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
        },
        {"time": "month", "value": "sum_gross_amount"},
        {"line"},
        derived_token=":monthly:order_date:gross_amount",
    ),
    Case(
        "Panier selon le pays",
        "Comment le montant du panier varie-t-il selon le pays ?",
        "group_comparison",
        "run_group_comparison",
        {
            "analytical_grain": "order_id",
            "group_column": "country",
            "value_column": "basket_amount",
            "aggregation_function": "none",
        },
        {"group": "country", "value": "basket_amount"},
        {"box", "boxplot"},
        {"descriptive_only"},
        derived_token=":session:order_id:gross_amount",
    ),
    Case(
        "Quantité × prix unitaire",
        "Existe-t-il une relation entre la quantité commandée et le prix unitaire ?",
        "quantitative_association",
        "run_quantitative_association",
        {
            "x_column": "quantity",
            "y_column": "unit_price",
            "aggregation_function": "none",
        },
        {"x": "quantity", "y": "unit_price"},
        {"scatter"},
        {"needs_specialized_method"},
        source_dataset=True,
    ),
    Case(
        "Segment × catégorie",
        "Existe-t-il une relation entre le segment client et la catégorie de produit ?",
        "categorical_association",
        "run_categorical_association",
        {
            "x_column": "segment",
            "y_column": "category",
            "aggregation_function": "none",
        },
        {"x": "segment", "y": "category"},
        {"heatmap"},
        {"descriptive_only"},
        source_dataset=True,
    ),
)


def require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"{name} doit être un objet JSON.")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AssertionError(f"{name} doit être une liste JSON.")
    return value


def contract_bindings(contract: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in require_list(contract.get("bindings"), "contract.bindings"):
        binding = require_dict(raw, "binding")
        role = binding.get("role")
        column = binding.get("column")
        if isinstance(role, str) and isinstance(column, str):
            result[role] = column
    return result


def planner_item(body: dict[str, Any], family: str) -> dict[str, Any]:
    planner = require_dict(body.get("planner"), "planner")
    matches = []
    for raw in require_list(planner.get("items"), "planner.items"):
        item = require_dict(raw, "planner item")
        proposal = item.get("proposal")
        if (
            item.get("validation_status") == "validated"
            and isinstance(proposal, dict)
            and proposal.get("family") == family
        ):
            matches.append(item)
    if len(matches) != 1:
        raise AssertionError(
            f"1 contrat planner validé attendu pour {family!r}, trouvé: {len(matches)}."
        )
    return matches[0]


def pipeline_item(body: dict[str, Any], family: str) -> dict[str, Any]:
    matches = [
        require_dict(raw, "pipeline item")
        for raw in require_list(body.get("items"), "items")
        if isinstance(raw, dict) and raw.get("family") == family
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"1 item pipeline attendu pour {family!r}, trouvé: {len(matches)}."
        )
    return matches[0]


def run_prompt(client: httpx.Client, objective: str) -> dict[str, Any]:
    # (None, value) force un multipart/form-data identique au FormData du frontend,
    # sans fournir de dataset_files. Après VALIDATE, le workflow_id reste autoritaire.
    multipart = [
        ("workflow_id", (None, WORKFLOW_ID)),
        ("objective", (None, objective)),
        ("planner_model", (None, PLANNER_MODEL)),
        ("tool_model", (None, TOOL_MODEL)),
    ]
    response = client.post("/planning/ai-native-run", files=multipart)
    try:
        payload = response.json()
    except Exception:
        payload = {"raw_text": response.text}
    if response.status_code != 200:
        raise AssertionError(
            f"HTTP {response.status_code}\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
        )
    return require_dict(payload, "response")


def validate(case: Case, body: dict[str, Any]) -> tuple[str, str]:
    assert body.get("status") == "ready"

    planner = require_dict(body.get("planner"), "planner")
    assert planner.get("validated_count") == 1, (
        "Le planner doit produire exactement 1 contrat validé. "
        f"validated={planner.get('validated_count')}, "
        f"ambiguous={planner.get('ambiguous_count')}, "
        f"blocked={planner.get('blocked_count')}, "
        f"rejected={planner.get('rejected_count')}"
    )

    p_item = planner_item(body, case.family)
    proposal = require_dict(p_item.get("proposal"), "proposal")
    for key, expected in case.proposal.items():
        actual = proposal.get(key)
        assert actual == expected, f"proposal.{key}: attendu {expected!r}, obtenu {actual!r}"

    dataset_id = proposal.get("dataset_id")
    assert isinstance(dataset_id, str)
    if case.derived_token:
        assert case.derived_token in dataset_id, (
            f"Mauvaise vue analytique: {dataset_id!r}; "
            f"token attendu: {case.derived_token!r}"
        )
    if case.source_dataset:
        assert not dataset_id.startswith("derived:"), (
            f"Une source validée était attendue, pas une vue dérivée: {dataset_id!r}"
        )

    contract = require_dict(p_item.get("contract"), "contract")
    assert contract.get("status") == "validated"
    assert contract.get("family") == case.family
    assert contract_bindings(contract) == case.bindings, (
        f"Bindings attendus {case.bindings!r}, "
        f"obtenus {contract_bindings(contract)!r}"
    )

    pipe = pipeline_item(body, case.family)
    assert pipe.get("pipeline_status") == "executed"

    native = require_dict(pipe.get("native_tool"), "native_tool")
    assert native.get("validation_status") == "validated", native.get("validation_errors")
    assert native.get("requested_tool") == case.tool

    execution = require_dict(native.get("execution"), "native_tool.execution")
    assert execution.get("execution_status") == "executed", execution.get("errors")
    assert execution.get("tool_name") == case.tool

    result = require_dict(execution.get("result"), "execution.result")
    assert result.get("family") == case.family
    assert result.get("execution_status") in case.result_statuses, (
        f"result.execution_status={result.get('execution_status')!r}, "
        f"attendu={sorted(case.result_statuses)!r}"
    )
    assert result.get("chart_type") in case.chart_types, (
        f"chart_type={result.get('chart_type')!r}, "
        f"attendu={sorted(case.chart_types)!r}"
    )
    chart_data = result.get("chart_data")
    assert isinstance(chart_data, list) and chart_data, "chart_data doit être non vide."

    analysis_id = body.get("analysis_id")
    assert isinstance(analysis_id, str) and analysis_id, (
        "Le résultat exécuté doit être enregistré comme artefact serveur."
    )

    return str(result.get("chart_type")), str(result.get("execution_status"))


def main() -> int:
    print("=" * 78)
    print("DATALENS ANALYSIS PROMPT E2E SMOKE v0.2")
    print("=" * 78)
    print(f"API      : {API_URL}")
    print(f"Workflow : {WORKFLOW_ID}")
    print(f"Planner  : {PLANNER_MODEL}")
    print(f"Tool     : {TOOL_MODEL}")
    print()

    failures: list[tuple[str, str]] = []

    with httpx.Client(base_url=API_URL, timeout=TIMEOUT_SECONDS) as client:
        for index, case in enumerate(CASES, start=1):
            print(f"[{index:02d}/{len(CASES):02d}] {case.name}")
            try:
                body = run_prompt(client, case.objective)
                chart_type, result_status = validate(case, body)
                print(
                    f"  [PASS] family={case.family} · tool={case.tool} · "
                    f"chart={chart_type} · result_status={result_status}"
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                failures.append((case.name, message))
                print(f"  [FAIL] {message}")
            print()

    print("=" * 78)
    if failures:
        print(f"FAIL - {len(failures)} / {len(CASES)} cas E2E en échec")
        for name, message in failures:
            print(f"- {name}: {message}")
        return 1

    print("PASS - les 9 questions traversent le vrai endpoint UI /planning/ai-native-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
